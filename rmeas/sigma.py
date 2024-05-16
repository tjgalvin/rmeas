"""Tooling around the rms and background estimation"""

from typing import Tuple, NamedTuple, Optional, Callable
import logging 

import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import minimize
from astropy.stats import sigma_clip

class Result(NamedTuple):
    """Container for function results"""
    rms: float
    """RMS constrained"""
    bkg: float 
    """BKG constrained"""
    valid_pixels: int 
    """Number of pixels constrained against"""


class FitGaussianCDF(NamedTuple):
    """Options for the fitting approach method"""
    linex_args: Tuple[float,float] = (0.1, 1.0)
    
    def perform(self, data: np.ndarray) -> Result:
        return fit_gaussian_cdf(data=data, linex_args=self.linex_args)


def softmax(x: np.ndarray) -> np.ndarray:
    """Activation function"""
    return np.log(1+np.exp(x))

def gaussian_cdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Calculate teh cumulative distribution given a mu and sig across x-values"""
    top = x - mu
    bottom = 1.412 * sig
    
    return 0.5 * (1. + erf(top / bottom))


def linear_squared(x: np.ndarray, a: float, b: float) -> np.ndarray:
    
    mask = x < 0
    
    return (a * -x + 1) * mask + (b * (1. + x))** 2. * ~mask

def make_fitness(
    x: np.ndarray, y: np.ndarray, a: float, b: float
) -> Callable:
    """Creates the cost function on the (x,y) values that will be used throughout minimisation. """
    
    def loss_function(args):
        mu, sig, scalar = args

        s = np.sum(
            linear_squared(
                gaussian_cdf(x, mu, sig) * softmax(scalar) - y, 
                a, 
                b, 
            )
        )
        
        return s
    return loss_function

def fit_gaussian_cdf(data: np.ndarray, linex_args: Tuple[float,float] = (0.1, 1.0)) -> Result:

    data = data[np.isfinite(data)].flatten()

    sort_data = np.sort(data)
    cdf_data = np.arange(start=0, stop=1, step=1./len(sort_data))

    func = make_fitness(sort_data, cdf_data, *linex_args)
    
    res = minimize(
        func, (0, 0.0001, 0.8),
        # tol=1e-7,
        method='Nelder-Mead'
    )

    bane_result = Result(rms=res.x[1], bkg=res.x[0], valid_pixels=int(res.x[2] * len(data)))

    return bane_result

class FittedSigmaClip(NamedTuple):
    """Arguments for the fitted_sigma_clip"""
    sigma: int = 3 
    """Threshhold before clipped"""
    
    def perform(self, data: np.ndarray) -> Result:
        return fitted_sigma_clip(data=data, sigma=self.sigma)
        
def fitted_mean(data: np.ndarray, axis: Optional[int] =None) -> float:
    """Internal function that returns the mean by fitting to pixel distribution data"""
    if axis is not None:
        # This is to make astropy sigma clip happy
        raise NotImplementedError("Unexpected axis keyword. ")
    
    mean, _ = norm.fit(data)
    
    return mean


def fitted_std(data: np.ndarray, axis: Optional[int]=None) -> float:
    """Internal function that retunrs the stf by fitting to the pixel distribution"""
    if axis is not None:
        # This is to make astropy sigma clip happy
        raise NotImplementedError("Unexpected axis keyword. ")
    
    _, std = norm.fit(data)
    
    return std

def fitted_sigma_clip(data: np.ndarray, sigma: int=3) -> Result:
    
    data = data[np.isfinite(data)]
    
    clipped_plane = sigma_clip(
        data.flatten(), 
        sigma=sigma, 
        cenfunc=fitted_mean, # type: ignore
        stdfunc=fitted_std,  # type: ignore
    )
    clipped_plane = np.reshape(np.array(clipped_plane), -1)
    
    bkg, rms = norm.fit(clipped_plane)

    result = Result(rms=float(rms), bkg=float(bkg), valid_pixels=len(clipped_plane))

    return result

class FitBkgRmsEstimate(NamedTuple):
    """Options for the fitting approach method"""
    clip_rounds: int = 3
    """Number of clipping rounds to perform"""
    bin_perc: float = 0.25
    """Minimum fraction of the histogram bins, or something"""
    outlier_thres: float = 3.0
    """Threshold that a data point should be at to be considered an outlier"""

    def perform(self, data: np.ndarray) -> Result:
        return fit_bkg_rms_estimate(data=data, clip_rounds=self.clip_rounds, bin_perc=self.bin_perc, outlier_thres=self.outlier_thres)

def mad(data, bkg=None):
    """Compute the median asbolute deviation. optionally provide a 
    precomuted background measure
    """
    bkg = bkg if bkg else np.median(data)
    return np.median(np.abs(data - bkg))

def fit_bkg_rms_estimate(
    data: np.ndarray,
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
) -> Result:
   
    data = data[np.isfinite(data)]

    cen_func = np.median

    bkg = cen_func(data)

    for i in range(clip_rounds):
        data = data[np.abs(data - bkg) < outlier_thres * 1.4826 * mad(data, bkg=bkg)]
        bkg = cen_func(data)

    # Attempts to ensure a sane number of bins to fit against
    mask_counts = 0
    loop = 1
    while True:
        counts, binedges = np.histogram(data, bins=50 * loop)

        mask = counts >= bin_perc * np.max(counts)
        mask_counts = np.sum(mask)
        loop += 1

        if not (mask_counts < 5 and loop < 5): 
            break

    binc = (binedges[:-1] + binedges[1:]) / 2
    p = np.polyfit(binc[mask], np.log10(counts[mask] / np.max(counts)), 2)
    a, b, c = p

    x1 = (-b + np.sqrt(b ** 2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    x2 = (-b - np.sqrt(b ** 2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    fwhm = np.abs(x1 - x2)
    noise = fwhm / 2.355

    result = Result(rms=float(noise), bkg=float(bkg), valid_pixels=len(data))

    return result



class SigmaClip(NamedTuple):
    """Container for the original sigma clipping method"""
    low: float = 3.0
    """Low sigma clip threshhold"""
    high: float = 3.0
    """High sigma clip threshhold"""

    def perform(self, data: np.ndarray) -> Result:
        return sigmaclip(arr=data, lo=self.low, hi=self.high)

def sigmaclip(arr, lo, hi, reps=10) -> Result:      
    
    clipped = np.array(arr)[np.isfinite(arr)]

    if len(clipped) < 1:
        return Result(rms=np.nan, bkg=np.nan, valid_pixels=0)

    std = np.std(clipped)
    mean = np.mean(clipped)
    prev_valid = len(clipped)
    count = reps
    for count in range(int(reps)):
        mask = (clipped > mean-std*lo) & (clipped < mean+std*hi)
        clipped = clipped[mask]

        curr_valid = len(clipped)
        if curr_valid < 1:
            break
        # No change in statistics if no change is noted
        if prev_valid == curr_valid:
            break
        std = np.std(clipped)
        mean = np.mean(clipped)
        prev_valid = curr_valid
    else:
        logging.debug(
            f"No stopping criteria was reached after {count} cycles"
        )

    result = Result(rms=float(std), bkg=float(mean), valid_pixels=len(clipped))

    return result
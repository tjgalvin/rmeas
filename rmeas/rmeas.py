import math
from argparse import ArgumentParser 
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, TypeAlias, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from astropy.io import fits

from rmeas.logging import logger
from rmeas.sigma import FitGaussianCDF, FitBkgRmsEstimate, SigmaClip, Result

ClippingModes: TypeAlias =  Union[SigmaClip, FitBkgRmsEstimate,FitGaussianCDF]

@dataclass
class Image:
    path: Path 
    nx: int 
    ny: int 
    original_shape: Tuple[int,...]
    data: np.ndarray

@dataclass
class GridStep:
    dx: int 
    dy: int

@dataclass
class GridPoint:
    x: int 
    y: int

GridPoints = Tuple[GridPoint,...]


@dataclass
class Box:
    x_min: int
    x_max: int 
    y_min: int 
    y_max: int 
    x: int 
    y: int 
    data: np.ndarray
    
    
def make_box(image: Image, grid_point: GridPoint) -> Box:
    xmin, xmax = grid_point.x - 50, grid_point.x + 50 
    xmin = max(0, xmin)
    xmax = min(image.nx, xmax)
    
    ymin, ymax = grid_point.y - 50, grid_point.y + 50 
    ymin = max(0, ymin)
    ymax = min(image.ny, ymax)
    
    data = image.data[ymin:ymax, xmin:xmax]
    
    return Box(
        x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, x=grid_point.x, y=grid_point.y, data=data
    )

def get_image_from_fits(image: Path) -> Image:
    logger.info(f"Loading {image}")
    with fits.open(name=image) as ofits:
        data: np.ndarray = ofits[0].data
    
        original_shape = data.shape
        data = np.squeeze(data)
    
    image_prop = Image(
        path=image, nx=data.shape[1], ny=data.shape[0], original_shape=original_shape, data=data
    )
    
    return image_prop


def create_box_centers(image: Image, grid_step: GridStep) -> GridPoints:

    grid_points = []

    x0 = math.floor(grid_step.dx / 2)
    y0 = math.floor(grid_step.dy / 2)

    for x in range(x0, image.nx, grid_step.dx):
        for y in range(y0, image.ny, grid_step.dy):
            grid_points.append(GridPoint(x=x, y=y))
        
    logger.info(f"Number of grid points: {len(grid_points)}")
    return tuple(grid_points)

def _evaluate(boxmode: Tuple[Box, ClippingModes]) -> Result:
    
    box, mode = boxmode
    result = mode.perform(data=box.data)
    
    print(result)
    
    return result
    
def execute_box_filter(image: Image, mode: ClippingModes, grid_points: GridPoints) -> None:
    
    boxes = [make_box(image=image, grid_point=gp) for gp in grid_points]
    args = [(box, mode) for box in boxes]
    
    logger.info(f"About to map thread pool across: {len(args)}")
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = pool.map(_evaluate, args)
    

def rmeas(image_path: Path) -> None:
    
    image = get_image_from_fits(image=image_path)
    
    grid_step = GridStep(dx=10, dy=10)
    grid_points = create_box_centers(image=image, grid_step=grid_step)

    mode: ClippingModes = SigmaClip(low=3, high=3)
    mode: ClippingModes = FitBkgRmsEstimate()
    mode: ClippingModes = FitGaussianCDF()

    logger.info(f"Selected {mode=}")

    execute_box_filter(image=image, mode=mode, grid_points=grid_points)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Measure the RMS of an image")

    parser.add_argument('image_path', type=Path, help="The path of the file to measure the rms and bkg from")

    return parser

def cli():
    parser = get_parser() 
    
    args = parser.parse_args() 
    print(f"{args=}")
    rmeas(
        image_path=args.image_path #type: ignore
    )

if __name__ == '__main__':
    cli()
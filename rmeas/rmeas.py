import math
from argparse import ArgumentParser 
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


import numpy as np
from astropy.io import fits

from rmeas.logging import logger

GridPoints = Tuple[Tuple[int,int],...]

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

def get_image_from_fits(image: Path) -> Image:
    
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

    for x in range(start=x0, stop=image.nx, step=grid_step.dx):
        for y in range(start=y0, stop=image.ny, step=grid_step.dy):
            grid_points.append(GridPoint(x=x, y=y))
        
    logger.info(f"{grid_points=}")
    return tuple(grid_points)


def rmeas(image_path: Path) -> None:
    
    image = get_image_from_fits(image=image_path)
    
    grid_step = GridStep(dx=100, dy=100)
    grid_points = create_box_centers(image=image, grid_step=grid_step)



def cli() -> ArgumentParser:
    parser = ArgumentParser(description="Measure the RMS of an image")
    parser.add_argument('image', type=Path, help="The path of the file to measure the rms and bkg from")


    return parser

if __name__ == '__main__':
    parser = cli() 
    
    args = parser.parse_args() 
    
    rmeas(
        image=args.image
    )
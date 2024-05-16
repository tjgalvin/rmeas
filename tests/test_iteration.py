from pathlib import Path 

import numpy as np

from rmeas.rmeas import create_box_centers, Image, GridStep, GridPoint


def test_iteration():

    img = np.ones((100,100))

    image = Image(
        path=Path("Does/not/exist.fits"), 
        ny=100, 
        nx=100, 
        original_shape=(100,100), 
        data=img
    )
    grid_step  = GridStep(dx=10, dy=10)
    
    grid_points = create_box_centers(image=image, grid_step=grid_step)

    assert isinstance(grid_points, tuple)
    assert all([isinstance(gp, GridPoint) for gp in grid_points])
    assert len(grid_points) == 100
    
    





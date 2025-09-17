import ee
import json
import time
from typing import List, Dict, Any

def sample_points_from_grids(grids_collection, temp_crop_mask, points_per_grid, seed):
    """Sample points from a collection of grids"""
    
    def sample_points_from_grid(grid_cell):
        """Sample random points from temporary crop areas within a grid cell"""
        cell_mask = temp_crop_mask.clip(grid_cell.geometry()).selfMask()
        
        # Sample random points where temp crop exists
        random_points = cell_mask.stratifiedSample(
            numPoints=points_per_grid,
            classBand='classification',
            region=grid_cell.geometry(),
            scale=10,
            classValues=[1],
            classPoints=[points_per_grid],
            seed=seed,
            geometries=True
        )
        
        # Add grid cell ID to each point
        grid_id = grid_cell.get('system:index')
        def add_grid_info(point):
            return point.set({
                'grid_id': grid_id,
                'temp_crop_area': grid_cell.get('tempCropArea'),
                'cell_area': grid_cell.get('cellArea'),
                'country': grid_cell.get('country'),
                'continent': grid_cell.get('continent')
            })
        
        return random_points.map(add_grid_info)
    
    # Apply sampling to all grids and flatten the results
    return grids_collection.map(sample_points_from_grid).flatten()
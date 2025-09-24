import ee
import json
import os
import time
import argparse
from typing import List, Dict, Any

ee.Initialize()

def sample_points_from_grids(geojson_path, points_per_grid=10, output_path=None):
    """
    Sample points from each grid in the GeoJSON file
    """
    
    if output_path is None:
        output_path = "data/sampled_points.geojson"
    
    print(f"Loading grids from {geojson_path}...")
    
    # Load the grids
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    total_grids = len(data['features'])
    print(f"Total grids to process: {total_grids:,}")
    print(f"Points per grid: {points_per_grid}")
    print(f"Expected total points: ~{total_grids * points_per_grid:,}")
    
    # Get the temp crop mask from WorldCereal
    print("Loading WorldCereal temporary crop data...")
    world_cereal = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    world_cereal_temp_crops = (world_cereal
                              .filterMetadata('product', 'equals', 'temporarycrops')
                              .mosaic())
    temp_crop_mask = world_cereal_temp_crops.select('classification').eq(100)
    
    # Process each grid sequentially
    all_sampled_points = []
    failed_grids = 0
    
    for i, grid_feature in enumerate(data['features']):
        try:
            print(f"Processing grid {i+1}/{total_grids} ({((i+1)/total_grids*100):.1f}%)")
            
            # Create EE geometry from the grid feature
            grid_geom = ee.Geometry(grid_feature['geometry'])
            grid_props = grid_feature['properties']
            
            # Clip temp crop mask to this grid
            grid_temp_crop = temp_crop_mask.clip(grid_geom).selfMask()
            
            # Sample random points within the temp crop areas of this grid
            sampled_points = grid_temp_crop.stratifiedSample(
                numPoints=points_per_grid,
                classBand='classification',
                region=grid_geom,
                scale=10,
                classValues=[1],  # Only sample where temp crop = 1
                classPoints=[points_per_grid],
                seed=42 + i,  # Different seed for each grid
                geometries=True
            )
            
            # Get the sampled points as a list
            points_info = sampled_points.getInfo()
            
            # Add grid information to each sampled point
            for point_feature in points_info['features']:
                point_feature['properties'].update({
                    'grid_id': i,
                    'country': grid_props.get('country'),
                    'continent': grid_props.get('continent'),
                    'tempCropArea': grid_props.get('tempCropArea'),
                    'cellArea': grid_props.get('cellArea')
                })
                
                all_sampled_points.append(point_feature)
            
            actual_points = len(points_info['features'])
            print(f"  Sampled {actual_points} points from grid {i+1}")
            
            # Progress update every 100 grids
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{total_grids} grids processed, {len(all_sampled_points):,} total points sampled")
            
            # Small delay to avoid overwhelming GEE
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ERROR processing grid {i+1}: {e}")
            failed_grids += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {total_grids - failed_grids}/{total_grids} grids")
    print(f"Failed grids: {failed_grids}")
    print(f"Total points sampled: {len(all_sampled_points):,}")
    
    # Create the output GeoJSON
    output_geojson = {
        "type": "FeatureCollection",
        "features": all_sampled_points
    }
    
    # Save the results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_geojson, f, indent=2)
    
    print(f"Saved {len(all_sampled_points):,} sampled points to {output_path}")
    
    return {
        'total_grids': total_grids,
        'failed_grids': failed_grids,
        'total_points': len(all_sampled_points),
        'output_file': output_path
    }

def sample_points_batch(geojson_path, start_idx, end_idx, points_per_grid=10, output_path=None):
    """
    Sample points from a batch of grids (for parallel processing)
    """
    
    print(f"Processing grids {start_idx} to {end_idx}")
    
    # Load the grids
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    # Get the batch of grids
    batch_grids = data['features'][start_idx:end_idx]
    
    print(f"Batch size: {len(batch_grids)} grids")
    
    # Get temp crop mask
    world_cereal = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    world_cereal_temp_crops = (world_cereal
                              .filterMetadata('product', 'equals', 'temporarycrops')
                              .mosaic())
    temp_crop_mask = world_cereal_temp_crops.select('classification').eq(100)
    
    # Process batch
    all_sampled_points = []
    
    for i, grid_feature in enumerate(batch_grids):
        actual_idx = start_idx + i
        try:
            print(f"  Processing grid {actual_idx}")
            
            grid_geom = ee.Geometry(grid_feature['geometry'])
            grid_props = grid_feature['properties']
            
            grid_temp_crop = temp_crop_mask.clip(grid_geom).selfMask()
            
            sampled_points = grid_temp_crop.stratifiedSample(
                numPoints=points_per_grid,
                classBand='classification',
                region=grid_geom,
                scale=10,
                classValues=[1],
                classPoints=[points_per_grid],
                seed=42 + actual_idx,
                geometries=True
            )
            
            points_info = sampled_points.getInfo()
            
            for point_feature in points_info['features']:
                point_feature['properties'].update({
                    'grid_id': actual_idx,
                    'country': grid_props.get('country'),
                    'continent': grid_props.get('continent'),
                    'tempCropArea': grid_props.get('tempCropArea'),
                    'cellArea': grid_props.get('cellArea')
                })
                
                all_sampled_points.append(point_feature)
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"    ERROR processing grid {actual_idx}: {e}")
            continue
    
    # Save batch results
    if output_path is None:
        output_path = f"data/sampled_points_batch_{start_idx}_{end_idx}.geojson"
    
    output_geojson = {
        "type": "FeatureCollection",
        "features": all_sampled_points
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_geojson, f, indent=2)
    
    print(f"Batch complete: {len(all_sampled_points)} points saved to {output_path}")
    
    return len(all_sampled_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample points from grids')
    parser.add_argument('--input', required=True, help='Path to grids GeoJSON')
    parser.add_argument('--points-per-grid', type=int, default=10, help='Points to sample per grid')
    parser.add_argument('--output', help='Output path for sampled points')
    parser.add_argument('--start-idx', type=int, help='Start index for batch processing')
    parser.add_argument('--end-idx', type=int, help='End index for batch processing')
    
    args = parser.parse_args()
    
    try:
        if args.start_idx is not None and args.end_idx is not None:
            # Batch processing mode
            result = sample_points_batch(
                args.input, 
                args.start_idx, 
                args.end_idx, 
                args.points_per_grid,
                args.output
            )
            print(f"Batch processing complete: {result} points sampled")
        else:
            # Full processing mode
            result = sample_points_from_grids(
                args.input, 
                args.points_per_grid,
                args.output
            )
            print(f"Full processing complete: {result['total_points']} points sampled")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        exit(1)

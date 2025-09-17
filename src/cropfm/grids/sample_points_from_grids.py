import ee
import json
import os
import time
from typing import List, Dict, Any

# Initialize Earth Engine
ee.Initialize()

def transform_to_geographic(collection):
    """Transform collection from EPSG:3857 to EPSG:4326 for export"""
    def transform_feature(feature):
        # Transform geometry to EPSG:4326
        transformed_geom = feature.geometry().transform('EPSG:4326', 1)  # 1m error tolerance
        return feature.setGeometry(transformed_geom)
    
    return collection.map(transform_feature)

def create_france_grid_sampling():
    """
    Sample random points from grid cells in France that contain temporary crops.
    Uses batching to avoid GEE collection size limits.
    """
    
    # Configuration parameters
    COUNTRY_NAME = 'France'
    GRID_SCALE = 5000  # 5km grid cells
    SAMPLE_POINTS_PER_GRID = 10  # You can adjust this value
    MIN_TEMP_CROP_AREA_THRESHOLD = 5000000  # 5 km² in square meters
    RANDOM_SEED = 42
    MAX_BATCH_SIZE = 1000  # Process grids in batches to avoid GEE limits
    MAX_POINTS_PER_BATCH = 4000  # Keep under 5000 limit
    GEOJSON_DIR = 'data/'
    
    print(f"Starting grid sampling for {COUNTRY_NAME}...")
    print(f"Sample points per grid: {SAMPLE_POINTS_PER_GRID}")
    print(f"Minimum temp crop area threshold: {MIN_TEMP_CROP_AREA_THRESHOLD} sq meters")
    print(f"Processing in batches of {MAX_BATCH_SIZE} grids")
    
    # Load datasets
    print("Loading datasets...")
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    world_cereal = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    
    # Create country boundary and grid
    print(f"Creating grid for {COUNTRY_NAME}...")
    country = countries.filter(ee.Filter.eq('country_na', COUNTRY_NAME)).geometry()
    
    # Create grid in EPSG:3857 for accurate area calculations
    grid = country.coveringGrid(ee.Projection('EPSG:3857').atScale(GRID_SCALE))
    
    # Process temporary crop areas (keep in EPSG:3857 for area calculations)
    print("Processing temporary crop areas...")
    world_cereal_temp_crops = (world_cereal
                              .filterMetadata('product', 'equals', 'temporarycrops')
                              .mosaic())
    temp_crop_mask = world_cereal_temp_crops.select('classification').eq(100)
    
    def process_grid_cell(grid_cell):
        temp_crop_area = ee.Number(
            temp_crop_mask
            .multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=grid_cell.geometry(),
                scale=10,
                maxPixels=1e9 
            ).get('classification')
        )
        
        cell_area = grid_cell.geometry().area(maxError=1)
        
        return grid_cell.set({
            'hasTempCrop': ee.Algorithms.If(temp_crop_area.gt(MIN_TEMP_CROP_AREA_THRESHOLD), 1, 0),
            'tempCropArea': temp_crop_area,
            'cellArea': cell_area
        })
    
    processed_grid = grid.map(process_grid_cell)
    grids_with_temp_crop = processed_grid.filter(ee.Filter.eq('hasTempCrop', 1))
    
    # Step 1: Export grid information (transform to EPSG:4326 first)
    print("Exporting grid information...")
    grid_info_filename = f"france_grids_info_{COUNTRY_NAME.lower()}.geojson"
    
    # Transform grids to geographic coordinates before export
    grids_geographic = transform_to_geographic(grids_with_temp_crop)
    
    # Export grids in smaller batches to get the actual count
    try:
        # Try to get a small sample first to estimate total
        sample_grids = grids_geographic.limit(100)
        sample_info = sample_grids.getInfo()
        
        if len(sample_info['features']) == 100:
            print("More than 100 qualifying grids found. Processing in batches...")
            # Export all grids to file first
            export_grids_to_file(grids_geographic, f'{GEOJSON_DIR}/{grid_info_filename}')
            
            # Read back the file to get actual count
            with open(f'{GEOJSON_DIR}/{grid_info_filename}', 'r') as f:
                grid_data = json.load(f)
                actual_grid_count = len(grid_data['features'])
        else:
            actual_grid_count = len(sample_info['features'])
            # Save the sample if it's all we have
            with open(f'{GEOJSON_DIR}/{grid_info_filename}', 'w') as f:
                json.dump(sample_info, f, indent=2)
            
    except Exception as e:
        print(f"Error getting grid count: {e}")
        print("Using estimated count...")
        actual_grid_count = 500  # Conservative estimate
    
    print(f"Actual number of qualifying grids: {actual_grid_count}")
    
    if actual_grid_count == 0:
        print("No grids found with temporary crops above the threshold.")
        return None
    
    # Step 2: Sample points in batches (keep in original projection for sampling)
    print(f"Sampling points from {actual_grid_count} grids in batches...")
    
    # Calculate batch size based on points per grid to stay under 5000 limit
    points_per_batch = min(MAX_POINTS_PER_BATCH, MAX_BATCH_SIZE * SAMPLE_POINTS_PER_GRID)
    grids_per_batch = min(MAX_BATCH_SIZE, points_per_batch // SAMPLE_POINTS_PER_GRID)
    
    print(f"Processing {grids_per_batch} grids per batch ({grids_per_batch * SAMPLE_POINTS_PER_GRID} points per batch)")
    
    all_sample_points = []
    total_processed_grids = 0
    batch_num = 0
    
    while total_processed_grids < actual_grid_count:
        batch_num += 1
        start_idx = total_processed_grids
        end_idx = min(total_processed_grids + grids_per_batch, actual_grid_count)
        
        print(f"Processing batch {batch_num}: grids {start_idx+1} to {end_idx}")
        
        # Get batch of grids (keep in EPSG:3857 for sampling)
        batch_grids = grids_with_temp_crop.toList(grids_per_batch, start_idx)
        batch_collection = ee.FeatureCollection(batch_grids)
        
        # Sample points from this batch
        batch_points = sample_points_from_grids(batch_collection, temp_crop_mask, SAMPLE_POINTS_PER_GRID, RANDOM_SEED)
        
        # Transform sample points to EPSG:4326 before export
        batch_points_geographic = transform_to_geographic(batch_points)
        
        try:
            # Get the points for this batch
            batch_points_data = batch_points_geographic.getInfo()
            if batch_points_data and 'features' in batch_points_data:
                all_sample_points.extend(batch_points_data['features'])
                print(f"  Added {len(batch_points_data['features'])} points from batch {batch_num}")
            else:
                print(f"  No points returned from batch {batch_num}")
                
        except Exception as e:
            print(f"  Error processing batch {batch_num}: {e}")
            # Continue with next batch
            
        total_processed_grids = end_idx
        
        # Add small delay to avoid rate limits
        time.sleep(1)
    
    total_points = len(all_sample_points)
    print(f"Total sample points generated: {total_points}")
    
    # Step 3: Save all points to GeoJSON (already in EPSG:4326)
    if total_points > 0:
        print("Saving sample points to GeoJSON...")
        
        # Create proper GeoJSON structure
        sample_points_geojson = {
            "type": "FeatureCollection",
            "features": all_sample_points
        }
        
        output_filename = f"france_crop_sample_points_{SAMPLE_POINTS_PER_GRID}per_grid_{actual_grid_count}grids_{total_points}points_wgs84.geojson"
        output_path = os.path.join(os.getcwd(), output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(sample_points_geojson, f, indent=2)
        
        print(f"Sample points saved to: {output_path}")
        
        # Print summary statistics
        print("\n=== SUMMARY ===")
        print(f"Country: {COUNTRY_NAME}")
        print(f"Grid scale: {GRID_SCALE}m")
        print(f"Minimum temp crop area threshold: {MIN_TEMP_CROP_AREA_THRESHOLD} sq meters")
        print(f"Qualifying grid cells: {actual_grid_count}")
        print(f"Sample points per grid: {SAMPLE_POINTS_PER_GRID}")
        print(f"Total sample points: {total_points}")
        print(f"Average points per grid: {total_points/actual_grid_count:.1f}")
        print(f"Sample points file: {output_filename} (EPSG:4326)")
        print(f"Grid info file: {grid_info_filename} (EPSG:4326)")
        
        return {
            'grid_count': actual_grid_count,
            'total_points': total_points,
            'output_file': output_path,
            'grid_info_file': grid_info_filename
        }
    else:
        print("No sample points were generated!")
        return None

def export_grids_to_file(grids_collection, filename):
    """Export grids to file in batches to handle large collections"""
    print(f"Exporting grids to {filename}...")
    
    # Try to export in batches
    batch_size = 1000
    all_features = []
    start = 0
    tmp_count = 0
    
    while True:
        try:
            batch = grids_collection.toList(batch_size, start)
            batch_collection = ee.FeatureCollection(batch)
            batch_data = batch_collection.getInfo()
            
            if not batch_data['features']:
                break
                
            all_features.extend(batch_data['features'])
            print(f"  Exported {len(batch_data['features'])} grids (total: {len(all_features)})")
            
            if len(batch_data['features']) < batch_size:
                break
                
            start += batch_size
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  Error exporting batch starting at {start}: {e}")
            break
        tmp_count += 1
        if tmp_count > 1:
            break # just for testing purposes
    
    # Save to file
    grid_geojson = {
        "type": "FeatureCollection", 
        "features": all_features
    }
    
    with open(f'{filename}', 'w') as f:
        json.dump(grid_geojson, f, indent=2)
    
    print(f"Saved {len(all_features)} grids to {filename}")

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
                'cell_area': grid_cell.get('cellArea')
            })
        
        return random_points.map(add_grid_info)
    
    # Apply sampling to all grids and flatten the results
    return grids_collection.map(sample_points_from_grid).flatten()

if __name__ == "__main__":
    try:
        results = create_france_grid_sampling()
        
        if results:
            print(f"\nSuccessfully completed sampling!")
            print(f"Generated {results['total_points']} points from {results['grid_count']} grids")
            print(f"Files saved in EPSG:4326 (WGS84):")
            print(f"   - Sample points: {os.path.basename(results['output_file'])}")
            print(f"   - Grid info: {results['grid_info_file']}")
        else:
            print("Sampling failed or no points generated")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Make sure you have authenticated with Google Earth Engine")
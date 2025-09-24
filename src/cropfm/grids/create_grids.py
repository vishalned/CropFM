'''
This script creates grids for a specific country.
'''

import ee
import json
import os
import sys
import time
import argparse
from typing import List, Dict, Any
from cropfm.utils.country_continent_mapping import get_continent

# Initialize Earth Engine
ee.Initialize()

def transform_to_geographic(collection):
    """Transform collection from EPSG:3857 to EPSG:4326 for export"""
    def transform_feature(feature):
        # Transform geometry to EPSG:4326
        transformed_geom = feature.geometry().transform('EPSG:4326', 1)  # 1m error tolerance
        return feature.setGeometry(transformed_geom)
    
    return collection.map(transform_feature)

def create_grid_for_country(country_name):
    """
    Create grid for a specific country with country and continent information
    """
    
    # Configuration parameters
    GRID_SCALE = 5000  # 5km grid cells
    MIN_TEMP_CROP_AREA_THRESHOLD = 5000000  # 5 km² in square meters
    GEOJSON_DIR = 'data/country_grids/'
    
    # Create output directory if it doesn't exist
    os.makedirs(GEOJSON_DIR, exist_ok=True)
    
    print(f"Starting grid creation for {country_name}...")
    print(f"Grid scale: {GRID_SCALE}m")
    print(f"Minimum temp crop area threshold: {MIN_TEMP_CROP_AREA_THRESHOLD} sq meters")

    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    world_cereal = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    
    country_geom = countries.filter(ee.Filter.eq('country_na', country_name)).geometry()
    
    # Create grid in EPSG:3857 for accurate area calculations
    print("Creating grids...")
    grid = country_geom.coveringGrid(ee.Projection('EPSG:3857').atScale(GRID_SCALE))
    
    # Process temporary crop areas (keep in EPSG:3857 for area calculations)
    world_cereal_temp_crops = (world_cereal
                              .filterMetadata('product', 'equals', 'temporarycrops')
                              .mosaic())
    temp_crop_mask = world_cereal_temp_crops.select('classification').eq(100)
    
    # Get continent info
    continent = get_continent(country_name)
    print(f"Country: {country_name}, Continent: {continent}")
    
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
            'country': country_name,
            'continent': continent,
            'hasTempCrop': ee.Algorithms.If(temp_crop_area.gt(MIN_TEMP_CROP_AREA_THRESHOLD), 1, 0),
            'tempCropArea': temp_crop_area,
            'cellArea': cell_area
        })
    
    processed_grid = grid.map(process_grid_cell)
    grids_with_temp_crop = processed_grid.filter(ee.Filter.eq('hasTempCrop', 1))
    
    # Transform to geographic coordinates before export
    grids_geographic = transform_to_geographic(grids_with_temp_crop)
    
    # Export grid information
    # Clean country name for filename
    clean_country_name = country_name.lower().replace(' ', '_').replace('&', 'and').replace(',', '').replace('\'', '').replace('.', '')
    grid_info_filename = f"{clean_country_name}_grids_info.geojson"
    output_path = os.path.join(GEOJSON_DIR, grid_info_filename)
    
    try:
        # Export grids to file
        export_grids_to_file(grids_geographic, output_path, country_name)
        
        # Read back to get count
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                grid_data = json.load(f)
                actual_grid_count = len(grid_data['features'])
        else:
            actual_grid_count = 0
            
    except Exception as e:
        print(f"Error processing {country_name}: {e}")
        return None
    
    print(f"Successfully processed {country_name}: {actual_grid_count} qualifying grids")
    
    return {
        'country': country_name,
        'continent': continent,
        'grid_count': actual_grid_count,
        'output_file': output_path
    }

def export_grids_to_file(grids_collection, filename, country_name):
    """Export grids to file in batches to handle large collections"""
    print(f"Exporting grids for {country_name} to {filename}...")
    
    # Try to export in smaller batches for individual countries
    batch_size = 1000  # Smaller batch size for individual countries
    all_features = []
    start = 0
    
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
            # Try smaller batch size
            if batch_size > 100:
                batch_size = 100
                print(f"  Retrying with smaller batch size: {batch_size}")
                continue
            else:
                break
    
    # Save to file
    grid_geojson = {
        "type": "FeatureCollection", 
        "features": all_features
    }
    
    with open(filename, 'w') as f:
        json.dump(grid_geojson, f, indent=2)
    
    print(f"Saved {len(all_features)} grids to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create grids for a specific country')
    parser.add_argument('--country', required=True, help='Country name to process')
    
    args = parser.parse_args()
    
    try:
        result = create_grid_for_country(args.country)
        
        if result:
            print(f"\nSuccessfully completed grid creation for {result['country']}!")
            print(f"Continent: {result['continent']}")
            print(f"Generated {result['grid_count']} qualifying grids")
            print(f"Output file: {result['output_file']}")
        else:
            print(f"Grid creation failed for {args.country}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)
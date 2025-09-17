"""
Generate a list of all countries from the GEE dataset and save it to a file.

"""
import ee
import json

# Initialize Earth Engine
ee.Initialize()

def get_all_countries():
    """Get list of all countries from GEE dataset"""
    print("Fetching country list from Google Earth Engine...")
    
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    
    # Get unique country names
    country_names = countries.aggregate_array('country_na').distinct().sort()
    country_list = country_names.getInfo()
    
    print(f"Found {len(country_list)} countries")
    
    # Save to file
    output_file = 'data/country_list.json'
    with open(output_file, 'w') as f:
        json.dump(country_list, f, indent=2)
    
    print(f"Country list saved to {output_file}")
    
    # Also save as text file for easy reading
    txt_file = 'data/country_list.txt'
    with open(txt_file, 'w') as f:
        for country in country_list:
            f.write(f"{country}\n")
    
    print(f"Country list also saved to {txt_file}")
    
    return country_list

if __name__ == "__main__":
    countries = get_all_countries()
    print("\nFirst 10 countries:")
    for i, country in enumerate(countries[:10]):
        print(f"{i+1}. {country}")
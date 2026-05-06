import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop"
    output_dir = os.path.join(base_dir, "03_Generation_Flux")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    basins_path = os.path.join(base_dir, "02_Basin_Masking", "02_basins_v1_mask.shp")
    points_path = os.path.join(base_dir, "raw_data", "Flux Hydro level 12 point center.shp")
    
    basins = gpd.read_file(basins_path)
    points = gpd.read_file(points_path)
    
    original_count = len(basins)
    assert original_count == 10226, f"Error: Initial basins should be 10226, got {original_count}"
    
    if basins.crs != points.crs:
        print("Reprojecting points to match basins...")
        points = points.to_crs(basins.crs)

    # In V4, we use mask_v1
    mask_col = 'mask_v1'
    
    print("Filtering geometries for valid ones to speed up spatial join...")
    valid_basins = basins[basins[mask_col] == 0].copy()
    
    # Identify points within valid basins
    print("Spatial Joining points to valid basins...")
    # It's critical to only extract subbasn to prevent dataframe bloat
    joined = gpd.sjoin(points, valid_basins[['subbasn', 'geometry']], how='inner', predicate='within')
    print(f"Joined {len(joined)} generation flux points to valid basins.")
    
    # Determine the flux column in points
    flux_col = 'Flux_Linea'
    if flux_col not in joined.columns:
        if 'withiflux' in joined.columns:
             flux_col = 'withiflux'
        else:
             print("Error: Could not find flux column!")
             return

    # Aggregate by strict integer ID
    print(f"Aggregating {flux_col} by subbasn...")
    aggregated = joined.groupby('subbasn')[flux_col].sum().reset_index()
    aggregated.rename(columns={flux_col: 'gen_flux'}, inplace=True)
    
    print("Merging metrics back onto exactly 10,226 original geometries...")
    # Ensure no duplicates by using 1:1 join
    result = basins.merge(aggregated, on='subbasn', how='left')
    result['gen_flux'] = result['gen_flux'].fillna(0)
    
    print(f"Final verify Row Count: {len(result)} (Should be exactly {original_count})")
    
    # Output 1: Shapefile
    out_shp = os.path.join(output_dir, "03_generation_flux_basins.shp")
    print(f"Saving Shapefile to {out_shp}...")
    result.to_file(out_shp)
    
    # Output 2: CSV Data
    out_csv = os.path.join(output_dir, "03_generation_flux_basins.csv")
    print(f"Saving CSV to {out_csv}...")
    result.drop(columns='geometry').to_csv(out_csv, index=False)
    
    print("Step 3 Complete.")

if __name__ == "__main__":
    main()

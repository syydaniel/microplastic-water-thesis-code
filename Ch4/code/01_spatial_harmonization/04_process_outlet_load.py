import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop"
    output_dir = os.path.join(base_dir, "04_Outlet_Flux")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    basins_path = os.path.join(base_dir, "03_Generation_Flux", "03_generation_flux_basins.shp")
    outlet_path = os.path.join(base_dir, "raw_data", "merge_flux_outlet.shp")
    
    basins = gpd.read_file(basins_path)
    outlets = gpd.read_file(outlet_path)
    
    original_count = len(basins)
    assert original_count == 10226, f"Error: Initial basins should be 10226, got {original_count}"

    print("Formatting and aggregating Outlet Fluxes by strict ID...")
    
    # Process outlet point properties
    if 'sum_flux' not in outlets.columns:
        print("Warning: sum_flux column not found in outlet points!")
        return
        
    # Aggregate any potential duplicate points by subbasin just in case, sum them
    aggregated = outlets.groupby('subbasin')['sum_flux'].sum().reset_index()
    aggregated.rename(columns={'sum_flux': 'out_flux', 'subbasin': 'subbasn'}, inplace=True)
    
    print("Merging metrics back into Valid Basin geometries (1:1 join)...")
    result = basins.merge(aggregated, on='subbasn', how='left')
    result['out_flux'] = result['out_flux'].fillna(0)
    
    print(f"Final verify Row Count: {len(result)} (Should be exactly {original_count})")
    
    # Output 1: Save Shapefile
    out_shp = os.path.join(output_dir, "04_outlet_flux_basins.shp")
    print(f"Saving Shapefile to {out_shp}...")
    result.to_file(out_shp)
    
    # Output 2: Save CSV
    out_csv = os.path.join(output_dir, "04_outlet_flux_basins.csv")
    print(f"Saving CSV to {out_csv}...")
    result.drop(columns='geometry').to_csv(out_csv, index=False)
    
    print("Step 4 Complete.")

if __name__ == "__main__":
    main()

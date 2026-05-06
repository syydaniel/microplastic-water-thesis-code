import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop"
    output_dir = os.path.join(base_dir, "02_Basin_Masking")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    basins_shp = os.path.join(base_dir, "raw_data", "basins_joined.shp")
    grid_shp = os.path.join(base_dir, "01_Base_Grid", "01_base_grid_05deg.shp")
    flux_csv = os.path.join(base_dir, "raw_data", "Flux Hydro level 12 point center.shp") # Actually it's an SHP
    
    basins = gpd.read_file(basins_shp)
    grid = gpd.read_file(grid_shp)
    points_df = gpd.read_file(flux_csv)
    
    # We must preserve exactly this row count (10226)
    original_count = len(basins)
    print(f"Original Basin Count: {original_count}")
    
    # We use 'subbasn' as the strictly unique integer ID
    assert basins['subbasn'].nunique() == original_count, "Error: subbasn is not strictly unique!"
    
    if basins.crs != grid.crs:
        print("Reprojecting grid to match basins...")
        grid = grid.to_crs(basins.crs)

    print("Identifying active grid cells (cells with flux points)...")
    if points_df.crs != grid.crs:
        points_df = points_df.to_crs(grid.crs)
    intersected_grids = gpd.sjoin(grid[['grid_id', 'geometry']], points_df[['geometry']], how='inner', predicate='intersects')
    active_grids = intersected_grids['grid_id'].unique()
    grid['has_data'] = grid['grid_id'].isin(active_grids)
    
    # Extract only necessary columns to keep memory low during intersection
    b_light = basins[['subbasn', 'geometry']].copy()
    g_light = grid[['grid_id', 'has_data', 'geometry']].copy()
    
    print("Performing Spatial Overlay (Intersection)...")
    # Intersection explodes the rows based on overlaps
    overlay = gpd.overlay(b_light, g_light, how='intersection')
    
    print("Calculating relative areas of intersection...")
    overlay['part_area'] = overlay.geometry.area
    
    print("Aggregating coverage by strict ID (subbasn)...")
    # For each subbasin, calculate total intersected area and area with valid data
    agg = overlay.groupby('subbasn').agg(
        total_intersect_area=('part_area', 'sum'),
        total_valid_area=('part_area', lambda x: x[overlay.loc[x.index, 'has_data']].sum()),
        grid_cells_intersected=('grid_id', 'nunique')
    ).reset_index()
    
    # Calculate covered percentage
    agg['data_cov'] = agg['total_valid_area'] / agg['total_intersect_area']
    agg['data_cov'] = agg['data_cov'].fillna(0)
    
    # Rule: Keep basins with >= 95% coverage
    agg['mask_v1'] = np.where(agg['data_cov'] < 0.95, 1, 0)
    
    print("Merging metrics back onto exactly 10,226 original geometries...")
    # This guarantees no duplication, because we merge a 10226-row DDM30 frame with grouped unique stats
    result = basins.merge(agg[['subbasn', 'data_cov', 'mask_v1', 'grid_cells_intersected']], on='subbasn', how='left')
    
    # Any basin that didn't even intersect the grid gets masked
    result['mask_v1'] = result['mask_v1'].fillna(1).astype(int)
    result['data_cov'] = result['data_cov'].fillna(0)
    result['grid_cells_intersected'] = result['grid_cells_intersected'].fillna(0)
    
    print(f"Final Row Count Verification: {len(result)} (Should be exactly {original_count})")
    
    masked_count = result[result['mask_v1'] == 1].shape[0]
    valid_count = result[result['mask_v1'] == 0].shape[0]
    print(f"Masked (Insufficient Data): {masked_count}")
    print(f"Valid (>= 95% Coverage): {valid_count}")
    
    out_shp = os.path.join(output_dir, "02_basins_v1_mask.shp")
    print(f"Saving explicitly sized 10,226 row Shapefile to {out_shp}...")
    result.to_file(out_shp)
    
    out_csv = os.path.join(output_dir, "02_basins_v1_mask.csv")
    result.drop(columns='geometry').to_csv(out_csv, index=False)
    
    print("Step 2 Complete.")

if __name__ == "__main__":
    main()

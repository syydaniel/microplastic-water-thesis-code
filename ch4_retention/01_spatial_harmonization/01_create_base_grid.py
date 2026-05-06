import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import os

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop"
    output_dir = os.path.join(base_dir, "01_Base_Grid")
    
    # Create Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("--- Creating Standalone Grid Level Shapefile (0.5 deg) ---")
    
    # Define Grid Parameters
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -60, 90
    res = 0.5
    half_res = res / 2
    
    # Create Grid Centers
    lons = np.arange(lon_min + half_res, lon_max, res)
    lats = np.arange(lat_min + half_res, lat_max, res)
    xx, yy = np.meshgrid(lons, lats)
    
    print(f"Generating {xx.size} grid cells (-180 to 180, -60 to 90)...")
    
    # Using numpy vectorization for faster box creation
    x_coords = xx.flatten()
    y_coords = yy.flatten()
    
    geometries = [box(x - half_res, y - half_res, x + half_res, y + half_res) 
                  for x, y in zip(x_coords, y_coords)]
                  
    grid_cells = gpd.GeoDataFrame(
        geometry=geometries, 
        crs="EPSG:4326"
    )
    
    # Add a unique Grid ID
    grid_cells['grid_id'] = grid_cells.index
    print(f"Total Grid Cells Created: {len(grid_cells)}")
    
    # Output 1: Shapefile
    out_shp_path = os.path.join(output_dir, "01_base_grid_05deg.shp")
    print(f"Saving Shapefile to {out_shp_path}...")
    grid_cells.to_file(out_shp_path)
    
    # Output 2: CSV Data
    out_csv_path = os.path.join(output_dir, "01_base_grid_05deg.csv")
    print(f"Saving Data CSV to {out_csv_path}...")
    # Add centroid columns for CSV export reference
    df_export = pd.DataFrame(grid_cells.drop(columns='geometry'))
    df_export['centroid_lon'] = x_coords
    df_export['centroid_lat'] = y_coords
    df_export.to_csv(out_csv_path, index=False)
    
    # Output 3: Visual Map (Just the outline/sample to prove it's there)
    out_png_path = os.path.join(output_dir, "01_map_base_grid.png")
    print(f"Generating Verification Map to {out_png_path}...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # We'll just plot a very faint black edge to show the mesh, it will be dense globally.
    grid_cells.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.01, alpha=0.5)
    ax.set_title("Generated 0.5-Degree Global Base Grid")
    ax.set_ylim([-60, 90])
    ax.set_xlim([-180, 180])
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()
    
    print("Step 1 Complete.")

if __name__ == "__main__":
    main()

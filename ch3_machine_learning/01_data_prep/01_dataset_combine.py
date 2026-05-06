import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os

# Set file paths
project_dir = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
result_dir = os.path.join(project_dir, "01_Data_Prep", "01_Data_Combine_Result")
os.makedirs(result_dir, exist_ok=True)

clean_data_path = os.path.join(project_dir, "Raw data", "Clean_Data.csv")
hb12_shp_path = os.path.join(project_dir, "Raw data", "HB12", "BasinATLAS_v10_lev12.shp")

hb12_output_path = os.path.join(result_dir, "HB12_combine.shp")
csv_output_path = os.path.join(result_dir, "data_combine.csv")

print(f"Results directory: {result_dir}")


# 1. Process HydroBASINS Level 12 (Chunked to avoid MemoryError)
print("Processing HydroBASINS Shapefile in chunks...")

# Define Transformations
transformations = {
    'dis_m3_pyr': ('Natural Discharge Upstream', 1.0),
    'run_mm_syr': ('Land Surface Runoff Local', 1.0),
    'lkv_mc_usu': ('Lake Volume Upstream', 1e6),
    'rev_mc_usu': ('Reservoir Volume Upstream', 1e6),
    'ria_ha_ssu': ('River Area Local', 10000.0),
    'ria_ha_usu': ('River Area Upstream', 10000.0),
    'riv_tc_ssu': ('River Volume Local', 1000.0),
    'riv_tc_usu': ('River Volume Upstream', 1000.0),
    'ele_mt_sav': ('Elevation Local', 1.0),
    'ele_mt_uav': ('Elevation Upstream', 1.0),
    'slp_dg_sav': ('Terrain Slope Local', 0.1),
    'slp_dg_uav': ('Terrain Slope Upstream', 0.1),
    'sgr_dk_sav': ('Stream Gradient Local', 0.1),
    'tmp_dc_syr': ('Temperature Local', 0.1),
    'tmp_dc_uyr': ('Temperature Upstream', 0.1),
    'pre_mm_syr': ('Precipitation Local', 1.0),
    'pre_mm_uyr': ('Precipitation Upstream', 1.0),
    'pet_mm_syr': ('Potential Evap Local', 1.0),
    'pet_mm_uyr': ('Potential Evap Upstream', 1.0),
    'aet_mm_syr': ('Actual Evap Local', 1.0),
    'aet_mm_uyr': ('Actual Evap Upstream', 1.0),
    'crp_pc_sse': ('Cropland Extent Local', 1.0),
    'crp_pc_use': ('Cropland Extent Upstream', 1.0),
    'pst_pc_sse': ('Pasture Extent Local', 1.0),
    'pst_pc_use': ('Pasture Extent Upstream', 1.0),
    'wet_pc_sg1': ('Wetland All Local', 1.0),
    'wet_pc_ug1': ('Wetland All Upstream', 1.0),
    'wet_pc_sg2': ('Wetland Land Local', 1.0),
    'wet_pc_ug2': ('Wetland Land Upstream', 1.0),
    'pop_ct_ssu': ('Population Local', 1000.0),
    'pop_ct_usu': ('Population Upstream', 1000.0),
    'urb_pc_sse': ('Urban Extent Local', 1.0),
    'urb_pc_use': ('Urban Extent Upstream', 1.0),
    'rdd_mk_sav': ('Road Density Local', 1.0),
    'rdd_mk_uav': ('Road Density Upstream', 1.0),
    'hft_ix_s09': ('Human Footprint Local', 0.1),
    'hft_ix_u09': ('Human Footprint Upstream', 0.1),
    'hdi_ix_sav': ('Human Dev Index Local', 0.001),
}

chunk_size = 50000
processed_chunks = []

start = 0
while True:
    try:
        # Use slice to read chunk
        gdf_chunk = gpd.read_file(hb12_shp_path, rows=slice(start, start + chunk_size))
    except Exception as e:
        # End of file or error
        print(f"Error reading chunk: {e}")
        break
        
    if gdf_chunk.empty:
        break
    
    print(f"Processing chunk {start} - {start + len(gdf_chunk)}...")

    processed_cols = ['HYBAS_ID', 'geometry']
    for orig_col, (new_name, scale) in transformations.items():
        if orig_col in gdf_chunk.columns:
            series = pd.to_numeric(gdf_chunk[orig_col], errors='coerce')
            series = series.replace(-9999, np.nan)
            gdf_chunk[new_name] = series * scale
            processed_cols.append(new_name)
    
    # Keep only processed columns to free memory
    gdf_slim = gdf_chunk[processed_cols].copy()
    processed_chunks.append(gdf_slim)
    
    rows_read = len(gdf_chunk)
    start += chunk_size
    del gdf_chunk
    
    if rows_read < chunk_size:
        break

print("Concatenating chunks...")
gdf_hb12_slim = pd.concat(processed_chunks, ignore_index=True)

# Save Slim Shapefile
print(f"Saving processed HB12 to {hb12_output_path}...")
gdf_hb12_slim.to_file(hb12_output_path)
print("Shapefile Processed.")


# 2. Load & Clean Main Data
try:
    df_clean = pd.read_csv(clean_data_path, encoding='latin-1')
except UnicodeDecodeError:
    df_clean = pd.read_csv(clean_data_path, encoding='cp1252')

# Standardize Missing Values to NaN (covers 'NR', 'NA', empty strings)
df_clean.replace(['NR', 'NA', 'nan', '', -9999, '-9999',-99.9, '-99.9'], np.nan, inplace=True)

# Filter: Remove Std_Value_m3 == 0 (and NaNs if any)
df_clean = df_clean[df_clean['Std_Value_m3'] != 0].dropna(subset=['Std_Value_m3']).copy()

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df_clean['Longitude'], df_clean['Latitude'])]
gdf_clean = gpd.GeoDataFrame(df_clean, geometry=geometry)
gdf_clean.set_crs(epsg=4326, inplace=True)

print(f"Cleaned Data: {gdf_clean.shape[0]} rows")

# 3. Spatial Join with Processed HB12
# Note: gdf_hb12_slim is already in memory from Step 1
if gdf_hb12_slim.crs != gdf_clean.crs:
    gdf_hb12_slim = gdf_hb12_slim.to_crs(gdf_clean.crs)

print("Joining Clean Data with Processed HB12...")
gdf_joined = gpd.sjoin(gdf_clean, gdf_hb12_slim, how="left", predicate="intersects")

print(f"Joined Rows: {gdf_joined.shape[0]}")

# 4. Final Selection & Save CSV
# Select ID, target, gear, and environmental variables (which are now properly named in HB12_Slim)
required_cols = ['Unique_ID', 'Std_Value_m3', 'Gear_Category']
env_cols = [t[0] for t in transformations.values()] + ['HYBAS_ID']

# Intersection to be safe (though join should have them)
final_cols = [c for c in required_cols + env_cols if c in gdf_joined.columns]

df_export = gdf_joined[final_cols].copy()

df_export.to_csv(csv_output_path, index=False)
print(f"CSV Saved to {csv_output_path}")
df_export.head()


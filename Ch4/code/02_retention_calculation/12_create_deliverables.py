import pandas as pd
import geopandas as gpd
import os

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop/10_Analysis_Report"
    
    print("Loading comprehensive Final Deliverables...")
    csv_path = os.path.join(base_dir, "10_V4_Analysis_Data_with_Discharge.csv")
    shp_path = os.path.join(r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop", "06_Imputed_Retention", "06_V4_Final_Retention_Basins.shp")
    
    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(shp_path)
    
    # Calculate final delivery ratio (inverse of retention)
    df['ratio_final'] = 1.0 - df['ret_final']
    
    # Define essential columns
    # We want to drop the dozens of intermediate DDM30 metadata columns and just leave the absolute core
    # subbasn (ID), area related, discharge, fluxes, ratio, and method.
    core_cols_csv = [
        'subbasn', 'area', 'ar_sqkm', 'grid_cells', 
        'mean_discharge_cms', 'gen_flux', 'out_flux', 
        'ratio_final', 'method'
    ]
    
    # For shapefile, we must ensure names don't exceed 10 chars, though these are mostly safe.
    # We will rename mean_discharge_cms to dis_cms to be safe for ESRI shapefile format
    core_cols_shp = core_cols_csv + ['geometry']
    
    print("Filtering down to essential columns...")
    df_clear = df[core_cols_csv].round(5) # keep numbers clean
    
    # Merge the discharge and ratio back to the gdf since they were computed later/adjusted
    gdf = gdf.merge(df[['subbasn', 'mean_discharge_cms', 'ratio_final']], on='subbasn', how='left')
    gdf_clear = gdf[core_cols_shp].copy()
    
    # Rename long columns for shapefile limits
    gdf_clear.rename(columns={'mean_discharge_cms': 'dis_cms', 'grid_cells': 'grd_cells'}, inplace=True)
    
    # Save clear versions
    out_csv = os.path.join(base_dir, "V4_Final_Ratio_Clear_Version.csv")
    out_shp = os.path.join(base_dir, "V4_Final_Ratio_Clear_Version.shp")
    
    print(f"Exporting Clear CSV to {out_csv}...")
    df_clear.to_csv(out_csv, index=False)
    
    print(f"Exporting Clear Shapefile to {out_shp}...")
    gdf_clear.to_file(out_shp)
    
    print("Done! Delivered clean versions.")

if __name__ == "__main__":
    main()

import geopandas as gpd
import pandas as pd
import numpy as np
import os

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop"
    output_dir = os.path.join(base_dir, "05_Retention_Ratios")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    basins_shp = os.path.join(base_dir, "02_Basin_Masking", "02_basins_v1_mask.shp")
    gen_csv = os.path.join(base_dir, "03_Generation_Flux", "03_generation_flux_basins.csv")
    out_csv = os.path.join(base_dir, "04_Outlet_Flux", "04_outlet_flux_basins.csv")
    
    basins = gpd.read_file(basins_shp)
    gen_df = pd.read_csv(gen_csv)
    out_df = pd.read_csv(out_csv)
    
    original_count = len(basins)
    assert original_count == 10226, f"Error: Initial basins should be 10226, got {original_count}"
    
    # Merge using strictly subbasn
    cols_to_keep = ['subbasn', 'mask_v1', 'grid_cells', 'geometry']
    # The shapefile writer truncated 'grid_cells_intersected' to 'grid_cells'
    
    print("Merging metrics...")
    merged_df = basins.merge(gen_df[['subbasn', 'gen_flux']], on='subbasn', how='left')
    merged_df = merged_df.merge(out_df[['subbasn', 'out_flux']], on='subbasn', how='left')
    
    merged_df['gen_flux'] = merged_df['gen_flux'].fillna(0)
    merged_df['out_flux'] = merged_df['out_flux'].fillna(0)
    
    print("Calculating Ratios...")
    def calculate_ratio(row):
        if row['gen_flux'] <= 0:
            return np.nan
        return row['out_flux'] / row['gen_flux']
        
    merged_df['ratio'] = merged_df.apply(calculate_ratio, axis=1)
    
    # Raw retention (uncapped mathematical result)
    merged_df['retention'] = 1.0 - merged_df['ratio']
    
    # Secondary Mask: Mask out basins where out_flux == 0 or gen_flux == 0
    # Also mask out basins where out >= gen (retention <= 0) because these physical anomalies
    # are better suited for size-based imputation (like the 0.75 for small coastal streams)
    merged_df['mask_v2'] = np.where((merged_df['out_flux'] == 0) | 
                                    (merged_df['gen_flux'] == 0) | 
                                    (merged_df['out_flux'] >= merged_df['gen_flux']), 1, 0)
    merged_df['mask_final'] = np.where((merged_df['mask_v1'] == 1) | (merged_df['mask_v2'] == 1), 1, 0)

    # For valid physical retention calculations, clip them nicely to 0
    # If they are masked, set them to NaN for now
    merged_df['ret_valid'] = np.where(merged_df['mask_final'] == 0, np.clip(merged_df['retention'], 0.0, 1.0), np.nan)
    
    print(f"Final Count Verification: {len(merged_df)}")
    print(f"Calculated Valid Physical Retentions: {(merged_df['mask_final'] == 0).sum()}")
    print(f"Masked (Need Imputation): {(merged_df['mask_final'] == 1).sum()}")
    
    # Output 1: Save final Shapefile
    out_shp = os.path.join(output_dir, "05_final_retention_basins.shp")
    print(f"Saving Shapefile to {out_shp}...")
    merged_df.to_file(out_shp)
    
    # Output 2: Save final CSV
    out_csv = os.path.join(output_dir, "05_final_retention.csv")
    print(f"Saving CSV to {out_csv}...")
    merged_df.drop(columns='geometry').to_csv(out_csv, index=False)
    
    print("Step 5 Complete.")

if __name__ == "__main__":
    main()

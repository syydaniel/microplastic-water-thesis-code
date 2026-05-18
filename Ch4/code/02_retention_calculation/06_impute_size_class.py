import geopandas as gpd
import pandas as pd
import numpy as np
import os

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Re_Analysis_V6_Develop"
    output_dir = os.path.join(base_dir, "06_Imputed_Retention")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading calculated retention datasets...")
    retention_shp = os.path.join(base_dir, "05_Retention_Ratios", "05_final_retention_basins.shp")
    retention_db = gpd.read_file(retention_shp)
    
    original_count = len(retention_db)
    assert original_count == 10226, f"Error: Initial basins should be 10226, got {original_count}"

    # Extract cell size column
    cell_col = 'grid_cells' if 'grid_cells' in retention_db.columns else 'grd_cll'
    
    print("Loading original DDM30 Raw File for metadata merge...")
    raw_shp = os.path.join(base_dir, "raw_data", "basins_joined.shp")
    original = gpd.read_file(raw_shp).drop(columns=['geometry'])
    
    overlap_cols = [c for c in retention_db.columns if c in original.columns and c != 'subbasn']
    retention_db = retention_db.drop(columns=overlap_cols, errors='ignore')
    retention_db = retention_db.merge(original, on='subbasn', how='left')

    print("Isolating Known vs Missing Data...")
    
    # Needs imputation: Anything logically flagged by mask_final
    # We exclude to_sea == 0 (endorheic) from the mean pool
    known_data = retention_db[(retention_db['mask_final'] == 0) & 
                              (retention_db['retention'] >= 0.0) & 
                              (retention_db['retention'] <= 1.0) &
                              (retention_db['to_sea'] == 1)].copy()
                              
    needs_impute = retention_db[retention_db['mask_final'] == 1].copy()

    # Calculate mean retention for each size category in known data
    size_means = known_data.groupby(cell_col)['retention'].mean().to_dict()
    available_sizes = np.array(list(size_means.keys()))
    
    def get_mean_for_size(cells):
        if pd.isna(cells):
            return 0.75
        if cells in size_means:
            return size_means[cells]
        if len(available_sizes) == 0:
            return 0.75
        # find nearest size
        idx = (np.abs(available_sizes - cells)).argmin()
        return size_means[available_sizes[idx]]

    # ==========================
    # Applying Imputation Logic
    # ==========================
    def impute_retention(row):
        cells = row[cell_col]
        
        # Rule 1: <= 4 grid cells -> Fixed 0.75
        if pd.isna(cells) or cells <= 4:
            return 0.75, "Fixed (Size <= 4 Grids)"
            
        # Rule 2: > 4 grids -> Size-Based Mean
        else:
            mean_val = get_mean_for_size(cells)
            mean_val = max(0.01, min(0.99, mean_val))
            return round(mean_val, 4), f"Size Mean (Size {cells})"

    print(f"Valid Source Basins to compute Means: {len(known_data)}")
    print(f"Basins to Impute: {len(needs_impute)}")

    impute_results = needs_impute.apply(impute_retention, axis=1)
    
    needs_impute['retention_imputed'] = [res[0] for res in impute_results]
    needs_impute['impute_method'] = [res[1] for res in impute_results]
    
    # Re-calculate out_flux for the missing ones to remain physically coherent
    needs_impute['out_flux_imputed'] = np.where(needs_impute['gen_flux'] > 0, 
                                                needs_impute['gen_flux'] * (1 - needs_impute['retention_imputed']), 
                                                0)

    # Merge results
    retention_db['ret_final'] = np.where(retention_db['mask_final'] == 0, retention_db['ret_valid'], np.nan)
    retention_db['out_final'] = np.where(retention_db['mask_final'] == 0, retention_db['out_flux'], 0)
    retention_db['method'] = np.where(retention_db['mask_final'] == 0, "Calculated", "")
    
    # Map the imputed values directly into the main database
    impute_dict_ret = dict(zip(needs_impute['subbasn'], needs_impute['retention_imputed']))
    impute_dict_out = dict(zip(needs_impute['subbasn'], needs_impute['out_flux_imputed']))
    impute_dict_meth = dict(zip(needs_impute['subbasn'], needs_impute['impute_method']))
    
    mask_to_update = retention_db['subbasn'].isin(needs_impute['subbasn'])
    
    retention_db.loc[mask_to_update, 'ret_final'] = retention_db.loc[mask_to_update, 'subbasn'].map(impute_dict_ret)
    retention_db.loc[mask_to_update, 'out_final'] = retention_db.loc[mask_to_update, 'subbasn'].map(impute_dict_out)
    retention_db.loc[mask_to_update, 'method'] = retention_db.loc[mask_to_update, 'subbasn'].map(impute_dict_meth)

    # -----------------------------------------------------
    # Apply Original DDM30 Native Masks
    # -----------------------------------------------------
    
    assert len(retention_db) == 10226, f"Final count error, should be 10226, got {len(retention_db)}"
    
    # Apply Original DDM30 Native Masks + to_sea Mask
    # User confirmed we solely rely on to_sea for V6
    mask_native = (retention_db['to_sea'] == 0)
    
    retention_db.loc[mask_native, 'ret_final'] = np.nan
    retention_db.loc[mask_native, 'method'] = "Masked (DDM30 Native)"

    # Output Shapefile
    out_shp = os.path.join(output_dir, "06_V4_Final_Retention_Basins.shp")
    print(f"Saving Shapefile to {out_shp}...")
    retention_db.to_file(out_shp)
    
    # Output CSV
    out_csv = os.path.join(output_dir, "06_V4_Final_Retention_Data.csv")
    retention_db.drop(columns=['geometry']).to_csv(out_csv, index=False)

    print("Step 6 Complete.")

if __name__ == "__main__":
    main()

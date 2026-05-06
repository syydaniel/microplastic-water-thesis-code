import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set global font to Times New Roman (Reference Style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

def generate_unified(base_dir):
    out_folder = os.path.join(base_dir, "04_Final_Outputs", "Final_Deliverables")
    os.makedirs(out_folder, exist_ok=True)
    
    print("--- Generating Final Unified Version ---")
    
    retention_shp = os.path.join(base_dir, "03_Intermediate_Steps", "05_final_retention_basins.shp")
    retention_db = gpd.read_file(retention_shp)
    
    raw_shp = os.path.join(base_dir, "02_Raw_Data", "basins_joined.shp")
    original = gpd.read_file(raw_shp).drop(columns=['geometry'])
    
    overlap_cols = [c for c in retention_db.columns if c in original.columns and c != 'subbasn']
    retention_db = retention_db.drop(columns=overlap_cols, errors='ignore')
    retention_db = retention_db.merge(original, on='subbasn', how='left')
    
    cell_col = 'grid_cells' if 'grid_cells' in retention_db.columns else 'grd_cll'
    
    # Isolate Greenland using Bounding Box (Lat > 58, Lon between -75 and -10)
    centroids = retention_db.geometry.centroid
    is_greenland = (centroids.y > 58) & (centroids.x > -75) & (centroids.x < -10) & (retention_db['mask'] == 1)
    
    # 1. Determine the logical exclusion pool for imputation
    # Exclude to_sea == 0 AND Greenland so they don't corrupt the imputation base
    known_data = retention_db[(retention_db['mask_final'] == 0) & 
                              (retention_db['retention'] >= 0.0) & 
                              (retention_db['retention'] <= 1.0) &
                              (retention_db['to_sea'] == 1) &
                              (~is_greenland)].copy()
                                  
    # NEVER impute 'Not Available' or 'Endorheic' regions
    needs_impute = retention_db[(retention_db['mask_final'] == 1) & (~is_greenland) & (retention_db['to_sea'] == 1)].copy()

    # Calculate means
    size_means = known_data.groupby(cell_col)['retention'].mean().to_dict()
    available_sizes = np.array(list(size_means.keys()))
    
    def get_mean_for_size(cells):
        if pd.isna(cells):
            return 0.75
        if cells in size_means:
            return size_means[cells]
        if len(available_sizes) == 0:
            return 0.75
        idx = (np.abs(available_sizes - cells)).argmin()
        return size_means[available_sizes[idx]]

    def impute_retention(row):
        cells = row[cell_col]
        if pd.isna(cells) or cells <= 4:
            return 0.75, "Fixed (Size <= 4 Grids)"
        else:
            mean_val = get_mean_for_size(cells)
            mean_val = max(0.01, min(0.99, mean_val))
            return round(mean_val, 4), f"Size Mean (Size {cells})"

    impute_results = needs_impute.apply(impute_retention, axis=1)
    needs_impute['retention_imputed'] = [res[0] for res in impute_results]
    needs_impute['impute_method'] = [res[1] for res in impute_results]

    retention_db['ret_final'] = np.where(retention_db['mask_final'] == 0, retention_db['ret_valid'], np.nan)
    retention_db['method'] = np.where(retention_db['mask_final'] == 0, "Calculated", "")
    
    impute_dict_ret = dict(zip(needs_impute['subbasn'], needs_impute['retention_imputed']))
    impute_dict_meth = dict(zip(needs_impute['subbasn'], needs_impute['impute_method']))
    
    mask_to_update = retention_db['subbasn'].isin(needs_impute['subbasn'])
    retention_db.loc[mask_to_update, 'ret_final'] = retention_db.loc[mask_to_update, 'subbasn'].map(impute_dict_ret)
    retention_db.loc[mask_to_update, 'method'] = retention_db.loc[mask_to_update, 'subbasn'].map(impute_dict_meth)

    # 2. Apply Custom Final Map Masking
    mask_native = (retention_db['to_sea'] == 0) | is_greenland

    retention_db.loc[mask_native, 'ret_final'] = np.nan
    retention_db.loc[is_greenland, 'method'] = "Masked (No Data)"
    # Endorheic are basins that don't go to sea, AND aren't Greenland
    retention_db.loc[(retention_db['to_sea'] == 0) & (~is_greenland), 'method'] = "Masked (Endorheic)"
    
    # Add Ratio
    retention_db['ratio_final'] = 1.0 - retention_db['ret_final']

    # Export full versions (including all metadata from basins_joined)
    full_csv = os.path.join(out_folder, "Data_Full_All_Columns.csv")
    full_shp = os.path.join(out_folder, "Shapefile_Full_All_Columns.shp")
    
    print(f"Exporting Full Metadata CSV to {full_csv}...")
    retention_db.drop(columns=['geometry']).to_csv(full_csv, index=False)
    
    print(f"Exporting Full Metadata SHP to {full_shp}...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore 10-char truncation warnings
        retention_db.to_file(full_shp)

    # Essential clears
    area_col = 'ar_sqkm' if 'ar_sqkm' in retention_db.columns else 'area_sqkm'
    
    # We won't have discharge here easily without re-running script 10, so let's stick to core fluxes
    core_cols = [
        'subbasn', area_col, cell_col,
        'gen_flux', 'out_flux', 
        'ret_final', 'ratio_final', 'method'
    ]
    
    df_clear = retention_db[core_cols].copy()
    gdf_clear = retention_db[core_cols + ['geometry']].copy()
    
    gdf_clear.rename(columns={cell_col: 'grd_cells'}, inplace=True)
    df_clear.rename(columns={cell_col: 'grd_cells'}, inplace=True)

    # Output separate retention and ratio clear CSVs and SHPs
    # Ratio 
    ratio_cols = ['subbasn', area_col, 'grd_cells', 'gen_flux', 'out_flux', 'ratio_final', 'method']
    ret_cols = ['subbasn', area_col, 'grd_cells', 'gen_flux', 'out_flux', 'ret_final', 'method']
    
    # Rename to keep within 10 chars
    ratio_shp = gdf_clear[ratio_cols + ['geometry']].rename(columns={'ratio_final':'ratio_fin'})
    ret_shp = gdf_clear[ret_cols + ['geometry']].rename(columns={'ret_final':'ret_fin'})

    df_clear[ratio_cols].to_csv(os.path.join(out_folder, "Data_Ratio_Clear.csv"), index=False)
    df_clear[ret_cols].to_csv(os.path.join(out_folder, "Data_Retention_Clear.csv"), index=False)
    
    ratio_shp.to_file(os.path.join(out_folder, "Shapefile_Ratio_Clear.shp"))
    ret_shp.to_file(os.path.join(out_folder, "Shapefile_Retention_Clear.shp"))

    # Map Generation
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
    bg_color = 'white'
    ax.set_facecolor(bg_color)
    
    valid_data = retention_db[~retention_db['ret_final'].isna()].copy()
    masked_data = retention_db[retention_db['ret_final'].isna()].copy()
    
    import warnings
    import matplotlib as mpl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        bounds = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
        cmap = plt.cm.get_cmap('Spectral_r', len(bounds)-1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # Recompute Greenland explicitly for the plot rendering
    plot_centroids = retention_db.geometry.centroid
    is_greenland_plot = (plot_centroids.y > 58) & (plot_centroids.x > -75) & (plot_centroids.x < -10) & (retention_db['mask'] == 1)
    
    # Plot ALL masked data as a single universal lightgrey background
    if len(masked_data) > 0:
        masked_data.plot(ax=ax, color='lightgrey', edgecolor='grey', linewidth=0.05)
        
    # Colorbar Setup using make_axes_locatable (Match Reference layout)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    # Valid Data Plot with thin grey boundaries
    p1 = valid_data.plot(column='ret_final', ax=ax, cmap=cmap, norm=norm, legend=False, edgecolor='grey', linewidth=0.05, alpha=1.0)
    
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=bounds)
    cbar.set_label('Riverine Microplastic Retention Rate', fontsize=16, fontweight='bold', fontname='Times New Roman')
    cbar.ax.tick_params(labelsize=14)
    
    # Exact styling from 05_Global_Distribution_Analysis.py
    ax.set_xlabel("Longitude", fontsize=18, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Optional Legend for the masks
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='lightgrey', label='Not Available')
    ]
    ax.legend(handles=patches, loc='lower left', frameon=True, framealpha=0.9, fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "Map_Retention_Global.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated {out_folder}")

def main():
    base_dir = r"/Volumes/TU200Pro/Chapter test/Active_V6_Pipeline"
    generate_unified(base_dir)

if __name__ == "__main__":
    main()

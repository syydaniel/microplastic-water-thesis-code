import pandas as pd
import geopandas as gpd
import numpy as np
import lightgbm as lgb
import joblib
import shap
import os
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "Raw data")
MODEL_DIR = os.path.join(PROJECT_DIR, "02_Training", "02_Model_Jin5_Results")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "02_Training", "03_Global_Results_Jin5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input Files
MODEL_PATH = os.path.join(MODEL_DIR, "final_model_Jin5.pkl")
LEV12_SHP_PATH = os.path.join(RAW_DATA_DIR, "HB12", "BasinATLAS_v10_lev12.shp")
LEV06_SHP_PATH = os.path.join(RAW_DATA_DIR, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev06.shp")

# Intermediate & Output Files
GLOBAL_INDEX_CSV = os.path.join(OUTPUT_DIR, "Global_Predictors_Lev12_Jin5.csv")
GLOBAL_RESULTS_LEV12_CSV = os.path.join(OUTPUT_DIR, "Global_Results_Lev12_Full_Jin5.csv")
GLOBAL_STATS_LEV6_CSV = os.path.join(OUTPUT_DIR, "Global_Stats_Lev6_Jin5.csv")

# Transformations (Same as data prep)
TRANSFORMATIONS = {
    'dis_m3_pyr': ('Natural_Discharge_Upstream', 1.0),
    'run_mm_syr': ('Land_Surface_Runoff_Local', 1.0),
    'lkv_mc_usu': ('Lake_Volume_Upstream', 1e6),
    'rev_mc_usu': ('Reservoir_Volume_Upstream', 1e6),
    'ria_ha_ssu': ('River_Area_Local', 10000.0),
    'ria_ha_usu': ('River_Area_Upstream', 10000.0),
    'riv_tc_ssu': ('River_Volume_Local', 1000.0),
    'riv_tc_usu': ('River_Volume_Upstream', 1000.0),
    'ele_mt_sav': ('Elevation_Local', 1.0),
    'ele_mt_uav': ('Elevation_Upstream', 1.0),
    'slp_dg_sav': ('Terrain_Slope_Local', 0.1),
    'slp_dg_uav': ('Terrain_Slope_Upstream', 0.1),
    'sgr_dk_sav': ('Stream_Gradient_Local', 0.1),
    'tmp_dc_syr': ('Temperature_Local', 0.1),
    'tmp_dc_uyr': ('Temperature_Upstream', 0.1),
    'pre_mm_syr': ('Precipitation_Local', 1.0),
    'pre_mm_uyr': ('Precipitation_Upstream', 1.0),
    'pet_mm_syr': ('Potential_Evap_Local', 1.0),
    'pet_mm_uyr': ('Potential_Evap_Upstream', 1.0),
    'aet_mm_syr': ('Actual_Evap_Local', 1.0),
    'aet_mm_uyr': ('Actual_Evap_Upstream', 1.0),
    'crp_pc_sse': ('Cropland_Extent_Local', 1.0),
    'crp_pc_use': ('Cropland_Extent_Upstream', 1.0),
    'pst_pc_sse': ('Pasture_Extent_Local', 1.0),
    'pst_pc_use': ('Pasture_Extent_Upstream', 1.0),
    'wet_pc_sg1': ('Wetland_All_Local', 1.0),
    'wet_pc_ug1': ('Wetland_All_Upstream', 1.0),
    'wet_pc_sg2': ('Wetland_Land_Local', 1.0),
    'wet_pc_ug2': ('Wetland_Land_Upstream', 1.0),
    'pop_ct_ssu': ('Population_Local', 1000.0),
    'pop_ct_usu': ('Population_Upstream', 1000.0),
    'urb_pc_sse': ('Urban_Extent_Local', 1.0),
    'urb_pc_use': ('Urban_Extent_Upstream', 1.0),
    'rdd_mk_sav': ('Road_Density_Local', 1.0),
    'rdd_mk_uav': ('Road_Density_Upstream', 1.0),
    'hft_ix_s09': ('Human_Footprint_Local', 0.1),
    'hft_ix_u09': ('Human_Footprint_Upstream', 0.1),
    'hdi_ix_sav': ('Human_Dev_Index_Local', 0.001),
}

# Feature Engineering Settings
PNV_COL = 'pnv_cl_smj'
EXCLUDE_PNV_CODES = [14, 99] # Desert, Greenland

# --- Phase 1: Build Global Predictor Index ---
def build_global_index():
    print("\n--- Phase 1: Building Global Predictor Index ---")
    if os.path.exists(GLOBAL_INDEX_CSV):
        print("Removing existing Global Index CSV to rebuild with correct feature names...")
        os.remove(GLOBAL_INDEX_CSV)

    chunk_size = 100000
    start = 0
    chunk_idx = 0
    total_rows = 0
    
    # Init CSV
    first_chunk = True
    
    while True:
        try:
            chunk = gpd.read_file(LEV12_SHP_PATH, rows=slice(start, start + chunk_size))
        except Exception as e:
            print(f"Error reading chunk at {start}: {e}")
            break
            
        if chunk.empty:
            break
            
        # Extract Predictors
        X_chunk = pd.DataFrame(index=chunk.index)
        
        # 1. IDs
        X_chunk['HYBAS_ID'] = chunk['HYBAS_ID']
        if 'PFAF_ID' in chunk.columns:
            pfaf_str = chunk['PFAF_ID'].astype(str)
            X_chunk['PFAF_ID'] = chunk['PFAF_ID']
            X_chunk['Lev6_ID'] = pfaf_str.str[:6]
        else:
            X_chunk['PFAF_ID'] = np.nan
            X_chunk['Lev6_ID'] = np.nan
            
        if PNV_COL in chunk.columns:
             X_chunk['PNV_Code'] = chunk[PNV_COL].fillna(0).astype(int)
        else:
             X_chunk['PNV_Code'] = 0
             
        # 2. Predictors
        for orig_col, (new_name, scale) in TRANSFORMATIONS.items():
            if orig_col in chunk.columns:
                # Handle missing values (-9999 and -99.9) BEFORE scaling
                val = pd.to_numeric(chunk[orig_col], errors='coerce')
                val = val.replace([-9999, -99.9], np.nan)
                X_chunk[new_name] = val * scale
            else:
                X_chunk[new_name] = np.nan

        # Save
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        X_chunk.to_csv(GLOBAL_INDEX_CSV, mode=mode, header=header, index=False)
        
        first_chunk = False
        total_rows += len(chunk)
        rows_read = len(chunk)
        chunk_idx += 1
        print(f"Index Builder: Processed Chunk {chunk_idx} ({total_rows} rows)")
        
        start += chunk_size
        if rows_read < chunk_size:
            break
            
    print(f"Phase 1 Complete. Index saved to {GLOBAL_INDEX_CSV}")

# --- Phase 2: Predict & SHAP ---
def run_predictions_and_shap():
    print("\n--- Phase 2: Running Predictions & SHAP ---")
    
    # Check if Phase 1 Output exists
    if not os.path.exists(GLOBAL_INDEX_CSV):
        print("Global Index missing. Cannot run Phase 2.")
        return

    # Check/Reset Output
    # Since previous run crashed, we should reset to ensure consistency
    if os.path.exists(GLOBAL_RESULTS_LEV12_CSV):
        print(f"Removing existing incomplete results file: {GLOBAL_RESULTS_LEV12_CSV}")
        os.remove(GLOBAL_RESULTS_LEV12_CSV)

    # Load Model
    print("Loading Model...")
    model = joblib.load(MODEL_PATH)
    features = model.feature_name()
    
    # Setup SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    # Load Data fully to memory to avoid file locking issues
    print(f"Loading Index CSV into memory: {GLOBAL_INDEX_CSV}")
    data_full = pd.read_csv(GLOBAL_INDEX_CSV)
    total_samples = len(data_full)
    print(f"Loaded {total_samples} rows.")
    
    chunk_size = 50000 
    
    # Iterate DataFrame manually
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk = data_full.iloc[start_idx:end_idx].copy()
        
        # CLEANING: -99.9 and -9999 to NaN
        chunk.replace([-99.9, -9999], np.nan, inplace=True)
        
        X_model = chunk[features]
        
        # 1. Predict
        preds = model.predict(X_model)
        
        # 2. SHAP
        shap_vals = explainer.shap_values(X_model)
        if isinstance(shap_vals, list):
             shap_vals = shap_vals[1]
             
        # 3. Create Result DataFrame
        result_chunk = pd.DataFrame()
        result_chunk['HYBAS_ID'] = chunk['HYBAS_ID']
        result_chunk['PFAF_ID'] = chunk['PFAF_ID']
        result_chunk['Lev6_ID'] = chunk['Lev6_ID']
        result_chunk['PNV_Code'] = chunk['PNV_Code']
        result_chunk['Log_Pred'] = preds
        
        # Add SHAP columns
        shap_df = pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in features], index=chunk.index)
        result_chunk = pd.concat([result_chunk, shap_df], axis=1)
        
        # 4. Mask Desert/Greenland
        mask = result_chunk['PNV_Code'].isin(EXCLUDE_PNV_CODES)
        result_chunk.loc[mask, 'Log_Pred'] = np.nan
        result_chunk.loc[mask, [c for c in result_chunk.columns if 'SHAP_' in c]] = np.nan
        
        # Save
        first_chunk = (start_idx == 0)
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        result_chunk.to_csv(GLOBAL_RESULTS_LEV12_CSV, mode=mode, header=header, index=False)
        
        print(f"Prediction: Processed Chunk {start_idx // chunk_size + 1} / {int(np.ceil(total_samples/chunk_size))}")
        
        # Garbage Collection
        del X_model, shap_vals, shap_df, result_chunk
        gc.collect()

    print(f"Phase 2 Complete. Results saved to {GLOBAL_RESULTS_LEV12_CSV}")

# --- Phase 3: Aggregation ---
def aggregate_level6():
    print("\n--- Phase 3: Aggregating to Level 6 (Chunked) ---")
    
    if not os.path.exists(GLOBAL_RESULTS_LEV12_CSV):
        print("Results file missing.")
        return

    # --- Step 1: Prediction Statistics (Fast, Low Memory) ---
    print("Step 1: Calculating Prediction Stats (Mean, Median, Std)...")
    try:
        # Load only necessary columns
        df_pred = pd.read_csv(GLOBAL_RESULTS_LEV12_CSV, usecols=['Lev6_ID', 'Log_Pred'])
    except ValueError:
        # Fallback if usecols fails (e.g. column name mismatch)
        print("Error reading columns. Reading full header to check names...")
        print(pd.read_csv(GLOBAL_RESULTS_LEV12_CSV, nrows=0).columns)
        return

    # Filter Valid key
    valid_pred = df_pred.dropna(subset=['Log_Pred']).copy()
    
    # Calculate counts for later normalization
    counts = valid_pred.groupby('Lev6_ID').size()
    
    # Restore to linear scale -> Average -> Log
    valid_pred['Linear_Pred'] = (10 ** valid_pred['Log_Pred']) - 1
    
    grouped = valid_pred.groupby('Lev6_ID')
    
    # Mean & Median of Linear
    agg_linear = grouped['Linear_Pred'].agg(['mean', 'median'])
    
    mean_linear = agg_linear['mean']
    median_linear = agg_linear['median']
    
    # Transform back to Log (Base 10)
    mean_log_conc = np.log10(mean_linear + 1)
    median_log_conc = np.log10(median_linear + 1)
    
    # Std of Log (Heterogeneity)
    std_log_conc = grouped['Log_Pred'].std()
    
    conc_stats = pd.DataFrame({
        'Lev6_ID': mean_log_conc.index,
        'Mean_Log_Conc': mean_log_conc.values,
        'Median_Log_Conc': median_log_conc.values,
        'Std_Log_Conc': std_log_conc.values
    })
    
    # Cleanup Step 1
    del df_pred, valid_pred, agg_linear, mean_linear, median_linear, mean_log_conc, median_log_conc, std_log_conc
    gc.collect()

    # --- Step 2: SHAP Statistics (Chunked for Memory Efficiency) ---
    print("Step 2: Calculating SHAP Stats (Chunked)...")
    
    # Identify SHAP columns
    all_cols = pd.read_csv(GLOBAL_RESULTS_LEV12_CSV, nrows=0).columns.tolist()
    shap_cols = [c for c in all_cols if 'SHAP_' in c]
    
    # Accumulators
    shap_sum = None
    shap_abs_sum = None
    
    chunk_size = 50000
    reader = pd.read_csv(GLOBAL_RESULTS_LEV12_CSV, usecols=['Lev6_ID', 'Log_Pred'] + shap_cols, chunksize=chunk_size)
    
    for i, chunk in enumerate(reader):
        # Filter same as Step 1 (valid predictions only)
        chunk = chunk.dropna(subset=['Log_Pred'])
        
        if chunk.empty:
            continue
            
        # Group by Lev6
        # Sum
        chunk_sum = chunk.groupby('Lev6_ID')[shap_cols].sum()
        # Abs Sum
        chunk_abs_sum = chunk[shap_cols].abs()
        chunk_abs_sum['Lev6_ID'] = chunk['Lev6_ID']
        chunk_abs_sum = chunk_abs_sum.groupby('Lev6_ID')[shap_cols].sum()
        
        if shap_sum is None:
            shap_sum = chunk_sum
            shap_abs_sum = chunk_abs_sum
        else:
            shap_sum = shap_sum.add(chunk_sum, fill_value=0)
            shap_abs_sum = shap_abs_sum.add(chunk_abs_sum, fill_value=0)
            
        if i % 10 == 0:
            print(f"  Processed SHAP chunk {i}...")
            
    # Normalize by counts
    # Align indices
    common_idx = counts.index.intersection(shap_sum.index)
    shap_sum = shap_sum.loc[common_idx]
    shap_abs_sum = shap_abs_sum.loc[common_idx]
    counts = counts.loc[common_idx]
    
    # Calculate Means
    shap_mean = shap_sum.div(counts, axis=0).reset_index()
    shap_abs_mean = shap_abs_sum.div(counts, axis=0)
    
    # Identify Dominant Driver (Max Mean Abs SHAP)
    dominant_driver = shap_abs_mean.idxmax(axis=1).to_frame(name='Dominant_Driver').reset_index()
    
    # --- Step 3: Merge ---
    final_stats = conc_stats.merge(shap_mean, on='Lev6_ID', how='left') \
                            .merge(dominant_driver, on='Lev6_ID', how='left')
    
    # Anomaly
    global_mean = final_stats['Mean_Log_Conc'].mean()
    final_stats['Anomaly'] = final_stats['Mean_Log_Conc'] - global_mean
    
    final_stats.to_csv(GLOBAL_STATS_LEV6_CSV, index=False)
    print(f"Phase 3 Complete. Stats saved to {GLOBAL_STATS_LEV6_CSV}")

# --- Phase 4: Mapping ---
def generate_maps():
    print("\n--- Phase 4: generating Maps ---")
    if not os.path.exists(GLOBAL_STATS_LEV6_CSV):
        print("Stats file missing.")
        return
        
    stats = pd.read_csv(GLOBAL_STATS_LEV6_CSV)
    lev6_gdf = gpd.read_file(LEV06_SHP_PATH)
    
    # Merge
    # Lev6 PFAF usually needed as string for join
    lev6_gdf['PFAF_ID_Str'] = lev6_gdf['PFAF_ID'].astype(str)
    # Stats Lev6_ID might be int or float or str
    # Clean it: remove .0 if float strings
    stats['Lev6_ID'] = stats['Lev6_ID'].astype(str).str.split('.').str[0]
    
    merged = lev6_gdf.merge(stats, left_on='PFAF_ID_Str', right_on='Lev6_ID', how='left')
    
    # --- Arid Filter (User Request) ---
    # FILTER REMOVED per User Request (Jan 2026)
    # if 'dis_m3_pyr' in merged.columns:
    #     print("Applying Arid Region Filter (dis_m3_pyr < 1)...")
    #     ... 

    # Styling
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['font.size'] = 12
    
    def plot_map(col, title, fname, cmap='viridis', vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(15, 8))
        merged.plot(ax=ax, color='lightgrey', edgecolor='none') # NA background
        
        valid = merged.dropna(subset=[col])
        if not valid.empty:
            valid.plot(ax=ax, column=col, cmap=cmap, legend=True, vmin=vmin, vmax=vmax,
                       legend_kwds={'label': title, 'orientation': 'horizontal', 'shrink': 0.6})
        
        ax.set_title(title, fontsize=16)
        # Enable Axis for Coordinates
        # ax.set_axis_off() 
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close()
        print(f"Saved {fname}")
        
    # 1. Mean
    plot_map('Mean_Log_Conc', 'Global Mean $Log_{10}(Abundance)$ (Level 6)', 'Map_Mean_Conc_Jin5.png', cmap='YlOrRd', vmin=0, vmax=7)

    # 1b. Median
    plot_map('Median_Log_Conc', 'Global Median $Log_{10}(Abundance)$ (Level 6)', 'Map_Median_Conc_Jin5.png', cmap='YlOrRd', vmin=0, vmax=7)
    
    # 2. Heterogeneity
    plot_map('Std_Log_Conc', 'Heterogeneity (Std Dev) (Level 6)', 'Map_Heterogeneity_Jin5.png', cmap='YlOrRd', vmin=0, vmax=1.5)
    
    plot_map('Anomaly', 'Abundance Anomaly', 'Map_Anomaly_Jin5.png', cmap='RdBu_r', vmin=-2.5, vmax=2.5)

    print("Phase 4 Complete.")

# --- Models ---
if __name__ == "__main__":
    # Validate contextily
    # Remove if not needed.
    
    build_global_index()
    run_predictions_and_shap()
    aggregate_level6()
    generate_maps()

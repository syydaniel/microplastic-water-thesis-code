import pandas as pd
import geopandas as gpd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
TRAIN_DIR = os.path.join(PROJECT_DIR, "02_Training")
COMP_DIR = os.path.join(PROJECT_DIR, "04_Model_Comparison")
STATS_DIR = os.path.join(COMP_DIR, "01_Stats_Files")
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "Raw data")
# Level 6 Shapefile path
LEV06_SHP_PATH = os.path.join(RAW_DATA_DIR, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev06.shp")

# Create Output Dirs
os.makedirs(STATS_DIR, exist_ok=True)

# Define Source Models
# Key: Model Output Name 
# Value: (Source Directory Name in 02_Training, Source Filename)
MODELS = {
    "Baseline": ("03_Global_Results_Refined", "Global_Stats_Lev6.csv"),
    "Jin5": ("03_Global_Results_Jin5", "Global_Stats_Lev6_Jin5.csv"),
    "SHAP5": ("03_Global_Results_SHAP_top5", "Global_Stats_Lev6_SHAP_top5.csv"),
    "Cluster3": ("03_Global_Results_Cluster3", "Global_Stats_Lev6_cluster3.csv"),
    "Cluster5": ("03_Global_Results_Cluster5", "Global_Stats_Lev6_cluster5.csv"),
    "Cluster7": ("03_Global_Results_Cluster7", "Global_Stats_Lev6_cluster7.csv")
}

def main():
    # --- 2. Copy Files ---
    print("--- Step 1: Copying Stats Files ---")
    copied_paths = {}
    
    for model_name, (folder, filename) in MODELS.items():
        src_path = os.path.join(TRAIN_DIR, folder, filename)
        dst_filename = f"Global_Stats_Lev6_{model_name}.csv"
        dst_path = os.path.join(STATS_DIR, dst_filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"[{model_name}] Copied to {dst_path}")
            copied_paths[model_name] = dst_path
        else:
            print(f"[{model_name}] ERROR: Source file not found: {src_path}")
            
    if not copied_paths:
        print("No files copied. Exiting.")
        return

    # --- 3. Merge Data ---
    print("\n--- Step 2: Merging Data ---")
    master_df = None
    model_columns = []
    
    for model_name, path in copied_paths.items():
        print(f"Processing {model_name}...")
        try:
            df = pd.read_csv(path)
            
            # Standardize Link Key
            if 'Lev6_ID' in df.columns:
                # Ensure string and remove potential decimals
                df['Lev6_ID'] = df['Lev6_ID'].astype(str).str.split('.').str[0]
            else:
                print(f"Warning: 'Lev6_ID' not found in {model_name}")
                continue
                
            # Rename Mean_Log_Conc
            col_name = f"Mean_Log_{model_name}"
            df_subset = df[['Lev6_ID', 'Mean_Log_Conc']].rename(columns={'Mean_Log_Conc': col_name})
            
            if master_df is None:
                master_df = df_subset
            else:
                master_df = master_df.merge(df_subset, on='Lev6_ID', how='outer')
                
            model_columns.append(col_name)
            
        except Exception as e:
            print(f"Error reading {model_name}: {e}")

    print(f"Merged {len(model_columns)} models: {model_columns}")
    
    # --- 4. Calculate CV (Refined: Linear Scale) ---
    print("\n--- Step 3: Calculating Coefficient of Variation (CV) on Linear Scale ---")
    
    # Convert Log Means to Linear Means for CV calculation
    # Logic: Mean_Log_Conc = Log10(Linear + 1)  =>  Linear = 10^(Mean_Log) - 1
    linear_cols = []
    for col in model_columns:
        lin_col = col.replace("Log", "Linear_Calc") # Avoid conflict
        master_df[lin_col] = (10 ** master_df[col]) - 1
        # Clip potential floating point negative errors
        master_df[lin_col] = master_df[lin_col].clip(lower=0)
        linear_cols.append(lin_col)

    # Calculate Mean and Std across models (Linear)
    master_df['Ensemble_Mean_Linear'] = master_df[linear_cols].mean(axis=1)
    master_df['Ensemble_Std_Linear'] = master_df[linear_cols].std(axis=1)
    
    # CV = Std / Mean (Linear)
    master_df['CoV'] = master_df['Ensemble_Std_Linear'] / master_df['Ensemble_Mean_Linear']
    
    # Save Results
    res_csv = os.path.join(COMP_DIR, "Model_Comparison_CV.csv")
    master_df.to_csv(res_csv, index=False)
    print(f"Saved Comparison Data to {res_csv}")
    
    # --- 5. Visualization ---
    print("\n--- Step 4: Generating CoV Map ---")
    
    if not os.path.exists(LEV06_SHP_PATH):
        print(f"Shapefile not found: {LEV06_SHP_PATH}")
        return
        
    gdf = gpd.read_file(LEV06_SHP_PATH)
    gdf['PFAF_ID_Str'] = gdf['PFAF_ID'].astype(str)
    
    # Merge with Shapefile
    merged_gdf = gdf.merge(master_df, left_on='PFAF_ID_Str', right_on='Lev6_ID', how='left')
    
    # Plotting
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def plot_cv_map(gdf, col, title, fname, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(15, 10))
        gdf.plot(ax=ax, color='lightgrey', edgecolor='none')
        
        valid = gdf.dropna(subset=[col])
        if not valid.empty:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            
            valid.plot(ax=ax, column=col, cmap='Spectral_r', legend=True, cax=cax,
                       vmin=vmin, vmax=vmax,
                       legend_kwds={'label': title, 'orientation': 'vertical'})
            
            cax.set_ylabel(title, fontsize=14, fontweight='bold', fontname='Times New Roman')
            cax.tick_params(labelsize=12)
        
        ax.set_title("Model Uncertainty (Coefficient of Variation across 6 Models)", fontsize=20, fontweight='bold', fontname='Times New Roman')
        
        # Refined Style: Grid and Axis Labels enabled
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Longitude", fontsize=16, fontweight='bold')
        ax.set_ylabel("Latitude", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(COMP_DIR, fname), dpi=300)
        plt.close()
        print(f"Map saved to {fname}")

    # Generate Map
    # Use merged_gdf (and handle potential NaNs)
    if not merged_gdf['CoV'].isna().all():
        v_max = merged_gdf['CoV'].quantile(0.95)
    else:
        v_max = 1.0
        
    plot_cv_map(merged_gdf, 'CoV', 'Coefficient of Variation (CoV)', "Map_Model_Comparison_CV.png", vmin=0, vmax=v_max)


    # --- 6. Trust Classification & Validation Summary ---
    print("\n--- Step 5: Trust Level Classification ---")
    
    # Define Thresholds
    # < 0.2: High Confidence (Legacy) -> Adjusted? No, keeping same categories but maybe renamed if needed.
    # User didn't ask to change these thresholds for the summary table, only for agreement map filter (1.0).
    # But let's renaming "CV" to "CoV" in the text strings.
    
    def classify_cv(cov):
        if pd.isna(cov): return "No Data"
        if cov < 0.2: return "High Confidence (CoV < 0.2)"
        elif cov < 0.5: return "Medium Confidence (0.2 <= CoV < 0.5)"
        else: return "Low Confidence (CoV >= 0.5)"
        
    master_df['Trust_Level'] = master_df['CoV'].apply(classify_cv)
    
    # Calculate Distibution
    trust_counts = master_df['Trust_Level'].value_counts()
    trust_pct = master_df['Trust_Level'].value_counts(normalize=True) * 100
    
    summary_df = pd.DataFrame({'Count': trust_counts, 'Percentage': trust_pct})
    print("\nTrust Level Summary:")
    print(summary_df)
    
    # Save Summary
    summary_csv = os.path.join(COMP_DIR, "Trust_Level_Summary.csv")
    summary_df.to_csv(summary_csv)
    
    # Visualize Distribution
    plt.figure(figsize=(10, 6))
    colors = {'High Confidence (CV < 0.2)': '#2ca02c', # Green
              'Medium Confidence (0.2 <= CV < 0.5)': '#ff7f0e', # Orange
              'Low Confidence (CV >= 0.5)': '#d62728', # Red
              'No Data': 'gray'}
              
    trust_counts.plot(kind='bar', color=[colors.get(x, 'gray') for x in trust_counts.index])
    plt.title("Distribution of Model Trust Levels (Basin Count)")
    plt.ylabel("Number of Basins")
    plt.xlabel("Trust Level")
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    dist_path = os.path.join(COMP_DIR, "Trust_Level_Distribution.png")
    plt.savefig(dist_path, dpi=300)
    print(f"Distribution plot saved to {dist_path}")
    plt.close()

if __name__ == "__main__":
    main()

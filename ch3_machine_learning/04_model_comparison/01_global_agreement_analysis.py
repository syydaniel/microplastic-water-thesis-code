import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# --- Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
TRAINING_DIR = os.path.join(PROJECT_DIR, "02_Training")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "04_Model_Comparison")
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "Raw data")
LEV06_SHP_PATH = os.path.join(RAW_DATA_DIR, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev06.shp")

# Thresholds (Items / m3)
THRESH_MEDIAN = 1200
THRESH_MEAN = 25101.55
# THRESH_HIGH removed as per request

# Input Models and File Paths
# Mapping: Model Name -> CSV Path
MODEL_DIRS = {
    "Cluster3": os.path.join(TRAINING_DIR, "03_Global_Results_Cluster3", "Global_Stats_Lev6_cluster3.csv"),
    "Cluster5": os.path.join(TRAINING_DIR, "03_Global_Results_Cluster5", "Global_Stats_Lev6_cluster5.csv"),
    "Cluster7": os.path.join(TRAINING_DIR, "03_Global_Results_Cluster7", "Global_Stats_Lev6_cluster7.csv"),
    "Jin5":     os.path.join(TRAINING_DIR, "03_Global_Results_Jin5", "Global_Stats_Lev6_Jin5.csv"),
    "Refined":  os.path.join(TRAINING_DIR, "03_Global_Results_Refined", "Global_Stats_Lev6.csv"),
    "SHAP_top5": os.path.join(TRAINING_DIR, "03_Global_Results_SHAP_top5", "Global_Stats_Lev6_SHAP_top5.csv"),
}

def load_and_classify():
    print("--- Loading and Classifying 6 Models ---")
    
    # Base DataFrame (Using Lev6_ID as key)
    first_key = list(MODEL_DIRS.keys())[0]
    path = MODEL_DIRS[first_key]
    if not os.path.exists(path):
        print(f"CRITICAL: Base file missing {path}")
        return None
        
    # Initialize with just IDs
    df_base = pd.read_csv(path)[['Lev6_ID']].copy()
    df_base['Lev6_ID'] = df_base['Lev6_ID'].astype(str).str.split('.').str[0]
    
    classification_frames = []
    final_df = df_base
    
    for name, filepath in MODEL_DIRS.items():
        if not os.path.exists(filepath):
            print(f"Warning: File missing for {name}: {filepath}")
            continue
            
        print(f"Processing {name}...")
        df = pd.read_csv(filepath)
        
        # Ensure Lev6_ID format
        df['Lev6_ID'] = df['Lev6_ID'].astype(str).str.split('.').str[0]
        
        # Calculate Abundance (10^Mean_Log_Conc)
        # Note: Mean_Log_Conc is typically Log10.
        if 'Mean_Log_Conc' in df.columns:
            df['Mean_Log_Conc'] = pd.to_numeric(df['Mean_Log_Conc'], errors='coerce')
            # Calculate Abundance (10^Mean_Log_Conc - 1)
            # Clip negative to 0 to handle small float errors
            df[f'Abundance_{name}'] = (10 ** df['Mean_Log_Conc']) - 1
            df.loc[df[f'Abundance_{name}'] < 0, f'Abundance_{name}'] = 0
            df[f'Log_{name}'] = df['Mean_Log_Conc'] # Store Log values
        else:
            print(f"  Error: 'Mean_Log_Conc' not found in {name}")
            continue
            
        # Classify
        # 0: < Median
        # 1: > Median (1200)
        # 2: > Mean (25101.55)
        
        df[f'Class_{name}'] = 0
        df.loc[df[f'Abundance_{name}'] > THRESH_MEDIAN, f'Class_{name}'] = 1
        df.loc[df[f'Abundance_{name}'] > THRESH_MEAN, f'Class_{name}'] = 2
        
        # Merge back to base
        # Keep Class, Abundance (for stats if needed), and Log (for CV)
        df_sub = df[['Lev6_ID', f'Class_{name}', f'Abundance_{name}', f'Log_{name}']]
        
        final_df = final_df.merge(df_sub, on='Lev6_ID', how='inner') 
            
        classification_frames.append(name)

    print(f"Merged {len(final_df)} sub-basins common to all {len(classification_frames)} models.")
    return final_df, classification_frames

def calculate_agreement(df, model_names):
    print("--- Calculating Agreement ---")
    
    class_cols = [f'Class_{name}' for name in model_names]
    abund_cols = [f'Abundance_{name}' for name in model_names]
    
    # 1. Calculate CoV across models for each sub-basin (LINEAR SPACE)
    # User requested Linear CoV calculation.
    # CoV = Std / Mean
    df['Mean_All'] = df[abund_cols].mean(axis=1)
    df['Std_All'] = df[abund_cols].std(axis=1)
    
    # Avoid division by zero
    df['CoV'] = df['Std_All'] / df['Mean_All']
    
    # 2. Determine Base Agreement (ignoring CoV for a moment)
    df['Min_Class'] = df[class_cols].min(axis=1)
    df['Max_Class'] = df[class_cols].max(axis=1)
    
    df['Agreement_Category'] = 0 # Default No Consensus
    
    # Logic:
    # 0: No Consensus (Default)
    # 1: All < Median (Max == 0)
    # 2: All > Median (Min >= 1)
    # 3: All > Mean (Min >= 2)
    
    df.loc[df['Max_Class'] == 0, 'Agreement_Category'] = 1 # Consensus Low
    df.loc[df['Min_Class'] >= 1, 'Agreement_Category'] = 2 # Consensus Median
    df.loc[df['Min_Class'] >= 2, 'Agreement_Category'] = 3 # Consensus Mean
    
    # 3. Apply CoV Filter (User Request: CoV < 1.0)
    # If CoV >= 1.0, it is No Consensus (0).
    
    mask_high_cov = df['CoV'] >= 1.0
    df.loc[mask_high_cov, 'Agreement_Category'] = 0
    
    # Count stats
    counts = df['Agreement_Category'].value_counts().sort_index()
    print("\nAgreement Categories (after CoV < 1.0 filter):")
    print(f"0 (No Consensus / CoV>=1.0): {counts.get(0, 0)}")
    print(f"1 (Consensus Low < 1200): {counts.get(1, 0)}")
    print(f"2 (Consensus Median > 1200): {counts.get(2, 0)}")
    print(f"3 (Consensus Mean > 25k): {counts.get(3, 0)}")
    
    return df

def plot_agreement_map(df):
    print("\n--- generating Map ---")
    
    # Load Geometry
    print(f"Loading Shapefile: {LEV06_SHP_PATH}")
    gdf = gpd.read_file(LEV06_SHP_PATH)
    
    # Join
    gdf['PFAF_ID_STR'] = gdf['PFAF_ID'].astype(str)
    # Ensure ID match
    merged = gdf.merge(df, left_on='PFAF_ID_STR', right_on='Lev6_ID', how='left')
    
    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "Global_Results_Agreement_Levels.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved Data to {out_csv}")
    
    # Visualization Styling
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold' 
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define Data Subsets
    valid_data = merged.dropna(subset=['Agreement_Category'])
    no_data = merged[merged['Agreement_Category'].isna()]
    
    # Plot Data First (Bottom Layer)
    # Colors Mapping:
    # 0: No Consensus (Light Blue)
    # 1: Low (Light Green)
    # 2: Median (Yellow)
    # 3: Mean (Orange) 
    
    colors = ['#ADD8E6', '#90EE90', '#fed976', '#fd8d3c'] 
    cmap = ListedColormap(colors)
    # Bounds: -0.5, 0.5, 1.5, 2.5, 3.5
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, len(colors))
    
    if not valid_data.empty:
        valid_data.plot(column='Agreement_Category', ax=ax, cmap=cmap, norm=norm, 
                        linewidth=0, edgecolor='none') 

    # Plot No Data on Top (Top Layer) to ensure masking
    if not no_data.empty:
        no_data.plot(ax=ax, color='#d3d3d3', edgecolor='none')
    
    # Custom Legend
    legend_patches = [
        mpatches.Patch(color=colors[3], label=r'Above Mean ($> 2.51 \times 10^4$ items m$^{-3}$)'),
        mpatches.Patch(color=colors[2], label=r'Above Median ($> 1.20 \times 10^3$ items m$^{-3}$)'),
        mpatches.Patch(color=colors[1], label=r'Below Median ($< 1.20 \times 10^3$ items m$^{-3}$)'),
        mpatches.Patch(color=colors[0], label=r'No Consensus (CoV $\geq$ 1.0)'),
        mpatches.Patch(color='#d3d3d3', label='No Data')
    ]
    
    legend = ax.legend(handles=legend_patches, 
              loc='lower left', 
              title="Agreement Level (6 Models)",
              title_fontsize=16,
              fontsize=14, 
              frameon=True,
              fancybox=False,
              edgecolor='black',
              facecolor='white')
    
    plt.setp(legend.get_title(), fontname='Times New Roman', fontweight='bold')
    for t in legend.get_texts():
        t.set_fontname('Times New Roman')
    
    ax.set_title("Global Microplastic Agreement Map (CoV < 1.0 Filter)", fontsize=24, fontweight='bold', fontname='Times New Roman')
    
    # Refined Style: Grid and Axis Labels
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Longitude", fontsize=16, fontweight='bold')
    ax.set_ylabel("Latitude", fontsize=16, fontweight='bold')
    
    out_png = os.path.join(OUTPUT_DIR, "Map_Global_Agreement_Levels_CoV1.0.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Map Saved to {out_png}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    result = load_and_classify()
    if result is not None:
        df, model_names = result
        if len(df) > 0:
            df_agreed = calculate_agreement(df, model_names)
            plot_agreement_map(df_agreed)
        else:
            print("No data found to process.")
    else:
        print("Skipping Agreement Analysis due to missing input files (Pipeline likely still running).")


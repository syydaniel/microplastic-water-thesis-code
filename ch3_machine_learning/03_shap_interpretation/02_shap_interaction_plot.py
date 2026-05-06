import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# --- CONFIGURATION ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
INPUT_CSV = os.path.join(PROJECT_DIR, "01_Data_Prep", "01_Data_Combine_Result", "data_combine.csv")
SHAP_PATH = os.path.join(PROJECT_DIR, "02_Training", "02_Model_Results", "shap_values.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "04_SHAP_Visual_Analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling (Global Scheme)
main_cmap_colors = ["#7B6C9F", "#e0e0e0", "#3AB5B3"]
custom_cmap = LinearSegmentedColormap.from_list('val_cmap', main_cmap_colors)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['axes.unicode_minus'] = False

aesthetic_params = {
    'ax_label_size': 16,
    'tick_label_size': 14,
    'legend_size': 14,
    'marker_size': 20,
    'edge_width': 0.6,
    'spine_width': 1.2
}

# Abbreviation Map (Bold Scientific)
ABBREVIATION_MAP = {
    'Land Surface Runoff Local': r'LSR$_{\mathbf{SUB}}$',
    'Potential Evap Local': r'PET$_{\mathbf{SUB}}$',
    'Cropland Extent Local': r'CROPL$_{\mathbf{SUB}}$',
    'Human Dev Index Local': r'HDI$_{\mathbf{SUB}}$'
}

def get_label(name):
    clean = name.replace('_', ' ').strip()
    return ABBREVIATION_MAP.get(clean, clean)

def apply_academic_style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(aesthetic_params['spine_width'])
        spine.set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=1.2, labelsize=aesthetic_params['tick_label_size'])

def main():
    print("--> Loading Data...")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_CSV)
    df = df[df['Std_Value_m3'] > 0].dropna(subset=['Std_Value_m3']) # Initial filter
    
    info_cols = ['Unique_ID', 'Std_Value_m3', 'Gear_Category', 'Target_Log', 'HYBAS_ID', 'geometry', 'index_right']
    predictors = [c for c in df.columns if c not in info_cols]
    X_full = df[predictors].reset_index(drop=True)
    
    # Handle -99 as NaN
    X_clean = X_full.replace(-99, np.nan)
    
    # 2. Load SHAP
    shap_df = pd.read_csv(SHAP_PATH)
    shap_values = shap_df.values
    
    if X_clean.shape[0] != shap_values.shape[0]:
        print(f"Warning: SHAP shape {shap_values.shape} mismatch with Data {X_clean.shape}. Proceeding cautiously.")

    # 3. Define Interaction Pairs
    # Requirement: Use HDI and Cropland as Color.
    # Format: (X_Axis_Feature, Color_Feature)
    # Pairs based on user history:
    # 1. Old: Cropland(X) vs Runoff(Color) -> New: Runoff(X) vs Cropland(Color)
    # 2. Old: HDI(X) vs Runoff(Color)      -> New: Runoff(X) vs HDI(Color)
    # 3. Old: HDI(X) vs PET(Color)         -> New: PET(X) vs HDI(Color)
    
    interaction_pairs = [
        ('Land Surface Runoff Local', 'Cropland Extent Local'),
        ('Land Surface Runoff Local', 'Human Dev Index Local'),
        ('Potential Evap Local', 'Human Dev Index Local')
    ]
    
    # 4. Plotting
    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    for i, (feat_x, feat_color) in enumerate(interaction_pairs):
        print(f"Plotting Pair {i+1}: X={feat_x}, Color={feat_color}")
        ax = fig.add_subplot(gs[0, i])
        
        # Get Data
        idx_x = X_clean.columns.get_loc(feat_x)
        x_data = X_clean[feat_x]
        shap_y = shap_values[:, idx_x] # Y-Axis is SHAP of X
        color_data = X_clean[feat_color]
        
        # Filter valid
        mask = ~np.isnan(x_data) & ~np.isnan(shap_y) & ~np.isnan(color_data)
        x_plot = x_data[mask]
        y_plot = shap_y[mask]
        c_plot = color_data[mask]
        
        # Add Zero Line
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
        
        # Scatter (X vs SHAP(X), Color=ColorFeature)
        # User requested NO truncation
        sc = ax.scatter(x_plot, y_plot, c=c_plot, cmap=custom_cmap, 
                        s=aesthetic_params['marker_size'], alpha=0.8,
                        edgecolors='white', linewidths=aesthetic_params['edge_width'], zorder=2)
        
        apply_academic_style(ax)
        
        # Labels
        ax.set_xlabel(get_label(feat_x), fontsize=aesthetic_params['ax_label_size'], fontweight='bold')
        if i == 0:
            ax.set_ylabel("SHAP Value", fontsize=aesthetic_params['ax_label_size'], fontweight='bold')
            
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(get_label(feat_color), fontsize=aesthetic_params['ax_label_size'])
        cbar.ax.tick_params(labelsize=aesthetic_params['tick_label_size'])
        cbar.outline.set_visible(False)
        
    out_file = os.path.join(OUTPUT_DIR, "SHAP_Interaction_Analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"--> Saved to: {out_file}")

if __name__ == "__main__":
    main()

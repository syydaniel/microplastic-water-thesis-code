
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

# --- Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
TRAINING_DIR = os.path.join(PROJECT_DIR, "02_Training")
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "Raw data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "04_Model_Comparison")
LEV06_SHP_PATH = os.path.join(RAW_DATA_DIR, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev06.shp")

# Model Configuration
# Key: Display Name
# Value: { 
#   'stats_file': Path to Global Stats CSV (Level 6) for Mapping
#   'preds_file': Path to Detailed Predictions CSV (Level 12) for Metrics
#   'col_name': Column name in Stats file for Mean Log Abundance
# }
MODELS = {
    "Baseline": {
        "stats_file": os.path.join(TRAINING_DIR, "03_Global_Results_Refined", "Global_Stats_Lev6.csv"),
        "preds_file": os.path.join(TRAINING_DIR, "02_Model_train_baseline_all_predictors.py"), # Dummy, predictions usually in results
        # Baseline preds might be in 02_Model_Results
        "preds_file_real": os.path.join(TRAINING_DIR, "02_Model_Results", "predictions_detailed.csv"),
        "col_name": "Mean_Log_Conc",
        "title": "(f) Baseline"
    },
    "Cluster3": {
        "stats_file": os.path.join(TRAINING_DIR, "03_Global_Results_Cluster3", "Global_Stats_Lev6_cluster3.csv"),
        "preds_file": os.path.join(TRAINING_DIR, "02_Model_Cluster3_Results", "predictions_detailed_cluster3.csv"),
        "col_name": "Mean_Log_Conc",
        "title": "(a) CT3 (n=15)"
    },
    "Cluster5": {
        "stats_file": os.path.join(TRAINING_DIR, "03_Global_Results_Cluster5", "Global_Stats_Lev6_cluster5.csv"),
        "preds_file": os.path.join(TRAINING_DIR, "02_Model_Cluster5_Results", "predictions_detailed_cluster5.csv"),
        "col_name": "Mean_Log_Conc",
        "title": "(b) CT5 (n=8)"
    },
    "Cluster7": {
        "stats_file": os.path.join(TRAINING_DIR, "03_Global_Results_Cluster7", "Global_Stats_Lev6_cluster7.csv"),
        "preds_file": os.path.join(TRAINING_DIR, "02_Model_Cluster7_Results", "predictions_detailed_cluster7.csv"),
        "col_name": "Mean_Log_Conc",
        "title": "(c) CT7 (n=5)"
    },
    "Jin5": {
        "stats_file": os.path.join(TRAINING_DIR, "03_Global_Results_Jin5", "Global_Stats_Lev6_Jin5.csv"),
        "preds_file": os.path.join(TRAINING_DIR, "02_Model_Jin5_Results", "predictions_detailed_Jin5.csv"),
        "col_name": "Mean_Log_Conc",
        "title": "(d) Jin5 (n=5)"
    },
    "SHAP5": {
        "stats_file": os.path.join(TRAINING_DIR, "03_Global_Results_SHAP_top5", "Global_Stats_Lev6_SHAP_top5.csv"),
        "preds_file": os.path.join(TRAINING_DIR, "02_Model_SHAP_top5_Results", "predictions_detailed_SHAP_top5.csv"),
        "col_name": "Mean_Log_Conc",
        "title": "(e) SHAP5 (n=5)"
    }
}

def calculate_metrics(preds_path):
    """Calculates R2 and RMSE from predictions file (CV predictions)."""
    if not os.path.exists(preds_path):
        # Try finding it based on standard pattern if generic path given
        return np.nan, np.nan
        
    try:
        df = pd.read_csv(preds_path)
        if 'Actual' in df.columns and 'CV_Pred' in df.columns:
            y_true = df['Actual']
            y_pred = df['CV_Pred']
        else:
            return np.nan, np.nan
            
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, rmse
    except Exception:
        return np.nan, np.nan

def plot_individual_maps():
    print("--- Generating Individual Maps ---")
    
    # Load Shapefile
    gdf = gpd.read_file(LEV06_SHP_PATH)
    gdf['PFAF_ID_Str'] = gdf['PFAF_ID'].astype(str)
    
    # Styling
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    
    for model_name, config in MODELS.items():
        if model_name == "Baseline": continue # Skip Baseline for individual maps if not requested (User asked for "contrast 5 and baseline" in stats, but maps for comparison usually imply the 5 variants. I'll stick to 5 maps + combined or maybe 6?)
        # User said "combine the other 5 except global". So skipping Baseline map here is correct.
        
        print(f"Processing {model_name}...")
        
        # 1. Calc Metrics
        r2, rmse = calculate_metrics(config.get('preds_file_real', config['preds_file']))
        
        # 2. Load Stats
        if not os.path.exists(config['stats_file']):
            print(f"Stats file missing for {model_name}")
            continue
            
        stats_df = pd.read_csv(config['stats_file'])
        stats_df['Lev6_ID'] = stats_df['Lev6_ID'].astype(str).str.split('.').str[0]
        
        # Merge
        merged = gdf.merge(stats_df, left_on='PFAF_ID_Str', right_on='Lev6_ID', how='left')
        
        # 3. Plot
        fig, ax = plt.subplots(figsize=(15, 10))
        merged.plot(ax=ax, color='lightgrey', edgecolor='none')
        
        # Valid Data
        col = config['col_name']
        valid = merged.dropna(subset=[col])
        
        if not valid.empty:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            
            valid.plot(ax=ax, column=col, cmap='YlOrRd', legend=True, cax=cax,
                       vmin=0, vmax=7,
                       legend_kwds={'label': r'Log$_{10}$(Abundance + 1) (items m$^{-3}$)', 'orientation': 'vertical'})
            
            cax.tick_params(labelsize=12)
            cax.set_ylabel(r'Log$_{10}$(Abundance + 1) (items m$^{-3}$)', fontsize=14, fontname='Times New Roman', fontweight='bold')

        # Annotation (R2, RMSE) bottom-left
        # Format: R² = 0.XX, RMSE = 0.XX (No Italics)
        annot_text = f"R\u00B2 = {r2:.2f}, RMSE = {rmse:.2f}"
        ax.text(0.02, 0.02, annot_text, transform=ax.transAxes, fontsize=16, 
                fontweight='bold', fontname='Times New Roman', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Title Logic: Split Tag and Name
        # config['title'] e.g. "(a) CT3 (n=15)"
        full_title = config['title']
        tag = full_title.split(' ')[0] # "(a)"
        name = " ".join(full_title.split(' ')[1:]) # "CT3 (n=15)"
        
        # 1. Tag at Top-Left (Outside)
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=20, fontweight='bold', fontname='Times New Roman', ha='left')
        
        # 2. Name Centered with Padding
        ax.set_title(name, fontsize=20, fontweight='bold', fontname='Times New Roman', loc='center', pad=20)
        
        # Grid and Axis
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Longitude", fontsize=16, fontweight='bold', fontname='Times New Roman')
        ax.set_ylabel("Latitude", fontsize=16, fontweight='bold', fontname='Times New Roman')
        
        outfile = os.path.join(OUTPUT_DIR, f"Map_Abundance_{model_name}_Refined.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {outfile}")

def plot_combined_map():
    print("\n--- Generating Combined 3x2 Map (Refined) ---")
    
    # Load Shapefile
    gdf = gpd.read_file(LEV06_SHP_PATH)
    gdf['PFAF_ID_Str'] = gdf['PFAF_ID'].astype(str)
    
    # 3x2 Grid
    # Tighter layout: adjust figsize height
    fig, axes = plt.subplots(3, 2, figsize=(20, 14)) 
    axes = axes.flatten()
    
    model_keys = ["Cluster3", "Cluster5", "Cluster7", "Jin5", "SHAP5"]
    
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Font Sizes (Increased and Bold)
    FONT_TITLE = 18
    FONT_LABEL = 16
    FONT_TICK = 14
    FONT_TAG = 20
    
    for i, model_name in enumerate(model_keys):
        ax = axes[i]
        config = MODELS[model_name]
        
        print(f"Adding {model_name} to combined plot...")
        
        # Metrics
        r2, rmse = calculate_metrics(config.get('preds_file_real', config['preds_file']))
        
        # Data
        if os.path.exists(config['stats_file']):
            stats_df = pd.read_csv(config['stats_file'])
            stats_df['Lev6_ID'] = stats_df['Lev6_ID'].astype(str).str.split('.').str[0]
            merged = gdf.merge(stats_df, left_on='PFAF_ID_Str', right_on='Lev6_ID', how='left')
            
            merged.plot(ax=ax, color='lightgrey', edgecolor='none')
            col = config['col_name']
            valid = merged.dropna(subset=[col])
            
            if not valid.empty:
                valid.plot(ax=ax, column=col, cmap='YlOrRd', vmin=0, vmax=7)
                
            # Annotation
            annot_text = f"R\u00B2 = {r2:.2f}, RMSE = {rmse:.2f}"
            ax.text(0.02, 0.05, annot_text, transform=ax.transAxes, fontsize=14, 
                    fontweight='bold', fontname='Times New Roman',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

        # Title Logic
        full_title = config['title']
        tag = full_title.split(' ')[0] # "(a)"
        name = " ".join(full_title.split(' ')[1:]) 

        # Tag: Left Diagonal Up (Outside)
        # x < 0 means left of y-axis. y > 1 means above.
        ax.text(-0.05, 1.05, tag, transform=ax.transAxes, fontsize=FONT_TAG, fontweight='bold', fontname='Times New Roman', ha='right')
        
        # Name Centered
        ax.set_title(name, fontsize=FONT_TITLE, fontweight='bold', fontname='Times New Roman', loc='center', pad=10)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Axis Labels (All plots)
        ax.set_xlabel("Longitude", fontname='Times New Roman', fontsize=FONT_LABEL, fontweight='bold')
        ax.set_ylabel("Latitude", fontname='Times New Roman', fontsize=FONT_LABEL, fontweight='bold')
        
        # Ticks
        ax.tick_params(axis='both', labelsize=FONT_TICK)
            
    # Hide the 6th empty subplot
    axes[5].axis('off')
    
    # Colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=7))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'Log$_{10}$(Abundance + 1) (items m$^{-3}$)', fontsize=18, fontweight='bold', fontname='Times New Roman')
    cbar.ax.tick_params(labelsize=16)
    
    # Tight layout logic
    # reduce hspace for compactness
    plt.subplots_adjust(wspace=0.15, hspace=0.25, right=0.9, top=0.95, left=0.08) 
    
    outfile = os.path.join(OUTPUT_DIR, "Map_Combined_Comparison.png")
    # Don't use bbox_inches='tight' if we manually adjusted subplots, or be careful
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Combined Map to {outfile}")

def plot_stats_comparison():
    print("\n--- Generating Statistical Comparison Chart ---")
    
    stats_data = []
    
    # Process all 6 models (Including Baseline)
    # Order: Baseline, CT3, CT5, CT7, Jin5, SHAP5
    order = ["Baseline", "Cluster3", "Cluster5", "Cluster7", "Jin5", "SHAP5"]
    
    for name in order:
        config = MODELS[name]
        path = config['stats_file']
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = config['col_name']
            if col in df.columns:
                valid_vals = df[col].dropna()
                
                # Linear Basis for Mean
                # Mean = Log10(Mean(10^x - 1) + 1)
                linear_vals = (10 ** valid_vals) - 1
                mean_lin = linear_vals.mean()
                mean_val = np.log10(mean_lin + 1)
                
                # SD: Keep as distribution spread in Log space (otherwise asymmetric)
                std_val = valid_vals.std()
                
                # Median/Quantiles (Invariant)
                median_val = valid_vals.median()
                q1 = valid_vals.quantile(0.25)
                q3 = valid_vals.quantile(0.75)
                iqr = q3 - q1
                
                # Use simplified label for chart
                if "Baseline" in name: display_name = "Baseline"
                elif "Cluster3" in name: display_name = "CT3"
                elif "Cluster5" in name: display_name = "CT5"
                elif "Cluster7" in name: display_name = "CT7"
                elif "Jin5" in name: display_name = "Jin5"
                elif "SHAP5" in name: display_name = "SHAP5"
                else: display_name = name
                
                stats_data.append({
                    "Model_Key": name,
                    "Model": display_name,
                    "Mean": mean_val,
                    "Std": std_val,
                    "Median": median_val,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr
                })
    
    if not stats_data:
        print("No stats data available.")
        return
        
    df_stats = pd.DataFrame(stats_data)
    
    # Save Detailed Stats to CSV
    stats_csv = os.path.join(OUTPUT_DIR, "Model_Comparison_Global_Stats.csv")
    df_stats.to_csv(stats_csv, index=False)
    print(f"Saved Global Stats to {stats_csv}")
    print(df_stats)
    print("Comparison Stats:")
    print(df_stats)
    
    # Plot Bar Chart with Error Bars
    plt.figure(figsize=(12, 7))
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Bar for Mean + Error Bar for SD
    bars = plt.bar(df_stats['Model'], df_stats['Mean'], yerr=df_stats['Std'], 
            capsize=5, color='skyblue', edgecolor='black', alpha=0.8, label='Mean ± SD')
            
    # Marker for Median
    plt.scatter(df_stats['Model'], df_stats['Median'], color='red', marker='D', s=50, zorder=3, label='Median')
    
    # Annotate R2 and RMSE at bottom of bars
    for i, row in df_stats.iterrows():
        # Use order list to retrieve the original key
        if i < len(order):
            m_key = order[i]
            config = MODELS[m_key]
            
            # Calculate metrics
            r2, rmse = calculate_metrics(config.get('preds_file_real', config['preds_file']))
            
            if not np.isnan(r2):
                label_text = f"R\u00B2={r2:.2f}\nRMSE={rmse:.2f}"
                plt.text(i, 0.1, label_text, ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    plt.ylabel(r"Global Mean Log$_{10}$(Abundance + 1) (items m$^{-3}$)", fontsize=18, fontweight='bold')
    plt.xlabel("Model", fontsize=18, fontweight='bold')
    plt.title("Comparison of Global Abundance Statistics", fontsize=20, fontweight='bold')
    
    # Ticks
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Legend Outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    out_chart = os.path.join(OUTPUT_DIR, "Comparison_Stats_Chart.png")
    plt.tight_layout()
    plt.savefig(out_chart, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Stats Chart to {out_chart}")

if __name__ == "__main__":
    plot_individual_maps()
    plot_combined_map()
    plot_stats_comparison()

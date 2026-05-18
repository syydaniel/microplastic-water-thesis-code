
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
TRAINING_DIR = os.path.join(PROJECT_DIR, "02_Training")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "04_Model_Comparison")

# Models Configuration (Comparing 5 Variants, No Baseline)
MODELS = {
    "Cluster3": {
        "preds_file_real": os.path.join(TRAINING_DIR, "02_Model_Cluster3_Results", "predictions_detailed_cluster3.csv"),
        "plot_title": "(a) CT3 (n=15)"
    },
    "Cluster5": {
        "preds_file_real": os.path.join(TRAINING_DIR, "02_Model_Cluster5_Results", "predictions_detailed_cluster5.csv"),
        "plot_title": "(b) CT5 (n=8)"
    },
    "Cluster7": {
        "preds_file_real": os.path.join(TRAINING_DIR, "02_Model_Cluster7_Results", "predictions_detailed_cluster7.csv"),
        "plot_title": "(c) CT7 (n=5)"
    },
    "Jin5": {
        "preds_file_real": os.path.join(TRAINING_DIR, "02_Model_Jin5_Results", "predictions_detailed_Jin5.csv"),
        "plot_title": "(d) Jin5 (n=5)"
    },
    "SHAP5": {
        "preds_file_real": os.path.join(TRAINING_DIR, "02_Model_SHAP_top5_Results", "predictions_detailed_SHAP_top5.csv"),
        "plot_title": "(e) SHAP5 (n=5)"
    }
}

ORDER = ["Cluster3", "Cluster5", "Cluster7", "Jin5", "SHAP5"]

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def main():
    print("Generating Combined Scatter Plot (Sensitivity Style)...")
    
    # 2 Rows, 3 Columns (5 plots + 1 empty)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # Matches sensitivity width/height ratio roughly
    axes = axes.flatten()
    
    # Matches Sensitivity Style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # Strict Math Font Settings
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['mathtext.default'] = 'regular'
    
    lims = [0, 8] # Match sensitivity range (0-8)
    
    for i, model_key in enumerate(ORDER):
        ax = axes[i]
        config = MODELS[model_key]
        path = config['preds_file_real']
        title = config['plot_title']
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Check cols
            if 'Actual' in df.columns and 'CV_Pred' in df.columns:
                y_true = df['Actual']
                y_pred = df['CV_Pred']
                
                # Metrics
                r2, rmse = calculate_metrics(y_true, y_pred)
                
                # Scatter Plot (Exact Sensitivity Colors)
                ax.scatter(y_true, y_pred, alpha=0.5, s=15, color='#1f77b4', label='CV Predictions')
                
                # 1:1 Line (Exact Sensitivity Style)
                ax.plot(lims, lims, 'k--', lw=2, label='1:1 Line')
                
                # Best Fit Line (Exact Sensitivity Style)
                slope, intercept = np.polyfit(y_true, y_pred, 1)
                fit_line = slope * np.array(lims) + intercept
                ax.plot(lims, fit_line, color='red', alpha=0.7, label=f'Best Fit (y={slope:.2f}x{intercept:+.2f})')
                
                # Style
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_aspect('equal')
                
                # Annotations
                # Title Logic (Split Tag and Name)
                parts = title.split(' ')
                tag = parts[0]
                name = " ".join(parts[1:])
                
                # Title Centered
                ax.set_title(name, fontsize=14, fontweight='bold', pad=10, fontname='Times New Roman')
                
                # Tag Top-Left Outside - Moved further top-left
                ax.text(-0.15, 1.1, tag, transform=ax.transAxes, fontsize=16, fontweight='bold', 
                        ha='left', va='top', fontname='Times New Roman')
                
                # Metrics Box (Exact Sensitivity Style)
                # "N = ..."
                stats_text = f"N = {len(df)}\nR\u00B2 = {r2:.2f}\nRMSE = {rmse:.2f}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                        fontsize=12, ha='left', va='top', fontname='Times New Roman',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Labels
                ax.set_xlabel(r"Observed Log$_{10}$(Abundance + 1)", fontsize=11, fontname='Times New Roman')
                ax.set_ylabel(r"Predicted Log$_{10}$(Abundance + 1)", fontsize=11, fontname='Times New Roman')
                    
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # Legend (Lower Right, small)
                ax.legend(loc='lower right', fontsize=9, frameon=True)
                
            else:
                ax.text(0.5, 0.5, "Columns Missing", ha='center')
        else:
            ax.text(0.5, 0.5, "File Missing", ha='center')
            print(f"File missing for {model_key}: {path}")

    # Hide the 6th empty subplot
    axes[5].axis('off')

    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Model_Performance_Scatter_Combined.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {outfile}")

if __name__ == "__main__":
    main()

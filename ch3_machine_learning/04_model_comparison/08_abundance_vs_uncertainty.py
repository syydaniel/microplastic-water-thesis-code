
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
COMP_DIR = os.path.join(PROJECT_DIR, "04_Model_Comparison")
INPUT_FILE = os.path.join(COMP_DIR, "Model_Comparison_CV.csv")
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "Raw data")
LEV06_SHP_PATH = os.path.join(RAW_DATA_DIR, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev06.shp")
OUTPUT_DIR = os.path.join(COMP_DIR, "02_Abundance_vs_Uncertainty")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("--- Abundance vs Uncertainty Analysis ---")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} basins.")
    
    # Use Baseline Linear Abundance
    abund_col = 'Mean_Linear_Calc_Baseline'
    cov_col = 'CoV'
    
    # 2. Calculate Thresholds
    # Log-transform for distribution visualization, but thresholds on Linear as requested?
    # User said: "Mean at ~900, Median at ~230". 
    # Let's calculate actuals.
    
    # Remove calculating limits from data, use user-defined limits
    # 2. Categorize Variables
    
    # A. CoV Groups (0 - 0.5, 0.5 - 1.0, 1.0 - 1.5, > 1.5)
    cov_bins = [0, 0.5, 1.0, 1.5, np.inf]
    cov_labels = ["CoV (0 - 0.5)", "CoV (0.5 - 1.0)", "CoV (1.0 - 1.5)", "CoV (> 1.5)"]
    df['CoV_Group'] = pd.cut(df[cov_col], bins=cov_bins, labels=cov_labels, right=False)
    
    # B. Abundance Groups (0-100, 100-1000, 1000-10000, >10000)
    abund_bins = [0, 100, 1000, 10000, np.inf]
    # User requested: Low, High, Very High, Extremely High
    abund_labels = ["Low (0 - 100)", "High (100 - 1,000)", "Very High (1,000 - 10,000)", "Extremely High (> 10,000)"]
    df['Abundance_Group'] = pd.cut(df[abund_col], bins=abund_bins, labels=abund_labels, right=False)
    
    # 3. Analysis: Cross-tabulation statistics
    print("\n--- Distribution of Abundance Groups within CoV Ranges ---")
    # Percentage Crosstab
    crosstab_pct = pd.crosstab(df['CoV_Group'], df['Abundance_Group'], normalize='index') * 100
    # Count Crosstab
    crosstab_count = pd.crosstab(df['CoV_Group'], df['Abundance_Group'])
    
    print(crosstab_pct)
    
    # Detailed stats for Excel
    cov_stats = df.groupby('CoV_Group', observed=True)[cov_col].describe()
    
    # 4. Visualizations
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['font.size'] = 12
    
    # Get total counts for labels
    group_counts = df['CoV_Group'].value_counts()
    new_index = [f"{label}\n(n={group_counts.get(label, 0)})" for label in crosstab_pct.index]
    crosstab_pct.index = new_index
    crosstab_count.index = new_index
    
    # A1. Stacked Bar Chart: Percentage
    ax = crosstab_pct.plot(kind='bar', stacked=True, figsize=(14, 10), colormap='viridis', width=0.7)
    
    # Tag (a)
    ax.text(-0.05, 1.05, "(a)", transform=ax.transAxes, fontsize=20, fontweight='bold', fontname='Times New Roman', ha='right')
    
    plt.title("Abundance Classification Composition (Percentage)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("")
    plt.ylabel("Percentage of Basins (%)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=0, ha='center', fontsize=12)
    plt.legend(title="Abundance Range (items m⁻³)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.subplots_adjust(bottom=0.35, right=0.75)
    plt.savefig(os.path.join(OUTPUT_DIR, "StackedBar_Abundance_Pct_CoV_Groups.png"), dpi=300)
    plt.close()
    
    # A2. Stacked Bar Chart: Absolute Counts
    ax = crosstab_count.plot(kind='bar', stacked=True, figsize=(14, 10), colormap='viridis', width=0.7)
    
    # Tag (b)
    ax.text(-0.05, 1.05, "(b)", transform=ax.transAxes, fontsize=20, fontweight='bold', fontname='Times New Roman', ha='right')
    
    plt.title("Abundance Classification Composition (Absolute Counts)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("")
    plt.ylabel("Number of Basins", fontsize=16, fontweight='bold')
    plt.xticks(rotation=0, ha='center', fontsize=12)
    plt.legend(title="Abundance Range (items m⁻³)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.subplots_adjust(bottom=0.35, right=0.75)
    plt.savefig(os.path.join(OUTPUT_DIR, "StackedBar_Abundance_Count_CoV_Groups.png"), dpi=300)
    plt.close()
    
    # B. Scatter Plot: PDP-style Bootstrap Fit (Abundance vs CoV)
    print("\n--- Generating Refined Scatter Plot (PDP Style) ---")
    plt.figure(figsize=(12, 8))
    
    # Data Preparation
    x_log = np.log10(df[abund_col] + 1)
    y_val = df[cov_col]
    
    # 1. Plot Background Points (Grey, low alpha, to show density)
    plt.scatter(x_log, y_val, alpha=0.15, s=15, color='grey', edgecolors='none', label='Basins')
    
    # 2. Bootstrap Smoothing (PDP Style)
    # User requested: n=20 bootstrap, 95% CI.
    # We will use Lowess (Locally Weighted Scatterplot Smoothing) to approximate the PDP curve.
    
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        has_statsmodels = True
    except ImportError:
        print("Statsmodels not found. Using Polynomial fit instead.")
        has_statsmodels = False
        
    n_boot = 20
    x_grid = np.linspace(x_log.min(), x_log.max(), 100)
    boot_preds = []
    
    # Explicit Bootstrap Loop
    np.random.seed(42) # Reproducibility
    indices = np.arange(len(x_log))
    
    for i in range(n_boot):
        # Resample
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        x_sample = x_log.iloc[boot_idx]
        y_sample = y_val.iloc[boot_idx]
        
        if has_statsmodels:
            # Lowess Fit (frac=0.3 for smoothing)
            # Returns sorted x and y
            z = lowess(y_sample, x_sample, frac=0.3, it=0)
            # Interpolate to grid
            from scipy.interpolate import interp1d
            # sort unique ps
            u_x, u_idx = np.unique(z[:, 0], return_index=True)
            u_y = z[u_idx, 1]
            f = interp1d(u_x, u_y, kind='linear', fill_value="extrapolate")
            y_pred = f(x_grid)
        else:
            # Polynomial Fit (Order 3)
            coeffs = np.polyfit(x_sample, y_sample, 3)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_grid)
            
        boot_preds.append(y_pred)
        
    boot_preds = np.array(boot_preds)
    
    # Calculate Mean and 95% CI
    y_mean = np.mean(boot_preds, axis=0)
    y_lower = np.percentile(boot_preds, 2.5, axis=0)
    y_upper = np.percentile(boot_preds, 97.5, axis=0)
    
    # Plot Mean Line
    plt.plot(x_grid, y_mean, color='black', linewidth=3, label='PDP Fit (Mean)')
    
    # Plot Confidence Interval
    plt.fill_between(x_grid, y_lower, y_upper, color='black', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel(r"Log$_{10}$(Abundance + 1) (items m$^{-3}$)", fontsize=16, fontweight='bold', fontname='Times New Roman')
    plt.ylabel("Coefficient of Variation (CoV)", fontsize=16, fontweight='bold', fontname='Times New Roman')
    plt.title("Relationship between Abundance and Uncertainty (PDP Fit)", fontsize=18, fontweight='bold', fontname='Times New Roman', pad=15)
    
    # Vertical Lines for Boundaries (Optional context)
    boundaries = [100, 1000, 10000]
    for b in boundaries:
        log_b = np.log10(b + 1)
        plt.axvline(log_b, color='black', linestyle=':', alpha=0.3, linewidth=1)
        
    plt.legend(fontsize=14, loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Tick Params
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Scatter_Abundance_vs_CoV_Refined.png"), dpi=300)
    plt.close()
    
    # C. Combined Figure (PDP Scatter + Stacked Bar)
    print("\n--- Generating Combined 1x2 Figure (Swapped) ---")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # --- Left Panel: PDP Scatter (ax1) ---
    ax1 = axes[0]
    
    # Reuse Data
    x_log = np.log10(df[abund_col] + 1)
    y_val = df[cov_col]
    
    # Background Points
    ax1.scatter(x_log, y_val, alpha=0.15, s=15, color='grey', edgecolors='none', label='Basins')
    
    # Bootstrap Fit logic
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        
    n_boot = 20
    x_grid = np.linspace(x_log.min(), x_log.max(), 100)
    boot_preds = []
    
    np.random.seed(42)
    indices = np.arange(len(x_log))
    
    for i in range(n_boot):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        x_sample = x_log.iloc[boot_idx]
        y_sample = y_val.iloc[boot_idx]
        
        if has_statsmodels:
            z = lowess(y_sample, x_sample, frac=0.3, it=0)
            from scipy.interpolate import interp1d
            u_x, u_idx = np.unique(z[:, 0], return_index=True)
            u_y = z[u_idx, 1]
            f = interp1d(u_x, u_y, kind='linear', fill_value="extrapolate")
            y_pred = f(x_grid)
        else:
            coeffs = np.polyfit(x_sample, y_sample, 3)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_grid)
        boot_preds.append(y_pred)
        
    boot_preds = np.array(boot_preds)
    y_mean = np.mean(boot_preds, axis=0)
    y_lower = np.percentile(boot_preds, 2.5, axis=0)
    y_upper = np.percentile(boot_preds, 97.5, axis=0)
    
    ax1.plot(x_grid, y_mean, color='black', linewidth=3, label='PDP Fit (Mean)')
    ax1.fill_between(x_grid, y_lower, y_upper, color='black', alpha=0.2, label='95% Confidence Interval')
    
    # Tag (a) - Moved up to avoid title overlap
    ax1.text(-0.05, 1.08, "(a)", transform=ax1.transAxes, fontsize=28, fontweight='bold', fontname='Times New Roman', ha='right')
    
    ax1.set_xlabel(r"Log$_{10}$(Abundance + 1) (items m$^{-3}$)", fontsize=20, fontweight='bold', fontname='Times New Roman', labelpad=15)
    ax1.set_ylabel("Coefficient of Variation (CoV)", fontsize=20, fontweight='bold', fontname='Times New Roman', labelpad=15)
    ax1.set_title("Relationship between Abundance and Uncertainty (PDP Fit)", fontsize=22, fontweight='bold', fontname='Times New Roman', pad=20)
    
    boundaries = [100, 1000, 10000]
    for b in boundaries:
        log_b = np.log10(b + 1)
        ax1.axvline(log_b, color='black', linestyle=':', alpha=0.3, linewidth=1)
        
    ax1.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, prop={'family':'Times New Roman', 'size':16})
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=16, pad=8)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname('Times New Roman')

    # --- Right Panel: Stacked Bar (ax2) ---
    ax2 = axes[1]
    crosstab_pct.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis', width=0.7)
    
    # Tag (b) - Moved up
    ax2.text(-0.05, 1.08, "(b)", transform=ax2.transAxes, fontsize=28, fontweight='bold', fontname='Times New Roman', ha='right')
    
    ax2.set_title("Abundance Classification Composition (Percentage)", fontsize=22, fontweight='bold', fontname='Times New Roman', pad=20)
    ax2.set_xlabel("") # No X Label
    ax2.set_ylabel("Percentage of Basins (%)", fontsize=20, fontweight='bold', fontname='Times New Roman', labelpad=15)
    ax2.tick_params(axis='x', rotation=0, labelsize=16, pad=8)
    ax2.tick_params(axis='y', labelsize=16, pad=8) 
    
    # Bold Tick Labels and ensure Font Family
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')
    for label in ax2.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax2.legend(title="Abundance Range (items m⁻³)", 
               bbox_to_anchor=(1.02, 1.0), 
               loc='upper left', 
               prop={'size':14, 'weight':'normal', 'family':'Times New Roman'}, 
               title_fontproperties={'size':16, 'weight':'normal', 'family':'Times New Roman'})
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    # Adjust spacing for legends
    plt.subplots_adjust(bottom=0.25, wspace=0.5, right=0.85)
    plt.savefig(os.path.join(OUTPUT_DIR, "Combined_Abundance_Uncertainty.png"), dpi=300)
    plt.close()

    print("\n--- Generating Maps ---")
    if os.path.exists(LEV06_SHP_PATH):
        gdf = gpd.read_file(LEV06_SHP_PATH)
        gdf['PFAF_ID_Str'] = gdf['PFAF_ID'].astype(str)
        df['Lev6_ID'] = df['Lev6_ID'].astype(str)
        
    # C. Refined Maps (Matching Global Structure)
    print("\n--- Generating Refined Maps ---")
    if os.path.exists(LEV06_SHP_PATH):
        gdf = gpd.read_file(LEV06_SHP_PATH)
        gdf['PFAF_ID_Str'] = gdf['PFAF_ID'].astype(str)
        df['Lev6_ID'] = df['Lev6_ID'].astype(str)
        
        merged = gdf.merge(df, left_on='PFAF_ID_Str', right_on='Lev6_ID', how='left')
        
        # Helper for Refined Map Plotting
        def plot_categorical_map(gdf, column, title, cmap_colors, labels, filename):
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Base Layer
            gdf.plot(ax=ax, color='lightgrey', edgecolor='none')
            
            # Create a numeric column for plotting with categorical colors
            # We map the categorical column labels to 0..N-1
            # Ensure the column is ordered categorical
            if not isinstance(gdf[column].dtype, pd.CategoricalDtype):
                 # Should be categorical already, but safety check
                 pass
            
            # Create Custom Colormap
            from matplotlib.colors import ListedColormap
            import matplotlib.patches as mpatches
            
            cmap = ListedColormap(cmap_colors)
            
            # Plot Valid Data
            valid = gdf.dropna(subset=[column])
            if not valid.empty:
                # We plot by mapping categories to integer codes
                # The colors will be assigned by index
                valid['plot_val'] = valid[column].cat.codes
                valid.plot(ax=ax, column='plot_val', cmap=cmap, legend=False)
                
                # Custom Legend
                patches = []
                for i, label in enumerate(labels):
                    color = cmap_colors[i] if i < len(cmap_colors) else 'black'
                    patches.append(mpatches.Patch(color=color, label=label))
                
                # Legend Position: Lower Left outside or inside?
                # Global structure: usually inside if possible or smart.
                # User asked for "global graph structure". 
                # Let's put it at lower left with frame.
                ax.legend(handles=patches, loc='lower left', title=title, 
                          fontsize=14, title_fontsize=16, frameon=True,
                          bbox_to_anchor=(0.02, 0.02))
            
            # Styling (Global Structure)
            ax.set_title(title, fontsize=20, fontweight='bold', fontname='Times New Roman', loc='center', pad=20)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel("Longitude", fontsize=16, fontweight='bold', fontname='Times New Roman')
            ax.set_ylabel("Latitude", fontsize=16, fontweight='bold', fontname='Times New Roman')
            
            # Tick Params
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {filename}")

        # 1. Abundance Map
        # Colors: Blue -> Cyan -> Orange -> Red
        abund_colors = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c'] 
        plot_categorical_map(
            merged, 
            'Abundance_Group', 
            "Global Microplastic Abundance Levels", 
            abund_colors, 
            abund_labels, 
            os.path.join(OUTPUT_DIR, "Map_Abundance_Categories_Refined.png")
        )
        
        # 2. CoV Map
        # Colors: Dark Green (Low) -> Light Green (Rel Low) -> Yellow (Med) -> Red (High)
        cov_colors = ['#1a9850', '#91cf60', '#fee08b', '#d73027']
        plot_categorical_map(
            merged, 
            'CoV_Group', 
            "Model Uncertainty Levels (CoV)", 
            cov_colors, 
            cov_labels, 
            os.path.join(OUTPUT_DIR, "Map_CoV_Categories_Refined.png")
        )
        
    # Excel Export
    xls_path = os.path.join(OUTPUT_DIR, "Abundance_Uncertainty_Refined.xlsx")
    with pd.ExcelWriter(xls_path) as writer:
        crosstab_pct.to_excel(writer, sheet_name='CoV_Abundance_Pct')
        crosstab_count.to_excel(writer, sheet_name='CoV_Abundance_Count')
        cov_stats.to_excel(writer, sheet_name='CoV_Stats')
        df[['Lev6_ID', abund_col, cov_col, 'CoV_Group', 'Abundance_Group']].to_excel(writer, sheet_name='Detailed_Data', index=False)
            
    print(f"Data saved to {xls_path}")

if __name__ == "__main__":
    main()

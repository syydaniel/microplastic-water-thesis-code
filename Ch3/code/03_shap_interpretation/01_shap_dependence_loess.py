import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os

# --- Configuration ---
# Fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 12

# Paths
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
INPUT_CSV = os.path.join(PROJECT_DIR, "01_Data_Prep", "01_Data_Combine_Result", "data_combine.csv")
SHAP_CSV = os.path.join(PROJECT_DIR, "02_Training", "02_Model_Results", "shap_values.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "04_SHAP_Visual_Analysis")
PLOT_DIR = os.path.join(OUTPUT_DIR, "Dependence_Plots_LOESS")
os.makedirs(PLOT_DIR, exist_ok=True)

# Abbreviation Map (Scientific + Units, Bold Upright)
ABBREVIATION_MAP = {
    'Natural Discharge Upstream': r'ND$_{\mathbf{UP}}$ (m$^{\mathbf{3}}$/s)',
    'Land Surface Runoff Local': r'LSR$_{\mathbf{SUB}}$ (mm)',
    'Lake Volume Upstream': r'LV$_{\mathbf{UP}}$ (km$^{\mathbf{3}}$)',
    'Reservoir Volume Upstream': r'ReV$_{\mathbf{UP}}$ (km$^{\mathbf{3}}$)',
    'River Area Local': r'RA$_{\mathbf{SUB}}$ (km$^{\mathbf{2}}$)',
    'River Area Upstream': r'RA$_{\mathbf{UP}}$ (km$^{\mathbf{2}}$)',
    'River Volume Local': r'RiV$_{\mathbf{SUB}}$ (km$^{\mathbf{3}}$)',
    'River Volume Upstream': r'RiV$_{\mathbf{UP}}$ (km$^{\mathbf{3}}$)',
    'Elevation Local': r'ELE$_{\mathbf{SUB}}$ (m)',
    'Elevation Upstream': r'ELE$_{\mathbf{UP}}$ (m)',
    'Terrain Slope Local': r'TS$_{\mathbf{SUB}}$ (deg)',
    'Terrain Slope Upstream': r'TS$_{\mathbf{UP}}$ (deg)',
    'Stream Gradient Local': r'SG$_{\mathbf{SUB}}$ (m/km)',
    'Temperature Local': r'TEMP$_{\mathbf{SUB}}$ ($^{\mathbf{\circ}}$C)',
    'Temperature Upstream': r'TEMP$_{\mathbf{UP}}$ ($^{\mathbf{\circ}}$C)',
    'Precipitation Local': r'PREP$_{\mathbf{SUB}}$ (mm)',
    'Precipitation Upstream': r'PREP$_{\mathbf{UP}}$ (mm)',
    'Potential Evap Local': r'PET$_{\mathbf{SUB}}$ (mm)',
    'Potential Evap Upstream': r'PET$_{\mathbf{UP}}$ (mm)',
    'Actual Evap Local': r'AET$_{\mathbf{SUB}}$ (mm)',
    'Actual Evap Upstream': r'AET$_{\mathbf{UP}}$ (mm)',
    'Cropland Extent Local': r'CROPL$_{\mathbf{SUB}}$ (%)',
    'Cropland Extent Upstream': r'CROPL$_{\mathbf{UP}}$ (%)',
    'Pasture Extent Local': r'PASTURE$_{\mathbf{SUB}}$ (%)',
    'Pasture Extent Upstream': r'PASTURE$_{\mathbf{UP}}$ (%)',
    'Wetland All Local': r'WLA$_{\mathbf{SUB}}$ (%)',
    'Wetland All Upstream': r'WLA$_{\mathbf{UP}}$ (%)',
    'Wetland Land Local': r'WLL$_{\mathbf{SUB}}$ (%)',
    'Wetland Land Upstream': r'WLL$_{\mathbf{UP}}$ (%)',
    'Population Local': r'POP$_{\mathbf{SUB}}$ (people/km$^{\mathbf{2}}$)', 
    'Population Upstream': r'POP$_{\mathbf{UP}}$ (people/km$^{\mathbf{2}}$)',
    'Urban Extent Local': r'URBAN$_{\mathbf{SUB}}$ (%)',
    'Urban Extent Upstream': r'URBAN$_{\mathbf{UP}}$ (%)',
    'Road Density Local': r'RD$_{\mathbf{SUB}}$ (km/km$^{\mathbf{2}}$)',
    'Road Density Upstream': r'RD$_{\mathbf{UP}}$ (km/km$^{\mathbf{2}}$)',
    'Human Footprint Local': r'HFI$_{\mathbf{SUB}}$',
    'Human Footprint Upstream': r'HFI$_{\mathbf{UP}}$',
    'Human Dev Index Local': r'HDI$_{\mathbf{SUB}}$',
    'Human Dev Index Upstream': r'HDI$_{\mathbf{UP}}$'
}

def get_label(name):
    # Normalize
    clean = name.replace('_', ' ').strip()
    return ABBREVIATION_MAP.get(clean, clean)

def main():
    print("--- 1. Loading Data ---")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found: {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    
    # Filter valid target (Exact logic from training script)
    df = df[df['Std_Value_m3'] > 0].copy()
    df = df.dropna(subset=['Std_Value_m3'])
    
    info_cols = ['Unique_ID', 'Std_Value_m3', 'Gear_Category', 'Target_Log', 'HYBAS_ID', 'geometry', 'index_right']
    predictors = [c for c in df.columns if c not in info_cols]
    
    X = df[predictors].reset_index(drop=True)
    print(f"Feature Data Loaded: {X.shape}")
    
    print("--- 2. Loading SHAP Values ---")
    if not os.path.exists(SHAP_CSV):
        print(f"Error: SHAP file not found: {SHAP_CSV}")
        return
        
    shap_df = pd.read_csv(SHAP_CSV)
    print(f"SHAP Data Loaded: {shap_df.shape}")
    
    if X.shape[0] != shap_df.shape[0]:
        print(f"WARNING: Shape mismatch! X={X.shape}, SHAP={shap_df.shape}. Visualizations may be incorrect if rows do not align.")
        if abs(X.shape[0] - shap_df.shape[0]) < 100:
             print("Small difference, proceeding (assuming mostly aligned).")
        else:
             print("Large difference. Please verify data sources.")
             # Assuming standard use case we proceed but matched by index
    
    # Ensure column alignment
    common_cols = [c for c in X.columns if c in shap_df.columns]
    if len(common_cols) < len(predictors) * 0.5:
        print("Warning: Column names seem different between X and SHAP. Assuming order is preserved.")
    
    # --- 3. Run for ALL Features (Ranked) ---
    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)
    all_features_ranked = mean_abs_shap.index.tolist()
    print(f"Generating Plots for {len(all_features_ranked)} features...")
    
    # --- 4. Dependence Plots (All Features) ---
    print("\n--- Generating Dependence Plots (LOESS + Linear) ---")
    
    for feature in all_features_ranked:
        if feature not in X.columns:
            print(f"Skipping {feature} (not in Input Data)")
            continue
            
        x_val = X[feature]
        y_val = shap_df[feature]
        
        mask = ~np.isnan(x_val) & ~np.isnan(y_val) & (x_val != -99)
        x_plot = x_val[mask]
        y_plot = y_val[mask]
        
        if len(x_plot) < 10:
            print(f"Skipping {feature} (Not enough valid data after filtering -99)")
            continue
            
        # --- Pre-processing: Two-Tailed 95% Truncation ---
        # Keep data between 2.5th and 97.5th percentiles to remove outliers
        lower_q = np.percentile(x_plot, 2.5)
        upper_q = np.percentile(x_plot, 97.5)
        
        # Create mask for the 95% range
        p_mask = (x_plot >= lower_q) & (x_plot <= upper_q)
        
        # Apply filter
        x_plot = x_plot[p_mask]
        y_plot = y_plot[p_mask]
        
        if len(x_plot) < 10:
             print(f"Skipping {feature} (Not enough data after 95% truncation)")
             continue
        
        plt.figure(figsize=(8, 6))
        
        # Check for Log Scale (if max > 1e6)
        use_log = False
        # Calculate new range stats
        x_min = x_plot.min()
        x_max = x_plot.max()
        
        if x_max > 1e6 and x_min > 0:
            use_log = True
        
        # 1. Scatter
        plt.scatter(x_plot, y_plot, alpha=0.3, color='#1f77b4', s=15, edgecolor='none', label='Data Points (95% Range)')
        
        # 2. Linear Fit & Stats
        # For linear fit, if log scale is used, we should fit on log(x)? 
        # User asked for "linear fit" generally, but fitting linear on log data visual might look curved on linear or straight on log.
        # Usually standard linear fit y = mx + c is on the raw data. 
        # But if we view in log scale, a straight line y=mx+c looks curved.
        # However, standard practice is to keep the stat on raw data unless specified. 
        # I will fit on raw data.
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_plot, y_plot)
        
        # Generate line points
        if use_log:
            # Create geomspace for smooth line on log scale
            line_x = np.geomspace(x_plot.min(), x_plot.max(), 100)
        else:
            line_x = np.linspace(x_plot.min(), x_plot.max(), 100)
            
        line_y = slope * line_x + intercept
        
        # Add Linear Line with Equation
        sign = "+" if intercept >= 0 else "-"
        eq_str = f"$y={slope:.2e}x {sign} {abs(intercept):.3f}$" # Use sci notation for slope as x includes 1e6
        if not use_log: # Only concise slope if small
             if abs(slope) > 0.001:
                 eq_str = f"$y={slope:.3f}x {sign} {abs(intercept):.3f}$"
                 
        plt.plot(line_x, line_y, color='black', linestyle='--', linewidth=2, 
                 label=f'Linear Fit: {eq_str}\n($R^2$={r_value**2:.2f})')
        
        # 3. LOESS Fit
        # frac=0.3 provides a reasonable smoothing window
        # Note: Lowess on log-scale data might surely be better if data spans orders of magnitude, 
        # but standard lowess on raw data is robust. 
        lowess = sm.nonparametric.lowess
        z = lowess(y_plot, x_plot, frac=0.3)
        
        # Sort for rolling calc
        simulated_data = pd.DataFrame({'x': x_plot, 'y': y_plot})
        simulated_data = simulated_data.sort_values(by='x')
        
        # Rolling Mean/SD for Range/CI visualization
        # Dynamic window size: 10% of data, but ensure min_periods to avoid gaps
        window = int(len(simulated_data) * 0.1)
        if window < 10: window = 10
        
        # min_periods=window/2 ensures we get values closer to edges
        rolling_mean = simulated_data['y'].rolling(window=window, center=True, min_periods=int(window/3)).mean()
        rolling_std = simulated_data['y'].rolling(window=window, center=True, min_periods=int(window/3)).std()
        
        upper_bound = rolling_mean + 1.96 * rolling_std
        lower_bound = rolling_mean - 1.96 * rolling_std
        
        # Plot LOESS Curve
        plt.plot(z[:,0], z[:,1], color='#d62728', linewidth=2.5, label='LOESS Trend')
        
        # Plot Range (95% CI)
        plt.fill_between(simulated_data['x'], lower_bound, upper_bound, color='#d62728', alpha=0.15, label='95% Confidence Interval')
        
        if use_log:
            # Use SymLog to handle 0 and show 0.01, 0.1...
            # linthresh=0.01 allows showing 0 and then starting log scale from 0.01
            plt.xscale('symlog', linthresh=0.01)
            
            # Generate ticks: 0, then powers of 10 from -2 up to log10(max)
            max_log = int(np.ceil(np.log10(x_plot.max())))
            # Start from 10^-2 (0.01)
            powers = np.arange(-2, max_log + 1)
            log_ticks = [10.0**p for p in powers]
            ticks = [0] + log_ticks
            
            plt.xticks(ticks)
            
            # Format Labels
            def log_fmt(x, pos):
                if x == 0: return "$0$"
                # Check for 0.01, 0.1, 1, 10...
                log_val = np.log10(x)
                if abs(log_val - round(log_val)) < 1e-5:
                   p = int(round(log_val))
                   return f"$10^{{{p}}}$"
                return ""
                
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
            print(f"  -> Applied SymLog Scale (0, 10^-2...) for {feature} (Max: {x_plot.max():.2e})")
        
        # Styling
        plt.xlabel(get_label(feature), fontsize=14, fontname='Times New Roman', fontweight='bold')
        plt.ylabel(f"SHAP Value\n(Impact on Log Abundance)", fontsize=14, fontname='Times New Roman', fontweight='bold')
        plt.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        # Legend (Standard Font)
        plt.legend(frameon=False, loc='best', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.4)
        
        # Save
        safe_name = feature.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(PLOT_DIR, f"Dependence_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    # --- 5. All Features Linear Fit Statistics ---
    print("\n--- Calculating Linear Fit Statistics for All Features ---")
    linear_stats = []
    
    for feature in shap_df.columns:
        if feature not in X.columns:
            continue
            
        x_val = X[feature]
        y_val = shap_df[feature]
        
        mask = ~np.isnan(x_val) & ~np.isnan(y_val)
        if mask.sum() < 10:
            continue
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_val[mask], y_val[mask])
        
        # Calculate RMSE of the linear fit
        y_pred = slope * x_val[mask] + intercept
        rmse = np.sqrt(((y_val[mask] - y_pred)**2).mean())
        
        linear_stats.append({
            'Feature': feature,
            'Slope': slope,
            'Intercept': intercept,
            'R2': r_value**2,
            'RMSE': rmse,
            'P_Value': p_value,
            'Std_Err': std_err
        })
        
    stats_df = pd.DataFrame(linear_stats)
    stats_df = stats_df.sort_values(by='R2', ascending=False)
    
    stats_out = os.path.join(OUTPUT_DIR, "Linear_Fit_Stats_All_Features.csv")
    stats_df.to_csv(stats_out, index=False)
    print(f"Saved Linear Stats to: {stats_out}")
    print("\nTop 5 Linear Associations (by R2):")
    print(stats_df[['Feature', 'R2', 'P_Value']].head())

if __name__ == "__main__":
    main()

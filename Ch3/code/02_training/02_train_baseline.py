import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set font family
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 12

# Paths
INPUT_CSV = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis\01_Data_Prep\01_Data_Combine_Result\data_combine.csv"
OUTPUT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis\02_Training\02_Model_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'final_model.pkl')
SHAP_PATH = os.path.join(OUTPUT_DIR, 'shap_values.csv')
PREDS_PATH = os.path.join(OUTPUT_DIR, 'predictions_detailed.csv')

# --- User Standardization Config ---
ABBREVIATION_MAP = {
    'Natural Discharge Upstream': r'$ND_{UP}$',
    'Land Surface Runoff Local': r'$LSR_{SUB}$',
    'Lake Volume Upstream': r'$LV_{UP}$',
    'Reservoir Volume Upstream': r'$ReV_{UP}$',
    'River Area Local': r'$RA_{SUB}$',
    'River Area Upstream': r'$RA_{UP}$',
    'River Volume Local': r'$RiV_{SUB}$',
    'River Volume Upstream': r'$RiV_{UP}$',
    'Elevation Local': r'$ELE_{SUB}$',
    'Elevation Upstream': r'$ELE_{UP}$',
    'Terrain Slope Local': r'$TS_{SUB}$',
    'Terrain Slope Upstream': r'$TS_{UP}$',
    'Stream Gradient Local': r'$SG_{SUB}$',
    'Temperature Local': r'$TEMP_{SUB}$',
    'Temperature Upstream': r'$TEMP_{UP}$',
    'Precipitation Local': r'$PREP_{SUB}$',
    'Precipitation Upstream': r'$PREP_{UP}$',
    'Potential Evap Local': r'$PET_{SUB}$',
    'Potential Evap Upstream': r'$PET_{UP}$',
    'Actual Evap Local': r'$AET_{SUB}$',
    'Actual Evap Upstream': r'$AET_{UP}$',
    'Cropland Extent Local': r'$CROPL_{SUB}$',
    'Cropland Extent Upstream': r'$CROPL_{UP}$',
    'Pasture Extent Local': r'$PASTURE_{SUB}$',
    'Pasture Extent Upstream': r'$PASTURE_{UP}$',
    'Wetland All Local': r'$WLA_{SUB}$',
    'Wetland All Upstream': r'$WLA_{UP}$',
    'Wetland Land Local': r'$WLL_{SUB}$',
    'Wetland Land Upstream': r'$WLL_{UP}$',
    'Population Local': r'$POP_{SUB}$', 
    'Population Upstream': r'$POP_{UP}$',
    'Urban Extent Local': r'$URBAN_{SUB}$',
    'Urban Extent Upstream': r'$URBAN_{UP}$',
    'Road Density Local': r'$RD_{SUB}$',
    'Road Density Upstream': r'$RD_{UP}$',
    'Human Footprint Local': r'$HFI_{SUB}$',
    'Human Footprint Upstream': r'$HFI_{UP}$',
    'Human Dev Index Local': r'$HDI_{SUB}$',
    'Human Dev Index Upstream': r'$HDI_{UP}$' # Added just in case
}

def get_abbreviation(name):
    clean_name = name.replace('_', ' ').strip()
    return ABBREVIATION_MAP.get(clean_name, clean_name)


# 1. Load and Prepare Data
print("Loading data...")
df = pd.read_csv(INPUT_CSV)

# Filter valid target
df = df[df['Std_Value_m3'] > 0].copy()
df = df.dropna(subset=['Std_Value_m3'])

# Log Transformation (Base 10)
df['Target_Log'] = np.log10(df['Std_Value_m3'] + 1)

# Define Predictors (Exclude ID, Target, Gear_Category)
info_cols = ['Unique_ID', 'Std_Value_m3', 'Gear_Category', 'Target_Log', 'HYBAS_ID', 'geometry', 'index_right']
predictors = [c for c in df.columns if c not in info_cols]

print(f"Number of predictors: {len(predictors)}")

X = df[predictors]
y = df['Target_Log']

# 2. Check for Existing Results vs Training
if os.path.exists(MODEL_PATH) and os.path.exists(SHAP_PATH) and os.path.exists(PREDS_PATH):
    print("Found existing model, SHAP values, and predictions. Skipping training...")
    final_model = joblib.load(MODEL_PATH)
    shap_df = pd.read_csv(SHAP_PATH)
    shap_values = shap_df.values
    preds_df = pd.read_csv(PREDS_PATH)
    
    # Reload metrics from df
    r2_cv = r2_score(y, preds_df['CV_Pred'])
    rmse_cv = np.sqrt(mean_squared_error(y, preds_df['CV_Pred']))
    r2_train = r2_score(y, preds_df['Train_Pred'])
    rmse_train = np.sqrt(mean_squared_error(y, preds_df['Train_Pred']))
    
else:
    print("Training model...")
    
    # Optuna (Condensed)
    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': 2000
        }
        cv = KFold(n_splits=10, shuffle=True, random_state=42) #n_splits=10 means 10 folds
        rmse_scores = []
        for train_idx, val_idx in cv.split(X, y):
            train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
            val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model = lgb.train(params, train_data, valid_sets=[val_data], callbacks=callbacks)
            preds = model.predict(X.iloc[val_idx])
            rmse_scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize', storage='sqlite:///optuna.db', study_name='lgbm_opt', load_if_exists=True)
    if len(study.trials) < 50: # Ensure at least 50 trials
         trials_needed = 50 - len(study.trials)
         print(f"Running {trials_needed} more optimization trials...")
         study.optimize(objective, n_trials=trials_needed)

    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbosity': -1, 'n_estimators': 2000})

    # Cross-Validation Predictions
    cv_predictions = np.zeros(len(X))
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in cv.split(X, y):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model_cv = lgb.train(best_params, train_data, valid_sets=[val_data], callbacks=callbacks)
        cv_predictions[val_idx] = model_cv.predict(X.iloc[val_idx])

    # Full Fit
    X_t, X_v, y_t, y_v =  pd.read_csv(INPUT_CSV, nrows=0), None, None, None # dummy
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.1, random_state=42)
    train_d = lgb.Dataset(X_t, label=y_t)
    val_d = lgb.Dataset(X_v, label=y_v, reference=train_d)
    callbacks = [lgb.early_stopping(stopping_rounds=50)]
    final_model = lgb.train(best_params, train_d, valid_sets=[val_d], callbacks=callbacks)
    
    train_predictions = final_model.predict(X)
    
    # Metrics
    r2_cv = r2_score(y, cv_predictions)
    rmse_cv = np.sqrt(mean_squared_error(y, cv_predictions))
    r2_train = r2_score(y, train_predictions)
    rmse_train = np.sqrt(mean_squared_error(y, train_predictions))
    
    # Save Everything
    preds_df = pd.DataFrame({'Actual': y, 'Train_Pred': train_predictions, 'CV_Pred': cv_predictions})
    preds_df.to_csv(PREDS_PATH, index=False)
    joblib.dump(final_model, MODEL_PATH)
    
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    pd.DataFrame(shap_values, columns=predictors).to_csv(SHAP_PATH, index=False)

print(f"CV R2: {r2_cv:.4f}, CV RMSE: {rmse_cv:.4f}")

# 3. Refined Visualizations

# 3. Refined Visualizations

# Load Category Mapping from Excel
# --- 3. Visualizations ---

# Hardcoded Category Map based on User Table
CATEGORY_MAP = {
    'Hydrology': ['Natural Discharge Upstream', 'Land Surface Runoff Local', 'Lake Volume Upstream', 'Reservoir Volume Upstream', 
                  'River Area Local', 'River Area Upstream', 'River Volume Local', 'River Volume Upstream',
                  'Natural_Discharge_Upstream', 'Land_Surface_Runoff_Local', 'Lake_Volume_Upstream', 'Reservoir_Volume_Upstream',
                  'River_Area_Local', 'River_Area_Upstream', 'River_Volume_Local', 'River_Volume_Upstream'],
    'Physiography': ['Elevation Local', 'Elevation Upstream', 'Terrain Slope Local', 'Terrain Slope Upstream', 'Stream Gradient Local',
                     'Elevation_Local', 'Elevation_Upstream', 'Terrain_Slope_Local', 'Terrain_Slope_Upstream', 'Stream_Gradient_Local'],
    'Climate': ['Temperature Local', 'Temperature Upstream', 'Precipitation Local', 'Precipitation Upstream', 
                'Potential Evap Local', 'Potential Evap Upstream', 'Actual Evap Local', 'Actual Evap Upstream',
                'Temperature_Local', 'Temperature_Upstream', 'Precipitation_Local', 'Precipitation_Upstream',
                'Potential_Evap_Local', 'Potential_Evap_Upstream', 'Actual_Evap_Local', 'Actual_Evap_Upstream'],
    'Landcover': ['Cropland Extent Local', 'Cropland Extent Upstream', 'Pasture Extent Local', 'Pasture Extent Upstream', 
                  'Wetland All Local', 'Wetland All Upstream', 'Wetland Land Local', 'Wetland Land Upstream',
                  'Cropland_Extent_Local', 'Cropland_Extent_Upstream', 'Pasture_Extent_Local', 'Pasture_Extent_Upstream',
                  'Wetland_All_Local', 'Wetland_All_Upstream', 'Wetland_Land_Local', 'Wetland_Land_Upstream'],
    'Anthropogenic': ['Population Local', 'Population Upstream', 'Urban Extent Local', 'Urban Extent Upstream', 
                      'Road Density Local', 'Road Density Upstream', 'Human Footprint Local', 'Human Footprint Upstream', 'Human Dev Index Local',
                      'Population_Local', 'Population_Upstream', 'Urban_Extent_Local', 'Urban_Extent_Upstream',
                      'Road_Density_Local', 'Road_Density_Upstream', 'Human_Footprint_Local', 'Human_Footprint_Upstream', 'Human_Dev_Index_Local']
}

# Invert Map for Lookup
FEATURE_TO_CAT = {}
for cat, feats in CATEGORY_MAP.items():
    for f in feats:
        FEATURE_TO_CAT[f] = cat
        FEATURE_TO_CAT[f.replace('_', ' ')] = cat # Ensure spaces

# Colors
CAT_COLORS = {
    'Hydrology': '#1f77b4',       # Blue
    'Physiography': '#7f7f7f',    # Grey
    'Climate': '#ff7f0e',         # Orange
    'Landcover': '#2ca02c',       # Green
    'Anthropogenic': '#d62728',   # Red
    'Other': '#333333'
}

def get_category_info(feature_name):
    clean_name = feature_name.replace('_', ' ').strip()
    cat = FEATURE_TO_CAT.get(clean_name, 'Other')
    return CAT_COLORS.get(cat, '#333333'), cat


# A. SHAP Bar Plot (All Features + Legend)
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({'Feature': predictors, 'Importance': mean_shap})
shap_importance = shap_importance.sort_values(by='Importance', ascending=True)

# Generate Abbreviated Labels for Plotting
shap_importance['Abbrev'] = shap_importance['Feature'].apply(get_abbreviation)

# Map Colors and Categories
final_colors = []
final_cats = []
for f in shap_importance['Feature']:
    c, cat = get_category_info(f)
    final_colors.append(c)
    final_cats.append(cat)

# Create Legend Handles
import matplotlib.patches as mpatches
unique_final_cats = list(set(final_cats))
unique_final_cats.sort()
legend_handles = []
for cat in unique_final_cats:
    legend_handles.append(mpatches.Patch(color=CAT_COLORS.get(cat, '#333333'), label=cat))

plt.figure(figsize=(15, 25)) 
plt.barh(shap_importance['Abbrev'], shap_importance['Importance'], color=final_colors)
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
plt.title("Feature Importance (All Predictors)")
plt.legend(handles=legend_handles, title="Category", loc='lower right')
plt.yticks(name='Times New Roman', fontsize=12) # Apply font explicitly to y-ticks just in case
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar_all_features.png'), dpi=300)
plt.close()


# B. Standard Prediction Scatter Plot (Actual vs Predicted)
# We use CV predictions for the main scatter as it represents generalization
plt.figure(figsize=(8, 8))
plt.scatter(y, preds_df['CV_Pred'], alpha=0.5, s=15, color='#1f77b4', label='CV Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='1:1 Line') # Identity line

# Add best fit line
m, b = np.polyfit(y, preds_df['CV_Pred'], 1)
plt.plot(y, m*y + b, color='red', alpha=0.7, label=f'Best Fit (y={m:.2f}x+{b:.2f})')

plt.xlim(0, 8)
plt.ylim(0, 8)
plt.xlabel(r"Observed $Log_{10}(Abundance + 1)$", fontsize=14, fontname='Times New Roman')
plt.ylabel(r"Predicted $Log_{10}(Abundance + 1)$", fontsize=14, fontname='Times New Roman')
plt.title(f"Model Performance (Cross-Validation)\n$R^2$ = {r2_cv:.2f}, RMSE = 0.67", fontname='Times New Roman', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_scatter_scientific.png'), dpi=300)
plt.close()

# C. SHAP Pie Charts (Nested Donut: Inner=Category, Outer=Scale)
# Hierarchical Aggregation
# Structure: {Category: {Scale: Importance}}
hier_data = {}
cat_total_imp = {}

for f, imp in zip(shap_importance['Feature'], shap_importance['Importance']):
    c, cat = get_category_info(f)
    
    # Determine Scale (SUB vs UP)
    # Heuristic based on name
    lower_f = f.lower()
    if 'local' in lower_f or 'sub' in lower_f or 'stream gradient' in lower_f or 'hdi' in lower_f:
        scale = 'SUB'
    elif 'upstream' in lower_f or 'up' in lower_f:
        scale = 'UP'
    else:
        scale = 'Other'
        
    if cat not in hier_data: hier_data[cat] = {}
    hier_data[cat][scale] = hier_data[cat].get(scale, 0) + imp
    cat_total_imp[cat] = cat_total_imp.get(cat, 0) + imp

# Prepare Vectors
sorted_cats = sorted(cat_total_imp, key=cat_total_imp.get, reverse=True)
inner_labels = sorted_cats
inner_sizes = [cat_total_imp[c] for c in sorted_cats]
inner_colors = [CAT_COLORS.get(c, '#cccccc') for c in sorted_cats]

outer_labels = []
outer_sizes = []
outer_colors = []

def adjust_color(c, scale):
    import matplotlib.colors as mc
    base = mc.to_rgb(c)
    if scale == 'SUB': 
        return tuple(max(0, x * 0.7) for x in base) # Darker
    elif scale == 'UP': 
        return (*base, 0.5) # Lighter
    else: return (*base, 0.3)

for cat in sorted_cats:
    scales_data = hier_data[cat]
    # Order: SUB, UP, Other
    unique_scales = sorted(scales_data.keys(), key=lambda x: (0 if x=='SUB' else 1 if x=='UP' else 2))
    
    for s in unique_scales:
        outer_labels.append(f"{cat} - {s}")
        outer_sizes.append(scales_data[s])
        outer_colors.append(adjust_color(inner_colors[inner_labels.index(cat)], s))

fig, ax = plt.subplots(figsize=(10, 12)) 
wedges_out, texts_out, autotexts_out = ax.pie(outer_sizes, radius=1.0, colors=outer_colors, 
                                              wedgeprops=dict(width=0.3, edgecolor='w'),
                                              autopct='%1.1f%%', pctdistance=0.85, labels=None)
plt.setp(autotexts_out, size=18, weight="bold", color="white")

wedges_in, texts_in, autotexts_in = ax.pie(inner_sizes, radius=0.7, colors=inner_colors, 
                                            wedgeprops=dict(width=0.3, edgecolor='w'),
                                            autopct='%1.1f%%', pctdistance=0.75, labels=None)
plt.setp(autotexts_in, size=24, weight="bold", color="white")

ax.set_title("Feature Importance: Category (Inner) & Scale (Outer)")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=CAT_COLORS.get(cat, 'gray'), label=cat) for cat in sorted_cats]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
          ncol=3, title="Category", fontsize=22, title_fontsize=20, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_nested_pie.png'), dpi=300)
plt.close()

# D. SHAP Beeswarm (All Features)
# Use Abbreviated Names
X_abbrev = X.copy()
X_abbrev.columns = [get_abbreviation(c) for c in X.columns]

plt.rcParams['font.family'] = 'Times New Roman' # Ensure Font
shap.summary_plot(shap_values, X_abbrev, max_display=len(predictors), show=False, plot_size=(9,11))

plt.title("SHAP Summary Plot (All Features)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm_all.png'), dpi=300)
plt.close()

print("Refined visualizations generated.")


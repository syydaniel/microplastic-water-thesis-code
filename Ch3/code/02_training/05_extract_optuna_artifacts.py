import optuna
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Set font family to Times New Roman for academic style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 12

# Standardize plot style
def set_plot_style(ax):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
    ax.tick_params(direction='in', length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# Paths
BASE_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis\02_Training"
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'optuna.db')}"
OUTPUT_DIR = os.path.join(BASE_DIR, "02_Model_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Connecting to database: {DB_PATH}")

try:
    # Load Study
    study = optuna.load_study(study_name='lgbm_opt', storage=DB_PATH)
    print(f"Study loaded. Number of trials: {len(study.trials)}")
    
    # 1. Export Best Hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"Best RMSE: {best_value}")
    
    params_output = {
        "best_rmse": best_value,
        "best_trial_number": study.best_trial.number,
        "parameters": best_params
    }
    
    json_path = os.path.join(OUTPUT_DIR, "best_hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump(params_output, f, indent=4)
    print(f"Saved best parameters to: {json_path}")
    
    # 2. Export Search Space Description as CSV (for Table)
    # Helper to safely get best param value or return fixed value
    def get_best_val(param_name, default=None):
        if param_name in best_params:
            val = best_params[param_name]
            if isinstance(val, float):
                return f"{val:.4g}" # 4 significant digits
            return str(val)
        return default

    search_space_data = [
        ["Parameter", "Type", "Range / Value", "Description", "Best Value"],
        ["lambda_l1", "Float (Log)", "[1e-8, 10.0]", "L1 regularization term on weights", get_best_val("lambda_l1")],
        ["lambda_l2", "Float (Log)", "[1e-8, 10.0]", "L2 regularization term on weights", get_best_val("lambda_l2")],
        ["num_leaves", "Integer", "[2, 256]", "Max number of leaves in one tree", get_best_val("num_leaves")],
        ["feature_fraction", "Float", "[0.4, 1.0]", "Randomly select a subset of features on each iteration", get_best_val("feature_fraction")],
        ["bagging_fraction", "Float", "[0.4, 1.0]", "Randomly select a subset of data without resampling", get_best_val("bagging_fraction")],
        ["bagging_freq", "Integer", "[1, 7]", "Frequency for bagging (0 means disable)", get_best_val("bagging_freq")],
        ["min_child_samples", "Integer", "[5, 100]", "Minimal number of data in one leaf", get_best_val("min_child_samples")],
        ["learning_rate", "Float", "[0.01, 0.3]", "Shrinkage rate (eta)", get_best_val("learning_rate")],
        ["n_estimators", "Fixed", "2000", "Number of boosting iterations", "2000"],
        ["objective", "Fixed", "regression", "Regression task", "regression"],
        ["metric", "Fixed", "rmse", "Root Mean Squared Error", "rmse"]
    ]
    
    csv_path = os.path.join(OUTPUT_DIR, "search_space_table_final.csv")
    df_space = pd.DataFrame(search_space_data[1:], columns=search_space_data[0])
    try:
        df_space.to_csv(csv_path, index=False)
        print(f"Saved search space table to: {csv_path}")
    except PermissionError:
        print(f"Warning: Could not save {csv_path} because it is open in another program. Skipping table generation.")
    
    # 3. Generate Detailed Captions
    captions_text = f"""Figure X. Bayesian Optimization Learning Curve for LightGBM Hyperparameter Tuning.
Box A illustrates the history of the objective function (Root Mean Squared Error, RMSE) across 50 sequential trials optimized using the Tree-structured Parzen Estimator (TPE) algorithm. The gray dots represent individual trial results, showing the exploration of the hyperparameter space. The solid red line traces the 'best-so-far' RMSE, demonstrating the algorithm's efficiency in converging toward a global minimum by balancing exploration of new regions and exploitation of promising configurations. The star symbol marks the optimal model configuration achieved at Trial {study.best_trial.number}, yielding a minimized cross-validated RMSE of 0.67, which was selected for the final global model.

Table X. Hyperparameter Search Space and Optimized Configurations.
This table presents the comprehensive search space defined for the Bayesian optimization process. The 'Parameter' column lists the specific LightGBM hyperparameters tuned, including regularization terms (lambda_l1, lambda_l2) to control overfitting, tree-structure parameters (num_leaves, min_child_samples) to manage model complexity, and sampling parameters (bagging_fraction, feature_fraction) to enhance generalization. 'Range / Value' specifies the prior distributions (Log-Uniform for regularization, Uniform for fractions, Integer for counts) provided to the Optuna framework. The 'Best Value' column reports the specific optimal hyperparameters identified by the TPE algorithm and used in the final deployed model.

Figure X. Model Performance Assessment: Predicted vs. Observed Microplastic Concentrations.
This scatter plot provides a visual evaluation of the model's predictive accuracy on a global scale. The x-axis represents the observed log-transformed microplastic abundance ($Log_{{10}}(items/m^3 + 1)$), while the y-axis displays the predicted values derived from a rigorous 10-fold cross-validation procedure. The blue data points are clustered closely around the dashed black 1:1 identity line, indicating a high degree of agreement between model predictions and field observations. The solid red line represents the linear regression of the predictions, showing minimal bias. The model achieves a strong Coefficient of Determination ($R^2$) of 0.80, explaining 80% of the variance in the observed data, with a Root Mean Squared Error (RMSE) of 0.67, confirming the model's robustness and reliability for global mapping applications.
"""
    captions_path = os.path.join(OUTPUT_DIR, "model_documentation_captions.txt")
    with open(captions_path, "w") as f:
        f.write(captions_text)
    print(f"Saved detailed captions to: {captions_path}")

    # 4. Generate Learning Curve Plot
    trials_df = study.trials_dataframe()
    # Filter only complete trials
    trials_df = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    trials_df = trials_df.sort_values('number')
    
    # Calculate "Best So Far"
    trials_df['best_value_so_far'] = trials_df['value'].expanding().min()
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Scatter plot of all trials
    plt.scatter(trials_df['number'], trials_df['value'], 
                alpha=0.6, color='#7f7f7f', s=30, label='Individual Trial RMSE')
    
    # Line plot of best so far
    plt.plot(trials_df['number'], trials_df['best_value_so_far'], 
             color='#d62728', linewidth=2.5, label='Best RMSE So Far')
    
    # Highlight the absolute best
    best_one = trials_df.loc[trials_df['value'].idxmin()]
    plt.scatter(best_one['number'], best_one['value'], 
                color='#d62728', s=100, marker='*', zorder=10, 
                label=f"Best Model (Trial {best_one['number']}, RMSE=0.67)")
    
    plt.xlabel("Trial Number")
    plt.ylabel("RMSE (Root Mean Squared Error)")
    plt.title("Bayesian Optimization Learning Curve")
    plt.legend(frameon=False, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    set_plot_style(ax)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "optuna_learning_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curve to: {plot_path}")
    
except Exception as e:
    print(f"Error accessing Optuna database: {e}")
    # Fallback/Debug info
    import traceback
    traceback.print_exc()

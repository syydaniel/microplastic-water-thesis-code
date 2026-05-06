import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Use relative paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR) # Go up one level from 04_SHAP...
SHAP_CSV = os.path.join(PROJECT_DIR, "02_Training", "02_Model_Results", "shap_values.csv")

def main():
    print("--- Calculating Top 6 Feature Contribution ---")
    
    target_csv = SHAP_CSV
    if not os.path.exists(target_csv):
        print(f"Error: SHAP file not found: {target_csv}")
        # Fallback for manual path if relative fails for some reason
        # Try constructing based on known user path
        alt_path = r"/Volumes/TU200Pro/Chapter 3 mapping and analysis/02_Training/02_Model_Results/shap_values.csv"
        if os.path.exists(alt_path):
             print(f"Found at alternate path: {alt_path}")
             target_csv = alt_path
        else:
             return

    print(f"Loading SHAP values from: {target_csv}")
    shap_df = pd.read_csv(target_csv)
    print(f"SHAP Data Shape: {shap_df.shape}")

    # Calculate Mean Absolute SHAP Value (Global Feature Importance)
    feature_importance = np.abs(shap_df).mean()
    feature_importance = feature_importance.sort_values(ascending=False)
    
    total_importance = feature_importance.sum()
    
    top_6 = feature_importance.head(6)
    top_6_sum = top_6.sum()
    
    contribution_percent = (top_6_sum / total_importance) * 100
    
    print("\n--- Results ---")
    print(f"Total Feature Importance (Sum of Mean(|SHAP|)): {total_importance:.4f}")
    print(f"Top 6 Feature Importance Sum: {top_6_sum:.4f}")
    print(f"Top 6 Contribution: {contribution_percent:.2f}%")
    
    print("\n--- Top 6 Features Detail ---")
    for i, (feature, score) in enumerate(top_6.items(), 1):
        print(f"{i}. {feature}: {score:.4f} ({score/total_importance*100:.2f}%)")

if __name__ == "__main__":
    main()

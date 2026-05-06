import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 0. Configuration
# =============================================================================
project_dir = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
corr_matrix_path = os.path.join(project_dir, "02_Training", "02_Feature_selecting_result", "correlation_matrix.csv")
shap_values_path = os.path.join(project_dir, "02_Training", "02_Model_Results", "shap_values.csv")
output_dir = os.path.join(project_dir, "02_Training", "02_Feature_selecting_result")

thresholds = [0.7, 0.3, 0.5] # Distance thresholds (dissimilarity = 1 - |r|)

# 1. Load Data
# =============================================================================
print("Loading data...")
# Load Correlation Matrix
# It seems the index might be the variable names, check first col
corr_df = pd.read_csv(corr_matrix_path, index_col=0)
print(f"Correlation matrix shape: {corr_df.shape}")

# Load SHAP Values
shap_df = pd.read_csv(shap_values_path)
print(f"SHAP values shape: {shap_df.shape}")

# 2. Calculate SHAP Importance
# =============================================================================
print("Calculating SHAP Importance...")
# Compute Mean Absolute SHAP value for each feature
shap_importance = shap_df.abs().mean().sort_values(ascending=False)
shap_importance_df = shap_importance.to_frame(name='mean_abs_shap')
print("Top 5 Variables by SHAP Importance:")
print(shap_importance_df.head())

# 3. Perform Hierarchical Clustering
# =============================================================================
print("Performing Hierarchical Clustering...")
# Ensure correlation matrix only contains the features present in SHAP (intersection)
# In case there are differences, align them.
common_features = [f for f in corr_df.index if f in shap_importance.index]
print(f"Number of common features: {len(common_features)}")

corr_matrix_aligned = corr_df.loc[common_features, common_features]
dissimilarity = 1 - np.abs(corr_matrix_aligned.values)

# Force diagonal to 0 just in case
np.fill_diagonal(dissimilarity, 0)

# Convert to condensed matrix for linkage (required by scipy)
condensed_dist = squareform(dissimilarity)

# Perform linkage using 'average' method as per previous script
linkage_matrix = linkage(condensed_dist, method='average')

# 4. Selection Logic
# =============================================================================
results = {}

for t in thresholds:
    print(f"\n--- Processing Threshold t={t} ---")
    # Form clusters
    # criterion='distance' means clusters are formed so that cophenetic distance between any two original observations in the same cluster is no more than t
    cluster_labels = fcluster(linkage_matrix, t=t, criterion='distance')
    
    # Create a DataFrame to manage clusters
    cluster_df = pd.DataFrame({
        'Feature': common_features,
        'Cluster': cluster_labels
    })
    
    # Add SHAP importance to this DF
    cluster_df['SHAP_Imp'] = cluster_df['Feature'].map(shap_importance)
    
    selected_features = []
    
    print(f"Found {cluster_df['Cluster'].nunique()} clusters.")
    
    for cluster_id in sorted(cluster_df['Cluster'].unique()):
        # Get all features in this cluster
        members = cluster_df[cluster_df['Cluster'] == cluster_id]
        
        # Sort by SHAP Importance descending
        members_sorted = members.sort_values(by='SHAP_Imp', ascending=False)
        
        # Pick the best one
        best_feature = members_sorted.iloc[0]['Feature']
        selected_features.append(best_feature)
        
        # Optional: Print detail if cluster has > 1 member
        if len(members) > 1:
            print(f"  Cluster {cluster_id}: Selected '{best_feature}' ({members_sorted.iloc[0]['SHAP_Imp']:.4f}) over {len(members)-1} others.")
            # print(f"    Others: {members_sorted.iloc[1:]['Feature'].tolist()}")
    
    results[t] = selected_features
    
    # Save selection list to CSV
    output_filename = f"selected_vars_threshold_{t}.csv"
    output_path = os.path.join(output_dir, output_filename)
    pd.Series(selected_features, name='Selected_Features').to_csv(output_path, index=False)
    print(f"Saved {len(selected_features)} selected features to {output_filename}")

# 5. Summary Output
# =============================================================================
print("\n=== Summary of Selections ===")
for t in thresholds:
    features = results[t]
    print(f"Threshold {t}: {len(features)} variables selected.")
    # print(features)

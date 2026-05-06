
import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "04_Model_Comparison")
DATA_DIR = os.path.join(BASE_DIR, "02_Training")

# 1. Define Model Paths
MODELS = {
    "Baseline": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Refined", "Global_Stats_Lev6.csv"),
        "col": "Mean_Log_Conc"
    },
    "Cluster3": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Cluster3", "Global_Stats_Lev6_cluster3.csv"),
        "col": "Mean_Log_Conc"
    },
    "Cluster5": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Cluster5", "Global_Stats_Lev6_cluster5.csv"),
        "col": "Mean_Log_Conc"
    },
    "Cluster7": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Cluster7", "Global_Stats_Lev6_cluster7.csv"),
        "col": "Mean_Log_Conc"
    },
    "Jin5": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Jin5", "Global_Stats_Lev6_Jin5.csv"),
        "col": "Mean_Log_Conc"
    },
    "SHAP5": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_SHAP_top5", "Global_Stats_Lev6_SHAP_top5.csv"),
        "col": "Mean_Log_Conc"
    },
    # Agreement File
    "Agreement": {
        "file": os.path.join(OUTPUT_DIR, "Global_Results_Agreement_Levels.csv"),
        "col": "Agreement_Category" # 1: Low, 2: Median, 3: Mean
    }
}

def load_data():
    data = {}
    for name, config in MODELS.items():
        if os.path.exists(config['file']):
            df = pd.read_csv(config['file'])
            # Ensure Lev6_ID is string and clean
            if 'Lev6_ID' in df.columns:
                df['Lev6_ID'] = df['Lev6_ID'].astype(str).str.split('.').str[0]
                df = df.set_index('Lev6_ID')
                data[name] = df[config['col']]
            elif 'PFAF_ID' in df.columns: # Agreement file might use PFAF_ID
                 df['PFAF_ID'] = df['PFAF_ID'].astype(str).str.split('.').str[0]
                 df = df.set_index('PFAF_ID')
                 data[name] = df[config['col']]
            else:
                print(f"Warning: ID column not found in {name}")
        else:
            print(f"Warning: File not found for {name}: {config['file']}")
    return data

def get_classifications(series, name):
    """
    Returns sets of indices for:
    - Above Mean
    - Above Median
    - Below Median (Strictly less than median)
    """
    if name == "Agreement":
        # Special logic for Agreement Category
        # 3: All > Mean
        # 2: All > Median (includes > Mean usually? In the script it was tiered)
        # 1: All < Median
        # 0: No Consensus
        
        # Let's assume inclusive definitions for "Above Median"
        above_mean = set(series[series == 3].index)
        above_median = set(series[series >= 2].index) # 2 and 3
        below_median = set(series[series == 1].index)
        
        # Note: 0 is ignored here
        return above_mean, above_median, below_median, series.mean(), series.median() # Mean/Median of categories is meaningless but returning for consistency structure
        
    else:
        # Standard Continuous Data
        valid = series.dropna()
        mean_val = valid.mean()
        median_val = valid.median()
        
        above_mean = set(valid[valid > mean_val].index)
        above_median = set(valid[valid > median_val].index)
        below_median = set(valid[valid <= median_val].index) # Or < ? User said "below". Usually <= median is the complement of > median.
        
        return above_mean, above_median, below_median, mean_val, median_val

def main():
    print("Loading data...")
    data_dict = load_data()
    
    if "Baseline" not in data_dict:
        print("Error: Baseline data missing.")
        return

    # 1. Get Baseline Sets
    base_mean_set, base_med_set, base_low_set, base_mean_val, base_med_val = get_classifications(data_dict["Baseline"], "Baseline")
    
    print(f"Baseline Stats: Mean={base_mean_val:.4f}, Median={base_med_val:.4f}")
    print(f"Baseline Counts: >Mean: {len(base_mean_set)}, >Median: {len(base_med_set)}, <Median: {len(base_low_set)}")
    
    results = []
    
    # 2. Compare Everyone to Baseline
    # Order: Agreement, then Models
    model_order = ["Agreement", "Cluster3", "Cluster5", "Cluster7", "Jin5", "SHAP5"]
    
    for model in model_order:
        if model not in data_dict: continue
        
        # Get Model's own sets
        m_mean_set, m_med_set, m_low_set, m_mean_val, m_med_val = get_classifications(data_dict[model], model)
        
        # Calculate Overlaps with Baseline
        # Note: Overlap means "Model says X is > Mean" AND "Baseline says X is > Mean"
        # Since Baseline is the truth, we want to know how many of the Model's "High" spots are actually "High" in Baseline?
        # Or how many of Baseline's "High" spots did the Model find? 
        # "agreement needs to compare with baseline... above mean (n globe has how many [Base], now agree has how many [Model], and overlap)"
        
        # Overlap > Mean
        overlap_mean = len(m_mean_set.intersection(base_mean_set))
        pct_mean = (overlap_mean / len(base_mean_set)) * 100 if len(base_mean_set) > 0 else 0 # Precision/Recall? Usually % of Baseline is Recall.
        # User asked for: "n globe has ... agree has ... coincide ?? ... ratio?"
        # Let's provide: n(Model), n(Baseline), n(Overlap), % Overlap (relative to Baseline)
        
        # Overlap > Median
        overlap_med = len(m_med_set.intersection(base_med_set))
        pct_med = (overlap_med / len(base_med_set)) * 100 if len(base_med_set) > 0 else 0
        
        # Overlap < Median
        overlap_low = len(m_low_set.intersection(base_low_set))
        pct_low = (overlap_low / len(base_low_set)) * 100 if len(base_low_set) > 0 else 0
        
        row = {
            "Model": model,
            "Total_Basins": len(data_dict[model].dropna()),
            
            # MEAN
            "Base_n_Above_Mean": len(base_mean_set),
            "Model_n_Above_Mean": len(m_mean_set),
            "Overlap_Above_Mean": overlap_mean,
            "Recall_Above_Mean (%)": round(pct_mean, 2),
            
            # MEDIAN
            "Base_n_Above_Median": len(base_med_set),
            "Model_n_Above_Median": len(m_med_set),
            "Overlap_Above_Median": overlap_med,
            "Recall_Above_Median (%)": round(pct_med, 2),
            
            # BELOW MEDIAN
            "Base_n_Below_Median": len(base_low_set), # Baseline Only
            "Model_n_Below_Median": len(m_low_set),
            "Overlap_Below_Median": overlap_low,
            "Recall_Below_Median (%)": round(pct_low, 2)
        }
        results.append(row)
        
    df_res = pd.DataFrame(results)
    
    # Save
    outfile = os.path.join(OUTPUT_DIR, "Detailed_Agreement_Statistics.xlsx")
    df_res.to_excel(outfile, index=False)
    print(f"Saved detailed stats to {outfile}")
    print(df_res)

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
import itertools

# Paths
BASE_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "04_Model_Comparison")
DATA_DIR = os.path.join(BASE_DIR, "02_Training")

# Output Paths
CSV_OUT = os.path.join(OUTPUT_DIR, "Grouped_Statistical_Stats.csv")
PLOT_OUT = os.path.join(OUTPUT_DIR, "Grouped_Significance_Comparison.png")

# Models
MODELS = {
    "Baseline": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Refined", "Global_Stats_Lev6.csv"),
        "col": "Mean_Log_Conc"
    },
    "CT3": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Cluster3", "Global_Stats_Lev6_cluster3.csv"),
        "col": "Mean_Log_Conc"
    },
    "CT5": {
        "file": os.path.join(DATA_DIR, "03_Global_Results_Cluster5", "Global_Stats_Lev6_cluster5.csv"),
        "col": "Mean_Log_Conc"
    },
    "CT7": {
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
    }
}

# Bins (Linear Scale)
# 0-10, 10-100, 100-1000, 1000-10000, >10000
BINS = [0, 10, 100, 1000, 10000, np.inf]
LABELS = ["0-10", "10-100", "100-1,000", "1,000-10,000", "> 10,000"]

def get_letters(p_values, models, alpha=0.05):
    """
    Generates Compact Letter Display (CLD) for significance.
    Simple logic: 
    1. Initialize all same letter 'a'.
    2. If sig diff, need different letters.
    This is complex to implement perfectly from scratch. 
    Simplified approach: 
    - Group models that are NOT significantly different.
    """
    # Placeholder for robust CLD:
    # 1. Create adjacency matrix (1=Not Sig Diff, 0=Sig Diff)
    # 2. Find maximal cliques?
    # Let's try a simpler greedy approach which is common in R's cld.
    
    # Initialize dictionary
    letters = {m: "" for m in models}
    
    # Sort models by median?
    # No, letters don't imply order, just grouping.
    
    # Full Pairwise Matrix
    # We have p_values for (m1, m2).
    # If p > alpha -> Same group.
    
    # Let's just output distinct letters for now based on a basic grouping set.
    # Actually, let's use a simpler visualization if CLD is too hard:
    # Just mark * vs Baseline?
    # User specifically asked for 'a', 'b'.
    
    # Attempting a basic CLD algorithm:
    # Start with one group containing all. 
    # If any pair in group is diff, split? No.
    
    # Alternative:
    # Use networkx to find cliques of "Indistinguishable" models.
    # Each clique gets a letter.
    # A model can be in multiple cliques (e.g., 'ab').
    
def get_letters(p_values, models, alpha=0.05):
    """
    Generates Compact Letter Display (CLD) for significance.
    Uses a simple clique-finding approach since N=6 (very small).
    Two models share a letter if they are NOT significantly different.
    1. Build adjacency matrix (1 = Not Sig Diff).
    2. Find all maximal cliques.
    3. Assign letters to cliques.
    """
    n = len(models)
    adj = {m: set() for m in models}
    
    # Build Graph
    for i in range(n):
        for j in range(i+1, n):
            m1, m2 = models[i], models[j]
            # Key might be (m1, m2) or (m2, m1)
            p = p_values.get((m1, m2), p_values.get((m2, m1), 1.0))
            if p > alpha: # Not Significant -> Share a letter (Edge)
                adj[m1].add(m2)
                adj[m2].add(m1)
    
    # Bron-Kerbosch Algorithm for Maximal Cliques
    def bron_kerbosch(R, P, X):
        if not P and not X:
            yield R
        while P:
            v = P.pop()
            yield from bron_kerbosch(R.union({v}), 
                                     P.intersection(adj[v]), 
                                     X.intersection(adj[v]))
            X.add(v)

    # Convert to set for the algo
    P = set(models)
    R = set()
    X = set()
    
    cliques = list(bron_kerbosch(R, P, X))
    
    # Sort cliques to make letter assignment somewhat deterministic/ordered
    # e.g., by the sort order of models contained
    cliques.sort(key=lambda c: sorted(list(c))[0])
    
    node_letters = {m: [] for m in models}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    for i, clq in enumerate(cliques):
        char = alphabet[i]
        for node in clq:
            node_letters[node].append(char)
            
    # Join
    final_letters = {}
    for m in models:
        # Sort letters to be nice "ab" instead of "ba"
        chars = sorted(list(set(node_letters[m])))
        final_letters[m] = "".join(chars)
        
    return final_letters

def main():
    # 1. Load Data
    print("Loading data...")
    merged = None
    
    for name, config in MODELS.items():
        if os.path.exists(config['file']):
            df = pd.read_csv(config['file'])
            # Clean ID
            if 'Lev6_ID' in df.columns:
                df['Lev6_ID'] = df['Lev6_ID'].astype(str).str.split('.').str[0]
                df = df.set_index('Lev6_ID')
            elif 'PFAF_ID' in df.columns:
                 df['PFAF_ID'] = df['PFAF_ID'].astype(str).str.split('.').str[0]
                 df = df.set_index('PFAF_ID')
            
            series = df[config['col']]
            series.name = name
            
            if merged is None:
                merged = pd.DataFrame(series)
            else:
                merged = merged.join(series, how='inner') # Intersection of basins
    
    merged = merged.dropna()
    print(f"Total Basins for comparison: {len(merged)}")
    
    # 2. Linear Conversion (for Binning)
    # Log10(x+1) -> 10^x - 1
    # We define bins based on "Baseline" Linear Abundance
    merged['Linear_Base'] = (10 ** merged['Baseline']) - 1
    
    # 3. Assign Groups
    merged['Group'] = pd.cut(merged['Linear_Base'], bins=BINS, labels=LABELS, right=True)
    
    # 4. Analysis per Group
    stats_rows = []
    
    # Prepare Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True) # 6 Groups?
    axes = axes.flatten()
    
    plt.rcParams['font.family'] = 'Times New Roman'
    
    model_cols = list(MODELS.keys()) # ["Baseline", "CT3", ...]
    
    for i, label in enumerate(LABELS):
        group_data = merged[merged['Group'] == label]
        n = len(group_data)
        fraction = (n / len(merged)) * 100
        
        print(f"\nProcessing Group: {label} (n={n}, {fraction:.2f}%)")
        
        stats_rows.append({
            "Group": label,
            "Count": n,
            "Percentage": fraction
        })
        
        if n < 5:
            # Too few samples for stats
            ax = axes[i]
            ax.text(0.5, 0.5, f"{label}\nn={n}\n(Too few data)", ha='center', va='center')
            continue
            
        # Stats Test (Friedman)
        # Check if samples are identical?
        data_arrays = [group_data[m] for m in model_cols]
        try:
            stat, p_global = friedmanchisquare(*data_arrays)
        except ValueError:
            p_global = 1.0 # Identical inputs
            
        # Pairwise (Wilcoxon) if global sig
        p_values = {}
        if p_global < 0.05:
            # Pairwise
            for m1, m2 in itertools.combinations(model_cols, 2):
                try:
                    s, p = wilcoxon(group_data[m1], group_data[m2])
                except ValueError: 
                    p = 1.0
                p_values[(m1, m2)] = p
                
            # Adjust P-values (Bonferroni)
            correction = len(p_values)
            p_values_adj = {k: min(v * correction, 1.0) for k, v in p_values.items()}
            
            # Get Letters
            letters = get_letters(p_values_adj, model_cols)
        else:
            letters = {m: "a" for m in model_cols}
            
        # --- Add Detailed Stats to CSV ---
        row_base = {
            "Group": label,
            "Count": n,
            "Percentage": f"{fraction:.2f}%"
        }
        # For each model, add Mean/Median/SD
        for m in model_cols:
            # Linear Basis for Mean
            # Mean = Log10(Mean(10^x - 1) + 1)
            lin_vals = (10 ** group_data[m]) - 1
            mean_lin = lin_vals.mean()
            row_base[f"{m}_Mean"] = np.log10(mean_lin + 1)
            
            row_base[f"{m}_Median"] = group_data[m].median()
            row_base[f"{m}_SD"] = group_data[m].std()
            row_base[f"{m}_Sig"] = letters[m]
            
        stats_rows.append(row_base)
            
        # Plotting
        ax = axes[i]
        
        # Melt for Seaborn
        melted = group_data[model_cols].melt(var_name='Model', value_name='LogAbundance')
        
        sns.boxplot(data=melted, x='Model', y='LogAbundance', ax=ax, color="skyblue", width=0.6, showfliers=False)
        
        # Annotate Letters
        # Get Y-axis limits
        y_max = melted['LogAbundance'].max()
        if pd.isna(y_max): y_max = 0
        y_min = melted['LogAbundance'].min()
        if pd.isna(y_min): y_min = 0
        y_range = y_max - y_min
        if y_range == 0: y_range = 1
        offset = y_range * 0.05
        
        for idx, m in enumerate(model_cols):
            l = letters[m]
            # Place above the max of that box
            m_max = group_data[m].max()
            # If outliers hidden, max might be q3 + 1.5iqr. 
            q1 = group_data[m].quantile(0.25)
            q3 = group_data[m].quantile(0.75)
            iqr = q3 - q1
            whisker = min(m_max, q3 + 1.5*iqr)
            
            if pd.isna(whisker): whisker = m_max
            
            ax.text(idx, whisker + offset, l, ha='center', va='bottom', fontweight='bold', fontsize=12)
            
        # Titles and Labels
        ax.set_title(f"Range: {label}\n(n={n}, {fraction:.1f}%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("")
        if i % 3 == 0:
            ax.set_ylabel(r"Log$_{10}$ Abundance", fontsize=12)
        else:
            ax.set_ylabel("")
            
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    print(f"Saved plot to {PLOT_OUT}")
    
    # Save CSV
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(CSV_OUT, index=False)
    print(f"Saved stats to {CSV_OUT}")

if __name__ == "__main__":
    main()

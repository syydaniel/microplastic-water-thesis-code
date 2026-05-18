
import pandas as pd
import numpy as np
import os

# --- Configuration ---
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
COMP_DIR = os.path.join(PROJECT_DIR, "04_Model_Comparison")
INPUT_FILE = os.path.join(COMP_DIR, "Model_Comparison_CV.csv")
OUTPUT_FILE = os.path.join(COMP_DIR, "CoV_Distribution_Stats.csv")

def main():
    print("--- Calculating CoV Distribution Statistics ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # Load Data
    df = pd.read_csv(INPUT_FILE)
    
    if 'CoV' not in df.columns:
        print("Error: 'CoV' column not found in input file.")
        return

    print(f"Total Basins: {len(df)}")
    
    # Define Bins
    # Ranges: 0-0.5, 0.5-1, 1-1.5, >1.5
    # Bins edges: [0, 0.5, 1.0, 1.5, np.inf]
    bins = [0, 0.5, 1.0, 1.5, np.inf]
    labels = ["0 - 0.5", "0.5 - 1.0", "1.0 - 1.5", "> 1.5"]
    
    # Categorize
    df['CoV_Range'] = pd.cut(df['CoV'], bins=bins, labels=labels, right=False)
    
    # Calculate Stats
    stats = df['CoV_Range'].value_counts().sort_index()
    percentages = df['CoV_Range'].value_counts(normalize=True).sort_index() * 100
    
    # Create Summary DataFrame
    summary_df = pd.DataFrame({
        'Range': stats.index,
        'Count': stats.values,
        'Percentage': percentages.values
    })
    
    # Format Percentage
    summary_df['Percentage'] = summary_df['Percentage'].map('{:.2f}%'.format)
    
    # Add Total Row
    total_row = pd.DataFrame({
        'Range': ['Total'], 
        'Count': [stats.sum()], 
        'Percentage': ['100.00%']
    })
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)
    
    print("\nCoV Distribution Summary:")
    print(summary_df)
    
    # Save to CSV
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nStats saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os

# Configuration
PROJECT_DIR = r"C:\Users\syyda\Desktop\Chapter 3 mapping and analysis"
INPUT_PATH = os.path.join(PROJECT_DIR, "01_Data_Prep", "01_Data_Combine_Result", "data_combine.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "01_Data_Prep", "02_Sensitivity_Data")

# Percentages to sample
PERCENTAGES = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
SEED = 42 # Fixed seed for reproducibility

def main():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total Rows: {len(df)}")
    
    # Filter for valid target if necessary (Training scripts usually do this too, but good to be consistent)
    # We will sample from the FULL dataset, and let training scripts handle their specific filtering
    # OR we sample from valid rows only? Usually better to sample from Raw and let pipeline handle it.
    # However, to ensure we have enough data, let's just sample rows.
    
    for pct in PERCENTAGES:
        # Calculate size
        sample_size = int(len(df) * pct)
        
        # Sample
        # random_state ensures we get the SAME 90% every time we run this script
        df_sample = df.sample(frac=pct, random_state=SEED)
        
        # Save
        pct_int = int(pct * 100)
        filename = f"data_sensitivity_{pct_int}.csv"
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        df_sample.to_csv(out_path, index=False)
        print(f"Saved {pct_int}% sample ({len(df_sample)} rows) to: {out_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()

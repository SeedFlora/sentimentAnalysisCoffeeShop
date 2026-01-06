import pandas as pd
import os

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

for csv_file in csv_files:
    print("\n" + "="*70)
    print(f"FILE: {csv_file}")
    print("="*70)
    try:
        df = pd.read_csv(csv_file)
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst row:")
        print(df.iloc[0].to_dict())
        
        # Check for sentiment/label column
        label_cols = [col for col in df.columns if any(x in col.lower() for x in ['sentiment', 'label', 'rating', 'review', 'text'])]
        if label_cols:
            print(f"\nPossible columns for labels: {label_cols}")
            for col in label_cols:
                print(f"  {col}: {df[col].nunique()} unique values")
                print(f"    Sample values: {df[col].unique()[:5].tolist()}")
    except Exception as e:
        print(f"Error: {e}")

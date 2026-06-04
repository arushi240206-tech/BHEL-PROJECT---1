import pandas as pd

df = pd.read_csv("10yearsdata_raw.csv")
date_cols = [c for c in df.columns if 'date' in c.lower() or 'Date' in c]

for col in date_cols:
    non_nulls = df[col].dropna()
    print(f"\nParsing {col}...")
    try:
        parsed = pd.to_datetime(non_nulls, format="%d-%b-%Y")
        print(f"Successfully parsed all {len(non_nulls)} non-null values with format %d-%b-%Y.")
    except Exception as e:
        print(f"Failed to parse with standard format: {e}")
        # Try generic parser
        try:
            parsed = pd.to_datetime(non_nulls, errors='coerce')
            nulls_after = parsed.isnull().sum()
            print(f"Generic parser nulls introduced: {nulls_after}")
            if nulls_after > 0:
                print("Failed values:")
                print(non_nulls[parsed.isnull()].head(10))
        except Exception as e2:
            print(f"Generic parser failed: {e2}")

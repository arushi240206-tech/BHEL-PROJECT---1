import pandas as pd

df = pd.read_csv("10yearsdata_raw.csv")

print("--- Date Samples ---")
date_cols = [c for c in df.columns if 'date' in c.lower() or 'Date' in c]
for col in date_cols:
    non_nulls = df[col].dropna()
    print(f"\n{col} (Null count: {df[col].isnull().sum()}):")
    if len(non_nulls) > 0:
        print("Sample values:", list(non_nulls.head(5)))

print("\n--- Categorical/String values ---")
categorical_cols = [
    'Status', 
    'Acceptance by Unit', 
    'Will milestone get affected (given by site)', 
    'Repetitive issues identified by unit(Y/N)',
    'Cost Debitable'
]
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} unique values:")
        print(df[col].value_counts(dropna=False).head(10))

# Print completely empty columns
empty_cols = [col for col in df.columns if df[col].isnull().sum() == len(df)]
print(f"\nCompletely empty columns ({len(empty_cols)}):", empty_cols)

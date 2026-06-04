import os
import pandas as pd

def convert_xls_to_csv(xls_path, csv_path):
    print(f"Reading {xls_path}...")
    # Load the Excel file
    xls = pd.ExcelFile(xls_path)
    print(f"Sheet names found: {xls.sheet_names}")
    
    # Read the first sheet
    sheet_name = xls.sheet_names[0]
    print(f"Loading sheet: {sheet_name}")
    df = pd.read_excel(xls_path, sheet_name=sheet_name)
    
    # Save raw data to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved raw CSV to {csv_path}")
    return df

def inspect_data(df):
    print("\n--- Data Inspection Summary ---")
    print(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows count:", df.duplicated().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe(include='all'))

if __name__ == "__main__":
    xls_file = "10yearsdata.xls"
    raw_csv = "10yearsdata_raw.csv"
    
    if os.path.exists(xls_file):
        df = convert_xls_to_csv(xls_file, raw_csv)
        inspect_data(df)
    else:
        print(f"Error: {xls_file} not found in current directory.")

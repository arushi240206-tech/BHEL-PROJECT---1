import os
import pandas as pd
import re

def verify():
    csv_path = "10yearsdata_cleaned.csv"
    assert os.path.exists(csv_path), "Cleaned CSV file does not exist!"
    
    df = pd.read_csv(csv_path)
    print(f"File loaded successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check row count
    assert df.shape[0] == 13034, f"Expected 13034 rows, but got {df.shape[0]}"
    # Check column count (original 59 - 2 dropped empty columns = 57)
    assert df.shape[1] == 57, f"Expected 57 columns, but got {df.shape[1]}"
    print("✓ Row and Column counts are correct.")
    
    # Check dropped columns
    dropped_cols = ['PEM Job no', 'PEM Job description', 'PEM Job No', 'PEM Job Description']
    for col in dropped_cols:
        assert col not in df.columns, f"Empty column {col} was not dropped!"
    print("✓ Completely empty columns were dropped successfully.")
    
    # Check `<br>` in column names
    for col in df.columns:
        assert '<br>' not in col, f"HTML tag found in column name: {col}"
    print("✓ HTML tags were removed from column names successfully.")
    
    # Check date formatting
    date_cols = [
        'Complaint Date',
        'First Disposition Date',
        'Last Disposition Date',
        'Last Date of Return by Site to Unit',
        'Anticipated Date of Action Completion (Unit)'
    ]
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    for col in date_cols:
        non_nulls = df[col].dropna()
        for idx, val in enumerate(non_nulls):
            assert date_pattern.match(str(val)), f"Invalid date format in {col} at index {idx}: {val}"
    print("✓ Date values are correctly formatted as YYYY-MM-DD.")
    
    # Check Y/N columns standardisation
    yn_cols = [
        'Acceptance by Unit',
        'Will Milestone Get Affected (Given by Site)',
        'Cost Debitable',
        'Repetitive Issues Identified by Unit (Y/N)'
    ]
    for col in yn_cols:
        unique_vals = set(df[col].dropna().unique())
        assert unique_vals.issubset({'Y', 'N'}), f"Unexpected values in Y/N column {col}: {unique_vals}"
    print("✓ Y/N columns successfully standardized to Y and N.")
    
    # Check Problem Nature Keywords trailing commas
    keywords = df['Problem Nature Keywords'].dropna()
    for idx, val in enumerate(keywords):
        assert not str(val).endswith(','), f"Trailing comma found in keyword at index {idx}: {val}"
    print("✓ Trailing commas removed from Problem Nature Keywords.")
    
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify()

import pandas as pd
import numpy as np

df = pd.read_csv("10yearsdata_raw.csv")

print("--- Checking numeric columns and possible string mixups ---")
for col in df.columns:
    # Check if this column has numbers mixed with strings
    non_null = df[col].dropna()
    if df[col].dtype == 'object':
        # See if we can convert it to numeric
        try:
            converted = pd.to_numeric(non_null)
            print(f"Column '{col}' is object type but can be converted to numeric (non-null count: {len(non_null)}).")
        except:
            pass

# Let's inspect Dispositioning authority Staffno
staff_no = df['Dispositioning authority Staffno'].dropna()
print(f"\nDispositioning authority Staffno sample: {list(staff_no.head(10))}")

# Let's inspect No. of times SAR/CAR is reopened
reopened = df['No. of times SAR/CAR is reopened'].dropna()
print(f"\nNo. of times SAR/CAR is reopened sample: {list(reopened.head(10))}")
print(f"Unique values of No. of times SAR/CAR is reopened: {reopened.unique()}")

# Let's check Vendor Code
vendor_code = df['Vendor Code'].dropna()
print(f"\nVendor Code sample: {list(vendor_code.head(10))}")

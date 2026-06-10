import pandas as pd
import numpy as np

def test_date_parsing():
    # Simulate the date parsing logic from old test_date_parsing.py
    data = pd.Series(["15-Jan-2020", "20-Feb-2021", "Invalid Date", np.nan])
    parsed = pd.to_datetime(data, format="%d-%b-%Y", errors='coerce')
    
    assert not pd.isna(parsed.iloc[0])
    assert parsed.iloc[0].year == 2020
    assert parsed.iloc[0].month == 1
    
    assert pd.isna(parsed.iloc[2]) # Invalid date should be NaT
    assert pd.isna(parsed.iloc[3]) # NaN should be NaT

def test_numeric_conversion():
    # Simulate logic from test_numeric.py
    data = pd.Series(["1,000.50", "500", "N/A", "2000"])
    
    # Clean and convert
    cleaned = data.replace(r'[^\d.]', '', regex=True)
    cleaned = pd.to_numeric(cleaned, errors='coerce')
    
    assert cleaned.iloc[0] == 1000.50
    assert cleaned.iloc[1] == 500.0
    assert pd.isna(cleaned.iloc[2])
    assert cleaned.iloc[3] == 2000.0

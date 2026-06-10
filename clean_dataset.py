import pandas as pd
import numpy as np

def clean_data():
    print("Loading raw CSV...")
    df = pd.read_csv("10yearsdata_raw.csv")
    
    # 1. Drop completely empty columns
    empty_cols = ['PEM Job no', 'PEM Job description']
    df = df.drop(columns=empty_cols, errors='ignore')
    print(f"Dropped completely empty columns: {empty_cols}")
    
    # 2. Standardize column names
    column_mapping = {
        'Sno': 'Sno',
        'Complaint no.': 'Complaint Number',
        'Complaint type': 'Complaint Type',
        'Complaint Date': 'Complaint Date',
        'Project': 'Project',
        'Product': 'Product',
        'Item': 'Item',
        'PGMA': 'PGMA',
        'DU': 'DU',
        'SHOP/BOI (given by site)': 'Shop/BOI (Given by Site)',
        'Problem description': 'Problem Description',
        'Site Recommendation': 'Site Recommendation',
        'Debit Claimed<br>(int INR)': 'Debit Claimed (INR)',
        'Debit Accepted by Unit': 'Debit Accepted by Unit',
        'Days taken for disposition': 'Days Taken for Disposition',
        'Status': 'Status',
        'Acceptance by Unit': 'Acceptance by Unit',
        'Unit disposition': 'Unit Disposition',
        'Defect type': 'Defect Type',
        'Defect Sub-type code': 'Defect Sub-type Code',
        'Defect Sub-type description': 'Defect Sub-type Description',
        'Nodal Agency': 'Nodal Agency',
        'Site Engineer name': 'Site Engineer Name',
        'Site Engineer email': 'Site Engineer Email',
        'Site HOS name': 'Site HOS Name',
        'Site HOS email': 'Site HOS Email',
        'PO Number': 'PO Number',
        'Vendor Code': 'Vendor Code',
        'Severity Rating (given by site)': 'Severity Rating (Given by Site)',
        'Severity Rating (given by unit)': 'Severity Rating (Given by Unit)',
        'EST_COST_BY_UNIT (given by unit)': 'Estimated Cost by Unit',
        'FINAL_COST_INCURRED_SITE (given by site)': 'Final Cost Incurred at Site',
        'First disposition date': 'First Disposition Date',
        'Last disposition date': 'Last Disposition Date',
        'Last Date of return by site to unit': 'Last Date of Return by Site to Unit',
        'Anticipated Date of action completion given by unit': 'Anticipated Date of Action Completion (Unit)',
        'Pending with': 'Pending With',
        'Reason of return by site': 'Reason of Return by Site',
        'No. of times SAR/CAR is reopened': 'No. of Times SAR/CAR Reopened',
        'Dispositioning authority Name': 'Dispositioning Authority Name',
        'Dispositioning authority Staffno': 'Dispositioning Authority Staff Number',
        'Region': 'Region',
        'Unit': 'Unit',
        'Activity Status (at registration time)': 'Activity Status (Registration)',
        'Type of Issue (given by site)': 'Type of Issue (Given by Site)',
        'Will milestone get affected (given by site)': 'Will Milestone Get Affected (Given by Site)',
        'Milestone Name (given by site)': 'Milestone Name (Given by Site)',
        'Type of Issue (given by unit)': 'Type of Issue (Given by Unit)',
        'Milestone affected (given during site action completion)': 'Milestone Affected (Given during Site Action Completion)',
        'Type of dispatch (given by unit)': 'Type of Dispatch (Given by Unit)',
        'Vendor name': 'Vendor Name',
        'Unit Workorder': 'Unit Work Order',
        'Cost Debitable': 'Cost Debitable',
        'NC Categorization': 'NC Categorization',
        'Learning Derived': 'Learning Derived',
        'Repetitive issues identified by unit(Y/N)': 'Repetitive Issues Identified by Unit (Y/N)',
        'Problem Nature Keywords': 'Problem Nature Keywords'
    }
    
    df = df.rename(columns=column_mapping)
    print("Renamed columns to standard Title Case format.")
    
    # 3. Trim whitespace and clean string values
    print("Cleaning string values and removing trailing commas...")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Trim whitespace
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings or string representation of nulls back to NaN
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, 'NONE': np.nan, '': np.nan})
            
            # If the column has values, strip trailing commas (e.g. in Problem Nature Keywords)
            df[col] = df[col].apply(lambda x: x.rstrip(',') if isinstance(x, str) else x)
            
    # 4. Standardize Y/N columns
    yn_cols = [
        'Acceptance by Unit',
        'Will Milestone Get Affected (Given by Site)',
        'Cost Debitable',
        'Repetitive Issues Identified by Unit (Y/N)'
    ]
    for col in yn_cols:
        if col in df.columns:
            df[col] = df[col].str.upper()
            # Verify they only contain Y, N or NaN
            df[col] = df[col].replace({'YES': 'Y', 'NO': 'N', 'TRUE': 'Y', 'FALSE': 'N'})
            
    # 5. Normalize Date Columns to YYYY-MM-DD
    date_cols = [
        'Complaint Date',
        'First Disposition Date',
        'Last Disposition Date',
        'Last Date of Return by Site to Unit',
        'Anticipated Date of Action Completion (Unit)'
    ]
    print("Parsing and normalizing date columns...")
    def fix_year(dt):
        if pd.isnull(dt):
            return dt
        if dt.year < 100:
            try:
                return dt.replace(year=2000 + dt.year)
            except ValueError:
                return dt
        return dt

    for col in date_cols:
        if col in df.columns:
            # Convert using standard %d-%b-%Y format
            parsed_dates = pd.to_datetime(df[col], format='%d-%b-%Y', errors='coerce')
            # Fix years < 100
            parsed_dates = parsed_dates.apply(fix_year)
            # Convert back to YYYY-MM-DD string format
            df[col] = parsed_dates.dt.strftime('%Y-%m-%d')
            
    # 6. Normalize integer columns with missing values to pandas nullable Int64
    int_cols = [
        'No. of Times SAR/CAR Reopened',
        'Dispositioning Authority Staff Number',
        'Sno',
        'Debit Claimed (INR)',
        'Days Taken for Disposition'
    ]
    print("Normalizing numeric columns...")
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
    # 7. Standardize specific categorical values (e.g., Status)
    if 'Status' in df.columns:
        status_mapping = {
            'Pending for action plan completion by unit': 'Pending for Action Plan Completion by Unit'
        }
        df['Status'] = df['Status'].replace(status_mapping)
        
    # 8. Merge PGMA Description from NS_LIST
    print("Enriching PGMA with NS_LIST master data...")
    try:
        ns_xls = pd.ExcelFile('NS_LIST.xlsx')
        pgma_df = pd.read_excel(ns_xls, sheet_name='PGMA')
        # Standardize join columns to string, ignoring leading zeros
        if 'PGMA' in df.columns and 'pgma' in pgma_df.columns:
            df['PGMA_join'] = df['PGMA'].astype(str).str.strip().str.lstrip('0')
            pgma_df['pgma_join'] = pgma_df['pgma'].astype(str).str.strip().str.lstrip('0')
            
            # Map description
            pgma_map = pgma_df.drop_duplicates(subset=['pgma_join']).set_index('pgma_join')['Description']
            df['PGMA Description'] = df['PGMA_join'].map(pgma_map)
            df.drop(columns=['PGMA_join'], inplace=True)
            
            # Create a unified Equipment Name (fallback to Product if PGMA is unknown)
            df['Equipment Name'] = df['PGMA Description'].fillna(df['Product'])
            print("Successfully merged PGMA Descriptions and created Equipment Name.")
        else:
            print("Warning: Could not find required PGMA columns for join.")
    except Exception as e:
        print(f"Warning: Failed to load NS_LIST.xlsx or merge PGMA. Error: {e}")
        
    # Write to cleaned CSV
    output_path = "10yearsdata_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to {output_path}!")
    
    # Print sample and verification info
    print("\n--- Cleaned Data Inspection Summary ---")
    print(f"Shape of cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nCleaned Column Types:")
    print(df.dtypes)
    print("\nSample Rows:")
    print(df[['Sno', 'Complaint Date', 'Debit Claimed (INR)', 'No. of Times SAR/CAR Reopened', 'Problem Nature Keywords']].head())

if __name__ == "__main__":
    clean_data()

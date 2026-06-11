import os
import pandas as pd
import numpy as np

class DataService:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.unique_metadata = {}

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Cleaned dataset not found at {self.csv_path}")

        print("Loading cleaned dataset for dashboard...")
        self.df = pd.read_csv(self.csv_path)

        # Pre-parse Complaint Year and Month
        self.df['Complaint Date'] = pd.to_datetime(self.df['Complaint Date'], errors='coerce')
        self.df['Complaint Year'] = self.df['Complaint Date'].dt.year.fillna(2020).astype(int)
        self.df['Complaint Month'] = self.df['Complaint Date'].dt.month.fillna(6).astype(int)

        # Fill missing values for categories in unique filters
        self.df['Project'] = self.df['Project'].fillna('Unknown')
        self.df['Status'] = self.df['Status'].fillna('Unknown')
        self.df['Product'] = self.df['Product'].fillna('Unknown')
        self.df['Region'] = self.df['Region'].fillna('Unknown')
        self.df['Unit'] = self.df['Unit'].fillna('Unknown')

        # Precompute global filter metadata
        self.unique_metadata = {
            'projects': sorted(self.df['Project'].unique().tolist()),
            'statuses': sorted(self.df['Status'].unique().tolist()),
            'products': sorted(self.df['Product'].unique().tolist()),
            'regions': sorted(self.df['Region'].unique().tolist()),
            'units': sorted(self.df['Unit'].unique().tolist()),
            'years': sorted([y for y in self.df['Complaint Year'].unique().tolist() if y > 0])
        }

    def get_metadata(self):
        return self.unique_metadata

    def filter_data(self, start_year, end_year, region, unit, project='', product='', status=''):
        f_df = self.df.copy()
        f_df = f_df[(f_df['Complaint Year'] >= start_year) & (f_df['Complaint Year'] <= end_year)]
        
        if region:
            f_df = f_df[f_df['Region'] == region]
        if unit:
            f_df = f_df[f_df['Unit'] == unit]
        if project:
            f_df = f_df[f_df['Project'] == project]
        if product:
            f_df = f_df[f_df['Product'] == product]
        if status:
            f_df = f_df[f_df['Status'] == status]
            
        return f_df

    def get_trends(self, filters):
        start_year = int(filters.get('start_year', 2014))
        end_year = int(filters.get('end_year', 2026))
        region = filters.get('region', '')
        unit = filters.get('unit', '')

        f_df = self.filter_data(start_year, end_year, region, unit)

        if f_df.empty:
            return {
                'kpis': {
                    'total_complaints': 0,
                    'avg_resolution_days': 0,
                    'pct_repetitive': 0,
                    'avg_severity': 0,
                    'total_cost_debitable': 0
                },
                'empty': True
            }

        # Compute KPIs
        total_complaints = len(f_df)
        avg_res = f_df['Days Taken for Disposition'].dropna()
        avg_resolution_days = float(avg_res.mean()) if not avg_res.empty else 0.0
        
        rep_series = f_df['Repetitive Issues Identified by Unit (Y/N)'].dropna()
        pct_repetitive = float((rep_series == 'Y').sum() / len(rep_series) * 100) if len(rep_series) > 0 else 0.0
        
        sev_series = f_df['Severity Rating (Given by Unit)'].dropna()
        avg_severity = float(sev_series.mean()) if not sev_series.empty else 0.0
        
        cost_debitable_df = f_df[f_df['Cost Debitable'] == 'Y']
        total_cost_debitable = float(cost_debitable_df['Debit Claimed (INR)'].sum()) if not cost_debitable_df.empty else 0.0
        
        kpis = {
            'total_complaints': total_complaints,
            'avg_resolution_days': round(avg_resolution_days, 1),
            'pct_repetitive': round(pct_repetitive, 1),
            'avg_severity': round(avg_severity, 2),
            'total_cost_debitable': round(total_cost_debitable, 0)
        }

        # 1. Complaint Volume Trends
        volume_grouped = f_df.groupby(['Complaint Year', 'Complaint Month']).size().reset_index(name='count')
        yoy_volume = {}
        years_present = sorted(volume_grouped['Complaint Year'].unique())
        months = list(range(1, 13))
        for yr in years_present:
            yr_data = volume_grouped[volume_grouped['Complaint Year'] == yr]
            yoy_volume[str(yr)] = [int(yr_data[yr_data['Complaint Month'] == m]['count'].iloc[0]) if not yr_data[yr_data['Complaint Month'] == m].empty else 0 for m in months]
            
        annual_volume = f_df.groupby('Complaint Year').size().sort_index()
        yoy_pct_change = {}
        prev_yr = None
        for yr, val in annual_volume.items():
            if prev_yr is not None and annual_volume[prev_yr] > 0:
                change = ((val - annual_volume[prev_yr]) / annual_volume[prev_yr]) * 100
                yoy_pct_change[str(yr)] = round(float(change), 1)
            else:
                yoy_pct_change[str(yr)] = 0.0
            prev_yr = yr
            
        f_df['YearMonth'] = f_df['Complaint Date'].dt.to_period('M')
        all_months = pd.period_range(start=f"{start_year}-01", end=f"{end_year}-12", freq='M')
        monthly_series = f_df.groupby('YearMonth').size().reindex(all_months, fill_value=0)
        rolling_3 = monthly_series.rolling(window=3, min_periods=1).mean().round(1).tolist()
        rolling_labels = [str(x) for x in all_months]
        
        f_df = f_df.drop(columns=['YearMonth'])
        
        # 2. Defect Analysis
        top_defect_types = f_df['Defect Type'].value_counts().head(10).to_dict()
        top_defect_subtypes = f_df['Defect Sub-type Description'].value_counts().head(10).to_dict()
        
        top_5_defects = list(f_df['Defect Type'].value_counts().head(5).index)
        defect_region_ct = pd.crosstab(f_df['Region'], f_df['Defect Type'])
        defect_region_ct = defect_region_ct[defect_region_ct.columns.intersection(top_5_defects)]
        defect_region_data = {
            'regions': list(defect_region_ct.index),
            'defects': {col: list(defect_region_ct[col].astype(int)) for col in defect_region_ct.columns}
        }
        
        # 3. Severity Trends
        avg_severity_year = f_df.groupby('Complaint Year')['Severity Rating (Given by Unit)'].mean().round(2).fillna(0.0).to_dict()
        
        top_5_products = list(f_df['Product'].value_counts().head(5).index)
        product_sev_df = f_df[f_df['Product'].isin(top_5_products)].copy()
        def group_severity(sev):
            if pd.isnull(sev): return 'Unknown'
            if sev <= 0.3: return 'Low'
            if sev <= 0.6: return 'Medium'
            return 'High'
        product_sev_df['Severity_Group'] = product_sev_df['Severity Rating (Given by Unit)'].apply(group_severity)
        product_sev_ct = pd.crosstab(product_sev_df['Product'], product_sev_df['Severity_Group'])
        product_sev_ct = product_sev_ct.reindex(columns=['Low', 'Medium', 'High', 'Unknown'], fill_value=0)
        product_sev_data = {
            'products': list(product_sev_ct.index),
            'Low': list(product_sev_ct['Low'].astype(int)),
            'Medium': list(product_sev_ct['Medium'].astype(int)),
            'High': list(product_sev_ct['High'].astype(int))
        }
        
        scatter_sev_days_df = f_df[['Severity Rating (Given by Unit)', 'Days Taken for Disposition']].dropna()
        np.random.seed(42)
        sample_size = min(300, len(scatter_sev_days_df))
        if sample_size > 0:
            scatter_sample = scatter_sev_days_df.sample(n=sample_size)
            scatter_sev_days = [{
                'severity': float(row['Severity Rating (Given by Unit)']),
                'days': int(row['Days Taken for Disposition'])
            } for _, row in scatter_sample.iterrows()]
        else:
            scatter_sev_days = []
            
        # 4. Resolution Performance
        avg_days_year = f_df.groupby('Complaint Year')['Days Taken for Disposition'].mean().round(1).fillna(0.0).to_dict()
        
        bins = [0, 30, 90, 180, 365, 99999]
        bin_labels = ['0-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
        resolution_bins = pd.cut(f_df['Days Taken for Disposition'], bins=bins, labels=bin_labels).value_counts().reindex(bin_labels, fill_value=0).to_dict()
        
        total_disposed = f_df['Days Taken for Disposition'].dropna().count()
        pct_30 = float((f_df['Days Taken for Disposition'] <= 30).sum() / total_disposed * 100) if total_disposed > 0 else 0.0
        pct_60 = float((f_df['Days Taken for Disposition'] <= 60).sum() / total_disposed * 100) if total_disposed > 0 else 0.0
        pct_90 = float((f_df['Days Taken for Disposition'] <= 90).sum() / total_disposed * 100) if total_disposed > 0 else 0.0
        resolved_ratios = {
            'within_30': round(pct_30, 1),
            'within_60': round(pct_60, 1),
            'within_90': round(pct_90, 1)
        }
        longest_resolution_units = f_df.groupby('Unit')['Days Taken for Disposition'].mean().round(1).sort_values(ascending=False).head(10).to_dict()
        
        # 5. Cost Analysis
        avg_claimed_year = f_df.groupby('Complaint Year')['Debit Claimed (INR)'].mean().round(0).fillna(0.0).to_dict()
        
        f_df['Debit_Accepted_Amount'] = np.where(f_df['Debit Accepted by Unit'] == 'Y', f_df['Debit Claimed (INR)'], 
                                                 np.where(f_df['Debit Accepted by Unit'] == 'N', 0, np.nan))
        avg_accepted_year = f_df.groupby('Complaint Year')['Debit_Accepted_Amount'].mean().round(0).fillna(0.0).to_dict()
        
        f_df['Cost_Overrun'] = f_df['Final Cost Incurred at Site'].fillna(0) - f_df['Estimated Cost by Unit'].fillna(0)
        avg_overrun_year = f_df.groupby('Complaint Year')['Cost_Overrun'].mean().round(0).fillna(0.0).to_dict()
        
        top_vendors_cost = f_df[f_df['Cost Debitable'] == 'Y'].groupby('Vendor Name')['Final Cost Incurred at Site'].sum().round(0).sort_values(ascending=False).head(10).to_dict()
        
        # 6. Repetitive Issues
        pct_repetitive_year = {}
        for yr in years_present:
            yr_df = f_df[f_df['Complaint Year'] == yr]
            yr_rep = yr_df['Repetitive Issues Identified by Unit (Y/N)'].dropna()
            pct_repetitive_year[str(yr)] = round(float((yr_rep == 'Y').sum() / len(yr_rep) * 100), 1) if len(yr_rep) > 0 else 0.0
            
        defect_rep_ct = pd.crosstab(f_df['Defect Type'], f_df['Repetitive Issues Identified by Unit (Y/N)'])
        defect_rep_ct = defect_rep_ct.reindex(columns=['Y', 'N'], fill_value=0)
        defect_rep_data = {
            'defects': list(defect_rep_ct.index),
            'Y': list(defect_rep_ct['Y'].astype(int)),
            'N': list(defect_rep_ct['N'].astype(int))
        }
        
        # 7. Milestone & Status
        pct_milestone_year = {}
        for yr in years_present:
            yr_df = f_df[f_df['Complaint Year'] == yr]
            milestone_series = yr_df['Will Milestone Get Affected (Given by Site)'].dropna()
            pct_milestone_year[str(yr)] = round(float((milestone_series == 'Y').sum() / len(milestone_series) * 100), 1) if len(milestone_series) > 0 else 0.0
            
        status_year_ct = pd.crosstab(f_df['Complaint Year'], f_df['Status'])
        status_columns = list(status_year_ct.columns)
        status_data = {
            'years': [str(y) for y in status_year_ct.index],
            'statuses': status_columns,
            'series': {status_col: list(status_year_ct[status_col].astype(int)) for status_col in status_columns}
        }
        
        # 8. RCA global distributions
        top_nc_types = f_df['NC Categorization'].value_counts().head(10).to_dict()
        
        product_defect_ct = pd.crosstab(f_df[f_df['Product'].isin(top_5_products)]['Product'], f_df['Defect Type'])
        product_defect_ct = product_defect_ct[product_defect_ct.columns.intersection(top_5_defects)]
        defect_product_data = {
            'products': list(product_defect_ct.index),
            'defects': {col: list(product_defect_ct[col].astype(int)) for col in product_defect_ct.columns}
        }
        
        learnings_series = f_df['Learning Derived'].dropna().astype(str).str.strip()
        learnings_series = learnings_series[~learnings_series.isin(['', '--', 'UNKNOWN', 'nan', 'None'])]
        top_learnings = list(learnings_series.unique()[:8])

        return {
            'kpis': kpis,
            'complaint_volume': {
                'yoy_volume': yoy_volume,
                'yoy_pct_change': yoy_pct_change,
                'rolling_labels': rolling_labels,
                'rolling_3': rolling_3
            },
            'defect_analysis': {
                'top_defect_types': top_defect_types,
                'top_defect_subtypes': top_defect_subtypes,
                'defect_region': defect_region_data
            },
            'severity_trends': {
                'avg_severity_year': avg_severity_year,
                'product_sev': product_sev_data,
                'scatter_sev_days': scatter_sev_days
            },
            'resolution_performance': {
                'avg_days_year': avg_days_year,
                'resolution_bins': resolution_bins,
                'resolved_ratios': resolved_ratios,
                'longest_resolution_units': longest_resolution_units
            },
            'cost_analysis': {
                'avg_claimed_year': avg_claimed_year,
                'avg_accepted_year': avg_accepted_year,
                'avg_overrun_year': avg_overrun_year,
                'top_vendors_cost': top_vendors_cost
            },
            'repetitive_issues': {
                'pct_repetitive_year': pct_repetitive_year,
                'defect_rep_data': defect_rep_data
            },
            'milestone_status': {
                'pct_milestone_year': pct_milestone_year,
                'status_data': status_data
            },
            'rca_global': {
                'top_nc_types': top_nc_types,
                'defect_product': defect_product_data,
                'learnings': top_learnings
            }
        }

    def get_equipment_analysis(self, filters):
        start_year = int(filters.get('start_year', 2014))
        end_year = int(filters.get('end_year', 2026))
        region = filters.get('region', '')
        unit = filters.get('unit', '')

        f_df = self.filter_data(start_year, end_year, region, unit)

        if f_df.empty:
            return []

        f_df = f_df.dropna(subset=['PGMA'])
        top_products = f_df['Item'].value_counts().head(5)
        
        equipment_data = []
        for product_name, count in top_products.items():
            if pd.isna(product_name) or product_name == 'Unknown':
                continue
                
            prod_df = f_df[f_df['Item'] == product_name]
            top_defects = prod_df['Defect Type'].value_counts().head(2)
            
            defects_list = []
            for defect_name, defect_count in top_defects.items():
                if pd.isna(defect_name) or defect_name == 'Unknown':
                    continue
                    
                defect_df = prod_df[prod_df['Defect Type'] == defect_name]
                dispositions = defect_df['Unit Disposition'].dropna().astype(str)
                dispositions = dispositions[~dispositions.isin(['', '--', 'UNKNOWN', 'nan', 'None'])]
                top_disposition = dispositions.mode()[0] if not dispositions.empty else "No standard disposition found."
                
                learnings = defect_df['Learning Derived'].dropna().astype(str)
                learnings = learnings[~learnings.isin(['', '--', 'UNKNOWN', 'nan', 'None'])]
                top_learning = learnings.mode()[0] if not learnings.empty else "No specific learnings recorded."
                
                defects_list.append({
                    'defect_name': defect_name,
                    'count': int(defect_count),
                    'disposition': top_disposition,
                    'learning': top_learning
                })
                
            if defects_list:
                equipment_data.append({
                    'product_name': product_name,
                    'total_complaints': int(count),
                    'defects': defects_list
                })
                
        return equipment_data

    def get_proactive_alerts(self):
        if self.df.empty:
            return []
            
        recent_df = self.df.sort_values('Complaint Date', ascending=False).head(100)
        recent_df = recent_df.dropna(subset=['Item'])
        freq = recent_df['Item'].value_counts()
        
        alerts = []
        for eq, count in freq.items():
            if count >= 3 and str(eq).lower() not in ['nan', 'unknown', 'none']:
                alerts.append({
                    'equipment': str(eq),
                    'recent_failures': int(count),
                    'message': f"Critical Alert: {eq} has failed {count} times recently."
                })
        return alerts[:5]

    def get_vendor_stats(self, vendor_name):
        if self.df.empty or not vendor_name:
            return None
            
        v_df = self.df[self.df['Vendor Name'].str.lower() == str(vendor_name).lower()]
        if v_df.empty:
            return None
            
        avg_res = v_df['Days Taken for Disposition'].mean()
        if pd.isna(avg_res):
            return None
            
        return round(float(avg_res), 1)
        
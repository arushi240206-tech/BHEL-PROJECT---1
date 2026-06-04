import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')

CSV_PATH = "10yearsdata_cleaned.csv"
METRICS_JSON_PATH = "model_evaluation_metrics.json"

# Global data holders
df = None
unique_metadata = {}
metrics_report = {}
tfidf_vectorizer = None
tfidf_matrix = None

# ML Deployed models
model_severity = None
model_disposition = None
model_repetitive = None

def initialize_dashboard_backend():
    global df, unique_metadata, metrics_report, model_severity, model_disposition, model_repetitive, tfidf_vectorizer, tfidf_matrix
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {CSV_PATH}")
        
    print("Loading cleaned dataset for dashboard...")
    df = pd.read_csv(CSV_PATH)
    
    # Pre-parse Complaint Year and Month
    df['Complaint Date'] = pd.to_datetime(df['Complaint Date'], errors='coerce')
    df['Complaint Year'] = df['Complaint Date'].dt.year.fillna(2020).astype(int)
    df['Complaint Month'] = df['Complaint Date'].dt.month.fillna(6).astype(int)
    
    # Fill missing values for categories in unique filters
    df['Project'] = df['Project'].fillna('Unknown')
    df['Status'] = df['Status'].fillna('Unknown')
    df['Product'] = df['Product'].fillna('Unknown')
    df['Region'] = df['Region'].fillna('Unknown')
    df['Unit'] = df['Unit'].fillna('Unknown')
    
    # Precompute global filter metadata
    unique_metadata = {
        'projects': sorted(df['Project'].unique().tolist()),
        'statuses': sorted(df['Status'].unique().tolist()),
        'products': sorted(df['Product'].unique().tolist()),
        'regions': sorted(df['Region'].unique().tolist()),
        'units': sorted(df['Unit'].unique().tolist()),
        'years': sorted([y for y in df['Complaint Year'].unique().tolist() if y > 0])
    }
    
    # Load metrics report if exists
    if os.path.exists(METRICS_JSON_PATH):
        print("Loading precomputed ML metrics...")
        with open(METRICS_JSON_PATH, 'r') as f:
            metrics_report = json.load(f)
            
    # Load ML models
    if os.path.exists("best_model_severity.joblib"):
        print("Loading deployed Severity model...")
        model_severity = joblib.load("best_model_severity.joblib")
    if os.path.exists("best_model_disposition.joblib"):
        print("Loading deployed Disposition Time model...")
        model_disposition = joblib.load("best_model_disposition.joblib")
    if os.path.exists("best_model_repetitive.joblib"):
        print("Loading deployed Repetitive Issue model...")
        model_repetitive = joblib.load("best_model_repetitive.joblib")

    # Fit TF-IDF Vectorizer on search corpus
    print("Fitting TF-IDF Vectorizer on search corpus...")
    corpus_columns = [
        'Problem Description',
        'Item',
        'Defect Type',
        'Defect Sub-type Description',
        'Problem Nature Keywords',
        'Product',
        'Project'
    ]
    search_corpus = df[corpus_columns].fillna('').agg(' '.join, axis=1)
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        sublinear_tf=True,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(search_corpus)
    print("TF-IDF fit complete. Matrix shape:", tfidf_matrix.shape)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify(unique_metadata)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not metrics_report:
        return jsonify({'error': 'Metrics not found. Please run model training first.'}), 404
    return jsonify(metrics_report)

@app.route('/api/trends', methods=['POST'])
def get_trends():
    try:
        data = request.json or {}
        start_year = int(data.get('start_year', 2014))
        end_year = int(data.get('end_year', 2026))
        region = data.get('region', '')
        unit = data.get('unit', '')
        
        # Filter dataframe
        f_df = df.copy()
        f_df = f_df[(f_df['Complaint Year'] >= start_year) & (f_df['Complaint Year'] <= end_year)]
        
        if region:
            f_df = f_df[f_df['Region'] == region]
        if unit:
            f_df = f_df[f_df['Unit'] == unit]
            
        if f_df.empty:
            return jsonify({
                'kpis': {
                    'total_complaints': 0,
                    'avg_resolution_days': 0,
                    'pct_repetitive': 0,
                    'avg_severity': 0,
                    'total_cost_debitable': 0
                },
                'empty': True
            })
            
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
        
        # 1. Complaint Volume Trends (Line chart YoY overlay)
        volume_grouped = f_df.groupby(['Complaint Year', 'Complaint Month']).size().reset_index(name='count')
        yoy_volume = {}
        years_present = sorted(volume_grouped['Complaint Year'].unique())
        months = list(range(1, 13))
        for yr in years_present:
            yr_data = volume_grouped[volume_grouped['Complaint Year'] == yr]
            yoy_volume[str(yr)] = [int(yr_data[yr_data['Complaint Month'] == m]['count'].iloc[0]) if not yr_data[yr_data['Complaint Month'] == m].empty else 0 for m in months]
            
        # YoY % change in annual frequency
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
            
        # 3-month rolling average
        f_df['YearMonth'] = f_df['Complaint Date'].dt.to_period('M')
        all_months = pd.period_range(start=f"{start_year}-01", end=f"{end_year}-12", freq='M')
        monthly_series = f_df.groupby('YearMonth').size().reindex(all_months, fill_value=0)
        rolling_3 = monthly_series.rolling(window=3, min_periods=1).mean().round(1).tolist()
        rolling_labels = [str(x) for x in all_months]
        
        # Remove YearMonth to avoid issues
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
        
        # Extract top unique learnings
        learnings_series = f_df['Learning Derived'].dropna().astype(str).str.strip()
        learnings_series = learnings_series[~learnings_series.isin(['', '--', 'UNKNOWN', 'nan', 'None'])]
        top_learnings = list(learnings_series.unique()[:8])
        
        return jsonify({
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
        })
    except Exception as e:
        print(f"Error serving trends: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def run_predictions():
    try:
        data = request.json or {}
        
        # Check if models are loaded
        if not model_severity or not model_disposition or not model_repetitive:
            return jsonify({'error': 'Models are not fully loaded on backend.'}), 500
            
        # Parse inputs
        product = str(data.get('product', 'UNKNOWN')).strip().upper()
        region = str(data.get('region', 'UNKNOWN')).strip().upper()
        defect_type = str(data.get('defect_type', 'UNKNOWN')).strip().upper()
        severity_site = float(data.get('severity_site', 0.5))
        complaint_type = str(data.get('complaint_type', 'UNKNOWN')).strip().upper()
        shop_boi = str(data.get('shop_boi', 'UNKNOWN')).strip().upper()
        unit = str(data.get('unit', 'UNKNOWN')).strip().upper()
        cost_debitable = str(data.get('cost_debitable', 'UNKNOWN')).strip().upper()
        milestone_affected = str(data.get('milestone_affected', 'UNKNOWN')).strip().upper()
        complaint_date_str = str(data.get('complaint_date', '2026-06-04'))
        debit_claimed = float(data.get('debit_claimed', 0))
        
        # DateTime features
        dt = pd.to_datetime(complaint_date_str, errors='coerce')
        if pd.isnull(dt):
            dt = pd.to_datetime('2026-06-04')
        year = dt.year
        month = dt.month
        quarter = dt.quarter
        dayofweek = dt.dayofweek
        
        # Estimate complaint frequency: monthly count in historical dataset or fallback
        hist_count = df[(df['Complaint Year'] == year) & (df['Complaint Date'].dt.month == month)]
        if not hist_count.empty:
            freq_monthly = float(len(hist_count))
        else:
            freq_monthly = float(df.groupby([df['Complaint Date'].dt.year, df['Complaint Date'].dt.month]).size().mean())
            
        # Get label mappings from metadata
        mappings = metrics_report['metadata']['label_mappings']
        
        def encode_val(col, val):
            mapping = mappings.get(col, {})
            return mapping.get(val, mapping.get('UNKNOWN', 0))
            
        # Build features dict
        feat_dict = {
            'Product': encode_val('Product', product),
            'Region': encode_val('Region', region),
            'Defect_Type': encode_val('Defect_Type', defect_type),
            'Severity_Rating_Given_by_Site': severity_site,
            'Complaint_Type': encode_val('Complaint_Type', complaint_type),
            'ShopBOI_Given_by_Site': encode_val('ShopBOI_Given_by_Site', shop_boi),
            'Unit': encode_val('Unit', unit),
            'Cost_Debitable': encode_val('Cost_Debitable', cost_debitable),
            'Will_Milestone_Get_Affected_Given_by_Site': encode_val('Will_Milestone_Get_Affected_Given_by_Site', milestone_affected),
            'Complaint_Year': year,
            'Complaint_Month': month,
            'Complaint_Quarter': quarter,
            'Complaint_DayOfWeek': dayofweek,
            'Complaint_Frequency_Monthly': freq_monthly,
            'Debit_Claimed_INR': debit_claimed
        }
        
        # Order features exactly as expected by models
        features_order = model_severity['features']
        feature_vec = np.array([[feat_dict[f] for f in features_order]])
        
        # 1. Predict Target A (Severity)
        scaler_a = model_severity['scaler']
        model_a = model_severity['model']
        target_map_a = model_severity['target_mapping']
        
        X_a = scaler_a.transform(feature_vec) if scaler_a else feature_vec
        pred_a_idx = int(model_a.predict(X_a)[0])
        pred_severity_val = float(target_map_a[pred_a_idx])
        
        if hasattr(model_a, "predict_proba"):
            probs_a = model_a.predict_proba(X_a)[0]
            conf_a = float(probs_a[pred_a_idx])
        else:
            conf_a = 1.0
            
        # 2. Predict Target B (Disposition Days)
        scaler_b = model_disposition['scaler']
        model_b = model_disposition['model']
        X_b = scaler_b.transform(feature_vec) if scaler_b else feature_vec
        pred_days = float(model_b.predict(X_b)[0])
        pred_days = max(0.0, pred_days) # disposition days cannot be negative
        
        # 3. Predict Target C (Repetitive Issue Y/N)
        scaler_c = model_repetitive['scaler']
        model_c = model_repetitive['model']
        X_c = scaler_c.transform(feature_vec) if scaler_c else feature_vec
        pred_c_idx = int(model_c.predict(X_c)[0])
        pred_repetitive = "Y" if pred_c_idx == 1 else "N"
        
        if hasattr(model_c, "predict_proba"):
            probs_c = model_c.predict_proba(X_c)[0]
            conf_c = float(probs_c[pred_c_idx])
        else:
            conf_c = 1.0
            
        return jsonify({
            'predicted_severity': pred_severity_val,
            'predicted_severity_confidence': round(conf_a * 100, 1),
            'predicted_disposition_days': round(pred_days, 1),
            'predicted_repetitive': pred_repetitive,
            'predicted_repetitive_confidence': round(conf_c * 100, 1)
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/nlp_search', methods=['POST'])
def run_nlp_search():
    try:
        data = request.json or {}
        query = data.get('query', '').strip()
        start_year = int(data.get('start_year', 2014))
        end_year = int(data.get('end_year', 2026))
        region = data.get('region', '')
        unit = data.get('unit', '')
        project = data.get('project', '')
        product = data.get('product', '')
        status = data.get('status', '')
        limit = int(data.get('limit', 100))
        
        # Filter dataframe first
        f_df = df.copy()
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
            
        if f_df.empty:
            return jsonify({
                'results': [],
                'total_results': 0,
                'query_tokens': [],
                'rca_summary': {
                    'top_defects': {},
                    'top_nc': {},
                    'top_vendors': {},
                    'avg_resolution_days': 0,
                    'avg_severity': 0,
                    'learnings': []
                }
            })
            
        indices = f_df.index.tolist()
        
        # Calculate similarities if query exists
        if query and tfidf_vectorizer is not None:
            query_vec = tfidf_vectorizer.transform([query])
            # Slice tfidf_matrix to only include filtered indices
            sliced_matrix = tfidf_matrix[indices]
            sims = cosine_similarity(query_vec, sliced_matrix).flatten()
            
            f_df = f_df.copy()
            f_df['score'] = sims
            
            # Filter out documents that have 0 similarity
            f_df = f_df[f_df['score'] > 0]
            
            # Sort by score descending, then by Sno descending
            f_df = f_df.sort_values(by=['score', 'Sno'], ascending=[False, False])
        else:
            f_df = f_df.copy()
            f_df['score'] = 0.0
            f_df = f_df.sort_values(by='Complaint Date', ascending=False)
            
        total_results = len(f_df)
        
        # Slice to limit
        top_results_df = f_df.head(limit)
        
        # Replace NaN with None for JSON compliance
        top_results_df = top_results_df.replace({np.nan: None})
        results = top_results_df.to_dict(orient='records')
        
        # Compute RCA summary of the matched subset
        matched_rca_df = f_df[f_df['score'] > 0] if query else f_df
        if matched_rca_df.empty:
            matched_rca_df = f_df
            
        top_defects = matched_rca_df['Defect Type'].value_counts().head(3).to_dict()
        top_nc = matched_rca_df['NC Categorization'].value_counts().head(3).to_dict()
        top_vendors = matched_rca_df['Vendor Name'].value_counts().head(3).to_dict()
        
        avg_res = matched_rca_df['Days Taken for Disposition'].dropna()
        avg_res_days = float(avg_res.mean()) if not avg_res.empty else 0.0
        
        avg_sev = matched_rca_df['Severity Rating (Given by Unit)'].dropna()
        avg_sev_val = float(avg_sev.mean()) if not avg_sev.empty else 0.0
        
        # Learnings sample
        learnings = matched_rca_df['Learning Derived'].dropna().astype(str).str.strip()
        learnings = learnings[~learnings.isin(['', '--', 'UNKNOWN', 'nan', 'None'])]
        learnings = list(learnings.unique()[:5])
        
        rca_summary = {
            'top_defects': top_defects,
            'top_nc': top_nc,
            'top_vendors': top_vendors,
            'avg_resolution_days': round(avg_res_days, 1),
            'avg_severity': round(avg_sev_val, 2),
            'learnings': learnings
        }
        
        # Extract query tokens for highlighting in frontend
        stop_words = tfidf_vectorizer.get_stop_words() or set() if tfidf_vectorizer else set()
        query_tokens = [w for w in query.lower().split() if w not in stop_words] if query else []
        
        return jsonify({
            'results': results,
            'total_results': total_results,
            'query_tokens': query_tokens,
            'rca_summary': rca_summary
        })
    except Exception as e:
        print(f"Error during NLP search: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_dashboard_backend()
    app.run(host='0.0.0.0', port=5001, debug=True)

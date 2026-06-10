import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_PATH = "10yearsdata_cleaned.csv"
METRICS_JSON_PATH = "model_evaluation_metrics_10x.json"
MODELS_DIR = "models_10x"

def load_and_preprocess():
    print("Loading cleaned dataset...")
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
    
    # Ensure columns exist
    df['Days_Taken_for_Disposition'] = df['Days_Taken_for_Disposition'].fillna(0)
    df['Severity_Rating_Given_by_Unit'] = df['Severity_Rating_Given_by_Unit'].fillna(df['Severity_Rating_Given_by_Unit'].median())
    df['Final_Cost_Incurred_at_Site'] = df['Final_Cost_Incurred_at_Site'].fillna(0)
    df['No_of_Times_SARCAR_Reopened'] = df['No_of_Times_SARCAR_Reopened'].fillna(0)
    
    # 1. Derived Target Features
    df['Is_Repetitive'] = np.where(df.get('Repetitive_Issues_Identified_by_Unit_YN', '') == 'Y', 1, 0)
    df['Reopened_Flag'] = (df['No_of_Times_SARCAR_Reopened'] > 0).astype(int)
    df['Is_Delayed'] = (df['Days_Taken_for_Disposition'] > 15).astype(int)
    df['Debit_Recovered'] = np.where(df.get('Debit_Accepted_by_Unit', '') == 'Y', 1, 0)
    
    # 2. Composite Targets
    df['Vendor_Risk_Score'] = (df['Is_Repetitive'] * 3) + (df['Is_Delayed'] * 2) + (df['Reopened_Flag'] * 2) + (df['Final_Cost_Incurred_at_Site'] > 0).astype(int) * 3
    
    max_sev = df['Severity_Rating_Given_by_Unit'].max()
    max_sev = max_sev if max_sev > 0 else 1
    severity_norm = df['Severity_Rating_Given_by_Unit'] / max_sev
    penalty = (df['Is_Repetitive'] * 20) + (severity_norm * 30) + (df['Is_Delayed'] * 20) + (df['Reopened_Flag'] * 15)
    df['Reliability_Index'] = 100 - penalty
    df['Reliability_Index'] = df['Reliability_Index'].clip(lower=0, upper=100)
    
    # 3. Categorical Fill
    cat_cols = ['Product', 'Equipment_Name', 'Item', 'Defect_Type', 'Complaint_Type', 'Unit', 'Region', 'Vendor_Code', 'Unit_Disposition']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('UNKNOWN').astype(str).str.strip().str.upper()
            
            # Collapse high cardinality to prevent massive tree counts in LightGBM
            if c == 'Unit_Disposition':
                counts = df[c].value_counts()
                if len(counts) > 20:
                    top_classes = counts.head(19).index
                    df[c] = df[c].apply(lambda x: x if x in top_classes else 'OTHER_DISPOSITION')
    
    df['Problem_Description'] = df.get('Problem_Description', '').fillna('No description')
    
    return df

def build_encoders(df, cat_cols):
    label_mappings = {}
    transformed_df = df.copy()
    for col in cat_cols:
        unique_vals = sorted(transformed_df[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        mapping['UNKNOWN'] = len(unique_vals)
        label_mappings[col] = mapping
        transformed_df[col] = transformed_df[col].map(mapping).fillna(mapping['UNKNOWN']).astype(int)
    return transformed_df, label_mappings

def train_target(task_name, df, feature_cols, target_col, is_classification, label_mappings, is_nlp=False):
    print(f"\n--- Training {task_name} ---")
    
    # Drop NAs in target
    df_clean = df.dropna(subset=[target_col]).copy()
    
    if is_nlp:
        # NLP Pipeline
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df_clean['Problem_Description'])
    else:
        X = df_clean[feature_cols]
    
    if is_classification:
        # Always map classification targets to 0, 1, 2... for LightGBM
        classes = sorted(df_clean[target_col].unique())
        target_map = {v: i for i, v in enumerate(classes)}
        inv_target_map = {i: str(v) for i, v in enumerate(classes)}
        y = df_clean[target_col].map(target_map)
    else:
        y = df_clean[target_col]
        inv_target_map = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling for non-NLP
    scaler = None
    if not is_nlp:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if is_classification:
        models = {
            'Logistic': LogisticRegression(max_iter=500),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
            'LightGBM': LGBMClassifier(n_estimators=50, random_state=42)
        }
    else:
        models = {
            'Linear': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=50, random_state=42)
        }

    best_score = -1 if is_classification else float('inf')
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            if is_classification:
                score = accuracy_score(y_test, preds)
                print(f"  {name} Accuracy: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model
            else:
                score = mean_absolute_error(y_test, preds)
                print(f"  {name} MAE: {score:.4f}")
                if score < best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model
        except Exception as e:
            print(f"  {name} failed: {e}")

    print(f"Best: {best_model_name} (Score: {best_score:.4f})")
    
    # Save package
    model_pkg = {
        'model': best_model,
        'scaler': scaler,
        'is_nlp': is_nlp,
        'features': feature_cols,
        'label_mappings': label_mappings,
        'inv_target_map': inv_target_map,
        'vectorizer': vectorizer if is_nlp else None
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_path = os.path.join(MODELS_DIR, f"{task_name}.joblib")
    joblib.dump(model_pkg, file_path)
    
    return {
        'task_name': task_name,
        'best_model': best_model_name,
        'score': float(best_score),
        'is_classification': is_classification,
        'features': feature_cols
    }

def main():
    df = load_and_preprocess()
    
    cat_cols = ['Product', 'Equipment_Name', 'Item', 'Defect_Type', 'Complaint_Type', 'Unit', 'Region', 'Vendor_Code', 'Unit_Disposition']
    df_encoded, label_mappings = build_encoders(df, cat_cols)
    
    std_features = ['Equipment_Name', 'Item', 'Complaint_Type', 'Unit', 'Region', 'Vendor_Code']
    
    tasks = [
        # task_name, target_col, is_classification, is_nlp, specific_features
        ('Resolution_Time', 'Days_Taken_for_Disposition', False, False, std_features + ['Defect_Type']),
        ('Severity', 'Severity_Rating_Given_by_Unit', True, False, std_features + ['Defect_Type']),
        ('Cost', 'Final_Cost_Incurred_at_Site', False, False, std_features + ['Defect_Type']),
        ('Repeat_Failure', 'Is_Repetitive', True, False, std_features + ['Defect_Type']),
        ('Vendor_Risk', 'Vendor_Risk_Score', False, False, std_features),
        ('Defect_Root_Cause', 'Defect_Type', True, True, []), # NLP
        ('Escalation', 'Reopened_Flag', True, False, std_features),
        ('Delay', 'Is_Delayed', True, False, std_features + ['Defect_Type']),
        ('Warranty_Recovery', 'Debit_Recovered', True, False, std_features),
        ('Reliability', 'Reliability_Index', False, False, std_features),
        ('Resolution_Strategy', 'Unit_Disposition', True, False, std_features + ['Defect_Type']),
    ]
    
    results = []
    for t in tasks:
        res = train_target(t[0], df_encoded, t[4], t[1], t[2], label_mappings, t[3])
        results.append(res)
        
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print("All 10 models trained successfully.")

if __name__ == "__main__":
    main()

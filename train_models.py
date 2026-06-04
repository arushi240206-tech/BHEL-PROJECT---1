import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

CSV_PATH = "10yearsdata_cleaned.csv"
METRICS_JSON_PATH = "model_evaluation_metrics.json"

def load_and_preprocess():
    print("Loading cleaned dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Strip and replace column headers to safe variable names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
    
    # 2. Parse date columns
    df['Complaint_Date'] = pd.to_datetime(df['Complaint_Date'], errors='coerce')
    df['First_Disposition_Date'] = pd.to_datetime(df['First_Disposition_Date'], errors='coerce')
    df['Last_Disposition_Date'] = pd.to_datetime(df['Last_Disposition_Date'], errors='coerce')
    
    # 3. Extract datetime features
    df['Complaint_Year'] = df['Complaint_Date'].dt.year.fillna(2020).astype(int)
    df['Complaint_Month'] = df['Complaint_Date'].dt.month.fillna(6).astype(int)
    df['Complaint_Quarter'] = df['Complaint_Date'].dt.quarter.fillna(2).astype(int)
    df['Complaint_DayOfWeek'] = df['Complaint_Date'].dt.dayofweek.fillna(2).astype(int)
    
    # 4. Resolution Lag (days)
    df['Resolution_Lag'] = (df['Last_Disposition_Date'] - df['First_Disposition_Date']).dt.days
    df['Resolution_Lag'] = df['Resolution_Lag'].fillna(df['Resolution_Lag'].median() if not df['Resolution_Lag'].empty else 0)
    
    # 5. Derived Features
    # Is_Repetitive
    df['Is_Repetitive'] = np.where(df['Repetitive_Issues_Identified_by_Unit_YN'] == 'Y', 1, 
                                   np.where(df['Repetitive_Issues_Identified_by_Unit_YN'] == 'N', 0, np.nan))
    
    # Debit_Gap = Debit Claimed - Debit Accepted
    debit_accepted_val = np.where(df['Debit_Accepted_by_Unit'] == 'Y', df['Debit_Claimed_INR'], 
                                  np.where(df['Debit_Accepted_by_Unit'] == 'N', 0, np.nan))
    df['Debit_Gap'] = df['Debit_Claimed_INR'].fillna(0) - pd.Series(debit_accepted_val).fillna(0)
    
    # Cost_Overrun = FINAL_COST_INCURRED_SITE - EST_COST_BY_UNIT
    df['Cost_Overrun'] = df['Final_Cost_Incurred_at_Site'].fillna(0) - df['Estimated_Cost_by_Unit'].fillna(0)
    
    # Reopened_Flag
    df['Reopened_Flag'] = (df['No_of_Times_SARCAR_Reopened'].fillna(0) > 0).astype(int)
    
    # Complaint_Frequency_Monthly
    df['YearMonth'] = df['Complaint_Date'].dt.to_period('M')
    monthly_counts = df.groupby('YearMonth').size().to_dict()
    df['Complaint_Frequency_Monthly'] = df['YearMonth'].map(monthly_counts).fillna(0).astype(float)
    df = df.drop(columns=['YearMonth'])
    
    # 6. Fill missing values for features
    df['Product'] = df['Product'].fillna('UNKNOWN')
    df['Region'] = df['Region'].fillna('UNKNOWN')
    df['Defect_Type'] = df['Defect_Type'].fillna('UNKNOWN')
    df['Complaint_Type'] = df['Complaint_Type'].fillna('UNKNOWN')
    df['ShopBOI_Given_by_Site'] = df['ShopBOI_Given_by_Site'].fillna('UNKNOWN')
    df['Unit'] = df['Unit'].fillna('UNKNOWN')
    df['Cost_Debitable'] = df['Cost_Debitable'].fillna('UNKNOWN')
    df['Will_Milestone_Get_Affected_Given_by_Site'] = df['Will_Milestone_Get_Affected_Given_by_Site'].fillna('UNKNOWN')
    
    # Numeric features fill with median
    df['Severity_Rating_Given_by_Site'] = df['Severity_Rating_Given_by_Site'].fillna(df['Severity_Rating_Given_by_Site'].median())
    df['Debit_Claimed_INR'] = df['Debit_Claimed_INR'].fillna(df['Debit_Claimed_INR'].median())
    
    return df

def build_encoders_and_transform(df):
    categorical_cols = [
        'Product', 'Region', 'Defect_Type', 'Complaint_Type',
        'ShopBOI_Given_by_Site', 'Unit', 'Cost_Debitable',
        'Will_Milestone_Get_Affected_Given_by_Site'
    ]
    
    label_mappings = {}
    transformed_df = df.copy()
    
    for col in categorical_cols:
        transformed_df[col] = transformed_df[col].astype(str).str.strip().str.upper()
        unique_vals = sorted(transformed_df[col].unique())
        
        # Mapping dict
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        # Add UNKNOWN key for unseen values at prediction time
        mapping['UNKNOWN'] = len(unique_vals)
        label_mappings[col] = mapping
        
        # Transform column
        transformed_df[col] = transformed_df[col].map(mapping)
        
    return transformed_df, label_mappings

def train_and_evaluate():
    # 1. Load data
    raw_df = load_and_preprocess()
    
    # 2. Encode categorical columns
    df_encoded, label_mappings = build_encoders_and_transform(raw_df)
    
    # Feature columns
    feature_cols = [
        'Product', 'Region', 'Defect_Type', 'Severity_Rating_Given_by_Site',
        'Complaint_Type', 'ShopBOI_Given_by_Site', 'Unit', 'Cost_Debitable',
        'Will_Milestone_Get_Affected_Given_by_Site', 'Complaint_Year',
        'Complaint_Month', 'Complaint_Quarter', 'Complaint_DayOfWeek',
        'Complaint_Frequency_Monthly', 'Debit_Claimed_INR'
    ]
    
    print(f"Features list: {feature_cols}")
    
    # Dictionary to collect all metrics
    metrics_report = {
        'metadata': {
            'features': feature_cols,
            'label_mappings': label_mappings
        },
        'targets': {}
    }
    
    # Setup training configurations
    # We will train 3 targets: Target A (Classification), Target B (Regression), Target C (Classification)
    
    # ------------------ TARGET A: SEVERITY RATING GIVEN BY UNIT ------------------
    print("\n================== TRAINING TARGET A (SEVERITY CLASSIFICATION) ==================")
    # Drop rows where target is missing
    df_a = df_encoded.dropna(subset=['Severity_Rating_Given_by_Unit'])
    
    # Encode Target A
    severity_classes = sorted(df_a['Severity_Rating_Given_by_Unit'].unique())
    severity_mapping = {val: idx for idx, val in enumerate(severity_classes)}
    inverse_severity_mapping = {idx: val for idx, val in enumerate(severity_classes)}
    
    y_a = df_a['Severity_Rating_Given_by_Unit'].map(severity_mapping)
    X_a = df_a[feature_cols]
    
    # Split
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
    
    # Scaler
    scaler_a = StandardScaler()
    X_train_scaled_a = scaler_a.fit_transform(X_train_a)
    X_test_scaled_a = scaler_a.transform(X_test_a)
    
    # Models to train
    clf_models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree': (DecisionTreeClassifier(max_depth=8, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), False),
        'Support Vector Machine': (SVC(probability=True, cache_size=1000, max_iter=2000, random_state=42), True),
        'K-Nearest Neighbors': (KNeighborsClassifier(n_neighbors=5), True),
        'Neural Network': (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42), True)
    }
    
    target_a_results = []
    best_acc = -1
    best_model_name_a = ""
    best_model_obj_a = None
    
    for name, (model, use_scaling) in clf_models.items():
        print(f"Training {name} for Target A...")
        X_tr = X_train_scaled_a if use_scaling else X_train_a
        X_te = X_test_scaled_a if use_scaling else X_test_a
        
        model.fit(X_tr, y_train_a)
        y_pred = model.predict(X_te)
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)
        else:
            y_prob = None
            
        # Metrics
        acc = accuracy_score(y_test_a, y_pred)
        prec = precision_score(y_test_a, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_a, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_a, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC
        if y_prob is not None:
            try:
                if len(severity_classes) == 2:
                    roc_auc = roc_auc_score(y_test_a, y_prob[:, 1], average='weighted')
                else:
                    roc_auc = roc_auc_score(y_test_a, y_prob, multi_class='ovr', average='weighted')
            except Exception as e:
                roc_auc = 0.0
        else:
            roc_auc = 0.0
            
        cm = confusion_matrix(y_test_a, y_pred).tolist()
        report = classification_report(y_test_a, y_pred, output_dict=True, zero_division=0)
        
        # Convert report class keys back to float representation
        formatted_report = {}
        for k, v in report.items():
            if k.isdigit():
                formatted_report[str(inverse_severity_mapping[int(k)])] = v
            else:
                formatted_report[k] = v
                
        # Feature importances (Random Forest or Decision Tree / Gradient Boosting)
        importances = []
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            importances = model.feature_importances_.tolist()
        elif name == 'Logistic Regression':
            # Use absolute coefficients averaged over classes
            importances = np.mean(np.abs(model.coef_), axis=0).tolist()
        else:
            importances = [0.0] * len(feature_cols)
            
        # Feature importances formatted
        feat_imp = sorted(
            [{'feature': f, 'importance': imp} for f, imp in zip(feature_cols, importances)],
            key=lambda x: x['importance'],
            reverse=True
        )
        
        target_a_results.append({
            'model_name': name,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm,
            'classification_report': formatted_report,
            'feature_importances': feat_imp[:15]
        })
        
        if acc > best_acc:
            best_acc = acc
            best_model_name_a = name
            best_model_obj_a = (model, scaler_a if use_scaling else None)
            
    print(f"--> Best model for Target A: {best_model_name_a} (Accuracy: {best_acc:.4f})")
    
    # Save deployed best model A
    deployed_model_a = {
        'model': best_model_obj_a[0],
        'scaler': best_model_obj_a[1],
        'target_mapping': inverse_severity_mapping,
        'features': feature_cols
    }
    joblib.dump(deployed_model_a, "best_model_severity.joblib")
    
    metrics_report['targets']['severity'] = {
        'results': target_a_results,
        'best_model': best_model_name_a,
        'classes': [float(c) for c in severity_classes]
    }
    
    # ------------------ TARGET B: DAYS TAKEN FOR DISPOSITION ------------------
    print("\n================== TRAINING TARGET B (DISPOSITION REGRESSION) ==================")
    # Outlier treatment: cap at 99th percentile
    df_b = df_encoded.copy()
    cap_val = df_b['Days_Taken_for_Disposition'].quantile(0.99)
    print(f"Capping Days_Taken_for_Disposition at 99th percentile: {cap_val} days")
    df_b['Days_Taken_for_Disposition_Capped'] = df_b['Days_Taken_for_Disposition'].clip(upper=cap_val)
    
    y_b = df_b['Days_Taken_for_Disposition_Capped']
    X_b = df_b[feature_cols]
    
    # Split
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2, random_state=42)
    
    # Scaler
    scaler_b = StandardScaler()
    X_train_scaled_b = scaler_b.fit_transform(X_train_b)
    X_test_scaled_b = scaler_b.transform(X_test_b)
    
    # Regression models
    reg_models = {
        'Linear Regression': (LinearRegression(), True),
        'Decision Tree': (DecisionTreeRegressor(max_depth=8, random_state=42), False),
        'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42), False),
        'Support Vector Machine': (SVR(cache_size=1000, max_iter=2000), True),
        'K-Nearest Neighbors': (KNeighborsRegressor(n_neighbors=5), True),
        'Neural Network': (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42), True)
    }
    
    target_b_results = []
    best_rmse = float('inf')
    best_model_name_b = ""
    best_model_obj_b = None
    
    for name, (model, use_scaling) in reg_models.items():
        print(f"Training {name} for Target B...")
        X_tr = X_train_scaled_b if use_scaling else X_train_b
        X_te = X_test_scaled_b if use_scaling else X_test_b
        
        model.fit(X_tr, y_train_b)
        y_pred = model.predict(X_te)
        
        # Metrics
        mae = mean_absolute_error(y_test_b, y_pred)
        mse = mean_squared_error(y_test_b, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_b, y_pred)
        
        # Actual vs Predicted scatter points (sample 400 points to keep json compact)
        np.random.seed(42)
        indices_sample = np.random.choice(len(y_test_b), size=min(400, len(y_test_b)), replace=False)
        actuals_sample = y_test_b.iloc[indices_sample].tolist()
        predictions_sample = y_pred[indices_sample].tolist()
        scatter_data = [{'actual': float(act), 'predicted': float(pred)} for act, pred in zip(actuals_sample, predictions_sample)]
        
        # Feature importances
        importances = []
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            importances = model.feature_importances_.tolist()
        elif name == 'Linear Regression':
            importances = np.abs(model.coef_).tolist()
        else:
            importances = [0.0] * len(feature_cols)
            
        feat_imp = sorted(
            [{'feature': f, 'importance': imp} for f, imp in zip(feature_cols, importances)],
            key=lambda x: x['importance'],
            reverse=True
        )
        
        target_b_results.append({
            'model_name': name,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'scatter_plot_sample': scatter_data,
            'feature_importances': feat_imp[:15]
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name_b = name
            best_model_obj_b = (model, scaler_b if use_scaling else None)
            
    print(f"--> Best model for Target B: {best_model_name_b} (RMSE: {best_rmse:.4f})")
    
    # Save deployed best model B
    deployed_model_b = {
        'model': best_model_obj_b[0],
        'scaler': best_model_obj_b[1],
        'features': feature_cols
    }
    joblib.dump(deployed_model_b, "best_model_disposition.joblib")
    
    metrics_report['targets']['disposition'] = {
        'results': target_b_results,
        'best_model': best_model_name_b
    }
    
    # ------------------ TARGET C: REPETITIVE ISSUE DETECTION ------------------
    print("\n================== TRAINING TARGET C (REPETITIVE BINARY CLASSIFICATION) ==================")
    # Drop rows where target is missing
    df_c = df_encoded.dropna(subset=['Is_Repetitive'])
    
    y_c = df_c['Is_Repetitive'].astype(int)
    X_c = df_c[feature_cols]
    
    # Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    
    # Scaler
    scaler_c = StandardScaler()
    X_train_scaled_c = scaler_c.fit_transform(X_train_c)
    X_test_scaled_c = scaler_c.transform(X_test_c)
    
    target_c_results = []
    best_acc_c = -1
    best_model_name_c = ""
    best_model_obj_c = None
    
    for name, (model, use_scaling) in clf_models.items():
        print(f"Training {name} for Target C...")
        X_tr = X_train_scaled_c if use_scaling else X_train_c
        X_te = X_test_scaled_c if use_scaling else X_test_c
        
        model.fit(X_tr, y_train_c)
        y_pred = model.predict(X_te)
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)
        else:
            y_prob = None
            
        # Metrics
        acc = accuracy_score(y_test_c, y_pred)
        prec = precision_score(y_test_c, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_c, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_c, y_pred, average='weighted', zero_division=0)
        
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test_c, y_prob[:, 1])
        else:
            roc_auc = 0.0
            
        cm = confusion_matrix(y_test_c, y_pred).tolist()
        report = classification_report(y_test_c, y_pred, output_dict=True, zero_division=0)
        
        # Feature importances
        importances = []
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            importances = model.feature_importances_.tolist()
        elif name == 'Logistic Regression':
            importances = np.abs(model.coef_[0]).tolist()
        else:
            importances = [0.0] * len(feature_cols)
            
        feat_imp = sorted(
            [{'feature': f, 'importance': imp} for f, imp in zip(feature_cols, importances)],
            key=lambda x: x['importance'],
            reverse=True
        )
        
        target_c_results.append({
            'model_name': name,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm,
            'classification_report': report,
            'feature_importances': feat_imp[:15]
        })
        
        if acc > best_acc_c:
            best_acc_c = acc
            best_model_name_c = name
            best_model_obj_c = (model, scaler_c if use_scaling else None)
            
    print(f"--> Best model for Target C: {best_model_name_c} (Accuracy: {best_acc_c:.4f})")
    
    # Save deployed best model C
    deployed_model_c = {
        'model': best_model_obj_c[0],
        'scaler': best_model_obj_c[1],
        'features': feature_cols
    }
    joblib.dump(deployed_model_c, "best_model_repetitive.joblib")
    
    metrics_report['targets']['repetitive'] = {
        'results': target_c_results,
        'best_model': best_model_name_c,
        'classes': [0, 1]
    }
    
    # Save all metrics to JSON
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Successfully saved evaluation metrics to {METRICS_JSON_PATH}!")
    print("Model training pipeline completed!")

if __name__ == "__main__":
    train_and_evaluate()

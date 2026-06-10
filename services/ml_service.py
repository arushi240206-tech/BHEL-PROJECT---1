import os
import json
import joblib
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MLService:
    def __init__(self, metrics_path):
        self.metrics_path = metrics_path
        self.metrics_report = {}
        self.models_10x = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def load_metrics(self):
        if os.path.exists(self.metrics_path):
            print("Loading precomputed ML metrics...")
            with open(self.metrics_path, 'r') as f:
                self.metrics_report = json.load(f)

    def load_models(self, models_dir="models_10x"):
        if os.path.exists(models_dir):
            print(f"Loading deployed 10x models from {models_dir}...")
            for filepath in glob.glob(f"{models_dir}/*.joblib"):
                task_name = os.path.basename(filepath).replace(".joblib", "")
                self.models_10x[task_name] = joblib.load(filepath)
                print(f" Loaded {task_name}...")

    def fit_tfidf(self, df):
        print("Fitting TF-IDF Vectorizer on search corpus...")
        corpus_columns = [
            'Problem Description',
            'Item',
            'Defect Type',
            'Defect Sub-type Description',
            'Problem Nature Keywords',
            'Product',
            'Project',
            'PGMA Description',
            'Equipment Name'
        ]
        search_corpus = df[corpus_columns].fillna('').agg(' '.join, axis=1)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(search_corpus)
        print("TF-IDF fit complete. Matrix shape:", self.tfidf_matrix.shape)

    def get_ml_metadata(self):
        available_models = {}
        for task_name, pkg in self.models_10x.items():
            available_models[task_name] = {
                'features': pkg.get('features', []),
                'is_nlp': pkg.get('is_nlp', False),
                'label_mappings': {k: list(v.keys()) for k, v in pkg.get('label_mappings', {}).items()}
            }
        return available_models

    def get_metrics(self):
        return self.metrics_report

    def predict_multiple(self, requested_targets, inputs):
        if not self.models_10x:
            raise Exception('10x Models are not loaded on backend.')
            
        results = {}
        
        for task_name in requested_targets:
            if task_name not in self.models_10x:
                results[task_name] = {'error': 'Model not found'}
                continue
                
            pkg = self.models_10x[task_name]
            model = pkg['model']
            scaler = pkg.get('scaler')
            is_nlp = pkg.get('is_nlp', False)
            features = pkg.get('features', [])
            label_mappings = pkg.get('label_mappings', {})
            inv_target_map = pkg.get('inv_target_map')
            vectorizer = pkg.get('vectorizer')
            
            if is_nlp:
                prob_desc = str(inputs.get('Problem_Description', ''))
                X_vec = vectorizer.transform([prob_desc])
                X_final = X_vec
            else:
                row = []
                for f in features:
                    val_str = str(inputs.get(f, 'UNKNOWN')).strip().upper()
                    mapping = label_mappings.get(f, {})
                    enc_val = mapping.get(val_str, mapping.get('UNKNOWN', 0))
                    row.append(enc_val)
                X_arr = np.array([row])
                X_final = scaler.transform(X_arr) if scaler else X_arr
                
            pred = model.predict(X_final)[0]
            
            if inv_target_map: # classification
                if isinstance(pred, (np.integer, int, float)) and int(pred) in inv_target_map:
                    pred_label = inv_target_map[int(pred)]
                else:
                    pred_label = inv_target_map.get(pred, str(pred))
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_final)[0]
                    idx = list(model.classes_).index(pred) if pred in model.classes_ else 0
                    conf = float(probs[idx])
                else:
                    conf = 1.0
                    
                results[task_name] = {
                    'prediction': str(pred_label),
                    'confidence': round(conf * 100, 1),
                    'type': 'classification'
                }
            else: # regression
                results[task_name] = {
                    'prediction': round(float(pred), 2),
                    'type': 'regression'
                }
                
        return results

    def run_nlp_search(self, query, f_df, limit):
        if f_df.empty:
            return {
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
            }
            
        indices = f_df.index.tolist()
        
        if query and self.tfidf_vectorizer is not None:
            query_vec = self.tfidf_vectorizer.transform([query])
            sliced_matrix = self.tfidf_matrix[indices]
            sims = cosine_similarity(query_vec, sliced_matrix).flatten()
            
            f_df = f_df.copy()
            f_df['score'] = sims
            
            f_df = f_df[f_df['score'] > 0]
            f_df = f_df.sort_values(by=['score', 'Sno'], ascending=[False, False])
        else:
            f_df = f_df.copy()
            f_df['score'] = 0.0
            if 'Complaint Date' in f_df.columns:
                f_df = f_df.sort_values(by='Complaint Date', ascending=False)
            
        total_results = len(f_df)
        top_results_df = f_df.head(limit)
        top_results_df = top_results_df.replace({np.nan: None})
        
        # Replace datetime objects with strings before to_dict to prevent JSON serialization errors
        for col in top_results_df.select_dtypes(include=['datetime64']).columns:
            top_results_df[col] = top_results_df[col].astype(str)

        results = top_results_df.to_dict(orient='records')
        
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
        
        stop_words = self.tfidf_vectorizer.get_stop_words() or set() if self.tfidf_vectorizer else set()
        query_tokens = [w for w in query.lower().split() if w not in stop_words] if query else []
        
        return {
            'results': results,
            'total_results': total_results,
            'query_tokens': query_tokens,
            'rca_summary': rca_summary
        }

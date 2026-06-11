import os
import hashlib
import pandas as pd
from datetime import datetime


class FeedbackEngine:
    """
    Manages engineer feedback on AI recommendations and the
    case approval workflow for continuous learning.

    Persistence: CSV files in the data/ directory (matching the
    project's existing architecture — no database required).
    """

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.feedback_path = os.path.join(data_dir, 'feedback.csv')
        self.approved_path = os.path.join(data_dir, 'approved_cases.csv')
        os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    def submit_feedback(self, query, recommendation_id, equipment_type,
                        defect_type, rating, notes=''):
        """
        Store a single feedback entry.

        rating: 'helpful' or 'not_helpful'
        """
        row = {
            'timestamp': datetime.now().isoformat(),
            'complaint_query': str(query)[:500],
            'recommendation_id': recommendation_id or self._hash(query),
            'equipment_type': equipment_type or '',
            'defect_type': defect_type or '',
            'rating': rating,
            'engineer_notes': str(notes)[:500]
        }

        df = pd.DataFrame([row])
        header = not os.path.exists(self.feedback_path) or os.path.getsize(self.feedback_path) < 10
        df.to_csv(self.feedback_path, mode='a', header=header, index=False)

    def get_feedback_weight(self, equipment_type, defect_type):
        """
        Compute the average helpfulness ratio for a given
        equipment + defect combination.

        Returns a float between 0.0 (all unhelpful) and 1.0 (all helpful).
        Returns 0.5 (neutral) if no feedback exists.
        """
        if not os.path.exists(self.feedback_path):
            return 0.5

        try:
            fb = pd.read_csv(self.feedback_path)
            if fb.empty:
                return 0.5

            # Filter to matching equipment/defect
            mask = pd.Series([True] * len(fb))
            if equipment_type:
                mask &= fb['equipment_type'].str.lower() == equipment_type.lower()
            if defect_type:
                mask &= fb['defect_type'].str.lower() == defect_type.lower()

            matched = fb[mask]
            if matched.empty:
                return 0.5

            helpful_count = (matched['rating'] == 'helpful').sum()
            total = len(matched)
            return round(helpful_count / total, 3)

        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Case Approval Workflow
    # ------------------------------------------------------------------
    def submit_resolved_case(self, case_data):
        """
        Submit a new resolved case to the pending queue.

        case_data should contain:
            problem_description, equipment_name, defect_type,
            severity, resolution, learning_derived
        """
        case_id = self._hash(
            case_data.get('problem_description', '') +
            datetime.now().isoformat()
        )[:12]

        row = {
            'timestamp': datetime.now().isoformat(),
            'case_id': case_id,
            'problem_description': case_data.get('problem_description', ''),
            'equipment_name': case_data.get('equipment_name', ''),
            'defect_type': case_data.get('defect_type', ''),
            'severity': case_data.get('severity', ''),
            'resolution': case_data.get('resolution', ''),
            'learning_derived': case_data.get('learning_derived', ''),
            'approved_by': '',
            'status': 'pending'
        }

        df = pd.DataFrame([row])
        header = not os.path.exists(self.approved_path) or os.path.getsize(self.approved_path) < 10
        df.to_csv(self.approved_path, mode='a', header=header, index=False)
        return case_id

    def get_pending_cases(self):
        """Return all cases with status='pending'."""
        if not os.path.exists(self.approved_path):
            return []

        try:
            df = pd.read_csv(self.approved_path)
            pending = df[df['status'] == 'pending']
            return pending.to_dict(orient='records')
        except Exception:
            return []

    def approve_case(self, case_id, approved_by='engineer'):
        """Mark a pending case as approved."""
        if not os.path.exists(self.approved_path):
            return False

        try:
            df = pd.read_csv(self.approved_path)
            mask = (df['case_id'].astype(str) == str(case_id)) & (df['status'] == 'pending')
            if mask.sum() == 0:
                return False

            df.loc[mask, 'status'] = 'approved'
            df.loc[mask, 'approved_by'] = approved_by
            df.to_csv(self.approved_path, index=False)
            return True
        except Exception:
            return False

    def merge_approved_cases(self, data_service, semantic_search_service=None):
        """
        Merge approved cases into the main dataset.

        1. Read approved cases with status='approved'.
        2. Map them to the main DataFrame columns.
        3. Append to data_service.df and re-save CSV.
        4. Rebuild semantic index if available.
        5. Mark merged cases as 'merged'.
        """
        if not os.path.exists(self.approved_path):
            return 0

        try:
            cases_df = pd.read_csv(self.approved_path)
            approved = cases_df[cases_df['status'] == 'approved']

            if approved.empty:
                return 0

            # Map case fields to main DataFrame columns
            new_rows = []
            for _, case in approved.iterrows():
                new_row = {
                    'Problem Description': case.get('problem_description', ''),
                    'Equipment Name': case.get('equipment_name', 'Unknown'),
                    'Defect Type': case.get('defect_type', 'Unknown'),
                    'Severity Rating (Given by Unit)': float(case.get('severity', 0)) if case.get('severity') else 0.0,
                    'Unit Disposition': case.get('resolution', ''),
                    'Learning Derived': case.get('learning_derived', ''),
                    'Complaint Date': datetime.now().strftime('%Y-%m-%d'),
                    'Status': 'Resolved',
                    'Region': 'Unknown',
                    'Unit': 'Unknown',
                    'Product': 'Unknown',
                    'Project': 'Unknown',
                    'Item': 'Unknown',
                    'Vendor Name': 'Unknown'
                }
                new_rows.append(new_row)

            new_df = pd.DataFrame(new_rows)
            count = len(new_df)

            # Append to main DataFrame
            data_service.df = pd.concat([data_service.df, new_df], ignore_index=True)

            # Re-parse dates for new rows
            data_service.df['Complaint Date'] = pd.to_datetime(
                data_service.df['Complaint Date'], errors='coerce'
            )
            data_service.df['Complaint Year'] = data_service.df['Complaint Date'].dt.year.fillna(2020).astype(int)
            data_service.df['Complaint Month'] = data_service.df['Complaint Date'].dt.month.fillna(6).astype(int)

            # Save updated CSV
            data_service.df.to_csv(data_service.csv_path, index=False)

            # Rebuild semantic index
            if semantic_search_service:
                semantic_search_service.rebuild(data_service.df)

            # Mark as merged
            cases_df.loc[cases_df['status'] == 'approved', 'status'] = 'merged'
            cases_df.to_csv(self.approved_path, index=False)

            print(f"[FeedbackEngine] Merged {count} approved cases into training data.")
            return count

        except Exception as e:
            print(f"[FeedbackEngine] Merge failed: {e}")
            return 0

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def get_feedback_analytics(self):
        """Return aggregated feedback statistics for the dashboard."""
        result = {
            'total_feedback': 0,
            'helpful_count': 0,
            'not_helpful_count': 0,
            'helpfulness_rate': 0.0,
            'by_equipment': {},
            'by_defect': {},
            'recent_trend': []
        }

        if not os.path.exists(self.feedback_path):
            return result

        try:
            fb = pd.read_csv(self.feedback_path)
            if fb.empty:
                return result

            result['total_feedback'] = len(fb)
            result['helpful_count'] = int((fb['rating'] == 'helpful').sum())
            result['not_helpful_count'] = int((fb['rating'] == 'not_helpful').sum())
            result['helpfulness_rate'] = round(
                result['helpful_count'] / result['total_feedback'] * 100, 1
            ) if result['total_feedback'] > 0 else 0.0

            # By equipment type
            eq_groups = fb.groupby('equipment_type')['rating'].apply(
                lambda x: round((x == 'helpful').sum() / len(x) * 100, 1)
            ).to_dict()
            result['by_equipment'] = eq_groups

            # By defect type
            def_groups = fb.groupby('defect_type')['rating'].apply(
                lambda x: round((x == 'helpful').sum() / len(x) * 100, 1)
            ).to_dict()
            result['by_defect'] = def_groups

            # Recent trend (last 10 entries)
            recent = fb.tail(10)[['timestamp', 'rating']].to_dict(orient='records')
            result['recent_trend'] = recent

            return result

        except Exception:
            return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _hash(text):
        return hashlib.md5(text.encode()).hexdigest()

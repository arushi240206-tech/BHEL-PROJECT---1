import numpy as np
import pandas as pd


class MaintenancePredictor:
    """
    Predictive maintenance intelligence layer.

    Uses historical complaint sequences to detect:
    - Repeated failures and emerging failure patterns
    - Equipment deterioration trends
    - Risk scores and maintenance priority rankings
    """

    # Risk score weights
    W_FAILURE_COUNT = 0.30
    W_TREND = 0.25
    W_SEVERITY = 0.20
    W_REPEAT_RATE = 0.15
    W_SLA = 0.10

    def __init__(self, data_service):
        self.data_service = data_service

    def compute_equipment_health(self):
        """
        Compute health/risk metrics for all equipment in the dataset.

        Returns:
            list of dicts sorted by risk_score descending, each containing:
            - equipment_name
            - total_complaints
            - recent_complaints (last 2 years)
            - failure_trend ("accelerating", "stable", "declining")
            - avg_severity
            - repeat_rate
            - avg_resolution_days
            - risk_score (0.0 - 1.0)
            - failure_probability (0 - 100)
            - maintenance_priority ("Critical", "High", "Medium", "Low")
        """
        df = self.data_service.df
        if df is None or df.empty:
            return []

        df_eq = df.dropna(subset=['Equipment Name'])
        df_eq = df_eq[~df_eq['Equipment Name'].isin(['Unknown', 'UNKNOWN', 'nan', 'None', ''])]

        if df_eq.empty:
            return []

        # Determine "recent" as the last 2 years in the dataset
        max_year = df_eq['Complaint Year'].max()
        recent_cutoff = max_year - 2
        recent_df = df_eq[df_eq['Complaint Year'] >= recent_cutoff]

        # Global stats for normalization
        equipment_list = df_eq['Equipment Name'].value_counts()
        max_total = equipment_list.max() if not equipment_list.empty else 1
        max_recent = recent_df['Equipment Name'].value_counts().max() if not recent_df.empty else 1

        results = []
        for eq_name in equipment_list.head(50).index:  # Top 50 equipment by complaint count
            eq_df = df_eq[df_eq['Equipment Name'] == eq_name]
            eq_recent = recent_df[recent_df['Equipment Name'] == eq_name]

            total_complaints = len(eq_df)
            recent_complaints = len(eq_recent)

            # Failure trend: linear regression on monthly complaint counts
            trend = self._compute_trend(eq_df)

            # Average severity
            sev = eq_df['Severity Rating (Given by Unit)'].dropna()
            avg_severity = float(sev.mean()) if not sev.empty else 0.0

            # Repeat rate
            rep = eq_df['Repetitive Issues Identified by Unit (Y/N)'].dropna()
            repeat_rate = float((rep == 'Y').sum() / len(rep)) if len(rep) > 0 else 0.0

            # Average resolution days
            res_days = eq_df['Days Taken for Disposition'].dropna()
            avg_res_days = float(res_days.mean()) if not res_days.empty else 0.0

            # Normalize components
            norm_failure = min(recent_complaints / max(max_recent, 1), 1.0)
            norm_trend = (trend + 1) / 2  # trend is -1 to 1, normalize to 0-1
            norm_severity = min(avg_severity, 1.0)  # already 0-1 scale
            norm_sla = min(avg_res_days / 365, 1.0)  # cap at 1 year

            # Composite risk score
            risk_score = (
                self.W_FAILURE_COUNT * norm_failure +
                self.W_TREND * norm_trend +
                self.W_SEVERITY * norm_severity +
                self.W_REPEAT_RATE * repeat_rate +
                self.W_SLA * norm_sla
            )
            risk_score = min(max(risk_score, 0.0), 1.0)

            # Failure probability (sigmoid-like mapping)
            failure_prob = int(100 / (1 + np.exp(-10 * (risk_score - 0.5))))

            # Maintenance priority
            if risk_score >= 0.7:
                priority = 'Critical'
            elif risk_score >= 0.5:
                priority = 'High'
            elif risk_score >= 0.3:
                priority = 'Medium'
            else:
                priority = 'Low'

            results.append({
                'equipment_name': eq_name,
                'total_complaints': total_complaints,
                'recent_complaints': recent_complaints,
                'failure_trend': self._trend_label(trend),
                'avg_severity': round(avg_severity, 3),
                'repeat_rate': round(repeat_rate * 100, 1),
                'avg_resolution_days': round(avg_res_days, 1),
                'risk_score': round(risk_score, 3),
                'failure_probability': failure_prob,
                'maintenance_priority': priority
            })

        # Sort by risk score descending
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results

    def detect_failure_patterns(self, equipment_name):
        """
        Analyze complaint sequences for a specific piece of equipment.

        Returns:
            dict with:
            - pattern_detected: bool
            - pattern_type: "accelerating", "stable", "declining"
            - recent_defects: list of (defect_type, count) tuples
            - monthly_counts: list of (month_label, count) for the last 24 months
            - recommendation: text summary
        """
        df = self.data_service.df
        if df is None or df.empty:
            return {'pattern_detected': False, 'pattern_type': 'unknown'}

        eq_df = df[df['Equipment Name'] == equipment_name]
        if eq_df.empty:
            return {'pattern_detected': False, 'pattern_type': 'unknown'}

        trend = self._compute_trend(eq_df)
        trend_label = self._trend_label(trend)

        # Recent defect breakdown
        max_year = eq_df['Complaint Year'].max()
        recent = eq_df[eq_df['Complaint Year'] >= max_year - 1]
        defect_counts = recent['Defect Type'].value_counts().head(5).to_dict()

        # Monthly complaint counts for the last 24 months
        eq_df = eq_df.copy()
        eq_df['YearMonth'] = eq_df['Complaint Date'].dt.to_period('M')
        monthly = eq_df.groupby('YearMonth').size()
        monthly = monthly.tail(24)
        monthly_data = [
            {'month': str(period), 'count': int(count)}
            for period, count in monthly.items()
        ]

        # Generate recommendation text
        if trend > 0.3:
            rec = f"ALERT: {equipment_name} shows an accelerating failure pattern. Schedule immediate inspection and consider proactive maintenance."
        elif trend > 0:
            rec = f"{equipment_name} shows a slightly increasing failure trend. Monitor closely and plan preventive maintenance."
        else:
            rec = f"{equipment_name} failure rate is stable or declining. Continue standard maintenance schedule."

        return {
            'pattern_detected': abs(trend) > 0.2,
            'pattern_type': trend_label,
            'recent_defects': [
                {'defect_type': dt, 'count': int(c)} for dt, c in defect_counts.items()
            ],
            'monthly_counts': monthly_data,
            'recommendation': rec,
            'total_complaints': len(eq_df)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_trend(self, eq_df):
        """
        Compute a linear trend slope on monthly complaint counts.

        Returns a value between -1 (strongly declining) and +1 (strongly accelerating).
        0 means stable.
        """
        if eq_df.empty:
            return 0.0

        eq_copy = eq_df.copy()
        eq_copy['YearMonth'] = eq_copy['Complaint Date'].dt.to_period('M')
        monthly = eq_copy.groupby('YearMonth').size()

        if len(monthly) < 3:
            return 0.0

        x = np.arange(len(monthly), dtype=float)
        y = monthly.values.astype(float)

        # Normalize x to [0, 1]
        x_norm = x / max(x.max(), 1)

        # Simple linear regression
        x_mean = x_norm.mean()
        y_mean = y.mean()
        numerator = ((x_norm - x_mean) * (y - y_mean)).sum()
        denominator = ((x_norm - x_mean) ** 2).sum()

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize slope to [-1, 1] using tanh
        normalized = float(np.tanh(slope / max(y_mean, 1)))
        return normalized

    @staticmethod
    def _trend_label(trend_value):
        if trend_value > 0.2:
            return 'accelerating'
        elif trend_value < -0.2:
            return 'declining'
        else:
            return 'stable'

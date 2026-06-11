import os
import json


class KnowledgeEngine:
    """
    Loads domain-specific engineering knowledge from JSON files and
    matches incoming complaint queries to the most relevant equipment
    type and failure mode.
    """

    def __init__(self, knowledge_dir='knowledge'):
        self.knowledge_dir = knowledge_dir
        self.knowledge_base = {}   # equipment_type -> full JSON object
        self.keyword_index = {}    # keyword (lowered) -> equipment_type
        self.equipment_types = []

    def load_knowledge(self):
        """Load all JSON knowledge files from the knowledge directory."""
        if not os.path.isdir(self.knowledge_dir):
            print(f"[KnowledgeEngine] Knowledge directory not found: {self.knowledge_dir}")
            return

        count = 0
        for filename in os.listdir(self.knowledge_dir):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(self.knowledge_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                eq_type = data.get('equipment_type', filename.replace('.json', ''))
                self.knowledge_base[eq_type] = data
                self.equipment_types.append(eq_type)

                # Index all keywords for this equipment type
                for kw in data.get('keywords', []):
                    self.keyword_index[kw.lower()] = eq_type

                count += 1
            except Exception as e:
                print(f"[KnowledgeEngine] Error loading {filename}: {e}")

        self.equipment_types.sort()
        print(f"[KnowledgeEngine] Loaded {count} knowledge files: {self.equipment_types}")

    def match_equipment(self, query):
        """
        Scan the query text against all keyword lists and return the
        best-matching equipment type based on keyword hit count.

        Returns:
            (equipment_type, match_count) or (None, 0) if no match.
        """
        if not query:
            return None, 0

        query_lower = query.lower()
        scores = {}

        for kw, eq_type in self.keyword_index.items():
            if kw in query_lower:
                scores[eq_type] = scores.get(eq_type, 0) + 1

        if not scores:
            return None, 0

        best = max(scores, key=scores.get)
        return best, scores[best]

    def get_knowledge(self, equipment_type, failure_keywords=None):
        """
        Retrieve the knowledge entry for a given equipment type.

        If failure_keywords are provided, attempt to match the most
        relevant failure mode within the equipment's knowledge base.

        Returns a dict with:
            - equipment_type
            - matched_failure (the best failure entry or None)
            - all_failures (list of failure names for context)
            - diagnostic_procedures
            - corrective_actions
            - preventive_actions
            - safety_precautions
        """
        if equipment_type not in self.knowledge_base:
            return None

        kb = self.knowledge_base[equipment_type]
        failures = kb.get('common_failures', [])

        result = {
            'equipment_type': equipment_type,
            'matched_failure': None,
            'all_failures': [f['failure'] for f in failures],
            'diagnostic_procedures': [],
            'corrective_actions': [],
            'preventive_actions': [],
            'safety_precautions': []
        }

        # Try to match a specific failure mode
        best_failure = None
        best_score = 0

        if failure_keywords:
            kw_lower = failure_keywords.lower() if isinstance(failure_keywords, str) else ' '.join(failure_keywords).lower()
            for failure_entry in failures:
                failure_text = failure_entry['failure'].lower()
                # Count overlapping words
                score = sum(1 for word in kw_lower.split() if word in failure_text)
                # Also check probable causes for deeper matching
                causes_text = ' '.join(failure_entry.get('probable_causes', [])).lower()
                score += sum(0.5 for word in kw_lower.split() if word in causes_text)

                if score > best_score:
                    best_score = score
                    best_failure = failure_entry

        # If no keyword match, use the first failure as default context
        if best_failure is None and failures:
            best_failure = failures[0]

        if best_failure:
            result['matched_failure'] = best_failure['failure']
            result['diagnostic_procedures'] = best_failure.get('diagnostic_procedures', [])
            result['corrective_actions'] = best_failure.get('corrective_actions', [])
            result['preventive_actions'] = best_failure.get('preventive_actions', [])
            result['safety_precautions'] = best_failure.get('safety_precautions', [])

        return result

    def get_equipment_types(self):
        """Return list of all known equipment types for UI dropdowns."""
        return self.equipment_types

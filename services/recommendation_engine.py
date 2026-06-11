import numpy as np
import hashlib


class RecommendationEngine:
    """
    Hybrid retrieval + ranking engine that combines:
    1. Semantic search (Sentence Transformers + FAISS)
    2. TF-IDF search (existing MLService)
    3. Engineering knowledge base
    4. Engineer feedback scores

    Produces structured, explainable recommendations that resemble
    guidance from an experienced BHEL reliability engineer.
    """

    # Weights for the composite ranking formula
    W_SEMANTIC = 0.50
    W_TFIDF = 0.20
    W_FEEDBACK = 0.15
    W_FREQUENCY = 0.15

    def __init__(self, semantic_search, ml_service, data_service,
                 knowledge_engine, feedback_engine):
        self.semantic = semantic_search
        self.ml_service = ml_service
        self.data_service = data_service
        self.knowledge = knowledge_engine
        self.feedback = feedback_engine

    def get_recommendations(self, query, equipment_type=None):
        """
        Main entry point. Produces a full structured recommendation.

        Args:
            query: Natural language problem description.
            equipment_type: Optional equipment type hint from the user.

        Returns:
            dict with structured recommendation + explainability.
        """
        df = self.data_service.df
        if df is None or df.empty:
            return self._empty_result()

        # ----------------------------------------------------------
        # Step 1: Semantic retrieval (primary)
        # ----------------------------------------------------------
        semantic_results = []
        if self.semantic and self.semantic.ready:
            raw = self.semantic.search(query, top_k=20)
            for idx, score in raw:
                semantic_results.append({'idx': idx, 'semantic_score': score})

        # ----------------------------------------------------------
        # Step 2: TF-IDF retrieval (secondary / fallback)
        # ----------------------------------------------------------
        tfidf_results = {}
        if self.ml_service and self.ml_service.tfidf_vectorizer is not None:
            tfidf_raw = self._search_tfidf(query, top_k=20)
            for idx, score in tfidf_raw:
                tfidf_results[idx] = score

        # ----------------------------------------------------------
        # Step 3: Merge & deduplicate
        # ----------------------------------------------------------
        merged = {}
        for item in semantic_results:
            idx = item['idx']
            merged[idx] = {
                'semantic_score': item['semantic_score'],
                'tfidf_score': tfidf_results.get(idx, 0.0)
            }

        for idx, score in tfidf_results.items():
            if idx not in merged:
                merged[idx] = {
                    'semantic_score': 0.0,
                    'tfidf_score': score
                }

        if not merged:
            return self._empty_result()

        # ----------------------------------------------------------
        # Step 4: Enrich with DataFrame columns
        # ----------------------------------------------------------
        enriched = []
        for idx, scores in merged.items():
            if idx not in df.index:
                continue
            row = df.loc[idx]
            enriched.append({
                'idx': idx,
                'semantic_score': scores['semantic_score'],
                'tfidf_score': scores['tfidf_score'],
                'complaint_number': str(row.get('Complaint Number', f'Row-{idx}')),
                'equipment': str(row.get('Equipment Name', 'Unknown')),
                'defect_type': str(row.get('Defect Type', 'Unknown')),
                'defect_subtype': str(row.get('Defect Sub-type Description', '')),
                'nc_categorization': str(row.get('NC Categorization', '')),
                'learning': str(row.get('Learning Derived', '')),
                'disposition': str(row.get('Unit Disposition', '')),
                'resolution_days': row.get('Days Taken for Disposition', None),
                'severity': row.get('Severity Rating (Given by Unit)', None),
                'problem_desc': str(row.get('Problem Description', '')),
                'product': str(row.get('Product', '')),
                'vendor': str(row.get('Vendor Name', '')),
            })

        # ----------------------------------------------------------
        # Step 5: Knowledge augmentation
        # ----------------------------------------------------------
        if not equipment_type:
            eq_match, _ = self.knowledge.match_equipment(query)
            equipment_type = eq_match

        knowledge_data = None
        if equipment_type:
            knowledge_data = self.knowledge.get_knowledge(equipment_type, query)

        # ----------------------------------------------------------
        # Step 6: Compute composite ranking
        # ----------------------------------------------------------
        # Frequency: count how often each defect_type appears in enriched set
        defect_counts = {}
        for item in enriched:
            dt = item['defect_type']
            defect_counts[dt] = defect_counts.get(dt, 0) + 1
        max_freq = max(defect_counts.values()) if defect_counts else 1

        for item in enriched:
            # Feedback weight
            fb_weight = self.feedback.get_feedback_weight(
                item['equipment'], item['defect_type']
            )
            # Frequency weight (normalized)
            freq_weight = defect_counts.get(item['defect_type'], 0) / max_freq

            item['feedback_weight'] = fb_weight
            item['frequency_weight'] = freq_weight
            item['final_score'] = (
                self.W_SEMANTIC * item['semantic_score'] +
                self.W_TFIDF * item['tfidf_score'] +
                self.W_FEEDBACK * fb_weight +
                self.W_FREQUENCY * freq_weight
            )

        # Sort descending by final_score
        enriched.sort(key=lambda x: x['final_score'], reverse=True)
        top_10 = enriched[:10]

        # ----------------------------------------------------------
        # Step 7: Generate structured recommendation
        # ----------------------------------------------------------
        recommendation = self._build_structured_output(top_10, knowledge_data, query)

        # ----------------------------------------------------------
        # Step 8: Build explainability
        # ----------------------------------------------------------
        explanation = self._build_explanation(top_10, knowledge_data, equipment_type, query)

        # Recommendation ID for feedback tracking
        rec_id = hashlib.md5((query + str(recommendation.get('recommended_actions', []))).encode()).hexdigest()[:16]

        return {
            'recommendation': recommendation,
            'similar_cases': self._format_cases(top_10),
            'explanation': explanation,
            'recommendation_id': rec_id,
            'equipment_type_detected': equipment_type
        }

    # ------------------------------------------------------------------
    # Structured output generation
    # ------------------------------------------------------------------
    def _build_structured_output(self, top_cases, knowledge_data, query):
        """Generate the structured recommendation dict."""

        # Probable root causes: combine defect types from top cases + knowledge
        defect_counts = {}
        for c in top_cases:
            dt = c['defect_type']
            if dt and dt not in ('Unknown', 'nan', 'None', ''):
                defect_counts[dt] = defect_counts.get(dt, 0) + 1

        root_causes = sorted(defect_counts.keys(), key=lambda k: defect_counts[k], reverse=True)[:5]

        # Diagnostic checks from knowledge base
        diagnostic_checks = []
        if knowledge_data:
            diagnostic_checks = knowledge_data.get('diagnostic_procedures', [])

        # Recommended actions: top unique dispositions
        dispositions = {}
        for c in top_cases:
            d = c['disposition']
            if d and d not in ('Unknown', 'nan', 'None', '', '--'):
                dispositions[d] = dispositions.get(d, 0) + c['final_score']

        recommended_actions = sorted(dispositions.keys(), key=lambda k: dispositions[k], reverse=True)[:5]

        # Preventive measures: combine knowledge + learnings
        preventive = []
        if knowledge_data:
            preventive.extend(knowledge_data.get('preventive_actions', []))

        learnings_seen = set()
        for c in top_cases:
            lrn = c['learning']
            if lrn and lrn not in ('Unknown', 'nan', 'None', '', '--') and lrn not in learnings_seen:
                preventive.append(lrn)
                learnings_seen.add(lrn)
                if len(preventive) >= 8:
                    break

        # Safety precautions from knowledge
        safety = []
        if knowledge_data:
            safety = knowledge_data.get('safety_precautions', [])

        # Confidence score
        confidence = self._compute_confidence(top_cases, knowledge_data)

        return {
            'probable_root_causes': root_causes,
            'diagnostic_checks': diagnostic_checks,
            'recommended_actions': recommended_actions,
            'preventive_measures': preventive[:8],
            'safety_precautions': safety,
            'confidence_score': confidence
        }

    def _compute_confidence(self, top_cases, knowledge_data):
        """
        Compute an overall confidence score (0-100) based on:
        - Quality of semantic matches
        - Consistency of defect types across matches
        - Whether knowledge base matched
        - Amount of historical data
        """
        if not top_cases:
            return 0

        # Factor 1: Average semantic score of top 5 (0-1)
        top_5_scores = [c['semantic_score'] for c in top_cases[:5]]
        avg_semantic = np.mean(top_5_scores) if top_5_scores else 0

        # Factor 2: Defect type consistency (how many agree on same type)
        defect_types = [c['defect_type'] for c in top_cases[:5] if c['defect_type'] not in ('Unknown', 'nan', '')]
        if defect_types:
            most_common_count = max(defect_types.count(dt) for dt in set(defect_types))
            consistency = most_common_count / len(defect_types)
        else:
            consistency = 0

        # Factor 3: Knowledge base match
        kb_bonus = 1.0 if knowledge_data and knowledge_data.get('matched_failure') else 0.5

        # Factor 4: Data volume
        data_factor = min(len(top_cases) / 10, 1.0)

        raw = (0.35 * avg_semantic + 0.25 * consistency + 0.2 * kb_bonus + 0.2 * data_factor)
        return min(100, max(0, int(raw * 100)))

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def _build_explanation(self, top_cases, knowledge_data, equipment_type, query):
        """Build the explainability dict."""
        if not top_cases:
            return {'reasoning': 'Insufficient data for recommendation.'}

        top_case = top_cases[0]
        num_cases = len(top_cases)

        # Count defect type agreement
        defect_types = [c['defect_type'] for c in top_cases if c['defect_type'] not in ('Unknown', 'nan', '')]
        if defect_types:
            from collections import Counter
            dt_counter = Counter(defect_types)
            most_common_dt, most_common_count = dt_counter.most_common(1)[0]
        else:
            most_common_dt, most_common_count = 'Unknown', 0

        # Count disposition agreement
        dispositions = [c['disposition'] for c in top_cases if c['disposition'] not in ('Unknown', 'nan', '', '--')]
        if dispositions:
            from collections import Counter
            disp_counter = Counter(dispositions)
            most_common_disp, disp_count = disp_counter.most_common(1)[0]
        else:
            most_common_disp, disp_count = 'N/A', 0

        # Build reasoning text
        reasoning_parts = []
        if top_case['semantic_score'] > 0.7:
            reasoning_parts.append(f"High confidence: the top match has a {top_case['semantic_score']:.0%} semantic similarity.")
        elif top_case['semantic_score'] > 0.4:
            reasoning_parts.append(f"Moderate confidence: the top match has a {top_case['semantic_score']:.0%} semantic similarity.")
        else:
            reasoning_parts.append(f"Low match quality: best semantic similarity is {top_case['semantic_score']:.0%}.")

        if most_common_count >= 3:
            reasoning_parts.append(
                f"{most_common_count} of {num_cases} similar cases share the defect type '{most_common_dt}'."
            )

        if disp_count >= 2:
            reasoning_parts.append(
                f"{disp_count} cases had the same resolution approach: '{most_common_disp[:80]}'."
            )

        if knowledge_data and knowledge_data.get('matched_failure'):
            reasoning_parts.append(
                f"The knowledge base confirms this as a known failure mode: '{knowledge_data['matched_failure']}'."
            )

        return {
            'method': 'Hybrid Semantic + TF-IDF retrieval with knowledge augmentation',
            'similar_cases_used': num_cases,
            'top_case': {
                'complaint_number': top_case['complaint_number'],
                'similarity': round(top_case['semantic_score'], 3),
                'equipment': top_case['equipment']
            },
            'knowledge_source': (
                f"{equipment_type} → {knowledge_data['matched_failure']}"
                if knowledge_data and knowledge_data.get('matched_failure') else None
            ),
            'root_causes_considered': list(set(
                c['defect_type'] for c in top_cases
                if c['defect_type'] not in ('Unknown', 'nan', '')
            ))[:5],
            'confidence_factors': {
                'semantic_match_quality': round(top_case['semantic_score'], 3),
                'historical_resolution_consistency': round(disp_count / max(num_cases, 1), 3),
                'knowledge_base_match': bool(knowledge_data and knowledge_data.get('matched_failure')),
                'feedback_rating': round(top_case.get('feedback_weight', 0.5), 3)
            },
            'reasoning': ' '.join(reasoning_parts)
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _format_cases(self, cases):
        """Format top cases for the frontend."""
        formatted = []
        for c in cases:
            formatted.append({
                'complaint_number': c['complaint_number'],
                'similarity_score': round(c['final_score'], 3),
                'semantic_score': round(c['semantic_score'], 3),
                'equipment': c['equipment'],
                'defect_type': c['defect_type'],
                'problem_summary': c['problem_desc'][:150],
                'resolution': c['disposition'][:200] if c['disposition'] else '',
                'learning': c['learning'][:200] if c['learning'] else '',
                'severity': c['severity'],
                'resolution_days': c['resolution_days'],
                'product': c['product'],
                'vendor': c['vendor']
            })
        return formatted

    def _search_tfidf(self, query, top_k=20):
        """
        Wrapper to get raw (index, score) tuples from the existing
        TF-IDF engine without the full RCA summary overhead.
        """
        if self.ml_service.tfidf_vectorizer is None or self.ml_service.tfidf_matrix is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.ml_service.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.ml_service.tfidf_matrix).flatten()

        # Get top_k indices with highest similarity
        top_indices = sims.argsort()[-top_k:][::-1]
        results = [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0]
        return results

    def _empty_result(self):
        """Return an empty recommendation structure."""
        return {
            'recommendation': {
                'probable_root_causes': [],
                'diagnostic_checks': [],
                'recommended_actions': [],
                'preventive_measures': [],
                'safety_precautions': [],
                'confidence_score': 0
            },
            'similar_cases': [],
            'explanation': {'reasoning': 'No data available for recommendation.'},
            'recommendation_id': '',
            'equipment_type_detected': None
        }

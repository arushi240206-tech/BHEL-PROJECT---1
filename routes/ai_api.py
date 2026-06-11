from flask import Blueprint, request, jsonify, current_app

ai_api_bp = Blueprint('ai_api', __name__)


@ai_api_bp.route('/recommend', methods=['POST'])
def recommend():
    """
    Main AI recommendation endpoint.

    Accepts: { "query": "...", "equipment_type": "..." (optional) }
    Returns: Structured recommendation with explainability.
    """
    try:
        data = request.json or {}
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'status': 'error', 'error': 'Query is required'}), 400

        equipment_type = data.get('equipment_type', None)
        engine = current_app.config.get('RECOMMENDATION_ENGINE')

        if engine is None:
            return jsonify({'status': 'error', 'error': 'Recommendation engine not initialized'}), 503

        result = engine.get_recommendations(query, equipment_type)
        return jsonify({'status': 'success', **result})

    except Exception as e:
        print(f"[AI Recommend] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback on a recommendation.

    Accepts: { "query": "...", "recommendation_id": "...",
               "equipment_type": "...", "defect_type": "...",
               "rating": "helpful"|"not_helpful", "notes": "..." }
    """
    try:
        data = request.json or {}
        query = data.get('query', '')
        rec_id = data.get('recommendation_id', '')
        equipment_type = data.get('equipment_type', '')
        defect_type = data.get('defect_type', '')
        rating = data.get('rating', '')
        notes = data.get('notes', '')

        if rating not in ('helpful', 'not_helpful'):
            return jsonify({'status': 'error', 'error': 'Rating must be helpful or not_helpful'}), 400

        feedback_engine = current_app.config.get('FEEDBACK_ENGINE')
        if feedback_engine is None:
            return jsonify({'status': 'error', 'error': 'Feedback engine not initialized'}), 503

        feedback_engine.submit_feedback(query, rec_id, equipment_type, defect_type, rating, notes)
        return jsonify({'status': 'success', 'message': 'Feedback recorded'})

    except Exception as e:
        print(f"[AI Feedback] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/submit_case', methods=['POST'])
def submit_case():
    """
    Submit a resolved case to the pending approval queue.

    Accepts: { "problem_description": "...", "equipment_name": "...",
               "defect_type": "...", "severity": "...",
               "resolution": "...", "learning_derived": "..." }
    """
    try:
        data = request.json or {}
        feedback_engine = current_app.config.get('FEEDBACK_ENGINE')

        if feedback_engine is None:
            return jsonify({'status': 'error', 'error': 'Feedback engine not initialized'}), 503

        case_id = feedback_engine.submit_resolved_case(data)
        return jsonify({'status': 'success', 'case_id': case_id, 'message': 'Case submitted for review'})

    except Exception as e:
        print(f"[AI Submit Case] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/pending_cases', methods=['GET'])
def pending_cases():
    """Return all cases pending engineer approval."""
    try:
        feedback_engine = current_app.config.get('FEEDBACK_ENGINE')
        if feedback_engine is None:
            return jsonify({'status': 'error', 'error': 'Feedback engine not initialized'}), 503

        cases = feedback_engine.get_pending_cases()
        return jsonify({'status': 'success', 'cases': cases})

    except Exception as e:
        print(f"[AI Pending Cases] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/approve_case', methods=['POST'])
def approve_case():
    """
    Approve a pending case.

    Accepts: { "case_id": "...", "approved_by": "..." }
    """
    try:
        data = request.json or {}
        case_id = data.get('case_id', '')
        approved_by = data.get('approved_by', 'engineer')

        feedback_engine = current_app.config.get('FEEDBACK_ENGINE')
        if feedback_engine is None:
            return jsonify({'status': 'error', 'error': 'Feedback engine not initialized'}), 503

        success = feedback_engine.approve_case(case_id, approved_by)
        if success:
            return jsonify({'status': 'success', 'message': 'Case approved'})
        else:
            return jsonify({'status': 'error', 'error': 'Case not found or already processed'}), 404

    except Exception as e:
        print(f"[AI Approve Case] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/merge_cases', methods=['POST'])
def merge_cases():
    """Merge all approved cases into the training dataset and rebuild indexes."""
    try:
        feedback_engine = current_app.config.get('FEEDBACK_ENGINE')
        data_service = current_app.config.get('DATA_SERVICE')
        semantic_search = current_app.config.get('SEMANTIC_SEARCH')

        if feedback_engine is None:
            return jsonify({'status': 'error', 'error': 'Feedback engine not initialized'}), 503

        count = feedback_engine.merge_approved_cases(data_service, semantic_search)

        # Re-fit TF-IDF as well
        ml_service = current_app.config.get('ML_SERVICE')
        if ml_service and data_service and data_service.df is not None:
            ml_service.fit_tfidf(data_service.df)

        return jsonify({
            'status': 'success',
            'merged_count': count,
            'message': f'{count} cases merged into training data. Indexes rebuilt.'
        })

    except Exception as e:
        print(f"[AI Merge Cases] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/feedback_analytics', methods=['GET'])
def feedback_analytics():
    """Return aggregated feedback statistics for the AI Analytics dashboard."""
    try:
        feedback_engine = current_app.config.get('FEEDBACK_ENGINE')
        if feedback_engine is None:
            return jsonify({'status': 'error', 'error': 'Feedback engine not initialized'}), 503

        analytics = feedback_engine.get_feedback_analytics()
        return jsonify({'status': 'success', **analytics})

    except Exception as e:
        print(f"[AI Feedback Analytics] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/maintenance_risk', methods=['POST'])
def maintenance_risk():
    """
    Get predictive maintenance analysis for a specific equipment.

    Accepts: { "equipment_name": "..." }
    """
    try:
        data = request.json or {}
        equipment_name = data.get('equipment_name', '').strip()

        predictor = current_app.config.get('MAINTENANCE_PREDICTOR')
        if predictor is None:
            return jsonify({'status': 'error', 'error': 'Maintenance predictor not initialized'}), 503

        result = predictor.detect_failure_patterns(equipment_name)
        return jsonify({'status': 'success', **result})

    except Exception as e:
        print(f"[AI Maintenance Risk] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/equipment_health', methods=['GET'])
def equipment_health():
    """Return health scores for all monitored equipment."""
    try:
        predictor = current_app.config.get('MAINTENANCE_PREDICTOR')
        if predictor is None:
            return jsonify({'status': 'error', 'error': 'Maintenance predictor not initialized'}), 503

        health_data = predictor.compute_equipment_health()
        return jsonify({'status': 'success', 'equipment': health_data})

    except Exception as e:
        print(f"[AI Equipment Health] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@ai_api_bp.route('/knowledge_types', methods=['GET'])
def knowledge_types():
    """Return the list of equipment types from the knowledge base for UI dropdowns."""
    try:
        knowledge_engine = current_app.config.get('KNOWLEDGE_ENGINE')
        if knowledge_engine is None:
            return jsonify({'status': 'success', 'types': []})

        return jsonify({'status': 'success', 'types': knowledge_engine.get_equipment_types()})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

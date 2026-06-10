from flask import Blueprint, request, jsonify, current_app

api_bp = Blueprint('api', __name__)

@api_bp.route('/metadata', methods=['GET'])
def get_metadata():
    data_service = current_app.config['DATA_SERVICE']
    return jsonify({
        'status': 'success',
        'filters': data_service.get_metadata()
    })

@api_bp.route('/ml_metadata', methods=['GET'])
def get_ml_metadata():
    try:
        ml_service = current_app.config['ML_SERVICE']
        return jsonify({
            'status': 'success',
            'models': ml_service.get_ml_metadata()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    ml_service = current_app.config['ML_SERVICE']
    metrics = ml_service.get_metrics()
    if not metrics:
        return jsonify({'error': 'Metrics not found. Please run model training first.'}), 404
    return jsonify(metrics)

@api_bp.route('/trends', methods=['POST'])
def get_trends():
    try:
        data = request.json or {}
        data_service = current_app.config['DATA_SERVICE']
        return jsonify(data_service.get_trends(data))
    except Exception as e:
        print(f"Error serving trends: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    try:
        data = request.json or {}
        requested_targets = data.get('targets', [])
        inputs = data.get('features', {})
        
        ml_service = current_app.config['ML_SERVICE']
        results = ml_service.predict_multiple(requested_targets, inputs)
        
        return jsonify({
            'status': 'success',
            'predictions': results
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/nlp_search', methods=['POST'])
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
        
        data_service = current_app.config['DATA_SERVICE']
        ml_service = current_app.config['ML_SERVICE']
        
        f_df = data_service.filter_data(start_year, end_year, region, unit, project, product, status)
        
        result = ml_service.run_nlp_search(query, f_df, limit)
        return jsonify(result)
    except Exception as e:
        print(f"Error during NLP search: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/equipment_analysis', methods=['POST'])
def equipment_analysis():
    try:
        data = request.json or {}
        data_service = current_app.config['DATA_SERVICE']
        
        result = data_service.get_equipment_analysis(data)
        
        return jsonify({
            'status': 'success',
            'equipment_data': result
        })
    except Exception as e:
        print(f"Error in equipment_analysis: {e}")
        return jsonify({'error': str(e)}), 500

import os
from flask import Flask, render_template

from routes.api import api_bp
from services.data_service import DataService
from services.ml_service import MLService

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    CSV_PATH = "10yearsdata_cleaned.csv"
    METRICS_JSON_PATH = "model_evaluation_metrics.json"

    # Initialize Services
    data_service = DataService(CSV_PATH)
    try:
        data_service.load_data()
    except Exception as e:
        print(f"Warning: DataService initialization failed: {e}")

    ml_service = MLService(METRICS_JSON_PATH)
    try:
        ml_service.load_metrics()
        ml_service.load_models()
        if data_service.df is not None:
            ml_service.fit_tfidf(data_service.df)
    except Exception as e:
        print(f"Warning: MLService initialization failed: {e}")

    # Inject services into app config so blueprints can access them
    app.config['DATA_SERVICE'] = data_service
    app.config['ML_SERVICE'] = ml_service

    # Register Blueprint
    app.register_blueprint(api_bp, url_prefix='/api')

    @app.route('/')
    def home():
        return render_template('index.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)

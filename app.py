import os
os.environ['OMP_NUM_THREADS'] = '1'

from flask import Flask, render_template

from routes.api import api_bp
from routes.ai_api import ai_api_bp
from services.data_service import DataService
from services.ml_service import MLService
from services.semantic_search import SemanticSearchService
from services.knowledge_engine import KnowledgeEngine
from services.feedback_engine import FeedbackEngine
from services.recommendation_engine import RecommendationEngine
from services.maintenance_predictor import MaintenancePredictor

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    CSV_PATH = "10yearsdata_cleaned.csv"
    METRICS_JSON_PATH = "model_evaluation_metrics.json"

    # ---- 1. DataService (existing) ----
    data_service = DataService(CSV_PATH)
    try:
        data_service.load_data()
    except Exception as e:
        print(f"Warning: DataService initialization failed: {e}")

    # ---- 2. MLService (existing) ----
    ml_service = MLService(METRICS_JSON_PATH)
    try:
        ml_service.load_metrics()
        ml_service.load_models()
        if data_service.df is not None:
            ml_service.fit_tfidf(data_service.df)
    except Exception as e:
        print(f"Warning: MLService initialization failed: {e}")

    # ---- 3. SemanticSearchService (NEW) ----
    semantic_search = SemanticSearchService(embeddings_dir='embeddings')
    try:
        semantic_search.load_index()
    except Exception as e:
        print(f"Warning: SemanticSearch initialization failed (TF-IDF fallback active): {e}")

    # ---- 4. KnowledgeEngine (NEW) ----
    knowledge_engine = KnowledgeEngine(knowledge_dir='knowledge')
    try:
        knowledge_engine.load_knowledge()
    except Exception as e:
        print(f"Warning: KnowledgeEngine initialization failed: {e}")

    # ---- 5. FeedbackEngine (NEW) ----
    feedback_engine = FeedbackEngine(data_dir='data')

    # ---- 6. RecommendationEngine (NEW) ----
    recommendation_engine = RecommendationEngine(
        semantic_search=semantic_search,
        ml_service=ml_service,
        data_service=data_service,
        knowledge_engine=knowledge_engine,
        feedback_engine=feedback_engine
    )

    # ---- 7. MaintenancePredictor (NEW) ----
    maintenance_predictor = MaintenancePredictor(data_service=data_service)

    # Inject all services into app config
    app.config['DATA_SERVICE'] = data_service
    app.config['ML_SERVICE'] = ml_service
    app.config['SEMANTIC_SEARCH'] = semantic_search
    app.config['KNOWLEDGE_ENGINE'] = knowledge_engine
    app.config['FEEDBACK_ENGINE'] = feedback_engine
    app.config['RECOMMENDATION_ENGINE'] = recommendation_engine
    app.config['MAINTENANCE_PREDICTOR'] = maintenance_predictor

    # Register Blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(ai_api_bp, url_prefix='/api/ai')

    @app.route('/')
    def home():
        return render_template('index.html')

    return app

if __name__ == '__main__':
    import sys
    print("Starting Flask app...")
    sys.stdout.flush()
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=False)

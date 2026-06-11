# BHEL Predictive Analytics Dashboard

A machine learning-powered predictive analytics dashboard for BHEL (Bharat Heavy Electricals Limited), providing insights into 10 years of operational data, predicting SLA delays, analyzing root causes (RCA), and performing semantic searches across unstructured complaint texts using an advanced AI Engineering Assistant.

## Features

- **Dynamic Trends Dashboard**: Visualize KPIs, year-over-year complaint volumes, severity distributions, and cost analysis charts.
- **Machine Learning Inference**: Real-time predictions for severity, expected resolution days, and repetitive failure likelihood using pre-trained `joblib` models.
- **AI Engineering Assistant**: 
  - **Hybrid Semantic Search**: Uses `SentenceTransformers` (all-MiniLM-L6-v2) and FAISS for fast approximate nearest-neighbor retrieval, falling back to TF-IDF when necessary.
  - **Knowledge Integration**: Enriches search results with diagnostic procedures and preventive measures from domain-specific engineering knowledge bases (e.g., Turbine, Boiler, Pump).
  - **Explainability**: Provides human-readable reasoning and confidence scores for recommended actions.
  - **Continuous Feedback Loop**: Engineers can approve or rate recommendations, feeding data back into the system to incrementally improve AI accuracy.
- **Predictive Maintenance**: Automatically identifies high-risk equipment and extracts failure patterns to warn against imminent failures.

## Tech Stack

- **Backend**: Python 3, Flask, Pandas, Scikit-Learn
- **AI / NLP**: PyTorch, SentenceTransformers, FAISS
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Chart.js
- **Containerization**: Docker, Docker Compose

## Project Structure

```text
.
├── app.py                      # Main Flask application entry point
├── routes/
│   ├── api.py                  # Standard API endpoints
│   └── ai_api.py               # AI Assistant and Feedback endpoints
├── services/
│   ├── data_service.py         # Data loading and KPI aggregation
│   ├── ml_service.py           # Model inference and TF-IDF fallback
│   ├── semantic_search.py      # FAISS and SentenceTransformer indexing
│   ├── recommendation_engine.py# Hybrid ranking and explainability logic
│   ├── knowledge_engine.py     # Domain-specific failure mode parsing
│   ├── feedback_engine.py      # Case approval and feedback loop
│   └── maintenance_predictor.py# Equipment risk analysis
├── knowledge/                  # JSON knowledge base files for various equipment
├── static/
│   ├── css/style.css           # Extracted styles
│   ├── js/main.js              # Frontend logic
│   └── logo.png                # Assets
├── templates/
│   └── index.html              # Main dashboard view
├── tests/                      # Pytest automated tests
├── models_10x/                 # Serialized .joblib ML models
├── 10yearsdata_cleaned.csv     # The cleaned dataset (required)
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Local development stack setup
└── requirements.txt            # Python dependencies
```

## Prerequisites

- Python 3.9+ (if running locally without Docker)
- Docker & Docker Compose (if containerizing)
- The dataset `10yearsdata_cleaned.csv` must be present in the root directory.
- Pre-trained models must be present in `models_10x/` directory.

> **Note for macOS users**: The application automatically sets `OMP_NUM_THREADS=1` to prevent silent thread-collision crashes between PyTorch (SentenceTransformers) and Scikit-Learn/XGBoost (Joblib).

## Installation & Running Locally

### Option 1: Using Python Virtual Environment (Recommended for Dev)

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:5001`

### Option 2: Using Docker (Recommended for Prod/Testing)

1. **Build and run using Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   Open your browser and navigate to `http://localhost:5001`

## Running Tests

This project uses `pytest` for automated integration and unit testing.

1. Ensure dependencies are installed.
2. Run the test suite:
   ```bash
   pytest
   ```
   
To see verbose output and coverage:
```bash
pytest -v
```

## Data Pipeline Scripts

This repository also contains several utility scripts for preprocessing raw Excel data into the cleaned CSV used by the dashboard, and training the models. 
- `clean_dataset.py`: Primary data sanitization logic.
- `convert_and_clean.py`: Utility wrapper.
- `train_10_models.py` & `train_models.py`: Scripts used to train the `.joblib` artifacts found in `models_10x/`.

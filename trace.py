import app
import sys

print("Start trace")
sys.stdout.flush()

try:
    ds = app.DataService('10yearsdata_cleaned.csv')
    print("Calling load_data")
    sys.stdout.flush()
    ds.load_data()
    print("load_data done")
    sys.stdout.flush()

    ml = app.MLService('model_evaluation_metrics.json')
    print("Calling load_metrics")
    sys.stdout.flush()
    ml.load_metrics()
    print("load_metrics done")
    sys.stdout.flush()

    print("Calling load_models")
    sys.stdout.flush()
    ml.load_models()
    print("load_models done")
    sys.stdout.flush()

    print("Calling fit_tfidf")
    sys.stdout.flush()
    if ds.df is not None:
        ml.fit_tfidf(ds.df)
    print("fit_tfidf done")
    sys.stdout.flush()

    print("Calling sem")
    sys.stdout.flush()
    sem = app.SemanticSearchService('embeddings')
    sem.load_index()
    print("sem done")
    sys.stdout.flush()

    print("Calling ken")
    sys.stdout.flush()
    ken = app.KnowledgeEngine('knowledge')
    ken.load_knowledge()
    print("ken done")
    sys.stdout.flush()

except Exception as e:
    print(f"Exception: {e}")

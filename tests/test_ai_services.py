import pytest
import os
import pandas as pd
from unittest.mock import MagicMock

# Import services
from services.knowledge_engine import KnowledgeEngine
from services.feedback_engine import FeedbackEngine
from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def mock_data_service():
    service = MagicMock()
    service.df = pd.DataFrame({
        'Complaint Number': ['C001', 'C002', 'C003'],
        'Equipment Name': ['HP Turbine', 'Boiler Feed Pump', 'Generator'],
        'Defect Type': ['Mechanical', 'Electrical', 'Instrumentation'],
        'Complaint Year': [2022, 2023, 2024],
        'Complaint Date': pd.to_datetime(['2022-01-01', '2023-01-01', '2024-01-01']),
        'Severity Rating (Given by Unit)': [0.8, 0.4, 0.9],
        'Repetitive Issues Identified by Unit (Y/N)': ['N', 'Y', 'N'],
        'Days Taken for Disposition': [10, 5, 20],
        'Unit Disposition': ['Weld tube', 'Replaced bearings', 'Rewound stator']
    })
    service.df.index = [0, 1, 2]
    return service


@pytest.fixture
def mock_ml_service():
    service = MagicMock()
    service.tfidf_vectorizer = MagicMock()
    service.tfidf_matrix = MagicMock()
    return service


@pytest.fixture
def mock_semantic_search():
    service = MagicMock()
    service.ready = True
    # Return index 0 and 2 with scores
    service.search.return_value = [(0, 0.85), (2, 0.65)]
    return service


@pytest.fixture
def knowledge_engine(tmp_path):
    import json
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    
    # Create mock boiler KB
    boiler_data = {
        "equipment_type": "Boiler",
        "keywords": ["boiler", "tube", "furnace"],
        "common_failures": [
            {
                "failure": "Tube leak",
                "diagnostic_procedures": ["Check pressure"],
                "corrective_actions": ["Weld tube"]
            }
        ]
    }
    (kb_dir / "boiler.json").write_text(json.dumps(boiler_data))
    
    engine = KnowledgeEngine(knowledge_dir=str(kb_dir))
    engine.load_knowledge()
    return engine


@pytest.fixture
def feedback_engine(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return FeedbackEngine(data_dir=str(data_dir))


def test_knowledge_engine_matching(knowledge_engine):
    # Test keyword matching
    eq, score = knowledge_engine.match_equipment("The boiler tube is leaking in the furnace")
    assert eq == "Boiler"
    assert score == 3  # Matches "boiler", "tube", "furnace"

    # Test missing match
    eq, score = knowledge_engine.match_equipment("Generator is vibrating")
    assert eq is None


def test_knowledge_engine_retrieval(knowledge_engine):
    data = knowledge_engine.get_knowledge("Boiler", "leak")
    assert data is not None
    assert data['equipment_type'] == "Boiler"
    assert data['matched_failure'] == "Tube leak"
    assert "Check pressure" in data['diagnostic_procedures']


def test_feedback_engine_submission(feedback_engine):
    # Submit feedback
    feedback_engine.submit_feedback("Test query", "rec_123", "Turbine", "Mechanical", "helpful")
    
    # Check weight
    weight = feedback_engine.get_feedback_weight("Turbine", "Mechanical")
    assert weight == 1.0  # 1 out of 1 helpful
    
    # Submit negative feedback
    feedback_engine.submit_feedback("Test query 2", "rec_124", "Turbine", "Mechanical", "not_helpful")
    
    # Check updated weight
    weight = feedback_engine.get_feedback_weight("Turbine", "Mechanical")
    assert weight == 0.5  # 1 helpful, 1 not_helpful


def test_case_approval_workflow(feedback_engine, mock_data_service):
    # Submit case
    case_data = {
        "problem_description": "Vibration",
        "equipment_name": "Turbine",
        "defect_type": "Mechanical",
        "severity": 0.5,
        "resolution": "Balanced rotor"
    }
    case_id = feedback_engine.submit_resolved_case(case_data)
    
    # Check pending
    pending = feedback_engine.get_pending_cases()
    assert len(pending) == 1
    assert pending[0]['case_id'] == case_id
    
    # Approve case
    success = feedback_engine.approve_case(case_id, "admin")
    assert success is True
    
    # Check pending again
    pending = feedback_engine.get_pending_cases()
    assert len(pending) == 0
    
    # Merge case
    merged_count = feedback_engine.merge_approved_cases(mock_data_service, semantic_search_service=None)
    assert merged_count == 1
    
    # Mock data service DF should now have 4 rows instead of 3
    assert len(mock_data_service.df) == 4


def test_recommendation_engine(mock_semantic_search, mock_ml_service, mock_data_service, knowledge_engine, feedback_engine):
    # Setup Recommendation Engine
    engine = RecommendationEngine(
        semantic_search=mock_semantic_search,
        ml_service=mock_ml_service,
        data_service=mock_data_service,
        knowledge_engine=knowledge_engine,
        feedback_engine=feedback_engine
    )
    
    # Ensure TF-IDF fallback doesn't crash if empty
    engine._search_tfidf = MagicMock(return_value=[(1, 0.4)])
    
    # Get recommendations
    result = engine.get_recommendations("Boiler tube leak", equipment_type="Boiler")
    
    assert result is not None
    assert 'recommendation' in result
    assert 'similar_cases' in result
    assert 'explanation' in result
    
    # Semantic search returned index 0 and 2, TF-IDF returned index 1
    # Check that all three are combined in the similar cases
    assert len(result['similar_cases']) == 3
    
    # Knowledge base integration check
    rec = result['recommendation']
    assert "Check pressure" in rec['diagnostic_checks']
    assert "Weld tube" in rec['recommended_actions']
    
    # Check explanation
    exp = result['explanation']
    assert exp['similar_cases_used'] == 3
    assert "method" in exp
    assert "reasoning" in exp

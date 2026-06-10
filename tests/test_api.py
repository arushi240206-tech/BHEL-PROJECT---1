import json

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_metadata_api(client):
    response = client.get('/api/metadata')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'filters' in data

def test_trends_api(client):
    response = client.post('/api/trends', json={
        "start_year": 2020,
        "end_year": 2023,
        "region": "",
        "unit": ""
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'kpis' in data
    assert 'complaint_volume' in data

def test_nlp_search_api(client):
    response = client.post('/api/nlp_search', json={
        "query": "pump failure",
        "limit": 10
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'results' in data
    assert 'total_results' in data

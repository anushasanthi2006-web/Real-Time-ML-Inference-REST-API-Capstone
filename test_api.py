from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["model_loaded"] is True

def test_prediction():
    response = client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}
    assert abs(sum(body["probabilities"].values()) - 1) < 0.001

def test_missing_field():
    assert client.post("/predict", json={"sepal_length": 5.1}).status_code == 422

def test_invalid_range():
    response = client.post("/predict", json={
        "sepal_length": -1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 422

def test_non_numeric():
    response = client.post("/predict", json={
        "sepal_length": "abc", "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 422

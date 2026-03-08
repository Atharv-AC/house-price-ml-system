from house_price_prediction.api import app, get_model
from fastapi.testclient import TestClient
import pytest


# ---------- Fixture ----------

@pytest.fixture
def client():
    app.state.model_loaded = True
    return TestClient(app)


# ---------- Fake Model ----------

class FakeModel:
    def predict(self, data):
        return 123456


def fake_get_model():
    return FakeModel()


# ---------- Tests ----------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_with_fake_model(client):
    # Override dependency only for this test
    app.dependency_overrides[get_model] = fake_get_model

    response = client.post("/predict", json={
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 2,
        "parking": 3,
        "area": 4235,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "yes",
        "furnishingstatus": "no"
    })

    assert response.status_code == 200
    assert response.json()["Prediction"] == 123456

    # Clean override
    app.dependency_overrides.clear()


def test_predict_when_model_not_loaded(client):
    # Ensure no override
    app.dependency_overrides.clear()

    # Simulate model not loaded
    app.state.model_loaded = False

    response = client.post("/predict", json={
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 2,
        "parking": 3,
        "area": 4235,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "yes",
        "furnishingstatus": "no"
    })

    assert response.status_code == 503
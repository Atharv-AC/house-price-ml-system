from house_price_prediction.api import app, get_model
from fastapi.testclient import TestClient
import pytest


# ---------- Fixture ----------

@pytest.fixture
def client():
    # Mark the app as ready so most tests exercise the happy-path API behavior.
    app.state.model_loaded = True
    return TestClient(app)


# ---------- Fake Model ----------

class FakeModel:
    def predict(self, data):
        # Return a fixed value so the test can assert a deterministic prediction.
        return 123456


def fake_get_model():
    # Provide the fake model through FastAPI's dependency override system.
    return FakeModel()


# ---------- Tests ----------

def test_home_page(client):
    # Verify the root endpoint is reachable.
    response = client.get("/")
    assert response.status_code == 200

def test_health(client):
    # Confirm the health endpoint responds and exposes model status details.
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_with_fake_model(client):
    # Override dependency only for this test
    app.dependency_overrides[get_model] = fake_get_model

    # Send a representative payload and validate the mocked prediction response.
    response = client.post("/predict-house", json={
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

    # Clean override so later tests use their own setup.
    app.dependency_overrides.clear()


def test_predict_when_model_not_loaded(client):
    # Ensure no override
    app.dependency_overrides.clear()

    # Simulate model not loaded
    app.state.model_loaded = False

    # The same payload should be rejected when the application reports no model.
    response = client.post("/predict-house", json={
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

def test_model_info_not_found(client, monkeypatch):

    def mock_open(*args, **kwargs):
        # Force the file read to fail so the error handling path is exercised.
        raise Exception("file missing")

    # Replace the built-in file opener only within this test.
    monkeypatch.setattr("builtins.open", mock_open)

    response = client.get("/model-info")

    # The endpoint should surface the failure as an internal server error.
    assert response.status_code == 500

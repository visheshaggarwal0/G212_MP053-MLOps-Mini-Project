"""API tests for the MP053 FastAPI prediction service."""

from starlette.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_REQUEST = {
    "students_registered": 700,
    "attendance_pct": 85.5,
    "menu_rating": 4.2,
    "temp_c": 28.5,
    "cooked_kg": 325.0,
    "plate_size_cm": 25.0,
    "meal": "lunch",
    "hostel": "Boys Hostel 1",
}


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_numeric_prediction():
    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))


def test_predict_rejects_more_than_2500_registered_students():
    invalid_request = {**VALID_REQUEST, "students_registered": 2501}

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422


def test_predict_rejects_temperature_below_4_degrees():
    invalid_request = {**VALID_REQUEST, "temp_c": 3.9}

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422


def test_predict_rejects_unexpected_request_fields():
    invalid_request = {**VALID_REQUEST, "record_id": "R50001"}

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422

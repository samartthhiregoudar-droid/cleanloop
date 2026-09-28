"""Unit tests for FastAPI backend routes and database storage."""
from fastapi.testclient import TestClient

from backend.database import init_db, save_incident
from backend.main import app

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "analytics" in data


def test_incidents_api():
    init_db()
    test_record = {
        "incident_id": "TEST_INC_123",
        "created_at": 1000.0,
        "object_id": 1,
        "person_name": "Person_1",
        "status": "PENDING_REVIEW",
        "state": "CONFIRMED",
        "confidence": 0.9,
        "label": "bottle",
        "bbox": [10, 20, 30, 40],
        "explanation": "Test incident explanation"
    }
    save_incident(test_record)

    response = client.get("/api/incidents")
    assert response.status_code == 200
    incidents = response.json()
    assert any(i["incident_id"] == "TEST_INC_123" for i in incidents)

    # Test review endpoint
    review_resp = client.post("/api/incidents/TEST_INC_123/review", json={
        "decision": "CONFIRMED_LITTER",
        "notes": "Verified by reviewer"
    })
    assert review_resp.status_code == 200
    updated = review_resp.json()
    assert updated["review_decision"] == "CONFIRMED_LITTER"

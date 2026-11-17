from fastapi import status
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test the root welcome/discovery endpoint."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "operational"
    assert "service" in data
    assert "version" in data


def test_get_health_root(client: TestClient):
    """Test the direct GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "APM-Nowcast API"
    assert "version" in data
    assert "timestamp" in data
    assert "environment" in data


def test_get_health_v1_api(client: TestClient):
    """Test the versioned GET /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "APM-Nowcast API"

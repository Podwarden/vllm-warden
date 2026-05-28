from fastapi.testclient import TestClient


def test_health_endpoint_returns_200(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

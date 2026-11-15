"""CORS middleware coverage for Pages + localhost origins."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_cors_preflight_allows_github_pages(client: TestClient) -> None:
    response = client.options(
        "/optimize-price",
        headers={
            "Origin": "https://arthursetiawan.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in {200, 204}
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://arthursetiawan.github.io"
    )


def test_cors_preflight_allows_localhost(client: TestClient) -> None:
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in {200, 204}
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_headers_on_actual_predict(client: TestClient, valid_quote: dict) -> None:
    response = client.post(
        "/predict",
        json=valid_quote,
        headers={"Origin": "https://arthursetiawan.github.io"},
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://arthursetiawan.github.io"
    )


def test_cors_rejects_unknown_origin(client: TestClient, valid_quote: dict) -> None:
    response = client.post(
        "/predict",
        json=valid_quote,
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") != "https://evil.example"

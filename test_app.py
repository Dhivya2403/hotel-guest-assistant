"""
Automated tests for the deterministic parts of the backend (availability logic
and input validation). These do NOT call the Gemini API, so they run without
needing GEMINI_API_KEY or network access — exactly the parts of the app that
should be reliable and testable without an LLM in the loop.

Run with: pytest test_app.py -v
"""
import pytest
from app import app, check_availability, _date_range


# ---------------------------------------------------------------------------
# Tests for deterministic business logic
# ---------------------------------------------------------------------------

def test_date_range_single_night():
    assert _date_range("2026-09-20", "2026-09-21") == ["2026-09-20"]


def test_date_range_multiple_nights():
    result = _date_range("2026-09-20", "2026-09-23")
    assert result == ["2026-09-20", "2026-09-21", "2026-09-22"]


def test_check_availability_finds_free_room():
    result = check_availability("2026-10-01", "2026-10-02", 2)
    assert "available_rooms" in result
    assert len(result["available_rooms"]) > 0


def test_check_availability_excludes_booked_room():
    # Standard room is booked for 2026-09-20 in the seed data
    result = check_availability("2026-09-20", "2026-09-21", 2)
    names = [r["name"] for r in result["available_rooms"]]
    assert "Standard Room" not in names


def test_check_availability_filters_by_capacity():
    # Only Family Suite fits 4 guests
    result = check_availability("2026-10-05", "2026-10-06", 4)
    names = [r["name"] for r in result["available_rooms"]]
    assert names == ["Family Suite"]


def test_check_availability_no_rooms_for_too_many_guests():
    result = check_availability("2026-10-05", "2026-10-06", 10)
    assert result["available_rooms"] == []


def test_check_availability_invalid_date_format():
    result = check_availability("20-09-2026", "2026-09-21", 2)
    assert "error" in result


def test_check_availability_checkout_before_checkin():
    result = check_availability("2026-09-22", "2026-09-20", 2)
    assert "error" in result


# ---------------------------------------------------------------------------
# Tests for API endpoint validation (using Flask test client)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_chat_rejects_empty_message(client):
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 400


def test_chat_rejects_missing_body(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 400


def test_availability_endpoint_requires_all_fields(client):
    resp = client.post("/availability", json={"check_in": "2026-10-01"})
    assert resp.status_code == 400


def test_availability_endpoint_rejects_invalid_adults(client):
    resp = client.post("/availability", json={
        "check_in": "2026-10-01", "check_out": "2026-10-02", "adults": "abc"
    })
    assert resp.status_code == 400


def test_availability_endpoint_happy_path(client):
    resp = client.post("/availability", json={
        "check_in": "2026-10-01", "check_out": "2026-10-02", "adults": 2
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "available_rooms" in data


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

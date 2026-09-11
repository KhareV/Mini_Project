import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_get_system_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/system/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ecg_model_status_identifies_release_boundary():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/model/status")
    assert response.status_code == 200
    assert response.json()["ecg"]["deployment_eligible"] is True
    assert response.json()["multimodal"]["deployment_eligible"] is False


@pytest.mark.asyncio
async def test_ecg_quality_gate_rejects_unreliable_signal():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/model/ecg/infer", json={"ecg": [0.0] * 2500, "quality": 0.2})
    assert response.status_code == 200
    assert response.json()["prediction"] == "UNRELIABLE_SIGNAL"

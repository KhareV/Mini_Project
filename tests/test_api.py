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
    assert response.json()["ecg"]["clinical_use_eligible"] is False
    assert response.json()["vitals"]["software_release_eligible"] is True
    assert response.json()["vitals"]["clinical_use_eligible"] is False
    assert response.json()["multimodal"]["deployment_eligible"] is False


@pytest.mark.asyncio
async def test_ecg_quality_gate_rejects_unreliable_signal():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/model/ecg/infer", json={"ecg": [0.0] * 2500, "quality": 0.2})
    assert response.status_code == 200
    assert response.json()["prediction"] == "UNRELIABLE_SIGNAL"


@pytest.mark.asyncio
async def test_vitals_inference_clean_synthetic_pulse():
    import math
    ppg = [1.0 - math.sin(2 * math.pi * 1.2 * i / 125.0) for i in range(1250)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/model/vitals/infer", json={"ppg": ppg, "spo2": 98.0})
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "NO_VITAL_ALERT"
    assert abs(result["pulse"]["pulse_bpm"] - 72.0) < 2.0
    assert result["clinical_use_eligible"] is False


@pytest.mark.asyncio
async def test_vitals_inference_reports_hypoxemia_threshold():
    import math
    ppg = [1.0 - math.sin(2 * math.pi * 1.2 * i / 125.0) for i in range(1250)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/model/vitals/infer", json={"ppg": ppg, "spo2": 92.0})
    assert response.status_code == 200
    assert response.json()["prediction"] == "POTENTIALLY_ABNORMAL_VITALS"
    assert "SPO2_BELOW_95_PERCENT" in response.json()["alerts"]


@pytest.mark.asyncio
async def test_system_inference_rejects_when_all_components_unreliable():
    payload = {
        "ecg": [0.0] * 2500,
        "ppg": [0.0] * 1250,
        "spo2": None,
        "ecg_quality": 0.1,
        "ppg_quality": 0.1,
        "spo2_quality": 0.1,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/model/system/infer", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "UNRELIABLE_SIGNAL"
    assert response.json()["clinical_use_eligible"] is False

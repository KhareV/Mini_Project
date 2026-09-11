from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.database import get_db
from backend.db.models import ModelPrediction
from backend.services.multimodal_inference import multimodal_inference_service
from backend.services.streaming_inference import streaming_inference_service
from backend.services.ecg_inference import ecg_inference_service

router = APIRouter(prefix="/model", tags=["multimodal-model"])


class MultimodalInferenceRequest(BaseModel):
    ecg: List[float] = Field(..., min_length=2500, max_length=2500)
    ppg: List[float] = Field(..., min_length=1250, max_length=1250)
    hr: float
    spo2: float
    presence_mask: List[float] = Field(default=[1, 1, 1, 1], min_length=4, max_length=4)
    quality_scores: List[float] = Field(default=[1, 1, 1, 1], min_length=4, max_length=4)


class StreamInferenceRequest(BaseModel):
    windows: List[MultimodalInferenceRequest] = Field(..., min_length=1, max_length=20)


class ECGInferenceRequest(BaseModel):
    ecg: List[float] = Field(..., min_length=2500, max_length=2500)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)


@router.get("/status")
def status():
    ecg = ecg_inference_service.status()
    multimodal = multimodal_inference_service.status()
    return {"ready": ecg["ready"] and multimodal["ready"], "model_version": "MODEL_V1 + multimodal research adapter", "ecg": ecg, "multimodal": multimodal}


@router.post("/ecg/infer")
def ecg_infer(request: ECGInferenceRequest):
    try:
        return ecg_inference_service.predict(**request.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/infer")
def infer(request: MultimodalInferenceRequest):
    try:
        return multimodal_inference_service.predict(**request.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/stream/{session_id}")
async def stream_infer(session_id: str, request: StreamInferenceRequest, db: AsyncSession = Depends(get_db)):
    """Process ordered canonical windows and persist each auditable decision."""
    try:
        response = streaming_inference_service.ingest(session_id, [item.model_dump() for item in request.windows])
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for decision in response["decisions"]:
        db.add(ModelPrediction(session_id=session_id, prediction=decision["prediction"], confidence=decision.get("model_confidence"), event_state=decision["event_state"], payload=decision))
    await db.commit()
    return response


@router.post("/stream/{session_id}/reset")
def reset_stream(session_id: str):
    streaming_inference_service.reset(session_id)
    return {"session_id": session_id, "event_state": "CLEAR"}

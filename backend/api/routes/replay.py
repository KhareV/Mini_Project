"""Versioned ECG replay endpoint backed by the centralized MODEL_V1 service."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.ml.ecg_replay_service import ECGReplayService

router = APIRouter(prefix="/replay", tags=["replay"])


class ECGReplayRequest(BaseModel):
    samples: list[float] = Field(..., min_length=1)
    sampling_rate: int = Field(500, ge=1, le=2000)


@lru_cache(maxsize=1)
def _service() -> ECGReplayService:
    checkpoint = os.getenv("NHM_ECG_MODEL_CHECKPOINT")
    normalization = os.getenv("NHM_ECG_NORMALIZATION_STATS")
    if not checkpoint or not normalization:
        raise RuntimeError(
            "MODEL_V1 artifacts are not configured; set "
            "NHM_ECG_MODEL_CHECKPOINT and NHM_ECG_NORMALIZATION_STATS"
        )
    return ECGReplayService.from_artifacts(checkpoint, normalization)


@router.post("/ecg")
def replay_ecg(request: ECGReplayRequest):
    try:
        return _service().predict_window(request.samples, request.sampling_rate)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

"""Centralized late-fusion ECG/PPG/HR/SpO2 model contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultimodalFusionConfig:
    signal_length: int = 2500
    scalar_features: int = 2  # HR, SpO2
    mask_features: int = 4    # ECG, PPG, HR, SpO2 presence/quality
    hidden: int = 64
    n_classes: int = 2
    model_version: str = "MULTIMODAL_FUSION_V1"


class _SignalEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(32, hidden)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.projection(self.net(signal).squeeze(-1))


class MultimodalFusionModel(nn.Module):
    """Late fusion with explicit presence/quality mask, never silent imputation."""

    def __init__(self, config: MultimodalFusionConfig | None = None):
        super().__init__()
        self.config = config or MultimodalFusionConfig()
        self.ecg_encoder = _SignalEncoder(self.config.hidden)
        self.ppg_encoder = _SignalEncoder(self.config.hidden)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.config.scalar_features, self.config.hidden), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden * 3 + self.config.mask_features, self.config.hidden),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(self.config.hidden, self.config.n_classes)
        )

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor,
                scalars: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        if presence_mask.shape[-1] != self.config.mask_features:
            raise ValueError("presence_mask has the wrong modality order/width")
        ecg_embedding = self.ecg_encoder(ecg) * presence_mask[:, 0:1]
        ppg_embedding = self.ppg_encoder(ppg) * presence_mask[:, 1:2]
        scalar_embedding = self.scalar_encoder(scalars)
        scalar_embedding = scalar_embedding * presence_mask[:, 2:3].clamp(0, 1)
        fused = torch.cat([ecg_embedding, ppg_embedding, scalar_embedding,
                           presence_mask], dim=1)
        return self.classifier(fused)

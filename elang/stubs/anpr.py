"""ANPR (Automatic Number Plate Recognition) — Phase 2 stub.

Target implementation (per DISHUB_Case_Analysis.md):
    - PaddleOCR for Indonesian plate characters (superior for non-Latin/mixed)
    - Two-stage: plate localisation -> OCR
    - CLAHE + denoising preprocessing for night/rain (adverse conditions)
    - Confidence threshold 0.8 to avoid false positives
    - Honest reporting: 85% accuracy ideal / 65% adverse

Activation: install paddleocr + paddlepaddle from requirements.txt
(currently commented), then replace the body of read_plate().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlateRead:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


def read_plate(plate_crop: np.ndarray) -> PlateRead | None:
    """Read characters from a cropped plate region.

    Args:
        plate_crop: BGR image of the plate region (after detection).

    Returns:
        PlateRead with normalised Indonesian plate format, or None
        if confidence is below threshold.
    """
    raise NotImplementedError("Phase 2: install paddleocr and implement.")


def preprocess_adverse(image: np.ndarray) -> np.ndarray:
    """CLAHE + denoising for night/rain footage."""
    raise NotImplementedError("Phase 2: implement CLAHE preprocessing.")

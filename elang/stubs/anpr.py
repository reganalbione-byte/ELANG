"""ANPR (Automatic Number Plate Recognition).

    - PaddleOCR (multi-language; Latin chars cover Indonesian plates)
    - Single-pass detect+recognise on the vehicle crop, then filter
      candidates against the Indonesian plate regex.
    - CLAHE + bilateral denoising preprocessing for adverse conditions
      (night, rain, low contrast).
    - Confidence threshold 0.8 to avoid false positives.
    - Expect 85% accuracy ideal / 65% adverse on real Jakarta footage —
      report honestly.

PaddleOCR is imported lazily so the module is inspectable without
the dep installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlateRead:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


PLATE_REGEX = re.compile(r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$")
PLATE_REGEX_RELAXED = re.compile(r"^[A-Z0-9]{4,9}$")
_DEFAULT_MIN_CONF = 0.8

_ocr_singleton = None


def _get_ocr():
    global _ocr_singleton
    if _ocr_singleton is not None:
        return _ocr_singleton
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "paddleocr is required for ANPR; "
            "uncomment paddleocr + paddlepaddle in requirements.txt and pip install."
        ) from e
    _ocr_singleton = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_singleton


def preprocess_adverse(image: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral filter — boosts contrast and denoises night/rain crops."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        boosted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        boosted = clahe.apply(image)
    return cv2.bilateralFilter(boosted, d=7, sigmaColor=50, sigmaSpace=50)


def _normalise(raw: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()


def _is_plausible_plate(text: str) -> bool:
    if not text:
        return False
    if PLATE_REGEX.match(text):
        return True
    compact = text.replace(" ", "")
    return bool(PLATE_REGEX_RELAXED.match(compact))


def read_plate(
    plate_crop: np.ndarray,
    min_confidence: float = _DEFAULT_MIN_CONF,
    enhance: bool = True,
) -> PlateRead | None:
    """Read characters from a vehicle / plate crop.

    Args:
        plate_crop: BGR image — either a tight plate ROI or a vehicle crop.
            When given a vehicle crop, PaddleOCR's built-in detector localises
            the plate; otherwise it just recognises.
        min_confidence: drop reads below this score.
        enhance: apply CLAHE + denoise before OCR (recommended for adverse).

    Returns:
        PlateRead with the highest-confidence plausible Indonesian plate,
        or None if nothing clears the threshold / regex.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    ocr = _get_ocr()
    image = preprocess_adverse(plate_crop) if enhance else plate_crop
    result = ocr.ocr(image, cls=True)

    if not result or not result[0]:
        return None

    best: PlateRead | None = None
    for line in result[0]:
        box, (raw_text, conf) = line
        text = _normalise(raw_text)
        if conf < min_confidence or not _is_plausible_plate(text):
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        if best is None or conf > best.confidence:
            best = PlateRead(text=text, confidence=float(conf), bbox=bbox)

    return best

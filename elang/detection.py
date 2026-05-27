"""YOLOv8 vehicle detection wrapper.

Lazy-loads the model on first call so import is cheap. Uses the
'n' (nano) variant by default for CPU-friendly inference; swap to
'm' or 'l' (per DISHUB_Case_Analysis.md) when GPU is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from .classes import VEHICLE_CLASS_IDS, coco_to_local


@dataclass
class Detection:
    label: str          # local label (motor, mobil, ...)
    confidence: float
    bbox: tuple[int, int, int, int]   # xyxy
    coco_id: int


@lru_cache(maxsize=1)
def _load_model(weights: str = "yolov8n.pt"):
    from ultralytics import YOLO
    return YOLO(weights)


def detect_vehicles(
    image: np.ndarray,
    conf: float = 0.4,
    weights: str = "yolov8n.pt",
) -> list[Detection]:
    """Run detection on a single image (H, W, 3 BGR or RGB)."""
    model = _load_model(weights)
    results = model.predict(image, conf=conf, verbose=False)
    if not results:
        return []

    out: list[Detection] = []
    boxes = results[0].boxes
    if boxes is None:
        return out

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
        if cid not in VEHICLE_CLASS_IDS:
            continue
        local = coco_to_local(cid)
        if local is None:
            continue
        out.append(Detection(
            label=local,
            confidence=float(c),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            coco_id=int(cid),
        ))
    return out


def detect_batch(frames: Iterable[np.ndarray], **kwargs) -> list[list[Detection]]:
    return [detect_vehicles(f, **kwargs) for f in frames]

"""Aggregate stats from detection results."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from .detection import Detection


def count_by_class(detections: Iterable[Detection]) -> dict[str, int]:
    return dict(Counter(d.label for d in detections))


def to_dataframe(detections: Iterable[Detection], frame_idx: int = 0) -> pd.DataFrame:
    rows = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        rows.append({
            "frame": frame_idx,
            "label": d.label,
            "confidence": round(d.confidence, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area": (x2 - x1) * (y2 - y1),
        })
    return pd.DataFrame(rows)


def summarise(detections: Iterable[Detection]) -> dict:
    dets = list(detections)
    counts = count_by_class(dets)
    return {
        "total": len(dets),
        "by_class": counts,
        "avg_confidence": round(
            sum(d.confidence for d in dets) / len(dets), 3
        ) if dets else 0.0,
    }

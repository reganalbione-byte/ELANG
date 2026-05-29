"""E-TLE (Electronic Traffic Law Enforcement) export adapter — mock.

Bridges ELANG violation detections to the JSON / CSV envelope that
POLRI / DISHUB expect for downstream ticketing. The real E-TLE API is
not publicly exposed, so this module produces format-compatible files
on disk instead of hitting a live endpoint.

Pure stdlib — no extra deps. Mock generator included for UI demos and
end-to-end format smoke-tests.
"""

from __future__ import annotations

import csv
import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

VEHICLE_CLASSES = ("motor", "mobil", "bus", "truk")
VIOLATION_TYPES = ("parkir_liar", "jalur_busway", "berhenti_sembarangan")
VALID_STATUS = ("pending", "submitted", "processed")
EXPORT_VERSION = "1.0"
EXPORT_SOURCE = "ELANG"

# E-TLE camera anchor points modelled on well-known Jakarta corridors.
# Mock generator jitters a few metres around these to look plausible.
_CAMERA_ANCHORS = [
    ("ETLE-JKT-001", -6.2088, 106.8456, "Bundaran HI"),
    ("ETLE-JKT-002", -6.2297, 106.8295, "Senayan"),
    ("ETLE-JKT-003", -6.1750, 106.8270, "Harmoni"),
    ("ETLE-JKT-004", -6.2370, 106.8307, "Semanggi"),
    ("ETLE-JKT-005", -6.2102, 106.8470, "Sudirman"),
    ("ETLE-JKT-006", -6.1944, 106.8229, "Tomang"),
    ("ETLE-JKT-007", -6.2614, 106.8101, "Permata Hijau"),
    ("ETLE-JKT-008", -6.1875, 106.8403, "Sarinah"),
]


@dataclass
class ViolationRecord:
    violation_id: str
    timestamp: str           # ISO 8601
    plate_number: str
    vehicle_class: str       # motor / mobil / bus / truk
    violation_type: str      # parkir_liar / jalur_busway / berhenti_sembarangan
    location_lat: float
    location_lon: float
    camera_id: str
    duration_seconds: int
    confidence_score: float
    evidence_frame_path: str
    status: str = "pending"  # pending / submitted / processed


CSV_FIELDS = [
    "violation_id", "timestamp", "plate_number", "vehicle_class",
    "violation_type", "location_lat", "location_lon", "camera_id",
    "duration_seconds", "confidence_score", "evidence_frame_path", "status",
]


def _random_plate(rng: random.Random) -> str:
    # B = Jakarta TNKB area code. Skip I and O to match real plate alphabet.
    suffix = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=3))
    return f"B {rng.randint(1000, 9999)} {suffix}"


def _random_timestamp(rng: random.Random, within_days: int = 7) -> str:
    delta = timedelta(seconds=rng.randint(0, within_days * 86400))
    return (datetime.now(timezone.utc) - delta).isoformat()


def generate_mock_violations(n: int = 10, seed: int | None = None) -> list[ViolationRecord]:
    """Realistic-looking dummy violations for UI demo + format smoke tests."""
    rng = random.Random(seed)
    out: list[ViolationRecord] = []
    for _ in range(n):
        cam_id, base_lat, base_lon, _ = rng.choice(_CAMERA_ANCHORS)
        vtype = rng.choice(VIOLATION_TYPES)
        # Duration is bounded by violation type — busway pelanggaran is brief,
        # parkir_liar lingers, berhenti_sembarangan is in between.
        if vtype == "jalur_busway":
            dur = rng.randint(3, 30)
        elif vtype == "berhenti_sembarangan":
            dur = rng.randint(30, 180)
        else:
            dur = rng.randint(60, 1800)
        vid = str(uuid.uuid4())
        out.append(ViolationRecord(
            violation_id=vid,
            timestamp=_random_timestamp(rng),
            plate_number=_random_plate(rng),
            vehicle_class=rng.choice(VEHICLE_CLASSES),
            violation_type=vtype,
            location_lat=round(base_lat + rng.uniform(-0.002, 0.002), 6),
            location_lon=round(base_lon + rng.uniform(-0.002, 0.002), 6),
            camera_id=cam_id,
            duration_seconds=dur,
            confidence_score=round(rng.uniform(0.72, 0.98), 3),
            evidence_frame_path=f"evidence/{cam_id}/{vid}.jpg",
            status="pending",
        ))
    return out


def _envelope(records: list[ViolationRecord]) -> dict:
    return {
        "version": EXPORT_VERSION,
        "source": EXPORT_SOURCE,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "violations": [asdict(r) for r in records],
    }


def export_to_etle_json(records: list[ViolationRecord], output_path: str) -> str:
    """Write the E-TLE envelope to disk as UTF-8 JSON. Returns the written path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_envelope(records), f, ensure_ascii=False, indent=2)
    return str(path)


def export_to_etle_csv(records: list[ViolationRecord], output_path: str) -> str:
    """Write a flat CSV (one row per violation) using CSV_FIELDS order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    return str(path)

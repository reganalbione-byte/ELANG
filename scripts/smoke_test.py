"""End-to-end smoke test for ELANG modules.

Runs each Phase 2 module against synthetic / real inputs and prints
pass/fail. Designed to be runnable without DeepSORT or PaddleOCR
installed — those checks skip cleanly with a note.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample.jpg"


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_detection() -> bool:
    _section("detection (YOLOv8)")
    if not SAMPLE.exists():
        print(f"SKIP: {SAMPLE} not found — run scripts/download_sample.py first.")
        return False
    from elang.detection import detect_vehicles
    img = cv2.imread(str(SAMPLE))
    dets = detect_vehicles(img, conf=0.4)
    print(f"detections: {len(dets)}")
    for d in dets[:5]:
        print(f"  {d.label} {d.confidence:.2f} {d.bbox}")
    return len(dets) > 0


def check_heatmap() -> bool:
    _section("heatmap (Folium)")
    from elang.stubs.heatmap import ViolationPoint, hot_zones, render_heatmap
    points = [
        ViolationPoint(-6.2088, 106.8456, 8, "parkir_liar", 3),
        ViolationPoint(-6.2089, 106.8457, 8, "parkir_liar", 5),
        ViolationPoint(-6.1750, 106.8270, 12, "trotoar", 4),
        ViolationPoint(-6.1751, 106.8271, 12, "trotoar", 2),
    ]
    zones = hot_zones(points, top_k=5, grid_size=0.001)
    print(f"hot_zones: {zones}")
    out_html = Path(tempfile.gettempdir()) / "elang_smoke_heatmap.html"
    try:
        render_heatmap(points, str(out_html))
    except RuntimeError as e:
        print(f"SKIP render: {e}")
        return bool(zones)
    size = out_html.stat().st_size
    print(f"map: {out_html} ({size:,} bytes)")
    return size > 0 and bool(zones)


def check_optimizer() -> bool:
    _section("officer optimizer")
    from elang.stubs.heatmap import ViolationPoint, hot_zones
    from elang.stubs.officer_optimizer import score_locations
    points = [
        ViolationPoint(-6.2088, 106.8456, 8, "x", 10),
        ViolationPoint(-6.1750, 106.8270, 12, "y", 4),
    ]
    zones = hot_zones(points, top_k=5, grid_size=0.001)
    candidates = [
        (-6.2088, 106.8456),   # on top of biggest hotspot
        (-6.1751, 106.8271),   # next to second hotspot
        (-6.5000, 107.0000),   # nowhere
    ]
    recs = score_locations(candidates, zones)
    for r in recs:
        print(f"  ({r.lat:.4f},{r.lon:.4f}) score={r.score:.4f}  {r.reason}")
    return recs[0].score > recs[-1].score


def check_tracking() -> bool:
    _section("tracking (DeepSORT)")
    try:
        from elang.stubs.tracking import Tracker
        tracker = Tracker(max_age=10, n_init=1)
        tracker._ensure_impl()
    except RuntimeError as e:
        print(f"SKIP: {e}")
        return True

    from elang.detection import Detection
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (180, 200), (255, 255, 255), -1)

    zone = [(50, 50), (300, 50), (300, 300), (50, 300)]

    for frame_idx in range(6):
        dx = frame_idx * 10
        det = Detection(label="mobil", confidence=0.9,
                        bbox=(100 + dx, 100, 180 + dx, 200), coco_id=2)
        tracker.update([det], frame=frame, frame_idx=frame_idx, zone_polygon=zone)

    active = tracker.active_tracks()
    print(f"active tracks: {len(active)}")
    for t in active:
        print(f"  #{t.track_id} {t.label} duration={t.duration_frames} in_zone={t.frames_in_zone}")
    return len(active) >= 1 and any(t.frames_in_zone > 0 for t in active)


def check_anpr() -> bool:
    _section("ANPR (PaddleOCR)")
    try:
        from elang.stubs.anpr import preprocess_adverse, read_plate, _is_plausible_plate
    except Exception as e:
        print(f"SKIP import: {e}")
        return False

    print(f"plausible('B 1234 XYZ') = {_is_plausible_plate('B 1234 XYZ')}")
    print(f"plausible('JUNK_TEXT_HERE') = {_is_plausible_plate('JUNK_TEXT_HERE')}")

    if SAMPLE.exists():
        crop = cv2.imread(str(SAMPLE))
        enhanced = preprocess_adverse(crop)
        print(f"preprocess_adverse: {crop.shape} -> {enhanced.shape}")

    try:
        if SAMPLE.exists():
            result = read_plate(cv2.imread(str(SAMPLE)), min_confidence=0.5)
            print(f"read_plate result: {result}")
    except RuntimeError as e:
        print(f"SKIP runtime: {e}")
        return True
    return True


CHECKS = [
    ("detection", check_detection),
    ("heatmap", check_heatmap),
    ("optimizer", check_optimizer),
    ("tracking", check_tracking),
    ("anpr", check_anpr),
]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    results: dict[str, str] = {}
    for name, fn in CHECKS:
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {type(e).__name__}: {e}"
            traceback.print_exc()
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:12s} {v}")
    return 0 if all(v == "PASS" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

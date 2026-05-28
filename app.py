"""ELANG MVP — Streamlit demo.

Run:
    streamlit run app.py

Tabs:
    📷 Image     vehicle detection (+ optional ANPR per crop)
    🎞️ Video     batch detection with optional DeepSORT tracking + zone
    🗺️ Heatmap   violation heatmap + officer placement optimizer
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from elang.detection import detect_vehicles
from elang.stats import count_by_class, summarise, to_dataframe
from elang.stubs.heatmap import ViolationPoint, hot_zones, render_heatmap
from elang.stubs.officer_optimizer import score_locations

# Heavy / optional deps: soft-import so the MVP still works without them.
try:
    from elang.stubs.tracking import Tracker
    _TRACKING_OK = True
    _TRACKING_ERR = None
except Exception as e:  # pragma: no cover
    Tracker = None  # type: ignore
    _TRACKING_OK = False
    _TRACKING_ERR = str(e)

try:
    from elang.stubs.anpr import read_plate
    _ANPR_OK = True
    _ANPR_ERR = None
except Exception as e:  # pragma: no cover
    read_plate = None  # type: ignore
    _ANPR_OK = False
    _ANPR_ERR = str(e)


st.set_page_config(
    page_title="ELANG — Intelligent Traffic Enforcement (MVP)",
    page_icon="🦅",
    layout="wide",
)

st.title("🦅 ELANG — Intelligent Traffic Enforcement (MVP)")
st.caption(
    "Electronic Enforcement & Analysis for Next-Gen traffic Governance. "
    "Reference architecture: `competition-analysis/DISHUB_Case_Analysis.md`."
)

with st.sidebar:
    st.header("Settings")
    weights = st.selectbox(
        "Model weights",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="Larger = more accurate, slower. nano works on CPU.",
    )
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    max_frames = st.number_input(
        "Max video frames to process",
        min_value=1, max_value=2000, value=120, step=10,
        help="Cap inference cost on long clips.",
    )
    stride = st.number_input(
        "Frame stride",
        min_value=1, max_value=30, value=5,
        help="Process every Nth frame.",
    )
    st.markdown("---")
    st.markdown("**Phase 2 modules**")
    st.markdown(
        f"- ANPR (PaddleOCR): {'✅ available' if _ANPR_OK else '⚠️ optional — install paddleocr'}\n"
        f"- Tracking (DeepSORT): {'✅ available' if _TRACKING_OK else '⚠️ optional — install deep-sort-realtime'}\n"
        f"- Heatmap (Folium): ✅ available when folium installed\n"
        f"- Officer optimizer: ✅ pure-Python, always on"
    )
    st.markdown(
        "**Phase 3 (still stubbed):** CRM classifier (Sentence-BERT). "
        "See `ROADMAP` in README."
    )


def _draw(image_bgr: np.ndarray, detections) -> np.ndarray:
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.label} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _draw_tracks(image_bgr: np.ndarray, tracks, zone_polygon=None) -> np.ndarray:
    out = image_bgr.copy()
    if zone_polygon and len(zone_polygon) >= 3:
        pts = np.array(zone_polygon, dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    for t in tracks:
        x1, y1, x2, y2 = t.last_bbox
        colour = (0, 165, 255) if t.frames_in_zone > 0 else (255, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"#{t.track_id} {t.label} [in-zone {t.frames_in_zone}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


tab_image, tab_video, tab_heatmap = st.tabs(["📷 Image", "🎞️ Video", "🗺️ Heatmap"])


# ───────────────────────────── IMAGE TAB ──────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        "Upload a traffic image",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )
    do_anpr = st.checkbox(
        "Run ANPR on each detected vehicle (Phase 2 — PaddleOCR)",
        value=False,
        disabled=not _ANPR_OK,
        help=None if _ANPR_OK else f"PaddleOCR not available: {_ANPR_ERR}",
    )

    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLOv8 inference..."):
            detections = detect_vehicles(bgr, conf=conf, weights=weights)

        annotated = _draw(bgr, detections)
        plate_reads: list[dict] = []

        if do_anpr and _ANPR_OK:
            with st.spinner("Running ANPR on each vehicle crop..."):
                for d in detections:
                    x1, y1, x2, y2 = d.bbox
                    crop = bgr[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    try:
                        plate = read_plate(crop, min_confidence=0.6)
                    except Exception as e:
                        plate = None
                        st.warning(f"ANPR failed on one crop: {e}")
                    if plate is not None:
                        plate_reads.append({
                            "vehicle": d.label,
                            "plate": plate.text,
                            "confidence": round(plate.confidence, 3),
                        })
                        px1, py1, _, _ = d.bbox
                        cv2.putText(
                            annotated, plate.text, (px1, py1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
                        )

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(annotated_rgb, caption="Detections", use_column_width=True)
        with col2:
            summary = summarise(detections)
            st.metric("Total vehicles", summary["total"])
            st.metric("Avg confidence", summary["avg_confidence"])
            st.subheader("By class")
            st.bar_chart(pd.Series(summary["by_class"]).sort_values(ascending=False))
            if plate_reads:
                st.subheader("Plates read")
                st.dataframe(pd.DataFrame(plate_reads))
            with st.expander("Raw detections"):
                st.dataframe(to_dataframe(detections))


# ───────────────────────────── VIDEO TAB ──────────────────────────────
with tab_video:
    uploaded_v = st.file_uploader(
        "Upload a short traffic video clip",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )

    col_track, col_zone = st.columns(2)
    with col_track:
        do_track = st.checkbox(
            "Enable DeepSORT tracking",
            value=False,
            disabled=not _TRACKING_OK,
            help=None if _TRACKING_OK else f"deep-sort-realtime not installed: {_TRACKING_ERR}",
        )
        min_violation_frames = st.number_input(
            "Min frames in zone → violation",
            min_value=1, max_value=500, value=10, step=1,
            disabled=not do_track,
        )
    with col_zone:
        zone_text = st.text_area(
            "Restricted-zone polygon (one `x,y` per line, ≥3 points)",
            value="",
            height=120,
            disabled=not do_track,
            help="Pixel coords in the source video frame. Leave empty to disable zone scoring.",
        )

    zone_polygon: list[tuple[int, int]] | None = None
    if do_track and zone_text.strip():
        try:
            zone_polygon = [
                tuple(int(x) for x in line.split(","))
                for line in zone_text.strip().splitlines()
                if line.strip()
            ]
            if any(len(p) != 2 for p in zone_polygon) or len(zone_polygon) < 3:
                raise ValueError("need ≥3 (x,y) points")
        except Exception as e:
            st.error(f"Could not parse zone polygon: {e}")
            zone_polygon = None

    if uploaded_v is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_v.name).suffix) as tmp:
            tmp.write(uploaded_v.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        st.info(f"Loaded video: {total} frames @ {fps:.1f} fps. "
                f"Processing every {stride} frame(s), capped at {max_frames}.")

        progress = st.progress(0.0, text="Detecting...")
        preview_slot = st.empty()
        all_counts: dict[str, int] = {}
        per_frame_rows = []
        tracker = Tracker() if (do_track and _TRACKING_OK) else None

        processed = 0
        frame_idx = 0
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                dets = detect_vehicles(frame, conf=conf, weights=weights)
                counts = count_by_class(dets)
                for k, v in counts.items():
                    all_counts[k] = all_counts.get(k, 0) + v
                per_frame_rows.append({"frame": frame_idx, **counts, "total": len(dets)})

                if tracker is not None:
                    tracks = tracker.update(dets, frame=frame, frame_idx=frame_idx,
                                            zone_polygon=zone_polygon)
                    if processed % 3 == 0:
                        annotated = cv2.cvtColor(
                            _draw_tracks(frame, tracks, zone_polygon),
                            cv2.COLOR_BGR2RGB,
                        )
                        preview_slot.image(annotated, caption=f"Frame {frame_idx}", use_column_width=True)
                elif processed % 3 == 0:
                    annotated = cv2.cvtColor(_draw(frame, dets), cv2.COLOR_BGR2RGB)
                    preview_slot.image(annotated, caption=f"Frame {frame_idx}", use_column_width=True)

                processed += 1
                progress.progress(min(processed / max_frames, 1.0),
                                  text=f"Processed {processed}/{max_frames} frames")
            frame_idx += 1
        cap.release()
        progress.empty()

        st.success(f"Done. {processed} frames inferenced.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total detections across frames")
            st.bar_chart(pd.Series(all_counts).sort_values(ascending=False))
        with col2:
            st.subheader("Per-frame timeline")
            df = pd.DataFrame(per_frame_rows).fillna(0).set_index("frame")
            st.line_chart(df.drop(columns=["total"]) if "total" in df.columns else df)

        if tracker is not None:
            st.subheader("Tracks")
            tracks_df = pd.DataFrame([
                {
                    "track_id": t.track_id,
                    "label": t.label,
                    "first_seen": t.first_seen_frame,
                    "last_seen": t.last_seen_frame,
                    "duration_frames": t.duration_frames,
                    "frames_in_zone": t.frames_in_zone,
                }
                for t in tracker.active_tracks()
            ])
            st.dataframe(tracks_df)
            if zone_polygon:
                violators = tracker.violators(int(min_violation_frames))
                st.metric("Violators (in-zone ≥ threshold)", len(violators))

        with st.expander("Per-frame raw counts"):
            st.dataframe(df)


# ──────────────────────────── HEATMAP TAB ─────────────────────────────
with tab_heatmap:
    st.markdown(
        "Upload a CSV of violations (columns: **lat, lon, hour, violation_type, count**). "
        "Or paste rows below. The map renders to HTML and the optimizer "
        "ranks officer-placement candidates."
    )

    sample_csv = (
        "lat,lon,hour,violation_type,count\n"
        "-6.2088,106.8456,8,parkir_liar,3\n"
        "-6.2095,106.8460,8,parkir_liar,5\n"
        "-6.2102,106.8470,17,lampu_merah,2\n"
        "-6.1750,106.8270,12,trotoar,4\n"
        "-6.1755,106.8275,12,trotoar,2\n"
    )
    csv_text = st.text_area("Violation CSV", value=sample_csv, height=180)

    uploaded_csv = st.file_uploader("…or upload a CSV", type=["csv"], key="hm_csv")
    if uploaded_csv is not None:
        csv_text = uploaded_csv.read().decode("utf-8")

    col_g, col_k = st.columns(2)
    with col_g:
        grid_size = st.number_input(
            "Grid cell size (degrees, ~111m at 0.001)",
            min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f",
        )
    with col_k:
        top_k = st.number_input("Top-K hotspots", min_value=1, max_value=100, value=10)

    cand_text = st.text_area(
        "Candidate officer/camera locations (`lat,lon` per line)",
        value="-6.2090,106.8458\n-6.2100,106.8470\n-6.1752,106.8272\n",
        height=120,
        key="hm_candidates",
    )

    if csv_text.strip():
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            required = {"lat", "lon", "hour", "violation_type"}
            missing = required - set(df.columns)
            if missing:
                st.error(f"CSV missing required columns: {sorted(missing)}")
            else:
                if "count" not in df.columns:
                    df["count"] = 1
                points = [
                    ViolationPoint(
                        lat=float(r.lat), lon=float(r.lon), hour=int(r.hour),
                        violation_type=str(r.violation_type), count=int(r["count"]),
                    )
                    for r in df.itertuples(index=False)
                ]

                zones = hot_zones(points, top_k=int(top_k), grid_size=float(grid_size))
                st.subheader(f"Top {len(zones)} hotspots")
                st.dataframe(pd.DataFrame(zones, columns=["lat", "lon", "count"]))

                tmp_html = Path(tempfile.gettempdir()) / "elang_heatmap.html"
                try:
                    render_heatmap(points, str(tmp_html))
                    st.components.v1.html(tmp_html.read_text(encoding="utf-8"), height=500)
                except RuntimeError as e:
                    st.warning(f"Skipped map render: {e}")

                st.divider()
                st.subheader("Officer placement optimizer")
                candidates: list[tuple[float, float]] = []
                for line in cand_text.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        try:
                            candidates.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            st.warning(f"Skipped malformed candidate: {line!r}")
                if candidates:
                    recs = score_locations(candidates, zones)
                    st.dataframe(pd.DataFrame([r.__dict__ for r in recs]))
                else:
                    st.info("Add at least one candidate location to score placements.")
        except Exception as e:
            st.error(f"Failed: {e}")

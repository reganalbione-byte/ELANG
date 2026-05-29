"""ELANG MVP — Streamlit demo.

Run:
    streamlit run app.py

Tabs:
    📷 Image     vehicle detection (+ optional ANPR per crop)
    🎞️ Video     batch detection with optional DeepSORT tracking + zone
    🗺️ Heatmap   violation heatmap + officer placement optimizer
    💬 CRM       citizen-report classifier (multilingual Sentence-BERT)
"""

from __future__ import annotations

import io
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from elang.detection import detect_vehicles
from elang.stats import count_by_class, summarise, to_dataframe
from elang.stubs.etle import (
    export_to_etle_csv,
    export_to_etle_json,
    generate_mock_violations,
)
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
    from elang.stubs.anpr import (
        INDONESIAN_REGION_CODES,
        preprocess_adverse,
        preprocess_plate_roi,
        read_plate,
        validate_indonesian_plate,
    )
    _ANPR_OK = True
    _ANPR_ERR = None
except Exception as e:  # pragma: no cover
    read_plate = None  # type: ignore
    preprocess_adverse = None  # type: ignore
    preprocess_plate_roi = None  # type: ignore
    validate_indonesian_plate = None  # type: ignore
    INDONESIAN_REGION_CODES = {}  # type: ignore
    _ANPR_OK = False
    _ANPR_ERR = str(e)

try:
    import sentence_transformers  # noqa: F401
    import sklearn  # noqa: F401
    from elang.stubs.crm_classifier import (
        CATEGORIES as CRM_CATEGORIES,
        classify_batch as crm_classify_batch,
        classify_report as crm_classify_report,
    )
    _CRM_OK = True
    _CRM_ERR = None
except Exception as e:  # pragma: no cover
    crm_classify_report = None  # type: ignore
    crm_classify_batch = None  # type: ignore
    CRM_CATEGORIES = []  # type: ignore
    _CRM_OK = False
    _CRM_ERR = str(e)

try:
    import yt_dlp  # noqa: F401
    _YTDLP_OK = True
except Exception:  # pragma: no cover
    yt_dlp = None  # type: ignore
    _YTDLP_OK = False


_RECORDING_MAX_FRAMES = 500


def _append_to_recording(frame_bgr: np.ndarray) -> None:
    """Buffer one annotated BGR frame for the demo recording (bounded)."""
    if not st.session_state.get("recording"):
        return
    frames = st.session_state.setdefault("recording_frames", [])
    if len(frames) >= _RECORDING_MAX_FRAMES:
        return
    frames.append(frame_bgr.copy())


def _save_recording() -> tuple[str | None, str | None]:
    """Encode buffered frames to MP4 in data/. Returns (path, error)."""
    frames = st.session_state.get("recording_frames", [])
    if not frames:
        return None, "No frames in buffer."
    h, w = frames[0].shape[:2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("data") / f"demo_recording_{ts}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (w, h))
    if not writer.isOpened():
        return None, "cv2.VideoWriter failed to open — mp4v codec unavailable."
    for f in frames:
        writer.write(f)
    writer.release()
    return str(out_path), None


def _resolve_stream_url(url: str) -> tuple[str | None, str | None]:
    """Resolve a YouTube / redirect URL to a direct stream URL via yt-dlp.

    Returns (resolved_url, error). For non-YouTube URLs, passes the input
    through unchanged.
    """
    if not url:
        return None, "URL is empty."
    is_youtube = ("youtube.com" in url) or ("youtu.be" in url)
    if not is_youtube:
        return url, None
    if not _YTDLP_OK:
        return None, (
            "yt-dlp is not installed; cannot resolve YouTube URL. "
            "Install with `pip install yt-dlp` or paste a direct RTSP/HTTP "
            "stream URL instead."
        )
    try:
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        direct = info.get("url") if info else None
        if not direct:
            return None, "yt-dlp returned no usable stream URL."
        return direct, None
    except Exception as e:
        return None, f"yt-dlp failed: {e}"


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
    st.markdown("**Phase 3 modules**")
    st.markdown(
        f"- CRM classifier (Sentence-BERT): "
        f"{'✅ available' if _CRM_OK else '⚠️ optional — install sentence-transformers + scikit-learn'}"
    )

    st.markdown("---")
    st.markdown("**🎬 Demo Recording**")
    st.caption(
        "Rekam demo terbaik sebagai backup sebelum presentasi ke juri. "
        f"Max {_RECORDING_MAX_FRAMES} frames; frame diambil dari tab Video saat preview update."
    )
    _rec_active = st.session_state.get("recording", False)
    _rec_count = len(st.session_state.get("recording_frames", []))

    rec_c1, rec_c2 = st.columns(2)
    with rec_c1:
        if st.button("▶ Start", key="rec_start", disabled=_rec_active,
                     use_container_width=True):
            st.session_state.recording = True
            st.session_state.recording_frames = []
            st.session_state.last_recording_path = None
            st.rerun()
    with rec_c2:
        if st.button("⏹ Stop & Save", key="rec_stop",
                     disabled=_rec_count == 0,
                     use_container_width=True):
            st.session_state.recording = False
            path, err = _save_recording()
            if err:
                st.session_state.last_recording_path = None
                st.session_state.last_recording_error = err
            else:
                st.session_state.last_recording_path = path
                st.session_state.last_recording_error = None
                st.session_state.recording_frames = []
            st.rerun()

    if _rec_active:
        st.caption(f"🔴 Recording — {_rec_count}/{_RECORDING_MAX_FRAMES} frames")
    elif _rec_count > 0:
        st.caption(f"📦 {_rec_count} frames buffered (klik Stop & Save untuk encode)")

    _last_path = st.session_state.get("last_recording_path")
    _last_err = st.session_state.get("last_recording_error")
    if _last_path:
        st.success(f"Saved: `{_last_path}`")
    elif _last_err:
        st.error(_last_err)


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


tab_image, tab_video, tab_heatmap, tab_crm = st.tabs(
    ["📷 Image", "🎞️ Video", "🗺️ Heatmap", "💬 CRM Classifier"]
)


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
    col_em, col_pp = st.columns(2)
    with col_em:
        enhance_mode = st.selectbox(
            "Enhance mode",
            ["auto", "clahe", "otsu", "none"],
            index=0,
            disabled=not do_anpr,
            help=(
                "auto / clahe = CLAHE + bilateral (best for full vehicle crops). "
                "otsu = upscale + Otsu + morphology (best for tight plate ROIs). "
                "none = pass crop through raw."
            ),
        )
    with col_pp:
        show_preprocess_preview = st.checkbox(
            "Show preprocessing preview",
            value=False,
            disabled=not do_anpr,
            help="Side-by-side: original vehicle crop vs. preprocessed input fed into PaddleOCR.",
        )

    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLOv8 inference..."):
            detections = detect_vehicles(bgr, conf=conf, weights=weights)

        annotated = _draw(bgr, detections)
        plate_reads: list[dict] = []
        plate_validations: list[dict] = []
        anpr_attempts = 0

        if do_anpr and _ANPR_OK:
            with st.spinner("Running ANPR on each vehicle crop..."):
                for d in detections:
                    x1, y1, x2, y2 = d.bbox
                    crop = bgr[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    anpr_attempts += 1
                    try:
                        plate = read_plate(
                            crop, min_confidence=0.6, enhance_mode=enhance_mode,
                        )
                    except Exception as e:
                        plate = None
                        st.warning(f"ANPR failed on one crop: {e}")
                    if plate is not None:
                        validation = validate_indonesian_plate(plate.text)
                        plate_reads.append({
                            "vehicle": d.label,
                            "plate": plate.text,
                            "formatted": validation["formatted"],
                            "region": validation["region_code"] or "—",
                            "type": validation["plate_type"],
                            "valid": validation["is_valid"],
                            "confidence": round(plate.confidence, 3),
                        })
                        plate_validations.append(validation)
                        px1, py1, _, _ = d.bbox
                        cv2.putText(
                            annotated, validation["formatted"] or plate.text,
                            (px1, py1 + 18),
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

        if do_anpr and _ANPR_OK and show_preprocess_preview and len(detections) > 0:
            x1, y1, x2, y2 = detections[0].bbox
            sample_crop = bgr[max(0, y1):y2, max(0, x1):x2]
            if sample_crop.size > 0:
                if enhance_mode == "otsu":
                    processed = preprocess_plate_roi(sample_crop)
                elif enhance_mode == "none":
                    processed = sample_crop
                else:
                    processed = preprocess_adverse(sample_crop)
                st.subheader("Preprocessing preview")
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.image(cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB),
                             caption="Original crop", use_column_width=True)
                with pc2:
                    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                             caption=f"After enhance_mode='{enhance_mode}'",
                             use_column_width=True)

        if do_anpr and _ANPR_OK and plate_validations:
            st.subheader("Plate validation")
            for v in plate_validations:
                fmt = v["formatted"]
                rtype = v["plate_type"]
                rcode = v["region_code"]
                region_name = INDONESIAN_REGION_CODES.get(rcode or "", "—")
                if not v["is_valid"]:
                    st.error(
                        f"🔴 Likely false positive: `{fmt}` — type=`{rtype}`"
                    )
                elif rtype == "reguler" and rcode in INDONESIAN_REGION_CODES:
                    st.success(
                        f"🟢 Valid: `{fmt}` — region **{rcode}** ({region_name}), type=`{rtype}`"
                    )
                else:
                    detail = f"type=`{rtype}`"
                    if rcode:
                        detail += f", region=`{rcode}` ({region_name})"
                    st.warning(f"🟡 Plausible but format off: `{fmt}` — {detail}")

        if do_anpr and _ANPR_OK and anpr_attempts > 0:
            valid_count = sum(1 for v in plate_validations if v["is_valid"])
            read_count = len(plate_validations)
            rejected = anpr_attempts - read_count
            st.divider()
            st.subheader("ANPR Accuracy Report")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Crops attempted", anpr_attempts)
            mc2.metric("Plates read", read_count)
            mc3.metric("Valid plates", valid_count)
            success_rate = valid_count / anpr_attempts
            st.progress(min(max(success_rate, 0.0), 1.0))
            st.caption(
                f"Success rate (valid / attempted): **{success_rate:.1%}**. "
                f"{rejected} crop(s) yielded no OCR read; "
                f"{read_count - valid_count} read(s) failed validation."
            )
            st.caption(
                "**Target akurasi yang jujur:** 85% pada kondisi normal "
                "(siang, kamera CCTV ETLE, plat menghadap kamera); "
                "65% pada kondisi buruk (malam, hujan, sudut miring, motor). "
                "Hasil aktual bergantung pada resolusi kamera dan cuaca."
            )


# ───────────────────────────── VIDEO TAB ──────────────────────────────
with tab_video:
    # Detect source-switch BEFORE the radio so we can clean up stream state.
    _prev_source = st.session_state.get("_prev_video_source")
    input_source = st.radio(
        "Input source",
        ["Upload File", "RTSP Stream", "Webcam"],
        horizontal=True,
        index=0,
        key="video_input_source",
    )
    if _prev_source is not None and _prev_source != input_source:
        st.session_state.stream_active = False
        for _k in ("stream_source", "stream_kind"):
            st.session_state.pop(_k, None)
    st.session_state["_prev_video_source"] = input_source

    uploaded_v = None
    stream_to_open: str | int | None = None

    if input_source == "Upload File":
        uploaded_v = st.file_uploader(
            "Upload a short traffic video clip",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_upload",
        )

    elif input_source == "RTSP Stream":
        preset = st.selectbox(
            "Quick presets",
            ["Custom URL", "CCTV Sample (YouTube stream)"],
            key="rtsp_preset",
        )
        preset_default = (
            "https://www.youtube.com/watch?v=ydYDqZQpim8"
            if preset == "CCTV Sample (YouTube stream)" else ""
        )
        rtsp_url = st.text_input(
            "RTSP / Stream URL",
            value=preset_default,
            placeholder="rtsp://username:password@192.168.1.1:554/stream",
            key="rtsp_url",
        )
        st.warning(
            "⚠️ RTSP stream membutuhkan network access ke kamera. "
            "Untuk demo offline, gunakan **Upload File**."
        )
        if preset == "CCTV Sample (YouTube stream)" and not _YTDLP_OK:
            st.info(
                "ℹ️ YouTube preset selected, but `yt-dlp` is not installed. "
                "Install with `pip install yt-dlp` to resolve YouTube URLs to "
                "direct stream URLs."
            )
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("▶ Start", key="rtsp_start",
                         disabled=not rtsp_url.strip(),
                         use_container_width=True):
                st.session_state.stream_active = True
                st.session_state.stream_source = rtsp_url.strip()
                st.session_state.stream_kind = "rtsp"
        with sc2:
            if st.button("⏹ Stop", key="rtsp_stop",
                         use_container_width=True):
                st.session_state.stream_active = False

        if (st.session_state.get("stream_active")
                and st.session_state.get("stream_kind") == "rtsp"):
            _src = st.session_state.get("stream_source")
            if _src and ("youtube.com" in _src or "youtu.be" in _src):
                with st.spinner("Resolving stream via yt-dlp..."):
                    resolved, err = _resolve_stream_url(_src)
                if err:
                    st.error(
                        f"{err}\n\n**Fallback:** switch to Upload File mode, "
                        f"or paste a direct RTSP/HTTP stream URL."
                    )
                    st.session_state.stream_active = False
                else:
                    stream_to_open = resolved
            else:
                stream_to_open = _src

    elif input_source == "Webcam":
        webcam_idx = st.selectbox(
            "Device index",
            [0, 1, 2],
            key="webcam_idx",
            help="0 is typically the built-in camera; 1 / 2 are external USB cams.",
        )
        st.caption(
            "Akses kamera lokal lewat OpenCV. Jika gagal, cek izin OS untuk "
            "Python / Streamlit mengakses kamera."
        )
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("▶ Start", key="webcam_start",
                         use_container_width=True):
                st.session_state.stream_active = True
                st.session_state.stream_source = int(webcam_idx)
                st.session_state.stream_kind = "webcam"
        with sc2:
            if st.button("⏹ Stop", key="webcam_stop",
                         use_container_width=True):
                st.session_state.stream_active = False

        if (st.session_state.get("stream_active")
                and st.session_state.get("stream_kind") == "webcam"):
            stream_to_open = st.session_state.get("stream_source")

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

    col_anpr_v, col_anpr_em = st.columns(2)
    with col_anpr_v:
        do_anpr_video = st.checkbox(
            "Run ANPR on video frames (slow on CPU)",
            value=False,
            disabled=not _ANPR_OK,
            help=(
                f"PaddleOCR not available: {_ANPR_ERR}" if not _ANPR_OK
                else "Runs PaddleOCR on each detection in every processed frame. "
                     "Expect ~1-2s/frame on CPU. Use small max_frames + larger stride."
            ),
        )
    with col_anpr_em:
        enhance_mode_v = st.selectbox(
            "Enhance mode (video ANPR)",
            ["auto", "clahe", "otsu", "none"],
            index=0,
            disabled=not do_anpr_video,
            key="enhance_mode_video",
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

    # ── Upload-file processing (existing flow, with recording hook) ─────────
    if input_source == "Upload File" and uploaded_v is not None:
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
        unique_plates: dict[str, dict] = {}

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

                if do_anpr_video and _ANPR_OK:
                    for d in dets:
                        x1, y1, x2, y2 = d.bbox
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size == 0:
                            continue
                        try:
                            plate = read_plate(
                                crop, min_confidence=0.6,
                                enhance_mode=enhance_mode_v,
                            )
                        except Exception:
                            plate = None
                        if plate is None:
                            continue
                        v_info = validate_indonesian_plate(plate.text)
                        key = v_info["formatted"] or plate.text
                        existing = unique_plates.get(key)
                        if existing is None:
                            unique_plates[key] = {
                                "plate": key,
                                "vehicle_class": d.label,
                                "first_seen_frame": frame_idx,
                                "confidence": round(float(plate.confidence), 3),
                                "type": v_info["plate_type"],
                                "region": v_info["region_code"] or "—",
                                "valid": v_info["is_valid"],
                            }
                        elif plate.confidence > existing["confidence"]:
                            existing["confidence"] = round(float(plate.confidence), 3)
                            existing["vehicle_class"] = d.label

                if tracker is not None:
                    tracks = tracker.update(dets, frame=frame, frame_idx=frame_idx,
                                            zone_polygon=zone_polygon)
                    if processed % 3 == 0:
                        annotated_bgr = _draw_tracks(frame, tracks, zone_polygon)
                        preview_slot.image(
                            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx}", use_column_width=True,
                        )
                        _append_to_recording(annotated_bgr)
                elif processed % 3 == 0:
                    annotated_bgr = _draw(frame, dets)
                    preview_slot.image(
                        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}", use_column_width=True,
                    )
                    _append_to_recording(annotated_bgr)

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

        if do_anpr_video and _ANPR_OK:
            st.subheader("Unique Plates Detected")
            if unique_plates:
                plates_df = (
                    pd.DataFrame(unique_plates.values())
                    .sort_values(["first_seen_frame", "confidence"],
                                 ascending=[True, False])
                    .reset_index(drop=True)
                )
                st.dataframe(plates_df, use_container_width=True)
                st.caption(
                    f"{len(plates_df)} unique plate(s) extracted across {processed} "
                    f"processed frame(s). Dedup key = canonical formatted plate; "
                    f"highest-confidence read kept per plate."
                )
            else:
                st.info("No plates passed the OCR confidence + format filter.")

        with st.expander("Per-frame raw counts"):
            st.dataframe(df)

    # ── RTSP / Webcam streaming (bounded by max_frames per Start click) ─────
    elif stream_to_open is not None:
        st.info(
            f"Streaming source `{stream_to_open}` — processing budget "
            f"**{int(max_frames)}** frames, then auto-stop. "
            "(Streamlit limitation: **Stop** only takes effect after the "
            "current batch finishes.)"
        )
        try:
            cap = cv2.VideoCapture(stream_to_open)
        except Exception as e:
            st.error(
                f"Could not create capture: {e}. "
                f"**Fallback:** switch to Upload File mode."
            )
            cap = None

        if cap is None or not cap.isOpened():
            st.error(
                "Failed to open stream / device. Possible causes: wrong URL, "
                "no network access to camera, device permissions blocked, or "
                "stream requires authentication. "
                "**Fallback:** switch to Upload File mode."
            )
            if cap is not None:
                cap.release()
            st.session_state.stream_active = False
        else:
            preview_slot = st.empty()
            progress = st.progress(0.0, text="Streaming...")
            tracker = Tracker() if (do_track and _TRACKING_OK) else None
            processed = 0
            frame_idx = 0
            budget = int(max_frames)
            while processed < budget:
                ret, frame = cap.read()
                if not ret:
                    st.warning(f"Stream ended after {processed} processed frames.")
                    break
                if frame_idx % stride == 0:
                    dets = detect_vehicles(frame, conf=conf, weights=weights)
                    if tracker is not None:
                        tracks = tracker.update(
                            dets, frame=frame, frame_idx=frame_idx,
                            zone_polygon=zone_polygon,
                        )
                        annotated_bgr = _draw_tracks(frame, tracks, zone_polygon)
                    else:
                        annotated_bgr = _draw(frame, dets)
                    if processed % 3 == 0:
                        preview_slot.image(
                            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx}",
                            use_column_width=True,
                        )
                    _append_to_recording(annotated_bgr)
                    processed += 1
                    progress.progress(
                        min(processed / budget, 1.0),
                        text=f"Streaming {processed}/{budget} frames",
                    )
                    time.sleep(0.033)
                frame_idx += 1
            cap.release()
            progress.empty()
            st.success(
                f"Stream batch complete: {processed} frames processed. "
                "Klik **▶ Start** lagi untuk lanjut, atau **⏹ Stop**."
            )
            st.session_state.stream_active = False


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

    st.divider()
    st.subheader("📤 Export ke E-TLE")
    st.caption(
        "Konversi violation detections menjadi paket file standar E-TLE POLRI. "
        "Mock generator dipakai untuk demo — di produksi, sumbernya adalah "
        "track-violator output dari tab Video + plate reads dari ANPR."
    )

    col_n, col_btn = st.columns([1, 2])
    with col_n:
        n_violations = st.number_input(
            "Jumlah mock violations",
            min_value=1, max_value=200, value=10, step=1,
            key="etle_n",
        )
    with col_btn:
        st.write("")
        if st.button("Generate Mock Violations", key="etle_generate"):
            st.session_state["etle_records"] = generate_mock_violations(int(n_violations))

    etle_records = st.session_state.get("etle_records", [])
    if etle_records:
        st.success(
            f"✅ Format-compatible dengan standar E-TLE POLRI "
            f"— {len(etle_records)} violations siap diexport"
        )

        df_etle = pd.DataFrame([asdict(r) for r in etle_records])
        st.dataframe(df_etle, use_container_width=True, height=320)

        tmp_dir = Path(tempfile.gettempdir())
        json_path = export_to_etle_json(etle_records, str(tmp_dir / "elang_etle_export.json"))
        csv_path = export_to_etle_csv(etle_records, str(tmp_dir / "elang_etle_export.csv"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        col_dj, col_dc = st.columns(2)
        with col_dj:
            st.download_button(
                "⬇️ Download JSON",
                data=Path(json_path).read_bytes(),
                file_name=f"elang_etle_{stamp}.json",
                mime="application/json",
                key="etle_dl_json",
                use_container_width=True,
            )
        with col_dc:
            st.download_button(
                "⬇️ Download CSV",
                data=Path(csv_path).read_bytes(),
                file_name=f"elang_etle_{stamp}.csv",
                mime="text/csv",
                key="etle_dl_csv",
                use_container_width=True,
            )

        with st.expander("Format envelope (JSON spec)"):
            st.code(
                '{\n'
                '  "version": "1.0",\n'
                '  "source": "ELANG",\n'
                '  "export_timestamp": "ISO 8601 UTC",\n'
                '  "total_records": N,\n'
                '  "violations": [ ViolationRecord, ... ]\n'
                '}',
                language="json",
            )
    else:
        st.info("Klik **Generate Mock Violations** untuk membuat sample data.")


# ─────────────────────────────── CRM TAB ──────────────────────────────
def _urgency_badge(urgency: str) -> None:
    if urgency == "high":
        st.error(f"🔴 Urgency: HIGH")
    elif urgency == "medium":
        st.warning(f"🟡 Urgency: MEDIUM")
    else:
        st.success(f"🟢 Urgency: LOW")


with tab_crm:
    st.markdown(
        "Klasifikasi laporan warga (kanal 112 / JAKI) ke kategori pelanggaran lalu lintas. "
        "Model: Sentence-BERT multilingual (`paraphrase-multilingual-MiniLM-L12-v2`) "
        "+ LogisticRegression yang dilatih di atas seed data Bahasa Indonesia."
    )

    if not _CRM_OK:
        st.warning(
            "⚠️ CRM classifier dependencies belum terinstall: "
            f"`{_CRM_ERR}`.\n\n"
            "Uncomment `sentence-transformers` + `scikit-learn` di "
            "`requirements.txt` lalu jalankan `pip install -r requirements.txt`."
        )
    else:
        st.caption("Kategori yang dikenali: " + ", ".join(f"`{c}`" for c in CRM_CATEGORIES))

        mode = st.radio(
            "Mode",
            ["Single report", "Batch (CSV)"],
            horizontal=True,
            key="crm_mode",
        )

        if mode == "Single report":
            report_text = st.text_area(
                "Laporan warga",
                value="",
                placeholder="Ada motor parkir di atas trotoar depan Indomaret Sudirman",
                height=120,
                key="crm_single_text",
            )
            if st.button("Classify", type="primary", key="crm_classify_btn"):
                if not report_text.strip():
                    st.info("Masukkan teks laporan dulu.")
                else:
                    with st.spinner("Embedding + classifying..."):
                        result = crm_classify_report(report_text)
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Kategori:** `{result.predicted_category}`")
                        st.markdown(f"**Confidence:** {result.confidence:.1%}")
                        st.progress(min(max(result.confidence, 0.0), 1.0))
                    with col_b:
                        _urgency_badge(result.urgency)

        else:
            st.markdown(
                "Upload CSV dengan kolom `text` (satu laporan per baris). "
                "Hasil ditampilkan sebagai tabel + distribusi kategori."
            )
            uploaded_csv = st.file_uploader(
                "CSV file",
                type=["csv"],
                key="crm_csv_upload",
            )
            sample = st.checkbox(
                "Atau pakai contoh inline (5 laporan)",
                value=False,
                key="crm_use_sample",
            )

            df_in: pd.DataFrame | None = None
            if uploaded_csv is not None:
                try:
                    df_in = pd.read_csv(uploaded_csv)
                except Exception as e:
                    st.error(f"Gagal membaca CSV: {e}")
            elif sample:
                df_in = pd.DataFrame({"text": [
                    "Ada motor parkir di atas trotoar depan Indomaret Sudirman",
                    "Mobil pribadi masuk jalur busway di Sudirman pagi tadi",
                    "Banyak motor terobos lampu merah di simpang Tomang",
                    "Lampu jalan mati di Jalan Kemang Raya",
                    "Kecelakaan parah antara motor dan truk di Cawang, ada korban luka",
                ]})

            if df_in is not None:
                if "text" not in df_in.columns:
                    st.error("CSV harus punya kolom `text`.")
                else:
                    texts = df_in["text"].astype(str).tolist()
                    with st.spinner(f"Classifying {len(texts)} reports..."):
                        results = crm_classify_batch(texts)

                    df_out = pd.DataFrame([
                        {
                            "text": r.text,
                            "category": r.predicted_category,
                            "confidence": round(r.confidence, 3),
                            "urgency": r.urgency,
                        }
                        for r in results
                    ])
                    st.subheader("Hasil klasifikasi")
                    st.dataframe(df_out, use_container_width=True)

                    st.subheader("Distribusi kategori")
                    dist = (
                        df_out["category"]
                        .value_counts()
                        .rename_axis("category")
                        .reset_index(name="count")
                    )
                    try:
                        import altair as alt
                        chart = (
                            alt.Chart(dist)
                            .mark_arc(innerRadius=40)
                            .encode(
                                theta=alt.Theta(field="count", type="quantitative"),
                                color=alt.Color(field="category", type="nominal"),
                                tooltip=["category", "count"],
                            )
                        )
                        st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        st.bar_chart(dist.set_index("category"))

                    urg_counts = df_out["urgency"].value_counts()
                    col_l, col_m, col_h = st.columns(3)
                    col_l.metric("🟢 Low", int(urg_counts.get("low", 0)))
                    col_m.metric("🟡 Medium", int(urg_counts.get("medium", 0)))
                    col_h.metric("🔴 High", int(urg_counts.get("high", 0)))

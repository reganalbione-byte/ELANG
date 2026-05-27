"""ELANG MVP — Streamlit demo for vehicle detection.

Run:
    streamlit run app.py

Upload an image or short video clip; the app runs YOLOv8 vehicle
detection and shows annotated output + per-class counts.
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
    st.markdown(
        "**Phase 2/3 (stubs):** ANPR · DeepSORT tracking · "
        "violation heatmap · CRM classifier · officer placement optimizer. "
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


tab_image, tab_video = st.tabs(["📷 Image", "🎞️ Video"])

with tab_image:
    uploaded = st.file_uploader(
        "Upload a traffic image",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLOv8 inference..."):
            detections = detect_vehicles(bgr, conf=conf, weights=weights)

        annotated = cv2.cvtColor(_draw(bgr, detections), cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(annotated, caption="Detections", use_column_width=True)
        with col2:
            summary = summarise(detections)
            st.metric("Total vehicles", summary["total"])
            st.metric("Avg confidence", summary["avg_confidence"])
            st.subheader("By class")
            st.bar_chart(pd.Series(summary["by_class"]).sort_values(ascending=False))
            with st.expander("Raw detections"):
                st.dataframe(to_dataframe(detections))


with tab_video:
    uploaded_v = st.file_uploader(
        "Upload a short traffic video clip",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )
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

                if processed % 3 == 0:
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
        with st.expander("Per-frame raw counts"):
            st.dataframe(df)

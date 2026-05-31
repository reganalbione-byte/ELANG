# ELANG — Operating Guide

How to **test**, how to **operate**, and **what each part does**. This is the
hands-on companion to `README.md` (which covers architecture and install notes
in prose). If you only read one file before a demo, read this one.

Built for the AI Open Innovation Challenge 2026 (DISHUB DKI Jakarta, Case 1).
Reference analysis: `../competition-analysis/DISHUB_Case_Analysis.md`.

---

## 1. What It Does (Function)

ELANG is a **traffic-enforcement pipeline** wrapped in a Streamlit app. It turns
CCTV footage and citizen reports into structured, E-TLE-ready violation records.

End-to-end flow:

```
CCTV frame ─▶ YOLOv8 detect vehicles ─▶ DeepSORT track + zone timer ─▶ violation
                     │                                                    │
                     └─▶ PaddleOCR read plate ─▶ validate TNKB format ────┘
                                                                          ▼
Citizen text ─▶ Sentence-BERT + LogReg classify ─▶ category + urgency    E-TLE
Violation CSV ─▶ grid hotspots ─▶ officer-placement optimizer        JSON / CSV export
```

### Modules (`elang/`)

| File | Function | Optional dep |
|------|----------|--------------|
| `detection.py` | YOLOv8 wrapper; COCO→local label, lazy-loaded, CPU-friendly nano default | — (ultralytics, core) |
| `classes.py` | Maps COCO IDs to local labels (`mobil`, `motor`, `bus`, `truk`, `sepeda`) | — |
| `stats.py` | Detection aggregation (counts, summary, dataframe) | — |
| `stubs/anpr.py` | PaddleOCR plate read + CLAHE/Otsu preprocessing + Indonesian TNKB validation (23 region codes, 4 plate types) | paddleocr + paddlepaddle |
| `stubs/tracking.py` | DeepSORT IDs + ray-cast point-in-polygon zone dwell-time → violation flag | deep-sort-realtime |
| `stubs/heatmap.py` | Grid-based hotspot aggregation + Folium heatmap render | folium |
| `stubs/officer_optimizer.py` | Rule-based candidate scoring by haversine proximity to hotspots | — (pure Python) |
| `stubs/crm_classifier.py` | Multilingual Sentence-BERT + LogisticRegression routing citizen reports into 6 categories + urgency heuristic | sentence-transformers + scikit-learn |
| `stubs/etle.py` | Mock violation generator + JSON/CSV export in POLRI E-TLE envelope | — (stdlib) |

Every optional module is **lazy-imported** — the app and the rest of the package
run even when its dep is missing; the feature just shows as unavailable.

### App tabs (`app.py`)

- **📷 Image** — detect vehicles in one frame; optional per-crop ANPR with an honest
  accuracy report (valid plates / crops attempted).
- **🎞️ Video** — Upload / RTSP / Webcam input; optional DeepSORT tracking + restricted-zone
  dwell counting; optional ANPR with a deduplicated unique-plates table.
- **🗺️ Heatmap** — violation CSV → hotspot grid → Folium map → officer-placement ranking →
  E-TLE JSON/CSV export.
- **💬 CRM Classifier** — single or batch citizen-report classification with category
  distribution and urgency tally.

The sidebar has a **demo recorder** (buffers annotated frames → MP4 in `data/`) as a
fallback if a live demo flakes on stage.

---

## 2. How to Operate

### Install (Python 3.10+)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_sample.py     # pulls data/sample.jpg
```

This installs the **core + Phase 3 (CRM)** stack. Phase 2 extras (tracking, ANPR) and
`yt-dlp` are commented in `requirements.txt` — uncomment what you need. Cold install with
torch + paddle is 1–2 GB and 10–15 min. See `README.md` → "Windows install notes" for the
two known Windows gotchas (paddle 3.x bug, torch/cv2 DLL order).

### Launch

```powershell
streamlit run app.py
```

First run downloads `yolov8n.pt` (~6 MB). First CRM call downloads MiniLM weights (~120 MB)
and caches the trained head to `elang/stubs/crm_model.pkl`.

### Operating each tab

**Image:** upload → see boxes + class chart. Tick *Run ANPR* for plates; pick **enhance mode**
(`auto`/`clahe` for full vehicle crops, `otsu` for tight plate ROIs). The Accuracy Report at
the bottom is the figure to show reviewers.

**Video:** pick input source. *Upload File* is the safe demo path. Tick *Enable DeepSORT* and
paste a polygon (`x,y` per line, ≥3 points) to flag vehicles dwelling in a restricted zone past
the frame threshold. ANPR-on-video is slow on CPU — keep `max_frames` small and `stride` large
(sidebar). RTSP needs network to the camera; have a recorded backup.

**Heatmap:** the tab ships with an inline sample CSV so it works with zero input. Feed
`lat,lon,hour,violation_type,count` rows → hotspots + map. Add `lat,lon` candidate lines →
placement ranking. Then *Generate Mock Violations* → download E-TLE JSON/CSV.

**CRM:** type a report (ID or EN) or upload a CSV with a `text` column. Output = category +
confidence + urgency, plus a distribution chart for batches.

### Tuning knobs (sidebar)

`Model weights` (nano→medium), `Confidence threshold`, `Max video frames`, `Frame stride`.
On GPU, swap to `yolov8m.pt`/`yolov8l.pt` for a large accuracy gain on small vehicles.

---

## 3. How to Test

### Automated smoke test

```powershell
python scripts/smoke_test.py
```

Exercises all six modules against the sample image / synthetic inputs and prints a summary.
Each check reports one of:

| Status | Meaning |
|--------|---------|
| **PASS** | Ran and met its assertion |
| **SKIP** | Couldn't run — optional dep or `data/sample.jpg` missing. **Not a failure.** |
| **FAIL** | Ran but the assertion failed — investigate |
| **ERROR** | Threw an unexpected exception (traceback printed) |

The process **exits 0** when every check is PASS or SKIP, and **exits non-zero** only on a
real FAIL/ERROR — so it's CI-safe even without the optional deps installed. Expected output
on a core-only install:

```
5 passed, 1 skipped, 0 failed.     # crm SKIP if sentence-transformers absent
```

What each check asserts:
- **detection** — ≥1 vehicle found in the sample image.
- **heatmap** — non-empty hotspot list; HTML map renders (or SKIP if folium absent).
- **optimizer** — the on-hotspot candidate scores higher than the far-away one.
- **tracking** — ≥1 confirmed track, with in-zone frames counted (SKIP if DeepSORT absent).
- **anpr** — plate-plausibility logic correct; preprocessing runs (SKIP if PaddleOCR absent).
- **crm** — every sample routes to a known category and the accident report is flagged
  `urgency=high` (SKIP if sentence-transformers/scikit-learn absent).

### Run a single check during development

```python
import sys; sys.path.insert(0, ".")
from scripts.smoke_test import check_crm, check_detection
check_detection()   # -> True / False / "SKIP"
```

### Manual UI test (pre-demo checklist)

1. `streamlit run app.py` — confirm sidebar shows correct ✅/⚠️ module availability.
2. **Image** tab: upload `data/sample.jpg` → boxes drawn, metrics populate.
3. **Video** tab (Upload File): short clip → preview updates, per-frame chart renders.
4. **Heatmap** tab: leave the inline CSV → hotspots table + map + optimizer table appear;
   *Generate Mock Violations* → JSON and CSV download buttons work.
5. **CRM** tab: paste a report → category + urgency badge (or the install hint if deps absent).
6. Sidebar: **▶ Start** recording, run the Video tab, **⏹ Stop & Save** → MP4 lands in `data/`.

### Smoke-test before every demo

Run `python scripts/smoke_test.py` on the **actual demo machine** with the **actual** set of
installed deps. A SKIP there tells you a feature won't be live on stage — decide whether to
install the dep or avoid that tab.

---

*Companion to `README.md`. Prototype status: end-to-end smoke-tested on Windows-CPU.*

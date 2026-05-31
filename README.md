# 🦅 ELANG — Intelligent Traffic Enforcement

**Electronic Enforcement & Analysis for Next-Gen traffic Governance**

A working proof-of-concept that turns a city's *existing* CCTV network into an
automated traffic-violation detection and analysis system. Built for the **AI Open
Innovation Challenge 2026** (DISHUB DKI Jakarta — Case 1: Intelligent Traffic
Enforcement & Behaviour Analysis).

`Python 3.10+` · `Streamlit` · `YOLOv8 · PaddleOCR · DeepSORT · Sentence-BERT` · Status: **working POC, CPU-verified**

---

## The problem

Jakarta doesn't lack traffic rules — it lacks the capacity to enforce them. Officers
can't be everywhere at once, violations happen in seconds, and the city's extensive
CCTV and E-TLE infrastructure is barely used for *automated* detection. The footage
exists; the intelligence layer to act on it does not.

ELANG is that layer. It watches the cameras that are already there, flags violations
with timestamped evidence, maps where and when discipline breaks down, and packages
everything in a format the existing E-TLE ticketing pipeline can ingest — complementing
Dishub's tools rather than replacing them.

---

## What it does

A single Streamlit app, four modules:

| Module | What it does |
|--------|--------------|
| 📷 **Image** | YOLOv8 detects and classifies vehicles in a frame. Optional ANPR reads each plate, validates it against the Indonesian TNKB format (23 region codes, 4 plate types), and reports per-batch accuracy honestly. |
| 🎞️ **Video** | Same detector across frames, with optional DeepSORT tracking. Define a restricted-zone polygon and ELANG counts how long each vehicle dwells inside it, flagging those over a threshold as violators. Works on uploaded clips, RTSP streams, or webcam. |
| 🗺️ **Heatmap** | Aggregates violations onto a spatial grid, renders a Folium hotspot map, and ranks candidate officer/camera placements by proximity to hotspots. Exports evidence as **E-TLE-compatible JSON/CSV**. |
| 💬 **CRM** | Classifies free-text citizen reports (the kind filed via *kanal 112* / JAKI) into six violation categories using multilingual Sentence-BERT, and assigns an urgency level. |

Together these mirror the three-layer architecture in the proposal:

```
  Detection layer        Intelligence layer        Decision-support layer
  ───────────────        ──────────────────        ──────────────────────
  YOLOv8  → vehicles      grid/DBSCAN hotspots       Streamlit dashboard
  PaddleOCR → plates      Sentence-BERT CRM          officer-placement optimizer
  DeepSORT → dwell time   dual-source signal         E-TLE JSON/CSV export
```

---

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_sample.py
streamlit run app.py
```

The first run downloads the YOLOv8-nano weights (~6 MB). The CRM tab pulls the
multilingual MiniLM model (~120 MB) into the HuggingFace cache the first time it runs,
then caches a trained classifier head locally so later calls skip training.

### Try it in two minutes

- **Heatmap** — upload `data/sample_violations.csv` (included). You'll get a Jakarta
  hotspot map, a ranked placement table, and a downloadable E-TLE export.
- **CRM** — switch to *Batch (CSV)* and upload `data/sample_crm_reports.csv` (included)
  for a categorised table, distribution chart, and urgency breakdown.
- **Image / Video** — drop in any traffic photo or clip. (Sample media is intentionally
  not committed; see *Notes on data* below.)

A click-by-click walkthrough of every tab lives in [`DEMO_GUIDE.md`](DEMO_GUIDE.md), and
a deeper operating/testing reference in [`OPERATING_GUIDE.md`](OPERATING_GUIDE.md).

---

## Optional capabilities

The core app (detection, heatmap, optimizer, E-TLE export, CRM) runs out of the box.
Three features are heavier and ship **commented** in `requirements.txt` — uncomment what
you need:

- **ANPR** — `paddleocr` + `paddlepaddle` (pin both `<3.0` on Windows).
- **DeepSORT tracking** — `deep-sort-realtime`.
- **YouTube stream preset** — `yt-dlp`.

Every optional module is lazy-imported, so the app and the rest of the package stay
usable even when a dependency isn't installed — the feature simply reports as unavailable.

### Windows install notes

Two issues are worth knowing in advance:

1. PaddleOCR 3.x fails on Windows-CPU (`ConvertPirAttribute2RuntimeAttribute not
   implemented`), so the requirements pin Paddle `<3.0`.
2. `torch>=2.10` can fail to load `shm.dll` when `cv2` is imported first, so the ANPR
   module imports torch before PaddleOCR to force the load order.

Expect 1–2 GB of dependencies once torch and PaddlePaddle are installed; a cold install
takes 10–15 minutes.

---

## Testing

```bash
python scripts/smoke_test.py
```

This exercises all six modules against real and synthetic inputs and prints a
`PASS / SKIP / FAIL` summary. Missing optional dependencies report **SKIP** (not a
failure), so the run is CI-safe even on a minimal install — it exits non-zero only on a
genuine fault.

---

## How accurate is it, really

We'd rather be honest than impressive:

- **Vehicle detection** targets **≥75% mAP** across lighting and weather, using YOLOv8
  with CLAHE + bilateral-filter preprocessing for night and rain.
- **Plate recognition (ANPR)** is realistically **~85%** on daytime CCTV with the plate
  facing the camera, dropping to **~65%** under adverse conditions (night, rain, sharp
  angle, motorcycle plates). The Image tab surfaces this number per batch, so reviewers
  see the same figure we do.
- Reads below the confidence threshold are queued for **manual review**, not silently
  accepted.

This POC runs YOLOv8-**nano** on CPU for portability; production deployment targets
YOLOv8-L / YOLOv11 on GPU. Angkot detection is on the roadmap via Indonesian-specific
fine-tuning — the current model covers car, motorcycle, bus, and truck from COCO weights.

---

## Tech stack

| Concern | Choice | Why |
|---------|--------|-----|
| Detection | YOLOv8 / YOLOv11 | Best speed/accuracy balance for real-time multi-feed video |
| Plate OCR | PaddleOCR (two-stage) | Strong on Asian plate formats; active ecosystem |
| Tracking | DeepSORT | Stable IDs across frames for dwell-time measurement |
| Hotspots | Grid aggregation (DBSCAN on roadmap) | Turns isolated events into actionable zones |
| Report NLP | Sentence-BERT multilingual + LogReg | Handles Bahasa Indonesia + English |
| UI / geo | Streamlit + Folium | Fast to build, interactive maps |

---

## Project layout

```
elang-prototype/
├── app.py                     Streamlit entry point (four tabs + demo recorder)
├── elang/
│   ├── detection.py           YOLOv8 wrapper
│   ├── classes.py             COCO → local label mapping
│   ├── stats.py               detection aggregation helpers
│   └── stubs/
│       ├── anpr.py            PaddleOCR + CLAHE/Otsu preprocessing + TNKB validation
│       ├── tracking.py        DeepSORT + point-in-polygon zone scoring
│       ├── heatmap.py         Folium grid hotspot aggregation
│       ├── officer_optimizer.py   haversine-proximity placement scoring
│       ├── crm_classifier.py  multilingual Sentence-BERT + logistic regression
│       └── etle.py            JSON/CSV export in the POLRI E-TLE envelope
├── scripts/
│   ├── download_sample.py     fetch a public-domain sample image
│   └── smoke_test.py          end-to-end module checks
├── data/                      sample CSVs (media is gitignored — see below)
├── DEMO_GUIDE.md              per-tab screenshot walkthrough
├── OPERATING_GUIDE.md         function / operate / test reference
└── requirements.txt
```

### Notes on data

Sample **CSVs** for the Heatmap and CRM tabs are committed so reviewers can test
immediately. Sample **images and video are not** — they're gitignored to keep the repo
light and to avoid redistributing third-party footage. Run `scripts/download_sample.py`
for a sample frame, or simply upload your own.

---

## Roadmap

- **Now (POC):** all four modules working and verified end to end on CPU.
- **Pilot (0–6 months):** deploy to 10–15 priority CCTV points; produce the first
  automated hotspot maps and E-TLE evidence batches.
- **Scale (6–18 months):** 50–100 cameras; Indonesian-specific fine-tuning (incl. angkot),
  DBSCAN behavioural clustering, and predictive officer deployment.

---

## License

YOLOv8 (Ultralytics) is **AGPL-3.0**, which carries redistribution obligations. Verify the
competition and licensing terms before publishing or commercialising. Sample media used
during development comes from public sources and should be replaced with properly licensed
footage for any public release.

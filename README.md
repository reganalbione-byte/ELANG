# ELANG — Intelligent Traffic Enforcement (Prototype)

> **Electronic Enforcement & Analysis for Next-Gen traffic Governance**
> AI Open Innovation Challenge 2026 — DISHUB DKI Jakarta, Case 1

Reference architecture and decision rationale:
[`../competition-analysis/DISHUB_Case_Analysis.md`](../competition-analysis/DISHUB_Case_Analysis.md)

---

## What ships in this MVP

✅ **Vehicle detection** — YOLOv8 (n/s/m), per-class counts mapped to
local categories (motor / mobil / bus / truk / sepeda).
✅ **Streamlit demo app** — image / video / heatmap tabs.
✅ **Phase 2 wired in** — ANPR (PaddleOCR), DeepSORT tracking with
in-zone duration scoring, Folium violation heatmap, rule-based
officer placement optimizer.
✅ **Modular package** — `elang/` separates detection, classes, stats;
Phase 2 modules live under `elang/stubs/` (directory kept for layout
continuity) with lazy imports so the MVP runs without their heavy deps.
✅ **CPU-friendly defaults** — `yolov8n.pt`, frame stride, frame cap.

## Module status

| Module | Phase | Target tech | Status |
|---|---|---|---|
| `elang/detection.py` | 1 | YOLOv8 | ✅ working |
| `elang/stubs/heatmap.py` | 2 | Folium | ✅ working (requires `folium`) |
| `elang/stubs/officer_optimizer.py` | 2 | Rule-based | ✅ working (pure Python) |
| `elang/stubs/tracking.py` | 2 | DeepSORT | ✅ wired (requires `deep-sort-realtime`) |
| `elang/stubs/anpr.py` | 2 | PaddleOCR + CLAHE | ✅ wired (requires `paddleocr`, `paddlepaddle`) |
| `elang/stubs/crm_classifier.py` | 3 | Sentence-BERT | ⏳ stub |

Optional Phase 2 deps are commented in `requirements.txt`; uncomment
the ones you want to enable in the demo.

---

## Quick start

```powershell
# 1. Install deps (Python 3.10+ recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. (Optional) download a sample image
python scripts/download_sample.py

# 3. Launch the demo
streamlit run app.py
```

First run downloads `yolov8n.pt` (~6 MB) into the working directory.
Subsequent runs are instant.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Project layout

```
elang-prototype/
├── app.py                       # Streamlit entry point
├── elang/
│   ├── detection.py             # YOLOv8 wrapper (WORKING)
│   ├── classes.py               # COCO -> local label mapping (WORKING)
│   ├── stats.py                 # aggregation helpers (WORKING)
│   └── stubs/                   # Phase 2 / 3 placeholders
│       ├── anpr.py
│       ├── tracking.py
│       ├── heatmap.py
│       ├── crm_classifier.py
│       └── officer_optimizer.py
├── scripts/
│   └── download_sample.py
├── data/                        # gitignored; demo media lives here
├── requirements.txt
└── README.md
```

---

## Roadmap

Priority tiers from `DISHUB_Case_Analysis.md`:

### MUST HAVE (80% effort) — pilot scope
- [x] Vehicle detection + classification (motor, mobil, bus, truk)
- [x] ANPR — plate number recognition (PaddleOCR + CLAHE preprocessing)
- [x] Duration tracking — time in restricted zone (DeepSORT + point-in-polygon)
- [x] Violation heatmap dashboard (Folium + grid hot-zone aggregation)

### SHOULD HAVE (15% effort)
- [ ] CRM report auto-classification (Sentence-BERT)
- [x] Officer / camera placement simulator (rule-based scoring, haversine proximity)

### NICE TO HAVE (5% effort → roadmap doc)
- [ ] Citizen engagement app concept
- [ ] Multi-city scalability vision
- [ ] Predictive violation forecasting

---

## Notes for the demo

- **Adverse conditions** — CLAHE preprocessing (Phase 2 ANPR stub) is the
  designated mitigation for night / rain footage. Until then, report
  accuracy honestly: 85% ideal / 65% adverse, per the analysis.
- **Demo backup** — never rely on live inference at presentation; record
  a pre-rendered run to MP4 as fallback.
- **GPU** — swap `yolov8n.pt` for `yolov8m.pt` or `yolov8l.pt` when a
  CUDA device is available (4–8× accuracy gain on small vehicles).

---

## License

TBD before public submission. The competition rules require all
third-party assets to be license-compatible — verify YOLOv8 (AGPL-3.0)
implications for your submission category before publishing.

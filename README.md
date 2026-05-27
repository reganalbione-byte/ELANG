# ELANG — Intelligent Traffic Enforcement (Prototype)

> **Electronic Enforcement & Analysis for Next-Gen traffic Governance**
> AI Open Innovation Challenge 2026 — DISHUB DKI Jakarta, Case 1

Reference architecture and decision rationale:
[`../competition-analysis/DISHUB_Case_Analysis.md`](../competition-analysis/DISHUB_Case_Analysis.md)

---

## What ships in this MVP

✅ **Vehicle detection** — YOLOv8 (n/s/m), per-class counts mapped to
local categories (motor / mobil / bus / truk / sepeda).
✅ **Streamlit demo app** — upload image or short clip, see annotated
output, per-class stats, per-frame timeline.
✅ **Modular package** — `elang/` separates detection, classes, stats
from Phase 2/3 stubs.
✅ **CPU-friendly defaults** — `yolov8n.pt`, frame stride, frame cap.

## What's stubbed (Phase 2 / Phase 3)

These modules have defined APIs and `NotImplementedError` bodies, so the
ELANG architecture surface is visible but no broken claims are made.

| Stub | Phase | Target tech | Trigger |
|---|---|---|---|
| `elang/stubs/anpr.py` | 2 | PaddleOCR + CLAHE | Indonesian plate sample set |
| `elang/stubs/tracking.py` | 2 | DeepSORT (deep-sort-realtime) | Duration-in-zone enforcement |
| `elang/stubs/heatmap.py` | 2 | Folium / Kepler.gl | Geo-tagged violation log |
| `elang/stubs/crm_classifier.py` | 3 | Sentence-BERT + classifier | Labeled CRM corpus |
| `elang/stubs/officer_optimizer.py` | 2 | Rule-based scoring | Heatmap output + coverage map |

Unlock by uncommenting the relevant lines in `requirements.txt` and
implementing the function bodies.

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
- [ ] ANPR — plate number recognition (PaddleOCR)
- [ ] Duration tracking — time in restricted zone (DeepSORT)
- [ ] Violation heatmap dashboard (Folium / Kepler.gl)

### SHOULD HAVE (15% effort)
- [ ] CRM report auto-classification (Sentence-BERT)
- [ ] Officer / camera placement simulator (rule-based scoring)

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

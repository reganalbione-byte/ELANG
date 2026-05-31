# ELANG

Traffic enforcement prototype for DKI Jakarta. Built for the AI Open Innovation Challenge 2026 (DISHUB Case 1).

Detects vehicles in CCTV footage, reads plates, tracks how long they sit in restricted zones, classifies citizen complaint text, and packages violations in a format the POLRI E-TLE system can ingest.

## What it does

A Streamlit app with four tabs.

**Image tab.** Upload a single frame. YOLOv8 marks every vehicle. If PaddleOCR is installed, it also reads plates on each crop, validates them against the Indonesian TNKB format (23 region codes covering Java and Bali), and labels each read as valid, plausible but off-format, or likely false positive. An accuracy report at the bottom tallies how many of N detections produced a valid plate.

**Video tab.** Same pipeline, three input modes:

1. Upload File. Standard MP4 / AVI / MOV / MKV.
2. RTSP Stream. Paste a camera URL or pick the YouTube preset if you have `yt-dlp` installed. Comes with a warning that this needs network access to the camera.
3. Webcam. Picks a local device index (0, 1, or 2).

Optional DeepSORT tracking keeps stable IDs across frames and counts how long each track sits inside a polygon you define. Tracks that exceed a frame threshold are flagged as violators. Optional ANPR runs on every detection and produces a deduplicated "Unique Plates Detected" table at the end.

**Heatmap tab.** Feed a CSV of `(lat, lon, hour, violation_type, count)` rows. The app aggregates them on a configurable grid (defaults to about 111m cells), renders a Folium heatmap with the top hotspots circled, and scores candidate camera placements by haversine proximity to those hotspots. Below the optimizer is the E-TLE export section: generates mock violations and offers JSON or CSV download in the POLRI envelope format.

**CRM Classifier tab.** Routes free-text citizen reports (the kind that come in through kanal 112 or JAKI) into six violation categories: `parkir_liar`, `pelanggaran_lampu_merah`, `kendaraan_kontrarus`, `trotoar_dipakai_kendaraan`, `jalur_busway_dilanggar`, `lain_lain`. Uses `paraphrase-multilingual-MiniLM-L12-v2` (Indonesian and English) plus a logistic regression head trained on roughly 54 seed examples. Adds an urgency label based on keyword heuristics (accident, collision, severe gridlock get marked high). Supports single text input and CSV batch.

A sidebar control records the annotated frames to MP4 as a backup. If the live demo flakes during a presentation, you have the recording.

## Running it

Needs Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_sample.py
streamlit run app.py
```

First run downloads `yolov8n.pt` (about 6 MB) into the project root. The CRM tab downloads multilingual MiniLM weights (around 120 MB) into the HuggingFace cache on first use, then caches the trained classifier head to `elang/stubs/crm_model.pkl` so subsequent calls skip the fit step.

Smoke tests before a demo:

```
python scripts/smoke_test.py
```

Exercises detection, heatmap, officer placement, tracking (if `deep-sort-realtime` is installed), ANPR (if PaddleOCR is installed), and the CRM classifier (if `sentence-transformers` is installed) against the sample image. Each check reports PASS / FAIL / SKIP; missing optional deps report SKIP and do **not** fail the run (exit code stays 0). The process exits non-zero only on a real FAIL/ERROR.

For a full hands-on walkthrough — what each module does, how to operate every tab, and the pre-demo test checklist — see `OPERATING_GUIDE.md`.

## Optional dependencies

Some modules need extra packages. They are commented in `requirements.txt`. Uncomment what you want.

* `deep-sort-realtime` for DeepSORT tracking in the Video tab.
* `paddleocr` plus `paddlepaddle` for plate reading. Pin both to `<3.0` on Windows.
* `yt-dlp` only if you want the YouTube preset to work in the RTSP tab.

Phase 3 packages (`sentence-transformers`, `scikit-learn`) ship uncommented so the CRM Classifier tab works without extra setup.

### Windows install notes

Two things broke during development and are worth knowing.

1. PaddleOCR 3.x fails on Windows CPU with `ConvertPirAttribute2RuntimeAttribute not implemented`. The requirements file pins paddle to `<3.0`.
2. `torch>=2.10` fails to load `shm.dll` if `cv2` is imported first. The ANPR module imports torch before paddleocr to force the right load order.

Expect 1 to 2 GB of dependencies once torch and paddlepaddle are both installed. Cold install takes 10 to 15 minutes.

## Project layout

```
elang-prototype/
  app.py                Streamlit entry point. Four tabs plus sidebar recording.
  elang/
    detection.py        YOLOv8 wrapper.
    classes.py          COCO ID to local label mapping.
    stats.py            Aggregation helpers.
    stubs/
      anpr.py           PaddleOCR plus CLAHE / Otsu preprocessing plus plate validation.
      tracking.py       DeepSORT plus point-in-polygon zone scoring.
      heatmap.py        Folium grid hot zone aggregation.
      officer_optimizer.py   Rule-based candidate scoring with haversine distance.
      crm_classifier.py Multilingual Sentence-BERT plus logistic regression.
      etle.py           JSON and CSV export adapter in POLRI envelope format.
  scripts/
    download_sample.py  Pull a public-domain sample image into data/.
    smoke_test.py       End-to-end module smoke checks.
  data/                 Sample media and saved demo recordings. Gitignored.
  requirements.txt
  pyrightconfig.json    Points Pylance at .venv.
  README.md             Architecture + install notes (prose).
  OPERATING_GUIDE.md    Function / how to operate / how to test (hands-on).
```

## ANPR accuracy

The plate reader is honest about its limits.

Default preprocessing is CLAHE plus a bilateral filter. This works fine on whole vehicle crops at typical CCTV resolution. For tight plate ROIs where you already cropped to the plate, switch to Otsu mode in the dropdown. Otsu upscales small crops to at least 200 px wide, then thresholds and runs morphological open / close to clean up.

Validation checks the OCR output against 23 Indonesian region codes and four plate types:

* `reguler`, the standard civilian format like `B 1234 XYZ`.
* `TNI_POLRI`, prefix `RI` followed by digits.
* `dinas`, prefix `CD` or `CC` for diplomatic and consular vehicles.
* `sementara`, anything plausible by shape but outside the strict format.

Realistic accuracy target: about 85 percent on daytime CCTV with the plate facing the camera, dropping to around 65 percent on adverse conditions (night, rain, sharp angle, motorcycle plates). The Image tab surfaces this number per batch so reviewers see the same figure you do.

## E-TLE envelope

JSON output:

```
{
  "version": "1.0",
  "source": "ELANG",
  "export_timestamp": "ISO 8601 UTC",
  "total_records": 10,
  "violations": [
    {
      "violation_id": "uuid",
      "timestamp": "ISO 8601",
      "plate_number": "B 1234 XYZ",
      "vehicle_class": "motor | mobil | bus | truk",
      "violation_type": "parkir_liar | jalur_busway | berhenti_sembarangan",
      "location_lat": -6.2088,
      "location_lon": 106.8456,
      "camera_id": "ETLE-JKT-001",
      "duration_seconds": 120,
      "confidence_score": 0.92,
      "evidence_frame_path": "evidence/ETLE-JKT-001/uuid.jpg",
      "status": "pending"
    }
  ]
}
```

CSV has one row per violation with the same field names as header.

In production the violation list comes from the track violator output in the Video tab combined with the ANPR plate reads. The Heatmap tab currently uses the mock generator for demo purposes.

## Demo tips

RTSP streams need network access to the camera. Venues are unpredictable. Test the URL beforehand and have a recorded backup.

Before going on stage, hit Start Recording in the sidebar, run the Video tab against your best clip, then Stop and Save. The MP4 lives in `data/demo_recording_YYYYMMDD_HHMMSS.mp4` at 15 fps with a 500 frame cap. If the live demo fails, switch to Upload File mode and play the recorded file.

The Heatmap tab has an inline sample CSV so it works without any input at all, which is useful when the network is down.

GPU helps a lot. Swap `yolov8n.pt` for `yolov8m.pt` or `yolov8l.pt` if you have CUDA. Small vehicle accuracy goes up roughly 4x to 8x.

## License

To be decided before submission. YOLOv8 is AGPL-3.0, which has implications for redistribution. Verify the competition rules before publishing.

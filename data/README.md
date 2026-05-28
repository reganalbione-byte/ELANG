# Sample data

Media files in this folder are gitignored (see root `.gitignore`).

## Quick start

```powershell
python scripts/download_sample.py
```

Downloads a public-domain sample image (`bus.jpg`, ~134 KB) for smoke testing.
Re-running `scripts/smoke_test.py` after this exercises detection + ANPR
against the saved file.

## For real demos

Replace with Jakarta CCTV / E-TLE footage. Recommended sources:

- DISHUB DKI Jakarta public CCTV streams (verify usage rights for competition)
- Open Images / KITTI / Cityscapes for non-Jakarta calibration
- Your own dashcam footage

Per the case analysis, adverse-condition (night / rain) samples are
critical for honest accuracy reporting (85% ideal / 65% adverse).

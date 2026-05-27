"""Download a small public-domain traffic sample for demo.

Uses an ultralytics-hosted sample image (Apache-2.0 licensed).
If the download fails, the app still works — just upload your own footage.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Ultralytics ships sample assets under the AGPL/Apache license.
# bus.jpg is bundled with the YOLO model release.
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = DATA_DIR / "sample.jpg"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_PATH.exists():
        print(f"Already exists: {SAMPLE_PATH}")
        return 0
    print(f"Downloading {SAMPLE_URL} -> {SAMPLE_PATH}")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("You can still use the app by uploading your own image/video.",
              file=sys.stderr)
        return 1
    print(f"Saved: {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

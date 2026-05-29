"""ANPR (Automatic Number Plate Recognition).

    - PaddleOCR (multi-language; Latin chars cover Indonesian plates)
    - Single-pass detect+recognise on the vehicle crop, then filter
      candidates against the Indonesian plate regex.
    - Two preprocessing modes:
        * CLAHE + bilateral denoising  → 'clahe' (default for full vehicle crops)
        * Upscale + Otsu + morphology  → 'otsu'  (preferred for tight plate ROIs)
    - Confidence threshold 0.8 to avoid false positives.
    - Expect 85% accuracy ideal / 65% adverse on real Jakarta footage —
      report honestly.

PaddleOCR is imported lazily so the module is inspectable without
the dep installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlateRead:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


PLATE_REGEX = re.compile(r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$")
PLATE_REGEX_RELAXED = re.compile(r"^[A-Z0-9]{4,9}$")
_DEFAULT_MIN_CONF = 0.8

# TNKB area codes — Java + Bali subset (covers all Jakarta-bound enforcement
# traffic). Sources: PP 80/2012, Korlantas POLRI plate gazette.
INDONESIAN_REGION_CODES: dict[str, str] = {
    "B": "DKI Jakarta / Tangerang / Bekasi / Depok",
    "D": "Bandung / Cimahi",
    "E": "Cirebon / Indramayu / Majalengka / Kuningan",
    "F": "Bogor / Sukabumi / Cianjur",
    "T": "Subang / Purwakarta / Karawang",
    "Z": "Garut / Sumedang / Tasikmalaya / Ciamis",
    "A": "Banten (Serang / Cilegon / Pandeglang)",
    "G": "Pekalongan / Tegal / Brebes / Batang / Pemalang",
    "H": "Semarang / Salatiga / Demak / Kendal",
    "K": "Pati / Kudus / Jepara / Rembang / Blora",
    "R": "Banyumas / Cilacap / Purbalingga / Banjarnegara",
    "AA": "Magelang / Temanggung / Wonosobo / Kebumen",
    "AB": "DI Yogyakarta",
    "AD": "Surakarta / Sukoharjo / Boyolali / Sragen / Klaten",
    "AE": "Madiun / Magetan / Ngawi / Ponorogo / Pacitan",
    "AG": "Kediri / Blitar / Tulungagung / Trenggalek / Nganjuk",
    "L": "Surabaya",
    "M": "Madura (Bangkalan / Sampang / Pamekasan / Sumenep)",
    "N": "Malang / Pasuruan / Probolinggo / Lumajang",
    "P": "Jember / Banyuwangi / Bondowoso / Situbondo",
    "S": "Bojonegoro / Mojokerto / Jombang / Lamongan / Tuban",
    "W": "Sidoarjo / Gresik",
    "DK": "Bali",
}

_PLATE_RI_REGEX = re.compile(r"^RI\s?(\d{1,3})$")
_PLATE_DINAS_REGEX = re.compile(r"^(CD|CC)\s?(\d{1,4})$")
_PLATE_REGULAR_REGEX = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$")

_ocr_singleton = None


def _get_ocr():
    global _ocr_singleton
    if _ocr_singleton is not None:
        return _ocr_singleton
    try:
        # On Windows, torch must be imported before cv2 to avoid an
        # shm.dll DLL-search conflict. paddleocr pulls torch via
        # albumentations, so we force the order here.
        import torch  # noqa: F401
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "paddleocr is required for ANPR; "
            "uncomment paddleocr + paddlepaddle in requirements.txt and pip install."
        ) from e
    _ocr_singleton = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_singleton


def preprocess_adverse(image: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral filter — boosts contrast and denoises night/rain crops."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        boosted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        boosted = clahe.apply(image)
    return cv2.bilateralFilter(boosted, d=7, sigmaColor=50, sigmaSpace=50)


def preprocess_plate_roi(image: np.ndarray, target_min_width: int = 200) -> np.ndarray:
    """Aggressive preprocessing for tight plate ROIs.

    Use when the input crop is already known to be a plate (not a whole
    vehicle): upscale small crops so PaddleOCR's detector sees enough pixels,
    apply Otsu thresholding to maximise character/background contrast, then
    morphologically open+close to remove specks and bridge broken strokes.

    Returns a 3-channel BGR image so the result is drop-in for read_plate().
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    if w < target_min_width and w > 0:
        scale = target_min_width / w
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _normalise(raw: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()


def _is_plausible_plate(text: str) -> bool:
    if not text:
        return False
    if PLATE_REGEX.match(text):
        return True
    compact = text.replace(" ", "")
    if not PLATE_REGEX_RELAXED.match(compact):
        return False
    # Indonesian plates always contain BOTH letters and digits.
    has_alpha = any(c.isalpha() for c in compact)
    has_digit = any(c.isdigit() for c in compact)
    return has_alpha and has_digit


def _format_plate(text: str) -> str:
    """Normalise OCR output to canonical 'B 1234 XYZ' / 'RI 1' / 'CD 12' shape."""
    text = (text or "").strip().upper()
    if not text:
        return ""
    compact = re.sub(r"[^A-Z0-9]", "", text)
    if not compact:
        return text

    m = _PLATE_RI_REGEX.match(compact)
    if m:
        return f"RI {m.group(1)}"

    m = _PLATE_DINAS_REGEX.match(compact)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Greedy "longest valid region code" — try two-letter, then one-letter.
    for prefix_len in (2, 1):
        if len(compact) > prefix_len:
            prefix = compact[:prefix_len]
            rest = compact[prefix_len:]
            m = re.match(r"^(\d{1,4})([A-Z]{0,3})$", rest)
            if m and prefix.isalpha():
                parts = [prefix, m.group(1)]
                if m.group(2):
                    parts.append(m.group(2))
                return " ".join(parts)

    return text


def _detect_plate_type(formatted: str) -> str:
    text = formatted.upper().strip()
    if _PLATE_RI_REGEX.match(re.sub(r"[^A-Z0-9]", "", text)):
        return "TNI_POLRI"
    if _PLATE_DINAS_REGEX.match(re.sub(r"[^A-Z0-9]", "", text)):
        return "dinas"
    compact = re.sub(r"[^A-Z0-9]", "", text)
    if _PLATE_REGULAR_REGEX.match(compact):
        return "reguler"
    if _is_plausible_plate(text):
        return "sementara"
    return "unknown"


def validate_indonesian_plate(text: str) -> dict:
    """Classify OCR output against the Indonesian plate conventions.

    Returns a dict with:
        is_valid    True iff the text matches a recognised plate shape
                    (reguler / TNI-POLRI / dinas / sementara). False means
                    OCR almost certainly mis-read.
        region_code Two-letter or one-letter prefix (e.g. 'B', 'AD') when
                    the plate is reguler, else None.
        plate_type  'reguler' / 'TNI_POLRI' / 'dinas' / 'sementara' / 'unknown'.
        formatted   Canonical 'B 1234 XYZ' shape, or the trimmed input if
                    no canonical form could be derived.
    """
    if not text:
        return {
            "is_valid": False,
            "region_code": None,
            "plate_type": "unknown",
            "formatted": "",
        }

    formatted = _format_plate(text)
    plate_type = _detect_plate_type(formatted)

    region_code: str | None = None
    if plate_type == "reguler":
        m = _PLATE_REGULAR_REGEX.match(re.sub(r"[^A-Z0-9]", "", formatted))
        if m:
            region_code = m.group(1)

    is_valid = plate_type in {"reguler", "TNI_POLRI", "dinas", "sementara"}

    return {
        "is_valid": is_valid,
        "region_code": region_code,
        "plate_type": plate_type,
        "formatted": formatted,
    }


def read_plate(
    plate_crop: np.ndarray,
    min_confidence: float = _DEFAULT_MIN_CONF,
    enhance: bool = True,
    enhance_mode: str = "auto",
) -> PlateRead | None:
    """Read characters from a vehicle / plate crop.

    Args:
        plate_crop: BGR image — either a tight plate ROI or a vehicle crop.
            When given a vehicle crop, PaddleOCR's built-in detector localises
            the plate; otherwise it just recognises.
        min_confidence: drop reads below this score.
        enhance: apply preprocessing before OCR. False = pass crop through raw.
        enhance_mode: which preprocessing pipeline to apply when enhance=True.
            - 'auto'  : CLAHE + bilateral (current default, safe for any crop)
            - 'clahe' : same as auto
            - 'otsu'  : upscale + Otsu + morphology (use for tight plate ROIs)
            - 'none'  : skip preprocessing (same effect as enhance=False)

    Returns:
        PlateRead with the highest-confidence plausible Indonesian plate,
        or None if nothing clears the threshold / regex.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    ocr = _get_ocr()

    if not enhance or enhance_mode == "none":
        image = plate_crop
    elif enhance_mode == "otsu":
        image = preprocess_plate_roi(plate_crop)
    else:  # 'auto' / 'clahe' / unknown → safe default
        image = preprocess_adverse(plate_crop)

    result = ocr.ocr(image, cls=True)

    if not result or not result[0]:
        return None

    best: PlateRead | None = None
    for line in result[0]:
        box, (raw_text, conf) = line
        text = _normalise(raw_text)
        if conf < min_confidence or not _is_plausible_plate(text):
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        if best is None or conf > best.confidence:
            best = PlateRead(text=text, confidence=float(conf), bbox=bbox)

    return best


def read_plate_two_stage(
    vehicle_crop: np.ndarray,
    plate_detector_model=None,
    min_confidence: float = _DEFAULT_MIN_CONF,
    enhance: bool = True,
    enhance_mode: str = "auto",
) -> PlateRead | None:
    """Two-stage plate read: detect plate ROI first, then OCR.

    Args:
        vehicle_crop: BGR image of a whole vehicle.
        plate_detector_model: ultralytics-style detector with .predict() that
            returns boxes for plate regions. None → fall back to single-stage
            read_plate() on the vehicle crop directly.
        min_confidence / enhance / enhance_mode: forwarded to read_plate().

    Behaviour: if the detector fails or returns no boxes, we fall through
    to single-stage rather than raising — callers get the same contract as
    read_plate(). When the detector does return a plate box, the tight crop
    is OCR'd with enhance_mode='otsu' by default (best for plate-ROI input)
    unless the caller forces a different mode.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None

    if plate_detector_model is None:
        return read_plate(
            vehicle_crop,
            min_confidence=min_confidence,
            enhance=enhance,
            enhance_mode=enhance_mode,
        )

    try:
        results = plate_detector_model.predict(vehicle_crop, verbose=False)
    except Exception:
        return read_plate(
            vehicle_crop,
            min_confidence=min_confidence,
            enhance=enhance,
            enhance_mode=enhance_mode,
        )

    if not results:
        return read_plate(vehicle_crop, min_confidence=min_confidence,
                          enhance=enhance, enhance_mode=enhance_mode)

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return read_plate(vehicle_crop, min_confidence=min_confidence,
                          enhance=enhance, enhance_mode=enhance_mode)

    try:
        conf = boxes.conf
        xyxy = boxes.xyxy
        best_idx = int(conf.argmax())
        x1, y1, x2, y2 = (int(v) for v in xyxy[best_idx])
    except Exception:
        return read_plate(vehicle_crop, min_confidence=min_confidence,
                          enhance=enhance, enhance_mode=enhance_mode)

    plate_crop = vehicle_crop[max(0, y1):y2, max(0, x1):x2]
    if plate_crop.size == 0:
        return read_plate(vehicle_crop, min_confidence=min_confidence,
                          enhance=enhance, enhance_mode=enhance_mode)

    tight_mode = "otsu" if enhance_mode == "auto" else enhance_mode
    return read_plate(
        plate_crop,
        min_confidence=min_confidence,
        enhance=enhance,
        enhance_mode=tight_mode,
    )

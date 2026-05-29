"""CRM citizen report classifier — Phase 3.

    - Sentence-BERT multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2,
      ~120MB, ID + EN support)
    - LogisticRegression head over embeddings, trained on seed examples of
      common Jakarta citizen reports
    - Categories cover the 5 most-routed violation types + 'lain_lain'
      for non-routable reports (infrastructure complaints etc.)
    - Urgency: keyword heuristic (accidents / collisions / severe gridlock → high,
      'lain_lain' → low, default → medium)

sentence-transformers + scikit-learn are imported lazily so the module
stays inspectable without the deps installed. The trained classifier is
pickled next to this file after first training; subsequent calls reload
from cache and skip the fit step.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = [
    "parkir_liar",
    "pelanggaran_lampu_merah",
    "kendaraan_kontrarus",
    "trotoar_dipakai_kendaraan",
    "jalur_busway_dilanggar",
    "lain_lain",
]

_HIGH_URGENCY_KEYWORDS = (
    "kecelakaan", "tabrak", "tabrakan", "macet parah", "korban",
    "luka", "darurat", "fatal", "terluka", "meninggal",
)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_CACHE_PATH = Path(__file__).with_name("crm_model.pkl")

# Realistic Indonesian-language seed examples modelled on 112 / JAKI reports.
SEED_DATA: dict[str, list[str]] = {
    "parkir_liar": [
        "Ada motor parkir di atas trotoar depan Indomaret Sudirman",
        "Mobil parkir sembarangan di Jalan Thamrin menghalangi jalur lambat",
        "Banyak motor parkir liar di bahu jalan Kemang Raya",
        "Mohon ditertibkan parkir liar di depan Stasiun Manggarai",
        "Truk parkir di pinggir Jalan Gatot Subroto bikin susah lewat",
        "Mobil parkir sembarangan di depan rumah saya menutupi gerbang",
        "Parkir liar di sekitar Pasar Senen sangat mengganggu pejalan kaki",
        "Bus pariwisata parkir di pinggir Jalan Wahid Hasyim mengambil satu lajur",
        "Banyak mobil parkir di pinggir Jalan Sabang padahal ada rambu dilarang parkir",
    ],
    "pelanggaran_lampu_merah": [
        "Banyak motor terobos lampu merah di perempatan Coca Cola Cempaka Putih",
        "Mobil sering menerobos lampu merah di simpang Tomang",
        "Pelanggaran lampu merah marak di perempatan Slipi Jaya",
        "Motor terobos lampu merah jam 10 malam di Mampang",
        "Tolong pasang ETLE di lampu merah Cawang, banyak yang nerobos",
        "Angkot menerobos lampu merah di dekat Halte Sarinah",
        "Pengendara motor nerobos lampu merah di perempatan Pancoran",
        "Banyak kendaraan terobos lampu merah saat tidak ada petugas di Senayan",
        "Lampu merah di simpang Grogol sering dilanggar pengendara motor pagi hari",
    ],
    "kendaraan_kontrarus": [
        "Banyak motor lawan arah di Jalan Casablanca menuju Kuningan",
        "Mobil lewat lawan arus di Jalan Sudirman tadi pagi",
        "Motor lawan arah di Jalan Wahid Hasyim sangat berbahaya",
        "Pengendara motor sering menyeberang lawan arus di Mampang Prapatan",
        "Mobil colt diesel jalan lawan arah di Bekasi Barat",
        "Banyak ojek online lawan arus di sekitar Pasar Tanah Abang",
        "Motor lawan arah di flyover Kuningan rawan tabrakan",
        "Pengendara motor lawan arus di Jalan Salemba Raya bikin kacau lalu lintas",
        "Sepeda motor sering melawan arus di Jalan Tebet Raya saat jam sibuk",
    ],
    "trotoar_dipakai_kendaraan": [
        "Motor naik trotoar di depan Stasiun Sudirman, pejalan kaki kesulitan jalan",
        "Banyak motor pakai trotoar untuk hindari macet di Thamrin",
        "Trotoar Tanah Abang dipakai motor seenaknya",
        "Motor melaju kencang di trotoar Jalan Casablanca",
        "Pejalan kaki tidak aman karena motor naik trotoar di Senen",
        "Trotoar di depan Plaza Indonesia dipakai motor untuk lewat",
        "Sepeda motor melaju di trotoar Jalan Kebon Sirih",
        "Tukang ojek pangkalan menaikkan motor ke trotoar Kuningan",
        "Motor lewat trotoar di Jalan Salemba sangat membahayakan anak sekolah",
    ],
    "jalur_busway_dilanggar": [
        "Mobil pribadi masuk jalur busway di Jalan Sudirman jam sibuk",
        "Banyak motor masuk jalur busway koridor 1",
        "Sepeda motor masuk jalur Transjakarta di Mampang setiap pagi",
        "Mobil dinas masuk jalur busway di koridor Blok M - Kota",
        "Ambulans bukan keadaan darurat lewat jalur busway di Sudirman",
        "Truk besar pakai jalur busway di koridor Pulogadung - Harmoni",
        "Banyak pelanggar jalur busway di koridor Cawang - Tendean",
        "Mobil mewah masuk jalur busway di Thamrin sore tadi",
        "Pengendara motor masuk jalur Transjakarta di koridor Pinang Ranti",
    ],
    "lain_lain": [
        "Ada lampu jalan mati di Jalan Kemang Raya",
        "Marka jalan sudah pudar di Jalan Casablanca, perlu dicat ulang",
        "Trotoar rusak di sekitar Stasiun Cikini",
        "Sampah menumpuk di taman dekat halte Transjakarta",
        "Saluran air mampet di Jalan Sabang",
        "Pohon tumbang di Jalan Senopati tadi pagi",
        "PJU mati di Jalan Letjen Suprapto",
        "Banjir tipis di kawasan Kemang Raya tadi malam",
        "Rambu lalu lintas roboh di Jalan Sudirman setelah hujan deras",
    ],
}


@dataclass
class CRMReport:
    text: str
    predicted_category: str
    confidence: float
    urgency: str   # low / medium / high


def _seed_fingerprint() -> str:
    flat = "|".join(
        f"{cat}::{ex}"
        for cat in CATEGORIES
        for ex in SEED_DATA.get(cat, [])
    )
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()[:16]


_sbert_singleton = None


def _get_sbert():
    global _sbert_singleton
    if _sbert_singleton is not None:
        return _sbert_singleton
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for CRM classifier; "
            "uncomment sentence-transformers in requirements.txt and pip install."
        ) from e
    _sbert_singleton = SentenceTransformer(_MODEL_NAME)
    return _sbert_singleton


_classifier_cache: dict | None = None


def _train_classifier() -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required for CRM classifier; "
            "uncomment scikit-learn in requirements.txt and pip install."
        ) from e

    sbert = _get_sbert()
    texts: list[str] = []
    labels: list[str] = []
    for cat in CATEGORIES:
        for ex in SEED_DATA.get(cat, []):
            texts.append(ex)
            labels.append(cat)

    embeddings = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(embeddings, labels)
    return {
        "classifier": clf,
        "categories": list(clf.classes_),
        "seed_fingerprint": _seed_fingerprint(),
        "model_name": _MODEL_NAME,
    }


def _load_or_train() -> dict:
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache

    fingerprint = _seed_fingerprint()
    if _CACHE_PATH.exists():
        try:
            with _CACHE_PATH.open("rb") as f:
                cached = pickle.load(f)
            if (
                cached.get("seed_fingerprint") == fingerprint
                and cached.get("model_name") == _MODEL_NAME
            ):
                _classifier_cache = cached
                return cached
        except Exception:
            pass

    trained = _train_classifier()
    try:
        with _CACHE_PATH.open("wb") as f:
            pickle.dump(trained, f)
    except OSError:
        pass
    _classifier_cache = trained
    return trained


def _urgency_for(text: str, category: str) -> str:
    if category == "lain_lain":
        return "low"
    lowered = text.lower()
    if any(kw in lowered for kw in _HIGH_URGENCY_KEYWORDS):
        return "high"
    return "medium"


def classify_report(text: str) -> CRMReport:
    """Classify a single citizen report."""
    cleaned = (text or "").strip()
    if not cleaned:
        return CRMReport(text=cleaned, predicted_category="lain_lain",
                         confidence=0.0, urgency="low")

    bundle = _load_or_train()
    sbert = _get_sbert()
    clf = bundle["classifier"]
    classes = bundle["categories"]

    emb = sbert.encode([cleaned], convert_to_numpy=True, show_progress_bar=False)
    probs = clf.predict_proba(emb)[0]
    best_idx = int(probs.argmax())
    category = classes[best_idx]
    confidence = float(probs[best_idx])
    return CRMReport(
        text=cleaned,
        predicted_category=category,
        confidence=confidence,
        urgency=_urgency_for(cleaned, category),
    )


def classify_batch(texts: list[str]) -> list[CRMReport]:
    """Classify many reports in one embedding pass."""
    cleaned = [(t or "").strip() for t in texts]
    if not cleaned:
        return []

    non_empty = [(i, t) for i, t in enumerate(cleaned) if t]
    results: list[CRMReport | None] = [None] * len(cleaned)

    if non_empty:
        bundle = _load_or_train()
        sbert = _get_sbert()
        clf = bundle["classifier"]
        classes = bundle["categories"]

        embeddings = sbert.encode(
            [t for _, t in non_empty],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_probs = clf.predict_proba(embeddings)

        for (orig_idx, txt), probs in zip(non_empty, all_probs):
            best_idx = int(probs.argmax())
            cat = classes[best_idx]
            results[orig_idx] = CRMReport(
                text=txt,
                predicted_category=cat,
                confidence=float(probs[best_idx]),
                urgency=_urgency_for(txt, cat),
            )

    for i, t in enumerate(cleaned):
        if results[i] is None:
            results[i] = CRMReport(
                text=t, predicted_category="lain_lain",
                confidence=0.0, urgency="low",
            )
    return results  # type: ignore[return-value]

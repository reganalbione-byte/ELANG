"""CRM citizen report classifier — Phase 3 stub.

Target implementation (per DISHUB_Case_Analysis.md):
    - Sentence-BERT embeddings (multilingual: ID + EN)
    - Lightweight classifier on top (logistic regression or kNN)
    - Categories: violation type, urgency, location-mentioned
    - Auto-routing to enforcement officers

Activation: install sentence-transformers + scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = [
    "parkir_liar",
    "pelanggaran_lampu_merah",
    "kendaraan_kontrarus",
    "trotoar_dipakai_kendaraan",
    "jalur_busway_dilanggar",
    "lain_lain",
]


@dataclass
class CRMReport:
    text: str
    predicted_category: str
    confidence: float
    urgency: str   # low / medium / high


def classify_report(text: str) -> CRMReport:
    """Classify a single citizen report."""
    raise NotImplementedError("Phase 3: install sentence-transformers.")


def classify_batch(texts: list[str]) -> list[CRMReport]:
    return [classify_report(t) for t in texts]

"""Officer / camera placement optimizer — Phase 2 stub.

Target implementation (per DISHUB_Case_Analysis.md):
    - Rule-based scoring (Phase 2): hotspot density * violation severity
      * (1 - existing officer coverage)
    - Optional Phase 3 upgrade: ILP / greedy facility location

Activation: implement score_locations() against the heatmap output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacementRecommendation:
    lat: float
    lon: float
    score: float
    reason: str


def score_locations(
    candidate_locations: list[tuple[float, float]],
    hot_zones: list[tuple[float, float, int]],
    existing_coverage: dict[tuple[float, float], float] | None = None,
) -> list[PlacementRecommendation]:
    """Score candidate placements; return sorted descending by score."""
    raise NotImplementedError("Phase 2: implement rule-based scoring.")

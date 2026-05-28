"""Officer / camera placement optimizer.

Rule-based scoring (per DISHUB_Case_Analysis.md):

    score(candidate) = sum over hotspots:
        violation_density(hotspot)
        * proximity_weight(candidate, hotspot)
        * (1 - existing_coverage(candidate))

Proximity weight decays with haversine distance using a configurable
radius (default 500 m). Density is the hotspot count normalised by
the max count across the input. Coverage is a 0..1 saturation per
candidate; unspecified candidates default to 0 (no current coverage).

A Phase 3 upgrade (ILP / greedy facility location) can replace this
without changing the function signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PlacementRecommendation:
    lat: float
    lon: float
    score: float
    reason: str


_EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def score_locations(
    candidate_locations: list[tuple[float, float]],
    hot_zones: list[tuple[float, float, int]],
    existing_coverage: dict[tuple[float, float], float] | None = None,
    radius_m: float = 500.0,
) -> list[PlacementRecommendation]:
    """Score candidate placements; return sorted descending by score."""
    if not candidate_locations or not hot_zones:
        return []

    max_count = max(c for _, _, c in hot_zones) or 1
    coverage = existing_coverage or {}

    recommendations: list[PlacementRecommendation] = []
    for cand in candidate_locations:
        score = 0.0
        contributing = 0
        for hz_lat, hz_lon, hz_count in hot_zones:
            distance = _haversine_m(cand, (hz_lat, hz_lon))
            if distance > radius_m * 3:
                continue
            proximity = math.exp(-distance / radius_m)
            density = hz_count / max_count
            score += density * proximity
            if proximity > 0.1:
                contributing += 1

        cov = max(0.0, min(1.0, coverage.get(cand, 0.0)))
        score *= 1.0 - cov

        reason = (
            f"{contributing} nearby hotspot(s) within ~{int(radius_m * 3)} m; "
            f"existing coverage {cov:.0%}"
        )
        recommendations.append(PlacementRecommendation(
            lat=cand[0], lon=cand[1], score=round(score, 4), reason=reason,
        ))

    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations

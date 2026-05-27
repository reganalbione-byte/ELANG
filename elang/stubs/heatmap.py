"""Spatial-temporal violation heatmap — Phase 2 stub.

Target implementation (per DISHUB_Case_Analysis.md):
    - Aggregate violations by (lat, lon, hour-of-day)
    - Render with Folium (simple) or Kepler.gl (rich, 3D)
    - Hot-zone identification feeds officer/camera placement optimizer

Activation: install folium / keplergl, then implement render().
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViolationPoint:
    lat: float
    lon: float
    hour: int
    violation_type: str
    count: int = 1


def render_heatmap(points: list[ViolationPoint], out_html: str) -> None:
    """Render a violation heatmap to an HTML file."""
    raise NotImplementedError("Phase 2: install folium and implement.")


def hot_zones(
    points: list[ViolationPoint],
    top_k: int = 10,
    grid_size: float = 0.001,
) -> list[tuple[float, float, int]]:
    """Return top-K (lat, lon, count) hotspots via grid aggregation."""
    raise NotImplementedError("Phase 2: implement grid clustering.")

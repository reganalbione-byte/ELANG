"""Spatial-temporal violation heatmap.

Implementation:
    - hot_zones(): grid clustering on (lat, lon) at configurable cell size,
      returns top-K cells by violation count weighted by ViolationPoint.count.
    - render_heatmap(): Folium HeatMap to a self-contained HTML file.

Folium is imported lazily so the rest of the package stays usable
without the dep installed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ViolationPoint:
    lat: float
    lon: float
    hour: int
    violation_type: str
    count: int = 1


def hot_zones(
    points: list[ViolationPoint],
    top_k: int = 10,
    grid_size: float = 0.001,
) -> list[tuple[float, float, int]]:
    """Return top-K (lat, lon, count) hotspots via grid aggregation.

    `grid_size` is in degrees; 0.001 ≈ 111 m at the equator.
    Each output coord is the centre of its grid cell.
    """
    if not points:
        return []
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    bins: dict[tuple[int, int], int] = defaultdict(int)
    for p in points:
        key = (int(p.lat // grid_size), int(p.lon // grid_size))
        bins[key] += max(1, int(p.count))

    ranked = sorted(bins.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    half = grid_size / 2.0
    return [
        (gx * grid_size + half, gy * grid_size + half, count)
        for (gx, gy), count in ranked
    ]


def render_heatmap(
    points: list[ViolationPoint],
    out_html: str,
    zoom_start: int = 12,
    radius: int = 15,
) -> str:
    """Render a Folium heatmap to `out_html`; returns the path written.

    If `points` is empty, writes an empty map centred on Jakarta
    (-6.2088, 106.8456) so the demo never silently produces nothing.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError as e:
        raise RuntimeError(
            "folium is required for render_heatmap(); "
            "uncomment folium in requirements.txt and pip install."
        ) from e

    if points:
        avg_lat = sum(p.lat for p in points) / len(points)
        avg_lon = sum(p.lon for p in points) / len(points)
    else:
        avg_lat, avg_lon = -6.2088, 106.8456

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_start, tiles="OpenStreetMap")

    if points:
        heat_data = [[p.lat, p.lon, max(1, int(p.count))] for p in points]
        HeatMap(heat_data, radius=radius, blur=radius, min_opacity=0.3).add_to(m)

        for lat, lon, count in hot_zones(points, top_k=10):
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="red",
                fill=True,
                fill_opacity=0.8,
                popup=f"Hotspot: {count} violations",
            ).add_to(m)

    m.save(out_html)
    return out_html

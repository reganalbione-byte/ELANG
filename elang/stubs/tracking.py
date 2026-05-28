"""Multi-object tracking via DeepSORT.

    - DeepSORT (deep-sort-realtime) on top of YOLO detections
    - Stable track IDs across frames
    - Optional zone polygon: counts frames each track sits inside the zone
    - Output: per-track time-in-zone for violation flagging

deep-sort-realtime is imported lazily so the module can be inspected
without the dep installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..detection import Detection


@dataclass
class Track:
    track_id: int
    label: str
    last_bbox: tuple[int, int, int, int]
    first_seen_frame: int
    last_seen_frame: int
    frames_in_zone: int = 0
    zone_history: list[tuple[int, bool]] = field(default_factory=list)

    @property
    def duration_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon. `polygon` is a list of (x, y)."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class Tracker:
    """DeepSORT wrapper. Stateful across frames."""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        self.max_age = max_age
        self.n_init = n_init
        self._tracks: dict[int, Track] = {}
        self._impl = None

    def _ensure_impl(self):
        if self._impl is not None:
            return
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:
            raise RuntimeError(
                "deep-sort-realtime is required; "
                "uncomment in requirements.txt and pip install."
            ) from e
        self._impl = DeepSort(max_age=self.max_age, n_init=self.n_init)

    def update(
        self,
        detections: list[Detection],
        frame,
        frame_idx: int,
        zone_polygon: list[tuple[int, int]] | None = None,
    ) -> list[Track]:
        """Update tracks with this frame's detections.

        `frame` is the BGR numpy array; DeepSORT uses it for appearance
        embedding. Returns the list of *confirmed* tracks observed this
        frame.
        """
        self._ensure_impl()

        ds_dets = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            ds_dets.append((
                [x1, y1, x2 - x1, y2 - y1],
                float(d.confidence),
                d.label,
            ))

        ds_tracks = self._impl.update_tracks(ds_dets, frame=frame)

        observed: list[Track] = []
        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = (int(v) for v in ltrb)
            label = getattr(t, "det_class", None) or "unknown"
            tid = int(t.track_id)

            existing = self._tracks.get(tid)
            if existing is None:
                existing = Track(
                    track_id=tid,
                    label=str(label),
                    last_bbox=(x1, y1, x2, y2),
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                )
                self._tracks[tid] = existing
            else:
                existing.last_bbox = (x1, y1, x2, y2)
                existing.last_seen_frame = frame_idx
                if label != "unknown":
                    existing.label = str(label)

            in_zone = False
            if zone_polygon:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                in_zone = _point_in_polygon((cx, cy), zone_polygon)
                if in_zone:
                    existing.frames_in_zone += 1
            existing.zone_history.append((frame_idx, in_zone))

            observed.append(existing)

        return observed

    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def violators(self, min_frames_in_zone: int) -> list[Track]:
        """Return tracks that exceeded the in-zone duration threshold."""
        return [t for t in self._tracks.values() if t.frames_in_zone >= min_frames_in_zone]

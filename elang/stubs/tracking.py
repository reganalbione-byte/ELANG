"""Multi-object tracking — Phase 2 stub.

Target implementation (per DISHUB_Case_Analysis.md):
    - DeepSORT (deep-sort-realtime) on top of YOLO detections
    - Stable track IDs across frames
    - Duration tracking: how long each vehicle stays in a defined zone
    - Output: per-track time-in-zone for violation flagging

Activation: install deep-sort-realtime, then implement update().
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


class Tracker:
    """DeepSORT wrapper. Stateful across frames."""

    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self._tracks: dict[int, Track] = {}

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        zone_polygon: list[tuple[int, int]] | None = None,
    ) -> list[Track]:
        """Update tracks with this frame's detections."""
        raise NotImplementedError("Phase 2: install deep-sort-realtime.")

    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

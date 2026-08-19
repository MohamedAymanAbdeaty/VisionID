"""
tracker.py
----------
Lightweight IoU-based face tracker with identity cache.
Assigns persistent track IDs to faces across frames and caches
recognition results to avoid repeated embedding + search calls.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """State for a single tracked face."""
    track_id: int
    bbox: np.ndarray           # [x1, y1, x2, y2]
    landmarks: Optional[np.ndarray]
    score: float
    age: int = 0               # frames since last association
    identity: Optional[str] = None
    identity_score: Optional[float] = None
    identity_timestamp: float = field(default_factory=time.time)
    hits: int = 1


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union for two boxes [x1, y1, x2, y2]."""
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class FaceTracker:
    """
    IoU-based tracker that:
    - assigns a unique track_id to each face
    - caches identity decisions for a configurable duration
    - prunes lost tracks after max_age frames

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to associate a detection with an existing track.
    max_age : int
        Frames a track can survive without a matching detection.
    cache_ttl : float
        Seconds before an identity result expires and requires re-recognition.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        cache_ttl: float = 5.0,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.cache_ttl = cache_ttl
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list) -> list[Track]:
        """
        Associate detections with existing tracks (greedy IoU matching).

        Parameters
        ----------
        detections : list[Detection]
            Output from FaceDetector.detect().

        Returns
        -------
        list[Track]
            Active tracks, each with its detection's landmarks/bbox updated.
        """
        # Age all existing tracks
        for t in self._tracks.values():
            t.age += 1

        matched_track_ids = set()

        for det in detections:
            best_iou = self.iou_threshold
            best_tid = None

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                score = iou(det.bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None:
                # Update existing track
                t = self._tracks[best_tid]
                t.bbox = det.bbox
                t.landmarks = det.landmarks
                t.score = det.score
                t.age = 0
                t.hits += 1
                matched_track_ids.add(best_tid)
                det.track_id = best_tid
            else:
                # Spawn new track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=det.bbox,
                    landmarks=det.landmarks,
                    score=det.score,
                )
                det.track_id = tid
                matched_track_ids.add(tid)

        # Prune stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.age > self.max_age]
        for tid in stale:
            logger.debug("Dropping stale track %d", tid)
            del self._tracks[tid]

        return [self._tracks[tid] for tid in matched_track_ids if tid in self._tracks]

    def set_identity(
        self,
        track_id: int,
        identity: Optional[str],
        similarity: Optional[float] = None,
    ) -> None:
        """Store an identity decision for a track."""
        if track_id in self._tracks:
            t = self._tracks[track_id]
            t.identity = identity
            t.identity_score = similarity
            t.identity_timestamp = time.time()

    def get_cached_identity(self, track_id: int) -> tuple[Optional[str], Optional[float], bool]:
        """
        Return (identity, similarity, is_valid) from the cache.
        is_valid is False when TTL has expired or track doesn't exist.
        """
        t = self._tracks.get(track_id)
        if t is None or t.identity is None:
            return None, None, False
        age = time.time() - t.identity_timestamp
        return t.identity, t.identity_score, age < self.cache_ttl

    def reset(self) -> None:
        """Clear all tracks (e.g. when switching cameras)."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    @property
    def track_count(self) -> int:
        return len(self._tracks)

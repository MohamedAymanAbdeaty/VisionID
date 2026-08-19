"""
pipeline.py
-----------
Orchestrates the end-to-end frame processing pipeline:
  capture → detect → track → align → embed → search → threshold → result
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import FaceDetector, Detection
from .aligner import FaceAligner
from .embedder import FaceEmbedder
from .tracker import FaceTracker

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result for a single processed frame."""
    frame_id: int
    timestamp: float
    detections: list = field(default_factory=list)     # list[FaceResult]
    detection_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class FaceResult:
    """Result for a single face in a frame."""
    track_id: int
    bbox: list                # [x1, y1, x2, y2]
    score: float
    identity: Optional[str] = None
    identity_score: Optional[float] = None
    from_cache: bool = False
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    metadata_ms: float = 0.0


class FacePipeline:
    """
    End-to-end face recognition pipeline.

    Parameters
    ----------
    detector : FaceDetector
    aligner : FaceAligner
    embedder : FaceEmbedder
    tracker : FaceTracker
    searcher : VectorSearcher subclass (or None for detection-only mode)
    repository : PersonRepository (or None)
    threshold : float
        Cosine similarity threshold (0-1). Below = UNKNOWN.
    ambiguity_margin : float
        If (top1 - top2) similarity < margin, result is AMBIGUOUS.
    top_k : int
        Number of candidates returned by the searcher.
    """

    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: FaceEmbedder,
        tracker: FaceTracker,
        searcher=None,
        repository=None,
        threshold: float = 0.35,
        ambiguity_margin: float = 0.05,
        top_k: int = 5,
    ):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.tracker = tracker
        self.searcher = searcher
        self.repository = repository
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self.top_k = top_k
        self._frame_count = 0

    def load_models(self) -> None:
        """Load detector + embedder (called once at startup)."""
        if not self.detector.is_loaded():
            self.detector.load()
        if not self.embedder.is_loaded():
            self.embedder.load()

    def warmup(self) -> None:
        """Warm up ONNX kernels."""
        self.detector.warmup()
        self.embedder.warmup()

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Run the full pipeline on one BGR frame.

        Returns a FrameResult with per-face identities and timing.
        """
        t_frame_start = time.perf_counter()
        self._frame_count += 1

        # ── 1. Detection ──────────────────────────────────────────────────────
        raw_detections, detection_ms = self.detector.detect(frame)

        # ── 2. Tracking ───────────────────────────────────────────────────────
        active_tracks = self.tracker.update(raw_detections)

        face_results: list[FaceResult] = []

        for track in active_tracks:
            face_res = FaceResult(
                track_id=track.track_id,
                bbox=track.bbox.tolist(),
                score=track.score,
            )

            # ── 3. Cache check ─────────────────────────────────────────────────
            identity, sim, cached = self.tracker.get_cached_identity(track.track_id)
            if cached:
                face_res.identity = identity
                face_res.identity_score = sim
                face_res.from_cache = True
                face_results.append(face_res)
                continue

            # ── 4. No searcher → detection-only mode ───────────────────────────
            if self.searcher is None or not self.searcher.is_ready():
                face_results.append(face_res)
                continue

            # ── 5. Alignment + embedding ───────────────────────────────────────
            aligned = self.aligner.align_from_detection(frame, track)
            embedding, emb_ms = self.embedder.embed(aligned)
            face_res.embedding_ms = emb_ms

            # ── 6. Vector search ───────────────────────────────────────────────
            t_search = time.perf_counter()
            candidates = self.searcher.search(embedding, k=self.top_k)
            face_res.search_ms = (time.perf_counter() - t_search) * 1000.0

            # ── 7. Threshold & ambiguity decision ──────────────────────────────
            identity, sim = self._decide_identity(candidates)

            # ── 8. Metadata lookup ─────────────────────────────────────────────
            if identity and identity not in ("UNKNOWN", "AMBIGUOUS") and self.repository:
                t_meta = time.perf_counter()
                profile = self.repository.get_person(identity)
                face_res.metadata_ms = (time.perf_counter() - t_meta) * 1000.0
                if profile:
                    face_res.identity = profile.get("display_name", identity)
                    face_res.identity_score = sim
                else:
                    face_res.identity = identity
                    face_res.identity_score = sim
            else:
                face_res.identity = identity or "UNKNOWN"
                face_res.identity_score = sim

            # ── 9. Cache result ────────────────────────────────────────────────
            self.tracker.set_identity(track.track_id, face_res.identity, sim)
            face_results.append(face_res)

        total_ms = (time.perf_counter() - t_frame_start) * 1000.0

        return FrameResult(
            frame_id=self._frame_count,
            timestamp=time.time(),
            detections=face_results,
            detection_ms=detection_ms,
            total_ms=total_ms,
        )

    def _decide_identity(self, candidates: list) -> tuple[Optional[str], Optional[float]]:
        """Apply threshold and ambiguity checks to the top-K candidates."""
        if not candidates:
            return "UNKNOWN", 0.0

        best_id, best_sim = candidates[0]
        best_sim = float(best_sim)

        if best_sim < self.threshold:
            return "UNKNOWN", best_sim

        if len(candidates) > 1:
            _, second_sim = candidates[1]
            if (best_sim - float(second_sim)) < self.ambiguity_margin:
                return "AMBIGUOUS", best_sim

        return best_id, best_sim

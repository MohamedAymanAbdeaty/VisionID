"""
recognition.py - Recognition service coordinating the full pipeline.
"""
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RecognitionService:
    """
    High-level recognition service wrapping FacePipeline.
    Provides additional utility methods beyond per-frame processing.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def recognize_image(self, image) -> list[dict]:
        """Run recognition on a single BGR image array."""
        result = self.pipeline.process_frame(image)
        return [
            {
                "track_id": f.track_id,
                "bbox": f.bbox,
                "score": f.score,
                "identity": f.identity,
                "similarity": f.identity_score,
                "from_cache": f.from_cache,
                "timing": {
                    "embedding_ms": f.embedding_ms,
                    "search_ms": f.search_ms,
                    "metadata_ms": f.metadata_ms,
                },
            }
            for f in result.detections
        ]

    def recognize_query_vector(self, embedding: np.ndarray) -> list[dict]:
        """
        Run vector search + threshold directly (no image needed).
        Useful for benchmarking.
        """
        if self.pipeline.searcher is None:
            return []
        t0 = time.perf_counter()
        candidates = self.pipeline.searcher.search(embedding, k=self.pipeline.top_k)
        search_ms = (time.perf_counter() - t0) * 1000.0
        identity, sim = self.pipeline._decide_identity(candidates)
        return [{"identity": identity, "similarity": sim, "search_ms": search_ms}]

"""
base.py
-------
Abstract base class for all VisionID vector searchers.
Every index implementation must conform to this interface so that
the recognition pipeline can swap indexes without code changes.
"""

import abc
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorSearcher(abc.ABC):
    """
    Common interface for FAISS-backed vector indexes.

    Subclasses implement build / add / search / save / load.
    The pipeline only calls search() and checks is_ready().
    """

    def __init__(self, dim: int = 512, metric: str = "ip"):
        """
        Parameters
        ----------
        dim : int
            Embedding dimension (512 for standard ArcFace).
        metric : str
            'ip'  → inner-product (cosine on normalised vectors)
            'l2'  → L2 / Euclidean distance
        """
        self.dim = dim
        self.metric = metric
        self._index = None
        self._id_map: dict[int, str] = {}      # faiss position → person_id
        self._reverse: dict[str, list[int]] = {}  # person_id → [positions]
        self._size: int = 0

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def build(self, vectors: np.ndarray, person_ids: list[str]) -> None:
        """
        Build index from scratch with the given vectors.

        Parameters
        ----------
        vectors : np.ndarray  shape (N, dim)  float32
        person_ids : list[str]  length N
        """

    @abc.abstractmethod
    def add(self, vectors: np.ndarray, person_ids: list[str]) -> None:
        """Add new vectors to an already-built index."""

    @abc.abstractmethod
    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        Return Top-K (person_id, similarity_score) pairs.

        Similarity is in [0, 1] for IP metric (higher = more similar).
        """

    @abc.abstractmethod
    def save(self, directory: str) -> None:
        """Persist the index and ID map to ``directory``."""

    @abc.abstractmethod
    def load(self, directory: str) -> None:
        """Load the index and ID map from ``directory``."""

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True if the index exists and contains at least one vector."""
        return self._index is not None and self._size > 0

    @property
    def size(self) -> int:
        return self._size

    def _register_ids(self, person_ids: list[str], start_pos: int) -> None:
        """Map sequential FAISS positions to person_ids."""
        for i, pid in enumerate(person_ids):
            pos = start_pos + i
            self._id_map[pos] = pid
            self._reverse.setdefault(pid, []).append(pos)

    def _positions_to_ids(
        self, faiss_indices: np.ndarray, distances: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Convert raw FAISS results → (person_id, score) list."""
        results = []
        seen_ids: set[str] = set()

        for idx, dist in zip(faiss_indices, distances):
            if idx < 0:          # FAISS returns -1 for unfilled slots
                continue
            pid = self._id_map.get(int(idx))
            if pid is None:
                continue
            if pid in seen_ids:   # de-duplicate by person (keep best)
                continue
            seen_ids.add(pid)

            # Convert distance → similarity score in [0, 1]
            if self.metric == "ip":
                score = float(np.clip(dist, 0.0, 1.0))
            else:
                # L2: convert to similarity; rough heuristic
                score = float(1.0 / (1.0 + dist))

            results.append((pid, score))
            if len(results) >= k:
                break

        return results

    def _save_id_map(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self._id_map, f)

    def _load_id_map(self, path: str) -> None:
        import json
        with open(path) as f:
            raw = json.load(f)
        self._id_map = {int(k): v for k, v in raw.items()}
        self._reverse = {}
        for pos, pid in self._id_map.items():
            self._reverse.setdefault(pid, []).append(pos)
        self._size = len(self._id_map)

    @property
    def name(self) -> str:
        return type(self).__name__

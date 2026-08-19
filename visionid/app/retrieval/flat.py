"""
flat.py
-------
FAISS IndexFlatIP – exact inner-product search.
This is the ground-truth baseline: always correct, O(N) per query.
"""

import logging
from pathlib import Path

import numpy as np
import faiss

from .base import VectorSearcher

logger = logging.getLogger(__name__)


class FlatSearcher(VectorSearcher):
    """
    Exact nearest-neighbour search with FAISS IndexFlatIP.

    Best for:  small galleries (< 100K) or as ground-truth reference.
    Parameters: none beyond dim.
    """

    def __init__(self, dim: int = 512):
        super().__init__(dim=dim, metric="ip")

    def _make_index(self) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(self.dim)

    def build(self, vectors: np.ndarray, person_ids: list[str]) -> None:
        vectors = self._prep(vectors)
        self._index = self._make_index()
        self._index.add(vectors)
        self._register_ids(person_ids, 0)
        self._size = len(person_ids)
        logger.info("FlatSearcher: built index with %d vectors (dim=%d)", self._size, self.dim)

    def add(self, vectors: np.ndarray, person_ids: list[str]) -> None:
        if self._index is None:
            self._index = self._make_index()
        vectors = self._prep(vectors)
        start = self._size
        self._index.add(vectors)
        self._register_ids(person_ids, start)
        self._size += len(person_ids)
        logger.info("FlatSearcher: added %d vectors (total=%d)", len(person_ids), self._size)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.is_ready():
            logger.warning("FlatSearcher: index not ready")
            return []
        q = self._prep(query[None])  # shape (1, dim)
        k_actual = min(k * 3, self._size)  # over-fetch for de-dup
        distances, indices = self._index.search(q, k_actual)
        return self._positions_to_ids(indices[0], distances[0], k)

    def save(self, directory: str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "flat.index"))
        self._save_id_map(str(p / "id_map.json"))
        logger.info("FlatSearcher: saved to %s", directory)

    def load(self, directory: str) -> None:
        p = Path(directory)
        self._index = faiss.read_index(str(p / "flat.index"))
        self._load_id_map(str(p / "id_map.json"))
        logger.info("FlatSearcher: loaded %d vectors from %s", self._size, directory)

    @staticmethod
    def _prep(vectors: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(vectors, dtype=np.float32)

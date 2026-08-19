"""
hnsw.py
-------
FAISS IndexHNSWFlat – Hierarchical Navigable Small World graph.
Primary ANN candidate for low-latency recognition.
"""

import logging
from pathlib import Path

import numpy as np
import faiss

from .base import VectorSearcher

logger = logging.getLogger(__name__)


class HNSWSearcher(VectorSearcher):
    """
    ANN search with FAISS IndexHNSWFlat.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    M : int
        Number of connections per layer (graph degree).
        Higher M → better recall but more memory and build time.
        Typical range: 16–64.
    ef_construction : int
        Beam width during index construction.
        Higher → better graph quality at cost of build time.
    ef_search : int
        Beam width at query time. Controls recall vs latency trade-off.
    """

    def __init__(
        self,
        dim: int = 512,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        super().__init__(dim=dim, metric="ip")
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def _make_index(self) -> faiss.IndexHNSWFlat:
        index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def build(self, vectors: np.ndarray, person_ids: list[str]) -> None:
        vectors = self._prep(vectors)
        self._index = self._make_index()
        self._index.add(vectors)
        self._register_ids(person_ids, 0)
        self._size = len(person_ids)
        logger.info(
            "HNSWSearcher: built index M=%d efC=%d efS=%d with %d vectors",
            self.M, self.ef_construction, self.ef_search, self._size,
        )

    def add(self, vectors: np.ndarray, person_ids: list[str]) -> None:
        if self._index is None:
            self._index = self._make_index()
        vectors = self._prep(vectors)
        start = self._size
        self._index.add(vectors)
        self._register_ids(person_ids, start)
        self._size += len(person_ids)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.is_ready():
            return []
        q = self._prep(query[None])
        self._index.hnsw.efSearch = self.ef_search
        k_actual = min(k * 3, self._size)
        distances, indices = self._index.search(q, k_actual)
        return self._positions_to_ids(indices[0], distances[0], k)

    def save(self, directory: str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "hnsw.index"))
        self._save_id_map(str(p / "id_map.json"))
        logger.info("HNSWSearcher: saved to %s", directory)

    def load(self, directory: str) -> None:
        p = Path(directory)
        self._index = faiss.read_index(str(p / "hnsw.index"))
        self._load_id_map(str(p / "id_map.json"))
        logger.info("HNSWSearcher: loaded %d vectors from %s", self._size, directory)

    @staticmethod
    def _prep(vectors: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(vectors, dtype=np.float32)

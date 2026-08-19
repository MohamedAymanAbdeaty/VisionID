"""
ivfpq.py - FAISS IndexIVFPQ compressed approximate search.
"""
import logging
from pathlib import Path
import numpy as np
import faiss
from .base import VectorSearcher

logger = logging.getLogger(__name__)


class IVFPQSearcher(VectorSearcher):
    """
    ANN + compression using FAISS IndexIVFPQ.
    Trades a little recall for much lower memory use.

    Parameters
    ----------
    dim      : Embedding dimension (must be divisible by M_pq)
    nlist    : Voronoi cells
    M_pq     : Number of sub-quantizers (product quantization)
    nbits    : Bits per sub-quantizer code (8 = 256 centroids)
    nprobe   : Cells probed at query time
    """
    def __init__(self, dim=512, nlist=100, M_pq=64, nbits=8, nprobe=10):
        super().__init__(dim=dim, metric="ip")
        self.nlist = nlist
        self.M_pq = min(M_pq, dim)
        self.nbits = nbits
        self.nprobe = nprobe

    def _make_index(self):
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.M_pq, self.nbits)
        index.nprobe = self.nprobe
        return index

    def build(self, vectors: np.ndarray, person_ids: list) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vectors)
        self.nlist = min(self.nlist, max(1, n // 4))
        self._index = self._make_index()
        self._index.train(vectors)
        self._index.add(vectors)
        self._register_ids(person_ids, 0)
        self._size = n
        logger.info("IVFPQSearcher: built nlist=%d M_pq=%d with %d vectors", self.nlist, self.M_pq, n)

    def add(self, vectors: np.ndarray, person_ids: list) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = self._size
        self._index.add(vectors)
        self._register_ids(person_ids, start)
        self._size += len(person_ids)

    def search(self, query: np.ndarray, k: int = 5) -> list:
        if not self.is_ready():
            return []
        self._index.nprobe = self.nprobe
        q = np.ascontiguousarray(query[None], dtype=np.float32)
        k_actual = min(k * 3, self._size)
        distances, indices = self._index.search(q, k_actual)
        return self._positions_to_ids(indices[0], distances[0], k)

    def save(self, directory: str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "ivfpq.index"))
        self._save_id_map(str(p / "id_map.json"))

    def load(self, directory: str) -> None:
        p = Path(directory)
        self._index = faiss.read_index(str(p / "ivfpq.index"))
        self._load_id_map(str(p / "id_map.json"))

"""
embedder.py
-----------
ArcFace-style face embedding.
Wraps InsightFace's recognition model to produce an L2-normalised 512-D vector.
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Loads an InsightFace recognition model once and provides fast inference.

    Parameters
    ----------
    model_name : str
        InsightFace model pack that contains a recognition sub-model
        (e.g. 'buffalo_l', 'buffalo_sc').
    providers : list[str] | None
        ONNX execution providers.
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        providers: Optional[list] = None,
    ):
        self.model_name = model_name
        self._model = None
        self._providers = providers or ["CPUExecutionProvider"]
        self._dim: Optional[int] = None

    def load(self) -> None:
        """Load the InsightFace recognition model."""
        try:
            import insightface
            from insightface.model_zoo import get_model
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError("insightface is not installed.") from e

        logger.info("Loading InsightFace embedding model '%s' ...", self.model_name)
        t0 = time.perf_counter()

        # We re-use the FaceAnalysis app but only the 'rec' (recognition) component.
        app = FaceAnalysis(name=self.model_name, providers=self._providers)
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Extract the recognition sub-model
        self._model = app.models.get("recognition")
        if self._model is None:
            # Try direct model-zoo path for some packs
            available = list(app.models.keys())
            logger.warning(
                "Recognition model not found by key 'recognition'. Available: %s",
                available,
            )
            # Fallback: use the first non-det model
            for k, m in app.models.items():
                if "rec" in k or "recognition" in k or "w600k" in k:
                    self._model = m
                    break
            if self._model is None and available:
                self._model = list(app.models.values())[-1]

        if self._model is not None and hasattr(self._model, "output_names"):
            # probe dimension
            dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
            try:
                emb = self._model.get_feat(dummy)
                self._dim = emb.shape[-1]
            except Exception:
                self._dim = 512
        else:
            self._dim = 512

        elapsed = time.perf_counter() - t0
        logger.info("Embedder loaded in %.3f s (dim=%d)", elapsed, self._dim)

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:
        return self._dim or 512

    def embed(self, face_chip: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute a normalised embedding from an aligned face chip.

        Parameters
        ----------
        face_chip : np.ndarray
            BGR image of shape (112, 112, 3) or (H, W, 3).

        Returns
        -------
        embedding : np.ndarray
            L2-normalised vector of shape (dim,).
        latency_ms : float
            Inference time in milliseconds.
        """
        if not self.is_loaded():
            self.load()

        import cv2

        # Resize if needed
        if face_chip.shape[:2] != (112, 112):
            face_chip = cv2.resize(face_chip, (112, 112))

        t0 = time.perf_counter()
        # InsightFace recognition model expects BGR uint8 or float32 input
        # get_feat handles preprocessing internally when given the face array
        embedding = self._model.get_feat(face_chip[None])  # shape (1, dim)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        embedding = embedding[0]  # shape (dim,)
        embedding = self._l2_normalize(embedding)
        return embedding, latency_ms

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            logger.warning("Near-zero embedding norm; returning zero vector")
            return np.zeros_like(vec, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def warmup(self, iterations: int = 3) -> None:
        """Pre-load ONNX kernels with dummy data."""
        if not self.is_loaded():
            self.load()
        logger.info("Warming up embedder (%d iterations) ...", iterations)
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.embed(dummy)
        logger.info("Embedder warmup complete.")

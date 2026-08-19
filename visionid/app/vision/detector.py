"""
detector.py
-----------
SCRFD face detector wrapper using InsightFace.
Loads the model once; returns bounding boxes, scores and 5-point landmarks.
"""

import time
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Detection:
    """Container for a single face detection result."""

    __slots__ = ("bbox", "score", "landmarks", "track_id")

    def __init__(self, bbox: np.ndarray, score: float, landmarks: np.ndarray):
        # bbox: [x1, y1, x2, y2]  (float32)
        self.bbox = bbox.astype(np.float32)
        self.score = float(score)
        # landmarks: (5, 2)  (float32)
        self.landmarks = landmarks.astype(np.float32) if landmarks is not None else None
        self.track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return (
            float((self.bbox[0] + self.bbox[2]) / 2),
            float((self.bbox[1] + self.bbox[3]) / 2),
        )

    def as_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "track_id": self.track_id,
        }


class FaceDetector:
    """
    Wrapper around InsightFace's SCRFD detector.

    Parameters
    ----------
    model_name : str
        InsightFace model name, e.g. 'buffalo_sc' (lightweight) or 'buffalo_l' (high accuracy).
    det_size : tuple[int, int]
        Detection input resolution. Smaller = faster; larger = better for small faces.
    det_thresh : float
        Minimum detection confidence threshold.
    providers : list[str] | None
        ONNX execution providers.  None = auto-detect (CPU first).
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        det_size: tuple = (640, 640),
        det_thresh: float = 0.5,
        providers: Optional[list] = None,
    ):
        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._app = None
        self._providers = providers or ["CPUExecutionProvider"]

        logger.info(
            "FaceDetector init: model=%s det_size=%s thresh=%.2f",
            model_name,
            det_size,
            det_thresh,
        )

    def load(self) -> None:
        """Download (first run) and load the InsightFace app pipeline."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is not installed. Run: pip install insightface"
            ) from e

        logger.info("Loading InsightFace model '%s' ...", self.model_name)
        t0 = time.perf_counter()
        self._app = FaceAnalysis(
            name=self.model_name,
            providers=self._providers,
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.3f s", elapsed)

    def is_loaded(self) -> bool:
        return self._app is not None

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """
        Detect faces in a BGR or RGB frame.

        Returns
        -------
        detections : list[Detection]
            Detected faces sorted by descending confidence.
        latency_ms : float
            Detection wall-clock time in milliseconds.
        """
        if not self.is_loaded():
            self.load()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame

        t0 = time.perf_counter()
        faces = self._app.get(rgb)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        for face in faces:
            bbox = face.bbox  # [x1, y1, x2, y2]
            score = float(face.det_score)
            kps = face.kps if hasattr(face, "kps") and face.kps is not None else None
            det = Detection(bbox=bbox, score=score, landmarks=kps)
            detections.append(det)

        # sort descending by score
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections, latency_ms

    def warmup(self, iterations: int = 3) -> None:
        """Run the detector on a blank frame to pre-load ONNX kernels."""
        logger.info("Warming up detector (%d iterations) ...", iterations)
        dummy = np.zeros((self.det_size[1], self.det_size[0], 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
        logger.info("Warmup complete.")

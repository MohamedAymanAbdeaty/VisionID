"""
aligner.py
----------
Landmark-based face alignment.
Applies a similarity transform to warp the detected face to a canonical
112 × 112 crop (ArcFace standard template).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ArcFace standard 5-point template (112 × 112)
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceAligner:
    """
    Aligns a face chip to the ArcFace 112×112 canonical template using
    the 5 detected landmarks (left-eye, right-eye, nose, left-mouth,
    right-mouth).

    Parameters
    ----------
    output_size : int
        Side length of the output square crop (default 112 for ArcFace).
    """

    def __init__(self, output_size: int = 112):
        self.output_size = output_size
        self._dst = ARCFACE_DST * (output_size / 112.0)

    def align(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Warp ``frame`` so that the 5 face landmarks match the ArcFace template.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR (or RGB) image from which the face should be cut.
        landmarks : np.ndarray
            Shape (5, 2) array of (x, y) landmark coordinates in ``frame`` pixel space.

        Returns
        -------
        np.ndarray
            BGR (or RGB) aligned face chip of shape (output_size, output_size, 3).
        """
        src = landmarks.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, self._dst, method=cv2.LMEDS)
        if M is None:
            # Fallback: use bounding-box crop if alignment fails
            logger.warning("Affine estimation failed; returning uncropped frame centre")
            return cv2.resize(frame, (self.output_size, self.output_size))

        aligned = cv2.warpAffine(
            frame,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned

    def align_from_detection(self, frame: np.ndarray, detection) -> np.ndarray:
        """
        Convenience wrapper that accepts a ``Detection`` object.

        Falls back to bounding-box crop if landmarks are unavailable.
        """
        if detection.landmarks is not None and detection.landmarks.shape == (5, 2):
            return self.align(frame, detection.landmarks)

        # Fallback: simple bounding-box crop
        logger.debug("No landmarks for detection; using bbox crop")
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (self.output_size, self.output_size))

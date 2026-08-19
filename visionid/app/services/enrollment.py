"""
enrollment.py - Offline enrollment pipeline.
Turns consented images into indexed face embeddings.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EnrollmentService:
    """
    Orchestrates the enrollment pipeline:
    image → detect → align → embed → add to searcher → save to DB
    """

    def __init__(self, detector, aligner, embedder, searcher, repository):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.searcher = searcher
        self.repository = repository

    def enroll_person(
        self,
        person_id: str,
        display_name: str,
        image_paths: list[str],
        age: int = None,
        demo_city: str = None,
        role: str = None,
        min_det_score: float = 0.6,
        max_images: int = 8,
    ) -> dict:
        """
        Enroll one person from multiple images.

        Returns a summary dict with accepted/rejected counts.
        """
        self.repository.add_person(
            person_id, display_name, age=age, demo_city=demo_city, role=role
        )

        accepted_vectors = []
        rejected = []

        for img_path in image_paths[:max_images]:
            img = cv2.imread(img_path)
            if img is None:
                rejected.append((img_path, "unreadable"))
                continue

            dets, _ = self.detector.detect(img)
            if not dets:
                rejected.append((img_path, "no_face"))
                continue

            # Keep highest-confidence detection
            best = dets[0]
            if best.score < min_det_score:
                rejected.append((img_path, f"low_score:{best.score:.2f}"))
                continue

            if len(dets) > 1 and dets[1].score > 0.5:
                rejected.append((img_path, "multiple_faces"))
                continue

            aligned = self.aligner.align_from_detection(img, best)
            embedding, _ = self.embedder.embed(aligned)
            accepted_vectors.append((embedding, img_path))

        if not accepted_vectors:
            logger.warning("Enrollment for %s: no accepted images", person_id)
            return {"person_id": person_id, "accepted": 0, "rejected": len(rejected)}

        # Add to FAISS
        vecs = np.array([v for v, _ in accepted_vectors], dtype=np.float32)
        ids = [person_id] * len(vecs)

        if self.searcher.is_ready():
            self.searcher.add(vecs, ids)
        else:
            self.searcher.build(vecs, ids)

        # Record in DB
        start_pos = self.searcher.size - len(vecs)
        for i, (_, img_path) in enumerate(accepted_vectors):
            self.repository.add_face_vector(person_id, start_pos + i, img_path)

        result = {
            "person_id": person_id,
            "accepted": len(accepted_vectors),
            "rejected": len(rejected),
            "rejected_detail": rejected,
        }
        logger.info("Enrolled %s: %d accepted, %d rejected", person_id, result["accepted"], result["rejected"])
        return result

    def enroll_from_folder(
        self, folder: str, person_id: str = None, display_name: str = None, **kwargs
    ) -> dict:
        """Convenience: enroll from a folder of images."""
        folder_path = Path(folder)
        if not person_id:
            person_id = f"DEMO_{uuid.uuid4().hex[:6].upper()}"
        if not display_name:
            display_name = folder_path.name.replace("_", " ").title()

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [str(p) for p in sorted(folder_path.iterdir()) if p.suffix.lower() in exts]
        return self.enroll_person(person_id, display_name, images, **kwargs)

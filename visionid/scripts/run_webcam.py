"""
run_webcam.py
Live webcam demo with real-time face recognition overlay.
Usage: python scripts/run_webcam.py [--source 0] [--method hnsw] [--threshold 0.35]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.vision.detector import FaceDetector
from app.vision.aligner import FaceAligner
from app.vision.embedder import FaceEmbedder
from app.vision.tracker import FaceTracker
from app.vision.pipeline import FacePipeline
from app.database import init_db, PersonRepository
from app.retrieval import FlatSearcher, HNSWSearcher, IVFSearcher, IVFPQSearcher

COLORS = {
    "match":     (52, 211, 153),   # green
    "unknown":   (244,  63,  94),  # rose
    "ambiguous": (251, 191,  36),  # amber
}

INDEX_MAP = {
    "flat":   FlatSearcher,
    "hnsw":   HNSWSearcher,
    "ivf":    IVFSearcher,
    "ivfpq":  IVFPQSearcher,
}


def draw_overlay(frame, face_result, detection_ms: float, total_ms: float, method: str, gallery_size: int):
    """Draw bounding box, identity label and timing overlay."""
    identity = face_result.identity or "UNKNOWN"
    sim = face_result.identity_score or 0.0
    bbox = [int(v) for v in face_result.bbox]
    x1, y1, x2, y2 = bbox

    is_match = identity not in ("UNKNOWN", "AMBIGUOUS")
    color_key = "match" if is_match else ("ambiguous" if identity == "AMBIGUOUS" else "unknown")
    color = COLORS[color_key]

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background + text
    label = f"{identity}"
    if is_match and sim > 0:
        label += f"  {sim:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Cache indicator
    if face_result.from_cache:
        cv2.putText(frame, "CACHED", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)


def draw_stats_panel(frame, frame_result, method: str, gallery_size: int):
    """Draw global stats in top-right corner."""
    h, w = frame.shape[:2]
    lines = [
        f"Index:   {method.upper()}",
        f"Gallery: {gallery_size:,}",
        f"Detect:  {frame_result.detection_ms:.1f} ms",
        f"Total:   {frame_result.total_ms:.1f} ms",
        f"Faces:   {len(frame_result.detections)}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.48, 1
    line_h = 20
    panel_w = 230
    panel_h = len(lines) * line_h + 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w - 8, 8), (w - 8, 8 + panel_h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (w - panel_w - 8, 8), (w - 8, 8 + panel_h), (99, 102, 241), 1)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (w - panel_w, 24 + i * line_h),
                    font, fs, (200, 200, 255), thick, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--method", default="flat", choices=list(INDEX_MAP))
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--skip-frames", type=int, default=2,
                        help="Process every N-th frame (1=all)")
    parser.add_argument("--det-size", type=int, default=320)
    args = parser.parse_args()

    # ── Init ────────────────────────────────────────────────────────────────
    init_db()
    repo = PersonRepository()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source '{source}'")
        sys.exit(1)

    det_size = (args.det_size, args.det_size)
    detector = FaceDetector(det_size=det_size)
    aligner  = FaceAligner()
    embedder = FaceEmbedder()
    tracker  = FaceTracker(cache_ttl=5.0)

    # Load index
    searcher_cls = INDEX_MAP[args.method]
    searcher = searcher_cls(dim=512)
    idx_path = f"data/indexes/{args.method}_demo"
    if Path(idx_path).exists():
        searcher.load(idx_path)
        print(f"Loaded {args.method.upper()} index: {searcher.size} vectors")
    else:
        flat_path = "data/indexes/flat_demo"
        if Path(flat_path).exists():
            flat = FlatSearcher(dim=512)
            flat.load(flat_path)
            searcher = flat
            print(f"Fallback to Flat index: {flat.size} vectors")
        else:
            print("WARNING: No index found. Run scripts/enroll_demo.py first.")

    pipeline = FacePipeline(
        detector=detector, aligner=aligner, embedder=embedder,
        tracker=tracker, searcher=searcher, repository=repo,
        threshold=args.threshold,
    )
    pipeline.load_models()
    pipeline.warmup()

    print(f"\nVisionID Webcam Demo  |  Index: {args.method.upper()}  |  Threshold: {args.threshold}")
    print("Press Q to quit, S to save screenshot\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % max(args.skip_frames, 1) == 0:
            result = pipeline.process_frame(frame)

            for face_r in result.detections:
                draw_overlay(frame, face_r, result.detection_ms, result.total_ms,
                             args.method, searcher.size)

            draw_stats_panel(frame, result, args.method, searcher.size)

        cv2.imshow("VisionID", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"results/screenshot_{int(time.time())}.jpg"
            Path("results").mkdir(exist_ok=True)
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

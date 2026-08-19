"""
main.py - VisionID FastAPI application entry point.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.database import init_db, PersonRepository
from app.retrieval import FlatSearcher, HNSWSearcher, IVFSearcher, IVFPQSearcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("visionid.main")

# ── Global singletons ──────────────────────────────────────────────────────────
_detector = None
_aligner = None
_embedder = None
_tracker = None
_pipeline = None
_searchers: dict = {}
_active_searcher_name: str = "flat"
_repository: PersonRepository = None

INDEX_DIR = Path("data/indexes")
DEMO_INDEX_DIR = INDEX_DIR / "flat_demo"


def _get_searcher(name: str):
    if name not in _searchers:
        cls_map = {
            "flat": FlatSearcher,
            "hnsw": HNSWSearcher,
            "ivf": IVFSearcher,
            "ivfpq": IVFPQSearcher,
        }
        if name not in cls_map:
            return None
        searcher = cls_map[name](dim=512)
        idx_path = INDEX_DIR / f"{name}_demo"
        if idx_path.exists():
            try:
                searcher.load(str(idx_path))
                logger.info("Loaded %s index from %s", name, idx_path)
            except Exception as e:
                logger.warning("Could not load %s index: %s", name, e)
        _searchers[name] = searcher
    return _searchers[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector, _aligner, _embedder, _tracker, _pipeline, _repository

    # Init DB
    init_db()
    _repository = PersonRepository()

    # Lazy-load AI models (first request triggers load)
    from app.vision.detector import FaceDetector
    from app.vision.aligner import FaceAligner
    from app.vision.embedder import FaceEmbedder
    from app.vision.tracker import FaceTracker
    from app.vision.pipeline import FacePipeline

    _detector = FaceDetector(model_name="buffalo_sc")
    _aligner = FaceAligner()
    _embedder = FaceEmbedder(model_name="buffalo_sc")
    _tracker = FaceTracker(cache_ttl=5.0)

    searcher = _get_searcher("flat")

    _pipeline = FacePipeline(
        detector=_detector,
        aligner=_aligner,
        embedder=_embedder,
        tracker=_tracker,
        searcher=searcher,
        repository=_repository,
        threshold=0.35,
    )

    logger.info("VisionID API ready.")
    yield
    logger.info("VisionID API shutting down.")


app = FastAPI(
    title="VisionID",
    description="Real-Time Face Identity Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static UI ──────────────────────────────────────────────────────────────────
ui_dir = Path(__file__).parent.parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "gallery_size": _searchers.get("flat", type("_", (), {"size": 0})()).size
        if _searchers else 0,
        "index_active": _active_searcher_name,
    }


# ── Gallery info ───────────────────────────────────────────────────────────────
@app.get("/api/gallery", tags=["gallery"])
def get_gallery():
    persons = _repository.list_persons() if _repository else []
    return {
        "count": len(persons),
        "persons": persons,
        "index": _active_searcher_name,
    }


# ── Switch index ───────────────────────────────────────────────────────────────
@app.post("/api/index/switch", tags=["index"])
def switch_index(method: str = Form(...)):
    global _active_searcher_name
    valid = ["flat", "hnsw", "ivf", "ivfpq"]
    if method not in valid:
        raise HTTPException(400, f"method must be one of {valid}")
    _active_searcher_name = method
    searcher = _get_searcher(method)
    if _pipeline:
        _pipeline.searcher = searcher
    return {"switched_to": method, "size": searcher.size if searcher else 0}


# ── Recognize from uploaded image ─────────────────────────────────────────────
@app.post("/api/recognize", tags=["recognition"])
async def recognize_image(file: UploadFile = File(...)):
    import cv2
    import io
    from PIL import Image

    data = await file.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if not _detector.is_loaded():
        _detector.load()
    if not _embedder.is_loaded():
        _embedder.load()

    result = _pipeline.process_frame(img_bgr)

    return {
        "frame_id": result.frame_id,
        "detection_ms": round(result.detection_ms, 2),
        "total_ms": round(result.total_ms, 2),
        "faces": [
            {
                "track_id": f.track_id,
                "bbox": f.bbox,
                "score": round(f.score, 3),
                "identity": f.identity,
                "similarity": round(f.identity_score, 4) if f.identity_score else None,
                "from_cache": f.from_cache,
                "embedding_ms": round(f.embedding_ms, 2),
                "search_ms": round(f.search_ms, 2),
            }
            for f in result.detections
        ],
    }


# ── Benchmark endpoint ─────────────────────────────────────────────────────────
@app.get("/api/benchmark/quick", tags=["benchmark"])
def quick_benchmark():
    """Run a fast in-memory benchmark and return results."""
    import time

    rng = np.random.default_rng(42)
    DIM = 512
    SIZE = 5000
    N_QUERIES = 100

    gallery = rng.standard_normal((SIZE, DIM)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    ids = [f"P{i}" for i in range(SIZE)]
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    results = {}
    for name, cls, kwargs in [
        ("Flat", FlatSearcher, {}),
        ("HNSW", HNSWSearcher, {"M": 32, "ef_search": 64}),
        ("IVF", IVFSearcher, {"nlist": 50, "nprobe": 5}),
    ]:
        try:
            s = cls(dim=DIM, **kwargs)
            s.build(gallery, ids)
            latencies = []
            for q in queries:
                t0 = time.perf_counter()
                s.search(q, k=5)
                latencies.append((time.perf_counter() - t0) * 1000)
            lats = np.array(latencies)
            results[name] = {
                "p50_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_ms": round(float(np.percentile(lats, 95)), 3),
                "mean_ms": round(float(lats.mean()), 3),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"gallery_size": SIZE, "n_queries": N_QUERIES, "results": results}


# ── Persons CRUD ───────────────────────────────────────────────────────────────
@app.get("/api/persons/{person_id}", tags=["persons"])
def get_person(person_id: str):
    p = _repository.get_person(person_id)
    if not p:
        raise HTTPException(404, "Person not found")
    return p


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

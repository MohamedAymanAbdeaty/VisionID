"""
build_index.py
Builds FAISS index from enrollment images or a pre-generated vector file.
Usage: python scripts/build_index.py --method hnsw --vectors data/generated/synthetic_10000x512.npy
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.retrieval import FlatSearcher, HNSWSearcher, IVFSearcher, IVFPQSearcher
from app.database import init_db, PersonRepository

INDEX_MAP = {
    "flat":   FlatSearcher,
    "hnsw":   HNSWSearcher,
    "ivf":    IVFSearcher,
    "ivfpq":  IVFPQSearcher,
}


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS vector index")
    parser.add_argument("--method", choices=list(INDEX_MAP), default="hnsw")
    parser.add_argument("--vectors", required=True, help="Path to .npy file of shape (N, dim)")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--out-dir", default="data/indexes")
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    args = parser.parse_args()

    vecs = np.load(args.vectors).astype(np.float32)
    print(f"Loaded {len(vecs)} vectors from {args.vectors}")

    person_ids = [f"SYNTH_{i:06d}" for i in range(len(vecs))]

    cls = INDEX_MAP[args.method]
    if args.method == "hnsw":
        searcher = cls(dim=args.dim, M=args.hnsw_m)
    elif args.method in ("ivf", "ivfpq"):
        searcher = cls(dim=args.dim, nlist=args.nlist, nprobe=args.nprobe)
    else:
        searcher = cls(dim=args.dim)

    import time
    t0 = time.perf_counter()
    searcher.build(vecs, person_ids)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Built {args.method} index in {build_ms:.1f} ms")

    out_dir = f"{args.out_dir}/{args.method}"
    searcher.save(out_dir)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()

"""
generate_synthetic_vectors.py
Generates synthetic L2-normalised 512-D vectors for FAISS benchmarking.
Usage: python scripts/generate_synthetic_vectors.py --count 100000 --dim 512
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate(count: int, dim: int, seed: int = 42, out_dir: str = "data/generated") -> str:
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic %d-D vectors ...", count, dim)
    vecs = rng.standard_normal((count, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-10)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/synthetic_{count}x{dim}.npy"
    np.save(out_path, vecs)
    logger.info("Saved to %s  (%.1f MB)", out_path, vecs.nbytes / 1e6)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic normalised face vectors")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/generated")
    args = parser.parse_args()
    generate(args.count, args.dim, args.seed, args.out_dir)


if __name__ == "__main__":
    main()

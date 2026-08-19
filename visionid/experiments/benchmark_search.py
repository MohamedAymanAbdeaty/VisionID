"""
benchmark_search.py
Benchmarks Flat / HNSW / IVF / IVF-PQ at multiple gallery sizes.
Outputs CSV and PNG plots to results/.

Usage: python experiments/benchmark_search.py
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.retrieval import FlatSearcher, HNSWSearcher, IVFSearcher, IVFPQSearcher

GALLERY_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
N_QUERIES = 200
WARMUP = 20
DIM = 512
TOP_K = 5


def build_searcher(method: str, dim: int, nlist: int = 100):
    if method == "Flat":
        return FlatSearcher(dim=dim)
    elif method == "HNSW":
        return HNSWSearcher(dim=dim, M=32, ef_construction=200, ef_search=64)
    elif method == "IVF":
        return IVFSearcher(dim=dim, nlist=nlist, nprobe=10)
    elif method == "IVFPQ":
        return IVFPQSearcher(dim=dim, nlist=nlist, M_pq=64, nbits=8, nprobe=10)
    raise ValueError(f"Unknown method: {method}")


def run_benchmark():
    rng = np.random.default_rng(42)
    methods = ["Flat", "HNSW", "IVF", "IVFPQ"]

    results = []

    for size in GALLERY_SIZES:
        print(f"\n{'='*50}")
        print(f"Gallery size: {size:,}")
        gallery = rng.standard_normal((size, DIM)).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        person_ids = [f"P{i:07d}" for i in range(size)]

        queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Ground truth via Flat
        gt_searcher = FlatSearcher(dim=DIM)
        gt_searcher.build(gallery, person_ids)
        gt_results = [gt_searcher.search(q, k=TOP_K) for q in queries]
        gt_top1 = [r[0][0] if r else None for r in gt_results]

        for method in methods:
            nlist = max(1, int(np.sqrt(size)))
            try:
                s = build_searcher(method, DIM, nlist)
                t0 = time.perf_counter()
                s.build(gallery, person_ids)
                build_ms = (time.perf_counter() - t0) * 1000

                # Warmup
                for q in queries[:WARMUP]:
                    s.search(q, k=TOP_K)

                # Benchmark
                latencies = []
                top1_preds = []
                for q in queries:
                    t0 = time.perf_counter()
                    r = s.search(q, k=TOP_K)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    top1_preds.append(r[0][0] if r else None)

                lats = np.array(latencies)
                recall1 = sum(p == g for p, g in zip(top1_preds, gt_top1)) / N_QUERIES

                row = {
                    "method": method,
                    "gallery_size": size,
                    "build_ms": f"{build_ms:.1f}",
                    "p50_ms": f"{np.percentile(lats, 50):.3f}",
                    "p95_ms": f"{np.percentile(lats, 95):.3f}",
                    "p99_ms": f"{np.percentile(lats, 99):.3f}",
                    "mean_ms": f"{lats.mean():.3f}",
                    "recall1": f"{recall1:.4f}",
                }
                results.append(row)
                print(f"  {method:8s} | build={build_ms:8.1f}ms | p50={float(row['p50_ms']):.3f}ms | p95={float(row['p95_ms']):.3f}ms | recall@1={recall1:.3f}")

            except Exception as e:
                print(f"  {method:8s} FAILED: {e}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = Path("results/csv/benchmark_search.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV → {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_latency(results)
    _plot_recall(results)
    print("Plots saved to results/plots/")


def _plot_latency(results):
    methods = sorted(set(r["method"] for r in results))
    sizes = sorted(set(r["gallery_size"] for r in results))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in zip(methods, colors):
        xs = [r["gallery_size"] for r in results if r["method"] == method]
        ys = [float(r["p50_ms"]) for r in results if r["method"] == method]
        ax.plot(xs, ys, "o-", label=method, color=color, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gallery Size", fontsize=13)
    ax.set_ylabel("P50 Search Latency (ms)", fontsize=13)
    ax.set_title("VisionID – Search Latency vs Gallery Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/plots/latency_vs_gallery.png", dpi=150)
    plt.close(fig)


def _plot_recall(results):
    methods = sorted(set(r["method"] for r in results))
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in zip(methods, colors):
        xs = [r["gallery_size"] for r in results if r["method"] == method]
        ys = [float(r["recall1"]) for r in results if r["method"] == method]
        ax.plot(xs, ys, "s-", label=method, color=color, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Gallery Size", fontsize=13)
    ax.set_ylabel("Recall@1 vs Flat Baseline", fontsize=13)
    ax.set_title("VisionID – Recall@1 vs Gallery Size", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/plots/recall_vs_gallery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()

"""
threshold_analysis.py
Sweeps cosine similarity thresholds and plots FAR/FRR curves.
Usage: python experiments/threshold_analysis.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.retrieval import FlatSearcher


def main():
    rng = np.random.default_rng(99)
    DIM = 512
    N_PERSONS = 50
    VEC_PER_PERSON = 4
    N_UNKNOWN = 200
    THRESHOLDS = np.linspace(0.0, 1.0, 101)

    # Build a small gallery
    bases = rng.standard_normal((N_PERSONS, DIM)).astype(np.float32)
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)

    gallery_vecs, gallery_ids = [], []
    query_vecs, query_labels = [], []

    for i, base in enumerate(bases):
        pid = f"PERSON_{i:03d}"
        for _ in range(VEC_PER_PERSON):
            v = base + rng.standard_normal(DIM).astype(np.float32) * 0.05
            v /= np.linalg.norm(v)
            gallery_vecs.append(v)
            gallery_ids.append(pid)

        # Genuine query (slightly different angle)
        q = base + rng.standard_normal(DIM).astype(np.float32) * 0.08
        q /= np.linalg.norm(q)
        query_vecs.append(q)
        query_labels.append(pid)

    # Unknown queries
    for _ in range(N_UNKNOWN):
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        query_vecs.append(v)
        query_labels.append("UNKNOWN")

    searcher = FlatSearcher(dim=DIM)
    searcher.build(np.array(gallery_vecs, dtype=np.float32), gallery_ids)

    # Get similarity scores
    sims = []
    for q in query_vecs:
        res = searcher.search(q, k=1)
        sims.append(res[0][1] if res else 0.0)

    genuine_sims = [s for s, l in zip(sims, query_labels) if l != "UNKNOWN"]
    unknown_sims = [s for s, l in zip(sims, query_labels) if l == "UNKNOWN"]

    # Sweep thresholds
    FARs, FRRs = [], []
    for thr in THRESHOLDS:
        fa = sum(s >= thr for s in unknown_sims) / max(len(unknown_sims), 1)
        fr = sum(s < thr for s in genuine_sims) / max(len(genuine_sims), 1)
        FARs.append(fa)
        FRRs.append(fr)

    # Find EER
    diffs = [abs(f - r) for f, r in zip(FARs, FRRs)]
    eer_idx = np.argmin(diffs)
    eer = (FARs[eer_idx] + FRRs[eer_idx]) / 2

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(THRESHOLDS, FARs, label="FAR (False Accept)", color="#e74c3c", linewidth=2)
    ax.plot(THRESHOLDS, FRRs, label="FRR (False Reject)", color="#2ecc71", linewidth=2)
    ax.axvline(THRESHOLDS[eer_idx], color="#3498db", linestyle="--", label=f"EER ≈ {eer:.3f} @ thr={THRESHOLDS[eer_idx]:.2f}")
    ax.set_xlabel("Similarity Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("FAR / FRR Threshold Sweep", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(genuine_sims, bins=30, alpha=0.6, label="Genuine pairs", color="#2ecc71")
    ax.hist(unknown_sims, bins=30, alpha=0.6, label="Unknown probes", color="#e74c3c")
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Similarity Score Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/plots/threshold_far_frr.png", dpi=150)
    print(f"EER ≈ {eer:.4f} at threshold {THRESHOLDS[eer_idx]:.2f}")
    print("Plot saved → results/plots/threshold_far_frr.png")


if __name__ == "__main__":
    main()

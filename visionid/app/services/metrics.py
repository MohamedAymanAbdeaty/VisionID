"""
metrics.py - Latency and quality metrics collection and persistence.
"""
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects timing and retrieval quality measurements."""

    def __init__(self, results_dir: str = "results"):
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        (self._results_dir / "csv").mkdir(exist_ok=True)
        (self._results_dir / "plots").mkdir(exist_ok=True)
        self._latencies: dict[str, list[float]] = defaultdict(list)

    def record(self, metric: str, value_ms: float) -> None:
        self._latencies[metric].append(value_ms)

    def record_frame(self, frame_result) -> None:
        self.record("detection_ms", frame_result.detection_ms)
        self.record("total_ms", frame_result.total_ms)
        for face in frame_result.detections:
            if not face.from_cache:
                if face.embedding_ms:
                    self.record("embedding_ms", face.embedding_ms)
                if face.search_ms:
                    self.record("search_ms", face.search_ms)

    def summary(self, metric: str) -> dict:
        vals = self._latencies.get(metric, [])
        if not vals:
            return {}
        arr = np.array(vals)
        return {
            "count": len(arr),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def all_summaries(self) -> dict:
        return {k: self.summary(k) for k in self._latencies}

    def save_csv(self, filename: str = "latency_summary.csv") -> str:
        path = self._results_dir / "csv" / filename
        summaries = self.all_summaries()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "count", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"])
            for metric, s in summaries.items():
                if s:
                    writer.writerow([
                        metric, s["count"], f"{s['mean']:.3f}", f"{s['p50']:.3f}",
                        f"{s['p95']:.3f}", f"{s['p99']:.3f}", f"{s['min']:.3f}", f"{s['max']:.3f}",
                    ])
        logger.info("Saved metrics CSV to %s", path)
        return str(path)

    def save_json(self, filename: str = "latency_summary.json") -> str:
        path = self._results_dir / "csv" / filename
        with open(path, "w") as f:
            json.dump(self.all_summaries(), f, indent=2)
        return str(path)

    def reset(self) -> None:
        self._latencies.clear()

    @staticmethod
    def compute_recall(
        predictions: list[str],
        ground_truth: list[str],
        top_k_lists: Optional[list[list[str]]] = None,
        k: int = 1,
    ) -> float:
        if k == 1:
            correct = sum(p == g for p, g in zip(predictions, ground_truth))
        else:
            correct = sum(g in topk for g, topk in zip(ground_truth, top_k_lists or []))
        return correct / max(len(ground_truth), 1)

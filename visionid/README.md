# VisionID 🔍

> **Real-Time Closed-Set Face-Based Identity Retrieval**  
> using Deep Face Embeddings (ArcFace) and Fast Vector Search (FAISS)  
> *Zero-Cost University Research Prototype*

---

## Architecture

```
Webcam/Image
    │
    ▼
SCRFD Face Detection  ──── latency tracked
    │
    ▼
IoU Tracker + Identity Cache
    │
    ▼
5-Point Face Alignment (112×112)
    │
    ▼
ArcFace Embedding (512-D) + L2 Normalise  ──── latency tracked
    │
    ▼
FAISS Search  [Flat | HNSW | IVF | IVF-PQ]  ──── latency tracked
    │
    ▼
Threshold Decision  →  MATCH / UNKNOWN / AMBIGUOUS
    │
    ▼
SQLite Profile Lookup  ──── latency tracked
    │
    ▼
Annotated Frame + Demo Profile
```

---

## Quick Start

### 1. Create environment
```bash
cd visionid
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Enroll demo profiles (synthetic – no real biometrics)
```bash
python scripts/enroll_demo.py
```

### 3. Start the API + Web UI
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# then open  http://localhost:8000/ui/
```

### 4. Live webcam demo
```bash
python scripts/run_webcam.py --method hnsw --threshold 0.35
```

### 5. Run search benchmarks
```bash
# Quick in-memory benchmark (no models needed)
python experiments/benchmark_search.py

# Threshold / FAR / FRR analysis
python experiments/threshold_analysis.py
```

### 6. Generate large synthetic vector sets (for scalability experiments)
```bash
python scripts/generate_synthetic_vectors.py --count 100000 --dim 512
python scripts/build_index.py --method hnsw --vectors data/generated/synthetic_100000x512.npy
```

---

## Project Structure

```
visionid/
├── app/
│   ├── vision/       detector · aligner · embedder · tracker · pipeline
│   ├── retrieval/    base · flat · hnsw · ivf · ivfpq
│   ├── database/     schema.sql · connection · repository
│   ├── services/     enrollment · recognition · metrics
│   └── main.py       FastAPI application
├── scripts/
│   ├── enroll_demo.py
│   ├── run_webcam.py
│   ├── build_index.py
│   └── generate_synthetic_vectors.py
├── experiments/
│   ├── benchmark_search.py      ← main latency & recall benchmark
│   └── threshold_analysis.py   ← FAR / FRR sweep
├── ui/               index.html · style.css · app.js
├── models/           (ONNX models auto-downloaded by InsightFace)
├── data/
│   ├── enrollment/   (put real/consented images here)
│   ├── generated/    (synthetic .npy files)
│   └── indexes/      (saved FAISS indexes)
├── results/
│   ├── csv/          benchmark output
│   └── plots/        PNG charts
└── requirements.txt
```

---

## FAISS Index Comparison

| Index      | Type        | Recall@1 | P50 Latency | Memory |
|------------|-------------|----------|-------------|--------|
| FlatIP     | Exact       | 1.000    | ↑ scales O(N) | medium |
| HNSWFlat   | Graph ANN   | ~0.99    | **very low** | high   |
| IVFFlat    | Cluster ANN | ~0.97    | low         | medium |
| IVF-PQ     | Compressed  | ~0.92    | low         | **low** |

---

## KPIs Tracked

| Metric        | Tool |
|---------------|------|
| Detection ms  | `MetricsCollector` |
| Embedding ms  | `MetricsCollector` |
| Search ms     | `MetricsCollector` |
| Total E2E ms  | `MetricsCollector` |
| Recall@1 / @5 | `experiments/benchmark_search.py` |
| FAR / FRR     | `experiments/threshold_analysis.py` |
| Index RAM     | `psutil` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + gallery size |
| GET | `/api/gallery` | List enrolled persons |
| POST | `/api/recognize` | Recognize face in uploaded image |
| POST | `/api/index/switch` | Switch active FAISS index |
| GET | `/api/benchmark/quick` | Run in-process latency benchmark |
| GET | `/api/persons/{id}` | Get person profile |

---

## Ethics & Privacy

- **Closed academic prototype** – uses only synthetic or explicitly consented identities.
- No real government database. No covert collection. No cloud upload.
- InsightFace pretrained models are used under their **non-commercial research** licence.
- Enrolled demo data should be deleted after grading if appropriate.

---

## References

- InsightFace / SCRFD / ArcFace: https://github.com/deepinsight/insightface
- FAISS: https://github.com/facebookresearch/faiss
- ArcFace paper (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
- SCRFD paper: https://arxiv.org/abs/2105.04714
- ONNX Runtime: https://onnxruntime.ai/

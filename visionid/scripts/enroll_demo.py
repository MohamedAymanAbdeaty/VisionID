"""
enroll_demo.py
Creates synthetic demo profiles and enrolls them for presentation.
No real biometric data – purely synthetic for academic demo.
Usage: python scripts/enroll_demo.py
"""
import logging
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.database import init_db, PersonRepository
from app.retrieval import FlatSearcher

DEMO_PROFILES = [
    ("DEMO_0001", "Alex Morgan", 28, "Dublin", "Software Engineer"),
    ("DEMO_0002", "Sam Rivera", 34, "Cork", "Data Scientist"),
    ("DEMO_0003", "Jordan Lee", 22, "Galway", "Student"),
    ("DEMO_0004", "Taylor Kim", 41, "Limerick", "Researcher"),
    ("DEMO_0005", "Casey Walsh", 30, "Waterford", "Engineer"),
]


def main():
    init_db()
    repo = PersonRepository()
    searcher = FlatSearcher(dim=512)

    rng = np.random.default_rng(1337)
    all_vecs = []
    all_ids = []

    for pid, name, age, city, role in DEMO_PROFILES:
        # Simulate 4 enrollment vectors per person (slight jitter)
        base_vec = rng.standard_normal(512).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)

        for _ in range(4):
            noise = rng.standard_normal(512).astype(np.float32) * 0.05
            v = base_vec + noise
            v /= np.linalg.norm(v)
            all_vecs.append(v)
            all_ids.append(pid)

        repo.add_person(pid, name, age=age, demo_city=city, role=role)
        print(f"  Registered: {pid} – {name}")

    vecs = np.array(all_vecs, dtype=np.float32)
    searcher.build(vecs, all_ids)

    out_dir = "data/indexes/flat_demo"
    searcher.save(out_dir)
    print(f"\nDemo gallery saved → {out_dir}/")
    print(f"Enrolled {len(DEMO_PROFILES)} identities, {len(all_vecs)} total vectors")


if __name__ == "__main__":
    main()

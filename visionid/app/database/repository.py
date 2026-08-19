"""
repository.py - CRUD operations for persons and face_vectors tables.
"""
import logging
import sqlite3
from typing import Optional
from .connection import get_connection

logger = logging.getLogger(__name__)


class PersonRepository:
    def __init__(self, db_path: str = None):
        self._conn: sqlite3.Connection = get_connection(db_path)

    # ── Persons ────────────────────────────────────────────────────────────────

    def add_person(self, person_id: str, display_name: str, age: int = None,
                   demo_city: str = None, role: str = None,
                   profile_image_path: str = None) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO persons
               (person_id, display_name, age, demo_city, role, profile_image_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (person_id, display_name, age, demo_city, role, profile_image_path),
        )
        self._conn.commit()

    def get_person(self, person_id: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM persons WHERE person_id = ?", (person_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_persons(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM persons ORDER BY created_at")
        return [dict(r) for r in cur.fetchall()]

    def delete_person(self, person_id: str) -> None:
        self._conn.execute("DELETE FROM face_vectors WHERE person_id = ?", (person_id,))
        self._conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
        self._conn.commit()

    def count_persons(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    # ── Face vectors ───────────────────────────────────────────────────────────

    def add_face_vector(self, person_id: str, faiss_position: int,
                        source_image: str = None) -> int:
        cur = self._conn.execute(
            """INSERT INTO face_vectors (person_id, faiss_position, source_image)
               VALUES (?, ?, ?)""",
            (person_id, faiss_position, source_image),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_vectors_for_person(self, person_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM face_vectors WHERE person_id = ? ORDER BY vector_id",
            (person_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def count_vectors(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM face_vectors").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

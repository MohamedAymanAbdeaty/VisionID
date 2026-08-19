-- VisionID SQLite schema
-- Run via: python -c "from app.database.connection import init_db; init_db()"

CREATE TABLE IF NOT EXISTS persons (
    person_id       TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    age             INTEGER,
    demo_city       TEXT,
    role            TEXT,
    profile_image_path TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS face_vectors (
    vector_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       TEXT NOT NULL REFERENCES persons(person_id),
    faiss_position  INTEGER,
    source_image    TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_face_vectors_person ON face_vectors(person_id);

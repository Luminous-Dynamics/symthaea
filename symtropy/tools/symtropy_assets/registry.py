import sqlite3
import os
from paths import get_registry_path

def create_registry(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # ... (rest of the schema)
    cursor.executescript("""
        CREATE TABLE assets (
            id TEXT PRIMARY KEY,
            title TEXT,
            type TEXT,
            source_name TEXT,
            source_url TEXT,
            creator TEXT,
            license_id TEXT,
            license_url TEXT,
            acquisition_method TEXT,
            ai_provenance_state TEXT,
            status TEXT,
            technical_status TEXT,
            style_status TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        );

        CREATE TABLE registry_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        INSERT INTO registry_meta (key, value, updated_at) VALUES ('schema_version', '1', DATETIME('now'));

        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            role TEXT,
            path TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            mime_type TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id)
        );

        CREATE TABLE licenses (
            id TEXT PRIMARY KEY,
            spdx_id TEXT,
            name TEXT,
            url TEXT,
            attribution_required BOOLEAN,
            commercial_allowed BOOLEAN,
            derivative_allowed BOOLEAN,
            share_alike BOOLEAN
        );

        CREATE TABLE provenance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            event_type TEXT,
            timestamp TIMESTAMP,
            actor TEXT,
            details_json TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id)
        );

        CREATE TABLE review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            review_type TEXT,
            reviewer TEXT,
            status TEXT,
            timestamp TIMESTAMP,
            notes TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id)
        );

        CREATE TABLE style_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            palette_id TEXT,
            material_family TEXT,
            biome TEXT,
            branch TEXT,
            reviewer TEXT,
            status TEXT,
            notes TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id)
        );

        CREATE TABLE behaviors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            role TEXT,
            parameters TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id)
        );
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = get_registry_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if not os.path.exists(db_path):
        create_registry(db_path)
        print(f"Created registry at {db_path}")
    else:
        print(f"Registry already exists at {db_path}")

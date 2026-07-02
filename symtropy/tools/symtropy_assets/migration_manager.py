import os
import sqlite3
import paths
import datetime

def bootstrap_schema(cursor):
    """Create the registry schema from scratch if it doesn't exist yet.

    Nothing else in this tool ever created these tables — the only working
    registry (assets/symtropy-foundry-data/registry/assets.sqlite) was
    clearly bootstrapped by hand at some point, so any fresh setup (new
    contributor, CI, a rebuilt asset root) hit "no such table: registry_meta"
    the first time `registry migrate` or `ingest` ran. Schema below mirrors
    that real registry's current `.schema` output exactly (version 2,
    including the mass_kg/com_*/inertia_* columns baked into `assets`
    directly, matching its actual current shape rather than replaying
    history that may never have existed as separate migration steps).
    """
    cursor.execute("""CREATE TABLE IF NOT EXISTS assets (
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
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            technical_status TEXT, style_status TEXT, mass_kg REAL, com_x REAL, com_y REAL, com_z REAL, inertia_ixx REAL, inertia_iyy REAL, inertia_izz REAL)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS registry_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            role TEXT,
            path TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            mime_type TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id))""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS licenses (
            id TEXT PRIMARY KEY,
            spdx_id TEXT,
            name TEXT,
            url TEXT,
            attribution_required BOOLEAN,
            commercial_allowed BOOLEAN,
            derivative_allowed BOOLEAN,
            share_alike BOOLEAN)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS provenance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            event_type TEXT,
            timestamp TIMESTAMP,
            actor TEXT,
            details_json TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id))""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            review_type TEXT,
            reviewer TEXT,
            status TEXT,
            timestamp TIMESTAMP,
            notes TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id))""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS style_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            palette_id TEXT,
            material_family TEXT,
            biome TEXT,
            branch TEXT,
            reviewer TEXT,
            status TEXT,
            notes TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id))""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS behaviors (
            id INTEGER PRIMARY KEY,
            asset_id TEXT,
            role TEXT,
            parameters TEXT,
            FOREIGN KEY(asset_id) REFERENCES assets(id))""")

    cursor.execute("SELECT value FROM registry_meta WHERE key = 'schema_version'")
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO registry_meta VALUES ('schema_version', '2', ?)",
            (datetime.datetime.now().isoformat(),),
        )

def run_migrations():
    registry_path = paths.get_registry_path()
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    conn = sqlite3.connect(registry_path)
    cursor = conn.cursor()

    bootstrap_schema(cursor)
    conn.commit()

    # Check current version
    cursor.execute("SELECT value FROM registry_meta WHERE key = 'schema_version'")
    version = int(cursor.fetchone()[0])

    if version < 2:
        print("Migrating to version 2: adding technical_status and style_status columns...")
        try:
            cursor.execute("ALTER TABLE assets ADD COLUMN technical_status TEXT")
        except sqlite3.OperationalError:
            pass # Column may already exist
        try:
            cursor.execute("ALTER TABLE assets ADD COLUMN style_status TEXT")
        except sqlite3.OperationalError:
            pass

        cursor.execute("UPDATE registry_meta SET value = '2', updated_at = ? WHERE key = 'schema_version'",
                       (datetime.datetime.now().isoformat(),))
        print("Migration to version 2 complete.")

    conn.commit()
    conn.close()

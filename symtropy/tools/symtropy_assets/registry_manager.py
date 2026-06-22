import json
import sqlite3
from datetime import datetime

import yaml

import paths

def ingest_manifest(manifest_path, status):
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)

    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()

    # Store asset
    asset_id = manifest.get('id')
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO assets (
            id, title, type, source_name, source_url, creator, license_id, license_url,
            acquisition_method, ai_provenance_state, status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            type = excluded.type,
            source_name = excluded.source_name,
            source_url = excluded.source_url,
            creator = excluded.creator,
            license_id = excluded.license_id,
            license_url = excluded.license_url,
            acquisition_method = excluded.acquisition_method,
            ai_provenance_state = excluded.ai_provenance_state,
            status = excluded.status,
            updated_at = excluded.updated_at
    """, (
        asset_id,
        manifest.get('title'),
        manifest.get('type'),
        manifest.get('source', {}).get('source_name'),
        manifest.get('source', {}).get('source_url'),
        manifest.get('source', {}).get('creator'),
        manifest.get('license', {}).get('id'),
        manifest.get('license', {}).get('url'),
        manifest.get('source', {}).get('acquisition_method'),
        manifest.get('ai', {}).get('provenance_state'),
        status,
        now,
        now,
    ))

    cursor.execute("""
        INSERT INTO provenance_events (asset_id, event_type, timestamp, actor, details_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        asset_id,
        "INGESTION",
        now,
        "symtropy_assets.ingest_manifest",
        json.dumps({"manifest_path": str(manifest_path), "status": status}),
    ))

    for file_entry in manifest.get("files", []):
        cursor.execute("""
            INSERT INTO files (asset_id, role, path, sha256, size_bytes, mime_type)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
        """, (
            asset_id,
            file_entry.get("role"),
            file_entry.get("path"),
            file_entry.get("sha256"),
            file_entry.get("size_bytes"),
            file_entry.get("mime_type"),
        ))

    # Ingest behaviors
    if "behaviors" in manifest:
        for behavior in manifest["behaviors"]:
            cursor.execute("""
                INSERT INTO behaviors (asset_id, role, parameters)
                VALUES (?, ?, ?)
            """, (asset_id, behavior["role"], json.dumps(behavior["parameters"])))

    conn.commit()
    conn.close()
    return asset_id

def register_file(asset_id, role, path, sha256=None, size_bytes=None, mime_type=None):
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()

    # Check if file with this role already exists for the asset
    cursor.execute("SELECT id FROM files WHERE asset_id = ? AND role = ?", (asset_id, role))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE files SET path = ?, sha256 = ?, size_bytes = ?, mime_type = ?
            WHERE id = ?
        """, (path, sha256, size_bytes, mime_type, existing[0]))
    else:
        cursor.execute("""
            INSERT INTO files (asset_id, role, path, sha256, size_bytes, mime_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (asset_id, role, path, sha256, size_bytes, mime_type))

    conn.commit()
    conn.close()

def get_asset(asset_id):
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM assets WHERE id = ?", (asset_id,))
    columns = [column[0] for column in cursor.description]
    result = cursor.fetchone()
    conn.close()
    if result:
        return dict(zip(columns, result))
    return None

def update_asset_audit_status(asset_id, tech_status=None, style_status=None):
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()
    if tech_status:
        cursor.execute("UPDATE assets SET technical_status = ?, updated_at = ? WHERE id = ?",
                       (tech_status, datetime.now().isoformat(), asset_id))
    if style_status:
        cursor.execute("UPDATE assets SET style_status = ?, updated_at = ? WHERE id = ?",
                       (style_status, datetime.now().isoformat(), asset_id))
    conn.commit()
    conn.close()

def get_assets_needing_conversion():
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()
    # Assets that have a source file but no optimized file, or technical_status is pending
    cursor.execute("""
        SELECT a.id, f.path
        FROM assets a
        JOIN files f ON a.id = f.asset_id
        LEFT JOIN files f_opt ON a.id = f_opt.asset_id AND f_opt.role = 'optimized'
        WHERE f.role = 'source'
          AND (f_opt.id IS NULL OR a.technical_status IS NULL OR a.technical_status = 'PENDING')
    """)
    results = cursor.fetchall()
    conn.close()
    return results

def get_assets_by_biome(biome):
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id FROM assets a
        JOIN style_reviews sr ON a.id = sr.asset_id
        WHERE sr.biome = ? AND sr.status = 'APPROVED'
    """, (biome,))
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results

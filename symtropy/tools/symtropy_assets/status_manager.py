import sqlite3
import paths

def get_registry_status():
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()

    status = {}

    # Get schema version
    try:
        cursor.execute("SELECT value FROM registry_meta WHERE key = 'schema_version'")
        status["schema_version"] = cursor.fetchone()[0]
    except Exception:
        status["schema_version"] = "unknown"

    # Get asset counts
    cursor.execute("SELECT status, COUNT(*) FROM assets GROUP BY status")
    status["asset_counts"] = dict(cursor.fetchall())

    # Get total assets
    cursor.execute("SELECT COUNT(*) FROM assets")
    status["total_assets"] = cursor.fetchone()[0]

    conn.close()
    return status

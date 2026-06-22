import sqlite3
import datetime
import paths

def get_pending_review():
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, status FROM assets WHERE status = 'QUARANTINE_REVIEW'")
    assets = cursor.fetchall()
    conn.close()
    return assets

def review_asset(asset_id, status, reviewer, notes):
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()

    # Update asset status
    cursor.execute("UPDATE assets SET status = ?, updated_at = ? WHERE id = ?",
                   (status, datetime.datetime.now().isoformat(), asset_id))

    # Record review event
    cursor.execute("""
        INSERT INTO review_events (asset_id, review_type, reviewer, status, timestamp, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (asset_id, "MANUAL_REVIEW", reviewer, status, datetime.datetime.now().isoformat(), notes))

    conn.commit()
    conn.close()
    return f"Asset {asset_id} updated to {status}"

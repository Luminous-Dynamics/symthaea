import sqlite3
import paths
import datetime

def run_migrations():
    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()

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

import sqlite3
import os
import json
import paths
import shutil

def generate_export_reports(pack_id, target_dir, biome_filter=None):
    os.makedirs(target_dir, exist_ok=True)
    models_dir = os.path.join(target_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    conn = sqlite3.connect(paths.get_registry_path())
    cursor = conn.cursor()

    # Fetch approved assets, prioritizing optimized files over source
    query = """
        SELECT a.id, a.title, a.creator, a.source_url, a.license_id, a.ai_provenance_state,
               COALESCE(f_opt.path, f_src.path) as effective_path,
               sr.material_family, sr.biome
        FROM assets a
        LEFT JOIN files f_src ON a.id = f_src.asset_id AND f_src.role = 'source'
        LEFT JOIN files f_opt ON a.id = f_opt.asset_id AND f_opt.role = 'optimized'
        LEFT JOIN style_reviews sr ON a.id = sr.asset_id
        WHERE a.status LIKE 'APPROVED%'
    """
    params = []
    if biome_filter:
        query += " AND sr.biome = ?"
        params.append(biome_filter)

    cursor.execute(query, params)
    assets = cursor.fetchall()

    index = []

    # Generate Attribution Report
    with open(os.path.join(target_dir, "ATTRIBUTION.md"), "w") as f:
        f.write("# Asset Attribution\n\n")
        for asset in assets:
            aid, title, creator, url, lic, ai, effective_path, mat_fam, biome = asset
            f.write(f"- {title} (ID: {aid}) by {creator}: {url} (License: {lic})\n")

            # Copy file if it exists
            if effective_path:
                # Use path discovery to find real source path
                asset_root = paths.get_asset_root()
                real_src = os.path.join(asset_root, effective_path)
                if os.path.exists(real_src):
                    dest = os.path.join(models_dir, os.path.basename(real_src))
                    shutil.copy2(real_src, dest)
                    index.append({
                        "id": aid,
                        "path": f"models/{os.path.basename(real_src)}",
                        "title": title,
                        "material_family": mat_fam,
                        "biome": biome
                    })

    # Generate AI Disclosure Report
    with open(os.path.join(target_dir, "AI_DISCLOSURE_REGISTER.md"), "w") as f:
        f.write("# AI Disclosure Register\n\n")
        for asset in assets:
            f.write(f"- {asset[1]}: {asset[5]}\n")

    # Generate Index
    with open(os.path.join(target_dir, "asset_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Generate Behavior Index
    cursor.execute("SELECT asset_id, role, parameters FROM behaviors")
    behavior_data = cursor.fetchall()
    behavior_index = {}
    for aid, role, params in behavior_data:
        if aid not in behavior_index:
            behavior_index[aid] = []
        behavior_index[aid].append({"role": role, "parameters": json.loads(params)})

    with open(os.path.join(target_dir, "asset_behaviors.json"), "w") as f:
        json.dump(behavior_index, f, indent=2)

    conn.close()
    return f"Exported {len(index)} assets and behaviors to {target_dir}"

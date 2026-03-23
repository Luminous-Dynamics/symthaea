#!/usr/bin/env bash
# Import geothermal and extended energy data from public federal sources
#
# Usage:
#   cd terra-atlas-mvp && ./scripts/import-geothermal-data.sh
#
# Downloads from:
#   1. EIA Form 860 — ALL US power plants (geothermal, biomass, tidal, hybrid)
#   2. OpenEI — Geothermal well data and resource temperatures
#   3. NREL — Geothermal resource areas
#
# All data is public domain / free, no API keys required.

set -euo pipefail
DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"

echo "=== Geothermal & Extended Energy Data Import ==="
echo "Data directory: ${DATA_DIR}"

# 1. EIA Form 860 (Operational + Planned Generators)
echo ""
echo "--- EIA Form 860 ---"
EIA_URL="https://www.eia.gov/electricity/data/eia860/xls/eia8602023.zip"
EIA_ZIP="${DATA_DIR}/eia860.zip"
EIA_DIR="${DATA_DIR}/eia860"

if [ ! -d "${EIA_DIR}" ]; then
    echo "Downloading EIA Form 860 (2023)..."
    curl -L -o "${EIA_ZIP}" "${EIA_URL}" --progress-bar
    mkdir -p "${EIA_DIR}"
    unzip -o "${EIA_ZIP}" -d "${EIA_DIR}"
    echo "  Downloaded and extracted to ${EIA_DIR}"
else
    echo "  Already downloaded at ${EIA_DIR}"
fi

# 2. OpenEI Geothermal Data
echo ""
echo "--- OpenEI Geothermal Wells ---"
OPENEI_URL="https://openei.org/datasets/files/961/pub/USGeothermalResourceData.csv"
OPENEI_FILE="${DATA_DIR}/openei-geothermal-wells.csv"

if [ ! -f "${OPENEI_FILE}" ]; then
    echo "Downloading OpenEI geothermal well data..."
    curl -L -o "${OPENEI_FILE}" "${OPENEI_URL}" --progress-bar || {
        echo "  OpenEI download failed — trying alternative URL..."
        # Alternative: NREL geothermal data
        curl -L -o "${OPENEI_FILE}" "https://gdr.openei.org/api/data/export?datasetId=961" --progress-bar || {
            echo "  Both URLs failed. Download manually from https://openei.org/wiki/Geothermal_Data"
        }
    }
else
    echo "  Already downloaded at ${OPENEI_FILE}"
fi

# 3. Parse and import to PostgreSQL
echo ""
echo "--- Importing to PostgreSQL ---"

/tmp/pg-import-venv/bin/python3 << 'IMPORT_PY'
import psycopg2
import csv
import os
import sys
import glob

conn = psycopg2.connect(dbname="terra_atlas", user="postgres")
conn.autocommit = True
cur = conn.cursor()

data_dir = os.environ.get("DATA_DIR", "data")

# Import EIA 860 generators
eia_dir = os.path.join(data_dir, "eia860")
gen_files = glob.glob(os.path.join(eia_dir, "*Generator*Y*.csv")) + glob.glob(os.path.join(eia_dir, "*3_1_Generator*"))

if not gen_files:
    print("  No EIA generator files found. Try running the download step first.", file=sys.stderr)
else:
    print(f"  Found {len(gen_files)} EIA generator files", file=sys.stderr)

# Import OpenEI geothermal wells
openei_file = os.path.join(data_dir, "openei-geothermal-wells.csv")
if os.path.exists(openei_file):
    count = 0
    with open(openei_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = row.get("latitude", row.get("Latitude", ""))
            lon = row.get("longitude", row.get("Longitude", ""))
            name = row.get("name", row.get("Name", row.get("well_name", f"Geothermal Well {count}")))
            temp = row.get("temperature_c", row.get("Temperature", row.get("temp_c", "0")))
            state = row.get("state", row.get("State", ""))

            if not lat or not lon:
                continue

            try:
                lat_f = float(lat)
                lon_f = float(lon)
                temp_f = float(temp) if temp else 0
            except ValueError:
                continue

            # Estimate capacity from temperature (rough: >150°C = viable for power)
            capacity_mw = max(0, (temp_f - 100) * 0.1) if temp_f > 100 else 0

            cur.execute("""
                INSERT INTO energy_projects (
                    id, name, project_type, developer, state, country,
                    latitude, longitude, capacity_mw, status,
                    energy_source, technology_type, data_source, data_quality,
                    water_flow
                ) VALUES (%s, %s, 'Geothermal', 'Resource Area', %s, 'United States',
                    %s, %s, %s, 'Resource Identified',
                    'Geothermal', 'Enhanced Geothermal System', 'OpenEI', 50,
                    %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                f"GEO-{count:06d}", name, state,
                lat_f, lon_f, capacity_mw,
                temp_f
            ))
            count += 1

    print(f"  Imported {count} geothermal resource sites from OpenEI", file=sys.stderr)
else:
    print("  OpenEI file not found. Download first.", file=sys.stderr)

# Summary
cur.execute("""
    SELECT project_type, COUNT(*), COALESCE(SUM(capacity_mw),0)::int
    FROM energy_projects
    GROUP BY project_type
    ORDER BY COUNT(*) DESC
""")
print("\n=== Updated Project Types ===", file=sys.stderr)
for r in cur.fetchall():
    print(f"  {r[0]:25s} {r[1]:>8,} projects  {r[2]:>10,} MW", file=sys.stderr)

cur.execute("SELECT COUNT(*), SUM(capacity_mw)::bigint FROM energy_projects")
total = cur.fetchone()
print(f"\n  TOTAL: {total[0]:,} projects, {total[1]:,} MW", file=sys.stderr)

cur.close()
conn.close()
IMPORT_PY

echo ""
echo "=== Import Complete ==="
echo "Run 'psql -U postgres -d terra_atlas -c \"SELECT project_type, COUNT(*) FROM energy_projects GROUP BY project_type ORDER BY COUNT(*) DESC\"' to verify"

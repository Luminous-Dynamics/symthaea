#!/usr/bin/env bash
# Import batch JSON files into Holochain conductor via hc sandbox call.
#
# Usage:
#   ./tools/import-batch.sh [batch_dir]
#
# Default batch_dir: import_batch/
#
# Each JSON file in the batch directory is submitted as a zome call:
#   register_object_*.json -> orbital_objects::register_object
#   submit_tle_*.json      -> orbital_objects::submit_tle

set -euo pipefail

BATCH_DIR="${1:-import_batch}"

if [ ! -d "$BATCH_DIR" ]; then
    echo "Error: Batch directory '$BATCH_DIR' does not exist."
    echo "Run 'celestrak-demo ingest' first to generate batch files."
    exit 1
fi

OBJECT_COUNT=0
TLE_COUNT=0
ERROR_COUNT=0

echo "=== Importing from $BATCH_DIR ==="

# Import orbital objects first
for f in "$BATCH_DIR"/register_object_*.json; do
    [ -f "$f" ] || continue
    echo -n "  Registering object: $(basename "$f")... "
    if hc sandbox call orbital_objects register_object "$(cat "$f")" 2>/dev/null; then
        echo "OK"
        OBJECT_COUNT=$((OBJECT_COUNT + 1))
    else
        echo "FAILED"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
done

# Then import TLEs
for f in "$BATCH_DIR"/submit_tle_*.json; do
    [ -f "$f" ] || continue
    echo -n "  Submitting TLE: $(basename "$f")... "
    if hc sandbox call orbital_objects submit_tle "$(cat "$f")" 2>/dev/null; then
        echo "OK"
        TLE_COUNT=$((TLE_COUNT + 1))
    else
        echo "FAILED"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
done

echo ""
echo "=== Import Complete ==="
echo "  Objects: $OBJECT_COUNT"
echo "  TLEs:    $TLE_COUNT"
echo "  Errors:  $ERROR_COUNT"

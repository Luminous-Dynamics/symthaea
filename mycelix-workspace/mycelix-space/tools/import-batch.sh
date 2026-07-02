#!/usr/bin/env bash
# Import batch JSON files into a Holochain conductor via hc sandbox call.
#
# Usage:
#   ./tools/import-batch.sh [OPTIONS] [batch_dir]
#
# Options:
#   --dry-run       Validate JSON files without submitting
#   --sandbox DIR   Path to hc sandbox (default: auto-detect)
#   --verbose       Show full JSON payloads
#   --continue      Skip files listed in .imported manifest
#   -h, --help      Show this help
#
# Default batch_dir: import_batch/
#
# File naming convention (set by celestrak-demo tool):
#   register_object_*.json       -> orbital_objects_coordinator::register_object
#   submit_tle_*.json            -> orbital_objects_coordinator::submit_tle
#   submit_observation_*.json    -> observations_coordinator::submit_observation
#
# Generate batch files:
#   cargo run -p celestrak-demo -- ingest --source stations

set -euo pipefail

# --- Configuration ---
ROLE="space_operator"
ZOME="orbital_objects_coordinator"

# --- Defaults ---
BATCH_DIR=""
DRY_RUN=false
VERBOSE=false
CONTINUE=false
SANDBOX_PATH=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --sandbox)
            SANDBOX_PATH="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --continue)
            CONTINUE=true
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            BATCH_DIR="$1"
            shift
            ;;
    esac
done

BATCH_DIR="${BATCH_DIR:-import_batch}"

# --- Validate environment ---
if [ ! -d "$BATCH_DIR" ]; then
    echo "Error: Batch directory '$BATCH_DIR' does not exist."
    echo ""
    echo "Generate batch files first:"
    echo "  cargo run -p celestrak-demo -- ingest --source stations"
    exit 1
fi

# Check for jq (used for JSON validation)
if ! command -v jq &>/dev/null; then
    echo "Warning: jq not found — skipping JSON validation"
    HAS_JQ=false
else
    HAS_JQ=true
fi

# Check for hc (unless dry-run)
if [ "$DRY_RUN" = false ] && ! command -v hc &>/dev/null; then
    echo "Error: 'hc' (Holochain CLI) not found."
    echo "Enter the dev shell first: nix develop"
    exit 1
fi

# --- Manifest for --continue mode ---
MANIFEST="$BATCH_DIR/.imported"
if [ "$CONTINUE" = true ] && [ -f "$MANIFEST" ]; then
    echo "Resuming from manifest ($(wc -l < "$MANIFEST") previously imported)"
else
    MANIFEST=""
fi

is_already_imported() {
    [ -n "$MANIFEST" ] && grep -qxF "$1" "$MANIFEST" 2>/dev/null
}

mark_imported() {
    if [ "$CONTINUE" = true ]; then
        echo "$1" >> "$BATCH_DIR/.imported"
    fi
}

# --- Auto-detect sandbox ---
SANDBOX_ARGS=""
if [ -n "$SANDBOX_PATH" ]; then
    SANDBOX_ARGS="--existing-paths $SANDBOX_PATH"
fi

# --- Counters ---
OBJECT_FILES=0
TLE_FILES=0
OBS_FILES=0
OBJECT_OK=0
TLE_OK=0
OBS_OK=0
SKIPPED=0
ERRORS=0

# --- Validate a JSON file ---
validate_json() {
    local file="$1"
    local expected_fields="$2"

    if [ "$HAS_JQ" = false ]; then
        return 0
    fi

    if ! jq empty "$file" 2>/dev/null; then
        echo "    INVALID JSON: $file"
        return 1
    fi

    # Check required field exists
    if [ -n "$expected_fields" ]; then
        local field
        for field in $expected_fields; do
            if ! jq -e ".$field" "$file" >/dev/null 2>&1; then
                echo "    Missing field '$field' in $file"
                return 1
            fi
        done
    fi

    return 0
}

# --- Submit a zome call ---
call_zome() {
    local fn_name="$1"
    local payload_file="$2"

    if [ "$VERBOSE" = true ]; then
        echo "    Payload: $(cat "$payload_file")"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "OK (dry-run)"
        return 0
    fi

    # hc sandbox call syntax:
    #   hc sandbox call [--existing-paths <dir>] <role> <zome> <fn> <payload>
    # The payload is the JSON string directly.
    local payload
    payload="$(cat "$payload_file")"

    # shellcheck disable=SC2086
    if hc sandbox call $SANDBOX_ARGS "$ROLE" "$ZOME" "$fn_name" "$payload" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

echo "=== Mycelix Space Batch Import ==="
echo "  Directory: $BATCH_DIR"
echo "  Role:      $ROLE"
echo "  Zome:      $ZOME"
[ "$DRY_RUN" = true ] && echo "  Mode:      DRY RUN (validation only)"
echo ""

# --- Phase 1: Register orbital objects ---
echo "--- Phase 1: Registering orbital objects ---"
for f in "$BATCH_DIR"/register_object_*.json; do
    [ -f "$f" ] || { echo "  No register_object files found."; break; }
    OBJECT_FILES=$((OBJECT_FILES + 1))
    basename_f="$(basename "$f")"

    if is_already_imported "$basename_f"; then
        SKIPPED=$((SKIPPED + 1))
        [ "$VERBOSE" = true ] && echo "  SKIP $basename_f (already imported)"
        continue
    fi

    echo -n "  $basename_f ... "

    if ! validate_json "$f" "norad_id name"; then
        ERRORS=$((ERRORS + 1))
        continue
    fi

    if call_zome "register_object" "$f"; then
        echo "OK"
        OBJECT_OK=$((OBJECT_OK + 1))
        mark_imported "$basename_f"
    else
        echo "FAILED"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# --- Phase 2: Submit TLEs ---
echo "--- Phase 2: Submitting TLEs ---"
for f in "$BATCH_DIR"/submit_tle_*.json; do
    [ -f "$f" ] || { echo "  No submit_tle files found."; break; }
    TLE_FILES=$((TLE_FILES + 1))
    basename_f="$(basename "$f")"

    if is_already_imported "$basename_f"; then
        SKIPPED=$((SKIPPED + 1))
        [ "$VERBOSE" = true ] && echo "  SKIP $basename_f (already imported)"
        continue
    fi

    echo -n "  $basename_f ... "

    if ! validate_json "$f" "norad_id line1 line2"; then
        ERRORS=$((ERRORS + 1))
        continue
    fi

    if call_zome "submit_tle" "$f"; then
        echo "OK"
        TLE_OK=$((TLE_OK + 1))
        mark_imported "$basename_f"
    else
        echo "FAILED"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# --- Phase 3: Submit observations ---
echo "--- Phase 3: Submitting observations ---"
OBS_ZOME="observations_coordinator"
for f in "$BATCH_DIR"/submit_observation_*.json; do
    [ -f "$f" ] || { echo "  No submit_observation files found."; break; }
    OBS_FILES=$((OBS_FILES + 1))
    basename_f="$(basename "$f")"

    if is_already_imported "$basename_f"; then
        SKIPPED=$((SKIPPED + 1))
        [ "$VERBOSE" = true ] && echo "  SKIP $basename_f (already imported)"
        continue
    fi

    echo -n "  $basename_f ... "

    if ! validate_json "$f" "sensor_id measurement"; then
        ERRORS=$((ERRORS + 1))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "OK (dry-run)"
        OBS_OK=$((OBS_OK + 1))
        mark_imported "$basename_f"
    else
        payload="$(cat "$f")"
        # shellcheck disable=SC2086
        if hc sandbox call $SANDBOX_ARGS "$ROLE" "$OBS_ZOME" "submit_observation" "$payload" 2>/dev/null; then
            echo "OK"
            OBS_OK=$((OBS_OK + 1))
            mark_imported "$basename_f"
        else
            echo "FAILED"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""

# --- Summary ---
echo "=== Import Summary ==="
echo "  Objects:      $OBJECT_OK / $OBJECT_FILES registered"
echo "  TLEs:         $TLE_OK / $TLE_FILES submitted"
echo "  Observations: $OBS_OK / $OBS_FILES submitted"
[ "$SKIPPED" -gt 0 ] && echo "  Skipped: $SKIPPED (already imported)"
echo "  Errors:  $ERRORS"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo "Some imports failed. Re-run with --verbose for details."
    echo "Use --continue to skip previously successful imports."
    exit 1
elif [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. All $((OBJECT_FILES + TLE_FILES + OBS_FILES)) files are valid."
    echo "Remove --dry-run to import into Holochain."
else
    echo "All imports successful."
fi

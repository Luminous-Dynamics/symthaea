#!/usr/bin/env bash
# Stage OTRF/Security-Datasets (Mordor-format JSON Lines inside zips) into a
# flat directory per tactic so the cluster_evtx example can pick them up.
#
# Output layout:
#   <dest>/
#     labels.csv              (basename,label — basename is a directory holding .json files)
#     <label>__<name>/
#       *.json
#
# The example's OTRF loader will walk each <label>__<name> directory and
# parse every .json inside as a JSONL stream.
#
# Usage:
#   stage_otrf_datasets.sh <src_root> <dest>
#
# <src_root> is the path to the Security-Datasets/datasets/atomic/windows
# subtree.

set -euo pipefail

SRC="${1:?usage: stage_otrf_datasets.sh <src> <dest>}"
DEST="${2:?usage: stage_otrf_datasets.sh <src> <dest>}"

mkdir -p "$DEST"
echo "basename,label" > "$DEST/labels.csv"

TACTICS=(
    "credential_access"
    "defense_evasion"
    "discovery"
    "execution"
    "lateral_movement"
    "persistence"
    "privilege_escalation"
)

count=0
for tactic in "${TACTICS[@]}"; do
    if [ ! -d "$SRC/$tactic" ]; then
        echo "warn: missing tactic dir: $tactic" >&2
        continue
    fi
    per_tactic=0
    while IFS= read -r zipf; do
        base=$(basename "$zipf" .zip)
        dest_dir="$DEST/${tactic}__${base}"
        mkdir -p "$dest_dir"
        # Extract only .json files from each zip
        unzip -q -o -j "$zipf" "*.json" -d "$dest_dir" 2>/dev/null || true
        # If nothing was extracted (zip had only pcap/cap), drop the dir
        if [ -z "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
            rmdir "$dest_dir"
            continue
        fi
        echo "${tactic}__${base},${tactic}" >> "$DEST/labels.csv"
        count=$((count + 1))
        per_tactic=$((per_tactic + 1))
    done < <(find "$SRC/$tactic" -name "*.zip" -type f)
    printf "%-22s %d zips with JSON extracted\n" "$tactic" "$per_tactic"
done

echo ""
echo "staged $count sample dirs into $DEST"
echo "labels.csv: $(wc -l < "$DEST/labels.csv") rows (including header)"

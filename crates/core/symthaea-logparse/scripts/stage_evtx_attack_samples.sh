#!/usr/bin/env bash
# Stage the sbousseaden/EVTX-ATTACK-SAMPLES repo into a flat corpus layout
# with labels.csv for the symthaea-logparse cluster_evtx example.
#
# Source layout (git repo):
#   EVTX-ATTACK-SAMPLES/
#     Command and Control/<file>.evtx
#     Credential Access/<file>.evtx
#     ...
#
# Output layout:
#   <dest>/
#     labels.csv          (filename,label header row)
#     <flat>.evtx         (one per source .evtx, prefixed to avoid collisions)
#
# Labels are MITRE ATT&CK tactic names, derived from the top-level directory.
#
# Usage:
#   scripts/stage_evtx_attack_samples.sh <repo_root> <dest_dir>

set -euo pipefail

SRC="${1:?usage: stage_evtx_attack_samples.sh <repo_root> <dest_dir>}"
DEST="${2:?usage: stage_evtx_attack_samples.sh <repo_root> <dest_dir>}"

if [ ! -d "$SRC" ]; then
    echo "error: source directory not found: $SRC" >&2
    exit 1
fi

mkdir -p "$DEST"
echo "filename,label" > "$DEST/labels.csv"

# MITRE ATT&CK tactic directories we consider labeled. "Other" and
# "AutomatedTestingTools" are intentionally excluded — they are not a single
# incident class.
TACTICS=(
    "Command and Control"
    "Credential Access"
    "Defense Evasion"
    "Discovery"
    "Execution"
    "Lateral Movement"
    "Persistence"
    "Privilege Escalation"
)

count=0
for tactic in "${TACTICS[@]}"; do
    label=$(echo "$tactic" | tr '[:upper:] ' '[:lower:]_')
    if [ ! -d "$SRC/$tactic" ]; then
        echo "warn: missing tactic dir: $tactic" >&2
        continue
    fi
    while IFS= read -r f; do
        base=$(basename "$f")
        # Prefix with label to avoid filename collisions across tactics
        dest_name="${label}__${base}"
        cp "$f" "$DEST/$dest_name"
        echo "$dest_name,$label" >> "$DEST/labels.csv"
        count=$((count + 1))
    done < <(find "$SRC/$tactic" -name "*.evtx" -type f)
    per_label=$(grep -c ",$label$" "$DEST/labels.csv" || true)
    printf "%-22s -> %s (%d files)\n" "$tactic" "$label" "$per_label"
done

echo ""
echo "staged $count files into $DEST"
echo "labels.csv: $(wc -l < "$DEST/labels.csv") rows (including header)"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 SOURCE_MANIFEST.json RESTORE_MANIFEST.json" >&2
  exit 64
fi
source_file="$1"
restore_file="$2"
[ -s "$source_file" ] && [ -s "$restore_file" ] || { echo "both manifests must exist and be non-empty" >&2; exit 66; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
jq -S . "$source_file" >"$tmp/source.json"
jq -S . "$restore_file" >"$tmp/restore.json"
if diff -u "$tmp/source.json" "$tmp/restore.json"; then
  echo "State manifests match."
else
  echo "State manifests diverge. This blocks a full-state recovery claim; investigate each typed surface." >&2
  exit 1
fi

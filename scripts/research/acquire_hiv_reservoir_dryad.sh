#!/usr/bin/env bash
set -euo pipefail

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Downloads the public Dryad deposit and extracts only its registered file names.
# It never invokes an archive extractor on arbitrary archive paths.

usage() {
  echo "Usage: $0 OUTPUT_DATA_DIR" >&2
  echo "Downloads Dryad DOI 10.5061/dryad.k3j9kd5dw using the documented public API." >&2
}

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -eq 1 ]] && exit 0
  exit 2
fi

for command in curl unzip sha256sum diff sort find awk date mktemp; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $command" >&2
    exit 2
  }
done

output_dir="$1"
parent_dir="$(dirname "$output_dir")"
base_dir="$(basename "$output_dir")"
mkdir -p "$parent_dir"
parent_dir="$(cd "$parent_dir" && pwd -P)"
output_dir="$parent_dir/$base_dir"
archive="$parent_dir/dryad-k3j9kd5dw-download.zip"
provenance="$parent_dir/dryad-k3j9kd5dw-download.provenance.txt"

if [[ -e "$output_dir" ]]; then
  if [[ ! -d "$output_dir" ]] || [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: output directory must not exist or must be empty: $output_dir" >&2
    exit 2
  fi
fi
if [[ -e "$archive" || -e "$provenance" ]]; then
  echo "ERROR: archive/provenance output already exists; refusing to overwrite it" >&2
  exit 2
fi

mkdir -p "$output_dir"
tmp="$(mktemp -d "$parent_dir/.dryad-hiv-download.XXXXXX")"
cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

url="https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.k3j9kd5dw/download"
echo "Downloading public Dryad deposit: $url"
curl --fail --location --retry 3 --retry-all-errors \
  --connect-timeout 20 --max-time 900 \
  --output "$tmp/dataset.zip" "$url"

if ! unzip -tq "$tmp/dataset.zip" >/dev/null 2>&1; then
  echo "ERROR: Dryad response was not a valid ZIP archive; no files extracted" >&2
  exit 2
fi

expected=(
  "README.md"
  "data_cd4_protein.csv"
  "data_clinical.csv"
  "data_plasma_protein.csv"
  "data_rnaseq_count.csv"
  "data_intact_dna.csv"
)

if ! diff -u \
  <(printf '%s\n' "${expected[@]}" | LC_ALL=C sort) \
  <(unzip -Z1 "$tmp/dataset.zip" | LC_ALL=C sort); then
  echo "ERROR: archive members do not exactly match the registered public file set; no files extracted" >&2
  echo "Review the current Dryad version and update the dataset register deliberately." >&2
  exit 2
fi

# Read only allow-listed member names. Never extract arbitrary paths from the archive.
for name in "${expected[@]}"; do
  unzip -p "$tmp/dataset.zip" "$name" > "$output_dir/$name"
  test -s "$output_dir/$name" || {
    echo "ERROR: extracted file is empty: $name" >&2
    exit 2
  }
done

archive_sha="$(sha256sum "$tmp/dataset.zip" | awk '{print $1}')"
mv "$tmp/dataset.zip" "$archive"
{
  printf 'dataset_doi=%s\n' '10.5061/dryad.k3j9kd5dw'
  printf 'source_url=%s\n' "$url"
  printf 'retrieved_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'archive_sha256=%s\n' "$archive_sha"
  printf 'archive_path=%s\n' "$archive"
} > "$provenance"

echo "Data files extracted to: $output_dir"
echo "Source archive: $archive"
echo "Archive SHA-256: $archive_sha"
echo "Provenance: $provenance"
echo "Next: capture a local audit receipt with symthaea-hiv-audit."

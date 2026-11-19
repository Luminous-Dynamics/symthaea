#!/usr/bin/env bash
# Reproducible external-codec setup for CDL-Bench.
# The script never clones or updates external repositories; it verifies exact
# pinned revisions in caller-provided working trees before building/running.
set -euo pipefail

readonly DEXOR_COMMIT="06128b3df57cc1e8c53681231e235e22eb508c8d"
readonly TERSETS_COMMIT="64abd7767f8fc6ac1ef4e932f2027a74cf8fb4a0"
readonly TARGET_TRIPLE="x86_64-linux"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${CDL_EXTERNAL_OUT:-$ROOT_DIR/target/cdl-external}"
DEXOR_DIR="${DEXOR_DIR:-}"
TERSETS_DIR="${TERSETS_DIR:-}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

verify_revision() {
  local dir="$1"
  local expected="$2"
  local name="$3"
  [[ -d "$dir/.git" ]] || fail "$name path is not a Git working tree: $dir"
  local actual
  actual="$(git -C "$dir" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || fail "$name revision mismatch: expected $expected, got $actual"
}

mkdir -p "$OUT_DIR"
require_command git
require_command sha256sum
require_command sort
require_command find
require_command xargs
require_command head
require_command sed
require_command cut

if [[ -n "$DEXOR_DIR" ]]; then
  require_command java
  require_command mvn
  verify_revision "$DEXOR_DIR" "$DEXOR_COMMIT" "DeXOR"

  java_line="$(java -version 2>&1 | head -n 1)"
  java_version="$(printf '%s\n' "$java_line" | sed -nE 's/.*version "([^"]+)".*/\1/p')"
  [[ -n "$java_version" ]] || fail "could not parse Java version: $java_line"
  java_major="${java_version%%.*}"
  if [[ "$java_major" == "1" ]]; then
    java_minor="$(printf '%s' "$java_version" | cut -d. -f2)"
    java_major="$java_minor"
  fi
  [[ "$java_major" == "8" ]] || fail "DeXOR README requires Java 8; found $java_version"

  dataset_dir="$DEXOR_DIR/datasets/Overall"
  [[ -d "$dataset_dir" ]] || fail "DeXOR dataset directory missing: $dataset_dir"
  mapfile -t csv_files < <(find "$dataset_dir" -maxdepth 1 -type f -name '*.csv' | sort)
  [[ "${#csv_files[@]}" -eq 22 ]] || fail "expected 22 DeXOR Overall CSV files, found ${#csv_files[@]}"

  dexor_out="$OUT_DIR/dexor"
  mkdir -p "$dexor_out/storage" "$dexor_out/log"
  (
    cd "$DEXOR_DIR"
    mvn --batch-mode clean install
  )
  jar="$DEXOR_DIR/target/OL-TSC-1.0-jar-with-dependencies.jar"
  [[ -f "$jar" ]] || fail "expected DeXOR jar missing after Maven build: $jar"

  (
    cd "$DEXOR_DIR"
    java -jar "$jar" \
      -in "$dataset_dir" \
      -out "$dexor_out/storage" \
      -log "$dexor_out/log" \
      -m DeXOR Gorilla Chimp Chimp128 Elf ElfPlus ALP 'Elf*' 'SElf*'
  )

  {
    printf 'repository=SuDIS-ZJU/DeXOR\n'
    printf 'commit=%s\n' "$DEXOR_COMMIT"
    printf 'java=%s\n' "$java_version"
    printf 'maven=%s\n' "$(mvn -version | head -n 1)"
    printf 'dataset=Overall\n'
    printf 'csv_count=%s\n' "${#csv_files[@]}"
    git -C "$DEXOR_DIR" status --short
  } > "$dexor_out/reproduction.txt"
  find "$dataset_dir" -maxdepth 1 -type f -name '*.csv' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$dexor_out/dataset-sha256.txt"
  printf 'DeXOR reproduction completed: %s\n' "$dexor_out"
else
  printf 'SKIP DeXOR: set DEXOR_DIR to a checkout at %s\n' "$DEXOR_COMMIT"
fi

if [[ -n "$TERSETS_DIR" ]]; then
  require_command zig
  require_command cc
  require_command tee
  verify_revision "$TERSETS_DIR" "$TERSETS_COMMIT" "TerseTS"
  tersets_out="$OUT_DIR/tersets"
  mkdir -p "$tersets_out"
  (
    cd "$TERSETS_DIR"
    zig build test -Dtarget="$TARGET_TRIPLE" -Doptimize=ReleaseFast
    zig build -Dtarget="$TARGET_TRIPLE" -Doptimize=ReleaseFast
  )
  library="$TERSETS_DIR/zig-out/lib/libtersets.so"
  [[ -f "$library" ]] || fail "TerseTS shared library missing after build: $library"

  gate_source="$ROOT_DIR/scripts/research/tersets_bitwise_gate.c"
  [[ -f "$gate_source" ]] || fail "TerseTS bitwise gate source missing: $gate_source"
  cc -std=c11 -O2 -Wall -Wextra -Werror \
    -I "$TERSETS_DIR/bindings/c" \
    "$gate_source" \
    -L "$TERSETS_DIR/zig-out/lib" \
    -Wl,-rpath,"$TERSETS_DIR/zig-out/lib" \
    -ltersets \
    -o "$tersets_out/tersets_bitwise_gate"
  "$tersets_out/tersets_bitwise_gate" | tee "$tersets_out/bitwise-gate.txt"

  {
    printf 'repository=cmcuza/TerseTS\n'
    printf 'commit=%s\n' "$TERSETS_COMMIT"
    printf 'zig=%s\n' "$(zig version)"
    printf 'cc=%s\n' "$(cc --version | head -n 1)"
    printf 'exact_lossless_whitelist=Uncompressed,Chimp64,Chimp128,Elf\n'
    printf 'conditional_methods=RunLengthEncoding,BitPackedDeltaEncoding\n'
    printf 'excluded_lossy_or_unverified=Camel,BitPackedQuantization,SerfQT,BitPackedBUFF,MacaqueS,MacaqueV,SwingFilter,SlidingWindow,BottomUp,DFT,and other approximation methods pending per-method proof\n'
  } > "$tersets_out/reproduction.txt"
  sha256sum "$library" > "$tersets_out/library-sha256.txt"
  printf 'TerseTS build and tests completed: %s\n' "$tersets_out"
else
  printf 'SKIP TerseTS: set TERSETS_DIR to a checkout at %s\n' "$TERSETS_COMMIT"
fi

cat <<'NOTICE'
Notes:
- DeXOR is documented as Java 8 / Maven. The README's TSFile 2.2.0-SNAPSHOT
  dependency is commented out in the pinned pom.xml; this script runs the CSV
  Overall path. The separate TSFile reproduction requires the upstream manual
  TSFile build/dependency step and is not silently enabled here.
- DeXOR currently documents only double support. Do not compare its results
  with non-f64 integer or bounded-error streams.
- TerseTS is beta. The C bitwise gate runs Uncompressed/Chimp64/Chimp128/Elf
  on IEEE-754 edge patterns and signed-zero transitions. RLE and BitPackedDelta
  are reported separately as conditional candidates, not in the strict whitelist.
- This script records environment and data hashes, but does not adapt method
  configuration, normalize timestamps, or substitute for the CDL-Bench paired
  manifest.
NOTICE

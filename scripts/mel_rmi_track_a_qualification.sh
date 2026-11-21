#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SOURCE_COMMIT="${1:?usage: scripts/mel_rmi_track_a_qualification.sh <exact-clean-head-sha>}"
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit must be a 40-character lowercase Git SHA" >&2
  exit 2
fi

ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$SOURCE_COMMIT" ]]; then
  echo "exact-head mismatch: expected $SOURCE_COMMIT, checked out $ACTUAL_HEAD" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "refusing to qualify a dirty checkout" >&2
  exit 2
fi

BASE_TMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
STAGE_ROOT="$BASE_TMP/mel-rmi-track-a-$SOURCE_COMMIT"
STAGE_CRATE="$STAGE_ROOT/crates/domains/symthaea-muse"
MANIFEST="$STAGE_CRATE/Cargo.toml"
EVIDENCE_DIR="${MEL_RMI_EVIDENCE_DIR:-$BASE_TMP/mel-rmi-track-a-evidence-$SOURCE_COMMIT}"
SEED_COUNT="${MEL_RMI_SEED_COUNT:-4}"

rm -rf "$STAGE_ROOT" "$EVIDENCE_DIR"
mkdir -p "$STAGE_CRATE/src/bin" "$EVIDENCE_DIR"

cp crates/domains/symthaea-muse/src/rmi_benchmark.rs "$STAGE_CRATE/src/rmi_benchmark.rs"
cp crates/domains/symthaea-muse/src/bin/rmi_bench_track_a.rs "$STAGE_CRATE/src/bin/rmi_bench_track_a.rs"
printf 'pub mod rmi_benchmark;\n' > "$STAGE_CRATE/src/lib.rs"

cat > "$MANIFEST" <<'CARGO_TOML'
[package]
name = "symthaea-muse"
version = "0.1.0"
edition = "2024"
publish = false
license = "AGPL-3.0-or-later"
description = "Isolated qualification harness for the Muse RMI Track A evaluator"

[lib]
path = "src/lib.rs"

[[bin]]
name = "rmi_bench_track_a"
path = "src/bin/rmi_bench_track_a.rs"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
CARGO_TOML

echo "staged_manifest=$MANIFEST"
echo "exact_source_head=$SOURCE_COMMIT"

cargo fmt --manifest-path "$MANIFEST" --check
cargo generate-lockfile --manifest-path "$MANIFEST"
cargo test --locked --manifest-path "$MANIFEST" --lib rmi_benchmark::tests
cargo test --locked --manifest-path "$MANIFEST" --bin rmi_bench_track_a
cargo clippy --locked --manifest-path "$MANIFEST" --all-targets -- -D warnings

REPORT="$EVIDENCE_DIR/melothaea-rmi-track-a.json"
cargo run --locked --manifest-path "$MANIFEST" --bin rmi_bench_track_a -- \
  --source-commit "$SOURCE_COMMIT" \
  --seed-count "$SEED_COUNT" \
  --output "$REPORT" \
  --self-check

cp "$MANIFEST" "$EVIDENCE_DIR/Cargo.toml"
cp "$STAGE_CRATE/Cargo.lock" "$EVIDENCE_DIR/Cargo.lock"
cp "$STAGE_CRATE/src/lib.rs" "$EVIDENCE_DIR/lib.rs"
cp "$STAGE_CRATE/src/rmi_benchmark.rs" "$EVIDENCE_DIR/rmi_benchmark.rs"
cp "$STAGE_CRATE/src/bin/rmi_bench_track_a.rs" "$EVIDENCE_DIR/rmi_bench_track_a.rs"
sha256sum \
  "$EVIDENCE_DIR/Cargo.toml" \
  "$EVIDENCE_DIR/Cargo.lock" \
  "$EVIDENCE_DIR/lib.rs" \
  "$EVIDENCE_DIR/rmi_benchmark.rs" \
  "$EVIDENCE_DIR/rmi_bench_track_a.rs" \
  "$REPORT" > "$EVIDENCE_DIR/SHA256SUMS"
printf 'exact_head=%s\nseed_count=%s\n' "$SOURCE_COMMIT" "$SEED_COUNT" > "$EVIDENCE_DIR/source-identities.txt"

echo "RMI Track A qualification completed."
echo "report=$REPORT"
echo "evidence=$EVIDENCE_DIR"

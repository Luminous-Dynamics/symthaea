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
STAGE_ROOT="$(mktemp -d "${BASE_TMP%/}/mel-rmi-track-a.XXXXXX")"
STAGE_CRATE="$STAGE_ROOT/crates/domains/symthaea-muse"
MANIFEST="$STAGE_CRATE/Cargo.toml"
EVIDENCE_DIR="${MEL_RMI_EVIDENCE_DIR:-$BASE_TMP/mel-rmi-track-a-evidence-$SOURCE_COMMIT}"
TRIALS_PER_TARGET="${MEL_RMI_TRIALS_PER_TARGET:-8}"
SEED_ROOT="${MEL_RMI_SEED_ROOT:-42}"

if [[ -e "$EVIDENCE_DIR" ]]; then
  echo "evidence destination already exists; refusing to overwrite: $EVIDENCE_DIR" >&2
  exit 2
fi
mkdir -p "$STAGE_CRATE/src/bin" "$STAGE_CRATE/tests" "$EVIDENCE_DIR"

cp crates/domains/symthaea-muse/src/rmi_benchmark.rs "$STAGE_CRATE/src/rmi_benchmark.rs"
cp crates/domains/symthaea-muse/src/rmi_track_a_runner.rs "$STAGE_CRATE/src/rmi_track_a_runner.rs"
cp crates/domains/symthaea-muse/src/bin/rmi_track_a.rs "$STAGE_CRATE/src/bin/rmi_track_a.rs"
cp crates/domains/symthaea-muse/tests/rmi_track_a_replay.rs "$STAGE_CRATE/tests/rmi_track_a_replay.rs"
printf 'pub mod rmi_benchmark;\npub mod rmi_track_a_runner;\n' > "$STAGE_CRATE/src/lib.rs"

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
name = "rmi_track_a"
path = "src/bin/rmi_track_a.rs"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
CARGO_TOML

echo "staged_manifest=$MANIFEST"
echo "exact_source_head=$SOURCE_COMMIT"

# Preserve original repository source bytes before formatting the isolated
# staging copy. The formatted staging copy is what is compiled and tested.
cp "$STAGE_CRATE/src/rmi_benchmark.rs" "$EVIDENCE_DIR/rmi_benchmark.source.rs"
cp "$STAGE_CRATE/src/rmi_track_a_runner.rs" "$EVIDENCE_DIR/rmi_track_a_runner.source.rs"
cp "$STAGE_CRATE/src/bin/rmi_track_a.rs" "$EVIDENCE_DIR/rmi_track_a.source.rs"
cp "$STAGE_CRATE/tests/rmi_track_a_replay.rs" "$EVIDENCE_DIR/rmi_track_a_replay.source.rs"
SOURCE_RMI_BENCHMARK_SHA256="$(sha256sum "$STAGE_CRATE/src/rmi_benchmark.rs" | awk '{print $1}')"
SOURCE_RMI_RUNNER_SHA256="$(sha256sum "$STAGE_CRATE/src/rmi_track_a_runner.rs" | awk '{print $1}')"
SOURCE_RMI_CLI_SHA256="$(sha256sum "$STAGE_CRATE/src/bin/rmi_track_a.rs" | awk '{print $1}')"
SOURCE_RMI_REPLAY_TEST_SHA256="$(sha256sum "$STAGE_CRATE/tests/rmi_track_a_replay.rs" | awk '{print $1}')"

# Rustfmt normalization is performed only in the temporary isolated crate.
# Both original and normalized source digests are retained in the evidence.
cargo fmt --manifest-path "$MANIFEST"
cargo fmt --manifest-path "$MANIFEST" --check
FORMATTED_RMI_BENCHMARK_SHA256="$(sha256sum "$STAGE_CRATE/src/rmi_benchmark.rs" | awk '{print $1}')"
FORMATTED_RMI_RUNNER_SHA256="$(sha256sum "$STAGE_CRATE/src/rmi_track_a_runner.rs" | awk '{print $1}')"
FORMATTED_RMI_CLI_SHA256="$(sha256sum "$STAGE_CRATE/src/bin/rmi_track_a.rs" | awk '{print $1}')"
FORMATTED_RMI_REPLAY_TEST_SHA256="$(sha256sum "$STAGE_CRATE/tests/rmi_track_a_replay.rs" | awk '{print $1}')"
cargo generate-lockfile --manifest-path "$MANIFEST"

LOCK_SHA256="$(sha256sum "$STAGE_CRATE/Cargo.lock" | awk '{print $1}')"
export SYMTHAEA_MUSE_GIT_REV="$SOURCE_COMMIT"
export RUSTC_VERSION="$(rustc --version)"
export MUSE_RMI_CARGO_LOCK_SHA256="$LOCK_SHA256"

cargo test --locked --manifest-path "$MANIFEST" --lib rmi_benchmark::tests
cargo test --locked --manifest-path "$MANIFEST" --lib rmi_track_a_runner::tests
cargo test --locked --manifest-path "$MANIFEST" --bin rmi_track_a
cargo test --locked --manifest-path "$MANIFEST" --test rmi_track_a_replay
cargo clippy --locked --manifest-path "$MANIFEST" --all-targets -- -D warnings

REPORT="$EVIDENCE_DIR/melothaea-rmi-track-a.json"
cargo run --locked --manifest-path "$MANIFEST" --bin rmi_track_a -- "$TRIALS_PER_TARGET" "$SEED_ROOT" "$REPORT"

cp "$MANIFEST" "$EVIDENCE_DIR/Cargo.toml"
cp "$STAGE_CRATE/Cargo.lock" "$EVIDENCE_DIR/Cargo.lock"
cp "$STAGE_CRATE/src/lib.rs" "$EVIDENCE_DIR/lib.rs"
cp "$STAGE_CRATE/src/rmi_benchmark.rs" "$EVIDENCE_DIR/rmi_benchmark.formatted.rs"
cp "$STAGE_CRATE/src/rmi_track_a_runner.rs" "$EVIDENCE_DIR/rmi_track_a_runner.formatted.rs"
cp "$STAGE_CRATE/src/bin/rmi_track_a.rs" "$EVIDENCE_DIR/rmi_track_a.formatted.rs"
cp "$STAGE_CRATE/tests/rmi_track_a_replay.rs" "$EVIDENCE_DIR/rmi_track_a_replay.formatted.rs"
(
  cd "$EVIDENCE_DIR"
  sha256sum \
    Cargo.toml \
    Cargo.lock \
    lib.rs \
    rmi_benchmark.source.rs \
    rmi_track_a_runner.source.rs \
    rmi_track_a.source.rs \
    rmi_track_a_replay.source.rs \
    rmi_benchmark.formatted.rs \
    rmi_track_a_runner.formatted.rs \
    rmi_track_a.formatted.rs \
    rmi_track_a_replay.formatted.rs \
    melothaea-rmi-track-a.json > SHA256SUMS
)
printf 'exact_head=%s\ntrials_per_target=%s\nseed_root=%s\nrustc=%s\ncargo_lock_sha256=%s\nsource_rmi_benchmark_sha256=%s\nsource_runner_sha256=%s\nsource_cli_sha256=%s\nsource_replay_test_sha256=%s\nformatted_rmi_benchmark_sha256=%s\nformatted_runner_sha256=%s\nformatted_cli_sha256=%s\nformatted_replay_test_sha256=%s\n' \
  "$SOURCE_COMMIT" "$TRIALS_PER_TARGET" "$SEED_ROOT" "$RUSTC_VERSION" "$LOCK_SHA256" \
  "$SOURCE_RMI_BENCHMARK_SHA256" "$SOURCE_RMI_RUNNER_SHA256" "$SOURCE_RMI_CLI_SHA256" "$SOURCE_RMI_REPLAY_TEST_SHA256" \
  "$FORMATTED_RMI_BENCHMARK_SHA256" "$FORMATTED_RMI_RUNNER_SHA256" "$FORMATTED_RMI_CLI_SHA256" "$FORMATTED_RMI_REPLAY_TEST_SHA256" \
  > "$EVIDENCE_DIR/source-identities.txt"

echo "RMI Track A qualification completed."
echo "report=$REPORT"
echo "evidence=$EVIDENCE_DIR"

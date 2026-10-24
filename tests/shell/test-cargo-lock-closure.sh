#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$ROOT/scripts/check_cargo_lock_closure.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

setup_case() {
  local case_root="$1"
  mkdir -p "$case_root/bin"
  cd "$case_root"
  git init --quiet
  git config user.name "Cargo lock closure test"
  git config user.email "cargo-lock-test@example.invalid"
  printf '[workspace]\n' > Cargo.toml
  printf 'committed-lock\n' > Cargo.lock
  git add Cargo.toml Cargo.lock
  git commit --quiet -m "fixture base"

  cat > "$case_root/bin/fake-cargo" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-V" || "${1:-}" == "--version" ]]; then
  echo "cargo 1.96.0 (test stub)"
  exit 0
fi
[[ "${1:-}" == "metadata" ]] || exit 99
locked=0
for arg in "$@"; do
  [[ "$arg" == "--locked" ]] && locked=1
done
mode="${CARGO_STUB_MODE:-drift}"
lock="$(cat Cargo.lock)"
case "$mode" in
  clean)
    if [[ "$locked" == "1" && "$lock" == "committed-lock" ]]; then
      printf '{"stub":true}\n'
      exit 0
    fi
    echo "unexpected clean-mode Cargo call" >&2
    exit 98
    ;;
  drift)
    if [[ "$locked" == "1" && "$lock" == "proposed-lock" ]]; then
      printf '{"stub":true}\n'
      exit 0
    fi
    if [[ "$locked" == "1" ]]; then
      echo "cannot update lock file because --locked was passed" >&2
      exit 101
    fi
    printf 'proposed-lock\n' > Cargo.lock
    printf '{"stub":true}\n'
    exit 0
    ;;
  resolution)
    echo "synthetic dependency resolution failure" >&2
    exit 101
    ;;
  *)
    echo "unknown stub mode" >&2
    exit 97
    ;;
esac
FAKE
  chmod +x "$case_root/bin/fake-cargo"
}

clean_root="$TMP/clean"
setup_case "$clean_root"
CARGO_LOCK_CLOSURE_ROOT="$clean_root" CARGO_BIN="$clean_root/bin/fake-cargo" CARGO_STUB_MODE=clean \
  bash "$HELPER" target/cargo-lock-closure > "$TMP/clean.stdout"
grep -q '"classification": "LOCK_CLEAN"' "$clean_root/target/cargo-lock-closure/report.json" \
  || fail "clean lock was not classified LOCK_CLEAN"
grep -q '"projection_metadata_exit_code": null' "$clean_root/target/cargo-lock-closure/report.json" \
  || fail "clean report should mark projection as not run"
grep -q '"proposed_locked_metadata_exit_code": null' "$clean_root/target/cargo-lock-closure/report.json" \
  || fail "clean report should mark proposed-lock validation as not run"
[[ -s "$clean_root/target/cargo-lock-closure/toolchain.txt" ]] || fail "toolchain provenance is missing"
clean_metadata_sha="$(sha256sum "$clean_root/target/cargo-lock-closure/metadata-locked.json" | awk '{print $1}')"
grep -q "\"locked_metadata_sha256\": \"$clean_metadata_sha\"" "$clean_root/target/cargo-lock-closure/report.json" \
  || fail "clean report is not bound to its metadata bytes"
[[ "$(cat "$clean_root/Cargo.lock")" == "committed-lock" ]] || fail "clean lock was altered"

drift_root="$TMP/drift"
setup_case "$drift_root"
set +e
CARGO_LOCK_CLOSURE_ROOT="$drift_root" CARGO_BIN="$drift_root/bin/fake-cargo" CARGO_STUB_MODE=drift \
  bash "$HELPER" target/cargo-lock-closure > "$TMP/drift.stdout" 2> "$TMP/drift.stderr"
drift_status=$?
set -e
[[ "$drift_status" -eq 1 ]] || fail "lock drift should exit 1, got $drift_status"
grep -q '"classification": "FAIL_LOCK_DRIFT"' "$drift_root/target/cargo-lock-closure/report.json" \
  || fail "lock drift was not classified FAIL_LOCK_DRIFT"
grep -q '"projection_metadata_exit_code": 0' "$drift_root/target/cargo-lock-closure/report.json" \
  || fail "drift report should record successful unlocked projection"
grep -q '"proposed_locked_metadata_exit_code": 0' "$drift_root/target/cargo-lock-closure/report.json" \
  || fail "drift report should record successful proposed-lock validation"
drift_metadata_sha="$(sha256sum "$drift_root/target/cargo-lock-closure/metadata-projection.json" | awk '{print $1}')"
proposed_metadata_sha="$(sha256sum "$drift_root/target/cargo-lock-closure/metadata-proposed-locked.json" | awk '{print $1}')"
grep -q "\"projection_metadata_sha256\": \"$drift_metadata_sha\"" "$drift_root/target/cargo-lock-closure/report.json" \
  || fail "drift report is not bound to the unlocked projection bytes"
grep -q "\"proposed_locked_metadata_sha256\": \"$proposed_metadata_sha\"" "$drift_root/target/cargo-lock-closure/report.json" \
  || fail "drift report is not bound to the proposed locked metadata bytes"
[[ "$(cat "$drift_root/Cargo.lock")" == "committed-lock" ]] || fail "helper did not restore the committed lock"
[[ "$(cat "$drift_root/target/cargo-lock-closure/proposed-Cargo.lock")" == "proposed-lock" ]] \
  || fail "Cargo-generated proposed lock was not retained"
[[ -s "$drift_root/target/cargo-lock-closure/Cargo.lock.patch" ]] || fail "lock diff was not retained"
[[ -z "$(git -C "$drift_root" diff --name-only)" ]] || fail "helper left a tracked worktree change"

resolution_root="$TMP/resolution"
setup_case "$resolution_root"
set +e
CARGO_LOCK_CLOSURE_ROOT="$resolution_root" CARGO_BIN="$resolution_root/bin/fake-cargo" CARGO_STUB_MODE=resolution \
  bash "$HELPER" target/cargo-lock-closure > "$TMP/resolution.stdout" 2> "$TMP/resolution.stderr"
resolution_status=$?
set -e
[[ "$resolution_status" -eq 2 ]] || fail "resolution failure should exit 2, got $resolution_status"
grep -q '"classification": "FAIL_METADATA_OR_RESOLUTION"' "$resolution_root/target/cargo-lock-closure/report.json" \
  || fail "resolution failure was not classified separately"
grep -q '"projection_metadata_exit_code": 101' "$resolution_root/target/cargo-lock-closure/report.json" \
  || fail "resolution report should preserve the projection exit code"
grep -q '"proposed_locked_metadata_exit_code": null' "$resolution_root/target/cargo-lock-closure/report.json" \
  || fail "resolution report should mark proposed-lock validation as not run"
[[ "$(cat "$resolution_root/Cargo.lock")" == "committed-lock" ]] || fail "resolution failure altered the committed lock"

echo "PASS: Cargo lock-closure helper preserves clean locks, captures Cargo-generated drift, restores the checkout, and distinguishes resolution failures"

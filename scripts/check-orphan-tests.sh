#!/usr/bin/env bash
# check-orphan-tests.sh — fail if tests/ holds a file cargo never builds.
#
# WHY: symthaea's Cargo.toml sets `autotests = false`, so a file in tests/ becomes
# a test target ONLY if it has an explicit `[[test]] name = "..."` entry. Without
# one it is not compiled, not run, and produces no warning of any kind. It looks
# exactly like a test — it lives in tests/, it is named like a test, it contains
# #[test] functions — and it is inert.
#
# This is the tests/ counterpart of check-orphan-modules.sh (the src/ variant).
# Same failure class, different location, and this one is much larger: measured
# against `cargo metadata` on 2026-07-30, 166 of 204 files in tests/ were dead.
#
# The cost is not hypothetical. Three known instances:
#   * embodiment_moral_contract.rs — a CROSS-PLATFORM SAFETY CONTRACT verifying
#     every EmbodimentBridge honours apply_moral_gate. It exists specifically to
#     catch a platform inheriting the trait's no-op default. It had never run.
#     Exactly that gap was found by hand in symthaea-gravcraft on 2026-07-29.
#   * binary_service.rs — a CI job invoked `cargo test --test binary_service` for
#     months against a target that did not exist, failing with "no test target
#     named `binary_service`" rather than running anything.
#   * vision_manifold_integration.rs + foveal_bridge_integration.rs — "24+22 tests
#     that never ran anywhere until registered" (Cargo.toml's own comment, 07-15).
#
# Each time, the two or three files in front of someone got registered and nobody
# measured the denominator. This script measures the denominator.
#
# USE:
#   scripts/check-orphan-tests.sh            # fail on any NEW orphan
#   scripts/check-orphan-tests.sh --list     # list every orphan, exit 0
#   scripts/check-orphan-tests.sh --count    # just the number, exit 0
#
# QUARANTINE: the 166 cannot be registered at once — many will not compile after
# months of drift, and registering a broken target turns a silent problem into a
# red build for everyone. So this script fails only on orphans NOT in the
# quarantine list below, i.e. it is a ratchet: it stops the population growing
# while the backlog is worked down. REMOVE entries as they are registered; never
# add one without a reason.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

MODE="${1:-check}"

# Authoritative: ask cargo what targets exist rather than grepping Cargo.toml,
# which would miss path-overridden or generated targets and mis-handle comments.
declared="$(cargo metadata --no-deps --format-version 1 2>/dev/null \
  | python3 -c '
import json,sys
try: m=json.load(sys.stdin)
except Exception: sys.exit(0)
for p in m.get("packages",[]):
    if p.get("name")!="symthaea": continue
    for t in p.get("targets",[]):
        if "test" in t.get("kind",[]): print(t["name"])
')"

if [[ -z "$declared" ]]; then
  echo "check-orphan-tests: cargo metadata produced no test targets for package 'symthaea'." >&2
  echo "  That usually means the workspace itself is broken — run" >&2
  echo "  scripts/check-workspace-targets.sh, which works when cargo metadata cannot." >&2
  exit 2
fi

orphans=()
for f in tests/*.rs; do
  [[ -e "$f" ]] || continue
  name="$(basename "$f" .rs)"
  grep -qxF "$name" <<<"$declared" && continue
  # Only count files that actually contain tests; a bare helper module in tests/
  # is not a dead test, it is (badly placed) shared code.
  grep -qE '^\s*#\[(test|tokio::test)\]' "$f" || continue
  orphans+=("$name")
done

total_files="$(find tests -maxdepth 1 -name '*.rs' | wc -l)"
n_declared="$(wc -l <<<"$declared")"

case "$MODE" in
  --count) echo "${#orphans[@]}"; exit 0 ;;
  --list)
    printf '%s\n' "${orphans[@]:-}"
    exit 0 ;;
esac

# ── Quarantine: the pre-existing backlog, recorded so the ratchet can bite ─────
# Generated 2026-07-31. Everything here is a KNOWN dead test file. The list is
# deliberately not curated by hand — shrinking it is the work.
QUARANTINE_FILE="scripts/orphan-tests-quarantine.txt"
if [[ ! -f "$QUARANTINE_FILE" ]]; then
  echo "check-orphan-tests: missing $QUARANTINE_FILE — cannot distinguish new orphans" >&2
  echo "  from the known backlog. Regenerate with:" >&2
  echo "    scripts/check-orphan-tests.sh --list > $QUARANTINE_FILE" >&2
  exit 2
fi

new=()
for o in "${orphans[@]:-}"; do
  grep -qxF "$o" "$QUARANTINE_FILE" || new+=("$o")
done

echo "tests/*.rs files:        $total_files"
echo "cargo test targets:      $n_declared"
echo "orphaned (never built):  ${#orphans[@]}"
echo "  of which quarantined:  $(( ${#orphans[@]} - ${#new[@]} ))"
echo "  of which NEW:          ${#new[@]}"

if (( ${#new[@]} > 0 )); then
  echo
  echo "FAIL: these tests/ files contain #[test] functions but no [[test]] target," >&2
  echo "so cargo never builds them and they will never run:" >&2
  printf '  - tests/%s.rs\n' "${new[@]}" >&2
  echo >&2
  echo "Fix: add to symthaea/Cargo.toml" >&2
  echo "  [[test]]" >&2
  echo "  name = \"<file stem>\"" >&2
  echo "  required-features = [...]   # omit if the file is internally #[cfg]-gated" >&2
  exit 1
fi

# Shrinking the quarantine is progress; report it so it is visible in CI logs.
stale=()
while IFS= read -r q; do
  [[ -z "$q" ]] && continue
  printf '%s\n' "${orphans[@]:-}" | grep -qxF "$q" || stale+=("$q")
done < "$QUARANTINE_FILE"
if (( ${#stale[@]} > 0 )); then
  echo
  echo "${#stale[@]} quarantine entries are no longer orphaned — registered or removed."
  echo "Please drop them from $QUARANTINE_FILE so the ratchet keeps tightening:"
  printf '  %s\n' "${stale[@]}"
  echo
  echo "  CAVEAT: re-run this before acting on the list. This repo runs 12+ concurrent"
  echo "  sessions, and orphan status is computed from \`cargo metadata\`, which reads"
  echo "  the WORKING-TREE Cargo.toml. If another session has an uncommitted [[test]]"
  echo "  entry in flight, a still-orphaned file is reported here as clean. Observed"
  echo "  2026-07-31: acting on one such entry immediately turned the gate red, which"
  echo "  is the ratchet catching the bad edit — but a re-run avoids the round trip."
fi

echo
echo "OK: no NEW orphaned test files."
exit 0

#!/usr/bin/env bash
# check-orphan-examples.sh — fail if examples/ holds a file cargo never builds.
#
# WHY: symthaea's Cargo.toml sets `autoexamples = false` (line 183), so a file in
# examples/ becomes a target ONLY with an explicit declaration. Without one it is
# not compiled, not run, and produces no warning of any kind. This is the
# examples/ counterpart of check-orphan-tests.sh (tests/) and
# check-orphan-modules.sh (src/) — same failure class, third location.
#
# Measured 2026-07-31: 454 files in examples/, 239 live, 215 dead.
#
# The cost is not hypothetical, and it reaches published claims:
#   * CLAUDE.md cited examples/benchmark_sleepstage.rs (Sleep-EDF 70-80%) and
#     examples/benchmark_arc_reasoning.rs (ARC-AGI 2-AFC+strict) as EXTERNAL
#     VALIDATION. Both are dead — they cannot be built or run at HEAD. So are
#     benchmark_moral_unified.rs and dmc_benchmark_report.rs.
#   * The one live benchmark, benchmark_ethics_moral_algebra, is precisely the
#     one that was independently re-run — and that re-run RETRACTED an inflated
#     94.5% down to an honest 56.2%. The unreproducible ones never got that
#     treatment. That asymmetry is the argument for this script.
#   * examples/evolve_consciousness_equation.rs is cited by
#     THE_SUBSTRATE_QUICKREF.md, which is @-imported into every session.
#
# ── WHY src_path MATCHING, NOT NAME MATCHING ─────────────────────────────────
# Liveness MUST be decided by `src_path` across ALL target kinds. Five
# kokoro_singing_* files live in examples/ but are declared `[[bin]]`
# (Cargo.toml:2007-2016) and DO compile. Comparing `[[example]]` names to
# filenames reports them as dead — an error made on 2026-07-31 that put a wrong
# number (220 instead of 215) into CLAUDE.md before it was caught. Counting by
# directory instead of by target is the same class of mistake this script exists
# to prevent, so it is worth stating twice.
#
# USE:
#   scripts/check-orphan-examples.sh          # ratchet: fails only on NEW orphans
#   scripts/check-orphan-examples.sh --count  # integer, for scripting
#   scripts/check-orphan-examples.sh --list   # names, for regenerating quarantine
#
# EXIT: 0 clean · 1 new orphan found · 2 harness could not run

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

MODE="${1:-check}"

# Authoritative: ask cargo which files are real targets. Emitting src_path (not
# name) is what makes the [[bin]]-in-examples/ case come out right.
declared="$(cargo metadata --no-deps --offline --format-version 1 2>/dev/null \
  | python3 -c '
import json,sys,os
try: m=json.load(sys.stdin)
except Exception: sys.exit(0)
for p in m.get("packages",[]):
    if p.get("name")!="symthaea": continue
    for t in p.get("targets",[]):
        sp=t.get("src_path","")
        if os.sep+"examples"+os.sep in sp:
            print(os.path.realpath(sp))
')"

if [[ -z "$declared" ]]; then
  echo "check-orphan-examples: cargo metadata produced no example targets for 'symthaea'." >&2
  echo "  That usually means the workspace itself is broken — run" >&2
  echo "  scripts/check-workspace-targets.sh, which works when cargo metadata cannot." >&2
  exit 2
fi

orphans=()
for f in examples/*.rs; do
  [[ -e "$f" ]] || continue
  real="$(realpath "$f")"
  grep -qxF "$real" <<<"$declared" && continue
  orphans+=("$(basename "$f" .rs)")
done

total_files="$(find examples -maxdepth 1 -name '*.rs' | wc -l)"
n_declared="$(wc -l <<<"$declared")"

case "$MODE" in
  --count) echo "${#orphans[@]}"; exit 0 ;;
  --list)
    printf '%s\n' "${orphans[@]:-}"
    exit 0 ;;
esac

# ── Quarantine: the pre-existing backlog, recorded so the ratchet can bite ─────
# Everything listed is a KNOWN dead example. Deliberately not hand-curated —
# shrinking it is the work. Same shape as orphan-tests-quarantine.txt.
QUARANTINE_FILE="scripts/orphan-examples-quarantine.txt"
if [[ ! -f "$QUARANTINE_FILE" ]]; then
  echo "check-orphan-examples: missing $QUARANTINE_FILE — cannot distinguish new" >&2
  echo "  orphans from the known backlog. Regenerate with:" >&2
  echo "    scripts/check-orphan-examples.sh --list > $QUARANTINE_FILE" >&2
  exit 2
fi

new=()
for o in "${orphans[@]:-}"; do
  grep -qxF "$o" "$QUARANTINE_FILE" || new+=("$o")
done

echo "examples/*.rs files:     $total_files"
echo "cargo targets in examples/: $n_declared"
echo "orphaned (never built):  ${#orphans[@]}"
echo "  of which quarantined:  $(( ${#orphans[@]} - ${#new[@]} ))"
echo "  of which NEW:          ${#new[@]}"

if (( ${#new[@]} > 0 )); then
  echo
  echo "FAIL: these examples/ files are not any cargo target, so they are never" >&2
  echo "built and any claim resting on them is unreproducible:" >&2
  printf '  - examples/%s.rs\n' "${new[@]}" >&2
  echo >&2
  echo "Fix: add to symthaea/Cargo.toml" >&2
  echo "  [[example]]" >&2
  echo "  name = \"<file stem>\"" >&2
  echo "  required-features = [...]   # omit if internally #[cfg]-gated" >&2
  echo >&2
  echo "Or delete the file if it is genuinely obsolete. Do not add it to the" >&2
  echo "quarantine to silence this — the quarantine is the backlog, not an" >&2
  echo "allowlist for new dead code." >&2
  exit 1
fi

# Shrinking the quarantine is progress; report it so it is visible in CI logs.
stale=()
while IFS= read -r q; do
  [[ -n "$q" ]] || continue
  printf '%s\n' "${orphans[@]:-}" | grep -qxF "$q" || stale+=("$q")
done < "$QUARANTINE_FILE"

if (( ${#stale[@]} > 0 )); then
  echo
  echo "PROGRESS: ${#stale[@]} quarantined example(s) are no longer orphaned."
  echo "Remove them from $QUARANTINE_FILE so the ratchet keeps its grip:"
  printf '  - %s\n' "${stale[@]}"
fi

echo
echo "OK: no new orphaned examples."

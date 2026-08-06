#!/usr/bin/env bash
# check-orphan-modules.sh — fail if a crate contains a .rs file no `mod` declares.
#
# WHY: rustc never warns about a source file that no module declares. The file
# is simply not part of the crate — it does not compile, its tests never run,
# and it silently bit-rots against APIs that move underneath it. Nothing in the
# normal build or test cycle can detect this, which is exactly why it persists.
#
# Three real instances found on 2026-07-28, all in shipped code:
#   * symthaea-muse-ui: harmony_view/motifs_view/orchestration_view — three
#     Research views, each landed in its own feat commit on 2026-07-16, dropped
#     from main.rs on 2026-07-21 while unbreaking the build, unnoticed for a
#     week. harmony_view had already rotted (it called an api:: function that
#     no longer existed) by the time it was found.
#   * symthaea-muse-protocol: catalog.rs — 1,257 lines holding the entire
#     "Muse 152" taxonomy plus its own `assert_eq!(CATALOG.len(), 152)` test.
#     `pub mod catalog;` was NEVER added; it has never compiled since the day
#     it landed, while a design doc described it as shipped.
#   * symthaea-music-theory: evidence_calibration.rs is a one-line
#     `// placeholder` sitting beside evidence_calibration/, a 27-file,
#     5,699-line subtree with no mod.rs — so the whole subtree is unreachable.
#     That second shape (empty module root + populated directory) is checked
#     separately below.
#
# All three arrived via "apply patch series" style commits, which is the common
# thread: hand-porting work between branches drops the wiring silently.
#
# USE:
#   scripts/check-orphan-modules.sh                    # every crate except QUARANTINE
#   scripts/check-orphan-modules.sh crates/domains/foo # one crate, quarantine ignored
#   scripts/check-orphan-modules.sh --all              # every crate, quarantine ignored
#   scripts/check-orphan-modules.sh --all --summary    # counts only, never fails
#
# SCOPE: the bare invocation is the CI gate — it enforces on all ~200 workspace
# members minus the QUARANTINE list. Naming a crate explicitly (or --all) ignores
# quarantine, so you can see a crate's true state while cleaning it. --summary
# is the backlog view and never fails.
#
# No build required — pure filesystem scan.
set -uo pipefail
shopt -s nullglob globstar
cd "$(dirname "$0")/.."  # -> symthaea/

# The gate enforces on EVERY workspace member except the quarantine below.
#
# It started opt-in (three muse crates) on the assumption most of the workspace
# was dirty. Measured 2026-07-29: 172 of 200 crates were already orphan-clean,
# so opt-in had it backwards — it left 172 crates unprotected to accommodate 28.
# Inverted: new crates are covered by default, and the quarantine is a visible,
# shrinking list rather than an invisible majority.
DEFAULT_SCOPE=(crates/core/* crates/domains/* crates/bridges/*)

# Crates with a pre-existing orphan backlog, exempt until cleaned. Deleting a
# line here should make the gate pass; if it doesn't, that crate regressed.
#
# Do NOT add to this list to make a build green — that is the exact reflex this
# whole gate exists to catch. Add the missing `mod`, or delete the file.
QUARANTINE=(
  crates/core/symthaea-hdc-ltc
  crates/domains/symthaea-aesthetic
  crates/domains/symthaea-broca
  crates/domains/symthaea-consciousness-resonance
  crates/domains/symthaea-consciousness-topology
  crates/domains/symthaea-dream
  crates/domains/symthaea-humanoid
  crates/domains/symthaea-perception
  crates/domains/symthaea-prime-gap-lab
  crates/domains/symthaea-psych-bench
  crates/domains/symthaea-quadruped
  crates/domains/symthaea-quantum-comp
  crates/domains/symthaea-spore
  crates/domains/symthaea-subterranean
  crates/domains/symthaea-vocal-tract
  crates/bridges/symthaea-bevy-dash
  crates/bridges/symthaea-mycelix-bridge
  crates/bridges/symthaea-web
)

is_quarantined() {
  local d="${1%/}" q
  for q in "${QUARANTINE[@]}"; do [[ "$d" == "$q" ]] && return 0; done
  return 1
}

# KNOWN, DELIBERATE orphans: reported on every run but do not fail the gate.
#
# The point of this list is that it is the OPPOSITE of how the 214 orphans got
# there. Silent orphaning is the bug; a named entry with a reason and a stated
# cost to fix is a decision. Keep it short — anything here that nobody can
# justify should be deleted, not carried.
KNOWN_ORPHANS=(
  # 154 lines. The evidence-backed VisualizationKind/EvidenceRequirement system,
  # deliberately deferred by the 2026-07-21 Listen Mode review as "bigger scope
  # than a follow-up round": wiring it means unifying it with the simpler
  # `VizMode` enum the Listen page actually uses. Its protocol imports all
  # exist, so it should still compile — this is a design decision, not rot.
  "crates/domains/symthaea-muse-ui/src/visualization.rs"
  # 331 lines. Arrived via patch bundle 82330c0c0a and is fully aspirational,
  # not merely unwired: it calls api::{fetch_foundry_qualification,
  # foundry_qualification_audio_url} and api::{FoundryQualificationEntry,
  # FoundryQualificationReveal}, none of which exist in api.rs on any branch
  # reachable from here, AND muse_studio has no matching endpoint (its only
  # "foundry" references are the unrelated use_motif_foundry compose flag).
  # Restoring it means building a backend endpoint first, so it is a feature,
  # not a repair.
  "crates/domains/symthaea-muse-ui/src/foundry_review_page.rs"
)

is_known_orphan() {
  local f="${1#./}" k
  for k in "${KNOWN_ORPHANS[@]}"; do [[ "$f" == "$k" ]] && return 0; done
  return 1
}

summary=0
explicit_scope=0
scope=()
for arg in "$@"; do
  case "$arg" in
    --summary) summary=1 ;;
    --all)     scope+=(crates/core/* crates/domains/* crates/bridges/*); explicit_scope=1 ;;
    -*)        echo "unknown flag: $arg" >&2; exit 2 ;;
    *)         scope+=("$arg"); explicit_scope=1 ;;
  esac
done
[[ ${#scope[@]} -eq 0 ]] && scope=("${DEFAULT_SCOPE[@]}")

# Is `mod <name>;` declared anywhere in this crate? Matches `mod x;`,
# `pub mod x;`, `pub(crate) mod x;` etc. Deliberately crate-wide rather than
# parent-file-exact: a false negative (missing a genuinely misplaced decl) is
# far cheaper here than a false positive that blocks CI.
declares_mod() {
  local crate="$1" name="$2"
  grep -rqE "^[[:space:]]*(pub(\([^)]*\))?[[:space:]]+)?mod[[:space:]]+${name}[[:space:]]*;" \
    --include='*.rs' "$crate/src" 2>/dev/null
}

orphans=0
stranded=0
quarantined=0
broken_examples=0
for dir in "${scope[@]}"; do
  [[ -f "$dir/Cargo.toml" && -d "$dir/src" ]] || continue

  # Quarantine applies only to the default sweep. An explicit path argument
  # always checks that crate, so you can work a crate out of quarantine and
  # see the real state without editing the list first.
  if [[ $explicit_scope -eq 0 ]] && is_quarantined "$dir"; then
    quarantined=$((quarantined + 1))
    continue
  fi

  # `#[path = "..."]` can attach any file to any module, so a crate using it
  # cannot be checked by name alone. Skip it rather than report noise.
  if grep -rq '#\[path' --include='*.rs' "$dir/src" 2>/dev/null; then
    [[ $summary -eq 1 ]] && echo "SKIP (uses #[path]): $dir"
    continue
  fi

  for f in "$dir"/src/**/*.rs; do
    base="$(basename "$f" .rs)"
    case "$base" in lib|main|mod) continue ;; esac
    # src/bin/*.rs are their own binary targets, not modules.
    [[ "$f" == "$dir/src/bin/"* ]] && continue

    if ! declares_mod "$dir" "$base"; then
      lines=$(wc -l < "$f")
      if is_known_orphan "$f"; then
        echo "known orphan (allowlisted): ${f#./}  (${lines} lines)"
      else
        echo "ORPHAN: ${f#./}  (${lines} lines — no 'mod ${base};' anywhere in ${dir}/src)"
        orphans=$((orphans + 1))
      fi
    fi

    # Empty module root beside a populated directory: `foo.rs` is the module
    # root for `foo/`, so if it is a stub the whole subtree is unreachable.
    if [[ -d "${f%.rs}" && ! -f "${f%.rs}/mod.rs" ]]; then
      root_lines=$(grep -cvE '^\s*(//.*)?$' "$f" || true)
      if [[ "$root_lines" -eq 0 ]]; then
        sub=$(cat "${f%.rs}"/**/*.rs 2>/dev/null | wc -l)
        echo "STRANDED SUBTREE: ${f#./} is an empty module root beside $(basename "${f%.rs}")/ (${sub} lines unreachable)"
        stranded=$((stranded + 1))
      fi
    fi
  done

  # ── examples/ — a second blind spot the module check above cannot see ──────
  #
  # Cargo AUTO-DISCOVERS `examples/*.rs` as binary targets, so no `mod`
  # declares them and the reachability check is structurally blind to them.
  # `cargo check -p <crate>` does not build them either. Both gates miss this.
  #
  # That is exactly how 35 of symthaea-music-theory's 45 examples sat broken:
  # 0d2a2d2090 correctly deleted the evidence_calibration subtree and verified
  # with `cargo check -p symthaea-music-theory` — clean — while every example
  # importing the deleted API stayed broken and invisible. Found hours later by
  # `cargo build --examples`. See docs/design/EVIDENCE_CALIBRATION_STRANDED_DESIGN_2026-07-29.md.
  #
  # Two distinct shapes, checked separately to avoid false positives:
  #   1. A TOP-LEVEL examples/*.rs with no `fn main` can never build.
  #   2. A `// placeholder` stub ANYWHERE under examples/, including support
  #      subdirectories.
  # Files in examples/<subdir>/ are NOT required to have `fn main` — that is
  # the legitimate shared-support-module pattern (e.g. symthaea-futures-ensemble's
  # examples/support/evolutionary_rescue_common.rs, 191 real lines).
  if [[ -d "$dir/examples" ]]; then
    for f in "$dir"/examples/*.rs; do
      [[ -e "$f" ]] || continue
      if ! grep -qE '(^|[^[:alnum:]_])fn[[:space:]]+main[[:space:]]*\(' "$f"; then
        lines=$(wc -l < "$f")
        echo "BROKEN EXAMPLE: ${f#./}  (${lines} lines — no 'fn main'; 'cargo build --examples' fails)"
        broken_examples=$((broken_examples + 1))
      fi
    done
    while IFS= read -r f; do
      [[ -n "$f" ]] || continue
      # Top-level stubs are already reported above by the fn-main check.
      [[ "$(dirname "$f")" == "$dir/examples" ]] && continue
      echo "PLACEHOLDER EXAMPLE: ${f#./}  (contains only '// placeholder')"
      broken_examples=$((broken_examples + 1))
    done < <(grep -rlxE '[[:space:]]*//[[:space:]]*placeholder[[:space:]]*' \
      --include='*.rs' "$dir/examples" 2>/dev/null)
  fi
done

if [[ $summary -eq 1 ]]; then
  echo ""
  echo "orphan modules: $orphans   stranded subtrees: $stranded   broken examples: $broken_examples"
  exit 0
fi

if [[ $orphans -ne 0 || $stranded -ne 0 || $broken_examples -ne 0 ]]; then
  echo ""
  echo "The file(s) above are not part of any crate: they do not compile, their"
  echo "tests never run, and they rot silently against APIs that move."
  echo "Fix by adding the missing 'mod <name>;' (and building — an orphan may"
  echo "already have rotted), or delete the file if it is genuinely dead."
  if [[ $broken_examples -ne 0 ]]; then
    echo ""
    echo "For examples specifically: 'cargo check -p <crate>' does NOT build them."
    echo "Verify with 'cargo build -p <crate> --examples' (or --all-targets)."
  fi
  exit 1
fi
echo "OK: no orphan modules or broken examples in ${#scope[@]} crate(s) checked, $quarantined quarantined."

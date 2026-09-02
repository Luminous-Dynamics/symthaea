#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Local exact-ref qualification for Spore Boot Ecology v0.3.x.
#
# This mirrors .github/workflows/spore-boot-ecology.yml while keeping a strict
# distinction between a formatter candidate and an exact committed qualification.
# If pinned rustfmt changes the target ref, a patch is exported and this script
# exits before any PASS claim can be made for that ref.

set -euo pipefail

FORMAT_PACKAGES=(
  symthaea-boot-ecology
  symthaea-boot-state
  symthaea-quicken-fb
)

usage() {
  cat <<'EOF'
Usage: scripts/qualify-spore-boot-ecology-local.sh [options]

Options:
  --ref REF          Commit/branch/tag to qualify (default: spore/boot-ecology-v0.3).
  --apply-format     Apply the resolver-produced rustfmt patch to this checkout.
                     Allowed only when REF resolves to the current checkout HEAD
                     and relevant files are clean.
  --output-dir PATH  Evidence/patch output root.
  --keep-worktree    Preserve the detached worktree for inspection.
  --skip-galleries   Run Rust checks/tests/probe but skip preview/gallery/lint/seal.
  -h, --help         Show this help.

Exit status:
  0  exact committed target passed all requested gates
  1  qualification/test/evidence failure
  2  tooling or precondition failure
  4  pinned rustfmt changes are required; patch exported, exact target not qualified
EOF
}

TARGET_REF="spore/boot-ecology-v0.3"
APPLY_FORMAT=0
KEEP_WORKTREE=0
SKIP_GALLERIES=0
OUTPUT_DIR=""

while (($#)); do
  case "$1" in
    --ref)
      [[ $# -ge 2 ]] || { echo "ERROR: --ref requires a value" >&2; exit 2; }
      TARGET_REF="$2"
      shift 2
      ;;
    --apply-format)
      APPLY_FORMAT=1
      shift
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires a path" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --keep-worktree)
      KEEP_WORKTREE=1
      shift
      ;;
    --skip-galleries)
      SKIP_GALLERIES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for tool in git sha256sum date; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool is required" >&2; exit 2; }
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: run from a Symthaea checkout" >&2
  exit 2
}
cd "$ROOT"

ROOT_HEAD="$(git rev-parse HEAD)"
TARGET_SHA="$(git rev-parse --verify "${TARGET_REF}^{commit}" 2>/dev/null)" || {
  echo "ERROR: could not resolve target ref: $TARGET_REF" >&2
  exit 2
}
TARGET_TREE="$(git show -s --format=%T "$TARGET_SHA")"
SHORT_SHA="${TARGET_SHA:0:12}"

if [[ "$APPLY_FORMAT" == 1 && "$TARGET_SHA" != "$ROOT_HEAD" ]]; then
  echo "ERROR: --apply-format requires REF to resolve to this checkout's HEAD" >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${TMPDIR:-/tmp}/spore-boot-ecology-${SHORT_SHA}"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

WORK_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/spore-ecology-worktree.XXXXXX")"
WORKTREE="$WORK_PARENT/tree"
FORMAT_PATCH="$OUTPUT_DIR/BOOT_ECOLOGY_RUSTFMT.patch"
RECEIPT="$OUTPUT_DIR/BOOT_ECOLOGY_LOCAL_QUALIFICATION.json"

cleanup() {
  local rc=$?
  if [[ "$KEEP_WORKTREE" == 1 ]]; then
    echo "Preserved qualification worktree: $WORKTREE"
  else
    git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
    rm -rf "$WORK_PARENT"
  fi
  return "$rc"
}
trap cleanup EXIT

git worktree add --detach "$WORKTREE" "$TARGET_SHA" >/dev/null

for required in \
  rust-toolchain.toml \
  Cargo.toml \
  Cargo.lock \
  scripts/spore_preview_gallery.py \
  scripts/spore_inoculation_gallery.py \
  scripts/spore_visual_lint.py \
  scripts/spore_visual_evidence.py; do
  [[ -f "$WORKTREE/$required" ]] || {
    echo "ERROR: target missing $required" >&2
    exit 2
  }
done

PINNED_RUST="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$WORKTREE/rust-toolchain.toml" | head -n1)"
[[ -n "$PINNED_RUST" ]] || { echo "ERROR: cannot read target Rust pin" >&2; exit 2; }

run_in_rust_env() {
  local command="$1"
  if command -v nix >/dev/null 2>&1; then
    (cd "$WORKTREE" && nix develop .#rust --command bash -c "$command")
    return
  fi
  if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    local actual
    actual="$(rustc --version | awk '{print $2}')"
    if [[ "$actual" != "$PINNED_RUST" ]]; then
      echo "ERROR: rustc $actual installed; target pins $PINNED_RUST" >&2
      return 2
    fi
    (cd "$WORKTREE" && bash -c "$command")
    return
  fi
  echo "ERROR: neither Nix nor a directly installed pinned Rust toolchain is available" >&2
  return 2
}

run_python() {
  if command -v python3 >/dev/null 2>&1; then
    (cd "$WORKTREE" && python3 "$@")
    return
  fi
  if command -v nix >/dev/null 2>&1; then
    (cd "$WORKTREE" && nix shell nixpkgs#python3 --command python3 "$@")
    return
  fi
  echo "ERROR: Python 3 is required for exact gallery/evidence tooling" >&2
  return 2
}

json_escape() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '%s' "$value"
}

write_receipt() {
  local status="$1"
  local exit_code="$2"
  local rustc_version="$3"
  local cargo_version="$4"
  local format_patch_sha="$5"
  local renderer_probe_sha="$6"
  local created
  created="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  {
    printf '{\n'
    printf '  "schema": "spore-boot-ecology-local-qualification-v1",\n'
    printf '  "qualified_ref": "%s",\n' "$(json_escape "$TARGET_REF")"
    printf '  "qualified_head": "%s",\n' "$TARGET_SHA"
    printf '  "qualified_tree": "%s",\n' "$TARGET_TREE"
    printf '  "created_at_utc": "%s",\n' "$created"
    printf '  "rust_channel": "%s",\n' "$(json_escape "$PINNED_RUST")"
    printf '  "rustc_version": "%s",\n' "$(json_escape "$rustc_version")"
    printf '  "cargo_version": "%s",\n' "$(json_escape "$cargo_version")"
    printf '  "status": "%s",\n' "$status"
    printf '  "exit_code": %d,\n' "$exit_code"
    if [[ -n "$format_patch_sha" ]]; then
      printf '  "rustfmt_patch_sha256": "%s",\n' "$format_patch_sha"
    else
      printf '  "rustfmt_patch_sha256": null,\n'
    fi
    if [[ -n "$renderer_probe_sha" ]]; then
      printf '  "renderer_probe_sha256": "%s",\n' "$renderer_probe_sha"
    else
      printf '  "renderer_probe_sha256": null,\n'
    fi
    printf '  "galleries_requested": %s\n' "$([[ "$SKIP_GALLERIES" == 1 ]] && printf false || printf true)"
    printf '}\n'
  } >"$RECEIPT"
}

# Capture tool versions from the same target environment used for formatting.
VERSION_TEXT="$(run_in_rust_env 'printf "%s\n" "$(rustc --version)" "$(cargo --version)"' 2>/dev/null)" || {
  write_receipt "precondition-error" 2 "unavailable" "unavailable" "" ""
  echo "ERROR: could not enter pinned Rust environment" >&2
  exit 2
}
RUSTC_VERSION="$(printf '%s\n' "$VERSION_TEXT" | sed -n '1p')"
CARGO_VERSION="$(printf '%s\n' "$VERSION_TEXT" | sed -n '2p')"

# First apply pinned rustfmt only inside the detached worktree. If bytes change,
# export the exact patch and stop: the committed target has not passed fmt.
run_in_rust_env 'cargo fmt -p symthaea-boot-ecology -p symthaea-boot-state -p symthaea-quicken-fb' || {
  write_receipt "format-error" 1 "$RUSTC_VERSION" "$CARGO_VERSION" "" ""
  exit 1
}

if ! git -C "$WORKTREE" diff --quiet -- \
  crates/core/symthaea-boot-ecology \
  crates/core/symthaea-boot-state \
  crates/domains/symthaea-quicken-fb; then
  git -C "$WORKTREE" diff --binary -- \
    crates/core/symthaea-boot-ecology \
    crates/core/symthaea-boot-state \
    crates/domains/symthaea-quicken-fb >"$FORMAT_PATCH"
  FORMAT_PATCH_SHA256="$(sha256sum "$FORMAT_PATCH" | awk '{print $1}')"
  echo "Pinned Rust $PINNED_RUST rustfmt changes are required."
  echo "Patch: $FORMAT_PATCH"
  echo "SHA-256: $FORMAT_PATCH_SHA256"

  if [[ "$APPLY_FORMAT" == 1 ]]; then
    if ! git diff --quiet -- \
      crates/core/symthaea-boot-ecology \
      crates/core/symthaea-boot-state \
      crates/domains/symthaea-quicken-fb \
      || ! git diff --cached --quiet -- \
      crates/core/symthaea-boot-ecology \
      crates/core/symthaea-boot-state \
      crates/domains/symthaea-quicken-fb; then
      write_receipt "precondition-error" 2 "$RUSTC_VERSION" "$CARGO_VERSION" "$FORMAT_PATCH_SHA256" ""
      echo "ERROR: refusing to apply formatting over existing boot-crate changes" >&2
      exit 2
    fi
    git apply "$FORMAT_PATCH"
    echo "Applied formatter patch to current checkout. Commit it, then rerun exact qualification."
  fi

  write_receipt "format-update-required" 4 "$RUSTC_VERSION" "$CARGO_VERSION" "$FORMAT_PATCH_SHA256" ""
  exit 4
fi

rm -f "$FORMAT_PATCH"

# The committed target is format-clean. Run the Rust gates from the dedicated
# hosted workflow, preserving exact package scope.
RUST_GATES='set -euo pipefail
cargo fmt --check -p symthaea-boot-ecology -p symthaea-boot-state -p symthaea-quicken-fb
cargo check -p symthaea-boot-ecology -p symthaea-boot-state -p symthaea-quicken-fb --all-targets
cargo clippy -p symthaea-boot-ecology -p symthaea-boot-state -p symthaea-quicken-fb --all-targets -- -D warnings
cargo test -p symthaea-boot-ecology -p symthaea-boot-state -p symthaea-quicken-fb
cargo run --release -p symthaea-quicken-fb --bin spore_render_probe -- --width 640 --height 360 --frames 24 --out spore-render-probe.json'

if ! run_in_rust_env "$RUST_GATES"; then
  write_receipt "rust-gate-failed" 1 "$RUSTC_VERSION" "$CARGO_VERSION" "" ""
  exit 1
fi

test -s "$WORKTREE/spore-render-probe.json" || {
  write_receipt "renderer-probe-missing" 1 "$RUSTC_VERSION" "$CARGO_VERSION" "" ""
  exit 1
}
RENDERER_PROBE_SHA256="$(sha256sum "$WORKTREE/spore-render-probe.json" | awk '{print $1}')"

# Validate the machine-readable probe using standard-library Python.
run_python -c '
import json, math
from pathlib import Path
report=json.loads(Path("spore-render-probe.json").read_text())
assert report["schema"] == "spore-render-probe-v1"
assert report["width"] == 640 and report["height"] == 360 and report["frames"] == 24
for key in ("mean_frame_ms","p50_frame_ms","p95_frame_ms","max_frame_ms"):
    value=float(report[key]); assert math.isfinite(value) and value >= 0.0
assert len(report["final_frame_blake3"]) == 64
assert report["policy"] == "evidence-only-no-performance-threshold"
' || {
  write_receipt "renderer-probe-invalid" 1 "$RUSTC_VERSION" "$CARGO_VERSION" "" "$RENDERER_PROBE_SHA256"
  exit 1
}

if [[ "$SKIP_GALLERIES" == 0 ]]; then
  run_in_rust_env 'set -euo pipefail
cargo run --release -p symthaea-quicken-fb --bin spore_boot_preview_matrix -- --out spore-boot-preview-matrix --width 320 --height 180 --fps 1
cargo run --release -p symthaea-quicken-fb --bin spore_inoculation_preview -- --out spore-inoculation-preview --width 320 --height 180
cargo run --release -p symthaea-quicken-fb --bin spore_inoculation_paths_preview -- --out spore-inoculation-paths-preview --width 320 --height 180' || {
    write_receipt "preview-render-failed" 1 "$RUSTC_VERSION" "$CARGO_VERSION" "" "$RENDERER_PROBE_SHA256"
    exit 1
  }

  run_python scripts/spore_preview_gallery.py spore-boot-preview-matrix
  run_python scripts/spore_inoculation_gallery.py spore-inoculation-preview
  run_python scripts/spore_inoculation_gallery.py spore-inoculation-paths-preview

  test "$(find "$WORKTREE/spore-boot-preview-matrix" -name preview-manifest.json | wc -l)" -eq 16
  test "$(find "$WORKTREE/spore-boot-preview-matrix" -name 'frame-*.ppm' | wc -l)" -ge 16
  test "$(find "$WORKTREE/spore-boot-preview-matrix" -name 'frame-*.png' | wc -l)" -ge 16
  test "$(find "$WORKTREE/spore-inoculation-preview" -name 'frame-*.ppm' | wc -l)" -eq 32
  test "$(find "$WORKTREE/spore-inoculation-preview" -name 'frame-*.png' | wc -l)" -eq 32
  test "$(find "$WORKTREE/spore-inoculation-paths-preview" -name 'frame-*.ppm' | wc -l)" -eq 18
  test "$(find "$WORKTREE/spore-inoculation-paths-preview" -name 'frame-*.png' | wc -l)" -eq 18

  run_python scripts/spore_visual_lint.py \
    spore-boot-preview-matrix spore-inoculation-preview spore-inoculation-paths-preview

  # Local evidence evaluates the exact target commit, not a synthetic PR merge.
  export SPORE_SOURCE_COMMIT="$TARGET_SHA"
  export GITHUB_SHA="$TARGET_SHA"
  export GITHUB_RUN_ID="local"
  export GITHUB_RUN_ATTEMPT="1"
  run_python scripts/spore_visual_evidence.py \
    spore-boot-preview-matrix spore-inoculation-preview spore-inoculation-paths-preview

  for evidence_root in \
    spore-boot-preview-matrix \
    spore-inoculation-preview \
    spore-inoculation-paths-preview; do
    test -s "$WORKTREE/$evidence_root/evidence-manifest.json"
    test -s "$WORKTREE/$evidence_root/EVIDENCE.sha256"
    (cd "$WORKTREE/$evidence_root" && sha256sum -c EVIDENCE.sha256)
    rm -rf "$OUTPUT_DIR/$evidence_root"
    cp -a "$WORKTREE/$evidence_root" "$OUTPUT_DIR/$evidence_root"
  done
fi

cp "$WORKTREE/spore-render-probe.json" "$OUTPUT_DIR/spore-render-probe.json"
write_receipt "passed" 0 "$RUSTC_VERSION" "$CARGO_VERSION" "" "$RENDERER_PROBE_SHA256"
echo "PASS: exact Boot Ecology target $SHORT_SHA passed all requested local gates."
echo "Receipt: $RECEIPT"
echo "Evidence: $OUTPUT_DIR"

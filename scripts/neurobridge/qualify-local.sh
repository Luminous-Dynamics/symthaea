#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Exact-head local/Nix qualification capsule for the NeuroBridge mechanism stack.
# This is an independent evidence lineage; it does not replace hosted CI.

set -Eeuo pipefail

NIX_FLAGS=(--extra-experimental-features "nix-command flakes")
SCRIPT_REL="scripts/neurobridge/qualify-local.sh"
SHELL_REL="nix/neurobridge-qualification-shell.nix"
CHECKER_REL="scripts/neurobridge/check_contracts.py"
DOC_REL="docs/neuroscience/NEUROBRIDGE_LOCAL_QUALIFICATION_V01.md"
PROFILE="symthaea-neurobridge-local-qualification-v0.1"

usage() {
  cat <<'EOF'
Usage: bash scripts/neurobridge/qualify-local.sh [--out DIR]

Qualifies the exact committed HEAD in a detached worktree using the repository's
locked Nix/Rust toolchain. Uncommitted caller bytes are never qualified.

Default evidence destination:
  target/neurobridge-qualification/local-<UTC timestamp>-<short HEAD>/

This capsule exercises synthetic/mechanism qualification only. It does not
acquire or process real HCP/BALSA data and does not replace hosted CI.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--inside" ]]; then
  ROOT="$(git rev-parse --show-toplevel)"
  HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD)"
  SHORT_SHA="${HEAD_SHA:0:12}"
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUT=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --out)
        [[ $# -ge 2 ]] || { echo "--out requires a directory" >&2; exit 64; }
        OUT="$2"
        shift 2
        ;;
      *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 64
        ;;
    esac
  done

  if [[ -z "$OUT" ]]; then
    OUT="$ROOT/target/neurobridge-qualification/local-${STAMP}-${SHORT_SHA}"
  elif [[ "$OUT" != /* ]]; then
    OUT="$PWD/$OUT"
  fi
  mkdir -p "$(dirname "$OUT")"

  command -v nix >/dev/null || {
    echo "nix is required for the locked NeuroBridge qualification environment" >&2
    exit 69
  }

  BOOTSTRAP_TMP="$(mktemp -d -t symthaea-neurobridge-bootstrap.XXXXXX)"
  WORKTREE="$BOOTSTRAP_TMP/repo"

  cleanup_outer() {
    set +e
    if [[ -d "$WORKTREE" ]]; then
      git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
    fi
    rm -rf "$BOOTSTRAP_TMP"
  }
  trap cleanup_outer EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  git -C "$ROOT" worktree add --detach "$WORKTREE" "$HEAD_SHA" >/dev/null
  for required in "$SCRIPT_REL" "$SHELL_REL" "$CHECKER_REL" "$DOC_REL"; do
    [[ -f "$WORKTREE/$required" ]] || {
      echo "exact HEAD does not contain $required" >&2
      exit 66
    }
  done

  read -r -d '' QUAL_SHELL_EXPR <<'NIX' || true
let
  flake = builtins.getFlake (toString ./.);
  system = builtins.currentSystem;
  pkgs = import flake.inputs.nixpkgs {
    inherit system;
    overlays = [ flake.inputs.rust-overlay.overlays.default ];
  };
  toolchainToml = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
  rustChannel = toolchainToml.toolchain.channel;
  rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
    extensions = [ "clippy" "rustfmt" ];
  };
in
  import ./nix/neurobridge-qualification-shell.nix {
    inherit pkgs rustToolchain;
  }
NIX

  set +e
  (
    cd "$WORKTREE"
    nix "${NIX_FLAGS[@]}" develop --impure --expr "$QUAL_SHELL_EXPR" -c \
      env \
        SYMTHEAEA_NEUROBRIDGE_QUAL_WORKTREE="$WORKTREE" \
        SYMTHEAEA_NEUROBRIDGE_QUAL_EVIDENCE="$OUT" \
        SYMTHEAEA_NEUROBRIDGE_QUAL_HEAD="$HEAD_SHA" \
        bash "$SCRIPT_REL" --inside
  )
  rc=$?
  set -e

  # A zero inner exit is insufficient without a finalized archive commitment.
  if [[ $rc -eq 0 && ( ! -f "${OUT}.tar.gz" || ! -f "${OUT}.tar.gz.sha256" ) ]]; then
    echo "qualification execution returned success without finalized evidence" >&2
    rc=74
  fi

  trap - EXIT INT TERM
  cleanup_outer

  echo "NeuroBridge local evidence: $OUT"
  if [[ -f "${OUT}.tar.gz.sha256" ]]; then
    echo "Evidence archive hash: $(cat "${OUT}.tar.gz.sha256")"
  fi
  exit "$rc"
fi

# ---------------------------------------------------------------------------
# Exact-HEAD inner qualification.
# ---------------------------------------------------------------------------

WORKTREE="${SYMTHEAEA_NEUROBRIDGE_QUAL_WORKTREE:?missing exact worktree}"
EVIDENCE="${SYMTHEAEA_NEUROBRIDGE_QUAL_EVIDENCE:?missing evidence destination}"
EXPECTED_HEAD="${SYMTHEAEA_NEUROBRIDGE_QUAL_HEAD:?missing expected HEAD}"
ARCHIVE="${EVIDENCE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
[[ "$PWD" == "$WORKTREE" ]] || { echo "qualification must run from detached exact HEAD" >&2; exit 70; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo "detached HEAD mismatch" >&2; exit 70; }
if [[ -e "$EVIDENCE" || -e "$ARCHIVE" || -e "$ARCHIVE_SHA" ]]; then
  echo "refusing to overwrite existing qualification evidence" >&2
  exit 73
fi

umask 077
mkdir -p "$EVIDENCE"
RUNTIME_TMP="$(mktemp -d -t symthaea-neurobridge-runtime.XXXXXX)"
export CARGO_TARGET_DIR="$RUNTIME_TMP/cargo-target"
export PYTHONPYCACHEPREFIX="$RUNTIME_TMP/pycache"
export PYTHONDONTWRITEBYTECODE=1
PHASE="bootstrap"
RESULT="RUNNING"

cleanup_runtime() {
  set +e
  rm -rf "$RUNTIME_TMP"
}

write_status() {
  local rc="$1"
  {
    echo "PROFILE=$PROFILE"
    echo "EXECUTION_RESULT=$RESULT"
    echo "EXECUTION_EXIT_CODE=$rc"
    echo "LAST_PHASE=$PHASE"
    echo "SOURCE_HEAD=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "SOURCE_TREE=$(git rev-parse HEAD^{tree} 2>/dev/null || true)"
  } > "$EVIDENCE/STATUS.env"
}

finalize() {
  local execution_rc=$?
  local final_rc=$execution_rc
  local archive_tmp="${ARCHIVE}.tmp.$$"
  local sha_tmp="${ARCHIVE_SHA}.tmp.$$"
  trap - EXIT INT TERM
  set +e

  if [[ "$RESULT" == "RUNNING" ]]; then
    RESULT="FAIL"
  fi
  write_status "$execution_rc"

  local manifest_ok=1 archive_ok=1 sha_ok=1
  (
    cd "$EVIDENCE" || exit 1
    find . -maxdepth 1 -type f ! -name MANIFEST.sha256 -printf '%P\0' \
      | sort -z \
      | xargs -0 -r sha256sum > MANIFEST.sha256
  ) || manifest_ok=0

  if [[ $manifest_ok -eq 1 ]]; then
    rm -f "$archive_tmp" "$sha_tmp"
    tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
      -C "$EVIDENCE" -cf - . | gzip -n > "$archive_tmp" || archive_ok=0
  else
    archive_ok=0
  fi

  if [[ $archive_ok -eq 1 ]]; then
    if ! ln "$archive_tmp" "$ARCHIVE"; then
      archive_ok=0
    fi
    rm -f "$archive_tmp"
  else
    rm -f "$archive_tmp"
  fi

  if [[ $archive_ok -eq 1 ]]; then
    (
      cd "$(dirname "$ARCHIVE")" || exit 1
      sha256sum "$(basename "$ARCHIVE")"
    ) > "$sha_tmp" || sha_ok=0
    if [[ $sha_ok -eq 1 ]]; then
      if ! ln "$sha_tmp" "$ARCHIVE_SHA"; then
        sha_ok=0
      fi
    fi
    rm -f "$sha_tmp"
  else
    sha_ok=0
  fi

  if [[ $manifest_ok -ne 1 || $archive_ok -ne 1 || $sha_ok -ne 1 ]]; then
    echo "evidence finalization failed" >&2
    final_rc=74
    rm -f "$ARCHIVE_SHA"
  fi

  cleanup_runtime
  exit "$final_rc"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_phase() {
  local name="$1"
  shift
  PHASE="$name"
  echo "==> $name"
  set +e
  "$@" > "$EVIDENCE/${name}.log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $name ($rc)" >&2
    return "$rc"
  fi
  printf '%s\tPASS\n' "$name" >> "$EVIDENCE/PHASES.tsv"
}

require_clean_worktree() {
  git status --porcelain=v1 --untracked-files=all > "$EVIDENCE/GIT_STATUS_${1}.txt"
  [[ ! -s "$EVIDENCE/GIT_STATUS_${1}.txt" ]]
}

record_source_identity() {
  [[ "$(git rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
  git rev-parse HEAD > "$EVIDENCE/GIT_HEAD.txt"
  git rev-parse HEAD^{tree} > "$EVIDENCE/GIT_TREE.txt"
  git show -s --format='%H%n%T%n%cI%n%s' HEAD > "$EVIDENCE/GIT_COMMIT.txt"
  sha256sum Cargo.lock flake.lock rust-toolchain.toml > "$EVIDENCE/SOURCE_LOCKS.sha256"
  sha256sum "$SCRIPT_REL" "$SHELL_REL" "$CHECKER_REL" "$DOC_REL" > "$EVIDENCE/QUALIFIER_FILES.sha256"
  sha256sum \
    .github/workflows/substrate-evidence-boundary.yml \
    .github/workflows/neural-benchmark-quarantine.yml \
    .github/workflows/fsaverage5-glasser-map-compiler.yml \
    .github/workflows/fsaverage5-glasser-crosscheck.yml \
    .github/workflows/fsaverage-hcpmmp-semantic-extractor.yml \
    .github/workflows/hcpmmp-neuromaps-lineage-b.yml \
    .github/workflows/hcpmmp-lineage-b-generator-provenance.yml \
    .github/workflows/hcpmmp-lineage-b-bundle-custody.yml \
    .github/workflows/hcpmmp-lineage-b-input-snapshot.yml \
    > "$EVIDENCE/FOCUSED_WORKFLOWS.sha256"
}

record_tool_identity() {
  {
    echo "rustc_path=$(command -v rustc)"
    rustc --version --verbose
    echo "cargo_path=$(command -v cargo)"
    cargo --version
    echo "rustfmt_path=$(command -v rustfmt)"
    rustfmt --version
    echo "clippy_path=$(command -v cargo-clippy)"
    cargo clippy --version
    echo "python_path=$(command -v python)"
    python --version
    echo "nix_path=$(command -v nix)"
    nix --version
    uname -a
  } > "$EVIDENCE/TOOLS.txt" 2>&1
}

python_syntax_contract() {
  python -m py_compile \
    "$CHECKER_REL" \
    scripts/compile_fsaverage5_glasser_map.py \
    scripts/test_compile_fsaverage5_glasser_map.py \
    scripts/compare_fsaverage5_glasser_maps.py \
    scripts/test_compare_fsaverage5_glasser_maps.py \
    scripts/extract_fsaverage_hcpmmp1_semantic_labels.py \
    scripts/test_extract_fsaverage_hcpmmp1_semantic_labels.py \
    scripts/hcpmmp_neuromaps_common.py \
    scripts/hcpmmp_neuromaps_gifti.py \
    scripts/derive_hcpmmp1_neuromaps_lineage_b.py \
    scripts/test_derive_hcpmmp1_neuromaps_lineage_b.py \
    scripts/test_hcpmmp_neuromaps_source_pair.py \
    scripts/test_hcpmmp_neuromaps_generator_provenance.py \
    scripts/test_hcpmmp_neuromaps_bundle_custody.py \
    scripts/hcpmmp_neuromaps_execution_snapshot.py \
    scripts/test_hcpmmp_neuromaps_execution_snapshot.py
}

cli_contracts() {
  python scripts/compile_fsaverage5_glasser_map.py --help
  python scripts/compile_fsaverage5_glasser_map.py compile --help
  python scripts/compile_fsaverage5_glasser_map.py validate --help
  python scripts/compare_fsaverage5_glasser_maps.py --help
  python scripts/extract_fsaverage_hcpmmp1_semantic_labels.py --help
  python scripts/extract_fsaverage_hcpmmp1_semantic_labels.py extract --help
  python scripts/extract_fsaverage_hcpmmp1_semantic_labels.py verify-source --help
  PYTHONPATH=scripts python scripts/derive_hcpmmp1_neuromaps_lineage_b.py --help
  PYTHONPATH=scripts python scripts/derive_hcpmmp1_neuromaps_lineage_b.py derive --help
  PYTHONPATH=scripts python scripts/derive_hcpmmp1_neuromaps_lineage_b.py verify-evidence --help
}

run_phase source-clean-before require_clean_worktree before
run_phase source-identity record_source_identity
run_phase tool-identity record_tool_identity
run_phase qualifier-shell-syntax bash -n "$SCRIPT_REL"
run_phase qualifier-static-contracts python "$CHECKER_REL"

# Rust gates mirror the focused hosted semantics and add --locked.
run_phase rust-core-format cargo fmt --check -p symthaea-core
run_phase rust-core-substrate cargo test --locked -p symthaea-core --lib --features neural_validation substrate_validation
run_phase rust-psych-tests cargo test --locked -p symthaea-psych-bench --lib --features neural_validation
run_phase rust-core-clippy cargo clippy --locked -p symthaea-core --lib --features neural_validation -- -D warnings
run_phase rust-psych-format cargo fmt --check -p symthaea-psych-bench
run_phase rust-psych-clippy cargo clippy --locked -p symthaea-psych-bench --lib --features neural_validation -- -D warnings

# Python mechanism gates present in this exact ancestry.
run_phase python-syntax python_syntax_contract
run_phase map-compiler python -m unittest scripts/test_compile_fsaverage5_glasser_map.py
run_phase map-crosscheck python -m unittest scripts/test_compare_fsaverage5_glasser_maps.py
run_phase fsaverage-extractor python -m unittest scripts/test_extract_fsaverage_hcpmmp1_semantic_labels.py
run_phase lineage-b-core env PYTHONPATH=scripts python -m unittest scripts/test_derive_hcpmmp1_neuromaps_lineage_b.py
run_phase lineage-b-source-pair env PYTHONPATH=scripts python -m unittest scripts/test_hcpmmp_neuromaps_source_pair.py
run_phase generator-provenance env PYTHONPATH=scripts python -m unittest scripts/test_hcpmmp_neuromaps_generator_provenance.py
run_phase bundle-custody env PYTHONPATH=scripts python -m unittest scripts/test_hcpmmp_neuromaps_bundle_custody.py
run_phase input-snapshot env PYTHONPATH=scripts python -m unittest scripts/test_hcpmmp_neuromaps_execution_snapshot.py
run_phase cli-contracts cli_contracts
run_phase source-clean-after require_clean_worktree after

PHASE="complete"
RESULT="PASS"
exit 0

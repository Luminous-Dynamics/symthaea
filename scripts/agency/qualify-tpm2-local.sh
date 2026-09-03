#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reproducible local qualification for the Agency TPM2 boundary.
#
# The outer invocation never qualifies the caller's working tree. It creates a
# detached worktree at the exact committed HEAD, evaluates the qualification
# toolchain from that worktree's flake.lock/rust-toolchain.toml, and re-enters
# this script from the detached worktree inside the locked Nix shell.

set -Eeuo pipefail

NIX_FLAGS=(--extra-experimental-features "nix-command flakes")
SCRIPT_REL="scripts/agency/qualify-tpm2-local.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/agency/qualify-tpm2-local.sh [--out DIR]

Qualifies the exact committed HEAD, never uncommitted working-tree bytes.
Default evidence destination:
  target/agency-qualification/tpm2-<UTC timestamp>-<short HEAD>/

Exit codes:
  0   qualification passed and checked-in Cargo.lock was fresh
  42  source/protocol qualification passed, but Cargo.lock was stale
  other nonzero values indicate a failed qualification phase
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
    OUT="$ROOT/target/agency-qualification/tpm2-${STAMP}-${SHORT_SHA}"
  elif [[ "$OUT" != /* ]]; then
    OUT="$PWD/$OUT"
  fi
  mkdir -p "$(dirname "$OUT")"

  command -v nix >/dev/null || {
    echo "nix is required to enter the locked qualification environment" >&2
    exit 69
  }

  BOOTSTRAP_TMP="$(mktemp -d -t symthaea-agency-tpm2-bootstrap.XXXXXX)"
  WORKTREE="$BOOTSTRAP_TMP/repo"

  cleanup_outer() {
    set +e
    if [[ -d "$WORKTREE" ]]; then
      git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
    fi
    rm -rf "$BOOTSTRAP_TMP"
  }
  trap cleanup_outer EXIT INT TERM

  git -C "$ROOT" worktree add --detach "$WORKTREE" "$HEAD_SHA" >/dev/null
  [[ -f "$WORKTREE/$SCRIPT_REL" ]] || {
    echo "exact HEAD does not contain $SCRIPT_REL" >&2
    exit 66
  }

  read -r -d '' QUAL_SHELL_EXPR <<'NIX' || true
let
  flake = builtins.getFlake (toString ./.);
  system = builtins.currentSystem;
  pkgs = import flake.inputs.nixpkgs {
    inherit system;
    overlays = [ (import flake.inputs.rust-overlay) ];
  };
  toolchainToml = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
  rustChannel = toolchainToml.toolchain.channel;
  rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
    extensions = [ "clippy" "rustfmt" ];
  };
in
  import ./nix/agency-tpm2-qualification-shell.nix {
    inherit pkgs rustToolchain;
  }
NIX

  set +e
  (
    cd "$WORKTREE"
    nix "${NIX_FLAGS[@]}" develop --impure --expr "$QUAL_SHELL_EXPR" -c \
      env \
        SYMTHEAEA_AGENCY_QUAL_WORKTREE="$WORKTREE" \
        SYMTHEAEA_AGENCY_QUAL_EVIDENCE="$OUT" \
        bash "$SCRIPT_REL" --inside
  )
  rc=$?
  set -e

  trap - EXIT INT TERM
  cleanup_outer

  echo "Agency TPM2 evidence: $OUT"
  if [[ -f "${OUT}.tar.gz.sha256" ]]; then
    echo "Evidence archive hash: $(cat "${OUT}.tar.gz.sha256")"
  fi
  exit "$rc"
fi

# ---------------------------------------------------------------------------
# Exact-HEAD inner qualification. This path runs only from the detached
# worktree inside the locked Nix shell constructed above.
# ---------------------------------------------------------------------------

WORKTREE="${SYMTHEAEA_AGENCY_QUAL_WORKTREE:?missing exact worktree}"
EVIDENCE="${SYMTHEAEA_AGENCY_QUAL_EVIDENCE:?missing evidence destination}"
[[ "$PWD" == "$WORKTREE" ]] || {
  echo "qualification must execute from the exact detached worktree" >&2
  exit 70
}

if [[ -e "$EVIDENCE" ]]; then
  echo "refusing to overwrite existing evidence path: $EVIDENCE" >&2
  exit 73
fi
mkdir -p "$EVIDENCE"

PHASE="bootstrap"
RESULT="RUNNING"
LOCK_STALE=0
SWTPM_PID=""
RUNTIME_TMP="$(mktemp -d -t symthaea-agency-tpm2-runtime.XXXXXX)"
ARCHIVE="${EVIDENCE}.tar.gz"

cleanup_runtime() {
  set +e
  if [[ -n "$SWTPM_PID" ]]; then
    kill "$SWTPM_PID" >/dev/null 2>&1 || true
    wait "$SWTPM_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$RUNTIME_TMP"
}

finalize() {
  rc=$?
  trap - EXIT INT TERM
  set +e
  cleanup_runtime

  if [[ "$RESULT" == "RUNNING" ]]; then
    RESULT="FAIL"
  fi

  printf '%s\n' "$RESULT" > "$EVIDENCE/RESULT.txt"
  printf '%s\n' "$PHASE" > "$EVIDENCE/LAST_PHASE.txt"
  printf '%s\n' "$rc" > "$EVIDENCE/EXIT_CODE.txt"
  printf '%s\n' "$LOCK_STALE" > "$EVIDENCE/CARGO_LOCK_STALE.txt"

  RESULT_VALUE="$RESULT" PHASE_VALUE="$PHASE" RC_VALUE="$rc" LOCK_VALUE="$LOCK_STALE" \
    python3 - <<'PY' > "$EVIDENCE/QUALIFICATION_RESULT.json"
import json, os
print(json.dumps({
    "schema": "symthaea.agency-tpm2-local-qualification.v1",
    "result": os.environ["RESULT_VALUE"],
    "last_phase": os.environ["PHASE_VALUE"],
    "exit_code": int(os.environ["RC_VALUE"]),
    "cargo_lock_stale": os.environ["LOCK_VALUE"] == "1",
}, sort_keys=True, indent=2))
PY

  (
    cd "$EVIDENCE" || exit 1
    find . -type f ! -name 'MANIFEST.sha256' -print0 \
      | sort -z \
      | xargs -0 sha256sum > MANIFEST.sha256
  )

  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -C "$EVIDENCE" -cf - . | gzip -n > "$ARCHIVE"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

  echo "qualification_result=$RESULT"
  echo "qualification_last_phase=$PHASE"
  echo "qualification_evidence=$EVIDENCE"
  echo "qualification_archive=$ARCHIVE"
  exit "$rc"
}
trap finalize EXIT INT TERM

run_capture() {
  local phase="$1"
  local log="$2"
  shift 2
  PHASE="$phase"
  "$@" > >(tee "$EVIDENCE/$log") 2>&1
}

HEAD_SHA="$(git rev-parse HEAD)"
TREE_SHA="$(git rev-parse HEAD^{tree})"
printf '%s\n' "$HEAD_SHA" > "$EVIDENCE/HEAD"
printf '%s\n' "$TREE_SHA" > "$EVIDENCE/TREE"
git status --porcelain=v1 --untracked-files=all > "$EVIDENCE/DETACHED_WORKTREE_STATUS.txt"
[[ ! -s "$EVIDENCE/DETACHED_WORKTREE_STATUS.txt" ]] || {
  echo "detached qualification worktree is unexpectedly dirty" >&2
  exit 71
}

rustc -Vv > "$EVIDENCE/RUSTC.txt"
cargo -V > "$EVIDENCE/CARGO.txt"
nix --version > "$EVIDENCE/NIX.txt"
uname -a > "$EVIDENCE/UNAME.txt"
sha256sum Cargo.lock > "$EVIDENCE/CARGO_LOCK_BEFORE_SHA256.txt"
sha256sum flake.lock > "$EVIDENCE/FLAKE_LOCK_SHA256.txt"
sha256sum rust-toolchain.toml > "$EVIDENCE/RUST_TOOLCHAIN_TOML_SHA256.txt"
cp Cargo.lock "$EVIDENCE/Cargo.lock.before"

PHASE="flake-metadata"
nix "${NIX_FLAGS[@]}" flake metadata --json . > "$EVIDENCE/FLAKE_METADATA.json"
python3 - "$EVIDENCE/FLAKE_METADATA.json" <<'PY' > "$EVIDENCE/NIXPKGS_LOCKED.json"
import json, pathlib, sys
meta = json.loads(pathlib.Path(sys.argv[1]).read_text())
locked = meta["locks"]["nodes"]["nixpkgs"]["locked"]
print(json.dumps(locked, sort_keys=True, indent=2))
PY

PHASE="cargo-lock-reconciliation"
cargo metadata --format-version 1 >/dev/null
cp Cargo.lock "$EVIDENCE/Cargo.lock.candidate"
git diff -- Cargo.lock > "$EVIDENCE/CARGO_LOCK_DIFF.patch"

python3 - "$EVIDENCE/Cargo.lock.before" Cargo.lock <<'PY' > "$EVIDENCE/LOCK_RECONCILIATION.txt"
import pathlib, sys, tomllib
before = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
after = tomllib.loads(pathlib.Path(sys.argv[2]).read_text())
key = lambda p: (p["name"], p["version"], p.get("source"))
b = {key(p): p for p in before.get("package", [])}
a = {key(p): p for p in after.get("package", [])}
removed = sorted(set(b) - set(a), key=repr)
changed = sorted((k for k in b.keys() & a if b[k] != a[k]), key=repr)
added = sorted(set(a) - set(b), key=repr)
sourced = [k for k in added if k[2] is not None]
print("removed:", removed)
print("changed:", changed)
print("new sourced packages:", sourced)
print("additive workspace/path nodes:", [k for k in added if k[2] is None])
if removed or changed or sourced:
    raise SystemExit(2)
PY

if ! cmp -s "$EVIDENCE/Cargo.lock.before" Cargo.lock; then
  LOCK_STALE=1
fi
sha256sum Cargo.lock > "$EVIDENCE/CARGO_LOCK_CANDIDATE_SHA256.txt"

PHASE="rustfmt-diagnostic"
set +e
cargo fmt --check --package symthaea-platform-attestation \
  > "$EVIDENCE/RUSTFMT.stdout" 2> "$EVIDENCE/RUSTFMT.stderr"
FMT_RC=$?
set -e
printf '%s\n' "$FMT_RC" > "$EVIDENCE/RUSTFMT_EXIT_CODE.txt"

run_capture "cargo-test" "CARGO_TEST.log" \
  cargo test --locked -p symthaea-platform-attestation
run_capture "cargo-clippy" "CARGO_CLIPPY.log" \
  cargo clippy --locked -p symthaea-platform-attestation --all-targets -- -D warnings
run_capture "probe-build" "PROBE_BUILD.log" \
  cargo build --locked -p symthaea-platform-attestation --bin tpm2_attestation_probe

PROBE="target/debug/tpm2_attestation_probe"
[[ -x "$PROBE" ]]
sha256sum "$PROBE" > "$EVIDENCE/PROBE_SHA256.txt"

TCTI='swtpm:host=127.0.0.1,port=2321'
PHASE="port-preflight"
python3 - <<'PY'
import socket
for port in (2321, 2322):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"qualification port {port} unavailable: {exc}")
    finally:
        s.close()
PY

PHASE="build-hermetic-verifier"
read -r -d '' VERIFIER_EXPR <<'NIX' || true
let
  flake = builtins.getFlake (toString ./.);
  pkgs = import flake.inputs.nixpkgs { system = builtins.currentSystem; };
in
  import ./nix/agency-tpm2-verifier-tools.nix {
    inherit pkgs;
    tcti = "swtpm:host=127.0.0.1,port=2321";
  }
NIX
VERIFIER_STORE="$(nix "${NIX_FLAGS[@]}" build --impure --no-link --print-out-paths --expr "$VERIFIER_EXPR")"
QUOTE_WRAPPER="$VERIFIER_STORE/bin/symthaea-tpm2-quote"
CHECK_WRAPPER="$VERIFIER_STORE/bin/symthaea-tpm2-checkquote"
[[ -x "$QUOTE_WRAPPER" && -x "$CHECK_WRAPPER" ]]

printf '%s\n' "$VERIFIER_STORE" > "$EVIDENCE/TPM2_VERIFIER_STORE.txt"
printf '%s\n' "$QUOTE_WRAPPER" > "$EVIDENCE/QUOTE_WRAPPER_PATH.txt"
printf '%s\n' "$CHECK_WRAPPER" > "$EVIDENCE/CHECKQUOTE_WRAPPER_PATH.txt"
sha256sum "$QUOTE_WRAPPER" > "$EVIDENCE/QUOTE_WRAPPER_SHA256.txt"
sha256sum "$CHECK_WRAPPER" > "$EVIDENCE/CHECKQUOTE_WRAPPER_SHA256.txt"
file "$QUOTE_WRAPPER" "$CHECK_WRAPPER" > "$EVIDENCE/TPM2_WRAPPER_FILE.txt"
readelf -l "$QUOTE_WRAPPER" > "$EVIDENCE/QUOTE_WRAPPER_ELF.txt"
readelf -l "$CHECK_WRAPPER" > "$EVIDENCE/CHECKQUOTE_WRAPPER_ELF.txt"
! grep -q 'INTERP' "$EVIDENCE/QUOTE_WRAPPER_ELF.txt"
! grep -q 'INTERP' "$EVIDENCE/CHECKQUOTE_WRAPPER_ELF.txt"
nix-store --query --references "$VERIFIER_STORE" | sort > "$EVIDENCE/TPM2_VERIFIER_REFERENCES.txt"

PHASE="override-rejection"
set +e
"$QUOTE_WRAPPER" -T device:/dev/null >/dev/null 2> "$EVIDENCE/QUOTE_TCTI_OVERRIDE.stderr"
QUOTE_TCTI_RC=$?
"$QUOTE_WRAPPER" -F values >/dev/null 2> "$EVIDENCE/QUOTE_FORMAT_OVERRIDE.stderr"
QUOTE_FORMAT_RC=$?
"$CHECK_WRAPPER" -T device:/dev/null >/dev/null 2> "$EVIDENCE/CHECK_TCTI_OVERRIDE.stderr"
CHECK_TCTI_RC=$?
set -e
[[ "$QUOTE_TCTI_RC" -eq 64 ]]
[[ "$QUOTE_FORMAT_RC" -eq 64 ]]
[[ "$CHECK_TCTI_RC" -eq 64 ]]
grep -q 'option override rejected' "$EVIDENCE/QUOTE_TCTI_OVERRIDE.stderr"
grep -q 'option override rejected' "$EVIDENCE/QUOTE_FORMAT_OVERRIDE.stderr"
grep -q 'option override rejected' "$EVIDENCE/CHECK_TCTI_OVERRIDE.stderr"

PHASE="start-swtpm"
mkdir -p "$RUNTIME_TMP/swtpm-state"
swtpm socket \
  --tpm2 \
  --tpmstate dir="$RUNTIME_TMP/swtpm-state" \
  --ctrl type=tcp,port=2322 \
  --server type=tcp,port=2321 \
  --flags not-need-init \
  > "$EVIDENCE/SWTPM.log" 2>&1 &
SWTPM_PID=$!
sleep 1
kill -0 "$SWTPM_PID"
swtpm --version > "$EVIDENCE/SWTPM_VERSION.txt" 2>&1 || true
tpm2_quote --version > "$EVIDENCE/TPM2_TOOLS_VERSION.txt" 2>&1 || true

PHASE="create-attestation-key"
tpm2_startup -T "$TCTI" -c
tpm2_createek -Q \
  -T "$TCTI" \
  -c "$RUNTIME_TMP/ek.ctx" \
  -G rsa \
  -u "$RUNTIME_TMP/ekpub.pem" \
  -f pem
tpm2_createak -Q \
  -T "$TCTI" \
  -C "$RUNTIME_TMP/ek.ctx" \
  -c "$RUNTIME_TMP/ak.ctx" \
  -G rsa \
  -s rsassa \
  -g sha256 \
  -u "$RUNTIME_TMP/akpub.pem" \
  -f pem \
  -n "$RUNTIME_TMP/ak.name"
sha256sum "$RUNTIME_TMP/akpub.pem" > "$EVIDENCE/AK_PUBLIC_SHA256.txt"

PHASE="baseline-pcr-profile"
set +e
env \
  TPM2TOOLS_TCTI='device:/definitely-not-the-swtpm' \
  LD_PRELOAD='/definitely/not/a/real/preload.so' \
  BASH_ENV='/definitely/not/a/bash/env' \
  "$QUOTE_WRAPPER" -Q \
    -c "$RUNTIME_TMP/ak.ctx" \
    -l sha256:16 \
    -m "$RUNTIME_TMP/baseline.msg" \
    -s "$RUNTIME_TMP/baseline.sig" \
    -o "$RUNTIME_TMP/baseline.pcrs" \
    -g sha256 \
    > "$EVIDENCE/HERMETIC_BASELINE.stdout" \
    2> "$EVIDENCE/HERMETIC_BASELINE.stderr"
BASELINE_RC=$?
set -e
[[ "$BASELINE_RC" -eq 0 ]]
! grep -Eqi 'LD_PRELOAD|cannot be preloaded|ld\.so' "$EVIDENCE/HERMETIC_BASELINE.stderr"

APPROVED_PCR_PROFILE="$("$PROBE" profile-digest "$RUNTIME_TMP/baseline.pcrs")"
[[ "${#APPROVED_PCR_PROFILE}" -eq 64 ]]
printf '%s\n' "$APPROVED_PCR_PROFILE" > "$EVIDENCE/APPROVED_PCR_PROFILE.txt"

PHASE="fresh-attestation"
TPM2TOOLS_TCTI='device:/ambient-tcti-must-not-win' \
OPENSSL_CONF='/definitely/not/an/openssl/config' \
"$PROBE" verify-nix \
  "$QUOTE_WRAPPER" \
  "$CHECK_WRAPPER" \
  "$RUNTIME_TMP/ak.ctx" \
  "$RUNTIME_TMP/akpub.pem" \
  "$APPROVED_PCR_PROFILE" \
  16 | tee "$EVIDENCE/TPM2_VERIFIED.txt"
grep -qx 'platform_attestation=verified' "$EVIDENCE/TPM2_VERIFIED.txt"

PHASE="pcr-mutation"
MUTATION="$(printf 'symthaea-pcr16-adversarial-mutation' | sha256sum | awk '{print $1}')"
tpm2_pcrextend -T "$TCTI" "16:sha256=$MUTATION"
set +e
"$PROBE" verify-nix \
  "$QUOTE_WRAPPER" \
  "$CHECK_WRAPPER" \
  "$RUNTIME_TMP/ak.ctx" \
  "$RUNTIME_TMP/akpub.pem" \
  "$APPROVED_PCR_PROFILE" \
  16 > "$EVIDENCE/TPM2_MUTATED.stdout" 2> "$EVIDENCE/TPM2_MUTATED.stderr"
MUTATED_RC=$?
set -e
[[ "$MUTATED_RC" -ne 0 ]]
grep -q 'PCR state is not an approved profile' "$EVIDENCE/TPM2_MUTATED.stderr"

PHASE="complete"
if [[ "$LOCK_STALE" -eq 1 ]]; then
  RESULT="FAIL_LOCK_STALE"
  exit 42
fi

RESULT="PASS"
exit 0

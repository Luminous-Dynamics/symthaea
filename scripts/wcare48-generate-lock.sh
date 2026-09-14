#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

WCARE47_PARENT="0321a22986e5c6bf1a6b099d30dc75ccc33d9deb"
MANIFEST="tools/wcare42_builder_attestation_verifier/Cargo.toml"
LOCK="tools/wcare42_builder_attestation_verifier/Cargo.lock"
RECEIPT="docs/release/evidence/WCARE48_LOCK_GENERATION_RECEIPT_V1.json"
PROTOCOL="docs/release/evidence/WCARE48_LOCK_GENERATION_PROTOCOL_V1.md"
WORKFLOW=".github/workflows/wcare48-lock-generation.yml"
SCRIPT="scripts/wcare48-generate-lock.sh"

: "${WCARE48_PREPARED_HEAD:?missing WCARE48_PREPARED_HEAD}"
: "${WCARE48_RUN_ID:?missing WCARE48_RUN_ID}"
: "${WCARE48_RUN_NUMBER:?missing WCARE48_RUN_NUMBER}"
: "${WCARE48_RUN_ATTEMPT:?missing WCARE48_RUN_ATTEMPT}"
: "${WCARE48_REPOSITORY:?missing WCARE48_REPOSITORY}"
: "${WCARE48_REF_NAME:?missing WCARE48_REF_NAME}"
: "${WCARE48_WORKFLOW_REF:?missing WCARE48_WORKFLOW_REF}"

fail() {
  printf '%s\n' "WCARE48_FAIL:$*" >&2
  exit 1
}

check_head_blob() {
  local path="$1" expected="$2" actual
  actual="$(git rev-parse "HEAD:$path" 2>/dev/null || true)"
  [[ "$actual" == "$expected" ]] || fail "blob_mismatch:$path:$actual"
}

HEAD_NOW="$(git rev-parse HEAD)"
[[ "$HEAD_NOW" == "$WCARE48_PREPARED_HEAD" ]] || fail "prepared_head_mismatch:$HEAD_NOW"
[[ "$(git rev-parse HEAD^)" == "$WCARE47_PARENT" ]] || fail "prepared_parent_mismatch"
[[ "$WCARE48_REPOSITORY" == "Luminous-Dynamics/symthaea" ]] || fail "repository_mismatch:$WCARE48_REPOSITORY"
[[ "$WCARE48_REF_NAME" == "wcare-48-observed-lock-generation" ]] || fail "branch_mismatch:$WCARE48_REF_NAME"

[[ ! -e "$LOCK" ]] || fail "candidate_lock_already_exists"
if git cat-file -e "HEAD:$LOCK" 2>/dev/null; then
  fail "candidate_lock_already_committed"
fi

[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "prepared_tree_not_clean"

check_head_blob "$MANIFEST" "5410040e5616241dd4ba581af8f297675d083830"
check_head_blob "tools/wcare42_builder_attestation_verifier/src/main.rs" "1c300a455f054d118e55556aac81b623824629bc"
check_head_blob "tools/wcare42_builder_attestation_verifier/tests/golden.rs" "666242f74fb302f9be2b62fa4b3050f3ee9ffefd"
check_head_blob "rust-toolchain.toml" "4f0430eac96d545bcfaa0df23ce475faf4aee96a"

PREPARED_TREE="$(git rev-parse 'HEAD^{tree}')"
SCRIPT_BLOB="$(git rev-parse "HEAD:$SCRIPT")"
WORKFLOW_BLOB="$(git rev-parse "HEAD:$WORKFLOW")"
PROTOCOL_BLOB="$(git rev-parse "HEAD:$PROTOCOL")"

RUSTC_VERSION="$(rustc --version)"
CARGO_VERSION="$(cargo --version)"
[[ "$RUSTC_VERSION" == rustc\ 1.96.0\ * ]] || fail "unexpected_rustc:$RUSTC_VERSION"
[[ "$CARGO_VERSION" == cargo\ 1.96.0\ * ]] || fail "unexpected_cargo:$CARGO_VERSION"
RUSTC_VERBOSE="$(rustc -Vv)"
CARGO_VERBOSE="$(cargo -Vv)"

export CARGO_TARGET_DIR="${RUNNER_TEMP:-/tmp}/wcare48-target-${WCARE48_RUN_ID}"
rm -rf "$CARGO_TARGET_DIR"

cargo generate-lockfile --manifest-path "$MANIFEST"

STATUS="$(git status --porcelain=v1 --untracked-files=all)"
[[ "$STATUS" == "?? $LOCK" ]] || fail "unexpected_post_generation_tree:$STATUS"

LOCK_SHA256="$(sha256sum "$LOCK" | awk '{print $1}')"
LOCK_GIT_BLOB="$(git hash-object "$LOCK")"

PACKAGE_COUNT="$(
  python3 - "$LOCK" <<'PY'
from pathlib import Path
import re
import sys
import tomllib

path = Path(sys.argv[1])
data = tomllib.loads(path.read_text(encoding="utf-8"))
if data.get("version") != 4:
    raise SystemExit("lock format is not v4")
packages = data.get("package")
if not isinstance(packages, list) or not packages:
    raise SystemExit("package census missing")
root_seen = False
hex64 = re.compile(r"^[0-9a-f]{64}$")
for package in packages:
    if not isinstance(package, dict):
        raise SystemExit("non-object package")
    name = package.get("name")
    source = package.get("source")
    checksum = package.get("checksum")
    if name == "wcare42-builder-attestation-verifier" and source is None:
        root_seen = True
        continue
    if not isinstance(source, str) or not source.startswith("registry+"):
        raise SystemExit(f"non-registry dependency: {name}")
    if not isinstance(checksum, str) or hex64.fullmatch(checksum) is None:
        raise SystemExit(f"missing/invalid checksum: {name}")
if not root_seen:
    raise SystemExit("standalone root package missing")
print(len(packages))
PY
)"

cargo metadata --manifest-path "$MANIFEST" --locked --format-version 1 > "${RUNNER_TEMP:-/tmp}/wcare48-metadata.json"
cargo test --manifest-path "$MANIFEST" --locked

REPEAT_PARENT="$(mktemp -d "${RUNNER_TEMP:-/tmp}/wcare48-repeat.XXXXXX")"
REPEAT_ROOT="$REPEAT_PARENT/repo"
cleanup() {
  if [[ -n "${REPEAT_ROOT:-}" ]] && git worktree list --porcelain | grep -Fq "worktree $REPEAT_ROOT"; then
    git worktree remove --force "$REPEAT_ROOT" || true
  fi
  rm -rf "${REPEAT_PARENT:-}"
}
trap cleanup EXIT

git worktree add --detach "$REPEAT_ROOT" "$WCARE48_PREPARED_HEAD" >/dev/null
cargo generate-lockfile --manifest-path "$REPEAT_ROOT/$MANIFEST"
cmp -s "$LOCK" "$REPEAT_ROOT/$LOCK" || fail "same_runner_repeat_generation_mismatch"
REPEAT_SHA256="$(sha256sum "$REPEAT_ROOT/$LOCK" | awk '{print $1}')"
[[ "$REPEAT_SHA256" == "$LOCK_SHA256" ]] || fail "same_runner_repeat_sha_mismatch"
git worktree remove --force "$REPEAT_ROOT"
REPEAT_ROOT=""

check_head_blob "$MANIFEST" "5410040e5616241dd4ba581af8f297675d083830"
check_head_blob "tools/wcare42_builder_attestation_verifier/src/main.rs" "1c300a455f054d118e55556aac81b623824629bc"
check_head_blob "tools/wcare42_builder_attestation_verifier/tests/golden.rs" "666242f74fb302f9be2b62fa4b3050f3ee9ffefd"
check_head_blob "rust-toolchain.toml" "4f0430eac96d545bcfaa0df23ce475faf4aee96a"
[[ "$(git rev-parse HEAD)" == "$WCARE48_PREPARED_HEAD" ]] || fail "head_drift_after_generation"

export PREPARED_TREE SCRIPT_BLOB WORKFLOW_BLOB PROTOCOL_BLOB
export RUSTC_VERSION CARGO_VERSION RUSTC_VERBOSE CARGO_VERBOSE
export LOCK_SHA256 LOCK_GIT_BLOB PACKAGE_COUNT REPEAT_SHA256
python3 - "$RECEIPT" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

receipt = {
    "authority": "MeasurementOnly",
    "protocol_version": "wcare48-observed-lock-generation-v1",
    "repository": os.environ["WCARE48_REPOSITORY"],
    "branch": os.environ["WCARE48_REF_NAME"],
    "prepared_head": os.environ["WCARE48_PREPARED_HEAD"],
    "prepared_parent": "0321a22986e5c6bf1a6b099d30dc75ccc33d9deb",
    "prepared_tree": os.environ["PREPARED_TREE"],
    "github_run_id": os.environ["WCARE48_RUN_ID"],
    "github_run_number": os.environ["WCARE48_RUN_NUMBER"],
    "github_run_attempt": os.environ["WCARE48_RUN_ATTEMPT"],
    "github_workflow_ref": os.environ["WCARE48_WORKFLOW_REF"],
    "runner_os": os.environ.get("RUNNER_OS"),
    "runner_arch": os.environ.get("RUNNER_ARCH"),
    "runner_name": os.environ.get("RUNNER_NAME"),
    "image_os": os.environ.get("ImageOS"),
    "image_version": os.environ.get("ImageVersion"),
    "rustc_version": os.environ["RUSTC_VERSION"],
    "cargo_version": os.environ["CARGO_VERSION"],
    "rustc_verbose": os.environ["RUSTC_VERBOSE"],
    "cargo_verbose": os.environ["CARGO_VERBOSE"],
    "generation_script_blob": os.environ["SCRIPT_BLOB"],
    "workflow_blob": os.environ["WORKFLOW_BLOB"],
    "protocol_blob": os.environ["PROTOCOL_BLOB"],
    "manifest_blob": "5410040e5616241dd4ba581af8f297675d083830",
    "verifier_source_blob": "1c300a455f054d118e55556aac81b623824629bc",
    "golden_test_blob": "666242f74fb302f9be2b62fa4b3050f3ee9ffefd",
    "rust_toolchain_blob": "4f0430eac96d545bcfaa0df23ce475faf4aee96a",
    "lock_sha256": os.environ["LOCK_SHA256"],
    "lock_git_blob_candidate": os.environ["LOCK_GIT_BLOB"],
    "lock_format": 4,
    "package_count": int(os.environ["PACKAGE_COUNT"]),
    "registry_checksum_policy_satisfied": True,
    "metadata_locked_passed": True,
    "tests_locked_passed": True,
    "same_runner_repeat_generation_match": True,
    "repeat_lock_sha256": os.environ["REPEAT_SHA256"],
    "source_postflight_unchanged": True,
    "github_host_observed_generation_lineage": True,
    "external_generation_provenance_established": False,
    "builder_authentication_established": False,
    "external_preregistration_established": False,
    "independent_host_reproducibility_established": False,
    "wcare42_executable_qualification_established": False,
    "runtime_authority_granted": False,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(sys.argv[1]).write_text(
    json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

EXPECTED_STATUS="$(printf '?? %s\n?? %s\n' "$RECEIPT" "$LOCK" | sort)"
ACTUAL_STATUS="$(git status --porcelain=v1 --untracked-files=all | sort)"
[[ "$ACTUAL_STATUS" == "$EXPECTED_STATUS" ]] || fail "unexpected_precommit_tree:$ACTUAL_STATUS"

printf '%s\n' "PASS_WCARE48_GENERATION_PRECOMMIT"
printf '%s\n' "LOCK_SHA256=$LOCK_SHA256"
printf '%s\n' "LOCK_GIT_BLOB_CANDIDATE=$LOCK_GIT_BLOB"
printf '%s\n' "PACKAGE_COUNT=$PACKAGE_COUNT"

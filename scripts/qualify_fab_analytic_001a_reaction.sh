#!/usr/bin/env bash
set -euo pipefail

SOURCE_HEAD="b14459e610b9e1fbb4676b7d0b5731883964efbf"
SOURCE_BLOB="4c2c8eaff894c51b7e0a3bc39d77e12eb1dd3dce"
MANIFEST_BLOB="e36c96a0e8f8fa390154db8cb0b1ae7383a08654"
SOURCE_PATH="crates/domains/symthaea-fabrication-kernel/src/analytical.rs"
MANIFEST_PATH="crates/domains/symthaea-fabrication-kernel/Cargo.toml"
SCRIPT_PATH="scripts/qualify_fab_analytic_001a_reaction.sh"
WORKFLOW_PATH=".github/workflows/fab-analytic-001a-reaction.yml"

HEAD="$(git rev-parse HEAD)"
PARENT="$(git rev-parse HEAD^)"

[[ "$PARENT" == "$SOURCE_HEAD" ]] || {
  echo "ERROR: qualifier parent $PARENT != frozen source $SOURCE_HEAD" >&2
  exit 1
}

[[ "$(git rev-list --count "$SOURCE_HEAD"..HEAD)" == "1" ]] || {
  echo "ERROR: qualifier must be exactly one commit over source" >&2
  exit 1
}

[[ "$(git rev-parse "$SOURCE_HEAD:$SOURCE_PATH")" == "$SOURCE_BLOB" ]] || {
  echo "ERROR: repaired analytical.rs blob drifted" >&2
  exit 1
}

[[ "$(git rev-parse "$SOURCE_HEAD:$MANIFEST_PATH")" == "$MANIFEST_BLOB" ]] || {
  echo "ERROR: package manifest blob drifted" >&2
  exit 1
}

mapfile -t ACTUAL < <(git diff --name-only "$SOURCE_HEAD"..HEAD | sort)
EXPECTED=("$WORKFLOW_PATH" "$SCRIPT_PATH")
mapfile -t EXPECTED_SORTED < <(printf '%s\n' "${EXPECTED[@]}" | sort)

if [[ "${#ACTUAL[@]}" -ne 2 ]] || ! diff -u <(printf '%s\n' "${EXPECTED_SORTED[@]}") <(printf '%s\n' "${ACTUAL[@]}"); then
  echo "ERROR: qualifier delta must contain exactly script + workflow" >&2
  exit 1
fi

echo "source=$SOURCE_HEAD"
echo "qualifier=$HEAD"
cat rust-toolchain.toml
rustc --version
cargo --version

cargo check --locked -p symthaea-fabrication-kernel
cargo test --locked -p symthaea-fabrication-kernel reaction_force_proxy_is_not_energy_state
cargo test --locked -p symthaea-fabrication-kernel reset_clears_reaction_force_proxy
cargo test --locked -p symthaea-fabrication-kernel test_physics_backend_step
cargo test --locked -p symthaea-fabrication-kernel

[[ -z "$(git status --porcelain)" ]] || {
  echo "ERROR: qualification execution dirtied the source tree" >&2
  git status --short >&2
  exit 1
}

git diff --exit-code

echo "PASS_FAB_ANALYTIC_001A_REACTION source=$SOURCE_HEAD qualifier=$HEAD"

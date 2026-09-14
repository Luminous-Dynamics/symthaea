#!/usr/bin/env bash
set -euo pipefail

package="symthaea-assurance-linux-tpm-ima-pcr-anchor"
manifest="crates/bridges/symthaea-assurance-linux-tpm-ima-pcr-anchor/Cargo.toml"
work="${RUNNER_TEMP:-/tmp}/assure-runtime-002-pcr-anchor"
pre_lock="$work/Cargo.lock.pre"
reconciled_lock="$work/Cargo.lock.reconciled"
lock_diff="$work/Cargo.lock.diff"
lock_audit="$work/Cargo.lock.audit.txt"
mkdir -p "$work"

[[ -f "$manifest" ]]
[[ -f Cargo.lock ]]
cp Cargo.lock "$pre_lock"
pre_lock_sha256="$(sha256sum Cargo.lock | awk '{print $1}')"

restore_lock() {
  cp "$pre_lock" Cargo.lock
  restored="$(sha256sum Cargo.lock | awk '{print $1}')"
  if [[ "$restored" != "$pre_lock_sha256" ]]; then
    echo "failed to restore source Cargo.lock" >&2
    exit 91
  fi
}
trap restore_lock EXIT

cargo fmt --manifest-path "$manifest" -- --check

# The new workspace package is intentionally source-only on this stacked tranche.
# Let Cargo reconcile the lock in the ephemeral worktree, then prove that the only
# semantic lock change is the new local package stanza.
cargo check --manifest-path "$manifest"
git diff -- Cargo.lock > "$lock_diff"
cp Cargo.lock "$reconciled_lock"

python3 - "$pre_lock" "$reconciled_lock" "$lock_audit" <<'PY'
import json
import pathlib
import sys
import tomllib

pre_path, post_path, audit_path = map(pathlib.Path, sys.argv[1:])
pre = tomllib.loads(pre_path.read_text())
post = tomllib.loads(post_path.read_text())

if pre.get("version") != post.get("version"):
    raise SystemExit("Cargo.lock format version changed")


def key(package):
    return (package["name"], package["version"], package.get("source", ""))

pre_packages = {key(package): package for package in pre.get("package", [])}
post_packages = {key(package): package for package in post.get("package", [])}

missing = sorted(set(pre_packages) - set(post_packages))
changed = sorted(
    package_key
    for package_key in set(pre_packages) & set(post_packages)
    if pre_packages[package_key] != post_packages[package_key]
)
extra = sorted(set(post_packages) - set(pre_packages))

if missing:
    raise SystemExit(f"existing lock packages removed: {missing!r}")
if changed:
    raise SystemExit(f"existing lock packages changed: {changed!r}")

expected_key = ("symthaea-assurance-linux-tpm-ima-pcr-anchor", "0.1.0", "")
if extra != [expected_key]:
    raise SystemExit(f"unexpected lock additions: {extra!r}")

new_package = post_packages[expected_key]
expected_dependencies = {
    "blake3",
    "serde",
    "sha2",
    "symthaea-assurance-tpm2-attestation-possession",
    "symthaea-assurance-tpm2-checkquote-adapter",
    "symthaea-assurance-tpm2-platform-qualification",
    "symthaea-assurance-tpm2-possession-policy-binding",
    "symthaea-linux-ima-replay",
}
actual_dependencies = {
    dependency.split()[0]
    for dependency in new_package.get("dependencies", [])
}
if actual_dependencies != expected_dependencies:
    raise SystemExit(
        "new package dependency closure mismatch: "
        f"expected={sorted(expected_dependencies)!r} actual={sorted(actual_dependencies)!r}"
    )

summary = {
    "existing_packages_preserved": len(pre_packages),
    "new_package": new_package,
    "added_package_count": len(extra),
}
audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(audit_path.read_text(), end="")
PY

reconciled_sha256="$(sha256sum Cargo.lock | awk '{print $1}')"
printf 'source_lock_sha256=%s\n' "$pre_lock_sha256"
printf 'reconciled_lock_sha256=%s\n' "$reconciled_sha256"

# From here on, dependency resolution is frozen to the audited ephemeral lock.
cargo check --locked -p "$package"
cargo test --locked -p "$package"
cargo clippy --locked -p "$package" --all-targets -- -D warnings

test "$(sha256sum Cargo.lock | awk '{print $1}')" = "$reconciled_sha256"
printf 'qualification_result=PASS\n'

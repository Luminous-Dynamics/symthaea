#!/usr/bin/env bash
set -Eeuo pipefail

TOOLCHAIN="${1:-1.100.0}"
EXPECTED="${2:-1.100.0}"
ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

RUSTC_VERSION="$(rustup run "$TOOLCHAIN" rustc --version)"
CARGO_VERSION="$(rustup run "$TOOLCHAIN" cargo --version)"
[[ "$RUSTC_VERSION" == "rustc $EXPECTED "* ]] || {
  echo "expected rustc $EXPECTED; got $RUSTC_VERSION" >&2
  exit 2
}
[[ "$CARGO_VERSION" == "cargo $EXPECTED "* ]] || {
  echo "expected Cargo $EXPECTED; got $CARGO_VERSION" >&2
  exit 2
}

# Audit a disposable exact Git snapshot. Add only the Cargo-native lint policy
# to the two package manifests in the temporary copy; never touch the checked
# out manifests or Cargo.lock.
git archive HEAD | tar -x -C "$WORK"
BEFORE="$(sha256sum "$WORK/Cargo.lock" | awk '{print $1}')"

for package in symthaea-core symthaea-fep; do
  case "$package" in
    symthaea-core) manifest="$WORK/crates/core/symthaea-core/Cargo.toml" ;;
    symthaea-fep) manifest="$WORK/crates/core/symthaea-fep/Cargo.toml" ;;
  esac
  [[ -f "$manifest" ]] || {
    echo "missing manifest for $package: $manifest" >&2
    exit 2
  }

  if grep -q '^\[lints\.cargo\]' "$manifest"; then
    echo "$package already declares [lints.cargo]; refusing to overwrite its policy" >&2
    exit 2
  fi

  cat >> "$manifest" <<'EOF'

# Added only in a disposable audit copy by audit-cargo-unused-dependencies.sh.
[lints.cargo]
unused_dependencies = "warn"
EOF

  LOG="$WORK/$package-unused-dependencies.log"
  echo "=== $package / Cargo unused_dependencies ==="
  set +e
  (
    cd "$WORK"
    rustup run "$TOOLCHAIN" cargo check \
      --locked --all-targets --package "$package" \
      --message-format=short
  ) > "$LOG" 2>&1
  status=$?
  set -e
  cat "$LOG"

  count="$(grep -Eic 'unused dependency' "$LOG" || true)"
  printf 'package=%s exit=%s unused_dependency_warnings=%s\n' "$package" "$status" "$count"

  if [[ $status -ne 0 ]]; then
    echo "AUDIT_INCOMPLETE: all-target Cargo check failed for $package" >&2
    exit "$status"
  fi
done

AFTER="$(sha256sum "$WORK/Cargo.lock" | awk '{print $1}')"
[[ "$BEFORE" == "$AFTER" ]] || {
  echo "the disposable audit changed Cargo.lock despite --locked" >&2
  exit 1
}

echo "rustc_identity=$RUSTC_VERSION"
echo "cargo_identity=$CARGO_VERSION"
echo "temporary_source_head=$(git rev-parse HEAD)"
echo "source_lock_sha256=$BEFORE"
echo "source_lock_unchanged=PASS"
echo "manifest_policy_scope=temporary archive only"
echo "default manifests and lockfile=untouched"
echo "cargo_unused_dependency_audit=COMPLETE"

#!/usr/bin/env bash
# Showroom Ring-0 dependency + license hygiene.
#
# The historical workflow described this as a "forbidden dependencies" check
# but searched every repository file for vocabulary such as Holochain, robotics,
# or PQC. Those concepts legitimately occur in documentation and other domains,
# so the repo-wide text scan was not a dependency theorem. This helper narrows
# authority to dependency declarations of the crate Showroom actually builds:
# symthaea-core.
set -euo pipefail

readonly SHOWROOM_MANIFEST="${SHOWROOM_MANIFEST:-crates/core/symthaea-core/Cargo.toml}"
readonly FORBIDDEN_RE='mycelix|prism|holochain|swarm|robotics|symtropy[-_]robotics|zero[-_.]?knowledge|zkp|pqc'

extract_dependency_declarations() {
  local manifest="$1"
  awk '
    /^\[/ {
      in_dependencies = ($0 ~ /dependencies\]$/)
      next
    }
    in_dependencies {
      line = $0
      sub(/[[:space:]]*#.*/, "", line)
      if (line ~ /[^[:space:]]/) {
        print FNR ":" line
      }
    }
  ' "$manifest"
}

# Return 0 for clean, 10 for a forbidden dependency declaration, and 20 for a
# scanner failure. Matches are printed so CI logs identify the declaration.
scan_manifest() {
  local manifest="$1"
  local extracted="$2"
  local status

  extract_dependency_declarations "$manifest" > "$extracted"
  if rg -n -i -- "$FORBIDDEN_RE" "$extracted"; then
    return 10
  else
    status=$?
    case "$status" in
      1) return 0 ;;
      *) return 20 ;;
    esac
  fi
}

self_test() {
  local tmpdir status
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' RETURN

  cat > "$tmpdir/clean.toml" <<'EOF'
[package]
name = "showroom-fixture"
description = "Documentation may discuss Holochain, robotics, or PQC."

[dependencies]
serde = "1"
# holochain-client is intentionally discussed here but not declared.

[target.'cfg(unix)'.dependencies]
libc = "0.2"
EOF
  if ! scan_manifest "$tmpdir/clean.toml" "$tmpdir/clean.dependencies"; then
    echo 'self-test failed: prose/comment vocabulary was treated as a dependency' >&2
    return 1
  fi

  cat > "$tmpdir/alias.toml" <<'EOF'
[package]
name = "showroom-fixture"

[dependencies]
bridge = { package = "holochain_client", version = "0.6" }
EOF
  if scan_manifest "$tmpdir/alias.toml" "$tmpdir/alias.dependencies" >/dev/null; then
    echo 'self-test failed: package alias was accepted' >&2
    return 1
  else
    status=$?
    if [[ "$status" -ne 10 ]]; then
      echo "self-test failed: package alias returned unexpected status=$status" >&2
      return 1
    fi
  fi

  cat > "$tmpdir/target.toml" <<'EOF'
[package]
name = "showroom-fixture"

[target.'cfg(target_os = "linux")'.dependencies]
robotics-bridge = "1"
EOF
  if scan_manifest "$tmpdir/target.toml" "$tmpdir/target.dependencies" >/dev/null; then
    echo 'self-test failed: target-specific forbidden dependency was accepted' >&2
    return 1
  else
    status=$?
    if [[ "$status" -ne 10 ]]; then
      echo "self-test failed: target-specific dependency returned unexpected status=$status" >&2
      return 1
    fi
  fi

  echo 'Showroom dependency-scope self-test: PASS'
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--self-test]" >&2
  exit 2
fi

if [[ ! -f "$SHOWROOM_MANIFEST" ]]; then
  echo "Showroom hygiene failed: missing manifest $SHOWROOM_MANIFEST" >&2
  exit 1
fi

readonly tmp_dependencies="$(mktemp)"
trap 'rm -f "$tmp_dependencies"' EXIT
if scan_manifest "$SHOWROOM_MANIFEST" "$tmp_dependencies"; then
  scan_status=0
else
  scan_status=$?
fi
case "$scan_status" in
  0) ;;
  10)
    echo "Showroom hygiene failed: forbidden Ring-0 dependency declaration in $SHOWROOM_MANIFEST" >&2
    exit 1
    ;;
  *)
    echo "Showroom hygiene failed: dependency scanner error (status=$scan_status)" >&2
    exit 1
    ;;
esac

head -n 5 LICENSE | grep -F 'GNU AFFERO GENERAL PUBLIC LICENSE' >/dev/null
echo 'Showroom Ring-0 dependency + AGPL hygiene: PASS'

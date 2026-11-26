#!/usr/bin/env sh
set -eu

expected_commit="dd1dc0393385048c26b2b47ac543859a1ba6cdd2"
if [ -z "$OAR_SOURCE_DIR" ]; then
  echo "Set OAR_SOURCE_DIR to a clone of openactionrecord/oar at the pinned commit" >&2
  exit 2
fi
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 127; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required" >&2; exit 127; }
command -v corepack >/dev/null 2>&1 || { echo "corepack is required to activate pinned pnpm" >&2; exit 127; }

actual_commit=$(git -C "$OAR_SOURCE_DIR" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
  echo "OAR source pin mismatch: expected $expected_commit, got $actual_commit" >&2
  exit 3
fi

node_version=$(node -p 'process.versions.node')
node_major=$(printf '%s' "$node_version" | cut -d. -f1)
node_minor=$(printf '%s' "$node_version" | cut -d. -f2)
if [ "$node_major" -lt 20 ] || { [ "$node_major" -eq 20 ] && [ "$node_minor" -lt 12 ]; }; then
  echo "Node.js >=20.12 is required; found $node_version" >&2
  exit 2
fi

cd "$OAR_SOURCE_DIR"
corepack pnpm@10.28.0 install --frozen-lockfile
corepack pnpm@10.28.0 build
corepack pnpm@10.28.0 typecheck
corepack pnpm@10.28.0 test

#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Structural coverage that must accompany focused Spore package qualification
# before any whole-crate source root may become eligible for focused-only CI.
#
# This does not replace the Rust check/test/Clippy lane. It transfers the cheap
# structural responsibilities that general CI currently provides and that Cargo
# compilation alone cannot prove.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANDIDATE_CRATES=(
  crates/domains/symthaea-boot-protocol
  crates/domains/symthaea-boot-observer
  crates/domains/symthaea-quicken-fb
  crates/domains/symthaea-boot-control
  crates/domains/symthaea-boot-input
  crates/domains/symthaea-boot-ecology-live
  crates/domains/symthaea-boot-visual-clock
  crates/domains/symthaea-boot-presentation
  crates/core/symthaea-spore-continuity
)

for crate in "${CANDIDATE_CRATES[@]}"; do
  [[ -f "$crate/Cargo.toml" ]] || {
    echo "missing candidate manifest: $crate/Cargo.toml" >&2
    exit 1
  }

  # check-orphan-modules.sh deliberately skips crates containing #[path] because
  # its name-based model cannot reason about arbitrary module attachment. A
  # general-CI ratchet may tolerate that skip; focused-only routing may not.
  if grep -rq '#\[path' --include='*.rs' "$crate/src" 2>/dev/null; then
    echo "$crate uses #[path]; focused orphan-module ownership is not proven" >&2
    exit 1
  fi
done

# General CI's workspace-target guard is build-free and catches both missing
# crate targets and manifest-declared targets whose files are not tracked.
echo '== workspace target integrity =='
bash scripts/check-workspace-targets.sh

# rustc/cargo cannot see a .rs file that was added but never wired with `mod`.
# Explicit path mode bypasses the historical quarantine and checks the actual
# candidate crate state.
for crate in "${CANDIDATE_CRATES[@]}"; do
  echo "== orphan module check: $crate =="
  bash scripts/check-orphan-modules.sh "$crate"
done

# The focused boot qualification currently exercises the package default feature
# sets only. The current nine candidate manifests intentionally define no local
# [features] tables. Make that assumption executable: introducing feature flags
# revokes structural coverage until the focused lane is deliberately upgraded to
# check/test/Clippy the relevant feature matrix.
python3 - "${CANDIDATE_CRATES[@]}" <<'PY'
import sys
import tomllib
from pathlib import Path

for crate_arg in sys.argv[1:]:
    manifest = Path(crate_arg) / 'Cargo.toml'
    value = tomllib.loads(manifest.read_text(encoding='utf-8'))
    features = value.get('features')
    if features:
        names = ', '.join(sorted(features))
        raise SystemExit(
            f'{manifest}: candidate defines feature flags ({names}); '
            'focused-only routing requires an explicitly qualified feature matrix first'
        )
PY

# These two checks are part of general CI's workspace-target job today. They are
# intentionally run unchanged rather than reimplemented here, so focused routing
# cannot bypass duplicate-capability adjudication or crate-registry integrity.
echo '== duplicate capability adjudication =='
cargo run --locked -q -p xtask -- duplicate-scan

echo '== crate registry integrity =='
cargo run --locked -q -p xtask -- crate-status

echo 'PASS: focused Spore structural coverage bundle'

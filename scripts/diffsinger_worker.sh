#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Provisions the onnxruntime/numpy Python environment diffsinger_worker.py
# needs (not part of the main Rust flake devShell -- this is an optional,
# separately-provisioned worker, see diffsinger.rs's module docs) and execs
# it. Point SYMTHAEA_DIFFSINGER_WORKER at THIS script, not the .py directly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec nix-shell -p "python3.withPackages(ps: with ps; [onnxruntime numpy])" \
  --run "python3 '$SCRIPT_DIR/diffsinger_worker.py' $*"

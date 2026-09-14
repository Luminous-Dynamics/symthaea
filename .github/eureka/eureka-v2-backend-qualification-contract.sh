#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Canonical executable command contract for EUREKA-002 V2 backend
# qualification. This script has no scientific-experiment execution authority.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <check|test|clippy>" >&2
  exit 2
fi

case "$1" in
  check)
    cargo check -p symthaea-psych-bench --features symthaea-backend --lib --tests
    ;;
  test)
    cargo test -p symthaea-psych-bench --features symthaea-backend --lib benchmarks::eureka -- --nocapture
    ;;
  clippy)
    cargo clippy -p symthaea-psych-bench --features symthaea-backend --lib --tests -- -D warnings
    ;;
  *)
    echo "unknown qualification phase: $1" >&2
    exit 2
    ;;
esac

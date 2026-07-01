#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""CLI wrapper for the shared NixOS configuration causal analyzer."""

from __future__ import annotations

import sys

from symthaea_research.nix import analyze_config


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/etc/nixos/configuration.nix"
    analyze_config(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

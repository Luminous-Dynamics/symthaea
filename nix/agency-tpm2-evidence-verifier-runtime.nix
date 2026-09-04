# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Production-shaped runtime inputs for symthaea-qualification-witness-service.
# The caller must construct `pkgs` from the repository's exact locked nixpkgs.
#
# We deliberately return the canonical Python interpreter and verifier script as
# two separate immutable Nix-store paths. The Rust witness service measures and
# commits both; no shell wrapper or mutable PATH lookup sits between it and the
# verifier process.
{ pkgs
, verifierSource ? ../scripts/agency/verify-tpm2-qualification-evidence.py
}:

let
  verifierScript = pkgs.writeText
    "symthaea-tpm2-evidence-verifier-v1.py"
    (builtins.readFile verifierSource);
in
{
  # nixpkgs' Python derivation exposes the concrete versioned interpreter path;
  # use that rather than the potentially symlinked `bin/python3` convenience
  # name because the Rust policy intentionally requires an already-canonical
  # reviewed executable path.
  pythonExecutable = pkgs.python3.interpreter;
  inherit verifierScript;
}

# Nixward authority-chain qualification manifest

This document records the exact integration surface being qualified by the `qualify/nixward-authority-chain-v1` branch.

Qualification is fail-closed: queued, skipped, cancelled, or mergeable states are not PASS evidence. A PASS requires executed checks against the exact integration head.

Included scoped hardening stages:
- CROSS-045: generic ActionIR / primitive command authority boundary.
- CROSS-048: legacy Sandbox authority boundary.
- CROSS-049: ungoverned `NixOSCommand::Custom` rejection.
- CROSS-050: no direct `NixOSCommand` tuple spawns outside the executor.
- CROSS-051: core service/generation/flake effects are typed; underspecified inference actions are not fabricated as read-only commands.
- CROSS-052: unconstrained `nixos-rebuild` `extra_args` are rejected for real execution.
- CROSS-053: `Custom` safety metadata is conservatively `Destructive`.

## Current exact integration head

`c186e69291970d496efd5359728e4d585afb20e4`

The integration branch now includes both authority architectures:
- top-level Symthaea ActionIR/root executor hardening (CROSS-032/045/056);
- top-level legacy Sandbox hardening (CROSS-048);
- `crates/core/nixward` typed-command hardening (CROSS-049/050/051/052/053).

Qualification is fail-closed: queued, skipped, cancelled, or mergeable states are not PASS evidence. A PASS requires executed checks against this exact integration head.

# Symthaea: Local Notes

(See the monorepo root `CLAUDE.md` for authoritative dev workflow rules —
this file only adds Symthaea-specific detail, it does not override the root.)

## Environment: NixOS 26.05 (Yarara)
- **Immutable Root**: NEVER attempt to install global packages via `pip`, `cargo install`, or `npm`.
- **Builds**: Direct `cargo build`/`cargo test` — no `nix develop` wrapper needed (root CLAUDE.md Rule 1). Use `nix develop` only for CUDA (e.g. `.#broca-gpu`), Python/PyPhi, or ONNX Runtime.
- **Acceleration**: `mold` and `sccache` are pre-configured system-wide (NixOS) for all Rust operations.

## Secrets & Credentials
- **Vault**: Use `~/.cargo/bin/bws secret get <id>` for tokens.
- **Crates.io**: Token lives in BWS under secret ID `736da236-a95f-4dd2-8efc-b42800c9106a` (this UUID is the vault lookup key, NOT the token itself — fetch with `bws secret get`).

## Coding Standards
- **Lie theory / su(2)**: `crates/core/symthaea-core/src/hdc/lie_theory.rs` implements real representation theory (sl(2), su(2), so(3), gl(2), root systems, Killing form, BCH) — 899 lines, 20 tests, "Production" per `MODULE_STATUS.md`. Extend this module rather than duplicating representation-theory math elsewhere.
- **Robotics traits**: every new platform crate MUST implement `symthaea-core/src/embodiment.rs:EmbodimentBridge` (see root CLAUDE.md's Robotics section for the full contract and crate template).

## Autonomous Protocol
- **Commits**: Commit after every phase (root CLAUDE.md Rule 8). Stage only authored files, never `git add -A`.
- **Memory**: Persistent cross-session memory lives at `~/.claude/projects/-srv-luminous-dynamics/memory/` (see root CLAUDE.md's auto-memory section) — there is no local `MEMORY.md` in this directory to maintain.

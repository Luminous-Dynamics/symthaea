# Symthaea Dependency Policy

**Integrity Goal:** Maintain a stable, reproducible, and verifiable dependency tree across the Symthaea ecosystem.

## 1. Path Normalization
- Workspace-level dependencies shared across projects (e.g., `mycelix-zkp-core`, `symthaea-core`) MUST use a single source of truth.
- For Symthaea development, the canonical path for shared Mycelix utilities is `../crates/`.
- Local project crates MUST use relative paths (e.g., `crates/symthaea-types`).

## 2. Patching Rules
- **No unrelated patching:** Never mutate patched crypto or network crates (e.g., `iroh`, `ed25519-dalek`) from feature branches.
- **Surgical Alignment:** Dependency version conflicts (like `sha2` v0.10 vs v0.11) must be resolved in dedicated `dependency/` branches with isolated test plans.
- **Lockfile Review:** Broad `cargo update` calls are prohibited without explicit lockfile diff review.

## 3. Toolchain & Security
- **Rust Version:** 1.95.0 (Latest stable).
- **ZKP Backend:** RISC0 + Winterfell (dual-backend integrity).
- **Crypto Audit:** Patched crates in `patches/` must be audited for security-sensitive changes to generic bounds or RNG assumptions.

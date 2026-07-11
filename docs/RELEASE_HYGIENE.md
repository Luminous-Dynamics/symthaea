# Release Hygiene & Showroom Policy

This document defines the policy for maintaining the public `Symthaea` repository (the "Showroom").

## 1. The Showroom Principle
The public repository is a curated, standalone workspace. It is **not** a mirror of the private monorepo "Workshop."

*   **Public Showroom:** Contains only Ring 0 buildable artifacts, verified documentation, and audited research results.
*   **Private Workshop:** Contains experimental Genesis/Morphogenesis work, strategy docs, and internal integration bridges.

## 2. Integrity Guards (CI/CD)
The Showroom is protected by a dedicated GitHub Action (`.github/workflows/showroom-integrity.yml`). This job must pass for any changes to be considered showroom-safe.

**Mandatory Checks:**
1. **Metadata:** `cargo metadata --no-deps` must pass in the standalone workspace.
2. **Build:** `cargo check -p symthaea-core --lib --offline` must pass.
3. **Hygiene:** Scan for private monorepo paths, secret leakage, and forbidden feature-gate usage.

## 3. Claim Discipline
All technical claims must be labeled with their provenance:
*   `Measured locally`
*   `Prototype benchmark`
*   `Experimental`
*   `Planned external validation`
*   `Not claimed`

## 4. Release Process
1.  **Stage:** Use `scripts/export_public_slice.sh` to generate the staging directory.
2.  **Audit:** Run the hygiene checklist (License, README, buildable integrity).
3.  **Sync:** Export to a clean-room export branch, never force-pushing to `main` without a clean-room verification.

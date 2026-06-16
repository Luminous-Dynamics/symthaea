# Symthaea Workspace Manifest Guide

This document defines the strict workspace path conventions required for the `symthaea` project. Following these rules prevents manifest resolution errors.

## 1. Path Convention
All crate dependencies in the workspace root `Cargo.toml` must follow the `path = "crates/<module>/<crate>"` pattern. 

*   **Core Modules**: Must reside in `crates/core/`.
*   **Domain Modules**: Must reside in `crates/domains/`.
*   **Bridge Modules**: Must reside in `crates/bridges/`.

## 2. Manifest Normalization
If the system fails to resolve a dependency, run the `scripts/maintenance/fix_manifest.sh` script.

## 3. Adding New Crates
When adding a new workspace member:
1. Create the crate directory structure according to the convention above.
2. Register the crate in `Cargo.toml` under the `[workspace] members` list.
3. If the crate depends on others within the workspace, use relative paths starting from the *root* of the workspace (relative to the `Cargo.toml` using that path).

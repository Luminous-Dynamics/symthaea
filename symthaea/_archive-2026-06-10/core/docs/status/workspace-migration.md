# Workspace Migration

Migration status date: 2026-06-15

## Canonical Layout

- Application package: `crates/symthaea-core`
- Reusable core package: `crates/core/symthaea-core`
- Domain packages: `crates/domains`
- Bridge packages: `crates/bridges`
- Workspace automation: `xtask`

The root `Cargo.toml` is now a virtual workspace manifest. Profiles, patches,
shared dependencies, membership, exclusions, and maturity tiers are owned
there.

## Preservation Decisions

- The pre-migration lockfile is retained as
  `Cargo.lock.pre-workspace-migration`.
- Legacy source trees were merged into canonical destinations without
  overwriting canonical files, preserving unique modules for later validation.
- Modified Broca gating work was explicitly retained in the canonical domain
  package.
- Broken migration symlinks were removed. Broken links under datasets, papers,
  generated demonstrations, MuJoCo installations, and build outputs are outside
  the source audit because those artifacts are optional or externally supplied.
- Nested or duplicate packages remain excluded until they can be validated as
  independent workspaces or removed in a dedicated cleanup.

## Invariants

`scripts/audit_paths.py` enforces:

- every canonical path dependency exists and is relative;
- canonical workspace package names are unique;
- source-tree symlinks are valid, with documented artifact exclusions;
- the application package remains a workspace member;
- every application integration-test source is visible to Cargo metadata.

Do not reintroduce root-level application sources or duplicate canonical package
names. New packages belong in the appropriate canonical subtree and must pass
the audit before merge.

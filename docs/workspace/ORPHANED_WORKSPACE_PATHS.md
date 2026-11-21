# Workspace paths excluded because they are not Cargo crates

This inventory records directories that are currently matched by the root Cargo workspace globs but do not contain a \`Cargo.toml\`. They are explicitly excluded until a reviewed crate manifest and package-level qualification exist.

**Important:** an exclusion is not a qualification pass. Source files in an excluded directory are not compiled or tested as workspace members by Cargo.

| Path | Current observed contents | Status / follow-up |
|---|---|---|
| \`crates/core/symthaea\` | Empty \`examples/unused.rs\` placeholder | Not the actual package; the real \`symthaea\` package is at repository root |
| \`crates/core/symthaea-iit\` | Tests-only directory; no \`src/\` or manifest | Test fixture cannot be a workspace crate on its own; decide whether to relocate tests to an owning crate or create the intended crate |
| \`crates/core/symthaea-observation-attestation\` | \`src/lib.rs\` without a manifest | Existing orphan-module work is tracked by #9361; add a package manifest and independent tests before re-inclusion |
| \`crates/domains/symthaea-architecture-engineering\` | \`src/\` Rust files without a manifest | Prototype/source-only; manifest, dependency ownership and package tests still required |
| \`crates/domains/symthaea-dishwasher-design\` | \`src/\` Rust file and docs without a manifest | Prototype/source-only; manifest, dependency ownership and package tests still required |
| \`crates/domains/symthaea-marine-geometry\` | \`src/\` Rust file without a manifest | Prototype/source-only; manifest, dependency ownership and package tests still required |
| \`crates/domains/symthaea-rea-compat\` | \`src/lib.rs\` without a manifest | Prototype/source-only; manifest, dependency ownership and package tests still required |
| \`crates/domains/symthaea-rea-evidence-verifier\` | README only; no \`src/\` or manifest | Documentation placeholder, not an executable crate |

Other excluded paths (for example \`crates/core/nixward-web\`, \`crates/domains/symthaea-lab\`, and \`crates/domains/symthaea-perception\`) have separate explicit workspace-boundary reasons documented adjacent to their manifest entries.

## Re-entry gate

A path may be removed from \`workspace.exclude\` only after:
1. its purpose and owner are identified;
2. a complete \`Cargo.toml\` exists with declared dependencies/features/targets;
3. the crate is independently buildable and its module API is intentional;
4. package tests, formatting and strict Clippy run at exact head;
5. any source-only prototype is not accidentally compiled as production code;
6. its workspace inclusion does not silently pull in abandoned, heavyweight or host-specific dependencies.

The regression checker \`scripts/check_workspace_glob_members.py\` ensures every directory selected by the workspace member globs either contains a Cargo manifest or is explicitly excluded. The exact-head workflow additionally runs Cargo metadata and STT package formatting, so a malformed member is caught before unrelated feature work is blocked.

## Relationship

This inventory was created after STT continuity PR #12870 exposed a workspace metadata failure before Rust formatting. The first missing manifest (\`crates/core/symthaea\`) was not the only affected path. #13080 tracks the workspace-glob repair; #9361 remains the source-specific owner for observation-attestation manifest completion.
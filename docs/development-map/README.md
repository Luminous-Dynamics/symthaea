# Development-map federation

This directory contains **pinned references to independently owned project maps**. It is not a merged graph, event store, trust root, authority engine, source registry or deployment controller.

## Canonical federation manifest

Use [federation-v0.1.1.json](federation-v0.1.1.json), validated by [federation-v0.1.1.schema.json](federation-v0.1.1.schema.json), for current federation work.

- **v0.1.1** adds an explicit manifest-schema identity pin and a two-sided compatibility-receipt contract. An edge cannot become `independently_qualified` without a receipt that binds both member revisions and trees, the exact contract blob, verifier build, configuration-world digest and a passing result.
- [federation-v0.1.json](federation-v0.1.json) and [federation-v0.1.schema.json](federation-v0.1.schema.json) are retained as the original v0.1 format for historical comparison. Do not silently reinterpret or overwrite old v0.1 snapshots.
- [luminous-ecosystem-index-v0.1.json](luminous-ecosystem-index-v0.1.json) is an earlier graph-shaped pointer-index prototype. It is historical context, not the canonical federation manifest.

The individual project maps use the separate [EGDDM snapshot schema v0.1.0](../research/development-map-v0.1.schema.json). Federation manifest version and member-map schema version are deliberately distinct.

## Current members

The v0.1.1 federation manifest pins four public project maps:

| Project | Map size | Subject snapshot scope | Review |
|---|---:|---|---|
| Symthaea | 13 nodes / 18 edges | Root README + selected public issue/workstream observations | [PR #12024](https://github.com/Luminous-Dynamics/symthaea/pull/12024) |
| Mycelix | 17 nodes / 14 edges | Root README maturity projection | [PR #6133](https://github.com/Luminous-Dynamics/mycelix/pull/6133) |
| Symtropy | 20 nodes / 29 edges | README + architecture document | [PR #5134](https://github.com/Luminous-Dynamics/symtropy/pull/5134) |
| Sol Atlas | 41 nodes / 51 edges | DEMO, Cargo, selected Rust paths, decision-reliance, TUF and Earth-observability docs | [PR #792](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/792) |

All four project maps are bounded **source-declared projections**. The 91 nodes and 112 edges describe the contents of these specific map artifacts; they are not a completeness or maturity score.

## Identity model

Each member separates two identities:

1. **Subject snapshot** — the exact source-repository revision and Git tree the map describes.
2. **Map artifact** — the exact branch commit, tree and Git blob that contain the map itself.

The v0.1.1 manifest also pins the exact federation schema artifact. These identities must not be conflated. Updating a map artifact does not, by itself, mean the source repository changed. Conversely, a source revision can change while the last published map remains pinned to the older revision.

## Cross-project compatibility

The v0.1.1 manifest currently contains 11 directional relationship statements. They remain source-declared or contract-versioned-but-unqualified; **none has an independent compatibility receipt**.

- A member map remains authoritative only within its owner project and declared source scope.
- Source-declared integration edges are not API compatibility receipts.
- Shared schema version strings are not conformance proofs; exact schema blob identity is required.
- An independently qualified relationship requires independent authority, a two-sided receipt for both exact member snapshots, contract identity, verifier identity, configuration-world identity and a passing conformance result.
- A missing, stale, unavailable or mismatched owner contract remains unknown/unjoined; never fall back to a moving branch head.
- Authority, finance, governance, identity, sensor, physical actuation and deployment powers never transfer through a federation edge.

## Validation tool

The standalone read-only Rust prototype is in [`tools/development-map-validator`](../../tools/development-map-validator/README.md). It has authored structural and adversarial tests for map snapshots, decision-time evidence, development-trajectory planning, federation manifests v0.1.0/v0.1.1, DYNRETR fixed-budget retrieval/exploration cases, a pure Cargo manifest projection library, and a syntax-only Rust source projection library with parser-derived source spans.

**The Rust source and its 95 unit-test cases have not been compiled or executed.** The prototype does not fetch/authenticate source objects, verify signatures, run builds/tests, inspect CI, or independently qualify facts. A structural pass is not JSON Schema 2020-12 meta-validation or runtime compatibility proof.

## Retrieval benchmark fixture

[DYNRETR-001](https://github.com/Luminous-Dynamics/symthaea/issues/12080) adds a public synthetic seven-case retrieval/exploration corpus and expected oracle under `fixtures/retrieval-exploration-v0.1.*.json`. It separates natural no-gold from wrong-repository counterfactual no-gold and includes feature/world, stale-revision, non-local impact and lexical-decoy controls. Each case pins its own repository revision/tree and fixed token/line budgets. This is authored synthetic data, not a measured retrieval score or an external benchmark reproduction.

## Refresh procedure

1. Fetch each repository's exact default-branch commit and Git tree OID.
2. Fetch the project map at its exact review-branch commit; record artifact commit, artifact tree and Git blob OID.
3. Parse and structurally validate the exact bytes. Keep source-declared and independently observed authorities distinct.
4. Update only the relevant member pin and source-backed relationships; never copy local graph claims into another project's map.
5. Independently fetch every pinned map path at its exact artifact commit and compare the resulting blob OID with the manifest.
6. Update the manifest's schema artifact pin when the federation contract changes; retain prior manifest versions.
7. Leave compiler, CI, runtime, physical, security and independent-adjudication states unchanged unless their own owner evidence has been obtained.

## Claim boundary

The federation is a directory of independent maps, not a merged graph or common authority. It does not establish that the four projects compile together, share compatible runtime contracts, pass end-to-end tests, or are jointly production-ready.


## Read-only Cargo projection

The validator crate now contains a library-only Cargo manifest projector at `src/cargo_manifest.rs`. It emits exact-anchor, `manifest_declared` facts for workspace patterns, package declarations, dependencies, target-specific dependencies, features, explicit targets and build-script declarations. It does not call Cargo, expand globs/features, resolve compiler semantics or execute build scripts. Ten test definitions cover virtual workspaces, packages, dependency kinds, target conditions, inherited declarations, build scripts and anchor failures; they are authored but not executed.


## Source-tree composition increment

The Symthaea research branch now includes a deterministic, read-only composition API that joins a caller-supplied recursive Git tree inventory to its Cargo manifest and Rust syntax projections by exact repository/revision/tree/path/blob identity. It reports partial projection coverage or rejects caller-labelled complete inventories with unprojected regular Cargo.toml / .rs files. Symlinks and gitlinks are not followed. This is still a source-anchor composition, not a compiler-resolved or executable repository twin. The tree inventory's completeness and Git object authenticity remain caller responsibilities.

The branch currently authors 95 Rust unit-test cases across the validator, Cargo projection, Rust syntax projection and tree composition. None has been compiled or executed in this environment; do not treat the test definitions as passing.

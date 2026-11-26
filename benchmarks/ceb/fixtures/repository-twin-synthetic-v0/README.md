# CEB Repository Twin Synthetic Fixture V0

**Status:** Design/development fixture only. Not compiled or executed.
**Task:** CEB-REPO-001.
**Trust boundary:** Static source and Cargo manifest inspection only. Do not run
build scripts, tests, procedural macros, binaries or dependency hooks as part of
fixture intake.

## Workspace

This is a five-crate Rust workspace:

- `domain`: user identifiers, user records and authorization trait/implementation.
- `store`: storage trait and in-memory store implementation.
- `service`: generic service using the storage and authorization abstractions.
- `api`: public API facade, integration tests, optional audit feature and a Windows-only adapter.
- `cli`: binary consumer of the API facade.

## Gold-label boundaries

`gold-graph.v0.json` contains fixture-declared nodes and expected relationships,
with a one-line declaration anchor for every node (1-based line number and exact
trimmed line text). Edges are classified as syntax-declared or expected semantic
relationships.
No edge is labeled compiler-verified: a future independent Rust Analyzer /
compiler pass must supply those receipts before such an authority can be assigned.
The anchors establish source-text provenance only; they are not complete AST spans
or compiler/LSP semantic locations. Static intake must also verify that the anchored declaration actually names the graph node; a line that exists and matches its text can still be assigned to the wrong symbol.

`worlds.v0.json` defines ten design-only Cargo configuration worlds: five
feature-selection modes on each of Linux and Windows. It includes default features
enabled with no explicit flags; no-default-features with no explicit features;
no-default-features with explicit `audit`; no-default-features with explicit
`windows-native`; and no-default-features with both features explicitly enabled.
This covers all four effective feature combinations while preserving the distinct
provenance of Cargo-default `audit` versus explicitly enabled `audit`.

Each world records `default_features_enabled`, `explicit_features`,
`effective_features`, and exact active/inactive sets for seven conditional graph
facts. A no-default/no-feature world ensures optional audit nodes disappear when
all optional features are disabled. The matrix also distinguishes feature
activation from target activation: `windows-native` may be enabled on Linux while
the Windows module and `windows-sys` dependency remain inactive. The contract
workflow verifies all ten distinct configurations, effective-feature sets, exact
target triples and active/inactive facts. Seven world-specific negative controls
include missing no-feature coverage, false/missed activation, forged effective
features, target mismatch and duplicate worlds. These are expected configuration
semantics, not compiler-produced evidence.

`mutations.v0.json` defines deterministic edit scenarios and expected impact
sets. They are scenario specifications, not pre-applied patches.

## Reproducibility status

The fixture files are committed source artifacts, but this initial design set
has not yet been finalized into an immutable fixture revision or tree digest.
The catalog entry remains `design_only`, is excluded from scores and must not
be used as a performance result.

See issue #10173 for gold-label review and qualification work.

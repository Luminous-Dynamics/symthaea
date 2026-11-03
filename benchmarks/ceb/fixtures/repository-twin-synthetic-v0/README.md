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

`worlds.v0.json` defines eight design-only Cargo configuration worlds: Linux
and Windows, each crossed with the default feature set, `audit`, `windows-native`,
and both features. The matrix distinguishes a feature being enabled from a
target-gated module/dependency actually being active. In particular,
`windows-native` may be enabled on Linux while the Windows module and
`windows-sys` dependency remain inactive. The contract workflow checks exact
active/inactive inventories for seven conditional graph facts and includes
negative controls for false activation, missed activation and duplicate worlds.
These are expected configuration semantics, not compiler-produced evidence.

`mutations.v0.json` defines deterministic edit scenarios and expected impact
sets. They are scenario specifications, not pre-applied patches.

## Reproducibility status

The fixture files are committed source artifacts, but this initial design set
has not yet been finalized into an immutable fixture revision or tree digest.
The catalog entry remains `design_only`, is excluded from scores and must not
be used as a performance result.

See issue #10173 for gold-label review and qualification work.

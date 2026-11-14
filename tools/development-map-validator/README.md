# development-map-validator

A dependency-light, read-only Rust validator for EGDDM v0.1 snapshots and project-federation manifests.

## Build and run

Run from this standalone crate directory:

    cargo run --release -- path/to/snapshot.json

Run its tests:

    cargo test

The crate has its own Cargo workspace so it is not implicitly pulled into Symthaea's large root workspace. It is not registered as a Symthaea workspace member.

## Supported document types

The CLI supports:
- **Map snapshots**: graph identity, node/edge shape, provenance, lifecycle, event references, evidence invalidation, authority promotion and decision-time evidence availability.
- **Retrieval/exploration cases and oracles**: per-case source revision/tree pins, fixed token/line budgets, candidate-region identity, availability, configuration-world and temporal-scope checks, distinct natural vs wrong-repository no-gold dispositions, and oracle relevant/forbidden-span consistency.
- **Cargo.toml** via the explicit `cargo-manifest` subcommand: workspace/package declarations, dependency kinds and target predicates, feature declarations, explicit targets and build-script declarations are preserved as manifest-declared facts.
- **Rust source** via the explicit `rust-file` subcommand: modules, declarations, functions, traits, methods, imports, visibility, attributes, opaque item-position macro syntax and parser-derived start/end line-column spans are preserved as syntax-extracted facts.
- **Federation manifests v0.1.0 and v0.1.1**: unique project/repository/map identities, exact map-blob and artifact-commit pins, source-repository references, relationship endpoints, shared-contract bindings, hub self-map identity and claim/authority consistency. V0.1.1 additionally requires a two-sided compatibility receipt before an edge may be independently qualified.

For federation manifests, a successful structural check does **not** fetch member repositories or prove that the pinned Git objects exist. It checks the manifest's internal references and identity formats only. Fetch each map by its exact artifact commit and compare the resulting blob OID independently when constructing or refreshing a federation.

## Cargo manifest projection library

The crate also exposes a pure read-only Rust API:

- `cargo_manifest::project_cargo_manifest(raw_toml, ManifestAnchor)`
- Output type: `CargoManifestProjection`, serializable through Serde.
- Authority is labelled `manifest_declared`; this is not compiler-resolved dependency, API or build evidence.
- The caller must provide the exact repository, revision, tree OID, manifest path and blob OID. The parser validates identity syntax but does not authenticate the Git objects.
- It preserves literal workspace member patterns, workspace dependency aliases, package inheritance declarations, normal/dev/build and target-specific dependencies, optional/default-feature declarations, feature expressions, explicit targets and build-script declarations.
- It does not expand workspace globs or feature expressions, resolve inherited workspace values, infer implicit targets without a separate source-tree view, run Cargo, inspect or execute build scripts, expand procedural macros, or invoke repository code.
- The Cargo and Rust syntax projectors are library APIs plus explicit CLI subcommands. Neither output is compiler-resolved or cryptographically authenticated.

## What it checks

- v0.1 top-level, node, edge, provenance, lifecycle and event field shape;
- supported node, relationship, authority, claim-state, lifecycle and event enumerations;
- revision and tree-digest format / algorithm agreement;
- unique graph and event identifiers;
- edge endpoints, event targets, invalidation-event references and event watermark;
- evidence invalidation/supersession consistency;
- DTEA actor-authority evidence references: owner record/node/event binding, actor/scope/policy match, source-observed versus actor-available ordering, availability by cutoff, and effective validity over the decision time;
- DTEA consulted-evidence validity: explicit complete/incomplete applicability, known open interval edges, reversed interval rejection and start/end boundary handling;
- proposed development actions remain proposals;
- passed_local / ci_passed / independently_qualified claims require the corresponding stronger evidence authority;
- containment edges are acyclic;
- retrieval regions marked available/inactive belong to the exact per-case repository revision; stale regions do not match it;
- inactive-configuration regions require a feature absent from the case world, and available regions require all declared features to be active;
- natural no-gold and wrong-repository counterfactual no-gold remain distinct in the oracle;
- federation members, role edges and source references resolve within the manifest;
- artifact URLs pin exact repository, commit and path rather than a moving branch;
- the hub self-map binds to the exact member map and subject revision/tree;
- shared-schema candidates bind the exact schema blob declared by the federation hub;
- relationship source evidence must come from one of its two endpoint repositories;
- an independently-qualified relationship requires independent authority, a conformance-tested contract binding, and a two-sided receipt matching both pinned member revisions/trees, contract digest, verifier build, configuration world and test outcome.

## Explicit non-goals

This is **not** a full JSON Schema 2020-12 implementation. It does not authenticate Git object IDs or external file digests, dereference source URIs, verify signatures, prove event-stream completeness, execute repository code, run tests, inspect CI, or independently adjudicate an assertion. A structural pass means only that the supplied JSON satisfies this validator's local contract checks.

The validator never executes commands from a repository or follows links from a snapshot. Its JSON validation mode reads one JSON file and produces deterministic diagnostics. Its Cargo mode parses one TOML file as data and emits a deterministic JSON projection; it does not run Cargo, execute build scripts, expand features, resolve workspace globs or verify supplied Git object IDs. Its Rust mode parses one Rust source file into syntax facts; it does not follow out-of-line modules, expand macros, resolve imports, evaluate cfg attributes or claim compiler-resolved spans/semantics.

## Status

V0.1 prototype source and negative-control tests are authored for map snapshots, decision-time authority evidence, development-trajectory planning, fixed-budget retrieval/exploration cases and four-project federation manifests (including versioned federation v0.1.1). Seventy-nine Rust unit-test cases are authored at the current branch head (59 validator/DTEA/federation/retrieval cases, 10 Cargo-projection cases and 10 Rust-syntax cases); compilation and execution remain pending. Do not treat unexecuted tests as passing.

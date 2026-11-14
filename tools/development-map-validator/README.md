# development-map-validator

A dependency-light, read-only Rust validator for EGDDM v0.1 snapshots and project-federation manifests.

## Build and run

Run from this standalone crate directory:

    cargo run --release -- path/to/snapshot.json

Run its tests:

    cargo test

The crate has its own Cargo workspace so it is not implicitly pulled into Symthaea's large root workspace. It is not registered as a Symthaea workspace member.

## Supported document types

The CLI autodetects:
- **Map snapshots**: graph identity, node/edge shape, provenance, lifecycle, event references, evidence invalidation, authority promotion and decision-time evidence availability.
- **Federation manifests v0.1.0 and v0.1.1**: unique project/repository/map identities, exact map-blob and artifact-commit pins, source-repository references, relationship endpoints, shared-contract bindings, hub self-map identity and claim/authority consistency. V0.1.1 additionally requires a two-sided compatibility receipt before an edge may be independently qualified.

For federation manifests, a successful structural check does **not** fetch member repositories or prove that the pinned Git objects exist. It checks the manifest's internal references and identity formats only. Fetch each map by its exact artifact commit and compare the resulting blob OID independently when constructing or refreshing a federation.

## What it checks

- v0.1 top-level, node, edge, provenance, lifecycle and event field shape;
- supported node, relationship, authority, claim-state, lifecycle and event enumerations;
- revision and tree-digest format / algorithm agreement;
- unique graph and event identifiers;
- edge endpoints, event targets, invalidation-event references and event watermark;
- evidence invalidation/supersession consistency;
- DTEA actor-authority evidence references: owner record/node/event binding, actor/scope/policy match, source-observed versus actor-available ordering, availability by cutoff, and effective validity over the decision time;
- proposed development actions remain proposals;
- passed_local / ci_passed / independently_qualified claims require the corresponding stronger evidence authority;
- containment edges are acyclic;
- federation members, role edges and source references resolve within the manifest;
- artifact URLs pin exact repository, commit and path rather than a moving branch;
- the hub self-map binds to the exact member map and subject revision/tree;
- shared-schema candidates bind the exact schema blob declared by the federation hub;
- relationship source evidence must come from one of its two endpoint repositories;
- an independently-qualified relationship requires independent authority, a conformance-tested contract binding, and a two-sided receipt matching both pinned member revisions/trees, contract digest, verifier build, configuration world and test outcome.

## Explicit non-goals

This is **not** a full JSON Schema 2020-12 implementation. It does not authenticate Git object IDs or external file digests, dereference source URIs, verify signatures, prove event-stream completeness, execute repository code, run tests, inspect CI, or independently adjudicate an assertion. A structural pass means only that the supplied JSON satisfies this validator's local contract checks.

The validator never executes commands from a repository or follows links from a snapshot. It reads one JSON file and produces deterministic diagnostics.

## Status

V0.1 prototype source and negative-control tests are authored for map snapshots, decision-time authority evidence, development-trajectory planning and four-project federation manifests (including versioned federation v0.1.1). Forty-one Rust unit-test cases are authored at the current branch head; compilation and execution remain pending. Do not treat unexecuted tests as passing.

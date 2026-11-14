# development-map-validator

A dependency-light, read-only Rust validator for the EGDDM v0.1 snapshot contract.

## Build and run

Run from this standalone crate directory:

    cargo run --release -- path/to/snapshot.json

Run its tests:

    cargo test

The crate has its own Cargo workspace so it is not implicitly pulled into Symthaea's large root workspace. It is not registered as a Symthaea workspace member.

## What it checks

- v0.1 top-level, node, edge, provenance, lifecycle and event field shape;
- supported node, relationship, authority, claim-state, lifecycle and event enumerations;
- revision and tree-digest format / algorithm agreement;
- unique graph and event identifiers;
- edge endpoints, event targets, invalidation-event references and event watermark;
- evidence invalidation/supersession consistency;
- proposed development actions remain proposals;
- passed_local / ci_passed / independently_qualified claims require the corresponding stronger evidence authority;
- containment edges are acyclic.

## Explicit non-goals

This is **not** a full JSON Schema 2020-12 implementation. It does not authenticate Git object IDs or external file digests, dereference source URIs, verify signatures, prove event-stream completeness, execute repository code, run tests, inspect CI, or independently adjudicate an assertion. A structural pass means only that the supplied JSON satisfies this validator's local contract checks.

The validator never executes commands from a repository or follows links from a snapshot. It reads one JSON file and produces deterministic diagnostics.

## Status

V0.1 prototype source and negative-control tests are authored. Compilation and test execution remain pending the repository's qualified execution-isolation path; do not treat unexecuted tests as passing.

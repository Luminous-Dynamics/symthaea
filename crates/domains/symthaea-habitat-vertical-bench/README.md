# symthaea-habitat-vertical-bench

Synthetic-only structural validator for the HABITAT-VERTICAL-001 household-goods factory/supply-chain fixture.

## Run

From the Symthaea workspace root:

```sh
cargo run -p symthaea-habitat-vertical-bench --bin validate-habitat-vertical
cargo test -p symthaea-habitat-vertical-bench
```

Validate an alternate fixture by passing its path:

```sh
cargo run -p symthaea-habitat-vertical-bench --bin validate-habitat-vertical -- path/to/fixture.json
```

The default fixture is embedded at compile time from `docs/research/fixtures/habitat-vertical-v0.1.json`. A successful run means only that the fixture parses and meets the structural invariants implemented here. The CLI explicitly states that the negative controls are declared but not executed by this binary.

## Checks

- required schema/program/evidence identity and synthetic-only claim ceiling;
- unique IDs for supplier, lot, resource, operation, target, metric, disruption, negative-control, owner and factory-zone records;
- every material lot's supplier reference resolves;
- every process operation's input lots, required resource, qualified resource skills and predecessors resolve;
- positive integer operation durations and non-empty output-state labels;
- directed acyclic process-predecessor graph using the existing `symthaea-graph-theory` crate;
- exactly one final inspection, requiring inspection and measurement skills and a required hold point;
- complete negative-control and disruption descriptions;
- proposed target status and the delivered-cost ratio range;
- synthetic-only physical-measurement, factory-commissioning, human-evaluation and replication states.

## Deliberate limits

This is a schema/consistency qualifier, not a scheduler, optimizer, production-execution engine, supplier verifier, material-property validator, factory code/safety reviewer, costing audit, LCA engine or product qualification system. It does not execute the negative-control scenarios, prove lot genealogy completeness, test cost formulas against observed books, verify a real factory layout, or show that any proposed target is achieved. Those require separate known-answer tests, exact current owner APIs, and eventually measured evidence.

The crate uses the existing graph-theory implementation rather than adding another graph engine. It is a research/benchmark consumer and must not be treated as physical execution authority.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8051

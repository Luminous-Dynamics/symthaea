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

The default fixture is embedded at compile time from `docs/research/fixtures/habitat-vertical-v0.1.json`. The CLI also validates the embedded definition-only campaign in `docs/research/fixtures/habitat-vertical-scenarios-v0.1.json` and the synthetic replay contract in `docs/research/fixtures/habitat-event-replay-v0.1.json`. A successful run means only that fixture structure, synthetic arithmetic controls, the eight scenario definitions, and event identity/causal replay are internally consistent. It does not run a scheduler or simulate a disruption. The CLI prints the synthetic cost ratio and mass-balance known answers; negative controls are declared and only the mutations covered by unit tests are exercised by the test suite.

## Checks

- delivered-cost known-answer arithmetic across 12 mandatory cost categories, positive accepted-unit denominators, exact currency matching, checked integer summation, and exact-ratio comparison without floating-point pass/fail rounding;
- single-stream material-balance accounting with explicit units/boundary, checked integer totals, required residual matching, tolerance enforcement, closure ratio and accepted-mass yield;
- required schema/program/evidence identity and synthetic-only claim ceiling;
- unique IDs for supplier, lot, resource, operation, target, metric, disruption, negative-control, owner and factory-zone records;
- every material lot's supplier reference resolves;
- every process operation's input lots, required resource, qualified resource skills and predecessors resolve;
- positive integer operation durations and non-empty output-state labels;
- directed acyclic process-predecessor graph using the existing `symthaea-graph-theory` crate;
- directed material-lot containment queries from lot → operation → work-in-process → accepted article → packed unit;
- exact known-answer descendant sets, including two affected packed units and an unrelated control unit;
- exactly one final inspection, requiring inspection and measurement skills and a required hold point;
- complete negative-control and disruption descriptions;
- a definition-only counterfactual campaign covering exactly the eight source disruption IDs, with per-case interventions, hard invariants, required outputs, metric references, and unresolved execution blockers;
- a synthetic resilience-options qualifier covering 6 strategy concepts × 8 disruptions = 48 challenge-oracle cells, exact source-scenario coverage, integer interval ordering, localized mitigation invariants, and precondition gates;
- explicit refusal to assign scenario probabilities, compute ROI/expected annual loss, monetize service loss, or run/select a robust frontier without the existing qualified owner APIs;
- a factory assumption-falsification qualifier covering 9 claims and 10 candidate tests, with source-linked option/scenario dependencies, concrete falsifiers, distinct support/failure/indeterminate outcomes, explicit not-run selection and execution states, and no numeric experiment cost/probability/results;\n- a manufacturing-event replay qualifier covering event identity, exact-redelivery idempotence, conflicting-ID rejection, strict UTC clock ordering, deterministic causal replay, source-graph references and causal object ancestry; it explicitly does not compute cryptographic digests or model quantities;
- no fabricated numeric campaign results, no implicit baseline/candidate policy binding, and no confusion between campaign definition validation and simulation;
- proposed target status and the delivered-cost ratio range;
- synthetic-only physical-measurement, factory-commissioning, human-evaluation and replication states.

## Tests and verification status

The source currently contains 84 authored Rust unit tests across four modules (31 structural/accounting + 13 scenario-campaign + 13 resilience-option + 13 assumption-falsification + 14 event-replay tests). They have not been compiled or executed in the current environment; do not describe them as passing. Run the commands above against the exact branch head before treating the qualifier as an executed test result. The scenario, resilience-option and assumption-falsification suites qualify definitions only, not scheduling, experiment selection, or resilience outcomes.

## Deliberate limits

This is a schema/consistency, synthetic arithmetic and scenario-definition qualifier, not a scheduler, optimizer, production-execution engine, supplier verifier, material-property validator, factory code/safety reviewer, costing audit, LCA engine or product qualification system. It checks the frozen graph's exact known-answer descendant sets and the eight disruption definitions; it does not prove physical lot-genealogy completeness, quarantine a real article, execute disruptions, compare baseline/candidate schedules, test cost formulas against observed books, verify a real factory layout, or show that any proposed target is achieved. The household-goods policy identities and scheduler adapter remain unbound. The option matrix is a constructed oracle for future interval-frontier tests, not realistic predicted outcomes. The assumption-falsification fixture is a proposed test plan: no experiment has been selected or executed, experiment cost/time is unmeasured, and outcome probabilities are not supplied. Those steps require exact current owner APIs and eventually measured evidence.

The crate uses the existing graph-theory implementation rather than adding another graph engine. It is a research/benchmark consumer and must not be treated as physical execution authority.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8051


## Manufacturing event replay (synthetic only)

The default CLI also validates `docs/research/fixtures/habitat-event-replay-v0.1.json`. The prototype detects conflicting duplicate identities, missing causal records, causal cycles/time inversions, unknown process/object references, and inputs outside declared causal ancestry. Exact redeliveries are counted and deduplicated, and the canonical event sequence is deterministic despite arrival-order permutation. The event protocol is inspired by GS1 EPCIS but is **not EPCIS-conformant**. Source authentication, signatures, payload digests, quantitative inventory allocation, and physical event truth are not implemented. These new tests are authored, not compiled or executed in the current environment.

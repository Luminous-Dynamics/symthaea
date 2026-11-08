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

The default fixture is embedded at compile time from `docs/research/fixtures/habitat-vertical-v0.1.json`. The CLI also validates the definition-only campaign, event replay, quantity-aware material ledger, interval-aware measurement gate, cross-layer audit bundle (now including the recall-response fixture), and recall-response drill. A successful run means only that fixture structure, synthetic arithmetic controls, scenario definitions, event identity/causal replay, the synthetic quantity ledger, measurement-decision known answers, and exact fixture-byte fingerprints are internally consistent. It does not run a scheduler or simulate a disruption. The CLI prints the synthetic cost ratio and mass-balance known answers; negative controls are declared and only the mutations covered by unit tests are exercised by the test suite.

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
- a factory assumption-falsification qualifier covering 9 claims and 10 candidate tests, with source-linked option/scenario dependencies, concrete falsifiers, distinct support/failure/indeterminate outcomes, explicit not-run selection and execution states, and no numeric experiment cost/probability/results;
- a manufacturing-event replay qualifier covering event identity, exact-redelivery idempotence, conflicting-ID rejection, strict UTC clock ordering, deterministic causal replay, source-graph references and causal object ancestry; it explicitly does not compute cryptographic digests or model quantities;
- a quantity-aware material ledger covering 16 accounts and nine ordered transactions, exact integer mg conservation, account overdraw prevention, source-event/operation references, byproduct and rework disposition, and frozen closing-balance known answers; the piece-equivalent-to-mg conversion remains explicitly blocked;
- an interval-aware measurement gate with five explicit dispositions, checked integer uncertainty bounds, unit and calibration-validity checks, and rejection of unsupported probability claims;
- a cross-layer audit bundle that pins exact UTF-8 byte lengths and SHA-256 fingerprints for the five source fixtures, reruns all component validators, and checks event/ledger/measurement summaries against an independent manifest; digest matches are not source authentication;
- no fabricated numeric campaign results, no implicit baseline/candidate policy binding, and no confusion between campaign definition validation and simulation;
- proposed target status and the delivered-cost ratio range;
- synthetic-only physical-measurement, factory-commissioning, human-evaluation and replication states.

## Tests and verification status

The source currently contains 170 authored Rust unit tests across ten modules (31 structural/accounting + 13 scenario-campaign + 13 resilience-option + 13 assumption-falsification + 15 event-replay + 15 material-ledger + 17 measurement-gate + 19 audit-bundle + 17 recall-drill + 17 quantity-genealogy tests). They have not been compiled or executed in the current environment; do not describe them as passing. Run the commands above against the exact branch head before treating the qualifier as an executed test result. The scenario, resilience-option and assumption-falsification suites qualify definitions only, not scheduling, experiment selection, or resilience outcomes.

## Deliberate limits

This is a schema/consistency, synthetic arithmetic and scenario-definition qualifier, not a scheduler, optimizer, production-execution engine, supplier verifier, material-property validator, factory code/safety reviewer, costing audit, LCA engine or product qualification system. It checks the frozen graph's exact known-answer descendant sets and the eight disruption definitions; it does not prove physical lot-genealogy completeness, quarantine a real article, execute disruptions, compare baseline/candidate schedules, test cost formulas against observed books, verify a real factory layout, or show that any proposed target is achieved. The household-goods policy identities and scheduler adapter remain unbound. The option matrix is a constructed oracle for future interval-frontier tests, not realistic predicted outcomes. The assumption-falsification fixture is a proposed test plan: no experiment has been selected or executed, experiment cost/time is unmeasured, and outcome probabilities are not supplied. Those steps require exact current owner APIs and eventually measured evidence.

The crate uses the existing graph-theory implementation rather than adding another graph engine. It is a research/benchmark consumer and must not be treated as physical execution authority.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8051


## Manufacturing event replay (synthetic only)

The default CLI also validates `docs/research/fixtures/habitat-event-replay-v0.1.json`. The prototype detects conflicting duplicate identities, missing causal records, causal cycles/time inversions, unknown process/object references, and inputs outside declared causal ancestry. Exact redeliveries are counted and deduplicated, and the canonical event sequence is deterministic despite arrival-order permutation. The event protocol is inspired by GS1 EPCIS but is **not EPCIS-conformant**. Source authentication, signatures, payload digests, quantitative inventory allocation, and physical event truth are not implemented. These new tests are authored, not compiled or executed in the current environment.


## Quantity-aware material ledger (synthetic only)

The default embedded-fixture CLI also validates `docs/research/fixtures/habitat-material-ledger-v0.1.json`. Its nine transactions account for a single textile-fiber component-mass stream through cutting, sewing, rework recovery, inspection, packaging, and dispatch. Each transaction must conserve exact integer milligrams and each account must remain nonnegative; final account balances and known-answer totals are checked. The fixture reuses event and traceability identities as contextual references, but its quantities are an independent synthetic overlay—not source measurements. Conversion from the base fixture's `piece_equivalent` to mg is blocked until a qualified conversion exists. These tests are authored, not compiled or executed in the current environment.


## Cross-layer audit bundle (synthetic only)

The default CLI validates `docs/research/fixtures/habitat-audit-bundle-v0.1.json`. It checks exact UTF-8 byte lengths and SHA-256 fingerprints of the base, event-replay, material-ledger, and measurement-gate fixtures, reruns each existing validator, and compares component summaries against the manifest. Hash matches establish byte identity only: there is no signature, trusted timestamp, source authentication, or proof of physical truth. These tests are authored, not compiled or executed.


## Recall-response closure drill (synthetic only)

The default CLI validates `docs/research/fixtures/habitat-recall-closure-v0.1.json` against the base traceability graph and the event-replay fixture. It derives affected packaged-unit IDs from causal event inputs and outputs; verifies the complete packaged/control target universe; requires exactly one explicit response record per target; rejects affected-as-unaffected outcomes and containment actions on unrelated controls; and refuses to derive a result when the event source is invalid. The known-answer drill is structurally valid but deliberately remains **OPEN_UNRESOLVED** because one affected packaged unit is not located. This is a synthetic oracle status, not a real recall, physical quarantine, customer contact or legal-compliance result. Seventeen new tests are authored but have not been compiled or executed.


The cross-layer audit bundle now fingerprints the recall-response fixture as a fifth source, reruns its causal recall qualifier, and independently checks the derived affected set and response closure state. A bundle may be internally valid while its recall state remains `OPEN_UNRESOLVED`; audit consistency must never be confused with physical recall closure. Three additional audit-bundle tests challenge stale recall bytes, tampered summary outcomes and component failures. They are authored, not yet compiled or executed.


The cross-layer audit bundle now fingerprints six inputs (base product/process, event replay, material ledger, measurement gate, recall response, and quantity genealogy) by exact byte length and SHA-256. It reruns all five relevant validators and independently compares the response closure counts/status. The expected recall state intentionally remains `OPEN_UNRESOLVED`; bundle validity means the synthetic sources and expected results agree, not that a real recall is closed.


## Quantity-aware genealogy challenge (synthetic only)

The default CLI also validates `docs/research/fixtures/habitat-quantity-genealogy-v0.1.json`. Its eight operations model splitting one suspect source lot into separate offcuts and scrap, partially consuming one offcut into a merge with a clean source, then assembling and packaging affected products alongside an unrelated clean control. Each input-to-output allocation includes explicit source quantities; the qualifier does not infer a mixing ratio. It checks input availability, allocation completeness, output quantities, same-unit mass conservation, source-attribution availability, remaining stock and the terminal affected set. The known answer preserves 2,700 mg across opening and final inventories and identifies two affected synthetic terminals. This is not physical genealogy or EPCIS conformance. Seventeen Rust tests are authored but not compiled or executed.


The six-source cross-layer audit bundle reruns both recall-response and quantity-genealogy validators. It checks exact source-byte fingerprints, source-attribution and mass known answers, the affected terminal set and the intentionally OPEN_UNRESOLVED recall status. Three audit mutation tests cover stale genealogy bytes, tampered mass known answers and a component-level unit mutation. Total documented authored tests: 170; none have been compiled or executed.

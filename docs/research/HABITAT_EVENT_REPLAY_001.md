# HABITAT-EVENT-REPLAY-001 — manufacturing event truth under imperfect delivery

**Status:** synthetic protocol and Rust qualifier added to HABITAT-VERTICAL-001; no physical event source is connected and no manufacturing execution is authorized.

## Paradigm shift

Treat operational history as a set of claims that must be **reconciled and replayed**, not as a spreadsheet or dashboard snapshot assumed to be current. Receiving, inspection, transformation, packaging, and dispatch records may arrive late or out of order; delivery may repeat an event; two payloads may disagree about the same stable event identity. A useful production system must distinguish those conditions from a valid event stream before it recalculates plans, containment, or service impact.

This slice adds no second scheduler, optimizer, event bus, cryptographic ledger, or autonomous control plane. It qualifies a small event envelope against the frozen HABITAT-VERTICAL-001 synthetic traceability and process graphs.

## Research grounding and interoperability boundary

- NIST's [Digital Twins Workshops Summary Report (NISTIR 8620, July 21, 2026)](https://www.nist.gov/publications/digital-twins-workshops-summary-report) identifies interoperability and verification, validation, and uncertainty quantification as persistent manufacturing-digital-twin challenges.
- GS1 describes [EPCIS](https://www.gs1.org/standards/epcis) as a way to exchange records of what happened to objects, when and where it happened, and the business context; the [EPCIS/CBV 2.0.1 artefacts](https://ref.gs1.org/standards/epcis/artefacts) provide current standard schemas and examples.

The fixture is **inspired by** those event concepts; it is not EPCIS JSON-LD and does **not** claim conformance. Real adoption must validate against the relevant GS1 schema/profile and partner agreements, identifiers and access controls. The fixture uses only synthetic local IDs.

## Executable contract

- Stable `event_id`; byte-structure-equivalent redelivery is idempotent when only `ingested_at_utc` differs. Same ID with a different payload fails closed.
- Separate event-effective, source-recorded, and ingestion timestamps, all strict UTC `Z` timestamps. Reject source/ingestion clock inversions and causal references to a later effective event.
- Deterministic causal topological replay; ties use effective time and event ID. Arrival order must not affect replay order.
- Every operation input and non-receiving subject must occur in the event's declared causal ancestry; all referenced objects and operation IDs must resolve against the base synthetic graph.
- Event types and dispositions are closed-world in this prototype. Unknown types, missing identities, malformed references, causal cycles and incomplete transformation inputs/outputs are findings, not defaults.
- Unknown, duplicate and conflicting conditions are reported separately; structural pass is never relabeled as physical qualification.

Embedded fixture: `docs/research/fixtures/habitat-event-replay-v0.1.json`. Rust module: `crates/domains/symthaea-habitat-vertical-bench/src/event_replay.rs`.

## Benchmark targets (proposed, not achieved)

1. **Replay determinism:** 100% byte-for-byte-equivalent canonical event-ID ordering for every permutation of the same valid event set.
2. **Idempotence:** zero change to canonical output from exact redelivery, with all duplicate deliveries counted.
3. **Conflict containment:** zero conflicting event identities admitted; never silently pick first/last writer.
4. **Causal/lineage integrity:** 100% of known-answer valid references admitted and 100% of missing, cyclic, inverted, or outside-ancestry negative controls rejected.
5. **Fail-closed reporting:** zero synthetic unknowns or blocked cases silently converted to physical facts, numeric zero, or pass.
6. **Scale target after profiling:** benchmark 1,000 and 10,000 events under release optimization, pin hardware/compiler/fixture digests, report median/p95 latency and peak memory. Set a service-level latency budget only after a reference implementation and workload are measured; no current performance result is claimed.

A future integrated benchmark should compare snapshot-only planning against replay-verified planning under identical disruptions, measuring time to correct containment, false quarantine, missed affected products, schedule invalidations, lost accepted-unit-hours, and operator review burden. No such paired run is executed here.

## Explicit limitations

- All actors, places, timestamps, IDs and business events are synthetic.
- Payload digests are **not computed**; this module provides no tamper-proofing, digital signatures, source authentication, trusted timestamping, or cryptographic identity.
- Quantity allocation, inventory depletion, partial consumption, material balance and reservation semantics are **not modeled**. A valid causal path is not proof that enough material existed or that the physical transformation occurred.
- No live EPCIS feed, Holochain/Mycelix event store, production scheduler, supplier, device, sensor, safety system or physical facility is connected.
- Passing this qualifier means only that the small event fixture is structurally and causally consistent with its known answer. It does not prove genealogy completeness in the real world, a recall is safe, production is feasible, or any cost/quality/service improvement has been achieved.

Next integration gate: agree a versioned adapter with Mycelix MFG-EVID-001 (#3084), MFG-PROC-001 (#3085) and LOG-COMMONS-001 (#4900); add real source attestation and digest rules under a separately reviewed protocol; then execute mutation/scale tests on pinned data before connecting the existing scheduler or resilience decision owners.

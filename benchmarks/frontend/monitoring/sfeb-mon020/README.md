# MON-020 — Provenance Event-Order Assurance

**Status:** research contract and local Go semantic prototype; runtime source adapters and independent qualification pending.

MON-020 addresses a subtle gap in ordinary DAG checks: an acyclic provenance graph can still describe an impossible history if an entity is used before it was generated. It qualifies the event order behind provenance relationships using authenticated source-local sequence domains and independently witnessed cross-source order.

## Research question

Can an independent verifier establish the required order between generation, use, invalidation, activity start/end and cross-source handoffs without trusting wall-clock timestamps or the producer's own sequence claims?

## Files

- [EVENT-ORDER-CONTRACT-v1.md](./EVENT-ORDER-CONTRACT-v1.md) — semantics, proof model, threat model and benchmarks.
- [SCENARIOS-v1.json](./SCENARIOS-v1.json) — synthetic adversarial seed corpus.
- [SCENARIOS-v1.schema.json](./SCENARIOS-v1.schema.json) — strict JSON Schema.
- [event-order/](./event-order/) — standard-library Go event-order oracle, adversarial tests and scale benchmarks.
- [BENCHMARKS-v0.1.md](./BENCHMARKS-v0.1.md) — local same-source and cross-source order measurements.

## Standards and prior art

W3C PROV Constraints specifies event ordering, including generation-before-use and activity-boundary constraints. MON-020 reuses that semantic requirement while adding an evidence-qualified method for establishing order across source domains. See [PROV Constraints](https://www.w3.org/TR/prov-constraints/).

## Claim ceiling

This prototype validates signed/verified event evidence and order witnesses supplied to it. It does not perform cryptographic verification itself, authenticate the transport for reports, prove factual truth, establish a complete source history outside the submitted event set, or prove destination/physical effects.

# HABITAT-RECALL-CLOSURE-001 — traceability is not recall closure

**Status:** synthetic response drill and Rust qualifier authored; it has not been compiled or executed, and no real recall or physical containment occurred.

## Paradigm shift

A traceability query answers which downstream products may be affected by a suspect lot. It does not show that the affected products were found, contained, withdrawn, or otherwise addressed. Operational recall closure is a distinct, evidence-bearing state transition.

This slice takes the frozen causal event stream and:
1. propagates a suspect material lot through event input/output relationships to the defined terminal packaged-product set;
2. compares the independently declared known-answer set with the computed set;
3. requires exactly one typed response for every affected and unaffected terminal target;
4. rejects an affected target marked as unaffected, a containment action assigned to an unaffected control, duplicate/missing target responses, unknown outcomes, and event streams that fail the causal replay validator; and
5. leaves the recall open while any affected target remains unlocated or pending.

The embedded synthetic challenge intentionally has one of two affected units quarantined, the other not located, and an unrelated control confirmed unaffected. The definition is internally consistent, but the expected operational state is **OPEN_UNRESOLVED**, not closed.

## Research grounding and industry comparison

- [ISO 10393:2013 — Consumer product recall](https://www.iso.org/standard/45968.html) is current (reviewed and confirmed in 2025) guidance for consumer-product recalls and corrective actions after a product leaves the manufacturing facility, including repair, replacement, disposal and notification. This prototype is not an ISO 10393 implementation or compliance judgment.
- [GS1 Global Traceability Standard](https://www.gs1.org/standards/gs1-global-traceability-standard/current-standard) provides product-neutral traceability guidance, including event completeness indicators and the distinction between batch/lot identity and serialized item identity. It identifies recall-notification and product-removal-confirmation flows as relevant records. This prototype does not claim GS1 or EPCIS conformance.
- [NIST's July 2026 digital-twin workshop report](https://www.nist.gov/publications/digital-twins-workshops-summary-report) emphasizes interoperability and verification, validation and uncertainty quantification as continuing manufacturing challenges. The response drill is designed as a falsifiable small oracle—not as proof of physical-model credibility.

Industry comparison must be empirical. The standards establish useful requirements and vocabulary, not a common performance leaderboard. We should compare a reference implementation and interoperable traceability products against the same frozen challenge fixtures and outcomes before making a “better than SOTA” claim.

## Benchmarks

**Hard correctness gates (proposed, not measured):**
- 100% exact agreement with the known affected terminal set on the frozen challenge and its mutation corpus.
- Zero missed affected terminal targets; no target with causal evidence may be declared `confirmed_not_affected`.
- Zero unrelated control targets assigned a quarantine, withdrawal, or customer stop-use action in the known-answer challenge.
- 100% of in-scope terminal targets have exactly one typed response; duplicates, absent response, unknown state, or out-of-scope target fail closed.
- An affected target recorded as `not_located` or `action_pending` keeps closure open. A schema-valid response log is not synonymous with a closed recall.

**Performance target (proposed, not measured):**
- 10,000 event records and 10,000 terminal targets; p95 under 5 seconds on a pinned reference CPU and compiler, while preserving exact set equality and every hard correctness gate. Record fixture digest, toolchain, hardware, median/p95 latency and peak memory. Do not claim a performance win before the baseline has been run.

**Operational metrics to measure only with authorized, real evidence:**
- incident-to-candidate-list latency;
- percentage of affected inventory/customer destinations found and response-recorded;
- elapsed time to verified first containment, final affected-item disposition and legally reviewed closure;
- missed affected units, false-positive containment, unresolved status, duplicate/contradictory reports and operator time.
No operational latency, customer-response, legal-compliance or safety result is claimed here.

## Execution design

The Rust module calls the existing event replay validator first and refuses to derive a recall set if the causal stream is invalid. It propagates the suspect identity through each replayed event whenever one or more inputs are affected, then checks the terminal target universe and typed response records. This is conservative for multi-input transformations: any output of an operation consuming a suspect input is considered affected. Real production would need qualified partial-consumption quantities, split/merge genealogy, rework semantics and the applicable risk assessment.

The benchmark fixture is `docs/research/fixtures/habitat-recall-closure-v0.1.json`; the implementation is `crates/domains/symthaea-habitat-vertical-bench/src/recall_drill.rs`. The existing four-source audit bundle has not yet been extended to fingerprint this fifth fixture; that is an explicit integration gap, not an implicit integrity claim.

## Claim ceiling / next gates

- All target states, source records and identities are synthetic.
- `quarantined_at_facility`, `withdrawal_confirmed_at_recipient`, and `customer_stop_use_confirmed` are just oracle labels. No location, recipient, quantity, signature, timestamp, stock ledger or customer acknowledgement is checked.
- The event schema remains an internal synthetic model inspired by event-lineage concepts; no EPCIS JSON-LD validation or standards-conformance claim is made.
- No real recall decision, public notification, supplier escalation, safety disposition, regulatory filing or destructive action is authorized.
- Next: include this fixture in the cross-layer audit bundle; adapt to a versioned event/quantity genealogy API; add partial-consumption and split/merge cases; then run differential benchmarks against an EPCIS-conformant reference, the canonical traceability owners, and adversarially mutated 10k+ event datasets.

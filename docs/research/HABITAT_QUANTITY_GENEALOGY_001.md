# HABITAT-QUANTITY-GENEALOGY-001 — partial consumption, split/merge and source attribution

**Status:** synthetic contract and Rust qualifier authored; not compiled or executed. No physical genealogy is established.

## Paradigm shift

Binary lot reachability is useful for conservative recall, but cannot express how much material was transferred through partial consumption, how source material split across outputs, how several source streams merged, or what remains in inventory. Conversely, scalar quantities alone do not establish lineage.

This benchmark combines both without guessing: each operation declares input and output quantities, explicit input-to-output allocations and source-lot quantities for every allocation. The validator checks conservation at operation and source-lot levels, prevents overdraw, tracks remaining stock and rejects unallocated, unknown or unsupported source attribution. It never infers a mixing ratio. A partial draw from a mixed material must explicitly declare which source quantities it carries.

## Interoperability grounding

GS1 EPCIS 2.0.1 defines TransformationEvents for inputs that may be fully or partially consumed and outputs that are produced; its input/output quantity lists and TransformationID describe linked transformations. The model also describes that inputs may contribute to outputs under a transformation event's semantics. See the [EPCIS 2.0.1 standard](https://ref.gs1.org/standards/epcis/2.0.1/), [TransformationEvent linked-data model](https://ref.gs1.org/epcis/TransformationEvent), and [GS1 implementation guideline](https://www.gs1.org/standards/epcis-and-cbv-implementation-guideline/current-standardd). This fixture is an internal quantity-flow oracle, not EPCIS JSON-LD and not a conformance claim.

Recall relevance is guided by [ISO 10393:2013](https://www.iso.org/standard/45968.html), confirmed current by ISO in 2025. Better genealogy can improve candidate scoping, but recall actions still require suitable risk assessment and trustworthy sources.

## Synthetic known answer

- Eight operations cover a split, partial consumption, merge, assembly, packaging and an unrelated control.
- Opening inventory is 2,700 mg; final stock, work-in-process, scrap and packaged units also total 2,700 mg.
- The suspect 1,000 mg source lot sends 700 mg through a split into two suspect offcuts and 50 mg scrap, leaving 300 mg unused in the source lot. Only 300 mg of the 400 mg first offcut is consumed later, leaving 100 mg as inventory.
- Finished affected units are PACKED-ARTICLE-A and PACKED-ARTICLE-B. PACKED-UNRELATED-CONTROL carries no suspect source lineage.

All values and source allocations are synthetic. No real production lot, material measurement or physical event is claimed.

## Proposed benchmark gates (not achieved)

1. 100% exact affected-terminal known-answer agreement across split, merge, partial use, leftover WIP, scrap and multi-source transformations.
2. Zero admitted input overdraw or mass-conservation failures.
3. 100% explicit source attribution for tracked mass; zero inferred mixing proportions.
4. 100% preservation of each opening source's lineage total across remaining inventory, scrap and terminal units.
5. Zero unrelated controls falsely included in the frozen known answer; fail closed when source allocation is missing or untrusted.
6. Mutation suite rejects duplicate/future IDs, unknown lineage sources, unit mismatches, allocation gaps, source overdraw and tampered known answers.
7. Scale target: 10,000 operations and 100,000 allocation edges; p95 below 5 seconds on pinned hardware. Proposed only, not measured.

## Claim ceiling and next gates

The current model covers one conserved same-unit mass stream. Real production can add/remove water, adhesives, solvents or component streams; these need explicit qualified boundary definitions. Cross-unit conversion, instrument validation, physical scanning, EPCIS profile validation, signatures and source authentication are not implemented.

Next: pin this fixture in the cross-layer audit bundle and add differential tests against EPCIS JSON-LD transformation examples without claiming this simplified model is conformant.

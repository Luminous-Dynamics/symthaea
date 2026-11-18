# Atelier Lightshell — Claim-Scoped Change Impact and Requalification Contract v0.1

## Purpose

Map authorized lifecycle changes to the garment qualification claims they affect. A panel repair should not silently inherit the old fit claim; it should also not automatically invalidate an independent sensor calibration or spacer airflow coupon result when their dependencies are unchanged and an owner policy explicitly permits carry-forward.

This is a Lightshell-specific adapter. Mycelix #5985 remains the canonical serialized repair/configuration event owner and #6092 the textile product-passport projection owner. Symtropy #3628 owns endurance/maintainability experiment semantics; Symthaea #12704 owns sensor-tile post-repair calibration/continuity. This contract does not make a second service ledger or maintenance-management system.

## Research context

Papaefstathiou, Elgendi & Menon, *From stretchable materials and functional fibers to intelligent textile sensing systems* (Communications Materials, 21 October 2026), reviews trade-offs between sensitivity, stretchability, durability, fabrication complexity and textile compatibility. Its conclusion that no single approach optimizes every dimension supports claim-scoped, application-specific validation rather than blanket claims. https://www.nature.com/articles/s43246-026-01333-4

A 2026 review on AI-driven sustainable textile materials and healthcare monitoring emphasizes detachable electronics, replaceable contact elements, calibration/continuity checks after repair, and testing under combined wear-and-care exposure. It warns that removable interfaces themselves can become electrical failure points. These are wearable-textile comparators, not robot-garment acceptance targets. https://www.mdpi.com/1996-1944/19/22/4813

A 2026 review of manufacturing-quality digital twins argues that physical intervention needs explicit authority, traceability, fallback and post-action verification; a connected record alone does not demonstrate that a physical intervention was safe or successful. https://doi.org/10.1186/s44147-026-01295-x

## Event and generation semantics

Each event binds a predecessor and successor configuration generation. The event references the canonical lifecycle source, event authority, physical evidence (for an assessed physical event), treatment state and exact predecessor/successor entities.

Replacement events must change the identity of the replaced entity. Reusing the predecessor panel ID as the successor panel ID is rejected. The prior article and its evidence remain immutable and historical.

## Claim dependency semantics

Each claim binds its evidence generation, evidence source and explicit entity dependencies: panel, seam/connection, sensor tile, material lot, process revision, robot geometry/joint, pose set, service feature or zone.

Impact is derived only from an exact intersection between the event's predecessor entity identity and the claim's dependency identity. Rules then match event kind, claim family and affected entity kind.

## Dispositions

- `preserved`: no dependency intersects the change, and the external engineering owner has explicitly authorized carry-forward under a versioned policy.
- `review_required`: an affected claim needs engineering review before reliance.
- `retest_required`: a new test must be run against the successor configuration with an exact plan, method selection, fixture revision and acceptance rule.
- `superseded`: the predecessor claim remains historical but cannot represent the successor component/configuration.
- `not_evaluated`: dependency mapping or impact rule is missing/ambiguous. It is never a preserve.

Multiple matching rules are a conflict and fail closed. An unaffected claim is not automatically preserved because two strings happen to differ; the explicit unaffected-claim owner policy is required.

## Illustrative panel-repair case

The synthetic example replaces a forearm panel and its adjoining seam. It routes the forearm fit claim to a new as-built/pose retest, the seam integrity claim to engineering review, and leaves sensor calibration and material-airflow claims as carry-forward candidates under an explicit unaffected-claim policy.

This is a software fixture, not a real repair. Its claim evidence and event identity are placeholders; it does not prove any claim was physically measured or that the physical repair occurred.

## Integration contracts

- Mycelix #5985: serialized article, repair event and configuration generation lineage.
- Mycelix #6092: evidence-qualified textile product-passport projection.
- Symtropy #3628: endurance, degradation and maintainability profile.
- Symthaea #12704: sensor-tile post-cleaning/post-repair calibration and continuity retention.
- Lightshell #12751/#12759/#12761: coupon evidence, method applicability and mission exposure.
- Lightshell #12785/#12800/#12802: topology, manufacturing route and as-built metrology.

## P0 adversarial controls

Authored tests cover missing event authority, generation mismatch, old evidence relabeled to the successor, replacement without a new entity identity, event/entity-kind mismatch, missing dependency scope, missing impact rule, conflicting rules, missing retest method/fixture/acceptance, sensor-only replacement, material-lot supersession and model-inferred event promotion.

## Benchmarks

- zero false-preservation dispositions against a frozen expert-adjudicated change/dependency oracle corpus;
- zero retest plans targeting predecessor article/configuration generations;
- 100% changed claims receive explicit review/retest/superseded/not-evaluated dispositions;
- 100% retest-required claims bind an applicable method, fixture revision and acceptance rule;
- unaffected-claim preservation accuracy and unnecessary retest burden against manual expert review;
- requalification lead time and repeat failure rate after repair by repair class;
- sensor calibration/continuity retention after replacement and combined cleaning/flex exposure.

Do not infer universal service intervals from published cycle counts. Report exposure, serial-unit variation, censoring and uncertainty.

## Claim boundary

An admitted plan routes engineering follow-up only. It does not prove an event occurred, authenticate its physical truth, execute tests or repair, certify a garment, authorize a robot restart or return-to-service, or grant operational/safety authority.
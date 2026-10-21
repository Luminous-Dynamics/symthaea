# Symthaea Landscape Alternative Review v0

**Status:** versioned artifact contract + adversarial fixture; not yet a production reasoning pipeline.  
**Parent:** [LAND-REVIEW-001 #8196](https://github.com/Luminous-Dynamics/symthaea/issues/8196).  
**Upstream editor:** [Sol-Atlas draft PR #191](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/191).  
**Scenario handoff:** [Symtropy draft PR #2322](https://github.com/Luminous-Dynamics/symtropy/pull/2322).  
**Stewardship handoff:** [Mycelix draft PR #5298](https://github.com/Luminous-Dynamics/mycelix/pull/5298).

## Paradigm shift

The system must not answer “which design is best?” before it has established “what is actually known, what is only proposed, what is modeled, and what is still missing?” For high-consequence landscape decisions, the correct answer can be **no defensible ranking**.

Treat the review as a replayable, read-only artifact that references exact design, metric registry, source catalog, scenario manifest and result digests. Every recommendation needs traceable support, declared uncertainty and a falsification path. Missing/conflicting evidence blocks a metric value instead of becoming zero, a default or an opaque score.

## Contract artifacts

- Machine-readable schema: [landscape-alternative-review-v0.schema.json](../schemas/landscape-alternative-review-v0.schema.json).
- Synthetic adversarial fixture: [review-unranked-v0.json](../fixtures/landscape/review-unranked-v0.json). Its hashes are sentinel values; it is not tied to real artifacts.
- Parent acceptance criteria and initial source profile: [LAND-REVIEW-001 #8196](https://github.com/Luminous-Dynamics/symthaea/issues/8196).

## Inputs and provenance

The v0 review pins:
- exact Sol-Atlas design snapshot SHA-256, schema URI, scope, coordinate-frame declaration, georeferencing state, and geometry status;
- exact landscape metric registry and source-catalog digests;
- candidate sources with catalog state, role and known limitations;
- Symtropy scenario manifest and optional result digests, model/version, purpose, evidence class and status;
- at least two alternatives, each tied to an exact snapshot digest.

The source catalog is metadata-only until individual datasets have been lawfully acquired, ingested, and validated. Catalog presence is not evidence that the source is current, licensed for reuse, CRS-compatible, accurate for the intended scale or suitable for the proposed metric.

## Review outputs

Every metric assessment has a status:
- **computed** — a value is permitted only with unit, evidence class, method reference, exact input digests and an uncertainty declaration;
- **blocked** — no value; a reason and list of unprovided inputs are required;
- **conflicted** — no value; at least two conflicting evidence references and a resolution/review path are required;
- **not-applicable** — no value; the reason must be explicit.

The comparison preserves trade-offs by dimension (heat, water, ecology, access/equity, material/lifecycle cost, stewardship, economics/community). It does not compute a universal sustainability score or silently impose weights. User/community priorities and acceptable trade-offs must be explicit project decisions.

The recommendation may be no-ranking, candidate-next-action, or prefer-alternative. The schema only allows prefer-alternative when the comparison says qualified-comparison; synthetic benchmark designs cannot qualify environmental/social metric outcomes.

## Safety and authority boundaries

- Review artifacts are read-only: no mutation of the authoritative project and no actuator authority.
- In the current Sol-Atlas v0 frame (LOCAL_CARTESIAN_DRAWING_UNITS, georeferenced=false), the review does not authorize analytical or scientific simulation claims or any computed spatial/physical outcome metric. A real coordinate transform and typed, validated geometry are prerequisites.
- Symtropy currently allows schema qualification and synthetic benchmarking for v0 geometry only. It blocks analytical/scientific claims with schematic anchors.
- A Mycelix decision/consent record may attest that an authorized decision was made; it does not prove physical outcomes, title, or ecological causality.
- Never represent synthetic/modelled evidence as observed evidence; preserve the evidence class in every display/export.
- No source ingestion or publication is assumed. Respect each source's license, consent and location sensitivity.

## Benchmark methodology

### Structural/adversarial conformance

Fixtures must cover:
1. missing or stale source digest;
2. unknown metric/source IDs;
3. candidate-source license review not complete;
4. a result whose design digest differs from the compared design;
5. a synthetic scenario submitted as observed;
6. contradictory scenario results;
7. one alternative missing required metrics;
8. a request to rank alternatives despite all consequence-bearing metrics being blocked;
9. a request to publish sensitive locations;
10. mutation or actuator authority requested by the review.

Acceptance: invalid/stale/mismatched inputs fail closed, preserve exact digest references in error reports and never yield a selected winner.

### Replay and decision-quality evaluation

- Pin analysis method/version, ruleset digest, normalized input digest and environment notes.
- Replay identical normalized inputs; require deterministic structured output.
- Measure source/claim traceability, calibration on blinded expert-labelled fixtures, rate of unsupported claims, quality of uncertainty/falsification questions, and time for a human reviewer to identify what would change the decision.
- Compare against a blinded no-assistant/standard checklist baseline for matched landscape-planning tasks. Preregister the protocol and outcome rubric.
- Report disagreement, false confidence and “no-ranking” appropriateness; do not select only successful demonstrations.
- Do not claim superiority to landscape architects without independent, preregistered, appropriately powered evaluation.

## Acceptance criteria

- The schema validates structurally and the canonical fixture round-trips under a strict validator.
- At least one alternative must reference the exact baseline-design SHA-256, and each genuinely different alternative must have its own distinct snapshot digest, scope and declared coordinate frame. Every scenario run must bind to one named alternative and its exact snapshot digest. Paired runs may be compared only when model ID/version, boundary-condition digest, evaluation-plan digest, metric-registry digest, seed and scope agree; the scenario-manifest digests and design-snapshot digests must remain distinct.
- Every computed metric names unit, method, evidence class, input digests and uncertainty; blocked/conflicted metrics cannot contain a value.
- Prefer-alternative cannot coexist with a non-qualified comparison; all-blocked outcomes produce no-defensible-ranking.
- Scenario-review records use only pinned Symtropy inputs/results and preserve status/evidence class.
- Read-only and no-actuation fields are fixed true/false invariants.
- Negative fixtures cover the cases above and are tested in Rust, not just manually reviewed.
- Structural qualification is not described as scientific validation, causal proof or evidence of landscape benefit.

## Current limitation

This branch defines the output contract and one “no ranking” fixture. It does not yet contain a live model, run source ingestion, independently compute landscape metrics, validate the JSON Schema at runtime, or establish browser/reviewer benchmark results.

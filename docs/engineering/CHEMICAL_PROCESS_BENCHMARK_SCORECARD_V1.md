# Chemical Process Benchmark Scorecard and Verification Ladder

**Status:** Research protocol and proposed targets; target values below are not measured results.  
**Date:** 2026-10-18  
**Parent:** [CHEM-BENCH-002 #7869](https://github.com/Luminous-Dynamics/symthaea/issues/7869) and [Tier-B registry PR #7872](https://github.com/Luminous-Dynamics/symthaea/pull/7872).  
**Scope:** Public Symthaea benchmark design. No plant operation, safety, product-quality, or physical qualification claim.

## Paradigm shift: benchmark claims, not just solver outputs

The primary unit of evaluation should be a **bounded engineering proposition**:

> Under this exact context of use, input/data lineage, model implementation, numerical method, and stated uncertainty, does the evidence support this particular claim—and what must remain unresolved?

A solver exit code, close numerical agreement, deterministic replay, and physical validity are different evidence. The benchmark must report them as separate dimensions; no weighted average may hide a failed hard invariant or an unearned claim.

This direction aligns with the scope of ASME V&V 20: quantify model accuracy for specified variables at specified validation points while accounting for uncertainty in both solution and data. That scope does not automatically justify extrapolation to unvalidated points. The October 2026 NIST report on computational modeling and simulation uncertainty likewise emphasizes risk-based credibility assessment scaled to model influence and decision consequence. This protocol adopts those principles as design inspiration; it does **not** claim formal compliance with either standard.

References:
- [ASME V&V 20 — Verification and Validation in CFD and Heat Transfer](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer)
- [ASME VVUQ Standards Hub, including VVUQ 20.1 (2024) and VVUQ 50.1 (2025)](https://www.asme.org/codes-standards/vvuq-standards)
- [NIST, *On Methods of Uncertainty Quantification in Computational Modeling and Simulation* (published 2026-10-05)](https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing)
- [W3C PROV Primer](https://www.w3.org/TR/prov-primer/), [SLSA build provenance](https://github.com/slsa-framework/slsa/blob/main/spec/build-provenance.md), and [in-toto Attestation Framework](https://github.com/in-toto/attestation/blob/main/spec/README.md) for provenance concepts. These references are not a claim that Tier-B manifests are authenticated or formally conformant.

## Five evidence levels—never infer a higher level from a lower one

| Level | Evidence demonstrated | Permitted conclusion | Not established |
|---|---|---|---|
| E0 — Contract integrity | Registry parses; IDs, states, blockers and field contracts are consistent | The benchmark record is structurally admissible | Simulator execution or model correctness |
| E1 — Analytical verification | Independent equations and hard invariants pass | Those invariants hold for the declared inputs and assumptions | Correct property data or physical fidelity |
| E2 — Reproducible execution | Exact inputs, versions, transitive dependencies and raw outputs are pinned; replay is compared | The named run can be reproduced within a preregistered criterion | Independence or physical validity |
| E3 — Independent numerical comparison | A comparator is genuinely separate, dependency overlap is audited, and discrepancies are reported | Agreement/disagreement within the matched context | Validation against reality |
| E4 — Empirical validation and UQ | Suitable experimental observations, measurement uncertainty and numerical/model discrepancy are accounted for | Estimated accuracy for specified quantities and validated points | Arbitrary extrapolation or broader qualification |
| E5 — Context-of-use qualification | Evidence is sufficient for a defined decision, domain and consequence; limitations are reviewed | Only the named decision within its declared domain | General safety, plant or product qualification |

Each case must record the level actually evidenced. E0–E3 cannot be advertised as E4/E5.

## Required receipt boundary

For any proposed frozen simulator output, the receipt must explicitly state:

- **Context of use:** what question the model is being used to answer.
- **Validity domain:** components, phases, conditions, units, model/property package, input range, and any other case-specific bounds needed to interpret the result.
- **Excluded contexts:** conditions or decisions that the receipt does not support.
- **Validation basis:** analytic invariant, numerical comparator, empirical observations, or a clearly described combination.
- **Claim ceiling:** the strongest conclusion supported by that evidence—not a marketing summary.

A nonempty prose ceiling alone is too easy to overread. Registry v1.1.0 makes the structured context/domain/exclusions/basis mandatory in the frozen receipt. Registry v1.2.0 binds the parsed receipt value to a domain-separated SHA-256 content digest using a repository-local canonical JSON form. Registry v1.3.0 adds a second envelope digest over registry schema identity and the full candidate record (case identity, source, configuration, invariants/hints, run status/version/output hashes, claim ceiling, receipt and its digest), excluding only the envelope digest field itself. This catches receipt mutation as well as partial edits or transplanting evidence to another case. This canonical form is not claimed to be RFC 8785/JCS compliant. The digest is unauthenticated: an editor who can rewrite the receipt and digest can recompute both. A future attestation must anchor the digest to a trusted signer identity; signatures establish origin/integrity under a trust policy, not scientific truth.

## Benchmark targets

All targets below are goals. Only the structural controls already encoded in the registry validator and its authored tests are implemented in this tranche; Rust tests have not yet been run in the current environment.

| Metric | Initial target | Gate |
|---|---:|---|
| Negative-control false accepts | **0** | Hard |
| NOT_RUN / SKIPPED / REFERENCE_UNAVAILABLE / OUT_OF_DOMAIN promoted to PASS | **0** | Hard |
| Frozen receipts with all required provenance/context fields | **100%** | Hard |
| Seeded changes to a frozen receipt without digest update are detected | **100%** | Hard |
| Digest policy drift from the compiled implementation is rejected | **100%** | Hard |
| Mutations to candidate identity, configuration, run metadata or output digest invalidate the frozen envelope | **100%** | Hard |
| Planted transitive shared-dependency challenge detected | **100%** of seeded cases | Hard |
| Malformed lineage graph challenge detected | **100%** of seeded cases | Hard |
| Analytic material/element/energy invariant escape | **0** for registered negative controls | Hard |
| Preregistered tolerance coverage | **100%** of frozen cases | Hard |
| Canonical replay agreement | **100%** for exact-environment repeat runs, or a stated deterministic numerical criterion | Hard where the solver supports it; otherwise reported as a limitation |
| Metamorphic hard-invariant escape | **0** | Hard |
| Experimental validation reporting | **100%** of validation points include context, reference uncertainty and model/solution discrepancy | Hard before an E4 claim |

These are not all aggregateable into a single leaderboard score. Report denominators, the exact corpus revision, excluded cases, uncertainty, and hard-gate failures alongside every number. A metric whose test has not been implemented or executed is **NOT_MEASURED**, never an implicit zero.

## Challenge families that can expose shortcut learning

1. **Provenance mutation:** change one input, property-data, mechanism or dependency digest while retaining a prior frozen receipt. Expected result: receipt invalidated or rerun required; never silently reused.
2. **Transitive common-mode dependency:** give two differently named solver outputs a shared upstream correlation/data file. Expected result: the common ancestor is reported and independent-corroboration claims are rejected.
3. **Malformed/incomplete lineage:** inject missing parents, a cycle, duplicate nodes or an incomplete manifest. Expected result: **UNRESOLVED**, not presumed independent.
4. **Metamorphic conservation:** scale every inlet mass flow by the same positive factor; permute stream declaration order where the physics is order-invariant; convert values between equivalent unit representations. Expected outcome: transformed outputs preserve the corresponding conservation relation within preregistered tolerance.
5. **Semantic perturbation:** change phase assumptions, composition basis, reference state, property-package revision or solver tolerance. Expected outcome: the context identity changes and a prior golden receipt cannot be reused unchanged.
6. **Domain-boundary challenge:** evaluate outside the declared validity domain or remove a required input. Expected outcome: **OUT_OF_DOMAIN**, **UNRESOLVED** or another specified non-pass state; never force a numeric answer to look valid.
7. **Hard-constraint versus objective trade-off:** provide a candidate with lower cost but a failed mass/energy/material constraint. Expected outcome: hard constraint wins; aggregate score cannot rescue it.
8. **Common-source “agreement” control:** have two implementations share property data or reference outputs. Expected outcome: similarity is reported as agreement, not independent corroboration.

The checker/oracle must be designed independently from the candidate model where practical. Negative controls should be seeded and scored before new candidate results are inspected. Preserve all failed and unresolved rows; do not drop them from the denominator. The current Rust integration suite now adds three **analytic-only** metamorphic controls for the mixer fixture: stream-order permutation, positive flow scaling, and equivalent kg/s-to-g/s conversion. These test the frozen arithmetic invariant only; they do not exercise a simulator or validate transformed simulator outputs.

## Physical validation and uncertainty

Numerical tolerances are case-specific—not a universal percentage. Freeze them before inspecting candidate results, with rationale tied to units, reference precision, solver error, measurement uncertainty and the intended context of use. Where physical data exists, report the validation quantity, data/measurement uncertainty, numerical uncertainty if estimated, observed discrepancy and how they combine. Do not collapse model discrepancy into measurement uncertainty or declare a model valid solely because a point lies inside an arbitrary band.

For later model families, add held-out validation observations, explicit train/calibration/test separation where data-driven components exist, coverage and sharpness of predictive intervals, performance near domain boundaries, and confidence intervals on benchmark metrics. A nominal interval coverage number is not meaningful without sample count and dependence assumptions. Riskier decisions require stronger and more directly relevant evidence.

## Execution roadmap

- **P0 — Contract:** complete registry, fail-closed statuses, structured context-of-use validity domain, dependency graph checks and negative controls. Current PR is at this level; structural code has not been tested with Rust in this environment.
- **P1 — Deterministic adversarial harness:** implement unit and metamorphic tests for lineage mutation, invalid graphs, unit transformations, stream permutations, scale relations and receipt invalidation; emit per-case machine-readable receipts.
- **P2 — Matched-condition external cases:** freeze at least three benign, matched-condition cases with pinned inputs/builds and independent analytic checks before optimization or leaderboard claims. The existing three DWSIM tutorial candidates are only candidates; they have not been executed here.
- **P3 — Independent numerical comparisons:** compare implementations only where equations, inputs, property data, reference states and quantities of interest actually match; show shared lineage and numerical discrepancy.
- **P4 — Empirical validation:** add inspectable measurements with uncertainty, blinded/held-out cases and explicit validity boundaries.
- **P5 — Decision-specific credibility:** tie evidence strength to the consequence and influence of the intended decision. Keep operational, safety, product-grade and economic claims outside the benchmark until their separate evidence requirements are met.

## Stop conditions

Never freeze a simulator golden output when the input lineage is incomplete, tolerance was chosen after seeing the result, the independent checker shares an undisclosed reference source, the solver did not converge, the case is out of domain, or the claimed conclusion exceeds the receipt's declared context. **SKIPPED**, **NOT_RUN**, queued CI, missing reference data, and an untested code change are not passes.

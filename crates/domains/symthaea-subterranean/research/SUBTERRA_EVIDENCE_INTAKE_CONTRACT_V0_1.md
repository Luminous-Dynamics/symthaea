# Subterra evidence intake and typed claim contract v0.1

**Date:** 2026-11-16  
**Status:** Research adapter and contract authored; Rust tests not executed  
**Consumer:** Symthaea Subterra scientific-belief and planning layer  
**Producer:** Symtropy GeoBio + Subterra research schemas in PR #5486

## 1. Purpose

The three Subterra repositories now need a stable boundary between research data and scientific reasoning. The first adapter increment turns Symtropy's site, non-uniform profile and cave-network JSON records into explicit typed Rust values inside Symthaea.

This boundary must preserve what the source says without inventing what it does not say. It is not a geology classifier, sensor-authentication system, posterior estimator, map merger, coordinate-transform engine, safety qualification or mission authorization service.

Upstream record/schema provenance:
- Site-evidence schema v0.1 blob: 970b48703a8147cd3047cce05c5fc0631d1f03d9.
- Cave-network schema v0.1 blob: bfe797f741d930ffcde2ce9d33a21c75396f46c0.
- Non-uniform profile schema v0.1 blob: 87d77b208f20227797ff59fd1f6d4c01bbb1c72c.
- Synthetic site fixture blob: 31813b808d596f9ddb78c4de49a15f1773026fc9.
- Synthetic cave-network fixture blob: 95fa93de34773569aa3abcb9c3f808f2f9986e97.
- Synthetic non-uniform profile fixture blob: a8305c9881f3483a762e3bc6717b86009d583039.
- Pinned schema/fixture manifest blob: c6c5773546335913175d74e2380ff6a9e6da01e1.
- Upstream research branch / draft: https://github.com/Luminous-Dynamics/symtropy/pull/5486.

The Symtropy typed semantic validator was also corrected in commit 82c2eccb49bcbd7567143b7b2e5ae98f7e61d916: directly observed connectivity no longer requires demonstrated access or a completed interior survey. This restores the original independent-claim model and has two new regression tests authored; it remains uncompiled/unexecuted.

## 2. The independent claim model

The site record carries separate assertions:

1. **Morphology:** what form is supported by surface or interior observations?
2. **Geological origin:** which mechanisms remain candidate, supported, contradicted or unknown?
3. **Connectivity:** what physical connection is candidate, geophysically supported, directly observed or contradicted?
4. **Accessibility:** what robot/human access and egress has separately been assessed or demonstrated?
5. **Interior survey:** what portion of the interior was actually surveyed?
6. **Engineering suitability:** for which declared use and envelope has evidence been assessed?
7. **Preservation:** what planetary-protection and activity permissions have been declared, and what authority review remains external?

No single scalar confidence is allowed to collapse these claims. No claim is inferred solely from another claim's state.

Examples:
- Radar-supported continuation does not imply directly observed connectivity.
- Directly observed connectivity does not imply a robot traversed the connection.
- A partially surveyed chamber does not imply all outgoing edges are traversable.
- Robot access does not imply human access.
- A qualified storage envelope does not imply qualified habitation.
- An activity permission string is not authenticated authorization.

The updated Symtropy semantic validator reflects this distinction. It still requires evidence references for a directly observed connectivity claim and retains use-specific prerequisites for pressurized/habitation engineering qualification.

## 3. Typed adapter behaviour

### Site evidence

The 'subterra-evidence-adapter/src/lib.rs' file:
- deserializes the v0.1 body, morphology, origin, connectivity, access, survey, engineering, preservation and observation values;
- rejects unsupported schema version, unknown fields, unknown enum values, malformed numeric ranges, duplicate observation IDs and missing required evidence references;
- converts to 'SiteBeliefInput' without calculating priors or posterior probabilities;
- retains source evidence references, native coordinate-frame strings, observation timestamps, provenance references and explicit unknowns;
- sets a separate 'planetary_protection_review_required' flag for non-Earth bodies;
- retains a declared activity-permission state but does not transform it into action authority.

The v0.1 preservation schema does not include a named authorization-case reference. A future schema version should add a stable authorization record identifier and issuer/scope metadata. Until then the external planetary-protection review boundary remains mandatory for all non-Earth activity proposals.

### Cave network

The typed graph has two different views:

- **Observed physical topology:** connected components use only directly observed connection edges, regardless of whether a robot has demonstrated traversal.
- **Demonstrated route topology:** route queries use only directly observed edges whose traversability is also demonstrated; edge direction is respected.

This avoids two symmetrical errors:
- promoting a GPR/candidate edge into confirmed topology;
- treating a connected physical cave as a robot-demonstrated route.

'NoDemonstratedRoute' is an evidence statement about the current record. It is not proof of physical disconnection.

### Non-uniform profile

The profile adapter preserves ordered stations, explicit 'cross_section: null' intervals, measured width/height, optional roof thickness, standard uncertainty and correlation-group metadata.

The initial summary reports:
- total and observed lengths;
- coverage fraction and status;
- observed-only length-weighted mean width/height;
- observed minima/maxima;
- roof-thickness measurement coverage.

Unknown intervals are excluded from dimension averages, not treated as zero and not interpolated. The source uncertainty/correlation metadata is retained, but this lightweight adapter summary does not yet propagate uncertainty through its mean statistics. Symtropy's separate morphometry kernel remains the reference implementation for uncertainty-aware summary calculations.

## 4. Planner interface

The adapter output is intentionally not yet a 'PlanningProblem'.

The next conversion must be a separately versioned, calibrated model layer:
- a geological hypothesis ontology;
- priors with explicit non-zero unknown mass;
- modality- and site-specific likelihood models;
- source-lineage/common-cause fault hypotheses;
- mission decision/loss matrices under declared units;
- model calibration / OOD evidence;
- hard external admissibility results and resource budgets.

Qualitative states such as 'supported' and 'candidate' cannot be mapped to arbitrary numbers like 0.7 and 0.4. Those values must come from reviewed calibration on suitable held-out observations. The adapter makes this separation visible by emitting qualitative belief inputs only.

## 5. Coordinate and physics boundaries

The site adapter preserves 'coordinate_frame', latitude/longitude/elevation and observation frame strings as provided. It does not transform them. Sol Atlas owns body/epoch/frame semantics and validated IAU/PDS/NAIF-SPICE conversions.

The profile adapter is a 1D along-profile record. It is not a 3D mesh, cave connectivity graph, structural model, roof stability calculation, pressurization model or access envelope.

### Science return

The additional science-return record enforces the lifecycle:

'Proposed → Executed → Acquired → IntegrityVerified → Persisted → Delivered → Admitted'.

Failure/cancellation are explicit terminal dispositions. 'StoreAndForwarded' is a distinct branch from 'Persisted' and is not counted as delivered. Integrity verification requires a calibration reference; persistence requires a custody reference; delivery requires both custody and a delivery receipt; scientific admission requires a separate admission reference. These references are still opaque, caller-supplied identifiers: this state machine does not authenticate signatures, prove durable storage, verify transport or establish custody authority.

The network adapter is a graph of evidence claims. It is not a SLAM map, path planner, geomechanics solver or robotic traversability certificate.

## 6. Source and evidence trust

A non-empty evidence reference is an opaque identifier. It does not prove that the object exists, is immutable, is correctly calibrated, was captured by the declared sensor, is independent of another reference or is authorized for use.

Science-return events also carry `clock_id` and `clock_uncertainty_ms`; the v0.1 event-log check accepts canonical UTC `YYYY-MM-DDTHH:MM:SSZ` only and flags same-clock regressions only when uncertainty intervals prove them. Event sequence remains authoritative, and distinct clocks are not directly compared.

An acquired science-return record now includes a canonical SHA-256 digest and exact byte length. `validate()` checks lifecycle structure and digest metadata; callers must separately invoke `verify_payload_bytes()` on the actual returned bytes to verify the digest and length before treating byte integrity as established. A pre-acquisition record cannot pass the byte-verification method without a declared payload. This establishes byte equality relative to the declared digest, not source authenticity; a trusted signature/key binding, calibration chain, clock attestation and custody verifier are still required.

A later source-trust bridge should bind:
- physical sensor and acquisition identity;
- observation ID and immutable content digest;
- calibration chain and version;
- capture/receipt clocks and uncertainty, bound to trusted clock sources;
- derived-product ancestry;
- integrity/signature status;
- independent source/modality lineage;
- freshness and replay sequence;
- verifier receipt.

Only then can the scientific hypothesis and measurement planner evaluate the observation with source-aware dependence assumptions.

## 7. Current implementation and tests

Standalone package:
'crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/'

Contents:
- site-evidence JSON-to-typed adapter;
- cave-network JSON-to-typed adapter and topology/route queries;
- non-uniform profile adapter and observed-only summary;
- science-return lifecycle validator and site/observation linkage;
- a separate site-disjoint likelihood-calibration reference crate for Training-only empirical likelihood estimates, Calibration/Evaluation proper scoring and a guarded conversion into EVSI inputs;
- copied synthetic site, network and profile fixtures;
- 21 site/network tests, 6 profile tests and 26 science-return tests (53 adapter cases total); 19 site-disjoint likelihood-calibration tests and 47 dependence-auditor tests in their separate packages.
- Pinned upstream schema/fixture manifest and explicit local verification script for the six nested Cargo packages.

Across the complete current Symthaea Subterra PR, the six standalone packages have 150 authored Rust unit tests (10 EVSI + 11 survival + 10 robust EVSI + 21 site/network + 6 profile + 26 science-return + 19 likelihood calibration + 47 dependence audit). This is a source count only; none has been compiled or executed in the current environment.

### Verification status

- Fixture and manifest JSON syntax: checked.
- Source/schema blob provenance: recorded in the pinned manifest.
- Local verifier script: AUTHORED; not executed because cargo, rustc and rustfmt are unavailable.
- Rust formatting: not executed.
- Rust compilation: not executed.
- Rust tests: not executed.
- Clippy: not executed.
- CI: skipped / not qualified.
- Real sensor evidence, robot-in-the-loop, cave analogue, flight and planetary validation: not performed.

## 8. Next increments

1. Add runtime/source-level enforcement of the pinned schema manifest and negative fixtures; reject unsupported version/blob drift.
2. Run the authored four-package verification script on an environment with the declared Rust toolchain and retain exact-head receipts.
2. Connect verified observation lineage and source-trust receipts; retain common-cause correlations.
3. Ingest independently adjudicated, site-disjoint observation cases into the calibration reference; extend beyond Dirichlet categorical counts to hierarchical/correlated sensor models only after held-out evidence supports them.
4. Add labelled observation-dataset adapters and source-lineage receipts; the current calibration-to-EVSI converter still requires caller-supplied priors, decision losses, action costs and external gate dispositions.
5. Extend science-return records with verifier-bound signatures, immutable custody receipts, transport-delivery reconciliation and cross-agent identity evidence; the current state machine only validates local event ordering and required reference fields.
6. Compare planner outputs against one-step EVSI, entropy, frontier, FEP and survival-conditioned baselines under identical evidence and resource budgets.
7. Only after exact-head source/test qualification should production runtime interfaces be proposed.

## 9. Non-claims

This adapter does not authenticate observations, establish geological truth, infer connectivity from surface morphology, demonstrate traversal, validate a planetary coordinate transform, certify structural safety or habitation, grant planetary-protection permission, issue robot commands, or prove SOTA superiority. It is a typed data boundary designed to prevent those claims from being silently conflated.


### Clock and chronology integrity

Event sequence numbers establish lifecycle order. Timestamps add an independent consistency check but do not replace sequence order. Every event declares a clock ID and an absolute uncertainty bound in milliseconds. Because v0.1 timestamps are whole-second values, uncertainty must be at least 1,000 ms to include timestamp quantization; smaller claims are rejected. Events from different clock IDs are never directly compared. For events from the same clock, a regression is rejected only when the current timestamp's upper uncertainty bound is earlier than the previous timestamp's lower bound. Overlapping uncertainty intervals are inconclusive, not proof of bad order.

The reference accepts only fixed-width UTC timestamps in `YYYY-MM-DDTHH:MM:SSZ` form. Timezone offsets, fractional seconds and leap-second notation are outside v0.1. This deliberately narrow representation avoids silently comparing local-clock strings or ambiguous formats.

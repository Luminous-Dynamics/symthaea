# symthaea-alternatives

Evidence-first industrial alternatives assessment.

This crate is deliberately read-only and decision-support oriented. It does not declare an alternative globally "better", mutate manufacturing records, or execute physical actions.

The model is:

1. Define the function that must be satisfied.
2. Bind the requirement to the exact product/component/design context being assessed.
3. Represent candidate material/process/product pathways.
4. Declare an explicit comparison scale (unit + functional/lifecycle scope + exact methodology/comparability basis identity) for every burden dimension; candidates cannot define their own cohort.
5. Represent functional-performance values as evidence-linked, scoped measurements tied to the exact comparison/test basis rather than bare claims.
6. Declare the physical operating envelope explicitly and bind its evidence to the exact engineering/test basis; a candidate must demonstrate coverage of the required ranges.
7. Keep multi-axis burdens separate.
8. Evaluate hard constraints fail-closed.
9. Compute a conservative Pareto frontier without an aggregate sustainability score.
10. Preserve evidence kind, confidence, canonical authority identity, artifact identity/digest, optional source-admission reference, contradictions, validity windows, observation time, and derivation lineage.
11. Derive an explicit qualification ceiling from evidence actually linked to every burden dimension and every required functional metric; higher global states additionally require tier-consistent functional and operating evidence.
12. Emit a deterministic assessment receipt, including the assessment timestamp and exact freshness policy when supplied.
13. Require reproducible derivation metadata for simulated/derived evidence.
14. Intersect eligibility across every required function while keeping each requirement assessment separate and auditable.
15. Carry externally qualified source-admission references without verifying or upgrading them inside Symthaea.
16. Identify a conservative next-measurement target from unresolved uncertainty.
17. Carry explicit experimental-design provenance when a proposed measurement is actually specified: hypothesis, targeted uncertainty identities, candidate discrimination set, exact protocol identity/basis, and stopping criteria.

Authority diversity is authority-scoped: multiple artifacts or rotated issuer keys under one authority do not become distinct authority groups. Only externally admitted authorities count toward higher qualification tiers. This is still not proof of epistemic or organizational independence, and the identity/admission contract is not an authenticity proof; Mycelix admission/attestation remains the authority boundary for source control. The requirement is bound to an exact subject/profile/revision/digest so the same generic function cannot be silently reused for a different BOM, product, or design revision. The design is intended to compose later with Mycelix manufacturing/BOM/routing records and a federated evidence graph, while remaining independent of Holochain versioning. The Pareto frontier is a candidate comparison set, not a recommendation or deployment authorization. Functional performance and operating capabilities use conservative intervals rather than midpoint-based pass/fail. A required operating envelope must be fully covered; partial temperature/pressure/load coverage is unresolved or failed rather than extrapolated. Time-bounded evidence is ignored by timeless assessments and evaluated only when the caller supplies an explicit assessment timestamp.

## Safety boundary

Model output is a hypothesis or assessment artifact, not a manufacturing authorization.

A simulation cannot promote a candidate to field-qualified status. Missing or conflicting evidence remains visible and lowers the qualification ceiling.


Freshness is deliberately separate from validity. An evidence record may remain valid for a declared interval while still being too old for a particular decision. A caller can therefore supply an explicit `EvidenceFreshnessPolicy` with a stable policy identity/digest and maximum age per evidence kind. A freshness-bounded assessment requires an explicit assessment timestamp and machine-readable `observed_at_epoch_seconds`; missing observation time, future observations, or observations older than the configured limit are conservatively unusable. There is no universal freshness default because evidence decay is decision-profile dependent. The freshness policy is carried into the assessment result and receipt, but its identity is not itself an authority or authenticity proof.

This temporal model follows a provenance-friendly boundary: W3C PROV treats entity lifetimes, generation, use, and invalidation as time-aware provenance events, so validity and recency should not be collapsed into an opaque source label. The JRC Safe and Sustainable by Design guidance likewise describes alternatives assessment as iterative and tiered as data availability and research knowledge increase.


Global qualification states are tier-consistent. Field qualification requires field-observed support across all burden dimensions and across every required functional/operating evidence surface; continuous monitoring applies the corresponding monitoring requirement. Manufacturing qualification likewise requires manufacturing-scale observations across every burden dimension. This prevents a candidate from being promoted by a strong burden dataset while its actual functional qualification remains at a lower evidence tier.


## Comparability boundary

Unit and scope equality are necessary but not sufficient for industrial comparison. Every burden, performance estimate, operating requirement, and candidate capability therefore carries a `ComparisonBasisRef` consisting of a stable basis ID, revision, and digest. The basis is intentionally opaque to this crate: it can identify the exact functional-unit definition, system boundary, allocation/normalisation rules, lifecycle accounting profile, or engineering test protocol selected by the authoritative assessment context. Cross-basis Pareto dominance, burden-transfer claims, functional constraint passes, and qualification are fail-closed.

This mirrors an important limitation in existing product accounting practice: the GHG Protocol Product Standard explains that additional accounting specifications are needed for defensible product comparisons, and comparative GHG work emphasizes using the same functional unit for like-for-like comparison.


Contradiction handling is also evidence-surface complete. Conflicting supported/contradicting records attached to required functional or operating evidence make the corresponding gate unresolved and cap qualification; stale/future records are excluded before contradiction/support evaluation. Burden contradictions remain visible and continue to cap qualification. This keeps “there is evidence” distinct from “the evidence agrees.”

The current GHG Protocol Product Standard explicitly notes that additional specifications such as accounting choices and data sources are needed for product comparison, rather than treating a generic product inventory as sufficient for comparative assertions. The 2026 ISO/GHG Protocol joint working group is updating and harmonising product-level GHG accounting, reinforcing the value of treating the exact comparison basis as versioned, digest-bound provenance rather than a free-text label.


Admission is also temporally scoped. When a source carries an admission validity window, the assessment timestamp must fall inside that window; timeless assessments conservatively reject time-bounded admissions. This prevents a once-admitted source authority from remaining silently qualified after its external admission has expired or been superseded.

## Candidate-generation provenance

Candidate generation is a separate provenance surface from evidence provenance. A generated pathway can carry optional reproducible derivation lineage describing the generating activity/method, exact input identities, and configuration hash. The lineage is validated structurally, copied into the candidate assessment, and included in the deterministic assessment receipt. This makes “how was this candidate proposed?” auditable without treating generation provenance as evidence quality, source authority, or physical qualification.

### Evidence-basis binding

Evidence records also carry their own exact `ComparisonBasisRef`. Every evidence record linked to a burden, functional-performance estimate, or operating capability must match the linked estimate's basis exactly. Unit and scope agreement alone is insufficient. A methodology/test-protocol mismatch fails closed during candidate validation, and the evidence basis is included in the canonical evidence digest and assessment receipt.

### Observation provenance

Physical and operational observation evidence (Observed, ManufacturingObserved, FieldObserved, ContinuouslyMonitored) must also carry an ObservationProvenanceRef identifying the exact observed specimen/lot/batch/product/process-run subject, the measurement/test activity, the underlying observation-record digest, the measurement system when applicable, and the declared calibration/traceability chain references. The observation provenance contract now also requires explicit measurand and documented measurement-procedure identities. This keeps the quantity actually measured and the procedure used to measure it from becoming implicit metadata. This is a structural traceability reference, not a claim that Symthaea has verified the chain. Missing observation provenance fails closed. This keeps the exact thing measured and the exact measurement activity separate from the source authority and from the comparison methodology.

Observation evidence also carries a quantitative uncertainty statement: standard or expanded uncertainty with an explicit unit, plus a coverage factor for expanded statements, and references to the uncertainty components/method/record. The evidence unit must match the stated uncertainty unit, and the uncertainty analysis must bind to the observation's exact measurand and measurement-procedure identity/digest. The uncertainty reference is also bound to the exact observation identity and observation-record digest, so it cannot be silently transplanted to a different result. Each uncertainty component now carries its own component-record digest plus the exact uncertainty-budget identity/digest and measurement-model identity/digest; component identities are canonicalized and committed as a set. The parent uncertainty additionally carries the component count and component-set digest attested by the exact uncertainty-budget record; Symthaea requires both to match the submitted canonical component frontier, closing the partial- or extra-component substitution seam without claiming that it independently knows the scientific budget is complete. The uncertainty also carries a separate canonical BLAKE3 binding digest over the reported combined uncertainty, observation lineage, budget record, measurement model, procedure, source-supplied uncertainty record digest, and canonical component set. This establishes an integrity binding between the declared combined result and the declared component lineage without pretending Symthaea recomputes or certifies the scientific uncertainty calculation. A same-unit uncertainty statement for a different measurand, procedure, budget, model, component record, or observation record fails closed. The engine remains a structural verifier; the authoritative uncertainty budget/model and metrological/scientific evaluation remain external.

The uncertainty evaluation lineage is explicit rather than a bare method label. Each measurement uncertainty identifies the exact evaluation record, the versioned method used to evaluate it, the versioned combination rule, the versioned covariance/dependence model, the exact probability-distribution treatment, the exact degrees-of-freedom record, and—when expanded uncertainty is reported—the exact coverage-factor method. These are structural external references whose identities and digests are committed into the uncertainty binding. This aligns with metrology guidance that an uncertainty budget should identify its measurement model, uncertainty components, covariances, probability-distribution treatment, degrees of freedom, evaluation type, and coverage factor, while keeping the scientific calculation and its authoritative records outside this crate.

The assessment receipt also binds a canonical BLAKE3 digest of the complete evidence bundle for each candidate. That prevents a provenance-only mutation—such as an artifact digest, admission reference, observation timestamp, validity window, or evidence derivation change—from becoming invisible merely because the downstream qualification result happens to remain unchanged. Evidence is therefore both evaluated and integrity-bound.

Source admission is authority-bound: an EvidenceSourceIdentity whose admission reference names a different authority is rejected. This keeps authority admission from becoming a detached capability token that could be attached to an unrelated source identity.


## Experimental-design provenance

The heuristic next-measurement target is intentionally weaker than an experiment specification. It identifies an unresolved dimension, candidate set, uncertainty references, and interval-overlap discrimination targets without inventing a laboratory protocol.

When an actual measurement campaign is proposed, ExperimentalDesignProvenance can be attached to the assessment. It requires an exact hypothesis, exact requirement identity, targeted uncertainty identities, candidate identities, explicit discrimination targets, an exact documented protocol identity/digest and comparison basis, and explicit stopping criteria. The assessment receipt commits to that design.

An observation may then carry the resulting experimental_design_id, closing the lineage from proposal to observed result. This is provenance, not scientific authorization: Symthaea does not certify protocol adequacy, sample size, statistical power, causal validity, metrological traceability, or experimental safety.


### Typed experimental discrimination

ExperimentalDesignProvenance uses typed discrimination targets rather than free-text pair descriptions. Each target identifies both candidate endpoints, the exact surface being discriminated (burden, performance metric, or operating condition), and a versioned decision-rule identity/digest. The target's surface must resolve to a declared requirement context and its comparison basis must exactly match the design and protocol basis.

Observations linked to an experimental design must also carry that exact design identity. Observations carrying a design identity without a supplied design are rejected, as are observations carrying a different design identity. This makes proposal-to-observation lineage bidirectional at the Symthaea boundary rather than merely advisory.


Experimental observations are now bound to the exact documented procedure prescribed by the experimental protocol. A design-bound observation must match both the design's protocol procedure identity and its exact procedure digest; procedure identity or semantic procedure drift fails closed. This keeps the planned protocol distinct from merely having an observation produced by some procedure.


### Requirement-semantic binding

ExperimentalDesignProvenance carries both the stable requirement ID and a canonical BLAKE3 digest of the complete FunctionalRequirement payload. The assessment recomputes that digest before accepting the design. This prevents a design from being replayed under the same requirement identifier after the underlying requirement semantics have changed.


Design-bound observations also have to match the target surface's declared unit, scope, and comparison basis. A correct design/target identity is therefore insufficient on its own: the observed evidence must be comparable to the exact requirement surface it purports to resolve. The typed target resolver returns the complete requirement scale (unit, scope, and exact basis) for burden, constrained-performance, and operating-condition surfaces; an undeclared or unconstrained performance surface remains unresolved and fails closed.


The heuristic next-measurement result also uses typed discrimination targets. Each target explicitly names the two frontier candidates and the burden dimension whose intervals overlap; the heuristic does not invent an experimental decision rule or protocol.


Uncertainty-based stopping criteria are target-specific rather than bare numeric widths. Each stopping bound names the exact discrimination target and the unit in which its maximum interval width is expressed, preventing ambiguity when a campaign covers multiple measurands or units.

# Neurosemantic Communication N1 Evidence Contract

Status: research / pre-registration scaffold

## Purpose

N1 evaluates a mediated communication pipeline whose input is a validated speech or inner-speech BCI signal and whose output is a reconstructed language representation. N1 is not a test of unrestricted thought decoding.

## Required separation

1. neural recording
2. derived neural features
3. decoded linguistic representation
4. grounded concept representation
5. emitted output

Each stage must have distinct provenance, consent scope, and retained-data policy.

## Minimum evaluation design

- participant holdout: no participant appears in both training and test identities;
- utterance holdout: test utterances are absent from training;
- temporal holdout where longitudinal data exist;
- preregistered metrics and thresholds;
- confidence calibration and uncertainty reporting;
- negative controls and permutation controls;
- explicit error taxonomy for substitutions, deletions, insertions, hallucinations, and abstentions;
- reproducible model/code/configuration hashes;
- deterministic evidence bundle with dataset and split manifests.

## Promotion boundary

N1 may support a bounded communication claim only for the tested population, task, acquisition modality, decoding target, and evaluation protocol. It must not be promoted to general semantic or private-thought decoding.

N2 begins only when the system is evaluated on conceptual representations that are not reducible to the original word sequence.

## Conformal N1 calibration contract

The existing N0 empirical calibration is deliberately not a conformal guarantee.
N1 must use a separately declared nonconformity score and a calibration split that
is disjoint from the final evaluation split.

For a calibration set of size n, significance level alpha, and sorted
nonconformity scores s_(1) <= ... <= s_(n), the set-valued split-conformal
threshold is the finite-sample order statistic:

k = ceil((n+1)(1-alpha))
q = s_(min(k,n))

A test candidate is admissible only when its nonconformity score is at most q.
The finite-sample coverage statement applies only under the pre-registered
exchangeability assumptions of the specific N1 experiment. It must never be
inferred merely from a passing implementation test.

The Symthaea implementation exposes this calculation as an isolated
HdcOntologyConformalCalibration primitive. It is intentionally not wired into
the N0 decoder's acceptance policy yet. This prevents the current empirical
score/margin calibration from inheriting a distribution-free claim by proximity.

The isolated primitive now also defines a versioned baseline nonconformity score,
cosine-nonconformity-v1:

s(y, x) = (1 - cosine(x, y)) / 2,

where cosine must be finite and lie in [-1, 1]. The score therefore lies in
[0, 1], with larger values meaning less conformity. Calibration is also bound to
a canonical hash of the complete candidate universe and its size, plus a hash of
the exact sorted multiset of calibration scores. At inference, the supplied
candidate universe must reproduce that hash exactly. Missing, extra, or duplicate
candidate IDs therefore cannot silently reuse the threshold; reordering is
allowed because the canonical universe hash is deliberately order-insensitive.
The external calibration scores can be independently checked against their stored
count and hash before an artifact is trusted. Verification also recomputes q from
the supplied score set and the artifact's alpha; a numerically well-formed but
forged threshold therefore does not validate merely because the score hash matches.
The primitive turns that complete, deterministic candidate universe into a set-valued
prediction by including every candidate whose nonconformity is at most the calibrated q. Invalid identifiers or
similarities fail closed, and the emitted set is canonicalized by stable
identifier.

This is deliberately a baseline score, not a claim that cosine alone is the
best N1 uncertainty statistic. Because (1 - cosine) / 2 is an affine monotone
transform of negative cosine, it retains the basic prototype-distance ordering
but does not model interactions among competing candidates. Recent ConformalHDC
work explicitly introduces HDC-specific conformity scores that combine similarity
with class-interaction information and supports set-valued abstention. Those
richer terms are candidates for a later preregistered score revision rather than
silent post-hoc tuning here. The full candidate universe must be part of the N1
protocol: a conformal set over an incomplete or opportunistically filtered
candidate pool cannot inherit the intended coverage statement.

Small calibration sets are structurally weak. At alpha=0.10, at least 19
calibration cases are needed before the finite-sample rank can fall below the
maximum observed nonconformity score; at alpha=0.05, the corresponding minimum
is 39. The implementation also caps calibration cases and candidate identifiers
at the same defensive resource boundary used by the N1 candidate universe. N1
studies therefore need materially larger, independently generated calibration
sets rather than the current three-case N0 calibration.

### Distribution-shift boundary

The current primitive assumes that the calibration and evaluation examples are
exchangeable under the experiment's declared sampling design. The exact candidate
universe requirement is a conservative operational restriction: it prevents a
caller from silently changing the label/candidate space after calibration, but it
does not establish robustness to participant, task, modality, temporal, covariate,
or label shift.

No N1 coverage claim should survive a detected exchangeability violation merely
because the implementation still returns a prediction set. A future shift-aware
revision must be preregistered with an explicit shift model, calibration rule,
and efficiency/coverage evaluation; weighted split-conformal methods are one
research direction under declared covariate-shift assumptions. Recent 2026 work
also shows that nominal conformal coverage can degrade under distribution shift,
which is why this boundary remains explicit here (Siahkali et al., arXiv:2602.14913;
Pournaderi, arXiv:2609.33456).

Serialized ontology artifacts should cross the byte-level trust boundary through the
bounded `from_json_bytes` constructors. Ontology manifests also reject duplicate
stable identities and duplicate grounding provenance references before canonical
hashing, so canonicalization cannot silently erase an ambiguous or duplicated
identity/provenance binding. These reject oversized raw JSON before
`serde_json` materialization and then apply the semantic/resource validators.
Direct unbounded deserialization of an untrusted manifest, codebook descriptor,
representation, or conformal artifact is outside this contract. Conformal artifact
parsing validates structural state, but score-set verification must additionally
recompute the threshold from the separately retained calibration scores and alpha.

The repository now provides a versioned machine-readable `HdcOntologyConformalEvidenceArtifact` binding around the isolated conformal primitive. The current artifact schema is v3; older schema-v1 and schema-v2 artifacts are intentionally rejected rather than migrated implicitly. The artifact also binds a canonical `study_protocol_hash` covering the preregistered evaluation/shift matrix and reporting rules, so changing the study protocol creates a new evidence identity. It records the model hash, an exact inference-configuration hash, distinct calibration and evaluation split-manifest hashes, the declared coverage target, exchangeability-assumption statement, exact execution revision, and evidence-bundle identity. The inference-configuration hash is required because changing preprocessing, decoder settings, or other runtime configuration can alter the calibration/test score distribution even when the model artifact is unchanged. The execution revision is fail-closed to a canonical 40- or 64-hex Git object ID, so placeholders such as `local` or arbitrary prose cannot satisfy the provenance field. This binding is provenance infrastructure only: selecting `LabelConditional` as a declared target does not create a label-conditional theorem, and distinct split hashes do not by themselves prove that the underlying populations are exchangeable or disjoint.

A conformal N1 artifact should bind at minimum:

- exact codebook/model hash;
- exact inference-configuration hash;
- exact study protocol hash, including the declared shift/ablation matrix and reporting rules;
- exact calibration and evaluation split-manifest hashes;
- the named nonconformity score definition and revision;
- the exact calibration-score hash;
- the exact candidate-universe hash and size;
- alpha, calibration count, and resulting threshold;
- coverage, abstention/set-size, and error results on the untouched evaluation split;
- the exchangeability/independence assumptions and any known violations;
- exact execution revision and deterministic evidence bundle identity.

The N1 result must report both coverage and efficiency (for example prediction-set
size or abstention rate). Under realistic deployment shifts, reporting should also
be stratified by the preregistered shift scenario (including mixed/partial shift
where applicable) rather than collapsing all conditions into one aggregate number. Coverage without efficiency can be made trivially safe
by returning an enormous prediction set, while efficiency without coverage does
not provide the intended statistical guarantee.

The preregistration must also declare the coverage target explicitly: marginal
coverage, label-conditional coverage, or another precisely defined criterion.
The declared inference configuration must remain fixed across calibration and
evaluation unless a preregistered configuration-shift analysis explicitly
models the change and re-establishes the intended statistical validity.
Marginal finite-sample validity must not be reported as subgroup-, participant-,
or feature-conditional validity. Conditional coverage can fail for specific
subpopulations even when marginal coverage is satisfactory, so any stronger
claim requires its own evaluation design and sufficient data rather than an
unstated interpretation of the same threshold.


## Post-generation lifecycle evidence

Lifecycle status is a separate provenance/evidence layer from derivation lineage and from authorization. The receipt schema v3 distinguishes:

- Requested
- Accepted
- Processing
- Applied
- IndependentlyVerified
- Rejected

Only Applied may carry effect evidence. IndependentlyVerified additionally binds an effect-agent identity, a distinct verifier identity, separate verification evidence, the exact effect-evidence hash inspected, and an explicit verification scope.

The N0 implementation uses a schema-validated, content-addressed target-set record for EnumeratedTargetSet. The record contains the root artifact and a unique bounded set of target artifact hashes. This establishes exactly which identities the verification artifact claims to cover; it does not prove the enumeration is globally complete.

The word independent is deliberately narrow: the protocol enforces that the verifier identity differs from the recorded effect agent. It does not establish that either identity is trustworthy or that the verifier followed its claimed procedure. A future deployment bridge must supply the relevant identity/key trust and audit semantics.

For model-derived remediation, an IndependentlyVerified lifecycle receipt is still insufficient to claim preservation of utility, safety, or fairness. Any such claim requires a separate preregistered impact-evaluation artifact binding the exact pre/post model identities, derivation lineage, evaluation split/protocol, subgroup metrics, and observed behavior.
## Model-remediation impact evidence

A lifecycle receipt and a model-remediation evaluation answer different questions. The lifecycle receipt binds the recorded downstream effect and independent verification; this artifact binds whether the resulting model was actually evaluated for the declared impact dimensions.

`NeurosemanticRemediationImpactArtifact` is schema v1 and requires:

- distinct pre- and post-remediation model content identities;
- independently bound pre- and post-remediation derivation lineage records;
- the exact independently-verified lifecycle receipt fingerprint;
- a fixed study-protocol hash and evaluation split-manifest hash;
- separate content-addressed evidence for forgetfulness, retained utility/behavior, and residual/recovery risk;
- optional fairness evidence only when the fairness dimension is explicitly declared;
- an explicit disposition of `WithinDeclaredBounds`, `OutsideDeclaredBounds`, or `Inconclusive`.

The implementation deliberately avoids one aggregate unlearning score. A passing lifecycle receipt or impact artifact does not establish complete erasure, legal compliance, safety, fairness, or absence of residual model influence.

The evaluation population is itself first-class evidence. `NeurosemanticRemediationEvaluationSetManifest` separately identifies a `Forget` set and `Retain` set, each with a source-dataset manifest hash and unique bounded member artifact identities. The verifier rejects role substitution and any overlap between the forget and retain sets. Membership fingerprints are canonicalized so harmless member reordering does not create a different population identity.

`NeurosemanticRemediationEvaluationManifest` then acts as the canonical study identity. It binds the source dataset manifest, forget-set identity, retain-set identity, study protocol, evaluation split, recovery-attack method, and representation-residual probe method into one content-addressed object. The impact artifact binds this manifest fingerprint, preventing an evaluator from mixing a valid result from one evaluation design with population or methodology identities from another. The concrete bundle verifier additionally checks that the manifest's source-dataset identity matches both parsed population manifests. A valid evaluation-manifest hash therefore cannot be paired with forget/retain populations from a different dataset lineage. The evaluation-assurance layer additionally records the evaluation agent, a distinct evaluation verifier, a content-addressed evaluation environment, and exact verification evidence. Evaluation roles must not collide with the remediation effect agent or lifecycle verifier. The environment is represented by `NeurosemanticRemediationEvaluationEnvironment`, a schema-validated descriptor containing platform, runtime, toolchain, dependency-lock, configuration, and execution-revision identities. Environment verification rejects both descriptor substitution and execution-revision drift. This turns evaluator independence and runtime identity into machine-checkable evidence rather than narrative metadata.

Evaluation methodology is also content-addressed. `NeurosemanticRemediationEvaluationMethod` identifies the recovery-attack method and representation-residual probe method, binds each to the exact study protocol hash, and requires the method implementation revision to match the impact artifact execution revision. Concrete method-byte verification is therefore distinct from merely recording a method name.

The impact artifact keeps `RecoveryRisk` and `RepresentationResidual` as separate evidence kinds. This is intentional: recovery robustness and residual internal representation leakage answer related but distinct questions. A future N1 bundle should report them separately rather than allowing a single residual-risk score to conceal which failure modes were actually tested.

`NeurosemanticRemediationMeasurementArtifact` is the structural coverage ledger for those results. It requires every core evaluation dimension to appear exactly once, records a positive sample count and bounded failure count for each dimension, and recomputes the overall status using an explicit worst-case rule. The caller cannot supply a more favorable aggregate status, omit a declared dimension, or duplicate a dimension without failing validation. If the impact artifact declares fairness evaluation, the measurement ledger must also contain the fairness measurement. The impact artifact also requires exact equality between its declared disposition and the ledger's recomputed worst-case disposition. Consequently, a favorable top-level summary cannot conceal an inconclusive or out-of-bounds required dimension.

Each measurement is now self-describing through `NeurosemanticRemediationMetricDefinition`: metric identity, estimand identity, evaluation scope, unit, aggregation rule, and direction are recorded as typed fields. The corresponding measurement records carry an explicit rational estimate representation and an uncertainty state. `NotEstimated` is an explicit state rather than an omitted field; interval uncertainty records lower/upper numerators, scale, and confidence level. Interval uncertainty now also records an explicit uncertainty-method reference, and the validator requires the interval to contain the point estimate at the same numeric scale. This prevents a bare confidence percentage from being interpreted as a reproducible statistical procedure. This keeps synthetic N0 from accidentally presenting fabricated confidence intervals while giving N1 a machine-readable place to store real uncertainty.

The measurement schema is v3, and the uncertainty-computation artifact is now schema v3 because a measurement ledger cannot be the final provenance layer for a reported statistic. Each metric must also name exactly one content-addressed `NeurosemanticRemediationMetricComputationArtifact`. The computation binds the metric-definition fingerprint, the observation-set fingerprint, the aggregation procedure, the execution revision, and the resulting fixed-point estimate and counts. The observation set is itself content-addressed and contains the complete eligible population plus the exact observed subject identities; duplicate subjects, observations outside the eligible population, and duplicate observation identities are rejected.

The observation set now also carries the content-addressed identity of its eligible population/split. During strong verification, the verifier must receive the concrete population-manifest bytes and confirm that their hash matches the observation set. For forgetfulness, recovery-risk, and representation-residual metrics, the eligible observation identities must equal the canonical `Forget` manifest membership exactly; for retained utility they must equal the canonical `Retain` membership exactly. This prevents a caller from keeping the same descriptive scope label while silently evaluating a cherry-picked or reduced population. Fairness observations are bound to the exact evaluation-split artifact hash; interpretation of subgroup membership remains part of that preregistered split specification.

Fairness now uses a typed `NeurosemanticRemediationEvaluationSplitManifest` rather than an opaque split label. The split is content-addressed over the source dataset identity, exact subject membership, and subgroup assignment for each subject. Strong computation verification requires the observation set's population hash to equal the impact artifact's canonical evaluation-split hash, requires exact split membership, and compares each observed subgroup assignment to the split manifest. Thus neither a different but structurally similar split nor a reassigned subgroup can be substituted behind the same descriptive scope.

The same canonical-identity rule is applied to the other populations: forgetfulness, recovery-risk, and representation-residual must use the impact artifact's exact `forget_set_manifest_hash`, while retained utility must use its exact `retain_set_manifest_hash`. Membership equality is necessary but not sufficient; the content identity of the canonical manifest must also match. N0 now exercises substitution with an alternate valid manifest containing the same members but a different identity.

The strong computation bundle therefore has seven explicit evidence collections: the structural measurement artifact, metric computation artifacts, observation-set artifacts, population/split manifest bytes, the statistical-design artifact bytes, the statistical sampling-frame bytes, and the statistical-execution bytes. The verifier does not resolve any of these through mutable external state. Missing population evidence, population substitution, membership reduction, statistical-design evidence, or statistical-execution evidence causes verification failure.

For interval uncertainty, the measurement now carries a content-addressed `NeurosemanticRemediationUncertaintyComputationArtifact`. It binds the exact metric-definition hash, observation-set hash, point estimate, interval endpoints, confidence level, named uncertainty method, typed inference scope, statistical-design artifact identity, assumptions identity, and execution revision. Strong verification receives the concrete uncertainty, statistical-design, statistical-sampling-frame, and statistical-execution bytes and checks their bindings. The statistical-design artifact is schema v2 and now carries `statistical_execution_hash`; the execution artifact in turn binds the exact metric, observations, source-derived sampling frame, selected subjects, inclusion probabilities, dependence assignments, study protocol, and execution revision. This is stronger than a design declaration: a verifier can now detect a claimed probability sample whose supplied selection population or inclusion-probability records do not match the observed execution. It remains an evidence-coherence guarantee, not proof that the declared design assumptions are empirically true.

The uncertainty computation artifact now also binds a content-addressed assumptions statement and an explicit assumptions reference. Strong verification receives the concrete assumptions bytes and checks the exact hash. This prevents a method label from being detached from the assumptions under which its interval was produced. It still does not establish that those assumptions hold in the evaluation data; that remains an experimental/statistical question.

This creates an explicit four-way boundary: **provenance** (the exact inputs, method, design, and assumptions were identified), **structural validity** (intervals and typed artifacts obey their declared numeric domain), **calculation integrity** (the reported point estimate and currently supported Wilson interval are reproduced from the observations), and **statistical validity** (the chosen uncertainty procedure is numerically and substantively appropriate for the estimand, sampling design, dependence structure, and data-generating assumptions). The statistical-execution layer adds a narrower intermediate guarantee: **execution coherence** (the declared frame, selected subjects, inclusion-probability records, dependence assignments, study protocol, and execution revision are mutually bound). None of these machine-checkable layers proves that the sampling frame covers the intended real-world target, that randomization was honestly performed, or that the independence/dependence declaration matches every latent correlation in the data. The first three layers plus execution coherence are exercised by synthetic N0; scientific/statistical validity remains deliberately unqualified.

Because the uncertainty oracle is currently defined over an observed `failure_count`, the supported failure-rate aggregations are required to declare `LowerIsBetter`. A metric cannot simultaneously be computed as the proportion of failures and semantically declared `HigherIsBetter`. This is now enforced in the typed metric-definition validator and exercised by N0. This prevents a direction field from becoming decorative metadata that reverses the interpretation of a correctly reproduced statistic.

The protocol now has one numerically qualified uncertainty method: `wilson-score-95-v1`. Proportion intervals are structurally constrained to the closed `[0,1]` domain at their declared fixed-point scale before any method-specific calculation is considered. For binary observations, the verifier reconstructs the Wilson score interval at 95% confidence and rounds the lower endpoint outward downward and the upper endpoint outward upward into the measurement's fixed-point scale. The method is deliberately limited to explicit single-proportion failure-rate aggregations (`per-item-rate`, `attack-success-rate`, and `probe-detection-rate`) with unit `proportion`. Its uncertainty computation artifact also carries a typed inference scope: `Superpopulation` is required for the current Wilson method, while `FixedEvaluationPopulation` remains a distinct descriptive target for future finite-population procedures. The method is therefore not applicable to `worst-subgroup-gap`, whose estimand is a difference between subgroup proportions and requires a distinct interval procedure. It is not silently substituted for clustered, hierarchical, dependent, finite-population, or shift-aware evaluation. N0 exercises the supported path on a synthetic 0/2 failure result and expects the outward-rounded interval `[0, 0.6577]`, while also constructing a valid-looking Wilson interval for a fairness gap and requiring rejection, plus a Wilson artifact marked as fixed-population inference and requiring rejection. These are numerical-method, method-applicability, and inference-target regressions, not evidence of meaningful coverage on a real evaluation population.

The current design contract now keeps four claims separate: the evaluator can identify a statistical design; the verifier can check that the selected uncertainty method is compatible with that declared design; the verifier can also check execution coherence against concrete frame/selection/dependence evidence; and an independent scientific study can determine whether the declared design assumptions actually hold in the world. Probability-sampling evidence is now explicit: the execution artifact records the full content-addressed frame, the selected observation identities, and a positive inclusion probability for every frame member. Each inclusion probability is encoded as a reduced positive rational, so mathematically equivalent values cannot acquire different evidence identities through alternate numerator/denominator encodings. This follows the United Nations sampling guidance that probability-sample inference requires non-zero, calculable inclusion probabilities for target-population units and selected units, while complex designs can additionally require first- and second-order inclusion probabilities or design-based variance information (see the UN *Handbook on Surveys in Developing Countries* and Statistics Canada guidance on second-order inclusion probabilities). NIST AI 800-3 (2026) notes that implicit assumptions and mismatched statistical models can invalidate uncertainty statements; NIST's measurement-science guidance likewise documents that autocorrelation invalidates independence-based uncertainty calculations. The implementation therefore treats the concrete execution artifact as evidence of protocol coherence, not as proof of coverage or independence.

This distinction is important because a syntactically valid interval can still be statistically invalid. NIST AI 800-3 emphasizes that uncertainty treatment depends on the evaluation model and estimand, and NIST's measurement-science guidance treats uncertainty as an explicit property of a measurement process rather than a confidence label attached after the fact. The protocol therefore keeps method identity, input identity, and numerical correctness as separate evidence claims.

For the currently supported computation rules, the verifier independently reconstructs the point estimate from the observation records rather than trusting the reported numerator. Item-rate, attack-success-rate, and probe-detection-rate are reconstructed as binary failure proportions; worst-subgroup-gap is reconstructed from the extrema of subgroup failure rates. The fixed-point estimate is compared by exact rational arithmetic, so an evaluator cannot change the reported statistic while preserving the surrounding counts. The computation verifier also binds the observation scope to the metric-definition scope and requires the computation execution revision to match the remediation impact artifact.

The strong verification path accepts the exact observation-set, population/split, statistical-design, sampling-frame, and statistical-execution bytes explicitly supplied by the caller. It does not consult a mutable repository, network service, or ambient resolver to discover evidence. This is intentional: a reproducibility claim should identify the bytes used in the calculation, not merely depend on whatever an external lookup returns at verification time. The statistical sampling frame is content-addressed to the source dataset identity, while the execution artifact binds the selected subjects and inclusion probabilities to that frame. Dependence assignments are likewise content-addressed to the same observed subject identities. These bindings close substitution routes without turning declarative provenance into empirical truth.

The N0 fixture exercises estimate forgery, observation substitution, observation omission, duplicate observation insertion, scope substitution, computation swapping, statistical-design substitution, sampling-frame/execution substitution, inclusion-probability tampering, and dependence-execution tampering. All of those controls are expected to reject. The current Wilson fixture uses a synthetic frame derived directly from the canonical forget-set manifest and assigns every frame member an inclusion probability of 1/1; that is deliberately a degenerate implementation fixture, not evidence of real probability sampling or superpopulation coverage. N0 remains synthetic and therefore the overall remediation disposition remains `Inconclusive`; the new evidence layer demonstrates machine-checked execution coherence and computational integrity, not scientific validity.

NIST AI 800-3 (2026) provides a direct motivation for this separation: evaluation metrics are estimates of explicitly defined estimands, and different estimands require appropriate estimation methods and uncertainty treatment rather than a one-size-fits-all formula. NIST's 2026 AITE program similarly emphasizes common data, metrics, and scoring over blind/sequestered data to improve comparability and reduce contamination. Those principles support keeping metric definitions, observation populations, and point-estimate calculations as separately inspectable evidence identities here.

These controls follow current unlearning verification research: output-level forget-set accuracy and retained utility can miss information preserved in internal representations, while recovery attacks can expose knowledge after apparently successful unlearning. See Cosma & Finke, “RULER: Representation-Level Verification of Machine Unlearning” (arXiv:2605.27569, 2026); Qian et al., “Leak-Resistant Unlearning” (arXiv:2608.04519, 2026); and Ebrahimpour-Boroojeny et al., “Unlearning Isn’t Forgetting” (PMLR 306, 2026).

NIST's 2026 TEVV-Athlon direction likewise emphasizes customizable, system-specific evaluation and explicitly treats testing, evaluation, verification, and validation as distinct activities. See NIST AI 200-2 draft, “The TEVV-Athlon Framework for Evaluating AI Systems” (August 2026). NIST's 2026 AI RMF Core also identifies independent assessors and documentation of test sets, metrics, and TEVV tooling as explicit evaluation measures. See NIST AI RMF Core, Measures 1.3 and 2.1–2.4 (2026). NIST's 2026 AI 800-3 work likewise warns that benchmark reporting can conflate distinct performance targets or leave uncertainty implicit, reinforcing the need to state exactly what a measurement covers. Its distinction between fixed benchmark accuracy and generalized accuracy is the rationale for making the estimand and scope explicit here rather than letting one score inherit a broader interpretation. See NIST AI 800-3, “Expanding the AI Evaluation Toolbox with Statistical Models” (February 2026). NIST also notes that different estimands require different estimation methods for valid uncertainty quantification, which motivates binding the uncertainty method to the interval rather than recording confidence alone.

The N0 example uses synthetic evidence and therefore reports `Inconclusive`. Its purpose is to demonstrate identity binding and fail-closed substitution controls, not to qualify a real remediation method. Concrete verification entry points cover the lifecycle receipt, pre/post lineage records, study-protocol bytes, evaluation-split bytes, and each declared impact-evidence class.

Research direction: RULER demonstrates that output-level unlearning checks can miss residual information in internal representations, while recent benchmarks show that recovery attacks can reveal knowledge after apparently successful unlearning. Accordingly, a future N1 remediation campaign should retain both output-level and representation/recovery evidence rather than selecting whichever metric is most favorable. (Cosma & Finke, RULER, arXiv:2605.27569; Qian et al., Leak-Resistant Unlearning, arXiv:2608.04519.)

Fairness and post-deployment effects must also be evaluated separately. Recent clinical-AI work found that forgetting medical records can interact with subgroup disparities, and NIST's 2026 monitoring guidance emphasizes that post-deployment monitoring is necessary because controlled pre-deployment evaluation cannot capture all real-world consequences. (Chen et al., *Nature Communications* 17, 6009, 2026, doi:10.1038/s41467-026-72601-7; NIST AI 800-4, 2026.)

Therefore a future model-impact evidence bundle should stratify the declared results by the preregistered evaluation matrix, record negative/recovery controls, and preserve enough artifact identity to reproduce exactly which model versions and data partitions were tested.

The next unresolved statistical boundary is stronger than the current content-addressed execution contract: **authenticated execution of the selection procedure and independent validation of frame coverage**. A `selection_procedure_ref` and a coherent inclusion-probability table do not prove that the claimed randomization actually occurred, nor that the sampling frame exhaustively represents the intended target population. A future N1 study should therefore preserve a replayable or independently attested selection trace (for example, a preregistered seed/procedure plus the resulting selected identities), and should retain whatever design-specific first- or second-order inclusion-probability or weighting information is required by the estimator. For clustered or repeated-measures studies, the dependence artifact should likewise carry authoritative grouping/time-structure evidence and be paired with a variance procedure appropriate to that dependence structure. Missingness remains another explicit boundary: currently any unobserved eligible cases force an `Inconclusive` disposition rather than silently assuming MCAR/MAR/MNAR.
## Privacy requirements

Consent must specify the permitted data class and inference class separately. A participant agreeing to communication assistance does not automatically authorize unrelated secondary inference, model training, affective inference, or commercial analytics.

Policy-bearing packets must additionally bind the declared data class to an intrinsic payload type. Opaque legacy payloads may remain readable for compatibility, but they must fail closed at the authorization boundary unless the payload type itself makes the declared data class machine-verifiable.

Revocation must have a defined effective time and an auditable downstream behavior. The transport layer must not be treated as the authority for semantic access.

## Current research anchor

Recent 2026 work demonstrates noninvasive sentence decoding from MEG/EEG in a 35-person healthy-volunteer cohort, including held-out sentences, which makes an N1-style evidence contract technically relevant while still leaving a substantial gap between bounded sentence decoding and unrestricted thought reading.

An October 2026 Nature Neuroscience ethics perspective argues that ethical clarity must keep pace with expanding implantable BCI capability and emphasizes meaningful clinical purpose and long-term obligations to participants.

## Data and inference authorization

Every N1 pipeline stage must declare its data class and inference classes separately from transport sensitivity.

At minimum, records should distinguish:

- raw neural recordings;
- derived neural features;
- semantic representations;
- decoded/reconstructed claims;
- personalized decoder/model state.

Inference authorization should independently identify potential disclosures such as signal patterns, linguistic content, semantic content, affective state, intent, and identity.

A communication purpose or transport permission must never imply permission for every inference available from the same artifact. Legacy or unknown classifications must fail closed.

This separation is consistent with recent iBCI governance analyses that distinguish these data products and identify conflated consent and limited misuse guardrails as key privacy gaps. (Sandbrink & Young, Communications Medicine, 27 July 2026, DOI 10.1038/s43856-026-01797-y; Young et al., Device, available 11 August 2026, DOI 10.1016/j.device.2026.101271.)
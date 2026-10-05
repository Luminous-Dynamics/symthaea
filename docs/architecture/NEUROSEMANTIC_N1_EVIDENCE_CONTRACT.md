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

The repository now provides a versioned machine-readable `HdcOntologyConformalEvidenceArtifact` binding around the isolated conformal primitive. The current artifact schema is v2; older schema-v1 artifacts are intentionally rejected rather than migrated implicitly. It records the model hash, an exact inference-configuration hash, distinct calibration and evaluation split-manifest hashes, the declared coverage target, exchangeability-assumption statement, exact execution revision, and evidence-bundle identity. The inference-configuration hash is required because changing preprocessing, decoder settings, or other runtime configuration can alter the calibration/test score distribution even when the model artifact is unchanged. The execution revision is fail-closed to a canonical 40- or 64-hex Git object ID, so placeholders such as `local` or arbitrary prose cannot satisfy the provenance field. This binding is provenance infrastructure only: selecting `LabelConditional` as a declared target does not create a label-conditional theorem, and distinct split hashes do not by themselves prove that the underlying populations are exchangeable or disjoint.

A conformal N1 artifact should bind at minimum:

- exact codebook/model hash;
- exact inference-configuration hash;
- exact calibration and evaluation split-manifest hashes;
- the named nonconformity score definition and revision;
- the exact calibration-score hash;
- the exact candidate-universe hash and size;
- alpha, calibration count, and resulting threshold;
- coverage, abstention/set-size, and error results on the untouched evaluation split;
- the exchangeability/independence assumptions and any known violations;
- exact execution revision and deterministic evidence bundle identity.

The N1 result must report both coverage and efficiency (for example prediction-set
size or abstention rate). Coverage without efficiency can be made trivially safe
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
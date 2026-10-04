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

Small calibration sets are structurally weak. At alpha=0.10, at least 19
calibration cases are needed before the finite-sample rank can fall below the
maximum observed nonconformity score; at alpha=0.05, the corresponding minimum
is 39. N1 studies therefore need materially larger, independently generated
calibration sets rather than the current three-case N0 calibration.

A conformal N1 artifact should bind at minimum:

- exact codebook/model hash;
- exact calibration and evaluation split-manifest hashes;
- the named nonconformity score definition and revision;
- alpha, calibration count, and resulting threshold;
- coverage, abstention/set-size, and error results on the untouched evaluation split;
- the exchangeability/independence assumptions and any known violations;
- exact execution revision and deterministic evidence bundle identity.

The N1 result must report both coverage and efficiency (for example prediction-set
size or abstention rate). Coverage without efficiency can be made trivially safe
by returning an enormous prediction set, while efficiency without coverage does
not provide the intended statistical guarantee.


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

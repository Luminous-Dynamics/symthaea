# Clinical Inference Envelope v2

## Purpose

`ClinicalInferenceEnvelopeV2` is the external-interoperability evolution of the v1 evidence-bound inference envelope.

v1 binds exact digests but does not say which canonicalization/digest namespace gave those bytes their semantic meaning. v2 makes that boundary explicit without granting any clinical authority.

## Core rule

A digest is not an evidence identity by itself.

Every evidence-like object is represented as:

```text
ClinicalEvidenceIdentityV2 {
    identity_version,
    namespace,
    artifact_id,
    digest,
}
```

`namespace` identifies the canonicalization/digest contract, not merely a human-readable data category.

Examples are illustrative and must correspond to actual qualified contracts before use:

- `mycelix/clinical-fact-snapshot/v1`
- `mycelix/patient-subject-binding-evidence/v1`
- `hl7/fhir-r4/resource-canonical/v1`
- `ohdsi/omop-artifact/v1`
- `symthaea/model-training-lineage/v1`
- `symthaea/model-evaluation-lineage/v1`
- `symthaea/model-calibration-evidence/v1`
- `symthaea/ood-detector-evidence/v1`

The same 32-byte digest under two different namespaces is intentionally a different evidence identity.

## Typed evidence boundaries

v2 types the evidence identity of:

- claim-supporting/opposing/context/contraindication evidence;
- subject-binding evidence;
- model training lineage;
- model evaluation lineage;
- model calibration evidence;
- inference calibration evidence;
- distribution/OOD detector evidence;
- all runtime execution evidence inputs.

Runtime and configuration digests remain artifact/execution-context identities rather than clinical evidence identities.

## Execution binding

Every claimed evidence reference must appear exactly in `execution.input_evidence`.

Likewise, subject-binding evidence must appear in `execution.input_evidence`.

Therefore changing only the namespace, artifact ID, or digest of a claimed evidence object breaks execution binding. A caller cannot take an inference produced from one typed evidence artifact and relabel it afterward as a different evidence domain.

## Model calibration

When an inference declares `Calibrated`:

- calibrated probability is required;
- typed inference calibration evidence is required;
- typed model calibration evidence is required;
- the two evidence identities must match exactly, including namespace, artifact ID and digest.

Matching raw digest bytes under different namespaces do not satisfy this rule.

## Distribution evidence

`InDistribution` and `OutOfDistribution` require typed detector evidence.

This remains producer evidence only. A downstream Mycelix deployment must independently qualify detector trust, currentness, policy compatibility and patient applicability. Producer-declared `InDistribution` is never clinical authority.

## Duplicate evidence identity

Within the envelope evidence list, `(namespace, artifact_id)` must be unique. Supplying the same named evidence artifact twice with different digests is rejected rather than interpreted as two independent facts.

## Subject semantics

`subject_namespace` identifies the subject identifier space (for example `fhir/Patient`).

`binding_evidence.namespace` identifies the canonicalization/digest contract for the evidence proving that subject binding.

These are separate concepts and must not be inferred from each other.

## Authority boundary

The v2 envelope is still evidence, not authority. It contains no mechanism to authorize:

- diagnosis;
- treatment selection;
- prescribing;
- dispensing;
- administration;
- autonomous therapeutic action;
- clinician presentation.

Those decisions remain downstream qualification/authority responsibilities.

## v1 compatibility

v2 does not silently reinterpret v1 bare digests.

v1 may remain useful for internal/preflight use under explicitly controlled assumptions. External medical interoperability should prefer v2 or a separately qualified crosswalk that proves the exact evidence namespace/canonicalization contract.

A v1 digest must never be type-cast into a v2 evidence identity merely because the digest bytes match.

## Next qualification layer

The semantic contract is necessary but not sufficient for cross-repository admission. Before external promotion, v2 still requires:

1. a frozen canonical wire/framing contract;
2. domain-separated v2 wire identity;
3. cross-repository conformance vectors;
4. an independent Mycelix v2 parser/verifier;
5. deployment admission policy over accepted evidence namespaces and exact lineages;
6. exact Mycelix evidence crosswalk verification;
7. distribution/runtime/currentness/human-authority gates.

# Clinical Inference Wire v2 / Binary Framing v1

## Purpose

This document freezes the first identity-bearing wire representation for `ClinicalInferenceEnvelopeV2`.

Unlike the experimental v1 JSON wire, the v2 external-interoperability wire is serializer-independent. Its exact bytes are determined by this specification rather than by a JSON implementation.

## Versioning

Two versions remain separate:

- envelope schema: `2`;
- binary wire framing: `1`.

Any incompatible framing change requires a new wire-framing version/domain. Any incompatible semantic change requires a new envelope schema version.

## Header

Every message begins with:

1. 8-byte magic: ASCII `SYMCLN2` followed by `0x00`;
2. unsigned big-endian `u16` wire version (`1`);
3. envelope payload.

Wrong magic, unknown wire version, truncation, or trailing bytes fail closed.

## Primitive framing

- `u8`: one byte.
- `u16`, `u32`, `u64`, `i64`: big-endian.
- `f64`: exact IEEE-754 `to_bits()` encoded as big-endian `u64`.
- fixed byte arrays: exact bytes, no length prefix.
- string: big-endian `u32` byte length followed by UTF-8 bytes.
- vector: big-endian `u32` item count followed by items in order.
- option: `0x00` for absent, `0x01` followed by value for present. Other tags are invalid.

Vector order is identity-bearing. Reordering evidence or execution inputs changes the exact wire bytes even if a higher-level consumer considers the set equivalent.

## Hard decoder bounds

- maximum complete message: 4 MiB;
- maximum textual narrative field: 256 KiB;
- maximum identifier-like string: 4 KiB at the wire layer (the semantic v2 layer imposes stricter namespace/artifact limits);
- maximum vector items: 4096.

The complete input byte slice is bounded before nested allocation. Invalid length claims fail closed.

## Digest framing

`ClinicalInferenceWireDigestV2` uses BLAKE3 derive-key mode with context:

`"symthaea.clinical.inference-wire-v2.v1"`

Digest material is framed as:

1. `u16` wire version;
2. `u16` domain-tag byte length;
3. domain tag `symthaea/clinical-inference-wire/v2/v1`;
4. `u64` canonical-wire byte length;
5. exact canonical wire bytes.

The digest proves exact wire identity only. It proves neither truth nor authority.

## Enum tags

### Digest algorithm

| Tag | Value |
| ---: | --- |
| 0 | BLAKE3-256 |

### Clinical claim kind

| Tag | Value |
| ---: | --- |
| 0 | CandidateSignal |
| 1 | Association |
| 2 | Prediction |
| 3 | RiskEstimate |
| 4 | CausalHypothesis |
| 5 | CausalEffectEstimate |
| 6 | DiagnosticSupport |
| 7 | TreatmentSupport |

### Evidence stage

| Tag | Value |
| ---: | --- |
| 0 | MechanisticHypothesis |
| 1 | SyntheticDemonstration |
| 2 | RetrospectiveInternal |
| 3 | RetrospectiveExternal |
| 4 | ProspectiveShadow |
| 5 | ProspectiveClinicalStudy |
| 6 | ReplicatedClinicalEvidence |

### Applicability

| Tag | Value |
| ---: | --- |
| 0 | Unestablished |
| 1 | EvaluatedCohortOnly |
| 2 | DefinedTargetPopulation |
| 3 | ValidatedTargetPopulation |

### Intended use

| Tag | Value |
| ---: | --- |
| 0 | ResearchOnly |
| 1 | ClinicalDecisionSupport |

### Evidence role

| Tag | Value |
| ---: | --- |
| 0 | Supports |
| 1 | Opposes |
| 2 | Context |
| 3 | Contraindication |

### Calibration status

| Tag | Value |
| ---: | --- |
| 0 | NotAssessed |
| 1 | Uncalibrated |
| 2 | Calibrated |

### Distribution status

| Tag | Value |
| ---: | --- |
| 0 | Unknown |
| 1 | InDistribution |
| 2 | OutOfDistribution |

### Missing-evidence criticality

| Tag | Value |
| ---: | --- |
| 0 | Informational |
| 1 | Important |
| 2 | Critical |

## Top-level payload order

The v2 payload is encoded in this exact order:

1. envelope schema version;
2. claim semantics;
3. optional subject binding;
4. statement;
5. evidence refs;
6. alternatives;
7. missing evidence;
8. uncertainty;
9. distribution assessment;
10. execution identity;
11. generated-at timestamp.

Nested object field order is the order implemented and documented by the v1 framing code. It is part of the contract and cannot be reordered without a new wire version.

## Canonicalization

There is exactly one valid byte representation for a given v2 object under binary framing v1.

The parser:

1. checks total length;
2. checks magic/version;
3. decodes bounded typed fields;
4. rejects invalid option/enum tags;
5. rejects invalid UTF-8;
6. rejects trailing bytes;
7. re-runs full v2 semantic validation.

`clinical_inference_wire_v2_digest_from_bytes` additionally re-encodes the decoded object and requires exact byte equality before hashing.

## Frozen conformance vector

`fixtures/clinical_inference_wire_v2.hex` is the first portable canonical vector.

- decoded byte length: 1260 bytes;
- format: ASCII hexadecimal with insignificant line breaks;
- the integration test requires the encoder to reproduce these exact bytes and the decoder to reproduce the exact typed object.

A downstream Mycelix verifier should copy the fixture bytes and independently reproduce decoding and digest framing; it must not import Symthaea implementation code merely to pass conformance.

## Non-claims

A valid canonical wire artifact establishes only that the bytes encode a structurally/semantically valid v2 Symthaea inference under this framing. It does not establish evidence truth, model validity, calibration quality, OOD detector validity, patient applicability, regulatory clearance, clinician-presentation authority, or treatment authority.

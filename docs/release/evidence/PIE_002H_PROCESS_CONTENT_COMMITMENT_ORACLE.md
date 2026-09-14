# PIE-002H-A Process Content Commitment Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

Freeze one canonical content-identity theorem for a full validated PIE `ProcessDefinition` without claiming currentness, provenance authenticity, truth, applicability, feasibility, dispatch, economics, or authority.

This tranche exists because PIE-002E proves invocation-current reprojection from the process object supplied to a call, but a stable process ID or revision coordinate does not prove what exact process content was supplied.

## Governing separation

```text
process_id
    != process content

revision / lineage coordinate
    != process content commitment

process content commitment
    != authority-currentness
    != truth
    != feasibility
```

## Exact commitment profile

Profile/domain:

`Symthaea PIE Process Content V1`

Domain-separation prefix bytes:

`symthaea.pie.process-content.v1\0`

Digest: SHA-256 over the exact canonical byte stream.

The baseline golden digest is:

`423cb41801ddfeb75e2c5d4a13d7edb3f108a2825f2965d4e669ee934736440d`

A future Rust/cross-language mirror must reproduce the canonical bytes and this vector, not merely hash equivalent-looking JSON.

## Committed ProcessDefinition surface

The profile commits the complete process-definition surface present in PIE:

- exact `process_id`;
- exact process `name`;
- every material input;
- every material output and output disposition;
- every utility declaration;
- every equipment requirement and its evidence;
- every environment constraint;
- every process-level evidence record.

Material grades, optional specification references, physical forms, input/output roles, utility kinds, dependency criticality, celestial-body constraints, ranges, evidence class/source/note, and explicit disposition targets are identity-bearing where present.

This is deliberately broader than a utility-projection commitment. A change to evidence, equipment, environment, input/output, name, or utility demand changes the full process-content ID even if PIE-002B would currently project the same electrical demand.

## Collection semantics

The current Rust structures use `Vec`, but no PIE theorem assigns semantic authority to insertion order for process inputs, outputs, utilities, equipment requirements, environment constraints, or evidence.

V1 therefore commits these collections as canonical multisets:

1. encode every element independently;
2. sort encoded element bytes lexicographically;
3. commit the exact item count and each length-prefixed encoded item.

This gives:

```text
permutation of the same records -> same content ID
additional duplicate record      -> different content ID
```

Multiplicity remains observable. This matters for utility declarations because duplicate peak-power/process-time entries remain semantically meaningful ambiguity under PIE-002B.

Evidence duplicate IDs are rejected before commitment, preserving the existing PIE evidence-validation rule.

## String encoding and durable IDs

All strings are UTF-8 with explicit big-endian length prefixes. Optional strings carry an explicit absent/present discriminator, so `None` and `Some("")` are distinct.

Human-readable names/notes/source strings are preserved exactly and are not silently case-folded or Unicode-normalized.

Commitment admission is intentionally stricter than legacy PIE `require_text` for stable IDs/references. The following must already be canonical lexical IDs:

- process ID;
- material join keys;
- equipment class keys;
- evidence IDs;
- sink/process disposition targets.

V1 rejects surrounding whitespace and ASCII control characters rather than trimming or mutating the identifier. This is the compatibility-safe strengthening tracked by PIE-ID-001 (#2870).

## Enum encoding

Enums use explicit one-byte V1 discriminants matching the frozen semantic variant sequence in the oracle. Encoding does not depend on Rust compiler/native enum layout.

Changing the commitment profile or variant mapping requires a new domain/profile version; it must not silently reuse V1.

## Numeric encoding

Validated finite non-negative range endpoints are encoded as fixed-endian IEEE-754 binary64.

`-0.0` is canonicalized to `+0.0` before encoding because PIE numeric equality treats them as the same semantic value.

NaN and positive/negative infinity fail closed before commitment. Reversed or negative ranges fail closed.

## Boundedness

The independent oracle freezes conservative admission budgets:

- max string bytes: 4,096;
- max items per committed collection: 1,024;
- max complete canonical process byte stream: 1,048,576 bytes.

Budget exhaustion fails before a digest is labeled a complete process commitment.

These are commitment-profile bounds, not claims that every lower PIE API has the same limits.

## Adversarial fixtures

The checked-in self-test proves at minimum:

1. baseline full-process golden SHA-256 vector;
2. independent permutation of Vec-backed collections preserves content identity;
3. utility multiplicity changes content identity;
4. process-name changes change identity;
5. utility range changes change identity;
6. material-input changes change identity;
7. material-output changes change identity;
8. equipment changes change identity;
9. environment changes change identity;
10. evidence-note changes change identity;
11. `-0.0` and `+0.0` produce the same canonical semantic encoding;
12. non-canonical process/material/evidence/reference IDs fail closed;
13. duplicate evidence IDs fail closed;
14. non-finite quantities fail closed;
15. collection budget exhaustion fails closed;
16. string budget exhaustion fails closed.

## Relationship to PIE-002E and PIE-002H-B

PIE-002E independently proves:

```text
process object supplied to invocation
    -> validation
    -> in-call reprojection
    -> explicit binding
```

PIE-002H-A adds:

```text
validated exact process content
    -> canonical ProcessContentIdV1
```

It does not resolve which content commitment is uniquely current.

PIE-002H-B should later adapt a qualified generic Symthaea append-only/fork-aware currentness primitive rather than copying a weaker PIE-specific `latest timestamp wins` resolver.

The intended higher path is eventually:

```text
canonical process ID
+ exact process-content commitment
+ admitted lineage/currentness evidence
    -> authority-current process subject
    -> PIE-002E in-call reprojection
    -> explicit context binding
    -> opaque witness (#2826)
```

## Non-claims

A matching SHA-256 digest does not prove who authored, measured, admitted, or currently endorses the process definition. It does not prove evidence applicability, thermodynamic realizability, process feasibility, equipment availability, economics, currentness, or execution authority.

Tracks #2867, #2870, #2782, #2783, #2618, #2785, #2826, #1610, #1647, and master #1604.

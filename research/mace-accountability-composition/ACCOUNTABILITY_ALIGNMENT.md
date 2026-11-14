# MACE accountability composition alignment (research note)

Date checked: 2026-11-13. This note is an implementation crosswalk, not a claim of IETF conformance.

## New adjacent work discovered

The IETF Internet-Draft `draft-mih-sato-agent-accountability-composition-02` (published 2026-11-02; Informational work in progress; expires 2027-05-06) defines independently verifiable CAN, WHO, WHAT and AUDIT slots composed by a shared action subject digest or explicit protected cross-reference. It calls for per-slot results to remain distinct, a typed result vocabulary (`pass`, `fail`, `not_evaluated`, `unsupported`, `indeterminate`), exact bundle pinning and frozen positive/negative/condition-removed vectors. A vector freezes only after recomputation by at least two independent implementations.

The same draft makes critical limits explicit: transparency registration anchors a signed statement but does not prove truth or completeness; a missing action cannot be detected by registration alone. Its current -02 cross-slot pack is still a candidate pending a second independent implementation. Therefore MACE should integrate this work, not claim the proposed cross-slot conformance is already achieved.

## MACE mapping

| Accountability question | Existing MACE/Luminous owner | Current gap |
|---|---|---|
| CAN — permitted action | #9490; independent task/policy pins | A receipt join is not a complete profile-level capability or policy appraisal |
| WHO — accountable human authority | no complete MACE slot yet | Agent/workload principal is not necessarily the accountable human; privacy-preserving principal commitments and delegation/quorum semantics remain |
| WHAT — recorded outcome and effect boundary | #11845 and #11870 | Current effect receipt verifies a source's signed statement; it does not prove external-world truth or unobserved effects |
| AUDIT — runtime enforcement | #11741/#11638, #9560/#3535 | Current prototype does not appraise measured platform state or prove observer completeness |
| Belief provenance extension | future Symthaea evidence slot | Must expose source/evidence lineage, not private chain-of-thought or fabricated introspection |

## Digest discipline

Do not treat MACE's current fixed-Go-struct JSON SHA-256 values as interoperable subject digests. The current implementation deliberately does not claim RFC 8785 JCS or a cross-language wire format.

A future composition adapter must pin, for every digest comparison: profile label/version, exact subject field projection, canonicalization, hash algorithm, domain-separation rule, byte/text representation and the exact preimage. Keep three roles distinct:

1. subject digest — identifies the action projection joined across profiles;
2. profile-tagged authority-reference digest — commits to native evidence supporting a slot;
3. transparency receipt-payload digest — commits to exact submitted bytes.

Do not infer equality or transitive coverage between digest roles. Reuse #9374's canonical signed-envelope work and #9560/#3535's transparency mechanisms.

## Required report shape

A composition report must preserve each native slot result unchanged and separately state each cross-slot join. Named checks use `pass`, `fail`, `not_evaluated`, `unsupported` or `indeterminate`; only `pass` is a pass. A failed prerequisite should leave dependent checks `not_evaluated`, not invent failures. No opaque `trusted=true` boolean or weighted score may hide a failed or unresolved required slot.

## Conformance pack proposal

Create immutable positive, negative and condition-removed controls. Each vector pins the exact action bytes, profile revisions, serialization suite, native evidence bytes/digests, digest contexts, protected references, expected per-slot and per-join results, and the expected terminal report. Every negative mutation changes one condition; its paired condition-removed control restores only that condition and must return to the positive result. Publish the input manifest, SHA-256 checksums, executable runner, raw stdout/stderr, source revision, toolchain identity and report digest.

Freeze only after an independently maintained second implementation consumes the same exact vector bytes and reproduces the result. The current Go implementation is one prototype; this package alone does not satisfy independent-party conformance.

## Proposed programme targets (not achieved)

- Full positive vector and every negative/condition-removed pair pass with no silently skipped required checks.
- 100% of changed subject bytes, digest context, profile labels, receipt payloads and protected references are detected.
- WHO and AUDIT remain `unsupported` until their native profiles are actually implemented; they are never represented as passes by mapping a machine identity or log receipt into them.
- Two independent implementations agree on all frozen vector results.
- Observer completeness measured against independent OS/API/network state ground truth with precision, recall, loss and coverage boundaries reported separately.
- Every published report carries immutable source, policy, fixture, toolchain and environment digests.

## Boundary

These Internet-Drafts are work in progress, not standards or certification. This research crosswalk is not evidence that the project conforms to them. A shared digest establishes linkage under a declared byte profile; it does not establish authorization, truth, completeness, sufficiency or policy acceptance.

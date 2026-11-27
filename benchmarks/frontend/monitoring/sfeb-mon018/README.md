# MON-018 — Mycelix Authority Evidence Composition

**Status:** research contract and synthetic qualification corpus; not a production adapter.

MON-018 defines a read-only boundary for composing Mycelix institutional identity/delegation/current-authority evidence with MON action evidence. It does not move authorization policy into MON, let Mycelix certify its own evidence, or let an authority verdict imply an external effect.

## Research question

Can an independent MON verifier establish that the exact principal, delegation, authority epoch, action, resource, destination and request were permitted at the destination's actual commit boundary, using cryptographically verified and sufficiently current evidence?

## Files

- [AUTHORITY-EVIDENCE-CONTRACT-v1.md](./AUTHORITY-EVIDENCE-CONTRACT-v1.md) — architecture, trust boundaries, semantics, adapter contract, qualification plan and benchmark design.
- [SCENARIOS-v1.json](./SCENARIOS-v1.json) — finite known-answer adversarial seed corpus.
- [SCENARIOS-v1.schema.json](./SCENARIOS-v1.schema.json) — strict JSON Schema for the corpus.
- [authority-join/](./authority-join/) — standard-library Go semantic reference and adversarial tests for a deliberately narrow v0.1 profile.

## Existing systems reused

- Mycelix ROS-005 delegated authority primitives: [PR #3660](https://github.com/Luminous-Dynamics/mycelix/pull/3660).
- Mycelix provider-authority timeline: [PR #1836](https://github.com/Luminous-Dynamics/mycelix/pull/1836).
- MON destination/effect and typed evidence foundations: [MON-013 #13763](https://github.com/Luminous-Dynamics/symthaea/pull/13763), [MON-014 #13768](https://github.com/Luminous-Dynamics/symthaea/pull/13768), [MON-015 #13817](https://github.com/Luminous-Dynamics/symthaea/pull/13817).
- SCITT transparency architecture: [RFC 9943](https://www.rfc-editor.org/rfc/rfc9943).

## Claim ceiling

This increment is a research design plus a synthetic corpus. It does not provide a Mycelix runtime adapter, cryptographic credential verifier, Holochain integration, PostgreSQL integration, independent source deployment, physical-effect qualification, CI pass, or production-readiness claim. Scenario counts are not security probabilities.

MON-018 authority qualification is one input to a later execution gate. It is never a dispatch command and never proof that an external effect occurred.
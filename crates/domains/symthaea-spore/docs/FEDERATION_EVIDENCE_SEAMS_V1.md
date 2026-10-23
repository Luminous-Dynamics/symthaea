# SPORE-FED-001H — Cross-Layer Evidence Seams

**Issue:** #6314
**Parent:** #6309 / #6311 / #6312
**Claim ceiling:** reproducible software/network semantics only

## Why deepen the witness?

The first witness proves useful boundaries, but a serious federation observatory needs to prevent a second class of category error: treating identity, a carried claim, an authorization decision, and an execution result as one object.

That distinction is consistent with mature external architectures:

- W3C Verifiable Credentials 2.0 defines issuer, holder, verifier, and verification as distinct roles; importantly, verification of a credential does not by itself establish the truth of the claims it contains.
- NIST SP 800-207 separates authentication from authorization and describes policy decision/enforcement as distinct functions.
- IETF RFC 9171 defines DTN as a store-carry-forward overlay and explicitly does not make bundle transport equivalent to guaranteed end-to-end application delivery.

## Five evidence layers

| Layer | Question | Positive evidence | Must not imply |
|---|---|---|---|
| Identity | Who produced this? | authenticated origin / stable node identity | authorization |
| Claim | What is being asserted/carried? | signed or provenance-bearing artifact | truth of the claim |
| Verification | Does the artifact satisfy its verification rules? | valid signature/status/format | truth or policy permission |
| Policy | May this actor authorize this action here and now? | explicit policy decision + authority/policy epoch | execution |
| Execution | Did the authorized action actually happen? | execution receipt / observed result | policy legitimacy or future success |

## Transport adds another independent seam

| Transport state | Meaning | Does it prove application execution? |
|---|---|---|
| queued | accepted for later transport | **No** |
| transmitted | handed to a convergence/transport layer | **No** |
| delivered | received by the destination transport/application boundary | **No** |
| accepted | domain accepted the artifact under its own rules | **No** |
| authorized | policy layer explicitly permitted the action | **No** |
| executed | action was actually performed and recorded | **Yes, for that recorded execution only** |

This is especially important for the interplanetary story. RFC 9171 describes DTN as handling intermittent connectivity and delay, while noting that BP itself does not ensure end-to-end delivery. The showcase therefore must never animate a moving packet directly into an execution result.

## Adversarial matrix

| Attack / failure | Expected disposition | Evidence preserved |
|---|---|---|
| forged issuer | reject | presented issuer + verification failure |
| valid credential from foreign authority | verify, then policy-reject | issuer, verifier, policy epoch |
| stale credential/status | reject as stale | credential generation/status evidence |
| valid observation with uncertain truth | retain as observation, not fact | provenance + epistemic status |
| analysis presented as authorization | policy-reject | artifact lineage + local policy |
| authorization without execution | remain authorized/not-executed | authorization receipt |
| transport delivered without domain acceptance | remain transport-delivered | transport receipt + domain disposition |
| replayed authorization | reject | sequence/idempotency evidence |
| stale policy epoch | reject | policy epoch mismatch |
| capability copied to another node | capability may transfer; identity/authority may not | source + recipient identities |

## Observatory UI consequence

The visual layer should become a **causal evidence graph**, not just a network map.

Every visible edge should answer one question:

```text
IDENTITY → CLAIM → VERIFICATION → POLICY → EXECUTION
                    │
                    └── epistemic status / provenance

TRANSPORT: QUEUED → TRANSMITTED → DELIVERED
                              │
                              └── independent of EXECUTION
```

A user should be able to click any node in this chain and see the evidence that justified that state.

## Stronger public demo

The adversarial demonstration should deliberately perform:

1. a legitimate capability share;
2. a forged-issuer attempt;
3. a valid-but-foreign credential;
4. an analysis recommendation presented as an authorization;
5. an authorization issued but no execution receipt produced;
6. a delayed DTN delivery;
7. a stale policy-epoch message after recovery;
8. replay of the original event ledger.

The visual result is not simply 'the network survived.' It is a visible proof that the system preserves distinctions while the network is degraded.

## External references

- W3C Verifiable Credentials Data Model 2.0: https://www.w3.org/TR/vc-data-model/
- NIST SP 800-207 Zero Trust Architecture: https://www.nist.gov/publications/zero-trust-architecture
- IETF RFC 9171 Bundle Protocol Version 7: https://www.rfc-editor.org/rfc/rfc9171.html

## Non-claims

This document does not establish:

- that a credential's claims are factually true merely because they verify;
- that an authorization is socially or politically legitimate;
- that transport delivery guarantees execution;
- that a simulated DTN profile is physical interplanetary communications;
- consciousness, sentience, governance outcomes, or economic outcomes.
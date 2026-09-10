# Health authentication wire v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-distributed-health-v2`

Parent exact head at branch creation:

`6154e5d2de14aa2ed05430c531a56d633952628b`

Exact code subject frozen before this evidence-only commit:

`bbe345202b761b0775af580daa09f59afcc4e9d2`

This subject defines stable canonical authentication bytes for the two exact local-health claim classes used by continuity:

```text
PostExecutionHealthClaimV1
    -> symthaea.continuity.post-execution-health-wire.v1\0
    -> canonical_post_execution_health_claim_bytes()

CrashSourceHealthClaimV1
    -> symthaea.continuity.crash-source-health-wire.v1\0
    -> canonical_crash_source_health_claim_bytes()
```

Each canonical payload validates the claim first and then commits the exact identity/provenance fields, verifier profile, health profile, observation time, health outcome, optional health-state digest with explicit presence tag, raw evidence digest, and canonical claim id.

The two claim classes have distinct authentication-purpose strings and distinct wire domains. A signature/attestation over one health semantic must not be interpretable as the other.

Serde/JSON/CBOR representation is explicitly not the signing or attestation contract.

Qualification must include golden byte/digest vectors before any external adapter treats these payloads as protocol-stable. Cross-language implementations must reproduce those vectors exactly.

This subject does not authenticate evidence by itself, define a cryptographic algorithm, prove health, establish distributed health, complete recovery, promote LKG, or grant execution/retry authority.

No test or CI pass is claimed here. Exact-head CI and all stacked parent qualification remain required.

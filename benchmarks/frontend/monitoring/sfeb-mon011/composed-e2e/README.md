# MON-011 composed trace preflight

This package is a byte-exact *composition* fixture across the MON-010 source assertion, typed claim, transformation and independent replay, a separately approved decision, an OAR-shaped action lifecycle, commit-time authority, the MON-009 sink record, and independent readback.

## Why this is not a new receipt format

The artifacts are adapters' already-authenticated source/decision/approval/authority/sink/observation records. This module hashes each exact supplied payload byte sequence and verifies the digest references and cross-layer joins. It does not sign records, issue a transparency receipt, verify COSE/SCITT, implement RFC 8785/JCS, or authenticate keys. The Go JSON encoder is used only to produce deterministic local test fixtures. Production adapters must preserve and verify the bytes of their actual profile (for example OAR JCS payloads) and pass the exact payload bytes and verified normalized fields to this checker.

## Composition checked

`source assertion -> claim -> transformation -> independent replay -> decision -> approval/display binding -> exact request bytes -> action intent -> authority re-check at commit -> atomic sink commit -> read-only independent observation`

The claim binds the exact source artifact and statement digests plus subject, unit, numerator/denominator, population, geography, period and methodology. The transformation binds source, claim, code, configuration, input and output digests. A second trust domain replays the same transform and must produce the same output. The decision then binds source, claim, transformation and replay digests. The request artifact is separately hashed from its exact bytes. The final chain also binds authority-set digest, policy digest, OAR-style instance/display commitments, approver binding, principal, tenant/resource/mutation path, operation ID, sink revision and before/after state commitments, then the observer's exact sink digest/revision/state commitment.

Three distinct times are explicit: source valid-time, when a correction became effective, and recorded/transaction time, when the verifier learned about it. A correction known by the audit cutoff and effective before commit challenges the decision while retaining the independently confirmed committed effect (`stale` + `committed`). A later-effective correction triggers a current-risk review without invalidating the earlier action. A correction not yet recorded by the audit cutoff does not leak into that as-of view. Readback freshness is measured from commit to observation, not from the date an auditor later reopens the bundle.

Outcome mapping follows OAR's `committed`, `refused`, `unresolved` classes. Only a matching read-only system-of-record observation yields `committed`. A missing confirmation remains `unresolved`. The internal status preserves stronger states: `qualified`, `indeterminate`, `contradicted`, `stale`, `unauthorized`.

## Run

```sh
./reproduce.sh
```

Go 1.23.2 / linux/amd64 local results:
- 30 named adversarial tests;
- `go test -race -count=5 ./...` passed;
- `go vet ./...` passed;
- 67.9% statement coverage.

The readback freshness window is measured from sink commit to observation, not from the date on which an auditor happens to re-open the bundle. A later audit must not retroactively invalidate an observation that was timely when made. Current source status is still reported separately for current-risk review.

## Interoperability boundary

See [OAR-INTEROP-v1.md](../OAR-INTEROP-v1.md) for field-by-field mapping to OAR's authority set, approval binding, action instance, execution commitments and outcome classes. See [MATCHED-BENCHMARK-PROTOCOL-v1.md](../MATCHED-BENCHMARK-PROTOCOL-v1.md) for the industry comparison and W0-W6 benchmark ladder.

The model is informed by Open Action Record `draft-malhotra-scitt-oar-01` (25 Nov 2026) and the public `openactionrecord/oar` reference implementation and vector corpus. The repository's OAR 0.3 / -00-era corpus has 18 main and 6 compatibility cases; the published manifest includes authority revocation before commit, approval flip, missing/removed records, tampering, disclosure erasure, and a successful API response whose outcome is not confirmed by readback. The public OAR -01 draft says its new constructions are not yet implemented. We map to OAR's action lifecycle and preserve its outcome classes; we do not claim full OAR -01 or full OAR 0.3 cryptographic conformance.

OAR's outcome confirmation may be performed by the issuer or another party with read access. MON additionally requires a separately identified trust domain where an independent observer is a policy requirement. This is a stricter deployment qualification, not a claim that OAR mandates that exact architecture. The source/claim/transformation/replay layer is MON's composition with MON-010: it is not part of OAR's action receipt and must not be represented as such.

## What this does not prove

- Source or decision truth.
- Publisher, approver, authority or issuer signature validity; those are adapter inputs.
- The revocation status itself; only its signed, independently verified status input is checked.
- The truth of the source figure or the semantic correctness of the transformation. The replay demonstrates deterministic reproducibility of the supplied pipeline, not factual correctness of its inputs or model.
- PostgreSQL transaction atomicity, sink-side operation uniqueness, transparency-log inclusion or physical effect.
- Universal canonicalization, trusted time, complete logs or non-equivocation.

Those remain separate profile adapters, database/physical tests and independent trust-anchor/checkpoint requirements.

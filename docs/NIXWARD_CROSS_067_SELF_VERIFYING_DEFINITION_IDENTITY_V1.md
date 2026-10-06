# Nixward CROSS-067 — Self-Verifying Definition Identity V1

## Purpose

CROSS-067 closes the remaining receipt-side opacity around systemd unit-definition identity.

A receipt previously persisted only an `observed_definition_digest`. That made the digest tamper-evident, but did not make the underlying FragmentPath/DropInPaths identity independently reconstructable from the serialized receipt.

## Persisted definition identity

The durable receipt now retains the exact:

- `FragmentPath`;
- ordered `DropInPaths` set.

The existing definition-identity type validates absolute paths and canonical deterministic ordering.

## Independent verification

During receipt validation:

1. the persisted definition identity is shape-validated;
2. its deterministic digest is recomputed against the exact target unit;
3. the recomputed digest must equal `observed_definition_digest`;
4. `observed_definition_digest` must equal `authorized_definition_digest`.

Therefore a serialized receipt cannot silently replace the observed source identity with another valid source while preserving an old or merely self-consistent digest.

A forged attacker would have to modify the authorized commitment as well, which is outside the receipt's independent trust boundary and is checked against the bound authorization/intent records.

## Durable digest

The receipt digest commits both the definition digests and the underlying FragmentPath/DropInPaths values.

Consequently, changing either the definition identity or its associated digest changes the durable receipt identity.

## Source identity is not content identity

CROSS-067 deliberately keeps the layers distinct:

```
FragmentPath + DropInPaths
        = systemd-reported source identity

content digest
        = exact bytes, when separately observed

NixOS generation / configuration identity
        = declarative provenance
```

CROSS-067 does not pretend that a path identity is a content hash.

CROSS-060 remains responsible for binding the authorized service-definition commitment into the authority chain.

## Claim ceiling

This tranche does not prove file bytes were unchanged after the observation, does not provide cryptographic attestation of systemd, and does not solve authorization-bound definition capture by itself.

It closes a narrower and testable gap: the durable receipt's observed definition digest is no longer an opaque claim; the verifier can reconstruct that identity from the receipt itself and check it against the authorization-bound digest.

## Qualification

Exact-head GitHub Actions remain the qualification authority. Mergeability, queued jobs, or local plausibility are not PASS evidence.

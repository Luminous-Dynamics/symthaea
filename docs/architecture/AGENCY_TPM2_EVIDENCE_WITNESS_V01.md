# Agency TPM2 evidence witness V0.1

## Status

Authored design and tests. **Not qualified yet.** This layer is post-verification evidence/notarization only; it does not create runtime authority.

## Purpose

`#423` can execute the exact-head TPM2 qualification locally and seal an evidence archive. `#431` independently interprets a complete PASS archive and emits a closed-world acceptance JSON.

This layer answers the next question:

> Which independent witnesses reviewed/accepted that exact interpreted release evidence under which witness policy?

It deliberately signs the **interpreted acceptance commitment**, not a free-form statement and not merely the tarball bytes.

## Strict input boundary

`parse_release_acceptance_v1` accepts only the exact #431 V1 schema and requires all of:

- `accepted = true`;
- `qualification_result = PASS`;
- `archive_hash_source = caller`;
- `external_head_bound = true`;
- `external_tree_bound = true`;
- `release_bound = true`.

Therefore a sidecar-only acceptance cannot later be upgraded into release evidence merely because two witnesses signed it.

The acceptance digest commits:

- archive SHA-256;
- evidence-manifest SHA-256;
- exact Git HEAD and tree;
- canonical commitment to the retained locked nixpkgs object;
- `flake.lock` SHA-256;
- `rust-toolchain.toml` SHA-256;
- approved PCR profile;
- platform-attestation policy digest;
- AK public-key digest;
- fresh TPM challenge digest;
- qualification-probe SHA-256;
- hermetic quote/checkquote launcher SHA-256 values;
- exact Nix verifier-store identity.

The nested nixpkgs object is canonicalized recursively with lexicographically sorted object keys and no floating-point numbers before domain-separated BLAKE3 commitment. JSON object key ordering therefore cannot alter the semantic acceptance digest.

## Witness policy

`QualificationWitnessPolicyV1` defines:

- policy identity;
- monotonic witness-policy epoch;
- witness threshold;
- minimum independent organizations;
- minimum independent services;
- exact allowed evidence-verifier implementation digests;
- canonical enrolled witness identities and Ed25519 public keys.

Witness identities are explicit triples:

`witness_id / organization_id / service_id`.

Two keys operated by one organization therefore do not satisfy a policy requiring two organizations. Likewise, two processes presenting the same enrolled witness ID do not count twice.

Rotating `witness_epoch` invalidates attestations from the previous policy/key generation without changing the underlying qualification acceptance.

## Signed statement

Each `QualificationWitnessAttestationV1` signs a domain-separated transcript containing:

- schema and signature algorithm;
- exact witness-policy digest;
- exact release-acceptance digest;
- exact evidence-verifier implementation digest;
- witness-policy epoch;
- witness/organization/service identity;
- witness-local monotonic sequence.

The sequence is **bound but not persisted by this crate**. A production witness service must own durable sequence state and reject regression. This crate does not pretend a caller-supplied sequence is itself an anti-rollback store.

## Key custody

The library signing API receives an already-instantiated `ed25519_dalek::SigningKey`. It deliberately defines no seed-file format, default development key, environment-variable key, or embedded key store.

Production key custody belongs in a dedicated witness boundary such as Xenia, TPM/HSM-backed signing, or another independently administered service.

## Quorum semantics

`verify_qualification_witness_quorum_v1` requires every attestation to bind the same:

- release acceptance;
- verifier implementation;
- witness policy;
- witness epoch.

It independently verifies Ed25519 signatures and then enforces:

- unique enrolled witnesses;
- threshold count;
- organization diversity;
- service diversity.

The successful return type is opaque and contains no execution capability.

## Security non-claims

A witness quorum does **not** establish that:

- the TPM was physical rather than swtpm;
- measured-boot or IMA event logs reconstruct correctly;
- the host kernel/hypervisor was uncompromised;
- the qualification producer or #431 verifier are bug-free;
- the witness signing keys are hardware protected;
- a witness sequence was durably monotonic unless its service proves that separately;
- Symthaea may execute an action.

It means only that the reviewed witness policy accepted signatures over one exact release-bound interpretation of the qualification evidence.

## Future composition

The natural next publication layer is:

```text
exact-head qualification
        ↓
independent evidence verifier
        ↓
release-bound acceptance digest
        ↓
threshold witness quorum
        ↓
Xenia / transparency / SCITT publication
```

Publication should preserve the distinction between evidence validity and execution authority. A qualification witness must never become a capability grant.
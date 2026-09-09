# symthaea-extension-admission

Typed, runtime-neutral admission contracts for Symthaea extensions.

This crate sits between **discovery** and **routing/execution**. It deliberately
performs no cryptography, package loading, networking, or cognitive work.

## Core theorem

```text
ExtensionManifest
    != TechnicalCompatibility
    != SignatureValidity
    != SignerAuthorization
    != AdmissionRecord
    != ActiveAdmission
    != InvocationAuthorization
```

An `AdmissionRecord` is immutable, serializable issuance evidence. It binds the
exact manifest SHA-256, executable/package payload SHA-256, admission-policy
SHA-256, issuer, optional already-resolved signer identity, trust floor,
capability grant, permission grant, and admission generation.

An `ActiveAdmission` is intentionally non-Serde and non-Clone. It can be created
only by revalidating an `AdmissionRecord` against:

1. the current exact `ExtensionManifest` value;
2. the current admission generation; and
3. live non-revocation state.

The active value retains the exact manifest it was activated against, so it
cannot be replayed against a different manifest that merely reuses the same
extension ID and version.

## Attenuation only

Admission may narrow an extension's declaration, never widen it:

- granted capabilities must be a subset of `manifest.provides`;
- network/filesystem/sensor/actuator permissions must be subsets of requested
  permissions;
- boolean authority such as GPU, wall clock, and randomness can only be granted
  when the manifest requested it.

Filesystem paths are treated as exact opaque grant strings here. Concrete hosts
remain responsible for symlink resolution, sandbox path mapping, and OS-level
containment.

## Point-of-use rule

Runtime registries may cache discovery metadata and quality telemetry. They
should not cache `ActiveAdmission` across revocation/generation checks.

The intended path is:

```text
serialized AdmissionRecord
        + current manifest
        + live generation/revocation context
        -> ActiveAdmission
        -> one routing/invocation decision
```

## Non-claims

This crate does not prove that:

- a package signature is valid;
- a signer is authorized;
- the manifest/payload digests were computed correctly;
- an admitted provider is scientifically accurate;
- a capability invocation is safe in its current physical context.

Those facts belong to the trust verifier, package/control host, evidence system,
and capability-specific safety policy respectively.

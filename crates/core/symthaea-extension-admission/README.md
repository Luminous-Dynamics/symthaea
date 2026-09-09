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
    != AdmissionRecordEvidence
    != AdmissionRecord
    != ActiveAdmission
    != InvocationAuthorization
```

`AdmissionRecord` is an **in-process host-issued authority record**. It binds the
exact manifest SHA-256, executable/package payload SHA-256, admission-policy
SHA-256, issuer, optional already-resolved signer identity, trust floor,
capability/permission grants, policy generation, and signer-trust generation.

It deliberately implements `Serialize` but **not `Deserialize`**.

Persisted admission JSON is parsed as `AdmissionRecordEvidence`. That type can be
structurally validated and inspected, but there is intentionally no conversion
from it back into `AdmissionRecord` or `ActiveAdmission`.

After a process restart the host must therefore rerun:

```text
exact package bytes
    -> technical inspection
    -> signer verification / current trust state
    -> local admission policy
    -> new AdmissionRecord
```

Stored positive evidence alone can never recreate runtime authority.

## Point-of-use authority

An `ActiveAdmission` is intentionally non-Serde and non-Clone. It is created only
from a live in-process `AdmissionRecord` after checking:

1. the current exact `ExtensionManifest` value;
2. the current policy/admission generation;
3. the current signer-trust generation; and
4. live non-revocation state.

The active value retains the exact manifest against which it was activated, so a
same-ID/same-version manifest substitution fails before routing.

The two generations answer different questions:

- **policy/admission generation** — is this still the current authority grant?
- **trust generation** — is the signer/key authorization state used to issue the
  grant still current?

Either changing invalidates the point-of-use admission.

## Attenuation only

Admission may narrow an extension declaration, never widen it:

- granted capabilities must be a subset of `manifest.provides`;
- network/filesystem/sensor/actuator permissions must be subsets of requested
  permissions;
- GPU, wall-clock, and randomness authority can only be granted when requested.

Filesystem paths are exact opaque grant strings here. Concrete hosts remain
responsible for symlink resolution, sandbox path mapping, and OS containment.

## Runtime caching rule

Runtime registries may cache discovery metadata and quality telemetry. They must
not cache `ActiveAdmission` across policy/trust/revocation checks.

```text
live AdmissionRecord
        + current exact manifest
        + current policy generation
        + current trust generation
        + live revocation state
        -> ActiveAdmission
        -> one routing/invocation decision
```

## Non-claims

This crate does not prove that:

- a package signature is valid;
- a signer is authorized;
- the manifest/payload digests were computed correctly;
- persisted admission evidence is authentic merely because it parses;
- an admitted provider is scientifically accurate;
- a capability invocation is safe in its current physical context.

Those facts belong to the trust verifier, package/control host, evidence system,
and capability-specific safety policy respectively.

# symthaea-extension-data-host

Fail-closed host for declarative `data_only` Symthaea extensions.

This path intentionally never executes package payload bytes. It reuses the same
technical-inspection/signature/admission pipeline as executable extensions, but
with a capability-specific data validator instead of Wasmtime.

```text
manifest + payload
      -> DataPackHost<V> technical inspection
      -> SignerVerifier
      -> AdmissionPolicy
      -> AdmissionRecord
      -> AdmissionAuthority + live currentness
      -> ScopedAdmission
      -> DataPackHost<V>::open
             |
             +-> host AuthorityScope identity check
             +-> live currentness recheck
             +-> exact byte commitments
             +-> capability-specific validator
      -> ValidatedDataPack
```

## Why a separate path

`ExtensionRegistry` may discover both executable providers and declarative packs,
but `ExtensionRouter` is an invocation router and categorically rejects
`RuntimeKind::DataOnly`. Declarative content therefore cannot become executable
authority accidentally.

## Capability-specific validation

The host does not define a universal arbitrary-JSON knowledge format.
`DataPackValidator` is supplied by the domain/capability owner and identifies the
semantic capability whose payload shape it understands.

The generic host enforces:

- manifest and payload byte-size ceilings;
- strict `ExtensionManifest` parsing/validation;
- `runtime == data_only`;
- the validator capability is actually declared and pure;
- exact manifest and payload SHA-256 binding;
- exact admitted manifest/capability authority;
- process-local host-authority identity;
- authoritative policy/trust/revocation currentness at the actual use site;
- capability-specific payload validation only after authority checks pass.

## Point-of-use rule

`DataPackHost` is constructed with an `AuthorityScope`. `open()` accepts a
`ScopedAdmission`, not a free-standing `ActiveAdmission`, and callers cannot
supply a replacement scope at the use site.

The host first proves that the admission was minted by the exact process-local
`AdmissionAuthority` to which the host is bound. It then asks the host-supplied
`AdmissionCurrentnessSource` to revalidate policy generation, signer-trust
generation and revocation state. Only after those authority checks pass does it
parse/hash the package and run the domain validator.

This means a caller cannot create a second local authority, recreate all visible
record fields, and use that token against a host bound to the first authority.
The scope identity is process-local and non-serializable.

`ValidatedDataPack` is intentionally an in-memory, non-Serde handle. The
resulting handle records the policy-admission and signer-trust generations used
at the point of use for downstream evidence.

Persisted admission evidence, foreign authority tokens, stale active tokens,
same-ID manifest substitution, or modified payload bytes cannot shortcut the
data path.

## Non-goals

- no Wasmtime or native code loading;
- no generic JSON query language;
- no claim that structurally valid data is epistemically true;
- no global schema registry yet;
- no capability routing for executable providers;
- no signer implementation or trust database;
- no persistent/global authority identifier.

The first real consumer is the strict epistemic global-claim pack adapter. Broader
data-pack abstractions should wait for measured experience from real domain
consumers.

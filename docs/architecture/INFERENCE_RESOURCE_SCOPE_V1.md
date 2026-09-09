# Symthaea Inference Fabric — IF-12 Resource Scope Coherence v1

Status: local resource-authority child of IF-11.

## Purpose

IF-11 binds the current provider profile, credential identity/epoch, and quota state
into execution evidence, but those are still independent namespaces.

IF-12 establishes a separate theorem:

> the current provider profile account scope, exact credential identity/epoch, and
> exact quota identity/epoch are all explicitly registered to the same local
> provider/deployment/account resource scope.

This is intentionally not inferred from identifier text.

## Opaque identifiers

Credential ids and quota scope ids are opaque names. For example, these strings:

`example-provider:account-a`

and:

`example-provider:account-a`

still establish no relationship merely because they happen to be equal.

The credential and quota identities must each have an explicit local registration
against an `InferenceResourceScopeKey` containing:

- provider id;
- deployment id;
- provider account-scope id.

The current IF-10 `QualifiedProviderCandidate` supplies the profile-side scope.

## Local scope registry

`InferenceResourceScopeRegistry` is identified by a non-zero 32-byte registry id.
That registry lineage is included in every proof digest.

Mutable access to the registry is itself an authority boundary. IF-12 assumes that
production wiring will keep registration mutation behind trusted local deployment or
future Xenia authority. The module does not claim that an arbitrary caller-created
registry is independently trustworthy.

Each credential/quota registration also carries:

- a non-zero local authority epoch;
- a digest of the local configuration/source snapshot that established the mapping.

Raw source material and credential secrets are not retained.

## Rollback resistance

Registrations are keyed by the exact credential id/epoch or quota scope id/epoch.
For a given credential id or quota scope id, installing another registration requires
a strictly newer resource epoch.

Once a newer credential/quota epoch is registered, verification using an older epoch
fails as superseded even though its historical registration remains in the registry.

This prevents a stale resource mapping from being replayed merely because it once
existed.

## Verification

`verify(profile, credential, quota)` fails closed unless:

1. the exact credential id/epoch has a current registration;
2. the exact quota scope id/epoch has a current registration;
3. the credential registration equals the profile's provider/deployment/account scope;
4. the quota registration equals the same profile scope.

No prefix, suffix, substring, naming convention, URL, API-key format, or provider
brand name participates in this decision.

## Proof digest

A successful verification produces `VerifiedInferenceResourceScope` and the domain:

`symthaea.inference.resource-scope-proof.v1`

The proof digest binds:

- local registry id;
- exact IF-10 provider-profile digest;
- provider/deployment/account scope;
- credential id + epoch;
- credential registration authority epoch + source digest;
- quota scope id + epoch;
- quota registration authority epoch + source digest.

Therefore changing registry lineage, provider-profile evidence, credential/quota
epochs, or local mapping evidence changes the proof even if visible account strings
remain the same.

## Qualification regressions

The focused IF-12 suite establishes:

- coherent explicit registrations verify;
- matching-looking identifier strings do not create authority;
- a credential mapped to another account fails;
- quota mapped to another deployment fails;
- newer credential epochs supersede old mappings;
- newer quota epochs supersede old mappings;
- registration rollback/equal-epoch replacement fails;
- registry lineage and local mapping evidence affect proof identity;
- the IF-10 provider-profile snapshot affects proof identity;
- a zero registry identity fails closed.

## Explicit non-claims

IF-12 does **not** yet establish:

- binding the resource-scope proof into IF-11/v3 execution permits;
- a v4 execution digest;
- trusted/cryptographic authorization for registry mutation;
- Xenia-backed scope registration;
- durable/distributed resource-scope consensus;
- provider-side proof that a credential actually owns an account;
- provider-side proof that quota observations belong to an account;
- automatic discovery of account/project/organization ids;
- deployment-aware IF-8 routing;
- provider presets or production runtime/factory changes.

The next theorem should compose this verified resource-scope proof into the final
execution binding and re-verify it adjacent to I/O, so credential/quota/profile scope
substitution becomes a one-use permit failure rather than merely a preflight check.

# Symthaea Inference Fabric — IF-11 Current Provider Profile Binding v1

Status: execution-binding child of IF-10.

## Purpose

IF-10 establishes that provider/model configuration is freshness-bounded evidence.
IF-11 carries that theorem through the final external-call boundary.

A profile that was valid during route/admission preparation is not automatically
valid at execution. The current profile for the exact provider/deployment/model key
is resolved and qualified again adjacent to network I/O, and its identity is part of
the one-use execution binding.

## v3 provider-state binding

IF-11 introduces the domain:

`symthaea.inference.provider-execution-binding.v3`

The v3 provider-state digest composes:

- the existing IF-4 v2 candidate + endpoint-state digest;
- IF-10 provider-profile digest;
- deployment id;
- account-scope id;
- profile epoch;
- profile validity horizon.

`qualified_at_tick` is deliberately excluded from digest identity. Requalifying the
same unchanged claim bundle later must reproduce the same binding. Freshness is a
separate current-time predicate checked on every bind/rebind.

## Current-profile resolution

`CurrentProfileCredentialExecutor` owns:

- the provider-profile resolver;
- one exact provider/deployment/model profile key;
- the qualification policy;
- the IF-9 credential-bound private executor;
- one shared trusted clock object.

Ordinary execution does not accept a caller-selected `QualifiedProviderCandidate`.
It resolves the current profile for its owned key immediately before forwarding to
the private executor.

A newer installed profile therefore supersedes an older prepared binding even if the
older profile has not reached its expiration tick.

## Construction derives the wire endpoint

The safe constructor is `from_current_profile()`.

Provider id, wire model, and base URL are derived from the currently qualified
registry profile. They are not independent caller parameters. This prevents a caller
from qualifying one provider profile while constructing the HTTP executor for a
different endpoint/model tuple.

If the registry later changes the endpoint, the existing executor fails closed during
profile/endpoint revalidation. A new executor configuration must be constructed from
the newer current profile before execution can proceed.

## One shared trusted clock

The safe constructor accepts one `Arc`-shared `InferenceTickSource`. The exact Arc is
retained for profile freshness and cloned into the underlying inference executor.

Therefore callers cannot submit a per-request freshness timestamp and profile
qualification/receipt timing cannot accidentally use two independently constructed
clock objects in the safe path.

## Supersession and expiry

Prepared authority is invalidated when any bound current-profile identity changes.
Examples include:

- profile epoch advancement;
- evidence snapshot/provenance changes that alter the profile digest;
- deployment/account-scope substitution;
- endpoint/model/provider drift;
- profile expiration.

Profile lookup/qualification failure consumes `PreparedInferenceExecution` into a
local verification-rejected receipt. The prepared authority is not returned to the
caller and no network evidence is attached.

When a newer still-fresh profile has the same endpoint semantics but a different
profile identity, the final v3 rebind produces `BoundStateChanged` before I/O.

## Compatibility

Legacy IF-4 v2 `execute()` and IF-9 credential-bound methods remain present for
stack compatibility. IF-11 adds a stricter profile-bound path rather than silently
changing older semantics.

Future runtime integration should prefer `CurrentProfileCredentialExecutor` for
remote provider execution.

## Qualification regressions

The focused IF-11 regression suite establishes:

- unchanged profile requalification at a later trusted tick reproduces the same v3 binding;
- a newer profile epoch invalidates still-fresh prepared authority before network I/O;
- expiration consumes the prepared attempt locally with no wire evidence;
- deployment/account-scope changes alter provider-state identity;
- endpoint drift after preparation is rejected before network I/O;
- safe construction derives provider/model/base URL from registry truth;
- an unchanged current profile completes a real localhost HTTP request and produces a receipt.

## Explicit non-claims

IF-11 does **not** yet establish:

- that a credential id cryptographically/administratively belongs to the profile's account-scope id;
- that quota authority belongs to the same account scope as the credential/profile;
- durable/distributed provider-registry consensus;
- signed deployment-manifest verification;
- deployment-aware IF-8 router identity;
- automatic executor reconstruction after endpoint/profile changes;
- production runtime module export or replacement of the current `LLMBackend` factory;
- provider presets or live provider discovery.

The next security theorem should bind **credential scope + quota scope + provider
profile account scope** into one locally verified resource identity before provider
presets are allowed to become a production route.

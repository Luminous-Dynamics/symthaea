# Symthaea Inference Fabric — IF-9 Credential-Bound Executor v1

Status: credential identity/secret binding child of IF-8.

## Purpose

IF-4 binds a non-secret `CredentialStateBinding` into execution evidence while the
OpenAI-compatible transport separately receives bearer material. IF-9 closes the
resulting identity gap for the safe integration path: one non-clone credential
lease carries both the exact credential identity/epoch and the wire credential
material.

## Credential handle

`InferenceCredentialHandle` is non-secret metadata:

- credential id;
- expected credential epoch;
- required credential mode (`Anonymous` or `Bearer`).

It is suitable for policy/configuration references. It contains no bearer value.

## Credential lease

`InferenceCredentialLease` is deliberately non-Clone and non-Serde. It owns:

- the exact `CredentialStateBinding` used in IF-3/IF-4 execution binding;
- credential mode;
- the transport credential material used on the wire.

Its custom `Debug` renders only credential id, epoch, and mode. Secret material is
never formatted.

## Resolver boundary

`InferenceCredentialResolver` is the future integration point for OS keyrings,
hardware-backed stores, or Xenia capability-bound secret access.

`resolve_credential()` rechecks that a resolver returned exactly the requested id,
epoch, and mode. A resolver cannot silently substitute another account/epoch.

## Safe executor wrapper

`CredentialBoundOpenAiExecutor` consumes one lease at construction. It moves the
lease's wire material into the existing private transport and retains only the
lease's non-secret credential binding beside it.

The wrapper exposes `credential_binding()` for admission/permit construction.
Its `execute()` and `execute_streaming()` methods deliberately do **not** accept a
caller-supplied credential binding. They always forward the binding paired with the
secret at construction.

Therefore a caller cannot prepare evidence for credential B and then ask this safe
wrapper to execute with secret A.

A regression constructs a permit against a forged credential binding and confirms
the wrapper rejects it as bound-state drift before network I/O.

## Compatibility and migration

The earlier raw `OpenAiInferenceExecutor` remains unchanged for stacked-contract
compatibility. Future runtime export/integration should prefer the credential-bound
wrapper and avoid exposing raw transport/executor constructors to ordinary callers.

## Secret lifetime

IF-9 fixes identity provenance but does not yet make bearer material request-scoped.
The underlying IF-1 transport still owns the credential for the executor lifetime.
A later secure-store tranche should resolve short-lived credential material adjacent
to request construction so idle executors do not retain long-lived secrets.

## Explicit non-claims

- no production OS keyring implementation;
- no Xenia secret-store integration;
- no hardware-backed key attestation;
- no per-request/zeroized bearer lease yet;
- no credential rotation in-place on a live executor;
- no current runtime module export;
- no provider presets;
- no current `LLMBackend` factory/routing change.

# Symthaea Inference Fabric — IF-1 OpenAI-Compatible Transport

Status: transport-only child of IF-0.

## Purpose

Many hosted and self-hosted inference services expose OpenAI-compatible chat
completion endpoints. Symthaea should not duplicate one HTTP client per provider.
IF-1 introduces one generic transport whose configuration carries provider id,
endpoint, model id, credential mode, and timeout.

## Security properties

- bearer credentials have redacted `Debug` output;
- transport/config debug output does not reveal bearer material;
- empty bearer credentials are rejected;
- only HTTP(S) base URLs are accepted;
- remote non-success bodies are not included in `TransportError`, because a
  provider error body may echo private prompt/context and later be logged;
- endpoint joining is canonicalized to avoid provider-specific slash handling.

## Authority boundary

This transport does not import or evaluate IF-0 policy and cannot mint an
`AdmittedInferenceRoute` or future `InferencePermit`. It simply performs a wire
operation when an authorized higher layer eventually invokes it.

The existing `OpenAiBackend`, provider factory, and environment-variable priority
remain unchanged. That prevents this transport tranche from silently changing
which data leaves the machine.

## Explicit non-claims

- not exported from `language` yet;
- not wired into `LLMBackend`;
- no provider presets;
- no provider-policy assertions;
- no quota/rate-limit telemetry;
- no retry logic or circuit breaker;
- no credential store;
- no current runtime behavior change.

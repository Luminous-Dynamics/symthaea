# Symthaea Inference Fabric — IF-4 Permit-Gated Executor v1

Status: transport-authority integration child of IF-3.

## New invariant

A production OpenAI-compatible HTTP transport must not be handed to ordinary
callers. The IF-4 executor privately owns it and requires ownership transfer of a
`PreparedInferenceExecution` before network I/O.

Immediately before I/O the executor re-runs admission and reconstructs the exact
v2 execution binding. Any request, policy, provider, credential, quota, sampling,
endpoint, model, timeout, credential-mode, or endpoint-config drift fails before
the HTTP request is sent.

## Why binding v2

IF-3 v1 established stable semantic digests. It deliberately did not yet bind
wire-only execution controls. IF-4 composes the v1 semantic request digest with:

- temperature represented as integer thousandths;
- streaming mode.

It composes the v1 provider candidate digest with:

- protocol id;
- canonicalized base URL produced by the transport config;
- wire model id;
- request timeout in milliseconds;
- credential mode (`None` or `Bearer`, never secret material);
- endpoint config epoch.

The final permit's request/provider digests are under the separate
`symthaea.inference.execution-binding.v2` domain, so v1 and v2 evidence cannot be
confused.

## Provider/model coherence

The IF-4 adapter currently requires `RemoteProvider` plus `ProviderAttested` model
identity whose provider/model exactly match the OpenAI-compatible endpoint config.
Content-verified local models and opaque endpoints remain outside this adapter and
must receive separately qualified execution paths.

## Wire derivation

The executor does not accept an arbitrary `TransportGenerationRequest`. It derives
the HTTP payload from the bound `InferenceRequest` and `InferenceGenerationControls`:

- prompt <- bound request prompt;
- system prompt <- bound request system prompt;
- max tokens <- bound request output limit;
- temperature <- bound integer-milli control.

Tool and structured-output requests fail closed because IF-1's transport does not
yet implement those wire contracts. Non-streaming and streaming execution have
separate entry points and the selected mode is part of the v2 request digest.

## Terminal evidence

Operational failures such as provider HTTP failure, decode failure, empty response,
or binding rejection return a terminal `InferenceExecutionOutcome` carrying a
failure receipt. Successful output is returned to the cognitive caller while the
receipt stores only its digest.

Receipt construction can still fail if the supplied qualified tick source regresses
behind the preparation tick. That is a fatal clock/evidence error, not an inference
failure, and no false terminal receipt is fabricated.

## Explicit non-claims

- no runtime export through `language::mod` yet;
- no replacement of the current `LLMBackend` factory;
- no provider presets;
- no credential store tying a secret handle cryptographically to credential epoch;
- no live quota discovery;
- no retries or circuit breaker;
- no signed/durable receipt ledger;
- no local/content-verified model adapter;
- no tool or structured-output wire execution;
- no claim that provider-attested model identity is independently verified;
- no claim that a successful response is true or safe to execute as an action.

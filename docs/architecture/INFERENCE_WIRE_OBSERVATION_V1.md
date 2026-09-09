# Symthaea Inference Fabric — IF-5 Wire Observation v1

Status: transport-observation child of IF-4.

## Purpose

IF-5 observes what an OpenAI-compatible endpoint actually reports without making
those provider claims authoritative. It adds a side-channel of bounded metadata
and quota hints beside the existing response-only transport API.

The legacy `generate()` and `generate_streaming()` APIs remain available and keep
their response/error shapes. New callers may opt into `generate_observed()` and
`generate_streaming_observed()`.

## Observed evidence

Successful responses may report:

- body response id;
- provider-declared model;
- system fingerprint;
- finish reason;
- provider-declared prompt/completion/total token usage;
- HTTP request id header;
- normalized request/token rate-limit limit, remaining, and reset durations;
- normalized `Retry-After` duration;
- observed transport latency.

All provider identifiers are bounded to 1024 bytes and rejected if they contain
control characters. Streaming metadata is accumulated conservatively; conflicting
claims set a conflict flag instead of silently replacing an earlier value.

## Error-body privacy invariant

Provider error bodies remain deliberately absent from `TransportError` and from
wire observations. A 429 may therefore preserve status, request id, Retry-After,
and rate-limit headers while discarding an error body that echoes the prompt.

Prompt, system prompt, and successful response text are not copied into
`TransportWireObservation`.

## Rate-limit normalization

IF-5 recognizes common OpenAI/Groq-style request/token headers and normalizes reset
values into durations from the observation point. It accepts bare seconds and
compound duration values such as `2m59.56s`.

Unknown/unparseable headers remain `None`; they are never guessed. These values are
provider observations, not quota authority. IF-6 will decide how observed quota
state may update a router/accounting state machine.

## Epistemic status

Provider response ids, model names, fingerprints, token usage, and rate-limit
headers are provider-observed claims. They do not independently prove which model
weights executed, correct token accounting, truthfulness, or provider compliance.

## Explicit non-claims

- no receipt integration yet;
- no provider-model mismatch rejection yet;
- no quota-state mutation;
- no circuit breaker or retry policy;
- no vendor preset;
- no provider-specific metadata opt-in;
- no raw provider error body retention;
- no current `LLMBackend` factory or routing change.

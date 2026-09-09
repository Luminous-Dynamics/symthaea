# Symthaea Inference Fabric — IF-6 Wire Evidence Receipt v1

Status: receipt-integration child of IF-5.

## Purpose

IF-6 connects IF-5's provider-observed wire metadata to IF-3/IF-4 terminal
inference receipts without turning provider claims into authority or creating a
long-lived raw provider-identifier ledger.

The execution chain becomes:

`bound request -> one-use permit -> prepared execution -> observed transport -> privacy projection -> terminal receipt`

## Provider identifier minimization

Raw provider request and response identifiers are useful transiently for support,
provider-console correlation, and incident investigation, so the immediate
`InferenceExecutionOutcome.wire_observation` may contain them.

Receipts do **not** retain those raw identifiers. `InferenceWireEvidence` stores
only domain-separated BLAKE3 digests under
`symthaea.inference.receipt.wire-id.v1` with distinct request-id and response-id
domains.

This permits explicit equality/correlation checks against an externally supplied
provider record while reducing the receipt ledger's value as a cross-system
tracking database.

## Receipt evidence

When safely observed, a terminal receipt may retain:

- provider response-id digest;
- provider request-id digest;
- provider-declared model string;
- provider system fingerprint;
- finish reason;
- provider-declared prompt/completion/total token usage;
- normalized request/token rate-limit observations;
- normalized Retry-After duration;
- observed transport latency;
- metadata conflict/rejection flags.

The successful response text remains represented only by its existing
BLAKE3 response digest.

## Failure evidence

Provider HTTP failures can carry privacy-safe wire evidence too. A 429 can produce
a failure receipt containing Retry-After and remaining-quota observations while
the provider error body remains discarded by IF-5.

Pre-network policy/binding failures have no provider wire evidence because no
provider interaction occurred.

## Debug/logging boundary

`InferenceExecutionOutcome` has a custom `Debug` implementation. It reports only
whether response text and wire observations are present, along with the receipt
and coarse failure class. It does not render generated response text or transient
raw provider identifiers.

This closes a logging gap that would otherwise undermine the receipt's deliberate
non-retention of model output.

## Epistemic boundary

Wire evidence remains provider-observed evidence. It does not independently prove:

- which model weights actually executed;
- that provider token accounting is correct;
- that a system fingerprint maps to specific weights;
- that rate-limit headers are globally complete;
- that provider privacy/retention promises were followed;
- that the model response is true or correct.

Receipt evidence is not quota authority and does not mutate routing state.

## Compatibility

The existing `InferenceReceipt::success()` and `InferenceReceipt::failure()`
constructors remain valid and produce receipts with no wire evidence. New observed
constructors are additive.

## Explicit non-claims

- no quota-state mutation;
- no router or circuit breaker;
- no provider-model mismatch enforcement from response metadata;
- no signed/durable receipt serialization;
- no cross-process receipt ledger;
- no provider-specific metadata API opt-in;
- no current `LLMBackend` factory or routing change;
- no claim that provider-observed metadata is independently verified truth.

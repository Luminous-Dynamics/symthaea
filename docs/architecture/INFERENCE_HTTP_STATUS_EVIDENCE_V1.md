# Symthaea Inference Fabric — Terminal HTTP Status Evidence v1

Status: evidence-only child of IF-13.

## Purpose

IF-7's resource guard intentionally distinguishes rate limiting, authentication
rejection, and generic transport/provider failure. Until this tranche, the transport
knew the exact failed HTTP status but the execution receipt collapsed all HTTP
failures into the coarse `ProviderHttp` class.

That was insufficient for a future execution-resource lease: settlement would have
to guess whether a provider returned 429, 401/403, or another status.

This tranche retains the **numeric terminal HTTP status as privacy-minimized evidence**
without retaining the provider response body.

## Evidence shape

`InferenceWireEvidence` gains a private:

`provider_http_status: Option<u16>`

The existing public constructor remains unchanged and initializes this field to
`None`. Only crate-internal executor projection can attach the status observed from
`TransportError::HttpStatus(u16)`.

A public read-only getter allows later settlement code to distinguish the terminal
condition represented by an already-produced receipt.

## Privacy invariant

Provider error bodies remain absent from:

- `TransportError`;
- `ObservedTransportFailure` evidence;
- `InferenceWireEvidence`;
- `InferenceReceipt`;
- ordinary `Debug` output.

The focused 429 regression deliberately returns an error body containing a sentinel
private string while also returning Retry-After and rate-limit headers. The receipt
retains only the status/header evidence and not the body.

## Failure distinctions preserved

A future settlement layer can now distinguish:

- HTTP 429 → rate-limited evidence;
- HTTP 401/403 → authentication/authorization rejection evidence;
- other HTTP status → provider HTTP failure evidence;
- request/connect/stream failure → no provider HTTP status.

The coarse `InferenceExecutionFailure::ProviderHttp` and
`InferenceFailureClass::ProviderHttp` remain unchanged for compatibility. The numeric
status is supplementary evidence rather than a new execution-authority enum.

## Explicit non-claims

This tranche does **not**:

- automatically settle or mutate `InferenceResourceGuard`;
- open or close circuit breakers;
- block credentials after 401/403;
- replenish or reduce local quota authority by itself;
- trust provider error bodies;
- retain success 2xx statuses;
- prove that a received HTTP status is truthful about provider-side account state;
- create an execution resource lease.

The next authority tranche can consume this evidence when settling a non-clone
pre-dispatch lease that already owns an IF-7 reservation and a verified IF-12 resource
scope.

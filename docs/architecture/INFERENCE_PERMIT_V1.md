# Symthaea Inference Fabric — IF-2 Execution Permit v1

Status: process-local authority-contract child of IF-1.

## Purpose

Route admission and external execution are different authority transitions. A
route can be valid when selected and invalid milliseconds later because provider,
credential, quota, privacy-policy, or request state changed.

IF-2 introduces a short-lived ownership-transfer boundary between those stages.

## Exact binding

Each permit binds non-zero digests of:

1. request;
2. admitted route;
3. inference policy;
4. provider state/profile;
5. credential state/epoch;
6. quota state.

The permit layer deliberately consumes digests instead of defining duplicate
copies of IF-0 semantic structures.

## Process-local capability semantics

`InferencePermit` is private-field, non-Clone, non-Copy, and non-Serde.
`PreparedInferenceExecution` has the same ownership property. Normal Rust code
must transfer ownership forward rather than duplicating the capability.

The issuer also keeps an active nonce/generation replay guard. A permit preparation
attempt removes that active entry *before* freshness or binding checks. Therefore
an expired, future-dated, or TOCTOU-raced permit cannot be retried after state
changes.

Validity is a half-open monotonic interval `[issued_at_tick, expires_at_tick)`.
The caller remains responsible for supplying a qualified monotonic/trusted time
source; IF-2 does not invent wall-clock authority.

## Intended integration

`IF-0 admission -> canonical digest binding -> IF-2 permit -> immediate TOCTOU recheck -> PreparedInferenceExecution -> IF-1 transport`

The eventual executor wrapper should own the transport and require
`PreparedInferenceExecution`, so callers cannot bypass preparation by directly
holding a production transport handle.

## Explicit non-claims

- no durable permit store;
- no cross-process replay protection;
- no distributed consensus;
- no cryptographic signature on permits;
- no secure time source;
- no canonical digest algorithm chosen yet;
- no runtime module export;
- no transport integration;
- no provider or credential implementation.

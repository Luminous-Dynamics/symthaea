# Symthaea Inference Fabric — IF-3 Canonical Binding + Receipt v1

Status: evidence-contract child of IF-2.

## Purpose

IF-0 defines admissible inference semantics. IF-2 binds six digests into a
short-lived execution permit. IF-3 defines exactly how those six digests are
constructed and how a consumed prepared execution becomes terminal evidence.

## Canonical digest theorem

Every binding digest is BLAKE3 over an explicit v1 domain plus explicit primitive
field encodings. IF-3 does **not** hash:

- Rust memory layout;
- `Debug` output;
- Serde JSON;
- bincode;
- pointer identity;
- map iteration order.

Strings and byte slices are length-prefixed. Integers are fixed-width little
endian. Options carry an explicit presence tag. Enums use fixed numeric tags.
Provider supported-purpose lists are canonicalized as sets by sorting and
deduplicating their tags before hashing.

The sensitive `InferenceRequest` remains non-Serde. Its prompt and optional system
prompt are fed directly into BLAKE3 and canonical prompt bytes are not exposed by
the public API.

## Six bound artifacts

`admit_and_bind()` performs admission itself and returns the admitted route plus:

1. request digest — prompt, system prompt, purpose, information class, token budget,
   and hard capability requirements;
2. route digest — selected provider/model/location/purpose/data class/admitted cost;
3. policy digest — the exact user/deployment privacy and monetary policy;
4. provider-state digest — the complete candidate capability/privacy/cost snapshot;
5. credential-state digest — opaque credential id + epoch, never secret material;
6. quota-state digest — opaque quota scope + epoch + known remaining/reset state.

Because `admit_and_bind()` creates the route internally, callers cannot bind a
route admitted for candidate A to candidate B's provider-state digest.

## Receipt boundary

`InferenceReceipt::success()` and `InferenceReceipt::failure()` consume
`PreparedInferenceExecution`. A success receipt stores only a domain-separated
BLAKE3 digest of response text; the raw response is not retained in the receipt.
A failure receipt records only a coarse failure class.

Receipts are evidence, not authority. They cannot be converted back into permits
or prepared executions.

Completion time must be greater than or equal to the preparation tick. As with
IF-2, ticks are caller-supplied monotonic/trusted-time observations; IF-3 does not
claim a secure clock.

## Why BLAKE3

The root Symthaea package already depends on BLAKE3 and uses it broadly for
content fingerprints. IF-3 makes the algorithm and the domain-separated v1 field
encoding explicit rather than inheriting an incidental serializer representation.

## Explicit non-claims

- no signed or durable receipt format yet;
- no cross-process receipt ledger;
- no secure time source;
- no credential store;
- no quota discovery;
- no provider profile registry;
- no transport execution wrapper yet;
- no runtime provider-selection change;
- no claim that a receipt proves model truthfulness or answer correctness.

The next tranche should place the generic transport behind a private executor that
accepts only a prepared execution, re-derives its transport request from the bound
semantic request, and always emits success/failure receipt evidence.

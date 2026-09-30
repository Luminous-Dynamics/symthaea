# HDC Associative Cleanup Task Harness

## Purpose

This is the third independent task family in the HDC dimension evidence chain. It tests **associative memory retrieval**, not direct prototype classification.

A memory stores key/value associations as a superposition of bound pairs:

`memory = bundle(bind(key_1, value_1), ..., bind(key_n, value_n))`

A query releases a value estimate with the queried key and then performs cleanup by nearest-codebook similarity:

`released = bind(query_key, memory)`

The experiment varies:

`dimension × memory load × query corruption → accuracy + mean decision margin`

This follows the established HDC distinction between binding, bundling/superposition, and cleanup/codebook retrieval. citeturn0search2turn0search1

## Important algebra boundary

The experiment deliberately converts deterministic random vectors to exact bipolar `{-1,+1}` components before binding. That is intentional: `ContinuousHV`'s ordinary real-valued binding is not self-inverse, so this harness does not silently assume a textbook bipolar-unbinding property that the continuous representation does not provide.

The harness therefore tests associative cleanup while making its algebraic assumption explicit and machine-checkable.

## Default matrix

- Dimensions: 1K through 256K.
- Memory loads: 2, 4, 8, 16 key/value pairs.
- Query corruption: 0%, 10%, 20%, 35%.
- Queries: 2 per stored key per cell.
- Every dimension/load/noise cell uses deterministic fixtures and a canonical experiment identity.

## What this can establish

- How retrieval accuracy changes as associative memory load increases.
- How cleanup margins change under query corruption.
- Whether those response surfaces differ across dimensions.
- Deterministic regressions in the associative-memory implementation.

## What this cannot establish

This is not a universal dimension recommendation, a production workload benchmark, or a proof that one associative-memory architecture dominates another. The task family remains separate from the direct-retrieval and noise-robustness families.

The literature emphasizes that superposition memory capacity is coupled to vector width and stored-item count, while cleanup/codebooks are part of the retrieval mechanism. citeturn0search2

## Next layer

Sequence/order retrieval should remain a fourth task family. The existing Symthaea sequence encoder already uses permutation before bundling specifically because plain binding is commutative and cannot preserve order. That makes it a natural next experiment: hold symbols fixed, pair deterministic sequences across dimensions, and measure position recovery and order discrimination separately.

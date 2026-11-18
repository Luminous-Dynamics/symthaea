# symthaea-algorithm-lab

Isolated proving ground for Symthaea's evidence-first algorithm discovery contracts.

## First target: exact BinaryHV Hamming distance

The v1 pilot searches a bounded, auditable implementation grammar over the existing 16,384-bit `BinaryHV`. Eight candidates cover:

- byte-wise XOR + popcount;
- 32-bit XOR + popcount;
- 64-bit XOR + popcount;
- 64-bit manually unrolled reduction;
- 64-bit reduction with four independent accumulators;
- 64-bit reduction with eight independent accumulators;
- 128-bit XOR + popcount;
- the current native Symthaea SIMD implementation.

This is the first parameterized search surface, not an exhaustive implementation grammar and not proof of autonomous algorithm invention. The width/reduction variants are explicit implementation hypotheses; the discovery system must still show that it selects useful candidates under independent measurements.

Before a candidate is benchmarked, it must match an intentionally simple bit-by-bit oracle on eight deterministic edge cases (including single-bit, high-bit, word-boundary and alternating-complement cases) and 128 seeded random pairs in both operand orders. Correctness evidence is opaque and self-validating.

Criterion compares every candidate across four deterministic input distributions:

- random balanced vectors;
- sparse two-bit differences;
- alternating complements;
- maximum-distance vectors.

No timing result is present yet. There is no claim that any candidate is faster than native SIMD, that the candidate set found a new algorithm, or that a winner generalizes across CPUs.
Run correctness tests with:

```bash
cargo test -p symthaea-algorithm-lab --lib -- --test-threads=1
```

Run the performance comparison with:

```bash
cargo bench -p symthaea-algorithm-lab --bench hdc_hamming_candidates
```

## Measurement boundary

Criterion output is a measurement, not a promotion decision. Do not copy a timing into the algorithm registry without also binding the exact source revision, toolchain, target/hardware profile, evaluator/oracle identities, and correctness evidence through `EvaluationReceipt`.

No unit test contains a fixed latency threshold. Timing varies by CPU, power state, compiler/codegen, scheduler noise, and build environment; pretending a single repository-wide nanosecond cutoff is portable would turn noise into policy.

The first benchmark is still a single-size microkernel test (16,384 bits). Before claiming algorithmic superiority, the next qualification stage must use multiple paired repetitions, randomized candidate order, separate warmup/measurement phases, and at least two target environments. Keep within-machine ranking separate from cross-machine performance; do not pool raw nanoseconds from different CPUs. Use the same fixed input corpus, baseline revision, compiler flags, Criterion settings, CPU governor/affinity policy, and background-load policy for all candidates in a comparison context.
## Authority boundary

This crate can construct descriptions, correctness evidence, discovery proposals, and evaluation receipts. It cannot modify tracked source, invoke Git, commit, merge, activate a candidate, or make a candidate production-eligible.

This pilot is ordinary deterministic computation. Cryptographic primitives, security-sensitive algorithms, and safety-critical controllers are intentionally outside the default automated discovery policy.

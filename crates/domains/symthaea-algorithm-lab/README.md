# symthaea-algorithm-lab

Isolated proving ground for Symthaea's evidence-first algorithm discovery contracts.

## First target: exact BinaryHV Hamming distance

The v1 pilot compares four deliberately small implementations over the existing 16,384-bit `BinaryHV`:

- byte-wise XOR + popcount;
- 64-bit XOR + popcount;
- four-word manually unrolled reduction;
- the current native Symthaea SIMD implementation.

Before a candidate is benchmarked, it must match an intentionally simple bit-by-bit oracle on fixed edge cases and deterministic seeded pairs in both operand orders. Correctness evidence is opaque and self-validating.

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

## Authority boundary

This crate can construct descriptions, correctness evidence, discovery proposals, and evaluation receipts. It cannot modify tracked source, invoke Git, commit, merge, activate a candidate, or make a candidate production-eligible.

This pilot is ordinary deterministic computation. Cryptographic primitives, security-sensitive algorithms, and safety-critical controllers are intentionally outside the default automated discovery policy.

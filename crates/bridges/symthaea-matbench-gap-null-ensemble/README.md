# symthaea-matbench-gap-null-ensemble

Deterministic pre-truth random-order null ensemble for the Matbench Benchmark Zero ladder.

V0 contains one deterministic random-order control. That is useful for pipeline parity, but one permutation can be accidentally easy or hard. V1 therefore freezes a finite ensemble of **256** random-order controls before truth is available.

This branch is intentionally not opened as a pull request while repository Actions admission remains saturated.

## No new seed choice

The ensemble does not introduce a second analyst-chosen seed.

Its master seed is exactly the `random_seed` already committed by the V0 `ComparisonFreezeReceipt`. Replicate seeds are deterministically derived as:

```text
SHA256(domain || master_seed_le_u64 || replicate_index_le_u64)
    -> first 8 digest bytes
    -> little-endian u64
```

for canonical replicate indices `0..256`.

All 256 derived seeds must be unique. A seed collision fails closed.

Different seeds are allowed to produce the same full ranking or same top-k selection. Rejecting such naturally occurring duplicates would condition the null ensemble after generation and would itself introduce selection bias.

## Compact pre-truth commitments

Storing 256 complete rankings over thousands of candidates would be unnecessarily large. Each replicate therefore freezes:

- canonical replicate index;
- derived seed;
- truth-free baseline screening-subject SHA-256;
- full ordered Benchmark Zero ranking digest;
- exact ordered top-k candidate IDs;
- top-k selection SHA-256.

`verify_null_ensemble_freeze` regenerates all 256 complete rankings from the truth-free exposure plan and requires the same compact commitments.

Every replicate must also:

- rank the full frozen candidate universe;
- have exactly the same candidate-id set as the V0 comparison;
- have a bit-identical candidate -> legacy-baseline prediction surface.

The null varies ordering only.

## Post-truth measurement

`measure_null_ensemble` does **not** call the baseline screening function.

It consumes:

- the V0 frozen comparison;
- the pre-truth null ensemble receipt;
- the already-restricted official truth receipt;
- the already-produced frozen-comparison measurement receipt.

Before using the observed heuristic/learned endpoint values, it independently calls the existing frozen-comparison measurement verifier against the same frozen rankings and restricted truth. It then evaluates each null replicate using only its pre-frozen top-k candidate IDs.

For every null replicate it records:

```text
top_k_hits
target_regret_ev
```

The complete 256-replicate vector is retained.

## Directional finite-ensemble references

For the legacy heuristic and composition-only learned policy separately, V1 records two exact add-one tail fractions:

```text
hits:
(1 + count(null_hits >= observed_hits)) / (256 + 1)

regret:
(1 + count(null_regret <= observed_regret)) / (256 + 1)
```

The numerator and denominator are stored alongside the floating-point value, and structural validation recomputes the extreme counts from the complete replicate vector.

These values are deliberately described as **finite-ensemble tail fractions**, not a generic significance certificate. They do not by themselves establish independence assumptions, multiple-comparison control, generalization, scientific superiority, or a publication-ready p-value theorem.

## Why MAE is excluded from the null reference

Random-order controls all preserve the exact same legacy predictor outputs. Reordering candidates therefore does not change mean absolute prediction error over the complete exact universe.

V1 explicitly records this boundary instead of fabricating a permutation distribution for an invariant endpoint.

## Authority boundary

A valid V1 receipt would establish that the heuristic and learned policy's ordering-dependent Benchmark Zero outcomes were compared against a deterministic 256-replicate random-order reference whose seeds and selections were fixed before truth.

It would not establish:

- a globally clean holdout;
- calibrated uncertainty;
- statistical significance in a broader inferential sense;
- scientific superiority;
- material novelty or feasibility;
- experiment authorization;
- candidate promotion.

Exact-head compile/test/rustfmt/strict-Clippy qualification remains required before using this branch as evidence.

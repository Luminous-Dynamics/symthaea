# ETH-027 — genuine PWMC semantic microbench v1

**Claim ceiling:** finite exact semantic fixture and exhaustive reference oracle. This is not a scalable compiler, SCPOG proof, formal checker, or solver-performance result.

## Contract

For each visible assignment v, support is true exactly when at least one hidden assignment satisfies the CNF. PWMC adds the exact visible literal-weight product once per supported visible assignment. It never multiplies by the number of hidden witnesses.

## Fixed cases

| Case | Full satisfying assignments | Projected support | Exact PWMC |
|---|---:|---:|---:|
| One hidden witness scope (F=d) | 2 | 1 | 2/3 |
| Twenty free hidden variables (F=d) | 1,048,576 | 1 | 2/3 |
| Asymmetric witness multiplicity (F=d OR h) | 3 | 2 | 1 |
| Free visible scope completion | 4 | 2 | 2/3 |
| Nonempty support with exact cancellation | 4 | 2 | 0 |
| Signed fractional weights | 2 | 2 | 1/6 |
| Empty projected support | 0 | 0 | 0 |

The second case has 20 hidden Boolean variables. All 1,048,576 hidden assignments witness the same single visible assignment. The projected support and PWMC are identical to the one-hidden-variable case.

## Different producer and verifier algorithms

The retained-support producer enumerates visible assignments and performs existential hidden search, stopping at the first satisfying hidden witness for each visible assignment.

The independent verifier enumerates the complete variable universe, collects satisfying assignments, and only then groups them by visible assignment. This is intentionally a different traversal order and projection procedure. Both are exhaustive on this micro-suite; the second is not a scalable verifier.

## Support reuse

The reusable fixture has variables (d,u,h), visible variables (d,u), hidden variable h, and CNF clause (d OR u). Its projected support is {01, 10, 11} and each supported visible assignment has two hidden witnesses.

One existential support compilation is evaluated under 32 distinct signed-rational visible weight maps. For every map, an independent full-assignment traversal reconstructs support and replays the exact rational result. All 32 results match.

The evaluator computes:

w_d(false)*w_u(true) + w_d(true)*w_u(false) + w_d(true)*w_u(true)

for this fixed support. The benchmark deliberately does not claim that cached evaluation is faster than replay; it demonstrates semantic reuse and records the separation needed for future performance measurements.

## Reproduction

From this directory, run:

python3 run_pwmc_semantic_microbench.py

The runner writes pwmc_semantic_microbench_results.v1.json. It records the input SHA-256, exact values, producer/verifier descriptions and the explicit claim ceiling.

Input SHA-256: c26471c634c9b956d7f229695f5fc3cceb4662501a6a73cba01d3916421d1707.

## Limitations

- The cases are small fixed DIMACS CNFs, not a curated competition benchmark.
- Both the producer and verifier are exhaustive exponential procedures.
- The support producer is not an SCPOG/d-DNNF compiler.
- No proof artifact is generated or checked.
- The result is a semantic regression suite, not a runtime/SOTA comparison.
- The 32 maps are a reuse correctness test, not evidence of a 32x speedup.

Next steps: add generator provenance and structural families; import the public 2024 projected benchmark fixtures; run the SCPOG artifact; then add Rust producer/replay implementations and differential checks.

# ETH-020: exact Gaussian-rational algebraic model counting

**Status:** Finite Python Q(i) reference arithmetic and bucket variable-elimination oracle committed. Ganak already provides exact complex-rational support, so Q(i) alone is not a novelty claim; Rust/Lean qualification, proof-carrying execution and competitive evaluation remain open.

## 1. Why this matters now

The MC Competition 2026 added:
- weighted model counting with negative weights;
- projected model counting;
- projected weighted model counting (mixed WMC/PMC/PWMC);
- algebraic model counting over a complex field (bonus/experimental track).

The exact ranking tier uses arbitrary precision. The official submission constraints specify a 3,600-second timeout and 32 GB memory limit:
- https://mccompetition.org/2026/mc_description.html
- https://mccompetition.org/2026/submission

The public Ganak 2026 entry reports 49/100 exact MPQ results on the complex-weighted track, and 120/200 exact WMC, 125/200 exact PMC and 135/200 exact PWMC. These are the Ganak entry's reported totals, not a global SOTA claim:
- https://github.com/meelgroup/mccomp-ganak-appmc-2026

The Ganak report explicitly distinguishes exact MPQ arithmetic from MPFR high-precision floating point: cancellation to exact zero can be incorrectly represented as a small nonzero number by floating point. This is why ETH-020 treats exact coefficient representation as a first-class capability, not a numeric formatting detail.

## 2. Initial exact coefficient domain: Q(i)

Start with Gaussian rationals:

\[
\mathbb{Q}(i)=\{a+bi\mid a,b\in\mathbb{Q},\ i^2=-1\}.
\]

This is an exact commutative field and a commutative Q-algebra. The representation stores two reduced rational numbers. It supports exact addition, subtraction, multiplication and conjugation.

Canonical wire representation:

    [numerator_real, denominator_real, numerator_imag, denominator_imag]

Both denominators must be positive. Each rational must be reduced by gcd. Zero has the unique encoding [0,1]. Booleans are rejected as integer fields. No float enters the arithmetic path.

**Scope boundary:** Q(i) is a strict subfield of C. It covers inputs whose real and imaginary parts are rational, not arbitrary computable complex numbers or all algebraic numbers. A future extension to algebraic number fields needs minimal polynomials, root/isolation identity and exact reduction rules; it must not be approximated by floating point and labelled exact.

## 3. Factor-network semantics

For finite-domain variables and local complex-rational factors:

\[
Z=\sum_{x_1,\ldots,x_n}\prod_j f_j(x_{\operatorname{scope}(f_j)}).
\]

The reference compares direct enumeration with bucket variable elimination over Q(i). It explicitly adds a unit unary factor for each variable absent from all factor scopes, so unconstrained variables contribute their domain multiplicity rather than silently disappearing. Scalar zero-scope factors are preserved.

Exact cancellation fixtures verify both real and imaginary cancellation to Z=0.

## 4. Executed reference experiment

Files:
- reference/gaussian_rational_amc.py
- reference/gaussian_rational_amc_results.json

Executed locally using Python standard-library Fraction arithmetic:
- 1,200 deterministic Gaussian-rational triples;
- 10,800 exact ring / conjugation / norm property checks;
- 180 random finite factor networks;
- 720 direct-summation versus variable-elimination checks, with four orders per network (forward, reverse, degree-like, random);
- 2 exact-zero cancellation fixtures;
- 5 malformed rational-encoding rejection controls.

These are finite reference tests, not a complexity or runtime result. The script is a small exact oracle, not a production WMC parser or competition solver.

## 5. Baseline correction: Q(i) is established capability

Ganak already supports exact complex-rational weights (MPQ) and broader algebraic model-counting modes, including polynomial/Laurent-polynomial and custom-field capability in its public code/documentation. Therefore, implementing Q(i) in isolation is not a new capability relative to the strongest relevant baseline.

ETH-020 must be positioned as a qualification and integration exercise:
- verify that Symthaea's exact algebra registry can represent and reason about Q(i);
- bind exact operations to field/Q-algebra laws and rule permissions;
- generate replayable factor-join and sum-out certificates;
- select plans using the coefficient growth and independent-checking cost;
- compare verified end-to-end execution against Ganak's exact MPQ mode on the same instances.

Ganak is a strong and necessary baseline, not a straw man. If it outperforms Symthaea, report that directly. The research value is established only if the certificate and planner integration gives a measurable advantage, or provides an independently valuable verification guarantee, without claiming arithmetic novelty.

Source: Ganak's 2026 MC competition submission and implementation:
- https://github.com/meelgroup/mccomp-ganak-appmc-2026
- https://github.com/meelgroup/ganak

## 6. What is needed to compete

1. **Rust backend:** arbitrary-precision numerator/denominator components and exact Gaussian-rational operations. Prefer an audited arbitrary-precision library; never use f64 for values or equality.
2. **Lean domain:** define Q(i) and prove commutative field / Q-algebra laws. The runtime registry may only expose the corresponding capabilities after its representation and operations are bound to these proofs.
3. **Proof-carrying VE:** each factor join and sum-out operation gets a canonical trace; an independent checker replays exact operations.
4. **MC parsers:** implement the exact Track 5B format separately from Boolean #SAT, ordinary WMC and projected counting. Validate headers and output semantics rather than trusting a file's advertised track.
5. **Benchmark protocol:** freeze the public suite, keep a family-held-out validation set, and use the same 3,600s / 32GB budget. Record exact solved count, producer time, proof generation, checker time/RSS, peak live factor size, coefficient bit lengths and zero-cancellation cases.
6. **Beyond Q(i):** inspect the actual benchmark coefficient sets. Add algebraic number fields only if needed, with exact root identity and canonical arithmetic.

## 7. Compatible baselines and targets

The direct baseline for the complex AMC track is Ganak's exact MPQ path, together with any other submitted solver that actually supports the same instance format and exact result contract. Do not use the approximate MPFR score as an exact baseline.

A first milestone is exact agreement on the public rational-complex instances and a complete, mutation-tested certificate checker. A later target is at least 10% more exact instances solved or 10% lower verified end-to-end time on a preregistered held-out subset under identical budgets. Both are aspirations, not observed results.

## 8. Relationship to the other ETH lines

- ETH-015 selects the requested algebra.
- ETH-016 certifies exact elimination.
- ETH-017 searches elimination orders with cost/certificate awareness.
- ETH-018 verifies algebra-domain-qualified rewrites.
- ETH-019 searches structural decompositions.
- ETH-020 adds an exact rational-complex coefficient domain to exercise the same architecture on the 2026 complex AMC track.

The key comparison is verified end-to-end computation, not arithmetic kernel speed in isolation.

## 9. Claim ceiling

This prototype demonstrates a Symthaea-side exact Q(i) reference and small factor-network VE equivalence; it does not establish a capability absent from Ganak. It does not establish full complex-field AMC coverage, projected counting, algebraic-number fields, Rust implementation, Lean proofs, competition performance, or better-than-SOTA results.

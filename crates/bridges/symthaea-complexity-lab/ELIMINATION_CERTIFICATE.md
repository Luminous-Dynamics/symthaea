# ETH-016: proof-carrying exact factor elimination

**Status:** Reference certificate format tested in Python; Rust integration and Lean proof remain open.

## Research positioning

Certified model counting is established prior art. Relevant systems include:

- Fichte, Hecher & Roland, *Proofs for Propositional Model Counting* (SAT 2022): proof traces for dynamic programming, counting-CDCL/component caching and Decision-DNNF counting. https://doi.org/10.4230/LIPIcs.SAT.2022.30
- Capelli, Lagniez & Marquis, *Certifying Top-Down Decision-DNNF Compilers* (AAAI 2021). https://doi.org/10.1609/aaai.v35i7.16776
- Bryant et al., *Certified Knowledge Compilation with Application to Formally Verified Model Counting* (JAIR 2025), POG/CPOG and a formally verified weighted/unweighted checker. https://arxiv.org/abs/2501.12906
- *Certifying Projected Knowledge Compilation* (SAT 2025). https://doi.org/10.4230/LIPIcs.SAT.2025.8
- Beyersdorff, Hoffmann & Kasche, ECCC TR25-127 (2025), on MICE and annotated Decision-DNNF proof systems. https://eccc.weizmann.ac.il/report/2025/127/

The target is not to claim novelty for certified #SAT. The research hypothesis is an independently checkable execution interface spanning several exact output algebras: Boolean feasibility, natural counts, optimum-plus-multiplicity (MaxCount), Pareto frontiers, and bounded polynomial profiles. We must reuse or interoperate with MICE / POG-CPOG on the overlapping Boolean and counting subset where possible.

## Certificate format v1

Identifier: `symthaea.exact-elimination-certificate/v1`.

A canonical instance binds variable cardinalities, algebra kind and parameters, ordered input-factor IDs, sorted scopes, and exact factor tables. A trace binds the instance digest, algebra, elimination order, ordered bucket-factor IDs and digests, every generated output scope/table/digest, final scalar-factor digests, result, and deterministic work counters.

For a variable (x) with bucket (B_x), the checker recomputes:

[
g_x(\mathbf y)=\bigoplus_{a\in D_x}\bigotimes_{f\in B_x}f(\mathbf y,a).
]

It rejects a false output table even when an attacker recomputes its digest. Hashes bind bytes; exact replay establishes correctness.

## Soundness boundary

For a finite factor system over a commutative semiring, each checked bucket step preserves the global product-of-factors summed over eliminated variables. An induction over the elimination order yields the final scalar result.

This argument still requires separate proofs that each supported algebra really satisfies its laws, that its encoding is faithful, and that the executable checker refines the mathematical checker. A caller-provided capability flag is not a proof.

The first trace is intentionally not succinct: it stores every intermediate factor table, and verification repeats most of the elimination arithmetic. This is an assurance boundary, not a speedup.

## Executed reference evidence

The local reproducibility package includes:
- `src/elimination_certificate.py`: producer, canonical input encoder, direct-enumeration oracle and mutation suite;
- `src/elimination_certificate_checker.py`: separately implemented checker with a standalone CLI;
- `src/elimination_certificate_results.json`: deterministic results.

Results: 7/7 positive examples, 96/96 randomized small count factor graphs (1–5 variables), 5/5 separate-process checker runs across all five algebra kinds, and 13/13 mutated/stale certificates rejected. The examples cover triangle independent-set count 4, triangle MaxCut optimum/count (2,6), full profile (2+6W^2), a two-objective Pareto frontier, Boolean feasibility, unconstrained variables and scalar-only factors. Random seed: `20261103`.

These are small Python reference checks only: no Rust execution, Lean proof, arbitrary user-defined algebra, succinct proof or scalable certificate result.

## Next work

1. Formalize algebra laws and the bucket-step/global-invariant theorem in Lean.
2. Add the producer and checker to ETH-015's Rust module; the checker must not import producer code.
3. Add exact encoding tests and mutation tests for every field, factor, scope, coefficient and algebra parameter.
4. Benchmark certificate bytes, producer/checker CPU and wall time, peak RSS, output-table entries, peak live table size and digest overhead.
5. Compare against MICE and POG/CPOG only on matched supported semantics; separate projected counting, MaxCount, Pareto and full profiles.
6. Evaluate streaming/parallel verification and sparse table encodings. Never replace table checking with digest-only acceptance.

Aspirational gate: zero accepted invalid certificates, 100% agreement with direct oracles on tractable cases, Lean soundness, and median producer-plus-checker overhead below 2x on a preregistered low-width slice. This is a target, not an observed result.

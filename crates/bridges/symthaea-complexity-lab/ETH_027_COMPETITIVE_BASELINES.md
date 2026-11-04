# ETH-027 — competitive baselines and benchmark plan

**Evidence snapshot:** 2026-11-03. These are source-defined baselines, not Symthaea measurements.

## MC 2026 official exact ranking

| Track | D4 | Ganak | Result |
|---|---:|---:|---|
| #SAT | 63 | 62 | D4 +1 |
| WMC | 60 | 59 | D4 +1 |
| PMC | 52 | 61 | Ganak +9 |
| Mixed Track 4 | 67 | 69 | Ganak +2 |
| Complex AMC | 49 | 44 | D4 +5 |

**Critical qualification:** the official Track 4 footnote reports MC, PMC and WMC instances, with zero genuine PWMC instances received. Do not cite Track 4 as a PWMC result. Official ranking counts and the Ganak entrant README's public/private submission totals are separate datasets with different denominators and must remain separate.

MC 2026 exact ranking used arbitrary precision, a 3,600-second per-instance limit and 32 GB RAM. Wrong exact answers are disqualifying for the exact ranking.

## Portfolio baselines

| System | Structure worth exploiting | Symthaea strategy |
|---|---|---|
| Ganak | Component caching, branching/phase selection | Preserve a cache-based backend; key caches by semantic contract, partition, weights and source identity |
| D4 | Top-down search, dynamic decomposition, hypergraph partitioning | Integrate as a solver and projected-compilation baseline |
| DPMC | Project-join trees, tree-decomposition planning, ADD/tensor execution | Add a project-join portfolio and track width/largest intermediate |
| SCPOG | Certified projected Boolean support | Reproduce and integrate as Layer A |
| CPOG | Proof-system interoperability and proof-complexity results | Treat as a distinct proof profile; theoretical proof-size simulation is not checker-speed evidence |

Do not try to replace all competitors with one algorithm. Choose a representation and backend using structural features and measured verified cost. Record end-to-end wall time separately from per-stage time because stages can overlap.

## Genuine PWMC suite

Publish a separate suite rather than inheriting MC 2026 Track 4's label. Hold out entire structural families, not just random seeds.

1. Hidden-witness multiplicity: one visible assignment with many hidden witnesses; add unconstrained hidden variables as an invariance test.
2. Projection ratio: nearly all-visible through heavily projected; include free variables and empty support.
3. Signed arithmetic: negative, zero, fractional and cancellation-producing rational weights.
4. Complex arithmetic: Gaussian-rational weights and independent real/imaginary cancellation.
5. Structure: sparse/dense CNF, disconnected components, high/low treewidth, nested separators and variable-order stress.
6. Transformations: definitional extensions, variable elimination, subsumption, component decomposition and re-encoding, each with its own preservation obligation.
7. Reuse: evaluate one certified support circuit under 1, 10 and 100 different visible weight maps without repeating projection.
8. Adversarial checks: hidden-weight leakage, witness addition, missing scope completion, false determinism, stale digest, malformed rational and unsupported coercion.

Each fixture binds source bytes, parser profile, visible/hidden map, contract, weights, known support/value when feasible, generator family and provenance.

## Mandatory measurements

Report solve, compile, proof generation, proof size, checker, arithmetic and end-to-end wall time; per-stage and total peak RSS; exact and independently verified instance counts; support cardinality; exact output and cancellation status; circuit nodes/edges/scope/width; proof-to-circuit size; checker failure/timeout; coefficient bit-height; serialization bytes; and marginal cost per reused weight map.

Primary result: exact instances independently verified under matched hardware and limits. Also report raw solved-but-unverified cases separately. False acceptance is a hard zero target.

## Aspirational targets — not achieved

| Metric | Target after reproducing prior work |
|---|---|
| Correctness | 100% agreement on supported exhaustive subset; zero false accepts |
| SCPOG reproduction | Reproduce published artifact success/failure and cost split first |
| Certification overhead | Median <=1.5x modified compile time; p95 <=5x on matched preregistered family |
| Proof size | Median Skolem-POG/circuit ratio <=10x versus published 34x median, on a matched subset |
| Independent checker memory | <=4 GB on public projected fixtures; competition ceiling remains 32 GB |
| Support reuse | 100 distinct weight maps from one certified projection, with independent exact replay |
| Competitive advantage | >=10% lower verified end-to-end time or >=10% more verified cases on held-out structural families |

These are falsifiable research goals, not achieved results. The SCPOG paper's 2024 benchmark environment and MC 2026 competition environment differ; do not compare raw timings without matched hardware and resource limits.

## Primary sources

- MC 2026 official description and rules: https://mccompetition.org/2026/mc_description
- MC 2026 official awards/results: https://mccompetition.org/assets/files/2026/MC2026_awards.pdf
- Ganak MC 2026 entrant report: https://github.com/meelgroup/mccomp-ganak-appmc-2026
- Bryant, Tan and Heule, Certifying Projected Knowledge Compilation, SAT 2025: https://doi.org/10.4230/LIPIcs.SAT.2025.8
- SCPOG supplementary software: https://zenodo.org/records/15628075
- DPMC project-join trees: https://arxiv.org/abs/2008.08748
- Relative strength of #SAT proof systems, 2026: https://doi.org/10.1007/s10817-026-09757-w

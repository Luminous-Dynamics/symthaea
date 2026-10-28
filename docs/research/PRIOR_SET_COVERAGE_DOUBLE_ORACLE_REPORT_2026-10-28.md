# Prior-set coverage and policy/world double oracle — 2026-10-28

**Status:** standalone exact synthetic finite-game experiment; 11/11 local tests pass; this report does not claim integration into Symthaea's production SCM evaluator or an external benchmark result.

## Executive result

A robust optimizer can be exact over hand-picked true-prior scenarios while still understating the value of robust decision-making over the full hypothesis simplex.

The synthetic game has 16 latent hypotheses (4 graph classes × 4 mechanism/kernel classes), two actions, binary observations, horizon two, and all eight deterministic complete policy trees. For graph, kernel and joint terminal categorical cross-entropy, expected policy risk is calculated at each of the 16 point-mass true priors. Under the declared shared-likelihood/strictly-positive-learner-prior assumptions, fixed-policy expected terminal categorical cross-entropy is linear in the true prior; thus the full simplex's worst-case point is one of its 16 vertices.

### Results

| Terminal objective | Six-prior game value (bits) | Full-simplex value (bits) | Value understatement (bits) | Six-prior solution's full-simplex regret (bits) | Double-oracle iterations | Outward-expanded interval width (bits) |
|---|---:|---:|---:|---:|---:|---:|
| Graph | 1.474374 | 1.815263 | 0.340889 | ~0 | 2 | 2.00e-9 |
| Kernel | 1.528545 | 1.853951 | 0.325406 | ~0 | 5 | 2.0003e-9 |
| Joint | 3.389536 | 3.794190 | 0.404654 | 0.081206 | 4 | 2.00e-9 |

Two quantities must be distinguished:

- **Game-value understatement:** the six-prior minimax value is 0.325–0.405 bits below the full-simplex optimum. The six-point scenario set therefore does not certify the full-simplex robust value.
- **Selected-policy regret:** the six-prior mixture happens to remain optimal for graph and kernel in this tiny game, but incurs 0.081206 bits of joint-objective regret. Value coverage and policy quality are related but distinct.

The prior/world double oracle began with the six legacy priors and one policy. It alternated a policy best response to the dual-induced prior mixture with the worst point-mass world for the current policy mixture. It converged in 2, 5 and 4 iterations. The maximum raw difference from the direct full-simplex LP was 3.2085445411667024e-13 bits.

The interval endpoints use an explicit 1e-9 bit outward margin on each side, giving a maximum interval width of about 2.000321e-9 bits. This is a numerical guard, **not a formal interval-arithmetic error proof**.

## Method

1. Construct a strictly positive action-dependent Bernoulli observation model over 16 graph/kernel hypotheses.
2. Enumerate all eight complete deterministic horizon-two policy trees.
3. Compute expected terminal cross-entropy for graph, kernel and joint class maps at each point-mass prior, producing a policy-by-world risk matrix.
4. Independently evaluate the six named mixture priors and check the linearity identity against the risk matrix.
5. Solve separate minimax matrix games for six selected priors and all 16 simplex vertices.
6. Run the double oracle; independently re-score its final mixture against every world and its dual-induced prior against every policy.
7. Stop only when the outward-expanded upper/lower value interval is within tolerance; fail closed on LP, input validation or certificate inconsistency.

## Regression evidence

Local command: python -m unittest -v from the standalone package directory.

Result: **11/11 tests passed** on Python 3.13.5 with NumPy 2.3.5 and SciPy 1.17.0. Tests include 300 seeded prior-linearity games, 100 convex-mixture checks, exact full-game/double-oracle agreement, vertex checks, objective coverage and invalid-input rejection.

The standalone benchmark source SHA-256 is d7ceb9b79ee06869e7835cfaec14f67a0bd59a017fc0f85fec972572dc0b8bb3; regression-test SHA-256 is 1e94d7884c315b93520cff5b8daa67918e085039e3ae30dc289467c9c4bb19e8; result JSON SHA-256 is b5aed02a4581054676f2602eef6f6577ec075f452e6987cd2895c757eddd2435. The standalone benchmark source, risk-linearity helper, and both regression-test modules are now committed on the research branch; their SHA-256 values and Git blob IDs are recorded in the companion summary. The exact 22,070-byte result JSON remains a local research-package artifact (SHA-256 `b5aed02a4581054676f2602eef6f6577ec075f452e6987cd2895c757eddd2435`) and is not represented as a committed repository artifact. The benchmark still uses a synthetic likelihood rather than Symthaea's repository SCM implementation.

## Efficiency boundary

The game is deliberately small (8 policies × 16 worlds). In the local environment, a direct full-simplex LP took roughly 2–3 ms per objective, while the double-oracle loop took roughly 5–12 ms. This is not a speedup claim. The result establishes a scenario-coverage diagnostic and a small-game numerical cross-check, not computational superiority.

## Scientific boundary

- The likelihood is a synthetic fixture, not Symthaea's repository SCM implementation.
- The full-simplex guarantee only covers prior-weight uncertainty over the fixed 16-hypothesis support under the declared shared likelihood. It cannot cover omitted mechanisms/graphs, misspecified likelihoods, latent confounders outside the support, or incorrect intervention semantics.
- Eight policies and horizon two do not support a scaling claim.
- No ACDB, CausalArena, TCD-Arena or industry baseline was run. Cross-entropy is not directly comparable with structural F1/SHD.
- No novelty, SOTA, production-readiness or general speedup claim is made.

## Connection to related work

- Li, Ju and Shroff, How to Find the Exact Pareto Front for Multi-Objective MDPs? (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/f8290ccc2905538be1a7f7914ccef629-Abstract-Conference.html. Its polytope/neighbor structure is relevant inspiration, but does not automatically transfer to this partially observed causal-probing problem.
- Active Causal Discovery Benchmark (ACDB): https://arxiv.org/abs/2609.31675. It separates skeleton recovery, directed-DAG recovery and intervention efficiency.
- CausalArena: https://arxiv.org/abs/2609.11897. Its varied SCM and real-data regimes motivate holding out entire graph/mechanism families.
- TCD-Arena: https://tcd-arena.github.io/. Its assumption-violation settings can stress-test beyond clean synthetic observation models.

## Next work and acceptance gates

1. Integrate this experiment against an exact Symthaea branch checkout and use repository likelihood/action semantics, rather than treating the toy result as integration proof.
2. Evaluate the existing six-prior-selected policy against all relevant worlds under identical risk semantics.
3. Add constrained ambiguity polytopes (probability floors, graph-marginal ranges, mechanism-family mass constraints and graph–kernel dependence constraints); verify LP worst cases independently whenever enumeration is feasible.
4. Compare exhaustive frontier, epsilon pruning with a proved bound, column generation and prior/world double oracle at equal tolerances across horizons 1–5 and 6/12/24/48 candidate atoms. Report p50/p95 latency, peak memory, state/oracle counts, value gap and policy support.
5. Expand to held-out 3–8 node DAG/mechanism families with noisy/missing/delayed telemetry, action costs, off-target interventions and task disturbance.
6. Run matched ACDB/CausalArena/TCD-Arena adapters and report skeleton F1, directed precision/recall/F1, SHD, posterior calibration/Brier/log loss, intervention efficiency and per-family uncertainty.
7. Treat NOT_CERTIFIED, missing numerical bounds and skipped validation as failures to qualify.

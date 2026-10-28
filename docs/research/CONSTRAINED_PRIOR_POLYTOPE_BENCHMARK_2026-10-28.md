# Constrained prior-polytope robust causal probing — 2026-10-28

**Status:** standalone synthetic finite-game pilot; local combined regression suite 17/17 passed. The likelihood fixture is not Symthaea's repository SCM implementation. The committed benchmark/test source versions have not yet been re-executed from an exact branch checkout; exact-head qualification remains open.

## Research question

Can Symthaea replace a hand-picked list of true-prior scenarios with an explicit ambiguity polytope encoding partial prior knowledge, then select a policy that minimizes worst-case expected terminal cross-entropy over that set?

This lane extends the prior-set coverage/double-oracle pilot. It is not a claim that ambiguity-set optimization or distributionally robust optimization is new. Bayesian ambiguity sets and decision/loss-aware ambiguity-set design are established directions; the experiment asks how these ideas behave in adaptive causal-probing policy selection.

## Model and protocol

- 16 fixed latent hypotheses: four graph classes × four mechanism/kernel classes.
- Eight complete deterministic policy trees, two binary-observation steps.
- Three terminal objectives: graph, kernel, and joint categorical cross-entropy.
- Shared synthetic, action-dependent Bernoulli likelihood with positive support and a uniform strictly-positive learner prior.
- Each risk vector is evaluated across the hypothesis atoms; prior-weight risk is linear for a fixed policy under this fixture, so worst-case risk over a declared convex prior polytope can be found with an LP.
- The robust game is solved as a finite zero-sum game over policy mixtures and the constrained prior set. An independent LP maximizes the selected policy mixture's risk over the same set, giving a lower/upper value interval.
- Outward margin is 1e-9 bits per side. This is a numerical guard, **not a formal interval-arithmetic error proof**.

Two explicitly declared polytopes were compared:
1. **bounded_marginals:** atom floor 0.005; each graph and kernel marginal in [0.10, 0.40]; graph-kernel diagonal and antidiagonal mass each in [0.10, 0.60].
2. **contains_all_legacy_priors:** same atom floor and association-mass limits, but every graph/kernel marginal in [0.05, 0.60]. The implementation asserts all six legacy priors are feasible in this set.

These are illustrative constraints, not empirically calibrated uncertainty regions.

## Results

All values are in bits of expected terminal cross-entropy.

| Ambiguity set | Objective | Constrained minimax value | Six-prior policy's worst-case risk over set | Six-prior policy regret |
|---|---|---:|---:|---:|
| bounded_marginals | Graph | 1.690035 | 1.690035 | 0.000000 |
| bounded_marginals | Kernel | 1.732918 | 1.732918 | ~0 |
| bounded_marginals | Joint | 3.670224 | 3.686422 | 0.016197 |
| contains_all_legacy_priors | Graph | 1.748079 | 1.748079 | 0.000000 |
| contains_all_legacy_priors | Kernel | 1.789665 | 1.789665 | ~0 |
| contains_all_legacy_priors | Joint | 3.727372 | 3.775266 | 0.047894 |

The six original priors are all inside the broader ambiguity set; only three of six are inside the narrower set. The broad set's robust value is higher than the narrow set's, as expected for nested uncertainty sets. The old scenario-optimized policy is still optimal for graph and kernel in these tiny examples but loses on joint prediction. The joint regret increases from 0.016197 bits for the narrower set to 0.047894 bits for the broader set.

For context, the full-simplex optimum from the previous pilot was 1.815263 (graph), 1.853951 (kernel), and 3.794190 bits (joint). These are on the same fixed support and synthetic likelihood; they should not be interpreted as real-world causal-recovery scores.

## Local verification and provenance boundary

- Local test suite: **17/17 passed** (Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0).
- Local constrained benchmark SHA-256: `0744255d0c42760d795a3d8639b8c3c6b001af649ee88e860301065c896c8400`.
- Local constrained regression SHA-256: `418ecbb4b62b9faa199228a1cee6cc2da0a454dc7d856755f865deb5125c7688`.
- Raw result artifact: 14,333 bytes, SHA-256 `8923e1a7b85d26be97fbfd790686972e8989074af2a691820ed3657255ac1da8`.
- Lossless `gzip -n` artifact: 2,453 bytes, SHA-256 `267eeede233f4042189f4e481b79b617fbdf7db72ed5c6f2a423952f4c59f4c2`, committed as `validation/CONSTRAINED_PRIOR_POLYTOPE_RESULTS_2026-10-28.json.gz` (Git blob `374f5b1217de29d7869f1ee65ddbe390606afe3c`).

Important: the local versions used for the 17/17 run are not currently byte-identical to the committed branch copies. Their local Git blob hashes are `8c68d2b8d65ef6586dd1234b293176157c17fb3a` and `ac5cd9c3ccc33a46692aba61ebbc7ba55cb8fff4`; the current branch blobs are `4d6dd29fbdf09de10cdfe7cba0c403a7d5ba640b` and `9d3723d9cbc0c20de182db4de8b8f71a3cfdb4f3`. Therefore, the local test result is evidence about the local source/test pair, **not exact-head execution evidence for the branch files**. This mismatch is a release-blocking provenance item, not a reason to claim green remote CI.

## Interpretation

This experiment supports three limited conclusions:

1. A constrained ambiguity set can encode prior information more precisely than either six unrelated scenarios or the full simplex.
2. Robust policy quality depends on the objective: in this fixture graph and kernel robust policies are unchanged, while joint prediction benefits from the constrained-polytope solution.
3. Increasing the uncertainty set increases the joint objective's regret for the six-prior-selected policy from 0.016197 to 0.047894 bits.

It does not demonstrate a speedup: all LPs here are tiny and wall-clock timing is sub-millisecond/millisecond scale. It does not show generalization to unrepresented graphs or mechanisms. No ACDB, CausalArena, or TCD-Arena comparison was executed.

## Additional experiment: ambiguity-width sweep

After the initial two-set comparison, I evaluated five nested graph/kernel marginal-bound widths while keeping the per-atom floor and diagonal/antidiagonal association bounds fixed. All three objectives were solved for each set (15 rows total). This is a separate local sweep and inherits the same synthetic-model and source-provenance limitations noted above.

| Graph/kernel marginal bounds | Graph value (bits) | Kernel value (bits) | Joint value (bits) | Old six-prior policy joint regret (bits) |
|---|---:|---:|---:|---:|
| [0.24, 0.26] | 1.520647 | 1.568321 | 3.438245 | ~0 |
| [0.20, 0.30] | 1.570999 | 1.616981 | 3.509153 | ~0 |
| [0.15, 0.35] | 1.631122 | 1.675454 | 3.597787 | ~0 |
| [0.10, 0.40] | 1.690035 | 1.732918 | 3.670224 | 0.016197 |
| [0.05, 0.60] | 1.748079 | 1.789665 | 3.727372 | 0.047894 |

The robust value rises monotonically as the admissible marginal range widens. In this fixture, the six-prior policy remains optimal for the graph and kernel objectives across all five widths, but becomes suboptimal for joint prediction at the two widest sets. This is a useful *sensitivity curve* for robustness versus conservatism; it is not a calibration curve because these bounds were chosen illustratively rather than estimated from data.

The exact 15-row local output is losslessly committed as `validation/CONSTRAINED_PRIOR_WIDTH_SWEEP_2026-10-28.json.gz` (compressed SHA-256 `98e4bfa4d824ec7c73bc0743f5a90302045b32b003fc91863c630a44f1e7b318`; Git blob `02b16e9103a59d5544cc68d12d79f8909dbe296e`). Its raw local JSON is 15 rows, 5,xxx bytes, SHA-256 `58e85876088d62fb9393551df359b995c58af7ec1d51f7551ba82b4b10ebd2d4`.

## Related work and benchmark implications

- Dellaporta, O'Hara and Damoulas, **Decision Making under the Exponential Family: Distributionally Robust Optimisation with Bayesian Ambiguity Sets** (ICML 2025), develops posterior-informed ambiguity sets and tractable dual formulations: https://proceedings.mlr.press/v267/dellaporta25a.html.
- Ohnemus et al., **Loss-Aware Distributionally Robust Optimization via Trainable Optimal Transport Ambiguity Sets** (ICML 2026), studies decision-focused ambiguity-set construction to avoid unnecessary conservatism: https://proceedings.mlr.press/v306/ohnemus26a.html.
- ACDB assesses budgeted interventions using separate skeleton, directed-DAG and intervention-efficiency metrics: https://arxiv.org/abs/2609.31675.
- CausalArena highlights ranking shifts across SCM families and protocols, motivating held-out mechanism/topology families: https://arxiv.org/abs/2609.11897.
- TCD-Arena evaluates time-series causal discovery under documented assumption violations: https://tcd-arena.github.io/.

The plausible research contribution is not “minimax/DRO is novel.” It is a carefully specified integration of explicit causal-hypothesis ambiguity sets, adaptive policy-tree optimization, policy/world response oracles, and independently verified certificates—qualified on held-out causal families.

## Next gates

1. Make local and committed source/test blobs byte-identical, then regenerate the result JSON from those exact files.
2. Add a digest-bound receipt linking model, polytope constraints, source/test blobs, results and numerical tolerances.
3. Replace this synthetic likelihood with Symthaea's exact repository SCM action/observation semantics.
4. Compare six-prior minimax, full-simplex, constrained-polytope, epsilon-pruned frontier and column generation at matched objective/certificate tolerance.
5. Test sensitivity to marginal bounds and atom floor; report the robust value-versus-conservatism curve.
6. Scale to more hypothesis atoms and horizons, then test held-out 3–8 node graph/mechanism families, noisy/missing/delayed telemetry, off-target interventions and intervention cost.
7. Run matched external benchmarks, reporting skeleton/directed F1, SHD, calibration/Brier/log loss, cost/intervention efficiency, p50/p95 runtime, peak memory and per-family uncertainty intervals.

No SOTA, novelty, production-readiness or general speedup claim is made.

# CMAQ-013: Selected-Action Uncertainty Calibration Under Planner-Induced Shift

**Status:** protocol v0.1 + executable Rust synthetic benchmark; draft  
**Tracking issue:** [#9127](https://github.com/Luminous-Dynamics/symthaea/issues/9127)  
**Claim ceiling:** deterministic synthetic calibration stress test only; no distribution-free guarantee under arbitrary adaptive actions, no production-safety or SOTA claim

## Research question

Does uncertainty calibrated on random behavior actions remain calibrated on the actions selected by a planner after searching a large candidate pool?

The key distinction is the distribution shift induced by selection itself. The planner selects the largest predicted value, so actions with positive prediction error are more likely to be selected. If model error also increases with candidate complexity, the chosen-action residual distribution can be much worse than the random-action residual distribution—even before the environment changes.

For a predicted value Vhat and realized value V, define the one-sided nonconformity score:

    s = Vhat - V

For quantile q, the lower confidence bound is:

    LCB = Vhat - q

The bound covers the realized value exactly when s <= q. This benchmark separately measures coverage on random behavior actions and on planner-selected actions.

## Experimental design

The dependency-free Rust executable evaluates **512 episodes × 48 decisions** for each of three scenarios, search widths 8/32/128 and rollout horizons 1/3/8 (27 scenario × width × horizon cells). The prediction-error standard deviation scales with the square root of rollout horizon, so the protocol tests both candidate-selection amplification and accumulated rollout uncertainty. Each decision uses the same deterministic synthetic candidate-generating process for all calibration methods.

Candidate complexity is coupled to latent action quality. More complex candidates have larger prediction-error variance. The planner selects the candidate with the highest predicted value, not the highest hidden true value. This creates a winner's-curse effect: the selected action's prediction error is biased upward relative to randomly sampled actions.

- **Stationary:** base residual distribution throughout.
- **Abrupt shift:** residual scale and bias increase at the episode midpoint.
- **Gradual shift:** residual scale and bias increase over the middle third.

For each width × horizon pair, a separate stationary development set of 512 × 16 decisions calibrates random-action and selected-action quantiles. Evaluation uses separate deterministic seeds. The qualification-like evaluation tape is intentionally public smoke data, not a sealed test set.

## Methods

1. **Fixed Gaussian:** one-sided 90% Gaussian-style bound using a fixed nominal scale.
2. **Random-action split conformal:** finite-sample upper quantile of nonconformity scores from randomly sampled behavior actions.
3. **Selected-action split conformal:** quantile from the top-predicted action selected at the same candidate-pool width. This calibrates the selection distribution under the stationary development regime.
4. **Online ACI-style quantile:** simplified stochastic online quantile update on selected-action residuals, initialized from the random-action calibration quantile. Labels arrive after four decisions and 10% are dropped. This is an ACI-inspired reduced baseline, not a reproduction of the full published ACI algorithm.
5. **Rolling residual quantile:** delayed selected-action scores in a 64-observation window, with a 16-score warm-up.
6. **Oracle selected quantile:** evaluator-only selected-action quantile computed from the evaluation tape. This is an upper-bound diagnostic, not an eligible method.

The 90% one-sided finite-sample order statistic uses rank ceil((n+1) × 0.9), capped at n.

## Measures

Each cell and method reports:
- selected-action lower-bound coverage;
- pre-shift and post-shift selected-action coverage (stationary scenarios report -1 for the inapplicable post-shift field);
- random-action lower-bound coverage;
- false-safe rate among accepted actions (LCB >= threshold but realized value < threshold);
- fallback rate;
- mean realized utility after fallback;
- mean interval width / lower-bound penalty q;
- delayed feedback updates per episode;
- fraction of shifted episodes that recover nominal rolling coverage;
- p90 recovery delay after shift.

The random-action coverage number must never substitute for selected-action coverage. The rollout-horizon axis tests whether longer prediction chains widen the selected-action residual distribution even at fixed search width. False-safe rate and tail utility matter in addition to nominal marginal coverage.

## Relation to current research

- [Adaptive Conformal Inference under Distribution Shift](https://papers.nips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) (Gibbs & Candès, NeurIPS 2021) motivates online adaptation to changing nonconformity scores.
- [Conformal Inference for Online Prediction with Arbitrary Distribution Shifts](https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf) adds adaptive step-size control and local regret analysis for shifting streams.
- [CAP: Calibration after Adaptive Pick](https://www.jmlr.org/beta/papers/v26/24-0452.html) develops selection-conditional coverage for adaptive picks. CMAQ-013's same-width selected split is a simpler selected-distribution baseline, not a CAP reproduction.
- [Conformal Optimistic Prediction](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5f8241131e6fd428aa49914da76b8ad0-Abstract-Conference.html) (ICLR 2026) studies online coverage/regret and tighter prediction sets. It is a future comparator, not implemented here.
- The [2026 Physical-AI world-model survey](https://link.springer.com/article/10.1007/s44163-026-02122-1) identifies policy-induced distribution shift, planner exploitation and uncertainty calibration as linked design tensions.

## Important limitations

- The selected-action split calibrator is trained on the stationary same-width selected distribution. It is expected to under-cover after a shift; the benchmark measures this rather than hiding it.
- The online method is intentionally labelled ACI-style. It uses a direct nonconformity-quantile update, not the complete ACI alpha-level process or CAP/COP.
- The simulator is synthetic and one-dimensional. It does not establish calibration on real MuJoCo, DMC, HumanoidBench, robot or Symthaea planner actions.
- The data-generating mechanism and seed construction are public. This is a smoke/research harness, not a sealed leaderboard.
- Coverage is reported empirically per scenario/width. No claim of arbitrary-distribution conditional coverage is made.

## Run

    cargo test --locked
    cargo run --quiet --locked

The JSON report is deterministic and includes all 27 cells (3 scenarios × 3 search widths × 3 rollout horizons) and six methods. A green workflow validates source, deterministic output and the declared synthetic measurement contract only.

## Next experiments

1. Add CAP and COP-faithful comparators with pinned published algorithms and assumptions.
2. Add joint stratification by search width, rollout horizon, candidate complexity and calibration feedback delay.
3. Compare one-sided conformal bounds with ensemble disagreement, epistemic/aleatoric decomposition and action-conditioned density estimates.
4. Carry the metrics into CMAQ-011 real environment adapters and CMAQ-012 paired direct/world-model ledgers.
5. Evaluate calibration cost against sharpness, fallback utility, false-safe tail risk and latency—not coverage alone.

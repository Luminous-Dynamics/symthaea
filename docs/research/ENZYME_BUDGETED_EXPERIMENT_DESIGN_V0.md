# Budgeted Enzyme Experiment Design V0

**Status:** exploratory synthetic benchmark specification and Rust implementation; not validated assay guidance.  
**Implementation:** `xtask enzyme-experiment-design`, in `xtask/src/enzyme_experiment_design.rs`.  
**Machine-readable output:** every campaign includes chosen substrate conditions, selection scores, generated observations, final model fits and the complete paired strategy summary.  
**Research context:** follows the abstention/falsification work in [PR #7533](https://github.com/Luminous-Dynamics/symthaea/pull/7533) and the calibration research in [Issue #7552](https://github.com/Luminous-Dynamics/symthaea/issues/7552).

## Paradigm shift

A common experimental-design objective is to collect more measurements where a parameter is most sensitive. That is not the same as choosing measurements that distinguish competing mechanisms.

For example, a substrate concentration near a fitted (K_m) may be informative about (K_m) under the Michaelis–Menten family. A much higher concentration may better separate Michaelis–Menten from substrate inhibition. Neither choice is universally best: the right condition depends on the scientific decision being targeted, the current parameter uncertainty, the measurement-noise model, constraints, and assay cost.

The benchmark therefore treats two objectives as distinct:

1. **Model discrimination:** select a condition where currently fitted competing model predictions differ most relative to a declared observation-noise scale.
2. **Parameter-estimation proxy:** select a condition with high local finite-difference sensitivity to (K_m), conditional on the Michaelis–Menten model family.

The reference strategies are a deterministic uniform-random control and a precommitted log-spaced sequence. All strategies use the same initial observations, candidate pool, simulated outcomes, measurement budget, and scenario seeds. Where two strategies select the same condition, they receive the same deterministic noisy outcome. This paired design reduces avoidable variation in the comparison.

This is a small test of the *decision logic*, not proof that either adaptive strategy improves real assay utility.

## Current executable matrix

The first version evaluates 64 campaigns:

- Two synthetic truths: Michaelis–Menten and substrate inhibition.
- Two declared noise SDs: 0.5 and 3.0 rate units.
- Four fixed seeds per scenario/noise combination.
- Four strategies per shared scenario: discrimination, local (K_m) sensitivity, deterministic uniform-random, and precommitted log grid.
- Three fixed initial substrate observations and four additional measurements per campaign.

Every campaign reports all seven observations, the order in which new substrate conditions were chosen, each selection score, both final model fits, BIC values, relative parameter errors where available, and whether the final BIC-favored model matches the known generator.

All concentrations and rates are synthetic and dimensionless for this benchmark. They are not assay recommendations.

## Strategy definitions

| Policy | Selection rule | What it tests | Main limitation |
|---|---|---|---|
| Adaptive model discrimination | Maximize the prediction difference between current fitted Michaelis–Menten and substrate-inhibition models, scaled by the assumed noise SD | Whether adaptive measurements probe model disagreement | Does not integrate over parameter uncertainty or calculate expected decision value |
| Adaptive (K_m) sensitivity | Maximize finite-difference (|partial v / partial K_m|), normalized by assumed noise SD, under the fitted Michaelis–Menten family | Whether local sensitivity can guide a parameter-estimation objective | Local, model-conditional proxy; not profile likelihood, Fisher-information optimality, or a confidence interval |
| Uniform-random control | Deterministic pseudo-random choice from remaining candidates | Reference policy at the matched measurement budget | One fixed seed set is not an estimate of the full stochastic distribution |
| Precommitted log grid | Select a predeclared spaced sequence from the candidate pool | Simple non-adaptive baseline | A single fixed sequence may be unusually good or poor for one scenario |

The current parameter grid and bounded-uniform synthetic noise are deliberately simple. The final BIC-favored candidate is reported as candidate-relative; it must not be read as proof of a mechanism. The present fit score uses an SSE-based BIC approximation whose Gaussian-error interpretation does not exactly match the bounded-uniform noise generator; this is explicitly a benchmark limitation to resolve in the likelihood-calibration work.

## Proposed benchmark gates

All targets below are future goals, not measured results.

| Benchmark | Proposed target | Required protocol |
|---|---|---|
| Paired accounting | 100% of campaign observations and budget steps accounted for exactly once | Machine-checked counts, no silent missing outcomes |
| Deterministic replay | Identical JSON report bytes for identical revision/toolchain/configuration | Pin the Rust toolchain and hash the input/configuration manifest |
| Model discrimination | ≥ 0.90 balanced accuracy on a locked matrix of separable mechanisms | At least 1,000 scenario instances per major error regime; include parameter variation and independent seeds |
| Parameter estimation | ≥ 25% lower median relative parameter error at matched budget, without worse tail error | Compare to each baseline with paired scenario/seed bootstrap intervals |
| False certainty | ≤ 5% decisive wrong-mechanism choices on *detectably* misspecified controls | Independently specify a detectability/power gate; report noise-obscured controls separately |
| Identifiability | Improved profile-likelihood 95% interval width in ≥ 80% of identifiable scenarios without degrading coverage | Use a valid likelihood/error model, profile-likelihood analysis, and report non-identifiable cases |
| Measurement efficiency | ≥ 25% fewer measurements to meet the same preregistered recovery and coverage criteria | Equal budget accounting; random and fixed-grid baselines; no degradation in uncertainty coverage |
| Robustness | Report utility separately under homoscedastic, heteroscedastic, correlated, outlier, censoring and model-misspecification regimes | Threshold tuning and evaluation manifests must be disjoint |
| Prospective utility | Positive cost-normalized decision benefit across at least two independent campaigns | Freeze candidate pool, strategy, selected measurements and scoring before outcomes are observed |

The exact statistical endpoint must match the objective. For model discrimination, use model-selection error and false-decisive rate. For parameter estimation, use relative error plus uncertainty-interval coverage and width. For prospective utility, use a preregistered decision-loss or yield/cost endpoint. Do not average these unlike endpoints into one aggregate "science score."

## Research basis

- Liu, Maini & Baker, [*Optimal experiment design for practical parameter identifiability and model discrimination*](https://www.sciencedirect.com/science/article/pii/S0025556426001008), *Mathematical Biosciences* 399 (September 2026), 109710. This research distinguishes practical identifiability (evaluated with profile-likelihood intervals) from model discrimination and discusses optimal-control formulations for experiment design.
- Hennecke et al., [*Closed-loop identification of enzyme kinetics applying model-based design of experiments*](https://pubs.rsc.org/en/content/articlehtml/2024/re/d4re00127c), *Reaction Chemistry & Engineering* (2024). This work describes integrating model-based optimal design into closed-loop enzyme-kinetic identification.
- Murphy, Gilmour & Crabbe, [*Effective experimental design: enzyme kinetics in the bioinformatics era*](https://www.sciencedirect.com/science/article/abs/pii/S135964460202384X), *Drug Discovery Today* (2002). A longstanding reminder that experiment design is crucial to extracting useful kinetic parameter information.
- [STRENDA guidelines](https://www.beilstein-strenda-db.org/strenda/public/guidelines.xhtml) and [EnzymeML V2](https://enzymeml.org/versions/v2/) should remain the reporting/interoperability foundations for later real-assay experiments; this benchmark is not a replacement.

## Reproduction

From the repository root:

```sh
cargo test -p xtask --bin xtask enzyme_experiment_design
cargo run -p xtask -- enzyme-experiment-design --out reports/enzyme_experiment_design_v0.json
```

These commands document independent reproduction. They are not evidence that this source has compiled or that the workflow has passed. Real-data and prospective claims remain out of scope until the error model is calibrated and independently reproduced assays are available.

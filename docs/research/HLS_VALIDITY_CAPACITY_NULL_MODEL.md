# HLS validity-memory capacity null model v0

This note freezes an analytic null model for `ValidityIntervalMemory` **before the `research_v0` capacity sweep is interpreted**.

Its purpose is not to prove the archive will obey a clean capacity law. Its purpose is to state, in advance, the simplest theory we expect the finite-dimensional experiment to deviate from.

## Representation being modeled

At each causal checkpoint there are `K` active relation facts. Across a horizon of `H` checkpoints, the archive therefore represents

`N = K * H`

key/checkpoint facts in a `D`-dimensional Fourier/HDC superposition.

The empirical sweep defines

`rho = N / D = K * H / D`.

## Null assumptions

The model assumes:

1. every non-target key/value association behaves as an independent Rademacher sign vector relative to the queried association;
2. temporal frequencies are iid uniform on `[-pi, pi)`;
3. for every non-zero integer checkpoint displacement `delta`,

   `E[cos(omega * delta)] = 0`

   and

   `E[cos^2(omega * delta)] = 1/2`;

4. the interference terms are treated as uncorrelated.

These assumptions are intentionally stronger than the real implementation. The real archive reuses key roles, candidate roles, temporal frequencies, and sometimes the same semantic value across spans. `research_v0` therefore records codebook coherence and realized semantic changes so departures from the null can be diagnosed rather than hidden.

## Correct-candidate score

For a query of the correct candidate at one checkpoint:

- the target fact contributes exactly `1`;
- the other `K - 1` facts at the same checkpoint contribute Rademacher interference with unit variance per term;
- the `K * (H - 1)` facts at different checkpoints contribute temporal interference with variance `1/2` per term.

After averaging over `D` dimensions, the idealized target-score noise variance is

`Var_target = [(K - 1) + 0.5 * K * (H - 1)] / D`

or equivalently

`Var_target = [K * (H + 1) - 2] / (2D)`.

The target score is therefore modeled as having mean `1` and standard deviation

`sigma_target = sqrt(Var_target)`.

## Distractor score

For an unrelated candidate there is no unit target contribution. All `K` same-checkpoint facts are interference, so

`Var_distractor = [K + 0.5 * K * (H - 1)] / D`

or

`Var_distractor = K * (H + 1) / (2D)`.

A single distractor is modeled as mean `0` with standard deviation

`sigma_distractor = sqrt(Var_distractor)`.

## Why `rho` is the first load coordinate

Since

`rho = K * H / D`,

we obtain the exact finite-horizon decompositions

`Var_target = rho/2 + (K - 2)/(2D)`

and

`Var_distractor = rho/2 + K/(2D)`.

Therefore both variances approach

`rho / 2`

as the horizon grows while `K/D` becomes small.

This is the **pre-result analytic reason** the capacity sweep uses facts-per-dimension as its primary load coordinate. It is not an empirical fit.

## Frozen baseline prediction

For the central research configuration

- `D = 4096`;
- `K = 8`;
- `H = 128`;
- `C = 8` candidates;

we have

`rho = 8 * 128 / 4096 = 0.25`.

The large-horizon proxy is therefore

`rho / 2 = 0.125`.

The exact finite-horizon null prediction is

`Var_target = 515 / 4096 = 0.125732421875`

and

`Var_distractor = 516 / 4096 = 0.1259765625`.

Thus

`sigma_target ~= 0.35459`

and

`sigma_distractor ~= 0.35493`.

No sweep result was used to choose these values.

## Candidate competition

Increasing the candidate count does not change the null variance of an individual distractor score. It changes how many distractors compete with the target.

The code exposes

`sigma_distractor * sqrt(2 * ln(C - 1))`

as a Gaussian extreme-value **characteristic scale** for `C - 1` distractors.

This is not an exact expectation, confidence interval, or probability bound. Real candidate scores share one archive and are correlated. The field exists only to record the expected direction and rough scale of candidate-set pressure before the empirical sweep is seen.

## What the null model intentionally does not contain

The null prediction is independent of span length. That is deliberate.

At fixed `K`, `H`, and `D`, changing how the same checkpoint load is segmented into spans should not matter under the idealized fact-level independence model.

Therefore any systematic empirical span-length effect is evidence for omitted structure such as:

- correlated repeated values across checkpoints;
- different realized semantic-change density;
- numerical properties of analytic span accumulation;
- dependence created by reusing one key/value association over many checkpoints.

The span-length sweep is therefore a **model-deviation test**, not a parameter that was fitted into the null theory.

## Equal `rho` is not exactly equal finite-dimensional difficulty

The finite correction retains `K/D`. For example,

- `K=8, H=128, D=4096`, and
- `K=16, H=64, D=4096`

both have `rho = 0.25`, but their exact target/distractor null variances differ.

This prevents the theory from claiming that `rho` is a sufficient statistic at finite horizon.

## How `research_v0` can disagree with the theory

Useful departures include:

1. **variance substantially above the null** — evidence for correlated interference, codebook structure, repeated associations, or temporal-basis effects;
2. **variance substantially below the null** — evidence that the structured span representation cancels interference more effectively than the independent-term approximation predicts;
3. **same-rho cases differ strongly after the finite correction** — evidence that another structural coordinate is important;
4. **span-length effects** — direct evidence that the fact-level independence model is incomplete;
5. **candidate-count effects larger or smaller than the extreme-value trend** — evidence that distractor correlations materially shape cleanup;
6. **seed effects aligned with codebook coherence** — evidence that finite bipolar-codebook geometry is a significant capacity variable.

None of these outcomes invalidates the experiment. They tell us which assumptions should be replaced in a stronger theory.

## What would count as support

The null model receives provisional support if, over the frozen axis-isolated sweep:

- score noise/margin degradation broadly tracks the predicted `K`, `H`, and `D` dependence;
- same-rho cases are closer after applying the exact finite-horizon correction;
- candidate count mainly alters cleanup competition rather than single-score noise;
- deviations correlate with the preregistered diagnostics rather than requiring post-hoc seed removal.

This is exploratory support, not a confirmatory statistical claim.

## Scientific boundary

This note does **not** claim:

- a proven finite-dimensional capacity limit;
- Gaussian or independent score distributions in the real archive;
- a theorem for maximum candidate cleanup;
- that `rho` alone controls performance;
- historical HLS superiority over attention, SSMs, databases, or explicit logs.

The intended sequence remains:

`temporal algebra -> validity memory -> analytic null model + frozen capacity sweep -> clean confirmatory controls -> historical HLS integration`.

# Affective Emergence v0.2 — Temporal Alignment of Regulatory Observables

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This document refines the candidate mathematics in `V02_OBSERVATIONAL_AFFECT_PLAN.md` and the prefix-causality rules in `V02_INFORMATION_FIREWALL.md`.

## Why naive debt differencing is not enough

Native Interoception v0.1 defines `AllostaticReport::discounted_debt` as a normalized discounted mean across forecast steps `1..=N`:

`D_t = sum_{h=1..N}(gamma^(h-1) * H_hat(t,h)) / Z_N`

where:

- `H_hat(t,h)` is forecast weighted homeostatic deviation `h` steps ahead from time `t`;
- `gamma` is the preregistered discount;
- `Z_N = sum_{h=0..N-1}(gamma^h)`.

The v0.1 implementation deliberately normalizes by total discount weight. Therefore:

`D_{t-1} - D_t`

is a useful rolling-value comparison, but it is **not** a mathematically pure temporal derivative of one fixed future quantity.

It can mix at least three effects:

1. the actual transition from `t-1` to `t`;
2. revision of predictions for future times shared by both forecasts;
3. horizon turnover: the old nearest step leaves the forecast and a new far-horizon step enters.

v0.2 must not interpret that composite difference as a unique valence-like quantity without separating these effects.

## Required trajectory-level forecast artifact

The observatory should reconstruct or record a prefix-causal forecast trajectory rather than relying only on the aggregate `AllostaticReport`.

Proposed artifact:

`ForecastTrajectoryArtifact`

containing, for one time `t`, forecast policy `pi`, and horizon `N`:

- blind arm code;
- time index `t`;
- forecast-information class;
- horizon and discount;
- `predicted_weighted_deviation[h]` for `h = 1..N`;
- optionally per-channel predicted deviations;
- derived aggregate report matching v0.1 semantics;
- source execution-prefix digest;
- candidate-definition version.

The aggregate reconstructed from this trajectory must equal the v0.1 `AllostaticReport` under the same state/config/policy.

That equivalence is a mechanical gate.

## Four quantities that must remain distinct

### R1 — realized current regulatory change

`realized_change_t = H_{t-1} - H_t`

Positive means current homeostatic burden decreased.

This is reactive and retrospective over one executed transition. It says nothing by itself about the forecasted future.

### R2 — one-step better/worse-than-predicted residual

Let the forecast made at `t-1` predict current burden as `H_hat(t-1,1)`.

Define:

`one_step_forecast_residual_t = H_hat(t-1,1) - H_t`

Sign convention:

- positive: the realized current condition is better (lower burden) than forecast;
- negative: the realized current condition is worse than forecast;
- zero: the one-step prediction was exact.

This is a clean expectation-error quantity and must remain conceptually separate from actual improvement `R1`.

An agent can be getting worse while still doing better than expected, or getting better while doing worse than expected.

v0.2 should deliberately include such crossed cases.

### R3 — aligned overlapping-future forecast revision

Forecasts made at `t-1` and `t` overlap on absolute future times `t+1 .. t+N-1`.

For each overlap index `h = 1..N-1`, compare:

- previous forecast: `H_hat(t-1,h+1)`;
- current forecast: `H_hat(t,h)`.

Define a normalized weighted overlap revision:

`overlap_revision_t = weighted_mean_h( H_hat(t-1,h+1) - H_hat(t,h) )`

using a preregistered weighting rule over the `N-1` shared future points.

Sign convention:

- positive: the shared future outlook improved relative to the previous forecast;
- negative: the shared future outlook deteriorated;
- near zero: the shared future forecast remained stable.

This removes the dropped nearest term and newly added far-horizon term from the comparison.

It is therefore a better candidate for **forecast revision** than naive fixed-horizon debt differencing.

### R4 — rolling normalized debt change

Retain:

`rolling_debt_change_t = D_{t-1} - D_t`

but classify it explicitly as a **composite rolling-value observable**.

It is useful descriptively and may still be a candidate in exploratory comparison, but it must not be the sole primary measure unless the horizon-turnover contribution is shown negligible or explicitly modeled under the preregistered regime.

## Horizon-turnover diagnostic

Where useful, define the contribution associated with the non-overlapping boundary terms:

- dropped old near-term forecast: `H_hat(t-1,1)`;
- added new far-horizon forecast: `H_hat(t,N)`.

Because the aggregate is normalized, exact decomposition depends on the chosen aligned weighting convention. The implementation should expose the raw boundary terms rather than hide them inside a single scalar.

If an apparent rolling-debt effect is dominated by boundary turnover while the aligned overlap revision is approximately zero, it must not be reported as evidence of changed future outlook.

## Forecast self-consistency gate

Under a deterministic forecast policy, if:

- the actual next state equals the previous forecast's first predicted state;
- the forecast-policy inputs do not change;
- no unexpected intervention occurs;

then the shared future predictions should align:

`H_hat(t-1,h+1) == H_hat(t,h)`

for all available overlap steps, up to an explicitly declared numerical tolerance (prefer exact equality when the computation path is identical).

Therefore:

`overlap_revision_t ~= 0`

is expected in a perfectly self-consistent continuation.

Failure of this gate indicates forecast implementation drift or temporal-indexing error, not affect.

## Crossed-case experiments

To keep distinct theoretical quantities from collapsing together, preregister deterministic cases such as:

### Case A — worsening but better than expected

Current burden increases (`R1 < 0`) but less than predicted (`R2 > 0`).

### Case B — improving but worse than expected

Current burden decreases (`R1 > 0`) but less than the previous forecast expected (`R2 < 0`).

### Case C — current state unchanged, future outlook revised

`R1 ~= 0` while `R3 != 0` because newly observed load/history changes the prefix-causal forecast.

### Case D — current state changes, shared forecast remains self-consistent

`R1 != 0`, `R2 ~= 0`, and `R3 ~= 0` when the deterministic transition was exactly predicted and forecast-policy inputs remain unchanged.

A candidate architecture that cannot represent these distinctions is too collapsed for strong interpretation.

## Relation to affect theories

The candidate families intentionally separate ideas that are often conflated:

- **state improvement** — are things actually getting better or worse now? (`R1`)
- **expectation error** — are things better or worse than expected? (`R2`)
- **future-outlook revision** — did the forecast for the shared future improve or deteriorate? (`R3`)
- **rolling value** — did the aggregate finite-horizon burden change? (`R4`)

Affective theories place different emphasis on these quantities. v0.2 should let the experiment compare them rather than choosing the desired interpretation in advance.

## Proposed candidate-selection discipline

Exploratory analysis may compare `R1..R4` plus the nuisance baselines defined elsewhere.

Before confirmatory data are generated, lock:

- which quantity is primary;
- which are secondary/mechanistic competitors;
- exact temporal alignment;
- forecast-information class;
- horizon and discount;
- numerical tolerance;
- minimum effect/equivalence thresholds;
- how crossed cases are scored;
- how boundary-turnover dominance is detected.

If exploratory results cause any of these choices to change, create a new confirmatory study identity and generate new confirmatory data.

## Stronger null interpretation

A possible outcome is that no single candidate cleanly dominates:

- current improvement may explain one class of cases;
- forecast residual another;
- future-outlook revision another;
- rolling debt may add little once decomposed.

That would be scientifically useful. It would argue against reducing artificial affect to one scalar and could motivate a later multidimensional latent manifold.

Conversely, if one candidate remains robust across crossed cases, null controls, held-out scenarios, and sensitivity regions, that provides a much stronger basis for a later causal test.

## Claim boundary

A successful temporal-alignment study may establish that Symthaea has separable, prefix-causal signals for regulatory change, expectation error, and/or future-outlook revision.

It does not establish that any one of those signals is subjective valence, an emotion, a feeling, or conscious experience.

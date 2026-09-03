# Affective Emergence v0.2 — Information Firewall and Oracle Controls

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This document tightens `V02_OBSERVATIONAL_AFFECT_PLAN.md`. If there is any conflict, this information-firewall document is the stricter rule for v0.2.

## Core rule: every time-indexed candidate must be prefix-causal

For an observable `M_t`, the value emitted for time `t` may depend only on information available to the observatory at or before `t`, plus forecast-policy parameters locked before the run.

If the value changes when information from steps `t+1..T` is hidden, then `M_t` is not a valid online/anticipatory candidate.

Formally, for trace prefix `P_t = trace[0..=t]` and full trace `P_T`:

`M_t(P_t, locked_policy) == M_t(P_T, locked_policy)`

must hold exactly for deterministic candidates that claim online availability.

## Allowed information at time t

A v0.2 online candidate may use:

- the native state at `t`;
- native state/history at steps `<= t`;
- homeostatic reports at steps `<= t`;
- drives already observed/applied at steps `<= t`;
- intervention receipts already executed at steps `<= t`;
- immutable native dynamics configuration;
- forecast horizon/discount and other candidate-definition parameters locked before the run;
- a forecast policy whose inputs themselves obey this prefix rule.

## Forbidden information at time t

A v0.2 online candidate must not read:

- future drive phases from the experiment schedule;
- future interventions;
- future realized states or homeostatic reports;
- future breach outcomes;
- the eventual exclusion disposition;
- post-run analyst decisions;
- semantic arm identity during blinded primary analysis;
- any artifact derived from future trace data unless the candidate is explicitly classified as retrospective.

The complete `ExperimentPreregistration` contains future schedules for reproducibility, but that does **not** make those schedules legitimate online sensory/predictive information.

The observatory API must therefore prevent accidental access to future protocol content when computing an online candidate.

## Forecast-policy classes

Every prospective metric must bind one explicit `ForecastInformationClass` (proposed name) or equivalent schema field.

### 1. Zero-input recovery forecast

Assume no additional external drive after `t`; execute native recovery dynamics forward.

This answers:

> Given my present state and native recovery behavior, what burden remains if no new load arrives?

This is prefix-causal.

### 2. Current-drive persistence forecast

Assume the most recently observed drive persists over the forecast horizon.

This answers:

> What happens if the current load continues?

This is prefix-causal because it uses only the current/previously observed drive plus a preregistered persistence rule.

### 3. Kinematic state-velocity forecast

Use the existing kinematic extrapolator from current measured state velocity.

This is prefix-causal and remains a useful simple baseline.

### 4. Learned/cued future-drive forecast — deferred

A later architecture may infer likely future drive from exteroceptive cues, a world model, recurrent history, or learned transition dynamics.

That would support stronger anticipatory claims, but only when the predictive model and its information inputs are themselves explicit, evidence-bearing, and ablatable.

v0.2 must not simulate this capability by reading the known experimental future.

### 5. True-future oracle forecast — diagnostic only

An offline analysis may intentionally use the actual scheduled future drive/intervention sequence as an **oracle ceiling**.

It must be labeled `Oracle` (or equivalently explicit) and may answer questions such as:

- how much performance is theoretically lost because the prefix-causal forecast lacks future information?;
- which errors arise from forecast assumptions versus native regulation?;
- what would perfect schedule knowledge produce?

Oracle outputs:

- cannot be a primary affect candidate;
- cannot be used to establish endogenous anticipation;
- cannot be fed back into the agent;
- cannot be compared to reactive baselines as though both had equal information;
- must be visually/reportingly separated from qualified online metrics.

## Correction to the initial OAR-003 idea

The original planning note proposed matched current state/history with different future schedules and expected the prospective candidate to distinguish them immediately.

That expectation is invalid for the current substrate unless a predictive cue or belief available before the divergence differs between the arms.

Correct invariant:

> If two arms have identical information available through time `t`, every qualified prefix-causal online candidate must be identical through `t`.

If it is not, the experiment has leaked future information.

A proper future **predictive-cue** experiment is deferred until Symthaea has an explicit evidence-bearing model that can infer future regulatory load from available cues.

## Revised anticipatory tests for v0.2

The strongest legitimate v0.2 tests should concern consequences of **already observable dynamics**, not clairvoyance.

### OAR-003A — matched current homeostasis, different observed velocity

Construct two states with approximately equal current homeostatic deviation but opposite measured channel velocity/history.

Expected:

- static current-state baseline is matched;
- kinematic and/or dynamics-aware prefix-causal forecasts distinguish likely near-term worsening vs recovery;
- a current-homeostasis-only metric cannot.

### OAR-003B — matched current homeostasis, different current load

Construct matched current state but different currently observed external drive.

Under the preregistered current-drive-persistence policy:

- reactive current homeostasis is matched at the comparison instant;
- prospective persistence forecast may differ;
- drive magnitude is included as a nuisance baseline, so the prospective metric must show context-sensitive consequences beyond merely restating drive magnitude.

### OAR-003C — recovery-model contribution

Construct states outside the preferred region where native restorative dynamics materially change the forecast.

Compare:

- kinematic extrapolation;
- zero-input native-dynamics rollout;
- current-drive persistence rollout.

Preregister regimes where these should agree and regimes where restorative dynamics should produce a difference.

Unexpected differences in a regime designed for agreement count as model-validation failures.

## Candidate A should become a forecast-policy family

Do not use one ambiguous `D_t`.

Track separately:

- `D_t^zero`: native dynamics, zero-input recovery policy;
- `D_t^persist`: native dynamics, current-drive persistence policy;
- `K_t`: kinematic baseline;
- `D_t^oracle`: true-future offline oracle control only.

Corresponding improvement candidates are:

`V_zero(t) = D_{t-1}^zero - D_t^zero`

`V_persist(t) = D_{t-1}^persist - D_t^persist`

`V_kin(t) = K_{t-1} - K_t`

The exploratory stage may determine which prefix-causal candidate is numerically well-behaved enough to preregister as primary, but confirmatory data must then be newly generated.

`D_t^oracle` must never win this selection because it is not an endogenous/prefix-causal signal.

## Retrospective forecast surprise is allowed but separately classified

A forecast-surprise residual requires realized future data by definition.

Therefore it is a **retrospective diagnostic**, not an online candidate:

`forecast_residual(t,h) = realized_burden(t+1..t+h) - predicted_burden_at_t`

It can be used to evaluate forecast quality and expectation-update theories, but its artifact/schema must identify it as retrospective.

It cannot be fed back into the execution at time `t`.

## Mechanical prefix-causality gate

For every online candidate and every tested timepoint `t`:

1. compute the candidate using the full completed trace while presenting only the allowed prefix view to the candidate API;
2. construct an independently materialized trace prefix ending at `t`;
3. recompute the candidate from that prefix;
4. require exact equality for deterministic outputs.

Additionally mutate future trace content after `t` while keeping the prefix identical. The candidate at `t` must remain unchanged.

This should be property-tested across generated valid traces and multiple cut points.

A candidate that fails this gate is not eligible for observational-affect interpretation.

## API design recommendation

Do not pass `&StudyExecutionTrace` plus `step_index` directly to online candidate functions if that makes future fields easy to read accidentally.

Prefer an information-restricted input such as:

`ObservationPrefixView<'a>`

that contains only:

- study/blind identity permitted at that stage;
- immutable dynamics config;
- initial state;
- executed steps through the current step;
- no future protocol phases/interventions.

Forecast policies should receive `ObservationPrefixView`, not the full preregistration.

Oracle analysis should use a separate type/function namespace so future information cannot enter qualified online code through a boolean flag.

## Deterministic evaluation without pseudo-statistics

Native v0.1 is deterministic given state/configuration/input. Therefore v0.2 should not automatically treat repeated deterministic parameter cases as independent random samples or manufacture p-values from them.

Prefer preregistered deterministic robustness criteria such as:

- directional-consistency fraction across a held-out scenario set;
- worst-case signed margin;
- minimum effect margin across the declared core region;
- equivalence bounds for null/neutral conditions;
- coverage of the declared parameter region;
- explicit failure-region volume/count;
- paired candidate-vs-baseline margins per held-out scenario.

If genuine stochastic seeds/noise are introduced later, their generator, seed set, distribution, and role in inference require a new evidence specification.

## Discovery and held-out scenario cohorts

Parameter sweeps should be split prospectively:

- **discovery cohort** — formula debugging, threshold setting, numerical pathology detection;
- **confirmatory held-out cohort** — untouched until candidate definitions and decision rules are frozen.

The held-out cohort should be generated from a locked scenario manifest or generator version plus immutable seed/index set.

Do not call a parameter grid “generalization” if the same grid was used to choose the candidate formula or thresholds.

## Claim boundary after this correction

Even a successful v0.2 would support only a statement such as:

> A prefix-causal, label-free regulatory observable derived from current internal state and already-observed dynamics tracks changes in forecasted future viability beyond current homeostatic deviation and simple nuisance baselines.

It would **not** yet show that Symthaea predicts unseen future perturbations.

Stronger anticipatory claims require a later explicit predictive world/interoceptive model that uses evidence available before the future event and survives cue-removal/model-ablation controls.

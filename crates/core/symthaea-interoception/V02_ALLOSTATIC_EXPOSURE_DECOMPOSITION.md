# Affective Emergence v0.2 — Allostatic Exposure Decomposition

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract addresses issue #269 without changing the qualifying Native Interoception
v0.1 semantics.

The key observation is that v0.1 `AllostaticReport.discounted_debt` is normalized by the
sum of discount weights before return. It is therefore a **discount-weighted mean
projected burden**, not a cumulative exposure integral.

That quantity remains useful, but v0.2 must not silently interpret it as duration-sensitive
"debt", persistence, mood, or accumulated suffering.

## 1. Principle

Keep temporal dimensions separate:

- intensity;
- cumulative exposure;
- peak;
- terminal condition;
- duration outside preferred range;
- duration outside viability;
- latency to first breach;
- recovery exposure.

A later candidate may combine them only after explicit preregistration and evidence.

## 2. Temporal aggregation axis

Every v0.2 regulatory-burden candidate should declare one temporal aggregation class.

### T0 — instantaneous

Burden at a single cut point.

Examples:

- current channel-deviation vector;
- current viability-weighted burden;
- current legacy precision×importance aggregate.

### T1 — discounted mean burden

Current v0.1 `discounted_debt` semantics:

`M_discount = sum_k B_k * gamma^(k-1) / sum_k gamma^(k-1)`

Properties:

- normalizes across the declared horizon;
- useful as average projected intensity;
- does not by itself encode total duration/exposure.

Keep this under a neutral/legacy identity.

### T2 — discounted cumulative exposure

Conceptual form:

`E_discount = sum_k B_k * gamma^(k-1) * dt`

Properties:

- accumulates burden across forecast duration;
- horizon-sensitive by design;
- binds `dt`, horizon, and discount in candidate identity;
- comparisons across different horizons require an explicitly justified rule.

### T3 — undiscounted cumulative exposure

`E_raw = sum_k B_k * dt`

A transparent area-under-burden curve useful as a control against discount-specific
behavior.

### T4 — peak burden

Maximum projected burden over the horizon.

Existing v0.1 peak deviation remains a candidate/input, with the weighting basis stated
explicitly where an aggregate peak is later introduced.

### T5 — terminal burden

Burden at the final forecast step.

Useful but path-insensitive; therefore always reported separately from cumulative
exposure.

### T6 — preferred-range exposure duration

Time/channel exposure outside preferred bands, even before viability is breached.

Possible forms:

- total channel×step exposure;
- unique affected channels;
- weighted exposure using the declared viability weights.

### T7 — viability-breach exposure duration

Builds on v0.1 `breach_exposures` and `unique_breached_channels` but binds explicit
abstract-time semantics.

A future duration field should make clear whether it counts:

- channel×step exposures;
- any-breach steps;
- weighted breach time;
- physical time (only if an explicit physical mapping later exists).

### T8 — breach latency

Time/step to first predicted viability breach.

Imminence belongs to urgency, not automatically to burden intensity.

### T9 — recovery exposure

Cumulative burden after a declared perturbation until:

- return to preferred range;
- return to viability;
- or a prospectively fixed cutoff.

The recovery condition and cutoff are part of candidate identity.

## 3. Weighting × temporal matrix

Issue #268 and this contract are orthogonal axes.

Each aggregate candidate should explicitly declare:

### Weighting basis

- `RawChannel`
- `ViabilityWeightOnly`
- `LegacyPrecisionTimesImportance`
- future separately qualified basis

### Temporal aggregation

- T0 instantaneous
- T1 discounted mean
- T2 discounted cumulative
- T3 undiscounted cumulative
- T4 peak
- T5 terminal
- T6 preferred exposure
- T7 breach exposure
- T8 latency
- T9 recovery exposure

This prevents a candidate called "allostatic debt change" from hiding both a weighting
choice and a temporal-integration choice.

## 4. Required neutral identities

Example candidate IDs:

- `t1_w1_discounted_mean_viability_burden`
- `t1_w2_discounted_mean_legacy_burden`
- `t2_w1_discounted_cumulative_viability_exposure`
- `t3_w1_raw_cumulative_viability_exposure`
- `t7_raw_breach_exposure`
- `t8_raw_first_breach_latency`

Names remain interpretation-neutral and do not contain `emotion`, `mood`, `pain`, or
`valence`.

## 5. Discriminating scenario family

The exploratory study should deliberately generate profiles where temporal aggregators
rank conditions differently.

### D1 — constant burden, different duration

Same nonzero projected burden for 8 vs 16 steps.

Expected:

- T1 mean may remain equal;
- T2/T3 cumulative exposure increases with duration;
- peak remains equal.

### D2 — short severe pulse vs long mild burden

Choose profiles with similar or matched average burden but different exposure/peak.

Purpose: prevent one measure from masquerading as all temporal semantics.

### D3 — equal terminal state, different path

Two trajectories end at the same final burden but take different routes.

Expected:

- T5 equal;
- T2/T3 may differ;
- T4 may differ.

### D4 — equal peak, different duration

Same maximum burden but different time near the peak.

Expected:

- T4 equal;
- cumulative exposure differs.

### D5 — same breach breadth, different latency

Same number of channels eventually breach, but one condition breaches much earlier.

Expected:

- breadth equal;
- T8 differs;
- exposure duration likely differs.

### D6 — recovery path divergence

Same perturbed initial state and same eventual recovered state, but different recovery
rates or intervention histories.

Expected:

- terminal state may equal;
- T9 recovery exposure differs.

### D7 — discount sensitivity

Same trajectory under several preregistered discount values, including 1.0.

Purpose: show which conclusions depend on future discounting rather than trajectory
shape itself.

### D8 — horizon-extension control

Extend a forecast by appending neutral/preferred states after full recovery.

A candidate's response to this extension must match its declared temporal semantics:
normalized means, cumulative exposure, and latency metrics need not behave identically.

## 6. Abstract-time contract

v0.1 `step_dt` / allostatic `dt` are abstract model time, not physical seconds.

Therefore:

- exposure units are initially "normalized burden × model-step";
- candidate definitions must bind exact `dt`;
- papers/receipts must not call the quantity seconds/minutes without a separate mapping;
- changing the abstract-to-physical mapping later creates a new interpretation/evidence
  contract.

## 7. Horizon comparability

Cumulative exposure is intentionally horizon-sensitive.

For confirmatory comparisons, prefer one of:

- fixed identical horizon across compared conditions;
- prospectively declared normalization;
- separate within-horizon analysis strata.

Do not compare cumulative exposure across different horizons and then interpret the
larger value as stronger regulatory burden without accounting for the extra exposure
time.

T1 discounted mean remains useful precisely because it normalizes horizon weight, but
that convenience must not be confused with duration sensitivity.

## 8. Forecast trajectory requirement

The first observatory should preserve trajectory-level burden values instead of only
receiving the final v0.1 aggregate report.

The trajectory artifact should expose, for every forecast step:

- cut-point-relative and absolute forecast index;
- state/prefix provenance permitted by the forecast contract;
- raw channel deviations;
- preferred/viable status;
- weighting-basis inputs needed by registered candidates;
- forecast policy identity;
- dt / horizon / discount identity.

The existing v0.1 aggregate allostatic report should be exactly reproducible from this
trajectory under the legacy formula as an equivalence gate.

That preserves backward compatibility while enabling explicit T1–T9 candidates.

## 9. Candidate-selection discipline

Do not prefer cumulative exposure merely because it produces more persistent-looking
signals.

Selection criteria should be preregistered, such as:

- structural correctness under D1–D8;
- neutrality under no-load scenarios;
- robustness across held-out parameter/scenario regions;
- explanatory value beyond current burden/peak/latency baselines;
- invariance expected under irrelevant horizon extensions;
- absence of pathological dependence on arbitrary cutoff choices.

Valid outcomes include:

- `MeanSufficient`
- `ExposureAddsStructure`
- `TemporalAggregationAmbiguous`
- `NoUniqueWinner`
- candidate-specific qualified null/failure.

## 10. Relation to persistence / mood-like claims

Duration-sensitive regulatory signals are a prerequisite for later persistence studies,
but they are not themselves mood.

A later mood-like tranche would additionally need evidence for:

- persistence beyond the immediate perturbation;
- state/history dependence;
- causal modulation of multiple cognitive functions;
- recovery/regulation dynamics;
- mechanism-specific ablation;
- separation from simple running averages or slowly decaying filters.

v0.2 must not jump from T2/T3 exposure to a mood claim.

## 11. Future native report changes

If a later native interoception semantics version is redesigned, prefer explicit fields
rather than changing the meaning of `discounted_debt` silently:

- `discounted_mean_burden`
- `discounted_cumulative_exposure`
- `undiscounted_cumulative_exposure`
- explicit breach-duration metrics

Such a native change requires the appropriate model/snapshot/evidence schema changes and
new qualification lineage.

## 12. Design-freeze consequence

Before implementation authorization, v0.2 should freeze:

- T1 current legacy semantics;
- trajectory-level artifact sufficient to derive T1–T9;
- D1–D8 exploratory discriminating scenarios;
- exact weighting × temporal candidate matrix eligible for exploratory comparison;
- fixed rules for horizon/dt/discount comparability;
- no interpretation of T2/T3/T9 as mood or subjective experience.

## 13. Claim boundary

Temporal decomposition can show that two regulatory trajectories differ in average
intensity, cumulative exposure, peak, duration, latency, or recovery cost.

It cannot by itself establish affect, emotion, mood, suffering, subjective valence,
sentience, or consciousness.

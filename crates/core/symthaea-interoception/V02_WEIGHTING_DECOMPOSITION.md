# Affective Emergence v0.2 — Weighting Decomposition Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract addresses issue #268 without changing Native Interoception v0.1 model semantics while its exact qualification candidate is pending.

The core concern is that v0.1 currently computes aggregate homeostatic deviation using `precision * importance` as the channel weight. That quantity can be a legitimate confidence-adjusted estimate, but v0.2 must not silently treat it as the unique intrinsic viability burden or as valence by construction.

This contract now defines only the **weighting axis**. Temporal aggregation is independently defined by `V02_ALLOSTATIC_EXPOSURE_DECOMPOSITION.md`, and the complete candidate coordinate is defined by `V02_CANDIDATE_FACTOR_SPACE.md`.

## 1. Principle

Keep **normative/regulatory burden** and **epistemic confidence** distinct until evidence supports a particular coupling.

For v0.2, the following are competing weighting observables, not aliases:

1. raw channel values/deviations;
2. viability/preference-weight-only aggregation;
3. legacy precision×importance aggregation;
4. precision/confidence itself;
5. future explicitly qualified uncertainty/epistemic summaries.

No candidate family may rename the legacy aggregate to `valence`, `badness`, `pain`, or another higher-level interpretation.

## 2. Existing v0.1 quantities

v0.1 already exposes enough structure to avoid changing its native semantics:

- per-channel normalized deviations in `HomeostaticReport.channel_deviations`;
- per-channel forecast quantities in `AllostaticReport.channel_debt`;
- per-channel `importance`;
- per-channel `precision`;
- raw breach count/breadth/timing metrics independent of aggregate weight.

Important semantic note: v0.1 `AllostaticReport.channel_debt` and aggregate `discounted_debt` are normalized by the total discount weight. They therefore behave as **discount-weighted mean projected burdens**, not cumulative exposure integrals. The temporal contract governs how these values are named and how cumulative alternatives are derived.

## 3. Weighting bases

### W0 — RawChannel

No cross-channel weighting or aggregation.

The vector of per-channel deviations remains available directly:

`d_i(t)`

Purpose:

- preserve native regulatory geometry;
- ensure aggregate weighting cannot erase a severe channel;
- support channel-specific, vector-valued, breach, and peak-channel baselines.

### W1 — ViabilityWeightOnly

Conceptual cross-channel aggregate:

`B_v = sum_i x_i * w_i / sum_i w_i`

where:

- `x_i` is the temporal quantity supplied by the independently declared temporal basis;
- `w_i` is the declared viability/preference significance currently represented by `importance`.

Properties:

- changing precision alone does not alter W1;
- changing importance can alter W1;
- zero total viability weight must be typed unavailable or handled by a prospectively frozen rule, never silently interpreted as zero burden.

### W2 — LegacyPrecisionTimesImportance

Exact v0.1 weighting hypothesis:

`B_legacy = sum_i x_i * precision_i * importance_i / sum_i precision_i * importance_i`

where `x_i` is again supplied by the temporal basis.

Do not call this intrinsic burden. Its scientific meaning is explicitly:

> a precision/confidence-adjusted aggregate under the v0.1 weighting hypothesis.

For T0 instantaneous deviation, W2 reproduces v0.1 `HomeostaticReport.weighted_deviation`.

For T1 discounted mean forecast burden, W2 reproduces v0.1 `AllostaticReport.discounted_debt` when the trajectory/legacy equivalence contract is satisfied.

### W3 — ConfidenceOnly

Precision/confidence is observed as its own quantity rather than being silently reclassified as viability burden.

Candidate summaries may include prospectively specified forms such as:

- importance-weighted mean precision;
- minimum precision over threatened channels;
- precision of the current peak-deviation channel;
- breadth of low-confidence channels;
- later entropy/variance measures once a probabilistic uncertainty contract exists.

Until uncertainty has a generative probabilistic contract, these are descriptive confidence candidates rather than calibrated posterior uncertainty.

### WFutureQualified

Reserved for a future weighting rule only after its exact semantics, fixtures, evidence identity, and sensitivity behavior are frozen.

It cannot be used as a generic extension escape hatch in an existing confirmatory lineage.

## 4. Weighting is independent of temporal aggregation

Weighting and temporal aggregation form separate axes.

Examples:

- `W1 × T0` — current viability-weighted burden;
- `W2 × T0` — current legacy confidence-weighted burden;
- `W1 × T1` — discounted mean viability-weighted forecast burden;
- `W2 × T1` — exact legacy v0.1 discounted mean aggregate;
- `W1 × T2` — discounted cumulative viability exposure;
- `W2 × T2` — discounted cumulative exposure under legacy precision weighting;
- `W0 × T8` — raw first-breach latency;
- `W3 × T0` — current confidence observable.

Do not use `debt` in a candidate ID without also specifying the temporal aggregation contract. Prefer `mean_burden`, `cumulative_exposure`, `peak`, `latency`, etc.

## 5. Regulatory relation candidates

R0/R1/R2/R3/R4/urgency candidates declare their weighting basis explicitly through the candidate coordinate.

Examples:

- `r0_w1_t0_viability_burden_v1`
- `r1_w1_t0_viability_change_v1`
- `r1_w2_t0_legacy_weighted_change_v1`
- `r4_w1_t2_viability_cumulative_change_v1`
- `r4_w2_t1_legacy_rolling_mean_change_v1`

Do not compare formulas while hiding the weighting difference inside one candidate ID.

Candidate identity must bind:

- weighting contract/version;
- exact weights used;
- treatment of zero weights;
- precision dependency yes/no;
- temporal aggregation identity;
- normalization;
- forecast basis/horizon/discount where applicable.

## 6. Falsification-oriented scenario family

The first v0.2 exploratory cohort should deliberately decorrelate burden and precision.

### P1 — fixed state, changed precision

Hold values, preferred/viable geometry, importance, velocity, and drive history fixed. Vary precision only.

Expected structural behavior:

- W0 unchanged;
- W1 unchanged;
- W2 may change;
- W3 changes by definition;
- raw breach measures unchanged.

If a supposed precision-independent burden candidate changes here, its contract is wrong or mislabeled.

### P2 — fixed precision, changed state deviation

Hold precision and weights fixed while increasing native deviation.

Expected:

- W0 responds monotonically per affected channel;
- W1 responds according to viability weight;
- W2 responds under the legacy rule;
- W3 remains fixed.

### P3 — severe low-confidence breach

Construct a severe native breach under deliberately low precision.

Required:

- raw breach evidence remains visible;
- W0/peak/breach-count do not disappear;
- W1 remains severe according to viability weight;
- W2 may attenuate, exposing the exact semantic consequence of precision weighting.

### P4 — mild high-confidence vs severe low-confidence

Cross burden and precision so W1 and W2 may rank conditions differently.

This is a required discriminating scenario, not an edge case.

### P5 — precision-only prospective fixture change

Hold native state/trajectory fixed while changing only precision in synthetic/reference fixtures.

Use this to prove which weighting candidates are epistemically sensitive and which are normatively invariant.

### P6 — importance-only change

Hold precision/state fixed and vary importance.

This tests whether W1 responds according to its normative weighting contract independently of confidence.

## 7. Candidate-selection rule

Do not choose W1 or W2 because it yields a more emotionally intuitive story.

Use preregistered structural criteria such as:

- invariance expected under precision-only manipulations;
- monotonicity under burden-only manipulations;
- robustness across sensitivity region;
- explanatory separation from raw current-state / urgency baselines;
- stability under held-out scenarios;
- absence of pathological ranking reversals unless prospectively predicted.

`NoUniqueWinner` and `WeightingAmbiguous` are valid outcomes.

## 8. Relationship to active inference

v0.2 should use active-inference concepts as competing computational motivations, not as an excuse to collapse them into one scalar.

At minimum preserve the conceptual distinction between:

- preferences / desired outcomes / viability significance;
- uncertainty / precision / reliability of beliefs or observations;
- epistemic value or uncertainty reduction;
- pragmatic value or preferred-state realization.

The exact Symthaea implementation does not have to reproduce one biological or formal active-inference model, but it should not use the word `precision` while giving it an undeclared normative meaning.

## 9. Future native-semantics change

If evidence/review later concludes that v0.1's native aggregate should itself be redefined, do not silently change `weighted_deviation` or `discounted_debt` in place.

Preferred future report shape may include separate fields such as:

- `viability_weighted_deviation`;
- `confidence_weighted_deviation`;
- `peak_deviation`;
- raw channel deviations;
- explicit confidence/uncertainty summary;
- `discounted_mean_burden`;
- explicit cumulative exposure fields.

Such a change alters scientific report semantics and must:

- increment `INTEROCEPTIVE_MODEL_SEMANTICS_VERSION` where native model/report meaning changes;
- use a new snapshot/report schema as required;
- start a new evidence lineage;
- rerun qualification;
- preserve old v0.1 artifacts under their original semantics.

## 10. Design-freeze consequence

Before v0.2 implementation is authorized:

- W0/W1/W2/W3 semantics are frozen;
- temporal aggregation is always separately identified;
- no primary affect precursor may assume W2 is intrinsically normative;
- the exploratory study includes P1–P6 discriminating scenarios;
- the factor-space compatibility validator rejects invalid/mislabeled combinations;
- confirmatory candidate choice remains open only until exploratory selection is frozen into a new confirmatory identity.

This resolves issue #268 at the design level without altering the qualifying v0.1 substrate.

## 11. Claim boundary

Separating weighting from temporal aggregation and confidence reduces a semantic confound. It does not prove that W1 is “true valence”, that W2 is wrong, or that any candidate corresponds to emotion, mood, suffering, or subjective feeling.

The point of v0.2 is precisely to let these interpretations compete under controlled, label-free evidence rather than choosing one by naming convention.

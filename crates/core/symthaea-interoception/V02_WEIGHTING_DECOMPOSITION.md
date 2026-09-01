# Affective Emergence v0.2 — Weighting Decomposition Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract addresses issue #268 without changing Native Interoception v0.1 model
semantics while its exact qualification candidate is pending.

The core concern is that v0.1 currently computes aggregate homeostatic deviation using
`precision * importance` as the channel weight. That quantity can be a legitimate
confidence-adjusted estimate, but v0.2 must not silently treat it as the unique
intrinsic viability burden or as valence by construction.

## 1. Principle

Keep **normative/regulatory burden** and **epistemic confidence** distinct until evidence
supports a particular coupling.

For v0.2, the following are competing observables, not aliases:

1. raw channel deviation;
2. viability/preference-weighted burden;
3. legacy precision×importance weighted aggregate;
4. precision/confidence itself;
5. optional explicit uncertainty/epistemic-pressure summaries.

No candidate family may rename the legacy aggregate to `valence`, `badness`, `pain`, or
another higher-level interpretation.

## 2. Existing v0.1 quantities

v0.1 already exposes enough structure to avoid changing its native semantics:

- per-channel normalized deviations in `HomeostaticReport.channel_deviations`;
- per-channel projected debt in `AllostaticReport.channel_debt`;
- per-channel `importance`;
- per-channel `precision`;
- raw breach count/breadth/timing metrics independent of aggregate weight.

Therefore v0.2 can construct alternative read-only aggregates from frozen artifacts
without rewriting v0.1.

## 3. Proposed neutral aggregates

Use interpretation-neutral candidate IDs and exact formula manifests.

### W0 — raw burden vector

The vector of per-channel normalized deviations:

`d_i(t)`

No cross-channel aggregation.

Purpose:

- preserve the full native regulatory geometry;
- ensure aggregate cancellation/weighting cannot erase a severe channel;
- support channel-specific and worst-channel baselines.

### W1 — viability-weighted burden

Conceptual form:

`B_v(t) = sum_i d_i(t) * w_i / sum_i w_i`

where `w_i` is the declared viability/preference significance currently represented by
`importance`.

Properties:

- changing precision alone does not alter `B_v`;
- changing importance can alter `B_v`;
- raw breach count/peak remain separately available;
- zero total viability weight must be typed unavailable or otherwise handled by a
  preregistered rule, never silently interpreted as zero burden.

### W2 — legacy confidence-weighted aggregate

The current v0.1 aggregate:

`B_legacy(t) = sum_i d_i(t) * precision_i * importance_i / sum_i precision_i * importance_i`

Retain it under a neutral identity as a competing hypothesis.

Do not call it intrinsic burden. Its scientific meaning is explicitly:

> a precision/confidence-adjusted aggregate under the v0.1 hypothesis.

### W3 — aggregate confidence / uncertainty

Do not infer a single universal formula prematurely.

Candidate families may include prospectively specified summaries such as:

- importance-weighted mean precision;
- minimum precision over threatened channels;
- precision over the currently peak-deviation channel;
- breadth of low-confidence channels;
- entropy/variance from a future explicit uncertainty model, once available.

Until uncertainty has a generative probabilistic contract, these remain descriptive
confidence candidates rather than calibrated posterior uncertainty.

### W4 — projected viability-weighted debt

Use the v0.1 per-channel `AllostaticReport.channel_debt` with the declared viability
weights:

`A_v(t) = sum_i channel_debt_i(t) * w_i / sum_i w_i`

This creates an importance-only prospective burden candidate without changing v0.1
allostatic dynamics.

### W5 — legacy projected debt

Retain v0.1 `discounted_debt` under a neutral legacy candidate identity.

W4 and W5 must be compared explicitly before either is used in a regulatory-improvement
or valence-motivated candidate.

## 4. Regulatory change candidates

R1/R2/R3/R4 candidate families should declare which weighting basis they use.

Examples:

- `r1_w1_viability_burden_change`
- `r1_w2_legacy_weighted_change`
- `r4_w4_viability_debt_change`
- `r4_w5_legacy_debt_change`

Do not compare formulas while hiding the weighting difference inside one candidate ID.

Candidate identity must bind:

- weighting contract/version;
- exact weights used;
- treatment of zero weights;
- precision dependency yes/no;
- normalization;
- forecast basis/horizon/discount where applicable.

## 5. Falsification-oriented scenario family

The first v0.2 exploratory cohort should deliberately decorrelate burden and precision.

### P1 — fixed state, changed precision

Hold values, preferred/viable geometry, importance, velocity, and drive history fixed.
Vary precision only.

Expected structural behavior:

- W0 unchanged;
- W1 unchanged;
- W2 may change;
- W3 changes by definition;
- raw breach measures unchanged.

If a supposed intrinsic-burden candidate changes here, its precision dependency is
explicit and must be defended rather than hidden.

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

Cross burden and precision so the two aggregation hypotheses may rank conditions
differently.

This is a required discriminating scenario, not an edge case.

### P5 — precision-only prospective change

Hold native state/trajectory fixed while changing only the declared precision field in
synthetic/reference fixtures.

Use this to prove which allostatic candidate families are epistemically sensitive and
which are normatively invariant.

### P6 — importance-only change

Hold precision/state fixed and vary importance.

This tests whether the declared viability-weighted candidate responds according to its
normative weighting contract independently of confidence.

## 6. Candidate-selection rule

Do not choose W1 or W2 because it yields a more emotionally intuitive story.

Use preregistered structural criteria such as:

- invariance expected under precision-only manipulations;
- monotonicity under burden-only manipulations;
- robustness across sensitivity region;
- explanatory separation from raw current-state / urgency baselines;
- stability under held-out scenarios;
- absence of pathological ranking reversals unless prospectively predicted.

`NoUniqueWinner` and `WeightingAmbiguous` are valid outcomes.

## 7. Relationship to active inference

v0.2 should use active-inference concepts as competing computational motivations, not as
an excuse to collapse them into one scalar.

At minimum preserve the conceptual distinction between:

- preferences / desired outcomes / viability significance;
- uncertainty / precision / reliability of beliefs or observations;
- epistemic value or uncertainty reduction;
- pragmatic value or preferred-state realization.

The exact Symthaea implementation does not have to reproduce one biological or formal
active-inference model, but it should not use the word `precision` while giving it an
undeclared normative meaning.

## 8. Future native-semantics change

If evidence/review later concludes that v0.1's native aggregate should itself be
redefined, do not silently change `weighted_deviation` in place.

Preferred future report shape:

- `viability_weighted_deviation`;
- `confidence_weighted_deviation`;
- `peak_deviation`;
- raw channel deviations;
- explicit confidence/uncertainty summary;
- corresponding separated allostatic debt fields.

Such a change alters the scientific meaning of homeostatic/allostatic reports and must:

- increment `INTEROCEPTIVE_MODEL_SEMANTICS_VERSION`;
- use a new snapshot/report schema as required;
- start a new evidence lineage;
- rerun qualification;
- preserve old v0.1 artifacts under their original semantics.

## 9. Design-freeze consequence

Before v0.2 implementation is authorized, classify weighting semantics as follows:

- v0.1 legacy aggregate remains available as W2/W5;
- v0.2 adds importance-only W1/W4 as observational candidates;
- precision/confidence is exposed separately as W3;
- no primary affect precursor is permitted to assume W2/W5 are intrinsically normative;
- the exploratory study must include P1–P6 discriminating scenarios;
- confirmatory candidate choice remains open until exploratory comparison is frozen into
  a new confirmatory identity.

This resolves issue #268 at the design level without altering the qualifying v0.1
substrate.

## 10. Claim boundary

Separating these quantities reduces a semantic confound. It does not prove that W1 is
"true valence", that W2 is wrong, or that any candidate corresponds to emotion or
subjective feeling.

The point of v0.2 is precisely to let these interpretations compete under controlled,
label-free evidence rather than choosing one by naming convention.

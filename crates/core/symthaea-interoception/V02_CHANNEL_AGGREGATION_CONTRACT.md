# Affective Emergence v0.2 — Cross-Channel Aggregation Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract addresses the denominator/channel-set ambiguity captured in issue #278 without changing Native Interoception v0.1 semantics.

The core concern is that a scalar burden aggregate must say not only **which weights** it uses, but also **which channels enter the scalar and how the denominator is formed**.

## 1. Principle

Keep these questions separate:

1. what is each channel's deviation?
2. which channels are projected into the candidate?
3. how are projected channels weighted?
4. is the scalar a sum, mean, maximum, count, or other prospectively defined reduction?
5. if it is a mean, what enters the denominator?

A change in any of these can alter the candidate even when the underlying deviated channel is unchanged.

## 2. Why denominator semantics matter

The v0.1 legacy aggregate is a weighted mean across all declared channels:

`B = sum_i d_i * w_i / sum_i w_i`

where legacy `w_i = precision_i * importance_i`.

A healthy channel with `d_i = 0` still contributes its weight to the denominator. Increasing that healthy channel's weight can therefore lower `B` without improving the deviated channel.

That behavior may be appropriate for an all-system average, but it must not be silently interpreted as recovery, reduced intrinsic burden, or valence improvement.

## 3. Proposed ChannelAggregationBasis

Extend the v0.2 candidate coordinate with an explicit cross-channel aggregation basis.

Initial classes:

- `A0FullVector` — preserve all channel values; no scalar reduction;
- `A1SingleChannel(channel)` — one prospectively named channel;
- `A2PeakDeviation` — maximum declared channel deviation;
- `A3WeightedMeanAllDeclared` — weighted mean over one fixed prospectively declared channel set;
- `A4WeightedSumAllDeclared` — weighted sum over the fixed declared channel set;
- `A5BreachBreadth` — count/breadth of channels meeting the declared breach rule;
- `A6ThreatenedSubsetDiagnostic` — optional diagnostic mean/sum over a prospectively defined threatened subset;
- `AFutureQualified` — reserved for later separately frozen aggregation semantics.

`A6ThreatenedSubsetDiagnostic` should not be a default primary candidate because membership can itself become an outcome-dependent selection rule. Any use must specify membership from allowed prefix information and pass selection/leakage audits.

## 4. Fixed channel-set identity

Every scalar candidate must bind an exact ordered/semantic channel set or a deterministic prospectively defined projection rule.

For v0.2 over the current substrate, the full declared native set is the eight v0.1 channels unless the candidate is explicitly channel-specific/vector/peak/breadth.

Candidate identity must bind:

- channel-set digest/version;
- projection rule;
- aggregation basis;
- denominator rule when applicable;
- weighting basis;
- zero-weight handling;
- unavailable-channel handling;
- membership rule for any subset reduction;
- normalization rule;
- reference fixtures.

A future native channel addition/removal does not silently preserve the same scalar candidate identity.

## 5. Mean vs sum semantics

### Weighted mean

Answers a question like:

> what is average declared burden under this fixed weighting/denominator policy?

Properties:

- invariant to uniform scaling of all included weights;
- sensitive to relative weights of healthy and deviated channels;
- can exhibit healthy-channel dilution;
- naturally bounded relative to constituent deviations under nonnegative weights.

### Weighted sum

Answers a different question:

> how much total weighted burden is present across the declared channel set?

Properties:

- changes under uniform scaling of weights;
- sensitive to channel count/set;
- does not normalize healthy-channel weight into the denominator;
- requires explicit interpretation of weight magnitude and cross-lineage comparability.

Neither is assumed superior. They are distinct hypotheses.

## 6. Peak/vector/breadth semantics

### Full vector

Preserves channel identity and avoids aggregation cancellation/dilution. It is the preferred audit substrate even when scalar candidates are also computed.

### Peak deviation

Tracks worst normalized channel deviation and is invariant to weights on unrelated healthy channels. It loses information about breadth and total burden.

### Breach breadth

Tracks how many channels cross a declared boundary. It ignores subthreshold severity and within-channel magnitude above the boundary unless paired with other quantities.

A useful candidate family may ultimately require multiple observables rather than forcing one scalar to encode severity, breadth, confidence, and duration.

## 7. Required denominator/distribution discriminators

The exploratory scenario set should include:

### A-D1 — healthy-channel weight dilution

Hold a deviated channel's value/geometry/own weight fixed. Change only the weight of a healthy channel.

Expected structural behavior:

- full vector: unchanged;
- single deviated channel: unchanged;
- peak deviation: unchanged;
- all-declared weighted mean: may change;
- weighted sum: may remain unchanged if the healthy channel has zero deviation and its own deviated-channel weight is fixed;
- breadth: unchanged.

### A-D2 — healthy-channel precision dilution

Same as A-D1 but alter precision only under the legacy weighting basis.

This separates issue #268's epistemic effect from denominator normalization.

### A-D3 — matched mean / different concentration

Construct one scenario with burden concentrated in one channel and another with burden spread across channels while matching the selected mean.

Peak, vector, breadth, and sum candidates should expose differences hidden by the matched mean.

### A-D4 — matched total / different distribution

Match weighted sum while redistributing burden across channels.

Use to show what mean/peak/breadth add beyond total burden.

### A-D5 — fixed peak / changed background

Keep the maximum deviation fixed while changing mild burden on other channels.

Peak should remain fixed while full-vector/sum/mean/breadth may differ according to their definitions.

### A-D6 — permutation symmetry

Under channels with identical declared geometry/weights, permuting otherwise equivalent channel values should preserve symmetric scalar aggregators while channel-specific/vector identities transform in the prospectively defined way.

### A-D7 — zero-weight behavior

A zero-weight channel may disappear from a weighted scalar but must not disappear from raw vector, peak, breach breadth, or evidence accounting.

### A-D8 — future channel-set diagnostic

Reference fixtures should document what happens if a neutral channel is added/removed in a future schema. The result is a **new candidate lineage** unless the candidate definition explicitly establishes a stable channel-set mapping.

## 8. Candidate-coordinate integration

`CandidateCoordinate` should refine `channel_projection` into or supplement it with an explicit `aggregation_basis`.

Examples:

- `R0 × W1 × T0 × A3WeightedMeanAllDeclared`;
- `R0 × W1 × T0 × A4WeightedSumAllDeclared`;
- `R0 × W0 × T0 × A2PeakDeviation`;
- `R1 × W0 × T0 × A1SingleChannel(compute_reserve)`.

The compatibility validator must reject contradictory combinations, for example:

- `A0FullVector` with scalar-only normalization fields;
- `A1SingleChannel` with a denominator across unrelated channels;
- `A6ThreatenedSubsetDiagnostic` whose membership needs future or semantic/unblinded information;
- `A5BreachBreadth` paired with a formula that claims continuous severity without another declared input.

## 9. Invariance declaration

Every candidate definition should declare which transformations are expected to leave its payload unchanged.

Potential invariances include:

- uniform scaling of all included weights;
- healthy-channel weight changes;
- channel permutation under symmetric configuration;
- addition/removal of zero-weight channels;
- redistribution of burden preserving a chosen total/mean;
- irrelevant semantic/blind-code changes;
- unseen future suffix changes for prefix-causal candidates.

The adversarial suite should test declared invariances rather than assuming them from names.

A violated declared invariance is a candidate/implementation failure. A transformation not declared invariant is a designed sensitivity, not automatically a defect.

## 10. Interaction with temporal and weighting factors

Cross-channel aggregation is orthogonal to:

- weighting basis from `V02_WEIGHTING_DECOMPOSITION.md`;
- temporal aggregation from `V02_ALLOSTATIC_EXPOSURE_DECOMPOSITION.md`;
- relation basis from `V02_TEMPORAL_ALIGNMENT.md` / candidate factor space.

Thus a candidate is more precisely understood as something like:

`relation × weighting × temporal aggregation × cross-channel aggregation × forecast/information policy`.

This factorization is intended for scientific attribution, not to authorize an unbounded Cartesian candidate search.

## 11. Confirmatory rule

Before confirmatory lock:

- choose/freeze the exact cross-channel aggregation of each primary/baseline candidate;
- freeze the channel-set/projection identity;
- freeze denominator semantics;
- retain raw vector/peak/breach evidence even when the primary is a mean/sum;
- ensure the discrimination manifest contains scenarios capable of separating the chosen primary aggregate from plausible simpler alternatives.

A denominator policy cannot change after confirmatory outputs are inspected.

## 12. Future native-semantics changes

If later evidence supports changing `HomeostaticReport.weighted_deviation` itself, do not silently alter the v0.1 field.

Prefer explicit future fields such as:

- `legacy_weighted_mean_deviation`;
- `viability_weighted_mean_deviation`;
- `viability_weighted_sum_deviation`;
- `peak_deviation`;
- raw channel deviations.

Changing native aggregate semantics requires the appropriate model/snapshot schema and evidence-lineage changes.

## 13. Claim boundary

Making aggregation and denominator semantics explicit can prevent dilution, concentration, or channel-set effects from being mislabeled as regulatory improvement.

It does not establish that any scalar aggregation is affect, emotion, subjective valence, suffering, sentience, or consciousness.
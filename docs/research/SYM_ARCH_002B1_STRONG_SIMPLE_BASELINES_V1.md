# SYM-ARCH-002B1 — Strong-Simple Analytic Baselines v1

**Status:** implementation contract; no architecture result

**Tracks:** #55

**Depends on:** the SYM-ARCH-002 experimental core and generated-task validity work. This tranche defines baseline mechanisms only; a later experiment binds them to a frozen DEV/CONFIRM/REPL manifest and benchmark generator.

## Question

Before adding GRU, trainable SSM, Mamba, liquid dynamics, or Hebbian plasticity, test the simplest serious alternative explanation for a future Symthaea architecture result:

> Is the apparent advantage already explained by a fixed representation plus a strong online analytic readout?

B1 deliberately contains no replay and no temporal state.

## Baseline ladder

All three conditions use the same `OnlineRlsBinary` update rule and frozen `RlsConfig`:

1. `one_hot_rls`
   - normalized categorical one-hot input;
   - no representation seed;
   - no encoded-dimension knob;
   - full-covariance online RLS readout.

2. `fixed_random_tanh_rls`
   - the same frozen categorical schema;
   - deterministic fixed random projection from pair-specific one-hot features;
   - `tanh` nonlinearity;
   - L2-normalized output;
   - full-covariance online RLS readout.

3. `vanilla_hdc_rls`
   - the same frozen categorical schema;
   - deterministic feature-role HVs;
   - shared categorical-value HVs across roles;
   - `ContinuousHV` role/value binding and bundling;
   - L2-normalized output;
   - the exact same full-covariance online RLS readout shape as the random condition when encoded dimensions match.

`MatchedBaselineFamilySpec` is the preferred construction path. It emits all three conditions from one feature-schema/RLS contract so the random and HDC controls cannot quietly receive different ridge, forgetting, bias, schema, or encoded dimensions.

The phrase **readout-matched** is reserved for a pair whose *effective* RLS input/state shape is actually equal. Sharing the same RLS algorithm/configuration is not by itself sufficient when the encoder output dimensions differ.

## Why include one-hot RLS?

The raw categorical condition tells us whether a strong analytic readout already solves the task without any high-dimensional representation.

Interpretation must respect the executable contrast audit:

- `hdc ≈ random` under a representation-level audit: HDC receives no evidence beyond generic fixed random features;
- `hdc > random` under a representation-level audit: representation-level evidence for the factorized HDC implementation;
- `random > hdc` under a representation-level audit: HDC is less useful in that regime;
- `one_hot` versus a higher-dimensional random/HDC condition is a **reference/capacity comparison** when effective readout dimensions differ;
- therefore `random > one_hot` does **not** by itself isolate a benefit from nonlinear expansion when the RLS trainable/covariance shape also increased;
- if one-hot and encoded dimensions happen to match and every other fairness invariant passes, the audit may elevate that pair to a representation-level contrast.

A B1 HDC advantage does **not** isolate HDC binding algebra from every other representation difference. The random encoder is pair-feature based, whereas HDC explicitly shares value identities across roles and composes them through binding. Exact algebraic attribution requires a later matched factorization/binding ablation, including the planned continuous-vs-exact bipolar/BinaryHV tranche.

## Executable contrast-fairness audit

`experiment_baseline_fairness::audit_baseline_contrast` validates both specs and records:

- exact spec digests;
- learner-visible feature-schema equality;
- exact RLS contract equality;
- effective feature dimensions;
- trainable readout state dimensions;
- inverse-covariance shapes;
- paired representation-seed index when both encoders are randomized;
- replay count (`0` for B1);
- temporal-state bytes (`0` for B1);
- explicit qualifiers for every mismatch.

It returns one of two non-scalar claim ceilings:

- `representation_level` — schema, RLS protocol, effective readout shape, and applicable random-seed pairing are matched;
- `reference_only` — capacity/protocol differs enough that a representation-only attribution is not admissible.

This audit intentionally does **not** require equal fixed-encoder storage. Encoder storage is part of the representation's resource cost and must be reported separately. A representation-level predictive contrast is therefore not automatically a resource-normalized comparison.

`audit_matched_family` audits all three pairwise contrasts emitted by one `MatchedBaselineFamilySpec`. In the usual B1 regime where the encoded random/HDC dimension is much larger than the categorical one-hot dimension, random↔HDC should be representation-level while one-hot↔random and one-hot↔HDC remain reference-only.

## This is RanDumb/F-OAL-inspired, not a reproduction

The fixed-random and analytic-readout controls are motivated by recent online continual-learning results showing that frozen random representations and forward-only analytic classifiers can be unexpectedly strong.

B1 does not claim to reproduce a named external method exactly. Its random transform, categorical schema, and RLS implementation are benchmark-local and explicitly versioned. Any external-method reproduction should be a separate adapter with its original algorithm/configuration documented.

## RLS contract

`OnlineRlsBinary` uses binary targets `{-1,+1}` and standard recursive least squares with:

- frozen positive ridge precision;
- forgetting factor in `(0, 1]`;
- optional bias term;
- symmetric rank-one inverse-covariance update;
- no replay buffer;
- no learned encoder.

The readout has trainable weight count `d (+ 1 when bias is enabled)` but maintains a full `f64` inverse-covariance matrix of the same state dimension.

### Replay-free is not memory-free

Full RLS state is `O(d^2)`.

Every B1 result must report separately:

- encoded feature dimension;
- fixed encoder bytes;
- readout weight bytes;
- inverse-covariance bytes;
- total persistent state bytes;
- trainable parameter count;
- replay examples (`0` here);
- temporal-state bytes (`0` here).

The implementation has a hard 512 MiB covariance ceiling. This is a safety invariant, not a scientific hyperparameter. A request above the ceiling fails before allocation. In particular, full-covariance RLS must not be naively instantiated at Symthaea's ordinary 16K HDC dimension.

Large-dimension comparisons need a separately specified bounded-state analytic readout (for example diagonal, block, sketch, or low-rank) with its own evidence tranche; they must not silently change the B1 algorithm.

## Frozen categorical schema

`CategoricalFeatureSchema` is strict:

- feature names are normalized, unique, and sorted;
- value domains are explicit, unique, and sorted;
- assignments must contain exactly the declared learner-visible features;
- out-of-domain values fail closed;
- one-hot vectors are normalized.

For claim-bearing experiments the schema comes from the preregistered generator/reference task specification. It must not be discovered by looking at CONFIRM labels or changed after observing CONFIRM behavior.

Task/world/boundary metadata remains subject to the A3/A4 validity policies; B1 does not authorize hidden task identity merely because a field can be represented categorically.

## Determinism and provenance

Random and HDC encoder state is deterministic from the versioned implementation plus the frozen:

- baseline kind;
- categorical schema;
- encoded dimension;
- representation seed.

The one-hot condition has no random representation. Its `encoded_dimension` and `representation_seed` fields are required to be exactly zero so irrelevant parameters cannot create fake experimental variants or degrees of freedom.

Each emitted baseline spec has a domain-separated BLAKE3 digest over its versioned schema/configuration. The spec digest does **not** pretend to be a git-revision digest. Exact source identity belongs in the experiment manifest's `code_revision` and later claim/evidence binding. A claim-bearing artifact must bind both the frozen baseline-spec digest(s) and the exact code revision used to execute them.

## What must be frozen before CONFIRM

At minimum:

- all emitted baseline spec digests;
- exact code revision and baseline schema version;
- exact categorical schema;
- encoded dimension for random/HDC;
- representation-seed manifest;
- RLS ridge;
- RLS forgetting factor;
- bias policy;
- stream/order/environment seeds;
- update count / observation budget;
- evaluation points;
- primary comparator and metric;
- SESOI and analysis rule;
- resource budget/fairness regime;
- expected contrast-fairness claim ceiling for every primary comparison.

DEV may be used to choose these values. CONFIRM may not be used to tune them.

`RlsConfig::default()` and unit-test dimensions are software conveniences, not preregistered scientific defaults.

## Fairness regimes

B1 supports distinct comparisons and they must not be conflated.

### Readout-matched representation comparison

`fixed_random_tanh_rls` vs `vanilla_hdc_rls` at the same encoded dimension, paired representation-seed index, exact same learner-visible schema, and exact same RLS configuration.

This is the primary representation comparison and should pass the executable fairness audit as `representation_level`.

### Simpler-model / capacity reference

`one_hot_rls` usually has a smaller feature/readout dimension than the high-dimensional conditions.

It is intentionally a lower-complexity baseline. If it is practically equivalent, the simpler model may still be preferable under a separately frozen complexity/resource policy. But a performance difference against a larger random/HDC readout is not, by itself, a clean representation-only effect.

Later resource-normalized tranches may additionally compare at matched persistent bytes or update compute. B1 does not invent a single weighted score across capability and resource dimensions.

## Acceptance tests

The implementation tranche is acceptable when the exact PR head demonstrates:

1. strict deterministic categorical-schema encoding;
2. online RLS learns a simple separator without replay;
3. oversized full-covariance RLS is rejected before allocation;
4. one-hot specs reject irrelevant seed/dimension knobs;
5. fixed-random encoding is seed deterministic;
6. vanilla HDC encoding is seed deterministic and normalized;
7. matched-family specs share one RLS/schema contract;
8. matched random/HDC conditions have identical readout state shape;
9. encoder state remains fixed while labels/readout updates change;
10. spec digests change when meaningful baseline configuration changes;
11. resource accounting exposes the full covariance cost;
12. random↔HDC receives a representation-level audit under the matched family;
13. dimension-mismatched one-hot↔random is downgraded to reference-only;
14. randomized representation contrasts with unpaired seed indices are downgraded;
15. RLS protocol mismatch is downgraded;
16. B1 Rust files pass rustfmt;
17. the psych-bench library compiles.

## Wording ceiling

Merging B1 supports only:

> Symthaea psych-bench contains deterministic, resource-audited strong-simple one-hot, fixed-random, and vanilla-HDC online-RLS baselines plus an executable contrast-fairness audit suitable for later preregistered architecture comparisons.

It does not support:

- an architecture-performance claim;
- a claim that HDC beats random features;
- a representation-only interpretation for a contrast that the fairness audit marks `reference_only`;
- a claim that RLS is a faithful reproduction of a named published method;
- a claim that HDC algebra has been isolated;
- a claim about liquid dynamics, Hebbian plasticity, or full Symthaea intelligence.

## Next step after infrastructure validation

Use B1 only after the A-series validity/statistics infrastructure is green enough to construct a DEV campaign. The first empirical goal is to measure whether fixed random nonlinear features plus RLS explain any apparent representation benefit before adding temporal or plastic mechanisms.

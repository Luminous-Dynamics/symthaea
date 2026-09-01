# Affective Emergence v0.2 — History Access and Native-State Sufficiency Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents a prefix-causal external observatory from creating an apparent
history-dependent or persistent affect signal and then attributing that persistence to
the native regulator itself.

Native Interoception v0.1 is a deterministic regulatory substrate. Its transition law
consumes the current native state, dynamics configuration, and current drive. The first
v0.2 observatory is allowed to replay a frozen historical trace, but historical
reconstructibility is not the same thing as endogenous memory or endogenous affective
state.

## 1. Principle

Keep three claims distinct:

1. **historically derivable** — an external replay process can compute a quantity from
   the execution prefix;
2. **currently state-derivable** — the same quantity is determined by the current native
   state/configuration and other explicitly current allowed inputs;
3. **endogenously represented/persistent** — the native system itself contains a state
   variable or qualified memory mechanism that carries the relevant history forward.

v0.2 may establish (1), and some candidates may establish (2). It is not designed to
establish (3).

A rolling window, exponentially weighted history, cumulative exposure integral, or
forecast-revision series implemented by the observatory is **observer state** unless an
independently qualified native mechanism stores an equivalent sufficient statistic.

## 2. v0.1 native-state boundary

Under the current v0.1 contract, the executable native state contains the eight
`ViabilityVariable`s and their current value/bounds/precision/velocity/importance.
`NativeInteroceptiveModel::step()` advances that state using the current state,
configuration, and current `InteroceptiveDrive`.

The primary v0.2 design must therefore not claim that arbitrary prior trace history is
part of the native agent's internal state merely because `ReplayHarness` can access it.

This is especially important for:

- cumulative exposure;
- recovery exposure;
- rolling R1/R2/R3/R4 summaries;
- historical volatility/trend candidates;
- repeated-breach counts;
- path-shape descriptors;
- any later persistence or mood-motivated statistic.

## 3. History-access basis

Add an explicit `history_access_basis` to each candidate identity.

Initial classes:

### `H0CurrentNativeStateOnly`

Candidate depends only on the current validated native state/configuration plus other
explicitly current allowed inputs such as currently observed drive.

No earlier trace samples are required.

### `H1ReplayedPrefixHistory`

Candidate may depend on the immutable execution prefix before the current cut point.

This is allowed for observational science, but its result is classified as an
**externally history-derived statistic**.

It is not evidence that the native regulator internally stores that statistic.

### `H2NativePersistedMemory`

Reserved for a future lineage in which a separately qualified native memory/state
mechanism persists history and exposes an evidence-bearing internal sufficient state.

Not available to initial v0.2 candidates.

### `H3RetrospectiveHistory`

Uses realized information after the declared cut point. Diagnostic only; not
prefix-causal.

### `H4OracleFuture`

Uses future experimental information. Oracle diagnostic only.

`H3` and `H4` remain ineligible for primary endogenous candidates.

## 4. Current-state sufficiency experiment

For every `H1ReplayedPrefixHistory` candidate, ask whether historical access actually
adds information beyond the current complete native state.

Construct history-paired scenarios that reach the same current native state and
current configuration at the comparison cut point but differ in their preceding
histories.

Where the current-drive/current-input contract also matters, match those inputs as
prospectively declared.

Then compare:

- H0 current-state baselines;
- H1 historical candidate.

Possible outcomes:

- `HistoryRedundantGivenCurrentState` — candidate is fully determined by current-state
  information across the locked discriminator set;
- `ObserverHistoryAddsInformation` — replayed history changes the external candidate
  while current native state is matched;
- `InsufficientStateMatching` — scenarios did not actually match all required current
  native variables/configuration;
- `Indeterminate`.

`ObserverHistoryAddsInformation` does **not** mean the native regulator has memory. It
means the external observatory computed a path-dependent statistic not contained in the
matched native state representation.

## 5. Restart / state-sufficiency gate

A strong structural test should compare two native executions initialized at the same
validated native state/configuration/cycle convention and then subjected to the same
future drive/intervention sequence.

If their future native trajectories differ solely because they arrived at that state by
different historical paths, then the declared native state is not actually sufficient
for deterministic restart and the v0.1 state contract needs investigation.

Under the current v0.1 hypothesis, expected result is exact native future equality when
all transition-relevant current state/configuration and future inputs are identical.

This gate is about native transition-state sufficiency. It is separate from whether an
H1 external historical candidate differs.

## 6. Candidate payload identity

Candidate manifests/payloads should record:

- `history_access_basis`;
- exact required prefix-history range/window when H1/H3 is used;
- history aggregation/update rule;
- whether the candidate can be reconstructed incrementally from a bounded sufficient
  statistic;
- if so, the sufficient-statistic definition/version;
- whether that statistic exists only in the observatory or in qualified native state;
- history truncation/warm-up behavior;
- reset/restart semantics.

Changing the history window, forgetting factor, sufficient statistic, or reset rule is
a new candidate identity.

## 7. Incremental-equivalence test

For an H1 candidate that can be maintained incrementally, require:

`batch(prefix_0..t) == incremental(update(...update(initial, x_0), ... x_t))`

under the declared numerical contract.

This distinguishes a legitimate causal streaming statistic from an implementation that
secretly relies on unavailable later data.

It still does not make the statistic native/endogenous unless the native model itself
stores and updates that state.

## 8. History permutation and path discriminators

Required scenario families should include:

1. **same current state, different prior path**;
2. **same cumulative exposure, different temporal ordering**;
3. **same peak/terminal state, different duration profile**;
4. **same recent window, different remote history**;
5. **same full past burden sequence but different semantic arm label**;
6. **history reset at a declared boundary**;
7. **state restart from matched native snapshot followed by identical future inputs**;
8. **history truncation/warm-up sensitivity**.

These reveal exactly which historical support a candidate uses.

## 9. Persistence / mood firewall

Initial v0.2 may describe an H1 signal as:

- persistent external regulatory statistic;
- path-dependent observational quantity;
- cumulative or history-sensitive candidate.

It must not describe such a signal as:

- native mood;
- endogenous affective persistence;
- remembered distress/reward;
- autobiographical affect;
- latent internal emotional state;

unless a later lineage introduces and qualifies a native persistent mechanism and
shows targeted ablation/state-transfer evidence.

A candidate continuing to be nonzero because the observatory's accumulator retains
prior samples is not evidence that Symthaea retains that state.

## 10. Factor-space integration

`history_access_basis` becomes an explicit candidate factor alongside:

- relation basis;
- weighting basis;
- cross-channel aggregation;
- temporal aggregation;
- forecast policy;
- information/execution class.

Compatibility examples:

- R0 current burden normally uses H0;
- cumulative exposure naturally uses H1 unless a future native accumulator is
  separately qualified;
- H2 is unavailable in initial v0.2;
- H3 implies retrospective diagnostic authority;
- H4 implies oracle authority;
- an `OfflinePrefixCausal` candidate may be H0 or H1, but the claim language differs.

## 11. Identifiability interaction

The candidate discrimination matrix should compare history-sensitive H1 candidates
against H0 current-state baselines on scenarios where current state is exactly matched.

If no locked scenario separates them, the history factor is not identifiable and the
more complex historical candidate cannot be promoted on the basis of implied
persistence.

If H1 wins on matched-state history contrasts, the supported claim is limited to:

> prior execution history contains additional information for this external regulatory
> observable beyond the declared current native state baseline.

That is scientifically useful without claiming endogenous memory.

## 12. Causal-contrast interaction

A history contrast must declare whether historical path is the manipulated factor and
which current-state variables are required equal at the cut point.

Do not match away a transition-relevant mediator when the question is the total effect
of a historical perturbation. Conversely, if the question is specifically whether
history adds information beyond current state, exact current-state matching is the
intended conditioning operation.

The causal question determines whether current-state matching is appropriate.

## 13. Evidence-root / design-freeze consequence

This contract is normative and architecture-blocking for any claim involving history,
persistence, cumulative exposure, or state sufficiency.

The design registry/freeze/evidence root should bind its digest/version.

The exploratory candidate-set manifest must classify every candidate by
`history_access_basis` before execution.

Any later transition from externally replayed history (H1) to native persisted memory
(H2) is a new architecture/model/evidence lineage, not an implementation detail.

## 14. Claim boundary

v0.2 can test whether prior regulatory history improves an **external observational
measurement** beyond current-state baselines, and whether native state is sufficient
for deterministic restart under the declared transition law.

It cannot establish that the native system remembers, feels, has a mood, or possesses a
persistent affective state merely because an external replay computation depends on
history.

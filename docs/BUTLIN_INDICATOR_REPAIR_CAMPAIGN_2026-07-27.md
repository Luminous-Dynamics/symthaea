# Butlin Indicator Repair Campaign — Future Work Roadmap

**Status**: proposal, not started. A distinct, larger undertaking than
`BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`'s minimal runner — do not begin this before that runner
exists and has actually proven the qualification contract holds on the 4 rows already eligible.
This document exists so the roadmap isn't lost, not as a commitment to start it now.

## The objective, stated precisely

Not "get more indicators to pass." The right target:

> **8-10 qualified direct probes, representing at least 6 independent evidence units across
> several theory families** — where "qualified" means the experiment is capable of returning a
> meaningful positive, null, *or* contradiction, not that it was tuned to look supportive.

**The honest rule this campaign must hold to for every repaired row:**

```text
Probe qualified + expected effect found  → eligible for causal/functional support
Probe qualified + no effect              → NotDemonstrated
Probe not qualified / manipulation failed → Inconclusive
```

A better experiment may leave the qualified-support count unchanged, or even reduce it. That's
still progress — the goal is increasing the number of indicators Symthaea can be proven *wrong*
about, not the number that can be made to look right.

## Highest-leverage shared investment: a typed experimental-input interface

Four blocked rows (`RPT-2`, `HOT-1`, `AST-1`, `AE-1`) all assume custom stimuli or state overrides
that `measure_indicator`'s real signature doesn't expose — it hardcodes a fixed 10-sentence
rotation with no override point. Building one shared interface unlocks all four at once, rather
than four one-off hacks:

```rust
// Sketch only.
pub struct ProbeProtocol {
    pub stimulus: StimulusPlan,
    pub state_override: Option<StateOverride>,
    pub expected_effect: ExpectedEffect,
    pub runtime_check: ManipulationCheck,
}
```

Needs to support: predictable vs. surprising streams; multiple labelled input categories;
controlled state injection; controlled action affordances; attention-target changes; confidence
corruption; and exact stimulus replay across baseline/target/sham/control arms (so every arm sees
identical content except for the one manipulated dimension).

## Priority order

### 1. HOT-2 — finish the confidence-injection path (cheapest legitimate gain)

Field and formula already formula-verified; only the injection path is unconfirmed. Add an
explicit, test-only confidence override and prove: the override was applied; confidence changes in
the expected direction; task correctness remains independently known; the metacognitive metric
detects the deliberate mismatch; disabling metacognition is distinguishable from merely injecting
bad confidence. **Likely gain: +1 qualified row**, without redefining the probe itself.

### 2. AE-1 — controlled agency instead of raw action diversity

Should not qualify merely by observing that Symthaea sometimes picks different actions. Build a
tiny environment with ≥2 genuine actions, different action-outcome contingencies, contingency
reversals, and matched sensory inputs where the choice actually matters. Measure: sensitivity to
consequences; adaptation after reversal; counterfactual difference between actions; diversity
conditioned on affordances, not raw randomness. Positive control: force one action ineffective or
swap outcome mappings, verify the metric responds. Rescue: reintroduce the missing action
information through a separate path, test whether adaptive choice returns. **Likely gain: +1
qualified row, and a relatively independent theory family.**

### 3. AST-1 — schema accuracy, not attention activity

Current danger: measuring "some attention signal was nonzero" rather than whether the system
models its own attentional state. Compare the system's internal attention-schema representation
against the attention mechanism's actual target/distribution; test changes from misleading or
delayed schema information. Interventions: change the real attention target; hold the schema
representation fixed; inject an incorrect schema; restore the correct schema through an alternate
route. This creates a real distinction between possessing attention, possessing information about
attention, and using that information to control cognition. **Likely gain: +1 qualified row.**

### 4. RPT-1 — build the recurrence control the formula actually needs

The frozen-input control was correctly rejected (`DegenerateGuardTest`) because a single repeated
category always returns 0 regardless of recurrence health — the formula needs ≥2 distinct-input
centroids. A proper design needs: ≥2 controlled stimulus categories; identical immediate inputs
following different histories; recurrence-preserved and recurrence-reset arms; a direct state
perturbation or known recurrent-carry signal. Key test: does identical current input produce
systematically different internal states because of prior context, and does that difference
disappear when recurrence is genuinely disabled? **Likely gain: +1, but the most implementation
work of the four.**

## Then: repair the construct-invalid probes

### GWT-3 — broadcast uptake, not module execution

`module_timings_us.gwt > 0` proves code ran, not that global broadcasting occurred. Real metric:
inject a hidden item into the workspace, test decoding/behavioral influence in language, memory,
planning, and action modules; compare local-only delivery, global delivery, and GWT-disabled
delivery; measure recipient count, uptake strength, task consequences. The important metric is
cross-module availability and use, not execution time.

### RPT-2 — binding quality, not module execution

`cross_modal_binding > 0` is the same class of proxy. Real task: paired features from two
modalities; congruent/incongruent pairs; feature swaps; delayed retrieval; unbinding/matching
accuracy. Key question: when binding is disabled, can the system still identify which feature
belonged with which object/event?

### GWT-2 — graded capacity and competition, not a boolean range check

Forcing coalition size to 0 validates the current predicate, not the theoretical interpretation.
Real experiment: vary the number of competing items; vary salience/priority; test whether selected
content is globally accessible; estimate capacity and replacement dynamics; compare against a
local-memory control. **Note even after repair**: `GWT-2` and `GWT-3` still share the identical
`enable_gwt=false` lever — they remain two outcomes from one causal intervention, not two
independent replications, regardless of how good their individual probes become.

## Separate HOT-3 and PP-1 into genuinely distinct constructs

Doesn't increase the raw qualifying count, but may increase the *real* evidence more than adding
another weak row. Both currently read `actual_effective_lr` — they need distinct probes:

- **PP-1** (predictive processing): prediction error; precision-weighted update magnitude;
  adaptation to predictable vs. volatile streams; calibration of surprise; improved future
  prediction after update.
- **HOT-3** (higher-order outcome-sensitive belief updating): confidence before an action; outcome
  observed; confidence/policy belief after the outcome; dependence on whether the outcome was
  self-generated; sensitivity to corrupted or withheld outcome information.

This lets predictive learning and higher-order outcome monitoring be tested for dissociation under
intervention, turning one shared raw field into two genuine constructs.

## Rescue tests — selective, not mandatory for all 14

A rescue arm (reintroduce the ablated information through a controlled alternate path, test
whether the function returns) should not be built for every indicator immediately. Strongest early
candidates, where the alternate information path is clear: `AST-1` (restore correct attention-state
info through an alternate channel), `AE-1` (restore action-outcome info without restoring the
original agency mechanism), `RPT-1` (inject history information without re-enabling native
recurrence), `GWT-3` (deliver workspace information to recipients directly without restoring
broadcast). A successful rescue helps establish the behavioral loss came from missing information,
not collateral damage from disabling a large subsystem.

## Sequenced campaign (if/when started)

1. Add the shared typed stimulus/override API.
2. Complete HOT-2's wrong-confidence injection.
3. Build AE-1's controlled action-contingency environment.
4. Build AST-1's attention-schema alignment probe.
5. Repair RPT-1 with multi-category, history-sensitive stimuli.
6. Split HOT-3 and PP-1 into distinct signals.
7. Replace GWT-3 and RPT-2's execution proxies.
8. Revisit GWT-2 with a graded capacity experiment.

Plausible outcome if fully executed: moving from 5 formula-verified controls / 4 statically
interpretation-eligible rows to 8-10 genuinely qualified rows and 6-8 independent evidence units.
Not guaranteed — per the honest rule above, a repaired probe finding `NotDemonstrated` instead of
support is still a successful repair.

## Explicitly not started

None of steps 1-8 above are implemented. This document is a roadmap, not a task list with any item
in progress. `BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`'s minimal runner (4 already-eligible rows) is
the actual next piece of work, and should be finished and proven correct before any part of this
campaign begins.

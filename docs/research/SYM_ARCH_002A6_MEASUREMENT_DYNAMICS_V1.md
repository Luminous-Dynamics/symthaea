# SYM-ARCH-002A6 — Online Measurement Dynamics v1

**Status:** measurement-infrastructure tranche; no architecture result

**Tracks:** #55

**Base:** SYM-ARCH-002A experimental core (#57)

## Why this exists

SYM-ARCH-001 exposed a stability/plasticity ambiguity: low measured forgetting can coexist with weak absolute acquisition. A learner that never acquires a task strongly may appear stable simply because there is little useful behavior to forget.

A6 therefore measures the *trajectory of learning itself*, not only final task scores.

It is architecture-agnostic and carries no Symthaea performance claim.

## Prequential measurement rule

Every online-learning step is ordered as:

1. present learner-visible input;
2. produce prediction **before consuming the current label**;
3. record correctness and inference latency;
4. apply the labeled update;
5. record update latency;
6. record post-update resource state.

Post-update evaluation on the same item must not be substituted for `correct_before_update` in claim-bearing acquisition traces.

This is the core anti-leakage rule for acquisition metrics.

## Trace provenance

Every `OnlineMeasurementTrace` is bound to:

- exact experiment-manifest digest;
- exact task-program digest;
- separately recorded runtime-context digest;
- human-readable contiguous `phase_id`;
- optional exact initial learner/spec-state digest;
- every ordered online step.

The trace itself receives a domain-separated BLAKE3 digest.

A continual run should normally emit a separate trace for each acquisition/adaptation phase rather than collapsing all task transitions into one global learning curve.

Examples:

- `world-a-acquisition`;
- `world-b-acquisition`;
- `post-reversal-recovery`;
- `drift-episode-03`.

## Acquisition metrics

### Examples to criterion

The criterion is frozen before claim-bearing evaluation:

- rolling-window size;
- accuracy threshold;
- number of consecutive qualifying windows.

The metric returns the number of examples observed when the sustained criterion is first reached, or `None` when it is never reached.

Requiring consecutive qualifying windows prevents one lucky local burst from being called acquisition.

The criterion must not be chosen after inspecting CONFIRM/REPL learning curves.

### Criterion admissibility against simple references

A threshold can be implemented perfectly and still be scientifically weak if a known chance, majority, marginal, or other preregistered shortcut reference can already satisfy it.

`experiment_measurement_validity::audit_acquisition_criterion` therefore freezes:

- the learning criterion;
- a preregistered reference-accuracy ceiling;
- the minimum practical excess required above that ceiling.

The audit also exposes the criterion's **finite-window resolution**. For a window of `n`, a nominal threshold is converted into the integer number correct actually required and the corresponding effective accuracy threshold.

It returns:

- `admissible` when the effective threshold clears the frozen reference ceiling by at least the required practical margin;
- `reference_confounded` otherwise, with explicit qualifiers.

This is a construct-validity guard, not a significance test. Passing it does not replace:

- A4's shortcut-control campaign;
- A2's paired/hierarchical uncertainty;
- prospective power;
- multiple-comparison control.

A claim-bearing examples-to-criterion result must not use a `reference_confounded` criterion.

### Overall prequential accuracy

Accuracy across every pre-update prediction in the phase.

This is an average over the entire learning history and is **not** described as final/terminal performance.

### Terminal-window accuracy

Accuracy over the final frozen criterion-window length. This separates end-of-phase performance from the average learning history.

### Cumulative-accuracy AUC

A6 defines normalized cumulative-accuracy AUC as the mean cumulative prequential accuracy across steps (a right-rectangle integral of the cumulative-accuracy curve).

It lies in `[0,1]` and rewards earlier acquisition. Two traces can have identical overall accuracy but different cumulative AUC when one learns useful behavior earlier.

This definition is frozen for A6; it must not later be silently replaced by ordinary mean correctness, post-update accuracy, or a different smoothing procedure.

## Latency and throughput

Each step records:

- inference latency in nanoseconds;
- update latency in nanoseconds.

Each latency series reports:

- sample count;
- total time;
- mean;
- p50;
- p95;
- observations/second when total measured time is nonzero.

Percentiles use deterministic linear interpolation over sorted observed nanosecond samples.

### Runtime-context boundary

Latency numbers are not hardware-independent model properties.

The trace therefore requires a `runtime_context_digest` representing separately frozen runtime information such as relevant CPU/GPU, operating environment, compiler/build profile, affinity/threading policy, and measurement protocol.

A latency superiority claim across different runtime-context digests requires an explicit cross-runtime comparison policy; A6 itself does not authorize one.

## Resource trace

After each update A6 records:

- trainable scalar parameters;
- total persistent state bytes;
- replay bytes;
- temporal/recurrent state bytes;
- optional process RSS.

The summary reports:

- final and peak trainable parameter counts;
- final and peak persistent state;
- peak replay state;
- peak temporal state;
- peak observed RSS.

Replay/temporal byte counts are components of persistent state and may not individually exceed the reported persistent-state total.

RSS is observational process-level memory, not a substitute for model-state accounting.

## What must be frozen before CONFIRM

For every claim-bearing acquisition comparison freeze at least:

- experiment manifest;
- task-program identity;
- phase boundaries;
- prequential ordering rule;
- learning criterion;
- reference-accuracy ceiling used to audit the criterion;
- minimum practical excess above that reference;
- observation budget;
- evaluation cadence;
- latency timing protocol;
- runtime-context schema;
- resource-accounting semantics;
- primary acquisition metric;
- comparator and SESOI;
- statistical analysis from A2.

DEV may be used to choose these values. CONFIRM/REPL may not tune them.

## Relationship to the R matrix

A6 does not replace `R[t_train][t_eval]`.

The two answer different questions:

- R matrix: what is retained/transferred across tasks?
- A6 trace: how quickly and efficiently is behavior acquired or recovered inside a phase?

A system with low forgetting but poor acquisition should therefore be visible as:

- apparently favorable retention/forgetting summaries;
- slow or absent examples-to-criterion;
- poor cumulative-accuracy AUC;
- weak terminal-window accuracy.

That combination must not be described as strong continual learning.

## Acceptance tests

The exact PR head must demonstrate:

1. invalid trace provenance fails closed;
2. sustained rolling criterion is required;
3. criterion miss returns `None` rather than an invented latency;
4. earlier acquisition raises cumulative AUC at matched overall accuracy;
5. terminal-window accuracy is distinct from overall prequential accuracy;
6. latency p50/p95 and throughput summaries are deterministic;
7. zero measured latency cannot create infinite throughput;
8. resource peaks/finals are reported correctly;
9. impossible replay/persistent-state accounting fails closed;
10. trace digest changes with order, manifest, task, runtime context, or phase;
11. summary binds acquisition, latency, resource, and trace identity;
12. criterion audit distinguishes admissible from reference-confounded thresholds;
13. finite-window criterion resolution is explicit;
14. the full psych-bench library compiles.

## Claim ceiling

Merging A6 supports only:

> Symthaea psych-bench contains provenance-bound prequential acquisition, criterion-validity, latency/throughput, and resource measurement primitives suitable for later preregistered continual-learning experiments.

It does **not** support:

- a Symthaea performance claim;
- a claim that any mechanism learns faster;
- a hardware-independent latency claim;
- a resource-efficiency claim across unmatched budgets;
- a claim that low forgetting implies successful learning;
- a claim-bearing acquisition threshold when the criterion audit is `reference_confounded`.

## Next use

Once A-series infrastructure is executable and green, use A6 traces in DEV with B1 and later baselines to identify sensible frozen acquisition criteria and estimate effect/variance structure before opening CONFIRM.

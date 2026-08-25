# SYM-ARCH-001 — Continual Compositional Adaptation

**Status:** preregistered experiment design, to be executed by the pull-request workflow before interpreting the result.

## Research question

Does the current Symthaea HDC-LTC mechanism provide a practically meaningful advantage over simpler controls on a shared continual-learning task family when data, dimensionality, update opportunities, and evaluation splits are held fixed?

This experiment is intentionally narrower than any claim about AGI, consciousness, language-model capability, or full live Symthaea. It is a mechanism-level architecture discrimination test.

## Why this is a new work unit

The frozen UAL-P4a-v2 result tested a benchmark-local HDC learner and did **not** demonstrate held-out compositional generalization in its qualified blocked arm. That result remains untouched.

The earlier live UAL-P1 reversal diagnostic also remains untouched. Its follow-up trace established that the `CognitiveLoopBenchmarkRunner` path based on `cycle_with_hv()` does not consume `provide_reward()` and therefore cannot support a confirmatory reward-driven reversal claim without separate production-harness work.

SYM-ARCH-001 therefore bypasses that invalid reward path and uses direct supervised online updates for every compared mechanism.

## Systems under test

All systems see the exact same deterministic encoded inputs in the exact same order for each seed.

1. `linear_sgd` — online linear logistic/SGD control.
2. `vanilla_hdc` — raw compositional HDC representation with an associative prototype readout.
3. `fixed_diagonal_ssm` — the existing Symthaea diagonal SSM state transform with the same associative prototype readout.
4. `hdc_ltc_frozen` — HDC-LTC temporal dynamics with frozen HDC-LTC weights and the same prototype readout.
5. `hdc_ltc_hebbian` — **candidate**: HDC-LTC temporal dynamics plus existing Hebbian plasticity and the same prototype readout.

The experiment does **not** claim that `linear_sgd` or `fixed_diagonal_ssm` is a frontier transformer/GRU/SSM baseline. A later external-baseline tranche is required before making industry-level efficiency claims.

## Task family

There are four sequential relational worlds, each operating on two four-valued factors. Inputs bind:

- a shared value hypervector,
- a shape role,
- a texture role, and
- a world context.

Each world defines a deterministic balanced binary relation over the 16 possible factor pairs. For every world:

- 12 combinations are training items: 6 positive, 6 negative;
- 4 combinations are held out: 2 positive, 2 negative;
- held-out items never appear during that world's training.

Worlds are learned strictly A → B → C → D. There is **no replay** of prior worlds while a later world is trained.

## Primary phenomena

### 1. Continual retention

After all four worlds are learned, compute mean accuracy on each world's 12 trained combinations.

Primary metric: `final_retention_accuracy`.

### 2. Compositional transfer

After all four worlds are learned, score the four balanced held-out combinations from every world.

Primary metric: `heldout_compositional_accuracy`.

### 3. Forgetting

After each world phase, evaluate every world seen so far without mutating agent state. For each world record its best observed trained-set accuracy, then subtract its final trained-set accuracy.

Primary metric: `mean_forgetting` (lower is better).

### 4. Contingency reversal

Clone the post-training state so reversal cannot contaminate the retention/composition measurements. In world D, invert the relation labels and present all 16 combinations repeatedly in deterministic shuffled order.

Predictions are scored **before** each supervised update.

Metrics:

- final accuracy over the trailing 32 predictions;
- first trial whose trailing 32-trial window reaches at least 75% accuracy; if never reached, report `reversal_trials + 1`.

This is a contingency-reversal test, not yet a full causal-intervention/do-calculus test.

## Fixed campaign configuration

- representation dimension: **512**;
- seeds: **16**;
- training epochs per world: **16**;
- reversal epochs: **12**;
- prototype update alpha: **0.15**;
- candidate Hebbian learning rate: **0.002**;
- practical win margin: **0.05** absolute accuracy;
- tolerated regression margin: **0.05** absolute accuracy.

Changing any of these after seeing the PR result creates a new experiment version rather than modifying the interpretation of SYM-ARCH-001 v1.

## Decision rule

Candidate: `hdc_ltc_hebbian`.

For retention, composition, and reversal-final accuracy, compare the candidate mean against the **strongest control mean for that metric**.

- **PASS:** candidate wins by at least 0.05 on at least two of the three target phenomena, has no target regression worse than 0.05, and mean forgetting is not worse than the best control by more than 0.05.
- **MIXED:** at least one target win of 0.05 or more, no target regression worse than 0.05, and no forgetting regression beyond tolerance.
- **NEGATIVE:** at least two target regressions worse than 0.05, or forgetting is worse than the best control by more than 0.05.
- **NULL:** all other outcomes.

The workflow reports this verdict mechanically. The threshold is not changed after results are observed.

## Statistics and evidence

For each agent and primary metric, report:

- all per-seed values;
- mean;
- normal-approximation 95% confidence interval across seeds.

The PR workflow emits `artifacts/sym-arch-001/report.json` and uploads it as a GitHub Actions artifact.

Wall-clock runtime is recorded as observational evidence only. The reproducible resource anchors are:

- representation dimension;
- number of training/update observations per agent;
- SSM state width.

No energy or parameter-count superiority claim is licensed by this first tranche.

## Claims this experiment can license

At most:

> Under the preregistered SYM-ARCH-001 synthetic continual-relational task family, the tested HDC-LTC+Hebbian mechanism did/did not show a practically meaningful advantage over the included controls.

It cannot by itself license:

- "Symthaea beats transformers";
- "Symthaea has solved continual learning";
- "Symthaea has demonstrated AGI";
- "Symthaea is more compute-efficient than frontier AI";
- a causal reasoning claim stronger than contingency adaptation.

## Follow-up if informative

A positive or mixed result should be followed by a separate preregistered tranche with stronger external baselines (small GRU/LSTM, trainable SSM, small transformer where practical), explicit compute/memory accounting, larger held-out relational families, and a genuine intervention-based causal adaptation task.

A null or negative result should be treated as architecture guidance: identify which ablation fails, preserve the evidence, and change the mechanism rather than loosening this experiment's criteria.

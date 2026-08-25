# SYM-ARCH-002A Experimental Core v1

**Status:** infrastructure-only implementation tranche  
**Scientific claim status:** none  
**Parent plan:** GitHub issue #55  

## Purpose

SYM-ARCH-002A builds reusable experimental infrastructure before adding stronger baselines or new Symthaea mechanisms. The goal is to make later architecture results difficult to overstate, accidentally contaminate, or attribute to the wrong source of variation.

This tranche deliberately does **not** run a confirmatory architecture comparison and does **not** alter SYM-ARCH-001 / PR #53.

## Added contracts

### Experiment manifest

`ExperimentManifest` records:

- experiment id/version;
- exact code revision;
- preregistration and generator digests;
- `DEV`, `CONFIRM`, or `REPL` namespace;
- tuning status;
- whether behavioral results from that namespace were already observed;
- independent environment / representation / learner / stream seed namespaces;
- primary hypothesis and comparator;
- smallest effect size of interest (SESOI).

Validation fails closed when, for example, a `confirmatory_first_use` run claims a previously observed `CONFIRM` stream.

### TaskProgram identity

`TaskProgram` is a typed, hashable description of an experimental world. v1 carries:

- task family and program id;
- a small typed logical rule AST;
- context visibility (`explicit`, `latent`, `task_free`);
- timing regime;
- positive/negative example counts;
- explicit train/evaluation support descriptions;
- oracle digest.

The runtime/generator remains separate. The program is the auditable identity and ground-truth contract that a generator/runtime must implement.

`validate_task_support` adds an explicit support-overlap policy. Empty or duplicate support descriptors are rejected, and confirmatory OOD/compositional tasks can fail closed on exact train/evaluation support overlap. Experiments that intentionally evaluate IID support must declare that overlap rather than receiving it silently.

### Continual-learning matrix

`PerformanceMatrix` uses the explicit convention:

- `T` tasks;
- `T + 1` rows;
- row 0 = pre-training evaluation;
- row `i + 1` = evaluation after training task `i`.

It exposes:

- final accuracy;
- average incremental accuracy;
- backward transfer;
- forward transfer;
- mean forgetting.

The matrix rejects non-finite, out-of-range, ragged, or dimensionally ambiguous inputs.

### Paired uncertainty and the independent unit

`paired_delta_bca` computes candidate-minus-control deltas **before** resampling and reuses the existing `psych-bench` BCa bootstrap implementation.

For confirmatory work, `paired_environment_delta_bca` is the stricter entry point. It requires one already-aggregated candidate/control observation per independently generated environment, binds each pair to a unique environment digest, rejects duplicate environment identities, and therefore prevents representation/learner/stream reruns from being flattened into pseudoreplicates.

Hierarchical resampling remains a follow-up once the nested environment/run result schema is concrete. Until then, confirmatory inference treats the environment aggregate as the independent observation.

### SESOI / practical-effect classification

`classify_practical_effect` interprets the complete paired confidence interval against a preregistered, strictly positive SESOI:

- `meaningful_gain` only when the entire interval exceeds `+SESOI`;
- `meaningful_regression` only when the entire interval is below `-SESOI`;
- `equivalent` only when the entire interval lies inside `[-SESOI, +SESOI]`;
- otherwise `inconclusive`.

A favorable point estimate alone therefore cannot establish a meaningful gain, and a low-powered null result cannot be mislabeled as evidence of equivalence.

### Claim ledger

`ClaimLedgerEntry` keeps independent evidence dimensions separate:

- outcome (`supported`, `equivalent`, `not_demonstrated`, `contradicted`, `inconclusive`);
- support kind (`architectural_only`, `observed`, `ablation_causal`, `intervention_causal`, `functionally_supported`);
- generalization (`dev_only`, `iid_confirm`, `ood_composition`, `ood_family`);
- replication state;
- resource state;
- provenance validity;
- explicit qualifiers.

There is intentionally **no scalar aggregate** across these dimensions.

`wording_ceiling()` prevents invalid or unsupported evidence from being surfaced with causal wording and reserves `replicated_causal` for valid, supported, causally identified evidence that replicated beyond DEV and is not carrying a resource regression.

## Evidence philosophy

This follows the same core lesson as the Butlin evidence-tier redesign: architectural existence, observed behavior, causal support, functional support, contradiction, and measurement failure are different kinds of evidence and must not be averaged into one flattering score.

## Acceptance gate for this PR

This tranche is acceptable when the exact PR head passes:

1. focused `experiment` and confirmatory-helper unit tests;
2. `cargo check -p symthaea-psych-bench --lib`;
3. changed-file rustfmt;
4. deterministic manifest/task hashes;
5. fail-closed confirmation-status tests;
6. reference-value continual-metric tests;
7. unique-environment paired-inference tests;
8. SESOI interval-classification tests;
9. train/evaluation support-policy tests;
10. ClaimLedger wording-ceiling tests.

No architecture-performance result is required for merge because this PR is measurement infrastructure, not confirmatory evidence.

## Explicit non-claims / deferred work

This v1 does not yet provide:

- procedural task execution;
- oracle execution or benchmark mutation tests;
- hierarchical environment-first bootstrap over nested runs;
- prospective power simulation;
- resource Pareto-front computation;
- diagnostic state traces or causal state patching;
- strong neural/SSM/Mamba baselines;
- a confirmatory seed manifest.

Those should land in subsequent focused 002A/002B tranches rather than turning this foundation PR into another monolith.

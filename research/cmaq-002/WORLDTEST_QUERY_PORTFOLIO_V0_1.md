# WorldTest Cross-Query Portfolio Evaluator v0.1

**Status:** bounded Rust evaluator and 25 deterministic tests authored; exact-head Rust 1.96 qualification pending.  
**Issue:** #13689.  
**Parent track:** WorldTest / AutumnBench adapter #13688.

## Research question

Does one frozen reward-free exploration history produce a reusable environment model that supports all three WorldTest query families—masked-frame prediction, change detection and planning—or does an exploration policy specialize in the query family it expects?

The primary comparison holds the candidate artifact, model checkpoint, exploration program/trace, exploration-derived checkpoint, seed, split, evaluator and exploration resource envelope fixed. The same learned state is then evaluated against a complete preregistered task inventory spanning all three query families.

This is deliberately stronger than averaging three unrelated single-family results. It asks whether the *same learned environment knowledge* transfers across query types.

## Input and evidence contract

A portfolio binds:
- candidate artifact and model/checkpoint;
- adapter/runtime and evaluator profile;
- exploration program, exact trace and frozen post-exploration checkpoint;
- seed commitment, budget manifest and evaluation split;
- measured exploration actions, resets, model queries, elapsed time and optional energy;
- separate action, reset, total-interaction, query, time and energy ceilings;
- one frozen family-specific scoring/normalization profile per query family;
- a complete expected task-ID set frozen before confirmatory results are opened.

Each test row binds its task ID, unique run ID, query family, evaluator-assigned `environment_family_id`, independent `environment_cluster_id`, challenge-program digest, family metric-profile digest, candidate artifact, shared exploration trace, shared exploration checkpoint, evaluator-assigned normalized score/disposition, and test resource usage. Multiple task rows from the same generated environment instance must reuse one cluster ID; they are not independent experimental subjects.

A challenge program must differ from the exploration program. The three test tasks can share a program digest where the benchmark defines multiple queries against one challenge environment; program uniqueness is therefore not required across tasks. They must nevertheless remain separate task/run records.

## Score semantics

Each task's score is an evaluator-supplied integer in [0, 1,000,000] on a frozen family-specific normalized utility scale, with higher values better. It is not recalibrated on the confirmatory cohort.

For family (f), the evaluator reports:

[
S_f = \frac{1}{N_f}\sum_{j=1}^{N_f} s_{fj},
]

using integer floor division. Failed, timed-out, invalid and abstained tasks require explicit evaluator-assigned scores and remain in (N_f). A missing task or family rejects the portfolio.

The certificate preserves sorted per-task family/cluster identity, per-cluster task counts and per-query-family counts within each cluster, plus the total number of distinct environment families and independent clusters. It therefore retains a statistical grouping key for cluster-aware downstream analysis; it does not itself compute confidence intervals or infer statistical significance. A cluster ID may map to only one environment-family ID.

The certificate reports the vector

[
(S_{\mathrm{MFP}}, S_{\mathrm{change}}, S_{\mathrm{planning}})
]

and the worst-family floor

[
S_{\min}=\min_f S_f,
]

plus the max-minus-min family gap. It does **not** report a grand mean across all three query families as the primary metric. The minimum-family descriptor exposes specialization but does not replace the vector or imply that the families have identical scientific meaning.

## Resource accounting

Exploration resources are charged once to the shared exploration history. Test interactions, model queries, elapsed time and optional energy are accumulated separately across query-task runs. The portfolio refuses an exploration or test budget overrun. If an energy ceiling is declared, every required measurement must be present.

This prevents one shared exploration run from being counted three times, while also preventing the follow-on query workloads from being treated as free.

## Falsification and controls

The Rust suite includes:
- one high-scoring query family with two weak families; the minimum-family floor must expose specialization;
- equal family scores and zero family gap;
- shuffled task order yielding the same canonical certificate;
- missing family/task and unexpected/duplicate task;
- duplicate run IDs;
- missing or untrimmed environment-family/cluster IDs;
- multiple query-task rows sharing one independent cluster are counted once for cluster-level analysis;
- one environment cluster assigned to conflicting environment families is rejected;
- same exploration/challenge program rejection;
- changed shared trace, checkpoint and candidate artifact rejection;
- mismatched family metric profile rejection;
- failed and abstained tasks retaining their assigned score in the family denominator;
- separate exploration versus test cost totals;
- test and total-exploration budget overruns;
- required energy missing;
- out-of-range normalized score.

No task outcome is silently omitted, and no family can disappear behind a pooled average.

## Experiment plan

Use the same exploration budget to compare:
1. no exploration / reactive baseline;
2. fixed-script and seeded random exploration;
3. novelty/entropy-driven exploration;
4. Bayesian information-gain exploration;
5. task-loss EVoI exploration;
6. query-specific exploration as a diagnostic upper bound only—not a shared-history result;
7. evaluator-only environment oracle, never candidate-visible.

For every shared-history arm, evaluate all three query families from the exact same frozen exploration checkpoint. Split by environment family and challenge-program family, not by task rows; keep each environment family intact across train/validation/test.

The primary results are family-specific means, worst-family floor, family gap and paired environment-family uncertainty. The task evaluator now preserves `environment_family_id` and `environment_cluster_id`; inference must resample/aggregate at the independent environment-cluster level, keep whole environment families intact for splits, and avoid treating task rows as independent. Report cluster counts and per-cell denominators alongside uncertainty. Report exploration actions, resets, go-to-test, model queries, time, memory and energy alongside downstream score. Do not reward raw information gain or reset count unless it translates into better held-out query performance.

## Ownership and claim ceiling

- #8908 owns EVoI exploration-policy semantics.
- #9094 owns matched external benchmark and resource comparisons.
- #9176 owns shared frozen-run manifests.
- #9437 owns evaluator/agent OS isolation and answer-key custody.
- #13688 owns WorldTest source, protocol and trace custody.

This evaluator scores supplied task outcomes. It does not run AutumnBench, authenticate receipts, prove causal understanding, establish a model advantage, perform statistical inference or claim SOTA.

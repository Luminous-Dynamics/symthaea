# WorldTest Frozen Mechanism-Transfer Evaluator v0.1

**Status:** Rust evaluator and 19 new deterministic tests authored; Rust 1.96 qualification passed at `86b9d7981637917d3060dd65514f130209d920ae`. CMAQ-002 is now pinned to Rust 1.100.0; exact-head Rust 1.100.0 qualification for this revision is pending.  
**Research owner:** [#13707 — CMAQ-018](https://github.com/Luminous-Dynamics/symthaea/issues/13707).  
**Base evaluator:** [#13711 — cluster-aware query portfolio](https://github.com/Luminous-Dynamics/symthaea/pull/13711).  
**Parent protocol:** [#13688 — WorldTest / AutumnBench adapter](https://github.com/Luminous-Dynamics/symthaea/pull/13688).

## Research question

Does one frozen, reward-free exploration history produce an environmental representation that remains useful across both different query families and held-out source-to-target mechanism relationships?

The evaluator crosses five transfer conditions with all three WorldTest query families, producing fifteen required cells:

\[
\mathcal S = \{S_{m,q}:m\in\mathcal M, q\in\mathcal Q\}
\]

where \(\mathcal M\) contains the five mechanism conditions and \(\mathcal Q\) contains masked-frame prediction, change detection and planning.

This is the **frozen-transfer / zero-shot reuse** evaluator. It does not implement the budgeted-adaptation track.

## Transfer-condition semantics

Each condition is bound to an evaluator-owned profile manifest, its exact `target_environment_family_id`, mechanism/observation relation types, and a digest. Same-family, nuisance and parameter conditions bind the exact source-family ID; compositional and structural conditions bind exact distinct evaluator-assigned target-family IDs. Every task row must match the condition profile's exact target family, not merely claim to be different from the source. The signed evaluator manifest must establish that the actual generator transformation conforms to the declared relation.

| Condition | Mechanism relation | Observation relation | Target family identity |
|---|---|---|---|
| Same-family holdout | Invariant | Invariant | Same source family; new independent instances/clusters |
| Nuisance shift | Invariant | Nuisance perturbed | Same source family |
| Parameter shift | Parameters perturbed, structure preserved | Invariant | Same source family |
| Compositional shift | Held-out combination of known mechanisms | Invariant | New composition family |
| Structural shift | New/replaced causal or transition structure | Invariant | New structural family |

Do not relabel a structural change as a nuisance shift. The source generator manifest, transfer design manifest, and one condition-profile manifest per condition are digest-bound in the context. The profile must also identify which relations are invariant, perturbed, composed or replaced.

## Frozen-source contract

One `WorldTestPortfolioContext` is reused unchanged for all fifteen cells. It binds:

- candidate artifact and model/checkpoint digests;
- adapter/runtime and evaluator profile;
- source exploration program, trace and checkpoint;
- seed commitment, budget manifest and evaluation split;
- measured exploration actions, resets, model queries, time and optional energy;
- resource limits and all three query-family metric profiles.

Every target task starts from that exact frozen source checkpoint. A target task cannot update the shared state used by a sibling task. The base #13711 evaluator validates candidate, trace, checkpoint, family-profile, task/run, cluster/family and resource provenance. A changed shared checkpoint or trace rejects the transfer portfolio.

The source environment-family identity and source/transfer manifest digests are separately required. Same-family, nuisance and parameter conditions must retain the source family identity. Compositional and structural conditions must use a distinct evaluator-assigned target family identity.

Environment identities and condition labels must originate in trusted generator/evaluator manifests and be bound by signed receipts. They are not candidate self-assertions.

## Independent unit and cell scoring

The independent experimental unit is the environment/mechanism-generation cluster—not the task row, action, rollout step or repeated query.

For query family \(q\), condition \(m\), and cluster \(c\), first compute the within-cluster mean:

\[
\bar s_{m,q,c}
=
\frac{1}{n_{m,q,c}}
\sum_{i=1}^{n_{m,q,c}}s_{m,q,c,i}
\]

Then compute the primary descriptive cell score with equal cluster weighting:

\[
S_{m,q}
=
\frac{1}{C_{m,q}}
\sum_{c=1}^{C_{m,q}}\bar s_{m,q,c}
\]

The task-row-weighted cell mean is retained only as a diagnostic. It is not the primary result because clusters with more task rows must not receive greater inferential weight.

The evaluator preserves:
- task and independent-cluster counts per cell;
- completed, failed, timed-out, invalid and abstained counts;
- task-weighted diagnostic mean;
- equal-cluster primary mean;
- minimum and maximum cluster means;
- per-cluster per-query-family task counts, score sums and means.

Failed, timed-out, invalid and abstained tasks stay in the cluster and cell denominators with explicit evaluator-assigned scores. Missing cluster/query-family cells remain absent, not zero-filled.

The preregistered minimum independent cluster count is a context field and is checked separately for all fifteen cells. The tests use two clusters per cell only as a mechanical fixture. Actual scientific sample size must follow the power/MDE analysis and the #8908 preregistration; the existing ten-seed-per-synthetic-lane requirement is a starting floor, not a universal guarantee of power.

## Robustness reporting

Report the complete fifteen-cell score matrix and all denominators. The certificate also gives the minimum and maximum cell and their gap:

\[
S_{\min}=\min_{m,q}S_{m,q}
\]

There is **no grand average across query families or mechanism conditions**. The minimum cell is a descriptive robustness floor, not a universal intelligence score or a confirmatory statistical test.

Superiority claims belong to the #8578 cluster-level inference layer. Use paired candidate/baseline cluster deltas and simultaneous confidence bounds across the preregistered cell family. Do not infer significance from task rows, or select the reported worst cell after inspecting results without multiplicity control.

## Exploration and representation arms

At matched action, reset, model-query, compute, memory and time budgets, compare:

1. No exploration / reactive baseline.
2. Seeded random and fixed-script exploration.
3. Novelty/entropy exploration.
4. Bayesian information-gain exploration.
5. Task-loss EVoI.
6. Shared-history EVoI.
7. Query-specific exploration as a diagnostic upper bound only; it is not a shared-history result.
8. Evaluator-only environment oracle; never candidate-visible.

Ablate:
- tabular/linear mechanism models;
- matched recurrent latent-dynamics/RSSM baselines;
- HDC-only;
- CfC/LTC-only;
- HDC + CfC/LTC with mechanism, instance and dynamical state separated;
- event-aware temporal segmentation (EAWM-inspired control).

No representation is presumed superior. Query adapters, splits, frozen-state semantics, interaction budgets and compute budgets must remain matched.

## Required qualification fixtures

The Rust tests cover:
- full fifteen-cell coverage and deterministic ordering;
- one weak structural-shift/planning cell exposed as the minimum;
- missing cell rejection;
- preregistered minimum clusters per cell;
- repeated tasks within one cluster do not create extra independent clusters;
- task-weighted and equal-cluster means differ as expected;
- one cluster cannot span transfer conditions;
- one cluster can cover multiple query families within the same condition;
- task-order invariance;
- failure/abstention retention in cluster and cell denominators;
- source family identity and source/transfer manifest digest validation;
- complete, unique, nonempty condition-profile validation;
- exact target-family identity is bound by the condition profile and checked on every task row;
- profile relation semantics must match the named condition;
- exact source-to-target family identity must match the evaluator-declared target family ID;
- mutated shared checkpoint is rejected by the base evaluator;
- shared exploration cost is charged once, separately from target-query costs.

## External benchmark positioning

- **WorldTest / AutumnBench (ICML 2026):** closest environment-level query track. Official paper describes 43 environments and 129 tasks across three query families. Do not claim reproduction until archive hashes, file inventory and license are verified.
- **EAWM (ICLR 2026):** event-aware representation is an ablation/comparator on its published Atari, Craftax and DeepMind Control tasks, not a score directly comparable to AutumnBench.
- **MRS.Q (ICML 2026):** planner-selected model error and pessimistic value estimation belong in the separate optimizer-exploitability stress track (#9396).
- **TD-MPC2 (ICLR 2024):** strong continuous-control baseline, evaluated in a separate numeric-control track with exact environment and resource matching.
- **WorldArena 2.0:** later visuo-tactile, interactive and real-robot qualification; do not pool its scores with grid-world query metrics.

## Adaptation is a separate experiment

This evaluator contains no target-specific learning or environment interaction after source exploration. A future adaptation evaluator must:
- create an isolated checkpoint copy per target cell;
- preregister and equalize each cell's adaptation budget;
- report initial frozen-transfer score, final post-adaptation score, delta and cost separately;
- preserve failed/abstained target episodes and verify that one target cell cannot mutate another cell's checkpoint.

Never pool frozen-transfer and adapted scores.

## Claim ceiling

A passing evaluator establishes that the fifteen cells, identities, cluster summaries, resource budgets and score semantics are internally complete. It does not establish causal understanding, benchmark performance, SOTA, real-world transfer, physical safety or a valid blind evaluation.

Blind execution remains blocked until #9437's actual OS/process isolation, oracle confidentiality and signed evidence controls are qualified. Dataset admission remains blocked until the official AutumnBench archive bytes and license are verified.

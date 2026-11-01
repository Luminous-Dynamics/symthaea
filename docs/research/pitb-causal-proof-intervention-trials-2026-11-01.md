# PITB research track: paired causal proof-intervention experiments (2026-11-01)

**Status:** preregistered-plan schema, deterministic assignment verifier and synthetic plan fixture added. No proof outcomes have been collected and no causal effect is measured.

## Research question

Does a PITB intervention actually improve the probability of an independently verified proof, or does it merely correlate with easier tasks, more context, more tokens, more attempts or favorable execution order?

Observational proof trajectories cannot answer this cleanly. A controller chooses interventions based on prior failures and proof-state features; those choices are confounded by difficulty and search history. PITB needs a randomized, matched intervention protocol that preserves the exact initial proof state and resource envelope across arms.

## First experiment: support-antichain causal ablation

Run four arms on the same task × initial proof-state × model-seed unit:

| Arm | Method | Interpretation |
|---|---|---|
| A0 | Full context + standard retrieval | Control |
| A1 | Predicted minimal-support antichain context | Primary PITB intervention |
| A2 | Oracle minimal support from exhaustive small-context lattice | Diagnostic upper bound only |
| A3 | Size-matched random support subset | Negative control for context-size reduction |

The primary contrast is A1 versus A0. A2 measures the remaining gap to an exhaustive oracle. A3 tests whether gains come merely from deleting context rather than selecting useful declarations.

Keep A2 to small audited contexts where exhaustive support lattices are affordable (initially 2–6 mutable declarations, with fixed context held constant). Never train on A2 oracle labels from the test cohort.

## Paired unit and isolation

One unit binds task ID, project/component ID, repository commit, exact target statement, initial proof-state receipt and model seed. Every arm starts from an independent copy of that exact initial state. Use a fresh Lean process per arm, discard mutable tactic/typeclass caches, and bind all arms to the same environment, Lean toolchain, model/version, seed, prompt/configuration, continuation policy, fixed evaluation horizon and token/model/tool/wall/Lean-CPU/cost caps.

Arm execution order is deterministically randomized using a frozen seed and domain-separated SHA-256 ordering. This is a reproducible carryover/cache control. It does not replace matched states or justify claims from observational logs. Never run paired arms in one shared mutable Lean process.

## Estimand and analysis

The primary estimand is the paired intention-to-treat risk difference:

\[
\widehat{\Delta}_{ITT}=\frac{1}{N}\sum_{i=1}^{N}(Y_{i,A1}-Y_{i,A0}),
\]

where Y=1 only for an independently replayed proof of the exact target passing kernel, semantic and applicable successor gates within the fixed horizon and budget.

A missing run, timeout, tool-unavailable result, rejected proof or no attempt remains in the assigned denominator as zero. Do not condition on attempts completed or tasks that reached a checker.

Secondary measures: paired cost difference; cost per verified proof; cost-quality AUC; success by project/family/difficulty; context declarations and tokens retained; retrieval precision/recall; Lean CPU, wall time, memory and branch work; support prediction precision/recall/exact-route match; oracle gap A2−A1; random-control gap A1−A3.

Use component/project-cluster bootstrap intervals and a preregistered paired randomization/permutation test. Resample by component ID, not theorem rows. Publish pair-level discordance and component-level results. Correct for multiplicity if more than one contrast is confirmatory. The controller's action-conditioned probabilities are not causal effect estimates; policy value comes from the paired experiment.

## Other intervention trials

Run separate preregistered trials for:
1. R5 live post-elaboration snapshot restore versus R4 cold reconstruction, holding branch search and proof policy constant.
2. Typed proof-repair operations versus source patching and whole-module rewriting, holding target blueprint and budget constant.
3. Multi-action adaptive routing versus fixed-step and threshold-only controls, holding base prover and total budget constant.
4. Full Lean diagnostics versus a frozen minimal diagnostic representation.
5. Query rewriting plus graph traversal versus each retrieval component alone.

Do not start with a giant factorial trial. Estimate each effect cleanly before testing interactions.

## Preregistration contract added

- data/benchmarks/proof-insight-transfer-v0/intervention-trials/proof-intervention-trial-plan-v0.schema.json
- data/benchmarks/proof-insight-transfer-v0/intervention-trials/fixtures/support-antichain-intervention.synthetic.json
- crates/bridges/symthaea-proof-insights/src/bin/verify_proof_intervention_trial.rs

The verifier recomputes each pair ID from exact task/component/state/statement/model-seed identities; recomputes deterministic randomized arm order; requires complete arm coverage; rejects duplicate units and insufficient component clusters; requires one control and at least one intervention; marks oracle arms diagnostic-only; and enforces fresh processes, cloned initial states, discarded mutable caches and matched environment/model seed/resource budgets.

The fixture is synthetic contract data, not theorem attempts or measured outcomes.

## Qualification targets (not results)

- A1 improves paired semantic-gated success over A0 by ≥5 percentage points, with the component-cluster interval excluding zero.
- A1 reduces cost per semantic-gated success by ≥20% at non-inferior success.
- A1 outperforms size-matched random support A3 by ≥3 percentage points.
- A1 recovers ≥70% of the success gap between full context and oracle support while retaining ≤50% of mutable declarations on the small-context subset.
- Same-direction effects on SorryDB, the frozen research-family split and a later/independent Aegis cohort.
- Zero false accepts across target/state/seed/arm/order/budget/receipt mutations.

A null result is valuable: it distinguishes the value of semantic support selection from generic context reduction.

## Remaining work

1. Execute schema and Rust verifier checks at the exact PR head.
2. Add an outcome-receipt schema requiring one exact run and independent checker/semantic/successor receipts per pair × arm.
3. Implement paired outcome analysis with cluster bootstrap, randomization inference, missingness checks and cost/success frontiers.
4. Generate a real exhaustive support lattice from a pinned Lean context and build the first reviewed cohort.
5. Run a small validation smoke trial; freeze models, prompts, contexts, policy and caps before held-out testing.

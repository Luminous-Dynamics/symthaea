# Symthaea RSI: Causal Lineage Attribution Protocol V0

**Date:** 2026-11-06  
**Status:** Research specification; no executable benchmark or causal result is claimed.  
**Owner:** #10717 — SYM-RSI-CAUSAL-LINEAGE-001  
**Scope:** Intervention-level attribution for evolutionary and multi-parent RSI candidates.

## 1. Research question

When an evolved candidate improves, which inherited change caused the improvement, which changes require interaction, and does the proposed mechanism survive independent transplant and replication?

A lineage graph records ancestry. Source/build provenance records what code produced an executable. Paired evaluation records an outcome difference. None of these, alone or together, identifies the causal contribution of a parent or patch.

The protocol supplies controlled intervention contrasts that consume the existing lineage, source/build, receipt, evaluator, and statistical infrastructure. It does not introduce another candidate archive, receipt schema, evaluator, signer, or promotion authority.

## 2. Evidence and claim ladder

The following claims must remain distinct:

| Level | Claim | Minimum evidence |
|---|---|---|
| L0 | Lineage | Parent/child identities and append-only ancestry |
| L1 | Change provenance | Exact source diff and source-to-executable binding |
| L2 | Outcome association | Matched child-versus-parent evaluation |
| L3 | Causal contribution | Same-base intervention contrast with identified treatment |
| L4 | Mechanism support | Preregistered intermediate prediction observed on the exact executable |
| L5 | Replicated mechanism | Independent seed/cluster replication and clean transplant or rescue |
| L6 | Transfer | Replicated effect on fresh task families or a different domain |

A result must not be described at L3–L6 when its evidence only supports L0–L2. Parentage, commit order, narrative plausibility, or a higher score is not causal attribution.

Causality and novelty are also separate. A previously known method can have a real causal effect in Symthaea; a novel-looking patch may have no effect. A scoped prior-art search can support “not found in the declared search” but cannot establish universal novelty.

## 3. Define the intervention unit before running it

The intervention manifest is a versioned experiment specification, **not** a new benchmark receipt or trust envelope. It must bind or reference:

- Frozen base candidate and all declared parent identities.
- Exact change units, source diff digests, affected symbols/files, origin parent, and proposed mechanism.
- A dependency/conflict graph. Semantically inseparable edits are one coupled block, not fabricated independent factors.
- Rollback, transplant, and (where meaningful) rescue procedures.
- Source-tree and executable/build identities, using #10569 and the existing evidence owners.
- Frozen checkpoint/model, harness/toolchain, task and cluster population, evaluator/verifier, split, seed plan, budget and analysis plan.
- Every planned and attempted variant, including build failures, invalid variants, rejected variants and incomplete runs.

A Git commit, file, PR, or parent edge is not automatically a valid causal unit. The change unit must correspond to an operation that can be independently manipulated and interpreted.

## 4. Same-base counterfactual design

### 4.1 Two-parent hybrid: full 2×2

For two independently addressable contributions A and B, generate all four siblings from the exact same frozen base:

| Variant | A | B | Meaning |
|---|---:|---:|---|
| Y00 | 0 | 0 | Base |
| Y10 | 1 | 0 | A only |
| Y01 | 0 | 1 | B only |
| Y11 | 1 | 1 | Hybrid A+B |

For a scalar outcome with higher values better, predeclare:

- A marginal effect: `[(Y10 + Y11) - (Y00 + Y01)] / 2`
- B marginal effect: `[(Y01 + Y11) - (Y00 + Y10)] / 2`
- A×B interaction: `Y11 - Y10 - Y01 + Y00`

Also report conditional effects (A when B=0 and A when B=1; likewise for B). A positive interaction means the joint intervention contributes beyond the sum of the isolated effects on the declared scale. Negative interactions are equally important.

These contrasts identify effects only for the declared task population, implementation, metric, and resource envelope. They do not imply universal parent quality.

### 4.2 More than two change units

For larger candidate sets, preregister a full factorial when feasible or a screening design whose alias structure is explicit. Confirm selected main effects and interactions with an identifiable follow-up design. An underpowered or aliased pilot cannot establish that an effect or interaction is absent.

Do not use Shapley-style allocations to fill missing interventions. They may be included as descriptive allocations only when the coalition evaluations and assumptions are explicit; they do not replace causal contrasts.

### 4.3 Blocking, pairing and assignment

- Start every sibling from identical base bytes and frozen research state.
- Match task assignments, seed blocks, toolchain, evaluator, training recipe, compute and wall-clock ceilings.
- Randomize execution order when thermal state, cache, queue, or temporal drift may matter.
- Treat independent task families, repositories, projects, or seed groups as the statistical units where appropriate; task rows within a cluster are not automatically IID.
- Charge all attempts, failures, retries, integration work and evaluation calls to the assigned arm.
- Report build/validity failure separately. Do not silently drop failed variants or invent a numeric capability score for an unexecuted candidate.

Use the existing paired, cluster-aware and population-inference owners (#8438, #9433, #10513) rather than creating another statistics backend.

## 5. Necessity, sufficiency and rescue are different tests

| Test | Intervention | Supported interpretation |
|---|---|---|
| Necessity | Roll back one contribution from the winning candidate | The contribution is needed in this tested context if the predicted outcome falls |
| Sufficiency | Transplant the isolated contribution to a clean matched base/sibling | The contribution reproduces an effect in the tested target context |
| Rescue | Restore a compatible missing component after a predicted interaction failure | The predicted joint mechanism is supported in that context |
| Negative control | Remove/transplant a no-op, unrelated or provenance-only change | Detects false attribution and analysis leakage |

Rollback-only attribution is incomplete: dependent changes may fail when separated. Transplant-only attribution is also incomplete: a change may be context-dependent. Record compatibility preconditions and treat invalid interventions as such; a failed transplant does not prove that the original contribution was ineffective.

## 6. Mechanism predictions

Before execution, preregister at least one observable intermediate prediction per claimed mechanism. Depending on the domain, this could be:

- A specified change in update direction, gradient/credit assignment or update variance.
- Improved sample efficiency under a fixed training budget.
- A predicted shift in experiment allocation or search branching.
- A memory retrieval, consolidation or invalidation behavior.
- A predicted failure-recovery path or verified execution trace.

The observation must be tied to the exact source/build/run and checked independently of the candidate's explanatory prose. Report terminal treatment effects separately from mediator observations. Do not claim formal mediation unless the design identifies it and its assumptions are justified.

## 7. Generalization and external confirmation

Report transfer lanes independently:

- **G0 — Replay:** original discovery conditions; diagnostic only.
- **G1 — Fresh instances:** unseen instances/seeds from the same task family.
- **G2 — Difficulty extrapolation:** harder instances or shifted resource regimes.
- **G3 — Fresh family:** a task family not used in discovery or selection, within the same broad capability.
- **G4 — Cross-domain:** a materially different capability or application domain.

The initial pilot should include at least one software/harness family and one clean-start algorithm-design or scientific-experiment family. Game-only transfer is not evidence of scientific R&D or algorithmic RSI.

Development and validation results may guide candidate selection. Fresh-family and final confirmation outcomes remain under independent custody and are not returned as detailed adaptive feedback. The independent evaluator and #10573 retain all promotion authority.

## 8. Outcomes — no opaque causal score

Report separately:

1. Parent/change-level treatment effects and preregistered interactions, with uncertainty.
2. Necessity, sufficiency and rescue success by intervention and context.
3. The fraction of the observed child gain explained by identified contrasts, plus an explicit unidentified remainder.
4. Mechanism-prediction accuracy and intermediate-effect calibration.
5. Replication rate across independent seeds and task-family clusters.
6. False-attribution rate on known-null, no-op, confounded and non-identifiable controls.
7. G0–G4 transfer and protected-capability regression profile.
8. Number of planned/attempted/valid/failed variants; compute or FLOPs proxy, tokens, evaluator calls, wall time, memory, energy where measurable, and human intervention.
9. Independently confirmed causal contributions per unit cost; cost per replicated contribution.
10. Best-so-far and final capability, so late regressions remain visible.

Do not collapse outcomes with incompatible units into a single “causal RSI” index. Publish raw contrasts, denominators, uncertainty, missingness, failed attempts and unassigned effects.

## 9. Minimum deterministic fixture corpus

The first tranche should use inert, non-executable fixtures with known expected outcomes:

1. One known-causal patch with an observable mediator.
2. Two-parent case where A contributes, B is a no-op, and interaction is zero.
3. Synergy case where neither parent alone helps but A+B does.
4. Harmful contribution masked by a stronger beneficial change.
5. Dependent edits that must be represented as one coupled block.
6. Source patch whose proposed mechanism is absent from the built executable.
7. Configuration-only change mislabeled as an algorithm change.
8. Hash-changing no-op refactor with unchanged behavior.
9. Score improvement that violates a protected capability floor.
10. Seed/task-family leakage or reuse.
11. Non-identifiable outcome under the available interventions.
12. Incompatible transplant with an unsatisfied precondition.

The first eight deterministic cases are recorded in machine-readable form at [SYM_RSI_CAUSAL_LINEAGE_2X2_FIXTURES_V0_2026-11-06.json](./fixtures/SYM_RSI_CAUSAL_LINEAGE_2X2_FIXTURES_V0_2026-11-06.json). The reference vectors deliberately exercise different causal shapes:

| Fixture | (Y00, Y10, Y01, Y11) | A marginal | B marginal | A×B interaction |
|---|---:|---:|---:|---:|
| A contributes; B no-op | (0, 2, 0, 2) | 2 | 0 | 0 |
| Pure synergy | (0, 0, 0, 4) | 2 | 2 | 4 |
| Harmful B masked by A | (0, 6, -2, 4) | 6 | -2 | 0 |
| Antagonistic interaction | (0, 2, 2, 1) | 0.5 | 0.5 | -3 |
| Both no-op | (0, 0, 0, 0) | 0 | 0 | 0 |

The corpus also includes a missing counterfactual, evaluation-context mismatch, and source-to-executable mismatch. The expected behavior is respectively “not identified,” reject the contrast, and block mechanism-level credit. The fixture file is a synthetic arithmetic/claim-gating oracle, not an empirical benchmark, signed receipt, or hidden holdout.

A structural manifest validator can establish consistency of declared identities and factor/dependency relations. It cannot establish that a proposed intervention is truly atomic, that an executable ran, or that the causal claim is true. Those require later execution and independent evidence.

## 10. Baselines and comparison

Compare:

- Parent-relative paired comparison only.
- Keep-one / rollback-only ablation.
- Single-parent mutation with source-backed change units.
- Multi-parent hybrid without causal intervention.
- Random matched-cost intervention selection.
- Human-authored causal plan where feasible.
- Same-base factorial plus rollback/transplant/rescue protocol.

The success criterion is not “more explanations.” It is a higher rate of correctly identified, independently replicated mechanisms, lower false attribution, and lower cost per validated contribution.

## 11. No-runner implementation tranche

Before execution isolation and independent evaluator prerequisites qualify:

1. Publish this protocol and a preregistration template.
2. Define inert intervention manifests and the fixture corpus.
3. Add only structural validation and offline reference calculations over synthetic outcomes, following existing validation conventions.
4. Add positive and adversarial negative-control tests.
5. Map the protocol to #10497, #10498, #10674, #10691, #10569 and #10573.

Do not execute candidate code, open sealed holdouts, or claim causal validity from structural tests. A later executable pilot must run through the existing isolated, attested, independently regraded evidence path.

## 12. Research landscape and Symthaea's differentiator

| Work | Established contribution | Symthaea extension |
|---|---|---|
| HyperAgents (Meta, 2026) | Makes the meta-improvement procedure editable and reports cross-domain improvement of meta-level behavior | Same-start matched budgets, intervention-level attribution, and independent confirmation |
| RSIGym (2026) | Shared research infrastructure and Data / Harness / Joint optimization tracks | Separate known-suite gains from fresh-family transfer and parent/change-level causal effects |
| RSI-Env (2026) | Persistent model/harness improvement with an independent Host Controller | Explicit causal variants, mechanism tests and cross-domain transfer of inherited changes |
| RSIBench (2026) | Fixed-model harness RSI, frozen tasks/verifiers/budgets, rollback and keep-one ablation | Multi-parent interactions, clean transplant/rescue and learning-algorithm mechanisms |
| SelfImprove-Bench (2026) | G0–G4 generalization-distance evaluation | Carry the ladder into software engineering, scientific research and algorithm design |
| AI4AI-Bench (2026) | Direct training-algorithm design task suite | Audit-resistant clean-start replication and causal evidence for the actual changed learning computation |
| FIRE-Bench (ICML 2026) | Execution-based rediscovery of established scientific insights | Separate rediscovery from novelty and use independent intervention/replication evidence |
| AIDE² (2026) | Reports seven successive self-improvements in an autonomous 8-day run and transfer to four held-out benchmarks, including an out-of-distribution weather-forecasting task | Include as a close R4 baseline; independently test same-start budget parity, inherited mechanisms and replication |
| METR Anthropic R&D assessment (2026-11-05) | Estimates 1.9× growth in Anthropic AECI benchmark capability since October 2025 (95% CI 1.5–2.8×); attributes much of the acceleration to AI-assisted R&D, while reporting no evidence yet of highly significant fully autonomous contributions to the most important training stages | Measure AI-assisted research productivity separately from autonomous recursive self-improvement |

### Human-supervised productivity is not autonomous RSI

METR's November 5, 2026 assessment is strategically important but must be interpreted carefully. It estimates that Anthropic's benchmark capability progress since October 2025 has proceeded at about 1.9× the prior rate, with a 95% interval of 1.5–2.8×. METR judges that a majority of current acceleration is likely attributable to AI use in R&D, but the report is high-uncertainty, includes substantial redactions, focuses on well-specified capability tasks, and says AI is not yet making highly significant fully autonomous contributions to the most important parts of model training.

This motivates a separate **R&D productivity frontier**, not a relaxation of the R4 definition. Compare at least:
1. Human-only researcher.
2. Human researcher with a fixed AI assistant.
3. Human researcher with a meta-improving Symthaea co-researcher.
4. Bounded autonomous Symthaea research loop with independent human/evaluator gates.

Measure independently confirmed research artifacts, accepted improvements, human active minutes and review/rework, compute/tokens, wall time, evaluation cost and cost per valid discovery. Publish quality-versus-human-time and quality-versus-total-resource frontiers. Keep expert oversight and intervention visible; self-reported speedup, code volume and faster patch production are not substitutes for independently confirmed research output.

AIDE² should be treated as a particularly relevant comparison: its paper reports seven successive improvements during an autonomous eight-day run and transfer to four held-out benchmarks, including an out-of-distribution weather-forecasting task. Those are the paper's reported results, not an independent replication by Symthaea. Our comparison should therefore use the same-start swap, matched total budget, inherited meta-level changes, fresh-family transfer and independent confirmation rather than claiming superiority from a different task suite.

Symthaea should not claim novelty for “an agent that edits itself,” “an editable meta-agent,” “a multi-parent archive,” or “rollback ablation.” The stronger contribution is a common, auditable method that ties **exact change → randomized intervention → mechanism prediction → independent replication → transfer** to the candidate's inherited research process.

### External references

- HyperAgents: https://arxiv.org/abs/2603.19461 and https://ai.meta.com/research/publications/hyperagents/
- RSIGym: https://arxiv.org/abs/2610.10310 and https://github.com/evolvent-ai/RSIGym
- RSI-Env: https://arxiv.org/abs/2611.01736
- RSIBench protocol: https://harness-rsibench.com/docs/
- SelfImprove-Bench: https://arxiv.org/abs/2611.02797
- AI4AI-Bench: https://arxiv.org/abs/2608.20318
- Independent AI4AI-Bench audit (2026-11-03): https://github.com/Open-Benchmark-Review/audits/blob/main/audits/ai4ai-bench/report.md
- FIRE-Bench: https://proceedings.mlr.press/v306/wang26d.html
- Red Queen Gödel Machine: https://arxiv.org/abs/2606.26294

## 13. Claim boundary

This specification does not establish that any current Symthaea improvement has a causal explanation. A positive result is conditional on the exact intervention, task population, evaluator and resource envelope. R4 still requires inherited improvement of the improvement process under matched budgets. R5 still requires a learning-algorithm change that improves clean-start successor training and transfers. Neither follows from ancestry, hybridization or this protocol alone.

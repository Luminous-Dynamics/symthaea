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

## 11. No-runner implementation tranche and current prototype

The first no-runner slice is now present in this PR branch:

- Standalone Rust package: `tools/sym-rsi-causal-reference/`. Its nested `[workspace]` keeps it outside the production root workspace and avoids relying on the still-unmerged #10472/#10496/#10499/#10515 draft stack.
- `Factorial2x2`: computes A/B marginal effects, conditional effects and A×B interaction from same-base outcomes.
- `summarize_equal_cluster_blocks`: requires one canonical context digest, retains block and cluster contrasts, and reports an equal-cluster point estimate. It does not calculate confidence intervals or assert cluster independence.
- `InterventionManifest`: checks digest syntax, unique identities, known parent/change references, dependency acyclicity, symmetric conflicts, coupled-group cardinality and exclusion of candidate-side evaluator/task-contract edits.
- Offline CLI: parses the inert 2×2 fixture corpus and synthetic manifest example.
- Sixteen Rust unit tests are authored for fixture arithmetic/gates, manifest validation, non-finite input, cluster weighting, context mismatch, dependency cycles/conflicts and evaluator-edit exclusion.

Commands from the repository root:

```sh
cargo fmt --manifest-path tools/sym-rsi-causal-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-causal-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-causal-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-causal-reference/Cargo.toml -- fixtures docs/research/fixtures/SYM_RSI_CAUSAL_LINEAGE_2X2_FIXTURES_V0_2026-11-06.json
cargo run --manifest-path tools/sym-rsi-causal-reference/Cargo.toml -- manifest docs/research/fixtures/SYM_RSI_INTERVENTION_MANIFEST_V0_EXAMPLE_2026-11-06.json
```

The current execution environment has no `rustc`, `cargo`, `rustfmt` or `nix`; the Rust tests are authored but **not compiled or executed**. A separate script parsed all eight 2×2 fixture cases and their expected arithmetic/claim-gating outputs; all eight matched. The example manifest's required fields, digest syntax, parent origins and unique change IDs were also checked. These are static/reference checks only—not a Rust test pass, CI pass, authenticated source/build proof or causal result.

The tool deliberately does not execute candidate code, access sealed holdouts, authenticate evidence refs, infer causal atomicity, prove statistical independence, calculate inferential uncertainty, or authorize promotion. Later execution must use the existing isolated, attested, independently regraded evidence path and the authority owners above.

## 12. Research landscape and Symthaea's differentiator

| Work | Current result / design (paper or project reported) | Symthaea extension and caveat |
|---|---|---|
| Darwin Gödel Machine | Open-ended coding-agent self-modification with an archive of candidate variants | Do not claim novelty for self-editing or archive search; require fresh-family transfer, causality and independent confirmation |
| HyperAgents (Meta, 2026) | Editable task agent and meta-agent in one self-referential program; reports meta-procedural improvement across coding, paper review, robotics reward design and proof grading | Same-start matched budgets, patch-level causal attribution, and independent confirmation are the differentiators—not editable meta-code alone |
| RSIGym (2026) | Reports a leading RSI-Index of 0.4809 (Opus 5) across five benchmarks and a $500 platform-service budget per run; 29/30 Joint runs improve | Treat 0.4809 as a known-suite infrastructure result, not direct unseen-transfer evidence: final evaluations reuse development-available tasks, each configuration is run once, and researcher inference/official evaluation are billed separately |
| RSI-Env (2026) | Persistent model/harness improvement with an independent Host Controller. Its Open-General Self setting reports GPQA Diamond 65.2→73.2 and AIME 2026 53.3→56.7; Knowledge Self reports SimpleQA 82→85→88 across updates | Measure cross-capability retention and task-family transfer under the same start and full cost boundary; published results are not independent Symthaea replications |
| RSIBench (2026) | Fixed-model harness evolution over five rounds, 84 train and 84 sealed test tasks across six domains; frozen verifiers/budgets and rollback/keep-one ablations | Strong harness-causality comparator. Its published protocol has no validation split; Symthaea should use discovery, validation and sealed confirmation, then add fresh-family and clean-start algorithm transfer |
| RSI-Anything (2026) | Uses predictive world models to rank actions/learning data before costly external feedback, then updates workflows and conditionally adapts the predictor under component/pair compatibility checks. Reports code-repair success 26.3→29.2→31.2% at comparable test usage, a 39.4% reduction in GUI actions, and 96.15% mean best observed reaction yield vs 82.16% for Bayesian optimization at 40 experiments | A strong efficiency and world-model comparator. Separate predictions from observed outcomes; test reliability under policy-induced distribution shift; reuse #5476 prospective VoC, #5478 causal ablations and #8721 allocation rather than adding another scheduler |
| SelfImprove-Bench (2026) | G0–G4 generalization ladder across Exploring/Planning/Reasoning game scenarios; reports inconsistent transfer across levels | Carry the ladder into software engineering, scientific research and algorithm design; game-only transfer is not general R&D competence |
| AI4AI-Bench (2026) | Ten training-algorithm repositories; paper reports mean 0.166 and best system 0.250, and method-changing submissions average 0.226 vs 0.126 for configuration-only changes | Useful algorithm-design starting point, but do not treat 0.250 as uncontested SOTA: an independent audit classified v1 as flawed, with score-changing defects in 4/10 tasks and final-test checkpoint selection; preserve paper and audit evidence as separate records |
| Red Queen Gödel Machine (2026) | Co-evolves agents and evaluators using frozen utility within epochs and controlled evaluator changes at boundaries | #10708 already owns this paradigm; #10717 must consume its independent anchor/confirmation authority, not create a second evaluator |
| AIDE² (2026) | Reports seven accepted inner-agent rewrites over eight days (0.703→0.778), two further runs accepting two and four rewrites, and transfer to four held-out benchmarks including OOD weather forecasting. Its ignition comparison of the evolved agent vs human-developed agent as the *outer-loop improver* is inconclusive: three seeds/arm, mean endpoints 0.780 vs 0.782 | Closest R4 comparator. Symthaea must separately pass the outer-loop ignition test; inner-loop gains and task transfer do not prove an improved self-improver |
| FIRE-Bench (ICML 2026) | Execution-grounded rediscovery of documented scientific insights by designing/running experiments and comparing conclusions with published findings | Add it as a scientific-process/rediscovery baseline; rediscovery is not novelty, and LLM-judge prose quality is not enough |
| RSI evidence audit (2026-10-20) | Full-text census of 114 eligible studies (90 self-labelled RSI): only 22% audit ≥2 full-loop reruns; of 26 studies claiming a better improver, only 4 compare earlier/later procedures from identical starts at matched budgets | Direct field-wide target: repeated full-loop replication and fair same-start improver swaps are under-reported; make them non-negotiable rather than another weighted leaderboard score |
| METR Anthropic R&D assessment (2026-11-05) | Estimates 1.9× growth in Anthropic AECI benchmark capability since October 2025 (95% CI 1.5–2.8×); judges much of the acceleration likely AI-assisted while noting no highly significant fully autonomous contribution to the most important training stages yet | Measure AI-assisted R&D productivity separately from autonomous recursion; the estimate is high-uncertainty, benchmark-scoped and partly informed by redacted evidence |

### The field-wide gap: test the improver, not only its output

Chen, Zhu and Zheng's October 20, 2026 census coded 114 eligible RSI studies, 90 of which described their own systems as RSI. Only 22% of those 90 audited two or more reruns of the complete loop. Of 26 papers claiming a better improvement procedure, 23 reported a related comparison, but only four compared earlier and later improvers from identical starting artifacts at matched budgets. These are frequencies of reported evaluation designs, not success rates. The implication is unusually actionable: a rigorous same-start, matched-budget and multi-seed R4 demonstration would address a documented methodological gap in the field, not just chase another benchmark number.

### The AIDE² ignition test should become Symthaea's R4 gate

AIDE²'s inner loop accepted seven rewrites in its main eight-day trajectory (grade 0.703 to 0.778), plus two and four accepted rewrites in two additional complete runs. It also reports transfer to four external benchmarks, including out-of-distribution weather forecasting. But its ignition test—using an evolved agent as the outer-loop researcher—compared AIDE47 against the human-developed AIDE agent from the same AIDE47 inner-loop start, with three seeds per arm and 50 outer-loop steps. Mean endpoints were 0.780 and 0.782 respectively. The authors explicitly regard the result as inconclusive because of seed variance and the cost of additional outer-loop runs and external evaluation.

Symthaea should make this distinction explicit in #10497:

1. **Product improvement:** the task-solving or inner research agent improves.
2. **Improver improvement:** the evolved researcher makes better candidates under the same starting state and total cost.
3. **Ignition:** the evolved researcher, when placed in the outer loop, produces a better next-generation improver than the incumbent/human-designed reference.
4. **Recursive progress:** the accepted outer-loop successor itself repeats the advantage over multiple complete, independently seeded cycles.

For ignition, freeze the starting task-agent, outer-loop state, model/backend, task populations, tools, evaluation rules, retry policy and total cost. Compare the evolved outer-loop agent against a fixed/human-authored outer-loop baseline on the same initial artifacts. Use several independent full-loop seeds, estimate variance before fixing confirmatory sample size, and evaluate the resulting successor researchers on fresh-family tasks. A positive inner-loop grade or a single successful outer-loop trajectory is insufficient.

### Predictive world models: improve experiment economy without treating prediction as truth

RSI-Anything is a timely adjacent approach. It uses action-conditioned predictive world models to rank candidate actions and select where to spend costly external feedback, then uses observed feedback to update the policy/workflow and conditionally adapt the predictor. It adds component and policy–predictor compatibility checks before retaining an updated pair. The paper reports gains in code repair, GUI efficiency, lab planning and reaction optimization, but these results are paper-reported and task-specific.

Symthaea should not build a competing generic value-of-computation scheduler. Instead compose:
- #5476 for forecast-before-outcome and realized-utility calibration;
- #5478 for intact/lesion/sham/rescue causal comparison of metareasoning mechanisms;
- #8721 for the scientific tool-allocation application.

Add one experimental comparison to #10497/#10674: no predictor, frozen predictor, reliability-gated adaptive predictor, and a synthetic oracle upper bound. Record every prediction before its outcome; retain predictor and policy as a versioned compatible pair; detect when policy evolution changes the predictor's error distribution; fall back to a simpler policy when calibration or compatibility fails. Do not count a predicted gain as observed improvement, and do not train on feedback hidden from the independent confirmation authority.

### Seven-dimensional RSI evidence profile

The 2026 RSI survey organizes evaluation around reliability, efficiency, diversity, persistence, adaptability, controllability and generalization. Use these as a reporting profile, not a new weighted composite. In particular, diversity must mean causal independence of hypotheses/interventions and differentiated failure coverage—not just varied wording. Persistence requires retained changes to remain useful in later cycles; adaptability requires explicit distribution-shift tests; controllability requires scoped authority, review and rollback; generalization includes transfer of the *improver*, not only the final product.

### Human-supervised productivity is not autonomous RSI

METR's November 5, 2026 assessment is strategically important but must be interpreted carefully. It estimates that Anthropic's benchmark capability progress since October 2025 has proceeded at about 1.9× the prior rate, with a 95% interval of 1.5–2.8×. METR judges that a majority of current acceleration is likely attributable to AI use in R&D, but the report is high-uncertainty, includes substantial redactions, focuses on well-specified capability tasks, and says AI is not yet making highly significant fully autonomous contributions to the most important parts of model training.

This motivates a separate **R&D productivity frontier**, not a relaxation of the R4 definition. Compare at least:
1. Human-only researcher.
2. Human researcher with a fixed AI assistant.
3. Human researcher with a meta-improving Symthaea co-researcher.
4. Bounded autonomous Symthaea research loop with independent human/evaluator gates.

Measure independently confirmed research artifacts, accepted improvements, human active minutes and review/rework, compute/tokens, wall time, evaluation cost and cost per valid discovery. Publish quality-versus-human-time and quality-versus-total-resource frontiers. Keep expert oversight and intervention visible; self-reported speedup, code volume and faster patch production are not substitutes for independently confirmed research output.

Symthaea should not claim novelty for “an agent that edits itself,” “an editable meta-agent,” “a multi-parent archive,” or “rollback ablation.” The stronger contribution is a common, auditable method that ties **exact change → randomized intervention → mechanism prediction → independent replication → transfer** to the candidate's inherited research process.

### External references

- Darwin Gödel Machine: https://arxiv.org/abs/2505.22954
- HyperAgents: https://ai.meta.com/research/publications/hyperagents/
- RSIGym: https://arxiv.org/abs/2610.10310 and https://github.com/evolvent-ai/RSIGym
- RSI-Env: https://arxiv.org/abs/2611.01736
- RSIBench protocol: https://harness-rsibench.com/docs/
- SelfImprove-Bench: https://arxiv.org/abs/2611.02797
- AI4AI-Bench paper: https://arxiv.org/abs/2608.20318
- Independent AI4AI-Bench audit (2026-11-03): https://github.com/Open-Benchmark-Review/audits/blob/main/audits/ai4ai-bench/report.md
- Red Queen Gödel Machine: https://arxiv.org/abs/2606.26294
- AIDE² paper and ignition test: https://arxiv.org/abs/2609.26457
- RSI evaluation-evidence census (2026-10-20): https://arxiv.org/abs/2610.27610
- RSI-Anything world-model framework: https://arxiv.org/abs/2610.13912
- RSI survey and seven-dimensional evaluation profile: https://lee1003-lee.github.io/Awesome-RSI-Research/
- FIRE-Bench: https://proceedings.mlr.press/v306/wang26d.html
- METR R&D acceleration assessment (2026-11-05): https://metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment/

## 13. Claim boundary

This specification does not establish that any current Symthaea improvement has a causal explanation. A positive result is conditional on the exact intervention, task population, evaluator and resource envelope. R4 still requires inherited improvement of the improvement process under matched budgets. R5 still requires a learning-algorithm change that improves clean-start successor training and transfers. Neither follows from ancestry, hybridization or this protocol alone.

# Architecture Search Qualification: Held-Out Landscape Suite v0.1

**Status:** protocol proposal only. The suite is not implemented, executed, or qualified.  
**Tracking issue:** [#8959 — qualify architecture search on held-out landscape families](https://github.com/Luminous-Dynamics/symthaea/issues/8959)  
**Parent program:** [#8888](https://github.com/Luminous-Dynamics/symthaea/issues/8888)  
**Current implementation under review:** [PR #8890](https://github.com/Luminous-Dynamics/symthaea/pull/8890)
**Related cost-aware follow-up:** [Multi-fidelity architecture search v0.1](./MULTI_FIDELITY_ARCHITECTURE_SEARCH_V0_1.md) ([#8994](https://github.com/Luminous-Dynamics/symthaea/issues/8994)); authored as a scheduling primitive, not yet compiled or qualified.

## 1. Why this is the next paradigm shift

The existing synthetic harness is useful as a unit-testable instrument: its objective is known, budgets are explicit, and its deceptive surface can expose a local-search failure. But two public, deterministic objective surfaces are not a qualification suite for a general search algorithm. A policy can be tuned to their topology, feature priors, or seed behavior.

The proposed change is to evaluate **the searcher across a distribution of optimization problems**, with a pre-registered primary outcome, equal evaluation budgets, held-out challenge instances, and a separate real-model confirmation stage.

The research hypothesis is not that Symthaea-guided search must win. It is that a candidate policy should be retained only if it beats strong, pinned baselines on unseen objective instances at a fixed budget, with measured uncertainty and without unacceptable worst-family regressions.

## 2. Evidence ladder

| Stage | Evidence question | Permitted claim |
|---|---|---|
| A. Deterministic smoke surfaces | Are objective oracles, budgets, metrics, and negative controls wired correctly? | Implementation-level synthetic checks only |
| B. Public development distribution | Can search policies be debugged and compared across diverse surface families? | Development results; not generalization evidence |
| C. Held-out challenge instances | Does the frozen policy generalize to unseen seeds/instances and proxy regimes? | Synthetic search-policy qualification on declared families |
| D. Real workload pilot | Does a search policy select better candidates on a pre-declared real task/hardware profile? | Workload-specific pilot result only |
| E. Confirmatory model evaluation | Does the selected architecture replicate under matched training/data/serving conditions and independent runs? | Only the explicitly supported architecture/model claim |

Passing an earlier stage does not imply a later stage. Stage C is not a substitute for stage E.

## 3. Surface families

Start with bounded, fully specified objective oracles so the global optimum and exact Pareto frontier are known. The public development suite should contain at least these distinct families:

1. **Smooth/separable:** additive objectives with a broad, well-behaved basin. Establishes basic search efficiency and expected random-search behavior.
2. **Deceptive/local-trap:** attractive local optima separated from the global optimum. Measures restart and escape behavior.
3. **Interaction-heavy/epistatic:** objective value depends on combinations of genes; single-coordinate improvements can be misleading.
4. **Constrained/discontinuous:** feasibility constraints, sharp cliffs, and rare feasible regions. Invalid candidates are outcomes, not silently dropped records.
5. **Noisy/heteroscedastic:** candidate estimates have different variances and rankings can reverse across evaluation seeds. Repeats consume budget and support uncertainty-aware decisions.
6. **Proxy drift and corruption:** proxy–quality rank correlation weakens, disappears, or reverses between development and held-out instances. Include shuffled and deliberately anti-correlated controls.
7. **Multi-objective topology:** convex, concave, disconnected, and boundary-heavy Pareto sets. The oracle supplies exact dominated/undominated classifications and hypervolume reference points.
8. **Workload-transfer:** architecture candidates are scored under two workload profiles whose quality/cost trade-offs differ. A policy tuned on profile A is tested on profile B without retuning.

Every surface must declare the number of configurations, variable domains, deterministic or stochastic semantics, objective direction and bounds, invalid-evaluation semantics, oracle version, and all metric normalization/reference points. Candidate IDs and ordering must not leak an answer.

The initial suite should remain small enough for exhaustive oracle calculation. Increase dimensionality only after the metric and budget controls are passing.

## 4. Holdout without false secrecy

A public Git repository cannot keep a generator or seed secret after committing it. Do not call a public, repeatedly inspected test file “hidden.”

Use three distinct mechanisms:

- **Public development suite:** fully committed generators, seeds, and oracles. These are explicitly tuneable and never counted as held-out evidence.
- **Committed challenge procedure:** publish the generator code and all scoring rules; pre-register a policy commit SHA; then generate challenge seeds after that SHA is frozen.
- **Seed commitment and release:** the steward publishes a SHA-256 commitment to the challenge seed before running the frozen policy, records the exact configuration and outputs, and reveals the seed after scoring so an independent party can reproduce it. If organizational independence is unavailable, label this as *prospective holdout*, not independent evaluation.

Create a new challenge version for confirmation. Once results from a challenge version have informed tuning, that version becomes development data for subsequent claims. Do not repeatedly inspect a score and still call the same suite held out.

## 5. Baselines and equal-budget accounting

Required policy set:

- Uniform random search.
- Local search with restarts.
- A conventional adaptive baseline such as TPE or Bayesian optimization, only when the exact implementation, version, search space, initialization, and stopping behavior are pinned.
- Proxy-ranked search with a feature-only proxy.
- Symthaea-guided search, using the candidate policy revision frozen before the challenge run.

Add evolutionary search when it can be implemented and pinned without making the comparison ill-defined. Avoid an untracked “industry baseline” that changes version or resource limits between runs.

Each full candidate evaluation costs one budget unit. Invalid configurations, OOM/timeouts, non-finite results, failed measurements, retries, and duplicate candidates must be present in the accounting. A repeated stochastic evaluation costs another unit. Report objective-call budget separately from wall-clock, CPU/GPU time, and total failed-evaluation cost; these quantities are not interchangeable.

Use common landscape instances and seeds across policies so comparisons are paired. Use budgets 16/32/64/128 initially, plus the exhaustive-oracle ceiling where helpful. Do not silently give one policy extra precomputation or free objective labels. Report a policy's development and tuning cost separately from challenge-time cost.

## 6. Metrics and statistical analysis

### Primary endpoints

Pre-register one primary endpoint per study. Recommended defaults:

- **Normalized simple regret** at each evaluation budget: oracle optimum minus best observed value, normalized by the declared objective range.
- **Anytime performance:** area under the best-so-far quality curve versus number of objective evaluations, with the integration rule fixed in advance.

For multi-objective studies, use normalized hypervolume with a pre-registered reference point as a primary or co-primary endpoint. Do not choose the primary metric after observing which policy wins.

### Secondary endpoints

- Exact-optimum hit rate when meaningful.
- Target hit rate and evaluations-to-target.
- Pareto-front recall and dominated-area regret.
- Best-so-far curve by budget, not just the final score.
- Proxy-quality rank correlation, calibration/uncertainty where the proxy emits uncertainty, and performance under proxy drift.
- Invalid/failed-evaluation rate, wall-clock, total compute, and seed-to-seed variance.
- Worst-family effect and workload-transfer degradation.

### Uncertainty and multiplicity

- For development qualification, run at least 30 paired seeds per family and budget; because synthetic runs should be cheap, aim for 100 paired seeds for the confirmatory comparison.
- Pair policies by identical landscape instance and seed. Report paired effect sizes and paired bootstrap 95% confidence intervals over the predeclared replication unit.
- Publish each per-seed result and the aggregate. If the unit includes multiple landscapes, avoid pretending the individual configurations within a single landscape are independent samples.
- Correct for multiple exploratory comparisons. Confirmatory analysis should name one primary contrast and metric in advance; confidence intervals do not correct post-hoc metric selection.
- Report pooled results, every-family results, and the worst-family outcome. A positive pooled mean must not hide catastrophic regressions in a critical family.

The 30/100-seed schedule is an initial engineering protocol, not a guarantee of statistical power for every effect. Estimate variance in development runs and increase the confirmatory count if the interval remains too wide.

## 7. Negative controls and invariants

A qualification implementation must fail closed on:

- Shuffled proxy labels that accidentally retain an oracle-derived ranking.
- Anti-correlated proxy labels reported as a useful signal.
- Duplicate or missing result records, duplicated seeds, and unbalanced policy × instance × seed coverage.
- Budget overflow, “free” retries, hidden duplicate evaluations, or dropped invalid candidates.
- Candidate ordering changing the result when order should be irrelevant.
- Oracle leakage into a candidate policy or a mismatch between the declared and executed objective version.
- Incomplete objective vectors, an unpinned hypervolume reference, NaNs/infinities, or silently suppressed failures.
- Challenge-seed use before the policy commit is frozen.
- Aggregate metrics whose receipt cannot be reconstructed from the released per-run artifacts.

A negative-control failure invalidates the associated comparison, even if the headline score looks favorable.

## 8. Provisional promotion target

The proposed engineering target—not a universal statistical law—is:

1. At equal challenge-time evaluation budget, at least **10% relative reduction in normalized simple regret** or **10% relative gain in normalized hypervolume** against the strongest pinned baseline on the pre-registered primary endpoint.
2. A paired 95% confidence interval for the primary effect excludes zero in the favorable direction.
3. No more than 5% relative regression on any pre-registered critical surface family, unless that trade-off was explicitly accepted in advance.
4. Shuffled-proxy control is indistinguishable from a no-signal baseline within uncertainty, while the anti-correlated control is not rewarded as positive information.
5. The direction of improvement replicates on a newly generated challenge version.
6. The full record is reproducible from released source, config, seeds, results, and hashes.

If the policy improves only one family, qualify only that family. Do not claim general search superiority based on one synthetic landscape, and never treat synthetic search metrics as evidence of better language-model quality.

## 9. The bridge to architecture experiments

Only after stage C passes should the policy be tested on model candidates. Before spending substantial GPU time, resolve or explicitly fence off [#8308 — learning-signal and benchmark validity](https://github.com/Luminous-Dynamics/symthaea/issues/8308): the training loss must respond to predictions, and external task evaluators must use the actual task inputs.

For model trials:

- Freeze a competent dense Transformer control and its training recipe.
- Compare candidates at matched data, tokenizer, training tokens/FLOPs, optimizer budget, and declared compute ceilings; report unavoidable differences rather than hiding them.
- Use development and validation splits for search, with final test locked until the candidate and analysis are frozen.
- Measure task quality jointly with prefill/TTFT, per-token latency, throughput at named concurrency, peak host/accelerator memory, energy source/uncertainty, invalid-run yield, and wall-clock.
- Evaluate at least one local-interactive and one throughput-oriented workload profile. The winner may differ by profile.
- Preserve all failed trials and negative results. Do not use Phi as the primary quality objective unless a pre-registered ablation demonstrates incremental predictive value for the task outcome.

### Industry comparison

For serving-system comparisons, align measurement with [MLPerf Endpoints v1.0](https://mlcommons.org/2026/10/open-submissions-endpoints-v1-0/) where practical: compare the quality-gated performance frontier across concurrency, per-user interactivity, aggregate throughput, P90 time-to-first-token, and provisioned power as specified by the pinned rules. Its [submission documentation](https://docs.mlcommons.org/endpoints-submission-docs/) tracks a developing v1.0 process; pin the rules and checker revision.

MLPerf Endpoints is a serving benchmark, not proof that an architecture search policy is superior. Architecture claims need controlled model training and workload experiments; serving claims need an explicit and comparable serving setup.

## 10. Research references and limits

- Phan & Luong, *Robust and Efficient Multi-Fidelity Neural Architecture Search with Zero-Cost Proxy-Guided Local Search*, ACM Transactions on Evolutionary Learning and Optimization, published June 29, 2026. The work explicitly studies proxy-guided local search and proxy fitness landscapes; see [the publisher record](https://doi.org/10.1145/3774940).
- *ZAPS* (September 2026 preprint) discusses correlated proxy errors and naive aggregation; its reported NAS-Bench-201 numbers belong to that setup, not Symthaea. [arXiv:2609.14184](https://arxiv.org/abs/2609.14184).
- [MLPerf Endpoints v1.0 announcement](https://mlcommons.org/2026/10/open-submissions-endpoints-v1-0/) and [live report portal](https://endpoints.mlcommons.org/).

These references motivate the evaluation design. They do not show that the Symthaea searcher has been implemented, run, or shown to outperform a baseline.

## 11. Definition of done

The suite is not qualified until: generator and oracle tests pass; per-budget policy results are produced on the exact source SHA; paired statistics and artifacts are independently recomputable; all negative controls behave as expected; a new challenge version replicates the direction of effect; and a reviewer accepts the claim ceiling. Until then, status remains **DESIGN_ONLY_NOT_EXECUTED_NOT_QUALIFIED**.

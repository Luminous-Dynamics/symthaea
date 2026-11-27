# Nano Policy Kernel — Benchmark Contract

**Status:** Preregistered evaluation plan for the selector kernel; no measurements have been run.  
**Implementation:** [symthaea-nano-policy](./README.md)  
**MOCLIP domain plan:** https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-001-atomistic-discovery/docs/engineering/NANOPHOTONICS_MOCLIP_BENCHMARK_SCORECARD.md  
**General benchmark charter:** https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-001-atomistic-discovery/docs/engineering/NANOTECHNOLOGY_BENCHMARK_CHARTER.md

## 1. What this kernel can and cannot establish

The crate implements four selector policies over a supplied, frozen descriptor pool: seeded random, farthest-first, family-balanced farthest-first, and cost-aware farthest-first. The JSONL CLI also offers a separate `--verify-receipt` mode for independent structural verification of request/response pairs. A separate post-outcome evaluator computes success rate, normalized success AUC, and paired best-loss reduction from a frozen observation ledger. For loss reduction, positive delta means baseline best loss minus candidate best loss; any target without a valid outcome in either arm yields an `insufficient_target_coverage` disposition and no confidence interval. The kernel is not an atomistic or electromagnetic predictor; it does not query an oracle; and it has no HDC/FEP integration yet.

The tests establish contract behavior only when executed. They cannot establish that descriptor-space diversity maps to physical utility. That must be measured using held-out outcomes in an appropriate external benchmark.

## 2. Tiered acceptance plan

### T0 — Contract and provenance

- Exact feature/preprocessing contract is frozen and identified by SHA-256.
- All feature transforms are fitted only on the permitted training partition.
- Each candidate has a stable ID; duplicates and missing family lineage are handled explicitly upstream.
- Target spectra/energies, oracle results, predictions, official split labels and post-outcome data are absent from the request.
- All source/data/model rights and artifact identities are recorded externally.

**Gate:** any target leakage, unresolved use-rights blocker, unaccounted candidate or ambiguous transform blocks a scientific comparison.

### T1 — Deterministic software invariants

The dedicated Rust unit tests cover:
- no duplicate IDs in selected candidates and no selection count above budget;
- repeatability for a fixed random seed and input;
- deterministic farthest-first behavior independent of input-row ordering;
- structural receipt verifier recomputes request/response SHA-256 binding, selected-ID membership, family/cost lineage, budget/rank invariants, and nearest-selected descriptor distances from the frozen request;
- cost-aware anchor/selection behavior and missing/non-positive cost rejection;
- expected family spread from the group-balanced policy;
- rejection of duplicate IDs, dimension mismatch, out-of-range/non-finite features and zero budgets;
- rejection of unrecognized fields, including a target-energy field;
- bounded JSONL input handling at the 16 MiB request limit, including oversized-line digest/accounting, record resynchronization and CRLF chunk-boundary fixtures;
- one-line request digesting and an end-to-end checked-in fixture.

**Execution status:** these tests are authored but have not yet run in the current environment because cargo/rustc are not installed locally. Treat the focused workflow result separately; queued or skipped remains unverified.

### T2 — Algorithm smoke/performance

Use a clearly labelled synthetic descriptor fixture (no material claim) to compare:
- 2,000 candidates × 64 dimensions with budgets 10, 100 and 250;
- up to 10,000 candidates × 128 dimensions with budgets 10, 100 and 250, subject to explicit resource caps.

Record exact CPU, runtime/toolchain, input digest, feature-contract digest, wall time, and peak RSS. Proposed engineering guardrail for the larger smoke case: p95 selector latency <=5 seconds on a named reference CPU, with all selected IDs unique and within the requested budget. This is a proposed feasibility threshold, not an achieved measurement and not a scientific quality target.

For reproducibility, run identical inputs twice and compare full responses after verifying the same request digest. Permuting candidate row order should not affect farthest-first selected IDs. The seeded random policy should be stable for a fixed candidate set, seed and implementation revision.

### T3 — Domain-specific decision utility

Do not use synthetic target scores as evidence for nanotechnology. Choose one real task only after source/data rights, official split/lineage and baseline are established.

**For MOCLIP-like photonics**
- Freeze geometry/structure inputs, target spectra, wavelength grid, polarizations, units, normalization, official split membership and query budget.
- Prefer held-out *target queries* and candidate geometries with mapped official train/validation/test assignment. The public 10k subset is not an official holdout until each row is mapped.
- Compare random selection, nearest spectral-library retrieval (if allowed), the relevant published MOCLIP/adjoint baseline, farthest-first, family-balanced farthest-first, and a future Symthaea-specific policy.
- Use one preregistered primary outcome: Success@K under the fixed spectral-MSE/constraint threshold, or best-so-far spectral MSE at a fixed evaluation budget.
- Re-simulate finalist designs with an independently pinned forward solver and track robustness under frozen geometry/wavelength perturbations.

**For MADE convex-hull discovery**
- Use the exact pinned MADE environment/oracle and source-defined discovery metrics.
- Keep policy-visible candidate features separate from hidden stability labels/convex-hull outcomes.
- Count invalid/duplicate candidates and oracle queries against every arm's budget.
- Report only results for the tested MADE task; this is not an OC22 or optical benchmark.

Never compare MOCLIP spectral MSE with MADE discovery counts or treat one as an aggregate cross-domain score.

## 3. Metrics

### Contract metrics (software)

- invalid-request rejection rate and error category;
- selected count and budget compliance;
- unique-ID rate (target 100%);
- deterministic replay rate (target 100% for identical request bytes and revision);
- input receipt verification rate (target 100%);
- p50/p95 selection latency and peak RSS.

### Descriptor diagnostics (not scientific outcomes)

- minimum pairwise descriptor distance among selected candidates;
- representation-space coverage and redundancy;
- number of declared families represented;
- family-count entropy or imbalance when family lineage is reliable.

These diagnose what a policy optimizes. They are not a proxy for energy, spectral performance, novelty, safety, or synthesizability.

### Scientific outcome metrics (domain-specific)

- **Photonics:** Success@K, wavelength/polarization-aware spectrum error, robustness under geometry perturbation, cost per valid design.
- **MADE:** source-implemented discovery curves and budgeted discovery utility, independently recomputed from trajectory receipts.
- **Measured optimization pools (e.g. NIST Fe–Co–Ni):** normalized simple-regret / best-loss reduction at budget, target-attainment curves and success-at-threshold; report Kerr and coercivity separately. The continuous-loss comparison requires complete paired target coverage, otherwise it emits no inferential interval.
- **Distribution-shift / atomistic studies:** per-group success and reliability curves, minimum (worst-group) success rate/AUC/coverage, shift-severity curve joins to a separately verified shift manifest, and adaptation cost by shift group.
- **Other domains:** task-specific metrics and units only; define them in their own scorecard before scoring.

A pooled metric is not a safe decision criterion when groups represent different materials, surface states, temperatures, or fabrication regimes. Report the weakest predeclared group beside pooled averages; the group manifest establishes split membership, not that its physical taxonomy is scientifically adequate.

## 4. Statistical protocol

- Random baseline: at least 30 fixed preregistered seeds when the system/oracle cost permits; retain every seed's outcome.
- Deterministic selectors: use multiple independent systems/target queries and group-aware bootstrap; repeat identical requests for software determinism, not as pseudo-replicates.
- Confidence intervals must resample the true independent unit (e.g. geometry family, chemistry system, batch, device, or target query), not correlated rows alone.
- Do not tune the descriptor transform, budget, policy parameters or primary metric against the final holdout.
- Report the whole budget curve when possible; do not report only the best final point.
- For shift studies, report each declared group and the worst-group success/AUC/coverage curve. Join group IDs to the frozen shift manifest; never infer shift severity from outcomes or reorder severity labels after unblinding.
- Use a fixed, domain-reviewed worst-group guard in addition to the primary pooled metric. A favorable pooled average cannot compensate for a predeclared critical group crossing its hard physical/feasibility floor or exceeding its regression margin. If per-group sample/power is insufficient, report the worst-group estimate as descriptive and do not claim a robust shift improvement.

## 5. Proposed go/no-go rule

For a genuine task that supports budgeted selection, compare the best Symthaea policy against the strongest eligible baseline under equal information, oracle/query budget, time constraints and candidate pool.

Provisional threshold: +5 percentage points in a preregistered success-at-budget metric **or** at least 20% lower median best-achieved error/cost, with a group-aware 95% confidence interval excluding no improvement, no leakage-control failure, and 100% reconstructible scored-run receipts. These are pilot targets, not universal standards; lock the primary metric, practical margin and sample/power rationale before opening the holdout.

If the policy only increases descriptor-space distance or family coverage but does not improve the task-specific outcome, report a diversity result—not a discovery improvement.

For shift-aware tasks, promotion additionally requires that no predeclared critical group violates its frozen non-inferiority/safety boundary. Do not choose that boundary universally: it depends on the scientific unit, measurement noise, risk of the predicted property, and powered sample size; write it into the task contract before evaluation.

## 6. Negative controls

- Add a forbidden target label to a request: deserialization must reject it.
- Permute descriptor dimensions or shuffle descriptor rows under a frozen recipe: any outcome improvement should disappear or be explicitly investigated.
- Permute family IDs: family-balanced gains should weaken if family metadata was doing meaningful work.
- Inject duplicate candidates: the input contract should reject duplicate IDs; upstream structure-equivalent duplicates need their own canonicalization report.
- Hide/reveal-label canary in the benchmark adapter: selector must not change before the oracle reveals the result.
- Enforce the exact same query budget and charge all failed/invalid/duplicate attempts as declared in the benchmark protocol.

## 7. Future Symthaea-specific arm

**Research hypothesis (not a result):** combine inexpensive HDC representation-space retrieval with an uncertainty-calibrated, cost-aware acquisition rule so Symthaea spends expensive reference calls where they can change the scientific decision. For trajectory tasks, a temporal model may help forecast deteriorating stability, but it is a separate component and must be tested against an explicitly defined temporal baseline. Novelty and temporal-model confidence must never bypass physical feasibility constraints.

The proposed two-stage policy is:
1. **Cheap screening:** encode admitted structure/metadata descriptors into a versioned HDC representation; calculate redundancy, candidate coverage and similarity to known valid/failed domains. Treat distance as a novelty diagnostic—not physical validity or confidence.
2. **Risk and value gate:** combine validated uncertainty or error-bound evidence with a task-specific estimate of expected value of information per all-in acquisition cost. If the candidate is clearly infeasible, reject it locally; if it is out-of-domain and low-information, abstain or ask for human review; prioritize candidates expected to change the decision or resolve a safety-critical uncertainty.
3. **Reference feedback:** after each oracle result, update only the training/acquisition state permitted by the frozen sequential protocol. Never feed post-outcome labels into the policy during a static held-out comparison.
4. **Independent audit:** emit exact selection receipts, local rejections, cache hits, external oracle calls, retries, wall/resource cost, and model/data revisions. A separate evaluator reconstructs utility and all-group outcomes from the frozen outcome ledger.

Mandatory arms for this hypothesis:
- random feasible candidate;
- max-min diversity / farthest-first;
- HDC novelty-only;
- uncertainty-only (where the model supports qualified uncertainty);
- cost-aware uncertainty or expected improvement;
- HDC + calibrated uncertainty + value-of-information/cost;
- passive/no-adaptation control, plus an oracle/selection negative control as applicable.

Keep the candidate pool, initial information, allowed features, reference oracle, compute/time cap, and total-cost denominator identical. Do component ablations (remove HDC novelty, calibrated uncertainty, or cost normalization one at a time) so the result shows what actually caused the change. Freeze all model training exposure and preprocessing before opening grouped test systems.

**Promotion targets:** target at least 20% fewer actual reference calls to reach a preregistered trajectory-reliability threshold; for error-triage experiments, target at least 90% severe-error recall while reviewing no more than 20% of windows. Both require no worsening of frozen hard-constraint violations, improvement across the predeclared worst-group guard, and adequate power across independent material/system groups. These are research targets; do not claim HDC, active inference, or a temporal model helps until this matched-budget ablation passes.

Do not label the present max-min selector as “Symthaea intelligence”; it is a deterministic baseline kernel intended to make future contributions falsifiable.

## 8. Research context: why the baseline set must expand

This kernel currently offers random, descriptor-diversity and family-balanced descriptor-diversity policies. That is an intentional first step, but it is **not yet a complete active-learning loop**: the current v1 selector does not update a predictive model or consume revealed labels between selection rounds.

External evidence argues for testing additional approaches:
- Wang et al. (2022), NIST, benchmarked active-learning acquisition strategies on a real Fe-Co-Ni thin-film library with X-ray diffraction, magnetic coercivity and Kerr-rotation properties; Expected Improvement performed best overall in that particular problem. This is a useful reference for task/metric design, not a universal winner: https://www.nist.gov/publications/benchmarking-active-learning-strategies-materials-optimization-and-discovery
- Bi et al. (2025), *Scientific Reports*, compared 17 active-learning strategies plus random sampling over nine small materials-formulation datasets. The study reports that uncertainty-driven and diversity-hybrid methods often led early, while gaps narrowed as labeled sets grew. This motivates reporting the **whole label-efficiency curve**, early-budget performance and final utility instead of only one endpoint: https://www.nature.com/articles/s41598-025-24613-4
- The public [Coactive Learning reference implementation](https://github.com/coactivescience/cost-aware-active-learning) at commit `33f29e85d3a464a75b4cda2d2202e3ef91b80c48` proposes a two-stage EC² pathway-identification plus cost-aware GP-UCB optimization policy. Its README explicitly characterizes the experiment as a CICERO-inspired *synthetic simulator*, not new wet-lab data. It is a useful architecture/control reference, but its outputs are not a nanophotonics or OC22 benchmark and it should not be presented as physical validation.
- The 2026 MatBench reproducibility analysis reported that only 2 of 28 published submissions installed/imported from original dependency metadata; 13 submissions completed a selected task after remediation. Reproducibility and metric regeneration must therefore be scored explicitly: https://www.oaepublish.com/articles/aiagent.2026.31

### Additional hypothesis: two-sided response-extreme sampling (future sequential loop, not selector v1)

A 2026 *Acta Physica Sinica* study reports that deliberately sampling both positive and negative response extrema improved generalization in its materials-discovery experiments compared with uniform or one-sided sampling: [Data Distribution Matters: Enhancing model generalization in Data-Driven Materials Discovery by intentionally exploiting both positive and negative data](https://doi.org/10.1016/j.actphy.2026.100291). This is a promising, task-specific hypothesis—not proof that extreme sampling always wins.

Keep its objective distinct from immediate design exploitation:
- **Exploitation:** query candidates predicted most likely to meet the target now.
- **Two-sided model coverage:** query selected candidates from both low- and high-predicted-response tails to improve the surrogate's learned response range and future generalization.
- **Uncertainty sampling:** query candidates for which a suitably validated estimator expects model uncertainty to be high.
- **Diversity:** cover descriptor/family regions independent of predicted response.

The present v1 kernel accepts neither target labels nor predicted properties, intentionally. Do not smuggle model predictions into its descriptor fields. Implement two-sided extremes only in a separately versioned sequential-loop layer after a task-specific model contract exists: train on revealed training labels, emit versioned predictions for the unqueried pool, choose tail candidates under a frozen policy, reveal selected labels only after the query, then log the model update and regenerate the next pool predictions.

Measure its effect as **model-learning quality**: MAE/RMSE across the full held-out response range, error by response quantile and family, calibration/coverage where probabilistic predictions are claimed, and learning curves by acquired-label count/cost. Separately report immediate Success@K or cost to target. Do not infer an improvement in candidate discovery from lower predictive error alone, or predictive generalization from a handful of successes.

### Required next baselines before claiming Symthaea advantage

Once a real task and data contract are admitted, extend the batch-policy comparison to include:
1. random selection with preregistered seeds;
2. deterministic farthest-first and family-balanced farthest-first;
3. task-appropriate conventional selector (e.g. nearest-neighbour retrieval, expected improvement, uncertainty sampling, or adjoint optimizer);
4. cost-aware farthest-first as a transparent baseline when query costs are available and defensibly estimated;
5. cost-aware active acquisition (e.g. cost-normalized expected improvement or expected value of information) where a suitable predictive model/cost model can be preregistered;
6. one Symthaea-specific HDC/active-inference arm and a component ablation;
7. leakage, shuffled-feature/family, duplicate, and hidden-label controls.

Keep primary outcomes distinct:
- **Discovery/selection:** success-at-budget, top-k regret/utility, best-so-far task loss, and cost to hit a predeclared goal.
- **Efficiency:** area under the budgeted success/improvement curve, oracle calls and dollar/time cost per valid success.
- **Coverage diagnostics:** descriptor distance and family coverage, explicitly not physical utility.
- **Reproducibility:** exact-source replay and independent score reconstruction.

For active-learning experiments, fit the surrogate/model only on revealed training observations, expose each query's label only after selection, and measure across a preregistered set of random seeds and held-out systems. A diversity-policy win on a synthetic geometry fixture is only a software behavior test, never a nanotechnology discovery claim.

## 9. Independent post-outcome metrics evaluator

A separate `src/metrics.rs` module and `nano-policy-metrics` binary now evaluate a frozen JSON outcome ledger after all selector/oracle actions are complete. The ledger includes task/source/evaluation manifest digests, policy revision and run/selection-receipt digests, paired target IDs, caller-declared independent groups, chronological candidate attempts, evaluated loss/status and per-call cost.

The evaluator rejects:
- malformed optional shift-manifest digests and mismatched target or group sets across policy arms;
- non-contiguous query ranks or query count beyond the common budget;
- missing/non-finite/negative loss on an evaluated attempt, or a loss attached to a failed attempt;
- non-finite/negative cost, invalid digests, duplicate arm/target IDs, and unknown fields;
- unbounded ledger size, arm count, target count, bootstrap replicates, or observation count.

Computed metrics include success-at-budget (all target queries are in the denominator; no usable outcome is failure), normalized success AUC over the full budget curve, coverage, valid/failure rate, repeated candidate attempts, best loss conditional on coverage, mean cost per target, and cost-to-first-success. Failure/timeout cost stays in the cost denominator. Metrics-report v2 also emits each group’s full success curve, target coverage, success rate and cost, plus explicit worst-group success rate/AUC/coverage. The input outcome ledger can carry an optional shift-manifest SHA-256, which the report preserves so detached group curves remain linked to the exact manifest. This prevents an improvement on easy groups from appearing sufficient when performance collapses on a held-out system family. Metrics reports are tied to the exact raw input JSON bytes with SHA-256.

For each non-baseline arm, it computes paired deltas for final-budget success, normalized success AUC, final-budget best-loss reduction (baseline best loss minus candidate best loss; positive means candidate improvement), and two robust endpoints: `worst_group_success_rate_at_budget` and `worst_group_normalized_success_auc`. Each robust endpoint is the minimum candidate-minus-baseline mean across the predeclared groups; it directly catches cases where the pooled average improves but at least one group regresses.

The loss-reduction comparison requires valid outcomes for every paired target in both arms; if any target lacks a valid outcome, it marks `insufficient_target_coverage` and emits no confidence interval. Pooled comparisons bootstrap whole groups with target pairs intact; fewer than five declared groups yields `insufficient_independent_groups`.

Worst-group comparisons use a stricter interval rule: at least five declared groups and at least two paired targets per group are required. The evaluator bootstraps paired target deltas within each frozen group, constructs Bonferroni-adjusted percentile bounds across groups, and takes the minimum lower and minimum upper group bound as a conservative interval for the worst-group effect. If any group has fewer than two paired targets, it reports a descriptive worst-group delta without an interval and marks `insufficient_within_group_replication`. A synthetic negative control makes the pooled success delta positive while one group is worse, and checks that the worst-group comparison detects the regression.

These minimum sample guards are **not** a power guarantee and do not prove group IDs are independent, shift labels physically valid, loss semantics correct or results scientifically useful. Intervals remain unadjusted across arms and across pooled/worst-group metrics; confirmatory claims must preregister a primary endpoint or use a declared multiplicity correction. The evaluator is separate from the receipt verifier, which checks selection-output structure only.

Unit tests use synthetic ledger fixtures to test budget success, full-curve AUC, all-target denominators, failure and repeat counting, cost aggregation, per-group curves and worst-group regressions, exact raw-byte digest binding, paired deterministic bootstrap, insufficient group count, best-loss reduction with complete coverage, refusal to infer loss improvement under incomplete coverage, mismatched target sets, query ranks, failed-loss rejection, observation-digest validation, unknown fields and malformed digests. The CLI's fixture is explicitly synthetic and is not a science result.

## 10. Current status

- The selector kernel, structural receipt verifier, outcome-ledger metrics module/CLI, request/response/receipt/outcome/report schemas, selector/outcome fixtures, README/benchmark contracts and focused workflow are committed on the draft PR branch.
- The outcome evaluator now has 20 authored unit tests, including paired worst-group comparisons with an adjusted interval that catches a regression hidden by a positive pooled result, plus optional shift-manifest digest propagation/validation. The report contract is now `nano-benchmark-metrics-v2`; the outcome-ledger input remains v1. An independent JavaScript arithmetic check over the checked-in synthetic ledger matches the test expectations for baseline success rate/AUC, failure accounting, candidate repetitions, all-in cost and paired success-rate delta. The JSON fixture and report schema were fetched back and parsed as JSON.
- The independent arithmetic and JSON parsing are source/fixture checks, **not** Rust compilation or test execution. No `cargo`, `rustc` or `rustfmt` is installed in the current environment, so formatting and tests remain unverified locally.
- The exact-head Rust workflow status must be read separately; queued/skipped is not a pass. No scientific data were scored and no selector improvement is claimed.

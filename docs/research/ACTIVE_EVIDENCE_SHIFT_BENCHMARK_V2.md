# Active-evidence selection under distribution shift: benchmark protocol v2.0

**Status:** research protocol proposal; not an implementation or performance claim.
**Date frozen for review:** 2026-10-28
**Related tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/8691
**Claim ceiling:** v1.9's 19×19 interpolated risk grid is synthetic sensitivity analysis. Its 361 cells are not independent experiments and do not establish real-world superiority.

## 1. Research question

Can decision-aware robust acquisition reduce cumulative downstream decision regret under previously unseen distribution shifts while retaining most of the value of cheaper, less conservative acquisition when the predictive model is correctly specified?

This is a two-sided question. A robust policy can fail by being too optimistic under shift, or by paying a robustness tax under IID conditions. We must measure both.

## 2. Experimental unit and leakage boundary

The primary experimental unit is a **task × held-out shift family × seed** trajectory, not a scenario-grid cell, individual query, or repeated row from one synthetic risk surface.

Partition data into development tasks for implementation/debugging, validation tasks/shifts for hyperparameters and ambiguity-set construction, and sealed test tasks and shift mechanisms used once after the protocol is frozen. Where possible, split by source, domain, environment, time, or causal mechanism. Never randomly split correlated rows from the same source into train and test. Keep the test generator seed and hidden shift settings outside policy code.

## 3. Baseline set

Every method gets the same initial labeled set, candidate pool, label oracle, per-query costs, compute allowance, and stopping rules.

1. Random/uniform sampling.
2. Entropy sampling.
3. Margin sampling.
4. Diversity/coverage baseline; evaluate Uncertainty Herding only if implementation matches its objective and budget semantics.
5. Cost-normalized uncertainty.
6. Plug-in EVSI using the same downstream action/loss model as the robust method.
7. Robust EVSI / predictive-ambiguity acquisition.
8. Optional R-IDeA-compatible arm only after its representativeness, informativeness, and error-de-amplification terms are implemented faithfully and documented.
9. Hindsight oracle that knows the true evaluation distribution, labelled **unattainable reference**, never counted as a deployable baseline.

No baseline may receive hidden test labels, target-domain prevalence, or true shift parameters unless that information is explicitly part of its deployment contract.

## 4. Shift matrix

Evaluate separate and compound regimes: IID/correct specification; class-prior shift; covariate shift; conditional/concept shift; class-conditional asymmetric measurement errors; rare classes and worst-group prevalence; class-dependent label noise; cost shift and missing/failed queries; in-ambiguity-set shift; out-of-ambiguity-set shift; and compound shift combining at least two mechanisms.

Use multiple severity levels fixed before test. State which assumptions make importance weighting or calibration valid. The out-of-set regime is essential: a robustness guarantee is conditional on the declared ambiguity set, not a guarantee against arbitrary shift.

## 5. Multi-round protocol

At each round, a policy receives only its permitted state: current labeled examples, unlabeled pool, allowed model outputs, costs, and declared uncertainty/ambiguity information. It selects one query or abstains. The label is revealed, the model/calibrator is updated by a frozen procedure, and the next round begins.

Freeze initial labeled-set sizes and class/group composition; pool construction and label-oracle behavior; update/retraining schedule; candidate-query cost model; policy random seeds and tie-breaking; total budgets and checkpoint budgets; handling of failed, invalid, duplicate, and abstained queries; stopping and model-selection rules.

Use a budget curve rather than a single budget. Prefer at least 20 paired seeds for an initial confirmatory run when computationally feasible; determine the final seed count using a predeclared power/precision analysis on development data, never by repeatedly inspecting held-out results.

## 6. Primary metrics

### Primary outcome
**Cumulative downstream decision regret per unit acquisition cost**, with regret defined against the best action under the evaluation truth/loss model, not the policy's own assumed model.

Report regret at fixed cumulative-cost checkpoints; area under regret-versus-cost curve; paired difference against the strongest non-oracle baseline; task/shift-level clustered confidence interval; mean and median plus 95th percentile, worst 5%, and CVaR where sample size supports it.

### Secondary outcomes
- Downstream task utility/loss.
- Label efficiency at predeclared target quality.
- Harmful-acquisition rate and magnitude (queries that increase expected downstream regret versus abstention).
- Abstention rate and decision coverage.
- Calibration and ambiguity-set coverage, separately from decision quality.
- Worst-group and rare-class performance.
- Query latency, wall-clock, peak memory, and acquisition compute.
- Regret under correctly specified IID conditions (robustness tax).

Do not collapse all outcomes into one score unless the scalar utility and weights are predeclared and sensitivity analysis is reported.

## 7. Statistical analysis

Use paired seeds and identical task/shift draws for all policies. Resample at the independent task/source/shift-family level; do not bootstrap query rows as if independent. Report per-regime estimates before any pooled estimate. Predeclare the primary contrast, interval method, multiplicity handling, and missing-run policy. Publish every seed and task result; aggregate metrics must be regenerated from those rows.

Distinguish confidence intervals for a population-level estimate from exact enumeration over a finite synthetic grid. Keep exploratory scenario discovery separate from the sealed confirmatory evaluation. If the number of independent task families is too small for reliable inference, label results exploratory and report descriptive intervals without overstating generalization.

## 8. Provisional gates

These are engineering goals, not industry standards. Freeze or revise them using development data before the confirmatory run.

- **Shift benefit:** at least 20% lower mean cumulative decision regret than the strongest non-oracle baseline on the prespecified held-out shift aggregate, with the paired interval excluding no improvement.
- **IID robustness tax:** no more than 1% relative degradation in the primary IID task metric against the strongest matched baseline.
- **Budget consistency:** improvement must persist at multiple predeclared budgets, not one favorable point.
- **Tail safety:** no material regression in worst-group/tail regret; set a numeric bound before test.
- **Efficiency:** report acquisition overhead and regret per cost; no success claim if savings disappear after accounting for query costs or compute.
- **Integrity:** all frozen-input checks and negative controls pass; no result qualifies if required inputs, rows, or digests are missing.

Failure of a gate is a valid result. Do not tune on the sealed test set or weaken gates after observing results.

## 9. Reproducibility and adversarial controls

The release artifact must contain source revision and exact file digests; frozen parent and child contracts as actual files, not just hashes; environment/toolchain metadata and one-command runner; raw per-trajectory outcomes, derived metrics, and an independent aggregate verifier; deterministic seeds and explicit tie-breaking; negative controls for missing contracts, one-byte contract mutation, swapped parent/child, altered output rows, duplicate trajectories, and leakage of test-only fields; and a clean-room run from a fresh checkout with no pre-existing generated files.

A digest match proves byte identity, not correctness. An aggregate verifier must recompute metrics from raw rows rather than trust the evaluator's summary JSON. Keep arithmetic/reference checks, Rust compilation/tests, workflow checks, and empirical evaluation as distinct evidence classes.

## 10. Decision rule for research claims

- **Synthetic sensitivity only:** deterministic scenarios derived from fixed assumptions.
- **Finite simulator result:** independently generated trajectories under a stated simulator, with uncertainty scoped to that simulator.
- **Benchmark result:** matched baselines, held-out task/shift families, preregistered metrics, raw outcomes, and statistical uncertainty.
- **Deployment evidence:** prospectively collected target-domain evidence and monitored operational outcomes.

Do not describe any earlier evidence class as a later one. A win over plug-in EVSI alone is insufficient; compare against strong simple baselines and test whether robustification helps when it should and abstains when it cannot justify a recommendation.

## 11. Literature alignment

- [Uncertainty Herding, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0302fb83c62991efbccf0a003e4f5a92-Abstract-Conference.html): compare only with a compatible implementation and matched budget.
- [R-IDeA, AISTATS 2026](https://proceedings.mlr.press/v300/tang26d.html): useful framing for representativeness, informativeness, and error de-amplification under misspecification; this protocol does not claim to reproduce it.
- [NIST AI 800-3](https://doi.org/10.6028/NIST.AI.800-3): motivates explicit estimands and statistical modeling rather than generalizing from one fixed benchmark.

## 12. Current blocker

The local v1.9 report states that the evaluator checks both the v1.9 contract and parent v1.8 digest, but the reproducible-package audit found two referenced frozen contract files missing. Recover the original exact bytes or mark the v1.9 package incomplete. Do not fabricate replacements from aggregate metrics. The v2.0 confirmatory run is blocked until the v1.9 provenance issue is resolved or the work is explicitly forked into a new version with new frozen inputs.

## 13. New exploratory algorithmic pilot: exact policy trees and a cost frontier

Tracking implementation: [PR #8723](https://github.com/Luminous-Dynamics/symthaea/pull/8723).

A dependency-free finite synthetic model now enumerates all 3,137 observation-contingent query trees up to depth three for a fixed three-hypothesis/four-query task. Unlike one-step greedy robust EVSI, the optimizer chooses a complete fixed query tree against a four-model ambiguity set and can additionally constrain worst-case expected acquisition cost. It does not switch the adversarial model independently at every branch.

Independent Python reference findings on this hand-defined model:
- Worst-case regret over the four ambiguity models at depth three: 1.7780 for one-step robust greedy versus 1.6090 for the selected full tree.
- At maximum expected-cost budget 2.632, best of the implemented greedy arms (random, entropy, plug-in EVSI, one-step robust EVSI) has worst-case regret 1.7780; the full tree has 1.6090.
- At expected-cost budget 3.0, entropy has worst-case regret 1.7359 while the full tree has 1.6090 at expected cost 2.632.
- The deliberately out-of-ambiguity-set measurement flip still raises full-tree regret to 3.5133 versus 2.3000 for no query.

This is an algorithmic pilot, **not confirmatory evidence**. The best greedy baseline is selected on the same four scenarios; only four hand-defined ambiguity models are used; the cost limit is on expected cost rather than each trajectory; and no Uncertainty Herding or R-IDeA implementation is included. The Rust binaries and independent CSV parity gate are committed, but Rust compilation/tests and exact-head workflow qualification have not yet been confirmed.

This pilot suggests the next research question should be: **can a sequentially calibrated, fixed-horizon decision-tree policy maintain its regret advantage when the ambiguity set is estimated from calibration data rather than hand-coded, and when the test shift family is held out?** The main confirmatory protocol above remains unchanged: this pilot may be used for development, not as the held-out test.


## 14. Separate prediction, decision, and discovery objectives

Do not report one blended "active-learning success" score. Pre-register distinct outcomes for:
- **Predictive learning:** held-out predictive risk/calibration after a fixed annotation budget.
- **Decision support:** cumulative downstream decision regret under a frozen task-loss model.
- **Discovery/exploitation:** recovery of high-value candidates under a preregistered utility definition.
- **Safety / selective prediction:** risk-coverage frontier and harmful-action rate, with abstention charged explicit operational costs.

A recent real-data active-learning benchmark on 1,202 experimental rare-earth extraction measurements with prospective validation on four independently synthesized ligands found that acquisition rankings depended on the objective: exploitation recovered top candidates but produced a weaker predictive model; broader exploration/random selection was better for predictive fit; and active learning did not improve prediction over random acquisition in that study. It also reported that row-level predictive R²=0.94 collapsed to 0.20 through -0.03 under leakage-controlled group splits by ligand/source/structural family. Those results are domain-specific, but they are a direct warning against random-row splits and conflating discovery with predictive quality. Reference: https://pubmed.ncbi.nlm.nih.gov/42626177/ (Harish, ACS Omega, 2026).

Implementation consequence: any real-task extension must split by task/source/domain/causal family, use a matched acquisition unit (including batches if the operational label unit is a group), and report each objective separately. If there is a genuine business or scientific utility tradeoff, report the Pareto frontier; do not hide the tradeoff in an unvalidated weighted sum.

The ICLR 2025 Uncertainty Herding work specifically targets the low-/high-label-budget tension, so every acquisition comparison must include low through high budgets and cannot tune the "low-budget" boundary after looking at test results: https://proceedings.iclr.cc/paper_files/paper/2025/hash/0302fb83c62991efbccf0a003e4f5a92-Abstract-Conference.html.

Finally, apply the generalization distinction emphasized by NIST AI 800-3: benchmark performance on a fixed scenario set and expected performance over a broader task population are different estimands. State which one is reported, and only generalize with an analysis design whose independent units match the population claim: https://doi.org/10.6028/NIST.AI.800-3.

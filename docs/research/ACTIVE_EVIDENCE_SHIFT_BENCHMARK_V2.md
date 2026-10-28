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


## 15. Sequential calibration must be anytime-valid

The calibration-derived ambiguity set can be queried and inspected repeatedly while the acquisition policy decides whether to stop. A fixed-sample confidence interval should not be reused as though it had its nominal coverage at every data-dependent stopping time. The implementation pilot now includes both:
- fixed-sample Hoeffding intervals, for a preregistered single-look calibration baseline;
- a transparent time-uniform Hoeffding/alpha-spending sequence with per-cell/time failure mass alpha/[M n(n+1)], where M is the number of likelihood cells.

Since the time weights sum to one, the latter retains familywise coverage over all scheduled looks under independent stationary Bernoulli observations per cell. It is a conservative correctness baseline. It does not solve model misspecification, nonstationarity, selection bias in the calibration sample, or arbitrary distribution shift.

For an efficiency frontier, compare the same acquisition objective under fixed-sample Hoeffding, alpha-spending Hoeffding, and a variance-adaptive empirical-Bernstein confidence sequence. Use matched label and compute budgets; report interval width and acquisition regret, not coverage alone. Relevant foundations: Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences* (Annals of Statistics, 2021, https://doi.org/10.1214/20-AOS1991), and Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting* (JRSS B, 2024, https://doi.org/10.1093/jrsssb/qkad009).

Exploratory implementation and independent Python checks live in [PR #8723](https://github.com/Luminous-Dynamics/symthaea/pull/8723). They are not yet Rust-qualified until exact-head workflow checks finish.


## 16. Feature-support-aware acquisition is a separate OOD problem

A peer-reviewed open-access article published 24 October 2026, “When active learning fails, uncalibrated out of distribution uncertainty quantification might be the problem” (npj Computational Materials, DOI https://doi.org/10.1038/s41524-026-02368-4), evaluates uncertainty estimation/calibration for solubility, bandgap, and formation-energy tasks. It reports calibrated uncertainties were generally unsuccessful at reducing data needs for OOD model generalization versus random acquisition and uncalibrated uncertainties, and points to uncertainty in feature-input space as a potential bottleneck. This is a domain-specific result, not proof that active learning generally fails; it does demonstrate that in-domain calibration alone is not enough to justify OOD acquisition claims.

Tracked experiment: [issue #8749](https://github.com/Luminous-Dynamics/symthaea/issues/8749).

Add matched arms for random, entropy/margin, Uncertainty Herding where compatible, plug-in/robust EVSI, feature-space support/coverage acquisition, support-gated robust EVSI, and a randomized sentinel-audit budget. Keep the oracle separate. All arms must share initial labels, query costs, pool, update schedule, compute allowance, and target data boundary.

The test matrix must split by task/source/domain rather than random rows only. Include in-support covariate shift, feature-support OOD, high-confidence conditional shift, irrelevant outliers, rare groups, and compound shifts. Report (a) predictive risk/calibration, (b) downstream decision regret, (c) feature-support coverage and high-risk uncovered mass, (d) false-confident errors in unsupported regions, (e) query-cost and regret frontiers, and (f) sentinel false alarms/delay and budget share. Do not collapse these into one weighted score without a prespecified utility and sensitivity analysis.

Critical falsification controls: an uncertainty estimator calibrated only in-domain; high-confidence OOD examples with a changed conditional mechanism; OOD anomalies irrelevant to target decisions; a held-out feature cluster; and corrupted/omitted support metadata. Sentinel benefits must be measured after including the labels and compute they consume. Use independent task/source families for uncertainty intervals, not query rows.

The proposed success criterion is not “lower calibration error.” It is a preregistered improvement in the selected primary OOD objective (predictive risk or decision regret) versus the strongest matched non-oracle baseline, accompanied by no material IID/in-support regression and a transparent robustness tax. This is a new research track, not a result established by the current synthetic Bernoulli fixtures.


## 17. Sequential shift monitoring is a separate acquisition-system metric

A selector may inspect calibration or monitoring evidence after each query, so fixed-time intervals reused at every look are not valid substitutes for anytime-valid evidence. Tracking implementation: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744).

The exact Bernoulli monitor microbenchmark compares four arms:
- one valid fixed-time look at a preregistered horizon;
- fixed-time Hoeffding intervals reused at every observation (negative control);
- time-uniform Hoeffding with explicit alpha spending across 12 cells;
- a predeclared Beta(1/2, 1/2) mixture likelihood-ratio e-process, with Ville threshold 1/(alpha/12).

On the synthetic p0=0.5 stationary null, exact DP gives false-alarm probability by horizon 1,000 of 0.000442 for the final-only interval, 0.009366 for repeatedly reused fixed-time intervals (above per-cell alpha 0.004167), 0.000000451 for alpha-spending Hoeffding, and 0.003059 for the Beta-mixture e-process. Under an abrupt p=0.5 to p=0.8 shift after 100 observations, the time-uniform Hoeffding detector's horizon-500 detection probability / conditional mean delay is 0.999999604 / 157.50 observations, versus 0.997539 / 94.76 for the Beta mixture. Across a wider shift-magnitude grid at horizon 1,000, the Beta mixture raises detection for p=0.40 from 30.95% (Hoeffding) to 94.28%, and for p=0.45 from approximately 0.05% to 11.84%. These are exact computations for a finite synthetic Bernoulli monitor, not production drift-detection performance.

This is an explicit **false-alarm / detection-delay / power frontier**, not a single winner: the Beta mixture is less conservative and faster for some alternatives while slightly reducing detection probability for the p=0.8 scenario. Repeated fixed-time peeking appears fastest but violates the requested null false-alarm budget and is only a negative control. The next monitor benchmark should also include gradual ramps, temporary bursts, serial dependence, cell multiplicity, and selection-biased streams.

The integrated future selector should report monitor false alarms and detection delay jointly with cumulative decision regret, feature-support coverage, label and compute costs, and the randomized sentinel budget. A shift alarm is not itself evidence that an additional label improves the target decision; the system should update the ambiguity set or abstain only under an explicitly validated policy.


## 18. Separate population shift from acquisition-induced selection shift

The sequential monitor now contains an exact finite-state counterexample showing why a shift alarm on adaptively acquired labels is not necessarily evidence that the target environment changed. In the closed-world fixture, two equally prevalent target strata have event rates 0.8 and 0.2, so the target-population rate is 0.5. If an acquisition policy queries the high-event stratum 90% of the time and the low-event stratum 10% of the time, the observed selected stream has rate 0.74 even though the target population is unchanged.

At horizon 1,000, the Beta-mixture e-process has exact alarm probability 0.003059 under a randomized target-population sentinel stream (per-cell alpha 0.004167), versus approximately 1.000000 under the selection-biased stream. This is not a contradiction of the e-process guarantee: under adaptive selection, the stream no longer satisfies the declared iid Bernoulli(p0=0.5) null. The finding is an identification warning about the observed-data process, not a claim that anytime-valid monitors fail under their assumptions.

Implementation: maintain separate evidence channels for (a) adaptively chosen labels and (b) randomized sentinel/audit labels drawn from the target population. If using known inclusion probabilities, evaluate the justified estimator and its overlap/positivity conditions explicitly; do not assume importance weighting repairs zero-support or unobserved strata. Charge sentinel queries and compute against the same total budget. Report environment-shift false alarms on the representative audit stream, selection-policy changes separately, and detection delay/power for genuine target-population shifts.

Tracking implementation and exact toy values: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744). The result is a mechanism-isolation fixture, not evidence that a specific sentinel rate or correction policy is optimal. Next tests should vary stratum prevalence, conditional rates, inclusion-probability misspecification, overlap failure, and budget shares, then carry the preregistered design to source-held-out real tasks.


## 19. Sentinel-budget frontier: spend labels to identify population drift

The executable monitor benchmark now emits a second frontier that treats randomized audits as a scarce resource. It fixes a total label budget of 1,000 and a target-population shift from p=0.5 to p=0.8 after 100 total labels. The sentinel channel is sampled on a regularly spaced, randomized audit cadence; non-sentinel labels remain available to the adaptive acquisition policy. Only sentinel labels are used for the target-drift monitor. The zero-sentinel arm instead monitors the adaptive selected stream and is a negative control for selection-induced alarms.

An independent exact-DP Python reference produced the following pilot results. Rust code and the independent CSV checker implement the same sweep, but these values are not Rust-qualified until the exact-head workflow completes.

| Sentinel budget share | Audit labels | Alarm by horizon when target is stable | Detection after genuine shift by horizon | Mean post-shift delay, total-label equivalents |
|---:|---:|---:|---:|---:|
| 0% | 0 | ~1.000 (spurious, selected stream only) | Not identifiable from representative audit evidence | — |
| 5% | 50 | 0.002118 | 57.69% | 614 |
| 10% | 100 | 0.002461 | 96.96% | 473 |
| 20% | 200 | 0.002711 | 99.85% | 285 |
| 25% | 250 | 0.002774 | 99.84% | 241 |
| 50% | 500 | 0.002938 | 99.79% | 148 |
| 100% | 1,000 | 0.003059 | 99.75% | 95 |

The table separates the false-alarm probability on the stable audit stream from false alarms before the true shift and post-shift detection. At larger sentinel shares, there are more opportunities for a pre-shift false alarm; post-shift detection probability alone can therefore decline slightly even while the monitor alarms somewhere with probability near one. It is not valid to optimize the post-shift detection column in isolation.

The 10% audit share reaches roughly 97% detection by the total budget for this large abrupt shift, whereas 5% reaches only about 58%. These figures are model-specific trade-offs, not a universal recommended rate. We must preregister the target shift family, primary objective, and total-label/compute budget before testing alternatives.

Acceptance for the next version: sweep audit fraction, shift magnitude and onset; test low-amplitude/gradual shifts, changing stratum prevalence, conditional shift, inclusion-probability errors, rare groups, and support overlap; report false alarms, missed shifts, post-change delay and tail risk jointly. Compare (a) random sentinel audits, (b) design-based estimators with known inclusion probabilities and explicit positivity/overlap checks, and (c) policy-conditioned monitoring. No weighting method may claim to recover unsupported regions where inclusion probability is zero. Always debit audits from the same acquisition budget and keep an oracle non-deployable.

Implementation: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744). Related OOD task and matched baseline matrix: [issue #8749](https://github.com/Luminous-Dynamics/symthaea/issues/8749).

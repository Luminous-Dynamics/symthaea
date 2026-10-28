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
9. Exploration-exploitation stratified sampling for label shift (EESAS), where its task assumptions match; implement and describe it faithfully before scoring it as a reproduced baseline. Wen et al., *Expert Systems with Applications* 328 (1 October 2026), article 132567, https://doi.org/10.1016/j.eswa.2026.132567, report label-balance problems under iterative uncertainty/diversity sampling and propose a stratified exploration–exploitation approach. This paper motivates the comparison; we have not reproduced it or claimed to beat it.
10. Hindsight oracle that knows the true evaluation distribution, labelled **unattainable reference**, never counted as a deployable baseline.

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


## 20. Shift magnitude must be crossed with sentinel share

The sentinel-budget executable now crosses sentinel shares {5%, 10%, 20%, 25%, 50%, 100%} with post-shift event rates {0.55, 0.60, 0.70, 0.80, 0.90}, with p0=0.5, a fixed 1,000-label budget, and the shift beginning after 100 total labels. The 0% sentinel arm is emitted once as a selection-bias negative control; it cannot provide a representative target-population audit channel.

Independent exact-DP Python-reference results show why no universal audit percentage is justified. At 10% audit share, post-shift detection by the total budget is 0.56% for p1=0.55, 3.07% for p1=0.60, 43.60% for p1=0.70, 96.96% for p1=0.80, and effectively 100% for p1=0.90. At 20% audit share, the corresponding probabilities are 0.82%, 10.01%, 89.73%, 99.85%, and 99.85%. Even auditing 100% of labels detects p1=0.60 with 94.28% probability and p1=0.55 with only 11.84% by this horizon. These are exact for this Bernoulli process and mixture-e test, not evidence of real-world power; gradual shifts, dependencies and alternate event models remain to be tested.

The EESAS paper is a timely comparator for label-shift risk during iterative active sampling. Include it where the task supports its assumptions instead of treating sentinel monitoring as the only possible response. Do not claim reproduction or superiority unless its stated method and evaluation semantics are actually implemented. Source: https://doi.org/10.1016/j.eswa.2026.132567.

Tracking implementation: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744). Full matched baselines and OOD axes: [issue #8749](https://github.com/Luminous-Dynamics/symthaea/issues/8749).


## 21. Compare predeclared anytime-valid prior portfolios

The shift-monitor benchmark now compares a fixed Beta(1/2,1/2) e-process, Beta(1,1), Beta(5,5), Beta(20,20), and an equal-weight portfolio of those four alternatives. All share the same Ville threshold 1/δ, δ=0.05/12, and the same fixed horizon in the reported finite fixture. A convex mixture of e-processes remains an e-process under the same null; portfolio weights must be frozen before observing the stream and must not be selected on the held-out shift.

Independent exact-DP Python calculations at horizon 1,000 under a change after 100 observations give:
- Jeffreys Beta(1/2,1/2): stable-null alarm 0.003059; post-shift detection 11.84% at p1=0.55 and 94.28% at p1=0.60.
- Equal-weight four-prior portfolio: stable-null alarm 0.002778; post-shift detection 17.93% at p1=0.55 and 96.75% at p1=0.60.
- Beta(20,20): stable-null alarm 0.001884; post-shift detection 22.83% at p1=0.55 and 97.92% at p1=0.60.

All stable-null alarm probabilities are below the allocated α/12=0.004167. These are probabilities for the stated synthetic Bernoulli process, not deployment evidence. Because the same Ville boundary is used but realised null alarm probability varies, this is an anytime-validity / power tradeoff comparison, **not** a matched-realised-false-alarm leaderboard. Report actual null alarm rates and pre-shift false alarms separately. A finite-horizon threshold tuned to use the alpha budget can be explored as a separate comparator, but it must be labelled finite-horizon calibrated and cannot inherit the unrestricted Ville guarantee merely because the score is an e-process.

Implementation: the third report `--eprocess-family-csv` in [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744) includes an exact-DP Rust evaluator, a separate standard-library Python reference, deterministic output comparison, label/metric assertions and a mutation negative control. Its current head still needs exact-source CI qualification. The next validation must test weaker and stronger alternatives, p0 misspecification, gradual ramps, bursts, serial dependence and a sealed set of shift families.


## 22. Nominal-rate misspecification must be an explicit negative control

The e-process null guarantee requires the nominal Bernoulli rate p0 to be correct. The executable now includes stable observed streams with actual event rate p ∈ {0.45, 0.48, 0.49, 0.51, 0.52, 0.55}, while every monitor continues to use declared p0=0.50. These are **stationary baseline-misspecification controls**: alarm probability is measured while the stream's actual rate is constant, not because the environment changes during observation.

For the Beta-prior e-process families at horizon 1,000 and log threshold log(240), an independent exact-DP Python prototype found that an actual stationary p=0.48 stream alarms with probability 0.90% (Jeffreys), 1.10% (uniform), 1.55% (Beta(5,5)), 1.73% (Beta(20,20)), and 1.36% (equal portfolio). All exceed the nominal per-cell allocation 0.4167%. At p=0.49, outcomes range from 0.40% to 0.48%, with some families already slightly exceeding the allocation. These are not violations of Ville's inequality; the data-generating rate is outside the declared null. The result demonstrates that an estimated or misspecified reference rate is a separate source of epistemic uncertainty.

Acceptance implications:
- Report performance on the exact null and under each misspecified stationary baseline separately.
- Do not call alarms under p≠p0 false-alarm-guarantee violations; label them assumption-sensitivity results.
- Test a separately collected reference stream, a reference-rate confidence sequence, and a valid composite-null betting process over a preregistered p0 interval.
- Keep baseline-estimation data, policy/tuning data, and held-out evaluation stream disjoint. Calibrating p0 and the threshold on the same stream is not independent validation.
- Preserve the selection-bias controls; a representative reference rate and an adaptively selected observed rate remain different estimands.

The point-null families still expose the limitation above; the composite-band method in Section 23 protects a separately justified interval, but it does not estimate that interval from the same monitored stream. Results are exact for the stated synthetic Bernoulli model only. Implementation and independent checker: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744).


## 23. Composite-null sequential monitoring under reference-rate uncertainty

Point-null e-processes assume a precisely specified baseline rate. The benchmark now adds a composite-null directional betting process for a preregistered Bernoulli interval p_t ∈ [0.48, 0.52]. Each fixed upper-direction factor is 1 + λ(X_t − 0.52); each fixed lower-direction factor is 1 − λ(X_t − 0.48). Under any predictable conditional rate p_t in the interval, each direction's conditional expectation is at most one. The process mixes λ ∈ {0.05, 0.10, 0.20, 0.40, 0.80} with fixed equal weights in each direction and then mixes the two directions equally. Nonnegative supermartingale mixtures preserve anytime-valid threshold control, provided the interval and stakes are fixed before inspecting the monitored stream.

This expands the baseline comparison to six families: the four fixed Beta alternatives, their equal portfolio, and the composite-band process. At horizon 1,000 with a per-cell threshold of 1/(0.05/12), independent exact-DP calculations report:
- At stationary p=0.48, the point-null priors alarm with probabilities 0.90%–1.73%, exceeding the allocated 0.4167%; the composite-band process alarms with probability 0.089%, because p=0.48 is within its declared null.
- At stationary p=0.52, the composite-band false-alarm probability is also 0.089%.
- At stationary p=0.45 or p=0.55, the band-null process alarms with probability 4.53%; those values are outside its declared interval, so this does not contradict its guarantee.
- Following a change from p=0.50 after 100 observations, the band-null process detects p1=0.55 with probability 1.77%, p1=0.60 with 73.53%, and p1=0.70 / 0.80 with approximately 99.99% by horizon 1,000. The original Jeffreys point-null e-process has higher power at p1=0.55 (11.84%) but does not control a stable p=0.48 stream under its point null.

This is an explicit robustness-versus-sensitivity trade-off; it is not a universal winner. The composite process is appropriate only when [0.48, 0.52] is justified from independent calibration knowledge. Its interval must not be selected from the held-out stream. In addition to stationary in-band/out-of-band controls, a new exact-DP scenario changes the conditional rate from p=0.49 to p=0.51 after observation 100; both values remain inside the accepted band. The composite process alarms with probability 0.0263% over the full horizon (0.0164% before the transition and 0.0099% after), below the per-cell budget. This is one predictable within-band stress case; broad serial/adaptive scenarios remain outstanding.

Acceptance requirements: preserve exact-null, selection-bias and point-null misspecification controls; test both band edges and outside-band alternatives; report false alarms and detection separately; validate any interval chosen from a baseline sample using disjoint baseline/tuning/evaluation streams; and include a serial/adversarial conditional-probability stress test. A composite-null alarm does not by itself identify which assumption failed. Implementation and checker: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744). The underlying betting confidence-sequence literature includes Waudby-Smith and Ramdas, “Estimating means of bounded random variables by betting,” https://arxiv.org/abs/2010.09686; this implementation is inspired by the general e-process idea but is not a claimed reproduction of that paper's method.


## 24. Conditional-rate transitions within the declared null

A fixed interval must be tested against more than stationary IID rates. The composite-null harness includes a piecewise predictable stream with p=0.49 for the first 100 observations and p=0.51 thereafter. Both conditional rates remain inside [0.48,0.52], and the process uses the same predeclared directional-betting mixture throughout. Independent exact-DP Python gives full-horizon alarm probability 0.0263%, comprising 0.0164% before the rate transition and 0.0099% after. A second schedule alternates p=0.48 and p=0.52 every observation; its full-horizon alarm probability is 0.02098%. Both are below the per-cell α/12=0.4167% allocation.

The new scenario is a narrow but important regression: a change in the conditional rate is not necessarily a rejection of the declared composite null. The evaluator must count an alarm on either side of the internal transition as an alarm for the composite family, not count post-transition alarms as successful detection of target-environment drift. For point-null families, the same stream is deliberately outside the exact p0=0.50 null before and after the transition; those methods are evaluated as sensitivity controls instead.

Next extend the synthetic null process to predictable sequences varying throughout the interval (including alternating endpoints), abrupt and gradual within-band ramps, and dependence/selection controls. The proof relies on a conditional expectation bound per observation, so these tests validate implementation and numeric stability but cannot replace the conditional-martingale argument. Implementation: [PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744).


## 25. Scenario-typed sequential-monitor metrics

The companion monitor benchmark must not collapse every alarm into “successful drift detection.” Every row carries a preregistered scenario class: `stationary_null`, `target_shift`, `selection_bias_control`, `baseline_misspecification`, or `within_band_transition_control`.

- For `target_shift`, report alarm probability up to the change point, post-change detection probability, and conditional post-change delay separately.
- For stationary, selection-bias, and baseline-misspecification controls, report total horizon alarm probability; post-change detection and delay are not applicable. A selected sample can differ from the target population without any target-environment change. An alarm for a stable baseline outside a detector's declared null is assumption sensitivity, not proof of when a shift occurred.
- For a predictable conditional-rate schedule entirely inside an accepted composite null, report total horizon alarm probability and mark target-shift detection as not applicable. In particular, the 0.49-to-0.51 transition and alternating 0.48/0.52 schedule are null stress controls, not positives for a drift-detection score.

The result export uses `scenario_or_pre_shift_alarm_probability` to expose this class-dependent interpretation, `post_shift_detection_probability` only for true target-shift scenarios, and `mean_post_shift_detection_delay` only when a target shift is defined. These names are deliberately more neutral than “false alarm” on out-of-null controls. Independent reference checks must verify the class, applicability, exact metric values, probability-mass conservation, and mutation rejection. See [monitor implementation PR #8744](https://github.com/Luminous-Dynamics/symthaea/pull/8744).

## 26. Industry baseline qualification: match the false-alarm budget before ranking power

Current synthetic e-process comparisons are not sufficient to claim state of the art. The next qualification lane must compare the composite-null process against faithful, version-pinned implementations of classic online baselines (CUSUM, EWMA, Page–Hinkley, ADWIN) and a distribution-sensitive baseline such as online kernel CUSUM. Where labels are unavailable, include unsupervised detectors whose published protocols match the task. The open-access survey of fully unsupervised detectors evaluated seven implementations on 11 real-world streams and found recommendations depend on the target metric ([Lukats et al., 2025](https://doi.org/10.1007/s41060-024-00620-y)); the 2026 online kernel-CUSUM paper specifically targets sensitivity to small distribution changes and constant computation/memory ([Wei & Xie, 2026](https://doi.org/10.1093/jrsssb/qkag020)).

Required protocol:

1. **Separate guarantee from empirical calibration.** Keep the composite e-process's conditional-null guarantee distinct from empirical ARL0/FPR estimates for heuristic detectors. Calibrate each baseline's threshold only on the calibration partition, then freeze it before evaluation.
2. **Match operating points.** Compare detection power and delay at common empirical false-alarm budgets (plus a range of average run length to false alarm, ARL0) rather than comparing defaults or a lone score. Report uncertainty on estimated ARL0; if there are too few no-change run-years to estimate a stringent FPR, label the estimate unresolved rather than passing it.
3. **Expand the shift matrix.** Include weak/moderate/large abrupt changes, gradual ramps, recurring/oscillating changes, autocorrelated outcomes, overdispersion, bursty outliers, conditional-rate schedules within the accepted band, stable target with selection shift, target shift with selection shift, delayed labels, and multi-cell sparse shifts.
4. **Use the same information and budget.** Give each method the same monitored observations and label availability; charge sentinel audits, feature extraction, kernel computation, and human review to the same budget. Record wall time, peak memory, and per-observation cost.
5. **Avoid benchmark leakage.** Split by dataset/source/time regime, freeze configuration and data digests, use paired runs, and report per-scenario plus aggregate results. Threshold selection cannot see held-out shift magnitudes or source identities.
6. **Rank on a frontier, not one scalar.** Publish (a) formal guarantee class, (b) false alarms per 1,000/10,000 observations and ARL0, (c) detection probability at fixed budgets, (d) median and worst-case delay conditional on detection, (e) calibration uncertainty, and (f) compute/memory cost. Keep downstream regret and target-population performance as separate outcomes.

**Acceptance criteria proposed for the next lane:** zero violations of the composite-null guarantee in the exact finite-state schedule tests; all baselines compared at preregistered matched empirical false-alarm operating points; no scenario-level detection claim where the target did not shift; repeatable paired evaluation on at least three source-distinct streams plus synthetic adversarial controls; and no “beats SOTA” claim until data, code, thresholds, and exact result receipts are independently reproduced. These are proposed gates, not results already achieved.


## 27. Matched-edge CUSUM reference pilot (synthetic, provisional)

The stacked implementation PR [#8822](https://github.com/Luminous-Dynamics/symthaea/pull/8822) adds upper/lower reflected Bernoulli CUSUM arms. Rather than giving CUSUM only a formal alpha allocation and then comparing it to the composite process at a different realized false-alarm rate, its threshold is selected by exact DP to match the existing composite-band betting e-process's *realized alarm probability at p=0.52* over 1,000 observations.

An independent local DP prototype obtained:
- reference composite edge alarm probability: 0.000891588;
- smallest CUSUM integer threshold: 1,374, score 54.96;
- CUSUM alarm probability at the boundary: 0.000885901; the previous threshold 1,373 gives 0.000894015 and is too liberal for this matched target;
- two-sided CUSUM union-bound receipt: 0.001771803, below the per-cell allocation 0.004166667.

At a change after observation 100 and horizon 1,000, the matched-edge prototype gives CUSUM upper-arm power of 7.226% at p1=.55 and 93.336% at p1=.60, compared with 1.769% and 73.527% for the composite-band e-process in the current exact-DP pilot. At p1=.70 and .80 both are near 100% by the horizon. CUSUM mean detection delay conditional on detection remains long (about 774 observations at p1=.55 and 623 at p1=.60), so the power gain is not a free win.

**Claim ceiling:** these values are from independent local exact-DP calculations, not from the stacked Rust PR's artifact or CI. The matched point is a stable constant edge rate, not an empirical matched-FPR evaluation over every predictable in-band schedule. The CUSUM family false-alarm bound is the directional union bound; this is distinct from pointwise realized FPR and from the composite process's formal alpha guarantee. The next qualification step is exact-head Rust/reference parity plus broader predictable schedules and source-held-out real streams. Do not call this state of the art or general superiority.

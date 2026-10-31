# Empirical VoC Qualification Gate v0.1

**Status:** Rust source and test declarations are committed on `research/voc-qualification-gate-v0.1`; compile, exact-head test execution, external independent-verifier agreement, and real-data qualification remain outstanding.

**Tracking:** [#9349](https://github.com/Luminous-Dynamics/symthaea/issues/9349); consumer planner [#9350](https://github.com/Luminous-Dynamics/symthaea/pull/9350).

**Implementation:** `crates/domains/symthaea-architecture-engineering/src/voc_qualification.rs`.

## 1. What this closes

The decision-focused planner computes exact one-step expected reduction in Bayes top-K regret under a supplied finite joint posterior and action-likelihood model. Its SHA-256 bindings and boolean qualification metadata do not establish that either model is empirically calibrated.

This module adds an empirical gate over frozen held-out replay episodes. The gate emits either `Qualified` or `Abstain` with a canonical receipt. The planner must consume a receipt that matches its exact posterior, action-likelihood, source, fidelity-ladder and cost-model digests; the planner must not infer qualification from a boolean.

## 2. Split and provenance contract

The manifest binds:

- benchmark and metric identity/direction;
- exact source-trajectory artifact SHA-256;
- posterior-model and action-likelihood-model SHA-256;
- ordered fidelity-ladder and fidelity-cost-model SHA-256;
- candidate-column order, target K and posterior particle weights;
- model-fit, calibration and evaluation task IDs;
- model-fit, calibration and evaluation architecture-family IDs;
- model-fit, calibration and evaluation seed IDs.

All three split dimensions must be pairwise disjoint. Candidate IDs are strictly sorted to make particle-matrix column identity unambiguous. Fidelity IDs remain in their declared order: sorting the fidelity ladder would change its semantics and is forbidden by the split digest.

The evaluation episodes must belong only to declared evaluation task, architecture-family, seed and fidelity IDs. The manifest also predeclares the expected rank regimes, learning-curve shapes and noise regimes. Every declared evaluation task, architecture family, fidelity rung and declared regime must contribute evidence; an absent regime is a hard error rather than a silent omission. Raw model scores and realized scores are normalized to [0,1] before metric-direction transformation; minimization is converted to maximize-is-better utility inside the scorer.

The split digest binds the complete manifest, including all model/artifact digests and posterior weights. The replay-evidence digest binds the exact held-out episodes, joint particle matrices, outcomes and costs. The receipt binds those digests, threshold digest, measured metrics and qualification decision.

A digest proves byte identity, not that the artifact is truthful. A separate acquisition process must verify dataset origin, task identities, seed provenance, candidate mapping, no test leakage and complete trajectory lineage.

## 3. Metrics and gate

### Joint posterior and candidate ranking

For every held-out episode, the scorer uses each joint posterior particle to derive:

- probability each candidate is best;
- marginal probability each candidate belongs to the true top-K;
- posterior predictive distribution for each candidate's final score.

It records top-1 accuracy, multiclass log loss, multiclass Brier score and top-label ECE. Top-K quality is scored with candidate-membership Brier error, its Brier skill score relative to the constant-prevalence forecast K/N, and empirical coverage of a fixed-threshold credible candidate set. The skill score is `1 - Brier / ((K/N) * (1 - K/N))`; zero means no improvement over the constant base-rate predictor, negative values are worse, and one is perfect. Requiring K < N avoids a degenerate reference score. This prevents a huge search space with a rare top-K positive class from making a near-zero raw Brier score look good by itself. Mean credible-set size is normalized by the candidate count to make the threshold comparable across search spaces.

The credible-set threshold is a pre-registered empirical rule, not a formal frequentist coverage guarantee. A set that includes every candidate can score perfect coverage; the maximum mean set-size fraction therefore prevents vacuous sets from qualifying.

### Posterior calibration and degeneracy

The scorer computes weighted central predictive intervals from posterior particle values, reports empirical coverage and mean width, and emits a weighted mid-rank PIT histogram for every candidate prediction. It also calculates effective sample size:

`ESS = 1 / sum(particle_weight^2)`.

The PIT histogram is diagnostic rather than an automatic uniformity test. Discrete/quantized scores and ties can create structure; tie handling is the deterministic weighted mid-rank. The interval-coverage gate checks closeness to the pre-registered nominal mass and limits average normalized width.

### Action likelihood calibration and decision value

For each replayed fidelity query, dormant reactivation and sentinel audit, the scorer evaluates the forecast distribution over the action's mutually exclusive outcomes using log loss, multiclass Brier score and top-label ECE. Outcome labels are canonicalized by name, and each action class must use one identical category set across its episodes; changing a label set or silently reordering probabilities is rejected. It requires evidence for all three action classes.

Each action record also contains predicted expected regret reduction, realized final-fidelity regret reduction, planned cost and measured cost. The gate measures absolute/signed VOI error and relative cost error. Realized regret reduction is permitted to be negative: an action can have positive predicted value and turn out badly on a particular replay.

This is deliberately stricter than ranking actions by Spearman correlation. High rank correlation can coexist with badly biased value estimates or poor final decisions.

### Conditional calibration

The same candidate-prediction checks are reported separately by:

- architecture family;
- fidelity rung;
- rank regime;
- learning-curve shape;
- noise regime.

Every stratum must meet the pre-registered minimum episode count and global ceilings/floors for top-1 ECE, top-K membership Brier skill against K/N, credible-set coverage and interval coverage. Undersized strata force abstention rather than disappearing from the report.

## 4. Fail-closed behavior

The scorer rejects malformed or mismatched evidence, non-disjoint splits, missing evaluation tasks, unnormalized posterior weights, mismatched candidate/particle dimensions, non-finite/out-of-range scores, missing action classes, malformed action distributions and invalid cost records.

A syntactically valid replay that misses a quality/calibration threshold receives an `Abstain` receipt with sorted failure reasons. The receipt does not turn a poor model into a passing model.

`verify_voc_qualification_receipt` recomputes the gate from the exact manifest, episode evidence and thresholds, then rejects any altered metric, decision, failure reason or digest. This is deterministic replay within the Rust implementation. A separate implementation (for example a small independently maintained verifier) must still agree on all receipt fields before the qualification pipeline is treated as independently verified.

## 5. Initial thresholds are policy, not evidence

The Rust defaults are conservative starting values to force a visible qualification decision. They are not universal statistical constants and should not be relaxed after seeing evaluation outcomes. Freeze a benchmark-specific threshold document before the evaluation split is opened.

The minimum task/episode/action counts should be increased for production. The default gate requires three fit tasks, three calibration tasks, five evaluation tasks, 50 replay episodes, 10 examples per calibration stratum and 20 action forecasts per action kind. These minimums are not a substitute for a power analysis; rare strata need more evidence.

## 6. Negative controls required before qualification

A benchmark harness should prove that the gate abstains under at least these controlled failures:

1. Reuse one task, architecture family or seed across fit/calibration/evaluation.
2. Swap candidate columns while keeping the same candidate list.
3. Reorder or rename fidelity rungs without changing the digest.
4. Replace the posterior with overconfident particles that miss late-learning candidates.
5. Use stale task priors or likelihoods from another benchmark family.
6. Report action outcomes with a misspecified or non-normalized distribution.
7. Make the sentinel-audit likelihood optimistic while the run-wide audit ledger stops pruning.
8. Understate measured action cost or alter the cost model after qualification.
9. Omit late learners, high-noise cases, failed runs or dormant-candidate reactivations.
10. Edit a replay episode, threshold or receipt field after scoring.

The current eighteen Rust tests cover basic pass/replay, receipt tampering, split leakage, invalid categorical probabilities, likelihood misspecification, ESS degeneration, minimize-is-better semantics, order-invariant replay evidence, consistent action-category semantics, reordered equivalent outcomes, fidelity-order digest binding, rejection of undeclared evaluation strata, Brier skill behavior against the K/N baseline, and family/seed leakage, missing-family coverage, and degenerate K=N controls. They are test declarations only until compiled and run.

## 7. Benchmark campaign and competitive baselines

Use SHA-256-pinned full learning-curve trajectories, not synthetic-only results. Start with NAS-Bench-201 CIFAR-10, CIFAR-100 and ImageNet16-120. Add NAS-Bench-101 and at least one separate benchmark family for transfer evaluation. Preserve the complete candidate/epoch/seed mapping and run the same sealed evaluator across all baselines.

Compare at equal measured total resource cost and separately at equal number of full-fidelity confirmations:

1. Random search and random early stopping.
2. Raw Successive Halving / Hyperband.
3. Calibrated lower-confidence-bound promotion.
4. Calibrated promotion with static optimistic guard slots.
5. Interval-width-per-cost VoC.
6. Decision-focused VoC without a qualification gate (diagnostic only; never production eligible).
7. Decision-focused VoC with this qualification gate and the run-wide pruning ledger.
8. IISE and Optimal Multi-Fidelity Best-Arm Identification for K=1 when their known-bias assumptions are satisfied.
9. DBCARE-inspired cost-risk stopping; do not claim DBCARE's theorem transfers to heterogeneous multi-fidelity NAS.
10. ParZC, AZ-NAS, ZAPS, CoRA-NAS, HiP-BO and MF-NAS where the original code, data and protocol can be reproduced.

Pinned literature/benchmark references:

- [NASLib NAS-Bench-201 full learning curves](https://automl.github.io/NASLib/search_spaces/nasbench_201/).
- [ParZC, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33793).
- [Optimal Multi-Fidelity Best-Arm Identification, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html).
- [Balancing Performance and Costs in Best-Arm Identification / DBCARE, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/b0dfe8236981ba84b4d98a7ed0426f0b-Abstract-Conference.html).
- [HiP-BO, online 16 October 2026](https://www.sciencedirect.com/science/article/pii/S092523122602802X).
- [MF-NAS, DOI 10.1145/3774940](https://doi.org/10.1145/3774940).

For proxy-ranking baselines, the current NAS-Bench-201 CIFAR-10 SOTA2 leaderboard reports ParZC at 90.4 Spearman / 70.6 Kendall and AZ-NAS at 89.2 / 71.2. This makes metric choice consequential: the leading method changes depending on Spearman versus Kendall. These scores are contextual references only; our primary comparison must reproduce the exact split, task, architecture population, preprocessing and correlation definition.

### Reported result anchors (not directly interchangeable)

| Method | Published/maintainer-reported reference | Correct interpretation for our comparison |
|---|---|---|
| ParZC / AZ-NAS | NAS-Bench-201 CIFAR-10: ParZC 90.4 Spearman / 70.6 Kendall; AZ-NAS 89.2 / 71.2 | Proxy rank correlation over a fixed benchmark population. It is not top-100 retrieval under a search budget. The leaderboard is a community-maintained secondary source; reproduce the primary implementation and protocol. |
| CoRA-Rank / CoRA-Refine | CoRA reports Spearman 0.927 for its zero-cost ranker and 0.946 after early-curve residual refinement, with the refinement costing approximately 1% of full-training-all-candidates cost; reported mean correlations across NB201, NB101, TransNAS and NATS-SSS are 0.946, 0.715, 0.786 and 0.894 | Ranking-correlation and selected-architecture quality track. CoRA's anchor sample is trained briefly (about 6.4% of the space); count all anchor training. Do not compare its correlation score directly to ZAPS P@100. |
| ZAPS | At B=200 evaluated architectures on NB201, P@100 = 52.3% CIFAR-10 / 65.8% CIFAR-100 | Search-retrieval track: how much of the true top-100 lies among architectures actually evaluated. Also report simple regret of the best evaluated candidate. Do not conflate this with a model's predicted top-100 precision. |
| R-MF-NAS | Author repository reports mean ± SD selected test accuracy: NB101-CIFAR10 92.95 ± 0.84; NB201-CIFAR10 93.70 ± 0.35; NB201-CIFAR100 71.46 ± 1.05; NB201-ImageNet16 45.06 ± 0.89 | Multi-fidelity final-architecture quality and cross-seed robustness. The public NB201 config samples up to 2,000 SynFlow-scored candidates, starts from 32 candidates and uses training rungs 12/25/50/100/200 epochs; compare under the same total metered budget rather than copying rung labels alone. |

ZAPS has a protocol nuance that must be replicated transparently: its paper describes offline ProxyFit selection using target-space proxy/accuracy correlations, then reports that transferred proxy subsets preserve performance within 5.2 percentage points at budgets of at least 200. Our benchmark must report both target-calibrated reproduction and a genuinely task-disjoint transferred-proxy condition; only the latter is eligible for a no-target-label leakage claim. This is an evaluation protocol distinction, not an accusation about the authors' reported experiments.

CoRA is an especially strong comparator because it combines proxy complementarity with early learning dynamics rather than using one signal family. Our opportunity is not to repeat that combination; it is to add a calibrated joint posterior, adaptive next-measurement choice by expected top-K regret reduction, dormant recovery, and run-wide audited pruning while matching the same cost frontier.


## 8. Primary outcome metrics

Report all of the following, not only correlation:

- final-confirmed P@1, P@10 and P@100;
- normalized final top-K regret and CVaR-10%;
- total measured cost, cost per final-confirmed top-K candidate and cost-to-regret curve;
- probability of premature stopping and under/over-spend;
- top-1/top-K calibration, action likelihood calibration and posterior interval coverage;
- predicted-versus-realized regret reduction error;
- dormant reactivation recall and false-reactivation cost;
- false-pruned fraction and run-wide audit-bound coverage;
- worst-task, worst-seed and held-out-family regression;
- quality/resource Pareto frontiers.

Use at least 30 independent search seeds per benchmark/task configuration for initial engineering estimates and report confidence intervals across seeds. Increase the number when power analysis shows the desired effect cannot be distinguished. Do not treat individual architectures from one search run as independent replicates.

### Pre-registered engineering targets (aspirations, not results)

- **ZAPS search-retrieval track:** at B=200, target P@100 of at least 60% on NB201-CIFAR10 and 73% on NB201-CIFAR100, compared against the reported 52.3% and 65.8% ZAPS references under the same eligible target-label protocol. Require a paired-seed 95% confidence interval for the P@100 difference above zero; if the target-calibrated proxy track is used, the result is not a transfer claim.
- **CoRA ranking track:** retain Spearman within 0.01 of CoRA on NB201 while improving the worst of NB101/TransNAS/NATS by at least 0.05 at no more than 1.25% of all-candidate full-training cost. Report Kendall as well; one correlation alone can hide top-of-list failures.
- **R-MF-NAS final-quality track:** at equal total metered cost, match or exceed its reported mean final quality on each comparable task and improve the worst-decile seed result by at least one percentage point where the task metric allows that interpretation. Do not use the paper's reported accuracy table as a pass/fail gate until its exact hardware, dataset version, test protocol and 500-run script are reproduced.
- **Decision/calibration track:** top-1 ECE ≤0.05 on sufficiently populated held-out strata; 90% interval coverage within ±5 percentage points; no declared stratum below its preregistered top-K coverage floor; action-outcome log loss/Brier/ECE within frozen limits; VOI MAE ≤0.05 regret units. Any underpowered stratum remains NOT_ESTIMABLE and forces abstention for that deployment slice.
- **Risk/anytime track:** the empirical run-wide false-prune undercoverage must remain within the declared run-level delta under preregistered stress tests; publish quality versus elapsed cost at every rung and include catastrophic late-learner recoveries as explicit failure cases.

These thresholds are intentionally hard targets and must be frozen before final test labels are opened. They are not universal theoretical guarantees.

## 9. Claim ceiling

This module can issue an empirical qualification decision for one exact frozen model, benchmark, metric, task distribution, evidence set and threshold set. It does not prove calibration on unobserved tasks, guarantee that the global optimum survives pruning, certify expected VOI as a frequentist bound, or demonstrate better-than-SOTA NAS performance.

No SOTA or calibrated-VoC claim is warranted until:
1. the exact branch head compiles, formats and passes Rust tests;
2. a separate verifier agrees with receipt contents;
3. NAS learning-curve artifacts are acquired and hash-pinned;
4. calibration and evaluation use genuinely disjoint task/family/seed partitions;
5. negative controls cause the prescribed abstention;
6. head-to-head replays beat the strongest eligible baseline under matched resource accounting.

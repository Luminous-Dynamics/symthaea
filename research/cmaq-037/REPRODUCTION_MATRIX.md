# CMAQ-037 — PRM / CCTM faithful reproduction and frontier study

## Research question

Does the ONS-adaptive predictive-rank martingale provide a measurable detection-delay advantage over conditional conformal test martingales (CCTM) at the same anytime false-alarm budget, calibration inventory, score process, and horizon? How does that trade-off change with calibration sample size, alpha, serial dependence, and shift morphology?

CMAQ-035 established an internal comparison between fixed and ONS betting. CMAQ-037 adds the external CCTM baseline and reproduces the 2026 PRM paper before extending the experiment.

## Method arms

| Arm | Description | Role |
|---|---|---|
| CCTM | Faithful implementation of the paper's conditional conformal test martingale | Published baseline |
| PRM-ONS | ONS-adaptive order and dispersion predictive-rank martingale | Published adaptive PRM |
| PRM-fixed-4 | Fixed two-sided four-component order / dispersion portfolio | Fixed-bet baseline |
| PRM-positive-only | Fixed positive Order and Dispersion bets | Directional ablation |
| PRM-spectrum | CMAQ-038 fixed-bet multi-resolution Haar contrasts over the complete finite rank grid | CMAQ exploratory extension; not part of the published PRM reproduction |
| PF-CCTM | Liu, Prinster & Orabona (2026), restarting universal portfolio for fixed-reference CCTM | New published late-shift baseline; must reproduce from the paper/reference implementation |
| PRM-restart-spectrum | CMAQ-040: Haar rank-spectrum features × heavy-tailed restart-time mixture × fixed betting-rate grid | CMAQ exploratory compositional extension; not the PF-CCTM algorithm |
| Operational outcome scorecard | CMAQ-041: warning/quarantine/fallback outcomes vs. independent recoverability deadline, grouped by common-cause block | Application scorecard; not a detector or safety certification |
| Reference-null finite-T gate | CMAQ-042: independent reference-path order-statistic boundary for frozen horizon | Optional finite-horizon gate only; retain Ville 1/alpha for unbounded monitoring; simple/pivotal null required |
| Fleet alpha allocation | CMAQ-043: hierarchical alpha_global/[i(i+1)j(j+1)k(k+1)] across asset/channel/deployment | Open-ended union-bound budget; does not assume stream independence; requires durable one-use ledger |
| Paired effective-block inference | CMAQ-044: H+1 miss penalty, matched baseline/candidate deltas, block bootstrap, family breakdown and false events/hour | Statistical comparison layer; bootstrap conditional on representative/exchangeable effective blocks |
| Conditional rank dependence | CMAQ-045: predictable history contexts × full dyadic rank spectrum | Targets temporal dependence with marginal rank invariance; finite context portfolio only |
| Convex-target approachability | CMAQ-046: predictable OCO separator × support-function residual × conditional-mean e-process | Adaptive conditional-mean null; distinct from exchangeable PRM/CCTM; fixed-bet subset |
| Controlled Blackwell action selection | CMAQ-047: OCO witness chooses from action-specific null-safe diagnostics using support-halfspace eligibility | Adaptive controlled experiment arm; requires authenticated action-conditional mean contracts and one-use outcome reconciliation |
| Fourier kernel mean embedding | CMAQ-048: fixed paired sine/cosine random Fourier features × CMAQ-046 approachability e-process | Detects selected higher-moment/shape changes with preserved raw mean; finite feature-bank approximation and frozen null embedding target |
| Contextual paired kernel discrepancy | CMAQ-049: one-hot predictable context blocks × paired finite Fourier response differences | Conditional paired-distribution equality within pre-response strata; blocks marginal cancellation, but finite context/feature approximation |
| Continuous-covariate paired kernel discrepancy | CMAQ-050: continuous covariate Fourier map × paired response Fourier difference | Smooth product-feature conditional comparison without hard bins; finite product feature map and dimension cap |
| CS-risk | Anytime confidence-sequence monitor of a matched running/rolling average risk | Complementary estimand, not a like-for-like per-instance drift detector |

Do not implement CCTM by multiplying repeated marginal conformal p-values or by using a rolling empirical breach rate. Implement its actual conditional calibration and martingale construction from the reference paper/code.

## Frozen experiment matrix

| Factor | Values |
|---|---|
| Calibration block count | 19, 49, 99, 199, 499, 999 |
| Alpha | 0.01, 0.05, 0.10 |
| Horizon | 100, 500, 2,000 |
| Null replications | At least 10,000 per registered cell |
| Shift replications | At least 5,000 per registered cell, preferably 10,000 |
| Score distributions | Continuous and discrete/tied |
| Dependence | IID independent blocks; explicitly dependent negative-control streams |
| Features | Order, dispersion, upper/lower tail, center concentration, higher-order shape |

N=19 is only the mathematical finite-sample floor for a non-randomized 95% split-conformal quantile; it is not a claim that 19 clusters gives adequate power. Report effective closure-block count from CMAQ-036, never raw episode count.

### Shift families

- Positive and negative location shifts.
- Dispersion expansion and contraction.
- Upper- and lower-tail contamination.
- Center concentration and symmetric heavy tails.
- Multimodal and trimodal shape changes.
- Gradual, delayed and transient shifts.
- Autocorrelated / common-shock score streams.
- Orthogonal-shape alternatives designed to evade pre-registered rank features.

Any serially dependent cell must be labelled assumption-violating for PRM unless a valid dependence-aware reference argument is established. Do not report an IID exchangeability guarantee on it.

## Metrics

**Null:** probability of ever crossing the anytime threshold; exact or Wilson 95% binomial confidence intervals; average run length. No early alarm is a “win” under the null.

**Alternatives:** detection power by horizon; median / p90 / p95 detection delay conditional on detection; fraction undetected by horizon; paired detection outcomes on shared random streams; worst-family result.

**Systems:** throughput, p50/p95/p99 per-observation latency, allocations, peak memory, and full end-to-end ingestion-to-quarantine cost.

**Coverage / utility:** calibration-size versus detection-power frontier; false-alarm versus detection-delay frontier; per-family results, never just pooled averages.

## Reproduction protocol

1. Pin the paper version, reference implementation commit, compiler, OS, and exact dependencies.
2. Recover published simulator equations, parameter settings, seeds and CCTM setup; record any missing information explicitly.
3. Implement CCTM faithfully and write fixed example vectors against the paper/reference code.
4. Reproduce the paper's headline curves before adding CMAQ-specific shifts.
5. Reproduce the PF-CCTM restarting universal portfolio separately from the published equations, including its three-asset wealth process, Dirichlet(1/2,1/2,1/2) universal portfolio, heavy-tailed start weights, and finite-reference confidence band. Do not label a simpler restart mixture as PF-CCTM.
6. Use a shared immutable calibration reference and common-random-number deployment streams for all PRM/CCTM arms.
7. Randomize ties using the same valid mechanism for calibration and deployment; simulation RNG remains clearly non-cryptographic.
8. Pre-register all cell combinations, replication counts, quantiles and exclusions.
9. Store per-replication alarm times / no-alarm indicators plus seed, configuration and source hashes. Aggregate CSVs alone are insufficient.
10. Publish code, artifact hashes, confidence intervals and the full run command.

## CCTM reproduction details that must be frozen

The official CCTM implementation maps each test score through an empirical CDF built from a fixed calibration set, applies a Dvoretzky–Kiefer–Wolfowitz confidence band, and updates its betting parameter with ONS. Its code exposes both a non-randomized ECDF using the count of calibration scores less than or equal to the test score divided by n, and a randomized ECDF for ties using a uniform tie-break and denominator n+1. These are not interchangeable implementation details. Pin the reference commit and reproduce the paper's exact tie policy, DKW parameter, ONS initialization/bounds, smoothing and threshold before comparing results.

For the documented DKW implementation, the band width is sqrt(log(2/ci_delta)/(2n)). Record both the finite-reference correction and the raw ECDF/tie mode in every run manifest. Do not silently use a deterministic <= ECDF for discrete/tied scores if the target protocol calls for randomized conformal ranks.

## External benchmark translation: from rank streams to control decisions

CMAQ is an assurance layer, not a trajectory predictor. Industry comparison therefore needs two linked evaluations:

1. **Statistical monitor evaluation:** score-to-rank calibration, null false alarms, distribution-shift detection, delay and dependence sensitivity under frozen evidence manifests.
2. **Decision/system evaluation:** feed a planner's frozen safety/recoverability score stream into the monitor, then measure whether quarantine and fallback happen before the safety margin is exhausted and whether useful operation is preserved.

Use existing public environments without mislabelling their scope:

- **Waymo Open Dataset:** its official site reports active leaderboards in 2026 despite no formal 2026 challenge, and the Motion Dataset contains 103,354 twenty-second segments at 10 Hz. Use motion, interaction-prediction and sim-agent scenarios to derive held-out environmental families and stress scores. The dataset is licensed for non-commercial research and explicitly should not be used to evaluate the real-life performance of a vehicle; it is not physical deployment evidence.
- **nuPlan:** use its open- and closed-loop simulation framework for long-horizon ego-planning evaluation. Its metric families include traffic-rule violations, collisions/off-road behavior, time-to-collision, vehicle-dynamics feasibility/comfort, goal achievement and scenario-specific interactions. Treat it as a research simulator, not a current live leaderboard claim.
- **CARLA:** the official Autonomous Driving Leaderboard is permanently closed and its evaluation servers are shut down. A locally reproducible CARLA suite can still supply scenario stress tests, but there is no live CARLA leaderboard submission to target.
- **Safety-process traceability:** map each scenario to explicit hazards, unsafe control actions, constraints and evidence required to justify fallback. SAE J3307_202503 documents STPA requirements across industries; use it as a process/traceability reference, not as a claim of certification.

For every simulator/data-source experiment, freeze route/scenario, map, weather, interaction class, vehicle/planner version, source trace, calibration split, effective CMAQ-036 dependency block, score definition, fallback policy and software digest. Split by scenario family and common-cause group—not random overlapping clips from the same recording. Record planner outcomes independently from the monitor's own decisions.

The external scorecard should report:
- collision / contact and traffic-rule violation rates, off-road rate, time-to-collision and minimum signed recoverability margin;
- intervention lead time: first monitor warning to margin exhaustion;
- fallback success and fallback completion before the margin is exhausted;
- spurious quarantine per operating hour or kilometre;
- safe useful progress / completion under the same fallback budget;
- p50/p95/p99 assurance latency, CPU/memory overhead and evidence-ingestion delay;
- worst environment-family results and confidence intervals, not only pooled averages.

The monitor must never be rewarded for avoiding violations by permanently refusing to operate. Report the safety–availability frontier: hazard exposure, unnecessary intervention, safe completion and useful progress together.

## CMAQ-038 exploratory feature-coverage arm

CMAQ-038 adds a pre-registered dyadic Haar rank-spectrum mixture with 2n directional components for n+1 rank buckets. It is evaluated against the actual CMAQ-031 fixed-four monitor on paired rank streams. Its deliberately trimodal rank-grid alternative is a diagnostic probe, not the raw-score generator used in CMAQ-035 and not a published-paper reproduction. Keep CMAQ-038 results in a separate extension table until the original PRM / CCTM reproduction is complete.

The additional feature bank can increase sensitivity to shape changes but distributes the e-process mixture mass over more components. It must therefore be assessed on the full null-alarm / detection-power / delay / runtime frontier; higher feature count is not itself evidence of improvement.

## Late-shift baseline added after the original matrix freeze

The 12 October 2026 paper by Liu, Prinster & Orabona, *Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales* (arXiv:2610.17343), is a direct and material comparator. It reframes the CCTM betting problem as a three-asset portfolio, then mixes restarted universal portfolios over every possible start time using

\[\pi_s=\ln 2\left(\frac{1}{\ln(s+1)}-\frac{1}{\ln(s+2)}\right),\qquad r_t=\frac{\ln 2}{\ln(t+2)}.\]

The paper proves anytime false-alarm control for its CCTM construction and gives a logarithmic-in-shift-time restart cost; it also reports substantially flatter delay across late shifts than the ONS CCTM baseline. Reproduce the paper's own method and assumptions before comparing it to PRM. In particular, its fixed-reference DKW correction and calibration-conditional validity statement are not interchangeable with the PRM's exact predictive-rank martingale and marginal anytime guarantee.

CMAQ-040 is a separate composition experiment: it applies a heavy-tailed mixture over possible restart times to fixed-bet predictive-rank Haar components, with a finite pre-registered betting-rate grid. Each feature/rate/direction account is itself a predictive-rank e-process; the restart mixture is a normalized mixture of those processes. This is not a reproduction of the paper's Dirichlet universal portfolio, does not inherit its optimality theorem, and should not be called parameter-free. Its purpose is to test whether feature-space coverage and late-onset adaptation compose effectively under the exact PRM null.

The late-shift matrix must include shift times 0, 125, 250, 500, 1,000, 2,000, 4,000 and 8,000, matching the PF-CCTM ImageNet-C protocol, and report delay from the true shift time—not from stream start. Report null wealth and false alarms over the *entire* deployment horizon, not just post-change windows. Use a long null prelude to detect performance degradation that short-horizon synthetic tests cannot expose.

## PF-CCTM reproduction target and caveats

The current PF-CCTM paper reports, for n=1,000 on ImageNet-C at t0=8,000 clean observations, median delay 50.5 rounds for its restarting universal portfolio versus 244.5 rounds for CCTM; detection power and misses are reported separately. On the synthetic sharp-shift setting with t0=4,000 and unit mean shift, the reported medians are 54 vs 216 rounds. These are paper-reported results, not independently replicated CMAQ results.

The paper's ImageNet-C grid is n in {100, 250, 500, 1,000, 4,000}, 10 seeds, 15 severity-5 corruptions, and t0 in {0, 125, 250, 500, 1,000, 2,000, 4,000, 8,000}; every change-point run observes 2,000 rounds after the shift. Its null check uses the remaining clean holdout and 100 seeds for n in {100, 500, 4,000}. Match the original data split, predictor outputs, calibration prefix, clean prelude, shift stream, and no-false-alarm conditioning before comparing. Count missed detections as infinity for medians and as horizon+1 in capped means; never drop misses from the denominator.

Important qualification: the PF-CCTM paper itself reports one false alarm among 300 clean null streams for its restarting portfolio at n=4,000; its CCTM baseline has zero in that table and standard CTM has one to three percent. Its other experiment tables and seeds should be reproduced as published rather than reducing the result to the headline late-shift ratio. The restart portfolio's false-alarm statistic, wealth statistic and power-after-no-false-alarm are distinct quantities.

The following are **paper replication targets**, not guaranteed CMAQ thresholds:
- t0=8,000, n=1,000, ImageNet-C: reproduce the reported PF-CCTM median around 50.5 and CCTM around 244.5 under the identical clean/shift partition and warm-up accounting.
- t0=4,000, n=1,000, synthetic sharp shift d=1: reproduce the reported PF-CCTM median around 54 and CCTM around 216.
- Report false-alarm rate, conditional power, unconditional miss fraction, median and capped mean delay together for each arm and cell.

Only after this faithful baseline reproduction should CMAQ-040's separate PRM restart mixture be compared. Theoretical validity differs: PF-CCTM's portfolio martingale and PRM's exact conditional predictive-rank martingale have different score/p-value assumptions and their guarantees must not be transferred between them.

## Current internal CMAQ-035 result is not SOTA evidence

At n=99, alpha=0.05 and horizon 500, an internal 10,000-replication experiment found:
- Null alarm rates of approximately 3.4–3.7% across ONS and fixed-bet variants.
- ONS median delays were substantially lower on location shifts and dispersion contraction.
- Dispersion expansion showed a faster ONS median but a slightly worse p90/p95 tail.
- A higher-order trimodal alternative exposed low detection power for all current feature portfolios.

This is enough to motivate broader replication, not enough to claim reproduction of the PRM paper or superiority over CCTM.

## Prior art

- Kuang & Xia (2026), *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales* — predictive rank law, order/dispersion features, ONS betting and CCTM comparisons.
- Liu, Prinster & Orabona (2026), *Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales* — three-asset portfolio formulation, restarting universal portfolio, heavy-tailed start-time allocation and logarithmic shift-time regret.
- Fang (2026), *Anytime Valid Conformal Prediction on Streaming Time Series* — confidence sequences for running / rolling average miscoverage; complementary target.
- WATCH (ICML 2025) — weighted conformal test martingales for covariate / concept shift diagnosis.
- Timans et al. (UAI 2025) — continuous monitoring of bounded risk violations under unknown shift with testing-by-betting.

Additional late-2026 comparator coverage:
- Zheng & Zhang (17 Nov 2026), *Sequential Conditional Paired-Sample Testing with Applications to Online Monitoring* (SCPT): anytime-valid conditional kernel discrepancy tests for paired responses sharing covariates; compare only on paired/covariate-matched streams and reproduce its dependence assumptions.
- Saha & Ramdas (30 Jul 2026), *Non-partitioned e-detectors for nonparametric sequential change detection*: aggregate point-null e-processes over candidate changepoints and take an infimum over candidate no-change distributions; it targets unknown pre/post distributions and is a comparator for reference/model uncertainty.
- Kalinke & Gavioli-Akilagun (NeurIPS 2025), *Optimal Online Change Detection via Random Fourier Features*: nonparametric online multivariate change detection with logarithmic time/space complexity and minimax delay guarantees; include as a direct RFF comparator.
- Njuki & Ning (21 Sep 2026), *A Sequential Nonparametric Test for Detecting Change in the Distribution Based on Energy Statistics*: windowed energy-statistic sequential detector; report its actual false-alarm criterion and avoid assuming anytime validity if the reproduced method does not provide it.
- Shaer et al. (ICML 2026), *Testing for Distribution Shifts with Conditional Conformal Test Martingales*: fixed-reference conditional validity and protection against post-change reference contamination.

These comparators have different nulls, data access, paired/covariate structure, and stopping criteria. Use separate matched benchmark families rather than forcing every method into one pooled leaderboard.

CMAQ-037 is a fair reproduction and benchmarking programme, not a new statistical theorem.

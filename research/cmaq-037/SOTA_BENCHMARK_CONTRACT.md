# CMAQ sequential-monitor benchmark contract

**Status:** proposed preregistration contract. All targets below are prospective; none are current achievement or SOTA claim.

## 1. Comparison arms

Keep these methods separate; do not merge their guarantees or silently borrow their best results.

| Arm | Role | Required status |
|---|---|---|
| Standard CTM | Baseline that updates its reference with test data | Faithful implementation |
| Conditional CTM (CCTM) | Fixed-reference, DKW-corrected conditional betting | Faithful paper reproduction |
| PRM | Exact predictive-rank martingale under its exchangeability construction | Faithful paper reproduction |
| PF-CCTM | Parameter-free restart/universal-portfolio CCTM | Faithful paper reproduction |
| CMAQ-031 | Fixed four-statistic / baseline PRM arm | Frozen internal baseline |
| CMAQ-033 | ONS/adaptive PRM arm | Frozen internal baseline |
| CMAQ-038 | Full dyadic rank-histogram spectrum | Exploratory CMAQ arm |
| CMAQ-040 | Rank spectrum × restart-time mixture | Exploratory CMAQ arm |
| CMAQ-045 | Rank spectrum × predictable history contexts | Exploratory CMAQ arm |
| CMAQ-046 | Predictable OCO witness × convex-target support residual × conditional-mean e-process | Exploratory conditional-mean arm; fixed-bet subset, distinct null contract |
| CMAQ-047 | Controlled Blackwell action selection with action-conditional null mean sets | Exploratory active diagnostic arm; selected action must satisfy current support-halfspace contract |

For every method, freeze source revision, dependencies, score mapping, reference set, ties, warm-up, alpha, horizon and stopping rule. The CCTM/PF-CCTM arms operate on their own stated score/p-value contracts; a PRM result cannot be presented as a reproduction of either.

## 2. Faithful PF-CCTM reproduction grid

The PF-CCTM paper's ImageNet-C protocol is the first external late-shift comparator:
- calibration/reference sizes n ∈ {100, 250, 500, 1,000, 4,000};
- 15 severity-5 ImageNet-C corruptions;
- 10 seeds;
- clean pre-change starts t0 ∈ {0, 125, 250, 500, 1,000, 2,000, 4,000, 8,000};
- 2,000 post-change observations per run;
- reproduce its predictor outputs, holdout partition, calibration construction, warm-up accounting and threshold definitions.

The paper reports a median detection delay of 50.5 rounds for the restarting portfolio against 244.5 for CCTM at n=1,000 and t0=8,000. Treat this as a replication target, not a CMAQ result. Reproduce the whole table, including misses and null behavior, before attempting a head-to-head claim.

The published PRM experiment is a separate arm. Reproduce its reference-rank procedure, exact randomized tie policy and null construction rather than translating continuous scores into an arbitrary rank approximation.

## 3. Primary synthetic design

Run a factorial grid over:

| Dimension | Values |
|---|---|
| Effective calibration blocks n_eff | 19, 49, 99, 199, 499, 999, 1,999, 3,999 |
| Local alpha | 0.01, 0.05, 0.10 |
| Pre-change delay t0 | 0, 125, 500, 2,000, 8,000 |
| Post-change horizon H | 500, 2,000, 10,000 |
| Seeds / independent streams | At least 10,000 null paths per primary cell; at least 5,000 paths per alternative cell |

A smaller smoke run may test wiring but cannot estimate a five-percent false-alarm budget tightly. With 10,000 independent null paths, the approximate 95% binomial half-width near 5% is 0.43 percentage points; with 25,000 it is about 0.27 points. Report exact Wilson or Clopper–Pearson intervals and the denominator.

### Nulls

1. Exact Pólya-urn predictive-rank null for each frozen n_eff.
2. Score-exchangeable nulls with atoms, using the declared randomized tie policy.
3. Clean stationary streams over the *entire* deployment horizon.
4. Independent replay seeds with common-cause groupings retained; resampling episodes from one group is not additional independent evidence.

### Alternatives

**Marginal rank-shape shifts:** location, scale, tail weight, central concentration, skew, bimodality, trimodality, sparse extreme mass, abrupt, gradual and transient shifts.

**Temporal dependence shifts:** stationary-uniform sticky Markov ranks, balanced alternation, burst/run persistence, conditional upper/lower tails, dependence changes with marginal-preserving transition matrices, and regime switching.

**Recoverability shifts:** increasing residual variance, decreasing barrier/recoverability margin, delayed actuation, fallback degradation, and simultaneous sensor/control disturbances. Ground-truth deadlines must come from independent plant/scenario evaluation, not from the detector.

For every generator, store its source, parameters, seed, split, effective block ID, rank/score receipt and exact build digest.

### Controlled action-selection experiments (CMAQ-047)

Treat this as a separate controlled-experiment family, not a passive detector baseline. At every round the monitor emits a predictable witness w_t, then an action is selected from a frozen catalog. For every admissible action a, the independently authenticated action-conditional null set S_a must satisfy h_(S_a)(w_t) <= h_S(w_t). Only then may the resulting residual vector be fed into the betting process.

Preregister:
- action-specific null-mean boxes and the authority receipts that justify them;
- alternative diagnostic signatures and the deterministic action ranking policy;
- target, scope, monitor build, observation bound, alpha, action inventory and execution semantics;
- one-use action sequence IDs and the result-reconciliation policy;
- baseline fixed action, random admissible action and oracle-only upper-bound action.

Measure the fraction of rounds with at least one admissible action, chosen-action alignment with the true alternative orientation, information/diagnostic utility per action, false alarms over the entire adaptive action sequence, detection power/delay/misses, action frequency and budget consumption, p99 selection latency, execution latency, unresolved-result frequency and recoverability/fallback consequences.

Null simulations must condition the next residual distribution on the action actually selected and keep its conditional mean in S_a. Sampling residuals from the global target S alone is not enough to validate an action-conditional policy. Alternatives may violate the action-specific null box, but the same action policy and observation model must be used for all arms.

Unknown or ambiguous action outcomes remain pending. Do not count a retry as an independent observation or select a second action until the first result is reconciled. Cross-process replay protection, authority signatures and physical action attestation require an external durable evidence ledger.

Compare adaptive controlled selection against a fixed safe action, a precommitted schedule, random choice among current admissible actions, and an oracle action only as a non-deployable upper bound. Pair these runs by effective block and use CMAQ-044. Any power improvement must be assessed alongside action availability, false alarms, execution delays, fallback timing and safety outcomes.

### Conditional-mean vector residual experiments (CMAQ-046)

These are a separate experimental family, not extra repetitions of the PRM rank-null grid.

Freeze a residual vector r_t, compact convex target S, dimension d, observation norm bound R, score/sensor build and target digest before each run. Under the null, test the adaptive conditional-mean restriction E[r_t | F_(t-1)] ∈ S. The benchmark alternatives should include:
- unknown-direction mean shifts toward all registered faces/corners of the box;
- one-coordinate and coordinated multi-coordinate violations;
- gradual and transient mean departures;
- cross-coordinate cancellation, where a fixed scalar projection can remain near zero;
- marginally unchanged noise variance/tails as negative controls, demonstrating that a conditional-mean detector need not detect scale-only changes;
- adaptive but valid conditional means that move within S over time;
- heteroskedastic and serially dependent noise whose conditional means remain in S.

Compare adaptive OCO witness, fixed compass-direction e-mixture, coordinate-wise scalar e-processes, and matched multivariate sequential baselines. All arms must use the same bounded residual vectors, target, norm bound, alpha and stopping horizon. The conditional-mean null must be generated and tested directly; PRM/CCTM guarantees cannot be transferred to this arm.

Report conditional-mean null full-horizon false-alarm probability, miss fraction, true-onset delay with H+1 miss penalty, worst orientation, pathwise distance/residual/regret identity error, p99 update latency, memory and dimension scaling. Use CMAQ-044 for paired effective-block inference. The empirical distance to S is a diagnostic, not a valid stopping rule; rejection uses the e-process threshold.

## 4. Primary estimands: no survivor-only reporting

Report all of the following per method and cell:

- full-horizon probability of ever raising an alarm under the null, with a binomial confidence interval;
- unconditional detection probability and unconditional miss fraction;
- power conditional on no pre-change false alarm, clearly labelled as conditional;
- delay measured from the true change point, never from stream start;
- median, p90 and p95 delay among detections *plus* miss count and the capped restricted delay;
- restricted delay with a miss penalty H+1 (or infinity for an uncapped survival estimand);
- worst morphology, worst scenario family and worst common-cause block;
- warning, quarantine and fallback completion as separate events;
- false warnings/quarantines per unioned safe operating hour, with raw counts and exposure denominator;
- safety violations, task success and safe useful progress;
- p50/p95/p99 update latency, memory, CPU and evidence-ingestion lag.

A low median among detections is not a win if the method misses more cases or alarms prematurely. A monitor that always quarantines cannot win by having zero post-change misses.

## 5. Statistical inference and independence

- The calibration sample size is the number of **effective independent calibration blocks** after CMAQ-036 common-cause closure, not the raw number of frames, clips, records or episodes.
- The evaluation unit is the effective block. CMAQ-044 pairs candidate and baseline on the same pre-registered block and resamples paired blocks, not episodes.
- For a primary family-level bootstrap interval, target at least 100 effective blocks per family. Fewer than two blocks cannot support a bootstrap interval; fewer than about 30 should be treated as exploratory. These are design targets, not guarantees of representative sampling.
- Keep tuning, validation and final holdout inventories disjoint by closure block. Freeze the candidate before opening final holdout results.
- Report every excluded block and why it was excluded. Do not exclude hard cases after observing outcomes.
- The bootstrap is conditional on effective blocks being representative/exchangeable; it is not a distribution-free interval under arbitrary dependence.
- Use CMAQ-043 for fleet-wide alpha allocation. Running many individually valid alpha=0.05 monitors does not imply a fleet-level 5% false-alert budget.

## 6. Industry and deployment evidence tiers

Do not collapse these evidence tiers:

| Tier | Data / environment | What it can establish | What it cannot establish |
|---|---|---|---|
| A | Exact synthetic Pólya null and registered alternatives | Mathematical wiring, finite-sample simulations, stress coverage of registered alternatives | Real-world completeness |
| B | ImageNet-C / Waymo Open Motion | Open-loop score/rank calibration and distribution-shift behavior on frozen datasets | Closed-loop driving safety |
| C | nuPlan closed-loop scenarios | Planner behavior under its scenario and simulation assumptions | Physical-vehicle safety |
| D | CARLA custom scenarios | Controlled scenario interventions, fault injection and reproducible simulator stress | Current public leaderboard standing or real-world safety |
| E | Hardware-in-the-loop | Timing, integration, sensor/actuator and fallback behavior on connected hardware | Full operational fleet safety |
| F | Physical trials | Field performance in the tested operational design domain | Generalization beyond the tested domain |

For nuPlan, report collisions, drivable-area/off-road violations, time-to-collision, traffic rules, comfort/dynamics, mission completion and interactions alongside monitor outcomes. Use Waymo Open Motion for open-loop motion-prediction/score streams, not as a substitute for closed-loop planning evaluation. CARLA can host local reproducible scenarios even though its legacy leaderboard should not be represented as an active public benchmark.

## 7. Prospective acceptance targets

These are ambitious preregistered goals to test—not measured achievements:

1. **Null integrity:** point estimate at or below the allocated alpha and one-sided 95% upper confidence bound no more than alpha + 0.5 percentage points in the primary null grid; separately preserve the theorem/assumption statement.
2. **Detection:** at least 20% lower paired H+1 restricted mean delay than the strongest matched baseline on at least three preregistered alternative families, with a paired interval excluding zero and no >2 percentage-point miss/power regression in any critical family.
3. **Temporal dependence:** demonstrate a clear gain on at least one marginal-preserving dependence family without increasing full-horizon null false alarms beyond the local risk allocation.
4. **Intervention:** at least 99.9% of registered fallback-required simulator episodes complete fallback before independent recoverability exhaustion; report the confidence interval and family worst case.
5. **Useful operation:** at least 5% improvement in safe useful progress or task completion without increased safety violations or worst-family regression.
6. **Runtime:** p99 monitor overhead below 1% of the actual control-cycle budget, measured on the target processor under concurrent evidence ingestion; report absolute p50/p95/p99 latency and memory.

A target is only met when its matched baseline, denominators, uncertainty interval, code/data digests and exact-head artifact are preserved. No target alone constitutes a safety certification.

## 8. Decision rule for a SOTA claim

A claim requires:
- faithful reproduction of relevant published baselines;
- an independently frozen final holdout;
- matched score/tie/reference/horizon/split contracts;
- null error and misses fully reported;
- paired effective-block inference;
- no material worst-family or operational-safety regression;
- reproducible artifacts and exact source/build digests.

Until then, describe results as exploratory engineering or internal synthetic evidence, never as SOTA or industry-leading performance.

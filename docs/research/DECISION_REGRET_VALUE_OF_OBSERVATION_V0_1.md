# Decision-Regret Value of Observation (v0.1)

**Tracking:** NEURO-SILICON-012 / #9563  
**Evaluator dependency:** #13152 decision-observability certificate  
**Candidate policy inputs:** #13151/#13167 risk-adjusted VOI and observation-specific resolvability  
**Mixed-strategy extension:** #13165  
**Joint resource integration:** #9666 / #13176  
**Calibration dependency:** #11367  
**Isolation dependency:** #9437  
**Status:** source-authored evaluator module and golden fixture; Rust execution pending

## 1. Research question

Does a sensor observation reduce the regret of the decision that must be made, rather than merely reduce entropy, prediction error, or a candidate confidence score?

This module composes the evaluator-owned hidden-world/action-loss contract in `decision_observability.rs`. It introduces no candidate sensing controller, new loss oracle, calibration service, acquisition executor, or resource allocator.

## 2. Formal contract

Let:

- (W) be one already-qualified observational equivalence class;
- (A) be its common admissible action set;
- (L(w,a)) be evaluator-owned loss in integer PPM;
- (t_d) be the frozen decision deadline;
- (q(y\mid w)) be the evaluator-owned probability of a candidate-visible sensor result by the deadline;
- (p(w)) be the independently qualified evaluator prior.

Per-world regret is:

\[
r(w,a)=L(w,a)-\min_{a'\in A}L(w,a')
\]

The current deterministic minimax-regret floor is:

\[
R(W)=\min_{a\in A}\max_{w\in W}r(w,a)
\]

The candidate-visible result space includes a single `no_result_by_deadline` outcome. A result available exactly at (t_d) is on time; a result strictly after (t_d), or no result, maps to that no-result outcome.

For every possible visible outcome:

\[
W_y=\{w\in W:q(y\mid w)>0\}
\]

The robust post-observation floor is:

\[
R_{post}^{robust}=\max_{y:W_y\ne\emptyset} R(W_y)
\]

and the guaranteed decision-regret reduction is:

\[
VOI_{robust}=R(W)-R_{post}^{robust}
\]

This is deliberately worst-case over all outcomes with nonzero support. A low-probability adverse result still matters to a worst-case guarantee.

For expected regret, use a separately qualified prior and observation model:

\[
ER_{before}=\min_{a\in A}\sum_w p(w)r(w,a)
\]

\[
ER_{after}=\sum_y\min_{a\in A}\sum_w p(w)q(y\mid w)r(w,a)
\]

\[
VOI_{expected}=ER_{before}-ER_{after}
\]

The exact integer numerator is reported over denominator (10^{12}), because both prior and likelihood use PPM. No floating-point posterior or rounding is used in the evaluator arithmetic.

## 3. Why both results are mandatory

A sensor with full-support noise can distinguish worlds probabilistically without eliminating any world from the possible-outcome set.

For example, with two equally likely worlds requiring opposite actions and a sensor that is 80% accurate:

- baseline worst-case regret is 1,000,000 PPM;
- after either sensor outcome, both worlds remain possible;
- robust worst-case regret remains 1,000,000 PPM;
- expected regret falls from 500,000 to 200,000 PPM;
- expected regret improvement is 300,000 PPM.

It is incorrect to report the expected improvement as a guaranteed safety improvement. This module reports both axes separately.

## 4. Input and qualification rules

The input is evaluator-only and binds:
- the already-qualified base equivalence class and its common candidate view/action contract;
- a nonzero digest of the exact sensor model/configuration;
- one positive PPM prior for every world, summing exactly to 1,000,000;
- a nonzero commitment digest identifying the prior/probability evidence artifact;
- conditional outcome likelihoods for every world, each summing exactly to 1,000,000 PPM;
- the decision deadline and candidate-visible result availability timestamps (after acquisition, transport, queueing and preprocessing);
- a SHA-256 digest of the complete candidate-visible transcript after each delivered result is appended.

Identical candidate-visible transcripts must have identical digests, including availability time and status. Late results are not allowed to reveal their future content to the candidate. They collapse to the no-result-by-deadline class. A delivered missing/error status can be represented as a normal candidate-visible transcript with its own digest.

Duplicate latent rows that produce the same candidate-visible transcript are combined by checked integer addition. Unknown world IDs, incomplete probability mass, out-of-range probabilities, empty delivered transcript digests and arithmetic overflow are rejected.

The prior commitment is provenance binding only; it does not prove calibration. Expected-value claims must consume the existing independently qualified calibration owner (#11367). A candidate's own prior or likelihood prediction must not be treated as evaluator ground truth.

Availability time means the instant the complete observation is actually accessible to the candidate policy, not the sensor event time or the first byte arrival. This keeps transport and preprocessing delay inside the decision deadline.

## 5. Output

For the complete observation model, the evaluator reports:
- baseline and post-observation deterministic minimax regret;
- guaranteed worst-case regret reduction;
- exact expected-regret numerator before and after observation;
- exact expected-regret improvement numerator and denominator;
- conservative integer lower bound on expected improvement;
- per-visible-outcome possible-world count;
- per-outcome robust minimax-regret action and floor;
- per-outcome Bayes-regret-optimal action;
- joint probability mass in PPM squared.

All per-outcome actions, likelihoods, hidden-world identities, priors and losses are evaluator-side outputs. They must never be included in the candidate-visible payload or handed to the candidate as a policy hint.

## 6. Controls

Eleven Rust unit tests are authored for:
1. perfect observation eliminating decision-relevant ambiguity;
2. noisy full-support observation with positive expected but zero robust value;
3. decision-irrelevant latent differences;
4. late observation collapsing to no-result;
5. delivery exactly at the deadline;
6. explicit missing observation;
7. invalid prior mass;
8. invalid per-world likelihood mass;
9. aggregation of identical candidate-visible transcripts;
10. unknown world rejection and missing provenance commitments.

The fixture supplies hand-calculated golden values for perfect, noisy, irrelevant and late-observation cases.

## 7. Integration experiment

Retain the existing #9563 policy arms and compare:
- fixed and random matched-budget sensing;
- event, novelty, uncertainty and residual-triggered sensing;
- ordinary VOI;
- Li & Parlikad's peer-reviewed value-based active-sensing baseline (IEEE TSMC, DOI 10.1109/TSMC.2026.3745855);
- #13151 risk-adjusted VOI;
- #13167 observation-specific resolvability;
- a future candidate decision-regret-aware arm, only after its candidate-visible prediction contract is frozen.

The evaluator score may be implemented before that last candidate policy exists. That allows the campaign to measure whether existing policy predictions correlate with actual decision-regret reduction.

Run paired hidden-stream replays, preserve late/missing/drop outcomes, and hold out source families. Integrate work, freshness and resource measurements via #13140/#13142 and #9666. Use #13176 for exact small factorial attribution and randomized multi-episode interventions for causal effects.

## 8. Claim ceiling

This is a finite, task-, action-, prior-, sensor- and deadline-specific evaluator certificate. It is not a universal theorem about all hidden processes, a safety certification, or a physical energy measurement.

The robust result is valid only for the declared finite possible-outcome support. The expected result is meaningful only when its prior and likelihood artifacts are independently qualified. Mixed-policy worst-case guarantees require #13165's mixed-strategy contract; this v0.1 module reports deterministic minimax regret and Bayes expected regret.

Package separation is not evaluator isolation. Confirmatory use still requires process/UID isolation, protected evaluator state and signed/replay-resistant receipts under #9437.

No Rust build, rustfmt, Clippy, unit-test execution, sensor campaign, hardware measurement, SOTA result or deployment qualification is claimed.

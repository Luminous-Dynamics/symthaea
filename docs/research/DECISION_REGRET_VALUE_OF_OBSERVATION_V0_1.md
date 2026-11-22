# Robust Decision-Regret Bounds for Sensor Outcomes (v0.1)

**Tracking:** NEURO-SILICON-012 / #9563  
**Base evaluator:** #13152 deterministic decision-observability bounds  
**Information-channel / probability owner:** #12567  
**Existing decision-predictability implementation:** #12546 / PR #12551  
**Common temporal benchmark:** #9954  
**Joint resource integration:** #9666 / #13176  
**Isolation dependency:** #9437  
**Status:** evaluator source and support fixture authored; Rust execution pending

## 1. Narrow research contribution

This module answers one question:

**Across every candidate-visible observation outcome that remains possible, how much can the deterministic minimax-regret floor improve by the decision deadline?**

It composes #13152's finite hidden-world/action-loss contract. It does not create a candidate sensing controller, expected-value oracle, probability calibrator, acquisition executor or resource allocator.

An important ownership distinction is required. #12567 already specifies noisy/delayed probes, information-channel integrity, posterior consistency, information gain versus decision value, and expected posterior Bayes-risk controls. PR #12551 already implements expected value of information from expected posterior Bayes risk, a Pareto cost vector, probe permissions and deadline admission. This PR does **not** duplicate those calculations.

The new complementary output is the robust worst-case regret envelope over the complete support of candidate-visible outcomes.

## 2. Formal contract

Let:

- (W) be an already-qualified observational equivalence class;
- (A) be its common admissible action set;
- (L(w,a)) be evaluator-owned loss;
- (r(w,a)=L(w,a)-\min_{a'\in A}L(w,a')) be per-world regret;
- (t_d) be the frozen decision deadline;
- (Y(w)) be the complete set of candidate-visible outcomes possible in world (w) by (t_d).

The baseline deterministic minimax-regret floor is:

\[
R(W)=\min_{a\in A}\max_{w\in W}r(w,a)
\]

For each candidate-visible outcome (y), define its compatible hidden-world set:

\[
W_y=\{w\in W:y\in Y(w)\}
\]

The robust post-observation floor is:

\[
R_{post}^{robust}=\max_{y:W_y\ne\emptyset}R(W_y)
\]

The finite-support guaranteed regret reduction is:

\[
VOI_{robust}=R(W)-R_{post}^{robust}
\]

This is a worst-case support bound, not an expectation. Every outcome with nonzero possibility remains in the maximum, however small its probability.

## 3. Outcome and deadline semantics

An outcome identity comprises the candidate-visible transcript digest and the time the complete result becomes accessible to the candidate policy. The availability timestamp is also a separate key component, so the evaluator will not silently merge outcomes that differ in decision-time freshness.

Availability includes acquisition, transport, queueing and preprocessing. It is not merely the sensor's event time or first-byte arrival.

A result available exactly at (t_d) is on time. A result strictly after (t_d), or no result, maps to one candidate-visible `no_result_by_deadline` outcome. A missing/error status that is actually delivered before the deadline is an ordinary visible result with its own transcript digest.

Repeated latent rows that map to the same visible digest and availability timestamp are deduplicated. All worlds must have at least one declared possible outcome.

## 4. Support completeness is a prerequisite

The support manifest must conservatively enumerate every candidate-visible outcome with nonzero possibility in each hidden world. If the support is incomplete, the robust guarantee can be overstated; the evaluator cannot infer omitted outcomes.

A nonzero support-evidence digest binds the artifact identity but does not certify its completeness. Confirmatory support qualification must compose with #12567's information-channel controls and an independently reviewed sensor/outcome model. Unknown support must remain unqualified, not be treated as impossible.

This module accepts no probabilities or priors. The likelihoods, posterior normalization, expected Bayes risk, expected VoI and calibration remain with #12567/#12551. This evaluator consumes only the positive-support relation.

## 5. Outputs

The evaluator reports:
- baseline deterministic minimax-regret floor;
- worst post-observation minimax-regret floor across all possible visible outcomes;
- guaranteed robust regret reduction;
- per-outcome possible-world count;
- per-outcome minimax-regret-optimal action and regret floor;
- visible transcript digest and availability timestamp, or the no-result class.

The outcome-specific actions and hidden-world sets are evaluator-only. They must never be included in candidate-visible payloads or used as policy hints.

## 6. Adversarial controls

Eleven Rust unit tests are authored for:
1. a perfect on-time observation removing decision-relevant ambiguity;
2. a noisy full-support observation providing no worst-case guarantee;
3. decision-irrelevant hidden variation;
4. late observations collapsing to no-result;
5. delivery exactly at the deadline;
6. absent observations remaining explicit;
7. duplicate rows with the same visible transcript being deduplicated;
8. distinct availability times remaining distinct outcome keys;
9. unknown world IDs being rejected;
10. worlds without declared possible outcomes being rejected;
11. empty sensor, support-evidence and delivered-transcript commitments being rejected.

The machine-readable fixture supplies hand-calculated perfect, noisy/full-support, irrelevant and late-result cases.

## 7. Experimental integration

Use the existing owners rather than adding another policy or benchmark:
- #9563 supplies the active-sensing policy arms and temporal observability controls.
- #13152 supplies the base deterministic minimax-regret certificate.
- #12567/#12551 supply noisy/delayed outcome integrity, posterior consistency and expected posterior Bayes-risk/VoI evaluation.
- #9954 owns the common event-time/availability-time temporal benchmark contract, source splits, baseline registry and multi-axis result receipts.
- #9666/#13176 own matched vector-budget joint allocation and exact factorial interaction attribution.
- #13140/#13142 own source-bound work and freshness receipts.
- #9437 owns the process-isolated oracle and signed evidence-receipt boundary.

Run paired hidden-stream replays and held-out source families. Report robust regret floors separately from expected posterior Bayes risk, temporal exposure, freshness, cost and physical energy. Use #13176 for exact finite factorial decomposition on small action subsets; randomized multi-episode intervention designs remain necessary for causal effects.

External comparison lanes remain separate: BITS/TIME for irregular forecasting, maintained NeuroBench for compatible algorithm/system work, MLPerf for compatible deployment workloads, and RoboDyna for asynchronous physical action under inference latency.

## 8. Claim ceiling

This is an exact finite-support deterministic minimax-regret calculation for the supplied action/loss matrix and declared outcome support. It is not a universal guarantee over omitted worlds, a probabilistic expected-value result, a mixed-strategy guarantee, a safety certification or a physical energy measurement.

Mixed-policy worst-case bounds remain with #13165. Expected posterior value remains with #12567/#12551. Confirmatory evaluation requires #9437 process isolation and protected evaluator state.

No Rust build, rustfmt, Clippy, unit-test execution, benchmark campaign, energy result, SOTA improvement or deployment qualification is claimed.

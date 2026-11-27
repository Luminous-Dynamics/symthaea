# SEMI-CA-OCRC-001 — capacity-aware delayed online conformal risk control v0.2

**Canonical issue:** #13578  
**Implementation branch:** feat/semi-capacity-aware-ocrc-v1  
**Implementation module:** symthaea-semiconductor::benchmark::capacity_aware_ocrc  
**Reference:** Wang & Simeone, arXiv:2611.17059 (12 November 2026), https://arxiv.org/abs/2611.17059  
**Status:** bounded-arbitrary-delay controller with K1 evidence receipts, independent scalar/queue oracle, and explicit guarantee/population status; fresh exact-head validation pending.

## 1. Scientific scope: three different probabilities, not one missingness variable

A semiconductor outcome passes through three distinct gates:

1. **Physical acquisition:** was this wafer actually selected for high-fidelity metrology?
2. **Evidence qualification:** did the acquired result pass calibration, identity, measurand, clock, numeric-value and uncertainty checks at K1?
3. **Feedback tracking:** was the existing outcome retained by the finite-capacity randomized proxy policy until its delayed result arrived?

CA-OCRC's randomized tracking and inverse-probability correction apply to stage 3. They do not create a physical outcome at stage 1, nor do they make a non-qualifying result at stage 2 statistically missing-at-random.

Therefore this implementation declares its population scope explicitly. The initial default is QualifiedMeasuredStreamOnly. A caller cannot silently reinterpret a qualified-label stream as all fab wafers.

Separate research remains active:
- #13579 — selective physical-metrology acquisition and tracking bias;
- #13646 — population coverage under selective qualification;
- #13328 — adaptive acquisition/EVSI and assay-adjusted prevalence;
- #13467 / #13461 — numeric label and K0/K1 snapshot evidence.

## 2. Paper assumptions and exact bounded-delay recursion

The first implementation covers arbitrary feedback delays with a known bound 0 <= d_t <= d_max. It does not yet implement the paper's IID stochastic-delay G-aware optimization.

A frozen global predictor and score S_t(X_t,y) in [0,B] produce a prediction set C_t={y:S_t(X_t,y)<=r_t}. The set loss is bounded in [0,1], with full-set loss 0 and empty-set loss 1. At most K pending outcomes may be tracked.

The proxy delay dbar_t has support {-1,0,...,d_max} with survival F(a)=Pr(dbar_t>=a), F(-1)=1. The probability mass is reconstructed as:
- Pr(dbar=-1)=1-F(0);
- Pr(dbar=a)=F(a)-F(a+1), for 0<=a<d_max;
- Pr(dbar=d_max)=F(d_max).

The proxy delay is sampled from a caller-supplied replayable uniform draw. No hidden PRNG or entropy source is used by the reference controller.

For each round t, the implementation uses this order:

1. Validate the immutable prediction and the exact contiguous round index.
2. Evict tracked i only when i+dbar_i<t. Feedback at age exactly dbar_i remains receivable.
3. For a nontrivial set, sample dbar_t, then admit t iff dbar_t>=0 and the post-eviction tracking set has free capacity.
4. Record the immutable prediction/interval record and tracking ticket.
5. Process all feedback delivered at round t, in deterministic stable order.
6. Apply every K1-qualified received loss exactly once, weighted by 1/F(d_i).
7. Apply the current known-loss term only for a trivial full/empty set and seal the round receipt.

The arbitrary bounded-delay update is:

r_(t+1)=r_t+eta*phi_t*(L_t-alpha)+eta*sum_{i in O_t}(L_i-alpha)/F(d_i),

where phi_t=1 only for a trivial set and O_t is the set of tracked, qualified outcomes received at this round. In the bounded setting q_G=1.

The production controller uses the frozen prediction interval to compute L_i=1{Y_i not in C_i} from the K1 corrected measurement. The true loss and actual delay are not supplied before delivery.

## 3. Theorem-claim gate and bound calculator

The arbitrary-delay tracking-load condition is ell_F=sum_(a=1..d_max)F(a)<=K. With h(x)=x-1-ln(x), the paper's risk-gap bound is:

RiskGap(T) <= B/(eta*T)+(K+1)/(F(d_max)*T)+exp(-K*h(ell_F/K)).

The bound helper refuses zero horizon, a proxy survival of zero at the maximum delay, and a tracking load above capacity. It reports the numerical bound and the parameter terms separately from measured empirical risk.

### Critical correction: spare capacity is not unlimited feedback

K>=d_max+1 eliminates capacity saturation only when the proxy policy also retains every nontrivial outcome: F(a)=1 for all a=0,...,d_max. If F(a)<1, some outcomes are deliberately untracked even with spare storage. The implementation now rejects the unlimited-OCRC lemma claim in that case. Do not call a configuration “unlimited feedback” merely because its queue is large.

### Run-level claim status

Each engine, round receipt and risk-bound result carries a declared population scope. The engine also retains a guarantee status:
- ConditionalOnDeclaredPaperAssumptions;
- InvalidatedByK1QualificationFiltering;
- InvalidatedByFeedbackDelayBound;
- InvalidatedByMalformedFeedbackEvidence.

A K1 UNKNOWN/BLOCKED label is never turned into a zero loss and never updates the threshold. It invalidates the inherited complete-feedback paper claim for the observed stream. A delay beyond the declared maximum likewise invalidates it. These statuses are conservative software guards; they do not prove physical source authenticity.

The initial population scope is QualifiedMeasuredStreamOnly. This is metadata describing the estimand, not evidence that this stream represents the full production population. A full-population claim requires a separate selection design and estimand validation.

## 4. Implemented contracts

The Rust module currently provides:
- versioned policy and engine;
- discrete proxy survival validation and inverse-CDF sampling from a recorded uniform draw;
- immutable prediction and interval receipts;
- exact event/sample/clock/model/policy identity checks;
- bounded tracking map and explicit admission states;
- strict proxy expiry ordering;
- deterministic feedback ordering for simultaneous arrivals;
- K1 label report digest, current frontier, eligibility, downstream permission, label ID/revision, measurement event, payload digest and numeric value checks;
- label availability, report availability and delivery chronology;
- duplicate feedback/idempotency handling;
- full/empty trivial sets with known loss and no tracking slot;
- inverse-survival weighted loss updates;
- atomic round state transition using a cloned engine;
- arbitrary-delay theorem bound and one-dimensional Bernoulli proxy optimizer.

The physical-selection probability and K1 qualification are not multiplied into the CA-OCRC IPW term. No doubly robust estimator, G-aware stochastic delay extension or clipped IPW policy is included.

## 5. Independent oracle and adversarial tests

The unit test module includes capacity saturation, expiry boundary, late feedback, zero delay, proxy delay -1, trivial full/empty sets, K1 UNKNOWN, inverse-survival weights, simultaneous arrivals, invalid survival tables, capacity >= d_max+1, interval tampering, revision/payload mismatch, bound calculations and optimizer cases.

A separate integration test file implements a scalar reference that independently computes:
- inverse-CDF proxy delays for a frozen survival distribution;
- interval miss bits;
- per-delay inverse-survival weights;
- the threshold recursion;
- deadline-inclusive reception/eviction order;
- the distinction between spare capacity and full proxy retention;
- guarantee-status downgrade for K1-filtered feedback.

The oracle deliberately does not call the production proxy sampler or threshold-update helper. Its exact-head formatter, strict Clippy, integration tests and reference workflow remain to be run on the current revision.

## 6. Required benchmark tournament

Compare at identical streams and declared capacities:
- OCRC with full tracking and F(a)=1;
- intermittent K=1;
- CA-OCRC with the paper's optimized Bernoulli proxy;
- CA-OCRC with a fixed, preregistered proxy distribution;
- FIFO/oldest-first, random admission and nearest-expiry heuristics as non-theorem baselines;
- naive drop-feedback without IPW as an adversarial baseline;
- evidence-qualified measured-stream CA-OCRC with theorem status invalidated when K1 feedback is excluded.

Report:
- empirical risk over all simulated rounds (including offline-known but untracked losses);
- absolute risk gap to target and paper's bound/slack;
- proxy draws, admissions, capacity denials, expiries, received and untracked counts;
- occupancy distribution, maximum occupancy and saturation frequency;
- IPW distribution, maximum weight and effective sample size;
- interval width/score and degenerate-set frequency;
- per-context risk and delay distribution;
- update compute p50/p95 and state memory;
- guarantee status and population scope separately from numeric risk.

The same random draw sequence must be replayed across controller variants when performing paired comparisons. Do not optimize F on the test stream.

## 7. Competitive positioning and research extensions

CA-OCRC is a risk-control/feedback-retention algorithm, not a VM point-prediction model. It should be compared against unlimited OCRC and intermittent OCRC under matched score/loss/delay assumptions.

The high-value semiconductor extension is the composition:
physical acquisition -> measurement qualification -> feedback tracking -> risk update.

But each stage has its own missingness mechanism.

The physical acquisition and qualification problems are not absorbed by the paper's randomized tracking correction. Before any population guarantee, require a randomized audit sample or a separately justified sequential propensity design with positivity and bounded weights. For MNAR/unobserved selection, report worst-case/partial-identification bounds instead of inventing an IPW point estimate.

Deferred work:
- stochastic delays and known G: reproduce the paper's Section VI equations and assumptions independently first;
- finite-capacity random tracking under estimated G;
- joint physical-selection + tracking estimators only after a theorem under explicit assumptions;
- K1 qualification-revocation effects on rolling guarantee windows;
- cross-tool/chamber audit design and panel-specific risk.

## 8. Claim ceiling

The code is a software reference for the paper's arbitrary bounded-delay algorithm under its declared conditions. It does not establish that fab metrology acquisition is random, that qualification failures are MAR, that source issuers are authentic, that production-wide conditional coverage holds, or that the controller improves yield. It grants no wafer-disposition, recipe, APC/R2R or equipment-control authority.

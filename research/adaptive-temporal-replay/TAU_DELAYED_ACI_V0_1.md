# Fixed-Horizon Delayed ACI Reference (tau-DACI v0.1)

**Status:** deterministic state-machine reference and evaluator metric only. No Rust compilation, test execution, interval-construction implementation, or coverage claim is made by this document.

## 1. Research anchor and exact recursion

El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic* (arXiv:2609.07251, 2026-09-07), study forecasts with a fixed lead time tau. A prediction issued at origin t is evaluated only when its target at t+tau arrives.

For target miscoverage alpha, adaptation step size gamma, the delayed recursion is:

    alpha_(t+tau) = alpha_t + gamma * (alpha - err_t)

where err_t is 1 if the prediction set issued at t fails to cover its target and 0 otherwise. The critical implementation detail is that the feedback update uses the alpha **stored with the prediction that is now being evaluated**, not a fresh global alpha selected after observing other phases.

The same recursion decomposes into tau independent interleaved phase sequences. For phase theta:

    alpha^(theta)_(m+1) = alpha^(theta)_m + gamma * (alpha - err^(theta)_m)

The paper's phase definition is t_theta(m) = t0 + theta + (m-1)tau. For tau=5, one phase contains times 1,6,11,16,...; another contains 2,7,12,17,... . Each phase uses its own alpha state.

Algorithm 2 performs, at each origin t:
1. If the corresponding target t is now due, dequeue its prediction set and stored alpha_t.
2. Compute err_t against the target outcome.
3. Set the phase level to alpha_t + gamma(alpha - err_t).
4. Construct the new prediction for target t+tau with that level.
5. Store the new prediction set and the level used, for later feedback.

Source: https://arxiv.org/abs/2609.07251

## 2. What the Symthaea implementation contains

The Rust source in research/adaptive-temporal-replay/src/lib.rs contains:

- DelayedAciBank, a fixed-horizon phase bank with exactly tau independent alpha states;
- a strict consecutive-origin API; it rejects feedback before the first target is due, missing due feedback, skipped origins, duplicate targets and zero prediction digests;
- a queue binding each issued target to its origin, phase, alpha used and prediction-set digest;
- feedback records binding the due target, prediction origin, alpha used, digest and miss/cover outcome;
- deterministic ordered receipts and a domain-separated history SHA-256;
- replay verification that reconstructs all phase states and pending predictions;
- AciSetRegime, which reports in-range, all-real (alpha < 0) or empty (alpha > 1) semantics. It does not clip the learned alpha.

This is intentionally a controller/replay substrate. The caller still constructs the actual interval and computes the miss/cover result. The controller does not silently invent an interval from a scalar.

Each step receipt distinguishes the alpha used by the prediction now being scored (`alpha_used_ppm`) from the updated phase alpha used for the newly issued prediction (`updated_alpha_ppm`). During the initial tau-step warm-up there is no due feedback, so both fields equal the initial phase alpha.

### Numerical claim boundary

The paper states its recursion over real-valued alpha. The Rust implementation stores alpha in integer parts-per-million and rounds each update to nearest PPM with ties away from zero. This makes the controller deterministic and auditable, but it is a numerical approximation. The paper's exact finite-sample theorem is **not automatically inherited** by this rounded implementation; a numerical error analysis or exact-rational implementation would be required before making that claim.

The current method supports only a fixed, guaranteed delay tau with one feedback bit arriving at each due target step. It does not support variable delays, missing feedback, retractions, superseding labels, or out-of-order outcomes. Those need distinct algorithms and filtrations. In particular, a later label revision must not be fed backward into the original update.

The earlier AciState remains a simple scalar no-delay baseline. It is not tau-DACI.

## 3. Theoretical guarantee and degenerate levels

The paper shows the phase-wise alpha sequence is bounded in [-gamma, 1+gamma] under the stated binary coverage construction. Combining the phase bounds yields a finite-sample long-run empirical miscoverage bound with an O(tau/T) term; as T grows, average miscoverage converges to alpha under its stated setup.

The implementation must preserve out-of-unit levels because they are part of the recursion:
- alpha < 0 means the signed-residual interval is all of R, so the next error is necessarily 0;
- alpha > 1 means the interval is empty, so the next error is necessarily 1.

This software does not clip alpha to [0,1]. The PPM rounding caveat above applies to the numerical form of the recursion. The paper's approximate **marginal** coverage result is separate and requires its additional stationary hidden-Markov/Lipschitz/unique-optimal-level assumptions; do not present it as distribution-free.

## 4. Evaluation metrics

### Worst-case local coverage error

For binary miscoverage outcomes err_1,...,err_T and window length k:

    LCE_k = max_{1 <= j <= T-k+1} | alpha - (1/k) * sum_{t=j}^{j+k-1} err_t |

The Rust helper worst_case_local_coverage_error_ppm computes this in O(T) time for one k using a rolling count, returning an integer-PPM value. When T < k it returns None, not zero. Evaluate every preregistered k and report the values separately; overlapping windows are descriptive diagnostics, not independent statistical replicates.

### Interval score

For an interval [l_t,u_t], outcome y_t and miscoverage alpha, the paper uses:

    IS_alpha = (u_t-l_t)
             + (2/alpha) * max(0,l_t-y_t)
             + (2/alpha) * max(0,y_t-u_t)

Report empirical coverage, LCE_k and mean interval score together. A method cannot win just by widening intervals to cover everything.

## 5. Comparators and industry reproduction

UP-OCP (Liu, Dobriban & Orabona, ICML 2026 / PMLR 306) develops a regret-to-coverage theory for interval online conformal prediction and a parameter-free two-asset universal-portfolio method, with finite-time miscoverage bounds. It is a required comparator, not a feature to imitate superficially:
- Paper: https://proceedings.mlr.press/v306/liu26y.html
- Public reference code: https://github.com/AdaMLLab/up-for-ocp
- Pinned public main commit inspected for this audit: 3575f1f94a14620e089422f52cf9e1a6766ec7b1
- The repository declares the MIT license. Its implementation has a UniversalPortfolio class with an explicit alpha correction, Dirichlet/uniform mixture options, wealth/betting updates and a quantile-radius pipeline. Reproduce paper-specific configurations and tests from the paper before interpreting this code as an exact theorem implementation.

The calibration comparison matrix also includes static split conformal (stationary control), ordinary ACI/FACI, SAOCP/SF-OGD, Gibbs-Candes local-regret adaptation, fixed sliding-window quantiles, Vaze's simultaneous coverage/efficiency method, delayed-ACI/MACP/AcMCP, PACE (heuristic) and Symthaea BMAC (unproven candidate). Formal guarantees and empirical results must remain in separate tables.

## 6. Verification status

Authored Rust tests cover:
- tau=1 equivalence to the simple scalar recursion while the level remains inside [0,1];
- three-phase independence and phase-specific updates;
- mandatory feedback at the exact fixed horizon;
- transactional failure on missing/skipped feedback;
- receipt replay and tamper detection;
- no alpha clipping and all-real/empty semantics;
- exact known-answer LCE_k windows.

No Rust test, build, formatting or Clippy run is claimed in this environment. No interval coverage, empirical performance, or theorem validation is claimed by the existence of this code.

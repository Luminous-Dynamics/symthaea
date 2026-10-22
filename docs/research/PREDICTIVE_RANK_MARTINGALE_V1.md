# Predictive-Rank Martingale Shift Benchmark v1

**Status:** draft synthetic benchmark; no exact-head qualification or external performance result is claimed until the Rust workflow passes.

## Motivation

The existing episode-max-calibrated residual CUSUM is useful, but it is not the last word in sequential monitoring. Recent work has identified a subtle statistical issue: if a monitor repeatedly ranks future observations against a fixed calibration set, the ranks are dependent conditional on that fixed set. Treating them as independent conformal p-values does not establish an anytime false-alarm guarantee.

Kuang and Xia (2026) derive the exact predictive null distribution for the next rank conditioned on previous ranks and construct a predictive-rank martingale (PRM). They use feature-based centered payoffs and adaptive online Newton-step betting; an equal-weight portfolio of order and dispersion features covers directional shifts and center-versus-tail shifts. Under the paper's assumptions, nonnegative martingale wealth and a Ville threshold provide finite-sample anytime marginal type-I error control.

This branch implements the predictive-rank conditioning, order/dispersion features, bounded predictable bets, and a portfolio. Its ONS-style gain coefficient (4.5) is a local prototype choice and has **not yet been independently reconciled with the paper's exact optimizer parameterization**. The construction should therefore be described as PRM-inspired, not an exact reproduction of Kuang and Xia's algorithm; their specific power/detection-delay results do not automatically transfer to this implementation. This does not remove the need to verify the rank null assumptions or the positive-factor bounds. It is not an implementation of every recent conformal detector or a claim of superior detection delay.

Two additional current baselines matter:
- Conditional conformal test martingales (CCTM) use a fixed null reference and account for reference estimation error.
- Bhattacharyya and Ramdas (2026) analyze limits of existing conformal changepoint procedures and propose other e-process/e-detector constructions with minimax delay results under their stated models.

The benchmark is meant to put these methods into an auditable comparison program, not to relabel the first implemented baseline as SOTA.

## Algorithm implemented

A fixed sample of (n) nominal score tokens is sorted once. Each token carries an independent tie-breaking key so rank order remains deterministic when score values tie. The calibration reference is never updated with test-time observations.

For test token (X_t), rank (R_tin{1,ldots,n+1}) is computed against that fixed sample. Let (N_{t-1,j}) be the number of previous test points with rank (j). The predictive null probability of the next rank is

\[
\pi_{t,j}=\frac{1+N_{t-1,j}}{n+t}.
\]

For a predeclared feature (h), the observed rank feature is centered by this predictive null mean:

\[
Z_t=h\!\left(\frac{R_t-1}{n}\right)
-\sum_{j=1}^{n+1}\pi_{t,j}
 h\!\left(\frac{j-1}{n}\right).
\]

An adaptive coefficient \(\lambda_t\), selected only from previous rank history, updates wealth by (M_t=M_{t-1}(1+\lambda_t Z_t)). The bounded ONS-style update uses a local fixed step-size coefficient and a conservative projection radius so the betting factors remain bounded away from zero. The exact coefficient and projection must be checked against the source algorithm before claiming faithful reproduction. The monitor alarms when its wealth reaches (1/\alpha). The portfolio is the equal-weight mean of the order and dispersion wealth processes, so it also starts at unit wealth and retains the component false-alarm guarantee when the component assumptions hold.

Features used here:

- **Order:** (h(u)=u-1/2). Positive and negative signals correspond to upper- and lower-ranked observations.
- **Dispersion:** (h(u)=|2u-1|-1/2). The two tails reinforce, so symmetric spread changes can be detected even when the order feature's signed mean cancels.
- **Portfolio:** the mean of the order and dispersion martingale wealths.

The implementation maintains the conditional feature expectation incrementally, so each observation does not require rescanning every rank bin.

## Benchmark design

The deterministic synthetic panel has:

- 159 independent-seed calibration episodes for the fixed reference sample;
- 159 full nominal calibration episodes for a fixed-horizon episode-maximum threshold;
- 300 held-out episode seeds per evaluation cell;
- a 256-step monitoring horizon;
- shift modes: upward score shift, downward score shift, scale/dispersion shift, and sensor-blind negative control;
- shift magnitudes: 0.04, 0.10 and 0.20;
- changepoints at steps 1, 17, 65 and 193, including long nominal prefixes.

This yields **14,400 evaluation streams** and 48 summary cells. The same base score and tie-key stream is reused for each matching seed across shift modes, magnitudes and onset labels.

The benchmark compares five alarm rules:
1. pointwise threshold calibrated at the per-observation nominal quantile, repeatedly applied across the full stream;
2. episode-maximum threshold calibrated from the maximum score of each nominal calibration episode;
3. PRM order feature;
4. PRM dispersion feature;
5. equal-weight PRM portfolio.

The pointwise baseline intentionally demonstrates the multiple-testing problem: a low per-observation false-alarm rate does not imply a low chance of at least one false alarm over a long stream. The episode-maximum baseline targets the stated finite horizon but must be recalibrated if the horizon or evaluation protocol changes. The PRM has an anytime guarantee only when the algorithm's assumptions about the nominal calibration sample and score stream are defensible.

## Metrics and negative controls

Each raw episode row records thresholds, first alarm steps for all five methods, and maximum portfolio wealth. Summary rows report pre-shift alarms, post-shift alarms, no post-shift alarm, and mean delay among streams whose first alarm occurs at or after the declared onset. Pre-shift false alarms are not re-labelled as successful detections.

The sensor-blind control changes the declared shift magnitude/onset but not the observation stream. For a given episode seed, all five alarm times and the portfolio-wealth trajectory must remain identical across sensor-blind magnitude and onset labels. This verifies that no label leaks into a score or detector.

The qualifier requires byte-identical repeated runs, exact row/schema coverage, the paired sensor-blind control, strong directional-shift responses for the order feature, a strong scale-shift response for the dispersion feature, and a source/output provenance receipt.

These synthetic score streams are deterministic software fixtures; their repeated seeds do not establish independent empirical trials. The run is a test of implementation and accounting, not a physical safety result and not independent verification of a mathematical proof.

## Research limits and next SOTA comparisons

- The current benchmark does not implement the parameter-free conditional conformal martingale of Liu, Prinster and Orabona (2026), the e-process constructions of Bhattacharyya and Ramdas (2026), or CCTM. These are next comparator targets after this prototype's exact betting update has been reconciled with the source PRM method and the tests are stable.
- For method-level SOTA comparisons, pin the authors' exact public revisions, reference-sample sizes, score functions, alpha allocation, shift distributions, changepoint positions and compute budget. Report false-alarm probability over the whole stream and detection-delay distributions; don't compare only average delay conditional on eventual detection.
- A distribution detector cannot identify a dynamics shift that leaves all of its observable inputs unchanged. A post-transition residual can provide new evidence only after a transition has happened; it may be too late for the action that produced it. The existing hazard-deadline frontier separately measures this causal timing problem.
- Even an anytime-valid shift alarm is not an action-safety certificate. The runtime-assurance layer must still route alarms to a separately verified reachability fallback and fail closed when the model, observation or certificate assumptions are invalid.

## References

- Kuang & Xia (2026), [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536).
- Bhattacharyya & Ramdas (2026), [Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods](https://arxiv.org/abs/2609.27179).
- Liu, Prinster & Orabona (2026), [Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales](https://arxiv.org/abs/2610.17343).
- Shaer et al. (ICML 2026), [Testing For Distribution Shifts with Conditional Conformal Test Martingales](https://proceedings.mlr.press/v306/shaer26a.html).
- Spoor, Plaat & Moerland (2026), [Evaluation Metrics for Safe Reinforcement Learning](https://arxiv.org/abs/2609.15315).

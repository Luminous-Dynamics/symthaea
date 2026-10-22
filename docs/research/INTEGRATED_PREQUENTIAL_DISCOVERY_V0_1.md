# Integrated Prequential PIT Audits in Symbolic-Law Discovery v0.1

**Status:** exploratory integration in the Symthaea physics-bridge simulator; exact-head Rust compile/test/format status is pending. It is a wiring and measurement-cost pilot, not evidence that the monitoring-enabled policy beats nominal EIG or state of the art.

## 1. Scope

This follow-on connects `PrequentialPitMartingale` to the existing active symbolic-law discovery loop in `integrated_discovery.rs`. It is intentionally narrower than a production scientific monitor: the simulator has a declared heteroscedastic Gaussian noise model and can therefore issue a conditional predictive distribution for a provisional singleton.

New selector path: `InformationGainWithPrequentialPitAudits`.

- While multiple hypotheses remain, select experiments by the existing nominal EIG-per-cost selector.
- Once exactly one provisional hypothesis remains, do **not** accept it immediately. Select the next admissible unused experiment randomly from the predeclared pool and score the result as a validation audit.
- Before each audit measurement, evaluate the provisional candidate and its predictive scale
  (\sigma(x)=\text{NOISE\_SIGMA\_FRACTION}\,(1+|\hat y(x)|)). After observing (y), compute the simulator's standard-normal PIT approximation
  (u=\Phi((y-\hat y(x))/\sigma(x))).
- Pass the PIT plus four bounded, pre-outcome context channels (normalized selected (m,v,r,t) coordinates) into the fixed-weight PIT martingale.
- Stop the provisional singleton run on a monitor alarm. The evaluator records detection separately from hidden truth; the detector receives only candidate predictions, observations, and the predeclared query contexts.

The simulator rejects the PIT-audit policy if CfC smoothing is enabled or adaptive selection is disabled. Smoothing changes the measurement distribution; treating its state estimate as one Gaussian draw would break the declared predictive-CDF contract. An unsupported configuration fails closed instead of silently reusing the PIT guarantee.

## 2. Important limit: alarm is not automatic model recovery

The current discovery simulator can stop accepting a provisional singleton and mark an adequacy alarm, but it does not yet synthesize, search, or validate newly expanded model families after that alarm. In a real loop the next action should be a typed transition to model-family expansion or an explicitly unresolved outcome. Do not interpret the current diagnostic flag as successful recovery of the true law.

The PIT monitor is only valid if the conditional predictive CDF is issued before the observation, its parameters/uncertainty protocol are frozen without leakage, and the context channels are based only on the selected action and permitted history. The current synthetic fixture has a known Gaussian noise model; this does not establish validity for real physics measurements, unknown noise, fitted-scale uncertainty, censored data, or adaptive repeated measurements.

## 3. Integration smoke test

The unit test `prequential_pit_audits_are_wired_into_adaptive_discovery` runs paired clean and discrepancy worlds for the same fixed seeds:

- nominal EIG baseline on the same generated world/seed;
- the new EIG-plus-prequential-PIT-audit selector;
- a declared discrepancy amplitude of 0.12 in the simulator's existing synthetic discrepancy basis;
- a 20-round cap and no smoothing, with audit measurement cost counted as a round/query;
- receipts reporting clean recovery, clean PIT alarms, false singleton outcomes, misspecification detection, actual audit queries, and protocol failures.

The test makes no assertion that the monitor improves detection rate or recovery. It asserts that the new branch actually reaches audit queries, that it never silently proceeds when the declared protocol is invalid, and that metrics remain within their possible counts. A second test rejects the smoothing configuration.

The following focused commands are added to the expected-information-gain workflow:

```sh
cargo test -p symthaea-physics-bridge prequential_pit_audits_are_wired_into_adaptive_discovery --lib -- --nocapture
cargo test -p symthaea-physics-bridge prequential_pit_policy_rejects_smoothed_measurements --lib
```

No Rust run is claimed until the exact-head checks finish.

## 4. What the comparison must measure next

Current smoke scale (32 paired seeds) only verifies integration. For a confirmatory test:

- Expand to at least 60 held-out structural worlds across six or more families and 20 paired seeds per world.
- Include null controls, omitted terms, scale and tail misspecification, context-specific shifts, query-region extrapolation, and inaccurate predictive uncertainty.
- Compare nominal EIG, scenario-robust EIG, scheduled random audits, batch conformal audits, finite-horizon spending, standalone PRM where its assumptions hold, and prequential PIT on a conditional-CDF-valid protocol.
- Keep identical candidate pools and noise streams paired where possible; bootstrap worlds first, seeds within worlds second.
- Publish clean structural recovery, false confident singleton rate, calibrated alarm rate, sensitivity by discrepancy magnitude, total measurement cost to valid recovery or explicit unresolved status, median/90th-percentile detection delay, compute and wall time. An alarm that stops forever is not a win unless valid recovery or an explicit unresolved report is counted.
- Preserve the model/query/score/CDF/config digests and exact source/toolchain receipts. Keep hidden truth and discrepancy labels in evaluator-only artifacts.

## 5. Comparison with published scientific-discovery systems

The intended external framing is not a single symbolic-accuracy leaderboard. SciLaws-Bench separates held-out predictive fit, scientific validity, and structural recovery for novel terms, with both fixed-data and active-query settings: https://yiyihum.github.io/SciLaws-Bench/. LLM-AutoSciLab / ActiveSciBench emphasizes a closed loop of hypothesis generation, query selection, and mechanism recovery; reported metrics include symbolic equation accuracy and exact graph recovery: https://scientific-discovery.github.io/llm-autoscilab-project/. LLM-SRBench evaluates transformed familiar equations and synthetic discovery tasks to reduce memorization effects: https://proceedings.mlr.press/v267/shojaee25a.html.

These scores are not directly comparable to the current Symthaea micro-simulator. Pin exact benchmark versions/splits, licenses, task prior knowledge, query/compute budgets and evaluator before a direct comparison. Use the scientific benchmarks to define the endpoint taxonomy; use the paired simulation to isolate the monitoring mechanism; then run a held-out active-discovery comparison to support any performance claim.

## 6. Advancement criteria (targets, not results)

Retain the PIT-audit selector as a candidate for broader evaluation only if the exact-head code passes its focused checks and subsequent matched-budget results show:

- no more than a 5-percentage-point decline in clean-world structural recovery;
- at least 20% relative reduction in false confident singletons under the preregistered misspecification mix;
- detection sensitivity at least 80% on alternatives chosen before opening the held-out split;
- at least 20% lower median measurement cost to valid recovery or explicit unresolved status than the best simple baseline, with paired world-clustered uncertainty intervals supporting the effect;
- no hidden-truth leakage, no invalid smoothed-PIT inputs, and no use of a non-alarm as proof of adequacy.

If a gate fails, report the negative result and retain nominal EIG or another audited policy where it does better. Do not adjust thresholds to manufacture a positive result.

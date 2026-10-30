# Anytime monitoring for assumption-source drift

**Date:** 2026-10-30  
**Status:** executed synthetic comparator pilot; no reproduction of published UAI risk-monitor code; not production validation.

## Research question

The issuer-reliability experiment in #9042 showed that a historical calibration can be completely invalidated by an unannounced issuer reliability shift. The next step is to monitor independently audited bound violations continuously, with an anytime false-alarm guarantee, and compare it to a fixed-look baseline.

The monitor operates on independently adjudicated audit outcomes. It does not infer the contamination bound's truth from the contaminated observations used by the downstream statistical certificate.

## Method

Let X_t=1 indicate that an independent audit found the issuer's declared upper bound failed to cover audited truth. The null is a conditional-probability claim:

`P(X_t=1 | F_{t-1}) <= p_0`

for every observation in the qualified scope/epoch. For each predeclared alternative q>p0, define a likelihood-ratio process

`E_t(q) = product_{s<=t} (q/p0)^X_s ((1-q)/(1-p0))^(1-X_s).`

For any q>p0, each one-step factor has conditional expectation at most one under the composite null, so E_t(q) is a nonnegative supermartingale. A fixed positive mixture over q is also a supermartingale. Ville's inequality gives an anytime false-alarm bound: crossing 1/alpha has probability at most alpha under the stated conditional null. The benchmark uses alternatives {0.015, 0.03, 0.05, 0.10, 0.20}, p0=0.01 and alpha=0.01.

The comparator is a 50-observation rolling exact binomial upper-tail test, evaluated every 25 observations, with Bonferroni alpha divided across all predeclared looks in the 500-observation horizon. This fixed-look comparator assumes the binomial null is valid within each window; the Bonferroni correction addresses repeated looks, not arbitrary temporal dependence. A static-reuse baseline never monitors and therefore never detects drift.

## Protocol

- 2,000 independent synthetic audit streams per scenario, seed 20261030.
- Horizon 500; shift begins after the first 100 audits.
- Stable control: issuer violation probability 1% throughout.
- Abrupt shift: 1% to 5%, and separately 1% to 10%.
- Gradual shift: linear increase from 1% to 10% over 100 observations after the change point, then remains at 10%.
- Same stream passed to each monitor within a replicate.
- Report false alarms before change, post-change detection probability, misses, and conditional detection-delay distribution. Monte Carlo rate intervals are Wilson 95% intervals.

## Results

### Stable 1% control

| Monitor | False alarms | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| Mixture e-process | 5/2,000 | 0.25% | 0.107%–0.584% |
| Rolling binomial + Bonferroni | 7/2,000 | 0.35% | 0.170%–0.721% |
| Static reuse | 0/2,000 | 0% | 0%–0.192% |

Both active monitors stayed below the 1% nominal false-alarm level in this finite simulation. This is not a proof of their false-alarm guarantees; those come from the respective mathematical assumptions.

### Abrupt shift from 1% to 5%

| Monitor | Post-change detection | Wilson 95% interval | Mean delay | P95 delay |
|---|---:|---:|---:|---:|
| Mixture e-process | 1,897/2,000 (94.85%) | 93.79%–95.74% | 181.9 | 347.4 |
| Rolling binomial + Bonferroni | 1,419/2,000 (70.95%) | 68.92%–72.90% | 184.0 | 375.0 |
| Static reuse | 0/2,000 (0%) | 0%–0.192% | — | — |

The e-process increased detection by 23.9 percentage points at this shift size, with similar conditional mean delay and lower P95 delay. This is a synthetic result for this alternative grid and horizon, not general superiority.

### Abrupt shift from 1% to 10%

| Monitor | Post-change detection | Wilson 95% interval | Mean delay | P95 delay |
|---|---:|---:|---:|---:|
| Mixture e-process | 1,994/2,000 (99.70%) | 99.35%–99.86% | 73.0 | 146 |
| Rolling binomial + Bonferroni | 1,998/2,000 (99.90%) | 99.64%–99.97% | 75.1 | 175 |
| Static reuse | 0/2,000 (0%) | 0%–0.192% | — | — |

Both active monitors detect nearly all large shifts within this horizon. The rolling comparator had slightly higher observed detection probability, while the e-process had lower mean and P95 conditional delay.

### Gradual shift from 1% to 10%

| Monitor | Post-change detection | Wilson 95% interval | Mean delay | P95 delay |
|---|---:|---:|---:|---:|
| Mixture e-process | 1,995/2,000 (99.75%) | 99.42%–99.89% | 127.9 | 199 |
| Rolling binomial + Bonferroni | 1,998/2,000 (99.90%) | 99.64%–99.97% | 132.6 | 225 |
| Static reuse | 0/2,000 (0%) | 0%–0.192% | — | — |

## Interpretation

The mixture e-process is promising as an assumption-source monitor: it out-detected the rolling-window comparator under the small 1%→5% shift, while matching it closely under larger/gradual shifts. The results also expose a sensitivity trade-off: at 1% nominal alpha the mixture false-alarm rate was 0.25% in the stable simulation, and finite-simulation uncertainty remains material.

The key architectural shift is from a static signed attestation to a **time-qualified assurance lease**:

- bind each assurance to issuer identity, risk scope, epoch, audit protocol and expiry;
- consume independently verified audit outcomes in a persistent sequential monitor;
- allocate a global false-alarm budget across issuers and epochs before monitoring begins;
- quarantine/revoke dependent certificates when the monitor crosses its threshold;
- require requalification under a new epoch before restoring trust;
- never reset the process or recycle alpha silently after an alarm.

For a countable family of issuer/epoch monitors, one simple global allocation is `alpha_{i,e} = alpha_total * w_i * 6/(pi^2 e^2)`, where issuer weights sum to one and the epoch weights also sum to one. This is one conservative proposal; actual system topology and repeated-use semantics must be made explicit. Resetting a process for every batch without alpha accounting would invalidate the global false-alarm claim.

## Direct SOTA/industry comparison still open

The closest published comparator is Timans et al., **On Continuous Monitoring of Risk Violations under Unknown Shift**, UAI 2025. It already applies testing-by-betting to continuous risk-violation monitoring and provides false-alarm control under its stated conditions. It has a public implementation repository with e-process variants and benchmark workloads:

- Paper: https://proceedings.mlr.press/v286/timans25a.html
- Official code: https://github.com/alextimans/risk-monitor

This PR does **not** reproduce or execute that official method. The current benchmark is a transparent Bernoulli special-case mixture e-process versus a simple Bonferroni rolling test. It would be incorrect to claim this beats the UAI 2025 method or SOTA. The official repository is AGPL-3.0; keep it as an external evaluator unless a separate integration/license review authorizes otherwise.

The UAI 2026 online martingale PAC-Bayes calibration work is another relevant baseline for dependent time-series calibration, with a predict-then-update protocol and model-agnostic online certificates:

https://proceedings.mlr.press/v337/huang26b.html

The next comparator gate is to reproduce Timans et al.'s risk-monitor on its own published tasks, then apply both monitors to paired issuer-reliability streams with identical audit labels, false-alarm budgets and shift schedules. Also test autocorrelation, clustered audit failures, multiple issuers, repeated epochs, label delays and adversarially targeted audit sampling.

## Reproduction and validation

Files:

- `validation/assurance_drift_monitor.py` — e-process, exact rolling-binomial baseline, scope/epoch preflight, benchmark runner.
- `validation/test_assurance_drift_monitor.py` — nine regression tests, including a direct one-step supermartingale expectation check for the composite null.
- `validation/ASSURANCE_DRIFT_MONITOR_RESULTS_2026-10-30_2000.json` — full results and Wilson intervals.

Local validation: **9/9 tests passed**. The exact source/test files committed to this branch match the locally executed files by Git blob SHA. The benchmark was run twice with the same seed; JSON output was byte-identical.

- Source SHA-256: `05c5b318a86f7e983c86d16def37382a6f4c9b479b48d47a428152ceacaf46b1`
- Test SHA-256: `b8a61ded0693788618e83e629908cca7cd5019650ba89b7f7a07ef87459947f1`
- Results SHA-256: `d367512ebb1994da44199e73000a22fffefb7a410a23f55959bad983628b660e`
- Raw JSON Git blob SHA: `ef92d260f2d05c89798d13fb6f2e6d253e842f26`

Reproduction:

```sh
PYTHONPATH=validation python -m unittest -v test_assurance_drift_monitor.py
PYTHONPATH=validation python assurance_drift_monitor.py \
  --output /tmp/assurance-drift-results.json --repeats 2000 --horizon 500
```

## Limitations and next gates

- This tests synthetic Bernoulli audit outcomes, not real issuer reliability or actual contamination data.
- The e-process requires a valid conditional upper bound on audit-violation probability and correctly adjudicated observations. It cannot prove that the labels are independent, unbiased or representative.
- The rolling baseline relies on an IID binomial model within each look window.
- No detector can guarantee discovery of every arbitrary or observationally hidden shift.
- The static baseline is a negative control, not an active statistical comparator.
- This is not a reproduction of the UAI 2025 official monitor, not production Rust integration, and not a SOTA claim.

Next gates: direct official-code reproduction; dependence/clustered failures; predictable multiple-issuer and multi-epoch alpha allocation; audit-label selection bias; change-detection delay versus false-alarm Pareto frontiers; then a typed Rust implementation with explicit lease/quarantine/requalification states.

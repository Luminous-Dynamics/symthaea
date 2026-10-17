# ENV-MOBILITY-010: Event-Aware Sensing and Energy-Bounded Survey Planning

**Status:** Research protocol and synthetic benchmark specification; no field deployment or measured energy saving is claimed.  
**Parent work:** ENV-MOBILITY-001 through -009.  
**Core question:** Can a survey network collect more decision-relevant evidence per joule without missing brief, localized, or safety-relevant events?

## 1. Paradigm shift: measure when information matters

Fixed-rate sampling is simple and reproducible, but spends the same sensing and communications budget during stable periods and during fast-changing conditions. Purely sparse sampling can save energy while missing short events. The design target is therefore not "sample as rarely as possible" or "maximize samples"; it is **maximize event-relevant information subject to explicit energy, latency, uncertainty, and coverage constraints**.

Recent research makes this tradeoff concrete. A 2026 study proposes sentinel-assisted anomaly monitoring with sparse prediction-driven sampling and local neighborhood wake-up; an earlier water-quality study evaluated data-driven adaptive sampling for energy saving while preserving reconstruction quality. These are promising baselines to test, not evidence that our design will outperform them:
- 2026 sentinel-assisted adaptive sampling preprint: https://arxiv.org/abs/2607.15235
- Adaptive sampling for automated water-quality monitoring: https://pmc.ncbi.nlm.nih.gov/articles/PMC5713192/
- 2026 adaptive sampling for irrigation monitoring: https://doi.org/10.1016/j.inpa.2026.07.009

## 2. Proposed architecture: sentinel + adaptive neighborhood + evidence-preserving backfill

Use three roles, assignable to one device or distributed across nodes:

1. **Sentinel:** one or more low-cost nodes sample at a conservative, validated minimum rate and run simple local change detectors. They are not ground truth; they are early-warning instruments with their own health and uncertainty status.
2. **Adaptive nodes:** use prediction uncertainty, residual energy, sensor health, and mission priority to choose among a small, auditable set of sampling rates. On a sentinel alert or local anomaly, temporarily increase sampling and wake only a bounded neighborhood.
3. **Evidence gateway:** stores raw observations, device time and clock-quality metadata, policy version, sample-rate transitions, alert provenance, and missingness. It must distinguish measured observations from interpolated/reconstructed values. Reconstruction never counts as a captured event.

Use a deterministic policy as the baseline before attempting reinforcement learning. A policy must have a hard energy reserve, maximum silence interval, local fail-safe rate, bounded wake-up fan-out, and explicit behavior when communication is unavailable. Learning may optimize within these constraints but must not remove them.

## 3. Benchmark corpus

Build a frozen, deterministic synthetic corpus before comparing policies. Generate multiple seeds, but keep train/tuning seeds disjoint from evaluation seeds. Include:
- smooth diurnal trends and slow drift;
- periodic and correlated spatial fields;
- abrupt local spikes, short-duration events, and moving fronts;
- rare but high-consequence threshold crossings;
- sensor bias, stuck-at values, missing packets, clock drift, and false alerts;
- depleted batteries, uncertain energy harvesting, cloudy periods, and communication partitions;
- node failure, sentinel failure, adversarial-but-plausible outliers, and sensor replacement;
- sites with different baselines so a policy cannot pass by memorizing one site.

For every event, record onset, duration, affected region, magnitude, mission severity, and the minimum evidence window needed to count as detected. Make event labels available only to the evaluator, not to the policy at decision time.

Retain negative controls: a no-event trace with noise, a sensor fault that resembles a real event, and a real event that is below a naive global threshold but locally meaningful.

## 4. Required baseline policies

Compare all policies on the same traces, energy model, starting states, and seeds:

A. Fixed high-rate sampling (quality/energy upper reference).  
B. Fixed low-rate sampling (energy lower reference).  
C. Threshold-triggered adaptive sampling.  
D. Prediction-uncertainty sampling (e.g. Kalman innovation / residual threshold).  
E. Sentinel plus bounded-neighborhood wake-up.  
F. Energy-aware adaptive policy with explicit reserve and harvest forecast.  
G. Optional learned policy, only after the deterministic baselines are implemented.

No policy may get an unfairly easier sensor model, better event labels, or different energy assumptions. Report both per-policy metrics and paired differences by seed/site/event class.

## 5. Metrics and proposed gates

Targets below are initial engineering targets to ratify for a particular mission; they are not results.

| Dimension | Metric | Initial gate |
|---|---|---|
| Event detection | Recall for mission-critical events within a predeclared detection latency | >= 99% on the frozen critical-event suite; report confidence interval and misses |
| Latency | Event onset to first valid, provenance-bearing observation | <= mission-specific bound; never count reconstructed data as detection |
| False alarms | False wakeups per node-day and false critical alerts per site-day | Report by event-free and fault-injected scenarios; set operational limit before test |
| Energy | Joules per valid observation and per correctly detected event | Beat fixed-high baseline on energy while preserving the critical-event gate |
| Energy autonomy | Fraction of simulated mission horizon without energy exhaustion | >= 99% for declared harvest/load traces; report worst-case scenarios separately |
| Reconstruction | MAE/RMSE on deliberately withheld samples, by event class | Must meet declared limit; separately report short-event miss rate |
| Coverage | Fraction of spatial-temporal mission cells with valid observations | Report mean and worst decile; do not let aggregate hide blind regions |
| Data integrity | 100% of reconstructed points marked as reconstructed; raw values immutable | Hard gate |
| Fault response | Detect stuck sensor, clock drift, and sentinel loss; enter safe fallback | 100% on frozen critical fault suite |
| Bounded coordination | Neighborhood wake-up fan-out and radio bytes per event | Hard upper bound set before test; no network-wide storm |
| Reproducibility | Exact same outputs for deterministic policy + seed + corpus version | 100% identical decisions and score outputs |
| Compute budget | Policy runtime, RAM, and energy per decision on target MCU | Fit the frozen device budget with explicit margin |

The 99% critical-event target is intentionally demanding and may be statistically unsupported by a small corpus. Use confidence intervals and enough independent events; if the corpus cannot support the claimed bound, report that limitation rather than asserting the target passed. Report miss counts and worst-case event classes, not only an aggregate score.

## 6. Score without hiding safety failures

Use a lexicographic acceptance order, not one weighted score that lets energy savings compensate for missed critical events:

1. Data provenance and event labels valid.
2. Critical-event recall and latency gates met.
3. Fault detection and safe fallback gates met.
4. Coverage and reconstruction limits met.
5. Only then compare energy, radio traffic, compute, and maintenance cost.

Any critical-event miss, unmarked reconstructed value, or failure to fall back after sentinel loss is a safety-gate failure regardless of energy savings. Publish Pareto frontiers for energy vs latency vs coverage only among candidates that satisfy the hard gates.

## 7. Energy model must be calibrated

Do not treat simulated joules as measured hardware. Initially model:
- sensor active/sleep power and wake-up stabilization;
- MCU inference/decision energy;
- radio transmit/receive and retry costs;
- storage writes and flash wear where relevant;
- power conversion losses, battery capacity derating, and reserve;
- solar/harvest uncertainty and site-specific time series.

Record model assumptions and perform sensitivity analysis. Before making hardware efficiency claims, measure representative state currents and radio energy on the actual device with a calibrated power monitor. Compare predicted and measured energy, then update the model with a versioned record.

## 8. Interoperability and evidence contract

Map observations to established concepts instead of inventing a closed data model. OGC SensorThings API 1.1 provides interoperable concepts for Things, Sensors, ObservedProperties, Datastreams, Observations, and time/location metadata: https://www.ogc.org/standards/sensorthings/

Keep project-specific fields for raw-data digest, clock quality, sample-policy revision, uncertainty, reconstruction status, calibration state, event-trigger provenance, and energy state. Document the mapping and export a standards-compatible representation; do not claim conformance without running applicable conformance tests.

Every evaluation artifact should include:
- corpus version and SHA-256;
- policy source revision and configuration;
- simulator/model revision;
- random seed and initial state;
- raw decision trace and observation trace;
- event-level detections/misses and reason codes;
- energy model and sensitivity range;
- independent recomputation output.

## 9. Implementation sequence

1. Implement a deterministic simulator and fixed-rate baseline first; no learned policy required.
2. Add event labels and negative controls, then test metric computation independently.
3. Add threshold and uncertainty-driven baselines.
4. Add sentinel wake-up with a strict fan-out cap, reserve floor, and timeout.
5. Run paired seeds and event-level analyses; preserve all misses.
6. Validate energy model against bench measurements before translating simulated savings to real deployment.
7. Only then evaluate learning methods under the same frozen interface and constraints.

**Stop condition:** If critical-event recall/latency or safe-fallback gates fail, do not recommend the policy for deployment even if it saves energy. If energy is only simulated, label all efficiency results simulated.

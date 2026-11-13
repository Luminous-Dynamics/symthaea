# UAP asynchronous trajectory reconstruction and calibration observability

Date: 2026-11-07  
Status: research contract / benchmark design; no estimator implementation or physical result claimed  
Parent: [UAP-ASYNC-001 #11029](https://github.com/Luminous-Dynamics/symthaea/issues/11029)  
Programme: [Causal–Temporal–Geometric Physics PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Executive thesis

The next useful advance for the Observatory is not a more elaborate object classifier. It is a system that knows when trajectory reconstruction is identifiable.

The current bearing triangulation primitive is deliberately narrower: it expects a common observation epoch, known sensor poses, and independent angular errors. The extension defined here treats time alignment, platform pose, boresight, target dynamics and measurement noise as coupled uncertain quantities.

The governing principle is:

> Observability must be established before estimation confidence is reported.

A low residual, a finite covariance, or a plausible-looking 3D track is not enough. The data-only information matrix must be inspected for gauge freedoms and weak directions. Priors may regularize the problem, but prior-dominated parameters must not be labelled as measured.

This is a measurement-science workstream. It does not identify UAPs, infer non-human origin, or propose a propulsion mechanism.

## Research basis and industry comparison

Current sensor-fusion practice already recognizes spatial and temporal calibration as first-order concerns. The 2025 review by Lyu et al. surveys camera/LiDAR/IMU extrinsic and temporal-offset calibration, including clock-source differences, triggering, transmission delay, jitter, drift and motion compensation (Sensors 25, 5409; DOI: 10.3390/s25175409).

A particularly relevant 2026 result is Weiss et al., *Joint Self-Calibration of Receiver Geometry, Timing, and Target Positions for Multistatic Radar Autofocus* (Sensors, 2026-08-05; DOI: 10.3390/s26154954). It jointly estimates receiver-position corrections, clock biases and target positions, then analyzes the remaining geometric null spaces. The important lesson is not merely that joint estimation is possible; it is that the anchor geometry determines which parameters can be identified at all.

Cheng et al., *Accuracy Analysis and Optimal Configuration for AOA Localization* (IEEE Signal Processing Letters, published 2026-09-07; DOI: 10.1109/LSP.2026.3731095), derives optimal 2D/3D bearings-only sensor configurations from Fisher-information eigenstructure and checks the Cramér–Rao bound with Monte Carlo simulations.

Earlier bearings-only tracking research also addresses asynchronous multi-sensor observations and uncertain sensor positions (e.g. *Bearings only multi-sensor maneuvering target tracking*, Systems & Control Letters 57, 2008, DOI: 10.1016/j.sysconle.2007.08.008).

Therefore, the proposed contribution is **not** “the first asynchronous triangulation” or “the first joint calibration.” It is an evidence-qualified, gauge-aware benchmark and reusable software contract that:
- makes observability a reported output rather than an implicit optimizer assumption;
- propagates nuisance-parameter uncertainty into trajectory uncertainty;
- tests explicit abstention on known unobservable configurations;
- binds calibration, timing, geometry, observations and results into reproducible evidence;
- compares against strong calibration-aware and conventional baselines under matched information.

Production autonomy stacks use hardware synchronization, calibration procedures, factor-graph/state-estimation methods and sensor-health checks. Symthaea should not claim to replace those systems based on a synthetic prototype. Its opportunity is to make the epistemic boundary and failure modes unusually explicit and independently auditable.

## Mathematical model

### Clock model

For sensor i, let tau_ik be the timestamp emitted by its local clock for observation k. Map it to a declared reference clock using

    t_ik = a_i * tau_ik + b_i + epsilon_ik

where:
- a_i is the clock-rate scale (drift relative to the reference);
- b_i is the clock offset;
- epsilon_ik is timestamp jitter/measurement uncertainty.

The reference clock is anchored with a_0 = 1 and b_0 = 0. These are gauge conventions, not discoveries from the observations. Preserve raw local timestamps and clock-domain identity in the evidence record. Never overwrite them with corrected values.

### Sensor and target state

Let x(t) describe the target trajectory, and let T_WS_i(t) describe sensor i's world pose. The observation model is

    z_ik = h(x(t_ik), T_WS_i(t_ik), T_SB_i, beta_i) + v_ik

where T_SB_i is the sensor-body extrinsic transform, beta_i is a boresight/calibration correction, and v_ik is measurement noise. For a bearing sensor, h returns azimuth/elevation in the sensor frame; for radar, the model may additionally include range or time-of-flight.

Target process uncertainty, platform-pose uncertainty, clock uncertainty, extrinsic uncertainty and measurement covariance remain distinct state/noise sources. Cross-sensor and cross-time covariance must be retained when common calibration, GNSS, timing or environmental errors induce correlation.

A corrected epoch is not a complete physical observation by itself: platform pose must be evaluated or interpolated at that same uncertain epoch, with interpolation uncertainty accounted for.

### Gauge-aware observability

Let theta contain the target trajectory parameters and calibration nuisance parameters. For whitened residual vector r(theta), compute the data-only Jacobian and information matrix:

    J = d r / d theta
    F_data = J^T J

If residuals are not already whitened, use the full covariance Sigma:

    F_data = J^T Sigma^{-1} J

Report the singular spectrum of the whitened Jacobian (or eigen-spectrum of F_data) **before adding priors**. Fix and document coordinate/time gauges first. Scale parameters by declared physical reference scales before comparing singular values; raw condition numbers across metres, radians and seconds are otherwise misleading.

If prior precision P^{-1} is used, report it separately:

    F_posterior = F_data + P^{-1}

A posterior may be numerically invertible while the observations themselves remain uninformative in one or more directions. Such a result is prior-dominated, not data-identified.

When calibration nuisance parameters eta are included, derive the target-state marginal covariance from the full joint system or the correct Schur complement. Do not report the inverse target-only block while ignoring target/calibration cross-correlation.

### Expected gauge and degeneracy cases

At minimum, expose these cases as named fixtures:

1. **Global time-origin gauge:** a common shift of all clocks and the target reference epoch is not observable without an external anchor.
2. **Global frame gauge:** if all sensor and target poses are free, a global rigid transform is not measured by internal bearings alone.
3. **Stationary target:** target motion cannot identify clock offsets through temporal change.
4. **No platform excitation:** some clock/pose/bearing-bias combinations become indistinguishable.
5. **Insufficient bearing diversity:** parallel or nearly parallel lines produce weak range/depth observability.
6. **Collinear/coplanar anchors:** may preserve continuous or discrete ambiguities even where a naive rank/condition check looks acceptable.
7. **Clock/boresight coupling:** a time shift can resemble an angular bias over a narrow time window or approximately constant bearing rate.
8. **Unknown baseline scale:** bearing-only measurements cannot recover absolute scale if all geometric baselines are also free and no metric anchor is supplied.

An observability status must distinguish at least:
- Identifiable under declared gauge constraints;
- Weakly observable (with weak parameter combinations);
- Unobservable (with null directions);
- Prior-dominated;
- Invalid input / insufficient evidence.

## Benchmark design

### Track A — deterministic synthetic oracle

Generate known 2D and 3D target trajectories, sensor platform trajectories, sensor-local clocks, clock drift, boresight offsets, pose perturbations, measurement noise and dropout/outlier tapes.

Hold out complete motion families and sensor geometries, not only random sample IDs. Freeze seed lists, generator version, physical units, noise covariance and train/qualification split digests.

Use target regimes:
- stationary;
- constant velocity;
- constant acceleration;
- maneuvering/curved trajectory;
- near-constant angular-rate segments.

Use sensor regimes:
- static array with well-conditioned baselines;
- moving sensors with diverse excitation;
- weakly excited or stationary platforms;
- collinear and coplanar configurations;
- mixed sensor rates and irregular sample intervals.

Corruptions:
- known constant clock offsets;
- clock-rate drift;
- timestamp jitter and delayed packet arrival;
- boresight error and drift;
- platform-position and attitude error;
- common-mode calibration errors;
- missing observations, duplicate sequence IDs and outliers.

### Track B — baseline comparison

All estimators must receive the same observation set, platform metadata, prior information and declared compute budget.

Compare:
1. exact-time/exact-calibration oracle (upper bound);
2. simultaneous-only triangulation (discarding unmatched asynchronous observations);
3. nearest-timestamp pairing;
4. interpolation to a common epoch with fixed calibration;
5. independent per-sensor tracking followed by late fusion;
6. joint batch MAP / factor-graph estimation of target state and clock offsets;
7. robust joint estimation with pose/boresight nuisance states and correlated uncertainty;
8. support-aware estimator that abstains on weakly identified parameter combinations.

Do not call nearest-timestamp pairing a principled synchronization method. Its error must be measured as a baseline.

### Track C — real instrument / public dataset

Do not start this track until licensing, calibration metadata, local clock semantics, sensor-pose provenance, synchronization uncertainty and ground-truth quality are pinned. Retain incomplete and uncalibrated records as explicit evidence-quality outcomes; do not silently discard them.

## Metrics

Report each metric by geometry, motion family, sample-rate regime and corruption level.

### Reconstruction accuracy
- position and velocity RMSE;
- acceleration error where acceleration is in the estimator's state model;
- angular residual and range residual;
- worst-decile and worst-case errors;
- trajectory likelihood / negative log-likelihood where its probabilistic assumptions hold.

### Calibration recovery
- clock offset error (seconds);
- clock-rate/drift error (ppm);
- boresight error (radians);
- sensor position error (metres);
- attitude/extrinsic error (radians and metres).

### Uncertainty calibration
- NEES and NIS with their assumptions stated;
- empirical 50%, 90%, 95% and 99% confidence-region coverage;
- interval/ellipsoid width;
- calibration error conditional on reported uncertainty;
- correlated-error coverage.

### Observability and integrity
- data-only numerical rank after declared gauge fixing;
- scaled minimum singular value and condition number;
- weak/null parameter combinations;
- prior-dominance ratio or an equivalent disclosed diagnostic;
- false-confident trajectory rate on unobservable fixtures;
- correct abstention rate and coverage-versus-error frontier;
- false track rate, track fragmentation and track continuity under dropout.

### Operational cost
- buffering/reorder delay;
- latency p50/p95;
- memory and compute;
- recovery time after sensor dropout;
- cost/accuracy frontier as sensor count and observation rate scale.

No single aggregate score may hide a false-confident result on a degenerate geometry.

## Prospective acceptance gates

These are proposed qualification rules, not achieved results:

- Exact fixtures recover known states within tolerances derived from the numerical implementation and measurement noise.
- Gauge fixtures demonstrate their expected null directions before and after anchoring.
- Known unobservable cases return a typed abstention or prior-dominated status; they do not publish a falsely precise data-only covariance.
- Joint uncertainty includes clock, pose and boresight contributions and their cross-correlations.
- Reordering packets without changing timestamps or observation identity cannot change the physical estimate.
- Duplicate IDs, clock-domain mixing, invalid clock maps and expired calibrations are rejected or preserved as explicit input-quality failures.
- Coverage is evaluated on sealed motion-family and geometry holdouts.
- Baselines use identical information and resource budgets.
- SOTA claims require a pinned method/dataset release, reproducible implementation, matched inputs and compute, and independent reproduction.

Numerical pass thresholds should be frozen after a pilot that establishes the null distributions and minimum detectable effects; they must not be selected after examining qualification labels.

## Proposed implementation sequence

1. Keep the existing common-epoch triangulator as a deterministic geometry oracle.
2. Add a gauge-aware observability analyzer, independently tested against small known Jacobians and hand-constructed null spaces.
3. Add a dependency-free synthetic event generator with local clocks and exact platform trajectories.
4. Add a minimal 2D constant-velocity joint clock-offset estimator.
5. Add target acceleration and clock-rate drift.
6. Add 3D bearing geometry, platform-pose uncertainty and boresight corrections.
7. Add correlated noise and robust residual models only after the independent reference agrees.
8. Bind exact observation manifests, generator versions, calibrations, seeds and estimator configuration into science-evidence receipts.
9. Compare robust target-region sensor placement against expected-FIM, A-optimal, E-optimal and worst-case CRLB objectives with timing/pose nuisance parameters included.
10. Promote to real instrument data only after independent calibration/rights gates and human review.

## Relationship to the rest of CTGP

- **Causal sets:** preserve event identity and partial ordering; sensor arrival order is not physical causal order.
- **Time-crystal analysis:** temporal/spectral anomalies are evaluated only after timing calibration and aliasing controls; periodicity is not a propulsion signature.
- **Metric geometry / GravCraft:** reconstructed trajectories may be compared with GR geodesic or Newtonian predictions, but no exotic metric is inferred from a trajectory residual alone.
- **Chain Theory:** candidate predictions are compared only where a quantitative observable is actually derived; the fine-structure arithmetic does not supply a macroscopic trajectory mechanism.
- **Science evidence:** every conclusion retains source data, calibration lineage, clock model, uncertainty, alternative hypotheses and an explicit unresolved state.

## Claim ceiling

A synthetic pass proves only that the algorithm and observability contract behave correctly for the declared generator and gauge conventions. It does not prove field accuracy, identify a UAP, establish anomalous dynamics, demonstrate new physics, or validate propulsion. If measurements do not identify a trajectory, the correct output is **unresolved due to insufficient observability**.

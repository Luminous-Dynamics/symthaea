// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Failure-mode scenarios (Phase 2 of `SPACE_AUTOMATION_PLAN_2026-07-06.md`).
//!
//! Per the crate-done criteria: benchmarks that expose failure modes, not
//! just happy paths. Each scenario runs a scripted baseline controller
//! against `SimpleOrbitalSimulator` and reports a distinguishable outcome.

use crate::simulator::{OrbitalPhysicsSimulator, SimpleOrbitalSimulator};
use crate::types::{OrbitalCommand, OrbitalConfig};
use orbital_mechanics::coordinates::wgs84::{A as EARTH_RADIUS_KM, MU};

/// Orbital period (s) for a circular orbit at the given altitude (km).
pub fn circular_period_s(altitude_km: f64) -> f64 {
    2.0 * std::f64::consts::PI * ((EARTH_RADIUS_KM + altitude_km).powi(3) / MU).sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StationKeepingOutcome {
    /// Held within `tolerance_km` of `target_altitude_km` for the whole run.
    Success,
    /// Altitude fell below `failure_altitude_km` — the corrective burns
    /// couldn't outrun drag decay (whether because of a too-small burn, too
    /// little budget, or just entering the loop already below tolerance).
    Decayed,
    /// Delta-v budget ran out while altitude was still below the tolerance
    /// band — decay would have continued, but is reported distinctly from
    /// `Decayed` because it's actionable differently (buy more propellant
    /// budget vs. redesign the control law).
    PropellantExhausted,
}

#[derive(Debug, Clone)]
pub struct StationKeepingConfig {
    /// Orbit configuration to run (altitude, drag, delta-v budget all come
    /// from here — see `OrbitalConfig::initial_altitude_km` etc.).
    pub orbital: OrbitalConfig,
    /// Maintain altitude within [target - tolerance, target + tolerance].
    /// Target defaults to `orbital.initial_altitude_km`.
    pub tolerance_km: f64,
    /// Hard failure floor — below this, the scenario reports `Decayed`
    /// regardless of remaining budget.
    pub failure_altitude_km: f64,
    /// Magnitude of each corrective prograde (along-velocity) burn, m/s.
    pub burn_mps: f32,
    /// How many orbital periods to attempt to hold station for.
    pub num_orbits: f64,
    /// Integration step, s. Needs to be finer than the 1.0s used by
    /// simulator.rs's whole-orbit sanity tests — at 1.0s, symplectic
    /// Euler's own numerical wobble (~1.2km within 335s, no-drag) swamps
    /// the km-scale tolerance this scenario checks against. 0.1s default
    /// keeps that noise well below `tolerance_km`.
    pub dt_s: f64,
}

impl StationKeepingConfig {
    pub fn new(orbital: OrbitalConfig) -> Self {
        Self {
            // NOTE (2026-07-07): tolerance_km/dt_s were originally 1.0/1.0,
            // which turned out to be tighter than symplectic Euler's own
            // per-step numerical wobble at 1s resolution (measured ~1.2km
            // of "altitude" noise within 335s at dt=1.0s, with drag
            // disabled entirely — i.e. pure integration artifact, not
            // physics). See scenarios::tests for the empirical values that
            // motivated these numbers; if you change dt_s, re-check the
            // no-drag test's min/max spread before trusting a tighter
            // tolerance_km against it.
            tolerance_km: 5.0,
            failure_altitude_km: orbital.initial_altitude_km - 30.0,
            burn_mps: 0.5,
            num_orbits: 3.0,
            dt_s: 0.1,
            orbital,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StationKeepingResult {
    pub outcome: StationKeepingOutcome,
    pub steps_run: usize,
    pub min_altitude_km: f64,
    pub max_altitude_km: f64,
    pub delta_v_used_m_s: f64,
    pub burns_fired: usize,
}

/// Scripted baseline: a simple bang-bang controller. If altitude has
/// dropped below the tolerance band's floor, fire a fixed prograde burn;
/// otherwise coast. This is the "beat this" baseline the plan calls for —
/// a learned controller should do at least this well on delta-v efficiency.
pub fn run_station_keeping(config: &StationKeepingConfig) -> StationKeepingResult {
    let mut sim = SimpleOrbitalSimulator::with_config(config.orbital.clone());
    let target = config.orbital.initial_altitude_km;
    let total_steps =
        (config.num_orbits * circular_period_s(target) / config.dt_s).round() as usize;

    let mut min_alt = f64::MAX;
    let mut max_alt = f64::MIN;
    let mut burns_fired = 0usize;

    for step in 0..total_steps {
        let alt = sim.state().altitude_km();
        min_alt = min_alt.min(alt);
        max_alt = max_alt.max(alt);

        if alt < config.failure_altitude_km {
            return StationKeepingResult {
                outcome: StationKeepingOutcome::Decayed,
                steps_run: step,
                min_altitude_km: min_alt,
                max_altitude_km: max_alt,
                delta_v_used_m_s: sim.state().delta_v_used_m_s,
                burns_fired,
            };
        }

        let remaining = sim
            .state()
            .delta_v_remaining_m_s(config.orbital.delta_v_budget_m_s);
        let below_tolerance = alt < target - config.tolerance_km;
        if below_tolerance && remaining <= 0.0 {
            return StationKeepingResult {
                outcome: StationKeepingOutcome::PropellantExhausted,
                steps_run: step,
                min_altitude_km: min_alt,
                max_altitude_km: max_alt,
                delta_v_used_m_s: sim.state().delta_v_used_m_s,
                burns_fired,
            };
        }

        let mut cmd = OrbitalCommand::zero();
        if below_tolerance {
            let v = sim.state().velocity_km_s;
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed > 1e-9 {
                for i in 0..3 {
                    cmd.translational_burn_mps[i] = (v[i] / speed) as f32 * config.burn_mps;
                }
                burns_fired += 1;
            }
        }
        sim.step(&cmd, config.dt_s);
    }

    StationKeepingResult {
        outcome: StationKeepingOutcome::Success,
        steps_run: total_steps,
        min_altitude_km: min_alt,
        max_altitude_km: max_alt,
        delta_v_used_m_s: sim.state().delta_v_used_m_s,
        burns_fired,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DesaturationOutcome {
    /// Held within `pointing_tolerance_rad_s` for the whole task, wheels
    /// never (or successfully) desaturated.
    Success,
    /// Bus angular rate exceeded the pointing tolerance — a saturated wheel
    /// lost authority (see `simulator.rs`'s saturation fix) and the
    /// scripted controller didn't desaturate in time.
    PointingViolated,
    /// A wheel crossed the desaturation trigger threshold but the
    /// desaturation propellant budget was already gone.
    DesaturationExhausted,
}

#[derive(Debug, Clone)]
pub struct MomentumDesaturationConfig {
    /// Orbit/reaction-wheel/desaturation-budget configuration.
    pub orbital: OrbitalConfig,
    /// Sustained joint-0 torque commanded every step — models "the arm
    /// doing its job" as a continuous task disturbance that keeps loading
    /// the wheel, rather than a one-off impulse.
    pub task_joint_torque: f32,
    /// Fire desaturation on an axis once |wheel momentum| on that axis
    /// exceeds this fraction of `reaction_wheel_capacity`.
    pub desaturation_threshold_fraction: f64,
    /// Desaturation torque magnitude commanded per axis when triggered, N·m.
    pub desaturation_rate_nm: f32,
    /// Max allowed |angular_velocity| component before reporting
    /// `PointingViolated`.
    pub pointing_tolerance_rad_s: f64,
    pub num_steps: usize,
    pub dt_s: f64,
}

impl MomentumDesaturationConfig {
    pub fn new(orbital: OrbitalConfig) -> Self {
        Self {
            // NOTE (2026-07-07): task_joint_torque needs to satisfy TWO
            // independent bounds, both against the wheel's rate law
            // `wt = (-50 * angular_velocity).clamp(-5.0, 5.0)`:
            //  1. max_joint_torques[0] * this value < 5.0 N·m, or the wheel
            //     can't even fully counter the disturbance while completely
            //     unsaturated (at 1.0 = 20 N·m, pointing violates almost
            //     immediately -- ~1300 steps -- for the wrong reason).
            //  2. The proportional law's own NOMINAL steady-state tracking
            //     error is v_ss ~= react/50 (solving wt_ss = -react for
            //     equilibrium, then v_ss = wt_ss/50). At react=4 N·m
            //     (task_joint_torque=0.2), v_ss ~= 0.08 rad/s -- which
            //     ALREADY exceeds a 0.05 rad/s pointing tolerance even with
            //     an unsaturated wheel working exactly as designed! That
            //     was a real tuning bug: the "pointing violated" outcome
            //     was firing every time regardless of saturation, because
            //     the tolerance was tighter than the control law's own
            //     nominal tracking error, not because of the momentum-
            //     saturation failure mode this scenario is meant to expose.
            // 0.075 (1.5 N·m) keeps v_ss ~= 0.03 rad/s, comfortably under
            // the 0.05 tolerance while unsaturated -- so a pointing
            // violation, when it happens, is actually caused by saturation.
            task_joint_torque: 0.075,
            desaturation_threshold_fraction: 0.5,
            desaturation_rate_nm: 100.0,
            pointing_tolerance_rad_s: 0.05,
            num_steps: 400_000,
            dt_s: 0.01,
            orbital,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MomentumDesaturationResult {
    pub outcome: DesaturationOutcome,
    pub steps_run: usize,
    pub max_angular_rate_rad_s: f64,
    pub desaturation_used_nms: f64,
    pub desaturation_steps: usize,
}

/// Scripted baseline: sustain the task disturbance every step; whenever any
/// wheel axis crosses `desaturation_threshold_fraction` of capacity, fire a
/// fixed desaturation torque on that axis until it's back under threshold.
/// This is the "beat this" baseline — a learned controller should desaturate
/// more propellant-efficiently (e.g. proportional rather than bang-bang) or
/// tolerate a tighter pointing budget.
pub fn run_momentum_desaturation(
    config: &MomentumDesaturationConfig,
) -> MomentumDesaturationResult {
    let mut sim = SimpleOrbitalSimulator::with_config(config.orbital.clone());
    let capacity = sim.config().reaction_wheel_capacity;
    let budget = sim.config().desaturation_budget_nms;
    let threshold = capacity * config.desaturation_threshold_fraction;

    let mut max_rate = 0.0f64;
    let mut desaturation_steps = 0usize;

    for step in 0..config.num_steps {
        let rate = sim
            .state()
            .spacecraft_angular_velocity
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        max_rate = max_rate.max(rate);

        if rate > config.pointing_tolerance_rad_s {
            return MomentumDesaturationResult {
                outcome: DesaturationOutcome::PointingViolated,
                steps_run: step,
                max_angular_rate_rad_s: max_rate,
                desaturation_used_nms: sim.state().desaturation_used_nms,
                desaturation_steps,
            };
        }

        let rwm = sim.reaction_wheel_momentum();
        let needs_desat = rwm.iter().any(|m| m.abs() > threshold);
        let remaining_budget = sim.state().desaturation_remaining_nms(budget);
        if needs_desat && remaining_budget <= 0.0 {
            return MomentumDesaturationResult {
                outcome: DesaturationOutcome::DesaturationExhausted,
                steps_run: step,
                max_angular_rate_rad_s: max_rate,
                desaturation_used_nms: sim.state().desaturation_used_nms,
                desaturation_steps,
            };
        }

        let mut cmd = OrbitalCommand::zero();
        cmd.joint_torques[0] = config.task_joint_torque;
        if needs_desat {
            for a in 0..3 {
                if rwm[a].abs() > threshold {
                    cmd.desaturation_torque_nm[a] = config.desaturation_rate_nm;
                }
            }
            desaturation_steps += 1;
        }
        sim.step(&cmd, config.dt_s);
    }

    MomentumDesaturationResult {
        outcome: DesaturationOutcome::Success,
        steps_run: config.num_steps,
        max_angular_rate_rad_s: max_rate,
        desaturation_used_nms: sim.state().desaturation_used_nms,
        desaturation_steps,
    }
}

use orbital_mechanics::conjunction::{ConjunctionAnalyzer, RiskLevel};
use orbital_mechanics::state::{
    DataSource, OrbitalState as TrackedObjectState, StateVector as LibStateVector,
};

/// Fixed deterministic epoch for conjunction assessments — the analyzer
/// only needs `primary.epoch == secondary.epoch` for a same-epoch
/// assessment, not real wall-clock time, so a constant avoids
/// nondeterminism across test runs.
fn fixed_epoch() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap()
}

/// Unit vector perpendicular to `v` (cross-track direction). Avoidance
/// burns fire cross-track rather than prograde/retrograde so they grow
/// miss distance without much altitude/period change.
fn cross_track_direction(v: [f64; 3]) -> [f64; 3] {
    let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if v_mag < 1e-9 {
        return [1.0, 0.0, 0.0];
    }
    let v_hat = [v[0] / v_mag, v[1] / v_mag, v[2] / v_mag];
    let z = [0.0, 0.0, 1.0];
    let cross = [
        v_hat[1] * z[2] - v_hat[2] * z[1],
        v_hat[2] * z[0] - v_hat[0] * z[2],
        v_hat[0] * z[1] - v_hat[1] * z[0],
    ];
    let cross_mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if cross_mag < 1e-9 {
        [1.0, 0.0, 0.0]
    } else {
        [
            cross[0] / cross_mag,
            cross[1] / cross_mag,
            cross[2] / cross_mag,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConjunctionOutcome {
    /// Assessed risk was down to `safe_risk_level` or better by TCA.
    Success,
    /// Needed to maneuver (risk >= `maneuver_trigger`) but the delta-v
    /// budget was already gone.
    DeltaVExhausted,
    /// Reached TCA with risk still above `safe_risk_level` despite
    /// maneuvering the whole way -- the avoidance burn wasn't strong
    /// enough / didn't start early enough for the available budget.
    CollisionRiskAtTca,
}

#[derive(Debug, Clone)]
pub struct ConjunctionAvoidanceConfig {
    /// Orbit/delta-v-budget configuration for our own spacecraft.
    pub orbital: OrbitalConfig,
    /// Predicted miss distance (km) at TCA if we do NOT maneuver at all --
    /// defines a fixed, non-maneuvering secondary object's predicted TCA
    /// position, offset cross-track from our own coast trajectory by this
    /// amount.
    pub baseline_miss_distance_km: f64,
    /// Hard-body radius for risk assessment, meters. Default (20m in
    /// `ConjunctionAnalyzer`) caps achievable risk at `High` for any
    /// realistic miss distance via the no-covariance Pc fallback
    /// (pc = exp(-x²/2) * (hbr_km)², x = miss_km; max pc at miss=0 is only
    /// (0.02)² = 4e-4, under the 1e-3 Emergency threshold) -- 100m is
    /// large enough that `Emergency` is reachable at sub-km miss
    /// distances, which this scenario needs to exercise.
    pub hard_body_radius_m: f64,
    /// Time until predicted closest approach (TCA), s.
    pub time_to_tca_s: f64,
    /// Fire the avoidance burn once assessed risk reaches at least this
    /// level.
    pub maneuver_trigger: RiskLevel,
    /// Report `Success` only once assessed risk is at or below this level.
    pub safe_risk_level: RiskLevel,
    /// Avoidance burn magnitude commanded per step while triggered, m/s,
    /// fired cross-track.
    pub avoidance_burn_mps: f32,
    pub dt_s: f64,
}

impl ConjunctionAvoidanceConfig {
    pub fn new(orbital: OrbitalConfig) -> Self {
        Self {
            baseline_miss_distance_km: 1.0,
            hard_body_radius_m: 100.0,
            time_to_tca_s: 3600.0,
            maneuver_trigger: RiskLevel::High,
            safe_risk_level: RiskLevel::Medium,
            avoidance_burn_mps: 0.02,
            dt_s: 0.1,
            orbital,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConjunctionAvoidanceResult {
    pub outcome: ConjunctionOutcome,
    pub steps_run: usize,
    pub initial_risk: RiskLevel,
    pub final_risk: RiskLevel,
    pub final_miss_distance_km: f64,
    pub delta_v_used_m_s: f64,
    pub maneuver_steps: usize,
}

/// Scripted baseline: coast until assessed risk (projected forward to TCA
/// via a simple linear coast, the standard screening-stage simplification
/// -- full nonlinear propagation is only warranted for TCA refinement, not
/// initial risk triage) reaches `maneuver_trigger`, then fire a fixed
/// cross-track burn every step until risk drops back below it or the
/// delta-v budget runs out.
pub fn run_conjunction_avoidance(
    config: &ConjunctionAvoidanceConfig,
) -> ConjunctionAvoidanceResult {
    let mut sim = SimpleOrbitalSimulator::with_config(config.orbital.clone());
    let analyzer = ConjunctionAnalyzer::new().with_hbr(config.hard_body_radius_m);
    let epoch = fixed_epoch();

    // Fixed, non-maneuvering secondary: where it will be at TCA, defined as
    // our own coast-projected TCA position offset cross-track by the
    // configured baseline miss distance.
    let p0 = sim.state().position_km;
    let v0 = sim.state().velocity_km_s;
    let coast_at_tca = [
        p0[0] + v0[0] * config.time_to_tca_s,
        p0[1] + v0[1] * config.time_to_tca_s,
        p0[2] + v0[2] * config.time_to_tca_s,
    ];
    let cross0 = cross_track_direction(v0);
    let secondary_pos_at_tca = [
        coast_at_tca[0] + cross0[0] * config.baseline_miss_distance_km,
        coast_at_tca[1] + cross0[1] * config.baseline_miss_distance_km,
        coast_at_tca[2] + cross0[2] * config.baseline_miss_distance_km,
    ];
    let secondary_state = LibStateVector::new(
        secondary_pos_at_tca[0],
        secondary_pos_at_tca[1],
        secondary_pos_at_tca[2],
        v0[0],
        v0[1],
        v0[2],
    );
    let secondary = TrackedObjectState::new(99999, epoch, secondary_state, DataSource::SpaceTrack);

    let assess_risk_at = |sim: &SimpleOrbitalSimulator, time_remaining_s: f64| -> RiskLevel {
        let p = sim.state().position_km;
        let v = sim.state().velocity_km_s;
        let projected = [
            p[0] + v[0] * time_remaining_s,
            p[1] + v[1] * time_remaining_s,
            p[2] + v[2] * time_remaining_s,
        ];
        let primary_state =
            LibStateVector::new(projected[0], projected[1], projected[2], v[0], v[1], v[2]);
        let primary = TrackedObjectState::new(1, epoch, primary_state, DataSource::SpaceTrack);
        analyzer.assess(&primary, &secondary).risk_level
    };

    let initial_risk = assess_risk_at(&sim, config.time_to_tca_s);
    let mut time_remaining = config.time_to_tca_s;
    let mut maneuver_steps = 0usize;
    let mut step = 0usize;

    while time_remaining > 0.0 {
        let risk = assess_risk_at(&sim, time_remaining);
        let mut cmd = OrbitalCommand::zero();
        if risk >= config.maneuver_trigger {
            let remaining_budget = sim
                .state()
                .delta_v_remaining_m_s(config.orbital.delta_v_budget_m_s);
            if remaining_budget <= 0.0 {
                let final_risk = assess_risk_at(&sim, time_remaining);
                let final_miss = (0..3)
                    .map(|i| (sim.state().position_km[i] - secondary_pos_at_tca[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                return ConjunctionAvoidanceResult {
                    outcome: ConjunctionOutcome::DeltaVExhausted,
                    steps_run: step,
                    initial_risk,
                    final_risk,
                    final_miss_distance_km: final_miss,
                    delta_v_used_m_s: sim.state().delta_v_used_m_s,
                    maneuver_steps,
                };
            }
            let cross = cross_track_direction(sim.state().velocity_km_s);
            for i in 0..3 {
                cmd.translational_burn_mps[i] = cross[i] as f32 * config.avoidance_burn_mps;
            }
            maneuver_steps += 1;
        }
        sim.step(&cmd, config.dt_s);
        time_remaining -= config.dt_s;
        step += 1;
    }

    let final_risk = assess_risk_at(&sim, 0.0);
    let final_miss = (0..3)
        .map(|i| (sim.state().position_km[i] - secondary_pos_at_tca[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let outcome = if final_risk <= config.safe_risk_level {
        ConjunctionOutcome::Success
    } else {
        ConjunctionOutcome::CollisionRiskAtTca
    };

    ConjunctionAvoidanceResult {
        outcome,
        steps_run: step,
        initial_risk,
        final_risk,
        final_miss_distance_km: final_miss,
        delta_v_used_m_s: sim.state().delta_v_used_m_s,
        maneuver_steps,
    }
}

use orbital_mechanics::{RelativeState, mean_motion, propagate_cw};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendezvousOutcome {
    /// Reached the capture window (position AND velocity both within
    /// tolerance) before the delta-v budget or step budget ran out.
    Docked,
    /// Drifted outside the tapering approach corridor -- a real docking
    /// approach aborts here rather than risk an uncontrolled collision.
    AbortedCorridorViolation,
    /// Needed to correct course but the delta-v budget was gone.
    DeltaVExhausted,
    /// Ran out of allotted time without reaching the capture window (and
    /// without violating the corridor or budget) -- e.g. too slow a
    /// glideslope gain for the time allotted.
    TimedOut,
}

#[derive(Debug, Clone, Copy)]
pub struct RendezvousDockingConfig {
    /// Reference (target) orbit altitude, km -- determines mean motion.
    pub reference_altitude_km: f64,
    /// Initial relative state: chaser relative to target in the target's
    /// LVLH frame (radial, along-track, cross-track), meters / m-s.
    /// Default starts behind the target on the along-track (V-bar) axis,
    /// the standard final-approach starting configuration.
    pub initial_state: RelativeState,
    /// Corridor half-width (radial/cross-track) at the START of the
    /// approach (|y| = |initial along-track offset|), meters. Tapers
    /// linearly to zero at the target -- a docking approach must get
    /// straighter as it gets closer, not just stay within a fixed tube.
    pub corridor_half_width_m: f64,
    /// Proportional gain (1/s) driving along-track closing velocity
    /// toward zero displacement: desired_vy = -k_glide * y.
    pub glideslope_gain: f64,
    /// Proportional gain (1/s) correcting radial/cross-track drift back
    /// toward the corridor centerline.
    pub lateral_gain: f64,
    /// Capture window: docking succeeds once |relative position| is under
    /// this AND |relative velocity| is under `capture_speed_mps` --
    /// position alone isn't enough (a fast flyby through the capture
    /// point is a miss, not a dock).
    pub capture_radius_m: f64,
    pub capture_speed_mps: f64,
    /// Total delta-v budget for the approach, m/s. Proximity-ops scale
    /// (single-digit m/s), NOT the whole-orbit delta_v_budget_m_s used by
    /// station-keeping/conjunction-avoidance scenarios elsewhere in this
    /// module -- deliberately a separate, scenario-local budget rather
    /// than threading OrbitalConfig through a relative-motion-only
    /// scenario that doesn't otherwise touch the absolute-state simulator.
    pub delta_v_budget_m_s: f64,
    pub dt_s: f64,
    pub max_steps: usize,
}

impl Default for RendezvousDockingConfig {
    fn default() -> Self {
        Self {
            reference_altitude_km: 400.0,
            initial_state: RelativeState {
                position_m: [0.0, -500.0, 0.0],
                velocity_mps: [0.0, 0.0, 0.0],
            },
            corridor_half_width_m: 50.0,
            glideslope_gain: 0.005,
            lateral_gain: 0.01,
            capture_radius_m: 2.0,
            capture_speed_mps: 0.1,
            delta_v_budget_m_s: 10.0,
            dt_s: 1.0,
            max_steps: 3000,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RendezvousDockingResult {
    pub outcome: RendezvousOutcome,
    pub steps_run: usize,
    pub final_distance_m: f64,
    pub final_speed_mps: f64,
    pub delta_v_used_m_s: f64,
}

/// Allowed lateral (radial/cross-track) deviation at along-track distance
/// `abs_y_m`, given the corridor's starting half-width at `start_abs_y_m`:
/// tapers linearly to zero at the target.
fn corridor_limit_m(abs_y_m: f64, start_abs_y_m: f64, half_width_m: f64) -> f64 {
    if start_abs_y_m < 1e-9 {
        return half_width_m;
    }
    half_width_m * (abs_y_m / start_abs_y_m).clamp(0.0, 1.0)
}

/// Scripted baseline: proportional glideslope control. Every step, compute
/// the velocity we'd need to be closing at (proportional to remaining
/// along-track distance, plus lateral correction toward the corridor
/// centerline), spend delta-v to correct our actual velocity toward it
/// (budget-clamped), then propagate the resulting relative state one step
/// via the closed-form CW solution (exact regardless of step size -- no
/// integration-noise tuning concern the way the absolute two-body+drag
/// simulator has).
pub fn run_rendezvous_docking(config: &RendezvousDockingConfig) -> RendezvousDockingResult {
    let n = mean_motion(orbital_mechanics::coordinates::wgs84::A + config.reference_altitude_km);
    let mut state = config.initial_state;
    let mut delta_v_used = 0.0f64;
    let start_abs_y = config.initial_state.position_m[1].abs();

    for step in 0..config.max_steps {
        let dist = state.distance_m();
        let speed = state.speed_mps();
        if dist < config.capture_radius_m && speed < config.capture_speed_mps {
            return RendezvousDockingResult {
                outcome: RendezvousOutcome::Docked,
                steps_run: step,
                final_distance_m: dist,
                final_speed_mps: speed,
                delta_v_used_m_s: delta_v_used,
            };
        }

        let limit = corridor_limit_m(
            state.position_m[1].abs(),
            start_abs_y,
            config.corridor_half_width_m,
        );
        if state.position_m[0].abs() > limit || state.position_m[2].abs() > limit {
            return RendezvousDockingResult {
                outcome: RendezvousOutcome::AbortedCorridorViolation,
                steps_run: step,
                final_distance_m: dist,
                final_speed_mps: speed,
                delta_v_used_m_s: delta_v_used,
            };
        }

        let desired_v = [
            -config.lateral_gain * state.position_m[0],
            -config.glideslope_gain * state.position_m[1],
            -config.lateral_gain * state.position_m[2],
        ];
        let needed: [f64; 3] = std::array::from_fn(|i| desired_v[i] - state.velocity_mps[i]);
        let needed_mag =
            (needed[0] * needed[0] + needed[1] * needed[1] + needed[2] * needed[2]).sqrt();
        let remaining_budget = (config.delta_v_budget_m_s - delta_v_used).max(0.0);
        if needed_mag > 1e-9 && remaining_budget <= 0.0 {
            return RendezvousDockingResult {
                outcome: RendezvousOutcome::DeltaVExhausted,
                steps_run: step,
                final_distance_m: dist,
                final_speed_mps: speed,
                delta_v_used_m_s: delta_v_used,
            };
        }
        let applied_mag = needed_mag.min(remaining_budget);
        let scale = if needed_mag > 1e-9 {
            applied_mag / needed_mag
        } else {
            0.0
        };
        for i in 0..3 {
            state.velocity_mps[i] += needed[i] * scale;
        }
        delta_v_used += applied_mag;

        state = propagate_cw(&state, n, config.dt_s);
    }

    RendezvousDockingResult {
        outcome: RendezvousOutcome::TimedOut,
        steps_run: config.max_steps,
        final_distance_m: state.distance_m(),
        final_speed_mps: state.speed_mps(),
        delta_v_used_m_s: delta_v_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_period_matches_400km_orbit_test() {
        // simulator.rs's test_orbit_roughly_closes_after_one_period hardcodes
        // this same formula inline — cross-check they agree.
        let p = circular_period_s(400.0);
        assert!(
            (p - 5554.0).abs() < 5.0,
            "expected ~5554s period at 400km, got {p}"
        );
    }

    #[test]
    fn test_no_drag_no_burns_needed() {
        // Zero drag area -> altitude shouldn't decay meaningfully -> the
        // scripted controller should never need to fire.
        let mut cfg = StationKeepingConfig::new(OrbitalConfig::default());
        cfg.orbital.drag_area_m2 = 0.0;
        let r = run_station_keeping(&cfg);
        assert_eq!(
            r.outcome,
            StationKeepingOutcome::Success,
            "min={} max={} steps={}",
            r.min_altitude_km,
            r.max_altitude_km,
            r.steps_run
        );
        assert_eq!(r.burns_fired, 0);
        assert_eq!(r.delta_v_used_m_s, 0.0);
    }

    #[test]
    fn test_high_drag_low_orbit_triggers_corrective_burns() {
        // A much lower, draggier orbit should decay enough within a few
        // periods that the scripted controller actually has to intervene —
        // proving the scenario isn't vacuously "always Success, never
        // burns" regardless of physics.
        let mut orbital = OrbitalConfig::default();
        orbital.initial_altitude_km = 180.0;
        orbital.drag_area_m2 = 10.0;
        orbital.drag_mass_kg = 50.0;
        orbital.delta_v_budget_m_s = 500.0;
        let mut cfg = StationKeepingConfig::new(orbital);
        cfg.tolerance_km = 0.5;
        cfg.failure_altitude_km = 100.0;
        cfg.num_orbits = 2.0;

        let r = run_station_keeping(&cfg);
        assert!(
            r.burns_fired > 0,
            "expected the scripted controller to fire corrective burns under strong drag"
        );
        assert!(r.delta_v_used_m_s > 0.0);
    }

    #[test]
    fn test_insufficient_budget_reports_propellant_exhausted() {
        // Same aggressive decay scenario, but with essentially no delta-v
        // budget -- the scripted controller wants to burn but can't, and
        // decay must eventually cross the failure floor or exhaust budget
        // detection first.
        let mut orbital = OrbitalConfig::default();
        orbital.initial_altitude_km = 180.0;
        orbital.drag_area_m2 = 10.0;
        orbital.drag_mass_kg = 50.0;
        orbital.delta_v_budget_m_s = 0.0; // no propellant at all
        let mut cfg = StationKeepingConfig::new(orbital);
        cfg.tolerance_km = 0.5;
        cfg.failure_altitude_km = 100.0;
        cfg.num_orbits = 5.0;

        let r = run_station_keeping(&cfg);
        assert!(
            matches!(
                r.outcome,
                StationKeepingOutcome::PropellantExhausted | StationKeepingOutcome::Decayed
            ),
            "expected a failure outcome with zero delta-v budget under drag, got {:?}",
            r.outcome
        );
    }

    #[test]
    fn test_default_config_holds_station_easily() {
        // At the default 400km/light-drag config, station-keeping over a
        // few orbits should be trivially achievable (sanity check that the
        // scenario harness itself isn't broken/inverted).
        let cfg = StationKeepingConfig::new(OrbitalConfig::default());
        let r = run_station_keeping(&cfg);
        assert_eq!(r.outcome, StationKeepingOutcome::Success);
    }

    #[test]
    fn test_desaturation_default_holds_pointing_and_actually_fires() {
        // The scripted baseline should keep pointing within tolerance AND
        // genuinely have to desaturate at least once — proving this isn't
        // a vacuous "always Success, never desaturates" scenario.
        //
        // The bang-bang baseline limit-cycles (load past 50% threshold,
        // dump back down, repeat) over this scenario's long duration --
        // OrbitalConfig's global default desaturation_budget_nms (100, a
        // "dump a saturated wheel twice" placeholder for occasional events)
        // isn't enough propellant for that many repeated dumps. Bump it
        // here to represent a mission that actually carries enough RCS
        // propellant for the task, rather than changing the global default
        // (which other tests deliberately override down to 0 to test
        // exhaustion) -- this is the config the "Success" case needs.
        let mut orbital = OrbitalConfig::default();
        orbital.desaturation_budget_nms = 10_000.0;
        let cfg = MomentumDesaturationConfig::new(orbital);
        let r = run_momentum_desaturation(&cfg);
        assert_eq!(
            r.outcome,
            DesaturationOutcome::Success,
            "max_rate={} steps={}",
            r.max_angular_rate_rad_s,
            r.steps_run
        );
        assert!(
            r.desaturation_steps > 0,
            "expected the sustained task disturbance to require desaturation \
             at least once within {} steps",
            cfg.num_steps
        );
    }

    #[test]
    fn test_no_desaturation_capability_eventually_violates_pointing() {
        // If the agent can never desaturate (rate=0), sustained disturbance
        // must eventually saturate the wheel; per the saturation-authority
        // fix in simulator.rs, a saturated wheel then lets disturbance
        // through, and pointing should be violated -- proving the failure
        // mode is real and reachable, not just theoretical.
        let mut cfg = MomentumDesaturationConfig::new(OrbitalConfig::default());
        cfg.desaturation_rate_nm = 0.0;
        cfg.num_steps = 1_500_000;
        let r = run_momentum_desaturation(&cfg);
        assert_eq!(
            r.outcome,
            DesaturationOutcome::PointingViolated,
            "max_rate={} steps={}",
            r.max_angular_rate_rad_s,
            r.steps_run
        );
    }

    #[test]
    fn test_zero_desaturation_budget_reports_exhausted() {
        let mut orbital = OrbitalConfig::default();
        orbital.desaturation_budget_nms = 0.0;
        let cfg = MomentumDesaturationConfig::new(orbital);
        let r = run_momentum_desaturation(&cfg);
        assert_eq!(
            r.outcome,
            DesaturationOutcome::DesaturationExhausted,
            "max_rate={} steps={}",
            r.max_angular_rate_rad_s,
            r.steps_run
        );
    }

    #[test]
    fn test_distant_object_stays_safe_no_maneuver_needed() {
        // A 50km predicted miss is nowhere near a collision risk -- the
        // scripted baseline should never need to fire.
        let mut cfg = ConjunctionAvoidanceConfig::new(OrbitalConfig::default());
        cfg.baseline_miss_distance_km = 50.0;
        let r = run_conjunction_avoidance(&cfg);
        assert_eq!(
            r.outcome,
            ConjunctionOutcome::Success,
            "initial={:?} final={:?} miss={}",
            r.initial_risk,
            r.final_risk,
            r.final_miss_distance_km
        );
        assert_eq!(r.maneuver_steps, 0);
        assert_eq!(r.delta_v_used_m_s, 0.0);
    }

    #[test]
    fn test_close_approach_starts_high_risk() {
        // Sanity check the risk-assessment wiring itself: the default
        // 1km/100m-HBR configuration must actually assess as High or worse
        // at t=TCA-time_to_tca, or the rest of this scenario is vacuous.
        let cfg = ConjunctionAvoidanceConfig::new(OrbitalConfig::default());
        let r = run_conjunction_avoidance(&cfg);
        assert!(
            r.initial_risk >= RiskLevel::High,
            "expected initial risk >= High for a 1km miss at 100m HBR, got {:?}",
            r.initial_risk
        );
    }

    #[test]
    fn test_close_approach_triggers_avoidance_maneuver() {
        let cfg = ConjunctionAvoidanceConfig::new(OrbitalConfig::default());
        let r = run_conjunction_avoidance(&cfg);
        assert!(
            r.maneuver_steps > 0,
            "expected the scripted controller to fire avoidance burns for a \
             close-approach conjunction"
        );
        assert!(r.delta_v_used_m_s > 0.0);
    }

    #[test]
    fn test_zero_budget_reports_delta_v_exhausted() {
        let mut orbital = OrbitalConfig::default();
        orbital.delta_v_budget_m_s = 0.0;
        let cfg = ConjunctionAvoidanceConfig::new(orbital);
        let r = run_conjunction_avoidance(&cfg);
        assert_eq!(
            r.outcome,
            ConjunctionOutcome::DeltaVExhausted,
            "initial={:?} final={:?} miss={}",
            r.initial_risk,
            r.final_risk,
            r.final_miss_distance_km
        );
    }

    #[test]
    fn test_rendezvous_default_config_docks_successfully() {
        let cfg = RendezvousDockingConfig::default();
        let r = run_rendezvous_docking(&cfg);
        assert_eq!(
            r.outcome,
            RendezvousOutcome::Docked,
            "steps={} dist={} speed={} dv={}",
            r.steps_run,
            r.final_distance_m,
            r.final_speed_mps,
            r.delta_v_used_m_s
        );
        assert!(r.delta_v_used_m_s > 0.0);
    }

    #[test]
    fn test_rendezvous_zero_budget_reports_delta_v_exhausted() {
        let mut cfg = RendezvousDockingConfig::default();
        cfg.delta_v_budget_m_s = 0.0;
        let r = run_rendezvous_docking(&cfg);
        assert_eq!(
            r.outcome,
            RendezvousOutcome::DeltaVExhausted,
            "steps={} dist={}",
            r.steps_run,
            r.final_distance_m
        );
    }

    #[test]
    fn test_rendezvous_lateral_start_outside_zero_corridor_aborts_immediately() {
        // Zero corridor half-width with a nonzero initial lateral offset
        // means we start already outside the (zero-width) corridor --
        // proves the abort path is reachable, not dead code.
        let mut cfg = RendezvousDockingConfig::default();
        cfg.corridor_half_width_m = 0.0;
        cfg.initial_state.position_m[0] = 10.0; // radial offset
        let r = run_rendezvous_docking(&cfg);
        assert_eq!(r.outcome, RendezvousOutcome::AbortedCorridorViolation);
        assert_eq!(r.steps_run, 0);
    }

    #[test]
    fn test_rendezvous_too_few_steps_times_out() {
        let mut cfg = RendezvousDockingConfig::default();
        cfg.max_steps = 5; // nowhere near enough to close a 500m approach
        let r = run_rendezvous_docking(&cfg);
        assert_eq!(
            r.outcome,
            RendezvousOutcome::TimedOut,
            "dist={} speed={}",
            r.final_distance_m,
            r.final_speed_mps
        );
    }

    #[test]
    fn test_rendezvous_corridor_limit_tapers_to_zero_at_target() {
        assert_eq!(corridor_limit_m(0.0, 500.0, 50.0), 0.0);
        assert_eq!(corridor_limit_m(500.0, 500.0, 50.0), 50.0);
        assert!((corridor_limit_m(250.0, 500.0, 50.0) - 25.0).abs() < 1e-9);
    }
}

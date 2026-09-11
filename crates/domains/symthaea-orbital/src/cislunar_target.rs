// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LL-004 moving-target cislunar rendezvous study primitives.
//!
//! This module deliberately consumes explicit inertial state/frame provenance
//! from `cislunar_oracle`. It does not convert a lunar surface site into an
//! inertial release state, fetch ephemerides, command guidance, or authorize a
//! launch. The first dynamics model is Moon-centered two-body propagation for
//! Phase-0 trade studies only.

use serde::{Deserialize, Serialize};

use crate::cislunar_oracle::{EphemerisSource, FrameContract, StateVectorKm};

const SECONDS_PER_DAY: f64 = 86_400.0;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CislunarTargetClass {
    LowLunarOrbitCatcher,
    EarthMoonL1Vicinity,
    EarthMoonL2Vicinity,
    Nrho,
    Depot,
    CatcherTug,
    DomainSpecific(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimedStateSample {
    pub epoch_jd: f64,
    pub state: StateVectorKm,
}

impl TimedStateSample {
    pub fn is_well_formed(&self) -> bool {
        self.epoch_jd.is_finite() && self.epoch_jd > 0.0 && self.state.is_well_formed()
    }
}

/// Provenance-bound moving-target track.
///
/// v0 uses cubic Hermite interpolation between position/velocity samples and
/// never extrapolates outside the declared sample interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CislunarTargetTrack {
    pub target_id: String,
    pub class: CislunarTargetClass,
    pub frame: FrameContract,
    pub source: EphemerisSource,
    pub samples: Vec<TimedStateSample>,
    /// Maximum permitted interval between adjacent reference samples.
    pub max_gap_s: f64,
    pub interpolation_ref: String,
    pub evidence_refs: Vec<String>,
}

impl CislunarTargetTrack {
    pub fn is_well_formed(&self) -> bool {
        if self.target_id.trim().is_empty()
            || !self.frame.is_well_formed()
            || !self.source.is_well_formed()
            || self.samples.len() < 2
            || !self.max_gap_s.is_finite()
            || self.max_gap_s <= 0.0
            || self.interpolation_ref.trim().is_empty()
            || self.samples.iter().any(|sample| !sample.is_well_formed())
            || self
                .evidence_refs
                .iter()
                .any(|reference| reference.trim().is_empty())
        {
            return false;
        }

        self.samples.windows(2).all(|pair| {
            let dt_days = pair[1].epoch_jd - pair[0].epoch_jd;
            dt_days.is_finite()
                && dt_days > 0.0
                && dt_days * SECONDS_PER_DAY <= self.max_gap_s
        })
    }

    pub fn start_epoch_jd(&self) -> Option<f64> {
        self.is_well_formed()
            .then(|| self.samples.first().expect("validated nonempty").epoch_jd)
    }

    pub fn end_epoch_jd(&self) -> Option<f64> {
        self.is_well_formed()
            .then(|| self.samples.last().expect("validated nonempty").epoch_jd)
    }

    /// Interpolate a target state without extrapolation.
    pub fn state_at(&self, epoch_jd: f64) -> Result<StateVectorKm, EncounterError> {
        if !self.is_well_formed() || !epoch_jd.is_finite() || epoch_jd <= 0.0 {
            return Err(EncounterError::InvalidTargetTrack);
        }
        let first = self.samples.first().expect("validated nonempty");
        let last = self.samples.last().expect("validated nonempty");
        let eps = 1.0e-12;
        if epoch_jd < first.epoch_jd - eps || epoch_jd > last.epoch_jd + eps {
            return Err(EncounterError::TargetEpochOutOfRange);
        }
        if (epoch_jd - first.epoch_jd).abs() <= eps {
            return Ok(first.state);
        }
        if (epoch_jd - last.epoch_jd).abs() <= eps {
            return Ok(last.state);
        }

        let upper = self.samples.partition_point(|sample| sample.epoch_jd < epoch_jd);
        if upper == 0 || upper >= self.samples.len() {
            return Err(EncounterError::TargetEpochOutOfRange);
        }
        let a = self.samples[upper - 1];
        let b = self.samples[upper];
        hermite_state(a, b, epoch_jd)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InertialReleaseState {
    pub release_id: String,
    pub epoch_jd: f64,
    pub frame: FrameContract,
    pub state: StateVectorKm,
    pub dynamics_ref: String,
    pub constants_ref: String,
    pub evidence_refs: Vec<String>,
}

impl InertialReleaseState {
    pub fn is_well_formed(&self) -> bool {
        !self.release_id.trim().is_empty()
            && self.epoch_jd.is_finite()
            && self.epoch_jd > 0.0
            && self.frame.is_well_formed()
            && self.state.is_well_formed()
            && !self.dynamics_ref.trim().is_empty()
            && !self.constants_ref.trim().is_empty()
            && self
                .evidence_refs
                .iter()
                .all(|reference| !reference.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncounterStudyInput {
    pub release: InertialReleaseState,
    pub target: CislunarTargetTrack,
    /// Must exactly match the declared frame origin in v0.
    pub central_body_origin: String,
    pub central_body_mu_km3_s2: f64,
    pub window_start_s: f64,
    pub window_end_s: f64,
    pub scan_step_s: f64,
    pub integration_step_s: f64,
    pub refinement_iterations: u32,
    /// Optional study-only distance threshold used to distinguish a nearby
    /// passage from `NoEncounterWithinWindow`.
    pub encounter_threshold_km: Option<f64>,
    pub study_ref: String,
}

impl EncounterStudyInput {
    pub fn validate(&self) -> Result<(), EncounterError> {
        if !self.release.is_well_formed() {
            return Err(EncounterError::InvalidReleaseState);
        }
        if !self.target.is_well_formed() {
            return Err(EncounterError::InvalidTargetTrack);
        }
        if self.release.frame != self.target.frame {
            return Err(EncounterError::FrameMismatch);
        }
        if self.central_body_origin.trim().is_empty()
            || self.release.frame.origin != self.central_body_origin
        {
            return Err(EncounterError::CentralBodyMismatch);
        }
        let scalars = [
            self.central_body_mu_km3_s2,
            self.window_start_s,
            self.window_end_s,
            self.scan_step_s,
            self.integration_step_s,
        ];
        if scalars.iter().any(|value| !value.is_finite()) {
            return Err(EncounterError::NonFiniteInput);
        }
        if self.central_body_mu_km3_s2 <= 0.0
            || self.window_start_s < 0.0
            || self.window_end_s <= self.window_start_s
            || self.scan_step_s <= 0.0
            || self.integration_step_s <= 0.0
            || self.refinement_iterations > 100
            || self.study_ref.trim().is_empty()
        {
            return Err(EncounterError::InvalidInput);
        }
        if let Some(threshold) = self.encounter_threshold_km {
            if !threshold.is_finite() || threshold < 0.0 {
                return Err(EncounterError::InvalidInput);
            }
        }

        let start_epoch = self.release.epoch_jd + self.window_start_s / SECONDS_PER_DAY;
        let end_epoch = self.release.epoch_jd + self.window_end_s / SECONDS_PER_DAY;
        if start_epoch < self.target.start_epoch_jd().ok_or(EncounterError::InvalidTargetTrack)?
            || end_epoch > self.target.end_epoch_jd().ok_or(EncounterError::InvalidTargetTrack)?
        {
            return Err(EncounterError::TargetEpochOutOfRange);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncounterOutcome {
    ClosestApproach,
    NoEncounterWithinWindow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncounterResult {
    pub outcome: EncounterOutcome,
    pub epoch_jd: f64,
    pub seconds_after_release: f64,
    pub miss_distance_km: f64,
    pub relative_speed_km_s: f64,
    pub pod_state: StateVectorKm,
    pub target_state: StateVectorKm,
    pub release_id: String,
    pub target_id: String,
    pub frame: FrameContract,
    pub dynamics_ref: String,
    pub constants_ref: String,
    pub target_source: EphemerisSource,
    pub interpolation_ref: String,
    pub study_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncounterError {
    NonFiniteInput,
    InvalidInput,
    InvalidReleaseState,
    InvalidTargetTrack,
    TargetEpochOutOfRange,
    FrameMismatch,
    CentralBodyMismatch,
    NumericalFailure,
}

/// Find the minimum separation from a moving target over the declared window.
///
/// The result is a Phase-0 study output only. It is not a catcher intercept,
/// rendezvous solution, guidance command, or launch authorization.
pub fn closest_approach(input: &EncounterStudyInput) -> Result<EncounterResult, EncounterError> {
    input.validate()?;

    let mut best_t = input.window_start_s;
    let mut best_distance = f64::INFINITY;
    let mut t = input.window_start_s;
    while t <= input.window_end_s + 1.0e-12 {
        let (distance, _) = separation_at(input, t)?;
        if distance < best_distance {
            best_distance = distance;
            best_t = t;
        }
        t += input.scan_step_s;
    }
    if best_t < input.window_end_s {
        let (distance, _) = separation_at(input, input.window_end_s)?;
        if distance < best_distance {
            best_t = input.window_end_s;
        }
    }

    if input.refinement_iterations > 0 {
        let lo = (best_t - input.scan_step_s).max(input.window_start_s);
        let hi = (best_t + input.scan_step_s).min(input.window_end_s);
        best_t = golden_section_minimum(input, lo, hi, input.refinement_iterations)?;
    }

    let (_, relative_speed) = separation_at(input, best_t)?;
    let pod_state = pod_state_at(input, best_t)?;
    let epoch_jd = input.release.epoch_jd + best_t / SECONDS_PER_DAY;
    let target_state = input.target.state_at(epoch_jd)?;
    let miss_distance_km = norm(sub(pod_state.position_km, target_state.position_km));
    let outcome = match input.encounter_threshold_km {
        Some(threshold) if miss_distance_km > threshold => EncounterOutcome::NoEncounterWithinWindow,
        _ => EncounterOutcome::ClosestApproach,
    };

    Ok(EncounterResult {
        outcome,
        epoch_jd,
        seconds_after_release: best_t,
        miss_distance_km,
        relative_speed_km_s: relative_speed,
        pod_state,
        target_state,
        release_id: input.release.release_id.clone(),
        target_id: input.target.target_id.clone(),
        frame: input.release.frame.clone(),
        dynamics_ref: input.release.dynamics_ref.clone(),
        constants_ref: input.release.constants_ref.clone(),
        target_source: input.target.source.clone(),
        interpolation_ref: input.target.interpolation_ref.clone(),
        study_ref: input.study_ref.clone(),
    })
}

fn golden_section_minimum(
    input: &EncounterStudyInput,
    mut lo: f64,
    mut hi: f64,
    iterations: u32,
) -> Result<f64, EncounterError> {
    if !(lo.is_finite() && hi.is_finite()) || hi < lo {
        return Err(EncounterError::InvalidInput);
    }
    if hi == lo {
        return Ok(lo);
    }
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut c = hi - (hi - lo) / phi;
    let mut d = lo + (hi - lo) / phi;
    let mut fc = separation_at(input, c)?.0;
    let mut fd = separation_at(input, d)?.0;
    for _ in 0..iterations {
        if fc <= fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - (hi - lo) / phi;
            fc = separation_at(input, c)?.0;
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + (hi - lo) / phi;
            fd = separation_at(input, d)?.0;
        }
    }
    Ok(0.5 * (lo + hi))
}

fn separation_at(input: &EncounterStudyInput, seconds_after_release: f64) -> Result<(f64, f64), EncounterError> {
    let pod = pod_state_at(input, seconds_after_release)?;
    let epoch_jd = input.release.epoch_jd + seconds_after_release / SECONDS_PER_DAY;
    let target = input.target.state_at(epoch_jd)?;
    let distance = norm(sub(pod.position_km, target.position_km));
    let relative_speed = norm(sub(pod.velocity_km_s, target.velocity_km_s));
    if !distance.is_finite() || !relative_speed.is_finite() {
        return Err(EncounterError::NumericalFailure);
    }
    Ok((distance, relative_speed))
}

fn pod_state_at(input: &EncounterStudyInput, seconds_after_release: f64) -> Result<StateVectorKm, EncounterError> {
    if !seconds_after_release.is_finite() || seconds_after_release < 0.0 {
        return Err(EncounterError::InvalidInput);
    }
    let mut position = input.release.state.position_km;
    let mut velocity = input.release.state.velocity_km_s;
    let mut elapsed = 0.0;
    while elapsed < seconds_after_release {
        let step = input.integration_step_s.min(seconds_after_release - elapsed);
        (position, velocity) = rk4(position, velocity, step, input.central_body_mu_km3_s2)?;
        elapsed += step;
    }
    Ok(StateVectorKm {
        position_km: position,
        velocity_km_s: velocity,
    })
}

fn hermite_state(a: TimedStateSample, b: TimedStateSample, epoch_jd: f64) -> Result<StateVectorKm, EncounterError> {
    let dt_s = (b.epoch_jd - a.epoch_jd) * SECONDS_PER_DAY;
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EncounterError::InvalidTargetTrack);
    }
    let u = ((epoch_jd - a.epoch_jd) * SECONDS_PER_DAY / dt_s).clamp(0.0, 1.0);
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    let dh00 = 6.0 * u2 - 6.0 * u;
    let dh10 = 3.0 * u2 - 4.0 * u + 1.0;
    let dh01 = -6.0 * u2 + 6.0 * u;
    let dh11 = 3.0 * u2 - 2.0 * u;

    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    for i in 0..3 {
        position[i] = h00 * a.state.position_km[i]
            + h10 * dt_s * a.state.velocity_km_s[i]
            + h01 * b.state.position_km[i]
            + h11 * dt_s * b.state.velocity_km_s[i];
        velocity[i] = (dh00 * a.state.position_km[i]
            + dh01 * b.state.position_km[i]) / dt_s
            + dh10 * a.state.velocity_km_s[i]
            + dh11 * b.state.velocity_km_s[i];
    }
    let state = StateVectorKm {
        position_km: position,
        velocity_km_s: velocity,
    };
    state
        .is_well_formed()
        .then_some(state)
        .ok_or(EncounterError::NumericalFailure)
}

fn rk4(
    position: [f64; 3],
    velocity: [f64; 3],
    dt_s: f64,
    mu_km3_s2: f64,
) -> Result<([f64; 3], [f64; 3]), EncounterError> {
    let k1r = velocity;
    let k1v = acceleration(position, mu_km3_s2)?;
    let k2r = add(velocity, scale(k1v, 0.5 * dt_s));
    let k2v = acceleration(add(position, scale(k1r, 0.5 * dt_s)), mu_km3_s2)?;
    let k3r = add(velocity, scale(k2v, 0.5 * dt_s));
    let k3v = acceleration(add(position, scale(k2r, 0.5 * dt_s)), mu_km3_s2)?;
    let k4r = add(velocity, scale(k3v, dt_s));
    let k4v = acceleration(add(position, scale(k3r, dt_s)), mu_km3_s2)?;

    let next_position = add(
        position,
        scale(add(add(k1r, scale(k2r, 2.0)), add(scale(k3r, 2.0), k4r)), dt_s / 6.0),
    );
    let next_velocity = add(
        velocity,
        scale(add(add(k1v, scale(k2v, 2.0)), add(scale(k3v, 2.0), k4v)), dt_s / 6.0),
    );
    if all_finite(next_position) && all_finite(next_velocity) {
        Ok((next_position, next_velocity))
    } else {
        Err(EncounterError::NumericalFailure)
    }
}

fn acceleration(position: [f64; 3], mu_km3_s2: f64) -> Result<[f64; 3], EncounterError> {
    let r = norm(position);
    if !r.is_finite() || r <= 0.0 {
        return Err(EncounterError::NumericalFailure);
    }
    Ok(scale(position, -mu_km3_s2 / r.powi(3)))
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], scalar: f64) -> [f64; 3] {
    [a[0] * scalar, a[1] * scalar, a[2] * scalar]
}

fn norm(value: [f64; 3]) -> f64 {
    value.into_iter().map(|component| component * component).sum::<f64>().sqrt()
}

fn all_finite(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cislunar_oracle::{TimeScale};

    fn frame(origin: &str) -> FrameContract {
        FrameContract {
            name: "synthetic-inertial".into(),
            origin: origin.into(),
            axes: "synthetic Cartesian inertial axes".into(),
            time_scale: TimeScale::Tdb,
            position_unit: "km".into(),
            velocity_unit: "km/s".into(),
            derivative_convention: "inertial derivative".into(),
            contract_ref: "ll004-synthetic-frame-v0".into(),
        }
    }

    fn source() -> EphemerisSource {
        EphemerisSource {
            provider: "synthetic".into(),
            product: "analytic-target".into(),
            version: "v0".into(),
            configuration_ref: "ll004-fixture".into(),
        }
    }

    #[test]
    fn hermite_reproduces_constant_velocity_track() {
        let epoch = 2_460_000.5;
        let track = CislunarTargetTrack {
            target_id: "linear".into(),
            class: CislunarTargetClass::CatcherTug,
            frame: frame("Moon"),
            source: source(),
            samples: vec![
                TimedStateSample {
                    epoch_jd: epoch,
                    state: StateVectorKm { position_km: [10.0, 0.0, 0.0], velocity_km_s: [0.1, 0.0, 0.0] },
                },
                TimedStateSample {
                    epoch_jd: epoch + 100.0 / SECONDS_PER_DAY,
                    state: StateVectorKm { position_km: [20.0, 0.0, 0.0], velocity_km_s: [0.1, 0.0, 0.0] },
                },
            ],
            max_gap_s: 101.0,
            interpolation_ref: "cubic-hermite-v0".into(),
            evidence_refs: vec!["analytic".into()],
        };
        let mid = track.state_at(epoch + 50.0 / SECONDS_PER_DAY).unwrap();
        assert!((mid.position_km[0] - 15.0).abs() < 1.0e-8);
        assert!((mid.velocity_km_s[0] - 0.1).abs() < 1.0e-10);
    }

    #[test]
    fn frame_mismatch_fails_closed() {
        let epoch = 2_460_000.5;
        let input = EncounterStudyInput {
            release: InertialReleaseState {
                release_id: "pod".into(), epoch_jd: epoch, frame: frame("Moon"),
                state: StateVectorKm { position_km: [2_000.0, 0.0, 0.0], velocity_km_s: [0.0, 1.0, 0.0] },
                dynamics_ref: "two-body".into(), constants_ref: "synthetic".into(), evidence_refs: vec![],
            },
            target: CislunarTargetTrack {
                target_id: "target".into(), class: CislunarTargetClass::Depot, frame: frame("Earth"), source: source(),
                samples: vec![
                    TimedStateSample { epoch_jd: epoch, state: StateVectorKm { position_km: [2_100.0, 0.0, 0.0], velocity_km_s: [0.0, 0.0, 0.0] } },
                    TimedStateSample { epoch_jd: epoch + 100.0 / SECONDS_PER_DAY, state: StateVectorKm { position_km: [2_100.0, 0.0, 0.0], velocity_km_s: [0.0, 0.0, 0.0] } },
                ],
                max_gap_s: 101.0, interpolation_ref: "cubic-hermite-v0".into(), evidence_refs: vec![],
            },
            central_body_origin: "Moon".into(), central_body_mu_km3_s2: 4_902.8,
            window_start_s: 0.0, window_end_s: 90.0, scan_step_s: 5.0, integration_step_s: 0.5,
            refinement_iterations: 10, encounter_threshold_km: None, study_ref: "test".into(),
        };
        assert_eq!(input.validate(), Err(EncounterError::FrameMismatch));
    }

    #[test]
    fn threshold_preserves_no_encounter_outcome() {
        let epoch = 2_460_000.5;
        let moon = frame("Moon");
        let target = CislunarTargetTrack {
            target_id: "far-target".into(), class: CislunarTargetClass::Depot, frame: moon.clone(), source: source(),
            samples: vec![
                TimedStateSample { epoch_jd: epoch, state: StateVectorKm { position_km: [50_000.0, 0.0, 0.0], velocity_km_s: [0.0, 0.0, 0.0] } },
                TimedStateSample { epoch_jd: epoch + 120.0 / SECONDS_PER_DAY, state: StateVectorKm { position_km: [50_000.0, 0.0, 0.0], velocity_km_s: [0.0, 0.0, 0.0] } },
            ],
            max_gap_s: 121.0, interpolation_ref: "cubic-hermite-v0".into(), evidence_refs: vec![],
        };
        let input = EncounterStudyInput {
            release: InertialReleaseState {
                release_id: "pod".into(), epoch_jd: epoch, frame: moon,
                state: StateVectorKm { position_km: [2_000.0, 0.0, 0.0], velocity_km_s: [0.0, 1.0, 0.0] },
                dynamics_ref: "moon-two-body-v0".into(), constants_ref: "synthetic".into(), evidence_refs: vec![],
            },
            target,
            central_body_origin: "Moon".into(), central_body_mu_km3_s2: 4_902.8,
            window_start_s: 0.0, window_end_s: 100.0, scan_step_s: 10.0, integration_step_s: 1.0,
            refinement_iterations: 8, encounter_threshold_km: Some(1.0), study_ref: "far-target-study".into(),
        };
        let result = closest_approach(&input).unwrap();
        assert_eq!(result.outcome, EncounterOutcome::NoEncounterWithinWindow);
        assert!(result.miss_distance_km > 1.0);
    }
}

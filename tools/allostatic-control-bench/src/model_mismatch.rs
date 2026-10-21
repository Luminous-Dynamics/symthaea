//! Separate nominal and true one-step action models for mismatch falsification.
//!
//! This module is a deterministic synthetic harness. The nominal model is the
//! controller estimate; each true model applies explicit additive residuals by
//! state channel and action. Episode scores maximize residuals across every
//! candidate action and the frozen stress grid, not only the selected action.

use super::calibration::{
    calibrate_action_specific_episode_radii, calibrate_episode_radii,
    enclose_action_specific_predicted_effects, enclose_action_specific_predicted_state,
    enclose_predicted_action_effects, enclose_predicted_state,
    ActionSpecificCalibratedEnvelopeRadii, CalibratedEnvelopeRadii,
};
use super::robust_viability::{
    next_state_envelope, robustly_viable, ActionEffectBounds, Interval, StateEnvelope,
    ViabilityLimits,
};
use super::Action;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionEffectError {
    pub reserve_delta_error: f64,
    pub load_delta_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MismatchProfile {
    pub reserve_state_error: f64,
    pub load_state_error: f64,
    pub push: ActionEffectError,
    pub maintain: ActionEffectError,
    pub emergency_recover: ActionEffectError,
}

impl MismatchProfile {
    pub fn new(
        reserve_state_error: f64,
        load_state_error: f64,
        push: ActionEffectError,
        maintain: ActionEffectError,
        emergency_recover: ActionEffectError,
    ) -> Option<Self> {
        let values = [
            reserve_state_error,
            load_state_error,
            push.reserve_delta_error,
            push.load_delta_error,
            maintain.reserve_delta_error,
            maintain.load_delta_error,
            emergency_recover.reserve_delta_error,
            emergency_recover.load_delta_error,
        ];
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(Self {
                reserve_state_error,
                load_state_error,
                push,
                maintain,
                emergency_recover,
            })
    }

    const fn error(self, action: Action) -> ActionEffectError {
        match action {
            Action::Push => self.push,
            Action::Maintain => self.maintain,
            Action::EmergencyRecover => self.emergency_recover,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeResidualScores {
    pub reserve_state: f64,
    pub load_state: f64,
    pub reserve_effect: f64,
    pub load_effect: f64,
}

/// Controller-side nominal action effect, independent of the episode's true
/// mismatch profile.
pub fn nominal_action_effect(action: Action, stress: f64) -> Option<ActionEffectBounds> {
    if !stress.is_finite() || !(0.0..=1.0).contains(&stress) {
        return None;
    }
    let (reserve_delta, load_delta) = match action {
        Action::Push => (-0.055 - 0.025 * stress, 0.026 + 0.032 * stress),
        Action::Maintain => (
            0.08 * (1.0 - 0.7 * stress),
            0.008 + stress * 0.032 - if stress < 0.25 { 0.055 } else { 0.020 },
        ),
        Action::EmergencyRecover => (0.10 * (1.0 - 0.5 * stress), -0.092 + 0.032 * stress),
    };
    ActionEffectBounds::new(
        Interval::point(reserve_delta)?,
        Interval::point(load_delta)?,
    )
}

/// Environment-side true action effect. Mismatch is additive and deliberately
/// separate from the nominal equation, so a point-estimate shield can be wrong.
pub fn true_action_effect(
    action: Action,
    stress: f64,
    mismatch: MismatchProfile,
) -> Option<(f64, f64)> {
    let nominal = nominal_action_effect(action, stress)?;
    let error = mismatch.error(action);
    Some((
        nominal.reserve_delta().lower() + error.reserve_delta_error,
        nominal.load_delta().lower() + error.load_delta_error,
    ))
}

/// Produce four episode scores. Each effect score is the maximum absolute
/// residual over all actions and all stress values in the supplied grid.
pub fn episode_residual_scores(
    mismatch: MismatchProfile,
    stress_grid: &[f64],
) -> Option<EpisodeResidualScores> {
    if stress_grid.is_empty()
        || stress_grid
            .iter()
            .any(|stress| !stress.is_finite() || !(0.0..=1.0).contains(stress))
    {
        return None;
    }

    let mut reserve_effect = 0.0_f64;
    let mut load_effect = 0.0_f64;
    for stress in stress_grid {
        for action in [Action::Push, Action::Maintain, Action::EmergencyRecover] {
            let error = mismatch.error(action);
            reserve_effect = reserve_effect.max(error.reserve_delta_error.abs());
            load_effect = load_effect.max(error.load_delta_error.abs());
            nominal_action_effect(action, *stress)?;
        }
    }

    Some(EpisodeResidualScores {
        reserve_state: mismatch.reserve_state_error.abs(),
        load_state: mismatch.load_state_error.abs(),
        reserve_effect,
        load_effect,
    })
}

/// Episode-level maxima with independent reserve/load effects for every action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionSpecificEpisodeResidualScores {
    pub reserve_state: f64,
    pub load_state: f64,
    pub push_reserve_effect: f64,
    pub push_load_effect: f64,
    pub maintain_reserve_effect: f64,
    pub maintain_load_effect: f64,
    pub recovery_reserve_effect: f64,
    pub recovery_load_effect: f64,
}

/// Maximize action-effect residuals per action across the complete stress grid.
/// Keeping these six residual families separate avoids charging Maintain with
/// the larger worst-case residual of Push when the two dynamics differ.
pub fn action_specific_episode_residual_scores(
    mismatch: MismatchProfile,
    stress_grid: &[f64],
) -> Option<ActionSpecificEpisodeResidualScores> {
    if stress_grid.is_empty()
        || stress_grid
            .iter()
            .any(|stress| !stress.is_finite() || !(0.0..=1.0).contains(stress))
    {
        return None;
    }

    let mut push_reserve = 0.0_f64;
    let mut push_load = 0.0_f64;
    let mut maintain_reserve = 0.0_f64;
    let mut maintain_load = 0.0_f64;
    let mut recovery_reserve = 0.0_f64;
    let mut recovery_load = 0.0_f64;
    for stress in stress_grid {
        nominal_action_effect(Action::Push, *stress)?;
        nominal_action_effect(Action::Maintain, *stress)?;
        nominal_action_effect(Action::EmergencyRecover, *stress)?;
        push_reserve = push_reserve.max(mismatch.push.reserve_delta_error.abs());
        push_load = push_load.max(mismatch.push.load_delta_error.abs());
        maintain_reserve = maintain_reserve.max(mismatch.maintain.reserve_delta_error.abs());
        maintain_load = maintain_load.max(mismatch.maintain.load_delta_error.abs());
        recovery_reserve =
            recovery_reserve.max(mismatch.emergency_recover.reserve_delta_error.abs());
        recovery_load = recovery_load.max(mismatch.emergency_recover.load_delta_error.abs());
    }

    Some(ActionSpecificEpisodeResidualScores {
        reserve_state: mismatch.reserve_state_error.abs(),
        load_state: mismatch.load_state_error.abs(),
        push_reserve_effect: push_reserve,
        push_load_effect: push_load,
        maintain_reserve_effect: maintain_reserve,
        maintain_load_effect: maintain_load,
        recovery_reserve_effect: recovery_reserve,
        recovery_load_effect: recovery_load,
    })
}

/// Calibrate episode-level radii from a supplied independent calibration panel.
/// There must be one worst-case score per independent episode, not per timestep.
pub fn calibrate_mismatch_panel(
    profiles: &[MismatchProfile],
    stress_grid: &[f64],
    target_coverage: f64,
) -> Option<CalibratedEnvelopeRadii> {
    if profiles.is_empty() {
        return None;
    }
    let scores: Vec<EpisodeResidualScores> = profiles
        .iter()
        .map(|profile| episode_residual_scores(*profile, stress_grid))
        .collect::<Option<Vec<_>>>()?;
    let reserve_state: Vec<f64> = scores.iter().map(|score| score.reserve_state).collect();
    let load_state: Vec<f64> = scores.iter().map(|score| score.load_state).collect();
    let reserve_effect: Vec<f64> = scores.iter().map(|score| score.reserve_effect).collect();
    let load_effect: Vec<f64> = scores.iter().map(|score| score.load_effect).collect();
    calibrate_episode_radii(
        &reserve_state,
        &load_state,
        &reserve_effect,
        &load_effect,
        target_coverage,
    )
}

/// Calibrate state and per-action effect radii from independent episode scores.
pub fn calibrate_action_specific_mismatch_panel(
    profiles: &[MismatchProfile],
    stress_grid: &[f64],
    target_coverage: f64,
) -> Option<ActionSpecificCalibratedEnvelopeRadii> {
    if profiles.is_empty() {
        return None;
    }
    let scores: Vec<ActionSpecificEpisodeResidualScores> = profiles
        .iter()
        .map(|profile| action_specific_episode_residual_scores(*profile, stress_grid))
        .collect::<Option<Vec<_>>>()?;
    let reserve_state: Vec<f64> = scores.iter().map(|score| score.reserve_state).collect();
    let load_state: Vec<f64> = scores.iter().map(|score| score.load_state).collect();
    let push_reserve: Vec<f64> = scores.iter().map(|score| score.push_reserve_effect).collect();
    let push_load: Vec<f64> = scores.iter().map(|score| score.push_load_effect).collect();
    let maintain_reserve: Vec<f64> =
        scores.iter().map(|score| score.maintain_reserve_effect).collect();
    let maintain_load: Vec<f64> =
        scores.iter().map(|score| score.maintain_load_effect).collect();
    let recovery_reserve: Vec<f64> =
        scores.iter().map(|score| score.recovery_reserve_effect).collect();
    let recovery_load: Vec<f64> =
        scores.iter().map(|score| score.recovery_load_effect).collect();

    calibrate_action_specific_episode_radii(
        &reserve_state,
        &load_state,
        &push_reserve,
        &push_load,
        &maintain_reserve,
        &maintain_load,
        &recovery_reserve,
        &recovery_load,
        target_coverage,
    )
}

/// Expand a nominal action effect by that action's own calibrated uncertainty.
pub fn action_specific_calibrated_action_effect(
    action: Action,
    stress: f64,
    radii: ActionSpecificCalibratedEnvelopeRadii,
) -> Option<ActionEffectBounds> {
    enclose_action_specific_predicted_effects(nominal_action_effect(action, stress)?, action, radii)
}

/// Estimated action effect expanded by its calibrated residual envelope.
pub fn calibrated_action_effect(
    action: Action,
    stress: f64,
    radii: CalibratedEnvelopeRadii,
) -> Option<ActionEffectBounds> {
    enclose_predicted_action_effects(nominal_action_effect(action, stress)?, radii)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SafetyAvailabilityMetrics {
    pub candidate_actions: usize,
    pub point_certified_actions: usize,
    pub robust_certified_actions: usize,
    pub true_safe_actions: usize,
    pub point_false_certifications: usize,
    pub robust_false_certifications: usize,
    pub robust_false_vetoes: usize,
}

/// Keep certification safety and availability cost separate. A filter cannot
/// look successful just by rejecting every candidate action.
pub fn evaluate_safety_availability(
    state: StateEnvelope,
    stress: f64,
    mismatch: MismatchProfile,
    radii: CalibratedEnvelopeRadii,
    limits: ViabilityLimits,
    shield_can_recover: bool,
) -> Option<SafetyAvailabilityMetrics> {
    let actions: &[Action] = if shield_can_recover {
        &[Action::Push, Action::Maintain, Action::EmergencyRecover]
    } else {
        &[Action::Push, Action::Maintain]
    };
    let mut metrics = SafetyAvailabilityMetrics {
        candidate_actions: actions.len(),
        ..SafetyAvailabilityMetrics::default()
    };
    for action in actions {
        let (point_certified, robust_certified, true_safe) =
            point_and_robust_vs_true(state, *action, stress, mismatch, radii, limits)?;
        metrics.point_certified_actions += usize::from(point_certified);
        metrics.robust_certified_actions += usize::from(robust_certified);
        metrics.true_safe_actions += usize::from(true_safe);
        metrics.point_false_certifications += usize::from(point_certified && !true_safe);
        metrics.robust_false_certifications += usize::from(robust_certified && !true_safe);
        metrics.robust_false_vetoes += usize::from(!robust_certified && true_safe);
    }
    Some(metrics)
}

fn make_effect_error(reserve: f64, load: f64) -> ActionEffectError {
    ActionEffectError {
        reserve_delta_error: reserve,
        load_delta_error: load,
    }
}

fn make_profile_from_coordinates(coordinates: [f64; 8]) -> MismatchProfile {
    MismatchProfile::new(
        0.01 * coordinates[0],
        0.02 * coordinates[1],
        make_effect_error(0.03 * coordinates[2], 0.06 * coordinates[3]),
        make_effect_error(0.01 * coordinates[4], 0.025 * coordinates[5]),
        make_effect_error(0.015 * coordinates[6], 0.04 * coordinates[7]),
    )
    .expect("finite synthetic mismatch")
}

fn deterministic_trit(seed: u64, dimension: usize) -> f64 {
    let mut value = seed.wrapping_add((dimension as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    [-1.0, 0.0, 1.0][(value % 3) as usize]
}

/// Deterministic calibration-design panel: zero error, both extrema on each of
/// eight coordinates, plus mixed perturbations. This is not empirical data.
pub fn calibration_panel() -> Vec<MismatchProfile> {
    let mut profiles = vec![make_profile_from_coordinates([0.0; 8])];
    for axis in 0..8 {
        for sign in [-1.0, 1.0] {
            let mut coordinates = [0.0; 8];
            coordinates[axis] = sign;
            profiles.push(make_profile_from_coordinates(coordinates));
        }
    }
    for case in 0..22_u64 {
        let coordinates =
            std::array::from_fn(|dimension| deterministic_trit(case + 100, dimension));
        profiles.push(make_profile_from_coordinates(coordinates));
    }
    profiles
}

/// 159-profile synthetic design panel for eight-family 95% joint-coverage code
/// tests. Includes zero error, both extrema on every coordinate, and 142 mixed
/// perturbations. It is a deterministic bounds fixture, not exchangeable data.
pub fn action_specific_calibration_panel() -> Vec<MismatchProfile> {
    let mut profiles = vec![make_profile_from_coordinates([0.0; 8])];
    for axis in 0..8 {
        for sign in [-1.0, 1.0] {
            let mut coordinates = [0.0; 8];
            coordinates[axis] = sign;
            profiles.push(make_profile_from_coordinates(coordinates));
        }
    }
    for case in 0..142_u64 {
        let coordinates =
            std::array::from_fn(|dimension| deterministic_trit(case + 10_000, dimension));
        profiles.push(make_profile_from_coordinates(coordinates));
    }
    profiles
}

/// Generate a deterministic held-out mismatch profile. Scale 1.0 stays within
/// the declared coordinate bounds; scale 2.0 is an out-of-family negative control.
pub fn synthetic_mismatch_profile(seed: u64, scale: f64) -> Option<MismatchProfile> {
    if !scale.is_finite() || !(0.0..=10.0).contains(&scale) {
        return None;
    }
    let coordinates = std::array::from_fn(|dimension| deterministic_trit(seed, dimension) * scale);
    Some(make_profile_from_coordinates(coordinates))
}

/// Compare point, action-specific robust, and realized one-step safety.
/// True safety evaluates the model at the estimated interval shifted by its
/// actual state error; it does not collapse a wider state set to its midpoint.
pub fn point_and_action_specific_robust_vs_true(
    state: StateEnvelope,
    action: Action,
    stress: f64,
    mismatch: MismatchProfile,
    radii: ActionSpecificCalibratedEnvelopeRadii,
    limits: ViabilityLimits,
) -> Option<(bool, bool, bool)> {
    let point_effect = nominal_action_effect(action, stress)?;
    let robust_effect = action_specific_calibrated_action_effect(action, stress, radii)?;
    let true_effect = true_action_effect(action, stress, mismatch)?;
    let true_state = StateEnvelope::new(
        Interval::new(
            (state.reserve().lower() + mismatch.reserve_state_error).clamp(0.0, 1.0),
            (state.reserve().upper() + mismatch.reserve_state_error).clamp(0.0, 1.0),
        )?,
        Interval::new(
            (state.load().lower() + mismatch.load_state_error).clamp(0.0, 1.0),
            (state.load().upper() + mismatch.load_state_error).clamp(0.0, 1.0),
        )?,
    )?;
    let robust_state = enclose_action_specific_predicted_state(state, radii)?;
    let point_certified = robustly_viable(state, point_effect, limits);
    let robust_certified = robustly_viable(robust_state, robust_effect, limits);
    let true_next = next_state_envelope(
        true_state,
        ActionEffectBounds::new(
            Interval::point(true_effect.0)?,
            Interval::point(true_effect.1)?,
        )?,
    )?;
    let true_safe = true_next.reserve().lower() >= limits.min_reserve()
        && true_next.load().upper() <= limits.max_load();
    Some((point_certified, robust_certified, true_safe))
}

/// Compare nominal certification, robust certification, and true one-step safety.
/// Input state is the controller estimate; the true model adds the profile's state
/// error. Robust certification includes the calibrated state and action-effect
/// envelopes, while the true transition is evaluated at the perturbed state.
pub fn point_and_robust_vs_true(
    state: StateEnvelope,
    action: Action,
    stress: f64,
    mismatch: MismatchProfile,
    radii: CalibratedEnvelopeRadii,
    limits: ViabilityLimits,
) -> Option<(bool, bool, bool)> {
    let point_effect = nominal_action_effect(action, stress)?;
    let robust_effect = calibrated_action_effect(action, stress, radii)?;
    let true_effect = true_action_effect(action, stress, mismatch)?;
    let true_state = StateEnvelope::new(
        Interval::new(
            (state.reserve().lower() + mismatch.reserve_state_error).clamp(0.0, 1.0),
            (state.reserve().upper() + mismatch.reserve_state_error).clamp(0.0, 1.0),
        )?,
        Interval::new(
            (state.load().lower() + mismatch.load_state_error).clamp(0.0, 1.0),
            (state.load().upper() + mismatch.load_state_error).clamp(0.0, 1.0),
        )?,
    )?;
    let robust_state = enclose_predicted_state(state, radii)?;
    let point_certified = robustly_viable(state, point_effect, limits);
    let robust_certified = robustly_viable(robust_state, robust_effect, limits);
    let true_next = next_state_envelope(
        true_state,
        ActionEffectBounds::new(
            Interval::point(true_effect.0)?,
            Interval::point(true_effect.1)?,
        )?,
    )?;
    let true_safe = true_next.reserve().lower() >= limits.min_reserve()
        && true_next.load().upper() <= limits.max_load();
    Some((point_certified, robust_certified, true_safe))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRESS_GRID: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

    fn error(reserve: f64, load: f64) -> ActionEffectError {
        ActionEffectError {
            reserve_delta_error: reserve,
            load_delta_error: load,
        }
    }

    fn profile(scale: f64) -> MismatchProfile {
        MismatchProfile::new(
            0.01 * scale,
            -0.02 * scale,
            error(-0.03 * scale, 0.06 * scale),
            error(0.01 * scale, -0.025 * scale),
            error(-0.015 * scale, 0.04 * scale),
        )
        .expect("finite synthetic mismatch")
    }

    fn state() -> StateEnvelope {
        StateEnvelope::new(
            Interval::point(0.22).expect("finite point"),
            Interval::point(0.80).expect("finite point"),
        )
        .expect("valid state")
    }

    fn limits() -> ViabilityLimits {
        ViabilityLimits::new(0.15, 0.88).expect("valid limits")
    }

    #[test]
    fn action_specific_panel_reaches_each_of_eight_residual_families() {
        let panel = action_specific_calibration_panel();
        assert_eq!(panel.len(), 159);
        let scores: Vec<ActionSpecificEpisodeResidualScores> = panel
            .iter()
            .map(|profile| {
                action_specific_episode_residual_scores(*profile, &STRESS_GRID)
                    .expect("valid action-specific profile scores")
            })
            .collect();
        assert!((scores.iter().map(|s| s.reserve_state).fold(0.0, f64::max) - 0.01).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.load_state).fold(0.0, f64::max) - 0.02).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.push_reserve_effect).fold(0.0, f64::max) - 0.03).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.push_load_effect).fold(0.0, f64::max) - 0.06).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.maintain_reserve_effect).fold(0.0, f64::max) - 0.01).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.maintain_load_effect).fold(0.0, f64::max) - 0.025).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.recovery_reserve_effect).fold(0.0, f64::max) - 0.015).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.recovery_load_effect).fold(0.0, f64::max) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn action_specific_certificate_preserves_safe_maintain_action_lost_by_global_margin() {
        let global_panel = calibration_panel();
        let global = calibrate_mismatch_panel(&global_panel, &STRESS_GRID, 0.90)
            .expect("global four-family calibration");
        let specific_panel = action_specific_calibration_panel();
        let specific = calibrate_action_specific_mismatch_panel(
            &specific_panel,
            &STRESS_GRID,
            0.95,
        )
        .expect("159 episodes meet the eight-family 95% finite-rank gate");
        let mismatch = profile(1.0);
        let (point, global_robust, global_true_safe) =
            point_and_robust_vs_true(state(), Action::Maintain, 0.5, mismatch, global, limits())
                .expect("global comparison");
        let (specific_point, specific_robust, specific_true_safe) =
            point_and_action_specific_robust_vs_true(
                state(),
                Action::Maintain,
                0.5,
                mismatch,
                specific,
                limits(),
            )
            .expect("action-specific comparison");

        assert_eq!(point, specific_point);
        assert_eq!(global_true_safe, specific_true_safe);
        assert!(point);
        assert!(!global_robust);
        assert!(specific_robust);
        assert!(specific_true_safe);
    }

    #[test]
    fn action_specific_certificates_are_sound_for_the_declared_coordinate_box() {
        let panel = action_specific_calibration_panel();
        let radii = calibrate_action_specific_mismatch_panel(&panel, &STRESS_GRID, 0.95)
            .expect("action-specific calibration panel");
        for mismatch in panel {
            for stress in STRESS_GRID {
                for action in [Action::Push, Action::Maintain, Action::EmergencyRecover] {
                    let (_, robust_certified, true_safe) =
                        point_and_action_specific_robust_vs_true(
                            state(),
                            action,
                            stress,
                            mismatch,
                            radii,
                            limits(),
                        )
                        .expect("valid action-specific certificate inputs");
                    assert!(
                        !robust_certified || true_safe,
                        "action-specific false certification: stress={stress} action={action:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatch_calibration_panel_reaches_each_independent_error_bound() {
        let panel = calibration_panel();
        assert_eq!(panel.len(), 39);
        let scores: Vec<EpisodeResidualScores> = panel
            .iter()
            .map(|mismatch| episode_residual_scores(*mismatch, &STRESS_GRID).expect("scores"))
            .collect();
        assert!((scores.iter().map(|s| s.reserve_state).fold(0.0, f64::max) - 0.01).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.load_state).fold(0.0, f64::max) - 0.02).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.reserve_effect).fold(0.0, f64::max) - 0.03).abs() < 1e-12);
        assert!((scores.iter().map(|s| s.load_effect).fold(0.0, f64::max) - 0.06).abs() < 1e-12);
    }

    #[test]
    fn zero_mismatch_matches_nominal_model_across_all_actions_and_stress_levels() {
        let zero = profile(0.0);
        for stress in STRESS_GRID {
            for action in [Action::Push, Action::Maintain, Action::EmergencyRecover] {
                let nominal = nominal_action_effect(action, stress).expect("nominal effect");
                let actual = true_action_effect(action, stress, zero).expect("true effect");
                assert!((nominal.reserve_delta().lower() - actual.0).abs() < 1e-12);
                assert!((nominal.load_delta().lower() - actual.1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn episode_score_covers_the_largest_error_across_all_actions() {
        let mismatch = MismatchProfile::new(
            -0.03,
            0.04,
            error(0.01, -0.02),
            error(-0.05, 0.03),
            error(0.02, 0.08),
        )
        .expect("finite profile");
        let scores = episode_residual_scores(mismatch, &STRESS_GRID).expect("valid grid");
        assert!((scores.reserve_state - 0.03).abs() < 1e-12);
        assert!((scores.load_state - 0.04).abs() < 1e-12);
        assert!((scores.reserve_effect - 0.05).abs() < 1e-12);
        assert!((scores.load_effect - 0.08).abs() < 1e-12);
    }

    #[test]
    fn malformed_profiles_or_stress_grids_fail_closed() {
        assert!(MismatchProfile::new(
            f64::NAN,
            0.0,
            error(0.0, 0.0),
            error(0.0, 0.0),
            error(0.0, 0.0),
        )
        .is_none());
        let valid = profile(1.0);
        assert!(episode_residual_scores(valid, &[]).is_none());
        assert!(episode_residual_scores(valid, &[0.0, f64::NAN]).is_none());
        assert!(nominal_action_effect(Action::Push, 1.1).is_none());
        assert!(synthetic_mismatch_profile(0, f64::NAN).is_none());
    }

    #[test]
    fn point_shield_false_positive_is_rejected_by_a_calibrated_envelope() {
        let calibration = calibration_panel();
        let radii = calibrate_mismatch_panel(&calibration, &STRESS_GRID, 0.90)
            .expect("39 episodes meet the 90% four-family finite-rank gate");
        let held_out = profile(1.0);

        let (point_certified, robust_certified, true_safe) =
            point_and_robust_vs_true(state(), Action::Push, 0.5, held_out, radii, limits())
                .expect("valid test contracts");

        assert!(
            point_certified,
            "point shield should demonstrate the false positive"
        );
        assert!(
            !robust_certified,
            "calibrated shield should reject the action"
        );
        assert!(
            !true_safe,
            "the held-out true transition should violate constraints"
        );
    }

    #[test]
    fn robustness_is_scored_with_false_certifications_and_false_vetoes_separate() {
        let calibration = calibration_panel();
        let radii =
            calibrate_mismatch_panel(&calibration, &STRESS_GRID, 0.90).expect("calibration panel");
        let metrics =
            evaluate_safety_availability(state(), 0.5, profile(1.0), radii, limits(), true)
                .expect("complete metrics");

        assert_eq!(metrics.candidate_actions, 3);
        assert_eq!(metrics.point_false_certifications, 1);
        assert_eq!(metrics.robust_false_certifications, 0);
        assert!(metrics.robust_certified_actions <= metrics.point_certified_actions);
        assert!(metrics.true_safe_actions > 0);
    }

    #[test]
    fn calibrated_certificate_matches_realized_safety_for_the_enclosed_model_panel() {
        let calibration = calibration_panel();
        let radii =
            calibrate_mismatch_panel(&calibration, &STRESS_GRID, 0.90).expect("calibration panel");
        for mismatch in calibration {
            for stress in STRESS_GRID {
                for action in [Action::Push, Action::Maintain, Action::EmergencyRecover] {
                    let (_, robust_certified, true_safe) = point_and_robust_vs_true(
                        state(),
                        action,
                        stress,
                        mismatch,
                        radii,
                        limits(),
                    )
                    .expect("valid enclosed model sample");
                    assert!(
                        !robust_certified || true_safe,
                        "robust false certification: stress={stress} action={action:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn model_error_outside_calibration_envelope_is_an_explicit_negative_control() {
        let calibration: Vec<MismatchProfile> =
            (1..=39).map(|i| profile(0.2 * i as f64 / 39.0)).collect();
        let radii =
            calibrate_mismatch_panel(&calibration, &STRESS_GRID, 0.90).expect("calibration panel");
        let out_of_family = profile(2.0);
        let scores = episode_residual_scores(out_of_family, &STRESS_GRID).expect("valid profile");

        assert!(scores.reserve_effect > radii.reserve_effect_radius());
        assert!(scores.load_effect > radii.load_effect_radius());
    }
}

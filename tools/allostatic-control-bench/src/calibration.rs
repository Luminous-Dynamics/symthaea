//! Episode-level split-conformal two-sided radii for robust state and action-effect envelopes.
//!
//! Each score is the maximum absolute prediction residual over one independent
//! calibration episode. Action-effect scores must take the maximum over every
//! candidate action and timestep in that episode. The Bonferroni correction
//! provides simultaneous coverage across four score families only under the
//! exchangeability/frozen-predictor assumptions described in the research note.

use super::robust_viability::{ActionEffectBounds, Interval, StateEnvelope};
use super::Action;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedEnvelopeRadii {
    reserve_state_radius: f64,
    load_state_radius: f64,
    reserve_effect_radius: f64,
    load_effect_radius: f64,
    target_coverage: f64,
    calibration_episodes: usize,
}

impl CalibratedEnvelopeRadii {
    pub const fn reserve_state_radius(self) -> f64 {
        self.reserve_state_radius
    }

    pub const fn load_state_radius(self) -> f64 {
        self.load_state_radius
    }

    pub const fn reserve_effect_radius(self) -> f64 {
        self.reserve_effect_radius
    }

    pub const fn load_effect_radius(self) -> f64 {
        self.load_effect_radius
    }

    pub const fn target_coverage(self) -> f64 {
        self.target_coverage
    }

    pub const fn calibration_episodes(self) -> usize {
        self.calibration_episodes
    }
}

/// Return the finite split-conformal order statistic for nonnegative scores.
///
/// None means the requested rank exceeds the available calibration sample or
/// the input/coverage contract is invalid. A finite radius is never guessed.
fn finite_conformal_radius(scores: &[f64], component_coverage: f64) -> Option<f64> {
    if scores.is_empty()
        || !component_coverage.is_finite()
        || component_coverage <= 0.0
        || component_coverage >= 1.0
        || scores
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
    {
        return None;
    }

    let rank = (((scores.len() + 1) as f64) * component_coverage).ceil() as usize;
    if rank == 0 || rank > scores.len() {
        return None;
    }

    let mut ordered = scores.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered.get(rank - 1).copied()
}

/// Calibrate simultaneous episode-level radii for four residual families:
/// reserve state, load state, reserve action-effect, and load action-effect.
///
/// Each score vector must contain one maximum absolute residual per independent
/// calibration episode. The action-effect scores must cover all candidate actions
/// and timesteps, not just the action selected by a policy. Four-way Bonferroni
/// adjustment targets joint coverage across these families.
pub fn calibrate_episode_radii(
    reserve_state_episode_scores: &[f64],
    load_state_episode_scores: &[f64],
    reserve_effect_episode_scores: &[f64],
    load_effect_episode_scores: &[f64],
    target_coverage: f64,
) -> Option<CalibratedEnvelopeRadii> {
    let n = reserve_state_episode_scores.len();
    if n == 0
        || load_state_episode_scores.len() != n
        || reserve_effect_episode_scores.len() != n
        || load_effect_episode_scores.len() != n
        || !target_coverage.is_finite()
        || target_coverage <= 0.0
        || target_coverage >= 1.0
    {
        return None;
    }

    // Union bound over all four score families; independence between families
    // is not assumed, but calibration/test episodes must be exchangeable.
    let component_coverage = 1.0 - (1.0 - target_coverage) / 4.0;
    Some(CalibratedEnvelopeRadii {
        reserve_state_radius: finite_conformal_radius(
            reserve_state_episode_scores,
            component_coverage,
        )?,
        load_state_radius: finite_conformal_radius(load_state_episode_scores, component_coverage)?,
        reserve_effect_radius: finite_conformal_radius(
            reserve_effect_episode_scores,
            component_coverage,
        )?,
        load_effect_radius: finite_conformal_radius(
            load_effect_episode_scores,
            component_coverage,
        )?,
        target_coverage,
        calibration_episodes: n,
    })
}

/// Expand a predicted state by calibrated state-error radii and clip to [0, 1].
/// Returns None for invalid radii rather than falling back to a point estimate.
pub fn enclose_predicted_state(
    predicted: StateEnvelope,
    radii: CalibratedEnvelopeRadii,
) -> Option<StateEnvelope> {
    if !radii.reserve_state_radius.is_finite()
        || !radii.load_state_radius.is_finite()
        || radii.reserve_state_radius < 0.0
        || radii.load_state_radius < 0.0
        || !(0.0..1.0).contains(&radii.target_coverage)
        || radii.calibration_episodes == 0
    {
        return None;
    }

    let predicted_reserve = predicted.reserve();
    let predicted_load = predicted.load();
    let reserve = Interval::new(
        (predicted_reserve.lower() - radii.reserve_state_radius).max(0.0),
        (predicted_reserve.upper() + radii.reserve_state_radius).min(1.0),
    )?;
    let load = Interval::new(
        (predicted_load.lower() - radii.load_state_radius).max(0.0),
        (predicted_load.upper() + radii.load_state_radius).min(1.0),
    )?;
    StateEnvelope::new(reserve, load)
}

/// Expand predicted reserve/load deltas by calibrated action-effect radii.
/// The physical domain bounds each state delta to [-1, 1] in this toy model.
/// Effect scores must cover every candidate action and timestep.
pub fn enclose_predicted_action_effects(
    predicted: ActionEffectBounds,
    radii: CalibratedEnvelopeRadii,
) -> Option<ActionEffectBounds> {
    if !radii.reserve_effect_radius.is_finite()
        || !radii.load_effect_radius.is_finite()
        || radii.reserve_effect_radius < 0.0
        || radii.load_effect_radius < 0.0
        || !(0.0..1.0).contains(&radii.target_coverage)
        || radii.calibration_episodes == 0
    {
        return None;
    }

    let reserve = predicted.reserve_delta();
    let load = predicted.load_delta();
    let reserve_effect = Interval::new(
        (reserve.lower() - radii.reserve_effect_radius).max(-1.0),
        (reserve.upper() + radii.reserve_effect_radius).min(1.0),
    )?;
    let load_effect = Interval::new(
        (load.lower() - radii.load_effect_radius).max(-1.0),
        (load.upper() + radii.load_effect_radius).min(1.0),
    )?;
    ActionEffectBounds::new(reserve_effect, load_effect)
}

/// Action-conditioned calibration radii prevent the worst error observed for
/// one action from being applied to every other action. The joint target covers
/// two state and six action-effect residual families (eight in total).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionSpecificCalibratedEnvelopeRadii {
    reserve_state_radius: f64,
    load_state_radius: f64,
    push_reserve_effect_radius: f64,
    push_load_effect_radius: f64,
    maintain_reserve_effect_radius: f64,
    maintain_load_effect_radius: f64,
    recovery_reserve_effect_radius: f64,
    recovery_load_effect_radius: f64,
    target_coverage: f64,
    calibration_episodes: usize,
}

impl ActionSpecificCalibratedEnvelopeRadii {
    pub const fn reserve_state_radius(self) -> f64 {
        self.reserve_state_radius
    }

    pub const fn load_state_radius(self) -> f64 {
        self.load_state_radius
    }

    pub const fn target_coverage(self) -> f64 {
        self.target_coverage
    }

    pub const fn calibration_episodes(self) -> usize {
        self.calibration_episodes
    }

    pub const fn effect_radii(self, action: Action) -> (f64, f64) {
        match action {
            Action::Push => (
                self.push_reserve_effect_radius,
                self.push_load_effect_radius,
            ),
            Action::Maintain => (
                self.maintain_reserve_effect_radius,
                self.maintain_load_effect_radius,
            ),
            Action::EmergencyRecover => (
                self.recovery_reserve_effect_radius,
                self.recovery_load_effect_radius,
            ),
        }
    }
}

/// Calibrate eight episode-level score families with a Bonferroni joint-coverage
/// target. Each input contains exactly one maximum absolute residual per
/// independent calibration episode. This does not make a deterministic design
/// panel exchangeable; distribution-free coverage only applies under the usual
/// exchangeability and frozen-predictor assumptions.
#[allow(clippy::too_many_arguments)]
pub fn calibrate_action_specific_episode_radii(
    reserve_state_episode_scores: &[f64],
    load_state_episode_scores: &[f64],
    push_reserve_episode_scores: &[f64],
    push_load_episode_scores: &[f64],
    maintain_reserve_episode_scores: &[f64],
    maintain_load_episode_scores: &[f64],
    recovery_reserve_episode_scores: &[f64],
    recovery_load_episode_scores: &[f64],
    target_coverage: f64,
) -> Option<ActionSpecificCalibratedEnvelopeRadii> {
    let n = reserve_state_episode_scores.len();
    let score_families = [
        reserve_state_episode_scores,
        load_state_episode_scores,
        push_reserve_episode_scores,
        push_load_episode_scores,
        maintain_reserve_episode_scores,
        maintain_load_episode_scores,
        recovery_reserve_episode_scores,
        recovery_load_episode_scores,
    ];
    if n == 0
        || score_families.iter().any(|scores| scores.len() != n)
        || !target_coverage.is_finite()
        || target_coverage <= 0.0
        || target_coverage >= 1.0
    {
        return None;
    }

    let component_coverage = 1.0 - (1.0 - target_coverage) / 8.0;
    Some(ActionSpecificCalibratedEnvelopeRadii {
        reserve_state_radius: finite_conformal_radius(
            reserve_state_episode_scores,
            component_coverage,
        )?,
        load_state_radius: finite_conformal_radius(load_state_episode_scores, component_coverage)?,
        push_reserve_effect_radius: finite_conformal_radius(
            push_reserve_episode_scores,
            component_coverage,
        )?,
        push_load_effect_radius: finite_conformal_radius(
            push_load_episode_scores,
            component_coverage,
        )?,
        maintain_reserve_effect_radius: finite_conformal_radius(
            maintain_reserve_episode_scores,
            component_coverage,
        )?,
        maintain_load_effect_radius: finite_conformal_radius(
            maintain_load_episode_scores,
            component_coverage,
        )?,
        recovery_reserve_effect_radius: finite_conformal_radius(
            recovery_reserve_episode_scores,
            component_coverage,
        )?,
        recovery_load_effect_radius: finite_conformal_radius(
            recovery_load_episode_scores,
            component_coverage,
        )?,
        target_coverage,
        calibration_episodes: n,
    })
}

/// Expand a state estimate with action-specific calibration's state radii.
pub fn enclose_action_specific_predicted_state(
    predicted: StateEnvelope,
    radii: ActionSpecificCalibratedEnvelopeRadii,
) -> Option<StateEnvelope> {
    let state_only = CalibratedEnvelopeRadii {
        reserve_state_radius: radii.reserve_state_radius,
        load_state_radius: radii.load_state_radius,
        reserve_effect_radius: 0.0,
        load_effect_radius: 0.0,
        target_coverage: radii.target_coverage,
        calibration_episodes: radii.calibration_episodes,
    };
    enclose_predicted_state(predicted, state_only)
}

/// Expand one action's nominal effect with only that action's residual radii.
pub fn enclose_action_specific_predicted_effects(
    predicted: ActionEffectBounds,
    action: Action,
    radii: ActionSpecificCalibratedEnvelopeRadii,
) -> Option<ActionEffectBounds> {
    if !radii.target_coverage.is_finite()
        || !(0.0..1.0).contains(&radii.target_coverage)
        || radii.calibration_episodes == 0
    {
        return None;
    }
    let (reserve_radius, load_radius) = radii.effect_radii(action);
    if !reserve_radius.is_finite()
        || !load_radius.is_finite()
        || reserve_radius < 0.0
        || load_radius < 0.0
    {
        return None;
    }

    let reserve = predicted.reserve_delta();
    let load = predicted.load_delta();
    ActionEffectBounds::new(
        Interval::new(
            (reserve.lower() - reserve_radius).max(-1.0),
            (reserve.upper() + reserve_radius).min(1.0),
        )?,
        Interval::new(
            (load.lower() - load_radius).max(-1.0),
            (load.upper() + load_radius).min(1.0),
        )?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("valid test interval")
    }

    fn state() -> StateEnvelope {
        StateEnvelope::new(interval(0.45, 0.50), interval(0.20, 0.40)).expect("valid test state")
    }

    fn nominal_effects() -> ActionEffectBounds {
        ActionEffectBounds::new(interval(-0.05, -0.04), interval(0.01, 0.02))
            .expect("valid test effects")
    }

    fn repeated(value: f64, n: usize) -> Vec<f64> {
        vec![value; n]
    }

    #[test]
    fn calibration_uses_episode_order_statistic_across_four_families() {
        let reserve_state: Vec<f64> = (1..=39).map(|i| i as f64 / 100.0).collect();
        let load_state: Vec<f64> = (1..=39).map(|i| i as f64 / 50.0).collect();
        let reserve_effect: Vec<f64> = (1..=39).map(|i| i as f64 / 1000.0).collect();
        let load_effect: Vec<f64> = (1..=39).map(|i| i as f64 / 500.0).collect();
        let radii = calibrate_episode_radii(
            &reserve_state,
            &load_state,
            &reserve_effect,
            &load_effect,
            0.50,
        )
        .expect("four-way Bonferroni-adjusted rank fits 39 episodes");

        // Component coverage = 0.875; rank = ceil(40 * 0.875) = 35.
        assert!((radii.reserve_state_radius() - 0.35).abs() < 1e-12);
        assert!((radii.load_state_radius() - 0.70).abs() < 1e-12);
        assert!((radii.reserve_effect_radius() - 0.035).abs() < 1e-12);
        assert!((radii.load_effect_radius() - 0.070).abs() < 1e-12);
        assert_eq!(radii.calibration_episodes(), 39);
    }

    #[test]
    fn insufficient_calibration_data_fails_closed_for_high_joint_coverage() {
        let eight = repeated(0.01, 8);
        assert!(calibrate_episode_radii(&eight, &eight, &eight, &eight, 0.90).is_none());

        let thirty_nine = repeated(0.01, 39);
        assert!(calibrate_episode_radii(
            &thirty_nine,
            &thirty_nine,
            &thirty_nine,
            &thirty_nine,
            0.90
        )
        .is_some());

        let seventy_eight = repeated(0.01, 78);
        assert!(calibrate_episode_radii(
            &seventy_eight,
            &seventy_eight,
            &seventy_eight,
            &seventy_eight,
            0.95
        )
        .is_none());

        let seventy_nine = repeated(0.01, 79);
        assert!(calibrate_episode_radii(
            &seventy_nine,
            &seventy_nine,
            &seventy_nine,
            &seventy_nine,
            0.95
        )
        .is_some());
    }

    #[test]
    fn invalid_scores_coverage_and_mismatched_episode_counts_fail_closed() {
        let one = [0.1];
        assert!(calibrate_episode_radii(&one, &[], &one, &one, 0.9).is_none());
        assert!(calibrate_episode_radii(&[f64::NAN], &one, &one, &one, 0.9).is_none());
        assert!(calibrate_episode_radii(&[-0.1], &one, &one, &one, 0.9).is_none());
        assert!(calibrate_episode_radii(&one, &one, &one, &one, 1.0).is_none());
        assert!(calibrate_episode_radii(&one, &one, &one, &one, 0.0).is_none());
    }

    #[test]
    fn calibrated_radii_expand_both_state_and_action_effect_envelopes() {
        let reserve_state = repeated(0.10, 39);
        let load_state = repeated(0.25, 39);
        let reserve_effect = repeated(0.02, 39);
        let load_effect = repeated(0.05, 39);
        let radii = calibrate_episode_radii(
            &reserve_state,
            &load_state,
            &reserve_effect,
            &load_effect,
            0.90,
        )
        .expect("39 episodes support 90% four-family coverage");

        let envelope = enclose_predicted_state(state(), radii).expect("valid state envelope");
        assert!((envelope.reserve().lower() - 0.35).abs() < 1e-12);
        assert!((envelope.reserve().upper() - 0.60).abs() < 1e-12);
        assert_eq!(envelope.load().lower(), 0.0);
        assert!((envelope.load().upper() - 0.65).abs() < 1e-12);

        let effects = enclose_predicted_action_effects(nominal_effects(), radii)
            .expect("valid effect envelope");
        assert!((effects.reserve_delta().lower() - (-0.07)).abs() < 1e-12);
        assert!((effects.reserve_delta().upper() - (-0.02)).abs() < 1e-12);
        assert!((effects.load_delta().lower() - (-0.04)).abs() < 1e-12);
        assert!((effects.load_delta().upper() - 0.07).abs() < 1e-12);
    }

    #[test]
    fn increasing_any_radius_never_shrinks_calibrated_envelopes() {
        let small = calibrate_episode_radii(
            &repeated(0.05, 39),
            &repeated(0.10, 39),
            &repeated(0.01, 39),
            &repeated(0.02, 39),
            0.90,
        )
        .expect("small valid calibration");
        let large = calibrate_episode_radii(
            &repeated(0.15, 39),
            &repeated(0.20, 39),
            &repeated(0.03, 39),
            &repeated(0.04, 39),
            0.90,
        )
        .expect("large valid calibration");
        let small_state = enclose_predicted_state(state(), small).expect("small state");
        let large_state = enclose_predicted_state(state(), large).expect("large state");
        let small_effects =
            enclose_predicted_action_effects(nominal_effects(), small).expect("small effects");
        let large_effects =
            enclose_predicted_action_effects(nominal_effects(), large).expect("large effects");

        assert!(large_state.reserve().lower() <= small_state.reserve().lower());
        assert!(large_state.reserve().upper() >= small_state.reserve().upper());
        assert!(large_state.load().lower() <= small_state.load().lower());
        assert!(large_state.load().upper() >= small_state.load().upper());
        assert!(large_effects.reserve_delta().lower() <= small_effects.reserve_delta().lower());
        assert!(large_effects.reserve_delta().upper() >= small_effects.reserve_delta().upper());
        assert!(large_effects.load_delta().lower() <= small_effects.load_delta().lower());
        assert!(large_effects.load_delta().upper() >= small_effects.load_delta().upper());
    }
}

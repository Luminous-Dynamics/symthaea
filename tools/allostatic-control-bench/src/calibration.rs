//! Episode-level split-conformal radii for robust state envelopes.
//!
//! Each score must be the maximum absolute prediction residual over one
//! independent calibration episode, not a per-timestep residual. This provides
//! a finite-sample marginal episode-level calibration contract under exchangeable
//! calibration/test episodes and a frozen predictor; it is not shift-robust.

use super::robust_viability::{Interval, StateEnvelope};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedEnvelopeRadii {
    reserve_radius: f64,
    load_radius: f64,
    target_coverage: f64,
    calibration_episodes: usize,
}

impl CalibratedEnvelopeRadii {
    pub const fn reserve_radius(self) -> f64 {
        self.reserve_radius
    }

    pub const fn load_radius(self) -> f64 {
        self.load_radius
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
/// the input/coverage contract is invalid. It is intentionally not replaced by
/// a guessed finite radius.
fn finite_conformal_radius(scores: &[f64], target_coverage: f64) -> Option<f64> {
    if scores.is_empty()
        || !target_coverage.is_finite()
        || !(0.0..1.0).contains(&target_coverage)
        || scores
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
    {
        return None;
    }

    let rank = (((scores.len() + 1) as f64) * target_coverage).ceil() as usize;
    if rank == 0 || rank > scores.len() {
        return None;
    }

    let mut ordered = scores.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered.get(rank - 1).copied()
}

/// Calibrate one-sided conservative radii from episode-level max-absolute-error
/// scores for reserve and load. Both score vectors must cover the same episodes.
pub fn calibrate_episode_radii(
    reserve_episode_scores: &[f64],
    load_episode_scores: &[f64],
    target_coverage: f64,
) -> Option<CalibratedEnvelopeRadii> {
    if reserve_episode_scores.len() != load_episode_scores.len() {
        return None;
    }

    // Bonferroni-adjust the per-component coverage so the pair of reserve/load
    // intervals has at least the declared joint coverage under the assumptions.
    let component_coverage = 1.0 - (1.0 - target_coverage) / 2.0;
    Some(CalibratedEnvelopeRadii {
        reserve_radius: finite_conformal_radius(reserve_episode_scores, component_coverage)?,
        load_radius: finite_conformal_radius(load_episode_scores, component_coverage)?,
        target_coverage,
        calibration_episodes: reserve_episode_scores.len(),
    })
}

/// Expand a nominal predicted state by calibrated radii and clip to [0, 1].
/// Returns None for an invalid calibration record rather than a point estimate.
pub fn enclose_predicted_state(
    predicted: StateEnvelope,
    radii: CalibratedEnvelopeRadii,
) -> Option<StateEnvelope> {
    if !radii.reserve_radius.is_finite()
        || !radii.load_radius.is_finite()
        || radii.reserve_radius < 0.0
        || radii.load_radius < 0.0
        || !(0.0..1.0).contains(&radii.target_coverage)
        || radii.calibration_episodes == 0
    {
        return None;
    }

    let predicted_reserve = predicted.reserve();
    let predicted_load = predicted.load();
    let reserve = Interval::new(
        (predicted_reserve.lower() - radii.reserve_radius).max(0.0),
        (predicted_reserve.upper() + radii.reserve_radius).min(1.0),
    )?;
    let load = Interval::new(
        (predicted_load.lower() - radii.load_radius).max(0.0),
        (predicted_load.upper() + radii.load_radius).min(1.0),
    )?;
    StateEnvelope::new(reserve, load)
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

    #[test]
    fn calibration_uses_episode_order_statistic_not_a_timestep_pool() {
        let radii = calibrate_episode_radii(
            &[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            &[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            0.50,
        )
        .expect("Bonferroni-adjusted rank fits the nine-episode calibration set");
        assert!((radii.reserve_radius() - 0.08).abs() < 1e-12);
        assert!((radii.load_radius() - 0.80).abs() < 1e-12);
        assert_eq!(radii.calibration_episodes(), 9);
    }

    #[test]
    fn insufficient_calibration_data_fails_closed_for_high_coverage() {
        let scores = [0.01; 8];
        assert!(calibrate_episode_radii(&scores, &scores, 0.90).is_none());
        let nineteen = [0.01; 19];
        assert!(calibrate_episode_radii(&nineteen, &nineteen, 0.90).is_some());
        let thirty_nine = [0.01; 39];
        assert!(calibrate_episode_radii(&thirty_nine, &thirty_nine, 0.95).is_some());
    }

    #[test]
    fn invalid_scores_and_mismatched_episode_counts_fail_closed() {
        assert!(calibrate_episode_radii(&[0.1], &[], 0.9).is_none());
        assert!(calibrate_episode_radii(&[f64::NAN], &[0.1], 0.9).is_none());
        assert!(calibrate_episode_radii(&[-0.1], &[0.1], 0.9).is_none());
        assert!(calibrate_episode_radii(&[0.1], &[0.2], 1.0).is_none());
    }

    #[test]
    fn calibrated_radii_expand_the_predicted_state_envelope() {
        let radii = calibrate_episode_radii(&[0.10; 19], &[0.25; 19], 0.90)
            .expect("nineteen episodes support this finite rank");
        let envelope = enclose_predicted_state(state(), radii).expect("valid envelope");

        assert!((envelope.reserve().lower() - 0.35).abs() < 1e-12);
        assert!((envelope.reserve().upper() - 0.60).abs() < 1e-12);
        assert_eq!(envelope.load().lower(), 0.0);
        assert!((envelope.load().upper() - 0.65).abs() < 1e-12);
    }

    #[test]
    fn larger_radii_never_shrink_the_conservative_state_envelope() {
        let small =
            calibrate_episode_radii(&[0.05; 19], &[0.10; 19], 0.90).expect("finite calibration");
        let large =
            calibrate_episode_radii(&[0.15; 19], &[0.20; 19], 0.90).expect("finite calibration");
        let small_envelope = enclose_predicted_state(state(), small).expect("small envelope");
        let large_envelope = enclose_predicted_state(state(), large).expect("large envelope");

        assert!(large_envelope.reserve().lower() <= small_envelope.reserve().lower());
        assert!(large_envelope.reserve().upper() >= small_envelope.reserve().upper());
        assert!(large_envelope.load().lower() <= small_envelope.load().lower());
        assert!(large_envelope.load().upper() >= small_envelope.load().upper());
    }
}

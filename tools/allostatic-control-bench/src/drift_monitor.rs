//! Episode-calibrated OOD monitor that switches to a robust reachability fallback.
//!
//! This module deliberately separates anomaly detection from safety certification.
//! The detector score is calibrated from episode-level nominal scores; an alarm
//! selects the multi-step viable set. A detected shift with no viable fallback
//! action returns None instead of silently reverting to the less-conservative
//! predictive controller.

use super::multistep_viability::HorizonViableActionSet;
use super::robust_viability::RobustActionSet;
use super::Action;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftThreshold {
    score_threshold: f64,
    target_false_alarm_rate: f64,
    calibration_episodes: usize,
}

impl DriftThreshold {
    pub const fn score_threshold(self) -> f64 {
        self.score_threshold
    }

    pub const fn target_false_alarm_rate(self) -> f64 {
        self.target_false_alarm_rate
    }

    pub const fn calibration_episodes(self) -> usize {
        self.calibration_episodes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMode {
    Predictive,
    ReachabilityFallback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftAssessment {
    observed_score: f64,
    score_threshold: f64,
    drift_detected: bool,
    mode: ControlMode,
}

impl DriftAssessment {
    pub const fn observed_score(self) -> f64 {
        self.observed_score
    }

    pub const fn score_threshold(self) -> f64 {
        self.score_threshold
    }

    pub const fn drift_detected(self) -> bool {
        self.drift_detected
    }

    pub const fn mode(self) -> ControlMode {
        self.mode
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallbackDecision {
    assessment: DriftAssessment,
    proposed_action: Action,
    selected_action: Option<Action>,
}

impl FallbackDecision {
    pub const fn assessment(self) -> DriftAssessment {
        self.assessment
    }

    pub const fn proposed_action(self) -> Action {
        self.proposed_action
    }

    pub const fn selected_action(self) -> Option<Action> {
        self.selected_action
    }

    pub const fn proposal_preserved(self) -> bool {
        matches!(self.selected_action, Some(action) if action as u8 == self.proposed_action as u8)
    }

    pub const fn no_certified_action(self) -> bool {
        self.selected_action.is_none()
    }
}

/// Calibrate a high-is-out-of-distribution score threshold from one scalar score
/// per independent nominal episode. The finite-sample rank is
/// ceil((n+1)*(1-alpha)); insufficient calibration data, invalid alpha, negative
/// scores or nonfinite values return None. Exchangeability is still required for
/// the statistical false-alarm interpretation.
pub fn calibrate_drift_threshold(
    nominal_episode_scores: &[f64],
    target_false_alarm_rate: f64,
) -> Option<DriftThreshold> {
    let n = nominal_episode_scores.len();
    if n == 0
        || !target_false_alarm_rate.is_finite()
        || !(0.0..1.0).contains(&target_false_alarm_rate)
        || nominal_episode_scores
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
    {
        return None;
    }

    let component_coverage = 1.0 - target_false_alarm_rate;
    let rank = (((n + 1) as f64) * component_coverage).ceil() as usize;
    if rank == 0 || rank > n {
        return None;
    }
    let mut ordered = nominal_episode_scores.to_vec();
    ordered.sort_by(f64::total_cmp);
    let score_threshold = *ordered.get(rank - 1)?;

    Some(DriftThreshold {
        score_threshold,
        target_false_alarm_rate,
        calibration_episodes: n,
    })
}

/// Detect drift when the current episode/window nonconformity score is strictly
/// greater than the nominal calibration quantile. Equality is deliberately
/// non-alarm, making ties conservative with respect to false alarms.
pub fn assess_drift(score: f64, threshold: DriftThreshold) -> Option<DriftAssessment> {
    if !score.is_finite()
        || score < 0.0
        || !threshold.score_threshold.is_finite()
        || threshold.score_threshold < 0.0
        || !threshold.target_false_alarm_rate.is_finite()
        || !(0.0..1.0).contains(&threshold.target_false_alarm_rate)
        || threshold.calibration_episodes == 0
    {
        return None;
    }

    let drift_detected = score > threshold.score_threshold;
    Some(DriftAssessment {
        observed_score: score,
        score_threshold: threshold.score_threshold,
        drift_detected,
        mode: if drift_detected {
            ControlMode::ReachabilityFallback
        } else {
            ControlMode::Predictive
        },
    })
}

/// Choose with the predictive one-step set in nominal mode and the multi-step
/// reachability set after a drift alarm. Crucially, an empty fallback set returns
/// no action; the detector never makes an uncertified predictive action safe.
pub fn select_control_action(
    score: f64,
    threshold: DriftThreshold,
    proposed_action: Action,
    predictive_viable: RobustActionSet,
    reachability_viable: HorizonViableActionSet,
) -> Option<FallbackDecision> {
    let assessment = assess_drift(score, threshold)?;
    let selected_action = match assessment.mode() {
        ControlMode::Predictive => predictive_viable.filter(proposed_action),
        ControlMode::ReachabilityFallback => reachability_viable.filter(proposed_action),
    };
    Some(FallbackDecision {
        assessment,
        proposed_action,
        selected_action,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multistep_viability::evaluate_horizon_action_set;
    use crate::robust_viability::{
        evaluate_action_set, ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope,
        ViabilityLimits,
    };

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("valid interval")
    }

    fn effect(
        reserve_delta: (f64, f64),
        load_delta: (f64, f64),
    ) -> ActionEffectBounds {
        ActionEffectBounds::new(
            interval(reserve_delta.0, reserve_delta.1),
            interval(load_delta.0, load_delta.1),
        )
        .expect("valid action effect")
    }

    fn test_sets() -> (RobustActionSet, HorizonViableActionSet) {
        let state = StateEnvelope::new(
            interval(0.17, 0.17),
            interval(0.81, 0.81),
        )
        .expect("valid state");
        let dynamics = DynamicsBounds::new(
            effect((-0.005, -0.005), (0.06, 0.06)),
            effect((0.005, 0.005), (0.02, 0.02)),
            effect((0.05, 0.05), (-0.12, -0.12)),
        );
        let limits = ViabilityLimits::new(0.15, 0.88).expect("valid limits");
        let predictive = evaluate_action_set(state, dynamics, limits, true);
        let fallback = evaluate_horizon_action_set(state, dynamics, limits, false, 2)
            .expect("valid horizon");
        (predictive, fallback)
    }

    fn nominal_scores(count: usize) -> Vec<f64> {
        (0..count)
            .map(|i| (i + 1) as f64 / (count + 2) as f64 * 0.25)
            .collect()
    }

    #[test]
    fn finite_rank_gate_rejects_too_small_sample_and_accepts_minimum() {
        let eighteen = nominal_scores(18);
        let nineteen = nominal_scores(19);
        assert!(calibrate_drift_threshold(&eighteen, 0.05).is_none());
        let threshold = calibrate_drift_threshold(&nineteen, 0.05)
            .expect("19 nominal episodes reach the finite 95th-percentile rank");
        assert_eq!(threshold.calibration_episodes(), 19);
        assert!((threshold.target_false_alarm_rate() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn invalid_scores_and_targets_fail_closed() {
        assert!(calibrate_drift_threshold(&[], 0.05).is_none());
        assert!(calibrate_drift_threshold(&[0.1, f64::NAN], 0.05).is_none());
        assert!(calibrate_drift_threshold(&[0.1, -0.1], 0.05).is_none());
        assert!(calibrate_drift_threshold(&[0.1, 0.2], 0.0).is_none());
        assert!(calibrate_drift_threshold(&[0.1, 0.2], 1.0).is_none());
        let threshold = calibrate_drift_threshold(&nominal_scores(19), 0.05)
            .expect("valid threshold");
        assert!(assess_drift(f64::NAN, threshold).is_none());
        assert!(assess_drift(-0.1, threshold).is_none());
    }

    #[test]
    fn threshold_tie_does_not_alarm_but_a_greater_score_switches_mode() {
        let threshold = calibrate_drift_threshold(&nominal_scores(19), 0.05)
            .expect("valid threshold");
        let tie = assess_drift(threshold.score_threshold(), threshold).expect("finite score");
        assert!(!tie.drift_detected());
        assert_eq!(tie.mode(), ControlMode::Predictive);

        let shifted = assess_drift(threshold.score_threshold() + 0.001, threshold)
            .expect("finite shifted score");
        assert!(shifted.drift_detected());
        assert_eq!(shifted.mode(), ControlMode::ReachabilityFallback);
    }

    #[test]
    fn nominal_mode_preserves_a_one_step_safe_proposal() {
        let threshold = calibrate_drift_threshold(&nominal_scores(19), 0.05)
            .expect("valid threshold");
        let (predictive, fallback) = test_sets();
        assert!(predictive.contains(Action::Push));
        assert!(!fallback.contains(Action::Push));
        assert!(fallback.contains(Action::Maintain));

        let decision = select_control_action(
            threshold.score_threshold() * 0.5,
            threshold,
            Action::Push,
            predictive,
            fallback,
        )
        .expect("valid drift assessment");
        assert_eq!(decision.assessment().mode(), ControlMode::Predictive);
        assert_eq!(decision.selected_action(), Some(Action::Push));
        assert!(decision.proposal_preserved());
    }

    #[test]
    fn detected_shift_selects_only_a_multistep_certified_fallback_action() {
        let threshold = calibrate_drift_threshold(&nominal_scores(19), 0.05)
            .expect("valid threshold");
        let (predictive, fallback) = test_sets();
        let shifted_score = threshold.score_threshold() + 0.1;
        let decision = select_control_action(
            shifted_score,
            threshold,
            Action::Push,
            predictive,
            fallback,
        )
        .expect("valid shifted assessment");

        assert_eq!(
            decision.assessment().mode(),
            ControlMode::ReachabilityFallback
        );
        assert_eq!(decision.selected_action(), Some(Action::Maintain));
        assert!(!decision.proposal_preserved());
        assert!(fallback.contains(decision.selected_action().expect("selected fallback")));
    }

    #[test]
    fn no_reachability_action_means_no_action_not_predictive_fallback() {
        let threshold = calibrate_drift_threshold(&nominal_scores(19), 0.05)
            .expect("valid threshold");
        let (predictive, _) = test_sets();
        let unsafe_state = StateEnvelope::new(
            interval(0.14, 0.16),
            interval(0.89, 0.95),
        )
        .expect("valid but unsafe initial state");
        let dynamics = DynamicsBounds::new(
            effect((-0.02, -0.01), (0.01, 0.02)),
            effect((0.0, 0.01), (0.0, 0.02)),
            effect((0.01, 0.02), (-0.2, -0.1)),
        );
        let limits = ViabilityLimits::new(0.15, 0.88).expect("valid limits");
        let no_fallback =
            evaluate_horizon_action_set(unsafe_state, dynamics, limits, true, 3)
                .expect("valid horizon and no viable action");
        assert_eq!(no_fallback.count(), 0);

        let decision = select_control_action(
            threshold.score_threshold() + 0.1,
            threshold,
            Action::Push,
            predictive,
            no_fallback,
        )
        .expect("valid drift assessment");
        assert!(decision.no_certified_action());
        assert_eq!(decision.selected_action(), None);
    }
}

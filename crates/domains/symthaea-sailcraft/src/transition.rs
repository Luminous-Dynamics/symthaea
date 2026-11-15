// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded tack/gybe transition-cost envelopes.
//!
//! This module adjusts a steady-course VMG interval for a named transition
//! using source-bound intervals for transition time and ground progress.
//! It does not model aerodynamic/hydrodynamic forces or emit actuator actions.

use super::{Evidence, EvidenceState, PlanStatus, evidence_gate};
use super::robust::{RobustCourseCandidate, ScalarInterval};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManeuverKind {
    Tack,
    Gybe,
}

/// A measured or validated transition envelope for one exact craft/polar/model.
/// ground_progress_m is signed progress toward the waypoint; negative progress
/// is valid and must not be clamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManeuverTransitionBound {
    pub kind: ManeuverKind,
    pub configuration_digest: [u8; 32],
    pub polar_artifact_digest: [u8; 32],
    pub actuator_model_digest: [u8; 32],
    pub duration_seconds: ScalarInterval,
    pub ground_progress_m: ScalarInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionEvaluationStatus {
    Adjusted,
    NotEvaluated,
    Conflicted,
    OutOfScope,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransitionAdjustedVmg {
    pub maneuver_kind: ManeuverKind,
    pub planning_horizon_seconds: f64,
    pub steady_course_vmg_mps: ScalarInterval,
    pub maneuver_duration_seconds: ScalarInterval,
    pub maneuver_ground_progress_m: ScalarInterval,
    pub post_maneuver_ground_progress_m: ScalarInterval,
    pub total_ground_progress_m: ScalarInterval,
    pub effective_ground_vmg_mps: ScalarInterval,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransitionEvaluation {
    pub status: TransitionEvaluationStatus,
    pub result: Option<TransitionAdjustedVmg>,
    pub reason: &'static str,
}

impl TransitionEvaluation {
    fn blocked(status: TransitionEvaluationStatus, reason: &'static str) -> Self {
        Self { status, result: None, reason }
    }
}

fn multiply(left: ScalarInterval, right: ScalarInterval) -> Option<ScalarInterval> {
    if !left.is_finite_ordered() || !right.is_finite_ordered() {
        return None;
    }
    let values = [
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    ];
    let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let result = ScalarInterval { lower, upper };
    result.is_finite_ordered().then_some(result)
}

fn add(left: ScalarInterval, right: ScalarInterval) -> Option<ScalarInterval> {
    if !left.is_finite_ordered() || !right.is_finite_ordered() {
        return None;
    }
    let result = ScalarInterval {
        lower: left.lower + right.lower,
        upper: left.upper + right.upper,
    };
    result.is_finite_ordered().then_some(result)
}

/// Compute an interval-bounded route-segment score after a tack or gybe.
///
/// For a fixed planning horizon H, the transition duration must fit strictly
/// inside it. Remaining steady-course duration is [H - t_max, H - t_min].
/// The transition's signed ground progress is retained without clamping.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_transition_adjusted_vmg(
    candidate: &RobustCourseCandidate,
    transition: &Evidence<ManeuverTransitionBound>,
    expected_kind: ManeuverKind,
    planning_horizon_seconds: f64,
    expected_actuator_model_digest: [u8; 32],
    expected_timebase_id: &str,
) -> TransitionEvaluation {
    let bound = match evidence_gate(transition) {
        Ok(value) => value,
        Err(PlanStatus::Conflicted) => {
            return TransitionEvaluation::blocked(
                TransitionEvaluationStatus::Conflicted,
                "maneuver transition evidence is conflicted",
            );
        }
        Err(_) => {
            return TransitionEvaluation::blocked(
                TransitionEvaluationStatus::NotEvaluated,
                "maneuver transition evidence is missing, stale or unqualified",
            );
        }
    };

    if transition.timebase_id != expected_timebase_id {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::NotEvaluated,
            "maneuver transition evidence uses a different timebase",
        );
    }

    if bound.configuration_digest != candidate.configuration_digest
        || bound.polar_artifact_digest != candidate.polar_artifact_digest
        || bound.actuator_model_digest != expected_actuator_model_digest
        || candidate.configuration_digest == [0; 32]
        || candidate.polar_artifact_digest == [0; 32]
        || expected_actuator_model_digest == [0; 32]
    {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::OutOfScope,
            "maneuver bound is not bound to the exact craft, polar and actuator model",
        );
    }

    if bound.kind != expected_kind {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::OutOfScope,
            "maneuver kind does not match the requested transition",
        );
    }

    let duration = bound.duration_seconds;
    let maneuver_progress = bound.ground_progress_m;
    if !planning_horizon_seconds.is_finite()
        || planning_horizon_seconds <= 0.0
        || !duration.is_finite_ordered()
        || !maneuver_progress.is_finite_ordered()
        || duration.lower < 0.0
        || duration.upper <= 0.0
        || duration.upper >= planning_horizon_seconds
        || !candidate.ground_velocity_made_good_mps.is_finite_ordered()
    {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::NotEvaluated,
            "maneuver duration, progress or route horizon is invalid",
        );
    }

    let remaining_time = ScalarInterval {
        lower: planning_horizon_seconds - duration.upper,
        upper: planning_horizon_seconds - duration.lower,
    };
    let Some(post_maneuver_progress) =
        multiply(candidate.ground_velocity_made_good_mps, remaining_time)
    else {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::NotEvaluated,
            "post-maneuver progress interval overflowed",
        );
    };
    let Some(total_progress) = add(maneuver_progress, post_maneuver_progress) else {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::NotEvaluated,
            "total route progress interval overflowed",
        );
    };

    let effective_vmg = ScalarInterval {
        lower: total_progress.lower / planning_horizon_seconds,
        upper: total_progress.upper / planning_horizon_seconds,
    };
    if !effective_vmg.is_finite_ordered() {
        return TransitionEvaluation::blocked(
            TransitionEvaluationStatus::NotEvaluated,
            "effective route VMG interval is non-finite",
        );
    }

    TransitionEvaluation {
        status: TransitionEvaluationStatus::Adjusted,
        result: Some(TransitionAdjustedVmg {
            maneuver_kind: expected_kind,
            planning_horizon_seconds,
            steady_course_vmg_mps: candidate.ground_velocity_made_good_mps,
            maneuver_duration_seconds: duration,
            maneuver_ground_progress_m: maneuver_progress,
            post_maneuver_ground_progress_m: post_maneuver_progress,
            total_ground_progress_m: total_progress,
            effective_ground_vmg_mps: effective_vmg,
        }),
        reason: "steady-course VMG adjusted using source-bound maneuver time/progress intervals",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TackSide;

    const CONFIG: [u8; 32] = [0x5a; 32];
    const POLAR: [u8; 32] = [0xb6; 32];
    const ACTUATOR: [u8; 32] = [0xd9; 32];
    const TIMEBASE: &str = "sail-transition-common-timebase";

    fn candidate(vmg: ScalarInterval) -> RobustCourseCandidate {
        RobustCourseCandidate {
            heading_deg: 45.0,
            tack_side: TackSide::Port,
            true_wind_angle_deg: ScalarInterval { lower: 45.0, upper: 50.0 },
            boat_speed_mps: ScalarInterval { lower: 4.0, upper: 5.0 },
            boat_velocity_made_good_mps: ScalarInterval { lower: 2.0, upper: 3.0 },
            current_velocity_made_good_mps: ScalarInterval { lower: 0.0, upper: 0.0 },
            ground_velocity_made_good_mps: vmg,
        }
    }

    fn evidence(
        kind: ManeuverKind,
        duration: ScalarInterval,
        progress: ScalarInterval,
    ) -> Evidence<ManeuverTransitionBound> {
        Evidence::qualified_with_freshness(
            ManeuverTransitionBound {
                kind,
                configuration_digest: CONFIG,
                polar_artifact_digest: POLAR,
                actuator_model_digest: ACTUATOR,
                duration_seconds: duration,
                ground_progress_m: progress,
            },
            [0xe1; 32],
            TIMEBASE,
            0.0,
            2.0,
        )
    }

    fn evaluate(
        bound: &Evidence<ManeuverTransitionBound>,
        kind: ManeuverKind,
        horizon: f64,
    ) -> TransitionEvaluation {
        evaluate_transition_adjusted_vmg(
            &candidate(ScalarInterval { lower: 2.0, upper: 3.0 }),
            bound,
            kind,
            horizon,
            ACTUATOR,
            TIMEBASE,
        )
    }

    #[test]
    fn valid_tack_cost_reduces_effective_vmg_below_steady_course_vmg() {
        let bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 10.0, upper: 20.0 },
            ScalarInterval { lower: -8.0, upper: -4.0 },
        );
        let result = evaluate(&bound, ManeuverKind::Tack, 100.0);
        assert_eq!(result.status, TransitionEvaluationStatus::Adjusted);
        let adjusted = result.result.unwrap();
        assert!(adjusted.effective_ground_vmg_mps.lower < 2.0);
        assert!(adjusted.effective_ground_vmg_mps.upper < 3.0);
    }

    #[test]
    fn negative_transition_progress_is_preserved() {
        let bound = evidence(
            ManeuverKind::Gybe,
            ScalarInterval { lower: 3.0, upper: 5.0 },
            ScalarInterval { lower: -20.0, upper: -10.0 },
        );
        let result = evaluate(&bound, ManeuverKind::Gybe, 60.0);
        assert_eq!(
            result.result.unwrap().maneuver_ground_progress_m,
            ScalarInterval { lower: -20.0, upper: -10.0 }
        );
    }

    #[test]
    fn duration_uncertainty_widens_effective_vmg_interval() {
        let narrow = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 10.0, upper: 10.0 },
            ScalarInterval { lower: -5.0, upper: -5.0 },
        );
        let wide = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 5.0, upper: 25.0 },
            ScalarInterval { lower: -5.0, upper: -5.0 },
        );
        let narrow_result = evaluate(&narrow, ManeuverKind::Tack, 100.0).result.unwrap();
        let wide_result = evaluate(&wide, ManeuverKind::Tack, 100.0).result.unwrap();
        assert!(
            wide_result.effective_ground_vmg_mps.upper
                - wide_result.effective_ground_vmg_mps.lower
                >= narrow_result.effective_ground_vmg_mps.upper
                    - narrow_result.effective_ground_vmg_mps.lower
        );
    }

    #[test]
    fn transition_duration_must_fit_strictly_inside_horizon() {
        let bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 5.0, upper: 100.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        let result = evaluate(&bound, ManeuverKind::Tack, 100.0);
        assert_eq!(result.status, TransitionEvaluationStatus::NotEvaluated);
    }

    #[test]
    fn zero_or_negative_horizon_is_rejected() {
        let bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 0.0).status,
            TransitionEvaluationStatus::NotEvaluated
        );
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, -1.0).status,
            TransitionEvaluationStatus::NotEvaluated
        );
    }

    #[test]
    fn configuration_polar_and_actuator_model_must_match_exactly() {
        let mut bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        bound.value.as_mut().unwrap().polar_artifact_digest = [0x44; 32];
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::OutOfScope
        );
        bound.value.as_mut().unwrap().polar_artifact_digest = POLAR;
        bound.value.as_mut().unwrap().actuator_model_digest = [0; 32];
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::OutOfScope
        );
    }

    #[test]
    fn transition_kind_mismatch_is_rejected() {
        let bound = evidence(
            ManeuverKind::Gybe,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::OutOfScope
        );
    }

    #[test]
    fn conflicting_and_stale_transition_evidence_remain_distinct() {
        let mut conflicted = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        conflicted.state = EvidenceState::Conflicted;
        assert_eq!(
            evaluate(&conflicted, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::Conflicted
        );
        let mut stale = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        stale.state = EvidenceState::Stale;
        assert_eq!(
            evaluate(&stale, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::NotEvaluated
        );
    }

    #[test]
    fn transition_evidence_requires_expected_common_timebase() {
        let mut bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -2.0, upper: 0.0 },
        );
        bound.timebase_id = "other-timebase".into();
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::NotEvaluated
        );
    }

    #[test]
    fn interval_products_cover_negative_vmg_and_transition_time() {
        let c = candidate(ScalarInterval { lower: -3.0, upper: -1.0 });
        let bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 10.0, upper: 20.0 },
            ScalarInterval { lower: -2.0, upper: 1.0 },
        );
        let result = evaluate_transition_adjusted_vmg(
            &c, &bound, ManeuverKind::Tack, 100.0, ACTUATOR, TIMEBASE
        );
        let adjusted = result.result.unwrap();
        assert!((adjusted.effective_ground_vmg_mps.lower + 2.72).abs() < 1e-9);
        assert!((adjusted.effective_ground_vmg_mps.upper + 0.79).abs() < 1e-9);
    }

    #[test]
    fn non_finite_and_unordered_intervals_are_rejected() {
        let mut bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: f64::NAN, upper: 0.0 },
        );
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::NotEvaluated
        );
        bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 2.0, upper: 1.0 },
            ScalarInterval { lower: -1.0, upper: 0.0 },
        );
        assert_eq!(
            evaluate(&bound, ManeuverKind::Tack, 50.0).status,
            TransitionEvaluationStatus::NotEvaluated
        );
    }

    #[test]
    fn overflowed_interval_arithmetic_is_not_returned() {
        let bound = evidence(
            ManeuverKind::Tack,
            ScalarInterval { lower: 1.0, upper: 2.0 },
            ScalarInterval { lower: -f64::MAX, upper: f64::MAX },
        );
        let c = candidate(ScalarInterval { lower: f64::MAX, upper: f64::MAX });
        assert_eq!(
            evaluate_transition_adjusted_vmg(
                &c, &bound, ManeuverKind::Tack, 50.0, ACTUATOR, TIMEBASE
            ).status,
            TransitionEvaluationStatus::NotEvaluated
        );
    }
}

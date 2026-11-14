// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-bounded sailing route candidates.
//!
//! This module complements the point-valued P0 planner. It does not select a
//! single route when uncertainty envelopes overlap, does not extrapolate the
//! polar surface, and does not emit actuator commands.
//!
//! For each polar interpolation cell, a bilinear surface reaches its extrema
//! on the cell-rectangle corners. The evaluator clips the requested wind/TWA
//! interval to every intersected grid cell and evaluates every clipped corner.
//! A source-bound absolute polar-speed error is then applied conservatively.
//!
//! The resulting interval is a deterministic envelope under the declared
//! input bounds. It is not a probabilistic forecast or physical validation.

use super::{Evidence, PolarDiagram, PolarError, TackSide, evidence_gate, signed_angle_deg};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ScalarInterval {
    pub fn is_finite_ordered(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }

    fn add(self, other: Self) -> Self {
        Self {
            lower: self.lower + other.lower,
            upper: self.upper + other.upper,
        }
    }

    fn scale(self, factor: f64) -> Self {
        if factor >= 0.0 {
            Self {
                lower: self.lower * factor,
                upper: self.upper * factor,
            }
        } else {
            Self {
                lower: self.upper * factor,
                upper: self.lower * factor,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionInterval {
    /// Centre bearing in degrees clockwise from true north.
    pub center_deg: f64,
    /// Symmetric half-width on the circle, in [0, 180] degrees.
    pub half_width_deg: f64,
}

impl DirectionInterval {
    fn validate(self) -> bool {
        self.center_deg.is_finite()
            && (0.0..360.0).contains(&self.center_deg)
            && self.half_width_deg.is_finite()
            && (0.0..=180.0).contains(&self.half_width_deg)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindUncertainty {
    /// Direction the wind comes FROM.
    pub from_direction: DirectionInterval,
    /// True-wind speed bounds in metres per second.
    pub speed_mps: ScalarInterval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurrentUncertainty {
    /// Direction the current flows TOWARD.
    pub toward_direction: DirectionInterval,
    /// Current speed bounds in metres per second.
    pub speed_mps: ScalarInterval,
}

/// Source-bound absolute error bound for the supplied polar surface.
///
/// The digest identifies the error-bound artifact; this module does not
/// authenticate its publisher or prove that the bound covers the physical
/// craft. Configuration equality prevents cross-craft reuse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolarErrorBound {
    pub source_artifact_digest: [u8; 32],
    pub configuration_digest: [u8; 32],
    pub maximum_absolute_speed_error_mps: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustRouteRequest {
    pub configuration_digest: [u8; 32],
    pub wind: Evidence<WindUncertainty>,
    pub current: Evidence<CurrentUncertainty>,
    pub target_bearing: Evidence<f64>,
    pub polar_error: PolarErrorBound,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustCourseCandidate {
    /// Advisory heading only; this is never a rudder or helm command.
    pub heading_deg: f64,
    pub tack_side: TackSide,
    /// Absolute true-wind-angle bounds for every wind direction in the input set.
    pub true_wind_angle_deg: ScalarInterval,
    pub boat_speed_mps: ScalarInterval,
    pub boat_velocity_made_good_mps: ScalarInterval,
    pub current_velocity_made_good_mps: ScalarInterval,
    /// Conservative ground-VMG envelope, including polar and current bounds.
    pub ground_velocity_made_good_mps: ScalarInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustPlanStatus {
    RobustCandidatesAvailable,
    RankingUnresolved,
    NoGuaranteedProgress,
    NoRobustFeasibleCourse,
    NotEvaluated,
    Conflicted,
    OutOfScope,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustRouteEvaluation {
    pub status: RobustPlanStatus,
    /// True only if every wind direction in the uncertainty set clears the no-go sector.
    pub direct_course_robustly_admissible: Option<bool>,
    /// At most one candidate per tack side; retain both when their envelopes overlap.
    pub candidates: Vec<RobustCourseCandidate>,
    /// Present only where one candidate's guaranteed VMG exceeds the other's possible VMG.
    pub unique_best_tack: Option<TackSide>,
    pub reason: &'static str,
}

impl RobustRouteEvaluation {
    fn blocked(status: RobustPlanStatus, reason: &'static str) -> Self {
        Self {
            status,
            direct_course_robustly_admissible: None,
            candidates: Vec::new(),
            unique_best_tack: None,
            reason,
        }
    }
}

fn polar_speed_envelope(
    polar: &PolarDiagram,
    wind_speed: ScalarInterval,
    twa: ScalarInterval,
    absolute_error_mps: f64,
) -> Result<ScalarInterval, PolarError> {
    if !wind_speed.is_finite_ordered()
        || !twa.is_finite_ordered()
        || !absolute_error_mps.is_finite()
        || absolute_error_mps < 0.0
    {
        return Err(PolarError::Invalid);
    }

    let wind_min = polar.true_wind_speeds_mps[0];
    let wind_max = *polar.true_wind_speeds_mps.last().ok_or(PolarError::Invalid)?;
    let twa_min = polar.true_wind_angles_deg[0];
    let twa_max = *polar.true_wind_angles_deg.last().ok_or(PolarError::Invalid)?;

    if wind_speed.lower < wind_min
        || wind_speed.upper > wind_max
        || twa.lower < twa_min
        || twa.upper > twa_max
    {
        return Err(PolarError::WindOutOfDomain);
    }

    let mut nominal_min = f64::INFINITY;
    let mut nominal_max = f64::NEG_INFINITY;
    let mut evaluated_cell = false;

    for wind_cell in 0..polar.true_wind_speeds_mps.len() - 1 {
        let cell_wind_lower = wind_speed.lower.max(polar.true_wind_speeds_mps[wind_cell]);
        let cell_wind_upper = wind_speed.upper.min(polar.true_wind_speeds_mps[wind_cell + 1]);
        if cell_wind_lower > cell_wind_upper {
            continue;
        }

        for angle_cell in 0..polar.true_wind_angles_deg.len() - 1 {
            let cell_twa_lower = twa.lower.max(polar.true_wind_angles_deg[angle_cell]);
            let cell_twa_upper = twa.upper.min(polar.true_wind_angles_deg[angle_cell + 1]);
            if cell_twa_lower > cell_twa_upper {
                continue;
            }

            evaluated_cell = true;
            // A bilinear function is linear in either coordinate when the
            // other is fixed, so its rectangular-cell extrema occur at corners.
            for wind_value in [cell_wind_lower, cell_wind_upper] {
                for angle_value in [cell_twa_lower, cell_twa_upper] {
                    let value = polar.speed_at(wind_value, angle_value)?;
                    nominal_min = nominal_min.min(value);
                    nominal_max = nominal_max.max(value);
                }
            }
        }
    }

    if !evaluated_cell || !nominal_min.is_finite() || !nominal_max.is_finite() {
        return Err(PolarError::Invalid);
    }

    Ok(ScalarInterval {
        lower: (nominal_min - absolute_error_mps).max(0.0),
        upper: nominal_max + absolute_error_mps,
    })
}

fn absolute_twa_interval(
    wind_from: DirectionInterval,
    heading_deg: f64,
) -> ScalarInterval {
    let centre_distance = signed_angle_deg(wind_from.center_deg, heading_deg).abs();
    ScalarInterval {
        lower: (centre_distance - wind_from.half_width_deg).max(0.0),
        upper: (centre_distance + wind_from.half_width_deg).min(180.0),
    }
}

fn cosine_envelope(direction: DirectionInterval, target_bearing_deg: f64) -> ScalarInterval {
    let centre_delta = signed_angle_deg(direction.center_deg, target_bearing_deg);
    let half_width = direction.half_width_deg;
    let lower_endpoint = (centre_delta - half_width).to_radians().cos();
    let upper_endpoint = (centre_delta + half_width).to_radians().cos();
    let mut lower = lower_endpoint.min(upper_endpoint);
    let mut upper = lower_endpoint.max(upper_endpoint);

    // cos reaches -1 at the antipode and +1 at the target direction.
    if 180.0 - centre_delta.abs() <= half_width {
        lower = -1.0;
    }
    if centre_delta.abs() <= half_width {
        upper = 1.0;
    }

    ScalarInterval { lower, upper }
}

fn direction_projection(
    speed: ScalarInterval,
    direction: DirectionInterval,
    target_bearing_deg: f64,
) -> ScalarInterval {
    let cosine = cosine_envelope(direction, target_bearing_deg);
    let products = [
        speed.lower * cosine.lower,
        speed.lower * cosine.upper,
        speed.upper * cosine.lower,
        speed.upper * cosine.upper,
    ];
    let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ScalarInterval { lower, upper }
}

fn best_for_tack(
    current: Option<RobustCourseCandidate>,
    candidate: RobustCourseCandidate,
) -> Option<RobustCourseCandidate> {
    match current {
        None => Some(candidate),
        Some(previous) => {
            let candidate_lower = candidate.ground_velocity_made_good_mps.lower;
            let previous_lower = previous.ground_velocity_made_good_mps.lower;
            if candidate_lower > previous_lower
                || (candidate_lower == previous_lower
                    && candidate.ground_velocity_made_good_mps.upper
                        > previous.ground_velocity_made_good_mps.upper)
            {
                Some(candidate)
            } else {
                Some(previous)
            }
        }
    }
}

/// Evaluate robust route candidates over bounded wind, current and polar error.
///
/// It returns one best *worst-case* candidate per tack side. If the resulting
/// VMG envelopes overlap, no unique winner is asserted. It never extrapolates
/// the polar surface or emits actuator commands.
pub fn evaluate_robust_route_candidates(
    polar: &PolarDiagram,
    request: RobustRouteRequest,
) -> RobustRouteEvaluation {
    let wind = match evidence_gate(&request.wind) {
        Ok(value) => value,
        Err(PlanStatus::Conflicted) => {
            return RobustRouteEvaluation::blocked(
                RobustPlanStatus::Conflicted,
                "wind uncertainty evidence is conflicted",
            );
        }
        Err(_) => {
            return RobustRouteEvaluation::blocked(
                RobustPlanStatus::NotEvaluated,
                "wind uncertainty evidence is missing, stale or unqualified",
            );
        }
    };
    let current = match evidence_gate(&request.current) {
        Ok(value) => value,
        Err(PlanStatus::Conflicted) => {
            return RobustRouteEvaluation::blocked(
                RobustPlanStatus::Conflicted,
                "current uncertainty evidence is conflicted",
            );
        }
        Err(_) => {
            return RobustRouteEvaluation::blocked(
                RobustPlanStatus::NotEvaluated,
                "current uncertainty evidence is missing, stale or unqualified",
            );
        }
    };
    let target_bearing = match evidence_gate(&request.target_bearing) {
        Ok(value) => *value,
        Err(PlanStatus::Conflicted) => {
            return RobustRouteEvaluation::blocked(
                RobustPlanStatus::Conflicted,
                "waypoint-bearing evidence is conflicted",
            );
        }
        Err(_) => {
            return RobustRouteEvaluation::blocked(
                RobustPlanStatus::NotEvaluated,
                "waypoint-bearing evidence is missing, stale or unqualified",
            );
        }
    };

    if request.wind.timebase_id != request.current.timebase_id
        || request.wind.timebase_id != request.target_bearing.timebase_id
    {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::NotEvaluated,
            "wind, current and waypoint evidence do not share a qualified timebase",
        );
    }

    if request.configuration_digest != polar.configuration_digest
        || request.polar_error.configuration_digest != polar.configuration_digest
    {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::OutOfScope,
            "craft or polar-error evidence belongs to a different configuration",
        );
    }

    let Some(polar_error) = request.polar_error.maximum_absolute_speed_error_mps else {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::NotEvaluated,
            "polar uncertainty is unknown; it cannot be treated as zero",
        );
    };
    if request.polar_error.source_artifact_digest == [0; 32]
        || !polar_error.is_finite()
        || polar_error < 0.0
    {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::NotEvaluated,
            "polar error bound is missing, malformed or unbound",
        );
    }

    if polar.validate().is_err() {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::NotEvaluated,
            "polar surface is malformed or lacks full TWA coverage",
        );
    }
    if !wind.from_direction.validate()
        || !wind.speed_mps.is_finite_ordered()
        || wind.speed_mps.lower <= 0.0
        || !current.toward_direction.validate()
        || !current.speed_mps.is_finite_ordered()
        || current.speed_mps.lower < 0.0
        || !target_bearing.is_finite()
        || !(0.0..360.0).contains(&target_bearing)
    {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::NotEvaluated,
            "one or more uncertainty intervals or navigation values are invalid",
        );
    }

    let wind_domain = ScalarInterval {
        lower: polar.true_wind_speeds_mps[0],
        upper: *polar.true_wind_speeds_mps.last().unwrap_or(&0.0),
    };
    if wind.speed_mps.lower < wind_domain.lower || wind.speed_mps.upper > wind_domain.upper {
        return RobustRouteEvaluation::blocked(
            RobustPlanStatus::OutOfScope,
            "true-wind uncertainty extends outside the polar domain; extrapolation is prohibited",
        );
    }

    let direct_twa = absolute_twa_interval(wind.from_direction, target_bearing);
    let direct_admissible = direct_twa.lower >= polar.no_go_angle_deg;
    let current_vmg = direction_projection(
        current.speed_mps,
        current.toward_direction,
        target_bearing,
    );

    let mut best_port = None;
    let mut best_starboard = None;

    for heading_integer in 0..360 {
        let heading_deg = f64::from(heading_integer);
        let twa = absolute_twa_interval(wind.from_direction, heading_deg);

        // A course is only robustly feasible if every wind direction in the
        // uncertainty set lies outside the craft's declared no-go sector.
        if twa.lower < polar.no_go_angle_deg {
            continue;
        }

        let boat_speed = match polar_speed_envelope(polar, wind.speed_mps, twa, polar_error) {
            Ok(value) => value,
            Err(PolarError::WindOutOfDomain) => {
                return RobustRouteEvaluation::blocked(
                    RobustPlanStatus::OutOfScope,
                    "wind/TWA uncertainty requests polar extrapolation",
                );
            }
            Err(PolarError::Invalid) => {
                return RobustRouteEvaluation::blocked(
                    RobustPlanStatus::NotEvaluated,
                    "polar uncertainty envelope could not be evaluated",
                );
            }
        };

        let heading_projection = (heading_deg - target_bearing).to_radians().cos();
        let boat_vmg = boat_speed.scale(heading_projection);
        let ground_vmg = boat_vmg.add(current_vmg);
        let signed_centre_twa =
            signed_angle_deg(wind.from_direction.center_deg, heading_deg);
        let tack_side = if signed_centre_twa > 0.0 {
            TackSide::Starboard
        } else if signed_centre_twa < 0.0 && signed_centre_twa > -180.0 {
            TackSide::Port
        } else {
            TackSide::DeadDownwind
        };
        if tack_side == TackSide::DeadDownwind {
            continue;
        }

        let candidate = RobustCourseCandidate {
            heading_deg,
            tack_side,
            true_wind_angle_deg: twa,
            boat_speed_mps: boat_speed,
            boat_velocity_made_good_mps: boat_vmg,
            current_velocity_made_good_mps: current_vmg,
            ground_velocity_made_good_mps: ground_vmg,
        };

        match tack_side {
            TackSide::Port => {
                best_port = best_for_tack(best_port, candidate);
            }
            TackSide::Starboard => {
                best_starboard = best_for_tack(best_starboard, candidate);
            }
            TackSide::DeadDownwind => {}
        }
    }

    let mut candidates = Vec::with_capacity(2);
    if let Some(candidate) = best_port {
        candidates.push(candidate);
    }
    if let Some(candidate) = best_starboard {
        candidates.push(candidate);
    }
    candidates.sort_by(|left, right| {
        right
            .ground_velocity_made_good_mps
            .lower
            .total_cmp(&left.ground_velocity_made_good_mps.lower)
    });

    if candidates.is_empty() {
        return RobustRouteEvaluation {
            status: RobustPlanStatus::NoRobustFeasibleCourse,
            direct_course_robustly_admissible: Some(direct_admissible),
            candidates,
            unique_best_tack: None,
            reason: "no heading remains outside the no-go sector for every declared wind direction",
        };
    }

    let guaranteed = candidates
        .iter()
        .filter(|candidate| candidate.ground_velocity_made_good_mps.lower > 0.0)
        .count();

    if guaranteed == 0 {
        return RobustRouteEvaluation {
            status: RobustPlanStatus::NoGuaranteedProgress,
            direct_course_robustly_admissible: Some(direct_admissible),
            candidates,
            unique_best_tack: None,
            reason: "no candidate guarantees positive ground VMG across the declared uncertainty bounds",
        };
    }

    let unique_best_tack = if candidates.len() == 1 {
        Some(candidates[0].tack_side)
    } else {
        let best = candidates[0];
        let other = candidates[1];
        (best.ground_velocity_made_good_mps.lower
            > other.ground_velocity_made_good_mps.upper)
            .then_some(best.tack_side)
    };

    RobustRouteEvaluation {
        status: if unique_best_tack.is_some() {
            RobustPlanStatus::RobustCandidatesAvailable
        } else {
            RobustPlanStatus::RankingUnresolved
        },
        direct_course_robustly_admissible: Some(direct_admissible),
        candidates,
        unique_best_tack,
        reason: if unique_best_tack.is_some() {
            "one tack dominates the alternative across the complete declared VMG envelopes"
        } else {
            "port/starboard VMG envelopes overlap; preserve both route candidates without forcing a winner"
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: [u8; 32] = [0x5a; 32];

    fn polar() -> PolarDiagram {
        PolarDiagram {
            configuration_digest: CONFIG,
            source_artifact_digest: [0xb6; 32],
            true_wind_speeds_mps: vec![4.0, 8.0, 12.0],
            true_wind_angles_deg: vec![0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0],
            boat_speed_mps: vec![
                vec![0.0, 0.8, 1.2, 1.8, 2.5, 2.1, 1.2, 0.8],
                vec![0.0, 1.6, 2.4, 3.6, 5.0, 4.2, 2.4, 1.6],
                vec![0.0, 2.4, 3.6, 5.4, 7.5, 6.3, 3.6, 2.4],
            ],
            no_go_angle_deg: 40.0,
        }
    }

    fn fresh<T>(value: T) -> Evidence<T> {
        Evidence::qualified_with_freshness(
            value,
            [0xc7; 32],
            "robust-synthetic-clock",
            0.0,
            2.0,
        )
    }

    fn request() -> RobustRouteRequest {
        RobustRouteRequest {
            configuration_digest: CONFIG,
            wind: fresh(WindUncertainty {
                from_direction: DirectionInterval {
                    center_deg: 0.0,
                    half_width_deg: 0.0,
                },
                speed_mps: ScalarInterval {
                    lower: 8.0,
                    upper: 8.0,
                },
            }),
            current: fresh(CurrentUncertainty {
                toward_direction: DirectionInterval {
                    center_deg: 90.0,
                    half_width_deg: 0.0,
                },
                speed_mps: ScalarInterval {
                    lower: 0.0,
                    upper: 0.0,
                },
            }),
            target_bearing: fresh(0.0),
            polar_error: PolarErrorBound {
                source_artifact_digest: [0xd8; 32],
                configuration_digest: CONFIG,
                maximum_absolute_speed_error_mps: Some(0.0),
            },
        }
    }

    #[test]
    fn robust_upwind_plan_preserves_both_tacks_when_vmg_envelopes_overlap() {
        let result = evaluate_robust_route_candidates(&polar(), request());
        assert_eq!(result.status, RobustPlanStatus::RankingUnresolved);
        assert_eq!(result.direct_course_robustly_admissible, Some(false));
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.unique_best_tack, None);
        assert!(result.candidates.iter().all(|candidate| {
            candidate.ground_velocity_made_good_mps.lower > 0.0
        }));
    }

    #[test]
    fn wind_direction_uncertainty_intersecting_no_go_rejects_candidate() {
        let mut input = request();
        input.wind.value.as_mut().unwrap().from_direction.half_width_deg = 15.0;
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert_eq!(result.status, RobustPlanStatus::RankingUnresolved);
        assert!(result
            .candidates
            .iter()
            .all(|candidate| candidate.true_wind_angle_deg.lower >= 40.0));
    }

    #[test]
    fn polar_uncertainty_is_not_silently_assumed_to_be_zero() {
        let mut input = request();
        input.polar_error.maximum_absolute_speed_error_mps = None;
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert_eq!(result.status, RobustPlanStatus::NotEvaluated);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn model_error_bound_is_applied_to_both_speed_extrema() {
        let mut input = request();
        input.polar_error.maximum_absolute_speed_error_mps = Some(0.25);
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert!(result
            .candidates
            .iter()
            .all(|candidate| candidate.boat_speed_mps.lower >= 0.0));
        assert!(result.candidates.iter().any(|candidate| {
            (candidate.boat_speed_mps.upper - candidate.boat_speed_mps.lower) >= 0.5
        }));
    }

    #[test]
    fn opposite_heading_uses_upper_boat_speed_for_worst_case_negative_vmg() {
        let mut data = polar();
        data.no_go_angle_deg = 10.0;
        let result = polar_speed_envelope(
            &data,
            ScalarInterval {
                lower: 8.0,
                upper: 8.0,
            },
            ScalarInterval {
                lower: 90.0,
                upper: 90.0,
            },
            0.5,
        )
        .unwrap();
        let negative_projection = result.scale(-1.0);
        assert!(negative_projection.lower < -5.0);
        assert!(negative_projection.upper > -6.0);
    }

    #[test]
    fn polar_interval_crossing_a_grid_knot_finds_interior_minimum() {
        let mut data = polar();
        for row in &mut data.boat_speed_mps {
            row[3] = 4.0;
        }
        data.boat_speed_mps[0][3] = 5.0;
        data.boat_speed_mps[1][3] = 0.4;
        data.boat_speed_mps[2][3] = 5.0;
        let result = polar_speed_envelope(
            &data,
            ScalarInterval {
                lower: 4.0,
                upper: 12.0,
            },
            ScalarInterval {
                lower: 60.0,
                upper: 60.0,
            },
            0.0,
        )
        .unwrap();
        assert!((result.lower - 0.4).abs() < 1e-9);
        assert_eq!(result.upper, 5.0);
    }

    #[test]
    fn a_tack_is_not_guaranteed_when_the_current_can_reverse_ground_vmg() {
        let mut input = request();
        input.current.value.as_mut().unwrap().speed_mps = ScalarInterval {
            lower: 0.0,
            upper: 4.0,
        };
        input.current.value.as_mut().unwrap().toward_direction.center_deg = 180.0;
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert_eq!(result.status, RobustPlanStatus::NoGuaranteedProgress);
        assert!(result.candidates.iter().all(|candidate| {
            candidate.ground_velocity_made_good_mps.lower <= 0.0
        }));
    }

    #[test]
    fn true_wind_interval_outside_polar_domain_is_out_of_scope() {
        let mut input = request();
        input.wind.value.as_mut().unwrap().speed_mps.upper = 13.0;
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert_eq!(result.status, RobustPlanStatus::OutOfScope);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn polar_error_bound_from_another_configuration_is_out_of_scope() {
        let mut input = request();
        input.polar_error.configuration_digest = [0x33; 32];
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert_eq!(result.status, RobustPlanStatus::OutOfScope);
    }

    #[test]
    fn timebase_conflict_still_fails_closed() {
        let mut input = request();
        input.current.timebase_id = "other-clock".to_string();
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert_eq!(result.status, RobustPlanStatus::NotEvaluated);
    }

    #[test]
    fn interval_directions_crossing_north_are_handled_circularly() {
        let mut input = request();
        input.wind.value.as_mut().unwrap().from_direction.center_deg = 359.0;
        input.wind.value.as_mut().unwrap().from_direction.half_width_deg = 5.0;
        let result = evaluate_robust_route_candidates(&polar(), input);
        assert!(result.candidates.iter().all(|candidate| {
            candidate.true_wind_angle_deg.lower >= polar().no_go_angle_deg
        }));
    }
}

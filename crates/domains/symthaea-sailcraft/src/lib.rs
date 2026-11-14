// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bounded point-of-sail and route-candidate kernel.
//!
//! This P0 crate consumes a craft-specific polar surface and qualified
//! wind/current/navigation observations. It does not simulate sail or hull
//! forces, train a policy, or emit helm/sheet actuator commands.
//!
//! All bearings use degrees clockwise from true north. Wind direction is the
//! direction the wind comes FROM. Current direction is the direction the
//! current flows TOWARD. Polar angles are absolute true-wind angles (TWA).

#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceState {
    Qualified,
    Missing,
    Stale,
    Unverified,
    Conflicted,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evidence<T> {
    pub state: EvidenceState,
    pub value: Option<T>,
    /// Digest of the exact source artifact; a digest is an identifier, not
    /// proof of publisher authenticity or physical validity.
    pub source_artifact_digest: [u8; 32],
    /// All P0 observations must be transformed to the same qualified timebase.
    pub timebase_id: String,
    pub age_seconds: f64,
    pub maximum_age_seconds: f64,
}

impl<T> Evidence<T> {
    pub fn qualified(value: T) -> Self {
        Self::qualified_with_freshness(
            value,
            [0xa5; 32],
            "synthetic-common-timebase",
            0.0,
            0.0,
        )
    }

    pub fn qualified_with_freshness(
        value: T,
        source_artifact_digest: [u8; 32],
        timebase_id: impl Into<String>,
        age_seconds: f64,
        maximum_age_seconds: f64,
    ) -> Self {
        Self {
            state: EvidenceState::Qualified,
            value: Some(value),
            source_artifact_digest,
            timebase_id: timebase_id.into(),
            age_seconds,
            maximum_age_seconds,
        }
    }

    pub fn missing() -> Self {
        Self {
            state: EvidenceState::Missing,
            value: None,
            source_artifact_digest: [0; 32],
            timebase_id: String::new(),
            age_seconds: 0.0,
            maximum_age_seconds: 0.0,
        }
    }

    pub fn stale(value: T) -> Self {
        let mut evidence = Self::qualified(value);
        evidence.state = EvidenceState::Stale;
        evidence
    }

    pub fn unverified(value: T) -> Self {
        let mut evidence = Self::qualified(value);
        evidence.state = EvidenceState::Unverified;
        evidence
    }

    pub fn conflicted(value: T) -> Self {
        let mut evidence = Self::qualified(value);
        evidence.state = EvidenceState::Conflicted;
        evidence
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindObservation {
    /// Direction the wind comes FROM, clockwise from true north.
    pub from_direction_deg: f64,
    pub speed_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurrentObservation {
    /// Direction the current flows TOWARD, clockwise from true north.
    pub toward_direction_deg: f64,
    pub speed_mps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolarDiagram {
    /// Exact craft + sail-plan + loading configuration identity.
    pub configuration_digest: [u8; 32],
    /// Exact polar source artifact identity (not a publisher attestation).
    pub source_artifact_digest: [u8; 32],
    /// Ordered true-wind-speed rows in metres per second.
    pub true_wind_speeds_mps: Vec<f64>,
    /// Ordered absolute true-wind angles in degrees; must span 0..=180.
    pub true_wind_angles_deg: Vec<f64>,
    /// Boat speed in m/s: [wind-speed row][TWA column].
    pub boat_speed_mps: Vec<Vec<f64>>,
    /// Craft-specific half-angle of the no-go sector, in degrees.
    pub no_go_angle_deg: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteRequest {
    pub configuration_digest: [u8; 32],
    pub wind: Evidence<WindObservation>,
    pub current: Evidence<CurrentObservation>,
    /// Bearing from the ownship's current location to the active waypoint.
    pub target_bearing: Evidence<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TackSide {
    Port,
    Starboard,
    DeadDownwind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CourseCandidate {
    /// Advisory course candidate only; never an actuator command.
    pub heading_deg: f64,
    /// Signed true-wind angle relative to this candidate heading.
    /// Positive means wind FROM starboard; negative means FROM port.
    pub signed_true_wind_angle_deg: f64,
    pub tack_side: TackSide,
    pub predicted_boat_speed_mps: f64,
    /// Ground-velocity projection toward the waypoint, including current.
    pub velocity_made_good_mps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanStatus {
    CandidatesAvailable,
    NoFeasibleProgress,
    NotEvaluated,
    Conflicted,
    OutOfScope,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteEvaluation {
    pub status: PlanStatus,
    pub direct_course_inside_no_go_sector: Option<bool>,
    /// At most one best candidate per tack side, ordered by VMG descending.
    pub candidates: Vec<CourseCandidate>,
    pub reason: &'static str,
}

impl RouteEvaluation {
    fn blocked(status: PlanStatus, reason: &'static str) -> Self {
        Self {
            status,
            direct_course_inside_no_go_sector: None,
            candidates: Vec::new(),
            reason,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolarError {
    Invalid,
    WindOutOfDomain,
}

impl PolarDiagram {
    fn validate(&self) -> Result<(), PolarError> {
        if self.configuration_digest == [0; 32]
            || self.source_artifact_digest == [0; 32]
            || !self.no_go_angle_deg.is_finite()
            || self.no_go_angle_deg <= 0.0
            || self.no_go_angle_deg >= 90.0
            || self.true_wind_speeds_mps.len() < 2
            || self.true_wind_angles_deg.len() < 2
            || self.boat_speed_mps.len() != self.true_wind_speeds_mps.len()
            || self.true_wind_angles_deg.first().copied() != Some(0.0)
            || self.true_wind_angles_deg.last().copied() != Some(180.0)
        {
            return Err(PolarError::Invalid);
        }

        if !strictly_increasing_nonnegative(&self.true_wind_speeds_mps)
            || !strictly_increasing_nonnegative(&self.true_wind_angles_deg)
            || self.true_wind_angles_deg.iter().any(|angle| *angle > 180.0)
            || self.boat_speed_mps.iter().any(|row| {
                row.len() != self.true_wind_angles_deg.len()
                    || row.iter().any(|speed| !speed.is_finite() || *speed < 0.0)
            })
        {
            return Err(PolarError::Invalid);
        }

        Ok(())
    }

    fn speed_at(&self, wind_speed_mps: f64, twa_deg: f64) -> Result<f64, PolarError> {
        let (wind_low, wind_fraction) = bracket(&self.true_wind_speeds_mps, wind_speed_mps)
            .ok_or(PolarError::WindOutOfDomain)?;
        let (angle_low, angle_fraction) = bracket(&self.true_wind_angles_deg, twa_deg)
            .ok_or(PolarError::Invalid)?;

        let wind_high = (wind_low + 1).min(self.true_wind_speeds_mps.len() - 1);
        let angle_high = (angle_low + 1).min(self.true_wind_angles_deg.len() - 1);

        let s00 = self.boat_speed_mps[wind_low][angle_low];
        let s01 = self.boat_speed_mps[wind_low][angle_high];
        let s10 = self.boat_speed_mps[wind_high][angle_low];
        let s11 = self.boat_speed_mps[wind_high][angle_high];

        let lower = lerp(s00, s01, angle_fraction);
        let upper = lerp(s10, s11, angle_fraction);
        Ok(lerp(lower, upper, wind_fraction))
    }
}

fn strictly_increasing_nonnegative(values: &[f64]) -> bool {
    values.len() >= 2
        && values.iter().all(|value| value.is_finite() && *value >= 0.0)
        && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn bracket(values: &[f64], query: f64) -> Option<(usize, f64)> {
    if !query.is_finite()
        || values.is_empty()
        || query < values[0]
        || query > *values.last()?
    {
        return None;
    }

    if query == *values.last()? {
        return Some((values.len() - 2, 1.0));
    }

    for index in 0..values.len() - 1 {
        let lower = values[index];
        let upper = values[index + 1];
        if (lower..=upper).contains(&query) {
            let fraction = (query - lower) / (upper - lower);
            return Some((index, fraction));
        }
    }

    None
}

fn lerp(a: f64, b: f64, fraction: f64) -> f64 {
    a + (b - a) * fraction
}

fn normalize_bearing(degrees: f64) -> f64 {
    degrees.rem_euclid(360.0)
}

fn signed_angle_deg(from_bearing: f64, to_bearing: f64) -> f64 {
    (from_bearing - to_bearing + 180.0).rem_euclid(360.0) - 180.0
}

fn tack_side(signed_twa_deg: f64) -> TackSide {
    if signed_twa_deg > 0.0 {
        TackSide::Starboard
    } else if signed_twa_deg < 0.0 && signed_twa_deg > -180.0 {
        TackSide::Port
    } else {
        TackSide::DeadDownwind
    }
}

fn ground_vmg(
    boat_speed_mps: f64,
    heading_deg: f64,
    current: CurrentObservation,
    target_bearing_deg: f64,
) -> f64 {
    let heading = heading_deg.to_radians();
    let current_heading = current.toward_direction_deg.to_radians();
    let target = target_bearing_deg.to_radians();

    let north_velocity =
        boat_speed_mps * heading.cos() + current.speed_mps * current_heading.cos();
    let east_velocity =
        boat_speed_mps * heading.sin() + current.speed_mps * current_heading.sin();

    north_velocity * target.cos() + east_velocity * target.sin()
}

fn evidence_gate<T>(evidence: &Evidence<T>) -> Result<&T, PlanStatus> {
    match evidence.state {
        EvidenceState::Conflicted => Err(PlanStatus::Conflicted),
        EvidenceState::Qualified => {
            let source_digest_valid = evidence.source_artifact_digest != [0; 32];
            let timebase_valid = !evidence.timebase_id.trim().is_empty();
            let freshness_valid = evidence.age_seconds.is_finite()
                && evidence.maximum_age_seconds.is_finite()
                && evidence.age_seconds >= 0.0
                && evidence.maximum_age_seconds >= 0.0
                && evidence.age_seconds <= evidence.maximum_age_seconds;
            if !source_digest_valid || !timebase_valid || !freshness_valid {
                return Err(PlanStatus::NotEvaluated);
            }
            evidence.value.as_ref().ok_or(PlanStatus::NotEvaluated)
        }
        EvidenceState::Missing
        | EvidenceState::Stale
        | EvidenceState::Unverified => Err(PlanStatus::NotEvaluated),
    }
}

/// Generate advisory course candidates from a qualified craft-specific polar.
///
/// The search evaluates integer-degree headings, interpolates only inside the
/// declared polar domain, and returns at most one best candidate for each tack.
/// It never outputs rudder, sail-sheet, throttle or actuator commands.
pub fn evaluate_route_candidates(
    polar: &PolarDiagram,
    request: RouteRequest,
) -> RouteEvaluation {
    let wind = match evidence_gate(&request.wind) {
        Ok(value) => value,
        Err(status) => {
            return RouteEvaluation::blocked(
                status,
                "wind evidence is not qualified, current and unambiguous",
            );
        }
    };
    let current = match evidence_gate(&request.current) {
        Ok(value) => value,
        Err(status) => {
            return RouteEvaluation::blocked(
                status,
                "current evidence is not qualified, current and unambiguous",
            );
        }
    };
    let target_bearing = match evidence_gate(&request.target_bearing) {
        Ok(value) => value,
        Err(status) => {
            return RouteEvaluation::blocked(
                status,
                "waypoint-bearing evidence is not qualified and unambiguous",
            );
        }
    };

    if request.wind.timebase_id != request.current.timebase_id
        || request.wind.timebase_id != request.target_bearing.timebase_id
    {
        return RouteEvaluation::blocked(
            PlanStatus::NotEvaluated,
            "wind, current and waypoint observations do not share the qualified P0 timebase",
        );
    }

    if request.configuration_digest != polar.configuration_digest {
        return RouteEvaluation::blocked(
            PlanStatus::OutOfScope,
            "polar surface belongs to a different craft configuration",
        );
    }

    if polar.validate().is_err() {
        return RouteEvaluation::blocked(
            PlanStatus::NotEvaluated,
            "polar surface is malformed or lacks full declared TWA coverage",
        );
    }

    if !wind.from_direction_deg.is_finite()
        || !(0.0..360.0).contains(&wind.from_direction_deg)
        || !wind.speed_mps.is_finite()
        || wind.speed_mps <= 0.0
        || !current.toward_direction_deg.is_finite()
        || !(0.0..360.0).contains(&current.toward_direction_deg)
        || !current.speed_mps.is_finite()
        || current.speed_mps < 0.0
        || !target_bearing.is_finite()
        || !(0.0..360.0).contains(&target_bearing)
    {
        return RouteEvaluation::blocked(
            PlanStatus::NotEvaluated,
            "one or more wind, current or navigation values are outside the typed contract",
        );
    }

    if wind.speed_mps < polar.true_wind_speeds_mps[0]
        || wind.speed_mps > *polar.true_wind_speeds_mps.last().unwrap_or(&0.0)
    {
        return RouteEvaluation::blocked(
            PlanStatus::OutOfScope,
            "true-wind speed is outside the polar surface domain; extrapolation is prohibited",
        );
    }

    let direct_twa =
        signed_angle_deg(wind.from_direction_deg, target_bearing).abs();
    let direct_inside_no_go = direct_twa < polar.no_go_angle_deg;

    let mut best_port: Option<CourseCandidate> = None;
    let mut best_starboard: Option<CourseCandidate> = None;

    // One-degree search is an explicit P0 numerical policy, not a steering rate.
    for heading_integer in 0..360 {
        let heading_deg = f64::from(heading_integer);
        let signed_twa =
            signed_angle_deg(wind.from_direction_deg, heading_deg);
        let absolute_twa = signed_twa.abs();

        if absolute_twa < polar.no_go_angle_deg {
            continue;
        }

        let boat_speed = match polar.speed_at(wind.speed_mps, absolute_twa) {
            Ok(value) => value,
            Err(PolarError::WindOutOfDomain) => {
                return RouteEvaluation::blocked(
                    PlanStatus::OutOfScope,
                    "polar interpolation requested outside its declared wind domain",
                );
            }
            Err(PolarError::Invalid) => {
                return RouteEvaluation::blocked(
                    PlanStatus::NotEvaluated,
                    "polar interpolation could not be evaluated",
                );
            }
        };

        let candidate = CourseCandidate {
            heading_deg,
            signed_true_wind_angle_deg: signed_twa,
            tack_side: tack_side(signed_twa),
            predicted_boat_speed_mps: boat_speed,
            velocity_made_good_mps: ground_vmg(
                boat_speed,
                heading_deg,
                current,
                target_bearing,
            ),
        };

        let slot = match candidate.tack_side {
            TackSide::Port => &mut best_port,
            TackSide::Starboard => &mut best_starboard,
            TackSide::DeadDownwind => continue,
        };

        if slot.is_none_or(|previous| {
            candidate.velocity_made_good_mps > previous.velocity_made_good_mps
        }) {
            *slot = Some(candidate);
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
            .velocity_made_good_mps
            .total_cmp(&left.velocity_made_good_mps)
    });

    if candidates.is_empty() {
        return RouteEvaluation {
            status: PlanStatus::NoFeasibleProgress,
            direct_course_inside_no_go_sector: Some(direct_inside_no_go),
            candidates,
            reason: "no candidate course was admitted by the declared polar surface",
        };
    }

    let has_positive_progress = candidates
        .iter()
        .any(|candidate| candidate.velocity_made_good_mps > 0.0);

    RouteEvaluation {
        status: if has_positive_progress {
            PlanStatus::CandidatesAvailable
        } else {
            PlanStatus::NoFeasibleProgress
        },
        direct_course_inside_no_go_sector: Some(direct_inside_no_go),
        candidates,
        reason: if has_positive_progress {
            "advisory port/starboard course candidates computed within the supplied polar domain"
        } else {
            "all admitted course candidates have non-positive predicted VMG"
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
            // Synthetic fixture value only; not a universal sailing threshold.
            no_go_angle_deg: 40.0,
        }
    }

    fn request(target_bearing_deg: f64) -> RouteRequest {
        RouteRequest {
            configuration_digest: CONFIG,
            wind: Evidence::qualified(WindObservation {
                from_direction_deg: 0.0,
                speed_mps: 8.0,
            }),
            current: Evidence::qualified(CurrentObservation {
                toward_direction_deg: 90.0,
                speed_mps: 0.0,
            }),
            target_bearing: Evidence::qualified(target_bearing_deg),
        }
    }

    #[test]
    fn head_to_wind_target_rejects_direct_course_and_returns_both_tacks() {
        let result = evaluate_route_candidates(&polar(), request(0.0));
        assert_eq!(result.status, PlanStatus::CandidatesAvailable);
        assert_eq!(result.direct_course_inside_no_go_sector, Some(true));
        assert_eq!(result.candidates.len(), 2);
        assert!(result.candidates.iter().any(|c| c.tack_side == TackSide::Port));
        assert!(result
            .candidates
            .iter()
            .any(|c| c.tack_side == TackSide::Starboard));
        assert!(result
            .candidates
            .iter()
            .all(|c| c.heading_deg != 0.0 && c.velocity_made_good_mps > 0.0));
    }

    #[test]
    fn reaching_target_is_not_confused_with_sailing_directly() {
        let result = evaluate_route_candidates(&polar(), request(90.0));
        assert_eq!(result.status, PlanStatus::CandidatesAvailable);
        assert_eq!(result.direct_course_inside_no_go_sector, Some(false));
    }

    #[test]
    fn stale_wind_never_produces_a_candidate() {
        let mut input = request(0.0);
        input.wind = Evidence::stale(WindObservation {
            from_direction_deg: 0.0,
            speed_mps: 8.0,
        });
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::NotEvaluated);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn conflicting_current_never_produces_a_candidate() {
        let mut input = request(0.0);
        input.current = Evidence::conflicted(CurrentObservation {
            toward_direction_deg: 90.0,
            speed_mps: 0.2,
        });
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::Conflicted);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn missing_waypoint_bearing_is_not_assumed() {
        let mut input = request(0.0);
        input.target_bearing = Evidence::missing();
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::NotEvaluated);
    }

    #[test]
    fn a_polar_from_another_configuration_is_out_of_scope() {
        let mut input = request(0.0);
        input.configuration_digest = [0x33; 32];
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::OutOfScope);
    }

    #[test]
    fn wind_speed_outside_polar_domain_is_not_extrapolated() {
        let mut input = request(0.0);
        input.wind = Evidence::qualified(WindObservation {
            from_direction_deg: 0.0,
            speed_mps: 14.0,
        });
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::OutOfScope);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn opposing_current_can_remove_all_positive_progress() {
        let mut input = request(0.0);
        input.current = Evidence::qualified(CurrentObservation {
            toward_direction_deg: 180.0,
            speed_mps: 20.0,
        });
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::NoFeasibleProgress);
        assert!(!result.candidates.is_empty());
        assert!(result
            .candidates
            .iter()
            .all(|c| c.velocity_made_good_mps <= 0.0));
    }

    #[test]
    fn observations_must_share_a_common_qualified_timebase() {
        let mut input = request(0.0);
        input.current = Evidence::qualified_with_freshness(
            CurrentObservation {
                toward_direction_deg: 90.0,
                speed_mps: 0.0,
            },
            [0xc7; 32],
            "different-clock-domain",
            0.0,
            1.0,
        );
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::NotEvaluated);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn evidence_older_than_its_declared_freshness_window_is_rejected() {
        let mut input = request(0.0);
        input.wind = Evidence::qualified_with_freshness(
            WindObservation {
                from_direction_deg: 0.0,
                speed_mps: 8.0,
            },
            [0xd8; 32],
            "synthetic-common-timebase",
            2.0,
            1.0,
        );
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::NotEvaluated);
    }

    #[test]
    fn zero_source_digest_is_not_qualified_evidence() {
        let mut input = request(0.0);
        input.wind = Evidence::qualified_with_freshness(
            WindObservation {
                from_direction_deg: 0.0,
                speed_mps: 8.0,
            },
            [0; 32],
            "synthetic-common-timebase",
            0.0,
            1.0,
        );
        let result = evaluate_route_candidates(&polar(), input);
        assert_eq!(result.status, PlanStatus::NotEvaluated);
    }

    #[test]
    fn invalid_polar_shapes_fail_closed() {
        let mut data = polar();
        data.boat_speed_mps[1].pop();
        let result = evaluate_route_candidates(&data, request(0.0));
        assert_eq!(result.status, PlanStatus::NotEvaluated);
    }

    #[test]
    fn polar_interpolation_is_bilinear_and_does_not_use_extrapolation() {
        let data = polar();
        assert!((data.speed_at(6.0, 75.0).unwrap() - 3.225).abs() < 1e-9);
        assert_eq!(data.speed_at(13.0, 75.0), Err(PolarError::WindOutOfDomain));
    }

    #[test]
    fn bearing_wraparound_is_consistent() {
        assert!((signed_angle_deg(359.0, 1.0) + 2.0).abs() < 1e-12);
        assert!((signed_angle_deg(1.0, 359.0) - 2.0).abs() < 1e-12);
    }
}

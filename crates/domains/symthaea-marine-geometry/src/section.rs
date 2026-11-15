//! Deterministic G2a section-family generator.
//!
//! This module expands bounded, dimensionless half-section control polylines
//! into physical starboard-side section samples. It does not loft a hull,
//! create a watertight surface, perform CAD fairing, calculate hydrostatics,
//! detect self-intersections, or qualify manufacturing feasibility.

use serde::{Deserialize, Serialize};

use crate::CoordinateFrame;

const SECTION_SCHEMA_VERSION: u32 = 1;
const ENDPOINT_TOLERANCE: f64 = 1.0e-12;
const MAX_STATIONS: usize = 512;
const MAX_CONTROL_POINTS_PER_STATION: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionHullSpec {
    pub schema_version: u32,
    pub candidate_id: String,
    /// Principal dimensions are expressed in metres.
    pub length_m: f64,
    pub beam_m: f64,
    pub draft_m: f64,
    pub coordinate_frame: CoordinateFrame,
    /// Number of linear samples per control-polyline segment (1..=64).
    pub samples_per_segment: u32,
    /// Longitudinal stations must start at 0, end at 1 and increase strictly.
    pub stations: Vec<SectionStation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionStation {
    pub x_over_length: f64,
    /// Local maximum half-breadth divided by the design maximum half-beam.
    pub half_breadth_fraction: f64,
    /// Local section depth divided by the design draft.
    pub draft_fraction: f64,
    /// Monotone keel-to-sheer half-section in normalized breadth/depth.
    /// First point is [0, 1]; last point is [1, 0].
    pub control_points: Vec<SectionControlPoint>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionControlPoint {
    pub breadth_fraction: f64,
    pub depth_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampledSection {
    pub station_index: usize,
    pub x_m: f64,
    /// Starboard-side samples from keel/centreline to sheer.
    /// Coordinates use x-forward, y-starboard, z-down and metres.
    pub keel_to_sheer_m: Vec<[f64; 3]>,
}

fn violation(violations: &mut Vec<SectionViolation>, code: &str, detail: impl Into<String>) {
    violations.push(SectionViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn close_enough(left: f64, right: f64) -> bool {
    (left - right).abs() <= ENDPOINT_TOLERANCE
}

/// Validate the bounded section-family schema without generating geometry.
pub fn validate_section_spec(spec: &SectionHullSpec) -> Result<(), Vec<SectionViolation>> {
    let mut violations = Vec::new();

    if spec.schema_version != SECTION_SCHEMA_VERSION {
        violation(
            &mut violations,
            "schema_version",
            format!(
                "section schema {} is unsupported; expected {}",
                spec.schema_version, SECTION_SCHEMA_VERSION
            ),
        );
    }

    if spec.candidate_id.trim().is_empty() {
        violation(&mut violations, "candidate_id", "candidate_id must not be empty");
    }

    for (code, value) in [
        ("length_m", spec.length_m),
        ("beam_m", spec.beam_m),
        ("draft_m", spec.draft_m),
    ] {
        if !finite_positive(value) {
            violation(
                &mut violations,
                code,
                format!("{code} must be finite and strictly positive"),
            );
        }
    }

    let frame = &spec.coordinate_frame;
    if frame.id.trim().is_empty()
        || frame.handedness != "right-handed"
        || !frame.axis_convention.contains("x-forward")
        || !frame.axis_convention.contains("y-starboard")
        || !frame.axis_convention.contains("z-down")
    {
        violation(
            &mut violations,
            "coordinate_frame",
            "G2a requires a named right-handed x-forward, y-starboard, z-down frame; no transform is inferred",
        );
    }

    if !(1..=64).contains(&spec.samples_per_segment) {
        violation(
            &mut violations,
            "samples_per_segment",
            "samples_per_segment must be between 1 and 64",
        );
    }

    if spec.stations.len() < 2 {
        violation(
            &mut violations,
            "station_count",
            "at least two longitudinal stations are required",
        );
    }
    if spec.stations.len() > MAX_STATIONS {
        violation(
            &mut violations,
            "station_count_limit",
            format!("station count exceeds the supported maximum of {MAX_STATIONS}"),
        );
    }

    if spec.stations.len() >= 2 {
        if !close_enough(spec.stations[0].x_over_length, 0.0) {
            violation(
                &mut violations,
                "first_station",
                "the first station must be at x/L = 0",
            );
        }
        if !close_enough(
            spec.stations[spec.stations.len() - 1].x_over_length,
            1.0,
        ) {
            violation(
                &mut violations,
                "last_station",
                "the last station must be at x/L = 1",
            );
        }
    }

    let mut previous_x: Option<f64> = None;
    for (station_index, station) in spec.stations.iter().enumerate() {
        if !station.x_over_length.is_finite()
            || !(0.0..=1.0).contains(&station.x_over_length)
        {
            violation(
                &mut violations,
                "station_x",
                format!("station {station_index} x/L must be finite and within [0, 1]"),
            );
        }
        if let Some(previous) = previous_x {
            if station.x_over_length <= previous {
                violation(
                    &mut violations,
                    "station_order",
                    format!("station {station_index} must be strictly aft of its predecessor"),
                );
            }
        }
        previous_x = Some(station.x_over_length);

        if !station.half_breadth_fraction.is_finite()
            || !(0.0..=1.0).contains(&station.half_breadth_fraction)
        {
            violation(
                &mut violations,
                "half_breadth_fraction",
                format!("station {station_index} half-breadth fraction must lie in [0, 1]"),
            );
        }
        if !station.draft_fraction.is_finite()
            || !(0.0..=1.0).contains(&station.draft_fraction)
            || station.draft_fraction <= 0.0
        {
            violation(
                &mut violations,
                "draft_fraction",
                format!("station {station_index} draft fraction must lie in (0, 1]"),
            );
        }

        if station.control_points.len() < 2 {
            violation(
                &mut violations,
                "section_control_count",
                format!("station {station_index} needs at least two half-section control points"),
            );
            continue;
        }
        if station.control_points.len() > MAX_CONTROL_POINTS_PER_STATION {
            violation(
                &mut violations,
                "section_control_count_limit",
                format!(
                    "station {station_index} exceeds the supported maximum of {MAX_CONTROL_POINTS_PER_STATION} control points"
                ),
            );
            continue;
        }

        let first = station.control_points[0];
        let last = station.control_points[station.control_points.len() - 1];
        if !close_enough(first.breadth_fraction, 0.0)
            || !close_enough(first.depth_fraction, 1.0)
        {
            violation(
                &mut violations,
                "keel_endpoint",
                format!("station {station_index} first control point must be keel [0, 1]"),
            );
        }
        if !close_enough(last.breadth_fraction, 1.0)
            || !close_enough(last.depth_fraction, 0.0)
        {
            violation(
                &mut violations,
                "sheer_endpoint",
                format!("station {station_index} last control point must be sheer [1, 0]"),
            );
        }

        for (point_index, point) in station.control_points.iter().enumerate() {
            if !point.breadth_fraction.is_finite()
                || !point.depth_fraction.is_finite()
                || !(0.0..=1.0).contains(&point.breadth_fraction)
                || !(0.0..=1.0).contains(&point.depth_fraction)
            {
                violation(
                    &mut violations,
                    "section_control_bounds",
                    format!(
                        "station {station_index} control point {point_index} must be finite and within the normalized unit square"
                    ),
                );
            }
        }

        for (pair_index, pair) in station.control_points.windows(2).enumerate() {
            let left = pair[0];
            let right = pair[1];
            if right.breadth_fraction < left.breadth_fraction
                || right.depth_fraction > left.depth_fraction
            {
                violation(
                    &mut violations,
                    "section_control_order",
                    format!(
                        "station {station_index} control segment {pair_index} must move monotonically from keel toward sheer"
                    ),
                );
            }
            if close_enough(left.breadth_fraction, right.breadth_fraction)
                && close_enough(left.depth_fraction, right.depth_fraction)
            {
                violation(
                    &mut violations,
                    "duplicate_section_control",
                    format!(
                        "station {station_index} control segment {pair_index} has duplicate consecutive points"
                    ),
                );
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Generate deterministic physical starboard-side section samples.
///
/// Each adjacent control-point pair is linearly sampled with
/// samples_per_segment subdivisions. The last control point is appended once.
/// This is a bounded polyline representation, not a fair spline.
pub fn generate_sections(
    spec: &SectionHullSpec,
) -> Result<Vec<SampledSection>, Vec<SectionViolation>> {
    validate_section_spec(spec)?;

    let mut output = Vec::with_capacity(spec.stations.len());

    for (station_index, station) in spec.stations.iter().enumerate() {
        let x_m = station.x_over_length * spec.length_m;
        let local_half_beam_m =
            0.5 * spec.beam_m * station.half_breadth_fraction;
        let local_draft_m = spec.draft_m * station.draft_fraction;

        let mut points = Vec::with_capacity(
            (station.control_points.len() - 1) * spec.samples_per_segment as usize + 1,
        );

        for pair in station.control_points.windows(2) {
            let start = pair[0];
            let end = pair[1];

            for step in 0..spec.samples_per_segment {
                let alpha = step as f64 / spec.samples_per_segment as f64;
                let breadth = start.breadth_fraction
                    + alpha * (end.breadth_fraction - start.breadth_fraction);
                let depth = start.depth_fraction
                    + alpha * (end.depth_fraction - start.depth_fraction);

                points.push([
                    x_m,
                    local_half_beam_m * breadth,
                    local_draft_m * depth,
                ]);
            }
        }

        let sheer = station
            .control_points
            .last()
            .expect("validated sections have at least two controls");
        points.push([
            x_m,
            local_half_beam_m * sheer.breadth_fraction,
            local_draft_m * sheer.depth_fraction,
        ]);

        output.push(SampledSection {
            station_index,
            x_m,
            keel_to_sheer_m: points,
        });
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_spec() -> SectionHullSpec {
        let shape = vec![
            SectionControlPoint {
                breadth_fraction: 0.0,
                depth_fraction: 1.0,
            },
            SectionControlPoint {
                breadth_fraction: 0.45,
                depth_fraction: 0.82,
            },
            SectionControlPoint {
                breadth_fraction: 0.82,
                depth_fraction: 0.36,
            },
            SectionControlPoint {
                breadth_fraction: 1.0,
                depth_fraction: 0.0,
            },
        ];
        SectionHullSpec {
            schema_version: 1,
            candidate_id: "section-fixture-001".into(),
            length_m: 100.0,
            beam_m: 20.0,
            draft_m: 6.0,
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            samples_per_segment: 4,
            stations: vec![
                SectionStation {
                    x_over_length: 0.0,
                    half_breadth_fraction: 0.05,
                    draft_fraction: 0.2,
                    control_points: shape.clone(),
                },
                SectionStation {
                    x_over_length: 0.5,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: shape.clone(),
                },
                SectionStation {
                    x_over_length: 1.0,
                    half_breadth_fraction: 0.05,
                    draft_fraction: 0.2,
                    control_points: shape,
                },
            ],
        }
    }

    #[test]
    fn produces_bounded_keel_to_sheer_sections() {
        let generated = generate_sections(&valid_spec()).unwrap();
        assert_eq!(generated.len(), 3);
        assert_eq!(generated[1].keel_to_sheer_m.len(), 13);
        assert_eq!(generated[1].x_m, 50.0);
        assert_eq!(generated[1].keel_to_sheer_m[0], [50.0, 0.0, 6.0]);
        assert_eq!(
            *generated[1].keel_to_sheer_m.last().unwrap(),
            [50.0, 10.0, 0.0]
        );
        assert!(generated.iter().all(|section| section
            .keel_to_sheer_m
            .iter()
            .all(|point| point.iter().all(|value| value.is_finite()))));
    }

    #[test]
    fn station_order_must_be_strict() {
        let mut spec = valid_spec();
        spec.stations[1].x_over_length = 0.0;
        let error = validate_section_spec(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "station_order"));
    }

    #[test]
    fn non_finite_dimensions_are_rejected() {
        let mut spec = valid_spec();
        spec.beam_m = f64::NAN;
        let error = generate_sections(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "beam_m"));
    }

    #[test]
    fn reversed_section_controls_are_rejected() {
        let mut spec = valid_spec();
        spec.stations[1].control_points[2].depth_fraction = 0.9;
        let error = validate_section_spec(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "section_control_order"));
    }

    #[test]
    fn wrong_frame_is_rejected_without_inferred_transform() {
        let mut spec = valid_spec();
        spec.coordinate_frame.axis_convention = "x-aft,y-port,z-up".into();
        let error = generate_sections(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "coordinate_frame"));
    }

    #[test]
    fn unsupported_sampling_resolution_is_rejected() {
        let mut spec = valid_spec();
        spec.samples_per_segment = 0;
        let error = validate_section_spec(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "samples_per_segment"));
    }

    #[test]
    fn generation_is_deterministic_for_identical_specification() {
        let spec = valid_spec();
        assert_eq!(
            generate_sections(&spec).unwrap(),
            generate_sections(&spec).unwrap()
        );
    }
}

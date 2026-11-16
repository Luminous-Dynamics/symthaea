//! Deterministic G2a section-family generator.
//!
//! This module expands bounded, dimensionless half-section control polylines
//! into physical starboard-side section samples. It does not loft a hull,
//! create a watertight surface, perform CAD fairing, calculate hydrostatics,
//! detect self-intersections, or qualify manufacturing feasibility.

use serde::{Deserialize, Serialize};

use crate::CoordinateFrame;

const SECTION_SCHEMA_VERSION: u32 = 2;
const ENDPOINT_TOLERANCE: f64 = 1.0e-12;
const MAX_STATIONS: usize = 512;
const MAX_CONTROL_POINTS_PER_STATION: usize = 128;
const MAX_TOTAL_OUTPUT_POINTS: usize = 1_000_000;

/// Declares how the first or last longitudinal section closes the displacement hull.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LongitudinalEndCondition {
    /// A positive-breadth section is closed by an explicit planar end cap.
    #[default]
    PlanarEndCap,
    /// The full keel-to-waterline sample column lies on y=0 and is shared by port/starboard.
    CollapsedCentreline,
}

/// Bow and stern are independent; interior stations are never collapsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalEndConditions {
    #[serde(default)]
    pub bow: LongitudinalEndCondition,
    #[serde(default)]
    pub stern: LongitudinalEndCondition,
}

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
    /// Missing field in v1 input JSON defaults to planar bow and stern.
    #[serde(default)]
    pub longitudinal_end_conditions: LongitudinalEndConditions,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionFamilyArtifact {
    pub artifact_schema_version: u32,
    pub candidate_id: String,
    pub source_spec_digest_blake3: String,
    pub sampled_sections_digest_blake3: String,
    #[serde(default)]
    pub section_family_manifest_digest_blake3: String,
    pub length_m: f64,
    pub beam_m: f64,
    pub draft_m: f64,
    pub units: String,
    pub coordinate_frame: CoordinateFrame,
    #[serde(default)]
    pub longitudinal_end_conditions: LongitudinalEndConditions,
    pub sections: Vec<SampledSection>,
}

#[derive(Serialize)]
struct SectionFamilyManifestIdentity<'a> {
    artifact_schema_version: u32,
    candidate_id: &'a str,
    source_spec_digest_blake3: &'a str,
    sampled_sections_digest_blake3: &'a str,
    length_m: f64,
    beam_m: f64,
    draft_m: f64,
    units: &'a str,
    coordinate_frame: &'a CoordinateFrame,
    longitudinal_end_conditions: LongitudinalEndConditions,
    station_count: usize,
}

fn compute_section_family_manifest_digest(
    artifact: &SectionFamilyArtifact,
) -> Result<String, serde_json::Error> {
    let identity = SectionFamilyManifestIdentity {
        artifact_schema_version: artifact.artifact_schema_version,
        candidate_id: &artifact.candidate_id,
        source_spec_digest_blake3: &artifact.source_spec_digest_blake3,
        sampled_sections_digest_blake3: &artifact.sampled_sections_digest_blake3,
        length_m: artifact.length_m,
        beam_m: artifact.beam_m,
        draft_m: artifact.draft_m,
        units: &artifact.units,
        coordinate_frame: &artifact.coordinate_frame,
        longitudinal_end_conditions: artifact.longitudinal_end_conditions,
        station_count: artifact.sections.len(),
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?)
        .to_hex()
        .to_string())
}

/// Verify section coordinates and all semantic metadata against their digests.
/// This is an integrity check, not proof that the source dimensions are correct.
pub fn verify_section_family_artifact_integrity(
    artifact: &SectionFamilyArtifact,
) -> Result<(), Vec<SectionViolation>> {
    let mut errors = Vec::new();
    if artifact.artifact_schema_version != 2 {
        violation(
            &mut errors,
            "artifact_schema_version",
            "section-family artifact schema v2 is required",
        );
    }
    if artifact.candidate_id.trim().is_empty() {
        violation(
            &mut errors,
            "candidate_id",
            "candidate ID must not be empty",
        );
    }
    if artifact.units != "m"
        || !finite_positive(artifact.length_m)
        || !finite_positive(artifact.beam_m)
        || !finite_positive(artifact.draft_m)
    {
        violation(
            &mut errors,
            "dimensions_or_units",
            "section-family dimensions must be finite, positive and expressed in metres",
        );
    }
    if artifact.source_spec_digest_blake3.len() != 64
        || !artifact
            .source_spec_digest_blake3
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        violation(
            &mut errors,
            "source_spec_digest",
            "source specification digest must be 64 hexadecimal characters",
        );
    }
    match compute_sampled_sections_digest(&artifact.sections) {
        Ok(actual) if actual.eq_ignore_ascii_case(&artifact.sampled_sections_digest_blake3) => {}
        Ok(_) => violation(
            &mut errors,
            "sampled_sections_digest",
            "sample coordinates do not match the declared sampled-sections digest",
        ),
        Err(error) => violation(
            &mut errors,
            "sampled_sections_digest",
            format!("sample coordinates cannot be serialized: {error}"),
        ),
    }
    let frame = &artifact.coordinate_frame;
    if frame.id.trim().is_empty()
        || frame.handedness != "right-handed"
        || !frame.axis_convention.contains("x-forward")
        || !frame.axis_convention.contains("y-starboard")
        || !frame.axis_convention.contains("z-down")
    {
        violation(
            &mut errors,
            "coordinate_frame",
            "section-family artifact requires explicit right-handed x-forward/y-starboard/z-down coordinates",
        );
    }
    if artifact.sections.len() < 2 || artifact.sections.len() > MAX_STATIONS {
        violation(
            &mut errors,
            "station_count",
            format!("section-family station count must be within 2..={MAX_STATIONS}"),
        );
    } else {
        for (index, section) in artifact.sections.iter().enumerate() {
            if section.station_index != index
                || !section.x_m.is_finite()
                || section.x_m < 0.0
                || section.x_m > artifact.length_m
                || section.keel_to_sheer_m.len() < 2
            {
                violation(
                    &mut errors,
                    "station_identity_or_coordinates",
                    format!("station {index} has invalid identity, x coordinate or sample count"),
                );
                continue;
            }
            if index > 0 && section.x_m <= artifact.sections[index - 1].x_m {
                violation(
                    &mut errors,
                    "station_order",
                    format!("station {index} is not strictly after its predecessor"),
                );
            }
            let endpoint = if index == 0 {
                Some(artifact.longitudinal_end_conditions.bow)
            } else if index + 1 == artifact.sections.len() {
                Some(artifact.longitudinal_end_conditions.stern)
            } else {
                None
            };
            let final_point = section.keel_to_sheer_m.last().copied().unwrap_or([0.0; 3]);
            match endpoint {
                Some(LongitudinalEndCondition::CollapsedCentreline)
                    if section.keel_to_sheer_m.iter().any(|point| point[1] != 0.0)
                        || final_point[1] != 0.0 =>
                {
                    violation(
                        &mut errors,
                        "collapsed_end_geometry",
                        format!(
                            "collapsed endpoint station {index} must have every transverse sample exactly on y=0"
                        ),
                    );
                }
                Some(LongitudinalEndCondition::PlanarEndCap) if final_point[1] <= 0.0 => {
                    violation(
                        &mut errors,
                        "planar_end_geometry",
                        format!(
                            "planar endpoint station {index} must have positive waterline half-breadth"
                        ),
                    );
                }
                None if final_point[1] <= 0.0 => {
                    violation(
                        &mut errors,
                        "interior_end_geometry",
                        format!(
                            "interior station {index} must have positive waterline half-breadth"
                        ),
                    );
                }
                _ => {}
            }
        }
        if artifact.sections.len() == 2
            && artifact.longitudinal_end_conditions.bow
                == LongitudinalEndCondition::CollapsedCentreline
            && artifact.longitudinal_end_conditions.stern
                == LongitudinalEndCondition::CollapsedCentreline
        {
            violation(
                &mut errors,
                "adjacent_collapsed_ends",
                "two collapsed endpoints require at least one positive-breadth interior station",
            );
        }
        if let (Some(first), Some(last)) = (artifact.sections.first(), artifact.sections.last()) {
            if !close_enough(first.x_m, 0.0) || !close_enough(last.x_m, artifact.length_m) {
                violation(
                    &mut errors,
                    "station_extent",
                    "section-family stations must span x=0 through the declared vessel length",
                );
            }
        }
    }

    match compute_section_family_manifest_digest(artifact) {
        Ok(actual)
            if actual.eq_ignore_ascii_case(&artifact.section_family_manifest_digest_blake3) => {}
        Ok(_) => violation(
            &mut errors,
            "section_family_manifest_digest",
            "section-family semantic metadata does not match its manifest digest",
        ),
        Err(error) => violation(
            &mut errors,
            "section_family_manifest_digest",
            format!("section-family metadata cannot be serialized: {error}"),
        ),
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
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

    if !(1..=SECTION_SCHEMA_VERSION).contains(&spec.schema_version) {
        violation(
            &mut violations,
            "schema_version",
            format!(
                "section schema {} is unsupported; supported versions are 1..={}",
                spec.schema_version, SECTION_SCHEMA_VERSION
            ),
        );
    }
    if spec.schema_version == 1
        && spec.longitudinal_end_conditions != LongitudinalEndConditions::default()
    {
        violation(
            &mut violations,
            "end_conditions_require_schema_v2",
            "non-planar longitudinal end conditions require section specification schema version 2",
        );
    }

    if spec.candidate_id.trim().is_empty() {
        violation(
            &mut violations,
            "candidate_id",
            "candidate_id must not be empty",
        );
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

    if spec.stations.len() == 2
        && spec.longitudinal_end_conditions.bow == LongitudinalEndCondition::CollapsedCentreline
        && spec.longitudinal_end_conditions.stern == LongitudinalEndCondition::CollapsedCentreline
    {
        violation(
            &mut violations,
            "adjacent_collapsed_ends",
            "two adjacent collapsed endpoints leave no positive interior station or displacement length",
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
        if !close_enough(spec.stations[spec.stations.len() - 1].x_over_length, 1.0) {
            violation(
                &mut violations,
                "last_station",
                "the last station must be at x/L = 1",
            );
        }
    }

    let mut previous_x: Option<f64> = None;
    let mut total_output_points = 0_usize;
    for (station_index, station) in spec.stations.iter().enumerate() {
        if !station.x_over_length.is_finite() || !(0.0..=1.0).contains(&station.x_over_length) {
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

        let declared_end = if station_index == 0 {
            Some(spec.longitudinal_end_conditions.bow)
        } else if station_index + 1 == spec.stations.len() {
            Some(spec.longitudinal_end_conditions.stern)
        } else {
            None
        };
        match declared_end {
            Some(LongitudinalEndCondition::CollapsedCentreline)
                if station.half_breadth_fraction != 0.0 =>
            {
                violation(
                    &mut violations,
                    "collapsed_end_breadth",
                    format!(
                        "collapsed endpoint station {station_index} must declare exactly zero half-breadth"
                    ),
                );
            }
            Some(LongitudinalEndCondition::PlanarEndCap)
                if station.half_breadth_fraction <= 0.0 =>
            {
                violation(
                    &mut violations,
                    "planar_end_breadth",
                    format!(
                        "planar endpoint station {station_index} must have positive half-breadth"
                    ),
                );
            }
            None if station.half_breadth_fraction <= 0.0 => {
                violation(
                    &mut violations,
                    "interior_station_breadth",
                    format!("interior station {station_index} must have positive half-breadth"),
                );
            }
            _ => {}
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
        if !close_enough(first.breadth_fraction, 0.0) || !close_enough(first.depth_fraction, 1.0) {
            violation(
                &mut violations,
                "keel_endpoint",
                format!("station {station_index} first control point must be keel [0, 1]"),
            );
        }
        if !close_enough(last.breadth_fraction, 1.0) || !close_enough(last.depth_fraction, 0.0) {
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

        let station_output_points = station
            .control_points
            .len()
            .checked_sub(1)
            .and_then(|segments| segments.checked_mul(spec.samples_per_segment as usize))
            .and_then(|points| points.checked_add(1));
        match station_output_points.and_then(|points| total_output_points.checked_add(points)) {
            Some(total) => {
                total_output_points = total;
                if total_output_points > MAX_TOTAL_OUTPUT_POINTS {
                    violation(
                        &mut violations,
                        "section_point_budget",
                        format!(
                            "generated section samples would exceed the {MAX_TOTAL_OUTPUT_POINTS}-point output budget"
                        ),
                    );
                }
            }
            None => violation(
                &mut violations,
                "section_point_budget",
                "generated section sample count overflows the supported size",
            ),
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Compute a deterministic digest of the normalized, typed section specification.
pub fn compute_section_spec_digest(spec: &SectionHullSpec) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(spec)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Compute a deterministic digest of generated section samples.
pub fn compute_sampled_sections_digest(
    sections: &[SampledSection],
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(sections)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
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
        let local_half_beam_m = 0.5 * spec.beam_m * station.half_breadth_fraction;
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
                let depth =
                    start.depth_fraction + alpha * (end.depth_fraction - start.depth_fraction);

                points.push([x_m, local_half_beam_m * breadth, local_draft_m * depth]);
            }
        }

        let Some(sheer) = station.control_points.last() else {
            return Err(vec![SectionViolation {
                code: "section_control_count".into(),
                detail: format!("station {station_index} lost its required control points"),
            }]);
        };
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

/// Generate a section-family artifact bound to both the exact specification
/// and the exact sampled coordinates. These hashes provide identity and
/// integrity, not signatures or proof of engineering correctness.
pub fn generate_section_artifact(
    spec: &SectionHullSpec,
) -> Result<SectionFamilyArtifact, Vec<SectionViolation>> {
    let sections = generate_sections(spec)?;
    let source_spec_digest_blake3 = compute_section_spec_digest(spec).map_err(|error| {
        vec![SectionViolation {
            code: "spec_serialization".into(),
            detail: format!("could not serialize section specification: {error}"),
        }]
    })?;
    let sampled_sections_digest_blake3 =
        compute_sampled_sections_digest(&sections).map_err(|error| {
            vec![SectionViolation {
                code: "sample_serialization".into(),
                detail: format!("could not serialize generated samples: {error}"),
            }]
        })?;

    let mut artifact = SectionFamilyArtifact {
        artifact_schema_version: 2,
        candidate_id: spec.candidate_id.clone(),
        source_spec_digest_blake3,
        sampled_sections_digest_blake3,
        section_family_manifest_digest_blake3: String::new(),
        length_m: spec.length_m,
        beam_m: spec.beam_m,
        draft_m: spec.draft_m,
        units: "m".into(),
        coordinate_frame: spec.coordinate_frame.clone(),
        longitudinal_end_conditions: spec.longitudinal_end_conditions,
        sections,
    };
    artifact.section_family_manifest_digest_blake3 =
        compute_section_family_manifest_digest(&artifact).map_err(|error| {
            vec![SectionViolation {
                code: "section_family_manifest_digest".into(),
                detail: format!("could not serialize section-family identity: {error}"),
            }]
        })?;
    Ok(artifact)
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
            longitudinal_end_conditions: LongitudinalEndConditions::default(),
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
        assert!(generated.iter().all(|section| {
            section
                .keel_to_sheer_m
                .iter()
                .all(|point| point.iter().all(|value| value.is_finite()))
        }));
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
        assert!(
            error
                .iter()
                .any(|item| item.code == "section_control_order")
        );
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
    fn section_artifact_binds_spec_and_generated_samples() {
        let spec = valid_spec();
        let first = generate_section_artifact(&spec).unwrap();
        let second = generate_section_artifact(&spec).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.source_spec_digest_blake3,
            compute_section_spec_digest(&spec).unwrap()
        );
        assert_eq!(
            first.sampled_sections_digest_blake3,
            compute_sampled_sections_digest(&first.sections).unwrap()
        );
        let mut changed = spec;
        changed.beam_m += 0.1;
        assert_ne!(
            first.source_spec_digest_blake3,
            compute_section_spec_digest(&changed).unwrap()
        );
    }

    #[test]
    fn oversized_sample_output_is_rejected_before_allocation() {
        let mut spec = valid_spec();
        spec.samples_per_segment = 64;
        let controls: Vec<SectionControlPoint> = (0..128)
            .map(|index| {
                let fraction = index as f64 / 127.0;
                SectionControlPoint {
                    breadth_fraction: fraction,
                    depth_fraction: 1.0 - fraction,
                }
            })
            .collect();
        spec.stations = (0..200)
            .map(|index| SectionStation {
                x_over_length: index as f64 / 199.0,
                half_breadth_fraction: 1.0,
                draft_fraction: 1.0,
                control_points: controls.clone(),
            })
            .collect();
        let error = validate_section_spec(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "section_point_budget"));
    }

    #[test]
    fn section_family_manifest_binds_longitudinal_end_conditions) {
        let mut artifact = generate_section_artifact(&valid_spec()).unwrap();
        verify_section_family_artifact_integrity(&artifact).unwrap();
        artifact.longitudinal_end_conditions.bow = LongitudinalEndCondition::CollapsedCentreline;
        let errors = verify_section_family_artifact_integrity(&artifact).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "section_family_manifest_digest")
        );
    }

    #[test]
    fn section_family_manifest_binds_dimensions_and_coordinate_frame() {
        let mut artifact = generate_section_artifact(&valid_spec()).unwrap();
        artifact.beam_m += 0.1;
        let errors = verify_section_family_artifact_integrity(&artifact).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "section_family_manifest_digest"));
    }

    #[test]
    fn collapsed_endpoint_semantics_are_checked_even_after_manifest_rebinding) {
        let mut spec = valid_spec();
        spec.schema_version = 2;
        spec.longitudinal_end_conditions.bow = LongitudinalEndCondition::CollapsedCentreline;
        spec.stations[0].half_breadth_fraction = 0.0;
        let mut artifact = generate_section_artifact(&spec).unwrap();

        artifact.sections[0].keel_to_sheer_m[2][1] = 0.125;
        artifact.sampled_sections_digest_blake3 =
            compute_sampled_sections_digest(&artifact.sections).unwrap();
        artifact.section_family_manifest_digest_blake3 =
            compute_section_family_manifest_digest(&artifact).unwrap();

        let errors = verify_section_family_artifact_integrity(&artifact).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "collapsed_end_geometry")
        );
    }

    #[test]
    fn legacy_v1_planar_section_specs_remain_supported() {
        let mut spec = valid_spec();
        spec.schema_version = 1;
        spec.longitudinal_end_conditions = LongitudinalEndConditions::default();
        assert!(generate_section_artifact(&spec).is_ok());
    }

    #[test]
    fn non_planar_end_conditions_require_spec_schema_v2() {
        let mut spec = valid_spec();
        spec.longitudinal_end_conditions.bow =
            LongitudinalEndCondition::CollapsedCentreline;
        spec.stations[0].half_breadth_fraction = 0.0;
        assert!(
            validate_section_spec(&spec)
                .unwrap_err()
                .iter()
                .any(|item| item.code == "end_conditions_require_schema_v2")
        );
    }

    #[test]
    fn legacy_v1_json_without_end_condition_field_defaults_to_planar) {
        let spec = valid_spec();
        let mut legacy = serde_json::to_value(&spec).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("longitudinal_end_conditions");
        let parsed: SectionHullSpec = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            parsed.longitudinal_end_conditions,
            LongitudinalEndConditions::default()
        );
        assert!(generate_section_artifact(&parsed).is_ok());
    }

    #[test]
    fn legacy_v1_section_artifact_parses_but_is_not_v2_admissible() {
        let artifact = generate_section_artifact(&valid_spec()).unwrap();
        let mut legacy = serde_json::to_value(&artifact).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.insert("artifact_schema_version".into(), serde_json::json!(1));
        object.remove("section_family_manifest_digest_blake3");
        let parsed: SectionFamilyArtifact = serde_json::from_value(legacy).unwrap();
        let errors = verify_section_family_artifact_integrity(&parsed).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "artifact_schema_version")
        );
    }

    #[test]
    fn collapsed_bow_requires_exact_zero_breadth() {
        let mut spec = valid_spec();
        spec.schema_version = 2;
        spec.longitudinal_end_conditions.bow =
            LongitudinalEndCondition::CollapsedCentreline;
        spec.stations[0].half_breadth_fraction = 0.0;
        let artifact = generate_section_artifact(&spec).unwrap();
        assert!(verify_section_family_artifact_integrity(&artifact).is_ok());
        assert!(
            artifact.sections[0]
                .keel_to_sheer_m
                .iter()
                .all(|point| point[1] == 0.0)
        );
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

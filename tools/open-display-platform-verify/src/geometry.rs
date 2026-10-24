// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Uncertainty-aware geometric screening for display measurement setup.
//!
//! The model assumes an ideal, centered rectangular angular field of view and a
//! planar target. It does not model real lens distortion, veiling glare, detector
//! angular response, spectral response, or calibrated optical performance.

use serde_json::Value;
use super::{EXPECTED_SCHEMA_DRAFT, sha256_hex};

#[derive(Debug, Clone, PartialEq)]
pub struct GeometryScreenInput {
    pub record_id: String,
    pub distance_mm: f64,
    pub horizontal_fov_deg: f64,
    pub vertical_fov_deg: f64,
    pub distance_standard_uncertainty_mm: f64,
    pub horizontal_fov_standard_uncertainty_deg: f64,
    pub vertical_fov_standard_uncertainty_deg: f64,
    pub coverage_factor: f64,
    pub uniform_patch_width_mm: f64,
    pub uniform_patch_height_mm: f64,
    pub centering_uncertainty_mm: f64,
    pub edge_guard_mm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometryScreenResult {
    pub record_id: String,
    pub nominal_footprint_width_mm: f64,
    pub nominal_footprint_height_mm: f64,
    pub horizontal_standard_uncertainty_mm: f64,
    pub vertical_standard_uncertainty_mm: f64,
    pub horizontal_expanded_uncertainty_mm: f64,
    pub vertical_expanded_uncertainty_mm: f64,
    pub conservative_footprint_width_mm: f64,
    pub conservative_footprint_height_mm: f64,
    pub total_required_width_mm: f64,
    pub total_required_height_mm: f64,
    pub horizontal_margin_mm: f64,
    pub vertical_margin_mm: f64,
    pub horizontal_utilization_fraction: f64,
    pub vertical_utilization_fraction: f64,
    pub uniform_patch_width_mm: f64,
    pub uniform_patch_height_mm: f64,
    pub horizontal_fits: bool,
    pub vertical_fits: bool,
    pub screen_status: String,
}

impl GeometryScreenResult {
    /// Stable report with a geometry-only claim ceiling.
    #[must_use]
    pub fn to_json(&self, input_sha256: &str, schema_sha256: &str) -> Value {
        serde_json::json!({
            "verifier": "open-display-platform-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "record_id": self.record_id,
            "model": "IDEAL_RECTANGULAR_FOV_PLANAR_TARGET",
            "screen_status": self.screen_status,
            "nominal_footprint_mm": {
                "width": self.nominal_footprint_width_mm,
                "height": self.nominal_footprint_height_mm
            },
            "first_order_standard_uncertainty_mm": {
                "width": self.horizontal_standard_uncertainty_mm,
                "height": self.vertical_standard_uncertainty_mm
            },
            "expanded_uncertainty_mm": {
                "width": self.horizontal_expanded_uncertainty_mm,
                "height": self.vertical_expanded_uncertainty_mm
            },
            "conservative_footprint_mm": {
                "width": self.conservative_footprint_width_mm,
                "height": self.conservative_footprint_height_mm
            },
            "required_patch_extent_mm": {
                "width": self.total_required_width_mm,
                "height": self.total_required_height_mm
            },
            "remaining_patch_margin_mm": {
                "width": self.horizontal_margin_mm,
                "height": self.vertical_margin_mm
            },
            "patch_utilization_fraction": {
                "width": self.horizontal_utilization_fraction,
                "height": self.vertical_utilization_fraction
            },
            "available_uniform_patch_mm": {
                "width": self.uniform_patch_width_mm,
                "height": self.uniform_patch_height_mm
            },
            "horizontal_fits": self.horizontal_fits,
            "vertical_fits": self.vertical_fits,
            "input_sha256": input_sha256,
            "schema_sha256": schema_sha256,
            "physical_validation": "NOT_PERFORMED",
            "claim_ceiling": "Geometric setup screening only; not a lens model, detector response model, calibrated measurement, or physical qualification."
        })
    }
}

/// Calculate a conservative footprint envelope and an independent first-order
/// standard-uncertainty estimate for the specified setup.
///
/// For the fit decision, both distance and angle are taken to their upper
/// coverage-factor bounds and the centering uncertainty and edge guard are
/// added explicitly. This is intentionally more conservative than deciding fit
/// from the linearized uncertainty alone.
///
/// # Errors
///
/// Returns validation errors when the geometry or uncertainty inputs are
/// missing, non-finite, physically invalid, or numerically unstable.
pub fn evaluate_geometry_screen(
    input: &GeometryScreenInput,
) -> Result<GeometryScreenResult, Vec<String>> {
    let mut errors = Vec::new();
    validate_positive(input.distance_mm, "distance_mm", &mut errors);
    validate_fov(input.horizontal_fov_deg, "horizontal_fov_deg", &mut errors);
    validate_fov(input.vertical_fov_deg, "vertical_fov_deg", &mut errors);
    validate_nonnegative(
        input.distance_standard_uncertainty_mm,
        "distance_standard_uncertainty_mm",
        &mut errors,
    );
    validate_nonnegative(
        input.horizontal_fov_standard_uncertainty_deg,
        "horizontal_fov_standard_uncertainty_deg",
        &mut errors,
    );
    validate_nonnegative(
        input.vertical_fov_standard_uncertainty_deg,
        "vertical_fov_standard_uncertainty_deg",
        &mut errors,
    );
    validate_positive(input.coverage_factor, "coverage_factor", &mut errors);
    validate_positive(
        input.uniform_patch_width_mm,
        "uniform_patch_width_mm",
        &mut errors,
    );
    validate_positive(
        input.uniform_patch_height_mm,
        "uniform_patch_height_mm",
        &mut errors,
    );
    validate_nonnegative(
        input.centering_uncertainty_mm,
        "centering_uncertainty_mm",
        &mut errors,
    );
    validate_nonnegative(input.edge_guard_mm, "edge_guard_mm", &mut errors);

    if input.record_id.trim().is_empty() {
        errors.push("record_id must not be empty".to_owned());
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let distance_upper = input.distance_mm
        + input.coverage_factor * input.distance_standard_uncertainty_mm;
    if !distance_upper.is_finite() || distance_upper <= 0.0 {
        errors.push("coverage-bounded distance must be finite and positive".to_owned());
    }

    let horizontal_angle_upper = input.horizontal_fov_deg
        + input.coverage_factor * input.horizontal_fov_standard_uncertainty_deg;
    let vertical_angle_upper = input.vertical_fov_deg
        + input.coverage_factor * input.vertical_fov_standard_uncertainty_deg;
    if !horizontal_angle_upper.is_finite() || horizontal_angle_upper >= 179.0 {
        errors.push("coverage-bounded horizontal FOV must remain below 179 degrees".to_owned());
    }
    if !vertical_angle_upper.is_finite() || vertical_angle_upper >= 179.0 {
        errors.push("coverage-bounded vertical FOV must remain below 179 degrees".to_owned());
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let horizontal_half_angle = input.horizontal_fov_deg.to_radians() / 2.0;
    let vertical_half_angle = input.vertical_fov_deg.to_radians() / 2.0;
    let nominal_width = 2.0 * input.distance_mm * horizontal_half_angle.tan();
    let nominal_height = 2.0 * input.distance_mm * vertical_half_angle.tan();

    // First-order propagation assumes independent distance and angular terms.
    // The angular derivatives are expressed per radian, so convert degrees.
    let distance_term_h = 2.0 * horizontal_half_angle.tan()
        * input.distance_standard_uncertainty_mm;
    let angle_term_h = input.distance_mm * (1.0 / horizontal_half_angle.cos().powi(2))
        * input.horizontal_fov_standard_uncertainty_deg.to_radians();
    let distance_term_v = 2.0 * vertical_half_angle.tan()
        * input.distance_standard_uncertainty_mm;
    let angle_term_v = input.distance_mm * (1.0 / vertical_half_angle.cos().powi(2))
        * input.vertical_fov_standard_uncertainty_deg.to_radians();

    let standard_h = distance_term_h.hypot(angle_term_h);
    let standard_v = distance_term_v.hypot(angle_term_v);
    let expanded_h = input.coverage_factor * standard_h;
    let expanded_v = input.coverage_factor * standard_v;

    // Exact upper envelope for the rectangular pinhole model, rather than
    // trusting the linearized uncertainty as a strict upper bound.
    let conservative_width = 2.0
        * distance_upper
        * (horizontal_angle_upper.to_radians() / 2.0).tan();
    let conservative_height = 2.0
        * distance_upper
        * (vertical_angle_upper.to_radians() / 2.0).tan();
    let guard_each_side = input.centering_uncertainty_mm + input.edge_guard_mm;
    let required_width = conservative_width + 2.0 * guard_each_side;
    let required_height = conservative_height + 2.0 * guard_each_side;

    let outputs = [
        nominal_width,
        nominal_height,
        standard_h,
        standard_v,
        expanded_h,
        expanded_v,
        conservative_width,
        conservative_height,
        required_width,
        required_height,
    ];
    if outputs.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(vec![
            "geometry or propagated uncertainty produced a non-finite or negative result"
                .to_owned(),
        ]);
    }

    let horizontal_margin = input.uniform_patch_width_mm - required_width;
    let vertical_margin = input.uniform_patch_height_mm - required_height;
    let horizontal_utilization = required_width / input.uniform_patch_width_mm;
    let vertical_utilization = required_height / input.uniform_patch_height_mm;
    if [
        horizontal_margin,
        vertical_margin,
        horizontal_utilization,
        vertical_utilization,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(vec![
            "patch margin or utilization produced a non-finite result".to_owned(),
        ]);
    }
    let horizontal_fits = horizontal_margin >= 0.0;
    let vertical_fits = vertical_margin >= 0.0;
    let screen_status = if horizontal_fits && vertical_fits {
        "WITHIN_GUARD_BAND"
    } else {
        "OUTSIDE_GUARD_BAND"
    }
    .to_owned();

    Ok(GeometryScreenResult {
        record_id: input.record_id.clone(),
        nominal_footprint_width_mm: nominal_width,
        nominal_footprint_height_mm: nominal_height,
        horizontal_standard_uncertainty_mm: standard_h,
        vertical_standard_uncertainty_mm: standard_v,
        horizontal_expanded_uncertainty_mm: expanded_h,
        vertical_expanded_uncertainty_mm: expanded_v,
        conservative_footprint_width_mm: conservative_width,
        conservative_footprint_height_mm: conservative_height,
        total_required_width_mm: required_width,
        total_required_height_mm: required_height,
        horizontal_margin_mm: horizontal_margin,
        vertical_margin_mm: vertical_margin,
        horizontal_utilization_fraction: horizontal_utilization,
        vertical_utilization_fraction: vertical_utilization,
        uniform_patch_width_mm: input.uniform_patch_width_mm,
        uniform_patch_height_mm: input.uniform_patch_height_mm,
        horizontal_fits,
        vertical_fits,
        screen_status,
    })
}

/// Validate an input record, validate the Draft 2020-12 schema, and run the
/// geometric screening model. A valid output remains a design-screen result.
pub fn validate_geometry_screen_str(
    record_text: &str,
    schema_text: &str,
) -> Result<(GeometryScreenResult, String, String), Vec<String>> {
    let record: Value = serde_json::from_str(record_text)
        .map_err(|error| vec![format!("geometry record JSON parse error: {error}")])?;
    let schema: Value = serde_json::from_str(schema_text)
        .map_err(|error| vec![format!("geometry schema JSON parse error: {error}")])?;
    let mut errors = Vec::new();

    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_DRAFT) {
        errors.push(format!("geometry schema must declare {EXPECTED_SCHEMA_DRAFT}"));
    }
    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&record) {
                errors.push(format!("geometry JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("geometry schema compilation failed: {error}")),
    }

    if record.get("record_state").and_then(Value::as_str)
        != Some("SYNTHETIC_DESIGN_SCREEN_ONLY")
    {
        errors.push("geometry v0.1 record must remain SYNTHETIC_DESIGN_SCREEN_ONLY".to_owned());
    }
    if record.get("physical_validation").and_then(Value::as_str) != Some("NOT_PERFORMED") {
        errors.push("geometry screen cannot self-promote physical validation".to_owned());
    }
    if record.get("claim_ceiling").and_then(Value::as_str)
        != Some("GEOMETRIC_SETUP_SCREEN_ONLY_NOT_PHYSICAL_QUALIFICATION")
    {
        errors.push("geometry claim ceiling is missing or unsupported".to_owned());
    }

    let geometry = record.get("geometry").unwrap_or(&Value::Null);
    let input = match input_from_json(&record, geometry) {
        Ok(input) => Some(input),
        Err(mut input_errors) => {
            errors.append(&mut input_errors);
            None
        }
    };
    let result = input.and_then(|value| match evaluate_geometry_screen(&value) {
        Ok(result) => Some(result),
        Err(mut eval_errors) => {
            errors.append(&mut eval_errors);
            None
        }
    });

    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok((
        result.expect("geometry errors must be empty when report is returned"),
        sha256_hex(record_text.as_bytes()),
        sha256_hex(schema_text.as_bytes()),
    ))
}

fn input_from_json(
    record: &Value,
    geometry: &Value,
) -> Result<GeometryScreenInput, Vec<String>> {
    fn get_number(value: &Value, key: &str, errors: &mut Vec<String>) -> f64 {
        match value.get(key).and_then(Value::as_f64) {
            Some(number) => number,
            None => {
                errors.push(format!("geometry.{key} must be numeric"));
                f64::NAN
            }
        }
    }

    let mut errors = Vec::new();
    let input = GeometryScreenInput {
        record_id: record.get("record_id").and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        distance_mm: get_number(geometry, "distance_mm", &mut errors),
        horizontal_fov_deg: get_number(geometry, "horizontal_fov_deg", &mut errors),
        vertical_fov_deg: get_number(geometry, "vertical_fov_deg", &mut errors),
        distance_standard_uncertainty_mm: get_number(
            geometry,
            "distance_standard_uncertainty_mm",
            &mut errors,
        ),
        horizontal_fov_standard_uncertainty_deg: get_number(
            geometry,
            "horizontal_fov_standard_uncertainty_deg",
            &mut errors,
        ),
        vertical_fov_standard_uncertainty_deg: get_number(
            geometry,
            "vertical_fov_standard_uncertainty_deg",
            &mut errors,
        ),
        coverage_factor: get_number(geometry, "coverage_factor", &mut errors),
        uniform_patch_width_mm: get_number(geometry, "uniform_patch_width_mm", &mut errors),
        uniform_patch_height_mm: get_number(geometry, "uniform_patch_height_mm", &mut errors),
        centering_uncertainty_mm: get_number(
            geometry,
            "centering_uncertainty_mm",
            &mut errors,
        ),
        edge_guard_mm: get_number(geometry, "edge_guard_mm", &mut errors),
    };
    if errors.is_empty() {
        Ok(input)
    } else {
        Err(errors)
    }
}

fn validate_positive(value: f64, name: &str, errors: &mut Vec<String>) {
    if !value.is_finite() || value <= 0.0 {
        errors.push(format!("{name} must be finite and positive"));
    }
}

fn validate_nonnegative(value: f64, name: &str, errors: &mut Vec<String>) {
    if !value.is_finite() || value < 0.0 {
        errors.push(format!("{name} must be finite and nonnegative"));
    }
}

fn validate_fov(value: f64, name: &str, errors: &mut Vec<String>) {
    if !value.is_finite() || value <= 0.0 || value >= 179.0 {
        errors.push(format!("{name} must be finite and in (0, 179) degrees"));
    }
}

#[cfg(test)]
mod tests {
    use super::{GeometryScreenInput, evaluate_geometry_screen, validate_geometry_screen_str};
    use serde_json::Value;

    const GEOMETRY_TEXT: &str =
        include_str!("../../../docs/engineering/display-measurement-geometry-synthetic-v0.1.json");
    const GEOMETRY_SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/display-measurement-geometry.schema.json");

    fn fixture() -> Value {
        serde_json::from_str(GEOMETRY_TEXT).expect("synthetic geometry fixture must parse")
    }

    fn geometry_input() -> GeometryScreenInput {
        GeometryScreenInput {
            record_id: "GEOM-SYNTH-001".to_owned(),
            distance_mm: 1000.0,
            horizontal_fov_deg: 10.0,
            vertical_fov_deg: 5.0,
            distance_standard_uncertainty_mm: 1.0,
            horizontal_fov_standard_uncertainty_deg: 0.1,
            vertical_fov_standard_uncertainty_deg: 0.1,
            coverage_factor: 2.0,
            uniform_patch_width_mm: 200.0,
            uniform_patch_height_mm: 100.0,
            centering_uncertainty_mm: 1.0,
            edge_guard_mm: 2.0,
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!((actual - expected).abs() <= tolerance, "{actual} != {expected}");
    }

    #[test]
    fn known_answer_ideal_fov_matches_tangent_geometry() {
        let result = evaluate_geometry_screen(&geometry_input())
            .expect("synthetic ideal geometry should be valid");
        close(
            result.nominal_footprint_width_mm,
            174.977_327_051_848_02,
            1e-10,
        );
        close(
            result.nominal_footprint_height_mm,
            87.321_885_817_024_11,
            1e-10,
        );
        assert!(result.horizontal_fits);
        assert!(result.vertical_fits);
        assert_eq!(result.screen_status, "WITHIN_GUARD_BAND");
    }

    #[test]
    fn conservative_envelope_includes_distance_angle_and_alignment_guards() {
        let input = geometry_input();
        let result = evaluate_geometry_screen(&input).expect("valid input");
        let upper_distance =
            input.distance_mm + input.coverage_factor * input.distance_standard_uncertainty_mm;
        let upper_horizontal_fov = input.horizontal_fov_deg
            + input.coverage_factor * input.horizontal_fov_standard_uncertainty_deg;
        let expected_width =
            2.0 * upper_distance * (upper_horizontal_fov.to_radians() / 2.0).tan();
        close(result.conservative_footprint_width_mm, expected_width, 1e-10);
        close(
            result.total_required_width_mm,
            expected_width + 2.0 * (input.centering_uncertainty_mm + input.edge_guard_mm),
            1e-10,
        );
    }

    #[test]
    fn larger_distance_and_angular_uncertainty_do_not_shrink_guarded_footprint() {
        let baseline = geometry_input();
        let baseline_result =
            evaluate_geometry_screen(&baseline).expect("baseline geometry is valid");

        let mut stressed = baseline;
        stressed.distance_mm += 100.0;
        stressed.horizontal_fov_standard_uncertainty_deg *= 2.0;
        stressed.vertical_fov_standard_uncertainty_deg *= 2.0;
        let stressed_result =
            evaluate_geometry_screen(&stressed).expect("stressed geometry remains valid");

        assert!(
            stressed_result.conservative_footprint_width_mm
                > baseline_result.conservative_footprint_width_mm
        );
        assert!(
            stressed_result.conservative_footprint_height_mm
                > baseline_result.conservative_footprint_height_mm
        );
        assert!(
            stressed_result.total_required_width_mm
                > baseline_result.total_required_width_mm
        );
        assert!(
            stressed_result.total_required_height_mm
                > baseline_result.total_required_height_mm
        );
    }

    #[test]
    fn uniform_patch_fit_boundary_is_inclusive_and_below_boundary_fails() {
        let mut input = geometry_input();
        let first = evaluate_geometry_screen(&input).expect("initial geometry must be valid");
        input.uniform_patch_width_mm = first.total_required_width_mm;
        input.uniform_patch_height_mm = first.total_required_height_mm;

        let exact_edge =
            evaluate_geometry_screen(&input).expect("exactly sufficient patch is valid");
        assert!(exact_edge.horizontal_fits);
        assert!(exact_edge.vertical_fits);
        close(exact_edge.horizontal_margin_mm, 0.0, 1e-12);
        close(exact_edge.vertical_margin_mm, 0.0, 1e-12);
        close(exact_edge.horizontal_utilization_fraction, 1.0, 1e-12);
        close(exact_edge.vertical_utilization_fraction, 1.0, 1e-12);
        assert_eq!(exact_edge.screen_status, "WITHIN_GUARD_BAND");

        input.uniform_patch_width_mm -= 1e-9;
        let just_too_small =
            evaluate_geometry_screen(&input).expect("just-too-small patch is still a valid measurement setup");
        assert!(!just_too_small.horizontal_fits);
        assert!(just_too_small.vertical_fits);
        assert!(just_too_small.horizontal_margin_mm < 0.0);
        assert!(just_too_small.vertical_margin_mm >= 0.0);
        assert!(just_too_small.horizontal_utilization_fraction > 1.0);
        assert!(just_too_small.vertical_utilization_fraction <= 1.0);
        assert_eq!(just_too_small.screen_status, "OUTSIDE_GUARD_BAND");
    }

    #[test]
    fn patch_fit_refuses_when_conservative_footprint_exceeds_uniform_region() {
        let mut input = geometry_input();
        input.uniform_patch_width_mm = 180.0;
        input.uniform_patch_height_mm = 90.0;
        let result = evaluate_geometry_screen(&input).expect("valid but insufficient patch");
        assert!(!result.horizontal_fits);
        assert!(!result.vertical_fits);
        assert_eq!(result.screen_status, "OUTSIDE_GUARD_BAND");
    }

    #[test]
    fn invalid_fov_and_uncertainty_inputs_fail_closed() {
        let mut input = geometry_input();
        input.horizontal_fov_deg = 180.0;
        assert!(evaluate_geometry_screen(&input).is_err());

        let mut input = geometry_input();
        input.vertical_fov_standard_uncertainty_deg = -0.01;
        assert!(evaluate_geometry_screen(&input).is_err());

        let mut input = geometry_input();
        input.horizontal_fov_deg = 178.0;
        input.horizontal_fov_standard_uncertainty_deg = 1.0;
        assert!(evaluate_geometry_screen(&input).is_err());
    }

    #[test]
    fn schema_fixture_remains_synthetic_and_unqualified() {
        let (result, input_hash, schema_hash) =
            validate_geometry_screen_str(GEOMETRY_TEXT, GEOMETRY_SCHEMA_TEXT)
                .expect("known-answer geometry fixture must validate");
        assert_eq!(result.record_id, "GEOM-SYNTH-001");
        assert_eq!(result.screen_status, "WITHIN_GUARD_BAND");
        close(result.horizontal_margin_mm, 15.147_764_779_274_72, 1e-9);
        close(result.vertical_margin_mm, 2.998_892_499_836_33, 1e-9);
        close(result.horizontal_utilization_fraction, 0.924_261_176_103_626_4, 1e-9);
        close(result.vertical_utilization_fraction, 0.970_011_075_001_636_7, 1e-9);
        assert_eq!(input_hash.len(), 64);
        assert_eq!(schema_hash.len(), 64);
    }

    #[test]
    fn schema_fixture_cannot_promote_physical_validation() {
        let mut record = fixture();
        record["physical_validation"] = Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        let text = serde_json::to_string(&record).expect("mutated record serializes");
        assert!(validate_geometry_screen_str(&text, GEOMETRY_SCHEMA_TEXT).is_err());
    }
}

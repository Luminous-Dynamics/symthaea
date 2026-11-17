// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative geometry admission for the first Robot Garment Compiler P0.
//!
//! This module evaluates candidate panel AABBs against explicitly declared
//! robot-local keep-out envelopes. It does not flatten surfaces, generate
//! sewing patterns, solve cloth mechanics, certify safety, or authorize
//! fabrication or robot motion.

use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotGarmentCompilerFixture {
    pub schema_version: String,
    pub compilation_id: String,
    pub state: String,
    pub units: String,
    pub robot: RobotGeometry,
    pub zones: Vec<ProtectionZone>,
    pub keepouts: Vec<KeepOutEnvelope>,
    pub candidate_panels: Vec<CandidatePanel>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotGeometry {
    pub robot_id: String,
    pub revision: Option<String>,
    pub geometry_state: String,
    pub geometry_digest: Option<String>,
    pub coordinate_frames: Vec<CoordinateFrame>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateFrame {
    pub frame_id: String,
    pub units: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds3 {
    pub min_mm: [f64; 3],
    pub max_mm: [f64; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectionZone {
    pub zone_id: String,
    pub coordinate_frame_ref: String,
    pub bounds: Bounds3,
    pub functions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeepOutEnvelope {
    pub keepout_id: String,
    pub coordinate_frame_ref: String,
    pub bounds: Bounds3,
    pub kind: String,
    pub affected_zone_ids: Vec<String>,
    pub required_clearance_mm: f64,
    pub uncertainty_mm: f64,
    pub severity: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatePanel {
    pub panel_id: String,
    pub zone_id: String,
    pub coordinate_frame_ref: String,
    pub bounds: Bounds3,
    pub material_evidence_state: String,
    pub material_profile_ref: Option<String>,
    pub pattern_revision_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
    pub blocking: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClearanceResult {
    pub panel_id: String,
    pub keepout_id: String,
    pub expanded_clearance_mm: f64,
    pub intersects: bool,
    pub severity: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub clearance_results: Vec<ClearanceResult>,
    /// Admitted only for a later physical fit-screen planning step. This does
    /// not mean the panel fits the physical robot or is safe.
    pub eligible_for_fit_screen: bool,
    /// True only when every panel binds measured coupon or qualified material
    /// evidence to an exact material profile.
    pub material_evidence_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        !self.findings.iter().any(|finding| finding.blocking)
    }

    pub fn has_geometric_intersections(&self) -> bool {
        self.clearance_results
            .iter()
            .any(|result| result.intersects && result.severity == "hard")
    }
}

pub fn parse_and_evaluate(
    input: &str,
) -> Result<ValidationReport, serde_json::Error> {
    let fixture: RobotGarmentCompilerFixture = serde_json::from_str(input)?;
    Ok(fixture.evaluate())
}

impl RobotGarmentCompilerFixture {
    pub fn evaluate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut clearance_results = Vec::new();
        let mut frame_ids = HashSet::new();
        let mut zone_ids = HashSet::new();
        let mut keepout_ids = HashSet::new();
        let mut panel_ids = HashSet::new();

        if self.schema_version != "0.1.0" {
            push(
                &mut findings,
                "unsupported_schema_version",
                "schema_version",
                "Only compiler fixture schema version 0.1.0 is admitted.",
                true,
            );
        }
        if self.compilation_id.trim().is_empty() || self.robot.robot_id.trim().is_empty() {
            push(
                &mut findings,
                "missing_compilation_identity",
                "compilation_id/robot.robot_id",
                "Compilation and robot identities must be non-empty.",
                true,
            );
        }
        if self.units != "mm" {
            push(
                &mut findings,
                "unsupported_units",
                "units",
                "P0 accepts millimetres only; implicit unit conversion is prohibited.",
                true,
            );
        }
        if !matches!(
            self.state.as_str(),
            "synthetic_fixture" | "inputs_pending" | "candidate_geometry"
        ) {
            push(
                &mut findings,
                "invalid_compilation_state",
                "state",
                "Unknown compiler fixture state.",
                true,
            );
        }
        if !matches!(
            self.robot.geometry_state.as_str(),
            "synthetic_fixture" | "cad_revision_bound"
        ) {
            push(
                &mut findings,
                "invalid_geometry_state",
                "robot.geometry_state",
                "Unknown geometry admission state.",
                true,
            );
        }
        if let Some(digest) = self.robot.geometry_digest.as_deref() {
            if !is_sha256_digest(digest) {
                push(
                    &mut findings,
                    "invalid_geometry_digest",
                    "robot.geometry_digest",
                    "A supplied geometry digest must be sha256: plus 64 hexadecimal characters.",
                    true,
                );
            }
        }
        if self.robot.geometry_state == "cad_revision_bound"
            && (self.robot.revision.as_deref().unwrap_or("").trim().is_empty()
                || !self
                    .robot
                    .geometry_digest
                    .as_deref()
                    .is_some_and(is_sha256_digest))
        {
            push(
                &mut findings,
                "cad_geometry_not_revision_bound",
                "robot",
                "CAD-bound geometry requires an exact revision and valid SHA-256 digest.",
                true,
            );
        }
        if self.state == "inputs_pending" {
            push(
                &mut findings,
                "compiler_inputs_pending",
                "state",
                "Inputs-pending fixtures are not eligible for geometric fit-screen admission.",
                true,
            );
        }
        if self.state == "candidate_geometry"
            && self.robot.geometry_state != "cad_revision_bound"
        {
            push(
                &mut findings,
                "candidate_geometry_without_cad",
                "robot.geometry_state",
                "A real candidate-geometry state requires revision-bound CAD geometry and digest; synthetic geometry must remain explicitly synthetic_fixture.",
                true,
            );
        }

        for (index, frame) in self.robot.coordinate_frames.iter().enumerate() {
            let path = format!("robot.coordinate_frames[{index}]");
            if frame.frame_id.trim().is_empty() {
                push(
                    &mut findings,
                    "empty_frame_id",
                    &path,
                    "Coordinate-frame IDs must be non-empty.",
                    true,
                );
            }
            if frame.units != "mm" {
                push(
                    &mut findings,
                    "frame_unit_mismatch",
                    format!("{path}.units"),
                    "Every P0 coordinate frame must use millimetres.",
                    true,
                );
            }
            if !frame_ids.insert(frame.frame_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_coordinate_frame",
                    &path,
                    "Coordinate-frame IDs must be unique.",
                    true,
                );
            }
        }
        if frame_ids.is_empty() {
            push(
                &mut findings,
                "missing_coordinate_frames",
                "robot.coordinate_frames",
                "At least one explicit coordinate frame is required.",
                true,
            );
        }

        for (index, zone) in self.zones.iter().enumerate() {
            let path = format!("zones[{index}]");
            if !zone_ids.insert(zone.zone_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_zone_id",
                    &path,
                    "Zone IDs must be unique.",
                    true,
                );
            }
            if !frame_ids.contains(zone.coordinate_frame_ref.as_str()) {
                push(
                    &mut findings,
                    "unknown_zone_frame",
                    format!("{path}.coordinate_frame_ref"),
                    "Zone frame is not declared by the robot geometry.",
                    true,
                );
            }
            validate_bounds(
                &zone.bounds,
                format!("{path}.bounds"),
                &mut findings,
            );
            if zone.functions.is_empty() {
                push(
                    &mut findings,
                    "zone_without_function",
                    format!("{path}.functions"),
                    "A zone must state at least one intended function.",
                    true,
                );
            }
        }
        if self.zones.is_empty() {
            push(
                &mut findings,
                "missing_protection_zones",
                "zones",
                "At least one protection zone is required.",
                true,
            );
        }

        for (index, keepout) in self.keepouts.iter().enumerate() {
            let path = format!("keepouts[{index}]");
            if !keepout_ids.insert(keepout.keepout_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_keepout_id",
                    &path,
                    "Keep-out IDs must be unique.",
                    true,
                );
            }
            if !frame_ids.contains(keepout.coordinate_frame_ref.as_str()) {
                push(
                    &mut findings,
                    "unknown_keepout_frame",
                    format!("{path}.coordinate_frame_ref"),
                    "Keep-out frame is not declared by the robot geometry.",
                    true,
                );
            }
            validate_bounds(
                &keepout.bounds,
                format!("{path}.bounds"),
                &mut findings,
            );
            if !matches!(
                keepout.kind.as_str(),
                "swept_motion"
                    | "pinch_shear"
                    | "cable_route"
                    | "sensor_aperture"
                    | "optical_field"
                    | "thermal_vent"
                    | "service_access"
                    | "emergency_access"
                    | "charging_interface"
            ) {
                push(
                    &mut findings,
                    "unknown_keepout_kind",
                    format!("{path}.kind"),
                    "Unknown keep-out category.",
                    true,
                );
            }
            if !matches!(keepout.severity.as_str(), "hard" | "advisory") {
                push(
                    &mut findings,
                    "unknown_keepout_severity",
                    format!("{path}.severity"),
                    "Keep-out severity must be hard or advisory.",
                    true,
                );
            }
            if !keepout.required_clearance_mm.is_finite()
                || keepout.required_clearance_mm < 0.0
                || !keepout.uncertainty_mm.is_finite()
                || keepout.uncertainty_mm < 0.0
            {
                push(
                    &mut findings,
                    "invalid_keepout_margin",
                    &path,
                    "Clearance and uncertainty must be finite non-negative millimetres.",
                    true,
                );
            }
            if keepout.affected_zone_ids.is_empty() {
                push(
                    &mut findings,
                    "keepout_without_zones",
                    format!("{path}.affected_zone_ids"),
                    "A keep-out must declare at least one affected zone.",
                    true,
                );
            }
            for zone_id in keepout.affected_zone_ids.iter() {
                if !zone_ids.contains(zone_id.as_str()) {
                    push(
                        &mut findings,
                        "keepout_unknown_zone",
                        format!("{path}.affected_zone_ids"),
                        format!("Keep-out references unknown zone {zone_id}."),
                        true,
                    );
                }
            }
        }

        let mut material_evidence_admitted = !self.candidate_panels.is_empty();
        for (index, panel) in self.candidate_panels.iter().enumerate() {
            let path = format!("candidate_panels[{index}]");
            if !panel_ids.insert(panel.panel_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_panel_id",
                    &path,
                    "Panel IDs must be unique.",
                    true,
                );
            }
            if !matches!(
                panel.material_evidence_state.as_str(),
                "unknown" | "supplier_declared" | "measured_coupon" | "qualified"
            ) {
                push(
                    &mut findings,
                    "unknown_material_evidence_state",
                    format!("{path}.material_evidence_state"),
                    "Unknown material-evidence state.",
                    true,
                );
            }
            let has_material_ref = panel
                .material_profile_ref
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            if matches!(
                panel.material_evidence_state.as_str(),
                "measured_coupon" | "qualified"
            ) && !has_material_ref
            {
                push(
                    &mut findings,
                    "material_evidence_without_profile",
                    &path,
                    "Measured or qualified material evidence requires an exact material-profile reference.",
                    true,
                );
            }
            if !matches!(
                panel.material_evidence_state.as_str(),
                "measured_coupon" | "qualified"
            ) || !has_material_ref
            {
                material_evidence_admitted = false;
            }

            let Some(zone) = self
                .zones
                .iter()
                .find(|zone| zone.zone_id == panel.zone_id)
            else {
                push(
                    &mut findings,
                    "panel_unknown_zone",
                    format!("{path}.zone_id"),
                    "Panel references an unknown protection zone.",
                    true,
                );
                continue;
            };

            if !frame_ids.contains(panel.coordinate_frame_ref.as_str()) {
                push(
                    &mut findings,
                    "unknown_panel_frame",
                    format!("{path}.coordinate_frame_ref"),
                    "Panel frame is not declared by the robot geometry.",
                    true,
                );
            }
            if panel.coordinate_frame_ref != zone.coordinate_frame_ref {
                push(
                    &mut findings,
                    "panel_zone_frame_mismatch",
                    &path,
                    "Panel and protection zone are expressed in different frames; no implicit transform is allowed.",
                    true,
                );
            }
            validate_bounds(
                &panel.bounds,
                format!("{path}.bounds"),
                &mut findings,
            );
            if !contains(&zone.bounds, &panel.bounds) {
                push(
                    &mut findings,
                    "panel_outside_zone",
                    &path,
                    "Candidate panel bounds extend outside the declared protection zone.",
                    true,
                );
            }

            for (keepout_index, keepout) in self.keepouts.iter().enumerate() {
                if !keepout.affected_zone_ids.iter().any(|id| id == &panel.zone_id) {
                    continue;
                }
                if panel.coordinate_frame_ref != keepout.coordinate_frame_ref {
                    push(
                        &mut findings,
                        "unresolved_keepout_frame_transform",
                        format!("{path}.keepout[{keepout_index}]"),
                        "Panel and keep-out use different frames; a declared and independently verified transform is required.",
                        true,
                    );
                    continue;
                }
                if !valid_bounds(&panel.bounds) || !valid_bounds(&keepout.bounds) {
                    continue;
                }
                let margin = keepout.required_clearance_mm + keepout.uncertainty_mm;
                if !margin.is_finite() {
                    push(
                        &mut findings,
                        "non_finite_expanded_margin",
                        format!("{path}.keepout[{keepout_index}]"),
                        "Clearance plus uncertainty overflowed the supported numeric range.",
                        true,
                    );
                    continue;
                }
                let intersects = intersects_expanded(&panel.bounds, &keepout.bounds, margin);
                clearance_results.push(ClearanceResult {
                    panel_id: panel.panel_id.clone(),
                    keepout_id: keepout.keepout_id.clone(),
                    expanded_clearance_mm: margin,
                    intersects,
                    severity: keepout.severity.clone(),
                });
                if intersects {
                    push(
                        &mut findings,
                        "panel_intersects_keepout",
                        format!("{path}.keepout[{keepout_index}]"),
                        format!(
                            "Panel {} intersects keep-out {} after adding {} mm clearance and uncertainty.",
                            panel.panel_id, keepout.keepout_id, margin
                        ),
                        keepout.severity == "hard",
                    );
                }
            }
        }

        if self.candidate_panels.is_empty() {
            push(
                &mut findings,
                "missing_candidate_panels",
                "candidate_panels",
                "At least one candidate panel is required for P0 evaluation.",
                true,
            );
        }

        let has_blocking_findings = findings.iter().any(|finding| finding.blocking);
        let eligible_for_fit_screen = !has_blocking_findings
            && matches!(
                self.state.as_str(),
                "synthetic_fixture" | "candidate_geometry"
            );
        ValidationReport {
            findings,
            clearance_results,
            eligible_for_fit_screen,
            material_evidence_admitted,
        }
    }
}

fn validate_bounds(bounds: &Bounds3, path: String, findings: &mut Vec<Finding>) {
    if !valid_bounds(bounds) {
        push(
            findings,
            "invalid_bounds",
            path,
            "Each bound must be finite and min_mm must be strictly below max_mm on every axis.",
            true,
        );
    }
}

fn valid_bounds(bounds: &Bounds3) -> bool {
    (0..3).all(|axis| {
        bounds.min_mm[axis].is_finite()
            && bounds.max_mm[axis].is_finite()
            && bounds.min_mm[axis] < bounds.max_mm[axis]
    })
}

fn contains(container: &Bounds3, candidate: &Bounds3) -> bool {
    (0..3).all(|axis| {
        candidate.min_mm[axis] >= container.min_mm[axis]
            && candidate.max_mm[axis] <= container.max_mm[axis]
    })
}

/// Conservative AABB intersection after expanding the keep-out on all axes.
/// Boundary contact counts as an intersection. Both boxes must already be
/// expressed in the same declared frame and in millimetres.
fn intersects_expanded(panel: &Bounds3, keepout: &Bounds3, margin_mm: f64) -> bool {
    (0..3).all(|axis| {
        panel.max_mm[axis] >= keepout.min_mm[axis] - margin_mm
            && panel.min_mm[axis] <= keepout.max_mm[axis] + margin_mm
    })
}

fn is_sha256_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn push(
    findings: &mut Vec<Finding>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
    blocking: bool,
) {
    findings.push(Finding {
        code,
        path: path.into(),
        message: message.into(),
        blocking,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(
        "../../../../docs/research/fixtures/robot_garment_compiler_v0_1/single_elbow.example.json"
    );

    fn fixture() -> RobotGarmentCompilerFixture {
        serde_json::from_str(FIXTURE).expect("synthetic elbow fixture must deserialize")
    }

    #[test]
    fn known_answer_elbow_fixture_is_eligible_for_fit_screen_only() {
        let report = fixture().evaluate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert!(report.eligible_for_fit_screen);
        assert!(!report.material_evidence_admitted);
        assert!(!report.has_geometric_intersections());
    }

    #[test]
    fn panel_entering_swept_joint_envelope_is_rejected() {
        let mut fixture = fixture();
        fixture.candidate_panels[0].bounds.min_mm[0] = 64.0;
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "panel_intersects_keepout"
                && finding.message.contains("elbow_swept_motion")
        }));
    }

    #[test]
    fn unknown_coordinate_frame_fails_closed() {
        let mut fixture = fixture();
        fixture.candidate_panels[0].coordinate_frame_ref = "unregistered_wrist_frame".into();
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "unknown_panel_frame"
        }));
    }

    #[test]
    fn panel_outside_declared_zone_is_rejected() {
        let mut fixture = fixture();
        fixture.candidate_panels[0].bounds.min_mm[0] = 59.0;
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "panel_outside_zone"
        }));
    }

    #[test]
    fn clearance_uncertainty_is_added_to_the_keepout() {
        let mut fixture = fixture();
        // The joint keep-out ends at x=62; clearance 2 + uncertainty 1
        // expands it to x=65. A panel starting at x=65 touches and is rejected.
        fixture.candidate_panels[0].bounds.min_mm[0] = 65.0;
        let report = fixture.evaluate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "panel_intersects_keepout"
                && finding.message.contains("elbow_swept_motion")
        }));
    }

    #[test]
    fn frame_mismatch_requires_an_explicit_transform() {
        let mut fixture = fixture();
        fixture.keepouts[0].coordinate_frame_ref = "another_declared_frame".into();
        fixture.robot.coordinate_frames.push(CoordinateFrame {
            frame_id: "another_declared_frame".into(),
            units: "mm".into(),
        });
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "unresolved_keepout_frame_transform"
        }));
    }

    #[test]
    fn candidate_geometry_cannot_be_admitted_from_synthetic_geometry() {
        let mut fixture = fixture();
        fixture.state = "candidate_geometry".into();
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "candidate_geometry_without_cad"
        }));
    }

    #[test]
    fn advisory_keepout_intersection_is_reported_but_does_not_block() {
        let mut fixture = fixture();
        fixture.keepouts[0].severity = "advisory".into();
        fixture.candidate_panels[0].bounds.min_mm[0] = 64.0;
        let report = fixture.evaluate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert!(report.eligible_for_fit_screen);
        assert!(report.clearance_results.iter().any(|result| {
            result.keepout_id == "elbow_swept_motion" && result.intersects
        }));
    }

    #[test]
    fn supplier_declared_material_is_not_measured_material_evidence() {
        let mut fixture = fixture();
        fixture.candidate_panels[0].material_evidence_state = "supplier_declared".into();
        fixture.candidate_panels[0].material_profile_ref = Some("supplier:material-001".into());
        let report = fixture.evaluate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert!(report.eligible_for_fit_screen);
        assert!(!report.material_evidence_admitted);
    }

    #[test]
    fn cad_geometry_requires_revision_and_digest() {
        let mut fixture = fixture();
        fixture.robot.geometry_state = "cad_revision_bound".into();
        fixture.robot.geometry_digest = None;
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "cad_geometry_not_revision_bound"
        }));
    }

    #[test]
    fn invalid_units_are_not_implicitly_converted() {
        let mut fixture = fixture();
        fixture.units = "inch".into();
        let report = fixture.evaluate();
        assert!(!report.eligible_for_fit_screen);
        assert!(report.findings.iter().any(|finding| {
            finding.code == "unsupported_units"
        }));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightshell spacer-textile experiment-design admission.
//!
//! This validates the Lightshell-specific v0.1 screening/refinement matrix.
//! It delegates randomization, blocking, statistical inference, active
//! experiment selection and physical qualification to the existing generic
//! SCI-EXP / MAT-ACTIVE / MAT-UQ / TEXTILE-INVERSE-DESIGN owners.

use serde::Deserialize;
use std::collections::HashSet;

const SCREEN_ARCHITECTURES: [&str; 4] = [
    "none",
    "float",
    "square_wave",
    "square_wave_anchored_tuck",
];
const FREQUENCIES: [&str; 3] = ["low", "reference", "high"];
const TRANSITIONS: [&str; 3] = ["one_needle", "two_needles", "three_needles"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpacerDoeManifest {
    pub schema_version: String,
    pub experiment_id: String,
    pub state: String,
    pub parent_article_id: String,
    pub process_capability: ProcessCapability,
    pub controls: Vec<Control>,
    pub factors: Vec<Factor>,
    pub candidate_structures: Vec<CandidateStructure>,
    pub response_domains: Vec<String>,
    pub specimen_allocation: SpecimenAllocation,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessCapability {
    pub machine_ref: Option<String>,
    pub capability_state: String,
    pub program_revision: Option<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    pub control_id: String,
    pub architecture: String,
    pub matching_basis: String,
    pub availability: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Factor {
    pub factor_id: String,
    pub unit_or_encoding: String,
    pub levels: Vec<String>,
    pub capability_dependent: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateStructure {
    pub candidate_id: String,
    pub stage: String,
    pub inlay_architecture: String,
    pub spacer_frequency: String,
    pub transition_geometry: String,
    pub state: String,
    pub material_lot_ref: Option<String>,
    pub process_revision_ref: Option<String>,
    pub replicate_plan: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecimenAllocation {
    pub randomization_seed: Option<u64>,
    pub blocking_variables: Vec<String>,
    pub destructive_tests_isolated: bool,
    pub confirmatory_holdout: bool,
    pub sample_size_rationale: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub material_lot_binding: bool,
    pub process_binding: bool,
    pub instrument_calibration_binding: bool,
    pub raw_data_digest_required: bool,
    pub uncertainty_required: bool,
    pub paper_results_not_lightshell_results: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub screen_count: usize,
    pub transition_refinement_count: usize,
    pub matrix_complete: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: SpacerDoeManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl SpacerDoeManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut candidate_ids = HashSet::new();

        if self.schema_version != "0.1.0" {
            push(
                &mut findings,
                "unsupported_schema_version",
                "schema_version",
                "Only Lightshell spacer DoE schema version 0.1.0 is admitted.",
            );
        }
        if self.experiment_id.trim().is_empty() || self.parent_article_id.trim().is_empty() {
            push(
                &mut findings,
                "missing_experiment_identity",
                "experiment_id/parent_article_id",
                "Experiment and parent article identities must be non-empty.",
            );
        }
        if !matches!(
            self.state.as_str(),
            "proposed" | "capability_pending" | "preregistered" | "running" | "complete" | "invalidated"
        ) {
            push(
                &mut findings,
                "invalid_experiment_state",
                "state",
                "Unknown spacer DoE lifecycle state.",
            );
        }
        if !matches!(
            self.process_capability.capability_state.as_str(),
            "unknown" | "preflight_pending" | "qualified_for_sample" | "not_capable"
        ) {
            push(
                &mut findings,
                "invalid_process_capability_state",
                "process_capability.capability_state",
                "Unknown process capability state.",
            );
        }
        if self.process_capability.capability_state == "qualified_for_sample"
            && (self
                .process_capability
                .machine_ref
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
                || self
                    .process_capability
                    .program_revision
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty())
        {
            push(
                &mut findings,
                "qualified_process_missing_identity",
                "process_capability",
                "Qualified sample capability requires machine and program-revision references.",
            );
        }
        if self.state == "preregistered"
            && self.process_capability.capability_state != "qualified_for_sample"
        {
            push(
                &mut findings,
                "preregistration_without_capability",
                "process_capability.capability_state",
                "The experimental matrix cannot be preregistered before the required sample process is capability-qualified.",
            );
        }

        let mut controls = HashSet::new();
        for (index, control) in self.controls.iter().enumerate() {
            let path = format!("controls[{index}]");
            if !controls.insert(control.control_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_control_id",
                    &path,
                    "Control IDs must be unique.",
                );
            }
            if control.control_id.trim().is_empty()
                || control.architecture.trim().is_empty()
                || control.matching_basis.trim().is_empty()
            {
                push(
                    &mut findings,
                    "incomplete_control_definition",
                    &path,
                    "Control identity, architecture and matching basis must be explicit.",
                );
            }
            if !matches!(
                control.availability.as_str(),
                "unknown" | "planned" | "available" | "not_comparable"
            ) {
                push(
                    &mut findings,
                    "invalid_control_availability",
                    format!("{path}.availability"),
                    "Unknown control availability state.",
                );
            }
        }
        for required in ["C0", "C1", "C2-T", "C2-M", "C3"] {
            if !controls.iter().any(|control_id| *control_id == required) {
                push(
                    &mut findings,
                    "missing_required_control",
                    "controls",
                    format!("Required comparison control {required} is missing."),
                );
            }
        }

        for (index, factor) in self.factors.iter().enumerate() {
            let path = format!("factors[{index}]");
            if factor.factor_id.trim().is_empty()
                || factor.unit_or_encoding.trim().is_empty()
                || factor.levels.is_empty()
            {
                push(
                    &mut findings,
                    "incomplete_factor_definition",
                    &path,
                    "Factor identity, encoding and levels must be explicit.",
                );
            }
            let unique_levels: HashSet<&str> =
                factor.levels.iter().map(String::as_str).collect();
            if unique_levels.len() != factor.levels.len() {
                push(
                    &mut findings,
                    "duplicate_factor_level",
                    format!("{path}.levels"),
                    "Factor levels must be unique.",
                );
            }
        }

        for required_factor in [
            "inlay_architecture",
            "spacer_frequency",
            "transition_geometry",
        ] {
            if !self
                .factors
                .iter()
                .any(|factor| factor.factor_id == required_factor)
            {
                push(
                    &mut findings,
                    "missing_required_factor",
                    "factors",
                    format!("Required factor {required_factor} is missing."),
                );
            }
        }
        let inlay_levels: Option<HashSet<String>> = self
            .factors
            .iter()
            .find(|factor| factor.factor_id == "inlay_architecture")
            .map(|factor| factor.levels.iter().cloned().collect());
        let frequency_levels: Option<HashSet<String>> = self
            .factors
            .iter()
            .find(|factor| factor.factor_id == "spacer_frequency")
            .map(|factor| factor.levels.iter().cloned().collect());
        let transition_levels: Option<HashSet<String>> = self
            .factors
            .iter()
            .find(|factor| factor.factor_id == "transition_geometry")
            .map(|factor| factor.levels.iter().cloned().collect());

        let mut screen_cells = HashSet::new();
        let mut refinement_cells = HashSet::new();
        let mut screen_count = 0;
        let mut transition_refinement_count = 0;

        for (index, candidate) in self.candidate_structures.iter().enumerate() {
            let path = format!("candidate_structures[{index}]");
            if candidate.candidate_id.trim().is_empty()
                || !candidate_ids.insert(candidate.candidate_id.as_str())
            {
                push(
                    &mut findings,
                    "duplicate_or_empty_candidate_id",
                    format!("{path}.candidate_id"),
                    "Candidate IDs must be non-empty and unique.",
                );
            }
            if !SCREEN_ARCHITECTURES.contains(&candidate.inlay_architecture.as_str()) {
                push(
                    &mut findings,
                    "unknown_inlay_architecture",
                    format!("{path}.inlay_architecture"),
                    "Unknown inlay architecture.",
                );
            }
            if !FREQUENCIES.contains(&candidate.spacer_frequency.as_str()) {
                push(
                    &mut findings,
                    "unknown_spacer_frequency",
                    format!("{path}.spacer_frequency"),
                    "Spacer frequency must be low, reference or high.",
                );
            }
            if !matches!(
                candidate.transition_geometry.as_str(),
                "not_applicable" | "one_needle" | "two_needles" | "three_needles"
            ) {
                push(
                    &mut findings,
                    "unknown_transition_geometry",
                    format!("{path}.transition_geometry"),
                    "Unknown transition geometry.",
                );
            }
            if let Some(levels) = inlay_levels.as_ref() {
                if !levels.contains(&candidate.inlay_architecture) {
                    push(
                        &mut findings,
                        "undeclared_inlay_level",
                        format!("{path}.inlay_architecture"),
                        "Candidate uses an inlay architecture absent from the factor definition.",
                    );
                }
            }
            if let Some(levels) = frequency_levels.as_ref() {
                if !levels.contains(&candidate.spacer_frequency) {
                    push(
                        &mut findings,
                        "undeclared_frequency_level",
                        format!("{path}.spacer_frequency"),
                        "Candidate uses a spacer-frequency level absent from the factor definition.",
                    );
                }
            }
            if let Some(levels) = transition_levels.as_ref() {
                if !levels.contains(&candidate.transition_geometry) {
                    push(
                        &mut findings,
                        "undeclared_transition_level",
                        format!("{path}.transition_geometry"),
                        "Candidate uses a transition-geometry level absent from the factor definition.",
                    );
                }
            }
            if candidate.replicate_plan.trim().is_empty() {
                push(
                    &mut findings,
                    "missing_replicate_plan",
                    format!("{path}.replicate_plan"),
                    "Every proposed structure needs an explicit replication plan.",
                );
            }
            if !matches!(
                candidate.state.as_str(),
                "proposed" | "not_manufacturable" | "fabricated" | "coupon_tested" | "invalidated"
            ) {
                push(
                    &mut findings,
                    "invalid_candidate_state",
                    format!("{path}.state"),
                    "Unknown sample lifecycle state.",
                );
            }
            if self.state == "proposed"
                && (candidate.state != "proposed"
                    || candidate.material_lot_ref.is_some()
                    || candidate.process_revision_ref.is_some())
            {
                push(
                    &mut findings,
                    "proposed_fixture_contains_execution_claim",
                    &path,
                    "The proposed seed must not imply fabrication, material-lot or process-revision evidence.",
                );
            }

            match candidate.stage.as_str() {
                "screen" => {
                    screen_count += 1;
                    let cell = (
                        candidate.inlay_architecture.as_str(),
                        candidate.spacer_frequency.as_str(),
                    );
                    if !screen_cells.insert(cell) {
                        push(
                            &mut findings,
                            "duplicate_screen_cell",
                            &path,
                            "Each inlay-architecture × spacer-frequency screen cell must appear exactly once.",
                        );
                    }
                    let expected_transition = match candidate.inlay_architecture.as_str() {
                        "none" | "float" => "not_applicable",
                        "square_wave" | "square_wave_anchored_tuck" => "two_needles",
                        _ => "",
                    };
                    if candidate.transition_geometry != expected_transition {
                        push(
                            &mut findings,
                            "screen_transition_confounded",
                            &path,
                            "Screen-stage transition geometry must be not_applicable for none/float and fixed at two needles for both square-wave families.",
                        );
                    }
                }
                "transition_refinement" => {
                    transition_refinement_count += 1;
                    if !matches!(
                        candidate.inlay_architecture.as_str(),
                        "square_wave" | "square_wave_anchored_tuck"
                    ) {
                        push(
                            &mut findings,
                            "invalid_transition_refinement_architecture",
                            &path,
                            "Transition refinement applies only to the two square-wave inlay families.",
                        );
                    }
                    if candidate.spacer_frequency != "reference" {
                        push(
                            &mut findings,
                            "transition_refinement_frequency_not_fixed",
                            &path,
                            "Transition-refinement cells must hold spacer frequency at reference.",
                        );
                    }
                    if !TRANSITIONS.contains(&candidate.transition_geometry.as_str()) {
                        push(
                            &mut findings,
                            "invalid_transition_refinement_level",
                            &path,
                            "Transition refinement must use one, two or three needle spacing.",
                        );
                    }
                    let cell = (
                        candidate.inlay_architecture.as_str(),
                        candidate.transition_geometry.as_str(),
                    );
                    if !refinement_cells.insert(cell) {
                        push(
                            &mut findings,
                            "duplicate_transition_refinement_cell",
                            &path,
                            "Each square-wave architecture × transition geometry refinement cell must appear exactly once.",
                        );
                    }
                }
                _ => push(
                    &mut findings,
                    "unknown_candidate_stage",
                    format!("{path}.stage"),
                    "Candidate stage must be screen or transition_refinement.",
                ),
            }
        }

        for architecture in SCREEN_ARCHITECTURES {
            for frequency in FREQUENCIES {
                if !screen_cells.contains(&(architecture, frequency)) {
                    push(
                        &mut findings,
                        "missing_screen_cell",
                        "candidate_structures",
                        format!("Missing screening cell {architecture} × {frequency}."),
                    );
                }
            }
        }
        for architecture in ["square_wave", "square_wave_anchored_tuck"] {
            for transition in TRANSITIONS {
                if !refinement_cells.contains(&(architecture, transition)) {
                    push(
                        &mut findings,
                        "missing_transition_refinement_cell",
                        "candidate_structures",
                        format!("Missing transition refinement {architecture} × {transition}."),
                    );
                }
            }
        }

        if self.response_domains.is_empty() {
            push(
                &mut findings,
                "missing_response_domains",
                "response_domains",
                "At least one response domain is required.",
            );
        }
        if !self.specimen_allocation.destructive_tests_isolated {
            push(
                &mut findings,
                "destructive_specimens_not_isolated",
                "specimen_allocation.destructive_tests_isolated",
                "Potentially destructive impact testing must use separately allocated specimens.",
            );
        }
        if self.specimen_allocation.sample_size_rationale.trim().is_empty() {
            push(
                &mut findings,
                "missing_sample_size_rationale",
                "specimen_allocation.sample_size_rationale",
                "Sample-size rationale must remain explicit, including when it is still pending.",
            );
        }
        if self.state == "preregistered" {
            if self.specimen_allocation.randomization_seed.is_none()
                || self.specimen_allocation.blocking_variables.is_empty()
                || !self.specimen_allocation.confirmatory_holdout
            {
                push(
                    &mut findings,
                    "incomplete_preregistration",
                    "specimen_allocation",
                    "Preregistration requires a fixed randomization seed, blocking variables and confirmatory holdout.",
                );
            }
        }

        if !self.evidence_policy.material_lot_binding
            || !self.evidence_policy.process_binding
            || !self.evidence_policy.instrument_calibration_binding
            || !self.evidence_policy.raw_data_digest_required
            || !self.evidence_policy.uncertainty_required
            || !self.evidence_policy.paper_results_not_lightshell_results
        {
            push(
                &mut findings,
                "weakened_evidence_policy",
                "evidence_policy",
                "Material lot, process, calibration, raw-data and uncertainty binding and paper/Lightshell claim separation are mandatory.",
            );
        }

        let matrix_complete = screen_count == 12
            && transition_refinement_count == 6
            && screen_cells.len() == 12
            && refinement_cells.len() == 6
            && candidate_ids.len() == 18;

        if !matrix_complete {
            push(
                &mut findings,
                "incomplete_spacer_matrix",
                "candidate_structures",
                format!(
                    "Expected 12 screening + 6 transition-refinement candidates; observed {} + {}.",
                    screen_count, transition_refinement_count
                ),
            );
        }

        ValidationReport {
            findings,
            screen_count,
            transition_refinement_count,
            matrix_complete,
        }
    }
}

fn push(
    findings: &mut Vec<Finding>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(Finding {
        code,
        path: path.into(),
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(
        "../../../../docs/research/fixtures/lightshell_spacer_doe_v0_1/lightshell_spacer_doe.example.json"
    );

    fn manifest() -> SpacerDoeManifest {
        serde_json::from_str(FIXTURE).expect("spacer DoE seed fixture must deserialize")
    }

    #[test]
    fn seed_matrix_is_valid_and_capability_pending() {
        let manifest = manifest();
        let report = manifest.validate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert!(report.matrix_complete);
        assert_eq!(report.screen_count, 12);
        assert_eq!(report.transition_refinement_count, 6);
        assert_eq!(manifest.process_capability.capability_state, "unknown");
    }

    #[test]
    fn screen_holds_square_wave_transition_geometry_constant() {
        let mut manifest = manifest();
        let candidate = manifest
            .candidate_structures
            .iter_mut()
            .find(|candidate| candidate.candidate_id == "A2-L")
            .unwrap();
        candidate.transition_geometry = "one_needle".into();
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "screen_transition_confounded"
        }));
    }

    #[test]
    fn transition_refinement_holds_spacer_frequency_constant() {
        let mut manifest = manifest();
        let candidate = manifest
            .candidate_structures
            .iter_mut()
            .find(|candidate| candidate.candidate_id == "A2-T-one_needle")
            .unwrap();
        candidate.spacer_frequency = "high".into();
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "transition_refinement_frequency_not_fixed"
        }));
    }

    #[test]
    fn missing_factorial_cell_is_rejected() {
        let mut manifest = manifest();
        manifest
            .candidate_structures
            .retain(|candidate| candidate.candidate_id != "A0-H");
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "missing_screen_cell"
        }));
        assert!(!report.matrix_complete);
    }

    #[test]
    fn duplicate_candidate_identity_is_rejected() {
        let mut manifest = manifest();
        let duplicate = manifest.candidate_structures[0].clone();
        manifest.candidate_structures.push(duplicate);
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "duplicate_or_empty_candidate_id"
        }));
    }

    #[test]
    fn required_factor_definition_cannot_be_omitted() {
        let mut manifest = manifest();
        manifest
            .factors
            .retain(|factor| factor.factor_id != "spacer_frequency");
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "missing_required_factor"
        }));
    }

    #[test]
    fn proposed_matrix_cannot_claim_a_material_lot_or_process_revision() {
        let mut manifest = manifest();
        manifest.candidate_structures[0].material_lot_ref =
            Some("material-lot-not-yet-procured".into());
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "proposed_fixture_contains_execution_claim"
        }));
    }

    #[test]
    fn preregistration_requires_process_capability_and_holdout() {
        let mut manifest = manifest();
        manifest.state = "preregistered".into();
        manifest.process_capability.capability_state = "unknown".into();
        manifest.specimen_allocation.randomization_seed = None;
        manifest.specimen_allocation.confirmatory_holdout = false;
        let report = manifest.validate();
        assert!(report.findings.iter().any(|finding| {
            finding.code == "preregistration_without_capability"
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == "incomplete_preregistration"
        }));
    }
}

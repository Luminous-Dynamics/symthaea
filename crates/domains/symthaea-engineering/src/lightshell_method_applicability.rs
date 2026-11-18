// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightshell-specific admission of textile test methods by edition,
//! specimen construction, tested surface, seam geometry and fixture.
//!
//! The catalogue pins public method metadata and known scope boundaries. It
//! does not reproduce licensed standard text, execute tests, accredit a lab,
//! or certify conformity.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodSelectionManifest {
    pub schema_version: String,
    pub campaign_id: String,
    pub state: String,
    pub specimens: Vec<SpecimenContext>,
    pub method_profiles: Vec<MethodProfile>,
    pub selections: Vec<MethodSelection>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecimenContext {
    pub specimen_id: String,
    pub candidate_id: String,
    pub construction_class: String,
    pub seam_geometry: String,
    pub thickness_mm: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodProfile {
    pub profile_id: String,
    pub profile_kind: String,
    pub method_id: String,
    pub edition: String,
    pub test_family: String,
    pub metadata_source_ref: Option<String>,
    pub metadata_state: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodSelection {
    pub selection_id: String,
    pub specimen_id: String,
    pub profile_id: String,
    pub test_family: String,
    pub selected_surface: String,
    pub fixture_class: String,
    pub fixture_ref: Option<String>,
    pub fixture_revision: Option<String>,
    pub fixture_validation_state: String,
    pub fixture_validation_ref: Option<String>,
    pub state: String,
    pub execution_mode: String,
    pub approved_deviation_ref: Option<String>,
    pub internal_method_validation_ref: Option<String>,
    pub protocol_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
    pub cleaning_profile: Option<CleaningProfile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleaningProfile {
    pub service_class: String,
    pub machine_type: Option<String>,
    pub detergent_ref: Option<String>,
    pub wash_procedure_ref: Option<String>,
    pub drying_procedure_ref: Option<String>,
    pub cycles: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub standard_edition_is_exact: bool,
    pub known_scope_exclusions_fail_closed: bool,
    pub spacer_fixture_validation_required: bool,
    pub modified_methods_are_not_standard_conformance: bool,
    pub internal_methods_are_not_iso_methods: bool,
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
    pub specimen_count: usize,
    pub profile_count: usize,
    pub selection_count: usize,
    pub ready_for_test_count: usize,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn can_start_tests(&self) -> bool {
        self.is_valid() && self.ready_for_test_count > 0
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: MethodSelectionManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl MethodSelectionManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut specimens: HashMap<&str, &SpecimenContext> = HashMap::new();
        let mut profiles: HashMap<&str, &MethodProfile> = HashMap::new();
        let mut selection_ids = HashSet::new();
        let mut ready_for_test_count = 0usize;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell method-applicability schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.campaign_id) {
            push(&mut findings, "missing_campaign_id", "campaign_id",
                "Campaign identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "method_review" | "ready_for_test") {
            push(&mut findings, "invalid_manifest_state", "state",
                "Manifest state must be proposed, method_review or ready_for_test.");
        }
        if self.specimens.is_empty() {
            push(&mut findings, "no_specimens", "specimens",
                "At least one specimen context must be declared.");
        }
        if self.method_profiles.is_empty() {
            push(&mut findings, "no_method_profiles", "method_profiles",
                "At least one versioned method profile must be declared.");
        }

        for (index, specimen) in self.specimens.iter().enumerate() {
            let path = format!("specimens[{index}]");
            if is_blank(&specimen.specimen_id) || is_blank(&specimen.candidate_id) {
                push(&mut findings, "missing_specimen_identity", &path,
                    "Specimen and candidate identities must be non-empty.");
            }
            if specimens.insert(specimen.specimen_id.as_str(), specimen).is_some() {
                push(&mut findings, "duplicate_specimen_id", format!("{path}.specimen_id"),
                    "Specimen IDs must be unique.");
            }
            if !is_known_construction_class(&specimen.construction_class) {
                push(&mut findings, "unknown_construction_class", format!("{path}.construction_class"),
                    "Unknown textile construction class.");
            }
            if !matches!(specimen.seam_geometry.as_str(), "not_applicable" | "straight" | "curved") {
                push(&mut findings, "invalid_seam_geometry", format!("{path}.seam_geometry"),
                    "Seam geometry must be not_applicable, straight or curved.");
            }
            if specimen.thickness_mm.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                push(&mut findings, "invalid_specimen_thickness", format!("{path}.thickness_mm"),
                    "Supplied specimen thickness must be finite and positive.");
            }
        }

        for (index, profile) in self.method_profiles.iter().enumerate() {
            let path = format!("method_profiles[{index}]");
            if is_blank(&profile.profile_id) || is_blank(&profile.method_id) || is_blank(&profile.edition) {
                push(&mut findings, "missing_method_identity", &path,
                    "Profile, method and edition identities must be non-empty.");
            }
            if profiles.insert(profile.profile_id.as_str(), profile).is_some() {
                push(&mut findings, "duplicate_method_profile_id", format!("{path}.profile_id"),
                    "Method profile IDs must be unique.");
            }
            validate_profile(profile, &path, &mut findings);
        }

        for (index, selection) in self.selections.iter().enumerate() {
            let path = format!("selections[{index}]");
            if is_blank(&selection.selection_id) {
                push(&mut findings, "blank_selection_id", format!("{path}.selection_id"),
                    "Selection identity must be non-empty.");
            }
            if !selection_ids.insert(selection.selection_id.as_str()) {
                push(&mut findings, "duplicate_selection_id", format!("{path}.selection_id"),
                    "Selection IDs must be unique.");
            }
            let specimen = specimens.get(selection.specimen_id.as_str()).copied();
            let profile = profiles.get(selection.profile_id.as_str()).copied();
            if specimen.is_none() {
                push(&mut findings, "selection_unknown_specimen", format!("{path}.specimen_id"),
                    "Selection references a specimen absent from this manifest.");
            }
            if profile.is_none() {
                push(&mut findings, "selection_unknown_profile", format!("{path}.profile_id"),
                    "Selection references a method profile absent from this manifest.");
            }
            if let Some(profile) = profile {
                if selection.test_family != profile.test_family {
                    push(&mut findings, "selection_family_mismatch", format!("{path}.test_family"),
                        "Selected test family differs from the method profile's declared family.");
                }
            }
            if !matches!(selection.state.as_str(), "proposed" | "ready_for_test" | "rejected") {
                push(&mut findings, "invalid_selection_state", format!("{path}.state"),
                    "Selection state must be proposed, ready_for_test or rejected.");
            }
            if !matches!(selection.fixture_validation_state.as_str(), "unknown" | "pending" | "qualified_generic" | "qualified_spacer" | "qualified_coated" | "qualified_article") {
                push(&mut findings, "invalid_fixture_validation_state", format!("{path}.fixture_validation_state"),
                    "Unknown fixture validation state.");
            }
            if !matches!(selection.execution_mode.as_str(), "normative_unmodified" | "modified_with_deviation" | "internal_protocol") {
                push(&mut findings, "invalid_execution_mode", format!("{path}.execution_mode"),
                    "Execution mode must distinguish unmodified standard, approved deviation and internal protocol.");
            }
            if let Some(profile) = profile {
                validate_selection_method(profile, selection, specimen, &path, &mut findings);
            }
            if selection.execution_mode == "modified_with_deviation" && is_blank_option(&selection.approved_deviation_ref) {
                push(&mut findings, "modified_method_missing_approval", format!("{path}.approved_deviation_ref"),
                    "A modified method requires an approved deviation reference and cannot be claimed as unmodified standard conformance.");
            }
            if selection.execution_mode == "internal_protocol" && is_blank_option(&selection.internal_method_validation_ref) {
                push(&mut findings, "internal_method_missing_validation", format!("{path}.internal_method_validation_ref"),
                    "An internal protocol requires an explicit method-validation reference.");
            }
            if selection.execution_mode == "normative_unmodified" && selection.approved_deviation_ref.is_some() {
                push(&mut findings, "unmodified_method_has_deviation", format!("{path}.approved_deviation_ref"),
                    "An unmodified standard method cannot carry an active deviation reference.");
            }

            if selection.state == "ready_for_test" {
                ready_for_test_count += 1;
                validate_ready_selection(selection, profile, specimen, &path, &mut findings);
            }
        }

        if self.state == "ready_for_test" {
            if self.selections.is_empty() || self.selections.iter().any(|selection| selection.state != "ready_for_test") {
                push(&mut findings, "manifest_has_unready_selections", "selections",
                    "A ready_for_test manifest cannot contain proposed or rejected selections.");
            }
            if ready_for_test_count == 0 {
                push(&mut findings, "manifest_without_ready_selection", "selections",
                    "A ready_for_test manifest requires at least one ready selection.");
            }
        }

        if !self.evidence_policy.standard_edition_is_exact
            || !self.evidence_policy.known_scope_exclusions_fail_closed
            || !self.evidence_policy.spacer_fixture_validation_required
            || !self.evidence_policy.modified_methods_are_not_standard_conformance
            || !self.evidence_policy.internal_methods_are_not_iso_methods
        {
            push(&mut findings, "unsafe_method_evidence_policy", "evidence_policy",
                "Exact editions, known exclusions, fixture validation and method-deviation boundaries must remain mandatory.");
        }

        ValidationReport {
            findings,
            specimen_count: self.specimens.len(),
            profile_count: self.method_profiles.len(),
            selection_count: self.selections.len(),
            ready_for_test_count,
        }
    }
}

fn validate_profile(profile: &MethodProfile, path: &str, findings: &mut Vec<Finding>) {
    if !matches!(profile.profile_kind.as_str(), "international_standard" | "internal_protocol") {
        push(findings, "invalid_profile_kind", format!("{path}.profile_kind"),
            "Profile kind must be international_standard or internal_protocol.");
    }
    if !matches!(profile.metadata_state.as_str(), "verified_current_metadata" | "superseded" | "review_pending" | "internal_validation_pending" | "internal_validated") {
        push(findings, "invalid_metadata_state", format!("{path}.metadata_state"),
            "Unknown method metadata or validation state.");
    }

    if profile.profile_kind == "international_standard" {
        if profile.metadata_source_ref.as_deref().is_none_or(is_blank) {
            push(findings, "standard_missing_metadata_source", format!("{path}.metadata_source_ref"),
                "A standard method profile must cite an official public metadata source.");
        }
        match known_standard(&profile.method_id) {
            Some(known) => {
                if profile.edition != known.edition {
                    push(findings, "standard_edition_mismatch", format!("{path}.edition"),
                        "The selected standard edition is not the current method edition pinned in the Lightshell metadata crosswalk.");
                }
                if profile.test_family != known.family {
                    push(findings, "standard_family_mismatch", format!("{path}.test_family"),
                        "The test family does not match the pinned standard method.");
                }
                if profile.metadata_source_ref.as_deref() != Some(known.source) {
                    push(findings, "standard_source_mismatch", format!("{path}.metadata_source_ref"),
                        "The metadata source must be the pinned official standard catalogue record.");
                }
                if profile.metadata_state != "verified_current_metadata" {
                    push(findings, "standard_not_current_metadata", format!("{path}.metadata_state"),
                        "A pinned current standard profile must be marked verified_current_metadata.");
                }
            }
            None => push(findings, "unknown_standard_method", format!("{path}.method_id"),
                "Standard method is not in the source-reviewed Lightshell metadata crosswalk; add it through research review."),
        }
    } else {
        if !profile.method_id.starts_with("luminous:") {
            push(findings, "internal_method_namespace_required", format!("{path}.method_id"),
                "An internal protocol must use the luminous: namespace and must not impersonate an ISO identifier.");
        }
        if profile.metadata_source_ref.is_some() {
            push(findings, "internal_method_has_standard_source", format!("{path}.metadata_source_ref"),
                "An internal protocol must not cite an ISO standard as its method identity.");
        }
        if profile.metadata_state != "internal_validation_pending" && profile.metadata_state != "internal_validated" {
            push(findings, "invalid_internal_method_state", format!("{path}.metadata_state"),
                "Internal protocols must be pending validation or internally validated.");
        }
    }
}

fn validate_selection_method(
    profile: &MethodProfile,
    selection: &MethodSelection,
    specimen: Option<&SpecimenContext>,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    if profile.profile_kind == "international_standard" && selection.execution_mode == "internal_protocol" {
        push(findings, "standard_selected_as_internal", format!("{path}.execution_mode"),
            "A registered international standard must not be relabelled as an internal protocol.");
    }
    if profile.profile_kind == "internal_protocol" && selection.execution_mode != "internal_protocol" {
        push(findings, "internal_protocol_claims_standard_mode", format!("{path}.execution_mode"),
            "An internal method cannot claim normative or standard-deviation execution mode.");
    }

    let Some(specimen) = specimen else { return; };
    match profile.method_id.as_str() {
        "ISO 13935-2" => {
            if specimen.seam_geometry == "curved" {
                push(findings, "iso_13935_2_curved_seam_out_of_scope", format!("{path}.specimen_id"),
                    "ISO 13935-2:2026 is for straight sewn seams; curved-seam use requires a separately validated method.");
            }
            if matches!(specimen.construction_class.as_str(), "nonwoven" | "coated_laminated") {
                push(findings, "iso_13935_2_construction_out_of_scope", format!("{path}.specimen_id"),
                    "ISO 13935-2:2026 is normally not applicable to nonwovens or coated fabrics.");
            }
            if selection.selected_surface != "seam" || selection.fixture_class != "cre_tensile" {
                push(findings, "iso_13935_2_fixture_or_surface_mismatch", path,
                    "ISO 13935-2:2026 selection requires seam testing on a CRE tensile fixture.");
            }
        }
        "ISO 12947-2" => {
            if specimen.construction_class == "coated_laminated" {
                push(findings, "iso_12947_2_coated_material_out_of_scope", format!("{path}.specimen_id"),
                    "ISO 12947-2:2016 excludes coated and laminated fabrics; use a coated-surface method where applicable.");
            }
            if selection.selected_surface != "uncoated_surface" || selection.fixture_class != "martindale_uncoated" {
                push(findings, "iso_12947_2_fixture_or_surface_mismatch", path,
                    "ISO 12947-2:2016 selection requires an uncoated textile surface and uncoated Martindale fixture profile.");
            }
        }
        "ISO 5470-2" => {
            if specimen.construction_class != "coated_laminated" || selection.selected_surface != "coated_surface" {
                push(findings, "iso_5470_2_surface_out_of_scope", path,
                    "ISO 5470-2:2021 is for abrasion of coated surfaces, not uncoated faces.");
            }
            if selection.fixture_class != "martindale_coated" {
                push(findings, "iso_5470_2_fixture_mismatch", path,
                    "ISO 5470-2:2021 selection requires the coated-surface Martindale fixture profile.");
            }
        }
        "ISO 9237" => {
            if selection.selected_surface != "through_thickness" || selection.fixture_class != "air_permeability" && selection.fixture_class != "spacer_airflow_adapter" {
                push(findings, "iso_9237_surface_or_fixture_mismatch", path,
                    "ISO 9237 selection requires through-thickness airflow measurement using an air-permeability fixture.");
            }
            if specimen.construction_class == "3d_spacer"
                && (selection.fixture_class != "spacer_airflow_adapter"
                    || selection.fixture_validation_state != "qualified_spacer"
                    || is_blank_option(&selection.fixture_validation_ref))
            {
                push(findings, "spacer_airflow_fixture_not_validated", path,
                    "A 3D spacer specimen requires an explicitly spacer-qualified airflow fixture and validation reference.");
            }
        }
        "ISO 6330" => {
            if selection.test_family != "cleaning_retention" {
                push(findings, "iso_6330_family_mismatch", format!("{path}.test_family"),
                    "ISO 6330 is a washing/drying procedure, not a property test family.");
            }
            match selection.cleaning_profile.as_ref() {
                Some(cleaning) => {
                    if cleaning.service_class != "domestic_wash" {
                        push(findings, "iso_6330_cleaning_scope_mismatch", format!("{path}.cleaning_profile.service_class"),
                            "ISO 6330 specifies domestic washing and drying procedures; do not claim it covers industrial or robot-specific cleaning.");
                    }
                    if cleaning.machine_type.as_deref().is_none_or(is_blank)
                        || cleaning.detergent_ref.as_deref().is_none_or(is_blank)
                        || cleaning.wash_procedure_ref.as_deref().is_none_or(is_blank)
                        || cleaning.drying_procedure_ref.as_deref().is_none_or(is_blank)
                        || cleaning.cycles == 0
                    {
                        push(findings, "iso_6330_profile_incomplete", format!("{path}.cleaning_profile"),
                            "ISO 6330 selection must bind machine type, detergent, wash procedure, drying procedure and nonzero cycle count.");
                    }
                    if !matches!(cleaning.machine_type.as_deref(), Some("A" | "B" | "C")) {
                        push(findings, "iso_6330_unknown_machine_type", format!("{path}.cleaning_profile.machine_type"),
                            "ISO 6330:2021 defines reference machine types A, B and C.");
                    }
                }
                None => push(findings, "iso_6330_cleaning_profile_missing", format!("{path}.cleaning_profile"),
                    "A washing-method selection must preserve its exact cleaning profile."),
            }
            if selection.fixture_class != "washing_machine" {
                push(findings, "iso_6330_fixture_mismatch", path,
                    "ISO 6330 requires a declared reference washing-machine fixture profile.");
            }
        }
        _ => {}
    }
}

fn validate_ready_selection(
    selection: &MethodSelection,
    profile: Option<&MethodProfile>,
    specimen: Option<&SpecimenContext>,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    for (field, value) in [
        ("fixture_ref", selection.fixture_ref.as_deref()),
        ("fixture_revision", selection.fixture_revision.as_deref()),
        ("fixture_validation_ref", selection.fixture_validation_ref.as_deref()),
        ("protocol_ref", selection.protocol_ref.as_deref()),
        ("acceptance_rule_ref", selection.acceptance_rule_ref.as_deref()),
    ] {
        if value.is_none_or(is_blank) {
            push(findings, "ready_selection_missing_evidence", format!("{path}.{field}"),
                "A ready-for-test selection must bind its exact fixture, validation, protocol and acceptance rule.");
        }
    }
    if selection.fixture_validation_state == "unknown" || selection.fixture_validation_state == "pending" {
        push(findings, "ready_selection_fixture_not_qualified", format!("{path}.fixture_validation_state"),
            "A ready-for-test selection requires a qualified fixture profile.");
    }
    if selection.state == "ready_for_test" {
        if let Some(profile) = profile {
            if profile.metadata_state == "superseded" || profile.metadata_state == "review_pending" {
                push(findings, "ready_selection_method_not_current", format!("{path}.profile_id"),
                    "A ready-for-test selection cannot use superseded or review-pending method metadata.");
            }
            if profile.profile_kind == "internal_protocol"
                && profile.metadata_state != "internal_validated"
            {
                push(findings, "ready_selection_internal_method_unvalidated", format!("{path}.profile_id"),
                    "A ready-for-test selection requires a validated internal method.");
            }
        }
        if let Some(specimen) = specimen {
            if specimen.construction_class == "3d_spacer"
                && selection.fixture_validation_state != "qualified_spacer"
            {
                push(findings, "ready_spacer_requires_spacer_fixture", format!("{path}.fixture_validation_state"),
                    "A ready 3D spacer test requires a qualified spacer-specific fixture.");
            }
        }
        if selection.execution_mode == "modified_with_deviation" && is_blank_option(&selection.approved_deviation_ref) {
            push(findings, "ready_modified_method_not_approved", format!("{path}.approved_deviation_ref"),
                "A modified method cannot become ready without approved deviation disposition.");
        }
        if selection.execution_mode == "internal_protocol" && is_blank_option(&selection.internal_method_validation_ref) {
            push(findings, "ready_internal_method_not_validated", format!("{path}.internal_method_validation_ref"),
                "An internal method requires its independent validation record.");
        }
    }
}

struct KnownStandard {
    edition: &'static str,
    family: &'static str,
    source: &'static str,
}

fn known_standard(method_id: &str) -> Option<KnownStandard> {
    match method_id {
        "ISO 13935-2" => Some(KnownStandard {
            edition: "2026",
            family: "seam_tensile",
            source: "https://www.iso.org/standard/88344.html",
        }),
        "ISO 12947-2" => Some(KnownStandard {
            edition: "2016",
            family: "abrasion",
            source: "https://www.iso.org/standard/61058.html",
        }),
        "ISO 5470-2" => Some(KnownStandard {
            edition: "2021",
            family: "abrasion",
            source: "https://www.iso.org/standard/77552.html",
        }),
        "ISO 9237" => Some(KnownStandard {
            edition: "1995",
            family: "airflow",
            source: "https://www.iso.org/standard/16869.html",
        }),
        "ISO 6330" => Some(KnownStandard {
            edition: "2021",
            family: "cleaning_retention",
            source: "https://www.iso.org/standard/75934.html",
        }),
        _ => None,
    }
}

fn is_known_construction_class(value: &str) -> bool {
    matches!(value,
        "woven_or_knit" | "nonwoven" | "coated_laminated" |
        "3d_spacer" | "made_up_article" | "integrated_assembly" | "unknown")
}

fn push(findings: &mut Vec<Finding>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn base_manifest() -> Value {
        json!({
            "schema_version": "0.1.0",
            "campaign_id": "LIGHTSHELL-METHOD-SYNTHETIC-001",
            "state": "proposed",
            "specimens": [
                {"specimen_id":"SP-UNCOATED","candidate_id":"A0-R","construction_class":"woven_or_knit","seam_geometry":"not_applicable","thickness_mm":0.8},
                {"specimen_id":"SP-COATED","candidate_id":"C-COATED","construction_class":"coated_laminated","seam_geometry":"not_applicable","thickness_mm":1.2},
                {"specimen_id":"SP-SPACER","candidate_id":"A2-R","construction_class":"3d_spacer","seam_geometry":"not_applicable","thickness_mm":8.0},
                {"specimen_id":"SP-SEAM","candidate_id":"SEAM-01","construction_class":"woven_or_knit","seam_geometry":"straight","thickness_mm":1.0},
                {"specimen_id":"SP-NONWOVEN","candidate_id":"NW-01","construction_class":"nonwoven","seam_geometry":"not_applicable","thickness_mm":2.0}
            ],
            "method_profiles": [
                {"profile_id":"M-ISO-13935-2-2026","profile_kind":"international_standard","method_id":"ISO 13935-2","edition":"2026","test_family":"seam_tensile","metadata_source_ref":"https://www.iso.org/standard/88344.html","metadata_state":"verified_current_metadata"},
                {"profile_id":"M-ISO-12947-2-2016","profile_kind":"international_standard","method_id":"ISO 12947-2","edition":"2016","test_family":"abrasion","metadata_source_ref":"https://www.iso.org/standard/61058.html","metadata_state":"verified_current_metadata"},
                {"profile_id":"M-ISO-5470-2-2021","profile_kind":"international_standard","method_id":"ISO 5470-2","edition":"2021","test_family":"abrasion","metadata_source_ref":"https://www.iso.org/standard/77552.html","metadata_state":"verified_current_metadata"},
                {"profile_id":"M-ISO-9237-1995","profile_kind":"international_standard","method_id":"ISO 9237","edition":"1995","test_family":"airflow","metadata_source_ref":"https://www.iso.org/standard/16869.html","metadata_state":"verified_current_metadata"},
                {"profile_id":"M-ISO-6330-2021","profile_kind":"international_standard","method_id":"ISO 6330","edition":"2021","test_family":"cleaning_retention","metadata_source_ref":"https://www.iso.org/standard/75934.html","metadata_state":"verified_current_metadata"}
            ],
            "selections": [
                {
                    "selection_id":"SEL-SEAM","specimen_id":"SP-SEAM","profile_id":"M-ISO-13935-2-2026","test_family":"seam_tensile","selected_surface":"seam","fixture_class":"cre_tensile","fixture_ref":null,"fixture_revision":null,"fixture_validation_state":"pending","fixture_validation_ref":null,"state":"proposed","execution_mode":"normative_unmodified","approved_deviation_ref":null,"internal_method_validation_ref":null,"protocol_ref":null,"acceptance_rule_ref":null,"cleaning_profile":null
                },
                {
                    "selection_id":"SEL-UNCOATED-ABRASION","specimen_id":"SP-UNCOATED","profile_id":"M-ISO-12947-2-2016","test_family":"abrasion","selected_surface":"uncoated_surface","fixture_class":"martindale_uncoated","fixture_ref":null,"fixture_revision":null,"fixture_validation_state":"pending","fixture_validation_ref":null,"state":"proposed","execution_mode":"normative_unmodified","approved_deviation_ref":null,"internal_method_validation_ref":null,"protocol_ref":null,"acceptance_rule_ref":null,"cleaning_profile":null
                },
                {
                    "selection_id":"SEL-COATED-ABRASION","specimen_id":"SP-COATED","profile_id":"M-ISO-5470-2-2021","test_family":"abrasion","selected_surface":"coated_surface","fixture_class":"martindale_coated","fixture_ref":null,"fixture_revision":null,"fixture_validation_state":"pending","fixture_validation_ref":null,"state":"proposed","execution_mode":"normative_unmodified","approved_deviation_ref":null,"internal_method_validation_ref":null,"protocol_ref":null,"acceptance_rule_ref":null,"cleaning_profile":null
                },
                {
                    "selection_id":"SEL-SPACER-AIRFLOW","specimen_id":"SP-SPACER","profile_id":"M-ISO-9237-1995","test_family":"airflow","selected_surface":"through_thickness","fixture_class":"spacer_airflow_adapter","fixture_ref":null,"fixture_revision":null,"fixture_validation_state":"pending","fixture_validation_ref":null,"state":"proposed","execution_mode":"normative_unmodified","approved_deviation_ref":null,"internal_method_validation_ref":null,"protocol_ref":null,"acceptance_rule_ref":null,"cleaning_profile":null
                }
            ],
            "evidence_policy": {
                "standard_edition_is_exact":true,
                "known_scope_exclusions_fail_closed":true,
                "spacer_fixture_validation_required":true,
                "modified_methods_are_not_standard_conformance":true,
                "internal_methods_are_not_iso_methods":true
            }
        })
    }

    #[test]
    fn proposed_method_portfolio_records_exact_editions_without_claiming_test_readiness() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.specimen_count, 5);
        assert_eq!(report.profile_count, 5);
        assert_eq!(report.selection_count, 4);
        assert_eq!(report.ready_for_test_count, 0);
        assert!(!report.can_start_tests());
    }

    #[test]
    fn iso_13935_2_2026_rejects_curved_seams() {
        let mut manifest = base_manifest();
        manifest["specimens"][3]["seam_geometry"] = json!("curved");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "iso_13935_2_curved_seam_out_of_scope"));
    }

    #[test]
    fn iso_13935_2_2026_rejects_nonwovens_and_coated_fabrics() {
        let mut manifest = base_manifest();
        let mut selection = manifest["selections"][0].clone();
        selection["selection_id"] = json!("SEL-NONWOVEN-SEAM");
        selection["specimen_id"] = json!("SP-NONWOVEN");
        manifest["selections"].as_array_mut().unwrap().push(selection);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "iso_13935_2_construction_out_of_scope"));
    }

    #[test]
    fn iso_12947_2_rejects_coated_laminated_fabric() {
        let mut manifest = base_manifest();
        let mut selection = manifest["selections"][1].clone();
        selection["selection_id"] = json!("SEL-WRONG-ABRASION");
        selection["specimen_id"] = json!("SP-COATED");
        manifest["selections"].as_array_mut().unwrap().push(selection);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "iso_12947_2_coated_material_out_of_scope"));
    }

    #[test]
    fn iso_5470_2_rejects_uncoated_surface() {
        let mut manifest = base_manifest();
        manifest["selections"][2]["selected_surface"] = json!("uncoated_surface");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "iso_5470_2_surface_out_of_scope"));
    }

    #[test]
    fn iso_9237_spacer_cannot_be_ready_without_qualified_spacer_fixture() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("ready_for_test");
        manifest["selections"][3]["state"] = json!("ready_for_test");
        manifest["selections"][3]["fixture_validation_state"] = json!("qualified_generic");
        manifest["selections"][3]["fixture_ref"] = json!("AIR-FIXTURE-01");
        manifest["selections"][3]["fixture_revision"] = json!("REV-A");
        manifest["selections"][3]["fixture_validation_ref"] = json!("GENERIC-FIXTURE-VALIDATION");
        manifest["selections"][3]["protocol_ref"] = json!("ISO-9237-PROTOCOL-REV-A");
        manifest["selections"][3]["acceptance_rule_ref"] = json!("ACCEPTANCE-REV-A");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "spacer_airflow_fixture_not_validated"));
        assert!(report.findings.iter().any(|f| f.code == "ready_spacer_requires_spacer_fixture"));
    }

    #[test]
    fn stale_2014_seam_method_is_not_silently_promoted() {
        let mut manifest = base_manifest();
        manifest["method_profiles"][0]["edition"] = json!("2014");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "standard_edition_mismatch"));
    }

    #[test]
    fn internal_protocol_cannot_impersonate_iso() {
        let mut manifest = base_manifest();
        manifest["method_profiles"][4]["profile_kind"] = json!("internal_protocol");
        manifest["method_profiles"][4]["method_id"] = json!("ISO 6330");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "internal_method_namespace_required"));
    }

    #[test]
    fn iso_6330_requires_domestic_wash_parameters_and_machine_type() {
        let mut manifest = base_manifest();
        manifest["selections"].push(json!({
            "selection_id":"SEL-WASH","specimen_id":"SP-UNCOATED","profile_id":"M-ISO-6330-2021",
            "test_family":"cleaning_retention","selected_surface":"whole_article","fixture_class":"washing_machine",
            "fixture_ref":null,"fixture_revision":null,"fixture_validation_state":"pending","fixture_validation_ref":null,
            "state":"proposed","execution_mode":"normative_unmodified","approved_deviation_ref":null,
            "internal_method_validation_ref":null,"protocol_ref":null,"acceptance_rule_ref":null,
            "cleaning_profile":{"service_class":"robot_cleaning","machine_type":null,"detergent_ref":null,"wash_procedure_ref":null,"drying_procedure_ref":null,"cycles":0}
        }));
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "iso_6330_cleaning_scope_mismatch"));
        assert!(report.findings.iter().any(|f| f.code == "iso_6330_profile_incomplete"));
    }

    #[test]
    fn modified_method_requires_approved_deviation() {
        let mut manifest = base_manifest();
        manifest["selections"][1]["execution_mode"] = json!("modified_with_deviation");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "modified_method_missing_approval"));
    }

    #[test]
    fn ready_selection_requires_exact_fixture_and_protocol_evidence() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("ready_for_test");
        manifest["selections"][0]["state"] = json!("ready_for_test");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "ready_selection_missing_evidence"));
        assert!(report.findings.iter().any(|f| f.code == "ready_selection_fixture_not_qualified"));
    }

    #[test]
    fn unknown_standard_method_is_not_assumed_applicable() {
        let mut manifest = base_manifest();
        manifest["method_profiles"][0]["method_id"] = json!("ISO 99999");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unknown_standard_method"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["selections"][0]["assume_applicable"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}

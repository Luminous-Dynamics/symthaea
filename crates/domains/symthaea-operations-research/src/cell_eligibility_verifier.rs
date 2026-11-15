// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent soundness verifier for positive workcell eligibility claims.
//!
//! This module intentionally does not call the eligibility evaluator or reuse
//! its collector/helpers. It independently recomputes hard predicates from the
//! same immutable owner-resolved facts. It is a soundness check for positive
//! claims, not a second planner and not source-document truth verification.

use std::collections::BTreeSet;

use crate::cell_eligibility::{
    AssetStatus, Availability, CellProfileSnapshot, EvidenceRecord, EvidenceStatus,
    EligibilityReport, EligibilityStatus, MaterialLotStatus, OperationRequirement,
    ProfileStatus, SnapshotResolution,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerificationFindingCode {
    ReportNotEligible,
    ReportIdentityMismatch,
    ReportContainsReasons,
    SnapshotNotComplete,
    MalformedRequirement,
    ProfileNotQualified,
    ProfileOutsideValidity,
    ProfileValidityUnknown,
    ProfileQualificationMissing,
    IndependentVerificationMissing,
    EvidenceInvalid,
    EvidenceReferenceMissing,
    ConfigurationMissing,
    ConfigurationAmbiguous,
    ConfigurationOutsideValidity,
    ConfigurationValidityUnknown,
    ConfigurationDigestInvalid,
    ConfigurationDigestMismatch,
    ConfigurationRevisionMismatch,
    OperationNotInConfiguration,
    InspectionPlanMismatch,
    AssetSetEmpty,
    AssetMissing,
    AssetAmbiguous,
    AssetNotAvailable,
    AssetCapabilityMismatch,
    AssetToolingMismatch,
    SkillRequirementInvalid,
    SkillMissing,
    SkillAmbiguous,
    SkillNotAvailable,
    SkillHeadcountUnknown,
    SkillHeadcountInsufficient,
    MaterialLotMissing,
    MaterialLotAmbiguous,
    MaterialLotNotAccepted,
    MaterialEnvelopeMismatch,
    UtilityMissing,
    UtilityAmbiguous,
    UtilityNotAvailable,
    UtilityUnitMismatch,
    UtilityCapacityInsufficient,
    ClearanceMissing,
    ClearanceAmbiguous,
    EvidenceSetMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VerificationFinding {
    pub code: VerificationFindingCode,
    pub subject_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Verified,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibleClaimVerification {
    pub operation_ref: String,
    pub profile_ref: String,
    pub verified_at_unix_seconds: i64,
    pub status: VerificationStatus,
    pub findings: Vec<VerificationFinding>,
    pub independently_resolved_evidence_refs: Vec<String>,
}

/// Verify a positive eligibility claim without trusting the evaluator's output.
///
/// The verifier recomputes identity, profile/configuration validity, exact
/// revisions and digest, asset/tooling, worker headcount, material admission,
/// utility capacity, clearances, and evidence currency. A report can be Verified
/// only when every hard predicate is satisfied and the evidence-reference set
/// matches the verifier's independently constructed set.
pub fn verify_eligible_claim(
    requirement: &OperationRequirement,
    cell: &CellProfileSnapshot,
    claim: &EligibilityReport,
) -> EligibleClaimVerification {
    let at = requirement.scheduled_at_unix_seconds;
    let mut findings = Vec::new();
    let mut evidence_refs = BTreeSet::new();

    if claim.status != EligibilityStatus::Eligible {
        add(&mut findings, VerificationFindingCode::ReportNotEligible, &claim.operation_ref);
    }
    if claim.operation_ref != requirement.operation_ref
        || claim.profile_ref != cell.profile_ref
        || claim.facility_ref != cell.facility_ref
        || claim.evaluated_at_unix_seconds != at
        || requirement.facility_ref != cell.facility_ref
    {
        add(&mut findings, VerificationFindingCode::ReportIdentityMismatch, &claim.profile_ref);
    }
    if !claim.reasons.is_empty() {
        add(&mut findings, VerificationFindingCode::ReportContainsReasons, &claim.profile_ref);
    }
    if cell.resolution != SnapshotResolution::Complete {
        add(&mut findings, VerificationFindingCode::SnapshotNotComplete, &cell.profile_ref);
    }

    for value in [
        requirement.operation_ref.as_str(),
        requirement.facility_ref.as_str(),
        requirement.product_revision_ref.as_str(),
        requirement.pattern_revision_ref.as_str(),
        requirement.process_revision_ref.as_str(),
        requirement.process_capability_ref.as_str(),
        requirement.process_configuration_ref.as_str(),
        requirement.required_parameter_digest_sha256.as_str(),
        requirement.inspection_plan_ref.as_str(),
        cell.profile_ref.as_str(),
        cell.facility_ref.as_str(),
    ] {
        if value.trim().is_empty() {
            add(&mut findings, VerificationFindingCode::MalformedRequirement, "required_reference");
        }
    }
    for reference in &requirement.required_tooling_refs {
        if reference.trim().is_empty() {
            add(&mut findings, VerificationFindingCode::MalformedRequirement, "required_tooling_ref");
        }
    }
    for reference in &requirement.required_material_lot_refs {
        if reference.trim().is_empty() {
            add(&mut findings, VerificationFindingCode::MalformedRequirement, "required_material_lot_ref");
        }
    }
    for reference in &requirement.required_clearance_refs {
        if reference.trim().is_empty() {
            add(&mut findings, VerificationFindingCode::MalformedRequirement, "required_clearance_ref");
        }
    }
    if requirement.required_material_envelope_ref.as_deref().is_some_and(|value| value.trim().is_empty()) {
        add(&mut findings, VerificationFindingCode::MalformedRequirement, "required_material_envelope_ref");
    }
    for demand in &requirement.utility_requirements {
        if demand.service_ref.trim().is_empty() || demand.unit_ref.trim().is_empty() {
            add(&mut findings, VerificationFindingCode::MalformedRequirement, "utility_requirement");
        }
    }

    if cell.status != ProfileStatus::Qualified {
        add(&mut findings, VerificationFindingCode::ProfileNotQualified, &cell.profile_ref);
    }
    match (cell.valid_from_unix_seconds, cell.valid_until_unix_seconds) {
        (Some(from), Some(until)) if from <= at && at < until => {}
        (Some(_), Some(_)) => add(&mut findings, VerificationFindingCode::ProfileOutsideValidity, &cell.profile_ref),
        _ => add(&mut findings, VerificationFindingCode::ProfileValidityUnknown, &cell.profile_ref),
    }

    if cell.qualification_evidence.is_empty() {
        add(&mut findings, VerificationFindingCode::ProfileQualificationMissing, &cell.profile_ref);
    }
    for evidence in &cell.qualification_evidence {
        verify_evidence(evidence, at, &mut findings, &mut evidence_refs);
    }

    if cell.independent_verification_evidence.is_empty() {
        add(&mut findings, VerificationFindingCode::IndependentVerificationMissing, &cell.profile_ref);
    }
    for evidence in &cell.independent_verification_evidence {
        verify_evidence(evidence, at, &mut findings, &mut evidence_refs);
    }

    let config_matches: Vec<_> = cell.process_configurations.iter()
        .filter(|item| item.configuration_ref == requirement.process_configuration_ref)
        .collect();
    let config = match config_matches.as_slice() {
        [] => {
            add(&mut findings, VerificationFindingCode::ConfigurationMissing, &requirement.process_configuration_ref);
            None
        }
        [only] => Some(*only),
        _ => {
            add(&mut findings, VerificationFindingCode::ConfigurationAmbiguous, &requirement.process_configuration_ref);
            None
        }
    };

    if let Some(config) = config {
        match (config.valid_from_unix_seconds, config.valid_until_unix_seconds) {
            (Some(from), Some(until)) if from <= at && at < until => {}
            (Some(_), Some(_)) => add(&mut findings, VerificationFindingCode::ConfigurationOutsideValidity, &config.configuration_ref),
            _ => add(&mut findings, VerificationFindingCode::ConfigurationValidityUnknown, &config.configuration_ref),
        }

        if !is_sha256_hex(&config.parameter_digest_sha256)
            || !is_sha256_hex(&requirement.required_parameter_digest_sha256)
        {
            add(&mut findings, VerificationFindingCode::ConfigurationDigestInvalid, &config.configuration_ref);
        } else if !config.parameter_digest_sha256.eq_ignore_ascii_case(&requirement.required_parameter_digest_sha256) {
            add(&mut findings, VerificationFindingCode::ConfigurationDigestMismatch, &config.configuration_ref);
        }

        if config.process_revision_ref != requirement.process_revision_ref
            || config.process_capability_ref != requirement.process_capability_ref
            || config.product_revision_ref != requirement.product_revision_ref
            || config.pattern_revision_ref != requirement.pattern_revision_ref
        {
            add(&mut findings, VerificationFindingCode::ConfigurationRevisionMismatch, &config.configuration_ref);
        }
        if !config.operation_refs.iter().any(|item| item == &requirement.operation_ref) {
            add(&mut findings, VerificationFindingCode::OperationNotInConfiguration, &requirement.operation_ref);
        }
        if config.inspection_plan_ref != requirement.inspection_plan_ref {
            add(&mut findings, VerificationFindingCode::InspectionPlanMismatch, &config.configuration_ref);
        }
        verify_evidence(&config.qualification_evidence, at, &mut findings, &mut evidence_refs);
        verify_evidence(&config.inspection_plan_evidence, at, &mut findings, &mut evidence_refs);

        if config.asset_refs.is_empty() {
            add(&mut findings, VerificationFindingCode::AssetSetEmpty, &config.configuration_ref);
        }
        for asset_ref in &config.asset_refs {
            if asset_ref.trim().is_empty() {
                add(&mut findings, VerificationFindingCode::AssetMissing, &config.configuration_ref);
                continue;
            }
            let matches: Vec<_> = cell.assets.iter().filter(|asset| asset.asset_ref == *asset_ref).collect();
            let asset = match matches.as_slice() {
                [] => {
                    add(&mut findings, VerificationFindingCode::AssetMissing, asset_ref);
                    None
                }
                [only] => Some(*only),
                _ => {
                    add(&mut findings, VerificationFindingCode::AssetAmbiguous, asset_ref);
                    None
                }
            };
            if let Some(asset) = asset {
                if asset.status != AssetStatus::Available {
                    add(&mut findings, VerificationFindingCode::AssetNotAvailable, asset_ref);
                }
                if !asset.capability_refs.iter().any(|item| item == &requirement.process_capability_ref) {
                    add(&mut findings, VerificationFindingCode::AssetCapabilityMismatch, asset_ref);
                }
                if !all_present(&asset.tooling_refs, &requirement.required_tooling_refs)
                    || !all_present(&config.tooling_refs, &requirement.required_tooling_refs)
                {
                    add(&mut findings, VerificationFindingCode::AssetToolingMismatch, asset_ref);
                }
                verify_evidence(&asset.calibration_evidence, at, &mut findings, &mut evidence_refs);
            }
        }

        for skill_requirement in &requirement.required_skills {
            if skill_requirement.skill_ref.trim().is_empty() || skill_requirement.minimum_people == 0 {
                add(&mut findings, VerificationFindingCode::SkillRequirementInvalid, &skill_requirement.skill_ref);
                continue;
            }
            let matches: Vec<_> = cell.skill_qualifications.iter()
                .filter(|skill| skill.skill_ref == skill_requirement.skill_ref)
                .collect();
            let skill = match matches.as_slice() {
                [] => {
                    add(&mut findings, VerificationFindingCode::SkillMissing, &skill_requirement.skill_ref);
                    None
                }
                [only] => Some(*only),
                _ => {
                    add(&mut findings, VerificationFindingCode::SkillAmbiguous, &skill_requirement.skill_ref);
                    None
                }
            };
            if let Some(skill) = skill {
                if skill.availability != Availability::Available {
                    add(&mut findings, VerificationFindingCode::SkillNotAvailable, &skill_requirement.skill_ref);
                }
                match skill.available_qualified_people {
                    Some(people) if people >= skill_requirement.minimum_people => {}
                    Some(_) => add(&mut findings, VerificationFindingCode::SkillHeadcountInsufficient, &skill_requirement.skill_ref),
                    None => add(&mut findings, VerificationFindingCode::SkillHeadcountUnknown, &skill_requirement.skill_ref),
                }
                verify_evidence(&skill.qualification_evidence, at, &mut findings, &mut evidence_refs);
            }
        }

        for lot_ref in &requirement.required_material_lot_refs {
            let matches: Vec<_> = cell.material_lots.iter().filter(|lot| lot.lot_ref == *lot_ref).collect();
            let lot = match matches.as_slice() {
                [] => {
                    add(&mut findings, VerificationFindingCode::MaterialLotMissing, lot_ref);
                    None
                }
                [only] => Some(*only),
                _ => {
                    add(&mut findings, VerificationFindingCode::MaterialLotAmbiguous, lot_ref);
                    None
                }
            };
            if let Some(lot) = lot {
                if lot.status != MaterialLotStatus::Accepted {
                    add(&mut findings, VerificationFindingCode::MaterialLotNotAccepted, lot_ref);
                }
                if requirement.required_material_envelope_ref.as_ref()
                    .is_some_and(|expected| expected != &lot.material_envelope_ref)
                {
                    add(&mut findings, VerificationFindingCode::MaterialEnvelopeMismatch, lot_ref);
                }
                verify_evidence(&lot.admission_evidence, at, &mut findings, &mut evidence_refs);
            }
        }

        for demand in &requirement.utility_requirements {
            let matches: Vec<_> = cell.utility_capacities.iter()
                .filter(|capacity| capacity.service_ref == demand.service_ref)
                .collect();
            let capacity = match matches.as_slice() {
                [] => {
                    add(&mut findings, VerificationFindingCode::UtilityMissing, &demand.service_ref);
                    None
                }
                [only] => Some(*only),
                _ => {
                    add(&mut findings, VerificationFindingCode::UtilityAmbiguous, &demand.service_ref);
                    None
                }
            };
            if let Some(capacity) = capacity {
                if capacity.availability != Availability::Available {
                    add(&mut findings, VerificationFindingCode::UtilityNotAvailable, &capacity.service_ref);
                }
                if capacity.unit_ref != demand.unit_ref {
                    add(&mut findings, VerificationFindingCode::UtilityUnitMismatch, &capacity.service_ref);
                }
                if capacity.capacity_units < demand.required_capacity_units {
                    add(&mut findings, VerificationFindingCode::UtilityCapacityInsufficient, &capacity.service_ref);
                }
                verify_evidence(&capacity.evidence, at, &mut findings, &mut evidence_refs);
            }
        }

        for clearance_ref in &requirement.required_clearance_refs {
            let matches: Vec<_> = cell.clearances.iter()
                .filter(|clearance| clearance.clearance_ref == *clearance_ref)
                .collect();
            let clearance = match matches.as_slice() {
                [] => {
                    add(&mut findings, VerificationFindingCode::ClearanceMissing, clearance_ref);
                    None
                }
                [only] => Some(*only),
                _ => {
                    add(&mut findings, VerificationFindingCode::ClearanceAmbiguous, clearance_ref);
                    None
                }
            };
            if let Some(clearance) = clearance {
                verify_evidence(&clearance.evidence, at, &mut findings, &mut evidence_refs);
            }
        }
    }

    let independently_resolved_evidence_refs: Vec<String> = evidence_refs.into_iter().collect();
    if claim.evidence_refs != independently_resolved_evidence_refs {
        add(&mut findings, VerificationFindingCode::EvidenceSetMismatch, &cell.profile_ref);
    }

    findings.sort();
    findings.dedup();
    let status = if findings.is_empty() {
        VerificationStatus::Verified
    } else {
        VerificationStatus::Rejected
    };

    EligibleClaimVerification {
        operation_ref: requirement.operation_ref.clone(),
        profile_ref: cell.profile_ref.clone(),
        verified_at_unix_seconds: at,
        status,
        findings,
        independently_resolved_evidence_refs,
    }
}

fn verify_evidence(
    evidence: &EvidenceRecord,
    at: i64,
    findings: &mut Vec<VerificationFinding>,
    evidence_refs: &mut BTreeSet<String>,
) {
    if evidence.evidence_ref.trim().is_empty() {
        add(findings, VerificationFindingCode::EvidenceReferenceMissing, "evidence_ref");
        return;
    }
    evidence_refs.insert(evidence.evidence_ref.clone());
    if evidence.status != EvidenceStatus::Current
        || evidence.issuer_ref.as_deref().unwrap_or("").trim().is_empty()
    {
        add(findings, VerificationFindingCode::EvidenceInvalid, &evidence.evidence_ref);
        return;
    }
    match (evidence.valid_from_unix_seconds, evidence.valid_until_unix_seconds) {
        (Some(from), Some(until)) if from <= at && at < until => {}
        _ => add(findings, VerificationFindingCode::EvidenceInvalid, &evidence.evidence_ref),
    }
}

fn all_present(available: &[String], required: &[String]) -> bool {
    required.iter().all(|item| !item.trim().is_empty() && available.iter().any(|value| value == item))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn add(
    findings: &mut Vec<VerificationFinding>,
    code: VerificationFindingCode,
    subject_ref: &str,
) {
    findings.push(VerificationFinding {
        code,
        subject_ref: subject_ref.to_owned(),
    });
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed workcell capability eligibility.
//!
//! This is a read-only eligibility evaluator, not a scheduler, registry, JSON
//! parser, machine controller, safety authority, or source-document verifier.
//! Callers resolve canonical owner records into immutable facts. A URI or digest
//! identifies evidence; it does not establish that the underlying assertion is true.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProfileStatus { Draft, Candidate, Qualified, Suspended, Expired, Revoked }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapshotResolution { Complete, Partial, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceStatus { Current, Expired, Revoked, Unknown, Missing }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssetStatus { Available, Maintenance, Quarantined, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Availability { Available, Unavailable, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaterialLotStatus { Accepted, Quarantined, Rejected, Unknown }

/// Unix-second validity window; valid_until is exclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub evidence_ref: String,
    pub status: EvidenceStatus,
    pub valid_from_unix_seconds: Option<i64>,
    pub valid_until_unix_seconds: Option<i64>,
    pub issuer_ref: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetSnapshot {
    pub asset_ref: String,
    pub status: AssetStatus,
    pub capability_refs: Vec<String>,
    pub tooling_refs: Vec<String>,
    pub calibration_evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessConfigurationSnapshot {
    pub configuration_ref: String,
    pub process_revision_ref: String,
    pub process_capability_ref: String,
    pub product_revision_ref: String,
    pub pattern_revision_ref: String,
    pub operation_refs: Vec<String>,
    pub asset_refs: Vec<String>,
    pub tooling_refs: Vec<String>,
    pub inspection_plan_ref: String,
    pub inspection_plan_evidence: EvidenceRecord,
    pub qualification_evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillQualificationSnapshot {
    pub skill_ref: String,
    pub availability: Availability,
    pub qualification_evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialLotSnapshot {
    pub lot_ref: String,
    pub status: MaterialLotStatus,
    pub material_envelope_ref: String,
    pub admission_evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityCapacitySnapshot {
    pub service_ref: String,
    pub unit_ref: String,
    pub availability: Availability,
    pub capacity_units: u64,
    pub evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClearanceSnapshot {
    pub clearance_ref: String,
    pub evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellProfileSnapshot {
    pub profile_ref: String,
    pub facility_ref: String,
    pub status: ProfileStatus,
    pub valid_from_unix_seconds: Option<i64>,
    pub valid_until_unix_seconds: Option<i64>,
    pub resolution: SnapshotResolution,
    pub qualification_evidence: Vec<EvidenceRecord>,
    pub independent_verification_evidence: Vec<EvidenceRecord>,
    pub assets: Vec<AssetSnapshot>,
    pub process_configurations: Vec<ProcessConfigurationSnapshot>,
    pub skill_qualifications: Vec<SkillQualificationSnapshot>,
    pub material_lots: Vec<MaterialLotSnapshot>,
    pub utility_capacities: Vec<UtilityCapacitySnapshot>,
    pub clearances: Vec<ClearanceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityRequirement {
    pub service_ref: String,
    pub unit_ref: String,
    pub required_capacity_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationRequirement {
    pub operation_ref: String,
    pub facility_ref: String,
    pub product_revision_ref: String,
    pub pattern_revision_ref: String,
    pub process_revision_ref: String,
    pub process_capability_ref: String,
    pub process_configuration_ref: String,
    pub required_tooling_refs: Vec<String>,
    pub required_skill_refs: Vec<String>,
    pub required_material_lot_refs: Vec<String>,
    pub required_material_envelope_ref: Option<String>,
    pub utility_requirements: Vec<UtilityRequirement>,
    pub required_clearance_refs: Vec<String>,
    pub inspection_plan_ref: String,
    pub scheduled_at_unix_seconds: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EligibilityReasonCode {
    EmptyRequiredReference,
    DuplicateResolvedReference,
    SnapshotIncomplete,
    FacilityMismatch,
    ProfileNotQualified,
    ProfileSuspended,
    ProfileExpired,
    ProfileRevoked,
    ProfileNotYetValid,
    ProfileValidityUnknown,
    ProfileEvidenceMissing,
    IndependentVerificationMissing,
    ProcessCapabilityMismatch,
    EvidenceMissing,
    EvidenceUnknown,
    EvidenceExpired,
    EvidenceRevoked,
    EvidenceNotYetValid,
    EvidenceValidityUnknown,
    ConfigurationMissing,
    ConfigurationOperationMismatch,
    ProductRevisionMismatch,
    PatternRevisionMismatch,
    ProcessRevisionMismatch,
    InspectionPlanMismatch,
    AssetMissing,
    AssetUnavailable,
    AssetCapabilityMismatch,
    AssetToolingMismatch,
    SkillMissing,
    SkillUnavailable,
    MaterialLotMissing,
    MaterialLotQuarantined,
    MaterialLotRejected,
    MaterialEnvelopeMismatch,
    UtilityMissing,
    UtilityUnavailable,
    UtilityCapacityInsufficient,
    UtilityUnitMismatch,
    ClearanceEvidenceMissing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EligibilitySeverity { Indeterminate, Blocked }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EligibilityReason {
    pub code: EligibilityReasonCode,
    pub severity: EligibilitySeverity,
    pub subject_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EligibilityStatus { Eligible, Blocked, Indeterminate }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibilityReport {
    pub operation_ref: String,
    pub profile_ref: String,
    pub facility_ref: String,
    pub evaluated_at_unix_seconds: i64,
    pub status: EligibilityStatus,
    pub reasons: Vec<EligibilityReason>,
    pub evidence_refs: Vec<String>,
}

/// Evaluate one immutable operation against one resolved cell snapshot.
///
/// Missing facts in a Complete snapshot are known absences (Blocked). Missing
/// facts in Partial/Unknown snapshots are Indeterminate. Any known hard violation
/// makes the overall result Blocked even if other facts remain unknown.
pub fn evaluate_cell_eligibility(
    requirement: &OperationRequirement,
    cell: &CellProfileSnapshot,
) -> EligibilityReport {
    let mut out = Collector::default();
    let at = requirement.scheduled_at_unix_seconds;

    for (name, value) in [
        ("operation_ref", requirement.operation_ref.as_str()),
        ("facility_ref", requirement.facility_ref.as_str()),
        ("product_revision_ref", requirement.product_revision_ref.as_str()),
        ("pattern_revision_ref", requirement.pattern_revision_ref.as_str()),
        ("process_revision_ref", requirement.process_revision_ref.as_str()),
        ("process_capability_ref", requirement.process_capability_ref.as_str()),
        ("process_configuration_ref", requirement.process_configuration_ref.as_str()),
        ("inspection_plan_ref", requirement.inspection_plan_ref.as_str()),
        ("profile_ref", cell.profile_ref.as_str()),
        ("cell_facility_ref", cell.facility_ref.as_str()),
    ] {
        if value.trim().is_empty() {
            out.add(EligibilityReasonCode::EmptyRequiredReference, EligibilitySeverity::Indeterminate, name);
        }
    }

    for (kind, references) in [
        ("required_tooling_ref", requirement.required_tooling_refs.as_slice()),
        ("required_skill_ref", requirement.required_skill_refs.as_slice()),
        ("required_material_lot_ref", requirement.required_material_lot_refs.as_slice()),
        ("required_clearance_ref", requirement.required_clearance_refs.as_slice()),
    ] {
        for reference in references {
            if reference.trim().is_empty() {
                out.add(EligibilityReasonCode::EmptyRequiredReference, EligibilitySeverity::Indeterminate, kind);
            }
        }
    }
    if requirement.required_material_envelope_ref.as_deref().is_some_and(|value| value.trim().is_empty()) {
        out.add(EligibilityReasonCode::EmptyRequiredReference, EligibilitySeverity::Indeterminate, "required_material_envelope_ref");
    }
    for demand in &requirement.utility_requirements {
        if demand.service_ref.trim().is_empty() {
            out.add(EligibilityReasonCode::EmptyRequiredReference, EligibilitySeverity::Indeterminate, "utility_service_ref");
        }
        if demand.unit_ref.trim().is_empty() {
            out.add(EligibilityReasonCode::EmptyRequiredReference, EligibilitySeverity::Indeterminate, "utility_unit_ref");
        }
    }

    if cell.resolution != SnapshotResolution::Complete {
        out.add(EligibilityReasonCode::SnapshotIncomplete, EligibilitySeverity::Indeterminate, &cell.profile_ref);
    }
    if requirement.facility_ref != cell.facility_ref {
        out.add(EligibilityReasonCode::FacilityMismatch, EligibilitySeverity::Blocked, &cell.facility_ref);
    }

    match cell.status {
        ProfileStatus::Qualified => {}
        ProfileStatus::Suspended => out.add(EligibilityReasonCode::ProfileSuspended, EligibilitySeverity::Blocked, &cell.profile_ref),
        ProfileStatus::Expired => out.add(EligibilityReasonCode::ProfileExpired, EligibilitySeverity::Blocked, &cell.profile_ref),
        ProfileStatus::Revoked => out.add(EligibilityReasonCode::ProfileRevoked, EligibilitySeverity::Blocked, &cell.profile_ref),
        ProfileStatus::Draft | ProfileStatus::Candidate => out.add(EligibilityReasonCode::ProfileNotQualified, EligibilitySeverity::Blocked, &cell.profile_ref),
    }

    match (cell.valid_from_unix_seconds, cell.valid_until_unix_seconds) {
        (Some(from), Some(until)) if from <= at && at < until => {}
        (Some(from), Some(_)) if at < from => out.add(EligibilityReasonCode::ProfileNotYetValid, EligibilitySeverity::Blocked, &cell.profile_ref),
        (Some(_), Some(until)) if at >= until => out.add(EligibilityReasonCode::ProfileExpired, EligibilitySeverity::Blocked, &cell.profile_ref),
        _ => out.add(EligibilityReasonCode::ProfileValidityUnknown, EligibilitySeverity::Indeterminate, &cell.profile_ref),
    }

    if cell.qualification_evidence.is_empty() {
        out.add(EligibilityReasonCode::ProfileEvidenceMissing, missing_severity(cell.resolution), &cell.profile_ref);
    } else {
        for evidence in &cell.qualification_evidence { out.check_evidence(evidence, at); }
    }
    if cell.independent_verification_evidence.is_empty() {
        out.add(EligibilityReasonCode::IndependentVerificationMissing, missing_severity(cell.resolution), &cell.profile_ref);
    } else {
        for evidence in &cell.independent_verification_evidence { out.check_evidence(evidence, at); }
    }

    let config = match unique_match(&cell.process_configurations, |candidate| {
        candidate.configuration_ref == requirement.process_configuration_ref
    }) {
        Ok(Some(config)) => config,
        Ok(None) => {
            out.add(EligibilityReasonCode::ConfigurationMissing, missing_severity(cell.resolution), &requirement.process_configuration_ref);
            return out.finish(requirement, cell, at);
        }
        Err(()) => {
            out.add(EligibilityReasonCode::DuplicateResolvedReference, EligibilitySeverity::Blocked, &requirement.process_configuration_ref);
            return out.finish(requirement, cell, at);
        }
    };

    if !config.operation_refs.contains(&requirement.operation_ref) {
        out.add(EligibilityReasonCode::ConfigurationOperationMismatch, EligibilitySeverity::Blocked, &requirement.operation_ref);
    }
    if config.product_revision_ref != requirement.product_revision_ref {
        out.add(EligibilityReasonCode::ProductRevisionMismatch, EligibilitySeverity::Blocked, &config.configuration_ref);
    }
    if config.pattern_revision_ref != requirement.pattern_revision_ref {
        out.add(EligibilityReasonCode::PatternRevisionMismatch, EligibilitySeverity::Blocked, &config.configuration_ref);
    }
    if config.process_revision_ref != requirement.process_revision_ref {
        out.add(EligibilityReasonCode::ProcessRevisionMismatch, EligibilitySeverity::Blocked, &config.configuration_ref);
    }
    if config.process_capability_ref != requirement.process_capability_ref {
        out.add(EligibilityReasonCode::ProcessCapabilityMismatch, EligibilitySeverity::Blocked, &config.configuration_ref);
    }
    if config.inspection_plan_ref != requirement.inspection_plan_ref {
        out.add(EligibilityReasonCode::InspectionPlanMismatch, EligibilitySeverity::Blocked, &config.configuration_ref);
    }
    out.check_evidence(&config.qualification_evidence, at);
    out.check_evidence(&config.inspection_plan_evidence, at);

    if config.asset_refs.is_empty() {
        out.add(EligibilityReasonCode::AssetMissing, EligibilitySeverity::Indeterminate, &config.configuration_ref);
    }
    for asset_ref in &config.asset_refs {
        if asset_ref.trim().is_empty() {
            out.add(EligibilityReasonCode::EmptyRequiredReference, EligibilitySeverity::Indeterminate, &config.configuration_ref);
            continue;
        }
        let asset = match unique_match(&cell.assets, |candidate| candidate.asset_ref == *asset_ref) {
            Ok(Some(asset)) => asset,
            Ok(None) => {
                out.add(EligibilityReasonCode::AssetMissing, missing_severity(cell.resolution), asset_ref);
                continue;
            }
            Err(()) => {
                out.add(EligibilityReasonCode::DuplicateResolvedReference, EligibilitySeverity::Blocked, asset_ref);
                continue;
            }
        };
        match asset.status {
            AssetStatus::Available => {}
            AssetStatus::Maintenance | AssetStatus::Quarantined =>
                out.add(EligibilityReasonCode::AssetUnavailable, EligibilitySeverity::Blocked, &asset.asset_ref),
            AssetStatus::Unknown =>
                out.add(EligibilityReasonCode::AssetUnavailable, EligibilitySeverity::Indeterminate, &asset.asset_ref),
        }
        if !asset.capability_refs.contains(&requirement.process_capability_ref) {
            out.add(EligibilityReasonCode::AssetCapabilityMismatch, EligibilitySeverity::Blocked, &asset.asset_ref);
        }
        if !contains_all(&asset.tooling_refs, &requirement.required_tooling_refs)
            || !contains_all(&config.tooling_refs, &requirement.required_tooling_refs) {
            out.add(EligibilityReasonCode::AssetToolingMismatch, EligibilitySeverity::Blocked, &asset.asset_ref);
        }
        out.check_evidence(&asset.calibration_evidence, at);
    }

    for skill_ref in &requirement.required_skill_refs {
        let skill = match unique_match(&cell.skill_qualifications, |candidate| candidate.skill_ref == *skill_ref) {
            Ok(Some(skill)) => skill,
            Ok(None) => {
                out.add(EligibilityReasonCode::SkillMissing, missing_severity(cell.resolution), skill_ref);
                continue;
            }
            Err(()) => {
                out.add(EligibilityReasonCode::DuplicateResolvedReference, EligibilitySeverity::Blocked, skill_ref);
                continue;
            }
        };
        match skill.availability {
            Availability::Available => {}
            Availability::Unavailable => out.add(EligibilityReasonCode::SkillUnavailable, EligibilitySeverity::Blocked, skill_ref),
            Availability::Unknown => out.add(EligibilityReasonCode::SkillUnavailable, EligibilitySeverity::Indeterminate, skill_ref),
        }
        out.check_evidence(&skill.qualification_evidence, at);
    }

    for lot_ref in &requirement.required_material_lot_refs {
        let lot = match unique_match(&cell.material_lots, |candidate| candidate.lot_ref == *lot_ref) {
            Ok(Some(lot)) => lot,
            Ok(None) => {
                out.add(EligibilityReasonCode::MaterialLotMissing, missing_severity(cell.resolution), lot_ref);
                continue;
            }
            Err(()) => {
                out.add(EligibilityReasonCode::DuplicateResolvedReference, EligibilitySeverity::Blocked, lot_ref);
                continue;
            }
        };
        match lot.status {
            MaterialLotStatus::Accepted => {}
            MaterialLotStatus::Quarantined => out.add(EligibilityReasonCode::MaterialLotQuarantined, EligibilitySeverity::Blocked, lot_ref),
            MaterialLotStatus::Rejected => out.add(EligibilityReasonCode::MaterialLotRejected, EligibilitySeverity::Blocked, lot_ref),
            MaterialLotStatus::Unknown => out.add(EligibilityReasonCode::MaterialLotMissing, EligibilitySeverity::Indeterminate, lot_ref),
        }
        if let Some(expected) = &requirement.required_material_envelope_ref {
            if &lot.material_envelope_ref != expected {
                out.add(EligibilityReasonCode::MaterialEnvelopeMismatch, EligibilitySeverity::Blocked, lot_ref);
            }
        }
        out.check_evidence(&lot.admission_evidence, at);
    }

    for demand in &requirement.utility_requirements {
        let capacity = match unique_match(&cell.utility_capacities, |candidate| candidate.service_ref == demand.service_ref) {
            Ok(Some(capacity)) => capacity,
            Ok(None) => {
                out.add(EligibilityReasonCode::UtilityMissing, missing_severity(cell.resolution), &demand.service_ref);
                continue;
            }
            Err(()) => {
                out.add(EligibilityReasonCode::DuplicateResolvedReference, EligibilitySeverity::Blocked, &demand.service_ref);
                continue;
            }
        };
        if capacity.unit_ref != demand.unit_ref {
            out.add(EligibilityReasonCode::UtilityUnitMismatch, EligibilitySeverity::Blocked, &capacity.service_ref);
        }
        match capacity.availability {
            Availability::Available => {}
            Availability::Unavailable => out.add(EligibilityReasonCode::UtilityUnavailable, EligibilitySeverity::Blocked, &capacity.service_ref),
            Availability::Unknown => out.add(EligibilityReasonCode::UtilityUnavailable, EligibilitySeverity::Indeterminate, &capacity.service_ref),
        }
        if capacity.capacity_units < demand.required_capacity_units {
            let severity = if capacity.availability == Availability::Unknown {
                EligibilitySeverity::Indeterminate
            } else { EligibilitySeverity::Blocked };
            out.add(EligibilityReasonCode::UtilityCapacityInsufficient, severity, &capacity.service_ref);
        }
        out.check_evidence(&capacity.evidence, at);
    }

    for clearance_ref in &requirement.required_clearance_refs {
        let clearance = match unique_match(&cell.clearances, |candidate| candidate.clearance_ref == *clearance_ref) {
            Ok(Some(clearance)) => clearance,
            Ok(None) => {
                out.add(EligibilityReasonCode::ClearanceEvidenceMissing, missing_severity(cell.resolution), clearance_ref);
                continue;
            }
            Err(()) => {
                out.add(EligibilityReasonCode::DuplicateResolvedReference, EligibilitySeverity::Blocked, clearance_ref);
                continue;
            }
        };
        out.check_evidence(&clearance.evidence, at);
    }

    out.finish(requirement, cell, at)
}

fn unique_match<'a, T, F>(items: &'a [T], predicate: F) -> Result<Option<&'a T>, ()>
where
    F: Fn(&T) -> bool,
{
    let mut matches = items.iter().filter(|item| predicate(item));
    let first = matches.next();
    if matches.next().is_some() {
        Err(())
    } else {
        Ok(first)
    }
}

fn contains_all(available: &[String], required: &[String]) -> bool {
    required.iter().all(|item| available.contains(item))
}

fn missing_severity(resolution: SnapshotResolution) -> EligibilitySeverity {
    match resolution {
        SnapshotResolution::Complete => EligibilitySeverity::Blocked,
        SnapshotResolution::Partial | SnapshotResolution::Unknown => EligibilitySeverity::Indeterminate,
    }
}

#[derive(Default)]
struct Collector {
    reasons: Vec<EligibilityReason>,
    evidence_refs: BTreeSet<String>,
}

impl Collector {
    fn add(&mut self, code: EligibilityReasonCode, severity: EligibilitySeverity, subject_ref: &str) {
        self.reasons.push(EligibilityReason { code, severity, subject_ref: subject_ref.to_owned() });
    }

    fn check_evidence(&mut self, evidence: &EvidenceRecord, at: i64) {
        if !evidence.evidence_ref.trim().is_empty() {
            self.evidence_refs.insert(evidence.evidence_ref.clone());
        }
        if evidence.evidence_ref.trim().is_empty() {
            self.add(EligibilityReasonCode::EvidenceMissing, EligibilitySeverity::Indeterminate, "evidence_ref");
            return;
        }
        match evidence.status {
            EvidenceStatus::Current => {}
            EvidenceStatus::Expired => { self.add(EligibilityReasonCode::EvidenceExpired, EligibilitySeverity::Blocked, &evidence.evidence_ref); return; }
            EvidenceStatus::Revoked => { self.add(EligibilityReasonCode::EvidenceRevoked, EligibilitySeverity::Blocked, &evidence.evidence_ref); return; }
            EvidenceStatus::Unknown => { self.add(EligibilityReasonCode::EvidenceUnknown, EligibilitySeverity::Indeterminate, &evidence.evidence_ref); return; }
            EvidenceStatus::Missing => { self.add(EligibilityReasonCode::EvidenceMissing, EligibilitySeverity::Indeterminate, &evidence.evidence_ref); return; }
        }
        match (evidence.valid_from_unix_seconds, evidence.valid_until_unix_seconds) {
            (Some(from), Some(until)) if from <= at && at < until => {}
            (Some(from), Some(_)) if at < from => self.add(EligibilityReasonCode::EvidenceNotYetValid, EligibilitySeverity::Blocked, &evidence.evidence_ref),
            (Some(_), Some(until)) if at >= until => self.add(EligibilityReasonCode::EvidenceExpired, EligibilitySeverity::Blocked, &evidence.evidence_ref),
            _ => self.add(EligibilityReasonCode::EvidenceValidityUnknown, EligibilitySeverity::Indeterminate, &evidence.evidence_ref),
        }
        if evidence.issuer_ref.as_deref().unwrap_or("").trim().is_empty() {
            self.add(EligibilityReasonCode::EvidenceUnknown, EligibilitySeverity::Indeterminate, &evidence.evidence_ref);
        }
    }

    fn finish(mut self, req: &OperationRequirement, cell: &CellProfileSnapshot, at: i64) -> EligibilityReport {
        self.reasons.sort();
        self.reasons.dedup();
        let status = if self.reasons.iter().any(|r| r.severity == EligibilitySeverity::Blocked) {
            EligibilityStatus::Blocked
        } else if !self.reasons.is_empty() {
            EligibilityStatus::Indeterminate
        } else {
            EligibilityStatus::Eligible
        };
        EligibilityReport {
            operation_ref: req.operation_ref.clone(),
            profile_ref: cell.profile_ref.clone(),
            facility_ref: cell.facility_ref.clone(),
            evaluated_at_unix_seconds: at,
            status,
            reasons: self.reasons,
            evidence_refs: self.evidence_refs.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: i64 = 1_800_000_000;

    fn evidence(reference: &str) -> EvidenceRecord {
        EvidenceRecord {
            evidence_ref: reference.to_owned(),
            status: EvidenceStatus::Current,
            valid_from_unix_seconds: Some(NOW - 100),
            valid_until_unix_seconds: Some(NOW + 100),
            issuer_ref: Some("urn:example:issuer:independent-review".to_owned()),
            sha256: Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned()),
        }
    }

    fn requirement() -> OperationRequirement {
        OperationRequirement {
            operation_ref: "urn:example:operation:join-shoulder".to_owned(),
            facility_ref: "urn:example:facility:partner-01".to_owned(),
            product_revision_ref: "urn:example:product:ao-01:r3".to_owned(),
            pattern_revision_ref: "urn:example:pattern:ao-01:r3".to_owned(),
            process_revision_ref: "urn:example:process:woven-sew:r2".to_owned(),
            process_capability_ref: "urn:example:capability:lockstitch-woven".to_owned(),
            process_configuration_ref: "urn:example:configuration:ao-01:r3".to_owned(),
            required_tooling_refs: vec!["urn:example:tooling:folder-01".to_owned()],
            required_skill_refs: vec!["urn:example:skill:lockstitch-operator".to_owned()],
            required_material_lot_refs: vec!["urn:example:lot:fabric-001".to_owned()],
            required_material_envelope_ref: Some("urn:example:material-envelope:woven-145gsm".to_owned()),
            utility_requirements: vec![UtilityRequirement {
                service_ref: "urn:example:utility:electricity".to_owned(),
                unit_ref: "urn:example:unit:watt".to_owned(),
                required_capacity_units: 500,
            }],
            required_clearance_refs: vec!["urn:example:clearance:maintenance-access".to_owned()],
            inspection_plan_ref: "urn:example:inspection:seam-check:r1".to_owned(),
            scheduled_at_unix_seconds: NOW,
        }
    }

    fn qualified_cell() -> CellProfileSnapshot {
        CellProfileSnapshot {
            profile_ref: "urn:example:cell-profile:woven-cell-01:r1".to_owned(),
            facility_ref: "urn:example:facility:partner-01".to_owned(),
            status: ProfileStatus::Qualified,
            valid_from_unix_seconds: Some(NOW - 100),
            valid_until_unix_seconds: Some(NOW + 100),
            resolution: SnapshotResolution::Complete,
            qualification_evidence: vec![evidence("urn:example:evidence:profile")],
            independent_verification_evidence: vec![evidence("urn:example:evidence:independent-profile-review")],
            assets: vec![AssetSnapshot {
                asset_ref: "urn:example:asset:sewing-machine-01".to_owned(),
                status: AssetStatus::Available,
                capability_refs: vec!["urn:example:capability:lockstitch-woven".to_owned()],
                tooling_refs: vec!["urn:example:tooling:folder-01".to_owned()],
                calibration_evidence: evidence("urn:example:evidence:calibration"),
            }],
            process_configurations: vec![ProcessConfigurationSnapshot {
                configuration_ref: "urn:example:configuration:ao-01:r3".to_owned(),
                process_revision_ref: "urn:example:process:woven-sew:r2".to_owned(),
                process_capability_ref: "urn:example:capability:lockstitch-woven".to_owned(),
                product_revision_ref: "urn:example:product:ao-01:r3".to_owned(),
                pattern_revision_ref: "urn:example:pattern:ao-01:r3".to_owned(),
                operation_refs: vec!["urn:example:operation:join-shoulder".to_owned()],
                asset_refs: vec!["urn:example:asset:sewing-machine-01".to_owned()],
                tooling_refs: vec!["urn:example:tooling:folder-01".to_owned()],
                inspection_plan_ref: "urn:example:inspection:seam-check:r1".to_owned(),
                inspection_plan_evidence: evidence("urn:example:evidence:inspection-plan"),
                qualification_evidence: evidence("urn:example:evidence:process-qualification"),
            }],
            skill_qualifications: vec![SkillQualificationSnapshot {
                skill_ref: "urn:example:skill:lockstitch-operator".to_owned(),
                availability: Availability::Available,
                qualification_evidence: evidence("urn:example:evidence:skill-qualification"),
            }],
            material_lots: vec![MaterialLotSnapshot {
                lot_ref: "urn:example:lot:fabric-001".to_owned(),
                status: MaterialLotStatus::Accepted,
                material_envelope_ref: "urn:example:material-envelope:woven-145gsm".to_owned(),
                admission_evidence: evidence("urn:example:evidence:lot-acceptance"),
            }],
            utility_capacities: vec![UtilityCapacitySnapshot {
                service_ref: "urn:example:utility:electricity".to_owned(),
                unit_ref: "urn:example:unit:watt".to_owned(),
                availability: Availability::Available,
                capacity_units: 1_000,
                evidence: evidence("urn:example:evidence:utility"),
            }],
            clearances: vec![ClearanceSnapshot {
                clearance_ref: "urn:example:clearance:maintenance-access".to_owned(),
                evidence: evidence("urn:example:evidence:clearance"),
            }],
        }
    }

    fn has_code(report: &EligibilityReport, code: EligibilityReasonCode) -> bool {
        report.reasons.iter().any(|reason| reason.code == code)
    }

    #[test]
    fn complete_current_capability_is_eligible() {
        let report = evaluate_cell_eligibility(&requirement(), &qualified_cell());
        assert_eq!(report.status, EligibilityStatus::Eligible);
        assert!(report.reasons.is_empty());
        assert!(report.evidence_refs.len() >= 6);
    }

    #[test]
    fn candidate_profile_is_blocked() {
        let mut cell = qualified_cell();
        cell.status = ProfileStatus::Candidate;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::ProfileNotQualified));
    }

    #[test]
    fn expired_calibration_is_blocked() {
        let mut cell = qualified_cell();
        cell.assets[0].calibration_evidence.status = EvidenceStatus::Expired;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::EvidenceExpired));
    }

    #[test]
    fn pattern_revision_mismatch_is_blocked() {
        let mut cell = qualified_cell();
        cell.process_configurations[0].pattern_revision_ref = "urn:example:pattern:ao-01:r2".to_owned();
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::PatternRevisionMismatch));
    }

    #[test]
    fn quarantined_material_is_blocked() {
        let mut cell = qualified_cell();
        cell.material_lots[0].status = MaterialLotStatus::Quarantined;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::MaterialLotQuarantined));
    }

    #[test]
    fn unknown_utility_is_indeterminate() {
        let mut cell = qualified_cell();
        cell.utility_capacities[0].availability = Availability::Unknown;
        cell.utility_capacities[0].evidence.status = EvidenceStatus::Unknown;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Indeterminate);
        assert!(has_code(&report, EligibilityReasonCode::UtilityUnavailable));
    }

    #[test]
    fn missing_skill_is_blocked_only_when_snapshot_is_complete() {
        let mut cell = qualified_cell();
        cell.skill_qualifications.clear();
        let complete = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(complete.status, EligibilityStatus::Blocked);
        assert!(has_code(&complete, EligibilityReasonCode::SkillMissing));

        cell.resolution = SnapshotResolution::Partial;
        let partial = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(partial.status, EligibilityStatus::Indeterminate);
        assert!(has_code(&partial, EligibilityReasonCode::SkillMissing));
    }

    #[test]
    fn unknown_asset_cannot_be_eligible() {
        let mut cell = qualified_cell();
        cell.assets[0].status = AssetStatus::Unknown;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Indeterminate);
        assert!(has_code(&report, EligibilityReasonCode::AssetUnavailable));
    }

    #[test]
    fn wrong_facility_is_blocked() {
        let mut req = requirement();
        req.facility_ref = "urn:example:facility:other-site".to_owned();
        let report = evaluate_cell_eligibility(&req, &qualified_cell());
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::FacilityMismatch));
    }

    #[test]
    fn missing_evidence_validity_is_indeterminate() {
        let mut cell = qualified_cell();
        cell.qualification_evidence[0].valid_until_unix_seconds = None;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Indeterminate);
        assert!(has_code(&report, EligibilityReasonCode::EvidenceValidityUnknown));
    }



    #[test]
    fn missing_inspection_plan_evidence_is_indeterminate() {
        let mut cell = qualified_cell();
        cell.process_configurations[0].inspection_plan_evidence.status = EvidenceStatus::Missing;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Indeterminate);
        assert!(has_code(&report, EligibilityReasonCode::EvidenceMissing));
    }

    #[test]
    fn utility_unit_mismatch_is_blocked() {
        let mut cell = qualified_cell();
        cell.utility_capacities[0].unit_ref = "urn:example:unit:kilowatt".to_owned();
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::UtilityUnitMismatch));
    }

    #[test]
    fn revoked_process_qualification_is_blocked() {
        let mut cell = qualified_cell();
        cell.process_configurations[0].qualification_evidence.status = EvidenceStatus::Revoked;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::EvidenceRevoked));
    }

    #[test]
    fn known_insufficient_utility_capacity_is_blocked() {
        let mut cell = qualified_cell();
        cell.utility_capacities[0].capacity_units = 100;
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::UtilityCapacityInsufficient));
    }

    #[test]
    fn unsupported_process_capability_is_blocked() {
        let mut cell = qualified_cell();
        cell.assets[0].capability_refs.clear();
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::AssetCapabilityMismatch));
    }



    #[test]
    fn empty_required_reference_cannot_match_empty_owner_reference() {
        let mut req = requirement();
        let mut cell = qualified_cell();
        req.required_skill_refs.push(String::new());
        cell.skill_qualifications.push(SkillQualificationSnapshot {
            skill_ref: String::new(),
            availability: Availability::Available,
            qualification_evidence: evidence("urn:example:evidence:empty-skill"),
        });
        let report = evaluate_cell_eligibility(&req, &cell);
        assert_ne!(report.status, EligibilityStatus::Eligible);
        assert!(has_code(&report, EligibilityReasonCode::EmptyRequiredReference));
    }

    #[test]
    fn missing_independent_profile_verification_is_blocked_in_complete_snapshot() {
        let mut cell = qualified_cell();
        cell.independent_verification_evidence.clear();
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::IndependentVerificationMissing));
    }

    #[test]
    fn process_configuration_capability_mismatch_is_blocked() {
        let mut cell = qualified_cell();
        cell.process_configurations[0].process_capability_ref =
            "urn:example:capability:wrong-process".to_owned();
        let report = evaluate_cell_eligibility(&requirement(), &cell);
        assert_eq!(report.status, EligibilityStatus::Blocked);
        assert!(has_code(&report, EligibilityReasonCode::ProcessCapabilityMismatch));
    }

    #[test]
    fn duplicate_configuration_reference_is_blocked_and_order_independent() {
        let req = requirement();
        let mut cell = qualified_cell();
        let mut duplicate = cell.process_configurations[0].clone();
        duplicate.pattern_revision_ref = "urn:example:pattern:conflicting-r9".to_owned();
        cell.process_configurations.push(duplicate);
        let first = evaluate_cell_eligibility(&req, &cell);
        assert_eq!(first.status, EligibilityStatus::Blocked);
        assert!(has_code(&first, EligibilityReasonCode::DuplicateResolvedReference));

        cell.process_configurations.reverse();
        let second = evaluate_cell_eligibility(&req, &cell);
        assert_eq!(first, second);
    }

    #[test]
    fn result_is_order_independent_and_preserves_evidence_refs() {
        let req = requirement();
        let cell = qualified_cell();
        let first = evaluate_cell_eligibility(&req, &cell);
        let mut reversed = cell;
        reversed.assets.reverse();
        reversed.qualification_evidence.reverse();
        reversed.skill_qualifications.reverse();
        reversed.material_lots.reverse();
        reversed.utility_capacities.reverse();
        reversed.clearances.reverse();
        let second = evaluate_cell_eligibility(&req, &reversed);
        assert_eq!(first, second);
    }
}

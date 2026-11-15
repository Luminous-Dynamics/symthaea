// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Soft-first modular manufacturing reconfiguration.
//!
//! Validates a declared target configuration and derives a deterministic
//! transition from a current configuration. Allocation-only changes are
//! distinguished from physical module, connection and layout changes.
//!
//! A successful result means only "structurally feasible for qualification".
//! Physical changes create independent requalification obligations. This
//! module is not a scheduler, machine controller, safety authority or source
//! document verifier and can never authorize production.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapshotResolution { Complete, Partial, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceStatus { Current, Expired, Revoked, Unknown, Missing }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub evidence_ref: String,
    pub status: EvidenceStatus,
    pub valid_from_unix_seconds: Option<i64>,
    pub valid_until_unix_seconds: Option<i64>,
    pub issuer_ref: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModuleStatus { Candidate, Qualified, Suspended, Expired, Revoked }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PortDirection { Input, Output, Bidirectional }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePort {
    pub port_ref: String,
    pub interface_type_ref: String,
    pub protocol_ref: String,
    pub medium_ref: String,
    pub unit_ref: String,
    pub direction: PortDirection,
    pub max_connections: u32,
    pub max_capacity_units: u64,
    pub required_peer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleCapability {
    pub capability_ref: String,
    pub process_revision_ref: String,
    pub parameter_digest_sha256: String,
    pub inspection_plan_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleProfile {
    pub module_ref: String,
    pub facility_ref: String,
    pub status: ModuleStatus,
    pub valid_from_unix_seconds: Option<i64>,
    pub valid_until_unix_seconds: Option<i64>,
    pub footprint_width_mm: u64,
    pub footprint_depth_mm: u64,
    pub capabilities: Vec<ModuleCapability>,
    pub ports: Vec<ModulePort>,
    pub qualification_evidence: EvidenceRecord,
    pub independent_verification_evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePlacement {
    pub module_ref: String,
    pub x_mm: u64,
    pub y_mm: u64,
    pub rotation_degrees: u16,
    pub coordinate_frame_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRect {
    pub rect_ref: String,
    pub x_mm: u64,
    pub y_mm: u64,
    pub width_mm: u64,
    pub depth_mm: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortConnection {
    pub connection_ref: String,
    pub from_module_ref: String,
    pub from_port_ref: String,
    pub to_module_ref: String,
    pub to_port_ref: String,
    pub capacity_units: u64,
    pub connection_evidence: EvidenceRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationDemand {
    pub operation_ref: String,
    pub capability_ref: String,
    pub process_revision_ref: String,
    pub parameter_digest_sha256: String,
    pub inspection_plan_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationAssignment {
    pub operation_ref: String,
    pub module_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GeometryBasis { Synthetic, Imported, Measured }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellConfiguration {
    pub configuration_ref: String,
    pub facility_ref: String,
    pub layout_revision_ref: String,
    pub coordinate_frame_ref: String,
    pub width_mm: u64,
    pub depth_mm: u64,
    pub geometry_basis: GeometryBasis,
    pub resolution: SnapshotResolution,
    pub placements: Vec<ModulePlacement>,
    pub keepouts: Vec<LayoutRect>,
    pub connections: Vec<PortConnection>,
    pub assignments: Vec<OperationAssignment>,
    /// Supports the declared geometry; it is not an egress or safety approval.
    pub layout_evidence: EvidenceRecord,
    /// Required when the physical installation has not changed.
    pub commissioning_evidence: Option<EvidenceRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransitionKind { NoChange, AllocationOnly, PhysicalReconfiguration }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanStatus { FeasibleForQualification, Blocked, Indeterminate }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransitionAction {
    Disconnect { connection_ref: String },
    RemoveModule { module_ref: String },
    RelocateModule {
        module_ref: String,
        from_x_mm: u64,
        from_y_mm: u64,
        to_x_mm: u64,
        to_y_mm: u64,
        from_rotation_degrees: u16,
        to_rotation_degrees: u16,
    },
    AddModule { module_ref: String },
    Connect { connection_ref: String },
    ReassignOperation {
        operation_ref: String,
        from_module_ref: Option<String>,
        to_module_ref: String,
    },
    UnassignOperation {
        operation_ref: String,
        from_module_ref: String,
    },
    ReviewLayout,
    CommissionAndIndependentlyRequalify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualificationObligation {
    RevalidateLayoutAndKeepouts,
    InspectPhysicalInterfaces,
    ReverifyUtilitiesAndServiceConnections,
    RequalifyAffectedProcessConfigurations,
    ReconfirmOperatorSkillsAndStaffing,
    ReconfirmInspectionPlan,
    IndependentCommissioningReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReconfigurationReasonCode {
    EmptyRequiredReference,
    DuplicateModuleProfile,
    DuplicateModulePlacement,
    DuplicateOperationDemand,
    DuplicateOperationAssignment,
    DuplicateConnection,
    DuplicatePortReference,
    CurrentConfigurationIncomplete,
    CurrentConfigurationAmbiguous,
    ConfigurationFacilityMismatch,
    ModuleProfileMissing,
    ModuleProfileAmbiguous,
    ModuleWrongFacility,
    ModuleNotQualified,
    ModuleNotYetValid,
    ModuleExpired,
    ModuleValidityUnknown,
    ModuleQualificationEvidenceInvalid,
    ModuleIndependentEvidenceInvalid,
    CapabilityMissing,
    CapabilityAmbiguous,
    CapabilityRevisionMismatch,
    CapabilityDigestInvalid,
    CapabilityDigestMismatch,
    CapabilityInspectionMismatch,
    AssignmentMissing,
    AssignmentUnknownOperation,
    AssignmentModuleMissing,
    PortMissing,
    PortAmbiguous,
    PortInterfaceMismatch,
    PortProtocolMismatch,
    PortMediumMismatch,
    PortUnitMismatch,
    PortDirectionMismatch,
    PortCapacityExceeded,
    PortConnectionCountExceeded,
    RequiredPortUnconnected,
    ConnectionEvidenceInvalid,
    GeometryFrameMismatch,
    UnsupportedRotation,
    ZeroFootprint,
    PlacementOutOfBounds,
    PlacementArithmeticOverflow,
    ModuleFootprintsOverlap,
    KeepoutArithmeticOverflow,
    KeepoutOutOfBounds,
    ModuleIntersectsKeepout,
    LayoutEvidenceInvalid,
    CurrentCommissioningEvidenceInvalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReasonSeverity { Indeterminate, Blocked }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReconfigurationReason {
    pub code: ReconfigurationReasonCode,
    pub severity: ReasonSeverity,
    pub subject_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconfigurationReport {
    pub configuration_ref: String,
    pub facility_ref: String,
    pub evaluated_at_unix_seconds: i64,
    pub transition_kind: TransitionKind,
    pub status: PlanStatus,
    pub actions: Vec<TransitionAction>,
    pub qualification_obligations: Vec<QualificationObligation>,
    pub requires_independent_requalification: bool,
    pub production_authorized: bool,
    pub reasons: Vec<ReconfigurationReason>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconfigurationInput<'a> {
    pub evaluated_at_unix_seconds: i64,
    pub current: Option<&'a CellConfiguration>,
    pub target: &'a CellConfiguration,
    pub module_profiles: &'a [ModuleProfile],
    pub module_pool_resolution: SnapshotResolution,
    pub operation_demands: &'a [OperationDemand],
}

#[derive(Default)]
struct Findings {
    reasons: Vec<ReconfigurationReason>,
    evidence_refs: BTreeSet<String>,
}

impl Findings {
    fn add(&mut self, code: ReconfigurationReasonCode, severity: ReasonSeverity, subject_ref: &str) {
        self.reasons.push(ReconfigurationReason {
            code, severity, subject_ref: subject_ref.to_owned(),
        });
    }

    fn check_evidence(
        &mut self,
        evidence: &EvidenceRecord,
        at: i64,
        code: ReconfigurationReasonCode,
        subject_ref: &str,
    ) -> bool {
        if !evidence.evidence_ref.trim().is_empty() {
            self.evidence_refs.insert(evidence.evidence_ref.clone());
        } else {
            self.add(code, ReasonSeverity::Indeterminate, subject_ref);
            return false;
        }

        match evidence.status {
            EvidenceStatus::Expired | EvidenceStatus::Revoked => {
                self.add(code, ReasonSeverity::Blocked, subject_ref);
                return false;
            }
            EvidenceStatus::Unknown | EvidenceStatus::Missing => {
                self.add(code, ReasonSeverity::Indeterminate, subject_ref);
                return false;
            }
            EvidenceStatus::Current => {}
        }

        if evidence.issuer_ref.as_deref().is_none_or(|issuer| issuer.trim().is_empty()) {
            self.add(code, ReasonSeverity::Indeterminate, subject_ref);
            return false;
        }

        match (evidence.valid_from_unix_seconds, evidence.valid_until_unix_seconds) {
            (Some(from), Some(until)) if from <= at && at < until => true,
            (Some(from), Some(_)) if at < from => {
                self.add(code, ReasonSeverity::Blocked, subject_ref);
                false
            }
            (Some(_), Some(until)) if at >= until => {
                self.add(code, ReasonSeverity::Blocked, subject_ref);
                false
            }
            _ => {
                self.add(code, ReasonSeverity::Indeterminate, subject_ref);
                false
            }
        }
    }

    fn finish(
        mut self,
        target: &CellConfiguration,
        at: i64,
        transition_kind: TransitionKind,
        actions: Vec<TransitionAction>,
        obligations: Vec<QualificationObligation>,
        requires_independent_requalification: bool,
    ) -> ReconfigurationReport {
        self.reasons.sort();
        self.reasons.dedup();
        let status = if self.reasons.iter().any(|reason| reason.severity == ReasonSeverity::Blocked) {
            PlanStatus::Blocked
        } else if !self.reasons.is_empty() {
            PlanStatus::Indeterminate
        } else {
            PlanStatus::FeasibleForQualification
        };
        ReconfigurationReport {
            configuration_ref: target.configuration_ref.clone(),
            facility_ref: target.facility_ref.clone(),
            evaluated_at_unix_seconds: at,
            transition_kind,
            status,
            actions,
            qualification_obligations: obligations,
            requires_independent_requalification,
            production_authorized: false,
            reasons: self.reasons,
            evidence_refs: self.evidence_refs.into_iter().collect(),
        }
    }
}

pub fn evaluate_reconfiguration(input: ReconfigurationInput<'_>) -> ReconfigurationReport {
    let at = input.evaluated_at_unix_seconds;
    let target = input.target;
    let mut findings = Findings::default();

    for (name, value) in [
        ("configuration_ref", target.configuration_ref.as_str()),
        ("facility_ref", target.facility_ref.as_str()),
        ("layout_revision_ref", target.layout_revision_ref.as_str()),
        ("coordinate_frame_ref", target.coordinate_frame_ref.as_str()),
    ] {
        if value.trim().is_empty() {
            findings.add(ReconfigurationReasonCode::EmptyRequiredReference, ReasonSeverity::Indeterminate, name);
        }
    }
    if target.width_mm == 0 || target.depth_mm == 0 {
        findings.add(ReconfigurationReasonCode::PlacementOutOfBounds, ReasonSeverity::Blocked, &target.configuration_ref);
    }
    if target.resolution != SnapshotResolution::Complete {
        findings.add(ReconfigurationReasonCode::ModuleProfileMissing, ReasonSeverity::Indeterminate, &target.configuration_ref);
    }
    if let Some(current) = input.current {
        if current.resolution != SnapshotResolution::Complete {
            findings.add(ReconfigurationReasonCode::CurrentConfigurationIncomplete, ReasonSeverity::Indeterminate, &current.configuration_ref);
        }
        if current.facility_ref != target.facility_ref {
            findings.add(ReconfigurationReasonCode::ConfigurationFacilityMismatch, ReasonSeverity::Blocked, &target.configuration_ref);
        }
        let mut module_ids = BTreeSet::new();
        for placement in &current.placements {
            if !module_ids.insert(placement.module_ref.as_str()) {
                findings.add(ReconfigurationReasonCode::CurrentConfigurationAmbiguous, ReasonSeverity::Blocked, &placement.module_ref);
            }
        }
        let mut connection_ids = BTreeSet::new();
        for connection in &current.connections {
            if !connection_ids.insert(connection.connection_ref.as_str()) {
                findings.add(ReconfigurationReasonCode::CurrentConfigurationAmbiguous, ReasonSeverity::Blocked, &connection.connection_ref);
            }
        }
        let mut operation_ids = BTreeSet::new();
        for assignment in &current.assignments {
            if !operation_ids.insert(assignment.operation_ref.as_str()) {
                findings.add(ReconfigurationReasonCode::CurrentConfigurationAmbiguous, ReasonSeverity::Blocked, &assignment.operation_ref);
            }
        }
    }

    let transition = derive_transition(input.current, target);
    let physical_change = transition.kind == TransitionKind::PhysicalReconfiguration;

    findings.check_evidence(
        &target.layout_evidence,
        at,
        ReconfigurationReasonCode::LayoutEvidenceInvalid,
        &target.configuration_ref,
    );
    if !physical_change {
        match &target.commissioning_evidence {
            Some(evidence) => {
                findings.check_evidence(
                    evidence,
                    at,
                    ReconfigurationReasonCode::CurrentCommissioningEvidenceInvalid,
                    &target.configuration_ref,
                );
            }
            None => findings.add(
                ReconfigurationReasonCode::CurrentCommissioningEvidenceInvalid,
                ReasonSeverity::Indeterminate,
                &target.configuration_ref,
            ),
        }
    }

    let mut profile_index = BTreeMap::<&str, Vec<&ModuleProfile>>::new();
    for profile in input.module_profiles {
        profile_index.entry(profile.module_ref.as_str()).or_default().push(profile);
    }
    for (module_ref, profiles) in &profile_index {
        if profiles.len() > 1 {
            findings.add(ReconfigurationReasonCode::DuplicateModuleProfile, ReasonSeverity::Blocked, module_ref);
        }
    }

    let mut placement_index = BTreeMap::<&str, &ModulePlacement>::new();
    for placement in &target.placements {
        if placement.module_ref.trim().is_empty() {
            findings.add(ReconfigurationReasonCode::EmptyRequiredReference, ReasonSeverity::Indeterminate, "module_ref");
            continue;
        }
        if placement_index.insert(placement.module_ref.as_str(), placement).is_some() {
            findings.add(ReconfigurationReasonCode::DuplicateModulePlacement, ReasonSeverity::Blocked, &placement.module_ref);
        }
    }
    if target.placements.is_empty() {
        findings.add(
            ReconfigurationReasonCode::ModuleProfileMissing,
            missing_severity(input.module_pool_resolution),
            &target.configuration_ref,
        );
    }

    let mut resolved_profiles = BTreeMap::<&str, &ModuleProfile>::new();
    for placement in &target.placements {
        let module_ref = placement.module_ref.as_str();
        let profile = match profile_index.get(module_ref).map(Vec::as_slice) {
            None | Some([]) => {
                findings.add(ReconfigurationReasonCode::ModuleProfileMissing, missing_severity(input.module_pool_resolution), module_ref);
                continue;
            }
            Some([one]) => *one,
            Some(_) => {
                findings.add(ReconfigurationReasonCode::ModuleProfileAmbiguous, ReasonSeverity::Blocked, module_ref);
                continue;
            }
        };
        resolved_profiles.insert(module_ref, profile);

        if profile.facility_ref != target.facility_ref {
            findings.add(ReconfigurationReasonCode::ModuleWrongFacility, ReasonSeverity::Blocked, module_ref);
        }
        if profile.status != ModuleStatus::Qualified {
            findings.add(ReconfigurationReasonCode::ModuleNotQualified, ReasonSeverity::Blocked, module_ref);
        }
        match (profile.valid_from_unix_seconds, profile.valid_until_unix_seconds) {
            (Some(from), Some(until)) if from <= at && at < until => {}
            (Some(from), Some(_)) if at < from => findings.add(ReconfigurationReasonCode::ModuleNotYetValid, ReasonSeverity::Blocked, module_ref),
            (Some(_), Some(until)) if at >= until => findings.add(ReconfigurationReasonCode::ModuleExpired, ReasonSeverity::Blocked, module_ref),
            _ => findings.add(ReconfigurationReasonCode::ModuleValidityUnknown, ReasonSeverity::Indeterminate, module_ref),
        }
        findings.check_evidence(
            &profile.qualification_evidence,
            at,
            ReconfigurationReasonCode::ModuleQualificationEvidenceInvalid,
            module_ref,
        );
        findings.check_evidence(
            &profile.independent_verification_evidence,
            at,
            ReconfigurationReasonCode::ModuleIndependentEvidenceInvalid,
            module_ref,
        );

        if placement.coordinate_frame_ref != target.coordinate_frame_ref {
            findings.add(ReconfigurationReasonCode::GeometryFrameMismatch, ReasonSeverity::Blocked, module_ref);
        }
        if !matches!(placement.rotation_degrees, 0 | 90 | 180 | 270) {
            findings.add(ReconfigurationReasonCode::UnsupportedRotation, ReasonSeverity::Blocked, module_ref);
        }
        if profile.footprint_width_mm == 0 || profile.footprint_depth_mm == 0 {
            findings.add(ReconfigurationReasonCode::ZeroFootprint, ReasonSeverity::Blocked, module_ref);
        }
        let mut port_ids = BTreeSet::new();
        for port in &profile.ports {
            if port.port_ref.trim().is_empty()
                || port.interface_type_ref.trim().is_empty()
                || port.protocol_ref.trim().is_empty()
                || port.medium_ref.trim().is_empty()
                || port.unit_ref.trim().is_empty()
            {
                findings.add(ReconfigurationReasonCode::EmptyRequiredReference, ReasonSeverity::Indeterminate, module_ref);
            }
            if !port_ids.insert(port.port_ref.as_str()) {
                findings.add(ReconfigurationReasonCode::DuplicatePortReference, ReasonSeverity::Blocked, &port.port_ref);
            }
        }
    }

    validate_geometry(target, &placement_index, &resolved_profiles, &mut findings);
    validate_connections(target, at, &resolved_profiles, &mut findings);
    validate_assignments(input, &resolved_profiles, &mut findings);

    let obligations = if physical_change {
        vec![
            QualificationObligation::RevalidateLayoutAndKeepouts,
            QualificationObligation::InspectPhysicalInterfaces,
            QualificationObligation::ReverifyUtilitiesAndServiceConnections,
            QualificationObligation::RequalifyAffectedProcessConfigurations,
            QualificationObligation::ReconfirmOperatorSkillsAndStaffing,
            QualificationObligation::ReconfirmInspectionPlan,
            QualificationObligation::IndependentCommissioningReview,
        ]
    } else if transition.kind == TransitionKind::AllocationOnly {
        vec![
            QualificationObligation::RequalifyAffectedProcessConfigurations,
            QualificationObligation::ReconfirmOperatorSkillsAndStaffing,
            QualificationObligation::ReconfirmInspectionPlan,
        ]
    } else {
        Vec::new()
    };

    findings.finish(
        target,
        at,
        transition.kind,
        transition.actions,
        obligations,
        physical_change,
    )
}

fn missing_severity(resolution: SnapshotResolution) -> ReasonSeverity {
    match resolution {
        SnapshotResolution::Complete => ReasonSeverity::Blocked,
        SnapshotResolution::Partial | SnapshotResolution::Unknown => ReasonSeverity::Indeterminate,
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn overlaps(ax1: u64, ay1: u64, ax2: u64, ay2: u64, bx1: u64, by1: u64, bx2: u64, by2: u64) -> bool {
    ax1 < bx2 && bx1 < ax2 && ay1 < by2 && by1 < ay2
}

fn validate_geometry(
    target: &CellConfiguration,
    placements: &BTreeMap<&str, &ModulePlacement>,
    profiles: &BTreeMap<&str, &ModuleProfile>,
    findings: &mut Findings,
) {
    let mut rectangles = Vec::<(String, u64, u64, u64, u64)>::new();

    for (module_ref, placement) in placements {
        let Some(profile) = profiles.get(module_ref) else { continue };
        let (width, depth) = match placement.rotation_degrees {
            0 | 180 => (profile.footprint_width_mm, profile.footprint_depth_mm),
            90 | 270 => (profile.footprint_depth_mm, profile.footprint_width_mm),
            _ => continue,
        };
        let Some(right) = placement.x_mm.checked_add(width) else {
            findings.add(ReconfigurationReasonCode::PlacementArithmeticOverflow, ReasonSeverity::Blocked, module_ref);
            continue;
        };
        let Some(bottom) = placement.y_mm.checked_add(depth) else {
            findings.add(ReconfigurationReasonCode::PlacementArithmeticOverflow, ReasonSeverity::Blocked, module_ref);
            continue;
        };
        if right > target.width_mm || bottom > target.depth_mm {
            findings.add(ReconfigurationReasonCode::PlacementOutOfBounds, ReasonSeverity::Blocked, module_ref);
        }
        rectangles.push(((*module_ref).to_owned(), placement.x_mm, placement.y_mm, right, bottom));
    }

    for (index, first) in rectangles.iter().enumerate() {
        for second in rectangles.iter().skip(index + 1) {
            if overlaps(first.1, first.2, first.3, first.4, second.1, second.2, second.3, second.4) {
                findings.add(ReconfigurationReasonCode::ModuleFootprintsOverlap, ReasonSeverity::Blocked, &format!("{}|{}", first.0, second.0));
            }
        }
    }

    for keepout in &target.keepouts {
        if keepout.rect_ref.trim().is_empty() || keepout.width_mm == 0 || keepout.depth_mm == 0 {
            findings.add(ReconfigurationReasonCode::KeepoutOutOfBounds, ReasonSeverity::Blocked, &keepout.rect_ref);
            continue;
        }
        let Some(right) = keepout.x_mm.checked_add(keepout.width_mm) else {
            findings.add(ReconfigurationReasonCode::KeepoutArithmeticOverflow, ReasonSeverity::Blocked, &keepout.rect_ref);
            continue;
        };
        let Some(bottom) = keepout.y_mm.checked_add(keepout.depth_mm) else {
            findings.add(ReconfigurationReasonCode::KeepoutArithmeticOverflow, ReasonSeverity::Blocked, &keepout.rect_ref);
            continue;
        };
        if right > target.width_mm || bottom > target.depth_mm {
            findings.add(ReconfigurationReasonCode::KeepoutOutOfBounds, ReasonSeverity::Blocked, &keepout.rect_ref);
        }
        for module in &rectangles {
            if overlaps(module.1, module.2, module.3, module.4, keepout.x_mm, keepout.y_mm, right, bottom) {
                findings.add(ReconfigurationReasonCode::ModuleIntersectsKeepout, ReasonSeverity::Blocked, &module.0);
            }
        }
    }
}

fn validate_connections(
    target: &CellConfiguration,
    at: i64,
    profiles: &BTreeMap<&str, &ModuleProfile>,
    findings: &mut Findings,
) {
    let mut connection_ids = BTreeSet::new();
    let mut port_usage = BTreeMap::<(String, String), (u32, u64)>::new();

    for connection in &target.connections {
        if connection.connection_ref.trim().is_empty() {
            findings.add(ReconfigurationReasonCode::EmptyRequiredReference, ReasonSeverity::Indeterminate, "connection_ref");
        }
        if !connection_ids.insert(connection.connection_ref.as_str()) {
            findings.add(ReconfigurationReasonCode::DuplicateConnection, ReasonSeverity::Blocked, &connection.connection_ref);
        }

        let (Some(from_profile), Some(to_profile)) = (
            profiles.get(connection.from_module_ref.as_str()).copied(),
            profiles.get(connection.to_module_ref.as_str()).copied(),
        ) else {
            findings.add(ReconfigurationReasonCode::PortMissing, ReasonSeverity::Indeterminate, &connection.connection_ref);
            continue;
        };

        let from_matches: Vec<_> = from_profile.ports.iter().filter(|port| port.port_ref == connection.from_port_ref).collect();
        let to_matches: Vec<_> = to_profile.ports.iter().filter(|port| port.port_ref == connection.to_port_ref).collect();
        let from_port = match from_matches.as_slice() {
            [one] => Some(*one),
            [] => {
                findings.add(ReconfigurationReasonCode::PortMissing, ReasonSeverity::Blocked, &connection.from_port_ref);
                None
            }
            _ => {
                findings.add(ReconfigurationReasonCode::PortAmbiguous, ReasonSeverity::Blocked, &connection.from_port_ref);
                None
            }
        };
        let to_port = match to_matches.as_slice() {
            [one] => Some(*one),
            [] => {
                findings.add(ReconfigurationReasonCode::PortMissing, ReasonSeverity::Blocked, &connection.to_port_ref);
                None
            }
            _ => {
                findings.add(ReconfigurationReasonCode::PortAmbiguous, ReasonSeverity::Blocked, &connection.to_port_ref);
                None
            }
        };

        if let (Some(from), Some(to)) = (from_port, to_port) {
            if from.interface_type_ref != to.interface_type_ref {
                findings.add(ReconfigurationReasonCode::PortInterfaceMismatch, ReasonSeverity::Blocked, &connection.connection_ref);
            }
            if from.protocol_ref != to.protocol_ref {
                findings.add(ReconfigurationReasonCode::PortProtocolMismatch, ReasonSeverity::Blocked, &connection.connection_ref);
            }
            if from.medium_ref != to.medium_ref {
                findings.add(ReconfigurationReasonCode::PortMediumMismatch, ReasonSeverity::Blocked, &connection.connection_ref);
            }
            if from.unit_ref != to.unit_ref {
                findings.add(ReconfigurationReasonCode::PortUnitMismatch, ReasonSeverity::Blocked, &connection.connection_ref);
            }
            if !matches!(from.direction, PortDirection::Output | PortDirection::Bidirectional)
                || !matches!(to.direction, PortDirection::Input | PortDirection::Bidirectional)
            {
                findings.add(ReconfigurationReasonCode::PortDirectionMismatch, ReasonSeverity::Blocked, &connection.connection_ref);
            }
            if connection.capacity_units == 0
                || connection.capacity_units > from.max_capacity_units
                || connection.capacity_units > to.max_capacity_units
            {
                findings.add(ReconfigurationReasonCode::PortCapacityExceeded, ReasonSeverity::Blocked, &connection.connection_ref);
            }
            for key in [
                (connection.from_module_ref.clone(), connection.from_port_ref.clone()),
                (connection.to_module_ref.clone(), connection.to_port_ref.clone()),
            ] {
                let usage = port_usage.entry(key).or_insert((0, 0));
                match (usage.0.checked_add(1), usage.1.checked_add(connection.capacity_units)) {
                    (Some(count), Some(capacity)) => {
                        usage.0 = count;
                        usage.1 = capacity;
                    }
                    _ => findings.add(
                        ReconfigurationReasonCode::PortCapacityExceeded,
                        ReasonSeverity::Blocked,
                        &connection.connection_ref,
                    ),
                }
            }
        }

        findings.check_evidence(
            &connection.connection_evidence,
            at,
            ReconfigurationReasonCode::ConnectionEvidenceInvalid,
            &connection.connection_ref,
        );
    }

    for (module_ref, profile) in profiles {
        for port in &profile.ports {
            let key = ((*module_ref).to_owned(), port.port_ref.clone());
            let (count, capacity) = port_usage.get(&key).copied().unwrap_or((0, 0));
            if port.required_peer && count == 0 {
                findings.add(ReconfigurationReasonCode::RequiredPortUnconnected, missing_severity(target.resolution), &format!("{module_ref}:{}", port.port_ref));
            }
            if count > port.max_connections || capacity > port.max_capacity_units {
                findings.add(ReconfigurationReasonCode::PortConnectionCountExceeded, ReasonSeverity::Blocked, &format!("{module_ref}:{}", port.port_ref));
            }
        }
    }
}

fn validate_assignments(
    input: ReconfigurationInput<'_>,
    profiles: &BTreeMap<&str, &ModuleProfile>,
    findings: &mut Findings,
) {
    let mut demands = BTreeMap::<&str, &OperationDemand>::new();
    for demand in input.operation_demands {
        if demand.operation_ref.trim().is_empty() || demand.capability_ref.trim().is_empty()
            || demand.process_revision_ref.trim().is_empty() || demand.inspection_plan_ref.trim().is_empty()
        {
            findings.add(ReconfigurationReasonCode::EmptyRequiredReference, ReasonSeverity::Indeterminate, &demand.operation_ref);
        }
        if demands.insert(demand.operation_ref.as_str(), demand).is_some() {
            findings.add(ReconfigurationReasonCode::DuplicateOperationDemand, ReasonSeverity::Blocked, &demand.operation_ref);
        }
        if !is_sha256_hex(&demand.parameter_digest_sha256) {
            findings.add(ReconfigurationReasonCode::CapabilityDigestInvalid, ReasonSeverity::Blocked, &demand.operation_ref);
        }
    }

    let mut assignments = BTreeMap::<&str, &OperationAssignment>::new();
    for assignment in &input.target.assignments {
        if assignment.operation_ref.trim().is_empty() || assignment.module_ref.trim().is_empty() {
            findings.add(ReconfigurationReasonCode::EmptyRequiredReference, ReasonSeverity::Indeterminate, &assignment.operation_ref);
        }
        if assignments.insert(assignment.operation_ref.as_str(), assignment).is_some() {
            findings.add(ReconfigurationReasonCode::DuplicateOperationAssignment, ReasonSeverity::Blocked, &assignment.operation_ref);
        }
        let Some(demand) = demands.get(assignment.operation_ref.as_str()) else {
            findings.add(ReconfigurationReasonCode::AssignmentUnknownOperation, ReasonSeverity::Blocked, &assignment.operation_ref);
            continue;
        };
        let Some(profile) = profiles.get(assignment.module_ref.as_str()) else {
            findings.add(ReconfigurationReasonCode::AssignmentModuleMissing, ReasonSeverity::Indeterminate, &assignment.module_ref);
            continue;
        };
        let matches: Vec<_> = profile.capabilities.iter().filter(|capability| capability.capability_ref == demand.capability_ref).collect();
        match matches.as_slice() {
            [] => findings.add(ReconfigurationReasonCode::CapabilityMissing, ReasonSeverity::Blocked, &assignment.module_ref),
            [capability] => {
                if capability.process_revision_ref != demand.process_revision_ref {
                    findings.add(ReconfigurationReasonCode::CapabilityRevisionMismatch, ReasonSeverity::Blocked, &assignment.operation_ref);
                }
                if !is_sha256_hex(&capability.parameter_digest_sha256) {
                    findings.add(ReconfigurationReasonCode::CapabilityDigestInvalid, ReasonSeverity::Blocked, &assignment.operation_ref);
                } else if !capability.parameter_digest_sha256.eq_ignore_ascii_case(&demand.parameter_digest_sha256) {
                    findings.add(ReconfigurationReasonCode::CapabilityDigestMismatch, ReasonSeverity::Blocked, &assignment.operation_ref);
                }
                if capability.inspection_plan_ref != demand.inspection_plan_ref {
                    findings.add(ReconfigurationReasonCode::CapabilityInspectionMismatch, ReasonSeverity::Blocked, &assignment.operation_ref);
                }
            }
            _ => findings.add(ReconfigurationReasonCode::CapabilityAmbiguous, ReasonSeverity::Blocked, &assignment.module_ref),
        }
    }

    for operation_ref in demands.keys() {
        if !assignments.contains_key(operation_ref) {
            findings.add(ReconfigurationReasonCode::AssignmentMissing, missing_severity(input.target.resolution), operation_ref);
        }
    }
}

struct TransitionDelta {
    kind: TransitionKind,
    actions: Vec<TransitionAction>,
}

fn derive_transition(current: Option<&CellConfiguration>, target: &CellConfiguration) -> TransitionDelta {
    let mut actions = BTreeSet::<TransitionAction>::new();
    let Some(current) = current else {
        for placement in &target.placements {
            actions.insert(TransitionAction::AddModule { module_ref: placement.module_ref.clone() });
        }
        for connection in &target.connections {
            actions.insert(TransitionAction::Connect { connection_ref: connection.connection_ref.clone() });
        }
        for assignment in &target.assignments {
            actions.insert(TransitionAction::ReassignOperation {
                operation_ref: assignment.operation_ref.clone(),
                from_module_ref: None,
                to_module_ref: assignment.module_ref.clone(),
            });
        }
        actions.insert(TransitionAction::ReviewLayout);
        actions.insert(TransitionAction::CommissionAndIndependentlyRequalify);
        return TransitionDelta { kind: TransitionKind::PhysicalReconfiguration, actions: actions.into_iter().collect() };
    };

    let current_modules: BTreeMap<&str, &ModulePlacement> = current.placements.iter().map(|p| (p.module_ref.as_str(), p)).collect();
    let target_modules: BTreeMap<&str, &ModulePlacement> = target.placements.iter().map(|p| (p.module_ref.as_str(), p)).collect();

    for (module_ref, placement) in &current_modules {
        match target_modules.get(module_ref) {
            None => {
                actions.insert(TransitionAction::RemoveModule { module_ref: (*module_ref).to_owned() });
            }
            Some(next) if placement.x_mm != next.x_mm || placement.y_mm != next.y_mm
                || placement.rotation_degrees != next.rotation_degrees
                || placement.coordinate_frame_ref != next.coordinate_frame_ref =>
            {
                actions.insert(TransitionAction::RelocateModule {
                    module_ref: (*module_ref).to_owned(),
                    from_x_mm: placement.x_mm,
                    from_y_mm: placement.y_mm,
                    to_x_mm: next.x_mm,
                    to_y_mm: next.y_mm,
                    from_rotation_degrees: placement.rotation_degrees,
                    to_rotation_degrees: next.rotation_degrees,
                });
            }
            _ => {}
        }
    }
    for module_ref in target_modules.keys() {
        if !current_modules.contains_key(module_ref) {
            actions.insert(TransitionAction::AddModule { module_ref: (*module_ref).to_owned() });
        }
    }

    let current_connections: BTreeMap<&str, &PortConnection> = current.connections.iter().map(|c| (c.connection_ref.as_str(), c)).collect();
    let target_connections: BTreeMap<&str, &PortConnection> = target.connections.iter().map(|c| (c.connection_ref.as_str(), c)).collect();
    for (connection_ref, connection) in &current_connections {
        if target_connections.get(connection_ref).map_or(true, |next| *next != *connection) {
            actions.insert(TransitionAction::Disconnect { connection_ref: (*connection_ref).to_owned() });
        }
    }
    for (connection_ref, connection) in &target_connections {
        if current_connections.get(connection_ref).map_or(true, |old| *old != *connection) {
            actions.insert(TransitionAction::Connect { connection_ref: (*connection_ref).to_owned() });
        }
    }

    let current_assignments: BTreeMap<&str, &str> = current.assignments.iter().map(|a| (a.operation_ref.as_str(), a.module_ref.as_str())).collect();
    let target_assignments: BTreeMap<&str, &str> = target.assignments.iter().map(|a| (a.operation_ref.as_str(), a.module_ref.as_str())).collect();
    for (operation_ref, module_ref) in &target_assignments {
        if current_assignments.get(operation_ref).copied() != Some(*module_ref) {
            actions.insert(TransitionAction::ReassignOperation {
                operation_ref: (*operation_ref).to_owned(),
                from_module_ref: current_assignments.get(operation_ref).map(|v| (*v).to_owned()),
                to_module_ref: (*module_ref).to_owned(),
            });
        }
    }
    for (operation_ref, module_ref) in &current_assignments {
        if !target_assignments.contains_key(operation_ref) {
            actions.insert(TransitionAction::UnassignOperation {
                operation_ref: (*operation_ref).to_owned(),
                from_module_ref: (*module_ref).to_owned(),
            });
        }
    }

    let physical_change = current.facility_ref != target.facility_ref
        || current.layout_revision_ref != target.layout_revision_ref
        || current.coordinate_frame_ref != target.coordinate_frame_ref
        || current.width_mm != target.width_mm
        || current.depth_mm != target.depth_mm
        || current.geometry_basis != target.geometry_basis
        || current.keepouts != target.keepouts
        || actions.iter().any(|action| !matches!(
            action,
            TransitionAction::ReassignOperation { .. } | TransitionAction::UnassignOperation { .. }
        ));
    if physical_change {
        actions.insert(TransitionAction::ReviewLayout);
        actions.insert(TransitionAction::CommissionAndIndependentlyRequalify);
        TransitionDelta { kind: TransitionKind::PhysicalReconfiguration, actions: actions.into_iter().collect() }
    } else if actions.is_empty() {
        TransitionDelta { kind: TransitionKind::NoChange, actions: Vec::new() }
    } else {
        TransitionDelta { kind: TransitionKind::AllocationOnly, actions: actions.into_iter().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_800_000_000;
    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn evidence(reference: &str) -> EvidenceRecord {
        EvidenceRecord {
            evidence_ref: reference.to_owned(),
            status: EvidenceStatus::Current,
            valid_from_unix_seconds: Some(NOW - 10),
            valid_until_unix_seconds: Some(NOW + 10),
            issuer_ref: Some("urn:example:issuer:independent".to_owned()),
            sha256: Some(DIGEST_A.to_owned()),
        }
    }

    fn port(reference: &str, direction: PortDirection, required: bool) -> ModulePort {
        ModulePort {
            port_ref: reference.to_owned(),
            interface_type_ref: "urn:example:interface:material-transfer".to_owned(),
            protocol_ref: "urn:example:protocol:atelier-port-v1".to_owned(),
            medium_ref: "urn:example:medium:fabric-panel".to_owned(),
            unit_ref: "urn:example:unit:panel-per-hour".to_owned(),
            direction,
            max_connections: 1,
            max_capacity_units: 100,
            required_peer: required,
        }
    }

    fn module(reference: &str, x: u64, capabilities: Vec<ModuleCapability>) -> (ModuleProfile, ModulePlacement) {
        let profile = ModuleProfile {
            module_ref: reference.to_owned(),
            facility_ref: "urn:example:facility:atelier-01".to_owned(),
            status: ModuleStatus::Qualified,
            valid_from_unix_seconds: Some(NOW - 10),
            valid_until_unix_seconds: Some(NOW + 10),
            footprint_width_mm: 1_000,
            footprint_depth_mm: 800,
            capabilities,
            ports: if reference.ends_with("cut") {
                vec![port("out", PortDirection::Output, false)]
            } else {
                vec![port("in", PortDirection::Input, false)]
            },
            qualification_evidence: evidence(&format!("urn:example:evidence:{reference}:qualification")),
            independent_verification_evidence: evidence(&format!("urn:example:evidence:{reference}:review")),
        };
        let placement = ModulePlacement {
            module_ref: reference.to_owned(),
            x_mm: x,
            y_mm: 0,
            rotation_degrees: 0,
            coordinate_frame_ref: "urn:example:frame:factory-01".to_owned(),
        };
        (profile, placement)
    }

    fn demand(operation: &str, capability: &str) -> OperationDemand {
        OperationDemand {
            operation_ref: operation.to_owned(),
            capability_ref: capability.to_owned(),
            process_revision_ref: "urn:example:process:cut-r1".to_owned(),
            parameter_digest_sha256: DIGEST_B.to_owned(),
            inspection_plan_ref: "urn:example:inspection:cut-r1".to_owned(),
        }
    }

    fn capability(capability_ref: &str) -> ModuleCapability {
        ModuleCapability {
            capability_ref: capability_ref.to_owned(),
            process_revision_ref: "urn:example:process:cut-r1".to_owned(),
            parameter_digest_sha256: DIGEST_B.to_owned(),
            inspection_plan_ref: "urn:example:inspection:cut-r1".to_owned(),
        }
    }

    fn base_config(placements: Vec<ModulePlacement>, assignments: Vec<OperationAssignment>) -> CellConfiguration {
        CellConfiguration {
            configuration_ref: "urn:example:cell-config:r1".to_owned(),
            facility_ref: "urn:example:facility:atelier-01".to_owned(),
            layout_revision_ref: "urn:example:layout:r1".to_owned(),
            coordinate_frame_ref: "urn:example:frame:factory-01".to_owned(),
            width_mm: 5_000,
            depth_mm: 3_000,
            geometry_basis: GeometryBasis::Synthetic,
            resolution: SnapshotResolution::Complete,
            placements,
            keepouts: vec![],
            connections: vec![],
            assignments,
            layout_evidence: evidence("urn:example:evidence:layout"),
            commissioning_evidence: Some(evidence("urn:example:evidence:commissioned")),
        }
    }

    fn input<'a>(
        current: Option<&'a CellConfiguration>,
        target: &'a CellConfiguration,
        profiles: &'a [ModuleProfile],
        demands: &'a [OperationDemand],
    ) -> ReconfigurationInput<'a> {
        ReconfigurationInput {
            evaluated_at_unix_seconds: NOW,
            current,
            target,
            module_profiles: profiles,
            module_pool_resolution: SnapshotResolution::Complete,
            operation_demands: demands,
        }
    }

    #[test]
    fn initial_configuration_is_feasible_only_for_qualification() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![capability("urn:example:capability:cut")]);
        let target = base_config(vec![placement], vec![OperationAssignment {
            operation_ref: "urn:example:operation:cut".to_owned(),
            module_ref: "urn:example:module:cut".to_owned(),
        }]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[demand("urn:example:operation:cut", "urn:example:capability:cut")]));
        assert_eq!(report.status, PlanStatus::FeasibleForQualification);
        assert_eq!(report.transition_kind, TransitionKind::PhysicalReconfiguration);
        assert!(report.requires_independent_requalification);
        assert!(!report.production_authorized);
        assert!(report.qualification_obligations.contains(&QualificationObligation::IndependentCommissioningReview));
    }

    #[test]
    fn unchanged_configuration_is_no_change() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![capability("urn:example:capability:cut")]);
        let assignments = vec![OperationAssignment {
            operation_ref: "urn:example:operation:cut".to_owned(),
            module_ref: "urn:example:module:cut".to_owned(),
        }];
        let current = base_config(vec![placement.clone()], assignments.clone());
        let target = base_config(vec![placement], assignments);
        let report = evaluate_reconfiguration(input(Some(&current), &target, &[profile], &[demand("urn:example:operation:cut", "urn:example:capability:cut")]));
        assert_eq!(report.status, PlanStatus::FeasibleForQualification);
        assert_eq!(report.transition_kind, TransitionKind::NoChange);
        assert!(!report.requires_independent_requalification);
        assert!(report.actions.is_empty());
    }

    #[test]
    fn operation_reassignment_is_soft_reconfiguration() {
        let (profile_a, placement_a) = module("urn:example:module:cut", 0, vec![capability("urn:example:capability:cut")]);
        let (profile_b, placement_b) = module("urn:example:module:cut-spare", 1_200, vec![capability("urn:example:capability:cut")]);
        let operation = "urn:example:operation:cut";
        let current = base_config(
            vec![placement_a.clone(), placement_b.clone()],
            vec![OperationAssignment { operation_ref: operation.to_owned(), module_ref: "urn:example:module:cut".to_owned() }],
        );
        let target = base_config(
            vec![placement_a, placement_b],
            vec![OperationAssignment { operation_ref: operation.to_owned(), module_ref: "urn:example:module:cut-spare".to_owned() }],
        );
        let report = evaluate_reconfiguration(input(Some(&current), &target, &[profile_a, profile_b], &[demand(operation, "urn:example:capability:cut")]));
        assert_eq!(report.status, PlanStatus::FeasibleForQualification);
        assert_eq!(report.transition_kind, TransitionKind::AllocationOnly);
        assert!(!report.requires_independent_requalification);
        assert!(report.actions.iter().any(|action| matches!(action, TransitionAction::ReassignOperation { .. })));
    }

    #[test]
    fn relocation_requires_physical_requalification() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![capability("urn:example:capability:cut")]);
        let current = base_config(vec![placement.clone()], vec![]);
        let mut moved = placement;
        moved.x_mm = 1_500;
        let target = base_config(vec![moved], vec![]);
        let report = evaluate_reconfiguration(input(Some(&current), &target, &[profile], &[]));
        assert_eq!(report.transition_kind, TransitionKind::PhysicalReconfiguration);
        assert!(report.requires_independent_requalification);
        assert!(report.actions.iter().any(|action| matches!(action, TransitionAction::RelocateModule { .. })));
    }

    #[test]
    fn module_profile_missing_in_complete_pool_is_blocked() {
        let (_, placement) = module("urn:example:module:missing", 0, vec![]);
        let target = base_config(vec![placement], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::ModuleProfileMissing));
    }

    #[test]
    fn module_profile_missing_in_partial_pool_is_indeterminate() {
        let (_, placement) = module("urn:example:module:missing", 0, vec![]);
        let target = base_config(vec![placement], vec![]);
        let mut args = input(None, &target, &[], &[]);
        args.module_pool_resolution = SnapshotResolution::Partial;
        let report = evaluate_reconfiguration(args);
        assert_eq!(report.status, PlanStatus::Indeterminate);
    }

    #[test]
    fn expired_module_qualification_is_blocked() {
        let (mut profile, placement) = module("urn:example:module:cut", 0, vec![]);
        profile.valid_until_unix_seconds = Some(NOW);
        let target = base_config(vec![placement], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::ModuleExpired));
    }

    #[test]
    fn wrong_facility_partner_module_is_blocked() {
        let (mut profile, placement) = module("urn:example:module:cut", 0, vec![]);
        profile.facility_ref = "urn:example:facility:partner-99".to_owned();
        let target = base_config(vec![placement], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::ModuleWrongFacility));
    }

    #[test]
    fn process_digest_mismatch_is_blocked() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![ModuleCapability {
            capability_ref: "urn:example:capability:cut".to_owned(),
            process_revision_ref: "urn:example:process:cut-r1".to_owned(),
            parameter_digest_sha256: DIGEST_A.to_owned(),
            inspection_plan_ref: "urn:example:inspection:cut-r1".to_owned(),
        }]);
        let target = base_config(vec![placement], vec![OperationAssignment {
            operation_ref: "urn:example:operation:cut".to_owned(),
            module_ref: "urn:example:module:cut".to_owned(),
        }]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[demand("urn:example:operation:cut", "urn:example:capability:cut")]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::CapabilityDigestMismatch));
    }

    #[test]
    fn overlapping_modules_are_blocked() {
        let (profile_a, placement_a) = module("urn:example:module:cut-a", 0, vec![]);
        let (profile_b, placement_b) = module("urn:example:module:cut-b", 500, vec![]);
        let target = base_config(vec![placement_a, placement_b], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile_a, profile_b], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::ModuleFootprintsOverlap));
    }

    #[test]
    fn keepout_conflict_is_blocked() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![]);
        let mut target = base_config(vec![placement], vec![]);
        target.keepouts.push(LayoutRect {
            rect_ref: "urn:example:keepout:maintenance".to_owned(),
            x_mm: 500,
            y_mm: 100,
            width_mm: 300,
            depth_mm: 300,
        });
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::ModuleIntersectsKeepout));
    }

    #[test]
    fn geometry_frame_mismatch_is_blocked() {
        let (profile, mut placement) = module("urn:example:module:cut", 0, vec![]);
        placement.coordinate_frame_ref = "urn:example:frame:other".to_owned();
        let target = base_config(vec![placement], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::GeometryFrameMismatch));
    }

    #[test]
    fn incompatible_port_protocol_is_blocked() {
        let (mut profile_a, placement_a) = module("urn:example:module:cut", 0, vec![]);
        let (mut profile_b, placement_b) = module("urn:example:module:sew", 1_200, vec![]);
        profile_a.ports = vec![port("out", PortDirection::Output, true)];
        profile_b.ports = vec![port("in", PortDirection::Input, true)];
        profile_b.ports[0].protocol_ref = "urn:example:protocol:other".to_owned();
        let mut target = base_config(vec![placement_a, placement_b], vec![]);
        target.connections.push(PortConnection {
            connection_ref: "urn:example:connection:cut-to-sew".to_owned(),
            from_module_ref: "urn:example:module:cut".to_owned(),
            from_port_ref: "out".to_owned(),
            to_module_ref: "urn:example:module:sew".to_owned(),
            to_port_ref: "in".to_owned(),
            capacity_units: 10,
            connection_evidence: evidence("urn:example:evidence:connection"),
        });
        let report = evaluate_reconfiguration(input(None, &target, &[profile_a, profile_b], &[]));
        assert_eq!(report.status, PlanStatus::Blocked);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::PortProtocolMismatch));
    }

    #[test]
    fn required_port_without_peer_is_indeterminate() {
        let (mut profile, placement) = module("urn:example:module:cut", 0, vec![]);
        profile.ports = vec![port("out", PortDirection::Output, true)];
        let target = base_config(vec![placement], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[]));
        assert_eq!(report.status, PlanStatus::Indeterminate);
        assert!(report.reasons.iter().any(|r| r.code == ReconfigurationReasonCode::RequiredPortUnconnected));
    }

    #[test]
    fn missing_commissioning_evidence_on_unchanged_cell_is_indeterminate() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![]);
        let mut current = base_config(vec![placement.clone()], vec![]);
        current.commissioning_evidence = None;
        let mut target = base_config(vec![placement], vec![]);
        target.commissioning_evidence = None;
        let report = evaluate_reconfiguration(input(Some(&current), &target, &[profile], &[]));
        assert_eq!(report.transition_kind, TransitionKind::NoChange);
        assert_eq!(report.status, PlanStatus::Indeterminate);
        assert!(!report.production_authorized);
    }

    #[test]
    fn evidence_references_are_sorted_and_deduplicated() {
        let (profile, placement) = module("urn:example:module:cut", 0, vec![]);
        let target = base_config(vec![placement], vec![]);
        let report = evaluate_reconfiguration(input(None, &target, &[profile], &[]));
        assert!(report.evidence_refs.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(report.evidence_refs.contains(&"urn:example:evidence:layout".to_owned()));
    }

    #[test]
    fn result_is_deterministic_under_profile_order_changes() {
        let (profile_a, placement_a) = module("urn:example:module:cut-a", 0, vec![]);
        let (profile_b, placement_b) = module("urn:example:module:cut-b", 1_200, vec![]);
        let target = base_config(vec![placement_a, placement_b], vec![]);
        let first = evaluate_reconfiguration(input(None, &target, &[profile_a.clone(), profile_b.clone()], &[]));
        let second = evaluate_reconfiguration(input(None, &target, &[profile_b, profile_a], &[]));
        assert_eq!(first, second);
    }
}

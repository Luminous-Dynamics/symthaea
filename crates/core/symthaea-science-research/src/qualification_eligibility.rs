// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-authorizing qualification-eligibility manifests.
//!
//! This module composes already content-addressed research receipts and answers a
//! deliberately narrow question: under one exact, caller-declared eligibility
//! profile, are the requested prerequisite gates present and non-blocking?
//!
//! `EligibleUnderDeclaredProfile` is **not** scientific qualification. The
//! profile itself is ordinary caller-constructible policy and therefore only
//! `Declared`. This module never creates `Qualified` authority, never authenticates
//! the profile issuer, and never turns survival of falsification into positive
//! evidence.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    AuthorityLevel, ClaimRelationBindingReport, EvidenceCoverageAuthorityReport,
    EvidenceDecisionBindingClosure, EvidenceDecisionBindingReport, ExecutionAudit,
    ExecutionConformance, FalsificationReport, FalsificationStatus, IndependenceDimension,
    RelationBindingClosure, ReplicationLineageClosure, ReplicationLineageReport, ResearchId,
    ScientificClaim, Sha256Digest, UncertaintyClosure, UncertaintyReport,
};

pub const QUALIFICATION_ELIGIBILITY_SCHEMA: &str =
    "symthaea.qualification-eligibility-profile.v1";
const PROFILE_DIGEST_DOMAIN: &str =
    "symthaea.qualification-eligibility-profile.identity.v1";
const MANIFEST_DIGEST_DOMAIN: &str =
    "symthaea.qualification-eligibility-manifest.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EligibilityGateKind {
    ExecutionConformance,
    Falsification,
    Uncertainty,
    ReplicationLineage,
}

/// Claim-specific eligibility policy.
///
/// This freezes *which* optional gates are required and which exact upstream
/// semantic/coverage receipts the eligibility decision applies to. It remains a
/// caller-declared policy; a later qualification system must authenticate which
/// profiles are acceptable for which claim classes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationEligibilityProfile {
    pub schema_version: String,
    pub profile_id: ResearchId,
    pub claim_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    pub relation_binding_sha256: Sha256Digest,
    pub evidence_coverage_report_sha256: Sha256Digest,
    pub evidence_decision_binding_sha256: Sha256Digest,
    pub required_gates: BTreeSet<EligibilityGateKind>,
    /// Relevant only when `ExecutionConformance` is required. AuthorityBlocked
    /// is forbidden here and can never be made acceptable by profile choice.
    pub accepted_execution_conformance: BTreeSet<ExecutionConformance>,
    /// Relevant only when `ReplicationLineage` is required. This checks exact
    /// declared dimensions in the graph-local replication receipt; it never
    /// upgrades them into global independence.
    pub required_replication_dimensions: BTreeSet<IndependenceDimension>,
    pub supersedes_profile_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationEligibilityProfileIssue {
    WrongSchemaVersion { found: String },
    ExecutionPolicyMissing,
    AuthorityBlockedExecutionCannotBeAccepted,
    ExecutionPolicyWithoutGate,
    ReplicationDimensionsWithoutGate,
}

impl QualificationEligibilityProfile {
    pub fn validate(&self) -> Vec<QualificationEligibilityProfileIssue> {
        let mut issues = Vec::new();
        if self.schema_version != QUALIFICATION_ELIGIBILITY_SCHEMA {
            issues.push(QualificationEligibilityProfileIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        let execution_required = self
            .required_gates
            .contains(&EligibilityGateKind::ExecutionConformance);
        if execution_required && self.accepted_execution_conformance.is_empty() {
            issues.push(QualificationEligibilityProfileIssue::ExecutionPolicyMissing);
        }
        if self
            .accepted_execution_conformance
            .contains(&ExecutionConformance::AuthorityBlocked)
        {
            issues.push(
                QualificationEligibilityProfileIssue::AuthorityBlockedExecutionCannotBeAccepted,
            );
        }
        if !execution_required && !self.accepted_execution_conformance.is_empty() {
            issues.push(QualificationEligibilityProfileIssue::ExecutionPolicyWithoutGate);
        }
        if !self
            .required_gates
            .contains(&EligibilityGateKind::ReplicationLineage)
            && !self.required_replication_dimensions.is_empty()
        {
            issues.push(QualificationEligibilityProfileIssue::ReplicationDimensionsWithoutGate);
        }
        issues
    }

    pub fn freeze(
        self,
    ) -> Result<FrozenQualificationEligibilityProfile, Vec<QualificationEligibilityProfileIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let profile_sha256 = profile_digest(&self);
        Ok(FrozenQualificationEligibilityProfile {
            profile: self,
            profile_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenQualificationEligibilityProfile {
    profile: QualificationEligibilityProfile,
    profile_sha256: Sha256Digest,
}

impl FrozenQualificationEligibilityProfile {
    pub fn profile(&self) -> &QualificationEligibilityProfile {
        &self.profile
    }

    pub fn profile_sha256(&self) -> &Sha256Digest {
        &self.profile_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EligibilityGateState {
    Satisfied,
    NotRequired,
    Missing,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum QualificationEligibilityFinding {
    ClaimIdentityMismatch,
    RelationBindingReceiptMismatch,
    EvidenceCoverageReceiptMismatch,
    EvidenceDecisionBindingReceiptMismatch,
    CoverageDoesNotBindRelationReceipt,
    DecisionBindingDoesNotBindCoverageReceipt,
    RelationBindingIncomplete,
    RelationBindingRejected,
    RelationBindingInvalid,
    BoundClaimAuthorityAbsent,
    EvidenceCoverageAccountingIncomplete,
    EvidenceCoverageAccountingInvalid,
    EvidenceCoverageSemanticBindingIncomplete,
    EvidenceCoverageSemanticBindingRejected,
    EvidenceCoverageSemanticBindingInvalid,
    ProtocolLocalCoverageNotEstablished,
    RequiredGateMissing { gate: EligibilityGateKind },
    ExecutionConformanceRejected { observed: ExecutionConformance },
    FalsificationIncomplete,
    FalsificationRefuted,
    UncertaintyIncomplete,
    UncertaintyInvalid,
    ReplicationIsReplay,
    ReplicationDimensionMissing { dimension: IndependenceDimension },
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum QualificationEligibilityClosure {
    EligibleUnderDeclaredProfile,
    Incomplete,
    Blocked,
    Invalid,
}

/// Optional content-addressed reports for profile-selected gates.
pub struct QualificationEligibilityInputs<'a> {
    pub execution: Option<&'a ExecutionAudit>,
    pub falsification: Option<&'a FalsificationReport>,
    pub uncertainty: Option<&'a UncertaintyReport>,
    pub replication_lineage: Option<&'a ReplicationLineageReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationEligibilityManifest {
    manifest_sha256: Sha256Digest,
    profile_sha256: Sha256Digest,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    relation_binding_sha256: Sha256Digest,
    evidence_coverage_report_sha256: Sha256Digest,
    evidence_decision_binding_sha256: Sha256Digest,
    execution_audit_sha256: Option<Sha256Digest>,
    falsification_report_sha256: Option<Sha256Digest>,
    uncertainty_report_sha256: Option<Sha256Digest>,
    replication_lineage_report_sha256: Option<Sha256Digest>,
    gate_states: Vec<(EligibilityGateKind, EligibilityGateState)>,
    findings: Vec<QualificationEligibilityFinding>,
    closure: QualificationEligibilityClosure,
    /// The policy selecting these prerequisites is caller-declared in SCI-013A.
    profile_binding_level: AuthorityLevel,
    /// Hard-coded false: eligibility is not qualification.
    qualification_established: bool,
    /// Hard-coded false: even graph-local no-shared-lineage is not global
    /// independence.
    global_independence_established: bool,
}

impl QualificationEligibilityManifest {
    pub fn manifest_sha256(&self) -> &Sha256Digest {
        &self.manifest_sha256
    }

    pub fn profile_sha256(&self) -> &Sha256Digest {
        &self.profile_sha256
    }

    pub fn closure(&self) -> QualificationEligibilityClosure {
        self.closure
    }

    pub fn gate_states(&self) -> &[(EligibilityGateKind, EligibilityGateState)] {
        &self.gate_states
    }

    pub fn findings(&self) -> &[QualificationEligibilityFinding] {
        &self.findings
    }

    pub fn profile_binding_level(&self) -> AuthorityLevel {
        self.profile_binding_level
    }

    pub fn qualification_established(&self) -> bool {
        self.qualification_established
    }

    pub fn global_independence_established(&self) -> bool {
        self.global_independence_established
    }
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_qualification_eligibility(
    claim: &ScientificClaim,
    relation_binding: &ClaimRelationBindingReport,
    evidence_coverage: &EvidenceCoverageAuthorityReport,
    evidence_decision_binding: &EvidenceDecisionBindingReport,
    frozen_profile: &FrozenQualificationEligibilityProfile,
    inputs: QualificationEligibilityInputs<'_>,
) -> QualificationEligibilityManifest {
    let profile = frozen_profile.profile();
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut blocked = false;
    let mut incomplete = false;

    if profile.claim_id != *claim.claim_id() || profile.subject_sha256 != *claim.subject_sha256() {
        findings.push(QualificationEligibilityFinding::ClaimIdentityMismatch);
        invalid = true;
    }
    if &profile.relation_binding_sha256 != relation_binding.binding_sha256() {
        findings.push(QualificationEligibilityFinding::RelationBindingReceiptMismatch);
        invalid = true;
    }
    if &profile.evidence_coverage_report_sha256 != evidence_coverage.report_sha256() {
        findings.push(QualificationEligibilityFinding::EvidenceCoverageReceiptMismatch);
        invalid = true;
    }
    if &profile.evidence_decision_binding_sha256 != evidence_decision_binding.report_sha256() {
        findings.push(QualificationEligibilityFinding::EvidenceDecisionBindingReceiptMismatch);
        invalid = true;
    }
    if evidence_coverage.claim_binding_sha256() != relation_binding.binding_sha256() {
        findings.push(QualificationEligibilityFinding::CoverageDoesNotBindRelationReceipt);
        invalid = true;
    }
    if evidence_decision_binding.source_coverage_report_sha256() != evidence_coverage.report_sha256() {
        findings.push(QualificationEligibilityFinding::DecisionBindingDoesNotBindCoverageReceipt);
        invalid = true;
    }

    match relation_binding.closure() {
        RelationBindingClosure::Bound => {
            if relation_binding.bound_claimable_authority().is_empty() {
                findings.push(QualificationEligibilityFinding::BoundClaimAuthorityAbsent);
                blocked = true;
            }
        }
        RelationBindingClosure::Incomplete => {
            findings.push(QualificationEligibilityFinding::RelationBindingIncomplete);
            incomplete = true;
        }
        RelationBindingClosure::Rejected => {
            findings.push(QualificationEligibilityFinding::RelationBindingRejected);
            blocked = true;
        }
        RelationBindingClosure::Invalid => {
            findings.push(QualificationEligibilityFinding::RelationBindingInvalid);
            invalid = true;
        }
    }

    match evidence_coverage.closure() {
        crate::EvidenceCoverageClosure::CompleteWithinProtocol => {}
        crate::EvidenceCoverageClosure::Incomplete => {
            findings.push(QualificationEligibilityFinding::EvidenceCoverageAccountingIncomplete);
            incomplete = true;
        }
        crate::EvidenceCoverageClosure::Invalid => {
            findings.push(QualificationEligibilityFinding::EvidenceCoverageAccountingInvalid);
            invalid = true;
        }
    }

    match evidence_decision_binding.closure() {
        EvidenceDecisionBindingClosure::Bound => {}
        EvidenceDecisionBindingClosure::Incomplete => {
            findings.push(
                QualificationEligibilityFinding::EvidenceCoverageSemanticBindingIncomplete,
            );
            incomplete = true;
        }
        EvidenceDecisionBindingClosure::Rejected => {
            findings.push(
                QualificationEligibilityFinding::EvidenceCoverageSemanticBindingRejected,
            );
            blocked = true;
        }
        EvidenceDecisionBindingClosure::Invalid => {
            findings.push(
                QualificationEligibilityFinding::EvidenceCoverageSemanticBindingInvalid,
            );
            invalid = true;
        }
    }
    if !evidence_decision_binding.protocol_local_coverage_established() {
        findings.push(QualificationEligibilityFinding::ProtocolLocalCoverageNotEstablished);
        incomplete = true;
    }

    let mut gate_states = Vec::new();
    let execution_state = evaluate_execution_gate(profile, inputs.execution, &mut findings);
    update_flags(execution_state, &mut invalid, &mut blocked, &mut incomplete);
    gate_states.push((EligibilityGateKind::ExecutionConformance, execution_state));

    let falsification_state =
        evaluate_falsification_gate(profile, inputs.falsification, &mut findings);
    update_flags(falsification_state, &mut invalid, &mut blocked, &mut incomplete);
    gate_states.push((EligibilityGateKind::Falsification, falsification_state));

    let uncertainty_state = evaluate_uncertainty_gate(profile, inputs.uncertainty, &mut findings);
    update_flags(uncertainty_state, &mut invalid, &mut blocked, &mut incomplete);
    gate_states.push((EligibilityGateKind::Uncertainty, uncertainty_state));

    let replication_state =
        evaluate_replication_gate(profile, inputs.replication_lineage, &mut findings);
    update_flags(replication_state, &mut invalid, &mut blocked, &mut incomplete);
    gate_states.push((EligibilityGateKind::ReplicationLineage, replication_state));

    gate_states.sort_by_key(|(gate, _)| *gate);
    findings.sort();
    findings.dedup();

    let closure = if invalid {
        QualificationEligibilityClosure::Invalid
    } else if blocked {
        QualificationEligibilityClosure::Blocked
    } else if incomplete {
        QualificationEligibilityClosure::Incomplete
    } else {
        QualificationEligibilityClosure::EligibleUnderDeclaredProfile
    };

    let execution_audit_sha256 = inputs.execution.map(|report| report.audit_sha256().clone());
    let falsification_report_sha256 =
        inputs.falsification.map(|report| report.report_sha256().clone());
    let uncertainty_report_sha256 =
        inputs.uncertainty.map(|report| report.report_sha256().clone());
    let replication_lineage_report_sha256 = inputs
        .replication_lineage
        .map(|report| report.report_sha256().clone());

    let manifest_sha256 = manifest_digest(
        frozen_profile.profile_sha256(),
        claim,
        relation_binding.binding_sha256(),
        evidence_coverage.report_sha256(),
        evidence_decision_binding.report_sha256(),
        execution_audit_sha256.as_ref(),
        falsification_report_sha256.as_ref(),
        uncertainty_report_sha256.as_ref(),
        replication_lineage_report_sha256.as_ref(),
        &gate_states,
        &findings,
        closure,
    );

    QualificationEligibilityManifest {
        manifest_sha256,
        profile_sha256: frozen_profile.profile_sha256().clone(),
        claim_id: claim.claim_id().clone(),
        subject_sha256: claim.subject_sha256().clone(),
        relation_binding_sha256: relation_binding.binding_sha256().clone(),
        evidence_coverage_report_sha256: evidence_coverage.report_sha256().clone(),
        evidence_decision_binding_sha256: evidence_decision_binding.report_sha256().clone(),
        execution_audit_sha256,
        falsification_report_sha256,
        uncertainty_report_sha256,
        replication_lineage_report_sha256,
        gate_states,
        findings,
        closure,
        profile_binding_level: AuthorityLevel::Declared,
        qualification_established: false,
        global_independence_established: false,
    }
}

fn evaluate_execution_gate(
    profile: &QualificationEligibilityProfile,
    report: Option<&ExecutionAudit>,
    findings: &mut Vec<QualificationEligibilityFinding>,
) -> EligibilityGateState {
    let required = profile
        .required_gates
        .contains(&EligibilityGateKind::ExecutionConformance);
    let Some(report) = report else {
        if required {
            findings.push(QualificationEligibilityFinding::RequiredGateMissing {
                gate: EligibilityGateKind::ExecutionConformance,
            });
            return EligibilityGateState::Missing;
        }
        return EligibilityGateState::NotRequired;
    };

    if report.conformance() == ExecutionConformance::AuthorityBlocked {
        findings.push(QualificationEligibilityFinding::ExecutionConformanceRejected {
            observed: report.conformance(),
        });
        return EligibilityGateState::Blocked;
    }
    if required && !profile.accepted_execution_conformance.contains(&report.conformance()) {
        findings.push(QualificationEligibilityFinding::ExecutionConformanceRejected {
            observed: report.conformance(),
        });
        return EligibilityGateState::Blocked;
    }
    if required {
        EligibilityGateState::Satisfied
    } else {
        EligibilityGateState::NotRequired
    }
}

fn evaluate_falsification_gate(
    profile: &QualificationEligibilityProfile,
    report: Option<&FalsificationReport>,
    findings: &mut Vec<QualificationEligibilityFinding>,
) -> EligibilityGateState {
    let required = profile.required_gates.contains(&EligibilityGateKind::Falsification);
    let Some(report) = report else {
        if required {
            findings.push(QualificationEligibilityFinding::RequiredGateMissing {
                gate: EligibilityGateKind::Falsification,
            });
            return EligibilityGateState::Missing;
        }
        return EligibilityGateState::NotRequired;
    };

    match report.status() {
        FalsificationStatus::Refuted => {
            findings.push(QualificationEligibilityFinding::FalsificationRefuted);
            EligibilityGateState::Blocked
        }
        FalsificationStatus::Incomplete => {
            findings.push(QualificationEligibilityFinding::FalsificationIncomplete);
            EligibilityGateState::Incomplete
        }
        FalsificationStatus::SurvivedChallenges => {
            if required {
                EligibilityGateState::Satisfied
            } else {
                EligibilityGateState::NotRequired
            }
        }
    }
}

fn evaluate_uncertainty_gate(
    profile: &QualificationEligibilityProfile,
    report: Option<&UncertaintyReport>,
    findings: &mut Vec<QualificationEligibilityFinding>,
) -> EligibilityGateState {
    let required = profile.required_gates.contains(&EligibilityGateKind::Uncertainty);
    let Some(report) = report else {
        if required {
            findings.push(QualificationEligibilityFinding::RequiredGateMissing {
                gate: EligibilityGateKind::Uncertainty,
            });
            return EligibilityGateState::Missing;
        }
        return EligibilityGateState::NotRequired;
    };

    match report.closure() {
        UncertaintyClosure::Complete => {
            if required {
                EligibilityGateState::Satisfied
            } else {
                EligibilityGateState::NotRequired
            }
        }
        UncertaintyClosure::Incomplete => {
            findings.push(QualificationEligibilityFinding::UncertaintyIncomplete);
            EligibilityGateState::Incomplete
        }
        UncertaintyClosure::Invalid => {
            findings.push(QualificationEligibilityFinding::UncertaintyInvalid);
            EligibilityGateState::Invalid
        }
    }
}

fn evaluate_replication_gate(
    profile: &QualificationEligibilityProfile,
    report: Option<&ReplicationLineageReport>,
    findings: &mut Vec<QualificationEligibilityFinding>,
) -> EligibilityGateState {
    let required = profile
        .required_gates
        .contains(&EligibilityGateKind::ReplicationLineage);
    let Some(report) = report else {
        if required {
            findings.push(QualificationEligibilityFinding::RequiredGateMissing {
                gate: EligibilityGateKind::ReplicationLineage,
            });
            return EligibilityGateState::Missing;
        }
        return EligibilityGateState::NotRequired;
    };

    if report.closure() == ReplicationLineageClosure::Replay && required {
        findings.push(QualificationEligibilityFinding::ReplicationIsReplay);
        return EligibilityGateState::Incomplete;
    }
    if required {
        for dimension in &profile.required_replication_dimensions {
            if !report.declared_dimensions().contains(dimension) {
                findings.push(QualificationEligibilityFinding::ReplicationDimensionMissing {
                    dimension: *dimension,
                });
            }
        }
        if profile
            .required_replication_dimensions
            .iter()
            .any(|dimension| !report.declared_dimensions().contains(dimension))
        {
            EligibilityGateState::Incomplete
        } else {
            EligibilityGateState::Satisfied
        }
    } else {
        EligibilityGateState::NotRequired
    }
}

fn update_flags(
    state: EligibilityGateState,
    invalid: &mut bool,
    blocked: &mut bool,
    incomplete: &mut bool,
) {
    match state {
        EligibilityGateState::Invalid => *invalid = true,
        EligibilityGateState::Blocked => *blocked = true,
        EligibilityGateState::Missing | EligibilityGateState::Incomplete => *incomplete = true,
        EligibilityGateState::Satisfied | EligibilityGateState::NotRequired => {}
    }
}

fn profile_digest(profile: &QualificationEligibilityProfile) -> Sha256Digest {
    let mut digest = FramedDigest::new(PROFILE_DIGEST_DOMAIN);
    digest.text(QUALIFICATION_ELIGIBILITY_SCHEMA);
    digest.text(profile.profile_id.as_str());
    digest.text(profile.claim_id.as_str());
    digest.text(profile.subject_sha256.as_str());
    digest.text(profile.relation_binding_sha256.as_str());
    digest.text(profile.evidence_coverage_report_sha256.as_str());
    digest.text(profile.evidence_decision_binding_sha256.as_str());
    for gate in &profile.required_gates {
        digest.text("required-gate");
        digest.text(gate_tag(*gate));
    }
    for conformance in &profile.accepted_execution_conformance {
        digest.text("accepted-execution");
        digest.text(execution_conformance_tag(*conformance));
    }
    for dimension in &profile.required_replication_dimensions {
        digest.text("replication-dimension");
        digest.text(independence_dimension_tag(*dimension));
    }
    digest.optional_sha(profile.supersedes_profile_sha256.as_ref());
    digest.finish()
}

#[allow(clippy::too_many_arguments)]
fn manifest_digest(
    profile_sha256: &Sha256Digest,
    claim: &ScientificClaim,
    relation_binding_sha256: &Sha256Digest,
    evidence_coverage_sha256: &Sha256Digest,
    evidence_decision_binding_sha256: &Sha256Digest,
    execution_sha256: Option<&Sha256Digest>,
    falsification_sha256: Option<&Sha256Digest>,
    uncertainty_sha256: Option<&Sha256Digest>,
    replication_sha256: Option<&Sha256Digest>,
    gate_states: &[(EligibilityGateKind, EligibilityGateState)],
    findings: &[QualificationEligibilityFinding],
    closure: QualificationEligibilityClosure,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(MANIFEST_DIGEST_DOMAIN);
    digest.text(profile_sha256.as_str());
    digest.text(claim.claim_id().as_str());
    digest.text(claim.subject_sha256().as_str());
    digest.text(relation_binding_sha256.as_str());
    digest.text(evidence_coverage_sha256.as_str());
    digest.text(evidence_decision_binding_sha256.as_str());
    digest.optional_sha(execution_sha256);
    digest.optional_sha(falsification_sha256);
    digest.optional_sha(uncertainty_sha256);
    digest.optional_sha(replication_sha256);
    for (gate, state) in gate_states {
        digest.text("gate");
        digest.text(gate_tag(*gate));
        digest.text(gate_state_tag(*state));
    }
    digest.text(eligibility_closure_tag(closure));
    for finding in findings {
        digest.text("finding");
        digest_finding(&mut digest, finding);
    }
    digest.text("profile-binding-declared");
    digest.text("qualification-not-established");
    digest.text("global-independence-not-established");
    digest.finish()
}

fn digest_finding(digest: &mut FramedDigest, finding: &QualificationEligibilityFinding) {
    match finding {
        QualificationEligibilityFinding::ClaimIdentityMismatch => digest.text("claim-identity-mismatch"),
        QualificationEligibilityFinding::RelationBindingReceiptMismatch => digest.text("relation-binding-receipt-mismatch"),
        QualificationEligibilityFinding::EvidenceCoverageReceiptMismatch => digest.text("evidence-coverage-receipt-mismatch"),
        QualificationEligibilityFinding::EvidenceDecisionBindingReceiptMismatch => digest.text("evidence-decision-binding-receipt-mismatch"),
        QualificationEligibilityFinding::CoverageDoesNotBindRelationReceipt => digest.text("coverage-does-not-bind-relation-receipt"),
        QualificationEligibilityFinding::DecisionBindingDoesNotBindCoverageReceipt => digest.text("decision-binding-does-not-bind-coverage-receipt"),
        QualificationEligibilityFinding::RelationBindingIncomplete => digest.text("relation-binding-incomplete"),
        QualificationEligibilityFinding::RelationBindingRejected => digest.text("relation-binding-rejected"),
        QualificationEligibilityFinding::RelationBindingInvalid => digest.text("relation-binding-invalid"),
        QualificationEligibilityFinding::BoundClaimAuthorityAbsent => digest.text("bound-claim-authority-absent"),
        QualificationEligibilityFinding::EvidenceCoverageAccountingIncomplete => digest.text("evidence-coverage-accounting-incomplete"),
        QualificationEligibilityFinding::EvidenceCoverageAccountingInvalid => digest.text("evidence-coverage-accounting-invalid"),
        QualificationEligibilityFinding::EvidenceCoverageSemanticBindingIncomplete => digest.text("evidence-coverage-semantic-binding-incomplete"),
        QualificationEligibilityFinding::EvidenceCoverageSemanticBindingRejected => digest.text("evidence-coverage-semantic-binding-rejected"),
        QualificationEligibilityFinding::EvidenceCoverageSemanticBindingInvalid => digest.text("evidence-coverage-semantic-binding-invalid"),
        QualificationEligibilityFinding::ProtocolLocalCoverageNotEstablished => digest.text("protocol-local-coverage-not-established"),
        QualificationEligibilityFinding::RequiredGateMissing { gate } => {
            digest.text("required-gate-missing");
            digest.text(gate_tag(*gate));
        }
        QualificationEligibilityFinding::ExecutionConformanceRejected { observed } => {
            digest.text("execution-conformance-rejected");
            digest.text(execution_conformance_tag(*observed));
        }
        QualificationEligibilityFinding::FalsificationIncomplete => digest.text("falsification-incomplete"),
        QualificationEligibilityFinding::FalsificationRefuted => digest.text("falsification-refuted"),
        QualificationEligibilityFinding::UncertaintyIncomplete => digest.text("uncertainty-incomplete"),
        QualificationEligibilityFinding::UncertaintyInvalid => digest.text("uncertainty-invalid"),
        QualificationEligibilityFinding::ReplicationIsReplay => digest.text("replication-is-replay"),
        QualificationEligibilityFinding::ReplicationDimensionMissing { dimension } => {
            digest.text("replication-dimension-missing");
            digest.text(independence_dimension_tag(*dimension));
        }
    }
}

const fn gate_tag(gate: EligibilityGateKind) -> &'static str {
    match gate {
        EligibilityGateKind::ExecutionConformance => "execution-conformance",
        EligibilityGateKind::Falsification => "falsification",
        EligibilityGateKind::Uncertainty => "uncertainty",
        EligibilityGateKind::ReplicationLineage => "replication-lineage",
    }
}

const fn gate_state_tag(state: EligibilityGateState) -> &'static str {
    match state {
        EligibilityGateState::Satisfied => "satisfied",
        EligibilityGateState::NotRequired => "not-required",
        EligibilityGateState::Missing => "missing",
        EligibilityGateState::Incomplete => "incomplete",
        EligibilityGateState::Blocked => "blocked",
        EligibilityGateState::Invalid => "invalid",
    }
}

const fn eligibility_closure_tag(closure: QualificationEligibilityClosure) -> &'static str {
    match closure {
        QualificationEligibilityClosure::EligibleUnderDeclaredProfile => {
            "eligible-under-declared-profile"
        }
        QualificationEligibilityClosure::Incomplete => "incomplete",
        QualificationEligibilityClosure::Blocked => "blocked",
        QualificationEligibilityClosure::Invalid => "invalid",
    }
}

const fn execution_conformance_tag(value: ExecutionConformance) -> &'static str {
    match value {
        ExecutionConformance::Exact => "exact",
        ExecutionConformance::WithNonMaterialDeviations => "with-non-material-deviations",
        ExecutionConformance::WithMaterialDeviations => "with-material-deviations",
        ExecutionConformance::AuthorityBlocked => "authority-blocked",
    }
}

const fn independence_dimension_tag(value: IndependenceDimension) -> &'static str {
    match value {
        IndependenceDimension::SameExecutionReplay => "same-execution-replay",
        IndependenceDimension::IndependentExecutionSameBinary => "independent-execution-same-binary",
        IndependenceDimension::IndependentImplementationSharedInputs => {
            "independent-implementation-shared-inputs"
        }
        IndependenceDimension::IndependentMethodSharedRawData => {
            "independent-method-shared-raw-data"
        }
        IndependenceDimension::IndependentDataReduction => "independent-data-reduction",
        IndependenceDimension::IndependentDataset => "independent-dataset",
        IndependenceDimension::IndependentSite => "independent-site",
        IndependenceDimension::IndependentOrganization => "independent-organization",
    }
}

struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    fn new(domain: &str) -> Self {
        let mut digest = Self { bytes: Vec::new() };
        digest.text(domain);
        digest
    }

    fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_blocked_execution_can_never_be_profile_accepted() {
        let profile = QualificationEligibilityProfile {
            schema_version: QUALIFICATION_ELIGIBILITY_SCHEMA.into(),
            profile_id: ResearchId::parse("PROFILE-1").unwrap(),
            claim_id: ResearchId::parse("CLAIM-1").unwrap(),
            subject_sha256: Sha256Digest::of_bytes(b"subject"),
            relation_binding_sha256: Sha256Digest::of_bytes(b"relation"),
            evidence_coverage_report_sha256: Sha256Digest::of_bytes(b"coverage"),
            evidence_decision_binding_sha256: Sha256Digest::of_bytes(b"decision"),
            required_gates: [EligibilityGateKind::ExecutionConformance]
                .into_iter()
                .collect(),
            accepted_execution_conformance: [ExecutionConformance::AuthorityBlocked]
                .into_iter()
                .collect(),
            required_replication_dimensions: BTreeSet::new(),
            supersedes_profile_sha256: None,
        };
        assert!(profile.validate().iter().any(|issue| matches!(
            issue,
            QualificationEligibilityProfileIssue::AuthorityBlockedExecutionCannotBeAccepted
        )));
    }

    #[test]
    fn eligibility_name_does_not_imply_qualification() {
        assert_ne!(
            eligibility_closure_tag(QualificationEligibilityClosure::EligibleUnderDeclaredProfile),
            "qualified"
        );
    }
}

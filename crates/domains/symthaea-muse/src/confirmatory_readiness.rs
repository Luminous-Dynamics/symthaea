// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final ready/not-ready gate before confirmatory participant collection.
//!
//! This gate is intentionally conservative. It cannot create evidence; it only
//! binds already-sealed build, pilot, external-review, governance, dry-run, and
//! reproduction evidence into one auditable decision.

use crate::confirmatory_amendment_control::{
    ConfirmatoryAmendmentDecision, ConfirmatoryAmendmentLedger,
    validate_confirmatory_amendment_ledger,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::external_review_completion::{
    ExternalReviewCompletionEvidence, validate_external_review_completion,
};
use crate::pilot_monitoring::PilotOperationalDecision;
use crate::pilot_report::{PilotReviewReport, pilot_review_report_commitment};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_READINESS_VERSION: &str = "symthaea-muse-confirmatory-readiness-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceTargetValidation {
    pub target: String,
    pub command_sha256: String,
    pub log_sha256: String,
    pub tests_passed: usize,
    pub tests_ignored: usize,
    pub succeeded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IgnoredTestEvidence {
    pub test_name: String,
    pub rationale: String,
    pub relevant_to_confirmatory_claim: bool,
    pub manually_executed_successfully: bool,
    pub scheduled_validation_lane: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceValidationEvidence {
    pub source_revision: String,
    pub source_tree_sha256: String,
    pub flake_lock_sha256: String,
    pub rustc_version: String,
    pub cargo_version: String,
    pub nix_version: String,
    pub target_triple: String,
    pub targets: Vec<WorkspaceTargetValidation>,
    pub ignored_tests: Vec<IgnoredTestEvidence>,
    pub cargo_fmt_passed: bool,
    pub cargo_clippy_all_targets_all_features_passed: bool,
    pub release_build_passed: bool,
    pub workspace_tree_clean: bool,
    pub evidence_uri: String,
    pub evidence_sha256: String,
}

#[derive(Serialize)]
struct WorkspaceValidationCommitment<'a> {
    source_revision: &'a str,
    source_tree_sha256: &'a str,
    flake_lock_sha256: &'a str,
    rustc_version: &'a str,
    cargo_version: &'a str,
    nix_version: &'a str,
    target_triple: &'a str,
    targets: &'a [WorkspaceTargetValidation],
    ignored_tests: &'a [IgnoredTestEvidence],
    cargo_fmt_passed: bool,
    cargo_clippy_all_targets_all_features_passed: bool,
    release_build_passed: bool,
    workspace_tree_clean: bool,
    evidence_uri: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanStudyGovernanceEvidence {
    pub consent_document_sha256: String,
    pub participant_instructions_sha256: String,
    pub privacy_notice_sha256: String,
    pub retention_policy_sha256: String,
    pub recruitment_plan_sha256: String,
    pub compensation_plan_sha256: String,
    pub ethics_determination: String,
    pub ethics_reviewer_id: String,
    pub ethics_receipt_uri: String,
    pub ethics_receipt_sha256: String,
    pub raw_contact_data_excluded_from_study_evidence: bool,
    pub participant_withdrawal_tested: bool,
    pub deletion_workflow_tested: bool,
    pub governance_sha256: String,
}

#[derive(Serialize)]
struct HumanStudyGovernanceCommitment<'a> {
    consent_document_sha256: &'a str,
    participant_instructions_sha256: &'a str,
    privacy_notice_sha256: &'a str,
    retention_policy_sha256: &'a str,
    recruitment_plan_sha256: &'a str,
    compensation_plan_sha256: &'a str,
    ethics_determination: &'a str,
    ethics_reviewer_id: &'a str,
    ethics_receipt_uri: &'a str,
    ethics_receipt_sha256: &'a str,
    raw_contact_data_excluded_from_study_evidence: bool,
    participant_withdrawal_tested: bool,
    deletion_workflow_tested: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryDryRunEvidence {
    pub synthetic_fixture_manifest_sha256: String,
    pub artifact_bundle_sha256: String,
    pub runner_package_set_sha256: String,
    pub session_log_set_sha256: String,
    pub compiled_dataset_sha256: String,
    pub rust_analysis_sha256: String,
    pub independent_analysis_sha256: String,
    pub analysis_crosscheck_sha256: String,
    pub policy_budget_evidence_sha256: String,
    pub failure_injection_report_sha256: String,
    pub release_root_sha256: String,
    pub synthetic_data_only: bool,
    pub all_pipeline_stages_succeeded: bool,
    pub independent_analysis_agreed: bool,
    pub equal_policy_budgets_verified: bool,
    pub corruption_was_detected: bool,
    pub no_real_participant_data: bool,
    pub dry_run_sha256: String,
}

#[derive(Serialize)]
struct ConfirmatoryDryRunCommitment<'a> {
    synthetic_fixture_manifest_sha256: &'a str,
    artifact_bundle_sha256: &'a str,
    runner_package_set_sha256: &'a str,
    session_log_set_sha256: &'a str,
    compiled_dataset_sha256: &'a str,
    rust_analysis_sha256: &'a str,
    independent_analysis_sha256: &'a str,
    analysis_crosscheck_sha256: &'a str,
    policy_budget_evidence_sha256: &'a str,
    failure_injection_report_sha256: &'a str,
    release_root_sha256: &'a str,
    synthetic_data_only: bool,
    all_pipeline_stages_succeeded: bool,
    independent_analysis_agreed: bool,
    equal_policy_budgets_verified: bool,
    corruption_was_detected: bool,
    no_real_participant_data: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentReproductionReadiness {
    pub attestation_sha256s: Vec<String>,
    pub independent_verifier_ids: Vec<String>,
    pub independent_organization_count: usize,
    pub exact_release_reproduction_count: usize,
    pub independent_analysis_reproduction_count: usize,
    pub blocking_limitations: Vec<String>,
    pub external_receipt_uri: String,
    pub reproduction_sha256: String,
}

#[derive(Serialize)]
struct IndependentReproductionCommitment<'a> {
    attestation_sha256s: &'a [String],
    independent_verifier_ids: &'a [String],
    independent_organization_count: usize,
    exact_release_reproduction_count: usize,
    independent_analysis_reproduction_count: usize,
    blocking_limitations: &'a [String],
    external_receipt_uri: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatoryReadinessGate {
    WorkspaceValidation,
    IgnoredTestDisposition,
    PilotClosed,
    ExternalReviewComplete,
    AmendmentLock,
    HumanStudyGovernance,
    EndToEndDryRun,
    IndependentReproduction,
    ExternalPreregistration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryReadinessCheck {
    pub gate: ConfirmatoryReadinessGate,
    pub blocking: bool,
    pub passed: bool,
    pub evidence_sha256: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryReadinessDecision {
    ReadyForConfirmatoryCollection,
    NotReady,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryReadinessReport {
    pub report_version: String,
    pub study_operations_release_sha256: String,
    pub authority_snapshot_sha256: String,
    pub pilot_report_sha256: String,
    pub external_review_completion_sha256: String,
    pub amendment_ledger_sha256: String,
    pub workspace_validation_sha256: String,
    pub human_governance_sha256: String,
    pub dry_run_sha256: String,
    pub reproduction_sha256: String,
    pub checks: Vec<ConfirmatoryReadinessCheck>,
    pub decision: ConfirmatoryReadinessDecision,
    pub decided_at_utc: String,
    pub decision_authority: String,
    pub external_receipt_uri: String,
    pub external_receipt_sha256: String,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ConfirmatoryReadinessReportCommitment<'a> {
    report_version: &'a str,
    study_operations_release_sha256: &'a str,
    authority_snapshot_sha256: &'a str,
    pilot_report_sha256: &'a str,
    external_review_completion_sha256: &'a str,
    amendment_ledger_sha256: &'a str,
    workspace_validation_sha256: &'a str,
    human_governance_sha256: &'a str,
    dry_run_sha256: &'a str,
    reproduction_sha256: &'a str,
    checks: &'a [ConfirmatoryReadinessCheck],
    decision: ConfirmatoryReadinessDecision,
    decided_at_utc: &'a str,
    decision_authority: &'a str,
    external_receipt_uri: &'a str,
    external_receipt_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryReadinessIssue {
    WrongVersion { found: String },
    SerializationFailed { field: String },
    InvalidDigest { field: String },
    EmptyField { field: String },
    WorkspaceEvidenceInvalid { reason: String },
    GovernanceEvidenceInvalid { reason: String },
    DryRunEvidenceInvalid { reason: String },
    ReproductionEvidenceInvalid { reason: String },
    PilotNotClosed,
    PilotRiskUnresolved,
    PilotDataLeakage,
    ExternalReviewInvalid { issue_count: usize },
    AmendmentLedgerInvalid { issue_count: usize },
    AcceptedAmendmentRequiresNewBaseline,
    CollectionAlreadyStarted,
    MissingReadinessGate { gate: ConfirmatoryReadinessGate },
    DuplicateReadinessGate { gate: ConfirmatoryReadinessGate },
    NonBlockingRequiredGate { gate: ConfirmatoryReadinessGate },
    GateEvidenceMismatch { gate: ConfirmatoryReadinessGate },
    ReadyDecisionWithFailedGate { gate: ConfirmatoryReadinessGate },
    NotReadyDecisionWithoutFailedGate,
    ReportDigestMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn build_confirmatory_readiness_report(
    study_operations_release_sha256: String,
    pilot_report: &PilotReviewReport,
    external_review: &ExternalReviewCompletionEvidence,
    amendments: &ConfirmatoryAmendmentLedger,
    workspace: &WorkspaceValidationEvidence,
    governance: &HumanStudyGovernanceEvidence,
    dry_run: &ConfirmatoryDryRunEvidence,
    reproduction: &IndependentReproductionReadiness,
    decided_at_utc: String,
    decision_authority: String,
    external_receipt_uri: String,
    external_receipt_sha256: String,
) -> Result<ConfirmatoryReadinessReport, serde_json::Error> {
    let workspace_issues = validate_workspace_validation_evidence(workspace);
    let governance_issues = validate_human_study_governance(governance);
    let dry_run_issues = validate_confirmatory_dry_run(dry_run);
    let reproduction_issues = validate_independent_reproduction_readiness(reproduction);
    let external_review_issues = validate_external_review_completion(external_review);
    let amendment_issues = validate_confirmatory_amendment_ledger(amendments);

    let pilot_commitment_valid = pilot_review_report_commitment(pilot_report)
        .is_ok_and(|value| value == pilot_report.report_sha256)
        && is_sha256(&pilot_report.report_sha256);
    let operations_release_valid = is_sha256(&study_operations_release_sha256);
    let pilot_passed = pilot_commitment_valid
        && pilot_report.operational_decision == PilotOperationalDecision::ReadyToClosePilot
        && pilot_report.unresolved_operational_risks.is_empty()
        && pilot_report.pilot_data_excluded_from_confirmatory_analysis
        && !pilot_report.confirmatory_quality_claim_made
        && !pilot_report.instrument_changes_required
        && !pilot_report.confirmatory_manifest_must_be_refrozen;
    let ignored_tests_passed = workspace.ignored_tests.iter().all(|test| {
        !test.relevant_to_confirmatory_claim
            || (test.manually_executed_successfully
                && !test.scheduled_validation_lane.trim().is_empty())
    });
    let no_accepted_amendments = amendments
        .amendments
        .iter()
        .all(|amendment| amendment.decision == ConfirmatoryAmendmentDecision::Rejected);
    let preregistration_passed = operations_release_valid
        && is_sha256(&amendments.baseline_authority.preregistration_receipt_sha256);

    let mut checks = vec![
        check(
            ConfirmatoryReadinessGate::WorkspaceValidation,
            workspace_issues.is_empty(),
            &workspace.evidence_sha256,
            "All affected crates, release builds, formatting, and Clippy are green.",
        ),
        check(
            ConfirmatoryReadinessGate::IgnoredTestDisposition,
            ignored_tests_passed,
            &workspace.evidence_sha256,
            "Every claim-relevant ignored test has an explicit successful execution and scheduled lane.",
        ),
        check(
            ConfirmatoryReadinessGate::PilotClosed,
            pilot_passed,
            &pilot_report.report_sha256,
            "Pilot is closed, claim-limited, operationally clean, and excluded from confirmation.",
        ),
        check(
            ConfirmatoryReadinessGate::ExternalReviewComplete,
            external_review_issues.is_empty(),
            &external_review.completion_sha256,
            "All required external roles completed review and every finding has an accepted disposition.",
        ),
        check(
            ConfirmatoryReadinessGate::AmendmentLock,
            amendment_issues.is_empty()
                && no_accepted_amendments
                && amendments.confirmatory_collection_started_at_utc.is_none(),
            &amendments.ledger_sha256,
            "No accepted post-review amendment remains; accepted changes require a new baseline and review cycle.",
        ),
        check(
            ConfirmatoryReadinessGate::HumanStudyGovernance,
            governance_issues.is_empty(),
            &governance.governance_sha256,
            "Consent, privacy, retention, recruitment, compensation, withdrawal, and deletion controls are sealed.",
        ),
        check(
            ConfirmatoryReadinessGate::EndToEndDryRun,
            dry_run_issues.is_empty(),
            &dry_run.dry_run_sha256,
            "The complete pipeline reproduced on synthetic data and detected deliberate corruption.",
        ),
        check(
            ConfirmatoryReadinessGate::IndependentReproduction,
            reproduction_issues.is_empty(),
            &reproduction.reproduction_sha256,
            "At least one independent organization reproduced both the release and primary analysis.",
        ),
        check(
            ConfirmatoryReadinessGate::ExternalPreregistration,
            preregistration_passed,
            &amendments.baseline_authority.preregistration_receipt_sha256,
            "The exact confirmatory authority snapshot has an external preregistration receipt.",
        ),
    ];
    checks.sort_by_key(|check| check.gate);
    let decision = if checks.iter().all(|check| !check.blocking || check.passed) {
        ConfirmatoryReadinessDecision::ReadyForConfirmatoryCollection
    } else {
        ConfirmatoryReadinessDecision::NotReady
    };
    let mut report = ConfirmatoryReadinessReport {
        report_version: CONFIRMATORY_READINESS_VERSION.into(),
        study_operations_release_sha256,
        authority_snapshot_sha256: amendments.baseline_authority.snapshot_sha256.clone(),
        pilot_report_sha256: pilot_report.report_sha256.clone(),
        external_review_completion_sha256: external_review.completion_sha256.clone(),
        amendment_ledger_sha256: amendments.ledger_sha256.clone(),
        workspace_validation_sha256: workspace.evidence_sha256.clone(),
        human_governance_sha256: governance.governance_sha256.clone(),
        dry_run_sha256: dry_run.dry_run_sha256.clone(),
        reproduction_sha256: reproduction.reproduction_sha256.clone(),
        checks,
        decision,
        decided_at_utc,
        decision_authority,
        external_receipt_uri,
        external_receipt_sha256,
        report_sha256: String::new(),
    };
    report.report_sha256 = confirmatory_readiness_report_commitment(&report)?;
    Ok(report)
}

fn check(
    gate: ConfirmatoryReadinessGate,
    passed: bool,
    evidence_sha256: &str,
    detail: &str,
) -> ConfirmatoryReadinessCheck {
    ConfirmatoryReadinessCheck {
        gate,
        blocking: true,
        passed,
        evidence_sha256: evidence_sha256.into(),
        detail: detail.into(),
    }
}

pub fn workspace_validation_commitment(
    evidence: &WorkspaceValidationEvidence,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&WorkspaceValidationCommitment {
        source_revision: &evidence.source_revision,
        source_tree_sha256: &evidence.source_tree_sha256,
        flake_lock_sha256: &evidence.flake_lock_sha256,
        rustc_version: &evidence.rustc_version,
        cargo_version: &evidence.cargo_version,
        nix_version: &evidence.nix_version,
        target_triple: &evidence.target_triple,
        targets: &evidence.targets,
        ignored_tests: &evidence.ignored_tests,
        cargo_fmt_passed: evidence.cargo_fmt_passed,
        cargo_clippy_all_targets_all_features_passed: evidence
            .cargo_clippy_all_targets_all_features_passed,
        release_build_passed: evidence.release_build_passed,
        workspace_tree_clean: evidence.workspace_tree_clean,
        evidence_uri: &evidence.evidence_uri,
    })
}

pub fn seal_workspace_validation_evidence(
    evidence: &mut WorkspaceValidationEvidence,
) -> Result<(), serde_json::Error> {
    evidence.targets.sort_by(|a, b| a.target.cmp(&b.target));
    evidence
        .ignored_tests
        .sort_by(|a, b| a.test_name.cmp(&b.test_name));
    evidence.evidence_sha256 = workspace_validation_commitment(evidence)?;
    Ok(())
}

pub fn validate_workspace_validation_evidence(
    evidence: &WorkspaceValidationEvidence,
) -> Vec<ConfirmatoryReadinessIssue> {
    let mut issues = Vec::new();
    for (field, value) in [
        ("source_revision", evidence.source_revision.as_str()),
        ("rustc_version", evidence.rustc_version.as_str()),
        ("cargo_version", evidence.cargo_version.as_str()),
        ("nix_version", evidence.nix_version.as_str()),
        ("target_triple", evidence.target_triple.as_str()),
        ("evidence_uri", evidence.evidence_uri.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryReadinessIssue::EmptyField {
                field: format!("workspace.{field}"),
            });
        }
    }
    for (field, digest) in [
        ("source_tree_sha256", evidence.source_tree_sha256.as_str()),
        ("flake_lock_sha256", evidence.flake_lock_sha256.as_str()),
        ("evidence_sha256", evidence.evidence_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
                field: format!("workspace.{field}"),
            });
        }
    }
    let mut targets = BTreeSet::new();
    for target in &evidence.targets {
        if !targets.insert(target.target.clone()) {
            issues.push(ConfirmatoryReadinessIssue::WorkspaceEvidenceInvalid {
                reason: format!("duplicate target {}", target.target),
            });
        }
        if target.target.trim().is_empty()
            || !is_sha256(&target.command_sha256)
            || !is_sha256(&target.log_sha256)
            || !target.succeeded
        {
            issues.push(ConfirmatoryReadinessIssue::WorkspaceEvidenceInvalid {
                reason: format!("invalid or failed target {}", target.target),
            });
        }
    }
    for required in [
        "symthaea-music-theory",
        "symthaea-fep",
        "symthaea-muse-lib",
        "cognitive-study-bin",
        "muse-studio-bin",
    ] {
        if !targets.contains(required) {
            issues.push(ConfirmatoryReadinessIssue::WorkspaceEvidenceInvalid {
                reason: format!("missing required target {required}"),
            });
        }
    }
    if !(evidence.cargo_fmt_passed
        && evidence.cargo_clippy_all_targets_all_features_passed
        && evidence.release_build_passed
        && evidence.workspace_tree_clean)
    {
        issues.push(ConfirmatoryReadinessIssue::WorkspaceEvidenceInvalid {
            reason: "format, Clippy, release build, or clean-tree gate failed".into(),
        });
    }
    match workspace_validation_commitment(evidence) {
        Ok(value) if value == evidence.evidence_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryReadinessIssue::WorkspaceEvidenceInvalid {
            reason: "workspace evidence digest mismatch".into(),
        }),
        Err(_) => issues.push(ConfirmatoryReadinessIssue::SerializationFailed {
            field: "workspace".into(),
        }),
    }
    issues
}

pub fn human_study_governance_commitment(
    evidence: &HumanStudyGovernanceEvidence,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&HumanStudyGovernanceCommitment {
        consent_document_sha256: &evidence.consent_document_sha256,
        participant_instructions_sha256: &evidence.participant_instructions_sha256,
        privacy_notice_sha256: &evidence.privacy_notice_sha256,
        retention_policy_sha256: &evidence.retention_policy_sha256,
        recruitment_plan_sha256: &evidence.recruitment_plan_sha256,
        compensation_plan_sha256: &evidence.compensation_plan_sha256,
        ethics_determination: &evidence.ethics_determination,
        ethics_reviewer_id: &evidence.ethics_reviewer_id,
        ethics_receipt_uri: &evidence.ethics_receipt_uri,
        ethics_receipt_sha256: &evidence.ethics_receipt_sha256,
        raw_contact_data_excluded_from_study_evidence: evidence
            .raw_contact_data_excluded_from_study_evidence,
        participant_withdrawal_tested: evidence.participant_withdrawal_tested,
        deletion_workflow_tested: evidence.deletion_workflow_tested,
    })
}

pub fn seal_human_study_governance(
    evidence: &mut HumanStudyGovernanceEvidence,
) -> Result<(), serde_json::Error> {
    evidence.governance_sha256 = human_study_governance_commitment(evidence)?;
    Ok(())
}

pub fn validate_human_study_governance(
    evidence: &HumanStudyGovernanceEvidence,
) -> Vec<ConfirmatoryReadinessIssue> {
    let mut issues = Vec::new();
    for (field, digest) in [
        (
            "consent_document_sha256",
            evidence.consent_document_sha256.as_str(),
        ),
        (
            "participant_instructions_sha256",
            evidence.participant_instructions_sha256.as_str(),
        ),
        (
            "privacy_notice_sha256",
            evidence.privacy_notice_sha256.as_str(),
        ),
        (
            "retention_policy_sha256",
            evidence.retention_policy_sha256.as_str(),
        ),
        (
            "recruitment_plan_sha256",
            evidence.recruitment_plan_sha256.as_str(),
        ),
        (
            "compensation_plan_sha256",
            evidence.compensation_plan_sha256.as_str(),
        ),
        (
            "ethics_receipt_sha256",
            evidence.ethics_receipt_sha256.as_str(),
        ),
        ("governance_sha256", evidence.governance_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
                field: format!("governance.{field}"),
            });
        }
    }
    for (field, value) in [
        (
            "ethics_determination",
            evidence.ethics_determination.as_str(),
        ),
        ("ethics_reviewer_id", evidence.ethics_reviewer_id.as_str()),
        ("ethics_receipt_uri", evidence.ethics_receipt_uri.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryReadinessIssue::EmptyField {
                field: format!("governance.{field}"),
            });
        }
    }
    if !(evidence.raw_contact_data_excluded_from_study_evidence
        && evidence.participant_withdrawal_tested
        && evidence.deletion_workflow_tested)
    {
        issues.push(ConfirmatoryReadinessIssue::GovernanceEvidenceInvalid {
            reason: "privacy, withdrawal, or deletion control is unverified".into(),
        });
    }
    match human_study_governance_commitment(evidence) {
        Ok(value) if value == evidence.governance_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryReadinessIssue::GovernanceEvidenceInvalid {
            reason: "governance evidence digest mismatch".into(),
        }),
        Err(_) => issues.push(ConfirmatoryReadinessIssue::SerializationFailed {
            field: "governance".into(),
        }),
    }
    issues
}

pub fn confirmatory_dry_run_commitment(
    evidence: &ConfirmatoryDryRunEvidence,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ConfirmatoryDryRunCommitment {
        synthetic_fixture_manifest_sha256: &evidence.synthetic_fixture_manifest_sha256,
        artifact_bundle_sha256: &evidence.artifact_bundle_sha256,
        runner_package_set_sha256: &evidence.runner_package_set_sha256,
        session_log_set_sha256: &evidence.session_log_set_sha256,
        compiled_dataset_sha256: &evidence.compiled_dataset_sha256,
        rust_analysis_sha256: &evidence.rust_analysis_sha256,
        independent_analysis_sha256: &evidence.independent_analysis_sha256,
        analysis_crosscheck_sha256: &evidence.analysis_crosscheck_sha256,
        policy_budget_evidence_sha256: &evidence.policy_budget_evidence_sha256,
        failure_injection_report_sha256: &evidence.failure_injection_report_sha256,
        release_root_sha256: &evidence.release_root_sha256,
        synthetic_data_only: evidence.synthetic_data_only,
        all_pipeline_stages_succeeded: evidence.all_pipeline_stages_succeeded,
        independent_analysis_agreed: evidence.independent_analysis_agreed,
        equal_policy_budgets_verified: evidence.equal_policy_budgets_verified,
        corruption_was_detected: evidence.corruption_was_detected,
        no_real_participant_data: evidence.no_real_participant_data,
    })
}

pub fn seal_confirmatory_dry_run(
    evidence: &mut ConfirmatoryDryRunEvidence,
) -> Result<(), serde_json::Error> {
    evidence.dry_run_sha256 = confirmatory_dry_run_commitment(evidence)?;
    Ok(())
}

pub fn validate_confirmatory_dry_run(
    evidence: &ConfirmatoryDryRunEvidence,
) -> Vec<ConfirmatoryReadinessIssue> {
    let mut issues = Vec::new();
    for (field, digest) in [
        (
            "synthetic_fixture_manifest_sha256",
            evidence.synthetic_fixture_manifest_sha256.as_str(),
        ),
        (
            "artifact_bundle_sha256",
            evidence.artifact_bundle_sha256.as_str(),
        ),
        (
            "runner_package_set_sha256",
            evidence.runner_package_set_sha256.as_str(),
        ),
        (
            "session_log_set_sha256",
            evidence.session_log_set_sha256.as_str(),
        ),
        (
            "compiled_dataset_sha256",
            evidence.compiled_dataset_sha256.as_str(),
        ),
        (
            "rust_analysis_sha256",
            evidence.rust_analysis_sha256.as_str(),
        ),
        (
            "independent_analysis_sha256",
            evidence.independent_analysis_sha256.as_str(),
        ),
        (
            "analysis_crosscheck_sha256",
            evidence.analysis_crosscheck_sha256.as_str(),
        ),
        (
            "policy_budget_evidence_sha256",
            evidence.policy_budget_evidence_sha256.as_str(),
        ),
        (
            "failure_injection_report_sha256",
            evidence.failure_injection_report_sha256.as_str(),
        ),
        ("release_root_sha256", evidence.release_root_sha256.as_str()),
        ("dry_run_sha256", evidence.dry_run_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
                field: format!("dry_run.{field}"),
            });
        }
    }
    if !(evidence.synthetic_data_only
        && evidence.all_pipeline_stages_succeeded
        && evidence.independent_analysis_agreed
        && evidence.equal_policy_budgets_verified
        && evidence.corruption_was_detected
        && evidence.no_real_participant_data)
    {
        issues.push(ConfirmatoryReadinessIssue::DryRunEvidenceInvalid {
            reason: "one or more dry-run truth gates failed".into(),
        });
    }
    match confirmatory_dry_run_commitment(evidence) {
        Ok(value) if value == evidence.dry_run_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryReadinessIssue::DryRunEvidenceInvalid {
            reason: "dry-run evidence digest mismatch".into(),
        }),
        Err(_) => issues.push(ConfirmatoryReadinessIssue::SerializationFailed {
            field: "dry_run".into(),
        }),
    }
    issues
}

pub fn independent_reproduction_commitment(
    evidence: &IndependentReproductionReadiness,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&IndependentReproductionCommitment {
        attestation_sha256s: &evidence.attestation_sha256s,
        independent_verifier_ids: &evidence.independent_verifier_ids,
        independent_organization_count: evidence.independent_organization_count,
        exact_release_reproduction_count: evidence.exact_release_reproduction_count,
        independent_analysis_reproduction_count: evidence.independent_analysis_reproduction_count,
        blocking_limitations: &evidence.blocking_limitations,
        external_receipt_uri: &evidence.external_receipt_uri,
    })
}

pub fn seal_independent_reproduction_readiness(
    evidence: &mut IndependentReproductionReadiness,
) -> Result<(), serde_json::Error> {
    evidence.attestation_sha256s.sort();
    evidence.attestation_sha256s.dedup();
    evidence.independent_verifier_ids.sort();
    evidence.independent_verifier_ids.dedup();
    evidence.blocking_limitations.sort();
    evidence.reproduction_sha256 = independent_reproduction_commitment(evidence)?;
    Ok(())
}

pub fn validate_independent_reproduction_readiness(
    evidence: &IndependentReproductionReadiness,
) -> Vec<ConfirmatoryReadinessIssue> {
    let mut issues = Vec::new();
    for digest in &evidence.attestation_sha256s {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
                field: "reproduction.attestation_sha256s".into(),
            });
        }
    }
    if !is_sha256(&evidence.reproduction_sha256) {
        issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
            field: "reproduction.reproduction_sha256".into(),
        });
    }
    if evidence.external_receipt_uri.trim().is_empty()
        || evidence.independent_verifier_ids.is_empty()
        || evidence.attestation_sha256s.len() < evidence.independent_verifier_ids.len()
        || evidence.independent_organization_count == 0
        || evidence.independent_organization_count > evidence.independent_verifier_ids.len()
        || evidence.exact_release_reproduction_count == 0
        || evidence.independent_analysis_reproduction_count == 0
        || !evidence.blocking_limitations.is_empty()
    {
        issues.push(ConfirmatoryReadinessIssue::ReproductionEvidenceInvalid {
            reason: "independence, exact release, analysis agreement, or limitation gate failed"
                .into(),
        });
    }
    match independent_reproduction_commitment(evidence) {
        Ok(value) if value == evidence.reproduction_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryReadinessIssue::ReproductionEvidenceInvalid {
            reason: "reproduction evidence digest mismatch".into(),
        }),
        Err(_) => issues.push(ConfirmatoryReadinessIssue::SerializationFailed {
            field: "reproduction".into(),
        }),
    }
    issues
}

pub fn confirmatory_readiness_report_commitment(
    report: &ConfirmatoryReadinessReport,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ConfirmatoryReadinessReportCommitment {
        report_version: &report.report_version,
        study_operations_release_sha256: &report.study_operations_release_sha256,
        authority_snapshot_sha256: &report.authority_snapshot_sha256,
        pilot_report_sha256: &report.pilot_report_sha256,
        external_review_completion_sha256: &report.external_review_completion_sha256,
        amendment_ledger_sha256: &report.amendment_ledger_sha256,
        workspace_validation_sha256: &report.workspace_validation_sha256,
        human_governance_sha256: &report.human_governance_sha256,
        dry_run_sha256: &report.dry_run_sha256,
        reproduction_sha256: &report.reproduction_sha256,
        checks: &report.checks,
        decision: report.decision,
        decided_at_utc: &report.decided_at_utc,
        decision_authority: &report.decision_authority,
        external_receipt_uri: &report.external_receipt_uri,
        external_receipt_sha256: &report.external_receipt_sha256,
    })
}

pub fn validate_confirmatory_readiness_report(
    report: &ConfirmatoryReadinessReport,
) -> Vec<ConfirmatoryReadinessIssue> {
    let mut issues = Vec::new();
    if report.report_version != CONFIRMATORY_READINESS_VERSION {
        issues.push(ConfirmatoryReadinessIssue::WrongVersion {
            found: report.report_version.clone(),
        });
    }
    for (field, digest) in [
        (
            "study_operations_release_sha256",
            report.study_operations_release_sha256.as_str(),
        ),
        (
            "authority_snapshot_sha256",
            report.authority_snapshot_sha256.as_str(),
        ),
        ("pilot_report_sha256", report.pilot_report_sha256.as_str()),
        (
            "external_review_completion_sha256",
            report.external_review_completion_sha256.as_str(),
        ),
        (
            "amendment_ledger_sha256",
            report.amendment_ledger_sha256.as_str(),
        ),
        (
            "workspace_validation_sha256",
            report.workspace_validation_sha256.as_str(),
        ),
        (
            "human_governance_sha256",
            report.human_governance_sha256.as_str(),
        ),
        ("dry_run_sha256", report.dry_run_sha256.as_str()),
        ("reproduction_sha256", report.reproduction_sha256.as_str()),
        (
            "external_receipt_sha256",
            report.external_receipt_sha256.as_str(),
        ),
        ("report_sha256", report.report_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("decided_at_utc", report.decided_at_utc.as_str()),
        ("decision_authority", report.decision_authority.as_str()),
        ("external_receipt_uri", report.external_receipt_uri.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryReadinessIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    let expected_evidence = [
        (
            ConfirmatoryReadinessGate::WorkspaceValidation,
            report.workspace_validation_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::IgnoredTestDisposition,
            report.workspace_validation_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::PilotClosed,
            report.pilot_report_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::ExternalReviewComplete,
            report.external_review_completion_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::AmendmentLock,
            report.amendment_ledger_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::HumanStudyGovernance,
            report.human_governance_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::EndToEndDryRun,
            report.dry_run_sha256.as_str(),
        ),
        (
            ConfirmatoryReadinessGate::IndependentReproduction,
            report.reproduction_sha256.as_str(),
        ),
    ]
    .into_iter()
    .collect::<std::collections::BTreeMap<_, _>>();
    let mut gates = BTreeSet::new();
    let mut failed_blocking = Vec::new();
    for check in &report.checks {
        if !gates.insert(check.gate) {
            issues.push(ConfirmatoryReadinessIssue::DuplicateReadinessGate { gate: check.gate });
        }
        if !check.blocking {
            issues.push(ConfirmatoryReadinessIssue::NonBlockingRequiredGate { gate: check.gate });
        }
        let evidence_matches = match check.gate {
            ConfirmatoryReadinessGate::ExternalPreregistration => is_sha256(&check.evidence_sha256),
            _ => expected_evidence
                .get(&check.gate)
                .is_some_and(|expected| *expected == check.evidence_sha256.as_str()),
        };
        if !evidence_matches {
            issues.push(ConfirmatoryReadinessIssue::GateEvidenceMismatch { gate: check.gate });
        }
        if !is_sha256(&check.evidence_sha256) || check.detail.trim().is_empty() {
            issues.push(ConfirmatoryReadinessIssue::InvalidDigest {
                field: format!("check.{:?}.evidence_sha256", check.gate),
            });
        }
        if check.blocking && !check.passed {
            failed_blocking.push(check.gate);
        }
    }
    for gate in [
        ConfirmatoryReadinessGate::WorkspaceValidation,
        ConfirmatoryReadinessGate::IgnoredTestDisposition,
        ConfirmatoryReadinessGate::PilotClosed,
        ConfirmatoryReadinessGate::ExternalReviewComplete,
        ConfirmatoryReadinessGate::AmendmentLock,
        ConfirmatoryReadinessGate::HumanStudyGovernance,
        ConfirmatoryReadinessGate::EndToEndDryRun,
        ConfirmatoryReadinessGate::IndependentReproduction,
        ConfirmatoryReadinessGate::ExternalPreregistration,
    ] {
        if !gates.contains(&gate) {
            issues.push(ConfirmatoryReadinessIssue::MissingReadinessGate { gate });
        }
    }
    match report.decision {
        ConfirmatoryReadinessDecision::ReadyForConfirmatoryCollection => {
            for gate in failed_blocking {
                issues.push(ConfirmatoryReadinessIssue::ReadyDecisionWithFailedGate { gate });
            }
        }
        ConfirmatoryReadinessDecision::NotReady if failed_blocking.is_empty() => {
            issues.push(ConfirmatoryReadinessIssue::NotReadyDecisionWithoutFailedGate);
        }
        ConfirmatoryReadinessDecision::NotReady => {}
    }
    match confirmatory_readiness_report_commitment(report) {
        Ok(value) if value == report.report_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryReadinessIssue::ReportDigestMismatch),
        Err(_) => issues.push(ConfirmatoryReadinessIssue::SerializationFailed {
            field: "report".into(),
        }),
    }
    issues
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_readiness_gates_are_blocking() {
        let gate = check(
            ConfirmatoryReadinessGate::WorkspaceValidation,
            true,
            &"a".repeat(64),
            "test",
        );
        assert!(gate.blocking);
    }

    #[test]
    fn ready_decision_is_distinct_from_not_ready() {
        assert_ne!(
            ConfirmatoryReadinessDecision::ReadyForConfirmatoryCollection,
            ConfirmatoryReadinessDecision::NotReady
        );
    }
    fn valid_report() -> ConfirmatoryReadinessReport {
        let workspace = "1".repeat(64);
        let pilot = "2".repeat(64);
        let review = "3".repeat(64);
        let amendments = "4".repeat(64);
        let governance = "5".repeat(64);
        let dry_run = "6".repeat(64);
        let reproduction = "7".repeat(64);
        let preregistration = "8".repeat(64);
        let mut report = ConfirmatoryReadinessReport {
            report_version: CONFIRMATORY_READINESS_VERSION.into(),
            study_operations_release_sha256: "9".repeat(64),
            authority_snapshot_sha256: "a".repeat(64),
            pilot_report_sha256: pilot.clone(),
            external_review_completion_sha256: review.clone(),
            amendment_ledger_sha256: amendments.clone(),
            workspace_validation_sha256: workspace.clone(),
            human_governance_sha256: governance.clone(),
            dry_run_sha256: dry_run.clone(),
            reproduction_sha256: reproduction.clone(),
            checks: vec![
                check(
                    ConfirmatoryReadinessGate::WorkspaceValidation,
                    true,
                    &workspace,
                    "ok",
                ),
                check(
                    ConfirmatoryReadinessGate::IgnoredTestDisposition,
                    true,
                    &workspace,
                    "ok",
                ),
                check(ConfirmatoryReadinessGate::PilotClosed, true, &pilot, "ok"),
                check(
                    ConfirmatoryReadinessGate::ExternalReviewComplete,
                    true,
                    &review,
                    "ok",
                ),
                check(
                    ConfirmatoryReadinessGate::AmendmentLock,
                    true,
                    &amendments,
                    "ok",
                ),
                check(
                    ConfirmatoryReadinessGate::HumanStudyGovernance,
                    true,
                    &governance,
                    "ok",
                ),
                check(
                    ConfirmatoryReadinessGate::EndToEndDryRun,
                    true,
                    &dry_run,
                    "ok",
                ),
                check(
                    ConfirmatoryReadinessGate::IndependentReproduction,
                    true,
                    &reproduction,
                    "ok",
                ),
                check(
                    ConfirmatoryReadinessGate::ExternalPreregistration,
                    true,
                    &preregistration,
                    "ok",
                ),
            ],
            decision: ConfirmatoryReadinessDecision::ReadyForConfirmatoryCollection,
            decided_at_utc: "2026-07-14T00:00:00Z".into(),
            decision_authority: "independent-chair".into(),
            external_receipt_uri: "https://example.invalid/readiness".into(),
            external_receipt_sha256: "b".repeat(64),
            report_sha256: String::new(),
        };
        report.checks.sort_by_key(|entry| entry.gate);
        report.report_sha256 = confirmatory_readiness_report_commitment(&report).unwrap();
        report
    }

    #[test]
    fn complete_ready_report_validates() {
        assert!(validate_confirmatory_readiness_report(&valid_report()).is_empty());
    }

    #[test]
    fn ready_report_cannot_point_a_gate_at_different_evidence() {
        let mut report = valid_report();
        report
            .checks
            .iter_mut()
            .find(|entry| entry.gate == ConfirmatoryReadinessGate::WorkspaceValidation)
            .unwrap()
            .evidence_sha256 = "f".repeat(64);
        report.report_sha256 = confirmatory_readiness_report_commitment(&report).unwrap();
        assert!(
            validate_confirmatory_readiness_report(&report)
                .iter()
                .any(|issue| {
                    matches!(
                        issue,
                        ConfirmatoryReadinessIssue::GateEvidenceMismatch {
                            gate: ConfirmatoryReadinessGate::WorkspaceValidation
                        }
                    )
                })
        );
    }

    #[test]
    fn failed_blocking_gate_forces_not_ready() {
        let mut report = valid_report();
        report
            .checks
            .iter_mut()
            .find(|entry| entry.gate == ConfirmatoryReadinessGate::EndToEndDryRun)
            .unwrap()
            .passed = false;
        report.report_sha256 = confirmatory_readiness_report_commitment(&report).unwrap();
        assert!(
            validate_confirmatory_readiness_report(&report)
                .iter()
                .any(|issue| {
                    matches!(
                        issue,
                        ConfirmatoryReadinessIssue::ReadyDecisionWithFailedGate {
                            gate: ConfirmatoryReadinessGate::EndToEndDryRun
                        }
                    )
                })
        );
    }
}

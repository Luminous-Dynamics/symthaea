// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-schema migration integrity.
//!
//! Assurance evidence may outlive the software that produced it. Migrations
//! therefore need explicit source/target identities, deterministic replay,
//! record accounting, lossy-field declarations, and rollback evidence.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MigrationValidationKind {
    SchemaValidation,
    RecordCount,
    DigestContinuity,
    SemanticEquivalence,
    DeterministicReplay,
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationValidationStatus {
    Passed,
    Failed,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationValidationEvidence {
    pub kind: MigrationValidationKind,
    pub status: MigrationValidationStatus,
    pub evidence_id: String,
    pub artifact_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceMigrationStep {
    pub step_id: String,
    pub from_schema: String,
    pub to_schema: String,
    pub transform_digest: String,
    pub deterministic: bool,
    pub reversible: bool,
    pub declared_lossy_fields: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceMigrationRun {
    pub migration_id: String,
    pub source_schema: String,
    pub target_schema: String,
    pub source_digest: String,
    pub target_digest: String,
    pub source_record_count: u64,
    pub target_record_count: u64,
    pub steps: Vec<EvidenceMigrationStep>,
    pub acknowledged_lossy_fields: BTreeSet<String>,
    pub validations: Vec<MigrationValidationEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceMigrationPolicy {
    pub permitted_schema_edges: BTreeSet<(String, String)>,
    pub required_validations: BTreeSet<MigrationValidationKind>,
    pub require_reversible: bool,
    pub require_record_count_preservation: bool,
    pub require_artifact_digest_for: BTreeSet<MigrationValidationKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceMigrationStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceMigrationIssue {
    EmptyIdentity,
    MissingStep,
    DuplicateStep(String),
    BrokenStepChain { expected: String, observed: String },
    DisallowedSchemaEdge { from: String, to: String },
    NonDeterministicStep(String),
    IrreversibleStep(String),
    UnacknowledgedLossyField { step_id: String, field: String },
    RecordCountMismatch { source: u64, target: u64 },
    DuplicateValidation(MigrationValidationKind),
    MissingValidation(MigrationValidationKind),
    FailedValidation(MigrationValidationKind),
    MissingValidationEvidence(MigrationValidationKind),
    MissingValidationDigest(MigrationValidationKind),
    SourceTargetDigestEqual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceMigrationReport {
    pub status: EvidenceMigrationStatus,
    pub migration_id: String,
    pub validated_kinds: Vec<MigrationValidationKind>,
    pub lossy_fields: Vec<String>,
    pub issues: Vec<EvidenceMigrationIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMigrationError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct EvidenceMigrationGate {
    policy: EvidenceMigrationPolicy,
}

impl EvidenceMigrationGate {
    pub fn new(policy: EvidenceMigrationPolicy) -> Result<Self, EvidenceMigrationError> {
        if policy.permitted_schema_edges.is_empty() || policy.required_validations.is_empty() {
            return Err(EvidenceMigrationError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, run: &EvidenceMigrationRun) -> EvidenceMigrationReport {
        let mut issues = Vec::new();
        if [
            run.migration_id.as_str(),
            run.source_schema.as_str(),
            run.target_schema.as_str(),
            run.source_digest.as_str(),
            run.target_digest.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
        {
            issues.push(EvidenceMigrationIssue::EmptyIdentity);
        }
        if run.steps.is_empty() {
            issues.push(EvidenceMigrationIssue::MissingStep);
        }
        if run.source_digest == run.target_digest && run.source_schema != run.target_schema {
            issues.push(EvidenceMigrationIssue::SourceTargetDigestEqual);
        }
        if self.policy.require_record_count_preservation
            && run.source_record_count != run.target_record_count
        {
            issues.push(EvidenceMigrationIssue::RecordCountMismatch {
                source: run.source_record_count,
                target: run.target_record_count,
            });
        }

        let mut step_ids = BTreeSet::new();
        let mut expected_from = run.source_schema.as_str();
        let mut lossy_fields = BTreeSet::new();
        for step in &run.steps {
            if step.step_id.trim().is_empty()
                || step.from_schema.trim().is_empty()
                || step.to_schema.trim().is_empty()
                || step.transform_digest.trim().is_empty()
            {
                issues.push(EvidenceMigrationIssue::EmptyIdentity);
            }
            if !step_ids.insert(step.step_id.as_str()) {
                issues.push(EvidenceMigrationIssue::DuplicateStep(step.step_id.clone()));
            }
            if step.from_schema != expected_from {
                issues.push(EvidenceMigrationIssue::BrokenStepChain {
                    expected: expected_from.to_string(),
                    observed: step.from_schema.clone(),
                });
            }
            if !self
                .policy
                .permitted_schema_edges
                .contains(&(step.from_schema.clone(), step.to_schema.clone()))
            {
                issues.push(EvidenceMigrationIssue::DisallowedSchemaEdge {
                    from: step.from_schema.clone(),
                    to: step.to_schema.clone(),
                });
            }
            if !step.deterministic {
                issues.push(EvidenceMigrationIssue::NonDeterministicStep(
                    step.step_id.clone(),
                ));
            }
            if self.policy.require_reversible && !step.reversible {
                issues.push(EvidenceMigrationIssue::IrreversibleStep(
                    step.step_id.clone(),
                ));
            }
            for field in &step.declared_lossy_fields {
                lossy_fields.insert(field.clone());
                if !run.acknowledged_lossy_fields.contains(field) {
                    issues.push(EvidenceMigrationIssue::UnacknowledgedLossyField {
                        step_id: step.step_id.clone(),
                        field: field.clone(),
                    });
                }
            }
            expected_from = step.to_schema.as_str();
        }
        if !run.steps.is_empty() && expected_from != run.target_schema {
            issues.push(EvidenceMigrationIssue::BrokenStepChain {
                expected: run.target_schema.clone(),
                observed: expected_from.to_string(),
            });
        }

        let mut validations = BTreeMap::new();
        let mut validated_kinds = Vec::new();
        for validation in &run.validations {
            if validations.insert(validation.kind, validation).is_some() {
                issues.push(EvidenceMigrationIssue::DuplicateValidation(validation.kind));
            }
            match validation.status {
                MigrationValidationStatus::Passed => {
                    validated_kinds.push(validation.kind);
                    if validation.evidence_id.trim().is_empty() {
                        issues.push(EvidenceMigrationIssue::MissingValidationEvidence(
                            validation.kind,
                        ));
                    }
                    if self
                        .policy
                        .require_artifact_digest_for
                        .contains(&validation.kind)
                        && validation
                            .artifact_digest
                            .as_ref()
                            .is_none_or(|digest| digest.trim().is_empty())
                    {
                        issues.push(EvidenceMigrationIssue::MissingValidationDigest(
                            validation.kind,
                        ));
                    }
                }
                MigrationValidationStatus::Failed => {
                    issues.push(EvidenceMigrationIssue::FailedValidation(validation.kind));
                }
                MigrationValidationStatus::Missing => {
                    issues.push(EvidenceMigrationIssue::MissingValidation(validation.kind));
                }
            }
        }
        for required in &self.policy.required_validations {
            if !validations.contains_key(required) {
                issues.push(EvidenceMigrationIssue::MissingValidation(*required));
            }
        }
        validated_kinds.sort();
        validated_kinds.dedup();

        let status = if issues.iter().any(is_failure) {
            EvidenceMigrationStatus::Fail
        } else if issues.is_empty() {
            EvidenceMigrationStatus::Pass
        } else {
            EvidenceMigrationStatus::Incomplete
        };
        EvidenceMigrationReport {
            status,
            migration_id: run.migration_id.clone(),
            validated_kinds,
            lossy_fields: lossy_fields.into_iter().collect(),
            issues,
        }
    }
}

fn is_failure(issue: &EvidenceMigrationIssue) -> bool {
    matches!(
        issue,
        EvidenceMigrationIssue::BrokenStepChain { .. }
            | EvidenceMigrationIssue::DisallowedSchemaEdge { .. }
            | EvidenceMigrationIssue::NonDeterministicStep(_)
            | EvidenceMigrationIssue::IrreversibleStep(_)
            | EvidenceMigrationIssue::UnacknowledgedLossyField { .. }
            | EvidenceMigrationIssue::RecordCountMismatch { .. }
            | EvidenceMigrationIssue::FailedValidation(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> EvidenceMigrationPolicy {
        EvidenceMigrationPolicy {
            permitted_schema_edges: BTreeSet::from([("v1".into(), "v2".into())]),
            required_validations: BTreeSet::from([
                MigrationValidationKind::SchemaValidation,
                MigrationValidationKind::RecordCount,
                MigrationValidationKind::DeterministicReplay,
                MigrationValidationKind::Rollback,
            ]),
            require_reversible: true,
            require_record_count_preservation: true,
            require_artifact_digest_for: BTreeSet::from([
                MigrationValidationKind::DeterministicReplay,
                MigrationValidationKind::Rollback,
            ]),
        }
    }

    fn run() -> EvidenceMigrationRun {
        let validations = policy()
            .required_validations
            .iter()
            .copied()
            .map(|kind| MigrationValidationEvidence {
                kind,
                status: MigrationValidationStatus::Passed,
                evidence_id: format!("evidence-{kind:?}"),
                artifact_digest: Some(format!("sha256:{kind:?}")),
            })
            .collect();
        EvidenceMigrationRun {
            migration_id: "migration-1".into(),
            source_schema: "v1".into(),
            target_schema: "v2".into(),
            source_digest: "sha256:source".into(),
            target_digest: "sha256:target".into(),
            source_record_count: 100,
            target_record_count: 100,
            steps: vec![EvidenceMigrationStep {
                step_id: "step-1".into(),
                from_schema: "v1".into(),
                to_schema: "v2".into(),
                transform_digest: "sha256:transform".into(),
                deterministic: true,
                reversible: true,
                declared_lossy_fields: BTreeSet::new(),
            }],
            acknowledged_lossy_fields: BTreeSet::new(),
            validations,
        }
    }

    #[test]
    fn complete_reversible_migration_passes() {
        let report = EvidenceMigrationGate::new(policy()).unwrap().assess(&run());
        assert_eq!(report.status, EvidenceMigrationStatus::Pass);
    }

    #[test]
    fn silent_loss_fails() {
        let mut run = run();
        run.steps[0]
            .declared_lossy_fields
            .insert("event_order".into());
        let report = EvidenceMigrationGate::new(policy()).unwrap().assess(&run);
        assert_eq!(report.status, EvidenceMigrationStatus::Fail);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Immutable study execution and deterministic protocol-conformance auditing.
//!
//! Protocols describe what was planned. Executions describe what was done.
//! Audits compare the two without mutating either object. Declared timestamps
//! remain declarations until a later provenance/qualification layer authenticates
//! them.

use crate::{
    ControlKind, FramedDigest, FrozenStudyProtocol, MultiplicityPolicy, OutcomeRole, ResearchId,
    Sha256Digest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const STUDY_EXECUTION_SCHEMA: &str = "symthaea.study-execution.v1";
const EXECUTION_DOMAIN: &str = "symthaea.study-execution.identity.v1";
const AUDIT_DOMAIN: &str = "symthaea.study-execution-audit.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviationCategory {
    Registration,
    DataAccess,
    Sampling,
    Exclusion,
    Stopping,
    Analysis,
    EstimatorOrTest,
    Code,
    Environment,
    Multiplicity,
    Outcome,
    Control,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeclaredDeviationSeverity {
    NonMaterial,
    Material,
}

/// Caller-declared metadata is retained for accountability, but conformance is
/// derived independently from the frozen protocol and actual execution fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredDeviation {
    pub deviation_id: ResearchId,
    pub category: DeviationCategory,
    pub severity: DeclaredDeviationSeverity,
    pub disclosed: bool,
    pub detail_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExecutedOutcome {
    pub outcome_id: ResearchId,
    pub artifact_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExecutedControl {
    pub control_id: ResearchId,
    pub artifact_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyExecution {
    pub schema_version: String,
    pub execution_id: ResearchId,
    pub protocol_sha256: Sha256Digest,
    pub subject_sha256: Sha256Digest,
    pub first_protected_observation_access_unix_ms: u64,
    pub collection_started_unix_ms: u64,
    pub collection_ended_unix_ms: u64,
    pub analysis_started_unix_ms: u64,
    pub analysis_ended_unix_ms: u64,
    pub data_snapshot_sha256: Sha256Digest,
    pub units_started: u64,
    pub units_completed: u64,
    pub units_excluded_after_observation: u64,
    pub recruitment_or_generation_sha256: Sha256Digest,
    pub exclusion_rule_sha256: Sha256Digest,
    pub stopping_rule_sha256: Sha256Digest,
    pub analysis_plan_sha256: Sha256Digest,
    pub estimator_or_test_sha256: Sha256Digest,
    pub code_sha256: Sha256Digest,
    pub environment_sha256: Sha256Digest,
    pub multiplicity_policy: MultiplicityPolicy,
    pub multiplicity_policy_sha256: Option<Sha256Digest>,
    pub outcomes: Vec<ExecutedOutcome>,
    pub controls: Vec<ExecutedControl>,
    pub declared_deviations: Vec<DeclaredDeviation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionIssue {
    WrongSchemaVersion { found: String },
    ProtocolBindingMismatch,
    SubjectBindingMismatch,
    InvalidCollectionWindow,
    InvalidAnalysisWindow,
    ProtectedAccessAfterCollectionEnd,
    AnalysisStartsBeforeProtectedAccess,
    ZeroStartedUnits,
    CompletedExceedsStarted,
    ExclusionsExceedCompleted,
    DuplicateOutcome { outcome_id: ResearchId },
    DuplicateControl { control_id: ResearchId },
    DuplicateDeclaredDeviation { deviation_id: ResearchId },
    MissingDomainSpecificMultiplicityCommitment,
}

impl StudyExecution {
    pub fn validate_against(&self, protocol: &FrozenStudyProtocol) -> Vec<ExecutionIssue> {
        let mut issues = Vec::new();
        if self.schema_version != STUDY_EXECUTION_SCHEMA {
            issues.push(ExecutionIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if &self.protocol_sha256 != protocol.protocol_sha256() {
            issues.push(ExecutionIssue::ProtocolBindingMismatch);
        }
        if self.subject_sha256 != protocol.protocol().subject_sha256 {
            issues.push(ExecutionIssue::SubjectBindingMismatch);
        }
        if self.collection_ended_unix_ms <= self.collection_started_unix_ms {
            issues.push(ExecutionIssue::InvalidCollectionWindow);
        }
        if self.analysis_ended_unix_ms <= self.analysis_started_unix_ms {
            issues.push(ExecutionIssue::InvalidAnalysisWindow);
        }
        if self.first_protected_observation_access_unix_ms > self.collection_ended_unix_ms {
            issues.push(ExecutionIssue::ProtectedAccessAfterCollectionEnd);
        }
        if self.analysis_started_unix_ms < self.first_protected_observation_access_unix_ms {
            issues.push(ExecutionIssue::AnalysisStartsBeforeProtectedAccess);
        }
        if self.units_started == 0 {
            issues.push(ExecutionIssue::ZeroStartedUnits);
        }
        if self.units_completed > self.units_started {
            issues.push(ExecutionIssue::CompletedExceedsStarted);
        }
        if self.units_excluded_after_observation > self.units_completed {
            issues.push(ExecutionIssue::ExclusionsExceedCompleted);
        }
        duplicate_ids(
            self.outcomes.iter().map(|item| &item.outcome_id),
            |id| ExecutionIssue::DuplicateOutcome { outcome_id: id },
            &mut issues,
        );
        duplicate_ids(
            self.controls.iter().map(|item| &item.control_id),
            |id| ExecutionIssue::DuplicateControl { control_id: id },
            &mut issues,
        );
        duplicate_ids(
            self.declared_deviations.iter().map(|item| &item.deviation_id),
            |id| ExecutionIssue::DuplicateDeclaredDeviation { deviation_id: id },
            &mut issues,
        );
        if self.multiplicity_policy == MultiplicityPolicy::DomainSpecific
            && self.multiplicity_policy_sha256.is_none()
        {
            issues.push(ExecutionIssue::MissingDomainSpecificMultiplicityCommitment);
        }
        issues
    }

    pub fn freeze_against(
        self,
        protocol: &FrozenStudyProtocol,
    ) -> Result<FrozenStudyExecution, Vec<ExecutionIssue>> {
        let issues = self.validate_against(protocol);
        if !issues.is_empty() {
            return Err(issues);
        }
        let execution_sha256 = self.compute_digest();
        Ok(FrozenStudyExecution {
            execution: self,
            execution_sha256,
        })
    }

    fn compute_digest(&self) -> Sha256Digest {
        let mut digest = FramedDigest::new(EXECUTION_DOMAIN);
        digest.text(STUDY_EXECUTION_SCHEMA);
        digest.text(self.execution_id.as_str());
        digest.text(self.protocol_sha256.as_str());
        digest.text(self.subject_sha256.as_str());
        for value in [
            self.first_protected_observation_access_unix_ms,
            self.collection_started_unix_ms,
            self.collection_ended_unix_ms,
            self.analysis_started_unix_ms,
            self.analysis_ended_unix_ms,
            self.units_started,
            self.units_completed,
            self.units_excluded_after_observation,
        ] {
            digest.text(&value.to_string());
        }
        for value in [
            &self.data_snapshot_sha256,
            &self.recruitment_or_generation_sha256,
            &self.exclusion_rule_sha256,
            &self.stopping_rule_sha256,
            &self.analysis_plan_sha256,
            &self.estimator_or_test_sha256,
            &self.code_sha256,
            &self.environment_sha256,
        ] {
            digest.text(value.as_str());
        }
        digest.text(multiplicity_policy_tag(self.multiplicity_policy));
        digest_optional_sha(&mut digest, self.multiplicity_policy_sha256.as_ref());

        let mut outcomes = self.outcomes.clone();
        outcomes.sort();
        for item in outcomes {
            digest.text("outcome");
            digest.text(item.outcome_id.as_str());
            digest.text(item.artifact_sha256.as_str());
        }
        let mut controls = self.controls.clone();
        controls.sort();
        for item in controls {
            digest.text("control");
            digest.text(item.control_id.as_str());
            digest.text(item.artifact_sha256.as_str());
        }
        let mut deviations = self.declared_deviations.clone();
        deviations.sort_by(|left, right| left.deviation_id.cmp(&right.deviation_id));
        for item in deviations {
            digest.text("declared-deviation");
            digest.text(item.deviation_id.as_str());
            digest.text(deviation_category_tag(item.category));
            digest.text(deviation_severity_tag(item.severity));
            digest.text(if item.disclosed { "disclosed" } else { "undisclosed" });
            digest.text(item.detail_sha256.as_str());
        }
        digest.digest()
    }
}

fn duplicate_ids<'a, I, F>(ids: I, issue: F, issues: &mut Vec<ExecutionIssue>)
where
    I: IntoIterator<Item = &'a ResearchId>,
    F: Fn(ResearchId) -> ExecutionIssue,
{
    let mut seen = BTreeSet::new();
    for id in ids {
        if !seen.insert(id.clone()) {
            issues.push(issue(id.clone()));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenStudyExecution {
    execution: StudyExecution,
    execution_sha256: Sha256Digest,
}

impl FrozenStudyExecution {
    pub fn execution(&self) -> &StudyExecution {
        &self.execution
    }
    pub fn execution_sha256(&self) -> &Sha256Digest {
        &self.execution_sha256
    }
    pub fn audit(
        &self,
        protocol: &FrozenStudyProtocol,
    ) -> Result<ExecutionAudit, Vec<ExecutionIssue>> {
        let issues = self.execution.validate_against(protocol);
        if !issues.is_empty() {
            return Err(issues);
        }
        Ok(ExecutionAudit::derive(protocol, self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FindingSeverity {
    NonMaterialDeviation,
    MaterialDeviation,
    AuthorityBlocking,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExecutionFindingCode {
    DeclaredRegistrationTiming,
    ObservationStartedBeforePlannedStart,
    SamplingCommitmentMismatch,
    TargetUnitMismatch,
    ExclusionRuleMismatch,
    StoppingRuleMismatch,
    AnalysisPlanMismatch,
    EstimatorOrTestMismatch,
    CodeMismatch,
    CodeWasUncommitted,
    EnvironmentMismatch,
    EnvironmentWasUncommitted,
    MultiplicityMismatch,
    MultiplicityCommitmentMismatch,
    MissingPrimaryOutcome { outcome_id: ResearchId },
    UnplannedOutcome { outcome_id: ResearchId },
    MissingPlannedControl { control_id: ResearchId, kind: ControlKind },
    UnplannedControl { control_id: ResearchId },
    DeclaredMaterialDeviation { deviation_id: ResearchId },
    UndisclosedDeclaredDeviation { deviation_id: ResearchId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionFinding {
    pub code: ExecutionFindingCode,
    pub severity: FindingSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExecutionConformance {
    Exact,
    WithNonMaterialDeviations,
    WithMaterialDeviations,
    AuthorityBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionAudit {
    protocol_sha256: Sha256Digest,
    execution_sha256: Sha256Digest,
    conformance: ExecutionConformance,
    findings: Vec<ExecutionFinding>,
    audit_sha256: Sha256Digest,
}

impl ExecutionAudit {
    fn derive(protocol: &FrozenStudyProtocol, execution: &FrozenStudyExecution) -> Self {
        let plan = protocol.protocol();
        let actual = execution.execution();
        let mut findings = Vec::new();

        if !matches!(
            plan.declared_registration_timing(actual.first_protected_observation_access_unix_ms),
            crate::RegistrationTiming::ProspectiveDeclared
        ) {
            findings.push(blocking(ExecutionFindingCode::DeclaredRegistrationTiming));
        }
        if plan
            .planned_protected_observation_start_unix_ms
            .is_some_and(|planned| actual.first_protected_observation_access_unix_ms < planned)
        {
            findings.push(material(
                ExecutionFindingCode::ObservationStartedBeforePlannedStart,
            ));
        }
        if actual.recruitment_or_generation_sha256 != plan.sampling.recruitment_or_generation_sha256 {
            findings.push(material(ExecutionFindingCode::SamplingCommitmentMismatch));
        }
        if plan
            .sampling
            .target_units
            .is_some_and(|target| actual.units_started != target)
        {
            findings.push(material(ExecutionFindingCode::TargetUnitMismatch));
        }
        if actual.exclusion_rule_sha256 != plan.sampling.exclusion_rule_sha256 {
            findings.push(material(ExecutionFindingCode::ExclusionRuleMismatch));
        }
        if actual.stopping_rule_sha256 != plan.sampling.stopping_rule_sha256 {
            findings.push(material(ExecutionFindingCode::StoppingRuleMismatch));
        }
        if actual.analysis_plan_sha256 != plan.analysis.analysis_plan_sha256 {
            findings.push(blocking(ExecutionFindingCode::AnalysisPlanMismatch));
        }
        if actual.estimator_or_test_sha256 != plan.analysis.estimator_or_test_sha256 {
            findings.push(blocking(ExecutionFindingCode::EstimatorOrTestMismatch));
        }
        match &plan.analysis.code_sha256 {
            Some(planned) if planned != &actual.code_sha256 => {
                findings.push(blocking(ExecutionFindingCode::CodeMismatch));
            }
            None => findings.push(non_material(ExecutionFindingCode::CodeWasUncommitted)),
            Some(_) => {}
        }
        match &plan.analysis.environment_sha256 {
            Some(planned) if planned != &actual.environment_sha256 => {
                findings.push(blocking(ExecutionFindingCode::EnvironmentMismatch));
            }
            None => findings.push(non_material(ExecutionFindingCode::EnvironmentWasUncommitted)),
            Some(_) => {}
        }
        if actual.multiplicity_policy != plan.analysis.multiplicity_policy {
            findings.push(blocking(ExecutionFindingCode::MultiplicityMismatch));
        }
        if actual.multiplicity_policy_sha256 != plan.analysis.multiplicity_policy_sha256 {
            findings.push(blocking(
                ExecutionFindingCode::MultiplicityCommitmentMismatch,
            ));
        }

        let actual_outcomes = actual
            .outcomes
            .iter()
            .map(|item| item.outcome_id.clone())
            .collect::<BTreeSet<_>>();
        let planned_outcomes = plan
            .outcomes
            .iter()
            .map(|item| item.outcome_id.clone())
            .collect::<BTreeSet<_>>();
        for outcome in plan
            .outcomes
            .iter()
            .filter(|item| item.role == OutcomeRole::Primary)
        {
            if !actual_outcomes.contains(&outcome.outcome_id) {
                findings.push(blocking(ExecutionFindingCode::MissingPrimaryOutcome {
                    outcome_id: outcome.outcome_id.clone(),
                }));
            }
        }
        for outcome_id in actual_outcomes.difference(&planned_outcomes) {
            findings.push(non_material(ExecutionFindingCode::UnplannedOutcome {
                outcome_id: outcome_id.clone(),
            }));
        }

        let actual_controls = actual
            .controls
            .iter()
            .map(|item| item.control_id.clone())
            .collect::<BTreeSet<_>>();
        let planned_controls = plan
            .controls
            .iter()
            .map(|item| item.control_id.clone())
            .collect::<BTreeSet<_>>();
        for control in &plan.controls {
            if !actual_controls.contains(&control.control_id) {
                findings.push(material(ExecutionFindingCode::MissingPlannedControl {
                    control_id: control.control_id.clone(),
                    kind: control.kind,
                }));
            }
        }
        for control_id in actual_controls.difference(&planned_controls) {
            findings.push(non_material(ExecutionFindingCode::UnplannedControl {
                control_id: control_id.clone(),
            }));
        }
        for deviation in &actual.declared_deviations {
            if deviation.severity == DeclaredDeviationSeverity::Material {
                findings.push(material(ExecutionFindingCode::DeclaredMaterialDeviation {
                    deviation_id: deviation.deviation_id.clone(),
                }));
            }
            if !deviation.disclosed {
                findings.push(blocking(
                    ExecutionFindingCode::UndisclosedDeclaredDeviation {
                        deviation_id: deviation.deviation_id.clone(),
                    },
                ));
            }
        }

        findings.sort_by(|left, right| {
            left.code
                .cmp(&right.code)
                .then(left.severity.cmp(&right.severity))
        });
        findings.dedup();
        let conformance = classify_conformance(&findings);
        let audit_sha256 = audit_digest(
            protocol.protocol_sha256(),
            execution.execution_sha256(),
            conformance,
            &findings,
        );
        Self {
            protocol_sha256: protocol.protocol_sha256().clone(),
            execution_sha256: execution.execution_sha256().clone(),
            conformance,
            findings,
            audit_sha256,
        }
    }

    pub fn protocol_sha256(&self) -> &Sha256Digest {
        &self.protocol_sha256
    }
    pub fn execution_sha256(&self) -> &Sha256Digest {
        &self.execution_sha256
    }
    pub fn conformance(&self) -> ExecutionConformance {
        self.conformance
    }
    pub fn findings(&self) -> &[ExecutionFinding] {
        &self.findings
    }
    pub fn audit_sha256(&self) -> &Sha256Digest {
        &self.audit_sha256
    }
}

fn classify_conformance(findings: &[ExecutionFinding]) -> ExecutionConformance {
    if findings
        .iter()
        .any(|item| item.severity == FindingSeverity::AuthorityBlocking)
    {
        ExecutionConformance::AuthorityBlocked
    } else if findings
        .iter()
        .any(|item| item.severity == FindingSeverity::MaterialDeviation)
    {
        ExecutionConformance::WithMaterialDeviations
    } else if findings
        .iter()
        .any(|item| item.severity == FindingSeverity::NonMaterialDeviation)
    {
        ExecutionConformance::WithNonMaterialDeviations
    } else {
        ExecutionConformance::Exact
    }
}

fn material(code: ExecutionFindingCode) -> ExecutionFinding {
    ExecutionFinding {
        code,
        severity: FindingSeverity::MaterialDeviation,
    }
}
fn blocking(code: ExecutionFindingCode) -> ExecutionFinding {
    ExecutionFinding {
        code,
        severity: FindingSeverity::AuthorityBlocking,
    }
}
fn non_material(code: ExecutionFindingCode) -> ExecutionFinding {
    ExecutionFinding {
        code,
        severity: FindingSeverity::NonMaterialDeviation,
    }
}

fn audit_digest(
    protocol_sha256: &Sha256Digest,
    execution_sha256: &Sha256Digest,
    conformance: ExecutionConformance,
    findings: &[ExecutionFinding],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(AUDIT_DOMAIN);
    digest.text(protocol_sha256.as_str());
    digest.text(execution_sha256.as_str());
    digest.text(conformance_tag(conformance));
    for finding in findings {
        digest.text(finding_code_tag(&finding.code));
        digest.text(finding_severity_tag(finding.severity));
        match &finding.code {
            ExecutionFindingCode::MissingPrimaryOutcome { outcome_id }
            | ExecutionFindingCode::UnplannedOutcome { outcome_id } => digest.text(outcome_id.as_str()),
            ExecutionFindingCode::MissingPlannedControl { control_id, kind } => {
                digest.text(control_id.as_str());
                digest.text(control_kind_tag(*kind));
            }
            ExecutionFindingCode::UnplannedControl { control_id } => digest.text(control_id.as_str()),
            ExecutionFindingCode::DeclaredMaterialDeviation { deviation_id }
            | ExecutionFindingCode::UndisclosedDeclaredDeviation { deviation_id } => {
                digest.text(deviation_id.as_str());
            }
            _ => {}
        }
    }
    digest.digest()
}

fn digest_optional_sha(digest: &mut FramedDigest, value: Option<&Sha256Digest>) {
    match value {
        Some(value) => {
            digest.text("some");
            digest.text(value.as_str());
        }
        None => digest.text("none"),
    }
}

const fn multiplicity_policy_tag(value: MultiplicityPolicy) -> &'static str {
    match value {
        MultiplicityPolicy::NotApplicable => "not-applicable",
        MultiplicityPolicy::NoneDeclared => "none-declared",
        MultiplicityPolicy::Bonferroni => "bonferroni",
        MultiplicityPolicy::Holm => "holm",
        MultiplicityPolicy::FalseDiscoveryRate => "false-discovery-rate",
        MultiplicityPolicy::Hierarchical => "hierarchical",
        MultiplicityPolicy::BayesianMultilevel => "bayesian-multilevel",
        MultiplicityPolicy::DomainSpecific => "domain-specific",
    }
}

const fn control_kind_tag(value: ControlKind) -> &'static str {
    match value {
        ControlKind::Positive => "positive",
        ControlKind::Negative => "negative",
        ControlKind::Placebo => "placebo",
        ControlKind::Sham => "sham",
        ControlKind::Shuffled => "shuffled",
        ControlKind::Baseline => "baseline",
        ControlKind::ActiveComparator => "active-comparator",
        ControlKind::Ablation => "ablation",
        ControlKind::NullModel => "null-model",
        ControlKind::Other => "other",
    }
}

const fn deviation_category_tag(value: DeviationCategory) -> &'static str {
    match value {
        DeviationCategory::Registration => "registration",
        DeviationCategory::DataAccess => "data-access",
        DeviationCategory::Sampling => "sampling",
        DeviationCategory::Exclusion => "exclusion",
        DeviationCategory::Stopping => "stopping",
        DeviationCategory::Analysis => "analysis",
        DeviationCategory::EstimatorOrTest => "estimator-or-test",
        DeviationCategory::Code => "code",
        DeviationCategory::Environment => "environment",
        DeviationCategory::Multiplicity => "multiplicity",
        DeviationCategory::Outcome => "outcome",
        DeviationCategory::Control => "control",
        DeviationCategory::Other => "other",
    }
}

const fn deviation_severity_tag(value: DeclaredDeviationSeverity) -> &'static str {
    match value {
        DeclaredDeviationSeverity::NonMaterial => "non-material",
        DeclaredDeviationSeverity::Material => "material",
    }
}

const fn finding_severity_tag(value: FindingSeverity) -> &'static str {
    match value {
        FindingSeverity::NonMaterialDeviation => "non-material-deviation",
        FindingSeverity::MaterialDeviation => "material-deviation",
        FindingSeverity::AuthorityBlocking => "authority-blocking",
    }
}

const fn conformance_tag(value: ExecutionConformance) -> &'static str {
    match value {
        ExecutionConformance::Exact => "exact",
        ExecutionConformance::WithNonMaterialDeviations => "with-non-material-deviations",
        ExecutionConformance::WithMaterialDeviations => "with-material-deviations",
        ExecutionConformance::AuthorityBlocked => "authority-blocked",
    }
}

const fn finding_code_tag(value: &ExecutionFindingCode) -> &'static str {
    match value {
        ExecutionFindingCode::DeclaredRegistrationTiming => "declared-registration-timing",
        ExecutionFindingCode::ObservationStartedBeforePlannedStart => "observation-started-before-planned-start",
        ExecutionFindingCode::SamplingCommitmentMismatch => "sampling-commitment-mismatch",
        ExecutionFindingCode::TargetUnitMismatch => "target-unit-mismatch",
        ExecutionFindingCode::ExclusionRuleMismatch => "exclusion-rule-mismatch",
        ExecutionFindingCode::StoppingRuleMismatch => "stopping-rule-mismatch",
        ExecutionFindingCode::AnalysisPlanMismatch => "analysis-plan-mismatch",
        ExecutionFindingCode::EstimatorOrTestMismatch => "estimator-or-test-mismatch",
        ExecutionFindingCode::CodeMismatch => "code-mismatch",
        ExecutionFindingCode::CodeWasUncommitted => "code-was-uncommitted",
        ExecutionFindingCode::EnvironmentMismatch => "environment-mismatch",
        ExecutionFindingCode::EnvironmentWasUncommitted => "environment-was-uncommitted",
        ExecutionFindingCode::MultiplicityMismatch => "multiplicity-mismatch",
        ExecutionFindingCode::MultiplicityCommitmentMismatch => "multiplicity-commitment-mismatch",
        ExecutionFindingCode::MissingPrimaryOutcome { .. } => "missing-primary-outcome",
        ExecutionFindingCode::UnplannedOutcome { .. } => "unplanned-outcome",
        ExecutionFindingCode::MissingPlannedControl { .. } => "missing-planned-control",
        ExecutionFindingCode::UnplannedControl { .. } => "unplanned-control",
        ExecutionFindingCode::DeclaredMaterialDeviation { .. } => "declared-material-deviation",
        ExecutionFindingCode::UndisclosedDeclaredDeviation { .. } => "undisclosed-declared-deviation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisPlan, ControlSpec, FalsifierSpec, HypothesisRole, HypothesisSpec, OutcomeSpec,
        PredictionDirection, PredictionSpec, SamplingPlan, StudyIntent, StudyProtocol,
        STUDY_PROTOCOL_SCHEMA,
    };

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }
    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn protocol() -> FrozenStudyProtocol {
        StudyProtocol {
            schema_version: STUDY_PROTOCOL_SCHEMA.into(),
            protocol_id: id("SCI-PROTOCOL-EXEC-001"),
            subject_sha256: sha("subject"),
            intent: StudyIntent::Confirmatory,
            registered_at_unix_ms: Some(100),
            registration_evidence_sha256: Some(sha("registration")),
            planned_protected_observation_start_unix_ms: Some(200),
            hypotheses: vec![HypothesisSpec {
                hypothesis_id: id("H-1"),
                role: HypothesisRole::Primary,
                statement_sha256: sha("h1"),
                null_statement_sha256: Some(sha("h1-null")),
                predictions: vec![PredictionSpec {
                    outcome_id: id("Y-1"),
                    direction: PredictionDirection::Increase,
                    quantitative_envelope_sha256: None,
                }],
            }],
            outcomes: vec![OutcomeSpec {
                outcome_id: id("Y-1"),
                role: OutcomeRole::Primary,
                measurement_sha256: sha("measurement"),
            }],
            controls: vec![ControlSpec {
                control_id: id("C-1"),
                kind: ControlKind::Negative,
                protocol_sha256: sha("control-plan"),
            }],
            falsifiers: vec![FalsifierSpec {
                falsifier_id: id("F-1"),
                hypothesis_id: id("H-1"),
                criterion_sha256: sha("falsifier"),
                required: true,
            }],
            analysis: AnalysisPlan {
                analysis_plan_sha256: sha("analysis"),
                estimator_or_test_sha256: sha("estimator"),
                code_sha256: Some(sha("code")),
                environment_sha256: Some(sha("env")),
                multiplicity_policy: MultiplicityPolicy::NotApplicable,
                multiplicity_policy_sha256: None,
            },
            sampling: SamplingPlan {
                target_units: Some(100),
                recruitment_or_generation_sha256: sha("sampling"),
                exclusion_rule_sha256: sha("exclusions"),
                stopping_rule_sha256: sha("stopping"),
            },
            deviation_policy_sha256: sha("deviation-policy"),
            supersedes_protocol_sha256: None,
        }
        .freeze()
        .unwrap()
    }

    fn execution(protocol: &FrozenStudyProtocol) -> StudyExecution {
        StudyExecution {
            schema_version: STUDY_EXECUTION_SCHEMA.into(),
            execution_id: id("SCI-EXEC-001"),
            protocol_sha256: protocol.protocol_sha256().clone(),
            subject_sha256: protocol.protocol().subject_sha256.clone(),
            first_protected_observation_access_unix_ms: 200,
            collection_started_unix_ms: 200,
            collection_ended_unix_ms: 300,
            analysis_started_unix_ms: 301,
            analysis_ended_unix_ms: 400,
            data_snapshot_sha256: sha("data"),
            units_started: 100,
            units_completed: 95,
            units_excluded_after_observation: 0,
            recruitment_or_generation_sha256: sha("sampling"),
            exclusion_rule_sha256: sha("exclusions"),
            stopping_rule_sha256: sha("stopping"),
            analysis_plan_sha256: sha("analysis"),
            estimator_or_test_sha256: sha("estimator"),
            code_sha256: sha("code"),
            environment_sha256: sha("env"),
            multiplicity_policy: MultiplicityPolicy::NotApplicable,
            multiplicity_policy_sha256: None,
            outcomes: vec![ExecutedOutcome {
                outcome_id: id("Y-1"),
                artifact_sha256: sha("y1-result"),
            }],
            controls: vec![ExecutedControl {
                control_id: id("C-1"),
                artifact_sha256: sha("c1-result"),
            }],
            declared_deviations: Vec::new(),
        }
    }

    #[test]
    fn exact_execution_is_exact() {
        let plan = protocol();
        let frozen = execution(&plan).freeze_against(&plan).unwrap();
        let audit = frozen.audit(&plan).unwrap();
        assert_eq!(audit.conformance(), ExecutionConformance::Exact);
        assert!(audit.findings().is_empty());
    }

    #[test]
    fn changed_analysis_is_authority_blocking() {
        let plan = protocol();
        let mut actual = execution(&plan);
        actual.analysis_plan_sha256 = sha("analysis-after-looking");
        let audit = actual.freeze_against(&plan).unwrap().audit(&plan).unwrap();
        assert_eq!(audit.conformance(), ExecutionConformance::AuthorityBlocked);
        assert!(audit.findings().iter().any(|finding| {
            matches!(&finding.code, ExecutionFindingCode::AnalysisPlanMismatch)
        }));
    }

    #[test]
    fn missing_primary_outcome_is_authority_blocking() {
        let plan = protocol();
        let mut actual = execution(&plan);
        actual.outcomes.clear();
        let audit = actual.freeze_against(&plan).unwrap().audit(&plan).unwrap();
        assert_eq!(audit.conformance(), ExecutionConformance::AuthorityBlocked);
    }

    #[test]
    fn unplanned_outcome_remains_non_primary() {
        let plan = protocol();
        let mut actual = execution(&plan);
        actual.outcomes.push(ExecutedOutcome {
            outcome_id: id("Y-EXPLORATORY"),
            artifact_sha256: sha("exploratory-result"),
        });
        let audit = actual.freeze_against(&plan).unwrap().audit(&plan).unwrap();
        assert_eq!(
            audit.conformance(),
            ExecutionConformance::WithNonMaterialDeviations
        );
        assert!(audit.findings().iter().any(|finding| {
            matches!(&finding.code, ExecutionFindingCode::UnplannedOutcome { .. })
        }));
    }

    #[test]
    fn omitting_deviation_declaration_cannot_hide_code_mismatch() {
        let plan = protocol();
        let mut actual = execution(&plan);
        actual.code_sha256 = sha("different-code");
        actual.declared_deviations.clear();
        let audit = actual.freeze_against(&plan).unwrap().audit(&plan).unwrap();
        assert_eq!(audit.conformance(), ExecutionConformance::AuthorityBlocked);
        assert!(audit.findings().iter().any(|finding| {
            matches!(&finding.code, ExecutionFindingCode::CodeMismatch)
        }));
    }

    #[test]
    fn wrong_protocol_binding_fails_before_audit() {
        let plan = protocol();
        let mut actual = execution(&plan);
        actual.protocol_sha256 = sha("wrong-protocol");
        assert!(actual
            .freeze_against(&plan)
            .unwrap_err()
            .contains(&ExecutionIssue::ProtocolBindingMismatch));
    }

    #[test]
    fn output_order_does_not_change_execution_identity() {
        let plan = protocol();
        let mut left = execution(&plan);
        left.outcomes.push(ExecutedOutcome {
            outcome_id: id("Y-EXTRA"),
            artifact_sha256: sha("extra"),
        });
        let mut right = left.clone();
        right.outcomes.reverse();
        assert_eq!(
            left.freeze_against(&plan).unwrap().execution_sha256(),
            right.freeze_against(&plan).unwrap().execution_sha256()
        );
    }
}

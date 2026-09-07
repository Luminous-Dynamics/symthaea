use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    evaluate_hypotheses, execute_preregistration, extract_blinded_metrics, BlindedMetricReport,
    ExecutionLimits, ExecutionTrace, ExperimentPreregistration, HypothesisEvaluationReport,
};

pub const STUDY_PREREGISTRATION_SCHEMA_VERSION: u16 = 1;
pub const STUDY_EXECUTION_SCHEMA_VERSION: u16 = 1;
pub const EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION: u16 = 1;
pub const STUDY_BLINDED_METRIC_SCHEMA_VERSION: u16 = 1;
pub const CONFIRMATORY_EVALUATION_SCHEMA_VERSION: u16 = 1;

/// Epistemic status fixed before a study is executed.
///
/// An exploratory study may inspect registered metrics, but only a confirmatory
/// study can pass the confirmatory-evaluation gate below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceRunClass {
    Exploratory,
    Confirmatory,
}

/// Study-level envelope around the mechanical experiment protocol.
///
/// Keeping this separate from `ExperimentPreregistration` avoids changing the
/// already-versioned native-regulation protocol merely to add inference policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyPreregistration {
    pub schema_version: u16,
    pub run_class: EvidenceRunClass,
    pub protocol: ExperimentPreregistration,
}

impl StudyPreregistration {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != STUDY_PREREGISTRATION_SCHEMA_VERSION {
            errors.push(format!(
                "study preregistration schema version mismatch: {}",
                self.schema_version
            ));
        }
        if let Err(protocol_errors) = self.protocol.validate() {
            errors.extend(
                protocol_errors
                    .into_iter()
                    .map(|error| format!("protocol: {error}")),
            );
        }
        if self.run_class == EvidenceRunClass::Confirmatory
            && !self.protocol.blind_arm_identity_during_primary_analysis
        {
            errors.push(
                "confirmatory studies must blind semantic arm identity during primary analysis"
                    .into(),
            );
        }
        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, Vec<String>> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| vec![format!("failed to serialize study preregistration: {error}")])
    }

    pub fn sha256(&self) -> Result<String, Vec<String>> {
        let bytes = self.canonical_json()?;
        Ok(hash_bytes(&bytes))
    }
}

/// Execution artifact that binds the mechanical trace to the study-level class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyExecutionTrace {
    pub schema_version: u16,
    pub run_class: EvidenceRunClass,
    pub study_preregistration_sha256: String,
    pub trace: ExecutionTrace,
}

pub fn execute_study(
    study: &StudyPreregistration,
    limits: ExecutionLimits,
) -> Result<StudyExecutionTrace, Vec<String>> {
    study.validate()?;
    let trace = execute_preregistration(&study.protocol, limits)?;
    Ok(StudyExecutionTrace {
        schema_version: STUDY_EXECUTION_SCHEMA_VERSION,
        run_class: study.run_class,
        study_preregistration_sha256: study.sha256()?,
        trace,
    })
}

impl StudyExecutionTrace {
    pub fn validation_errors_against(
        &self,
        study: &StudyPreregistration,
        limits: ExecutionLimits,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != STUDY_EXECUTION_SCHEMA_VERSION {
            errors.push(format!(
                "study execution schema version mismatch: {}",
                self.schema_version
            ));
        }
        if self.run_class != study.run_class {
            errors.push("study execution run class does not match preregistration".into());
        }
        match study.sha256() {
            Ok(expected) if expected == self.study_preregistration_sha256 => {}
            Ok(_) => errors.push("study preregistration digest mismatch in execution trace".into()),
            Err(study_errors) => errors.extend(study_errors),
        }
        if let Err(trace_errors) = self.trace.validate_against(&study.protocol, limits) {
            errors.extend(
                trace_errors
                    .into_iter()
                    .map(|error| format!("execution trace: {error}")),
            );
        }
        errors
    }

    pub fn validate_against(
        &self,
        study: &StudyPreregistration,
        limits: ExecutionLimits,
    ) -> Result<(), Vec<String>> {
        let errors = self.validation_errors_against(study, limits);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn sha256(&self) -> Result<String, String> {
        hash_json(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExclusionDecisionStatus {
    NotTriggered,
    Triggered,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExclusionCriterionDecision {
    pub criterion_id: String,
    pub status: ExclusionDecisionStatus,
    /// Digest of the evidence used to make this criterion decision.
    /// Required even for `NotTriggered`, so inclusion also leaves an audit trail.
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunDisposition {
    Include,
    Exclude,
    Indeterminate,
}

/// Immutable analyst decision record for every preregistered exclusion criterion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExclusionDecisionReceipt {
    pub schema_version: u16,
    pub run_class: EvidenceRunClass,
    pub study_preregistration_sha256: String,
    pub study_execution_sha256: String,
    pub decisions: Vec<ExclusionCriterionDecision>,
}

impl ExclusionDecisionReceipt {
    pub fn validation_errors_against(
        &self,
        study: &StudyPreregistration,
        execution: &StudyExecutionTrace,
        limits: ExecutionLimits,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION {
            errors.push(format!(
                "exclusion decision receipt schema version mismatch: {}",
                self.schema_version
            ));
        }
        if self.run_class != study.run_class {
            errors.push("exclusion receipt run class does not match study".into());
        }
        if let Err(execution_errors) = execution.validate_against(study, limits) {
            errors.extend(
                execution_errors
                    .into_iter()
                    .map(|error| format!("study execution: {error}")),
            );
        }
        match study.sha256() {
            Ok(expected) if expected == self.study_preregistration_sha256 => {}
            Ok(_) => errors.push("exclusion receipt study digest mismatch".into()),
            Err(study_errors) => errors.extend(study_errors),
        }
        match execution.sha256() {
            Ok(expected) if expected == self.study_execution_sha256 => {}
            Ok(_) => errors.push("exclusion receipt execution digest mismatch".into()),
            Err(error) => errors.push(format!("failed to hash study execution: {error}")),
        }

        let known: BTreeSet<&str> = study
            .protocol
            .exclusions
            .iter()
            .map(|criterion| criterion.criterion_id.as_str())
            .collect();
        let mut seen = BTreeSet::new();
        for decision in &self.decisions {
            if !known.contains(decision.criterion_id.as_str()) {
                errors.push(format!(
                    "exclusion decision references unknown criterion: {}",
                    decision.criterion_id
                ));
            }
            if !seen.insert(decision.criterion_id.as_str()) {
                errors.push(format!(
                    "duplicate exclusion decision for criterion: {}",
                    decision.criterion_id
                ));
            }
            if !is_lower_hex(&decision.evidence_sha256, 64) {
                errors.push(format!(
                    "exclusion decision {} must include a lowercase SHA-256 evidence digest",
                    decision.criterion_id
                ));
            }
        }
        for criterion in &study.protocol.exclusions {
            if !seen.contains(criterion.criterion_id.as_str()) {
                errors.push(format!(
                    "missing exclusion decision for criterion: {}",
                    criterion.criterion_id
                ));
            }
        }
        errors
    }

    pub fn validate_against(
        &self,
        study: &StudyPreregistration,
        execution: &StudyExecutionTrace,
        limits: ExecutionLimits,
    ) -> Result<(), Vec<String>> {
        let errors = self.validation_errors_against(study, execution, limits);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn disposition_against(
        &self,
        study: &StudyPreregistration,
        execution: &StudyExecutionTrace,
        limits: ExecutionLimits,
    ) -> Result<RunDisposition, Vec<String>> {
        self.validate_against(study, execution, limits)?;
        if self
            .decisions
            .iter()
            .any(|decision| decision.status == ExclusionDecisionStatus::Triggered)
        {
            Ok(RunDisposition::Exclude)
        } else if self
            .decisions
            .iter()
            .any(|decision| decision.status == ExclusionDecisionStatus::Indeterminate)
        {
            Ok(RunDisposition::Indeterminate)
        } else {
            Ok(RunDisposition::Include)
        }
    }

    pub fn sha256(&self) -> Result<String, String> {
        hash_json(self)
    }
}

/// Blinded primary metrics with the study class and exclusion disposition bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyBlindedMetricReport {
    pub schema_version: u16,
    pub run_class: EvidenceRunClass,
    pub study_preregistration_sha256: String,
    pub exclusion_decision_sha256: String,
    pub disposition: RunDisposition,
    pub blinded: BlindedMetricReport,
}

pub fn extract_study_blinded_metrics(
    study: &StudyPreregistration,
    execution: &StudyExecutionTrace,
    exclusions: &ExclusionDecisionReceipt,
    limits: ExecutionLimits,
) -> Result<StudyBlindedMetricReport, Vec<String>> {
    execution.validate_against(study, limits)?;
    let disposition = exclusions.disposition_against(study, execution, limits)?;
    let blinded = extract_blinded_metrics(&execution.trace, &study.protocol, limits)?;
    Ok(StudyBlindedMetricReport {
        schema_version: STUDY_BLINDED_METRIC_SCHEMA_VERSION,
        run_class: study.run_class,
        study_preregistration_sha256: study.sha256()?,
        exclusion_decision_sha256: exclusions
            .sha256()
            .map_err(|error| vec![format!("failed to hash exclusion receipt: {error}")])?,
        disposition,
        blinded,
    })
}

impl StudyBlindedMetricReport {
    pub fn confirmatory_eligible(&self) -> bool {
        self.run_class == EvidenceRunClass::Confirmatory
            && self.disposition == RunDisposition::Include
    }

    pub fn sha256(&self) -> Result<String, String> {
        hash_json(self)
    }
}

/// Confirmatory-only evaluation artifact.
///
/// Exploratory studies and excluded/indeterminate runs cannot produce this
/// artifact through the qualified study API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryHypothesisEvaluation {
    pub schema_version: u16,
    pub study_preregistration_sha256: String,
    pub exclusion_decision_sha256: String,
    pub study_blinded_metric_sha256: String,
    pub evaluation: HypothesisEvaluationReport,
}

pub fn evaluate_confirmatory_study(
    study: &StudyPreregistration,
    blinded: &StudyBlindedMetricReport,
) -> Result<ConfirmatoryHypothesisEvaluation, Vec<String>> {
    study.validate()?;
    if study.run_class != EvidenceRunClass::Confirmatory {
        return Err(vec![
            "exploratory studies cannot be promoted through confirmatory evaluation".into(),
        ]);
    }
    if blinded.run_class != study.run_class {
        return Err(vec!["blinded report run class does not match study".into()]);
    }
    if !blinded.confirmatory_eligible() {
        return Err(vec![
            "confirmatory evaluation requires an included, non-indeterminate run".into(),
        ]);
    }
    let expected_study_sha = study.sha256()?;
    if blinded.study_preregistration_sha256 != expected_study_sha {
        return Err(vec!["blinded report study digest mismatch".into()]);
    }

    let evaluation = evaluate_hypotheses(&study.protocol, &blinded.blinded)?;
    Ok(ConfirmatoryHypothesisEvaluation {
        schema_version: CONFIRMATORY_EVALUATION_SCHEMA_VERSION,
        study_preregistration_sha256: expected_study_sha,
        exclusion_decision_sha256: blinded.exclusion_decision_sha256.clone(),
        study_blinded_metric_sha256: blinded
            .sha256()
            .map_err(|error| vec![format!("failed to hash study blinded report: {error}")])?,
        evaluation,
    })
}

impl ConfirmatoryHypothesisEvaluation {
    pub fn sha256(&self) -> Result<String, String> {
        hash_json(self)
    }
}

fn hash_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(hash_bytes(&bytes))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

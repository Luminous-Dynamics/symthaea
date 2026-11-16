//! SciLaws-Real evaluator-side candidate-selection audit.
//!
//! This module does not select a candidate for the live research policy. It
//! validates the strict development-only selection packet, joins it to the
//! evaluator's frozen post-submission outcomes, and uses the existing
//! architecture-engineering Pareto objective engine to measure whether the
//! submitted Real-track candidate was dominated in (S_N, S_V).
//!
//! Evaluator outcomes in this module are never valid live-selector inputs.
//! Failed and non-evaluable candidates remain in the attempt denominator and
//! are not imputed as zero-valued objective vectors.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::pareto_objectives::{
    ObjectiveDirection, ParetoAnalysisPlan, ParetoAnalysisSummary, ParetoCandidate,
    ParetoObjectiveSpec, analyze_pareto,
};

const LIVE_SCHEMA: &str = "symthaea.scilaws.live-selection-packet";
const LIVE_VERSION: &str = "0.1.0";
const EVALUATOR_SCHEMA: &str = "symthaea.scilaws.real-evaluator-pool";
const EVALUATOR_VERSION: &str = "0.1.0";
const AUDIT_SCHEMA: &str = "symthaea.scilaws.real-selection-audit";
const AUDIT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScilawsSelectionError {
    InvalidJson(String),
    Serialization(String),
    UnsupportedLiveSchema,
    UnsupportedEvaluatorSchema,
    EmptyCampaignId,
    InvalidSelectorPolicyId,
    EmptyCandidatePool,
    EmptyCandidateId,
    DuplicateCandidateId(String),
    InvalidAssumptionCheckId { candidate_id: String },
    InvalidDigest { field: String },
    InvalidDevelopmentRmse { candidate_id: String },
    InvalidComplexity { candidate_id: String },
    CampaignMismatch,
    TaskManifestMismatch,
    LiveCandidateMissingEvaluatorRecord(String),
    EvaluatorCandidateMissingLiveRecord(String),
    CandidateModelDigestMismatch(String),
    DuplicateEvaluatorCandidateId(String),
    IncompleteMetrics { candidate_id: String },
    UnexpectedMetricsForNonComplete { candidate_id: String },
    InvalidMetricRange { candidate_id: String, metric: String },
    SubmittedCandidateMissing(String),
    Pareto(String),
}

impl fmt::Display for ScilawsSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) => write!(f, "invalid JSON: {message}"),
            Self::Serialization(message) => write!(f, "serialization failed: {message}"),
            Self::UnsupportedLiveSchema => write!(f, "unsupported live selection schema/version"),
            Self::UnsupportedEvaluatorSchema => {
                write!(f, "unsupported evaluator pool schema/version")
            }
            Self::EmptyCampaignId => write!(f, "campaign ID must not be empty"),
            Self::InvalidSelectorPolicyId => write!(f, "selector policy ID must not be empty"),
            Self::EmptyCandidatePool => write!(f, "candidate pool must not be empty"),
            Self::EmptyCandidateId => write!(f, "candidate ID must not be empty"),
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate ID: {id}"),
            Self::InvalidAssumptionCheckId { candidate_id } => {
                write!(f, "assumption check IDs must be non-empty and unique for candidate {candidate_id}")
            }
            Self::InvalidDigest { field } => {
                write!(f, "{field} must be a lowercase 64-character SHA-256 digest")
            }
            Self::InvalidDevelopmentRmse { candidate_id } => {
                write!(f, "development RMSE is invalid for candidate {candidate_id}")
            }
            Self::InvalidComplexity { candidate_id } => {
                write!(f, "symbolic complexity is invalid for candidate {candidate_id}")
            }
            Self::CampaignMismatch => write!(f, "live and evaluator campaign IDs differ"),
            Self::TaskManifestMismatch => write!(f, "live and evaluator task-manifest digests differ"),
            Self::LiveCandidateMissingEvaluatorRecord(id) => {
                write!(f, "live candidate has no evaluator record: {id}")
            }
            Self::EvaluatorCandidateMissingLiveRecord(id) => {
                write!(f, "evaluator record has no live candidate: {id}")
            }
            Self::CandidateModelDigestMismatch(id) => {
                write!(f, "candidate model digest differs between live/evaluator records: {id}")
            }
            Self::DuplicateEvaluatorCandidateId(id) => {
                write!(f, "duplicate evaluator candidate ID: {id}")
            }
            Self::IncompleteMetrics { candidate_id } => {
                write!(f, "complete candidate lacks S_N or S_V: {candidate_id}")
            }
            Self::UnexpectedMetricsForNonComplete { candidate_id } => write!(
                f,
                "failed/non-evaluable candidate must not carry placeholder metrics: {candidate_id}"
            ),
            Self::InvalidMetricRange {
                candidate_id,
                metric,
            } => write!(f, "{metric} must be finite and in [0,1] for candidate {candidate_id}"),
            Self::SubmittedCandidateMissing(id) => {
                write!(f, "submitted candidate is absent from the frozen live pool: {id}")
            }
            Self::Pareto(message) => write!(f, "Pareto evaluator failed: {message}"),
        }
    }
}

impl Error for ScilawsSelectionError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveSelectionPacket {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub task_manifest_sha256: String,
    pub development_split_sha256: String,
    pub selector_policy_id: String,
    pub selector_policy_sha256: String,
    pub selection_trace_sha256: String,
    pub resource_receipt_sha256: String,
    pub candidates: Vec<LiveCandidateEvidence>,
    /// None is an explicit abstention. This field records the actual submitted
    /// candidate; it is not a request for this evaluator module to select one.
    pub submitted_candidate_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveCandidateEvidence {
    pub candidate_id: String,
    pub model_sha256: String,
    pub development_evidence_sha256: String,
    /// Development-only RMSE, computed from the declared training/development
    /// folds. It is not the benchmark's held-out S_N.
    pub development_rmse: f64,
    pub symbolic_complexity: u64,
    pub declared_assumption_check_ids: Vec<String>,
}

impl LiveSelectionPacket {
    /// Strictly parse a live-selector packet. Unknown fields such as S_N, S_V,
    /// S_S, held-out results, rubric judgments and hidden labels are rejected by
    /// serde's deny_unknown_fields contract.
    pub fn parse_json(input: &str) -> Result<Self, ScilawsSelectionError> {
        let packet: Self = serde_json::from_str(input)
            .map_err(|error| ScilawsSelectionError::InvalidJson(error.to_string()))?;
        packet.validate()?;
        Ok(packet)
    }

    pub fn validate(&self) -> Result<(), ScilawsSelectionError> {
        if self.schema != LIVE_SCHEMA || self.version != LIVE_VERSION {
            return Err(ScilawsSelectionError::UnsupportedLiveSchema);
        }
        if self.campaign_id.trim().is_empty() {
            return Err(ScilawsSelectionError::EmptyCampaignId);
        }
        for (field, digest) in [
            ("task_manifest_sha256", &self.task_manifest_sha256),
            ("development_split_sha256", &self.development_split_sha256),
            ("selector_policy_sha256", &self.selector_policy_sha256),
            ("selection_trace_sha256", &self.selection_trace_sha256),
            ("resource_receipt_sha256", &self.resource_receipt_sha256),
        ] {
            validate_digest(digest, field.to_owned())?;
        }
        if self.selector_policy_id.trim().is_empty() {
            return Err(ScilawsSelectionError::InvalidSelectorPolicyId);
        }
        if self.candidates.is_empty() {
            return Err(ScilawsSelectionError::EmptyCandidatePool);
        }

        let mut candidate_ids = BTreeSet::new();
        for candidate in &self.candidates {
            if candidate.candidate_id.trim().is_empty() {
                return Err(ScilawsSelectionError::EmptyCandidateId);
            }
            if !candidate_ids.insert(candidate.candidate_id.as_str()) {
                return Err(ScilawsSelectionError::DuplicateCandidateId(
                    candidate.candidate_id.clone(),
                ));
            }
            validate_digest(
                &candidate.model_sha256,
                format!("candidates[{}].model_sha256", candidate.candidate_id),
            )?;
            validate_digest(
                &candidate.development_evidence_sha256,
                format!(
                    "candidates[{}].development_evidence_sha256",
                    candidate.candidate_id
                ),
            )?;
            if !candidate.development_rmse.is_finite() || candidate.development_rmse < 0.0 {
                return Err(ScilawsSelectionError::InvalidDevelopmentRmse {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
            if candidate.symbolic_complexity == 0 {
                return Err(ScilawsSelectionError::InvalidComplexity {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
            let mut checks = BTreeSet::new();
            for check_id in &candidate.declared_assumption_check_ids {
                if check_id.trim().is_empty() || !checks.insert(check_id.as_str()) {
                    return Err(ScilawsSelectionError::InvalidAssumptionCheckId {
                        candidate_id: candidate.candidate_id.clone(),
                    });
                }
            }
        }

        if let Some(submitted_id) = &self.submitted_candidate_id {
            if !candidate_ids.contains(submitted_id.as_str()) {
                return Err(ScilawsSelectionError::SubmittedCandidateMissing(
                    submitted_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Returns byte-identical canonical model-artifact groups without dropping
    /// attempts. This is content identity, not a claim of semantic equivalence.
    pub fn duplicate_model_artifact_groups(&self) -> Vec<Vec<String>> {
        let mut groups = BTreeMap::<&str, Vec<String>>::new();
        for candidate in &self.candidates {
            groups
                .entry(candidate.model_sha256.as_str())
                .or_default()
                .push(candidate.candidate_id.clone());
        }
        let mut duplicates = groups
            .into_values()
            .filter(|ids| ids.len() > 1)
            .collect::<Vec<_>>();
        for ids in &mut duplicates {
            ids.sort();
        }
        duplicates.sort();
        duplicates
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluatorCandidateStatus {
    Complete,
    Failed,
    NotEvaluable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealEvaluatorPool {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub task_manifest_sha256: String,
    pub evaluator_manifest_sha256: String,
    pub candidates: Vec<RealEvaluatorCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealEvaluatorCandidate {
    pub candidate_id: String,
    pub model_sha256: String,
    pub evaluation_evidence_sha256: String,
    pub status: EvaluatorCandidateStatus,
    /// Native SciLaws-Real numeric fit. Missing values are not zero.
    pub s_n: Option<f64>,
    /// Native SciLaws-Real scientific validity. Missing values are not zero.
    pub s_v: Option<f64>,
    /// Hard submission-contract gate, distinct from S_V.
    pub contract_valid: bool,
    /// Frozen anti-hacking gate, distinct from the numeric validity score.
    pub anti_hacking_pass: bool,
}

impl RealEvaluatorPool {
    /// Strictly parse the evaluator-only post-submission result bundle.
    pub fn parse_json(input: &str) -> Result<Self, ScilawsSelectionError> {
        let pool: Self = serde_json::from_str(input)
            .map_err(|error| ScilawsSelectionError::InvalidJson(error.to_string()))?;
        pool.validate()?;
        Ok(pool)
    }

    pub fn validate(&self) -> Result<(), ScilawsSelectionError> {
        if self.schema != EVALUATOR_SCHEMA || self.version != EVALUATOR_VERSION {
            return Err(ScilawsSelectionError::UnsupportedEvaluatorSchema);
        }
        if self.campaign_id.trim().is_empty() {
            return Err(ScilawsSelectionError::EmptyCampaignId);
        }
        validate_digest(
            &self.task_manifest_sha256,
            "evaluator.task_manifest_sha256".to_owned(),
        )?;
        validate_digest(
            &self.evaluator_manifest_sha256,
            "evaluator.evaluator_manifest_sha256".to_owned(),
        )?;
        if self.candidates.is_empty() {
            return Err(ScilawsSelectionError::EmptyCandidatePool);
        }
        let mut ids = BTreeSet::new();
        for candidate in &self.candidates {
            if candidate.candidate_id.trim().is_empty() {
                return Err(ScilawsSelectionError::EmptyCandidateId);
            }
            if !ids.insert(candidate.candidate_id.as_str()) {
                return Err(ScilawsSelectionError::DuplicateEvaluatorCandidateId(
                    candidate.candidate_id.clone(),
                ));
            }
            validate_digest(
                &candidate.model_sha256,
                format!("evaluator[{}].model_sha256", candidate.candidate_id),
            )?;
            validate_digest(
                &candidate.evaluation_evidence_sha256,
                format!(
                    "evaluator[{}].evaluation_evidence_sha256",
                    candidate.candidate_id
                ),
            )?;

            match candidate.status {
                EvaluatorCandidateStatus::Complete => {
                    let (Some(s_n), Some(s_v)) = (candidate.s_n, candidate.s_v) else {
                        return Err(ScilawsSelectionError::IncompleteMetrics {
                            candidate_id: candidate.candidate_id.clone(),
                        });
                    };
                    validate_score(s_n, &candidate.candidate_id, "S_N")?;
                    validate_score(s_v, &candidate.candidate_id, "S_V")?;
                }
                EvaluatorCandidateStatus::Failed | EvaluatorCandidateStatus::NotEvaluable => {
                    if candidate.s_n.is_some() || candidate.s_v.is_some() {
                        return Err(ScilawsSelectionError::UnexpectedMetricsForNonComplete {
                            candidate_id: candidate.candidate_id.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SubmittedCandidateDisposition {
    Abstained,
    NonEvaluable,
    Infeasible,
    StrictlyDominated { dominator_candidate_id: String },
    ParetoNondominated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorAttemptAudit {
    pub candidate_id: String,
    pub status: EvaluatorCandidateStatus,
    pub s_n: Option<f64>,
    pub s_v: Option<f64>,
    pub contract_valid: bool,
    pub anti_hacking_pass: bool,
    pub evaluation_evidence_sha256: String,
    pub dominated_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScilawsRealSelectionAudit {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub task_manifest_sha256: String,
    pub evaluator_manifest_sha256: String,
    pub live_packet_sha256: String,
    pub attempt_count: usize,
    pub duplicate_model_artifact_groups: Vec<Vec<String>>,
    pub attempt_results: Vec<EvaluatorAttemptAudit>,
    pub complete_count: usize,
    pub failed_count: usize,
    pub not_evaluable_count: usize,
    pub infeasible_count: usize,
    pub dominated_candidate_count: usize,
    pub nondominated_candidate_ids: Vec<String>,
    pub submitted_candidate_id: Option<String>,
    pub submitted_disposition: SubmittedCandidateDisposition,
    /// Best evaluator-only values among all complete, feasible candidates.
    /// Kept coordinate-wise; there is no blended S_N/S_V winner score.
    pub best_available_s_n: Option<f64>,
    pub best_available_s_v: Option<f64>,
    pub submitted_s_n: Option<f64>,
    pub submitted_s_v: Option<f64>,
    pub s_n_regret: Option<f64>,
    pub s_v_regret: Option<f64>,
    /// None only when there were no complete evaluator rows. No score is
    /// synthesized for failed or non-evaluable candidates.
    pub pareto_summary: Option<ParetoAnalysisSummary>,
    pub audit_sha256: String,
}

#[derive(Serialize)]
struct AuditReceiptPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    task_manifest_sha256: &'a str,
    evaluator_manifest_sha256: &'a str,
    live_packet_sha256: &'a str,
    attempt_count: usize,
    duplicate_model_artifact_groups: &'a [Vec<String>],
    attempt_results: &'a [EvaluatorAttemptAudit],
    complete_count: usize,
    failed_count: usize,
    not_evaluable_count: usize,
    infeasible_count: usize,
    dominated_candidate_count: usize,
    nondominated_candidate_ids: &'a [String],
    submitted_candidate_id: &'a Option<String>,
    submitted_disposition: &'a SubmittedCandidateDisposition,
    best_available_s_n: Option<f64>,
    best_available_s_v: Option<f64>,
    submitted_s_n: Option<f64>,
    submitted_s_v: Option<f64>,
    s_n_regret: Option<f64>,
    s_v_regret: Option<f64>,
    pareto_receipt_sha256: Option<&'a str>,
}

/// Audit a frozen SciLaws-Real candidate pool after the actual submission has
/// been fixed. The two objective dimensions remain separate: maximize native
/// S_N and S_V. Contract/anti-hacking failures are a hard feasibility gate.
///
/// This function uses evaluator-only scores and must never be called inside
/// the live candidate-selection policy. It returns a post-submission audit,
/// not a selector recommendation.
pub fn audit_real_candidate_selection(
    live: &LiveSelectionPacket,
    evaluator: &RealEvaluatorPool,
) -> Result<ScilawsRealSelectionAudit, ScilawsSelectionError> {
    live.validate()?;
    evaluator.validate()?;
    if live.campaign_id != evaluator.campaign_id {
        return Err(ScilawsSelectionError::CampaignMismatch);
    }
    if live.task_manifest_sha256 != evaluator.task_manifest_sha256 {
        return Err(ScilawsSelectionError::TaskManifestMismatch);
    }

    let live_by_id = live
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect::<BTreeMap<_, _>>();
    let evaluator_by_id = evaluator
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect::<BTreeMap<_, _>>();

    for (candidate_id, live_candidate) in &live_by_id {
        let evaluator_candidate = evaluator_by_id
            .get(candidate_id)
            .ok_or_else(|| {
                ScilawsSelectionError::LiveCandidateMissingEvaluatorRecord(
                    (*candidate_id).to_owned(),
                )
            })?;
        if live_candidate.model_sha256 != evaluator_candidate.model_sha256 {
            return Err(ScilawsSelectionError::CandidateModelDigestMismatch(
                (*candidate_id).to_owned(),
            ));
        }
    }
    for candidate_id in evaluator_by_id.keys() {
        if !live_by_id.contains_key(candidate_id) {
            return Err(ScilawsSelectionError::EvaluatorCandidateMissingLiveRecord(
                (*candidate_id).to_owned(),
            ));
        }
    }

    let mut pareto_candidates = Vec::new();
    let mut complete_count = 0;
    let mut failed_count = 0;
    let mut not_evaluable_count = 0;
    let mut infeasible_count = 0;

    for candidate in &evaluator.candidates {
        match candidate.status {
            EvaluatorCandidateStatus::Failed => failed_count += 1,
            EvaluatorCandidateStatus::NotEvaluable => not_evaluable_count += 1,
            EvaluatorCandidateStatus::Complete => {
                complete_count += 1;
                let s_n = candidate.s_n.expect("complete metrics validated");
                let s_v = candidate.s_v.expect("complete metrics validated");
                let feasible = candidate.contract_valid && candidate.anti_hacking_pass;
                let mut violations = Vec::new();
                if !candidate.contract_valid {
                    violations.push("submission_contract_invalid".to_owned());
                }
                if !candidate.anti_hacking_pass {
                    violations.push("anti_hacking_gate_failed".to_owned());
                }
                if !feasible {
                    infeasible_count += 1;
                }
                pareto_candidates.push(ParetoCandidate {
                    candidate_id: candidate.candidate_id.clone(),
                    objective_values: vec![s_n, s_v],
                    feasible,
                    constraint_violations: violations,
                    evidence_sha256: candidate.evaluation_evidence_sha256.clone(),
                });
            }
        }
    }

    let pareto_summary = if pareto_candidates.is_empty() {
        None
    } else {
        let plan = ParetoAnalysisPlan::new(
            evaluator.campaign_id.clone(),
            vec![
                ParetoObjectiveSpec {
                    objective_id: "S_N".to_owned(),
                    direction: ObjectiveDirection::Maximize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "S_V".to_owned(),
                    direction: ObjectiveDirection::Maximize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
            ],
            vec![0.0, 0.0],
            pareto_candidates,
        )
        .map_err(|error| ScilawsSelectionError::Pareto(error.to_string()))?;
        Some(
            analyze_pareto(&plan)
                .map_err(|error| ScilawsSelectionError::Pareto(error.to_string()))?,
        )
    };

    let (dominated_candidate_count, nondominated_candidate_ids) = pareto_summary
        .as_ref()
        .map(|summary| {
            (
                summary.dominated_candidate_count,
                summary.nondominated_candidate_ids.clone(),
            )
        })
        .unwrap_or_else(|| (0, Vec::new()));

    let duplicate_model_artifact_groups = live.duplicate_model_artifact_groups();
    let mut attempt_results = evaluator
        .candidates
        .iter()
        .map(|candidate| {
            let dominated_by = pareto_summary.as_ref().and_then(|summary| {
                summary
                    .candidate_results
                    .iter()
                    .find(|row| row.candidate_id == candidate.candidate_id)
                    .and_then(|row| row.dominated_by.clone())
            });
            EvaluatorAttemptAudit {
                candidate_id: candidate.candidate_id.clone(),
                status: candidate.status,
                s_n: candidate.s_n,
                s_v: candidate.s_v,
                contract_valid: candidate.contract_valid,
                anti_hacking_pass: candidate.anti_hacking_pass,
                evaluation_evidence_sha256: candidate.evaluation_evidence_sha256.clone(),
                dominated_by,
            }
        })
        .collect::<Vec<_>>();
    attempt_results.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let best_available_s_n = evaluator
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.status == EvaluatorCandidateStatus::Complete
                && candidate.contract_valid
                && candidate.anti_hacking_pass
        })
        .filter_map(|candidate| candidate.s_n)
        .max_by(f64::total_cmp);
    let best_available_s_v = evaluator
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.status == EvaluatorCandidateStatus::Complete
                && candidate.contract_valid
                && candidate.anti_hacking_pass
        })
        .filter_map(|candidate| candidate.s_v)
        .max_by(f64::total_cmp);

    let (submitted_s_n, submitted_s_v) = live
        .submitted_candidate_id
        .as_deref()
        .and_then(|submitted_id| evaluator_by_id.get(submitted_id))
        .filter(|candidate| {
            candidate.status == EvaluatorCandidateStatus::Complete
                && candidate.contract_valid
                && candidate.anti_hacking_pass
        })
        .map(|candidate| (candidate.s_n, candidate.s_v))
        .unwrap_or((None, None));
    let s_n_regret = best_available_s_n
        .zip(submitted_s_n)
        .map(|(best, submitted)| (best - submitted).max(0.0));
    let s_v_regret = best_available_s_v
        .zip(submitted_s_v)
        .map(|(best, submitted)| (best - submitted).max(0.0));

    let submitted_disposition = match live.submitted_candidate_id.as_deref() {
        None => SubmittedCandidateDisposition::Abstained,
        Some(submitted_id) => {
            let candidate = evaluator_by_id
                .get(submitted_id)
                .expect("submitted candidate was checked against the live candidate pool")
                .to_owned();
            match candidate.status {
                EvaluatorCandidateStatus::Failed | EvaluatorCandidateStatus::NotEvaluable => {
                    SubmittedCandidateDisposition::NonEvaluable
                }
                EvaluatorCandidateStatus::Complete
                    if !candidate.contract_valid || !candidate.anti_hacking_pass =>
                {
                    SubmittedCandidateDisposition::Infeasible
                }
                EvaluatorCandidateStatus::Complete => {
                    let dominated_by = pareto_summary
                        .as_ref()
                        .and_then(|summary| {
                            summary
                                .candidate_results
                                .iter()
                                .find(|row| row.candidate_id == submitted_id)
                                .and_then(|row| row.dominated_by.clone())
                        });
                    match dominated_by {
                        Some(dominator_candidate_id) => {
                            SubmittedCandidateDisposition::StrictlyDominated {
                                dominator_candidate_id,
                            }
                        }
                        None => SubmittedCandidateDisposition::ParetoNondominated,
                    }
                }
            }
        }
    };

    let schema = AUDIT_SCHEMA.to_owned();
    let version = AUDIT_VERSION.to_owned();
    let attempt_count = evaluator.candidates.len();
    let live_bytes = serde_json::to_vec(live)
        .map_err(|error| ScilawsSelectionError::Serialization(error.to_string()))?;
    let live_packet_sha256 = sha256_hex(&live_bytes);
    let pareto_receipt_sha256 = pareto_summary
        .as_ref()
        .map(|summary| summary.receipt_sha256.as_str());
    let payload = AuditReceiptPayload {
        schema: &schema,
        version: &version,
        campaign_id: &evaluator.campaign_id,
        task_manifest_sha256: &evaluator.task_manifest_sha256,
        evaluator_manifest_sha256: &evaluator.evaluator_manifest_sha256,
        live_packet_sha256: &live_packet_sha256,
        attempt_count,
        duplicate_model_artifact_groups: &duplicate_model_artifact_groups,
        attempt_results: &attempt_results,
        complete_count,
        failed_count,
        not_evaluable_count,
        infeasible_count,
        dominated_candidate_count,
        nondominated_candidate_ids: &nondominated_candidate_ids,
        submitted_candidate_id: &live.submitted_candidate_id,
        submitted_disposition: &submitted_disposition,
        best_available_s_n,
        best_available_s_v,
        submitted_s_n,
        submitted_s_v,
        s_n_regret,
        s_v_regret,
        pareto_receipt_sha256,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| ScilawsSelectionError::Serialization(error.to_string()))?;
    let audit_sha256 = sha256_hex(&bytes);

    Ok(ScilawsRealSelectionAudit {
        schema,
        version,
        campaign_id: evaluator.campaign_id.clone(),
        task_manifest_sha256: evaluator.task_manifest_sha256.clone(),
        evaluator_manifest_sha256: evaluator.evaluator_manifest_sha256.clone(),
        live_packet_sha256,
        attempt_count,
        duplicate_model_artifact_groups,
        attempt_results,
        complete_count,
        failed_count,
        not_evaluable_count,
        infeasible_count,
        dominated_candidate_count,
        nondominated_candidate_ids,
        submitted_candidate_id: live.submitted_candidate_id.clone(),
        submitted_disposition,
        best_available_s_n,
        best_available_s_v,
        submitted_s_n,
        submitted_s_v,
        s_n_regret,
        s_v_regret,
        pareto_summary,
        audit_sha256,
    })
}

/// Parse the public/live packet and separate evaluator-only bundle, then
/// produce a deterministic post-submission audit report. The evaluator bundle
/// is never projected into the live packet or passed to a selector.
pub fn audit_real_candidate_selection_json(
    live_json: &str,
    evaluator_json: &str,
) -> Result<String, ScilawsSelectionError> {
    let live = LiveSelectionPacket::parse_json(live_json)?;
    let evaluator = RealEvaluatorPool::parse_json(evaluator_json)?;
    let audit = audit_real_candidate_selection(&live, &evaluator)?;
    serde_json::to_string_pretty(&audit)
        .map_err(|error| ScilawsSelectionError::Serialization(error.to_string()))
}

fn validate_digest(value: &str, field: String) -> Result<(), ScilawsSelectionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ScilawsSelectionError::InvalidDigest { field });
    }
    Ok(())
}

fn validate_score(
    value: f64,
    candidate_id: &str,
    metric: &str,
) -> Result<(), ScilawsSelectionError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ScilawsSelectionError::InvalidMetricRange {
            candidate_id: candidate_id.to_owned(),
            metric: metric.to_owned(),
        });
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn live_candidate(
        id: &str,
        model_digest: char,
        development_rmse: f64,
    ) -> LiveCandidateEvidence {
        LiveCandidateEvidence {
            candidate_id: id.to_owned(),
            model_sha256: digest(model_digest),
            development_evidence_sha256: digest('d'),
            development_rmse,
            symbolic_complexity: 12,
            declared_assumption_check_ids: vec!["units".to_owned(), "range".to_owned()],
        }
    }

    fn live_packet(
        candidates: Vec<LiveCandidateEvidence>,
        submitted: Option<&str>,
    ) -> LiveSelectionPacket {
        LiveSelectionPacket {
            schema: LIVE_SCHEMA.to_owned(),
            version: LIVE_VERSION.to_owned(),
            campaign_id: "scilaws-real-fixture-001".to_owned(),
            task_manifest_sha256: digest('f'),
            development_split_sha256: digest('a'),
            selector_policy_id: "fixture-selector-v1".to_owned(),
            selector_policy_sha256: digest('d'),
            selection_trace_sha256: digest('c'),
            resource_receipt_sha256: digest('b'),
            candidates,
            submitted_candidate_id: submitted.map(str::to_owned),
        }
    }

    fn evaluator_candidate(
        id: &str,
        model_digest: char,
        s_n: Option<f64>,
        s_v: Option<f64>,
        contract_valid: bool,
        anti_hacking_pass: bool,
        status: EvaluatorCandidateStatus,
        evidence_digest: char,
    ) -> RealEvaluatorCandidate {
        RealEvaluatorCandidate {
            candidate_id: id.to_owned(),
            model_sha256: digest(model_digest),
            evaluation_evidence_sha256: digest(evidence_digest),
            status,
            s_n,
            s_v,
            contract_valid,
            anti_hacking_pass,
        }
    }

    fn evaluator_pool(candidates: Vec<RealEvaluatorCandidate>) -> RealEvaluatorPool {
        RealEvaluatorPool {
            schema: EVALUATOR_SCHEMA.to_owned(),
            version: EVALUATOR_VERSION.to_owned(),
            campaign_id: "scilaws-real-fixture-001".to_owned(),
            task_manifest_sha256: digest('f'),
            evaluator_manifest_sha256: digest('a'),
            candidates,
        }
    }

    #[test]
    fn live_packet_rejects_evaluator_only_metrics() {
        let packet = r#"{
            "schema":"symthaea.scilaws.live-selection-packet",
            "version":"0.1.0",
            "campaign_id":"c1",
            "task_manifest_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "development_split_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "selector_policy_id":"fixture-selector-v1",
            "selector_policy_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "selection_trace_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "resource_receipt_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "candidates":[{
                "candidate_id":"a",
                "model_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "development_evidence_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "development_rmse":0.2,
                "symbolic_complexity":10,
                "declared_assumption_check_ids":["units"],
                "s_n":0.9
            }],
            "submitted_candidate_id":"a"
        }"#;
        assert!(matches!(
            LiveSelectionPacket::parse_json(packet),
            Err(ScilawsSelectionError::InvalidJson(_))
        ));
    }

    #[test]
    fn evaluator_finds_a_strictly_dominated_submitted_candidate() {
        let live = live_packet(
            vec![
                live_candidate("submitted", 'a', 0.2),
                live_candidate("dominator", 'b', 0.25),
                live_candidate("tradeoff", 'c', 0.3),
            ],
            Some("submitted"),
        );
        let evaluator = evaluator_pool(vec![
            evaluator_candidate("submitted", 'a', Some(0.70), Some(0.80), true, true, EvaluatorCandidateStatus::Complete, '1'),
            evaluator_candidate("dominator", 'b', Some(0.78), Some(0.82), true, true, EvaluatorCandidateStatus::Complete, '2'),
            evaluator_candidate("tradeoff", 'c', Some(0.90), Some(0.50), true, true, EvaluatorCandidateStatus::Complete, '3'),
        ]);

        let audit = audit_real_candidate_selection(&live, &evaluator).unwrap();
        assert_eq!(audit.attempt_count, 3);
        assert_eq!(audit.complete_count, 3);
        assert_eq!(audit.dominated_candidate_count, 1);
        assert_eq!(
            audit.submitted_disposition,
            SubmittedCandidateDisposition::StrictlyDominated {
                dominator_candidate_id: "dominator".to_owned()
            }
        );
        assert_eq!(audit.audit_sha256.len(), 64);
        assert_eq!(audit.best_available_s_n, Some(0.90));
        assert_eq!(audit.best_available_s_v, Some(0.82));
        assert_eq!(audit.submitted_s_n, Some(0.70));
        assert_eq!(audit.submitted_s_v, Some(0.80));
        assert!((audit.s_n_regret.unwrap() - 0.20).abs() < 1.0e-12);
        assert!((audit.s_v_regret.unwrap() - 0.02).abs() < 1.0e-12);
    }

    #[test]
    fn fit_validity_tradeoff_is_not_collapsed_to_a_scalar_winner() {
        let live = live_packet(
            vec![live_candidate("fit", 'a', 0.1), live_candidate("valid", 'b', 0.2)],
            Some("fit"),
        );
        let evaluator = evaluator_pool(vec![
            evaluator_candidate("fit", 'a', Some(0.95), Some(0.55), true, true, EvaluatorCandidateStatus::Complete, '1'),
            evaluator_candidate("valid", 'b', Some(0.75), Some(0.92), true, true, EvaluatorCandidateStatus::Complete, '2'),
        ]);

        let audit = audit_real_candidate_selection(&live, &evaluator).unwrap();
        assert_eq!(audit.dominated_candidate_count, 0);
        assert_eq!(audit.nondominated_candidate_ids, vec!["fit", "valid"]);
        assert_eq!(
            audit.submitted_disposition,
            SubmittedCandidateDisposition::ParetoNondominated
        );
    }

    #[test]
    fn anti_hacking_failure_is_infeasible_even_with_high_numeric_fit() {
        let live = live_packet(
            vec![live_candidate("hacked", 'a', 0.01), live_candidate("valid", 'b', 0.2)],
            Some("hacked"),
        );
        let evaluator = evaluator_pool(vec![
            evaluator_candidate("hacked", 'a', Some(0.99), Some(0.02), true, false, EvaluatorCandidateStatus::Complete, '1'),
            evaluator_candidate("valid", 'b', Some(0.70), Some(0.91), true, true, EvaluatorCandidateStatus::Complete, '2'),
        ]);

        let audit = audit_real_candidate_selection(&live, &evaluator).unwrap();
        assert_eq!(audit.infeasible_count, 1);
        assert_eq!(
            audit.submitted_disposition,
            SubmittedCandidateDisposition::Infeasible
        );
    }

    #[test]
    fn failed_and_non_evaluable_attempts_stay_in_the_denominator() {
        let live = live_packet(
            vec![
                live_candidate("complete", 'a', 0.2),
                live_candidate("failed", 'b', 0.3),
                live_candidate("unknown", 'c', 0.4),
            ],
            Some("complete"),
        );
        let evaluator = evaluator_pool(vec![
            evaluator_candidate("complete", 'a', Some(0.8), Some(0.8), true, true, EvaluatorCandidateStatus::Complete, '1'),
            evaluator_candidate("failed", 'b', None, None, false, false, EvaluatorCandidateStatus::Failed, '2'),
            evaluator_candidate("unknown", 'c', None, None, false, false, EvaluatorCandidateStatus::NotEvaluable, '3'),
        ]);

        let audit = audit_real_candidate_selection(&live, &evaluator).unwrap();
        assert_eq!(audit.attempt_count, 3);
        assert_eq!(audit.complete_count, 1);
        assert_eq!(audit.failed_count, 1);
        assert_eq!(audit.not_evaluable_count, 1);
        assert_eq!(
            audit.submitted_disposition,
            SubmittedCandidateDisposition::ParetoNondominated
        );
    }

    #[test]
    fn missing_evaluator_metric_is_not_imputed_as_zero() {
        let pool = evaluator_pool(vec![evaluator_candidate(
            "a",
            'a',
            Some(0.8),
            None,
            true,
            true,
            EvaluatorCandidateStatus::Complete,
            '1',
        )]);
        assert!(matches!(
            pool.validate(),
            Err(ScilawsSelectionError::IncompleteMetrics { .. })
        ));
    }

    #[test]
    fn no_admissible_candidate_can_end_in_explicit_abstention() {
        let live = live_packet(
            vec![live_candidate("invalid", 'a', 0.2)],
            None,
        );
        let evaluator = evaluator_pool(vec![evaluator_candidate(
            "invalid",
            'a',
            Some(0.9),
            Some(0.1),
            false,
            false,
            EvaluatorCandidateStatus::Complete,
            '1',
        )]);

        let audit = audit_real_candidate_selection(&live, &evaluator).unwrap();
        assert_eq!(audit.infeasible_count, 1);
        assert_eq!(
            audit.submitted_disposition,
            SubmittedCandidateDisposition::Abstained
        );
    }

    #[test]
    fn duplicate_model_artifacts_are_retained_and_flagged() {
        let live = live_packet(
            vec![live_candidate("first", 'a', 0.1), live_candidate("second", 'a', 0.2)],
            Some("first"),
        );
        live.validate().unwrap();
        assert_eq!(
            live.duplicate_model_artifact_groups(),
            vec![vec!["first".to_owned(), "second".to_owned()]]
        );
    }

    #[test]
    fn mismatched_task_manifests_fail_closed() {
        let live = live_packet(vec![live_candidate("a", 'a', 0.1)], Some("a"));
        let mut evaluator = evaluator_pool(vec![evaluator_candidate(
            "a",
            'a',
            Some(0.8),
            Some(0.8),
            true,
            true,
            EvaluatorCandidateStatus::Complete,
            '1',
        )]);
        evaluator.task_manifest_sha256 = digest('e');
        assert_eq!(
            audit_real_candidate_selection(&live, &evaluator).err(),
            Some(ScilawsSelectionError::TaskManifestMismatch)
        );
    }

    #[test]
    fn json_audit_replays_deterministically_from_separate_packets() {
        let live = live_packet(
            vec![live_candidate("a", 'a', 0.1), live_candidate("b", 'b', 0.2)],
            Some("a"),
        );
        let evaluator = evaluator_pool(vec![
            evaluator_candidate("a", 'a', Some(0.8), Some(0.9), true, true, EvaluatorCandidateStatus::Complete, '1'),
            evaluator_candidate("b", 'b', Some(0.9), Some(0.7), true, true, EvaluatorCandidateStatus::Complete, '2'),
        ]);
        let live_json = serde_json::to_string(&live).unwrap();
        let evaluator_json = serde_json::to_string(&evaluator).unwrap();
        let first = audit_real_candidate_selection_json(&live_json, &evaluator_json).unwrap();
        let replay = audit_real_candidate_selection_json(&live_json, &evaluator_json).unwrap();
        assert_eq!(first, replay);
        let report: serde_json::Value = serde_json::from_str(&first).unwrap();
        let regret = report["s_n_regret"].as_f64().unwrap();
        assert!((regret - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn live_selector_policy_metadata_is_required_and_digest_bound() {
        let mut live = live_packet(vec![live_candidate("a", 'a', 0.1)], Some("a"));
        live.selector_policy_id.clear();
        assert_eq!(
            live.validate(),
            Err(ScilawsSelectionError::InvalidSelectorPolicyId)
        );
        live.selector_policy_id = "fixture-selector-v1".to_owned();
        live.selection_trace_sha256 = "not-a-digest".to_owned();
        assert!(matches!(
            live.validate(),
            Err(ScilawsSelectionError::InvalidDigest { .. })
        ));
    }

    #[test]
    fn live_and_evaluator_candidate_sets_must_match_exactly() {
        let live = live_packet(
            vec![live_candidate("a", 'a', 0.1), live_candidate("b", 'b', 0.2)],
            Some("a"),
        );
        let evaluator = evaluator_pool(vec![evaluator_candidate(
            "a",
            'a',
            Some(0.8),
            Some(0.8),
            true,
            true,
            EvaluatorCandidateStatus::Complete,
            '1',
        )]);
        assert_eq!(
            audit_real_candidate_selection(&live, &evaluator).err(),
            Some(ScilawsSelectionError::LiveCandidateMissingEvaluatorRecord(
                "b".to_owned()
            ))
        );
    }
}

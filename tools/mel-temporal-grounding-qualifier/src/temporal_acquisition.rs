//! Source-balanced active refinement and frozen probability-audit selection.
//!
//! Two streams are selected independently:
//! - Active refinement prioritizes known model errors, human disagreement,
//!   panel/stratum coverage gaps and hidden repeats.
//! - Probability audit samples source works first, then one item uniformly
//!   within each selected work, recording exact inclusion probabilities.
//!
//! The active stream never changes the probability-audit draw. Rights and
//! consent receipt digests are structurally checked but not authenticated.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::evidence_digest::canonical_json_sha256;

const PLANNER_SCHEMA_V1: &str = "mel-annotation-acquisition-plan-v1";
const PLANNER_VERSION_V1: &str = "mel-source-balanced-annotation-planner-v1";
const PPM_SCALE: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorizationStateV1 {
    Authorized,
    Denied,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationCohortRoleV1 {
    Development,
    ActiveRefinement,
    ConfirmatoryAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ActivePriorityBandV1 {
    ModelFailure,
    HumanDisagreement,
    CoverageGap,
    HiddenRepeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStreamV1 {
    ActiveRefinement,
    ProbabilityAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionReasonV1 {
    ModelErrorWithStablePanel,
    HumanOrBoundaryDisagreement,
    PanelOrPredictionCoverageGap,
    HiddenRepeatControl,
    StratumMinimumQuota,
    UniformSourceWorkProbabilityDraw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateRejectionReasonV1 {
    RightsDenied,
    RightsUnknown,
    MissingAuthorizationReceipt,
    InvalidAuthorizationReceipt,
    PanelAlreadyComplete,
    NoMatchingStratumQuota,
    ActiveStreamDisabled,
    ProbabilityAuditStreamDisabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateNotSelectedReasonV1 {
    NoActivePrioritySignal,
    ActiveBudgetOrQuotaExhausted,
    NotDrawnInProbabilityAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuotaStreamV1 {
    ActiveRefinement,
    ProbabilityAudit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationAcquisitionCandidateV1 {
    pub item_id: String,
    pub query_id: String,
    pub source_work_id_sha256: String,
    pub recording_sha256: String,
    /// A source-work-level stratum (e.g. tradition × renderer cohort).
    /// Every candidate from the same source work must use the same value.
    pub source_stratum_id: String,
    /// A finer query/task family; it may vary within one source work.
    pub task_family: String,
    pub purpose_scope: String,
    /// Digest of the exact #13560 per-item model-vs-annotator report.
    pub model_grounding_report_sha256: String,
    /// Digest of the exact #13617 reliability report/annotation cohort.
    pub inter_annotator_reliability_report_sha256: String,
    /// Digest binding this item's frozen Development/Active/Audit cohort role.
    pub cohort_assignment_manifest_sha256: String,
    pub cohort_role: AnnotationCohortRoleV1,
    pub authorization_state: AuthorizationStateV1,
    pub authorization_receipt_sha256: Option<String>,
    pub current_annotation_count: u16,
    pub maximum_annotation_count: u16,
    pub annotation_cost_units: u64,
    /// Normalized diagnostics supplied by the upstream scorer; 0..=1,000,000.
    pub model_error_ppm: u32,
    pub human_disagreement_ppm: u32,
    pub occurrence_count_disagreement_ppm: u32,
    pub prediction_coverage_ppm: u32,
    pub boundary_dispersion_ms: u32,
    pub hidden_repeat_control: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationStratumQuotaV1 {
    pub source_stratum_id: String,
    pub active_minimum_items: u32,
    pub active_maximum_items: u32,
    /// Probability audit selects source works, then one item within each work.
    pub audit_source_work_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationAcquisitionConfigV1 {
    pub schema: String,
    pub planner_id: String,
    pub seed: u64,
    pub minimum_panel_count: u16,
    pub maximum_active_items_per_source_work: u32,
    /// Number of query items sampled within each audit source work; at least two,
    /// with a census when a source work has fewer eligible query items.
    pub audit_items_per_source_work: u32,
    pub total_budget_units: u64,
    pub active_budget_units: u64,
    pub probability_audit_budget_units: u64,
    pub model_error_threshold_ppm: u32,
    pub human_disagreement_threshold_ppm: u32,
    pub occurrence_count_disagreement_threshold_ppm: u32,
    pub minimum_prediction_coverage_ppm: u32,
    pub boundary_dispersion_threshold_ms: u32,
    pub stratum_quotas: Vec<AnnotationStratumQuotaV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProbabilityFractionV1 {
    pub numerator: u64,
    pub denominator: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnnotationSelectionRecordV1 {
    pub item_id: String,
    pub query_id: String,
    pub source_work_id_sha256: String,
    pub source_stratum_id: String,
    pub task_family: String,
    pub purpose_scope: String,
    pub stream: SelectionStreamV1,
    pub annotation_cost_units: u64,
    pub priority_band: Option<ActivePriorityBandV1>,
    pub reason: SelectionReasonV1,
    pub model_error_ppm: u32,
    pub human_disagreement_ppm: u32,
    pub occurrence_count_disagreement_ppm: u32,
    pub prediction_coverage_ppm: u32,
    pub boundary_dispersion_ms: u32,
    /// Null for the deterministic adaptive refinement stream.
    pub source_work_inclusion_probability: Option<ProbabilityFractionV1>,
    /// Probability of drawing this item conditional on its source work.
    pub item_given_source_work_probability: Option<ProbabilityFractionV1>,
    /// Product of source-work and conditional item probabilities.
    pub item_inclusion_probability: Option<ProbabilityFractionV1>,
    pub eligible_source_works_in_stratum: Option<usize>,
    pub selected_source_works_in_stratum: Option<usize>,
    pub eligible_items_in_source_work: Option<usize>,
    pub selected_items_in_source_work: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProbabilityAuditStratumDesignV1 {
    pub source_stratum_id: String,
    pub eligible_source_works: usize,
    pub eligible_query_items: usize,
    pub selected_source_works: usize,
    pub selected_query_items: usize,
    pub items_per_source_work_cap: u32,
    /// Fixed worst-case reservation computed before random sampling.
    pub worst_case_reserved_cost_units: u64,
    pub actual_spent_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RejectedAnnotationCandidateV1 {
    pub item_id: String,
    pub source_work_id_sha256: String,
    pub reasons: Vec<CandidateRejectionReasonV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnselectedAnnotationCandidateV1 {
    pub item_id: String,
    pub source_work_id_sha256: String,
    pub stream: SelectionStreamV1,
    pub reason: CandidateNotSelectedReasonV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnmetAnnotationQuotaV1 {
    pub source_stratum_id: String,
    pub stream: QuotaStreamV1,
    pub requested: u32,
    pub selected: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnnotationAcquisitionBudgetUsageV1 {
    pub total_budget_units: u64,
    pub active_budget_units: u64,
    pub probability_audit_budget_units: u64,
    pub active_spent_units: u64,
    pub probability_audit_spent_units: u64,
    pub total_spent_units: u64,
    pub total_remaining_units: u64,
    pub active_selected_items: usize,
    pub probability_audit_selected_items: usize,
    pub probability_audit_selected_source_works: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnnotationAcquisitionReportV1 {
    pub schema: &'static str,
    pub planner_version: &'static str,
    pub dataset_id: String,
    pub seed: u64,
    pub candidate_count: usize,
    pub candidate_manifest_sha256: String,
    pub normalized_config_sha256: String,
    pub active_refinement: Vec<AnnotationSelectionRecordV1>,
    pub probability_audit: Vec<AnnotationSelectionRecordV1>,
    pub probability_audit_design: Vec<ProbabilityAuditStratumDesignV1>,
    pub rejected_candidates: Vec<RejectedAnnotationCandidateV1>,
    pub eligible_but_unselected: Vec<UnselectedAnnotationCandidateV1>,
    pub unmet_quotas: Vec<UnmetAnnotationQuotaV1>,
    pub budget_usage: AnnotationAcquisitionBudgetUsageV1,
    pub selection_receipt_sha256: String,
    pub source_work_sets_are_disjoint: bool,
    pub probability_audit_is_adaptive: bool,
    pub active_stream_is_probability_sample: bool,
    pub authorization_receipts_authenticated: bool,
    pub selection_receipt_is_authentication: bool,
}

#[derive(Serialize)]
struct SelectionReceiptPayloadV1<'a> {
    planner_version: &'static str,
    dataset_id: &'a str,
    seed: u64,
    candidate_manifest_sha256: &'a str,
    normalized_config_sha256: &'a str,
    active_refinement: &'a [AnnotationSelectionRecordV1],
    probability_audit: &'a [AnnotationSelectionRecordV1],
    probability_audit_design: &'a [ProbabilityAuditStratumDesignV1],
    rejected_candidates: &'a [RejectedAnnotationCandidateV1],
    eligible_but_unselected: &'a [UnselectedAnnotationCandidateV1],
    unmet_quotas: &'a [UnmetAnnotationQuotaV1],
    budget_usage: &'a AnnotationAcquisitionBudgetUsageV1,
}

pub fn plan_annotation_acquisition_json(
    input: &str,
) -> Result<AnnotationAcquisitionReportV1, String> {
    let dataset: AnnotationAcquisitionDatasetV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid acquisition dataset JSON: {error}"))?;
    plan_annotation_acquisition(&dataset)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationAcquisitionDatasetV1 {
    pub dataset_id: String,
    pub config: AnnotationAcquisitionConfigV1,
    pub candidates: Vec<AnnotationAcquisitionCandidateV1>,
}

pub fn plan_annotation_acquisition(
    dataset: &AnnotationAcquisitionDatasetV1,
) -> Result<AnnotationAcquisitionReportV1, String> {
    validate_config(&dataset.config)?;
    require_text(&dataset.dataset_id, "dataset_id")?;
    if dataset.candidates.is_empty() {
        return Err("annotation acquisition candidate set must not be empty".into());
    }

    let mut candidates = dataset.candidates.clone();
    candidates.sort_by(|left, right| left.item_id.cmp(&right.item_id));
    validate_candidates(&candidates)?;

    let mut config = dataset.config.clone();
    config
        .stratum_quotas
        .sort_by(|left, right| left.source_stratum_id.cmp(&right.source_stratum_id));

    let candidate_manifest_sha256 = canonical_json_sha256(&candidates)
        .map_err(|error| format!("failed to hash candidate manifest: {error}"))?;
    let normalized_config_sha256 = canonical_json_sha256(&config)
        .map_err(|error| format!("failed to hash normalized config: {error}"))?;

    let quotas: BTreeMap<&str, &AnnotationStratumQuotaV1> = config
        .stratum_quotas
        .iter()
        .map(|quota| (quota.source_stratum_id.as_str(), quota))
        .collect();

    let mut rejected_candidates = Vec::new();
    let mut eligible_active = Vec::new();
    let mut eligible_audit = Vec::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let reasons = hard_rejections(candidate, quotas.get(candidate.source_stratum_id.as_str()).copied());
        if !reasons.is_empty() {
            rejected_candidates.push(RejectedAnnotationCandidateV1 {
                item_id: candidate.item_id.clone(),
                source_work_id_sha256: candidate.source_work_id_sha256.clone(),
                reasons,
            });
            continue;
        }

        let quota = quotas
            .get(candidate.source_stratum_id.as_str())
            .copied()
            .ok_or_else(|| "candidate passed hard gate without a stratum quota".to_string())?;
        match candidate.cohort_role {
            AnnotationCohortRoleV1::Development | AnnotationCohortRoleV1::ActiveRefinement => {
                if quota.active_maximum_items > 0 {
                    eligible_active.push(index);
                }
            }
            AnnotationCohortRoleV1::ConfirmatoryAudit => {
                if quota.audit_source_work_count > 0 {
                    eligible_audit.push(index);
                }
            }
        }
    }

    let mut random_state = config.seed;
    let (
        probability_audit,
        audit_unmet,
        audit_spent,
        audit_source_works,
        audit_design,
    ) = select_probability_audit(&candidates, &eligible_audit, &config, &mut random_state)?;

    let audit_source_set: BTreeSet<_> = probability_audit
        .iter()
        .map(|selection| selection.source_work_id_sha256.as_str())
        .collect();

    let (active_refinement, active_unmet, active_spent, active_source_counts) = select_active(
        &candidates,
        &eligible_active,
        &config,
        &audit_source_set,
    )?;

    let mut unmet_quotas = active_unmet;
    unmet_quotas.extend(audit_unmet);
    unmet_quotas.sort_by(|left, right| {
        left.source_stratum_id
            .cmp(&right.source_stratum_id)
            .then_with(|| (left.stream as u8).cmp(&(right.stream as u8)))
    });

    let selected_items: BTreeSet<_> = active_refinement
        .iter()
        .chain(&probability_audit)
        .map(|selection| selection.item_id.as_str())
        .collect();

    let rejected_items: BTreeSet<_> = rejected_candidates
        .iter()
        .map(|rejection| rejection.item_id.as_str())
        .collect();

    let mut eligible_but_unselected = Vec::new();
    for candidate in &candidates {
        if selected_items.contains(candidate.item_id.as_str())
            || rejected_items.contains(candidate.item_id.as_str())
        {
            continue;
        }
        let stream = match candidate.cohort_role {
            AnnotationCohortRoleV1::Development | AnnotationCohortRoleV1::ActiveRefinement => {
                SelectionStreamV1::ActiveRefinement
            }
            AnnotationCohortRoleV1::ConfirmatoryAudit => SelectionStreamV1::ProbabilityAudit,
        };
        let reason = match stream {
            SelectionStreamV1::ActiveRefinement => {
                if active_priority(candidate, &config).is_none() {
                    CandidateNotSelectedReasonV1::NoActivePrioritySignal
                } else {
                    CandidateNotSelectedReasonV1::ActiveBudgetOrQuotaExhausted
                }
            }
            SelectionStreamV1::ProbabilityAudit => {
                CandidateNotSelectedReasonV1::NotDrawnInProbabilityAudit
            }
        };
        eligible_but_unselected.push(UnselectedAnnotationCandidateV1 {
            item_id: candidate.item_id.clone(),
            source_work_id_sha256: candidate.source_work_id_sha256.clone(),
            stream,
            reason,
        });
    }

    let total_spent = active_spent
        .checked_add(audit_spent)
        .ok_or_else(|| "total annotation budget usage overflow".to_string())?;
    if total_spent > config.total_budget_units {
        return Err("planner exceeded the total annotation budget".into());
    }

    let budget_usage = AnnotationAcquisitionBudgetUsageV1 {
        total_budget_units: config.total_budget_units,
        active_budget_units: config.active_budget_units,
        probability_audit_budget_units: config.probability_audit_budget_units,
        active_spent_units: active_spent,
        probability_audit_spent_units: audit_spent,
        total_spent_units: total_spent,
        total_remaining_units: config.total_budget_units - total_spent,
        active_selected_items: active_refinement.len(),
        probability_audit_selected_items: probability_audit.len(),
        probability_audit_selected_source_works: audit_source_works,
    };

    let payload = SelectionReceiptPayloadV1 {
        planner_version: PLANNER_VERSION_V1,
        dataset_id: &dataset.dataset_id,
        seed: config.seed,
        candidate_manifest_sha256: &candidate_manifest_sha256,
        normalized_config_sha256: &normalized_config_sha256,
        active_refinement: &active_refinement,
        probability_audit: &probability_audit,
        probability_audit_design: &audit_design,
        rejected_candidates: &rejected_candidates,
        eligible_but_unselected: &eligible_but_unselected,
        unmet_quotas: &unmet_quotas,
        budget_usage: &budget_usage,
    };
    let selection_receipt_sha256 = canonical_json_sha256(&payload)
        .map_err(|error| format!("failed to hash selection receipt: {error}"))?;

    let report = AnnotationAcquisitionReportV1 {
        schema: "mel-annotation-acquisition-report-v1",
        planner_version: PLANNER_VERSION_V1,
        dataset_id: dataset.dataset_id.clone(),
        seed: config.seed,
        candidate_count: candidates.len(),
        candidate_manifest_sha256,
        normalized_config_sha256,
        active_refinement,
        probability_audit,
        probability_audit_design: audit_design,
        rejected_candidates,
        eligible_but_unselected,
        unmet_quotas,
        budget_usage,
        selection_receipt_sha256,
        source_work_sets_are_disjoint: true,
        probability_audit_is_adaptive: false,
        active_stream_is_probability_sample: false,
        authorization_receipts_authenticated: false,
        selection_receipt_is_authentication: false,
    };

    // The active selector must never use any source work that entered the
    // probability audit. This second check protects future edits to either arm.
    let active_sources: BTreeSet<_> = report
        .active_refinement
        .iter()
        .map(|selection| selection.source_work_id_sha256.as_str())
        .collect();
    if !active_sources.is_disjoint(&audit_source_set) {
        return Err("active and probability-audit source-work sets overlap".into());
    }

    // Keep this local counter observable in the debug/test report path; it is
    // also useful for assertions that source-work caps were honored.
    if active_source_counts
        .values()
        .any(|count| *count > config.maximum_active_items_per_source_work)
    {
        return Err("active selector exceeded the source-work cap".into());
    }
    Ok(report)
}

fn validate_config(config: &AnnotationAcquisitionConfigV1) -> Result<(), String> {
    if config.schema != PLANNER_SCHEMA_V1 {
        return Err(format!("unsupported acquisition planner schema: {}", config.schema));
    }
    require_text(&config.planner_id, "planner_id")?;
    if config.seed == 0 {
        return Err("planner seed must be nonzero".into());
    }
    if config.minimum_panel_count == 0 || config.maximum_active_items_per_source_work == 0 {
        return Err("panel minimum and active source-work cap must be positive".into());
    }
    if config.audit_items_per_source_work < 2 {
        return Err("audit_items_per_source_work must be at least two".into());
    }
    if config.total_budget_units == 0 {
        return Err("total annotation budget must be positive".into());
    }
    let reserved = config
        .active_budget_units
        .checked_add(config.probability_audit_budget_units)
        .ok_or_else(|| "reserved annotation budgets overflow".to_string())?;
    if reserved > config.total_budget_units {
        return Err("active and audit budgets exceed the total budget".into());
    }
    for (label, value) in [
        ("model_error_threshold_ppm", config.model_error_threshold_ppm),
        ("human_disagreement_threshold_ppm", config.human_disagreement_threshold_ppm),
        (
            "occurrence_count_disagreement_threshold_ppm",
            config.occurrence_count_disagreement_threshold_ppm,
        ),
        ("minimum_prediction_coverage_ppm", config.minimum_prediction_coverage_ppm),
    ] {
        if value > PPM_SCALE {
            return Err(format!("{label} must be within 0..=1,000,000"));
        }
    }

    let mut strata = BTreeSet::new();
    for quota in &config.stratum_quotas {
        require_text(&quota.source_stratum_id, "quota.source_stratum_id")?;
        if !strata.insert(quota.source_stratum_id.as_str()) {
            return Err(format!("duplicate source stratum quota: {}", quota.source_stratum_id));
        }
        if quota.active_minimum_items > quota.active_maximum_items {
            return Err(format!(
                "active minimum exceeds maximum for stratum {}",
                quota.source_stratum_id
            ));
        }
    }
    if config.stratum_quotas.is_empty() {
        return Err("at least one source stratum quota is required".into());
    }
    Ok(())
}

fn validate_candidates(candidates: &[AnnotationAcquisitionCandidateV1]) -> Result<(), String> {
    let mut item_ids = BTreeSet::new();
    let mut query_ids = BTreeSet::new();
    let mut source_metadata: BTreeMap<&str, (AnnotationCohortRoleV1, &str)> = BTreeMap::new();

    for candidate in candidates {
        for (label, value) in [
            ("item_id", candidate.item_id.as_str()),
            ("query_id", candidate.query_id.as_str()),
            ("source_stratum_id", candidate.source_stratum_id.as_str()),
            ("task_family", candidate.task_family.as_str()),
            ("purpose_scope", candidate.purpose_scope.as_str()),
        ] {
            require_text(value, label)?;
        }
        validate_digest(&candidate.source_work_id_sha256, "source_work_id_sha256")?;
        validate_digest(&candidate.recording_sha256, "recording_sha256")?;
        validate_digest(
            &candidate.model_grounding_report_sha256,
            "model_grounding_report_sha256",
        )?;
        validate_digest(
            &candidate.inter_annotator_reliability_report_sha256,
            "inter_annotator_reliability_report_sha256",
        )?;
        validate_digest(
            &candidate.cohort_assignment_manifest_sha256,
            "cohort_assignment_manifest_sha256",
        )?;
        if !item_ids.insert(candidate.item_id.as_str()) {
            return Err(format!("duplicate acquisition item ID: {}", candidate.item_id));
        }
        if !query_ids.insert(candidate.query_id.as_str()) {
            return Err(format!("duplicate acquisition query ID: {}", candidate.query_id));
        }
        if candidate.maximum_annotation_count == 0
            || candidate.current_annotation_count > candidate.maximum_annotation_count
        {
            return Err(format!(
                "candidate {} has invalid current/maximum panel counts",
                candidate.item_id
            ));
        }
        if candidate.annotation_cost_units == 0 {
            return Err(format!("candidate {} has zero annotation cost", candidate.item_id));
        }
        for (label, value) in [
            ("model_error_ppm", candidate.model_error_ppm),
            ("human_disagreement_ppm", candidate.human_disagreement_ppm),
            (
                "occurrence_count_disagreement_ppm",
                candidate.occurrence_count_disagreement_ppm,
            ),
            ("prediction_coverage_ppm", candidate.prediction_coverage_ppm),
        ] {
            if value > PPM_SCALE {
                return Err(format!(
                    "candidate {} field {label} is outside 0..=1,000,000",
                    candidate.item_id
                ));
            }
        }
        if let Some(receipt) = candidate.authorization_receipt_sha256.as_deref() {
            if !is_digest(receipt) {
                return Err(format!(
                    "candidate {} has an invalid authorization receipt digest",
                    candidate.item_id
                ));
            }
        }
        if let Some((role, stratum)) = source_metadata.get(candidate.source_work_id_sha256.as_str()) {
            if *role != candidate.cohort_role || *stratum != candidate.source_stratum_id {
                return Err(format!(
                    "source work {} crosses cohort roles or source strata",
                    candidate.source_work_id_sha256
                ));
            }
        } else {
            source_metadata.insert(
                candidate.source_work_id_sha256.as_str(),
                (candidate.cohort_role, candidate.source_stratum_id.as_str()),
            );
        }
    }
    Ok(())
}

fn hard_rejections(
    candidate: &AnnotationAcquisitionCandidateV1,
    quota: Option<&AnnotationStratumQuotaV1>,
) -> Vec<CandidateRejectionReasonV1> {
    let mut reasons = Vec::new();
    match candidate.authorization_state {
        AuthorizationStateV1::Authorized => {
            if candidate.authorization_receipt_sha256.is_none() {
                reasons.push(CandidateRejectionReasonV1::MissingAuthorizationReceipt);
            } else if candidate
                .authorization_receipt_sha256
                .as_deref()
                .is_none_or(|value| !is_digest(value))
            {
                reasons.push(CandidateRejectionReasonV1::InvalidAuthorizationReceipt);
            }
        }
        AuthorizationStateV1::Denied => reasons.push(CandidateRejectionReasonV1::RightsDenied),
        AuthorizationStateV1::Unknown => reasons.push(CandidateRejectionReasonV1::RightsUnknown),
    }
    if candidate.current_annotation_count >= candidate.maximum_annotation_count {
        reasons.push(CandidateRejectionReasonV1::PanelAlreadyComplete);
    }
    match (candidate.cohort_role, quota) {
        (_, None) => reasons.push(CandidateRejectionReasonV1::NoMatchingStratumQuota),
        (
            AnnotationCohortRoleV1::Development | AnnotationCohortRoleV1::ActiveRefinement,
            Some(quota),
        ) if quota.active_maximum_items == 0 => {
            reasons.push(CandidateRejectionReasonV1::ActiveStreamDisabled)
        }
        (AnnotationCohortRoleV1::ConfirmatoryAudit, Some(quota))
            if quota.audit_source_work_count == 0 =>
        {
            reasons.push(CandidateRejectionReasonV1::ProbabilityAuditStreamDisabled)
        }
        _ => {}
    }
    reasons
}

fn active_priority(
    candidate: &AnnotationAcquisitionCandidateV1,
    config: &AnnotationAcquisitionConfigV1,
) -> Option<(ActivePriorityBandV1, SelectionReasonV1)> {
    let human_disagreement = candidate.human_disagreement_ppm
        >= config.human_disagreement_threshold_ppm
        || candidate.occurrence_count_disagreement_ppm
            >= config.occurrence_count_disagreement_threshold_ppm
        || candidate.boundary_dispersion_ms >= config.boundary_dispersion_threshold_ms;
    let coverage_gap = candidate.current_annotation_count < config.minimum_panel_count
        || candidate.prediction_coverage_ppm < config.minimum_prediction_coverage_ppm;
    // A likely model failure is only elevated when the existing human panel
    // is stable and prediction coverage is adequate. Disagreement instead
    // triggers annotation refinement; it is not used as a clean model-error
    // label.
    if candidate.model_error_ppm >= config.model_error_threshold_ppm
        && !human_disagreement
        && !coverage_gap
    {
        return Some((
            ActivePriorityBandV1::ModelFailure,
            SelectionReasonV1::ModelErrorWithStablePanel,
        ));
    }
    if human_disagreement {
        return Some((
            ActivePriorityBandV1::HumanDisagreement,
            SelectionReasonV1::HumanOrBoundaryDisagreement,
        ));
    }
    if coverage_gap {
        return Some((
            ActivePriorityBandV1::CoverageGap,
            SelectionReasonV1::PanelOrPredictionCoverageGap,
        ));
    }
    if candidate.hidden_repeat_control {
        return Some((
            ActivePriorityBandV1::HiddenRepeat,
            SelectionReasonV1::HiddenRepeatControl,
        ));
    }
    None
}

fn select_probability_audit(
    candidates: &[AnnotationAcquisitionCandidateV1],
    eligible: &[usize],
    config: &AnnotationAcquisitionConfigV1,
    rng: &mut u64,
) -> Result<
    (
        Vec<AnnotationSelectionRecordV1>,
        Vec<UnmetAnnotationQuotaV1>,
        u64,
        usize,
    ),
    String,
> {
    let mut by_stratum: BTreeMap<&str, BTreeMap<&str, Vec<usize>>> = BTreeMap::new();
    for index in eligible {
        let candidate = &candidates[*index];
        by_stratum
            .entry(candidate.source_stratum_id.as_str())
            .or_default()
            .entry(candidate.source_work_id_sha256.as_str())
            .or_default()
            .push(*index);
    }
    for works in by_stratum.values_mut() {
        for items in works.values_mut() {
            items.sort_by(|left, right| {
                candidates[*left]
                    .item_id
                    .cmp(&candidates[*right].item_id)
            });
        }
    }

    let quotas: BTreeMap<_, _> = config
        .stratum_quotas
        .iter()
        .map(|quota| (quota.source_stratum_id.as_str(), quota))
        .collect();

    // Determine every stratum's fixed source-work sample size before drawing
    // any random source or item. Reserve worst-case item cost for each planned
    // source. Thus the sample size in later strata cannot depend on whether an
    // earlier random draw happened to select a cheap or expensive item.
    let mut reserved_budget = config.probability_audit_budget_units;
    let mut sample_size_by_stratum: BTreeMap<&str, usize> = BTreeMap::new();
    for (stratum, source_items) in &by_stratum {
        let quota = quotas
            .get(stratum)
            .copied()
            .ok_or_else(|| format!("eligible audit stratum {stratum} has no quota"))?;
        let maximum_item_cost = source_items
            .values()
            .flat_map(|items| items.iter().map(|index| candidates[*index].annotation_cost_units))
            .max()
            .unwrap_or(0);
        if maximum_item_cost == 0 {
            sample_size_by_stratum.insert(*stratum, 0);
            continue;
        }
        let selected_sources = (quota.audit_source_work_count as usize)
            .min(source_items.len())
            .min((reserved_budget / maximum_item_cost) as usize);
        let reservation = (selected_sources as u64)
            .checked_mul(maximum_item_cost)
            .ok_or_else(|| "audit worst-case budget reservation overflow".to_string())?;
        reserved_budget = reserved_budget
            .checked_sub(reservation)
            .ok_or_else(|| "audit worst-case budget reservation underflow".to_string())?;
        sample_size_by_stratum.insert(*stratum, selected_sources);
    }

    let mut selections = Vec::new();
    let mut unmet = Vec::new();
    let mut spent = 0_u64;
    let mut selected_source_count = 0_usize;

    for (stratum, quota) in quotas {
        let requested = quota.audit_source_work_count as usize;
        if requested == 0 {
            continue;
        }
        let Some(source_items) = by_stratum.get(stratum) else {
            unmet.push(UnmetAnnotationQuotaV1 {
                source_stratum_id: stratum.to_string(),
                stream: QuotaStreamV1::ProbabilityAudit,
                requested: requested as u32,
                selected: 0,
            });
            continue;
        };
        let eligible_sources = source_items.len();
        let selected_sources = sample_size_by_stratum
            .get(stratum)
            .copied()
            .unwrap_or(0);

        let mut source_ids: Vec<_> = source_items.keys().copied().collect();
        for index in (1..source_ids.len()).rev() {
            let selected = unbiased_index(rng, index + 1);
            source_ids.swap(index, selected);
        }

        for source_id in source_ids.into_iter().take(selected_sources) {
            let items = source_items
                .get(source_id)
                .ok_or_else(|| "selected audit source work has no candidate items".to_string())?;
            let item_index = items[unbiased_index(rng, items.len())];
            let candidate = &candidates[item_index];
            let denominator = (eligible_sources as u64)
                .checked_mul(items.len() as u64)
                .ok_or_else(|| "audit item inclusion probability denominator overflow".to_string())?;
            if denominator == 0 {
                return Err("audit item inclusion probability denominator is zero".into());
            }
            let probability = ProbabilityFractionV1 {
                numerator: selected_sources as u64,
                denominator,
            };
            selections.push(selection_record(
                candidate,
                SelectionStreamV1::ProbabilityAudit,
                None,
                SelectionReasonV1::UniformSourceWorkProbabilityDraw,
                Some(ProbabilityFractionV1 {
                    numerator: selected_sources as u64,
                    denominator: eligible_sources as u64,
                }),
                Some(ProbabilityFractionV1 {
                    numerator: 1,
                    denominator: items.len() as u64,
                }),
                Some(probability),
            ));
            spent = spent
                .checked_add(candidate.annotation_cost_units)
                .ok_or_else(|| "audit budget usage overflow".to_string())?;
            selected_source_count += 1;
        }

        if selected_sources < requested {
            unmet.push(UnmetAnnotationQuotaV1 {
                source_stratum_id: stratum.to_string(),
                stream: QuotaStreamV1::ProbabilityAudit,
                requested: requested as u32,
                selected: selected_sources as u32,
            });
        }
    }

    if spent > config.probability_audit_budget_units {
        return Err("probability audit exceeded its reserved budget".into());
    }
    // Preserve randomized source-work draw order in the replayable receipt.
    Ok((selections, unmet, spent, selected_source_count))
}

fn select_active(
    candidates: &[AnnotationAcquisitionCandidateV1],
    eligible: &[usize],
    config: &AnnotationAcquisitionConfigV1,
    audit_sources: &BTreeSet<&str>,
) -> Result<
    (
        Vec<AnnotationSelectionRecordV1>,
        Vec<UnmetAnnotationQuotaV1>,
        u64,
        BTreeMap<String, u32>,
    ),
    String,
> {
    let quotas: BTreeMap<&str, &AnnotationStratumQuotaV1> = config
        .stratum_quotas
        .iter()
        .map(|quota| (quota.source_stratum_id.as_str(), quota))
        .collect();
    let mut selected = BTreeSet::new();
    let mut selections = Vec::new();
    let mut spent = 0_u64;
    let mut source_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut stratum_counts: BTreeMap<String, u32> = BTreeMap::new();

    // First satisfy per-stratum minimums. Items without a model/disagreement
    // priority can enter only under an explicit minimum-coverage quota.
    for quota in &config.stratum_quotas {
        while stratum_counts
            .get(&quota.source_stratum_id)
            .copied()
            .unwrap_or(0)
            < quota.active_minimum_items
        {
            let mut options: Vec<_> = eligible
                .iter()
                .copied()
                .filter(|index| {
                    let candidate = &candidates[*index];
                    candidate.source_stratum_id == quota.source_stratum_id
                        && !selected.contains(index)
                        && !audit_sources.contains(candidate.source_work_id_sha256.as_str())
                })
                .collect();
            options.sort_by(|left, right| active_candidate_cmp(*left, *right, candidates, config));
            let Some(index) = options.into_iter().find(|index| {
                active_candidate_fits(
                    *index,
                    candidates,
                    config,
                    &source_counts,
                    &stratum_counts,
                    spent,
                    quota,
                )
            }) else {
                break;
            };
            let candidate = &candidates[index];
            let (band, reason) = active_priority(candidate, config).unwrap_or((
                ActivePriorityBandV1::CoverageGap,
                SelectionReasonV1::StratumMinimumQuota,
            ));
            add_active_selection(
                index,
                candidates,
                band,
                reason,
                &mut selected,
                &mut selections,
                &mut spent,
                &mut source_counts,
                &mut stratum_counts,
            )?;
        }
    }

    // Then round-robin through priority bands and source strata. This prevents
    // one dense source family from monopolizing the remaining annotation budget.
    for band in [
        ActivePriorityBandV1::ModelFailure,
        ActivePriorityBandV1::HumanDisagreement,
        ActivePriorityBandV1::CoverageGap,
        ActivePriorityBandV1::HiddenRepeat,
    ] {
        let mut by_stratum: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for index in eligible {
            if selected.contains(index)
                || audit_sources.contains(candidates[*index].source_work_id_sha256.as_str())
                || active_priority(&candidates[*index], config).map(|value| value.0) != Some(band)
            {
                continue;
            }
            by_stratum
                .entry(candidates[*index].source_stratum_id.as_str())
                .or_default()
                .push(*index);
        }
        for items in by_stratum.values_mut() {
            items.sort_by(|left, right| active_candidate_cmp(*left, *right, candidates, config));
        }

        loop {
            let mut progress = false;
            for (stratum, items) in &by_stratum {
                let quota = quotas
                    .get(stratum)
                    .copied()
                    .ok_or_else(|| "active candidate has no stratum quota".to_string())?;
                if stratum_counts.get(*stratum).copied().unwrap_or(0)
                    >= quota.active_maximum_items
                {
                    continue;
                }
                let next = items.iter().copied().find(|index| {
                    !selected.contains(index)
                        && active_candidate_fits(
                            *index,
                            candidates,
                            config,
                            &source_counts,
                            &stratum_counts,
                            spent,
                            quota,
                        )
                });
                if let Some(index) = next {
                    let candidate = &candidates[index];
                    let (_, reason) = active_priority(candidate, config)
                        .ok_or_else(|| "selected active candidate has no priority reason".to_string())?;
                    add_active_selection(
                        index,
                        candidates,
                        band,
                        reason,
                        &mut selected,
                        &mut selections,
                        &mut spent,
                        &mut source_counts,
                        &mut stratum_counts,
                    )?;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
    }

    let unmet = config
        .stratum_quotas
        .iter()
        .filter_map(|quota| {
            let selected_count = stratum_counts
                .get(&quota.source_stratum_id)
                .copied()
                .unwrap_or(0);
            (selected_count < quota.active_minimum_items).then(|| UnmetAnnotationQuotaV1 {
                source_stratum_id: quota.source_stratum_id.clone(),
                stream: QuotaStreamV1::ActiveRefinement,
                requested: quota.active_minimum_items,
                selected: selected_count,
            })
        })
        .collect();

    Ok((selections, unmet, spent, source_counts))
}

fn active_candidate_fits(
    index: usize,
    candidates: &[AnnotationAcquisitionCandidateV1],
    config: &AnnotationAcquisitionConfigV1,
    source_counts: &BTreeMap<String, u32>,
    stratum_counts: &BTreeMap<String, u32>,
    spent: u64,
    quota: &AnnotationStratumQuotaV1,
) -> bool {
    let candidate = &candidates[index];
    let source_count = source_counts
        .get(&candidate.source_work_id_sha256)
        .copied()
        .unwrap_or(0);
    let stratum_count = stratum_counts
        .get(&candidate.source_stratum_id)
        .copied()
        .unwrap_or(0);
    source_count < config.maximum_active_items_per_source_work
        && stratum_count < quota.active_maximum_items
        && candidate.annotation_cost_units
            <= config.active_budget_units.saturating_sub(spent)
}

fn add_active_selection(
    index: usize,
    candidates: &[AnnotationAcquisitionCandidateV1],
    band: ActivePriorityBandV1,
    reason: SelectionReasonV1,
    selected: &mut BTreeSet<usize>,
    selections: &mut Vec<AnnotationSelectionRecordV1>,
    spent: &mut u64,
    source_counts: &mut BTreeMap<String, u32>,
    stratum_counts: &mut BTreeMap<String, u32>,
) -> Result<(), String> {
    if !selected.insert(index) {
        return Err("active selector attempted to schedule a duplicate candidate".into());
    }
    let candidate = &candidates[index];
    *spent = spent
        .checked_add(candidate.annotation_cost_units)
        .ok_or_else(|| "active budget usage overflow".to_string())?;
    *source_counts
        .entry(candidate.source_work_id_sha256.clone())
        .or_default() += 1;
    *stratum_counts
        .entry(candidate.source_stratum_id.clone())
        .or_default() += 1;
    selections.push(selection_record(
        candidate,
        SelectionStreamV1::ActiveRefinement,
        Some(band),
        reason,
        None,
        None,
        None,
    ));
    Ok(())
}

fn active_candidate_cmp(
    left: usize,
    right: usize,
    candidates: &[AnnotationAcquisitionCandidateV1],
    config: &AnnotationAcquisitionConfigV1,
) -> Ordering {
    let a = &candidates[left];
    let b = &candidates[right];
    let band_a = active_priority(a, config).map(|value| value.0);
    let band_b = active_priority(b, config).map(|value| value.0);
    let priority_rank = |band: Option<ActivePriorityBandV1>| match band {
        Some(ActivePriorityBandV1::ModelFailure) => 0_u8,
        Some(ActivePriorityBandV1::HumanDisagreement) => 1,
        Some(ActivePriorityBandV1::CoverageGap) => 2,
        Some(ActivePriorityBandV1::HiddenRepeat) => 3,
        None => 4,
    };
    priority_rank(band_a)
        .cmp(&priority_rank(band_b))
        .then_with(|| match band_a {
            Some(ActivePriorityBandV1::ModelFailure) => b.model_error_ppm.cmp(&a.model_error_ppm),
            Some(ActivePriorityBandV1::HumanDisagreement) => b
                .human_disagreement_ppm
                .cmp(&a.human_disagreement_ppm)
                .then_with(|| {
                    b.occurrence_count_disagreement_ppm
                        .cmp(&a.occurrence_count_disagreement_ppm)
                })
                .then_with(|| b.boundary_dispersion_ms.cmp(&a.boundary_dispersion_ms)),
            Some(ActivePriorityBandV1::CoverageGap) => a
                .prediction_coverage_ppm
                .cmp(&b.prediction_coverage_ppm)
                .then_with(|| a.current_annotation_count.cmp(&b.current_annotation_count)),
            Some(ActivePriorityBandV1::HiddenRepeat) | None => Ordering::Equal,
        })
        .then_with(|| {
            deterministic_tie_key(config.seed, &a.item_id)
                .cmp(&deterministic_tie_key(config.seed, &b.item_id))
        })
        .then_with(|| a.item_id.cmp(&b.item_id))
}

fn selection_record(
    candidate: &AnnotationAcquisitionCandidateV1,
    stream: SelectionStreamV1,
    priority_band: Option<ActivePriorityBandV1>,
    reason: SelectionReasonV1,
    source_work_probability: Option<ProbabilityFractionV1>,
    item_given_source_probability: Option<ProbabilityFractionV1>,
    item_probability: Option<ProbabilityFractionV1>,
) -> AnnotationSelectionRecordV1 {
    AnnotationSelectionRecordV1 {
        item_id: candidate.item_id.clone(),
        query_id: candidate.query_id.clone(),
        source_work_id_sha256: candidate.source_work_id_sha256.clone(),
        source_stratum_id: candidate.source_stratum_id.clone(),
        task_family: candidate.task_family.clone(),
        purpose_scope: candidate.purpose_scope.clone(),
        stream,
        annotation_cost_units: candidate.annotation_cost_units,
        priority_band,
        reason,
        model_error_ppm: candidate.model_error_ppm,
        human_disagreement_ppm: candidate.human_disagreement_ppm,
        occurrence_count_disagreement_ppm: candidate.occurrence_count_disagreement_ppm,
        prediction_coverage_ppm: candidate.prediction_coverage_ppm,
        boundary_dispersion_ms: candidate.boundary_dispersion_ms,
        source_work_inclusion_probability: source_work_probability,
        item_given_source_work_probability: item_given_source_probability,
        item_inclusion_probability: item_probability,
    }
}

fn unbiased_index(state: &mut u64, upper_bound: usize) -> usize {
    let bound = upper_bound as u64;
    let limit = u64::MAX - (u64::MAX % bound);
    loop {
        *state = (*state).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        if value < limit {
            return (value % bound) as usize;
        }
    }
}

fn deterministic_tie_key(seed: u64, item_id: &str) -> u64 {
    let mut state = item_id
        .bytes()
        .fold(seed, |state, byte| state.rotate_left(5) ^ byte as u64);
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if !is_digest(value) {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn config() -> AnnotationAcquisitionConfigV1 {
        AnnotationAcquisitionConfigV1 {
            schema: PLANNER_SCHEMA_V1.into(),
            planner_id: "synthetic-annotation-plan-v1".into(),
            seed: 0x13624,
            minimum_panel_count: 3,
            maximum_active_items_per_source_work: 1,
            audit_items_per_source_work: 2,
            total_budget_units: 100,
            active_budget_units: 60,
            probability_audit_budget_units: 40,
            model_error_threshold_ppm: 300_000,
            human_disagreement_threshold_ppm: 250_000,
            occurrence_count_disagreement_threshold_ppm: 100_000,
            minimum_prediction_coverage_ppm: 750_000,
            boundary_dispersion_threshold_ms: 80,
            stratum_quotas: vec![
                AnnotationStratumQuotaV1 {
                    source_stratum_id: "stratum-a".into(),
                    active_minimum_items: 1,
                    active_maximum_items: 2,
                    audit_source_work_count: 2,
                },
                AnnotationStratumQuotaV1 {
                    source_stratum_id: "stratum-b".into(),
                    active_minimum_items: 1,
                    active_maximum_items: 2,
                    audit_source_work_count: 2,
                },
            ],
        }
    }

    fn candidate(
        item_id: &str,
        source_work: char,
        stratum: &str,
        cohort_role: AnnotationCohortRoleV1,
        model_error: u32,
        disagreement: u32,
    ) -> AnnotationAcquisitionCandidateV1 {
        AnnotationAcquisitionCandidateV1 {
            item_id: item_id.into(),
            query_id: format!("query-{item_id}"),
            source_work_id_sha256: digest(source_work),
            recording_sha256: digest(source_work.to_ascii_lowercase()),
            source_stratum_id: stratum.into(),
            task_family: "instrument-entry".into(),
            purpose_scope: "evaluation-only".into(),
            model_grounding_report_sha256: digest('b'),
            inter_annotator_reliability_report_sha256: digest('c'),
            cohort_assignment_manifest_sha256: digest('d'),
            cohort_role,
            authorization_state: AuthorizationStateV1::Authorized,
            authorization_receipt_sha256: Some(digest('f')),
            current_annotation_count: 3,
            maximum_annotation_count: 5,
            annotation_cost_units: 5,
            model_error_ppm: model_error,
            human_disagreement_ppm: disagreement,
            occurrence_count_disagreement_ppm: disagreement / 2,
            prediction_coverage_ppm: 900_000,
            boundary_dispersion_ms: disagreement / 10_000,
            hidden_repeat_control: false,
        }
    }

    fn synthetic_dataset() -> AnnotationAcquisitionDatasetV1 {
        let mut active_a3 = candidate(
            "active-a3",
            'c',
            "stratum-a",
            AnnotationCohortRoleV1::Development,
            0,
            0,
        );
        active_a3.prediction_coverage_ppm = 500_000;
        let candidates = vec![
            candidate("active-a1", 'a', "stratum-a", AnnotationCohortRoleV1::Development, 700_000, 50_000),
            candidate("active-a2", 'b', "stratum-a", AnnotationCohortRoleV1::Development, 100_000, 800_000),
            active_a3,
            candidate("active-b1", 'd', "stratum-b", AnnotationCohortRoleV1::ActiveRefinement, 900_000, 20_000),
            candidate("active-b2", 'e', "stratum-b", AnnotationCohortRoleV1::ActiveRefinement, 100_000, 700_000),
            candidate("audit-a1", 'f', "stratum-a", AnnotationCohortRoleV1::ConfirmatoryAudit, 0, 0),
            candidate("audit-a2", 'g', "stratum-a", AnnotationCohortRoleV1::ConfirmatoryAudit, 0, 0),
            candidate("audit-a3", 'h', "stratum-a", AnnotationCohortRoleV1::ConfirmatoryAudit, 0, 0),
            candidate("audit-b1", 'i', "stratum-b", AnnotationCohortRoleV1::ConfirmatoryAudit, 0, 0),
            candidate("audit-b2", 'j', "stratum-b", AnnotationCohortRoleV1::ConfirmatoryAudit, 0, 0),
            candidate("audit-b3", 'k', "stratum-b", AnnotationCohortRoleV1::ConfirmatoryAudit, 0, 0),
        ];
        AnnotationAcquisitionDatasetV1 {
            dataset_id: "synthetic-acquisition-dataset".into(),
            config: config(),
            candidates,
        }
    }

    #[test]
    fn active_refinement_and_probability_audit_are_separate_and_replayable() {
        let dataset = synthetic_dataset();
        let first = plan_annotation_acquisition(&dataset).unwrap();
        let second = plan_annotation_acquisition(&dataset).unwrap();
        assert_eq!(first.selection_receipt_sha256, second.selection_receipt_sha256);
        assert_eq!(first.active_refinement, second.active_refinement);
        assert_eq!(first.probability_audit, second.probability_audit);
        assert!(first.source_work_sets_are_disjoint);
        assert!(!first.probability_audit_is_adaptive);
        assert!(!first.active_stream_is_probability_sample);
        assert!(!first.authorization_receipts_authenticated);
        assert!(!first.selection_receipt_is_authentication);
        assert!(first.probability_audit.iter().all(|item| {
            item.item_inclusion_probability.is_some()
                && item.source_work_inclusion_probability.is_some()
                && item.item_given_source_work_probability.is_some()
        }));
        assert!(first.active_refinement.iter().all(|item| {
            item.item_inclusion_probability.is_none()
                && item.priority_band.is_some()
        }));
        assert_eq!(first.budget_usage.active_spent_units, 20);
        assert_eq!(first.active_refinement.len(), 4);
        assert_eq!(first.budget_usage.probability_audit_selected_source_works, 4);
        assert_eq!(first.budget_usage.probability_audit_spent_units, 20);
        assert_eq!(first.probability_audit.len(), 4);
        assert!(first.probability_audit.iter().all(|item| {
            item.item_inclusion_probability.unwrap().numerator == 2
                && item.item_inclusion_probability.unwrap().denominator == 3
        }));
    }

    #[test]
    fn active_priority_distinguishes_stable_panel_model_error_from_uncertain_labels() {
        let mut candidate = candidate(
            "priority-control",
            'z',
            "stratum-a",
            AnnotationCohortRoleV1::Development,
            800_000,
            0,
        );
        assert_eq!(
            active_priority(&candidate, &config()).unwrap().0,
            ActivePriorityBandV1::ModelFailure
        );

        candidate.human_disagreement_ppm = 900_000;
        assert_eq!(
            active_priority(&candidate, &config()).unwrap().0,
            ActivePriorityBandV1::HumanDisagreement
        );

        candidate.human_disagreement_ppm = 0;
        candidate.model_error_ppm = 0;
        candidate.prediction_coverage_ppm = 500_000;
        assert_eq!(
            active_priority(&candidate, &config()).unwrap().0,
            ActivePriorityBandV1::CoverageGap
        );

        candidate.prediction_coverage_ppm = 900_000;
        candidate.hidden_repeat_control = true;
        assert_eq!(
            active_priority(&candidate, &config()).unwrap().0,
            ActivePriorityBandV1::HiddenRepeat
        );
    }

    #[test]
    fn source_work_cap_and_source_disjointness_are_enforced() {
        let report = plan_annotation_acquisition(&synthetic_dataset()).unwrap();
        let active_sources: BTreeSet<_> = report.active_refinement
            .iter()
            .map(|item| item.source_work_id_sha256.as_str())
            .collect();
        assert_eq!(active_sources.len(), report.active_refinement.len());
        let audit_sources: BTreeSet<_> = report.probability_audit
            .iter()
            .map(|item| item.source_work_id_sha256.as_str())
            .collect();
        assert!(active_sources.is_disjoint(&audit_sources));
    }

    #[test]
    fn rights_and_consent_fail_closed() {
        let mut dataset = synthetic_dataset();
        dataset.candidates[0].authorization_state = AuthorizationStateV1::Unknown;
        dataset.candidates[1].authorization_state = AuthorizationStateV1::Denied;
        dataset.candidates[2].authorization_receipt_sha256 = None;
        let report = plan_annotation_acquisition(&dataset).unwrap();
        assert_eq!(report.rejected_candidates.len(), 3);
        assert!(report.rejected_candidates.iter().all(|item| !item.reasons.is_empty()));
        assert!(!report.authorization_receipts_authenticated);
    }

    #[test]
    fn confirmatory_source_work_cannot_be_scheduled_into_active_stream() {
        let mut dataset = synthetic_dataset();
        // Reuse a ConfirmatoryAudit source-work digest for a Development
        // candidate. The same source work may not cross the two streams.
        dataset.candidates[0].source_work_id_sha256 =
            dataset.candidates[5].source_work_id_sha256.clone();
        assert!(plan_annotation_acquisition(&dataset).is_err());
    }

    #[test]
    fn audit_probability_is_exact_when_items_are_sampled_within_source_works() {
        let mut dataset = synthetic_dataset();
        dataset.config.stratum_quotas[0].audit_source_work_count = 1;
        dataset.config.stratum_quotas[1].audit_source_work_count = 1;

        // stratum-a now has two eligible source works: work f has two items,
        // while work h has one. The sampled item's conditional probability
        // must reflect its work's eligible item count.
        dataset.candidates[6].source_work_id_sha256 =
            dataset.candidates[5].source_work_id_sha256.clone();
        dataset.candidates[6].recording_sha256 =
            dataset.candidates[5].recording_sha256.clone();

        let report = plan_annotation_acquisition(&dataset).unwrap();
        assert_eq!(report.probability_audit.len(), 2);
        for item in &report.probability_audit {
            assert_eq!(
                item.source_work_inclusion_probability.unwrap(),
                ProbabilityFractionV1 {
                    numerator: 1,
                    denominator: 2,
                }
            );
            let conditional = item.item_given_source_work_probability.unwrap();
            let expected = if item.source_work_id_sha256
                == dataset.candidates[5].source_work_id_sha256
            {
                ProbabilityFractionV1 {
                    numerator: 1,
                    denominator: 2,
                }
            } else {
                ProbabilityFractionV1 {
                    numerator: 1,
                    denominator: 1,
                }
            };
            assert_eq!(conditional, expected);
            let item_probability = item.item_inclusion_probability.unwrap();
            let lhs = item_probability.numerator
                * (item.source_work_inclusion_probability.unwrap().denominator)
                * conditional.denominator;
            let rhs = item.source_work_inclusion_probability.unwrap().numerator
                * conditional.numerator
                * item_probability.denominator;
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn later_stratum_sample_size_does_not_depend_on_random_earlier_item_cost() {
        for seed in 1_u64..=64 {
            let mut dataset = synthetic_dataset();
            dataset.config.seed = seed;
            dataset.config.probability_audit_budget_units = 9;
            dataset.config.stratum_quotas[0].audit_source_work_count = 1;
            dataset.config.stratum_quotas[1].audit_source_work_count = 1;
            // Work f has two candidate queries with different costs. One is
            // cheap, one expensive. Its random item draw must not change the
            // pre-reserved source-work sample size in stratum-b.
            dataset.candidates[6].source_work_id_sha256 =
                dataset.candidates[5].source_work_id_sha256.clone();
            dataset.candidates[6].recording_sha256 =
                dataset.candidates[5].recording_sha256.clone();
            dataset.candidates[6].annotation_cost_units = 1;

            let report = plan_annotation_acquisition(&dataset).unwrap();
            assert_eq!(report.probability_audit.len(), 1);
            assert_eq!(report.probability_audit[0].source_stratum_id, "stratum-a");
            assert!(report.budget_usage.probability_audit_spent_units <= 5);
            assert_eq!(
                report.unmet_quotas
                    .iter()
                    .find(|quota| {
                        quota.stream == QuotaStreamV1::ProbabilityAudit
                            && quota.source_stratum_id == "stratum-b"
                    })
                    .map(|quota| quota.selected),
                Some(0)
            );
        }
    }

    #[test]
    fn budgets_and_unmet_minimums_are_reported_without_relaxation() {
        let mut dataset = synthetic_dataset();
        dataset.config.active_budget_units = 5;
        dataset.config.probability_audit_budget_units = 5;
        dataset.config.total_budget_units = 10;
        let report = plan_annotation_acquisition(&dataset).unwrap();
        assert!(report.budget_usage.total_spent_units <= 10);
        assert!(!report.unmet_quotas.is_empty());
    }

    #[test]
    fn invalid_or_missing_diagnostic_lineage_digests_fail_closed() {
        let mut dataset = synthetic_dataset();
        dataset.candidates[0].model_grounding_report_sha256 = "invalid".into();
        assert!(plan_annotation_acquisition(&dataset).is_err());

        let mut dataset = synthetic_dataset();
        dataset.candidates[0].inter_annotator_reliability_report_sha256.clear();
        assert!(plan_annotation_acquisition(&dataset).is_err());

        let mut dataset = synthetic_dataset();
        dataset.candidates[0].cohort_assignment_manifest_sha256 =
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffG".into();
        assert!(plan_annotation_acquisition(&dataset).is_err());
    }

    #[test]
    fn invalid_ids_and_metrics_fail_closed() {
        let mut dataset = synthetic_dataset();
        dataset.candidates[1].item_id = dataset.candidates[0].item_id.clone();
        assert!(plan_annotation_acquisition(&dataset).is_err());

        let mut dataset = synthetic_dataset();
        dataset.candidates[0].human_disagreement_ppm = 1_000_001;
        assert!(plan_annotation_acquisition(&dataset).is_err());

        let mut dataset = synthetic_dataset();
        dataset.config.stratum_quotas[0].active_minimum_items = 3;
        dataset.config.stratum_quotas[0].active_maximum_items = 2;
        assert!(plan_annotation_acquisition(&dataset).is_err());
    }
}

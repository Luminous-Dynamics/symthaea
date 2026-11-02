#![forbid(unsafe_code)]

//! Cross-stage evidence-chain consistency for connectome proofreading workflows.
//!
//! This crate does not run models, acquire views, allocate reviews, apply edits or hash external
//! bytes. It validates that independently generated stage receipts refer to the same operation,
//! task, dataset and graph lineage. Missing stages remain pending; failed stages remain failed.
//! A caller cannot replace a rejected, quarantined or unstable stage with a claimed success.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSplit {
    Train,
    Validation,
    Test,
    Adaptation,
    Calibration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskRoute {
    FalseSplitCorrection,
    FalseMergeIdentification,
    SplitMaskGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitAuditOutcome {
    GroupDisjoint,
    LeakageDetected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewKind {
    EmXy,
    EmXz,
    EmYz,
    EmBest,
    GeometrySingle,
    GeometryDual,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetProvenance {
    pub repository: String,
    pub revision: String,
    pub revision_commit_sha: String,
    pub card_sha256: String,
    pub shard_manifest_sha256: String,
    pub operation_manifest_sha256: String,
    pub license_review_sha256: String,
    pub license_approved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationIdentity {
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub split: DataSplit,
    pub source_archive_id: String,
    pub source_volume_id: String,
    pub source_operation_id: String,
    pub spatial_cube_id: String,
    pub neuron_root_ids: Vec<String>,
    pub animal_id: String,
    pub session_id: String,
    pub species: String,
    pub workflow_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitAuditReceipt {
    pub receipt_sha256: String,
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub split: DataSplit,
    pub policy_sha256: String,
    pub input_manifest_sha256: String,
    pub operation_assignment_sha256: String,
    pub report_sha256: String,
    pub outcome: SplitAuditOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictionReceipt {
    pub receipt_sha256: String,
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub model_checkpoint_sha256: String,
    pub inference_config_sha256: String,
    pub calibration_profile_sha256: String,
    pub prediction_artifact_sha256: String,
    pub error_probability: f64,
    pub calibrated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewDecisionOutcome {
    Acquired {
        view_id: String,
        view_kind: ViewKind,
        sample_hash: String,
        task_route: TaskRoute,
        artifact_sha256: String,
        acquisition_cost_units: String,
    },
    Stopped {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewDecisionReceipt {
    pub receipt_sha256: String,
    pub operation_id: String,
    pub task_route: TaskRoute,
    pub policy_sha256: String,
    pub scenario_set_sha256: String,
    pub loss_matrix_sha256: String,
    pub acquisition_cost_profile_sha256: String,
    pub loss_unit_id: String,
    pub outcome: ViewDecisionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewAllocationDecision {
    Selected,
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewDisposition {
    Approved,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewReceipt {
    pub receipt_sha256: String,
    pub policy_run_sha256: String,
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub allocation: ReviewAllocationDecision,
    pub reviewer_id: Option<String>,
    pub disposition: Option<ReviewDisposition>,
    pub review_seconds: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditTransactionOutcome {
    Committed,
    Rejected,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditTransactionReceipt {
    pub receipt_sha256: String,
    pub event_stream_sha256: String,
    pub transaction_id: String,
    /// Independently computed digest binding the transaction to this sample/task operation.
    pub operation_binding_sha256: String,
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub base_snapshot_sha256: String,
    pub resulting_snapshot_sha256: Option<String>,
    pub verification_artifact_sha256: String,
    pub verification_passed: bool,
    pub annotation_propagation_sha256: String,
    pub annotation_propagation_verified: bool,
    pub outcome: EditTransactionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryStabilityOutcome {
    InsufficientHistory,
    StableOverObservedWindow,
    UnstableOverObservedWindow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryStabilityReceipt {
    pub receipt_sha256: String,
    pub report_sha256: String,
    pub query_id: String,
    pub query_definition_sha256: String,
    pub analysis_code_sha256: String,
    pub uncertainty_set_sha256: String,
    pub source_volume_id: String,
    pub first_graph_snapshot_sha256: String,
    pub latest_graph_snapshot_sha256: String,
    pub outcome: QueryStabilityOutcome,
    pub future_stability_certified: bool,
    pub universal_connectome_completeness_claimed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowEnvelope {
    pub envelope_id: String,
    pub dataset: DatasetProvenance,
    pub operation: OperationIdentity,
    pub base_graph_snapshot_sha256: String,
    pub split_audit: Option<SplitAuditReceipt>,
    pub prediction: Option<PredictionReceipt>,
    pub view_decision: Option<ViewDecisionReceipt>,
    pub review: Option<ReviewReceipt>,
    pub edit_transaction: Option<EditTransactionReceipt>,
    pub query_stability: Option<QueryStabilityReceipt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowStatus {
    DatasetLicensePending,
    SplitAuditPending,
    BlockedBySplitLeakage,
    PredictionPending,
    PredictionUncalibrated,
    ViewDecisionPending,
    ReviewAllocationPending,
    ReviewPending,
    ReviewDeferred,
    ReviewRejected,
    EditPending,
    EditRejected,
    EditQuarantined,
    QueryAuditPending,
    QueryInsufficientHistory,
    QueryUnstable,
    QueryReadyObservedWindow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowAudit {
    pub envelope_id: String,
    pub status: WorkflowStatus,
    pub completed_stages: Vec<&'static str>,
    pub unresolved_stages: Vec<&'static str>,
    pub base_graph_snapshot_sha256: String,
    pub canonical_graph_snapshot_sha256: String,
    pub query_id: Option<String>,
    /// Always false: observed query stability is not a future forecast.
    pub future_stability_certified: bool,
    /// Always false: one query's stability does not imply universal graph completeness.
    pub universal_connectome_completeness_claimed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    InvalidIdentifier { kind: &'static str, value: String },
    InvalidSampleHash(String),
    InvalidSha256 { kind: &'static str, value: String },
    InvalidCommitSha(String),
    DuplicateNeuronRoot(String),
    OperationMismatch { stage: &'static str, field: &'static str },
    QuerySourceVolumeMismatch,
    ViewTaskMismatch,
    ViewSampleMismatch,
    LeakingViewForTask,
    ViewCostUnitMismatch,
    InvalidProbability,
    InvalidReviewReceipt,
    EditBaseSnapshotMismatch,
    EditResultMissing,
    EditNoopSnapshot,
    QueryFirstSnapshotMismatch,
    EditVerificationContradiction,
    EditAnnotationContradiction,
    QuerySnapshotMismatch,
    QueryFutureClaimForbidden,
    QueryUniversalClaimForbidden,
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::InvalidSampleHash(value) => {
                write!(f, "invalid 32-character lowercase sample hash: {value:?}")
            }
            Self::InvalidSha256 { kind, value } => {
                write!(f, "invalid SHA-256 for {kind}: {value:?}")
            }
            Self::InvalidCommitSha(value) => write!(f, "invalid 40-character commit SHA: {value:?}"),
            Self::DuplicateNeuronRoot(value) => write!(f, "duplicate neuron root ID: {value}"),
            Self::OperationMismatch { stage, field } => {
                write!(f, "{stage} receipt does not match operation field {field}")
            }
            Self::QuerySourceVolumeMismatch => {
                write!(f, "query-stability report source volume differs from operation")
            }
            Self::ViewTaskMismatch => write!(f, "acquired view task route differs from operation"),
            Self::ViewSampleMismatch => write!(f, "acquired view sample hash differs from operation"),
            Self::LeakingViewForTask => {
                write!(f, "em_best is forbidden for false-merge identification")
            }
            Self::ViewCostUnitMismatch => {
                write!(f, "view acquisition cost unit differs from decision-loss unit")
            }
            Self::InvalidProbability => write!(f, "prediction probability is not finite in [0,1]"),
            Self::InvalidReviewReceipt => {
                write!(f, "review receipt allocation/disposition/identity fields are inconsistent")
            }
            Self::EditBaseSnapshotMismatch => {
                write!(f, "edit transaction base snapshot differs from envelope base graph")
            }
            Self::EditResultMissing => write!(f, "committed edit lacks resulting snapshot digest"),
            Self::EditNoopSnapshot => write!(f, "committed edit result snapshot equals its base snapshot"),
            Self::QueryFirstSnapshotMismatch => {
                write!(f, "query report first snapshot does not match envelope base snapshot")
            }
            Self::EditVerificationContradiction => {
                write!(f, "committed edit lacks a passing independent verifier")
            }
            Self::EditAnnotationContradiction => {
                write!(f, "committed edit lacks verified annotation propagation")
            }
            Self::QuerySnapshotMismatch => {
                write!(f, "query report latest snapshot does not match canonical graph snapshot")
            }
            Self::QueryFutureClaimForbidden => {
                write!(f, "observed-window query report cannot certify future stability")
            }
            Self::QueryUniversalClaimForbidden => {
                write!(f, "query-specific report cannot claim universal connectome completeness")
            }
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Validate the references that are present and calculate the actual workflow status.
///
/// This does not read or hash the bytes behind receipt references. The ingestion adapter must
/// verify every digest against the referenced bytes and authenticate actor identity before
/// constructing the envelope.
pub fn audit_workflow(envelope: &WorkflowEnvelope) -> Result<WorkflowAudit, EnvelopeError> {
    validate_envelope_identity(envelope)?;

    let mut completed_stages = vec!["DATASET_IDENTITY", "OPERATION_IDENTITY"];
    let mut unresolved_stages = Vec::new();

    validate_sha("license review", &envelope.dataset.license_review_sha256)?;
    if !envelope.dataset.license_approved {
        unresolved_stages.push("LICENSE_APPROVAL");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::DatasetLicensePending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    }

    completed_stages.push("LICENSE_APPROVED");
    let Some(split) = &envelope.split_audit else {
        unresolved_stages.push("SPLIT_AUDIT");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::SplitAuditPending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    };
    validate_split_receipt(split, &envelope.operation)?;
    match split.outcome {
        SplitAuditOutcome::LeakageDetected => {
            unresolved_stages.push("GROUP_DISJOINTNESS");
            return Ok(build_audit(
                envelope,
                WorkflowStatus::BlockedBySplitLeakage,
                completed_stages,
                unresolved_stages,
                envelope.base_graph_snapshot_sha256.clone(),
                None,
            ));
        }
        SplitAuditOutcome::GroupDisjoint => completed_stages.push("SPLIT_AUDIT"),
    }

    let Some(prediction) = &envelope.prediction else {
        unresolved_stages.push("PREDICTION");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::PredictionPending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    };
    validate_prediction_receipt(prediction, &envelope.operation)?;
    if !prediction.calibrated {
        unresolved_stages.push("CALIBRATION");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::PredictionUncalibrated,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    }
    completed_stages.push("PREDICTION");

    let Some(view_decision) = &envelope.view_decision else {
        unresolved_stages.push("VIEW_DECISION");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::ViewDecisionPending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    };
    validate_view_receipt(view_decision, &envelope.operation)?;
    completed_stages.push("VIEW_DECISION");

    let Some(review) = &envelope.review else {
        unresolved_stages.push("REVIEW_ALLOCATION");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::ReviewAllocationPending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    };
    validate_review_receipt(review, &envelope.operation)?;
    match review.allocation {
        ReviewAllocationDecision::Deferred => {
            completed_stages.push("REVIEW_ALLOCATION");
            unresolved_stages.push("HUMAN_REVIEW");
            return Ok(build_audit(
                envelope,
                WorkflowStatus::ReviewDeferred,
                completed_stages,
                unresolved_stages,
                envelope.base_graph_snapshot_sha256.clone(),
                None,
            ));
        }
        ReviewAllocationDecision::Selected => completed_stages.push("REVIEW_ALLOCATION"),
    }

    let Some(disposition) = review.disposition else {
        unresolved_stages.push("HUMAN_REVIEW");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::ReviewPending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    };
    match disposition {
        ReviewDisposition::Rejected => {
            completed_stages.push("HUMAN_REVIEW");
            unresolved_stages.push("EDIT_NOT_AUTHORIZED");
            return Ok(build_audit(
                envelope,
                WorkflowStatus::ReviewRejected,
                completed_stages,
                unresolved_stages,
                envelope.base_graph_snapshot_sha256.clone(),
                None,
            ));
        }
        ReviewDisposition::Approved => completed_stages.push("HUMAN_REVIEW"),
    }

    let Some(edit) = &envelope.edit_transaction else {
        unresolved_stages.push("EDIT_TRANSACTION");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::EditPending,
            completed_stages,
            unresolved_stages,
            envelope.base_graph_snapshot_sha256.clone(),
            None,
        ));
    };
    validate_edit_receipt(edit, &envelope.operation, &envelope.base_graph_snapshot_sha256)?;
    match edit.outcome {
        EditTransactionOutcome::Rejected => {
            completed_stages.push("EDIT_REJECTED");
            unresolved_stages.push("GRAPH_UPDATE");
            return Ok(build_audit(
                envelope,
                WorkflowStatus::EditRejected,
                completed_stages,
                unresolved_stages,
                envelope.base_graph_snapshot_sha256.clone(),
                None,
            ));
        }
        EditTransactionOutcome::Quarantined => {
            completed_stages.push("EDIT_QUARANTINED");
            unresolved_stages.push("GRAPH_UPDATE");
            return Ok(build_audit(
                envelope,
                WorkflowStatus::EditQuarantined,
                completed_stages,
                unresolved_stages,
                envelope.base_graph_snapshot_sha256.clone(),
                None,
            ));
        }
        EditTransactionOutcome::Committed => completed_stages.push("EDIT_COMMITTED"),
    }
    let resulting_snapshot = edit
        .resulting_snapshot_sha256
        .as_ref()
        .expect("committed edit validated with resulting digest")
        .clone();

    let Some(query) = &envelope.query_stability else {
        unresolved_stages.push("QUERY_STABILITY_AUDIT");
        return Ok(build_audit(
            envelope,
            WorkflowStatus::QueryAuditPending,
            completed_stages,
            unresolved_stages,
            resulting_snapshot,
            None,
        ));
    };
    validate_query_receipt(
        query,
        &envelope.operation,
        &envelope.base_graph_snapshot_sha256,
        &resulting_snapshot,
    )?;
    completed_stages.push("QUERY_STABILITY_AUDIT");

    match query.outcome {
        QueryStabilityOutcome::InsufficientHistory => {
            unresolved_stages.push("QUERY_READINESS");
            Ok(build_audit(
                envelope,
                WorkflowStatus::QueryInsufficientHistory,
                completed_stages,
                unresolved_stages,
                resulting_snapshot,
                Some(query.query_id.clone()),
            ))
        }
        QueryStabilityOutcome::UnstableOverObservedWindow => {
            unresolved_stages.push("QUERY_READINESS");
            Ok(build_audit(
                envelope,
                WorkflowStatus::QueryUnstable,
                completed_stages,
                unresolved_stages,
                resulting_snapshot,
                Some(query.query_id.clone()),
            ))
        }
        QueryStabilityOutcome::StableOverObservedWindow => Ok(build_audit(
            envelope,
            WorkflowStatus::QueryReadyObservedWindow,
            completed_stages,
            unresolved_stages,
            resulting_snapshot,
            Some(query.query_id.clone()),
        )),
    }
}

fn build_audit(
    envelope: &WorkflowEnvelope,
    status: WorkflowStatus,
    completed_stages: Vec<&'static str>,
    unresolved_stages: Vec<&'static str>,
    canonical_graph_snapshot_sha256: String,
    query_id: Option<String>,
) -> WorkflowAudit {
    WorkflowAudit {
        envelope_id: envelope.envelope_id.clone(),
        status,
        completed_stages,
        unresolved_stages,
        base_graph_snapshot_sha256: envelope.base_graph_snapshot_sha256.clone(),
        canonical_graph_snapshot_sha256,
        query_id,
        future_stability_certified: false,
        universal_connectome_completeness_claimed: false,
    }
}

fn validate_envelope_identity(envelope: &WorkflowEnvelope) -> Result<(), EnvelopeError> {
    validate_id("envelope", &envelope.envelope_id)?;
    validate_id("repository", &envelope.dataset.repository)?;
    validate_id("revision", &envelope.dataset.revision)?;
    validate_commit_sha(&envelope.dataset.revision_commit_sha)?;
    validate_sha("dataset card", &envelope.dataset.card_sha256)?;
    validate_sha("dataset shard manifest", &envelope.dataset.shard_manifest_sha256)?;
    validate_sha("dataset operation manifest", &envelope.dataset.operation_manifest_sha256)?;
    validate_sha("license review", &envelope.dataset.license_review_sha256)?;

    validate_id("operation", &envelope.operation.operation_id)?;
    validate_sample_hash(&envelope.operation.sample_hash)?;
    for (kind, value) in [
        ("source archive", &envelope.operation.source_archive_id),
        ("source volume", &envelope.operation.source_volume_id),
        ("source operation", &envelope.operation.source_operation_id),
        ("spatial cube", &envelope.operation.spatial_cube_id),
        ("animal", &envelope.operation.animal_id),
        ("session", &envelope.operation.session_id),
        ("species", &envelope.operation.species),
        ("workflow", &envelope.operation.workflow_id),
    ] {
        validate_id(kind, value)?;
    }
    let mut neuron_ids = std::collections::BTreeSet::new();
    for root_id in &envelope.operation.neuron_root_ids {
        validate_id("neuron root", root_id)?;
        if !neuron_ids.insert(root_id) {
            return Err(EnvelopeError::DuplicateNeuronRoot(root_id.clone()));
        }
    }
    validate_sha("base graph snapshot", &envelope.base_graph_snapshot_sha256)?;
    Ok(())
}

fn validate_split_receipt(
    receipt: &SplitAuditReceipt,
    operation: &OperationIdentity,
) -> Result<(), EnvelopeError> {
    validate_sha("split receipt", &receipt.receipt_sha256)?;
    if receipt.operation_id != operation.operation_id {
        return Err(EnvelopeError::OperationMismatch {
            stage: "split audit",
            field: "operation_id",
        });
    }
    if receipt.sample_hash != operation.sample_hash {
        return Err(EnvelopeError::OperationMismatch {
            stage: "split audit",
            field: "sample_hash",
        });
    }
    if receipt.task_route != operation.task_route {
        return Err(EnvelopeError::OperationMismatch {
            stage: "split audit",
            field: "task_route",
        });
    }
    if receipt.split != operation.split {
        return Err(EnvelopeError::OperationMismatch {
            stage: "split audit",
            field: "split",
        });
    }
    validate_sha("split policy", &receipt.policy_sha256)?;
    validate_sha("split input manifest", &receipt.input_manifest_sha256)?;
    validate_sha("split operation assignment", &receipt.operation_assignment_sha256)?;
    validate_sha("split report", &receipt.report_sha256)?;
    Ok(())
}

fn validate_prediction_receipt(
    receipt: &PredictionReceipt,
    operation: &OperationIdentity,
) -> Result<(), EnvelopeError> {
    validate_sha("prediction receipt", &receipt.receipt_sha256)?;
    validate_id("prediction operation", &receipt.operation_id)?;
    validate_sha("model checkpoint", &receipt.model_checkpoint_sha256)?;
    validate_sha("inference config", &receipt.inference_config_sha256)?;
    validate_sha("calibration profile", &receipt.calibration_profile_sha256)?;
    validate_sha("prediction artifact", &receipt.prediction_artifact_sha256)?;
    if receipt.operation_id != operation.operation_id {
        return Err(EnvelopeError::OperationMismatch {
            stage: "prediction",
            field: "operation_id",
        });
    }
    if receipt.sample_hash != operation.sample_hash {
        return Err(EnvelopeError::OperationMismatch {
            stage: "prediction",
            field: "sample_hash",
        });
    }
    if receipt.task_route != operation.task_route {
        return Err(EnvelopeError::OperationMismatch {
            stage: "prediction",
            field: "task_route",
        });
    }
    if !receipt.error_probability.is_finite()
        || !(0.0..=1.0).contains(&receipt.error_probability)
    {
        return Err(EnvelopeError::InvalidProbability);
    }
    Ok(())
}

fn validate_view_receipt(
    receipt: &ViewDecisionReceipt,
    operation: &OperationIdentity,
) -> Result<(), EnvelopeError> {
    validate_sha("view receipt", &receipt.receipt_sha256)?;
    validate_sha("view policy", &receipt.policy_sha256)?;
    validate_sha("view scenario set", &receipt.scenario_set_sha256)?;
    validate_sha("view decision loss matrix", &receipt.loss_matrix_sha256)?;
    validate_sha("view acquisition cost profile", &receipt.acquisition_cost_profile_sha256)?;
    validate_id("view loss unit", &receipt.loss_unit_id)?;
    if receipt.operation_id != operation.operation_id {
        return Err(EnvelopeError::OperationMismatch {
            stage: "view decision",
            field: "operation_id",
        });
    }
    if receipt.task_route != operation.task_route {
        return Err(EnvelopeError::OperationMismatch {
            stage: "view decision",
            field: "task_route",
        });
    }
    if let ViewDecisionOutcome::Acquired {
        view_id,
        view_kind,
        sample_hash,
        task_route,
        artifact_sha256,
        acquisition_cost_units,
    } = &receipt.outcome
    {
        validate_id("view ID", view_id)?;
        validate_id("acquisition cost unit", acquisition_cost_units)?;
        validate_sha("acquired view artifact", artifact_sha256)?;
        if sample_hash != &operation.sample_hash {
            return Err(EnvelopeError::ViewSampleMismatch);
        }
        if task_route != &operation.task_route {
            return Err(EnvelopeError::ViewTaskMismatch);
        }
        if operation.task_route == TaskRoute::FalseMergeIdentification
            && *view_kind == ViewKind::EmBest
        {
            return Err(EnvelopeError::LeakingViewForTask);
        }
        if acquisition_cost_units != &receipt.loss_unit_id {
            return Err(EnvelopeError::ViewCostUnitMismatch);
        }
    } else if let ViewDecisionOutcome::Stopped { reason } = &receipt.outcome {
        validate_id("view stop reason", reason)?;
    }
    Ok(())
}

fn validate_review_receipt(
    receipt: &ReviewReceipt,
    operation: &OperationIdentity,
) -> Result<(), EnvelopeError> {
    validate_sha("review receipt", &receipt.receipt_sha256)?;
    validate_sha("review policy run", &receipt.policy_run_sha256)?;
    if receipt.operation_id != operation.operation_id
        || receipt.sample_hash != operation.sample_hash
        || receipt.task_route != operation.task_route
    {
        return Err(EnvelopeError::OperationMismatch {
            stage: "review",
            field: "operation identity",
        });
    }
    match receipt.allocation {
        ReviewAllocationDecision::Deferred => {
            if receipt.reviewer_id.is_some()
                || receipt.disposition.is_some()
                || receipt.review_seconds.is_some()
            {
                return Err(EnvelopeError::InvalidReviewReceipt);
            }
        }
        ReviewAllocationDecision::Selected => {
            if let Some(reviewer) = &receipt.reviewer_id {
                validate_id("reviewer", reviewer)?;
            }
            if let Some(seconds) = receipt.review_seconds {
                if seconds == 0 {
                    return Err(EnvelopeError::InvalidReviewReceipt);
                }
            }
            if receipt.disposition.is_some()
                && (receipt.reviewer_id.is_none() || receipt.review_seconds.is_none())
            {
                return Err(EnvelopeError::InvalidReviewReceipt);
            }
        }
    }
    Ok(())
}

fn validate_edit_receipt(
    receipt: &EditTransactionReceipt,
    operation: &OperationIdentity,
    base_snapshot: &str,
) -> Result<(), EnvelopeError> {
    validate_sha("edit receipt", &receipt.receipt_sha256)?;
    validate_sha("edit event stream", &receipt.event_stream_sha256)?;
    validate_sha("edit operation binding", &receipt.operation_binding_sha256)?;
    validate_id("transaction ID", &receipt.transaction_id)?;
    validate_sha("edit base snapshot", &receipt.base_snapshot_sha256)?;
    validate_sha("edit verifier artifact", &receipt.verification_artifact_sha256)?;
    validate_sha("annotation propagation receipt", &receipt.annotation_propagation_sha256)?;
    if receipt.operation_id != operation.operation_id
        || receipt.sample_hash != operation.sample_hash
        || receipt.task_route != operation.task_route
    {
        return Err(EnvelopeError::OperationMismatch {
            stage: "edit transaction",
            field: "operation identity",
        });
    }
    if receipt.base_snapshot_sha256 != base_snapshot {
        return Err(EnvelopeError::EditBaseSnapshotMismatch);
    }
    if receipt.outcome == EditTransactionOutcome::Committed {
        let result = receipt
            .resulting_snapshot_sha256
            .as_ref()
            .ok_or(EnvelopeError::EditResultMissing)?;
        validate_sha("edit result snapshot", result)?;
        if result == &receipt.base_snapshot_sha256 {
            return Err(EnvelopeError::EditNoopSnapshot);
        }
        if !receipt.verification_passed {
            return Err(EnvelopeError::EditVerificationContradiction);
        }
        if !receipt.annotation_propagation_verified {
            return Err(EnvelopeError::EditAnnotationContradiction);
        }
    } else if let Some(result) = &receipt.resulting_snapshot_sha256 {
        validate_sha("non-committed edit result snapshot", result)?;
    }
    Ok(())
}

fn validate_query_receipt(
    receipt: &QueryStabilityReceipt,
    operation: &OperationIdentity,
    base_snapshot: &str,
    canonical_snapshot: &str,
) -> Result<(), EnvelopeError> {
    validate_sha("query receipt", &receipt.receipt_sha256)?;
    validate_sha("query report", &receipt.report_sha256)?;
    validate_id("query ID", &receipt.query_id)?;
    validate_sha("query definition", &receipt.query_definition_sha256)?;
    validate_sha("query analysis", &receipt.analysis_code_sha256)?;
    validate_sha("query uncertainty set", &receipt.uncertainty_set_sha256)?;
    validate_sha("query first snapshot", &receipt.first_graph_snapshot_sha256)?;
    validate_sha("query latest snapshot", &receipt.latest_graph_snapshot_sha256)?;
    if receipt.source_volume_id != operation.source_volume_id {
        return Err(EnvelopeError::QuerySourceVolumeMismatch);
    }
    if receipt.first_graph_snapshot_sha256 != base_snapshot {
        return Err(EnvelopeError::QueryFirstSnapshotMismatch);
    }
    if receipt.latest_graph_snapshot_sha256 != canonical_snapshot {
        return Err(EnvelopeError::QuerySnapshotMismatch);
    }
    if receipt.future_stability_certified {
        return Err(EnvelopeError::QueryFutureClaimForbidden);
    }
    if receipt.universal_connectome_completeness_claimed {
        return Err(EnvelopeError::QueryUniversalClaimForbidden);
    }
    Ok(())
}

fn validate_sample_hash(value: &str) -> Result<(), EnvelopeError> {
    if value.len() != 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EnvelopeError::InvalidSampleHash(value.to_owned()));
    }
    Ok(())
}

fn validate_sha(kind: &'static str, value: &str) -> Result<(), EnvelopeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EnvelopeError::InvalidSha256 {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_commit_sha(value: &str) -> Result<(), EnvelopeError> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EnvelopeError::InvalidCommitSha(value.to_owned()));
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), EnvelopeError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(EnvelopeError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn envelope() -> WorkflowEnvelope {
        WorkflowEnvelope {
            envelope_id: "env-1".to_owned(),
            dataset: DatasetProvenance {
                repository: "jeffbbrown2/ConnectomeBench2".to_owned(),
                revision: "v2".to_owned(),
                revision_commit_sha: COMMIT.to_owned(),
                card_sha256: SHA_A.to_owned(),
                shard_manifest_sha256: SHA_B.to_owned(),
                operation_manifest_sha256: SHA_C.to_owned(),
                license_review_sha256: SHA_D.to_owned(),
                license_approved: true,
            },
            operation: OperationIdentity {
                operation_id: "op-1".to_owned(),
                sample_hash: "0123456789abcdef0123456789abcdef".to_owned(),
                task_route: TaskRoute::FalseMergeIdentification,
                split: DataSplit::Test,
                source_archive_id: "archive-human".to_owned(),
                source_volume_id: "volume-1".to_owned(),
                source_operation_id: "archive-human::source-op-1".to_owned(),
                spatial_cube_id: "volume-1::cube-50um-1".to_owned(),
                neuron_root_ids: vec!["root-1".to_owned(), "root-2".to_owned()],
                animal_id: "animal-1".to_owned(),
                session_id: "session-1".to_owned(),
                species: "human".to_owned(),
                workflow_id: "workflow-v1".to_owned(),
            },
            base_graph_snapshot_sha256: SHA_D.to_owned(),
            split_audit: None,
            prediction: None,
            view_decision: None,
            review: None,
            edit_transaction: None,
            query_stability: None,
        }
    }

    fn split_receipt(outcome: SplitAuditOutcome) -> SplitAuditReceipt {
        SplitAuditReceipt {
            receipt_sha256: SHA_A.to_owned(),
            operation_id: "op-1".to_owned(),
            sample_hash: "0123456789abcdef0123456789abcdef".to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            split: DataSplit::Test,
            policy_sha256: SHA_B.to_owned(),
            input_manifest_sha256: SHA_C.to_owned(),
            operation_assignment_sha256: SHA_D.to_owned(),
            report_sha256: SHA_A.to_owned(),
            outcome,
        }
    }

    fn prediction_receipt() -> PredictionReceipt {
        PredictionReceipt {
            receipt_sha256: SHA_A.to_owned(),
            operation_id: "op-1".to_owned(),
            sample_hash: "0123456789abcdef0123456789abcdef".to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            model_checkpoint_sha256: SHA_B.to_owned(),
            inference_config_sha256: SHA_C.to_owned(),
            calibration_profile_sha256: SHA_D.to_owned(),
            prediction_artifact_sha256: SHA_A.to_owned(),
            error_probability: 0.92,
            calibrated: true,
        }
    }

    fn view_receipt() -> ViewDecisionReceipt {
        ViewDecisionReceipt {
            receipt_sha256: SHA_A.to_owned(),
            operation_id: "op-1".to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            policy_sha256: SHA_B.to_owned(),
            scenario_set_sha256: SHA_C.to_owned(),
            loss_matrix_sha256: SHA_D.to_owned(),
            acquisition_cost_profile_sha256: SHA_A.to_owned(),
            loss_unit_id: "risk-units".to_owned(),
            outcome: ViewDecisionOutcome::Acquired {
                view_id: "view-xy".to_owned(),
                view_kind: ViewKind::EmXy,
                sample_hash: "0123456789abcdef0123456789abcdef".to_owned(),
                task_route: TaskRoute::FalseMergeIdentification,
                artifact_sha256: SHA_A.to_owned(),
                acquisition_cost_units: "risk-units".to_owned(),
            },
        }
    }

    fn review_receipt() -> ReviewReceipt {
        ReviewReceipt {
            receipt_sha256: SHA_A.to_owned(),
            policy_run_sha256: SHA_B.to_owned(),
            operation_id: "op-1".to_owned(),
            sample_hash: "0123456789abcdef0123456789abcdef".to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            allocation: ReviewAllocationDecision::Selected,
            reviewer_id: Some("reviewer-1".to_owned()),
            disposition: Some(ReviewDisposition::Approved),
            review_seconds: Some(90),
        }
    }

    fn edit_receipt() -> EditTransactionReceipt {
        EditTransactionReceipt {
            receipt_sha256: SHA_A.to_owned(),
            event_stream_sha256: SHA_B.to_owned(),
            transaction_id: "tx-1".to_owned(),
            operation_binding_sha256: SHA_D.to_owned(),
            operation_id: "op-1".to_owned(),
            sample_hash: "0123456789abcdef0123456789abcdef".to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            base_snapshot_sha256: SHA_D.to_owned(),
            resulting_snapshot_sha256: Some(SHA_C.to_owned()),
            verification_artifact_sha256: SHA_A.to_owned(),
            verification_passed: true,
            annotation_propagation_sha256: SHA_B.to_owned(),
            annotation_propagation_verified: true,
            outcome: EditTransactionOutcome::Committed,
        }
    }

    fn query_receipt(latest: &str, outcome: QueryStabilityOutcome) -> QueryStabilityReceipt {
        QueryStabilityReceipt {
            receipt_sha256: SHA_A.to_owned(),
            report_sha256: SHA_B.to_owned(),
            query_id: "pathway-1".to_owned(),
            query_definition_sha256: SHA_C.to_owned(),
            analysis_code_sha256: SHA_D.to_owned(),
            uncertainty_set_sha256: SHA_A.to_owned(),
            source_volume_id: "volume-1".to_owned(),
            first_graph_snapshot_sha256: SHA_D.to_owned(),
            latest_graph_snapshot_sha256: latest.to_owned(),
            outcome,
            future_stability_certified: false,
            universal_connectome_completeness_claimed: false,
        }
    }

    fn through_split_and_prediction() -> WorkflowEnvelope {
        let mut value = envelope();
        value.split_audit = Some(split_receipt(SplitAuditOutcome::GroupDisjoint));
        value.prediction = Some(prediction_receipt());
        value
    }

    fn through_review() -> WorkflowEnvelope {
        let mut value = through_split_and_prediction();
        value.view_decision = Some(view_receipt());
        value.review = Some(review_receipt());
        value
    }

    fn through_edit() -> WorkflowEnvelope {
        let mut value = through_review();
        value.edit_transaction = Some(edit_receipt());
        value
    }

    #[test]
    fn empty_envelope_reports_split_audit_pending_not_success() {
        let report = audit_workflow(&envelope()).unwrap();
        assert_eq!(report.status, WorkflowStatus::SplitAuditPending);
        assert_eq!(report.unresolved_stages, vec!["SPLIT_AUDIT"]);
        assert!(!report.future_stability_certified);
        assert!(!report.universal_connectome_completeness_claimed);
    }

    #[test]
    fn license_pending_blocks_all_downstream_stages() {
        let mut value = envelope();
        value.dataset.license_approved = false;
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::DatasetLicensePending);
        assert_eq!(report.unresolved_stages, vec!["LICENSE_APPROVAL"]);
    }

    #[test]
    fn split_leakage_blocks_prediction_claims() {
        let mut value = envelope();
        value.split_audit = Some(split_receipt(SplitAuditOutcome::LeakageDetected));
        value.prediction = Some(prediction_receipt());
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::BlockedBySplitLeakage);
        assert!(!report.completed_stages.contains(&"PREDICTION"));
    }

    #[test]
    fn calibrated_prediction_and_group_disjoint_split_advance_to_view_decision_pending() {
        let report = audit_workflow(&through_split_and_prediction()).unwrap();
        assert_eq!(report.status, WorkflowStatus::ViewDecisionPending);
        assert!(report.completed_stages.contains(&"SPLIT_AUDIT"));
        assert!(report.completed_stages.contains(&"PREDICTION"));
    }

    #[test]
    fn split_audit_must_reference_the_same_operation_sample_task_and_split() {
        let mut value = envelope();
        let mut receipt = split_receipt(SplitAuditOutcome::GroupDisjoint);
        receipt.sample_hash = "ffffffffffffffffffffffffffffffff".to_owned();
        value.split_audit = Some(receipt);
        assert_eq!(
            audit_workflow(&value),
            Err(EnvelopeError::OperationMismatch {
                stage: "split audit",
                field: "sample_hash"
            })
        );
    }

    #[test]
    fn uncalibrated_probability_cannot_advance_into_decision_policies() {
        let mut value = through_split_and_prediction();
        value.prediction.as_mut().unwrap().calibrated = false;
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::PredictionUncalibrated);
        assert_eq!(report.unresolved_stages, vec!["CALIBRATION"]);
    }

    #[test]
    fn false_merge_em_best_is_rejected_even_if_a_stage_receipt_claims_acquisition() {
        let mut value = through_split_and_prediction();
        let mut receipt = view_receipt();
        if let ViewDecisionOutcome::Acquired { view_kind, .. } = &mut receipt.outcome {
            *view_kind = ViewKind::EmBest;
        }
        value.view_decision = Some(receipt);
        assert_eq!(audit_workflow(&value), Err(EnvelopeError::LeakingViewForTask));
    }

    #[test]
    fn acquired_view_must_match_sample_task_and_loss_cost_units() {
        let mut value = through_split_and_prediction();
        let mut receipt = view_receipt();
        if let ViewDecisionOutcome::Acquired {
            sample_hash,
            acquisition_cost_units,
            ..
        } = &mut receipt.outcome
        {
            *sample_hash = "ffffffffffffffffffffffffffffffff".to_owned();
            *acquisition_cost_units = "milliseconds".to_owned();
        }
        value.view_decision = Some(receipt);
        assert_eq!(audit_workflow(&value), Err(EnvelopeError::ViewSampleMismatch));
    }

    #[test]
    fn stopped_view_decision_preserves_stop_reason_and_can_continue_to_review() {
        let mut value = through_split_and_prediction();
        let mut receipt = view_receipt();
        receipt.outcome = ViewDecisionOutcome::Stopped {
            reason: "NO_POSITIVE_NET_VALUE".to_owned(),
        };
        value.view_decision = Some(receipt);
        value.review = Some(review_receipt());
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::EditPending);
        assert!(report.completed_stages.contains(&"VIEW_DECISION"));
    }

    #[test]
    fn allocated_review_without_disposition_remains_review_pending() {
        let mut value = through_split_and_prediction();
        value.view_decision = Some(view_receipt());
        let mut review = review_receipt();
        review.disposition = None;
        review.reviewer_id = None;
        review.review_seconds = None;
        value.review = Some(review);
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::ReviewPending);
        assert!(report.unresolved_stages.contains(&"HUMAN_REVIEW"));
    }

    #[test]
    fn deferred_review_cannot_be_represented_as_completed_review_or_edit() {
        let mut value = through_split_and_prediction();
        value.view_decision = Some(view_receipt());
        let mut review = review_receipt();
        review.allocation = ReviewAllocationDecision::Deferred;
        review.reviewer_id = None;
        review.disposition = None;
        review.review_seconds = None;
        value.review = Some(review);
        value.edit_transaction = Some(edit_receipt());
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::ReviewDeferred);
        assert!(!report.completed_stages.contains(&"EDIT_COMMITTED"));
    }

    #[test]
    fn rejected_review_cannot_advance_to_edit_commit() {
        let mut value = through_split_and_prediction();
        value.view_decision = Some(view_receipt());
        let mut review = review_receipt();
        review.disposition = Some(ReviewDisposition::Rejected);
        value.review = Some(review);
        value.edit_transaction = Some(edit_receipt());
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::ReviewRejected);
    }

    #[test]
    fn committed_edit_and_stable_query_produce_scoped_observed_window_readiness() {
        let mut value = through_edit();
        value.query_stability = Some(query_receipt(
            SHA_C,
            QueryStabilityOutcome::StableOverObservedWindow,
        ));
        let report = audit_workflow(&value).unwrap();
        assert_eq!(report.status, WorkflowStatus::QueryReadyObservedWindow);
        assert_eq!(report.canonical_graph_snapshot_sha256, SHA_C);
        assert_eq!(report.query_id.as_deref(), Some("pathway-1"));
        assert!(!report.future_stability_certified);
        assert!(!report.universal_connectome_completeness_claimed);
    }

    #[test]
    fn committed_edit_without_query_audit_remains_pending() {
        let report = audit_workflow(&through_edit()).unwrap();
        assert_eq!(report.status, WorkflowStatus::QueryAuditPending);
        assert_eq!(report.canonical_graph_snapshot_sha256, SHA_C);
    }

    #[test]
    fn quarantined_or_rejected_edit_never_reaches_query_ready() {
        for (outcome, expected) in [
            (EditTransactionOutcome::Rejected, WorkflowStatus::EditRejected),
            (EditTransactionOutcome::Quarantined, WorkflowStatus::EditQuarantined),
        ] {
            let mut value = through_review();
            let mut edit = edit_receipt();
            edit.outcome = outcome;
            edit.resulting_snapshot_sha256 = None;
            edit.verification_passed = false;
            edit.annotation_propagation_verified = false;
            value.edit_transaction = Some(edit);
            value.query_stability = Some(query_receipt(
                SHA_D,
                QueryStabilityOutcome::StableOverObservedWindow,
            ));
            let report = audit_workflow(&value).unwrap();
            assert_eq!(report.status, expected);
            assert!(!report.completed_stages.contains(&"QUERY_STABILITY_AUDIT"));
        }
    }

    #[test]
    fn unstable_or_insufficient_query_audits_do_not_become_query_ready() {
        for (outcome, expected) in [
            (QueryStabilityOutcome::UnstableOverObservedWindow, WorkflowStatus::QueryUnstable),
            (QueryStabilityOutcome::InsufficientHistory, WorkflowStatus::QueryInsufficientHistory),
        ] {
            let mut value = through_edit();
            value.query_stability = Some(query_receipt(SHA_C, outcome));
            assert_eq!(audit_workflow(&value).unwrap().status, expected);
        }
    }

    #[test]
    fn query_latest_snapshot_must_match_committed_edit_result() {
        let mut value = through_edit();
        value.query_stability = Some(query_receipt(
            SHA_B,
            QueryStabilityOutcome::StableOverObservedWindow,
        ));
        assert_eq!(
            audit_workflow(&value),
            Err(EnvelopeError::QuerySnapshotMismatch)
        );
    }

    #[test]
    fn query_first_snapshot_must_match_the_base_graph_snapshot() {
        let mut value = through_edit();
        let mut query = query_receipt(
            SHA_C,
            QueryStabilityOutcome::StableOverObservedWindow,
        );
        query.first_graph_snapshot_sha256 = SHA_A.to_owned();
        value.query_stability = Some(query);
        assert_eq!(
            audit_workflow(&value),
            Err(EnvelopeError::QueryFirstSnapshotMismatch)
        );
    }

    #[test]
    fn committed_edit_must_change_the_graph_snapshot() {
        let mut value = through_review();
        let mut edit = edit_receipt();
        edit.resulting_snapshot_sha256 = Some(SHA_D.to_owned());
        value.edit_transaction = Some(edit);
        assert_eq!(
            audit_workflow(&value),
            Err(EnvelopeError::EditNoopSnapshot)
        );
    }

    #[test]
    fn query_cannot_claim_future_stability_or_universal_completeness() {
        let mut future = through_edit();
        let mut query = query_receipt(
            SHA_C,
            QueryStabilityOutcome::StableOverObservedWindow,
        );
        query.future_stability_certified = true;
        future.query_stability = Some(query);
        assert_eq!(
            audit_workflow(&future),
            Err(EnvelopeError::QueryFutureClaimForbidden)
        );

        let mut universal = through_edit();
        let mut query = query_receipt(
            SHA_C,
            QueryStabilityOutcome::StableOverObservedWindow,
        );
        query.universal_connectome_completeness_claimed = true;
        universal.query_stability = Some(query);
        assert_eq!(
            audit_workflow(&universal),
            Err(EnvelopeError::QueryUniversalClaimForbidden)
        );
    }

    #[test]
    fn dataset_operation_and_receipt_digests_are_validated() {
        let mut value = envelope();
        value.dataset.revision_commit_sha = "short".to_owned();
        assert!(matches!(
            audit_workflow(&value),
            Err(EnvelopeError::InvalidCommitSha(_))
        ));

        value = through_split_and_prediction();
        value.prediction.as_mut().unwrap().task_route = TaskRoute::FalseSplitCorrection;
        assert_eq!(
            audit_workflow(&value),
            Err(EnvelopeError::OperationMismatch {
                stage: "prediction",
                field: "task_route"
            })
        );
    }
}

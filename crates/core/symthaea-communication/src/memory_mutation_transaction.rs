// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional, content-blind memory-graph mutation receipts.
//!
//! This module proves execution semantics for a predeclared mutation plan. It
//! does not decide whether the plan is semantically correct and does not mutate
//! a production store itself.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MEMORY_MUTATION_TRANSACTION_SCHEMA_V1: &str =
    "symthaea.communication.memory-mutation-transaction.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PlannedMemoryMutationKindV1 {
    NoEffectOutsideScope,
    ShadowForCurrentScope,
    Invalidate,
    RecomputeRequired,
    SupersedeWithFreshIdentity,
    RetainIndependentBasis,
    BlockedUnknownDependency,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedMemoryMutationV1 {
    pub artifact_id: String,
    pub action: PlannedMemoryMutationKindV1,
    pub expected_before_artifact_commitment: String,
    pub fresh_identity_required: bool,
    pub external_copy_possible: bool,
}

impl PlannedMemoryMutationV1 {
    pub fn new(
        artifact_id: impl Into<String>,
        action: PlannedMemoryMutationKindV1,
        expected_before_artifact_commitment: impl Into<String>,
        fresh_identity_required: bool,
        external_copy_possible: bool,
    ) -> Result<Self, MemoryMutationTransactionErrorV1> {
        let artifact_id = canonical_id(artifact_id.into())?;
        let expected_before_artifact_commitment = expected_before_artifact_commitment.into();
        if !is_blake3_commitment(&expected_before_artifact_commitment) {
            return Err(MemoryMutationTransactionErrorV1::InvalidArtifactCommitment);
        }
        let action_requires_fresh = matches!(
            action,
            PlannedMemoryMutationKindV1::RecomputeRequired
                | PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity
        );
        if fresh_identity_required != action_requires_fresh {
            return Err(MemoryMutationTransactionErrorV1::InvalidFreshIdentityPolicy);
        }
        Ok(Self {
            artifact_id,
            action,
            expected_before_artifact_commitment,
            fresh_identity_required,
            external_copy_possible,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMutationPlanRefV1 {
    pub operation_id: String,
    pub operation_epoch: u64,
    pub plan_schema: String,
    pub plan_commitment: String,
    pub expected_before_snapshot_commitment: String,
    pub planned: BTreeMap<String, PlannedMemoryMutationV1>,
    pub reference_commitment: String,
}

impl MemoryMutationPlanRefV1 {
    pub fn new(
        operation_id: impl Into<String>,
        operation_epoch: u64,
        plan_schema: impl Into<String>,
        plan_commitment: impl Into<String>,
        expected_before_snapshot_commitment: impl Into<String>,
        planned: Vec<PlannedMemoryMutationV1>,
    ) -> Result<Self, MemoryMutationTransactionErrorV1> {
        let operation_id = canonical_id(operation_id.into())?;
        if operation_epoch == 0 {
            return Err(MemoryMutationTransactionErrorV1::InvalidOperationEpoch);
        }
        let plan_schema = canonical_key(plan_schema.into())?;
        let plan_commitment = plan_commitment.into();
        let expected_before_snapshot_commitment = expected_before_snapshot_commitment.into();
        if !is_blake3_commitment(&plan_commitment)
            || !is_blake3_commitment(&expected_before_snapshot_commitment)
        {
            return Err(MemoryMutationTransactionErrorV1::InvalidPlanCommitment);
        }
        if planned.is_empty() {
            return Err(MemoryMutationTransactionErrorV1::EmptyPlan);
        }
        let mut planned_map = BTreeMap::new();
        for mutation in planned {
            let key = mutation.artifact_id.clone();
            if planned_map.insert(key, mutation).is_some() {
                return Err(MemoryMutationTransactionErrorV1::DuplicatePlannedArtifact);
            }
        }
        let mut value = Self {
            operation_id,
            operation_epoch,
            plan_schema,
            plan_commitment,
            expected_before_snapshot_commitment,
            planned: planned_map,
            reference_commitment: String::new(),
        };
        value.reference_commitment = plan_reference_commitment(&value);
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), MemoryMutationTransactionErrorV1> {
        canonical_id(self.operation_id.clone())?;
        canonical_key(self.plan_schema.clone())?;
        if self.operation_epoch == 0 {
            return Err(MemoryMutationTransactionErrorV1::InvalidOperationEpoch);
        }
        if self.planned.is_empty() {
            return Err(MemoryMutationTransactionErrorV1::EmptyPlan);
        }
        if !is_blake3_commitment(&self.plan_commitment)
            || !is_blake3_commitment(&self.expected_before_snapshot_commitment)
            || !is_blake3_commitment(&self.reference_commitment)
        {
            return Err(MemoryMutationTransactionErrorV1::InvalidPlanCommitment);
        }
        for (artifact_id, mutation) in &self.planned {
            if artifact_id != &mutation.artifact_id {
                return Err(MemoryMutationTransactionErrorV1::PlannedArtifactKeyMismatch);
            }
            PlannedMemoryMutationV1::new(
                mutation.artifact_id.clone(),
                mutation.action,
                mutation.expected_before_artifact_commitment.clone(),
                mutation.fresh_identity_required,
                mutation.external_copy_possible,
            )?;
        }
        if self.reference_commitment != plan_reference_commitment(self) {
            return Err(MemoryMutationTransactionErrorV1::PlanReferenceCommitmentMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppliedMutationDispositionV1 {
    Applied,
    VerifiedNoEffect,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedMemoryMutationV1 {
    pub artifact_id: String,
    pub action: PlannedMemoryMutationKindV1,
    pub disposition: AppliedMutationDispositionV1,
    pub before_artifact_commitment: String,
    pub after_artifact_id: Option<String>,
    pub after_artifact_commitment: Option<String>,
    pub external_deletion_unproven: bool,
}

impl AppliedMemoryMutationV1 {
    pub fn new(
        artifact_id: impl Into<String>,
        action: PlannedMemoryMutationKindV1,
        disposition: AppliedMutationDispositionV1,
        before_artifact_commitment: impl Into<String>,
        after_artifact_id: Option<String>,
        after_artifact_commitment: Option<String>,
        external_deletion_unproven: bool,
    ) -> Result<Self, MemoryMutationTransactionErrorV1> {
        let artifact_id = canonical_id(artifact_id.into())?;
        let before_artifact_commitment = before_artifact_commitment.into();
        if !is_blake3_commitment(&before_artifact_commitment) {
            return Err(MemoryMutationTransactionErrorV1::InvalidArtifactCommitment);
        }
        let after_artifact_id = after_artifact_id
            .map(canonical_id)
            .transpose()?;
        if let Some(commitment) = &after_artifact_commitment {
            if !is_blake3_commitment(commitment) {
                return Err(MemoryMutationTransactionErrorV1::InvalidArtifactCommitment);
            }
        }
        let value = Self {
            artifact_id,
            action,
            disposition,
            before_artifact_commitment,
            after_artifact_id,
            after_artifact_commitment,
            external_deletion_unproven,
        };
        validate_applied_shape(&value)?;
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryMutationTransactionOutcomeV1 {
    Committed,
    AbortedRolledBack,
    AbortedRollbackUnproven,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMutationAbortEvidenceV1 {
    pub failure_artifact_id: Option<String>,
    pub failure_reason_ref: String,
    pub attempted_artifact_ids: BTreeSet<String>,
}

impl MemoryMutationAbortEvidenceV1 {
    pub fn new(
        failure_artifact_id: Option<String>,
        failure_reason_ref: impl Into<String>,
        attempted_artifact_ids: impl IntoIterator<Item = String>,
    ) -> Result<Self, MemoryMutationTransactionErrorV1> {
        let failure_artifact_id = failure_artifact_id.map(canonical_id).transpose()?;
        let failure_reason_ref = canonical_ref(failure_reason_ref.into())?;
        let mut attempted = BTreeSet::new();
        for artifact_id in attempted_artifact_ids {
            attempted.insert(canonical_id(artifact_id)?);
        }
        Ok(Self {
            failure_artifact_id,
            failure_reason_ref,
            attempted_artifact_ids: attempted,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMutationTransactionReceiptV1 {
    pub schema: String,
    pub transaction_id: String,
    pub plan_reference_commitment: String,
    pub before_snapshot_commitment: String,
    pub after_snapshot_commitment: String,
    pub started_at_ns: u64,
    pub completed_at_ns: u64,
    pub outcome: MemoryMutationTransactionOutcomeV1,
    pub applied: BTreeMap<String, AppliedMemoryMutationV1>,
    pub abort: Option<MemoryMutationAbortEvidenceV1>,
    pub receipt_commitment: String,
}

impl MemoryMutationTransactionReceiptV1 {
    pub fn committed(
        transaction_id: impl Into<String>,
        plan: &MemoryMutationPlanRefV1,
        live_before_snapshot_commitment: impl Into<String>,
        after_snapshot_commitment: impl Into<String>,
        started_at_ns: u64,
        completed_at_ns: u64,
        applied: Vec<AppliedMemoryMutationV1>,
    ) -> Result<Self, MemoryMutationTransactionErrorV1> {
        plan.validate()?;
        let live_before_snapshot_commitment = live_before_snapshot_commitment.into();
        let after_snapshot_commitment = after_snapshot_commitment.into();
        if live_before_snapshot_commitment != plan.expected_before_snapshot_commitment {
            return Err(MemoryMutationTransactionErrorV1::BeforeSnapshotMismatch);
        }
        if !is_blake3_commitment(&after_snapshot_commitment) {
            return Err(MemoryMutationTransactionErrorV1::InvalidSnapshotCommitment);
        }
        validate_time_window(started_at_ns, completed_at_ns)?;
        let applied = canonicalize_applied(applied)?;
        validate_committed_results(plan, &applied, &after_snapshot_commitment)?;
        let mut receipt = Self {
            schema: MEMORY_MUTATION_TRANSACTION_SCHEMA_V1.into(),
            transaction_id: canonical_id(transaction_id.into())?,
            plan_reference_commitment: plan.reference_commitment.clone(),
            before_snapshot_commitment: live_before_snapshot_commitment,
            after_snapshot_commitment,
            started_at_ns,
            completed_at_ns,
            outcome: MemoryMutationTransactionOutcomeV1::Committed,
            applied,
            abort: None,
            receipt_commitment: String::new(),
        };
        receipt.receipt_commitment = transaction_receipt_commitment(&receipt);
        Ok(receipt)
    }

    pub fn aborted(
        transaction_id: impl Into<String>,
        plan: &MemoryMutationPlanRefV1,
        live_before_snapshot_commitment: impl Into<String>,
        after_snapshot_commitment: impl Into<String>,
        started_at_ns: u64,
        completed_at_ns: u64,
        rollback_proven: bool,
        abort: MemoryMutationAbortEvidenceV1,
    ) -> Result<Self, MemoryMutationTransactionErrorV1> {
        plan.validate()?;
        let live_before_snapshot_commitment = live_before_snapshot_commitment.into();
        let after_snapshot_commitment = after_snapshot_commitment.into();
        if live_before_snapshot_commitment != plan.expected_before_snapshot_commitment {
            return Err(MemoryMutationTransactionErrorV1::BeforeSnapshotMismatch);
        }
        if !is_blake3_commitment(&after_snapshot_commitment) {
            return Err(MemoryMutationTransactionErrorV1::InvalidSnapshotCommitment);
        }
        validate_time_window(started_at_ns, completed_at_ns)?;
        for artifact_id in &abort.attempted_artifact_ids {
            if !plan.planned.contains_key(artifact_id) {
                return Err(MemoryMutationTransactionErrorV1::UnexpectedAttemptedArtifact);
            }
        }
        if let Some(failure_artifact_id) = &abort.failure_artifact_id {
            if !plan.planned.contains_key(failure_artifact_id) {
                return Err(MemoryMutationTransactionErrorV1::UnexpectedAttemptedArtifact);
            }
        }
        let outcome = if rollback_proven {
            if after_snapshot_commitment != live_before_snapshot_commitment {
                return Err(MemoryMutationTransactionErrorV1::RollbackSnapshotMismatch);
            }
            MemoryMutationTransactionOutcomeV1::AbortedRolledBack
        } else {
            MemoryMutationTransactionOutcomeV1::AbortedRollbackUnproven
        };
        let mut receipt = Self {
            schema: MEMORY_MUTATION_TRANSACTION_SCHEMA_V1.into(),
            transaction_id: canonical_id(transaction_id.into())?,
            plan_reference_commitment: plan.reference_commitment.clone(),
            before_snapshot_commitment: live_before_snapshot_commitment,
            after_snapshot_commitment,
            started_at_ns,
            completed_at_ns,
            outcome,
            applied: BTreeMap::new(),
            abort: Some(abort),
            receipt_commitment: String::new(),
        };
        receipt.receipt_commitment = transaction_receipt_commitment(&receipt);
        Ok(receipt)
    }

    pub fn validate(
        &self,
        plan: &MemoryMutationPlanRefV1,
    ) -> Result<(), MemoryMutationTransactionErrorV1> {
        plan.validate()?;
        if self.schema != MEMORY_MUTATION_TRANSACTION_SCHEMA_V1
            || self.plan_reference_commitment != plan.reference_commitment
            || self.before_snapshot_commitment != plan.expected_before_snapshot_commitment
        {
            return Err(MemoryMutationTransactionErrorV1::ReceiptPlanMismatch);
        }
        canonical_id(self.transaction_id.clone())?;
        if !is_blake3_commitment(&self.before_snapshot_commitment)
            || !is_blake3_commitment(&self.after_snapshot_commitment)
            || !is_blake3_commitment(&self.receipt_commitment)
        {
            return Err(MemoryMutationTransactionErrorV1::InvalidSnapshotCommitment);
        }
        validate_time_window(self.started_at_ns, self.completed_at_ns)?;
        match self.outcome {
            MemoryMutationTransactionOutcomeV1::Committed => {
                if self.abort.is_some() {
                    return Err(MemoryMutationTransactionErrorV1::UnexpectedAbortEvidence);
                }
                validate_committed_results(plan, &self.applied, &self.after_snapshot_commitment)?;
            }
            MemoryMutationTransactionOutcomeV1::AbortedRolledBack => {
                if !self.applied.is_empty() || self.abort.is_none() {
                    return Err(MemoryMutationTransactionErrorV1::InvalidAbortReceipt);
                }
                if self.after_snapshot_commitment != self.before_snapshot_commitment {
                    return Err(MemoryMutationTransactionErrorV1::RollbackSnapshotMismatch);
                }
                validate_abort_against_plan(plan, self.abort.as_ref().expect("checked above"))?;
            }
            MemoryMutationTransactionOutcomeV1::AbortedRollbackUnproven => {
                if !self.applied.is_empty() || self.abort.is_none() {
                    return Err(MemoryMutationTransactionErrorV1::InvalidAbortReceipt);
                }
                validate_abort_against_plan(plan, self.abort.as_ref().expect("checked above"))?;
            }
        }
        if self.receipt_commitment != transaction_receipt_commitment(self) {
            return Err(MemoryMutationTransactionErrorV1::ReceiptCommitmentMismatch);
        }
        Ok(())
    }
}

fn canonicalize_applied(
    applied: Vec<AppliedMemoryMutationV1>,
) -> Result<BTreeMap<String, AppliedMemoryMutationV1>, MemoryMutationTransactionErrorV1> {
    let mut map = BTreeMap::new();
    for value in applied {
        validate_applied_shape(&value)?;
        let artifact_id = value.artifact_id.clone();
        if map.insert(artifact_id, value).is_some() {
            return Err(MemoryMutationTransactionErrorV1::DuplicateAppliedArtifact);
        }
    }
    Ok(map)
}

fn validate_committed_results(
    plan: &MemoryMutationPlanRefV1,
    applied: &BTreeMap<String, AppliedMemoryMutationV1>,
    after_snapshot_commitment: &str,
) -> Result<(), MemoryMutationTransactionErrorV1> {
    if applied.len() != plan.planned.len() {
        return Err(MemoryMutationTransactionErrorV1::IncompleteCommittedResultSet);
    }
    let mut graph_must_change = false;
    for (artifact_id, planned) in &plan.planned {
        let result = applied
            .get(artifact_id)
            .ok_or(MemoryMutationTransactionErrorV1::MissingAppliedArtifact)?;
        if result.action != planned.action {
            return Err(MemoryMutationTransactionErrorV1::ActionSubstitution);
        }
        if result.before_artifact_commitment != planned.expected_before_artifact_commitment {
            return Err(MemoryMutationTransactionErrorV1::BeforeArtifactCommitmentMismatch);
        }
        validate_result_against_plan(planned, result)?;
        if action_changes_graph(planned.action) {
            graph_must_change = true;
        }
    }
    for artifact_id in applied.keys() {
        if !plan.planned.contains_key(artifact_id) {
            return Err(MemoryMutationTransactionErrorV1::UnexpectedAppliedArtifact);
        }
    }
    if graph_must_change && after_snapshot_commitment == plan.expected_before_snapshot_commitment {
        return Err(MemoryMutationTransactionErrorV1::MutatingCommitPreservedSnapshot);
    }
    Ok(())
}

fn validate_result_against_plan(
    planned: &PlannedMemoryMutationV1,
    result: &AppliedMemoryMutationV1,
) -> Result<(), MemoryMutationTransactionErrorV1> {
    validate_applied_shape(result)?;
    let destructive = matches!(
        planned.action,
        PlannedMemoryMutationKindV1::Invalidate
            | PlannedMemoryMutationKindV1::RecomputeRequired
            | PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity
    );
    if planned.external_copy_possible && destructive && !result.external_deletion_unproven {
        return Err(MemoryMutationTransactionErrorV1::ExternalDeletionLimitationLost);
    }
    if !planned.external_copy_possible && result.external_deletion_unproven {
        return Err(MemoryMutationTransactionErrorV1::UnexpectedExternalDeletionLimitation);
    }
    match planned.action {
        PlannedMemoryMutationKindV1::NoEffectOutsideScope
        | PlannedMemoryMutationKindV1::RetainIndependentBasis => {
            if result.disposition != AppliedMutationDispositionV1::VerifiedNoEffect
                || result.after_artifact_id.as_deref() != Some(planned.artifact_id.as_str())
                || result.after_artifact_commitment.as_deref()
                    != Some(planned.expected_before_artifact_commitment.as_str())
            {
                return Err(MemoryMutationTransactionErrorV1::NoEffectPostconditionFailed);
            }
        }
        PlannedMemoryMutationKindV1::BlockedUnknownDependency => {
            if result.disposition != AppliedMutationDispositionV1::Blocked
                || result.after_artifact_id.as_deref() != Some(planned.artifact_id.as_str())
                || result.after_artifact_commitment.as_deref()
                    != Some(planned.expected_before_artifact_commitment.as_str())
            {
                return Err(MemoryMutationTransactionErrorV1::BlockedPostconditionFailed);
            }
        }
        PlannedMemoryMutationKindV1::ShadowForCurrentScope => {
            if result.disposition != AppliedMutationDispositionV1::Applied
                || result.after_artifact_id.as_deref() != Some(planned.artifact_id.as_str())
                || result.after_artifact_commitment.as_deref()
                    != Some(planned.expected_before_artifact_commitment.as_str())
            {
                return Err(MemoryMutationTransactionErrorV1::ShadowPostconditionFailed);
            }
        }
        PlannedMemoryMutationKindV1::Invalidate => {
            if result.disposition != AppliedMutationDispositionV1::Applied
                || result.after_artifact_id.is_some()
                || result.after_artifact_commitment.is_some()
            {
                return Err(MemoryMutationTransactionErrorV1::InvalidatePostconditionFailed);
            }
        }
        PlannedMemoryMutationKindV1::RecomputeRequired
        | PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity => {
            if result.disposition != AppliedMutationDispositionV1::Applied {
                return Err(MemoryMutationTransactionErrorV1::ReplacementPostconditionFailed);
            }
            let after_id = result
                .after_artifact_id
                .as_ref()
                .ok_or(MemoryMutationTransactionErrorV1::FreshIdentityMissing)?;
            let after_commitment = result
                .after_artifact_commitment
                .as_ref()
                .ok_or(MemoryMutationTransactionErrorV1::ReplacementCommitmentMissing)?;
            if after_id == &planned.artifact_id {
                return Err(MemoryMutationTransactionErrorV1::FreshIdentityReused);
            }
            if after_commitment == &planned.expected_before_artifact_commitment {
                return Err(MemoryMutationTransactionErrorV1::ReplacementContentUnchanged);
            }
        }
    }
    Ok(())
}

fn validate_applied_shape(
    value: &AppliedMemoryMutationV1,
) -> Result<(), MemoryMutationTransactionErrorV1> {
    canonical_id(value.artifact_id.clone())?;
    if !is_blake3_commitment(&value.before_artifact_commitment) {
        return Err(MemoryMutationTransactionErrorV1::InvalidArtifactCommitment);
    }
    if value.after_artifact_id.is_some() != value.after_artifact_commitment.is_some() {
        return Err(MemoryMutationTransactionErrorV1::IncompleteAfterArtifact);
    }
    if let Some(after_id) = &value.after_artifact_id {
        canonical_id(after_id.clone())?;
    }
    if let Some(after_commitment) = &value.after_artifact_commitment {
        if !is_blake3_commitment(after_commitment) {
            return Err(MemoryMutationTransactionErrorV1::InvalidArtifactCommitment);
        }
    }
    Ok(())
}

fn validate_abort_against_plan(
    plan: &MemoryMutationPlanRefV1,
    abort: &MemoryMutationAbortEvidenceV1,
) -> Result<(), MemoryMutationTransactionErrorV1> {
    canonical_ref(abort.failure_reason_ref.clone())?;
    for artifact_id in &abort.attempted_artifact_ids {
        if !plan.planned.contains_key(artifact_id) {
            return Err(MemoryMutationTransactionErrorV1::UnexpectedAttemptedArtifact);
        }
    }
    if let Some(failure_artifact_id) = &abort.failure_artifact_id {
        if !plan.planned.contains_key(failure_artifact_id) {
            return Err(MemoryMutationTransactionErrorV1::UnexpectedAttemptedArtifact);
        }
    }
    Ok(())
}

fn action_changes_graph(action: PlannedMemoryMutationKindV1) -> bool {
    matches!(
        action,
        PlannedMemoryMutationKindV1::ShadowForCurrentScope
            | PlannedMemoryMutationKindV1::Invalidate
            | PlannedMemoryMutationKindV1::RecomputeRequired
            | PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity
    )
}

fn validate_time_window(
    started_at_ns: u64,
    completed_at_ns: u64,
) -> Result<(), MemoryMutationTransactionErrorV1> {
    if started_at_ns == 0 || completed_at_ns < started_at_ns {
        return Err(MemoryMutationTransactionErrorV1::InvalidTimeWindow);
    }
    Ok(())
}

fn plan_reference_commitment(plan: &MemoryMutationPlanRefV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-memory-mutation-plan-ref-v1");
    put_str(&mut hasher, &plan.operation_id);
    put_u64(&mut hasher, plan.operation_epoch);
    put_str(&mut hasher, &plan.plan_schema);
    put_str(&mut hasher, &plan.plan_commitment);
    put_str(&mut hasher, &plan.expected_before_snapshot_commitment);
    put_u64(&mut hasher, plan.planned.len() as u64);
    for mutation in plan.planned.values() {
        put_str(&mut hasher, &mutation.artifact_id);
        put_u8(&mut hasher, action_tag(mutation.action));
        put_str(&mut hasher, &mutation.expected_before_artifact_commitment);
        put_bool(&mut hasher, mutation.fresh_identity_required);
        put_bool(&mut hasher, mutation.external_copy_possible);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn transaction_receipt_commitment(receipt: &MemoryMutationTransactionReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-memory-mutation-transaction-receipt-v1");
    put_str(&mut hasher, &receipt.schema);
    put_str(&mut hasher, &receipt.transaction_id);
    put_str(&mut hasher, &receipt.plan_reference_commitment);
    put_str(&mut hasher, &receipt.before_snapshot_commitment);
    put_str(&mut hasher, &receipt.after_snapshot_commitment);
    put_u64(&mut hasher, receipt.started_at_ns);
    put_u64(&mut hasher, receipt.completed_at_ns);
    put_u8(&mut hasher, outcome_tag(receipt.outcome));
    put_u64(&mut hasher, receipt.applied.len() as u64);
    for value in receipt.applied.values() {
        put_str(&mut hasher, &value.artifact_id);
        put_u8(&mut hasher, action_tag(value.action));
        put_u8(&mut hasher, disposition_tag(value.disposition));
        put_str(&mut hasher, &value.before_artifact_commitment);
        put_opt_str(&mut hasher, value.after_artifact_id.as_deref());
        put_opt_str(&mut hasher, value.after_artifact_commitment.as_deref());
        put_bool(&mut hasher, value.external_deletion_unproven);
    }
    match &receipt.abort {
        Some(abort) => {
            put_bool(&mut hasher, true);
            put_opt_str(&mut hasher, abort.failure_artifact_id.as_deref());
            put_str(&mut hasher, &abort.failure_reason_ref);
            put_u64(&mut hasher, abort.attempted_artifact_ids.len() as u64);
            for artifact_id in &abort.attempted_artifact_ids {
                put_str(&mut hasher, artifact_id);
            }
        }
        None => put_bool(&mut hasher, false),
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_id(value: String) -> Result<String, MemoryMutationTransactionErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 192 {
        return Err(MemoryMutationTransactionErrorV1::InvalidIdentifier);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(MemoryMutationTransactionErrorV1::InvalidIdentifier);
    }
    Ok(value)
}

fn canonical_key(value: String) -> Result<String, MemoryMutationTransactionErrorV1> {
    canonical_id(value.to_ascii_lowercase())
}

fn canonical_ref(value: String) -> Result<String, MemoryMutationTransactionErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 512 {
        return Err(MemoryMutationTransactionErrorV1::InvalidReference);
    }
    Ok(value)
}

fn is_blake3_commitment(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn put_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn put_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_bool(hasher: &mut blake3::Hasher, value: bool) {
    put_u8(hasher, u8::from(value));
}

fn put_opt_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            put_bool(hasher, true);
            put_str(hasher, value);
        }
        None => put_bool(hasher, false),
    }
}

fn action_tag(value: PlannedMemoryMutationKindV1) -> u8 {
    match value {
        PlannedMemoryMutationKindV1::NoEffectOutsideScope => 0,
        PlannedMemoryMutationKindV1::ShadowForCurrentScope => 1,
        PlannedMemoryMutationKindV1::Invalidate => 2,
        PlannedMemoryMutationKindV1::RecomputeRequired => 3,
        PlannedMemoryMutationKindV1::SupersedeWithFreshIdentity => 4,
        PlannedMemoryMutationKindV1::RetainIndependentBasis => 5,
        PlannedMemoryMutationKindV1::BlockedUnknownDependency => 6,
    }
}

fn disposition_tag(value: AppliedMutationDispositionV1) -> u8 {
    match value {
        AppliedMutationDispositionV1::Applied => 0,
        AppliedMutationDispositionV1::VerifiedNoEffect => 1,
        AppliedMutationDispositionV1::Blocked => 2,
    }
}

fn outcome_tag(value: MemoryMutationTransactionOutcomeV1) -> u8 {
    match value {
        MemoryMutationTransactionOutcomeV1::Committed => 0,
        MemoryMutationTransactionOutcomeV1::AbortedRolledBack => 1,
        MemoryMutationTransactionOutcomeV1::AbortedRollbackUnproven => 2,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryMutationTransactionErrorV1 {
    InvalidIdentifier,
    InvalidReference,
    InvalidOperationEpoch,
    InvalidPlanCommitment,
    InvalidArtifactCommitment,
    InvalidSnapshotCommitment,
    InvalidFreshIdentityPolicy,
    EmptyPlan,
    DuplicatePlannedArtifact,
    PlannedArtifactKeyMismatch,
    PlanReferenceCommitmentMismatch,
    BeforeSnapshotMismatch,
    DuplicateAppliedArtifact,
    MissingAppliedArtifact,
    UnexpectedAppliedArtifact,
    IncompleteCommittedResultSet,
    ActionSubstitution,
    BeforeArtifactCommitmentMismatch,
    IncompleteAfterArtifact,
    NoEffectPostconditionFailed,
    BlockedPostconditionFailed,
    ShadowPostconditionFailed,
    InvalidatePostconditionFailed,
    ReplacementPostconditionFailed,
    FreshIdentityMissing,
    FreshIdentityReused,
    ReplacementCommitmentMissing,
    ReplacementContentUnchanged,
    ExternalDeletionLimitationLost,
    UnexpectedExternalDeletionLimitation,
    MutatingCommitPreservedSnapshot,
    InvalidTimeWindow,
    UnexpectedAttemptedArtifact,
    RollbackSnapshotMismatch,
    UnexpectedAbortEvidence,
    InvalidAbortReceipt,
    ReceiptPlanMismatch,
    ReceiptCommitmentMismatch,
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Deterministic Forge validation bound to live pre-release cgroup verification.
//!
//! This bridge does not implement another scorer. It carries the strongest current evaluator
//! containment theorem into validation: exact cgroup limits and membership were re-read while the
//! model remained blocked, the launcher then re-observed the same process identity before release,
//! and terminal cgroup cleanup was verified before deterministic validation evidence is minted.

use serde::Serialize;
use symthaea_algorithms::ContentId;
use symthaea_forge::{
    ForgeProposalCorpusManifest, ForgeProposalEvaluationRequest, ForgeProposalExecutableModelBinding,
    ForgeProposalFitPermit, ForgeProposalFrozenModel, ForgeProposalHoldoutSeal,
    ForgeProposalMetricScore, ForgeProposalStudySpec, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationGateSpec,
    ForgeProposalValidationScorePermit, ForgeProposalValidationSet, ForgeProposalValidationTargetSet,
};
use symthaea_forge_kernel_validation::{
    KernelGatedModelHoldoutPermit, KernelGatedModelValidationReceipt, KernelGatedValidationError,
};
use symthaea_forge_linux_cgroup_control::CgroupV2ResourcePolicy;
use symthaea_forge_linux_cgroup_live_verification::{
    LiveCgroupGatedRun, LiveCgroupVerificationError,
};
use symthaea_forge_linux_observed_exec::KernelGatedEvaluatorPolicy;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LiveCgroupGatedValidationError {
    #[error(transparent)]
    LiveCgroup(#[from] LiveCgroupVerificationError),
    #[error(transparent)]
    Kernel(#[from] KernelGatedValidationError),
    #[error("live-cgroup validation receipt does not bind supplied execution/validation evidence")]
    ValidationScopeMismatch,
    #[error("live-cgroup validation receipt identity is non-canonical")]
    ValidationIdentityMismatch,
    #[error("live-cgroup holdout permit does not bind supplied execution/validation evidence")]
    HoldoutScopeMismatch,
    #[error("live-cgroup holdout permit identity is non-canonical")]
    HoldoutIdentityMismatch,
}

/// Deterministic validation evidence retaining the exact live cgroup read-back lineage.
#[derive(Debug, Clone, Serialize)]
pub struct LiveCgroupGatedModelValidationReceipt {
    id: ContentId,
    live_execution_receipt_id: ContentId,
    live_verification_receipt_id: ContentId,
    kernel_validation_receipt: KernelGatedModelValidationReceipt,
}

impl LiveCgroupGatedModelValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
        live_run: &LiveCgroupGatedRun,
        cgroup_policy: &CgroupV2ResourcePolicy,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
    ) -> Result<Self, LiveCgroupGatedValidationError> {
        validate_live_execution(live_run, cgroup_policy, kernel_policy, binding, model, request)?;
        let kernel_validation_receipt = KernelGatedModelValidationReceipt::record_brier(
            kernel_policy,
            live_run.execution(),
            live_run.observation(),
            live_run.gate(),
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            live_run.response(),
        )?;
        let id = derive_validation_id(
            live_run.receipt().id(),
            live_run.live_receipt().id(),
            kernel_validation_receipt.id(),
        );
        Ok(Self {
            id,
            live_execution_receipt_id: live_run.receipt().id().clone(),
            live_verification_receipt_id: live_run.live_receipt().id().clone(),
            kernel_validation_receipt,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn live_execution_receipt_id(&self) -> &ContentId { &self.live_execution_receipt_id }
    pub fn live_verification_receipt_id(&self) -> &ContentId {
        &self.live_verification_receipt_id
    }
    pub fn kernel_validation_receipt_id(&self) -> &ContentId {
        self.kernel_validation_receipt.id()
    }
    pub fn model_id(&self) -> &ContentId { self.kernel_validation_receipt.model_id() }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore {
        self.kernel_validation_receipt.primary_score()
    }
    pub fn passed(&self) -> bool { self.kernel_validation_receipt.passed() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        live_run: &LiveCgroupGatedRun,
        cgroup_policy: &CgroupV2ResourcePolicy,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
    ) -> Result<(), LiveCgroupGatedValidationError> {
        validate_live_execution(live_run, cgroup_policy, kernel_policy, binding, model, request)?;
        self.kernel_validation_receipt.validate_for(
            kernel_policy,
            live_run.execution(),
            live_run.observation(),
            live_run.gate(),
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            live_run.response(),
        )?;
        if self.live_execution_receipt_id != *live_run.receipt().id()
            || self.live_verification_receipt_id != *live_run.live_receipt().id()
        {
            return Err(LiveCgroupGatedValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_id(
            &self.live_execution_receipt_id,
            &self.live_verification_receipt_id,
            self.kernel_validation_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(LiveCgroupGatedValidationError::ValidationIdentityMismatch)
        }
    }

    fn kernel_validation(&self) -> &KernelGatedModelValidationReceipt {
        &self.kernel_validation_receipt
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_live_execution(
    live_run: &LiveCgroupGatedRun,
    cgroup_policy: &CgroupV2ResourcePolicy,
    kernel_policy: &KernelGatedEvaluatorPolicy,
    binding: &ForgeProposalExecutableModelBinding,
    model: &ForgeProposalFrozenModel,
    request: &ForgeProposalEvaluationRequest,
) -> Result<(), LiveCgroupGatedValidationError> {
    live_run.receipt().validate_for(
        kernel_policy,
        binding,
        model,
        request,
        live_run.response(),
        live_run.observation(),
        live_run.gate(),
        live_run.execution(),
        live_run.admission(),
        cgroup_policy,
        live_run.base_receipt(),
        live_run.strict_receipt(),
        live_run.live_receipt(),
        live_run.teardown_receipt(),
    )?;
    Ok(())
}

fn derive_validation_id(
    live_execution_id: &ContentId,
    live_verification_id: &ContentId,
    kernel_validation_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-live-cgroup-gated-model-validation-receipt.v1",
        [
            live_execution_id.as_str().as_bytes(),
            live_verification_id.as_str().as_bytes(),
            kernel_validation_id.as_str().as_bytes(),
        ],
    )
}

/// Identity-only holdout prerequisite retaining the exact live cgroup validation lineage.
#[derive(Debug, Clone, Serialize)]
pub struct LiveCgroupGatedModelHoldoutPermit {
    id: ContentId,
    live_validation_receipt_id: ContentId,
    live_execution_receipt_id: ContentId,
    live_verification_receipt_id: ContentId,
    inner: KernelGatedModelHoldoutPermit,
}

impl LiveCgroupGatedModelHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        live_run: &LiveCgroupGatedRun,
        cgroup_policy: &CgroupV2ResourcePolicy,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        receipt: &LiveCgroupGatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, LiveCgroupGatedValidationError> {
        receipt.validate_for(
            live_run,
            cgroup_policy,
            kernel_policy,
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
        )?;
        let inner = KernelGatedModelHoldoutPermit::issue(
            kernel_policy,
            live_run.execution(),
            live_run.observation(),
            live_run.gate(),
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            live_run.response(),
            receipt.kernel_validation(),
            holdout,
        )?;
        let id = derive_holdout_id(
            inner.id(),
            receipt.id(),
            live_run.receipt().id(),
            live_run.live_receipt().id(),
        );
        Ok(Self {
            id,
            live_validation_receipt_id: receipt.id().clone(),
            live_execution_receipt_id: live_run.receipt().id().clone(),
            live_verification_receipt_id: live_run.live_receipt().id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn live_validation_receipt_id(&self) -> &ContentId { &self.live_validation_receipt_id }
    pub fn live_execution_receipt_id(&self) -> &ContentId { &self.live_execution_receipt_id }
    pub fn live_verification_receipt_id(&self) -> &ContentId {
        &self.live_verification_receipt_id
    }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        live_run: &LiveCgroupGatedRun,
        cgroup_policy: &CgroupV2ResourcePolicy,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        receipt: &LiveCgroupGatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), LiveCgroupGatedValidationError> {
        receipt.validate_for(
            live_run,
            cgroup_policy,
            kernel_policy,
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
        )?;
        self.inner.validate_for(
            kernel_policy,
            live_run.execution(),
            live_run.observation(),
            live_run.gate(),
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            live_run.response(),
            receipt.kernel_validation(),
            holdout,
        )?;
        if self.live_validation_receipt_id != *receipt.id()
            || self.live_execution_receipt_id != *live_run.receipt().id()
            || self.live_verification_receipt_id != *live_run.live_receipt().id()
            || self.inner.model_id() != model.id()
            || self.inner.holdout_seal_id() != holdout.id()
        {
            return Err(LiveCgroupGatedValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_id(
            self.inner.id(),
            &self.live_validation_receipt_id,
            &self.live_execution_receipt_id,
            &self.live_verification_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(LiveCgroupGatedValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_id(
    inner_id: &ContentId,
    validation_receipt_id: &ContentId,
    live_execution_id: &ContentId,
    live_verification_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-live-cgroup-gated-model-holdout-permit.v1",
        [
            inner_id.as_str().as_bytes(),
            validation_receipt_id.as_str().as_bytes(),
            live_execution_id.as_str().as_bytes(),
            live_verification_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_identity_changes_with_live_verification() {
        let execution = ContentId::derive("test-live-exec", [b"exec".as_slice()]);
        let live_a = ContentId::derive("test-live-verification", [b"a".as_slice()]);
        let live_b = ContentId::derive("test-live-verification", [b"b".as_slice()]);
        let kernel = ContentId::derive("test-kernel-validation", [b"kernel".as_slice()]);
        assert_ne!(
            derive_validation_id(&execution, &live_a, &kernel),
            derive_validation_id(&execution, &live_b, &kernel)
        );
    }
}

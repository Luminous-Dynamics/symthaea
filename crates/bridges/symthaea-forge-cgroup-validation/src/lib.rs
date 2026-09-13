// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Deterministic Forge validation bound to pre-release cgroup resource containment.
//!
//! This bridge does not implement another scorer or another sandbox. It composes the existing
//! deterministic kernel-gated validation theorem with the stronger cgroup-gated execution theorem,
//! so validation/holdout evidence can retain proof that the exact evaluator was admitted to the
//! frozen cgroup-v2 resource policy, live resource state was re-read while blocked, and the cgroup
//! reached `populated 0` before execution evidence was promoted into validation evidence.

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
use symthaea_forge_linux_cgroup_observed_exec::{
    CgroupObservedExecError, CgroupResourceGatedRun,
};
use symthaea_forge_linux_observed_exec::KernelGatedEvaluatorPolicy;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CgroupGatedValidationError {
    #[error(transparent)]
    Cgroup(#[from] CgroupObservedExecError),
    #[error(transparent)]
    Kernel(#[from] KernelGatedValidationError),
    #[error("cgroup-gated validation receipt does not bind supplied execution/validation evidence")]
    ValidationScopeMismatch,
    #[error("cgroup-gated validation receipt identity is non-canonical")]
    ValidationIdentityMismatch,
    #[error("cgroup-gated holdout permit does not bind supplied execution/validation evidence")]
    HoldoutScopeMismatch,
    #[error("cgroup-gated holdout permit identity is non-canonical")]
    HoldoutIdentityMismatch,
}

#[derive(Debug, Clone, Serialize)]
pub struct CgroupGatedModelValidationReceipt {
    id: ContentId,
    cgroup_execution_receipt_id: ContentId,
    kernel_validation_receipt: KernelGatedModelValidationReceipt,
}

impl CgroupGatedModelValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
        cgroup_run: &CgroupResourceGatedRun,
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
    ) -> Result<Self, CgroupGatedValidationError> {
        cgroup_run.receipt().validate_for(
            kernel_policy,
            binding,
            model,
            request,
            cgroup_run.response(),
            cgroup_run.observation(),
            cgroup_run.gate(),
            cgroup_run.execution(),
            cgroup_run.admission(),
            cgroup_policy,
            cgroup_run.base_receipt(),
            cgroup_run.strict_receipt(),
            cgroup_run.live_verification(),
            cgroup_run.teardown_receipt(),
        )?;
        let kernel_validation_receipt = KernelGatedModelValidationReceipt::record_brier(
            kernel_policy,
            cgroup_run.execution(),
            cgroup_run.observation(),
            cgroup_run.gate(),
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
            cgroup_run.response(),
        )?;
        let id = derive_validation_id(
            cgroup_run.receipt().id(),
            kernel_validation_receipt.id(),
        );
        Ok(Self {
            id,
            cgroup_execution_receipt_id: cgroup_run.receipt().id().clone(),
            kernel_validation_receipt,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn cgroup_execution_receipt_id(&self) -> &ContentId {
        &self.cgroup_execution_receipt_id
    }
    pub fn kernel_validation_receipt_id(&self) -> &ContentId {
        self.kernel_validation_receipt.id()
    }
    pub fn model_id(&self) -> &ContentId {
        self.kernel_validation_receipt.model_id()
    }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore {
        self.kernel_validation_receipt.primary_score()
    }
    pub fn passed(&self) -> bool {
        self.kernel_validation_receipt.passed()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        cgroup_run: &CgroupResourceGatedRun,
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
    ) -> Result<(), CgroupGatedValidationError> {
        cgroup_run.receipt().validate_for(
            kernel_policy,
            binding,
            model,
            request,
            cgroup_run.response(),
            cgroup_run.observation(),
            cgroup_run.gate(),
            cgroup_run.execution(),
            cgroup_run.admission(),
            cgroup_policy,
            cgroup_run.base_receipt(),
            cgroup_run.strict_receipt(),
            cgroup_run.live_verification(),
            cgroup_run.teardown_receipt(),
        )?;
        self.kernel_validation_receipt.validate_for(
            kernel_policy,
            cgroup_run.execution(),
            cgroup_run.observation(),
            cgroup_run.gate(),
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
            cgroup_run.response(),
        )?;
        if self.cgroup_execution_receipt_id != *cgroup_run.receipt().id() {
            return Err(CgroupGatedValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_id(
            &self.cgroup_execution_receipt_id,
            self.kernel_validation_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupGatedValidationError::ValidationIdentityMismatch)
        }
    }

    fn kernel_validation(&self) -> &KernelGatedModelValidationReceipt {
        &self.kernel_validation_receipt
    }
}

fn derive_validation_id(
    cgroup_execution_id: &ContentId,
    kernel_validation_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-gated-model-validation-receipt.v1",
        [
            cgroup_execution_id.as_str().as_bytes(),
            kernel_validation_id.as_str().as_bytes(),
        ],
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct CgroupGatedModelHoldoutPermit {
    id: ContentId,
    cgroup_validation_receipt_id: ContentId,
    cgroup_execution_receipt_id: ContentId,
    inner: KernelGatedModelHoldoutPermit,
}

impl CgroupGatedModelHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        cgroup_run: &CgroupResourceGatedRun,
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
        receipt: &CgroupGatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, CgroupGatedValidationError> {
        receipt.validate_for(
            cgroup_run,
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
            cgroup_run.execution(),
            cgroup_run.observation(),
            cgroup_run.gate(),
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
            cgroup_run.response(),
            receipt.kernel_validation(),
            holdout,
        )?;
        let id = derive_holdout_id(
            inner.id(),
            receipt.id(),
            cgroup_run.receipt().id(),
        );
        Ok(Self {
            id,
            cgroup_validation_receipt_id: receipt.id().clone(),
            cgroup_execution_receipt_id: cgroup_run.receipt().id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn cgroup_validation_receipt_id(&self) -> &ContentId {
        &self.cgroup_validation_receipt_id
    }
    pub fn cgroup_execution_receipt_id(&self) -> &ContentId {
        &self.cgroup_execution_receipt_id
    }
    pub fn model_id(&self) -> &ContentId {
        self.inner.model_id()
    }
    pub fn holdout_seal_id(&self) -> &ContentId {
        self.inner.holdout_seal_id()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        cgroup_run: &CgroupResourceGatedRun,
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
        receipt: &CgroupGatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), CgroupGatedValidationError> {
        receipt.validate_for(
            cgroup_run,
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
            cgroup_run.execution(),
            cgroup_run.observation(),
            cgroup_run.gate(),
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
            cgroup_run.response(),
            receipt.kernel_validation(),
            holdout,
        )?;
        if self.cgroup_validation_receipt_id != *receipt.id()
            || self.cgroup_execution_receipt_id != *cgroup_run.receipt().id()
            || self.inner.model_id() != model.id()
            || self.inner.holdout_seal_id() != holdout.id()
        {
            return Err(CgroupGatedValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_id(
            self.inner.id(),
            &self.cgroup_validation_receipt_id,
            &self.cgroup_execution_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupGatedValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_id(
    inner_id: &ContentId,
    validation_receipt_id: &ContentId,
    cgroup_execution_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-gated-model-holdout-permit.v1",
        [
            inner_id.as_str().as_bytes(),
            validation_receipt_id.as_str().as_bytes(),
            cgroup_execution_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_identity_changes_with_cgroup_execution() {
        let a = ContentId::derive("test-cgroup-exec", [b"a".as_slice()]);
        let b = ContentId::derive("test-cgroup-exec", [b"b".as_slice()]);
        let kernel = ContentId::derive("test-kernel-validation", [b"kernel".as_slice()]);
        assert_ne!(derive_validation_id(&a, &kernel), derive_validation_id(&b, &kernel));
    }
}

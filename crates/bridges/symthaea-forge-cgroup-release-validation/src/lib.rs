// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Deterministic Forge validation bound to release-verified cgroup containment.
//!
//! This bridge adds no scorer and no runtime authority. It composes the existing cgroup-gated
//! deterministic validation theorem with the stronger execution proposition that CPU/memory/PID,
//! no-swap/OOM, and exact cgroup membership were freshly re-observed in the launcher's final
//! admission-verification phase before model release.

use serde::Serialize;
use symthaea_algorithms::ContentId;
use symthaea_forge::{
    ForgeProposalCorpusManifest, ForgeProposalEvaluationRequest, ForgeProposalExecutableModelBinding,
    ForgeProposalFitPermit, ForgeProposalFrozenModel, ForgeProposalHoldoutSeal,
    ForgeProposalMetricScore, ForgeProposalStudySpec, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationGateSpec,
    ForgeProposalValidationScorePermit, ForgeProposalValidationSet, ForgeProposalValidationTargetSet,
};
use symthaea_forge_cgroup_validation::{
    CgroupGatedModelHoldoutPermit, CgroupGatedModelValidationReceipt, CgroupGatedValidationError,
};
use symthaea_forge_linux_cgroup_control::CgroupV2ResourcePolicy;
use symthaea_forge_linux_cgroup_observed_exec::{
    CgroupObservedExecError, CgroupResourceGatedRun,
};
use symthaea_forge_linux_observed_exec::KernelGatedEvaluatorPolicy;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CgroupReleaseVerifiedValidationError {
    #[error(transparent)]
    Execution(#[from] CgroupObservedExecError),
    #[error(transparent)]
    Validation(#[from] CgroupGatedValidationError),
    #[error("release-verified cgroup validation receipt does not bind supplied evidence")]
    ValidationScopeMismatch,
    #[error("release-verified cgroup validation receipt identity is non-canonical")]
    ValidationIdentityMismatch,
    #[error("release-verified cgroup holdout permit does not bind supplied evidence")]
    HoldoutScopeMismatch,
    #[error("release-verified cgroup holdout permit identity is non-canonical")]
    HoldoutIdentityMismatch,
}

#[derive(Debug, Clone, Serialize)]
pub struct CgroupReleaseVerifiedModelValidationReceipt {
    id: ContentId,
    release_verified_execution_receipt_id: ContentId,
    inner: CgroupGatedModelValidationReceipt,
}

impl CgroupReleaseVerifiedModelValidationReceipt {
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
    ) -> Result<Self, CgroupReleaseVerifiedValidationError> {
        validate_release_verified_execution(
            cgroup_run,
            cgroup_policy,
            kernel_policy,
            binding,
            model,
            request,
        )?;
        let inner = CgroupGatedModelValidationReceipt::record_brier(
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
        let release_verified_execution_receipt_id =
            cgroup_run.release_verified_receipt().id().clone();
        let id = derive_validation_id(&release_verified_execution_receipt_id, inner.id());
        Ok(Self {
            id,
            release_verified_execution_receipt_id,
            inner,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn release_verified_execution_receipt_id(&self) -> &ContentId {
        &self.release_verified_execution_receipt_id
    }
    pub fn cgroup_validation_receipt_id(&self) -> &ContentId {
        self.inner.id()
    }
    pub fn model_id(&self) -> &ContentId {
        self.inner.model_id()
    }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore {
        self.inner.primary_score()
    }
    pub fn passed(&self) -> bool {
        self.inner.passed()
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
    ) -> Result<(), CgroupReleaseVerifiedValidationError> {
        validate_release_verified_execution(
            cgroup_run,
            cgroup_policy,
            kernel_policy,
            binding,
            model,
            request,
        )?;
        self.inner.validate_for(
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
        if self.release_verified_execution_receipt_id
            != *cgroup_run.release_verified_receipt().id()
        {
            return Err(CgroupReleaseVerifiedValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_id(
            &self.release_verified_execution_receipt_id,
            self.inner.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupReleaseVerifiedValidationError::ValidationIdentityMismatch)
        }
    }

    fn inner(&self) -> &CgroupGatedModelValidationReceipt {
        &self.inner
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CgroupReleaseVerifiedModelHoldoutPermit {
    id: ContentId,
    release_verified_validation_receipt_id: ContentId,
    release_verified_execution_receipt_id: ContentId,
    inner: CgroupGatedModelHoldoutPermit,
}

impl CgroupReleaseVerifiedModelHoldoutPermit {
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
        receipt: &CgroupReleaseVerifiedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, CgroupReleaseVerifiedValidationError> {
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
        let inner = CgroupGatedModelHoldoutPermit::issue(
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
            receipt.inner(),
            holdout,
        )?;
        let release_verified_execution_receipt_id =
            cgroup_run.release_verified_receipt().id().clone();
        let id = derive_holdout_id(
            inner.id(),
            receipt.id(),
            &release_verified_execution_receipt_id,
        );
        Ok(Self {
            id,
            release_verified_validation_receipt_id: receipt.id().clone(),
            release_verified_execution_receipt_id,
            inner,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn release_verified_validation_receipt_id(&self) -> &ContentId {
        &self.release_verified_validation_receipt_id
    }
    pub fn release_verified_execution_receipt_id(&self) -> &ContentId {
        &self.release_verified_execution_receipt_id
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
        receipt: &CgroupReleaseVerifiedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), CgroupReleaseVerifiedValidationError> {
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
            receipt.inner(),
            holdout,
        )?;
        if self.release_verified_validation_receipt_id != *receipt.id()
            || self.release_verified_execution_receipt_id
                != *cgroup_run.release_verified_receipt().id()
            || self.inner.model_id() != model.id()
            || self.inner.holdout_seal_id() != holdout.id()
        {
            return Err(CgroupReleaseVerifiedValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_id(
            self.inner.id(),
            &self.release_verified_validation_receipt_id,
            &self.release_verified_execution_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupReleaseVerifiedValidationError::HoldoutIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_release_verified_execution(
    cgroup_run: &CgroupResourceGatedRun,
    cgroup_policy: &CgroupV2ResourcePolicy,
    kernel_policy: &KernelGatedEvaluatorPolicy,
    binding: &ForgeProposalExecutableModelBinding,
    model: &ForgeProposalFrozenModel,
    request: &ForgeProposalEvaluationRequest,
) -> Result<(), CgroupReleaseVerifiedValidationError> {
    cgroup_run.release_verified_receipt().validate_for(
        cgroup_run.receipt(),
        cgroup_run.release_verification(),
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
        cgroup_run.teardown_receipt(),
    )?;
    Ok(())
}

fn derive_validation_id(
    release_verified_execution_id: &ContentId,
    cgroup_validation_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-release-verified-model-validation-receipt.v1",
        [
            release_verified_execution_id.as_str().as_bytes(),
            cgroup_validation_id.as_str().as_bytes(),
        ],
    )
}

fn derive_holdout_id(
    inner_id: &ContentId,
    validation_receipt_id: &ContentId,
    release_verified_execution_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-release-verified-model-holdout-permit.v1",
        [
            inner_id.as_str().as_bytes(),
            validation_receipt_id.as_str().as_bytes(),
            release_verified_execution_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_identity_changes_with_release_verified_execution() {
        let a = ContentId::derive("test-release-execution", [b"a".as_slice()]);
        let b = ContentId::derive("test-release-execution", [b"b".as_slice()]);
        let validation = ContentId::derive("test-cgroup-validation", [b"validation".as_slice()]);
        assert_ne!(
            derive_validation_id(&a, &validation),
            derive_validation_id(&b, &validation)
        );
    }
}

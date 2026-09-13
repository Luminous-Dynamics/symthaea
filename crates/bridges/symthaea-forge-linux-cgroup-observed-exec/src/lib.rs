// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Cgroup-v2 resource admission for Forge's parent-observed evaluator launcher.
//!
//! This bridge composes independently reviewable propositions rather than silently widening them:
//!
//! 1. `symthaea-forge-linux-observed-exec` proves the exact evaluator remained behind `block-fd`
//!    through parent-observed kernel isolation and a verified pre-release admission seam;
//! 2. `symthaea-forge-linux-cgroup-control` proves finite CPU/memory/PID limits were applied to the
//!    exact host-visible sandbox PID in a delegated cgroup-v2 leaf;
//! 3. `symthaea-forge-linux-cgroup-strict` proves swap was disabled, grouped OOM semantics were
//!    requested, and the leaf reached `populated 0` before removal;
//! 4. this bridge re-reads all of those live controls and exact process membership in the launcher's
//!    final verification hook, after post-admission kernel re-observation and before release.
//!
//! A release-verified receipt therefore means the resource controls were re-observed at the final
//! admission verification point before model execution. It does not claim exact seccomp filter
//! semantics, Landlock, VM isolation, kernel correctness, model quality, search-policy authority,
//! algorithm superiority, or promotion authority.

use serde::Serialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use symthaea_algorithms::ContentId;
use symthaea_forge::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalExecutableModelBinding, ForgeProposalFrozenModel,
};
use symthaea_forge_linux_cgroup_control::{
    apply_cgroup_v2_resource_policy, CgroupV2ResourceError, CgroupV2ResourcePolicy,
    CgroupV2ResourceReceipt,
};
use symthaea_forge_linux_cgroup_strict::{
    harden_cgroup_v2_lease, StrictCgroupV2Error, StrictCgroupV2Lease, StrictCgroupV2Receipt,
    StrictCgroupV2TeardownReceipt,
};
use symthaea_forge_linux_kernel_attestation::{KernelIsolationGate, KernelSandboxObservation};
use symthaea_forge_linux_observed_exec::{
    pre_release_admission_protocol_id, run_kernel_gated_evaluator_with_admission,
    KernelGatedEvaluatorPolicy, KernelGatedExecutionReceipt, ObservedEvaluatorError,
    PreReleaseAdmission, PreReleaseAdmissionReceipt,
};
use thiserror::Error;

const ADMISSION_TRANSPORT_ERROR: &str = "cgroup pre-release admission failed";
const CGROUP_V2_ROOT: &str = "/sys/fs/cgroup";

#[derive(Debug, Error)]
pub enum CgroupObservedExecError {
    #[error(transparent)]
    Observed(#[from] ObservedEvaluatorError),
    #[error(transparent)]
    Resource(#[from] CgroupV2ResourceError),
    #[error(transparent)]
    Strict(#[from] StrictCgroupV2Error),
    #[error("cgroup pre-release admission failed: {detail}")]
    AdmissionFailed { detail: String },
    #[error("cgroup pre-release admission failed after resource admission, with verified cgroup cleanup: {detail}")]
    AdmissionFailedAfterCleanup {
        detail: String,
        teardown_receipt: StrictCgroupV2TeardownReceipt,
    },
    #[error("cgroup admission/verification failed and descendant-wide cleanup also failed; admission={admission}; teardown={teardown}")]
    AdmissionAndTeardownFailed { admission: String, teardown: String },
    #[error("cgroup admission completed but did not retain all required live evidence")]
    AdmissionStateMissing,
    #[error("observed evaluator failed after cgroup admission: {source}")]
    ExecutionAfterAdmission {
        #[source]
        source: ObservedEvaluatorError,
        teardown_receipt: StrictCgroupV2TeardownReceipt,
    },
    #[error("observed evaluator failed after cgroup admission and descendant-wide cgroup teardown also failed; execution={execution}; teardown={teardown}")]
    ExecutionAndTeardownFailed { execution: String, teardown: String },
    #[error("cgroup release verification receipt does not bind supplied evidence")]
    ReleaseVerificationScopeMismatch,
    #[error("cgroup release verification receipt identity is non-canonical")]
    ReleaseVerificationIdentityMismatch,
    #[error("cgroup-gated execution receipt does not bind supplied evidence")]
    ReceiptScopeMismatch,
    #[error("cgroup-gated execution receipt identity is non-canonical")]
    ReceiptIdentityMismatch,
    #[error("release-verified cgroup execution receipt does not bind supplied evidence")]
    ReleaseVerifiedScopeMismatch,
    #[error("release-verified cgroup execution receipt identity is non-canonical")]
    ReleaseVerifiedIdentityMismatch,
}

pub fn cgroup_release_verification_protocol_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-release-verification-protocol.v1",
        [b"post-kernel-reobserve+memory+pids+cpu+swap+oom+cgroup-procs+proc-cgroup".as_slice()],
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CgroupReleaseVerificationReceipt {
    id: ContentId,
    protocol_id: ContentId,
    cgroup_policy_id: ContentId,
    base_receipt_id: ContentId,
    strict_receipt_id: ContentId,
    sandbox_host_pid: u32,
    observation_id: ContentId,
    gate_id: ContentId,
    leaf_path: String,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    swap_max_bytes: u64,
    oom_group: bool,
    proc_membership: String,
}

impl CgroupReleaseVerificationReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn protocol_id(&self) -> &ContentId { &self.protocol_id }
    pub fn sandbox_host_pid(&self) -> u32 { self.sandbox_host_pid }
    pub fn observation_id(&self) -> &ContentId { &self.observation_id }
    pub fn gate_id(&self) -> &ContentId { &self.gate_id }
    pub fn leaf_path(&self) -> &str { &self.leaf_path }

    pub fn validate_for(
        &self,
        cgroup_policy: &CgroupV2ResourcePolicy,
        base: &CgroupV2ResourceReceipt,
        strict: &StrictCgroupV2Receipt,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
    ) -> Result<(), CgroupObservedExecError> {
        cgroup_policy.validate()?;
        base.validate_for(cgroup_policy)?;
        strict.validate_for(cgroup_policy, base)?;
        gate.validate_for(observation).map_err(ObservedEvaluatorError::from)?;
        if self.protocol_id != cgroup_release_verification_protocol_id()
            || self.cgroup_policy_id != *cgroup_policy.id()
            || self.base_receipt_id != *base.id()
            || self.strict_receipt_id != *strict.id()
            || self.sandbox_host_pid != base.host_pid()
            || self.sandbox_host_pid != observation.host_pid()
            || self.observation_id != *observation.id()
            || self.gate_id != *gate.id()
            || self.leaf_path != base.leaf_path()
            || self.leaf_path != strict.leaf_path()
            || self.memory_max_bytes != cgroup_policy.memory_max_bytes()
            || self.pids_max != cgroup_policy.pids_max()
            || self.cpu_quota_us != cgroup_policy.cpu_quota_us()
            || self.cpu_period_us != cgroup_policy.cpu_period_us()
            || self.swap_max_bytes != 0
            || !self.oom_group
            || self.proc_membership != base.proc_membership()
        {
            return Err(CgroupObservedExecError::ReleaseVerificationScopeMismatch);
        }
        let expected = derive_release_verification_id(
            &self.protocol_id,
            &self.cgroup_policy_id,
            &self.base_receipt_id,
            &self.strict_receipt_id,
            self.sandbox_host_pid,
            &self.observation_id,
            &self.gate_id,
            &self.leaf_path,
            self.memory_max_bytes,
            self.pids_max,
            self.cpu_quota_us,
            self.cpu_period_us,
            self.swap_max_bytes,
            self.oom_group,
            &self.proc_membership,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupObservedExecError::ReleaseVerificationIdentityMismatch)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CgroupResourceGatedExecutionReceipt {
    id: ContentId,
    execution_receipt_id: ContentId,
    admission_receipt_id: ContentId,
    admission_protocol_id: ContentId,
    cgroup_policy_id: ContentId,
    base_receipt_id: ContentId,
    strict_receipt_id: ContentId,
    teardown_receipt_id: ContentId,
    sandbox_host_pid: u32,
    post_observation_id: ContentId,
    post_gate_id: ContentId,
    residual_descendant_kill_requested: bool,
}

impl CgroupResourceGatedExecutionReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn execution_receipt_id(&self) -> &ContentId { &self.execution_receipt_id }
    pub fn admission_receipt_id(&self) -> &ContentId { &self.admission_receipt_id }
    pub fn cgroup_policy_id(&self) -> &ContentId { &self.cgroup_policy_id }
    pub fn base_receipt_id(&self) -> &ContentId { &self.base_receipt_id }
    pub fn strict_receipt_id(&self) -> &ContentId { &self.strict_receipt_id }
    pub fn teardown_receipt_id(&self) -> &ContentId { &self.teardown_receipt_id }
    pub fn sandbox_host_pid(&self) -> u32 { self.sandbox_host_pid }
    pub fn residual_descendant_kill_requested(&self) -> bool {
        self.residual_descendant_kill_requested
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        observed_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        model: &ForgeProposalFrozenModel,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
        execution: &KernelGatedExecutionReceipt,
        admission: &PreReleaseAdmissionReceipt,
        cgroup_policy: &CgroupV2ResourcePolicy,
        base: &CgroupV2ResourceReceipt,
        strict: &StrictCgroupV2Receipt,
        teardown: &StrictCgroupV2TeardownReceipt,
    ) -> Result<(), CgroupObservedExecError> {
        execution.validate_for(observed_policy, binding, model, request, response, observation, gate)?;
        admission.validate_for(observation, gate)?;
        base.validate_for(cgroup_policy)?;
        strict.validate_for(cgroup_policy, base)?;
        teardown.validate_for(strict, cgroup_policy, base)?;
        if self.execution_receipt_id != *execution.id()
            || self.admission_receipt_id != *admission.id()
            || self.admission_protocol_id != pre_release_admission_protocol_id()
            || self.admission_protocol_id != *admission.protocol_id()
            || self.cgroup_policy_id != *cgroup_policy.id()
            || self.base_receipt_id != *base.id()
            || self.strict_receipt_id != *strict.id()
            || self.teardown_receipt_id != *teardown.id()
            || self.sandbox_host_pid != execution.sandbox_host_pid()
            || self.sandbox_host_pid != admission.sandbox_host_pid()
            || self.sandbox_host_pid != base.host_pid()
            || self.sandbox_host_pid != observation.host_pid()
            || self.post_observation_id != *observation.id()
            || self.post_observation_id != *admission.post_observation_id()
            || self.post_gate_id != *gate.id()
            || self.post_gate_id != *admission.post_gate_id()
            || self.residual_descendant_kill_requested != teardown.descendant_kill_requested()
        {
            return Err(CgroupObservedExecError::ReceiptScopeMismatch);
        }
        let expected = derive_execution_receipt_id(
            &self.execution_receipt_id,
            &self.admission_receipt_id,
            &self.admission_protocol_id,
            &self.cgroup_policy_id,
            &self.base_receipt_id,
            &self.strict_receipt_id,
            &self.teardown_receipt_id,
            self.sandbox_host_pid,
            &self.post_observation_id,
            &self.post_gate_id,
            self.residual_descendant_kill_requested,
        );
        if expected == self.id { Ok(()) } else { Err(CgroupObservedExecError::ReceiptIdentityMismatch) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CgroupReleaseVerifiedExecutionReceipt {
    id: ContentId,
    resource_gated_execution_receipt_id: ContentId,
    release_verification_receipt_id: ContentId,
}

impl CgroupReleaseVerifiedExecutionReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn resource_gated_execution_receipt_id(&self) -> &ContentId {
        &self.resource_gated_execution_receipt_id
    }
    pub fn release_verification_receipt_id(&self) -> &ContentId {
        &self.release_verification_receipt_id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        resource: &CgroupResourceGatedExecutionReceipt,
        verification: &CgroupReleaseVerificationReceipt,
        observed_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        model: &ForgeProposalFrozenModel,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
        execution: &KernelGatedExecutionReceipt,
        admission: &PreReleaseAdmissionReceipt,
        cgroup_policy: &CgroupV2ResourcePolicy,
        base: &CgroupV2ResourceReceipt,
        strict: &StrictCgroupV2Receipt,
        teardown: &StrictCgroupV2TeardownReceipt,
    ) -> Result<(), CgroupObservedExecError> {
        resource.validate_for(
            observed_policy, binding, model, request, response, observation, gate, execution,
            admission, cgroup_policy, base, strict, teardown,
        )?;
        verification.validate_for(cgroup_policy, base, strict, observation, gate)?;
        if self.resource_gated_execution_receipt_id != *resource.id()
            || self.release_verification_receipt_id != *verification.id()
        {
            return Err(CgroupObservedExecError::ReleaseVerifiedScopeMismatch);
        }
        let expected = derive_release_verified_execution_id(
            &self.resource_gated_execution_receipt_id,
            &self.release_verification_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupObservedExecError::ReleaseVerifiedIdentityMismatch)
        }
    }
}

pub struct CgroupResourceGatedRun {
    response: ForgeProposalEvaluationResponse,
    observation: KernelSandboxObservation,
    gate: KernelIsolationGate,
    execution: KernelGatedExecutionReceipt,
    admission: PreReleaseAdmissionReceipt,
    base_receipt: CgroupV2ResourceReceipt,
    strict_receipt: StrictCgroupV2Receipt,
    release_verification: CgroupReleaseVerificationReceipt,
    teardown_receipt: StrictCgroupV2TeardownReceipt,
    receipt: CgroupResourceGatedExecutionReceipt,
    release_verified_receipt: CgroupReleaseVerifiedExecutionReceipt,
}

impl CgroupResourceGatedRun {
    pub fn response(&self) -> &ForgeProposalEvaluationResponse { &self.response }
    pub fn observation(&self) -> &KernelSandboxObservation { &self.observation }
    pub fn gate(&self) -> &KernelIsolationGate { &self.gate }
    pub fn execution(&self) -> &KernelGatedExecutionReceipt { &self.execution }
    pub fn admission(&self) -> &PreReleaseAdmissionReceipt { &self.admission }
    pub fn base_receipt(&self) -> &CgroupV2ResourceReceipt { &self.base_receipt }
    pub fn strict_receipt(&self) -> &StrictCgroupV2Receipt { &self.strict_receipt }
    pub fn release_verification(&self) -> &CgroupReleaseVerificationReceipt {
        &self.release_verification
    }
    pub fn teardown_receipt(&self) -> &StrictCgroupV2TeardownReceipt { &self.teardown_receipt }
    pub fn receipt(&self) -> &CgroupResourceGatedExecutionReceipt { &self.receipt }
    pub fn release_verified_receipt(&self) -> &CgroupReleaseVerifiedExecutionReceipt {
        &self.release_verified_receipt
    }
}

#[derive(Debug)]
struct CgroupAdmissionState<'a> {
    policy: &'a CgroupV2ResourcePolicy,
    delegation_root: PathBuf,
    base_receipt: Option<CgroupV2ResourceReceipt>,
    strict_receipt: Option<StrictCgroupV2Receipt>,
    release_verification: Option<CgroupReleaseVerificationReceipt>,
    lease: Option<StrictCgroupV2Lease>,
    failure: Option<String>,
}

impl<'a> CgroupAdmissionState<'a> {
    fn new(policy: &'a CgroupV2ResourcePolicy, delegation_root: &Path) -> Self {
        Self {
            policy,
            delegation_root: delegation_root.to_path_buf(),
            base_receipt: None,
            strict_receipt: None,
            release_verification: None,
            lease: None,
            failure: None,
        }
    }

    fn fail(&mut self, detail: String) -> Result<(), ObservedEvaluatorError> {
        self.failure = Some(detail);
        Err(ObservedEvaluatorError::Control(io::Error::other(ADMISSION_TRANSPORT_ERROR)))
    }
}

impl PreReleaseAdmission for CgroupAdmissionState<'_> {
    fn admit(
        &mut self,
        sandbox_pid: u32,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
    ) -> Result<(), ObservedEvaluatorError> {
        if observation.host_pid() != sandbox_pid {
            return self.fail("initial kernel observation PID differs from admission PID".into());
        }
        if let Err(error) = gate.validate_for(observation) {
            return self.fail(format!("initial kernel gate revalidation failed: {error}"));
        }
        let base = match apply_cgroup_v2_resource_policy(
            self.policy,
            &self.delegation_root,
            sandbox_pid,
        ) {
            Ok(value) => value,
            Err(error) => return self.fail(format!("base cgroup admission failed: {error}")),
        };
        let base_receipt = base.receipt().clone();
        let strict = match harden_cgroup_v2_lease(self.policy, base) {
            Ok(value) => value,
            Err(error) => return self.fail(format!("strict cgroup hardening failed: {error}")),
        };
        let strict_receipt = strict.receipt().clone();
        if let Err(error) = strict_receipt.validate_for(self.policy, &base_receipt) {
            let cleanup = strict.kill_and_cleanup(30_000);
            let detail = match cleanup {
                Ok(_) => format!("strict cgroup receipt revalidation failed: {error}"),
                Err(cleanup_error) => format!(
                    "strict cgroup receipt revalidation failed: {error}; fail-closed cleanup also failed: {cleanup_error}"
                ),
            };
            return self.fail(detail);
        }
        self.base_receipt = Some(base_receipt);
        self.strict_receipt = Some(strict_receipt);
        self.lease = Some(strict);
        Ok(())
    }

    fn verify_before_release(
        &mut self,
        sandbox_pid: u32,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
    ) -> Result<(), ObservedEvaluatorError> {
        let Some(base) = self.base_receipt.clone() else {
            return self.fail("base cgroup receipt missing at final release verification".into());
        };
        let Some(strict) = self.strict_receipt.clone() else {
            return self.fail("strict cgroup receipt missing at final release verification".into());
        };
        let Some(leaf_path) = self
            .lease
            .as_ref()
            .and_then(StrictCgroupV2Lease::leaf_path)
            .map(Path::to_path_buf)
        else {
            return self.fail("live strict cgroup lease missing at final release verification".into());
        };
        if sandbox_pid != base.host_pid()
            || sandbox_pid != observation.host_pid()
            || leaf_path != PathBuf::from(base.leaf_path())
        {
            return self.fail("final cgroup verification scope differs from admitted sandbox".into());
        }
        match capture_release_verification(self.policy, &base, &strict, observation, gate, &leaf_path) {
            Ok(receipt) => {
                self.release_verification = Some(receipt);
                Ok(())
            }
            Err(detail) => self.fail(detail),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_cgroup_resource_gated_evaluator(
    observed_policy: &KernelGatedEvaluatorPolicy,
    binding: &ForgeProposalExecutableModelBinding,
    request: &ForgeProposalEvaluationRequest,
    model: &ForgeProposalFrozenModel,
    bubblewrap_executable: impl AsRef<Path>,
    model_executable: impl AsRef<Path>,
    model_args: &[String],
    cgroup_policy: &CgroupV2ResourcePolicy,
    delegation_root: impl AsRef<Path>,
) -> Result<CgroupResourceGatedRun, CgroupObservedExecError> {
    cgroup_policy.validate()?;
    let mut admission = CgroupAdmissionState::new(cgroup_policy, delegation_root.as_ref());
    let observed = run_kernel_gated_evaluator_with_admission(
        observed_policy,
        binding,
        request,
        model,
        bubblewrap_executable,
        model_executable,
        model_args,
        &mut admission,
    );

    let (response, observation, gate, execution, admission_receipt) = match observed {
        Ok(value) => value,
        Err(source) => {
            let detail = admission.failure.take();
            if let Some(lease) = admission.lease.take() {
                return match lease.kill_and_cleanup(observed_policy.teardown_timeout_ms()) {
                    Ok(teardown_receipt) => match detail {
                        Some(detail) => Err(CgroupObservedExecError::AdmissionFailedAfterCleanup {
                            detail,
                            teardown_receipt,
                        }),
                        None => Err(CgroupObservedExecError::ExecutionAfterAdmission {
                            source,
                            teardown_receipt,
                        }),
                    },
                    Err(teardown_error) => match detail {
                        Some(detail) => Err(CgroupObservedExecError::AdmissionAndTeardownFailed {
                            admission: detail,
                            teardown: teardown_error.to_string(),
                        }),
                        None => Err(CgroupObservedExecError::ExecutionAndTeardownFailed {
                            execution: source.to_string(),
                            teardown: teardown_error.to_string(),
                        }),
                    },
                };
            }
            if let Some(detail) = detail {
                return Err(CgroupObservedExecError::AdmissionFailed { detail });
            }
            return Err(CgroupObservedExecError::Observed(source));
        }
    };

    let state = (
        admission.base_receipt.take(),
        admission.strict_receipt.take(),
        admission.release_verification.take(),
        admission.lease.take(),
    );
    let (base_receipt, strict_receipt, release_verification, lease) = match state {
        (Some(base), Some(strict), Some(verification), Some(lease)) => {
            (base, strict, verification, lease)
        }
        (_, _, _, Some(lease)) => {
            return match lease.kill_and_cleanup(observed_policy.teardown_timeout_ms()) {
                Ok(_) => Err(CgroupObservedExecError::AdmissionStateMissing),
                Err(teardown_error) => Err(CgroupObservedExecError::ExecutionAndTeardownFailed {
                    execution: "release verification state missing after successful evaluator execution".into(),
                    teardown: teardown_error.to_string(),
                }),
            };
        }
        _ => return Err(CgroupObservedExecError::AdmissionStateMissing),
    };
    let teardown_receipt = lease.finalize_after_sandbox_exit(observed_policy.teardown_timeout_ms())?;

    let admission_protocol_id = pre_release_admission_protocol_id();
    let id = derive_execution_receipt_id(
        execution.id(),
        admission_receipt.id(),
        &admission_protocol_id,
        cgroup_policy.id(),
        base_receipt.id(),
        strict_receipt.id(),
        teardown_receipt.id(),
        execution.sandbox_host_pid(),
        observation.id(),
        gate.id(),
        teardown_receipt.descendant_kill_requested(),
    );
    let receipt = CgroupResourceGatedExecutionReceipt {
        id,
        execution_receipt_id: execution.id().clone(),
        admission_receipt_id: admission_receipt.id().clone(),
        admission_protocol_id,
        cgroup_policy_id: cgroup_policy.id().clone(),
        base_receipt_id: base_receipt.id().clone(),
        strict_receipt_id: strict_receipt.id().clone(),
        teardown_receipt_id: teardown_receipt.id().clone(),
        sandbox_host_pid: execution.sandbox_host_pid(),
        post_observation_id: observation.id().clone(),
        post_gate_id: gate.id().clone(),
        residual_descendant_kill_requested: teardown_receipt.descendant_kill_requested(),
    };
    receipt.validate_for(
        observed_policy,
        binding,
        model,
        request,
        &response,
        &observation,
        &gate,
        &execution,
        &admission_receipt,
        cgroup_policy,
        &base_receipt,
        &strict_receipt,
        &teardown_receipt,
    )?;
    release_verification.validate_for(cgroup_policy, &base_receipt, &strict_receipt, &observation, &gate)?;
    let release_verified_id = derive_release_verified_execution_id(receipt.id(), release_verification.id());
    let release_verified_receipt = CgroupReleaseVerifiedExecutionReceipt {
        id: release_verified_id,
        resource_gated_execution_receipt_id: receipt.id().clone(),
        release_verification_receipt_id: release_verification.id().clone(),
    };
    release_verified_receipt.validate_for(
        &receipt,
        &release_verification,
        observed_policy,
        binding,
        model,
        request,
        &response,
        &observation,
        &gate,
        &execution,
        &admission_receipt,
        cgroup_policy,
        &base_receipt,
        &strict_receipt,
        &teardown_receipt,
    )?;

    Ok(CgroupResourceGatedRun {
        response,
        observation,
        gate,
        execution,
        admission: admission_receipt,
        base_receipt,
        strict_receipt,
        release_verification,
        teardown_receipt,
        receipt,
        release_verified_receipt,
    })
}

fn capture_release_verification(
    policy: &CgroupV2ResourcePolicy,
    base: &CgroupV2ResourceReceipt,
    strict: &StrictCgroupV2Receipt,
    observation: &KernelSandboxObservation,
    gate: &KernelIsolationGate,
    leaf: &Path,
) -> Result<CgroupReleaseVerificationReceipt, String> {
    policy.validate().map_err(|error| error.to_string())?;
    base.validate_for(policy).map_err(|error| error.to_string())?;
    strict.validate_for(policy, base).map_err(|error| error.to_string())?;
    gate.validate_for(observation).map_err(|error| error.to_string())?;

    let memory_max = read_u64(leaf.join("memory.max"), "memory.max")?;
    let pids_max = read_u64(leaf.join("pids.max"), "pids.max")?;
    let (cpu_quota, cpu_period) = read_cpu_max_live(&leaf.join("cpu.max"))?;
    let swap_max = read_u64(leaf.join("memory.swap.max"), "memory.swap.max")?;
    let oom_group = match read_trimmed(leaf.join("memory.oom.group"))?.as_str() {
        "1" => true,
        "0" => false,
        _ => return Err("memory.oom.group is malformed during release verification".into()),
    };
    if memory_max != policy.memory_max_bytes()
        || pids_max != policy.pids_max()
        || cpu_quota != policy.cpu_quota_us()
        || cpu_period != policy.cpu_period_us()
        || swap_max != 0
        || !oom_group
    {
        return Err("cgroup controller values changed before evaluator release".into());
    }

    let members = read_trimmed(leaf.join("cgroup.procs"))?;
    if !members
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .any(|pid| pid == base.host_pid())
    {
        return Err("sandbox PID left the evaluator cgroup before release".into());
    }
    let proc_membership = read_proc_membership_live(base.host_pid())?;
    let expected_membership = membership_for_leaf(leaf)?;
    if proc_membership != expected_membership || proc_membership != base.proc_membership() {
        return Err("/proc cgroup membership changed before evaluator release".into());
    }

    let protocol_id = cgroup_release_verification_protocol_id();
    let leaf_path = leaf.to_string_lossy().into_owned();
    let id = derive_release_verification_id(
        &protocol_id,
        policy.id(),
        base.id(),
        strict.id(),
        base.host_pid(),
        observation.id(),
        gate.id(),
        &leaf_path,
        memory_max,
        pids_max,
        cpu_quota,
        cpu_period,
        swap_max,
        oom_group,
        &proc_membership,
    );
    let receipt = CgroupReleaseVerificationReceipt {
        id,
        protocol_id,
        cgroup_policy_id: policy.id().clone(),
        base_receipt_id: base.id().clone(),
        strict_receipt_id: strict.id().clone(),
        sandbox_host_pid: base.host_pid(),
        observation_id: observation.id().clone(),
        gate_id: gate.id().clone(),
        leaf_path,
        memory_max_bytes: memory_max,
        pids_max,
        cpu_quota_us: cpu_quota,
        cpu_period_us: cpu_period,
        swap_max_bytes: swap_max,
        oom_group,
        proc_membership,
    };
    receipt
        .validate_for(policy, base, strict, observation, gate)
        .map_err(|error| error.to_string())?;
    Ok(receipt)
}

fn read_trimmed(path: PathBuf) -> Result<String, String> {
    fs::read_to_string(&path)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("failed to read {path:?} during release verification: {error}"))
}

fn read_u64(path: PathBuf, name: &'static str) -> Result<u64, String> {
    read_trimmed(path)?
        .parse::<u64>()
        .map_err(|_| format!("{name} is malformed during release verification"))
}

fn read_cpu_max_live(path: &Path) -> Result<(u64, u64), String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {path:?} during release verification: {error}"))?;
    let mut fields = text.split_whitespace();
    let quota = fields
        .next()
        .ok_or_else(|| "cpu.max quota missing during release verification".to_string())?
        .parse::<u64>()
        .map_err(|_| "cpu.max quota malformed during release verification".to_string())?;
    let period = fields
        .next()
        .ok_or_else(|| "cpu.max period missing during release verification".to_string())?
        .parse::<u64>()
        .map_err(|_| "cpu.max period malformed during release verification".to_string())?;
    if fields.next().is_some() {
        return Err("cpu.max has extra fields during release verification".into());
    }
    Ok((quota, period))
}

fn read_proc_membership_live(host_pid: u32) -> Result<String, String> {
    let path = PathBuf::from(format!("/proc/{host_pid}/cgroup"));
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {path:?} during release verification: {error}"))?;
    text.lines()
        .find_map(|line| {
            let mut fields = line.splitn(3, ':');
            let hierarchy = fields.next()?;
            let controllers = fields.next()?;
            let membership = fields.next()?;
            if hierarchy == "0" && controllers.is_empty() && membership.starts_with('/') {
                Some(membership.trim().to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| "unified cgroup v2 membership missing during release verification".into())
}

fn membership_for_leaf(leaf: &Path) -> Result<String, String> {
    let relative = leaf
        .strip_prefix(Path::new(CGROUP_V2_ROOT))
        .map_err(|_| "evaluator cgroup leaf is outside /sys/fs/cgroup".to_string())?;
    Ok(format!(
        "/{}",
        relative.to_string_lossy().trim_start_matches('/')
    ))
}

#[allow(clippy::too_many_arguments)]
fn derive_release_verification_id(
    protocol_id: &ContentId,
    cgroup_policy_id: &ContentId,
    base_receipt_id: &ContentId,
    strict_receipt_id: &ContentId,
    sandbox_host_pid: u32,
    observation_id: &ContentId,
    gate_id: &ContentId,
    leaf_path: &str,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    swap_max_bytes: u64,
    oom_group: bool,
    proc_membership: &str,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-release-verification-receipt.v1",
        [
            protocol_id.as_str().as_bytes(),
            cgroup_policy_id.as_str().as_bytes(),
            base_receipt_id.as_str().as_bytes(),
            strict_receipt_id.as_str().as_bytes(),
            sandbox_host_pid.to_be_bytes().as_slice(),
            observation_id.as_str().as_bytes(),
            gate_id.as_str().as_bytes(),
            leaf_path.as_bytes(),
            memory_max_bytes.to_be_bytes().as_slice(),
            pids_max.to_be_bytes().as_slice(),
            cpu_quota_us.to_be_bytes().as_slice(),
            cpu_period_us.to_be_bytes().as_slice(),
            swap_max_bytes.to_be_bytes().as_slice(),
            &[u8::from(oom_group)],
            proc_membership.as_bytes(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_execution_receipt_id(
    execution_receipt_id: &ContentId,
    admission_receipt_id: &ContentId,
    admission_protocol_id: &ContentId,
    cgroup_policy_id: &ContentId,
    base_receipt_id: &ContentId,
    strict_receipt_id: &ContentId,
    teardown_receipt_id: &ContentId,
    sandbox_host_pid: u32,
    post_observation_id: &ContentId,
    post_gate_id: &ContentId,
    residual_descendant_kill_requested: bool,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-resource-gated-execution-receipt.v1",
        [
            execution_receipt_id.as_str().as_bytes(),
            admission_receipt_id.as_str().as_bytes(),
            admission_protocol_id.as_str().as_bytes(),
            cgroup_policy_id.as_str().as_bytes(),
            base_receipt_id.as_str().as_bytes(),
            strict_receipt_id.as_str().as_bytes(),
            teardown_receipt_id.as_str().as_bytes(),
            sandbox_host_pid.to_be_bytes().as_slice(),
            post_observation_id.as_str().as_bytes(),
            post_gate_id.as_str().as_bytes(),
            &[u8::from(residual_descendant_kill_requested)],
        ],
    )
}

fn derive_release_verified_execution_id(
    resource_execution_id: &ContentId,
    release_verification_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-release-verified-execution-receipt.v1",
        [
            resource_execution_id.as_str().as_bytes(),
            release_verification_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_identity_separates_residual_descendant_kill() {
        let execution = ContentId::derive("test-execution", [b"execution".as_slice()]);
        let admission = ContentId::derive("test-admission", [b"admission".as_slice()]);
        let protocol = pre_release_admission_protocol_id();
        let policy = ContentId::derive("test-policy", [b"policy".as_slice()]);
        let base = ContentId::derive("test-base", [b"base".as_slice()]);
        let strict = ContentId::derive("test-strict", [b"strict".as_slice()]);
        let teardown = ContentId::derive("test-teardown", [b"teardown".as_slice()]);
        let observation = ContentId::derive("test-observation", [b"observation".as_slice()]);
        let gate = ContentId::derive("test-gate", [b"gate".as_slice()]);
        let graceful = derive_execution_receipt_id(
            &execution, &admission, &protocol, &policy, &base, &strict, &teardown, 42,
            &observation, &gate, false,
        );
        let residual_kill = derive_execution_receipt_id(
            &execution, &admission, &protocol, &policy, &base, &strict, &teardown, 42,
            &observation, &gate, true,
        );
        assert_ne!(graceful, residual_kill);
    }

    #[test]
    fn release_verified_identity_binds_fresh_verification() {
        let resource = ContentId::derive("test-resource", [b"resource".as_slice()]);
        let a = ContentId::derive("test-release-verification", [b"a".as_slice()]);
        let b = ContentId::derive("test-release-verification", [b"b".as_slice()]);
        assert_ne!(
            derive_release_verified_execution_id(&resource, &a),
            derive_release_verified_execution_id(&resource, &b)
        );
    }
}

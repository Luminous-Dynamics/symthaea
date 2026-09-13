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
//! 4. `symthaea-forge-linux-cgroup-live-verify` independently re-reads the live controller values
//!    and exact PID membership while the evaluator is still blocked.
//!
//! A successful v2 receipt therefore means resource admission/hardening happened before model
//! release and the same state was re-observed before the admission callback completed. A later
//! generic launcher verification hook can move the same live check to the literal final release
//! boundary without changing the resource-control theorem itself.

use serde::Serialize;
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
use symthaea_forge_linux_cgroup_live_verify::{
    verify_live_cgroup_v2, CgroupLiveVerificationReceipt, CgroupLiveVerifyError,
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

#[derive(Debug, Error)]
pub enum CgroupObservedExecError {
    #[error(transparent)]
    Observed(#[from] ObservedEvaluatorError),
    #[error(transparent)]
    Resource(#[from] CgroupV2ResourceError),
    #[error(transparent)]
    Strict(#[from] StrictCgroupV2Error),
    #[error(transparent)]
    Live(#[from] CgroupLiveVerifyError),
    #[error("cgroup pre-release admission failed: {detail}")]
    AdmissionFailed { detail: String },
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
    #[error("cgroup-gated execution receipt does not bind supplied evidence")]
    ReceiptScopeMismatch,
    #[error("cgroup-gated execution receipt identity is non-canonical")]
    ReceiptIdentityMismatch,
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
    live_verification_receipt_id: ContentId,
    teardown_receipt_id: ContentId,
    sandbox_host_pid: u32,
    post_observation_id: ContentId,
    post_gate_id: ContentId,
    residual_descendant_kill_requested: bool,
}

impl CgroupResourceGatedExecutionReceipt {
    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn execution_receipt_id(&self) -> &ContentId {
        &self.execution_receipt_id
    }
    pub fn admission_receipt_id(&self) -> &ContentId {
        &self.admission_receipt_id
    }
    pub fn cgroup_policy_id(&self) -> &ContentId {
        &self.cgroup_policy_id
    }
    pub fn base_receipt_id(&self) -> &ContentId {
        &self.base_receipt_id
    }
    pub fn strict_receipt_id(&self) -> &ContentId {
        &self.strict_receipt_id
    }
    pub fn live_verification_receipt_id(&self) -> &ContentId {
        &self.live_verification_receipt_id
    }
    pub fn teardown_receipt_id(&self) -> &ContentId {
        &self.teardown_receipt_id
    }
    pub fn sandbox_host_pid(&self) -> u32 {
        self.sandbox_host_pid
    }
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
        live: &CgroupLiveVerificationReceipt,
        teardown: &StrictCgroupV2TeardownReceipt,
    ) -> Result<(), CgroupObservedExecError> {
        execution.validate_for(
            observed_policy,
            binding,
            model,
            request,
            response,
            observation,
            gate,
        )?;
        admission.validate_for(observation, gate)?;
        base.validate_for(cgroup_policy)?;
        strict.validate_for(cgroup_policy, base)?;
        live.validate_for(cgroup_policy, base, strict)?;
        teardown.validate_for(strict, cgroup_policy, base)?;

        if self.execution_receipt_id != *execution.id()
            || self.admission_receipt_id != *admission.id()
            || self.admission_protocol_id != pre_release_admission_protocol_id()
            || self.admission_protocol_id != *admission.protocol_id()
            || self.cgroup_policy_id != *cgroup_policy.id()
            || self.base_receipt_id != *base.id()
            || self.strict_receipt_id != *strict.id()
            || self.live_verification_receipt_id != *live.id()
            || self.teardown_receipt_id != *teardown.id()
            || self.sandbox_host_pid != execution.sandbox_host_pid()
            || self.sandbox_host_pid != admission.sandbox_host_pid()
            || self.sandbox_host_pid != base.host_pid()
            || self.sandbox_host_pid != live.host_pid()
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
            &self.live_verification_receipt_id,
            &self.teardown_receipt_id,
            self.sandbox_host_pid,
            &self.post_observation_id,
            &self.post_gate_id,
            self.residual_descendant_kill_requested,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(CgroupObservedExecError::ReceiptIdentityMismatch)
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
    live_verification: CgroupLiveVerificationReceipt,
    teardown_receipt: StrictCgroupV2TeardownReceipt,
    receipt: CgroupResourceGatedExecutionReceipt,
}

impl CgroupResourceGatedRun {
    pub fn response(&self) -> &ForgeProposalEvaluationResponse {
        &self.response
    }
    pub fn observation(&self) -> &KernelSandboxObservation {
        &self.observation
    }
    pub fn gate(&self) -> &KernelIsolationGate {
        &self.gate
    }
    pub fn execution(&self) -> &KernelGatedExecutionReceipt {
        &self.execution
    }
    pub fn admission(&self) -> &PreReleaseAdmissionReceipt {
        &self.admission
    }
    pub fn base_receipt(&self) -> &CgroupV2ResourceReceipt {
        &self.base_receipt
    }
    pub fn strict_receipt(&self) -> &StrictCgroupV2Receipt {
        &self.strict_receipt
    }
    pub fn live_verification(&self) -> &CgroupLiveVerificationReceipt {
        &self.live_verification
    }
    pub fn teardown_receipt(&self) -> &StrictCgroupV2TeardownReceipt {
        &self.teardown_receipt
    }
    pub fn receipt(&self) -> &CgroupResourceGatedExecutionReceipt {
        &self.receipt
    }
}

#[derive(Debug)]
struct CgroupAdmissionState<'a> {
    policy: &'a CgroupV2ResourcePolicy,
    delegation_root: PathBuf,
    base_receipt: Option<CgroupV2ResourceReceipt>,
    strict_receipt: Option<StrictCgroupV2Receipt>,
    live_verification: Option<CgroupLiveVerificationReceipt>,
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
            live_verification: None,
            lease: None,
            failure: None,
        }
    }

    fn fail(&mut self, detail: String) -> Result<(), ObservedEvaluatorError> {
        self.failure = Some(detail);
        Err(ObservedEvaluatorError::Control(io::Error::other(
            ADMISSION_TRANSPORT_ERROR,
        )))
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

        let live_verification = match verify_live_cgroup_v2(
            self.policy,
            &base_receipt,
            &strict_receipt,
        ) {
            Ok(value) => value,
            Err(error) => {
                let cleanup = strict.kill_and_cleanup(30_000);
                let detail = match cleanup {
                    Ok(_) => format!("live cgroup re-verification failed: {error}"),
                    Err(cleanup_error) => format!(
                        "live cgroup re-verification failed: {error}; fail-closed cleanup also failed: {cleanup_error}"
                    ),
                };
                return self.fail(detail);
            }
        };

        self.base_receipt = Some(base_receipt);
        self.strict_receipt = Some(strict_receipt);
        self.live_verification = Some(live_verification);
        self.lease = Some(strict);
        Ok(())
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
            if let Some(detail) = admission.failure.take() {
                return Err(CgroupObservedExecError::AdmissionFailed { detail });
            }
            if let Some(lease) = admission.lease.take() {
                return match lease.kill_and_cleanup(observed_policy.teardown_timeout_ms()) {
                    Ok(teardown_receipt) => Err(CgroupObservedExecError::ExecutionAfterAdmission {
                        source,
                        teardown_receipt,
                    }),
                    Err(teardown_error) => Err(
                        CgroupObservedExecError::ExecutionAndTeardownFailed {
                            execution: source.to_string(),
                            teardown: teardown_error.to_string(),
                        },
                    ),
                };
            }
            return Err(CgroupObservedExecError::Observed(source));
        }
    };

    let state = (
        admission.base_receipt.take(),
        admission.strict_receipt.take(),
        admission.live_verification.take(),
        admission.lease.take(),
    );
    let (base_receipt, strict_receipt, live_verification, lease) = match state {
        (Some(base), Some(strict), Some(live), Some(lease)) => (base, strict, live, lease),
        (_, _, _, Some(lease)) => {
            return match lease.kill_and_cleanup(observed_policy.teardown_timeout_ms()) {
                Ok(_) => Err(CgroupObservedExecError::AdmissionStateMissing),
                Err(teardown_error) => Err(CgroupObservedExecError::ExecutionAndTeardownFailed {
                    execution: "admission state missing after successful evaluator execution".into(),
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
        live_verification.id(),
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
        live_verification_receipt_id: live_verification.id().clone(),
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
        &live_verification,
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
        live_verification,
        teardown_receipt,
        receipt,
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_execution_receipt_id(
    execution_receipt_id: &ContentId,
    admission_receipt_id: &ContentId,
    admission_protocol_id: &ContentId,
    cgroup_policy_id: &ContentId,
    base_receipt_id: &ContentId,
    strict_receipt_id: &ContentId,
    live_verification_receipt_id: &ContentId,
    teardown_receipt_id: &ContentId,
    sandbox_host_pid: u32,
    post_observation_id: &ContentId,
    post_gate_id: &ContentId,
    residual_descendant_kill_requested: bool,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-cgroup-resource-gated-execution-receipt.v2",
        [
            execution_receipt_id.as_str().as_bytes(),
            admission_receipt_id.as_str().as_bytes(),
            admission_protocol_id.as_str().as_bytes(),
            cgroup_policy_id.as_str().as_bytes(),
            base_receipt_id.as_str().as_bytes(),
            strict_receipt_id.as_str().as_bytes(),
            live_verification_receipt_id.as_str().as_bytes(),
            teardown_receipt_id.as_str().as_bytes(),
            sandbox_host_pid.to_be_bytes().as_slice(),
            post_observation_id.as_str().as_bytes(),
            post_gate_id.as_str().as_bytes(),
            &[u8::from(residual_descendant_kill_requested)],
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_identity_separates_live_verification_and_residual_kill() {
        let execution = ContentId::derive("test-execution", [b"execution".as_slice()]);
        let admission = ContentId::derive("test-admission", [b"admission".as_slice()]);
        let protocol = pre_release_admission_protocol_id();
        let policy = ContentId::derive("test-policy", [b"policy".as_slice()]);
        let base = ContentId::derive("test-base", [b"base".as_slice()]);
        let strict = ContentId::derive("test-strict", [b"strict".as_slice()]);
        let live_a = ContentId::derive("test-live", [b"live-a".as_slice()]);
        let live_b = ContentId::derive("test-live", [b"live-b".as_slice()]);
        let teardown = ContentId::derive("test-teardown", [b"teardown".as_slice()]);
        let observation = ContentId::derive("test-observation", [b"observation".as_slice()]);
        let gate = ContentId::derive("test-gate", [b"gate".as_slice()]);
        let graceful = derive_execution_receipt_id(
            &execution,
            &admission,
            &protocol,
            &policy,
            &base,
            &strict,
            &live_a,
            &teardown,
            42,
            &observation,
            &gate,
            false,
        );
        let changed_live = derive_execution_receipt_id(
            &execution,
            &admission,
            &protocol,
            &policy,
            &base,
            &strict,
            &live_b,
            &teardown,
            42,
            &observation,
            &gate,
            false,
        );
        let residual_kill = derive_execution_receipt_id(
            &execution,
            &admission,
            &protocol,
            &policy,
            &base,
            &strict,
            &live_a,
            &teardown,
            42,
            &observation,
            &gate,
            true,
        );
        assert_ne!(graceful, changed_live);
        assert_ne!(graceful, residual_kill);
    }
}

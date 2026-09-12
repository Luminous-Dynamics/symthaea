// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Live cgroup-v2 verification inside Forge's pre-release evaluator admission window.
//!
//! The admission callback applies the base cgroup policy, upgrades it to strict no-swap/OOM
//! hardening, then re-reads every enforced controller plus exact PID membership while the model is
//! still held behind Bubblewrap's `--block-fd`. Only after this callback returns does the existing
//! observed launcher re-observe the same process identity and release the model.
//!
//! This narrows the remaining host-policy race substantially, but does not remove the host/system
//! manager from the TCB: a privileged external actor could still mutate cgroup state after the live
//! read-back. Exact release-time immutability therefore remains a separate theorem.

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

const CGROUP_V2_ROOT: &str = "/sys/fs/cgroup";
const ADMISSION_TRANSPORT_ERROR: &str = "live cgroup pre-release verification failed";

#[derive(Debug, Error)]
pub enum LiveCgroupVerificationError {
    #[error(transparent)]
    Observed(#[from] ObservedEvaluatorError),
    #[error(transparent)]
    Resource(#[from] CgroupV2ResourceError),
    #[error(transparent)]
    Strict(#[from] StrictCgroupV2Error),
    #[error("live cgroup verification IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("live cgroup controller or membership value is malformed")]
    MalformedLiveState,
    #[error("live cgroup state does not match the admitted policy/receipts")]
    LiveStateMismatch,
    #[error("live cgroup verification receipt is out of scope")]
    VerificationScopeMismatch,
    #[error("live cgroup verification receipt identity is non-canonical")]
    VerificationIdentityMismatch,
    #[error("live cgroup admission failed: {detail}")]
    AdmissionFailed { detail: String },
    #[error("live cgroup admission failed and descendant-wide cleanup also failed; admission={detail}; cleanup={cleanup}")]
    AdmissionFailedAndCleanupFailed { detail: String, cleanup: String },
    #[error("live cgroup admission completed but retained state is incomplete")]
    AdmissionStateMissing,
    #[error("incomplete admission state could not be cleaned up fail-closed: {0}")]
    AdmissionStateCleanupFailed(String),
    #[error("evaluator failed after live cgroup admission: {source}")]
    ExecutionAfterAdmission {
        #[source]
        source: ObservedEvaluatorError,
        teardown_receipt: StrictCgroupV2TeardownReceipt,
    },
    #[error("evaluator failed after live cgroup admission and cgroup teardown also failed; execution={execution}; teardown={teardown}")]
    ExecutionAndTeardownFailed { execution: String, teardown: String },
    #[error("live-cgroup execution receipt does not bind supplied evidence")]
    ExecutionScopeMismatch,
    #[error("live-cgroup execution receipt identity is non-canonical")]
    ExecutionIdentityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveCgroupVerificationReceipt {
    id: ContentId,
    cgroup_policy_id: ContentId,
    base_receipt_id: ContentId,
    strict_receipt_id: ContentId,
    host_pid: u32,
    process_start_time_ticks: u64,
    pre_observation_id: ContentId,
    pre_gate_id: ContentId,
    leaf_path: String,
    proc_membership: String,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    swap_max_bytes: u64,
    oom_group: bool,
    populated: u64,
}

impl LiveCgroupVerificationReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn cgroup_policy_id(&self) -> &ContentId { &self.cgroup_policy_id }
    pub fn base_receipt_id(&self) -> &ContentId { &self.base_receipt_id }
    pub fn strict_receipt_id(&self) -> &ContentId { &self.strict_receipt_id }
    pub fn host_pid(&self) -> u32 { self.host_pid }
    pub fn process_start_time_ticks(&self) -> u64 { self.process_start_time_ticks }
    pub fn pre_observation_id(&self) -> &ContentId { &self.pre_observation_id }
    pub fn pre_gate_id(&self) -> &ContentId { &self.pre_gate_id }
    pub fn leaf_path(&self) -> &str { &self.leaf_path }
    pub fn proc_membership(&self) -> &str { &self.proc_membership }
    pub fn populated(&self) -> u64 { self.populated }

    pub fn validate_for(
        &self,
        policy: &CgroupV2ResourcePolicy,
        base: &CgroupV2ResourceReceipt,
        strict: &StrictCgroupV2Receipt,
    ) -> Result<(), LiveCgroupVerificationError> {
        base.validate_for(policy)?;
        strict.validate_for(policy, base)?;
        if self.cgroup_policy_id != *policy.id()
            || self.base_receipt_id != *base.id()
            || self.strict_receipt_id != *strict.id()
            || self.host_pid != base.host_pid()
            || self.leaf_path != base.leaf_path()
            || self.leaf_path != strict.leaf_path()
            || self.proc_membership != base.proc_membership()
            || self.memory_max_bytes != policy.memory_max_bytes()
            || self.pids_max != policy.pids_max()
            || self.cpu_quota_us != policy.cpu_quota_us()
            || self.cpu_period_us != policy.cpu_period_us()
            || self.swap_max_bytes != strict.swap_max_bytes()
            || self.oom_group != strict.oom_group()
            || self.swap_max_bytes != 0
            || !self.oom_group
            || self.populated != 1
        {
            return Err(LiveCgroupVerificationError::VerificationScopeMismatch);
        }
        let expected = derive_live_receipt_id(
            &self.cgroup_policy_id,
            &self.base_receipt_id,
            &self.strict_receipt_id,
            self.host_pid,
            self.process_start_time_ticks,
            &self.pre_observation_id,
            &self.pre_gate_id,
            &self.leaf_path,
            &self.proc_membership,
            self.memory_max_bytes,
            self.pids_max,
            self.cpu_quota_us,
            self.cpu_period_us,
            self.swap_max_bytes,
            self.oom_group,
            self.populated,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(LiveCgroupVerificationError::VerificationIdentityMismatch)
        }
    }
}

pub fn observe_live_cgroup(
    policy: &CgroupV2ResourcePolicy,
    base: &CgroupV2ResourceReceipt,
    strict: &StrictCgroupV2Receipt,
    observation: &KernelSandboxObservation,
    gate: &KernelIsolationGate,
) -> Result<LiveCgroupVerificationReceipt, LiveCgroupVerificationError> {
    policy.validate()?;
    base.validate_for(policy)?;
    strict.validate_for(policy, base)?;
    gate.validate_for(observation)?;
    if base.host_pid() != observation.host_pid()
        || base.leaf_path() != strict.leaf_path()
        || observation.host_pid() == 0
    {
        return Err(LiveCgroupVerificationError::LiveStateMismatch);
    }

    let leaf = PathBuf::from(base.leaf_path());
    let expected_membership = membership_for_leaf(&leaf)?;
    let memory_max_bytes = read_u64(&leaf.join("memory.max"))?;
    let pids_max = read_u64(&leaf.join("pids.max"))?;
    let (cpu_quota_us, cpu_period_us) = read_cpu_max(&leaf.join("cpu.max"))?;
    let swap_max_bytes = read_u64(&leaf.join("memory.swap.max"))?;
    let oom_group = match read_trimmed(&leaf.join("memory.oom.group"))?.as_str() {
        "1" => true,
        "0" => false,
        _ => return Err(LiveCgroupVerificationError::MalformedLiveState),
    };
    let populated = read_populated(&leaf.join("cgroup.events"))?;
    let members = read_trimmed(&leaf.join("cgroup.procs"))?;
    let pid_present = members
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .any(|pid| pid == observation.host_pid());
    let proc_membership = read_proc_membership(observation.host_pid())?;

    if memory_max_bytes != policy.memory_max_bytes()
        || pids_max != policy.pids_max()
        || cpu_quota_us != policy.cpu_quota_us()
        || cpu_period_us != policy.cpu_period_us()
        || swap_max_bytes != 0
        || !oom_group
        || populated != 1
        || !pid_present
        || proc_membership != expected_membership
        || proc_membership != base.proc_membership()
    {
        return Err(LiveCgroupVerificationError::LiveStateMismatch);
    }

    let leaf_path = base.leaf_path().to_string();
    let id = derive_live_receipt_id(
        policy.id(),
        base.id(),
        strict.id(),
        observation.host_pid(),
        observation.process_start_time_ticks(),
        observation.id(),
        gate.id(),
        &leaf_path,
        &proc_membership,
        memory_max_bytes,
        pids_max,
        cpu_quota_us,
        cpu_period_us,
        swap_max_bytes,
        oom_group,
        populated,
    );
    let receipt = LiveCgroupVerificationReceipt {
        id,
        cgroup_policy_id: policy.id().clone(),
        base_receipt_id: base.id().clone(),
        strict_receipt_id: strict.id().clone(),
        host_pid: observation.host_pid(),
        process_start_time_ticks: observation.process_start_time_ticks(),
        pre_observation_id: observation.id().clone(),
        pre_gate_id: gate.id().clone(),
        leaf_path,
        proc_membership,
        memory_max_bytes,
        pids_max,
        cpu_quota_us,
        cpu_period_us,
        swap_max_bytes,
        oom_group,
        populated,
    };
    receipt.validate_for(policy, base, strict)?;
    Ok(receipt)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveCgroupGatedExecutionReceipt {
    id: ContentId,
    execution_receipt_id: ContentId,
    admission_receipt_id: ContentId,
    admission_protocol_id: ContentId,
    live_verification_receipt_id: ContentId,
    cgroup_policy_id: ContentId,
    base_receipt_id: ContentId,
    strict_receipt_id: ContentId,
    teardown_receipt_id: ContentId,
    sandbox_host_pid: u32,
    post_observation_id: ContentId,
    post_gate_id: ContentId,
    residual_descendant_kill_requested: bool,
}

impl LiveCgroupGatedExecutionReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn execution_receipt_id(&self) -> &ContentId { &self.execution_receipt_id }
    pub fn admission_receipt_id(&self) -> &ContentId { &self.admission_receipt_id }
    pub fn live_verification_receipt_id(&self) -> &ContentId {
        &self.live_verification_receipt_id
    }
    pub fn cgroup_policy_id(&self) -> &ContentId { &self.cgroup_policy_id }
    pub fn teardown_receipt_id(&self) -> &ContentId { &self.teardown_receipt_id }
    pub fn residual_descendant_kill_requested(&self) -> bool {
        self.residual_descendant_kill_requested
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        model: &ForgeProposalFrozenModel,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        post_observation: &KernelSandboxObservation,
        post_gate: &KernelIsolationGate,
        execution: &KernelGatedExecutionReceipt,
        admission: &PreReleaseAdmissionReceipt,
        cgroup_policy: &CgroupV2ResourcePolicy,
        base: &CgroupV2ResourceReceipt,
        strict: &StrictCgroupV2Receipt,
        live: &LiveCgroupVerificationReceipt,
        teardown: &StrictCgroupV2TeardownReceipt,
    ) -> Result<(), LiveCgroupVerificationError> {
        execution.validate_for(
            kernel_policy,
            binding,
            model,
            request,
            response,
            post_observation,
            post_gate,
        )?;
        admission.validate_for(post_observation, post_gate)?;
        live.validate_for(cgroup_policy, base, strict)?;
        teardown.validate_for(strict, cgroup_policy, base)?;
        if self.execution_receipt_id != *execution.id()
            || self.admission_receipt_id != *admission.id()
            || self.admission_protocol_id != pre_release_admission_protocol_id()
            || self.admission_protocol_id != *admission.protocol_id()
            || self.live_verification_receipt_id != *live.id()
            || self.cgroup_policy_id != *cgroup_policy.id()
            || self.base_receipt_id != *base.id()
            || self.strict_receipt_id != *strict.id()
            || self.teardown_receipt_id != *teardown.id()
            || self.sandbox_host_pid != execution.sandbox_host_pid()
            || self.sandbox_host_pid != admission.sandbox_host_pid()
            || self.sandbox_host_pid != base.host_pid()
            || self.sandbox_host_pid != live.host_pid()
            || self.sandbox_host_pid != post_observation.host_pid()
            || live.process_start_time_ticks() != admission.process_start_time_ticks()
            || live.pre_observation_id() != admission.pre_observation_id()
            || live.pre_gate_id() != admission.pre_gate_id()
            || self.post_observation_id != *post_observation.id()
            || self.post_observation_id != *admission.post_observation_id()
            || self.post_gate_id != *post_gate.id()
            || self.post_gate_id != *admission.post_gate_id()
            || self.residual_descendant_kill_requested != teardown.descendant_kill_requested()
        {
            return Err(LiveCgroupVerificationError::ExecutionScopeMismatch);
        }
        let expected = derive_execution_receipt_id(
            &self.execution_receipt_id,
            &self.admission_receipt_id,
            &self.admission_protocol_id,
            &self.live_verification_receipt_id,
            &self.cgroup_policy_id,
            &self.base_receipt_id,
            &self.strict_receipt_id,
            &self.teardown_receipt_id,
            self.sandbox_host_pid,
            &self.post_observation_id,
            &self.post_gate_id,
            self.residual_descendant_kill_requested,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(LiveCgroupVerificationError::ExecutionIdentityMismatch)
        }
    }
}

pub struct LiveCgroupGatedRun {
    response: ForgeProposalEvaluationResponse,
    observation: KernelSandboxObservation,
    gate: KernelIsolationGate,
    execution: KernelGatedExecutionReceipt,
    admission: PreReleaseAdmissionReceipt,
    base_receipt: CgroupV2ResourceReceipt,
    strict_receipt: StrictCgroupV2Receipt,
    live_receipt: LiveCgroupVerificationReceipt,
    teardown_receipt: StrictCgroupV2TeardownReceipt,
    receipt: LiveCgroupGatedExecutionReceipt,
}

impl LiveCgroupGatedRun {
    pub fn response(&self) -> &ForgeProposalEvaluationResponse { &self.response }
    pub fn observation(&self) -> &KernelSandboxObservation { &self.observation }
    pub fn gate(&self) -> &KernelIsolationGate { &self.gate }
    pub fn execution(&self) -> &KernelGatedExecutionReceipt { &self.execution }
    pub fn admission(&self) -> &PreReleaseAdmissionReceipt { &self.admission }
    pub fn base_receipt(&self) -> &CgroupV2ResourceReceipt { &self.base_receipt }
    pub fn strict_receipt(&self) -> &StrictCgroupV2Receipt { &self.strict_receipt }
    pub fn live_receipt(&self) -> &LiveCgroupVerificationReceipt { &self.live_receipt }
    pub fn teardown_receipt(&self) -> &StrictCgroupV2TeardownReceipt { &self.teardown_receipt }
    pub fn receipt(&self) -> &LiveCgroupGatedExecutionReceipt { &self.receipt }
}

#[derive(Debug)]
struct LiveAdmissionState<'a> {
    policy: &'a CgroupV2ResourcePolicy,
    delegation_root: PathBuf,
    teardown_timeout_ms: u64,
    base_receipt: Option<CgroupV2ResourceReceipt>,
    strict_receipt: Option<StrictCgroupV2Receipt>,
    live_receipt: Option<LiveCgroupVerificationReceipt>,
    lease: Option<StrictCgroupV2Lease>,
    failure: Option<String>,
}

impl<'a> LiveAdmissionState<'a> {
    fn new(
        policy: &'a CgroupV2ResourcePolicy,
        delegation_root: &Path,
        teardown_timeout_ms: u64,
    ) -> Self {
        Self {
            policy,
            delegation_root: delegation_root.to_path_buf(),
            teardown_timeout_ms,
            base_receipt: None,
            strict_receipt: None,
            live_receipt: None,
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

impl PreReleaseAdmission for LiveAdmissionState<'_> {
    fn admit(
        &mut self,
        sandbox_pid: u32,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
    ) -> Result<(), ObservedEvaluatorError> {
        if observation.host_pid() != sandbox_pid {
            return self.fail("kernel observation PID differs from admission PID".into());
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

        self.base_receipt = Some(base_receipt.clone());
        self.strict_receipt = Some(strict_receipt.clone());
        self.lease = Some(strict);

        let live_receipt = match observe_live_cgroup(
            self.policy,
            &base_receipt,
            &strict_receipt,
            observation,
            gate,
        ) {
            Ok(value) => value,
            Err(error) => return self.fail(format!("live cgroup verification failed: {error}")),
        };
        self.live_receipt = Some(live_receipt);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_live_cgroup_gated_evaluator(
    kernel_policy: &KernelGatedEvaluatorPolicy,
    binding: &ForgeProposalExecutableModelBinding,
    request: &ForgeProposalEvaluationRequest,
    model: &ForgeProposalFrozenModel,
    bubblewrap_executable: impl AsRef<Path>,
    model_executable: impl AsRef<Path>,
    model_args: &[String],
    cgroup_policy: &CgroupV2ResourcePolicy,
    delegation_root: impl AsRef<Path>,
) -> Result<LiveCgroupGatedRun, LiveCgroupVerificationError> {
    cgroup_policy.validate()?;
    let mut admission = LiveAdmissionState::new(
        cgroup_policy,
        delegation_root.as_ref(),
        kernel_policy.teardown_timeout_ms(),
    );
    let observed = run_kernel_gated_evaluator_with_admission(
        kernel_policy,
        binding,
        request,
        model,
        bubblewrap_executable,
        model_executable,
        model_args,
        &mut admission,
    );

    let (response, post_observation, post_gate, execution, admission_receipt) = match observed {
        Ok(value) => value,
        Err(source) => {
            if let Some(detail) = admission.failure.take() {
                if let Some(lease) = admission.lease.take() {
                    return match lease.kill_and_cleanup(kernel_policy.teardown_timeout_ms()) {
                        Ok(_) => Err(LiveCgroupVerificationError::AdmissionFailed { detail }),
                        Err(cleanup_error) => {
                            Err(LiveCgroupVerificationError::AdmissionFailedAndCleanupFailed {
                                detail,
                                cleanup: cleanup_error.to_string(),
                            })
                        }
                    };
                }
                return Err(LiveCgroupVerificationError::AdmissionFailed { detail });
            }
            if let Some(lease) = admission.lease.take() {
                return match lease.kill_and_cleanup(kernel_policy.teardown_timeout_ms()) {
                    Ok(teardown_receipt) => Err(
                        LiveCgroupVerificationError::ExecutionAfterAdmission {
                            source,
                            teardown_receipt,
                        },
                    ),
                    Err(teardown_error) => Err(
                        LiveCgroupVerificationError::ExecutionAndTeardownFailed {
                            execution: source.to_string(),
                            teardown: teardown_error.to_string(),
                        },
                    ),
                };
            }
            return Err(LiveCgroupVerificationError::Observed(source));
        }
    };

    let base = admission.base_receipt.take();
    let strict = admission.strict_receipt.take();
    let live = admission.live_receipt.take();
    let lease = admission.lease.take();
    let (base_receipt, strict_receipt, live_receipt, lease) = match (base, strict, live, lease) {
        (Some(base), Some(strict), Some(live), Some(lease)) => (base, strict, live, lease),
        (_, _, _, Some(lease)) => {
            return match lease.kill_and_cleanup(kernel_policy.teardown_timeout_ms()) {
                Ok(_) => Err(LiveCgroupVerificationError::AdmissionStateMissing),
                Err(error) => Err(LiveCgroupVerificationError::AdmissionStateCleanupFailed(
                    error.to_string(),
                )),
            };
        }
        _ => return Err(LiveCgroupVerificationError::AdmissionStateMissing),
    };

    let teardown_receipt = lease.finalize_after_sandbox_exit(kernel_policy.teardown_timeout_ms())?;
    let admission_protocol_id = pre_release_admission_protocol_id();
    let id = derive_execution_receipt_id(
        execution.id(),
        admission_receipt.id(),
        &admission_protocol_id,
        live_receipt.id(),
        cgroup_policy.id(),
        base_receipt.id(),
        strict_receipt.id(),
        teardown_receipt.id(),
        execution.sandbox_host_pid(),
        post_observation.id(),
        post_gate.id(),
        teardown_receipt.descendant_kill_requested(),
    );
    let receipt = LiveCgroupGatedExecutionReceipt {
        id,
        execution_receipt_id: execution.id().clone(),
        admission_receipt_id: admission_receipt.id().clone(),
        admission_protocol_id,
        live_verification_receipt_id: live_receipt.id().clone(),
        cgroup_policy_id: cgroup_policy.id().clone(),
        base_receipt_id: base_receipt.id().clone(),
        strict_receipt_id: strict_receipt.id().clone(),
        teardown_receipt_id: teardown_receipt.id().clone(),
        sandbox_host_pid: execution.sandbox_host_pid(),
        post_observation_id: post_observation.id().clone(),
        post_gate_id: post_gate.id().clone(),
        residual_descendant_kill_requested: teardown_receipt.descendant_kill_requested(),
    };
    receipt.validate_for(
        kernel_policy,
        binding,
        model,
        request,
        &response,
        &post_observation,
        &post_gate,
        &execution,
        &admission_receipt,
        cgroup_policy,
        &base_receipt,
        &strict_receipt,
        &live_receipt,
        &teardown_receipt,
    )?;

    Ok(LiveCgroupGatedRun {
        response,
        observation: post_observation,
        gate: post_gate,
        execution,
        admission: admission_receipt,
        base_receipt,
        strict_receipt,
        live_receipt,
        teardown_receipt,
        receipt,
    })
}

fn read_trimmed(path: &Path) -> Result<String, LiveCgroupVerificationError> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_string())
        .map_err(|source| LiveCgroupVerificationError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn read_u64(path: &Path) -> Result<u64, LiveCgroupVerificationError> {
    read_trimmed(path)?
        .parse::<u64>()
        .map_err(|_| LiveCgroupVerificationError::MalformedLiveState)
}

fn read_cpu_max(path: &Path) -> Result<(u64, u64), LiveCgroupVerificationError> {
    let value = read_trimmed(path)?;
    let mut fields = value.split_whitespace();
    let quota = fields
        .next()
        .ok_or(LiveCgroupVerificationError::MalformedLiveState)?
        .parse::<u64>()
        .map_err(|_| LiveCgroupVerificationError::MalformedLiveState)?;
    let period = fields
        .next()
        .ok_or(LiveCgroupVerificationError::MalformedLiveState)?
        .parse::<u64>()
        .map_err(|_| LiveCgroupVerificationError::MalformedLiveState)?;
    if fields.next().is_some() {
        return Err(LiveCgroupVerificationError::MalformedLiveState);
    }
    Ok((quota, period))
}

fn read_populated(path: &Path) -> Result<u64, LiveCgroupVerificationError> {
    let value = read_trimmed(path)?;
    value
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            match (fields.next(), fields.next(), fields.next()) {
                (Some("populated"), Some(value), None) => value.parse::<u64>().ok(),
                _ => None,
            }
        })
        .ok_or(LiveCgroupVerificationError::MalformedLiveState)
}

fn read_proc_membership(host_pid: u32) -> Result<String, LiveCgroupVerificationError> {
    let path = PathBuf::from(format!("/proc/{host_pid}/cgroup"));
    let text = read_trimmed(&path)?;
    text.lines()
        .find_map(|line| {
            let mut fields = line.splitn(3, ':');
            match (fields.next(), fields.next(), fields.next()) {
                (Some("0"), Some(""), Some(path)) if path.starts_with('/') => {
                    Some(path.to_string())
                }
                _ => None,
            }
        })
        .ok_or(LiveCgroupVerificationError::MalformedLiveState)
}

fn membership_for_leaf(leaf: &Path) -> Result<String, LiveCgroupVerificationError> {
    let relative = leaf
        .strip_prefix(Path::new(CGROUP_V2_ROOT))
        .map_err(|_| LiveCgroupVerificationError::LiveStateMismatch)?;
    Ok(format!(
        "/{}",
        relative.to_string_lossy().trim_start_matches('/')
    ))
}

#[allow(clippy::too_many_arguments)]
fn derive_live_receipt_id(
    policy_id: &ContentId,
    base_receipt_id: &ContentId,
    strict_receipt_id: &ContentId,
    host_pid: u32,
    process_start_time_ticks: u64,
    pre_observation_id: &ContentId,
    pre_gate_id: &ContentId,
    leaf_path: &str,
    proc_membership: &str,
    memory_max_bytes: u64,
    pids_max: u64,
    cpu_quota_us: u64,
    cpu_period_us: u64,
    swap_max_bytes: u64,
    oom_group: bool,
    populated: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-live-cgroup-verification-receipt.v1",
        [
            policy_id.as_str().as_bytes(),
            base_receipt_id.as_str().as_bytes(),
            strict_receipt_id.as_str().as_bytes(),
            host_pid.to_be_bytes().as_slice(),
            process_start_time_ticks.to_be_bytes().as_slice(),
            pre_observation_id.as_str().as_bytes(),
            pre_gate_id.as_str().as_bytes(),
            leaf_path.as_bytes(),
            proc_membership.as_bytes(),
            memory_max_bytes.to_be_bytes().as_slice(),
            pids_max.to_be_bytes().as_slice(),
            cpu_quota_us.to_be_bytes().as_slice(),
            cpu_period_us.to_be_bytes().as_slice(),
            swap_max_bytes.to_be_bytes().as_slice(),
            &[u8::from(oom_group)],
            populated.to_be_bytes().as_slice(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_execution_receipt_id(
    execution_receipt_id: &ContentId,
    admission_receipt_id: &ContentId,
    admission_protocol_id: &ContentId,
    live_verification_receipt_id: &ContentId,
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
        "symthaea.forge-live-cgroup-gated-execution-receipt.v1",
        [
            execution_receipt_id.as_str().as_bytes(),
            admission_receipt_id.as_str().as_bytes(),
            admission_protocol_id.as_str().as_bytes(),
            live_verification_receipt_id.as_str().as_bytes(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_max_parser_is_closed_shape() {
        let temp = std::env::temp_dir().join(format!(
            "symthaea-cpu-max-test-{}",
            std::process::id()
        ));
        fs::write(&temp, b"50000 100000\n").unwrap();
        assert_eq!(read_cpu_max(&temp).unwrap(), (50_000, 100_000));
        fs::write(&temp, b"max 100000\n").unwrap();
        assert!(read_cpu_max(&temp).is_err());
        let _ = fs::remove_file(temp);
    }

    #[test]
    fn execution_identity_separates_live_verification() {
        let execution = ContentId::derive("test-execution", [b"execution".as_slice()]);
        let admission = ContentId::derive("test-admission", [b"admission".as_slice()]);
        let protocol = pre_release_admission_protocol_id();
        let live_a = ContentId::derive("test-live", [b"a".as_slice()]);
        let live_b = ContentId::derive("test-live", [b"b".as_slice()]);
        let policy = ContentId::derive("test-policy", [b"policy".as_slice()]);
        let base = ContentId::derive("test-base", [b"base".as_slice()]);
        let strict = ContentId::derive("test-strict", [b"strict".as_slice()]);
        let teardown = ContentId::derive("test-teardown", [b"teardown".as_slice()]);
        let observation = ContentId::derive("test-observation", [b"observation".as_slice()]);
        let gate = ContentId::derive("test-gate", [b"gate".as_slice()]);
        let a = derive_execution_receipt_id(
            &execution, &admission, &protocol, &live_a, &policy, &base, &strict, &teardown,
            42, &observation, &gate, false,
        );
        let b = derive_execution_receipt_id(
            &execution, &admission, &protocol, &live_b, &policy, &base, &strict, &teardown,
            42, &observation, &gate, false,
        );
        assert_ne!(a, b);
    }
}

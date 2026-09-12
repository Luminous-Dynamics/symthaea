// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed execution-substrate policy for simulation releases.
//!
//! Wasmtime fuel, Store limits, and epoch interruption bound guest execution but
//! do not bound Component compilation performed before a Store exists. Until a
//! separately supervised compiler/execution worker is available, this crate
//! makes that gap explicit at the production release entrypoint:
//!
//! - default policy rejects in-process Wasm selection;
//! - an operator may explicitly allow in-process Wasm only with a `Trusted`
//!   routing floor or higher;
//! - `Native` and `Remote` remain eligible according to the caller's other
//!   routing constraints;
//! - the effective policy and actual selected-admission trust are committed to
//!   the underlying sealed simulation release so later audit can distinguish
//!   containment from explicit risk acceptance.
//!
//! `AllowTrustedInProcess` is **not** compiler containment. It is a deliberate
//! residual-risk exception for controlled/trusted packages.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_extension_admission::{ActiveAdmission, AdmissionCurrentnessSource, TrustLevel};
use symthaea_extension_core::RuntimeKind;
use symthaea_extension_router::RoutingConstraints;
use symthaea_sim_bridge::{SimulationRequest, SimulationResult};
use symthaea_sim_extension_routing::LazySimulationRegistry;
use symthaea_sim_release::{
    ReleasedSimulation, SimulationReleaseError, SimulationReleaseEvidence, run_released,
};
use thiserror::Error;

pub const SIMULATION_EXECUTION_POLICY_PROFILE_V1: &str =
    "symthaea.simulation.execution-policy.v1";

const POLICY_DOMAIN_V1: &[u8] = b"symthaea.simulation.execution-policy.v1\0";

/// Policy for Wasm compilation that occurs inside the Symthaea host process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WasmCompilationPolicy {
    /// Fail closed: do not allow an in-process Wasm provider to be selected.
    #[default]
    RejectInProcess,
    /// Explicit residual-risk exception. The effective routing trust floor is
    /// raised to at least `Trusted`, but compilation is still not contained.
    AllowTrustedInProcess,
}

/// Host policy applied before routing/execution begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimulationExecutionPolicy {
    pub wasm_compilation: WasmCompilationPolicy,
}

impl SimulationExecutionPolicy {
    pub const fn reject_in_process_wasm() -> Self {
        Self {
            wasm_compilation: WasmCompilationPolicy::RejectInProcess,
        }
    }

    pub const fn allow_trusted_in_process_wasm() -> Self {
        Self {
            wasm_compilation: WasmCompilationPolicy::AllowTrustedInProcess,
        }
    }

    /// Normalize caller constraints into the exact runtime/trust envelope this
    /// policy will pass to the router.
    pub fn apply(
        self,
        mut constraints: RoutingConstraints,
    ) -> Result<RoutingConstraints, SimulationExecutionPolicyError> {
        let caller_runtimes = constraints.allowed_runtimes.clone();
        constraints.allowed_runtimes = match self.wasm_compilation {
            WasmCompilationPolicy::RejectInProcess => normalize_runtime_intersection(
                &caller_runtimes,
                &[RuntimeKind::Native, RuntimeKind::Remote],
            ),
            WasmCompilationPolicy::AllowTrustedInProcess => {
                if constraints.minimum_trust < TrustLevel::Trusted {
                    constraints.minimum_trust = TrustLevel::Trusted;
                }
                normalize_runtime_intersection(
                    &caller_runtimes,
                    &[RuntimeKind::Native, RuntimeKind::Wasm, RuntimeKind::Remote],
                )
            }
        };
        if constraints.allowed_runtimes.is_empty() {
            return Err(SimulationExecutionPolicyError::NoInvocableRuntime);
        }
        Ok(constraints)
    }
}

/// Persistable audit representation of the execution-substrate decision plus
/// the underlying sealed simulation release evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyReleaseEvidence {
    pub profile: String,
    pub wasm_compilation: WasmCompilationPolicy,
    pub effective_minimum_trust: TrustLevel,
    pub selected_trust: TrustLevel,
    pub effective_allowed_runtimes: Vec<RuntimeKind>,
    pub underlying_release_sha256: String,
    pub policy_release_sha256: String,
    pub release: SimulationReleaseEvidence,
}

#[derive(Debug)]
struct ExecutionPolicyReceipt {
    wasm_compilation: WasmCompilationPolicy,
    effective_minimum_trust: TrustLevel,
    selected_trust: TrustLevel,
    effective_allowed_runtimes: Vec<RuntimeKind>,
    underlying_release_sha256: [u8; 32],
    policy_release_sha256: [u8; 32],
}

/// A sealed simulation release together with the exact compilation policy that
/// constrained provider selection.
#[derive(Debug)]
pub struct PolicyReleasedSimulation {
    release: ReleasedSimulation,
    policy: ExecutionPolicyReceipt,
}

impl PolicyReleasedSimulation {
    pub fn release(&self) -> &ReleasedSimulation {
        &self.release
    }

    pub fn result(&self) -> &SimulationResult {
        self.release.result()
    }

    pub fn evidence(&self) -> PolicyReleaseEvidence {
        PolicyReleaseEvidence {
            profile: SIMULATION_EXECUTION_POLICY_PROFILE_V1.into(),
            wasm_compilation: self.policy.wasm_compilation,
            effective_minimum_trust: self.policy.effective_minimum_trust,
            selected_trust: self.policy.selected_trust,
            effective_allowed_runtimes: self.policy.effective_allowed_runtimes.clone(),
            underlying_release_sha256: hex_digest(self.policy.underlying_release_sha256),
            policy_release_sha256: hex_digest(self.policy.policy_release_sha256),
            release: self.release.evidence(),
        }
    }

    /// Verify the underlying release and the policy-to-release commitment.
    /// This is consistency verification only; it does not re-query current
    /// authority or retroactively contain an in-process compiler.
    pub fn verify(
        &self,
        request: &SimulationRequest,
    ) -> Result<(), SimulationExecutionPolicyError> {
        self.release.verify(request)?;
        if self.release.receipt().release_sha256() != self.policy.underlying_release_sha256 {
            return Err(SimulationExecutionPolicyError::UnderlyingReleaseMismatch);
        }
        if !self
            .policy
            .effective_allowed_runtimes
            .contains(&self.release.receipt().runtime())
        {
            return Err(SimulationExecutionPolicyError::ReleasedRuntimeNotAllowed);
        }
        if self.policy.selected_trust < self.policy.effective_minimum_trust {
            return Err(SimulationExecutionPolicyError::SelectedTrustBelowEffectiveMinimum);
        }
        if policy_release_sha256_v1(&self.policy) != self.policy.policy_release_sha256 {
            return Err(SimulationExecutionPolicyError::PolicyReceiptMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SimulationExecutionPolicyError {
    #[error("execution policy leaves no invocable runtime")]
    NoInvocableRuntime,
    #[error(transparent)]
    Release(#[from] SimulationReleaseError),
    #[error("selected admission is missing, duplicated, or no longer matches the release decision")]
    SelectedAdmissionMismatch,
    #[error("selected admission trust is below the effective execution-policy minimum")]
    SelectedTrustBelowEffectiveMinimum,
    #[error("released provider runtime is outside the effective execution-policy runtime set")]
    ReleasedRuntimeNotAllowed,
    #[error("execution-policy receipt no longer matches the sealed simulation release")]
    UnderlyingReleaseMismatch,
    #[error("execution-policy receipt digest does not match")]
    PolicyReceiptMismatch,
}

/// Apply execution-substrate policy before routing, execute through the sealed
/// release boundary, and bind the effective policy to the resulting release.
pub fn run_with_execution_policy(
    registry: &LazySimulationRegistry,
    request: &SimulationRequest,
    constraints: RoutingConstraints,
    admissions: &[ActiveAdmission],
    currentness: &dyn AdmissionCurrentnessSource,
    policy: SimulationExecutionPolicy,
) -> Result<PolicyReleasedSimulation, SimulationExecutionPolicyError> {
    let effective = policy.apply(constraints)?;
    let effective_minimum_trust = effective.minimum_trust;
    let effective_allowed_runtimes = effective.allowed_runtimes.clone();
    let release = run_released(registry, request, effective, admissions, currentness)?;
    let selected_trust = selected_admission_trust(&release, admissions)?;
    if selected_trust < effective_minimum_trust {
        return Err(SimulationExecutionPolicyError::SelectedTrustBelowEffectiveMinimum);
    }

    let mut receipt = ExecutionPolicyReceipt {
        wasm_compilation: policy.wasm_compilation,
        effective_minimum_trust,
        selected_trust,
        effective_allowed_runtimes,
        underlying_release_sha256: release.receipt().release_sha256(),
        policy_release_sha256: [0; 32],
    };
    receipt.policy_release_sha256 = policy_release_sha256_v1(&receipt);

    let sealed = PolicyReleasedSimulation {
        release,
        policy: receipt,
    };
    sealed.verify(request)?;
    Ok(sealed)
}

fn selected_admission_trust(
    release: &ReleasedSimulation,
    admissions: &[ActiveAdmission],
) -> Result<TrustLevel, SimulationExecutionPolicyError> {
    let decision = release.decision();
    let mut matches = admissions
        .iter()
        .filter(|admission| admission.extension() == &decision.selected);
    let admission = matches
        .next()
        .ok_or(SimulationExecutionPolicyError::SelectedAdmissionMismatch)?;
    if matches.next().is_some()
        || admission.generation() != decision.selected_admission_generation
        || admission.trust_generation() != decision.selected_trust_generation
        || admission.manifest_sha256() != decision.selected_manifest_sha256
        || admission.payload_sha256() != decision.selected_payload_sha256
        || admission.policy_sha256() != decision.selected_policy_sha256
    {
        return Err(SimulationExecutionPolicyError::SelectedAdmissionMismatch);
    }
    Ok(admission.trust())
}

fn normalize_runtime_intersection(
    caller: &[RuntimeKind],
    policy_allowed: &[RuntimeKind],
) -> Vec<RuntimeKind> {
    let mut runtimes = if caller.is_empty() {
        policy_allowed.to_vec()
    } else {
        caller
            .iter()
            .copied()
            .filter(|runtime| policy_allowed.contains(runtime))
            .collect()
    };
    runtimes.sort_by_key(|runtime| runtime_tag(*runtime));
    runtimes.dedup();
    runtimes
}

fn policy_release_sha256_v1(receipt: &ExecutionPolicyReceipt) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(POLICY_DOMAIN_V1);
    hasher.update([wasm_policy_tag(receipt.wasm_compilation)]);
    hasher.update([trust_tag(receipt.effective_minimum_trust)]);
    hasher.update([trust_tag(receipt.selected_trust)]);
    hasher.update((receipt.effective_allowed_runtimes.len() as u64).to_le_bytes());
    for runtime in &receipt.effective_allowed_runtimes {
        hasher.update([runtime_tag(*runtime)]);
    }
    hasher.update(receipt.underlying_release_sha256);
    hasher.finalize().into()
}

const fn wasm_policy_tag(policy: WasmCompilationPolicy) -> u8 {
    match policy {
        WasmCompilationPolicy::RejectInProcess => 0,
        WasmCompilationPolicy::AllowTrustedInProcess => 1,
    }
}

const fn trust_tag(trust: TrustLevel) -> u8 {
    match trust {
        TrustLevel::Untrusted => 0,
        TrustLevel::Community => 1,
        TrustLevel::Trusted => 2,
        TrustLevel::Privileged => 3,
    }
}

const fn runtime_tag(runtime: RuntimeKind) -> u8 {
    match runtime {
        RuntimeKind::Native => 0,
        RuntimeKind::Wasm => 1,
        RuntimeKind::Remote => 2,
        RuntimeKind::DataOnly => 3,
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_removes_in_process_wasm_and_data_only() {
        let effective = SimulationExecutionPolicy::default()
            .apply(RoutingConstraints::default())
            .unwrap();
        assert_eq!(
            effective.allowed_runtimes,
            vec![RuntimeKind::Native, RuntimeKind::Remote]
        );
        assert_eq!(effective.minimum_trust, TrustLevel::Untrusted);
    }

    #[test]
    fn explicit_in_process_exception_raises_trust_floor() {
        let effective = SimulationExecutionPolicy::allow_trusted_in_process_wasm()
            .apply(RoutingConstraints {
                minimum_trust: TrustLevel::Community,
                ..RoutingConstraints::default()
            })
            .unwrap();
        assert_eq!(effective.minimum_trust, TrustLevel::Trusted);
        assert_eq!(
            effective.allowed_runtimes,
            vec![RuntimeKind::Native, RuntimeKind::Wasm, RuntimeKind::Remote]
        );
    }

    #[test]
    fn explicit_wasm_only_request_fails_under_default_policy() {
        let error = SimulationExecutionPolicy::default()
            .apply(RoutingConstraints {
                allowed_runtimes: vec![RuntimeKind::Wasm],
                ..RoutingConstraints::default()
            })
            .unwrap_err();
        assert!(matches!(
            error,
            SimulationExecutionPolicyError::NoInvocableRuntime
        ));
    }

    #[test]
    fn runtime_sets_are_canonicalized_for_policy_receipts() {
        let effective = SimulationExecutionPolicy::allow_trusted_in_process_wasm()
            .apply(RoutingConstraints {
                allowed_runtimes: vec![
                    RuntimeKind::Remote,
                    RuntimeKind::Wasm,
                    RuntimeKind::Native,
                    RuntimeKind::Wasm,
                ],
                ..RoutingConstraints::default()
            })
            .unwrap();
        assert_eq!(
            effective.allowed_runtimes,
            vec![RuntimeKind::Native, RuntimeKind::Wasm, RuntimeKind::Remote]
        );
    }
}

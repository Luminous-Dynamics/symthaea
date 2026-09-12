// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parent-observed seccomp state for a Forge evaluator sandbox.
//!
//! This bridge intentionally does **not** claim that a particular seccomp policy was loaded.
//! It composes the existing strong kernel-isolation gate with two additional facts already exposed
//! by `/proc/<pid>/status` for the exact paused sandbox process:
//!
//! - `Seccomp: 2` — Linux reports filter mode is active;
//! - `NoNewPrivs: 1` — the process may not gain privileges through `execve`.
//!
//! The exact cBPF bytes, architecture checks, compiler lineage and syscall-policy semantics remain a
//! separate theorem. A future launcher must bind those bytes to Bubblewrap's `--seccomp FD` before
//! this observed state can be upgraded into a policy-semantic claim.

use serde::Serialize;
use symthaea_algorithms::ContentId;
use symthaea_forge_linux_kernel_attestation::{
    KernelAttestationError, KernelIsolationGate, KernelSandboxObservation,
};
use thiserror::Error;

const SECCOMP_MODE_FILTER: u32 = 2;
const NO_NEW_PRIVS_ENABLED: u32 = 1;

#[derive(Debug, Error)]
pub enum SeccompObservationError {
    #[error(transparent)]
    Kernel(#[from] KernelAttestationError),
    #[error("sandbox is not observed in Linux seccomp filter mode: {observed:?}")]
    FilterModeNotObserved { observed: Option<u32> },
    #[error("sandbox is not observed with NoNewPrivs=1: {observed:?}")]
    NoNewPrivsNotObserved { observed: Option<u32> },
    #[error("seccomp observation gate does not bind the supplied kernel evidence")]
    ScopeMismatch,
    #[error("seccomp observation gate identity does not match canonical fields")]
    IdentityMismatch,
}

/// Deliberate nonclaim: procfs proves that *a* seccomp filter is active, not which filter semantics
/// produced that state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SeccompFilterSemanticsStatus {
    NotEstablishedV1,
}

/// Stronger parent-observed kernel gate requiring both the existing isolation theorem and active
/// seccomp filter/no-new-privileges state on the same exact process identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelSeccompObservationGate {
    id: ContentId,
    kernel_gate_id: ContentId,
    observation_id: ContentId,
    host_pid: u32,
    process_start_time_ticks: u64,
    seccomp_mode: u32,
    no_new_privs: u32,
    filter_semantics: SeccompFilterSemanticsStatus,
}

impl KernelSeccompObservationGate {
    pub fn issue(
        kernel_gate: &KernelIsolationGate,
        observation: &KernelSandboxObservation,
    ) -> Result<Self, SeccompObservationError> {
        kernel_gate.validate_for(observation)?;
        let seccomp_mode = observation.seccomp_mode().ok_or(
            SeccompObservationError::FilterModeNotObserved { observed: None },
        )?;
        if seccomp_mode != SECCOMP_MODE_FILTER {
            return Err(SeccompObservationError::FilterModeNotObserved {
                observed: Some(seccomp_mode),
            });
        }
        let no_new_privs = observation.no_new_privs().ok_or(
            SeccompObservationError::NoNewPrivsNotObserved { observed: None },
        )?;
        if no_new_privs != NO_NEW_PRIVS_ENABLED {
            return Err(SeccompObservationError::NoNewPrivsNotObserved {
                observed: Some(no_new_privs),
            });
        }

        let id = derive_gate_id(
            kernel_gate.id(),
            observation.id(),
            observation.host_pid(),
            observation.process_start_time_ticks(),
            seccomp_mode,
            no_new_privs,
        );
        Ok(Self {
            id,
            kernel_gate_id: kernel_gate.id().clone(),
            observation_id: observation.id().clone(),
            host_pid: observation.host_pid(),
            process_start_time_ticks: observation.process_start_time_ticks(),
            seccomp_mode,
            no_new_privs,
            filter_semantics: SeccompFilterSemanticsStatus::NotEstablishedV1,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn kernel_gate_id(&self) -> &ContentId { &self.kernel_gate_id }
    pub fn observation_id(&self) -> &ContentId { &self.observation_id }
    pub fn host_pid(&self) -> u32 { self.host_pid }
    pub fn process_start_time_ticks(&self) -> u64 { self.process_start_time_ticks }
    pub fn seccomp_mode(&self) -> u32 { self.seccomp_mode }
    pub fn no_new_privs(&self) -> u32 { self.no_new_privs }
    pub fn filter_semantics(&self) -> SeccompFilterSemanticsStatus { self.filter_semantics }

    pub fn validate_for(
        &self,
        kernel_gate: &KernelIsolationGate,
        observation: &KernelSandboxObservation,
    ) -> Result<(), SeccompObservationError> {
        let rebuilt = Self::issue(kernel_gate, observation)?;
        if self.kernel_gate_id != *kernel_gate.id()
            || self.observation_id != *observation.id()
            || self.host_pid != observation.host_pid()
            || self.process_start_time_ticks != observation.process_start_time_ticks()
            || self.seccomp_mode != SECCOMP_MODE_FILTER
            || self.no_new_privs != NO_NEW_PRIVS_ENABLED
            || self.filter_semantics != SeccompFilterSemanticsStatus::NotEstablishedV1
        {
            return Err(SeccompObservationError::ScopeMismatch);
        }
        if rebuilt == *self {
            Ok(())
        } else {
            Err(SeccompObservationError::IdentityMismatch)
        }
    }
}

fn derive_gate_id(
    kernel_gate_id: &ContentId,
    observation_id: &ContentId,
    host_pid: u32,
    process_start_time_ticks: u64,
    seccomp_mode: u32,
    no_new_privs: u32,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-kernel-seccomp-observation-gate.v1",
        [
            kernel_gate_id.as_str().as_bytes(),
            observation_id.as_str().as_bytes(),
            host_pid.to_be_bytes().as_slice(),
            process_start_time_ticks.to_be_bytes().as_slice(),
            seccomp_mode.to_be_bytes().as_slice(),
            no_new_privs.to_be_bytes().as_slice(),
            b"filter-semantics-not-established-v1",
        ],
    )
}

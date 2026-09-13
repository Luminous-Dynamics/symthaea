// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only TPM2 counter evidence adapter.

#![deny(unsafe_code)]

mod parser;
#[cfg(target_os = "linux")]
mod system;

use std::path::Path;

use serde::{Deserialize, Serialize};
use symthaea_assurance_policy_lineage_anchor::PolicyLineageAnchor;
use symthaea_assurance_trust_store::{TrustStoreCheckpoint, TrustStoreProfile};

pub use parser::{parse_nv_public, Tpm2NvPublicEvidence};
#[cfg(target_os = "linux")]
pub use system::SystemTpm2ToolsExecutor;

const TPM_NV_HANDLE_PREFIX: u32 = 0x0100_0000;
const TPM_NV_HANDLE_MASK: u32 = 0xff00_0000;
pub const COUNTER_BYTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tpm2ReadHierarchy {
    Owner,
    Platform,
    Index,
}

impl Tpm2ReadHierarchy {
    fn command_value(self, nv_index: u32) -> String {
        match self {
            Self::Owner => "o".into(),
            Self::Platform => "p".into(),
            Self::Index => format!("{nv_index:#x}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2ToolsAdapterPolicy {
    pub schema_version: String,
    pub adapter_id: String,
    pub logical_store_id: String,
    pub trust_store_ref: String,
    pub counter_epoch: String,
    pub nv_index: u32,
    pub tcti: String,
    pub read_hierarchy: Tpm2ReadHierarchy,
    pub nvreadpublic_path: String,
    pub nvread_path: String,
    pub expected_nvreadpublic_blake3: String,
    pub expected_nvread_blake3: String,
    pub minimum_counter_value: u64,
    pub evidence_refs: Vec<String>,
}

impl Tpm2ToolsAdapterPolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.adapter_id.trim().is_empty()
            && !self.logical_store_id.trim().is_empty()
            && !self.trust_store_ref.trim().is_empty()
            && !self.counter_epoch.trim().is_empty()
            && self.nv_index & TPM_NV_HANDLE_MASK == TPM_NV_HANDLE_PREFIX
            && matches!(self.tcti.as_str(), "device:/dev/tpmrm0" | "device:/dev/tpm0")
            && Path::new(&self.nvreadpublic_path).is_absolute()
            && Path::new(&self.nvread_path).is_absolute()
            && valid_blake3_digest(&self.expected_nvreadpublic_blake3)
            && valid_blake3_digest(&self.expected_nvread_blake3)
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecution {
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

pub trait Tpm2ToolsExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, Tpm2AdapterError>;
    fn execute(&self, executable: &str, args: &[String]) -> Result<ToolExecution, Tpm2AdapterError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2NvCounterObservation {
    pub adapter_id: String,
    pub logical_store_id: String,
    pub trust_store_ref: String,
    pub counter_epoch: String,
    pub nv_index: u32,
    pub tcti: String,
    pub counter_value: u64,
    pub observed_at_ms: u64,
    pub nvreadpublic_blake3: String,
    pub nvread_blake3: String,
    pub public_evidence: Tpm2NvPublicEvidence,
    pub raw_counter_blake3: String,
    pub evidence_refs: Vec<String>,
}

impl Tpm2NvCounterObservation {
    pub fn validate(&self, policy: &Tpm2ToolsAdapterPolicy) -> bool {
        policy.validate()
            && self.adapter_id == policy.adapter_id
            && self.logical_store_id == policy.logical_store_id
            && self.trust_store_ref == policy.trust_store_ref
            && self.counter_epoch == policy.counter_epoch
            && self.nv_index == policy.nv_index
            && self.tcti == policy.tcti
            && self.counter_value >= policy.minimum_counter_value
            && self.nvreadpublic_blake3 == policy.expected_nvreadpublic_blake3
            && self.nvread_blake3 == policy.expected_nvread_blake3
            && self.public_evidence.validate(policy.nv_index)
            && valid_blake3_digest(&self.raw_counter_blake3)
            && !self.evidence_refs.is_empty()
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointBindingContext {
    pub checkpoint_id: String,
    pub store_revision: u64,
    pub predecessor_checkpoint_digest: Option<String>,
    pub attestation_ref: String,
    pub independent_verification_ref: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tpm2AdapterError {
    InvalidPolicy,
    ExecutableDigestMismatch,
    Io(String),
    ToolFailed,
    ToolEmittedStderr,
    InvalidPublicOutput,
    NvIndexNotPresent,
    MissingAttributes,
    MissingDataSize,
    NvIndexIsNotCounter,
    UnexpectedDataSize(usize),
    UnexpectedCounterReadLength(usize),
    CounterBelowProvisionedFloor,
    InvalidObservation,
    InvalidCheckpointBindingContext,
    TrustStoreProfileMismatch,
    AnchorMismatch,
    CheckpointValidationFailed,
}

pub fn read_tpm2_nv_counter(
    policy: &Tpm2ToolsAdapterPolicy,
    observed_at_ms: u64,
    executor: &impl Tpm2ToolsExecutor,
) -> Result<Tpm2NvCounterObservation, Tpm2AdapterError> {
    if !policy.validate() { return Err(Tpm2AdapterError::InvalidPolicy); }
    verify_tool(executor, &policy.nvreadpublic_path, &policy.expected_nvreadpublic_blake3)?;
    verify_tool(executor, &policy.nvread_path, &policy.expected_nvread_blake3)?;

    let handle = format!("{:#x}", policy.nv_index);
    let public_run = executor.execute(
        &policy.nvreadpublic_path,
        &["-T".into(), policy.tcti.clone(), handle.clone()],
    )?;
    clean_success(&public_run)?;
    let public_evidence = parse_nv_public(&public_run.stdout, policy.nv_index)?;

    let read_run = executor.execute(
        &policy.nvread_path,
        &[
            "-T".into(), policy.tcti.clone(),
            "-C".into(), policy.read_hierarchy.command_value(policy.nv_index),
            "-s".into(), COUNTER_BYTES.to_string(), handle,
        ],
    )?;
    clean_success(&read_run)?;
    if read_run.stdout.len() != COUNTER_BYTES {
        return Err(Tpm2AdapterError::UnexpectedCounterReadLength(read_run.stdout.len()));
    }
    let mut bytes = [0_u8; COUNTER_BYTES];
    bytes.copy_from_slice(&read_run.stdout);
    let counter_value = u64::from_be_bytes(bytes);
    if counter_value < policy.minimum_counter_value {
        return Err(Tpm2AdapterError::CounterBelowProvisionedFloor);
    }

    let observation = Tpm2NvCounterObservation {
        adapter_id: policy.adapter_id.clone(),
        logical_store_id: policy.logical_store_id.clone(),
        trust_store_ref: policy.trust_store_ref.clone(),
        counter_epoch: policy.counter_epoch.clone(),
        nv_index: policy.nv_index,
        tcti: policy.tcti.clone(),
        counter_value,
        observed_at_ms,
        nvreadpublic_blake3: policy.expected_nvreadpublic_blake3.clone(),
        nvread_blake3: policy.expected_nvread_blake3.clone(),
        public_evidence,
        raw_counter_blake3: blake3_digest(&read_run.stdout),
        evidence_refs: policy.evidence_refs.clone(),
    };
    observation.validate(policy).then_some(observation).ok_or(Tpm2AdapterError::InvalidObservation)
}

pub fn bind_observation_to_checkpoint(
    observation: &Tpm2NvCounterObservation,
    policy: &Tpm2ToolsAdapterPolicy,
    profile: &TrustStoreProfile,
    anchor: &PolicyLineageAnchor,
    context: &CheckpointBindingContext,
) -> Result<TrustStoreCheckpoint, Tpm2AdapterError> {
    if !observation.validate(policy) { return Err(Tpm2AdapterError::InvalidObservation); }
    if context.checkpoint_id.trim().is_empty()
        || context.store_revision == 0
        || context.attestation_ref.trim().is_empty()
        || context.independent_verification_ref.trim().is_empty()
        || context.evidence_refs.is_empty()
    { return Err(Tpm2AdapterError::InvalidCheckpointBindingContext); }
    if context.store_revision == 1 && context.predecessor_checkpoint_digest.is_some() {
        return Err(Tpm2AdapterError::InvalidCheckpointBindingContext);
    }
    if context.store_revision > 1 && context.predecessor_checkpoint_digest.is_none() {
        return Err(Tpm2AdapterError::InvalidCheckpointBindingContext);
    }
    if !profile.validate()
        || profile.store_id != policy.logical_store_id
        || profile.trust_store_ref != policy.trust_store_ref
        || profile.initial_counter_epoch != policy.counter_epoch
    { return Err(Tpm2AdapterError::TrustStoreProfileMismatch); }
    if !anchor.validate() || anchor.trust_store_ref != policy.trust_store_ref {
        return Err(Tpm2AdapterError::AnchorMismatch);
    }

    let mut evidence_refs = context.evidence_refs.clone();
    evidence_refs.push(format!("tpm2-counter:{}", observation.raw_counter_blake3));
    let checkpoint = TrustStoreCheckpoint {
        schema_version: "1".into(),
        checkpoint_id: context.checkpoint_id.clone(),
        store_id: profile.store_id.clone(),
        store_revision: context.store_revision,
        counter_epoch: observation.counter_epoch.clone(),
        counter_value: observation.counter_value,
        anchor_revision: anchor.anchor_revision,
        anchor_digest: anchor.anchor_digest(),
        policy_tip_revision: anchor.tip_revision,
        predecessor_checkpoint_digest: context.predecessor_checkpoint_digest.clone(),
        recorded_at_ms: observation.observed_at_ms,
        attestation_ref: context.attestation_ref.clone(),
        independent_verification_ref: context.independent_verification_ref.clone(),
        evidence_refs,
    };
    checkpoint.validate(profile).then_some(checkpoint).ok_or(Tpm2AdapterError::CheckpointValidationFailed)
}

fn verify_tool(executor: &impl Tpm2ToolsExecutor, path: &str, expected: &str) -> Result<(), Tpm2AdapterError> {
    (executor.executable_blake3(path)? == expected)
        .then_some(())
        .ok_or(Tpm2AdapterError::ExecutableDigestMismatch)
}

fn clean_success(run: &ToolExecution) -> Result<(), Tpm2AdapterError> {
    if run.exit_code != Some(0) { return Err(Tpm2AdapterError::ToolFailed); }
    if !run.stderr.is_empty() { return Err(Tpm2AdapterError::ToolEmittedStderr); }
    Ok(())
}

pub(crate) fn blake3_digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

pub(crate) fn valid_blake3_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

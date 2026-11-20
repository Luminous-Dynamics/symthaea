// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure host-side validation and failure-state tracking for COMPUTE-EXEC-001.
//!
//! This module deliberately performs no MMIO, DMA mapping, cache maintenance,
//! device discovery, cryptographic attestation, or hardware qualification.
//! It validates protocol objects before a driver is permitted to act on them.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const PROTOCOL_VERSION: &str = "0.1.0";
pub const REPRESENTATION_ID: &str = "luminous.binaryhv.bipolar.16384.v1";
pub const DIMENSION_BITS: u64 = 16_384;
pub const VECTOR_BYTES: u64 = 2_048;
pub const RESULT_RECORD_BYTES: u64 = 16;
const MAX_TIMEOUT_MS: u32 = 60_000;
const MAX_CANDIDATES: u64 = u32::MAX;

pub type ValidationResult<T> = Result<T, ValidationError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub code: &'static str,
    pub detail: String,
}

impl ValidationError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for ValidationError {}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceCapabilities {
    pub schema: String,
    pub schema_version: String,
    pub device_id: String,
    pub device_epoch: String,
    pub backend: String,
    pub execution_class: String,
    pub supported_operations: Vec<String>,
    pub supported_representations: Vec<String>,
    pub max_transfer_bytes: String,
    pub required_alignment_bytes: u64,
    pub max_inflight_commands: u64,
    pub dma_isolation: String,
    pub cache_coherency: String,
    pub implementation_sha256: String,
    pub hardware_profile_sha256: Option<String>,
    pub firmware_sha256: Option<String>,
    pub bitstream_sha256: Option<String>,
    pub hardware_identity: Option<HardwareIdentity>,
    pub qualification_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareIdentity {
    pub manufacturer: String,
    pub board_model: String,
    pub board_revision: String,
    pub device_part: String,
    pub som_ordering_code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticContract {
    pub contract_id: String,
    pub contract_version: String,
    pub contract_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Representation {
    pub representation_id: String,
    pub dimension_bits: u64,
    pub payload_bytes: u64,
    pub bit_order: String,
    pub byte_order: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandParameters {
    pub candidate_count: Option<u64>,
    pub tie_break_policy: Option<String>,
    pub result_record_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferDescriptor {
    pub role: String,
    pub handle_id: String,
    pub generation: String,
    pub offset_bytes: String,
    pub length_bytes: String,
    pub access: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HdcCommand {
    pub schema: String,
    pub schema_version: String,
    pub command_id: String,
    pub device_id: String,
    pub device_epoch: String,
    pub sequence: String,
    pub operation_id: String,
    pub semantic_contract: SemanticContract,
    pub representation: Representation,
    pub workload_manifest_sha256: String,
    pub implementation_sha256: String,
    pub timeout_ms: u32,
    pub parameters: CommandParameters,
    pub buffers: Vec<BufferDescriptor>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputBufferRef {
    pub role: String,
    pub handle_id: String,
    pub generation: String,
    pub offset_bytes: String,
    pub length_bytes: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionResult {
    pub output_sha256: String,
    pub winner_stable_id: Option<String>,
    pub hamming_distance_bits: Option<u64>,
    pub result_record_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HdcCompletion {
    pub schema: String,
    pub schema_version: String,
    pub command_id: String,
    pub device_id: String,
    pub device_epoch: String,
    pub sequence: String,
    pub operation_id: String,
    pub semantic_contract_sha256: String,
    pub implementation_sha256: String,
    pub status: String,
    pub error_code: Option<String>,
    pub error_detail: Option<String>,
    pub result: Option<CompletionResult>,
    pub output_buffer: Option<OutputBufferRef>,
    pub device_profile_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPermissions {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl BufferPermissions {
    fn permits(self, access: &str) -> bool {
        matches!(
            (self, access),
            (Self::ReadOnly, "read")
                | (Self::WriteOnly, "write")
                | (Self::ReadWrite, "read")
                | (Self::ReadWrite, "write")
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferOwnership {
    HostOwned,
    /// DMA returned the buffer, but digest and independent-oracle checks have
    /// not completed. The buffer is not reusable until verification succeeds.
    OutputUnverified {
        command_id: String,
        device_epoch: String,
        sequence: u64,
    },
    DeviceOwned {
        device_epoch: String,
        sequence: u64,
    },
    Quarantined {
        device_epoch: String,
        sequence: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferKey {
    handle_id: String,
    generation: u64,
}

#[derive(Debug, Clone)]
struct RegisteredBuffer {
    byte_length: u64,
    permissions: BufferPermissions,
    ownership: BufferOwnership,
}

#[derive(Debug, Default)]
pub struct BufferRegistry {
    buffers: HashMap<BufferKey, RegisteredBuffer>,
}

impl BufferRegistry {
    /// Register an opaque handle and generation. This records host metadata;
    /// it does not pin pages, map an IOMMU, or create a DMA mapping.
    pub fn register(
        &mut self,
        handle_id: impl Into<String>,
        generation: u64,
        byte_length: u64,
        permissions: BufferPermissions,
    ) -> ValidationResult<()> {
        let handle_id = handle_id.into();
        if handle_id.is_empty() || handle_id.len() > 128 || byte_length == 0 {
            return Err(ValidationError::new(
                "INVALID_BUFFER_REGISTRATION",
                "handle ID and byte length must be non-empty and bounded",
            ));
        }
        if generation == 0 {
            return Err(ValidationError::new(
                "INVALID_BUFFER_GENERATION",
                "registered buffer generations start at one",
            ));
        }
        let key = BufferKey {
            handle_id,
            generation,
        };
        if self.buffers.contains_key(&key) {
            return Err(ValidationError::new(
                "DUPLICATE_BUFFER_GENERATION",
                "the handle generation is already registered",
            ));
        }
        self.buffers.insert(
            key,
            RegisteredBuffer {
                byte_length,
                permissions,
                ownership: BufferOwnership::HostOwned,
            },
        );
        Ok(())
    }

    pub fn ownership(&self, handle_id: &str, generation: u64) -> Option<&BufferOwnership> {
        self.buffers
            .get(&BufferKey {
                handle_id: handle_id.to_owned(),
                generation,
            })
            .map(|buffer| &buffer.ownership)
    }

    fn entry(&self, key: &BufferKey) -> ValidationResult<&RegisteredBuffer> {
        self.buffers.get(key).ok_or_else(|| {
            ValidationError::new(
                "UNREGISTERED_BUFFER",
                format!(
                    "no registered buffer for handle {} generation {}",
                    key.handle_id, key.generation
                ),
            )
        })
    }

    fn set_ownership(&mut self, key: &BufferKey, ownership: BufferOwnership) {
        if let Some(buffer) = self.buffers.get_mut(key) {
            buffer.ownership = ownership;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionDisposition {
    /// DMA ownership has returned to the host and the output is available.
    /// The caller must still verify output bytes/digest and oracle parity.
    CompletedOutputAvailable,
    RejectedBeforeExecution,
    Quarantined,
}

/// Proof token created only by verify_exact_output_bytes after the actual
/// output bytes pass digest, operation-layout and independent-oracle checks.
/// Private fields prevent callers from constructing a token themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExactOutput {
    command_id: String,
    device_id: String,
    device_epoch: String,
    sequence: u64,
    operation_id: String,
    output_handle_id: String,
    output_generation: u64,
    output_sha256: String,
    output_bytes: u64,
}

impl VerifiedExactOutput {
    pub fn output_sha256(&self) -> &str {
        &self.output_sha256
    }

    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }
}

#[derive(Debug, Clone)]
struct PendingCommand {
    command_id: String,
    operation_id: String,
    semantic_contract_sha256: String,
    implementation_sha256: String,
    output: BufferDescriptor,
    buffers: Vec<BufferKey>,
}

/// A deterministic per-device command ledger.
///
/// The ledger is intentionally conservative: ambiguous outcomes, failed or
/// cancelled work, and reset-interrupted commands retain buffer quarantine
/// until the caller explicitly confirms DMA quiescence for the old epoch.
#[derive(Debug)]
pub struct ExecutionLedger {
    device_id: String,
    device_epoch: String,
    last_sequence: u64,
    max_inflight_commands: usize,
    /// A timeout, reset race or ambiguous device failure blocks all new work
    /// in this epoch until a new epoch is established through reset/recovery.
    submission_blocked: bool,
    /// Old epochs whose DMA contexts must be independently proven quiescent.
    quiescence_required_epochs: HashSet<String>,
    /// A new epoch must be established after an uncertain outcome before
    /// confirmation of old-epoch quiescence can resume dispatch.
    recovery_epoch_pending: bool,
    pending: HashMap<u64, PendingCommand>,
    seen_command_ids: HashSet<String>,
}

impl ExecutionLedger {
    pub fn new(
        device_id: impl Into<String>,
        device_epoch: impl Into<String>,
        max_inflight_commands: usize,
    ) -> ValidationResult<Self> {
        let device_id = device_id.into();
        let device_epoch = device_epoch.into();
        validate_epoch(&device_epoch)?;
        if device_id.is_empty() || max_inflight_commands == 0 {
            return Err(ValidationError::new(
                "INVALID_LEDGER_CONFIGURATION",
                "device ID and in-flight limit must be non-empty/non-zero",
            ));
        }
        Ok(Self {
            device_id,
            device_epoch,
            last_sequence: 0,
            max_inflight_commands,
            submission_blocked: false,
            quiescence_required_epochs: HashSet::new(),
            recovery_epoch_pending: false,
            pending: HashMap::new(),
            seen_command_ids: HashSet::new(),
        })
    }

    pub fn current_epoch(&self) -> &str {
        &self.device_epoch
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Validate the command and reserve every referenced handle before a
    /// driver is allowed to submit any work to a device.
    pub fn submit(
        &mut self,
        command: &HdcCommand,
        capabilities: &DeviceCapabilities,
        registry: &mut BufferRegistry,
    ) -> ValidationResult<()> {
        if self.submission_blocked {
            return Err(ValidationError::new(
                "DEVICE_EPOCH_REQUIRES_RECONCILIATION",
                "no new work is admitted in an uncertain device epoch; reset/reconcile first",
            ));
        }
        let sequence = parse_decimal_u64(&command.sequence, false, "sequence")?;

        if self.device_id != command.device_id || self.device_id != capabilities.device_id {
            return Err(ValidationError::new(
                "DEVICE_ID_MISMATCH",
                "command, capability manifest and ledger must name the same device",
            ));
        }
        if self.device_epoch != command.device_epoch
            || self.device_epoch != capabilities.device_epoch
        {
            return Err(ValidationError::new(
                "STALE_DEVICE_EPOCH",
                "command/capability epoch differs from the active ledger epoch",
            ));
        }
        if sequence <= self.last_sequence {
            return Err(ValidationError::new(
                "NON_MONOTONIC_SEQUENCE",
                "sequence must increase strictly within a device epoch",
            ));
        }
        let capability_limit = usize::try_from(capabilities.max_inflight_commands)
            .unwrap_or(usize::MAX);
        if self.pending.len() >= self.max_inflight_commands.min(capability_limit) {
            return Err(ValidationError::new(
                "QUEUE_FULL",
                "the configured/capability in-flight command limit has been reached",
            ));
        }
        if self.seen_command_ids.contains(&command.command_id) {
            return Err(ValidationError::new(
                "DUPLICATE_COMMAND_ID",
                "command IDs may not be reused during this ledger lifetime",
            ));
        }

        validate_command(command, capabilities, registry)?;
        let keys = collect_buffer_keys(&command.buffers)?;
        for key in &keys {
            let entry = registry.entry(key)?;
            if entry.ownership != BufferOwnership::HostOwned {
                return Err(ValidationError::new(
                    "BUFFER_NOT_HOST_OWNED",
                    format!("buffer {} is not available to the host", key.handle_id),
                ));
            }
        }

        for key in &keys {
            registry.set_ownership(
                key,
                BufferOwnership::DeviceOwned {
                    device_epoch: self.device_epoch.clone(),
                    sequence,
                },
            );
        }

        let output = command
            .buffers
            .iter()
            .find(|buffer| buffer.role == output_role(&command.operation_id))
            .cloned()
            .ok_or_else(|| {
                ValidationError::new("MISSING_OUTPUT_BUFFER", "validated output disappeared")
            })?;

        self.pending.insert(
            sequence,
            PendingCommand {
                command_id: command.command_id.clone(),
                operation_id: command.operation_id.clone(),
                semantic_contract_sha256: command.semantic_contract.contract_sha256.clone(),
                implementation_sha256: command.implementation_sha256.clone(),
                output,
                buffers: keys,
            },
        );
        self.last_sequence = sequence;
        self.seen_command_ids.insert(command.command_id.clone());
        Ok(())
    }

    /// Accept only a completion matching the active command, device epoch,
    /// sequence and exact output buffer generation/range.
    pub fn accept_completion(
        &mut self,
        completion: &HdcCompletion,
        capabilities: &DeviceCapabilities,
        registry: &mut BufferRegistry,
    ) -> ValidationResult<CompletionDisposition> {
        validate_completion_envelope(completion)?;
        let sequence = parse_decimal_u64(&completion.sequence, false, "sequence")?;

        if completion.device_id != self.device_id || completion.device_epoch != self.device_epoch {
            return Err(ValidationError::new(
                "STALE_OR_FOREIGN_COMPLETION",
                "completion belongs to another device or device epoch",
            ));
        }

        validate_capabilities(capabilities)?;
        if capabilities.device_id != self.device_id || capabilities.device_epoch != self.device_epoch {
            return Err(ValidationError::new(
                "CAPABILITY_IDENTITY_MISMATCH",
                "completion validation requires the active device capability profile",
            ));
        }

        let pending = self.pending.get(&sequence).cloned().ok_or_else(|| {
            ValidationError::new(
                "UNKNOWN_COMPLETION",
                "no pending command exists for this device epoch and sequence",
            )
        })?;

        if capabilities.implementation_sha256 != pending.implementation_sha256 {
            return Err(ValidationError::new(
                "IMPLEMENTATION_PROFILE_CHANGED",
                "device implementation identity changed while a command was in flight",
            ));
        }
        if completion.implementation_sha256 != pending.implementation_sha256 {
            return Err(ValidationError::new(
                "COMPLETION_IMPLEMENTATION_MISMATCH",
                "completion implementation digest differs from the submitted command",
            ));
        }
        if completion.semantic_contract_sha256 != pending.semantic_contract_sha256 {
            return Err(ValidationError::new(
                "COMPLETION_SEMANTIC_CONTRACT_MISMATCH",
                "completion semantic-contract digest differs from the submitted command",
            ));
        }
        if completion.operation_id != pending.operation_id {
            return Err(ValidationError::new(
                "COMPLETION_OPERATION_MISMATCH",
                "completion operation differs from the submitted semantic operation",
            ));
        }

        if completion.command_id != pending.command_id {
            return Err(ValidationError::new(
                "COMMAND_ID_MISMATCH",
                "completion command ID does not match the submitted command",
            ));
        }

        match completion.status.as_str() {
            "completed" => {
                validate_success_result(completion, &pending, capabilities)?;
                let output_generation = parse_decimal_u64(
                    &pending.output.generation,
                    false,
                    "command.output.generation",
                )?;
                let output_key = BufferKey {
                    handle_id: pending.output.handle_id.clone(),
                    generation: output_generation,
                };
                for key in &pending.buffers {
                    if *key == output_key {
                        registry.set_ownership(
                            key,
                            BufferOwnership::OutputUnverified {
                                command_id: pending.command_id.clone(),
                                device_epoch: self.device_epoch.clone(),
                                sequence,
                            },
                        );
                    } else {
                        registry.set_ownership(key, BufferOwnership::HostOwned);
                    }
                }
                self.pending.remove(&sequence);
                Ok(CompletionDisposition::CompletedOutputAvailable)
            }
            "rejected" => {
                // Contract meaning: the device rejected the descriptor before
                // execution and therefore never acquired DMA ownership.
                for key in &pending.buffers {
                    registry.set_ownership(key, BufferOwnership::HostOwned);
                }
                self.pending.remove(&sequence);
                Ok(CompletionDisposition::RejectedBeforeExecution)
            }
            "failed" | "cancelled" | "outcome_unknown" => {
                self.submission_blocked = true;
                self.recovery_epoch_pending = true;
                self.quiescence_required_epochs.insert(self.device_epoch.clone());
                for key in &pending.buffers {
                    registry.set_ownership(
                        key,
                        BufferOwnership::Quarantined {
                            device_epoch: self.device_epoch.clone(),
                            sequence,
                        },
                    );
                }
                self.pending.remove(&sequence);
                Ok(CompletionDisposition::Quarantined)
            }
            _ => Err(ValidationError::new(
                "UNKNOWN_COMPLETION_STATUS",
                "completion status is not admitted by protocol v0.1",
            )),
        }
    }

    /// Return an output buffer to reusable host ownership only when a proof
    /// token produced by verify_exact_output_bytes binds the exact command,
    /// device epoch, output handle and generation.
    pub fn confirm_output_verified(
        &mut self,
        verified: &VerifiedExactOutput,
        registry: &mut BufferRegistry,
    ) -> ValidationResult<()> {
        let key = BufferKey {
            handle_id: verified.output_handle_id.clone(),
            generation: verified.output_generation,
        };
        let ownership = registry.entry(&key)?.ownership.clone();
        match ownership {
            BufferOwnership::OutputUnverified {
                command_id,
                device_epoch,
                sequence,
            } if command_id == verified.command_id
                && device_epoch == verified.device_epoch
                && sequence == verified.sequence
                && self.device_id == verified.device_id
                && self.device_epoch == verified.device_epoch =>
            {
                registry.set_ownership(&key, BufferOwnership::HostOwned);
                Ok(())
            }
            _ => Err(ValidationError::new(
                "OUTPUT_VERIFICATION_OWNERSHIP_MISMATCH",
                "verification token does not match an unverified output owned by this ledger",
            )),
        }
    }

    /// Move a timed-out command into quarantine. A timeout is not proof that
    /// the device did not execute or stop touching the submitted buffers.
    pub fn mark_timeout(
        &mut self,
        sequence: u64,
        registry: &mut BufferRegistry,
    ) -> ValidationResult<()> {
        let pending = self.pending.get(&sequence).cloned().ok_or_else(|| {
            ValidationError::new("UNKNOWN_TIMEOUT", "sequence is not pending")
        })?;
        self.submission_blocked = true;
        self.recovery_epoch_pending = true;
        self.quiescence_required_epochs.insert(self.device_epoch.clone());
        for key in &pending.buffers {
            registry.set_ownership(
                key,
                BufferOwnership::Quarantined {
                    device_epoch: self.device_epoch.clone(),
                    sequence,
                },
            );
        }
        self.pending.remove(&sequence);
        Ok(())
    }

    /// A reset invalidates every outstanding command and quarantines its
    /// buffers under the old epoch. The new epoch starts a fresh sequence.
    pub fn reset(
        &mut self,
        new_epoch: impl Into<String>,
        registry: &mut BufferRegistry,
    ) -> ValidationResult<()> {
        let new_epoch = new_epoch.into();
        validate_epoch(&new_epoch)?;
        if new_epoch == self.device_epoch {
            return Err(ValidationError::new(
                "EPOCH_NOT_ROTATED",
                "a reset must establish a distinct device epoch",
            ));
        }
        let old_epoch = self.device_epoch.clone();
        let mut reset_interrupted_work = false;
        for (sequence, pending) in self.pending.drain() {
            reset_interrupted_work = true;
            for key in pending.buffers {
                registry.set_ownership(
                    &key,
                    BufferOwnership::Quarantined {
                        device_epoch: old_epoch.clone(),
                        sequence,
                    },
                );
            }
        }
        let mut unverified_output_interrupted = false;
        for buffer in registry.buffers.values_mut() {
            if let BufferOwnership::OutputUnverified {
                device_epoch,
                sequence,
                ..
            } = buffer.ownership.clone()
            {
                if device_epoch == old_epoch {
                    unverified_output_interrupted = true;
                    buffer.ownership = BufferOwnership::Quarantined {
                        device_epoch: old_epoch.clone(),
                        sequence,
                    };
                }
            }
        }
        if reset_interrupted_work
            || unverified_output_interrupted
            || registry.buffers.values().any(|buffer| {
                matches!(
                    &buffer.ownership,
                    BufferOwnership::Quarantined { device_epoch, .. } if device_epoch == old_epoch
                )
            })
        {
            self.quiescence_required_epochs.insert(old_epoch);
        }

        self.device_epoch = new_epoch;
        self.last_sequence = 0;
        self.recovery_epoch_pending = false;
        self.submission_blocked = !self.quiescence_required_epochs.is_empty();
        Ok(())
    }

    /// Release only buffers quarantined under the supplied epoch. The caller
    /// must invoke this after an independent driver/platform quiescence check;
    /// this pure state machine cannot establish that physical fact itself.
    /// Dispatch resumes only when a new epoch has also been established.
    pub fn confirm_dma_quiescent(
        &mut self,
        old_epoch: &str,
        registry: &mut BufferRegistry,
    ) -> usize {
        if !self.quiescence_required_epochs.remove(old_epoch) {
            return 0;
        }

        let mut released = 0usize;
        for buffer in registry.buffers.values_mut() {
            if matches!(
                &buffer.ownership,
                BufferOwnership::Quarantined { device_epoch, .. } if device_epoch == old_epoch
            ) {
                buffer.ownership = BufferOwnership::HostOwned;
                released += 1;
            }
        }
        self.submission_blocked =
            self.recovery_epoch_pending || !self.quiescence_required_epochs.is_empty();
        released
    }
}

pub fn validate_capabilities(capabilities: &DeviceCapabilities) -> ValidationResult<()> {
    if capabilities.schema != "luminous.hdc-device-capabilities"
        || capabilities.schema_version != PROTOCOL_VERSION
    {
        return Err(ValidationError::new(
            "UNSUPPORTED_CAPABILITY_SCHEMA",
            "capability schema identity/version is not supported",
        ));
    }
    if capabilities.device_id.is_empty() || capabilities.device_id.len() > 128 {
        return Err(ValidationError::new(
            "INVALID_DEVICE_ID",
            "device ID must contain 1..=128 bytes",
        ));
    }
    validate_epoch(&capabilities.device_epoch)?;
    validate_sha256(&capabilities.implementation_sha256, "implementation_sha256")?;

    let max_transfer =
        parse_decimal_u64(&capabilities.max_transfer_bytes, false, "max_transfer_bytes")?;
    if max_transfer == 0
        || capabilities.required_alignment_bytes == 0
        || !capabilities.required_alignment_bytes.is_power_of_two()
        || capabilities.max_inflight_commands == 0
        || capabilities.max_inflight_commands > u16::MAX as u64
    {
        return Err(ValidationError::new(
            "INVALID_CAPABILITY_LIMIT",
            "transfer, power-of-two alignment and in-flight limits must be positive and bounded",
        ));
    }

    let allowed_operations = [
        "hdc.bind_xor.exact.v1",
        "hdc.hamming.exact.v1",
        "hdc.top1.exact.v1",
    ];
    if capabilities.supported_operations.is_empty() {
        return Err(ValidationError::new(
            "EMPTY_SUPPORTED_OPERATIONS",
            "a device must advertise at least one exact operation",
        ));
    }
    let mut seen_operations = HashSet::new();
    for operation in &capabilities.supported_operations {
        if !allowed_operations.contains(&operation.as_str())
            || !seen_operations.insert(operation.as_str())
        {
            return Err(ValidationError::new(
                "INVALID_SUPPORTED_OPERATIONS",
                "supported operations must be unique members of the v0.1 exact operation set",
            ));
        }
    }
    let unique_representations: HashSet<&str> = capabilities
        .supported_representations
        .iter()
        .map(String::as_str)
        .collect();
    if unique_representations.len() != capabilities.supported_representations.len()
        || !capabilities
            .supported_representations
            .iter()
            .any(|representation| representation == REPRESENTATION_ID)
    {
        return Err(ValidationError::new(
            "UNSUPPORTED_REPRESENTATION",
            "the canonical 16,384-bit BinaryHV representation is not advertised",
        ));
    }

    let backend_class_matches = matches!(
        (capabilities.backend.as_str(), capabilities.execution_class.as_str()),
        ("cpu_reference", "software_reference")
            | ("cpu_simd", "software_accelerator")
            | ("vulkan", "software_accelerator")
            | ("webgpu", "software_accelerator")
            | ("fpga", "rtl_simulation")
            | ("fpga", "synthesis_estimate")
            | ("fpga", "fpga_hardware")
            | ("asic", "synthesis_estimate")
            | ("asic", "asic_hardware")
    );
    if !backend_class_matches {
        return Err(ValidationError::new(
            "BACKEND_EXECUTION_CLASS_MISMATCH",
            "backend and execution class are not a supported pairing",
        ));
    }

    if !matches!(
        capabilities.dma_isolation.as_str(),
        "iommu_domain" | "dedicated_bounded_pool" | "none"
    ) || !matches!(
        capabilities.cache_coherency.as_str(),
        "coherent" | "explicit_flush_invalidate" | "unknown"
    ) {
        return Err(ValidationError::new(
            "INVALID_MEMORY_PROFILE",
            "DMA isolation or cache-coherency value is outside the protocol enum",
        ));
    }

    if matches!(
        capabilities.execution_class.as_str(),
        "fpga_hardware" | "asic_hardware"
    ) {
        let hardware_profile = capabilities.hardware_profile_sha256.as_deref().ok_or_else(|| {
            ValidationError::new("MISSING_HARDWARE_PROFILE", "physical profile digest is required")
        })?;
        validate_sha256(hardware_profile, "hardware_profile_sha256")?;
        let firmware = capabilities.firmware_sha256.as_deref().ok_or_else(|| {
            ValidationError::new("MISSING_FIRMWARE_IDENTITY", "firmware digest is required")
        })?;
        validate_sha256(firmware, "firmware_sha256")?;

        let identity = capabilities.hardware_identity.as_ref().ok_or_else(|| {
            ValidationError::new("MISSING_HARDWARE_IDENTITY", "physical board identity is required")
        })?;
        if identity.manufacturer.is_empty()
            || identity.board_model.is_empty()
            || identity.board_revision.is_empty()
            || identity.device_part.is_empty()
        {
            return Err(ValidationError::new(
                "INCOMPLETE_HARDWARE_IDENTITY",
                "manufacturer, model, revision and FPGA/programmable-logic identity are required",
            ));
        }

        if capabilities.backend == "fpga" {
            let bitstream = capabilities.bitstream_sha256.as_deref().ok_or_else(|| {
                ValidationError::new("MISSING_BITSTREAM_IDENTITY", "FPGA bitstream digest is required")
            })?;
            validate_sha256(bitstream, "bitstream_sha256")?;
        }
        if !matches!(
            capabilities.dma_isolation.as_str(),
            "iommu_domain" | "dedicated_bounded_pool"
        ) {
            return Err(ValidationError::new(
                "UNSAFE_DMA_ISOLATION",
                "physical DMA requires an IOMMU domain or dedicated bounded DMA pool",
            ));
        }
        if !matches!(
            capabilities.cache_coherency.as_str(),
            "coherent" | "explicit_flush_invalidate"
        ) {
            return Err(ValidationError::new(
                "UNKNOWN_CACHE_COHERENCY",
                "physical DMA requires coherent memory or explicit cache maintenance",
            ));
        }
    }
    Ok(())
}

/// Validate protocol shape plus operation-specific semantic, capability,
/// buffer-range, ownership and aliasing constraints. No device action occurs.
pub fn validate_command(
    command: &HdcCommand,
    capabilities: &DeviceCapabilities,
    registry: &BufferRegistry,
) -> ValidationResult<()> {
    validate_capabilities(capabilities)?;

    if command.schema != "luminous.hdc-device-command"
        || command.schema_version != PROTOCOL_VERSION
    {
        return Err(ValidationError::new(
            "UNSUPPORTED_COMMAND_SCHEMA",
            "command schema identity/version is not supported",
        ));
    }
    if command.command_id.is_empty()
        || command.command_id.len() > 64
        || !command
            .command_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(ValidationError::new(
            "INVALID_COMMAND_ID",
            "command ID contains unsupported characters or length",
        ));
    }
    validate_epoch(&command.device_epoch)?;
    parse_decimal_u64(&command.sequence, false, "sequence")?;

    if command.device_id != capabilities.device_id
        || command.device_epoch != capabilities.device_epoch
    {
        return Err(ValidationError::new(
            "CAPABILITY_IDENTITY_MISMATCH",
            "command device ID and epoch must match the capability manifest",
        ));
    }
    if !capabilities
        .supported_operations
        .iter()
        .any(|operation| operation == &command.operation_id)
    {
        return Err(ValidationError::new(
            "UNSUPPORTED_OPERATION",
            "device did not advertise this exact operation",
        ));
    }
    if command.representation.representation_id != REPRESENTATION_ID
        || command.representation.dimension_bits != DIMENSION_BITS
        || command.representation.payload_bytes != VECTOR_BYTES
        || command.representation.bit_order != "lsb0"
        || command.representation.byte_order != "little"
    {
        return Err(ValidationError::new(
            "NON_CANONICAL_REPRESENTATION",
            "only the canonical 16,384-bit, 2,048-byte little-endian BinaryHV is admitted",
        ));
    }
    if command.timeout_ms == 0 || command.timeout_ms > MAX_TIMEOUT_MS {
        return Err(ValidationError::new(
            "INVALID_TIMEOUT",
            "timeout must be in 1..=60,000 milliseconds",
        ));
    }
    validate_sha256(
        &command.workload_manifest_sha256,
        "workload_manifest_sha256",
    )?;
    validate_sha256(&command.implementation_sha256, "implementation_sha256")?;
    if command.implementation_sha256 != capabilities.implementation_sha256 {
        return Err(ValidationError::new(
            "IMPLEMENTATION_IDENTITY_MISMATCH",
            "command and capability implementation digests differ",
        ));
    }
    validate_sha256(
        &command.semantic_contract.contract_sha256,
        "semantic_contract.contract_sha256",
    )?;
    if command.semantic_contract.contract_version != "1.0.0"
        || command.semantic_contract.contract_id != expected_contract_id(&command.operation_id)?
    {
        return Err(ValidationError::new(
            "SEMANTIC_CONTRACT_MISMATCH",
            "semantic contract ID/version does not match the admitted operation",
        ));
    }
    if command.semantic_contract.contract_sha256
        != expected_contract_sha256(&command.operation_id)?
    {
        return Err(ValidationError::new(
            "NON_CANONICAL_SEMANTIC_DIGEST",
            "semantic-contract digest is not the host-pinned canonical operation descriptor",
        ));
    }

    let expected = expected_buffers(command)?;
    if command.buffers.len() != expected.len() {
        return Err(ValidationError::new(
            "BUFFER_ROLE_CARDINALITY",
            "command must contain exactly the operation's required buffer roles",
        ));
    }

    let mut roles = HashMap::new();
    for buffer in &command.buffers {
        if roles.insert(buffer.role.as_str(), buffer).is_some() {
            return Err(ValidationError::new(
                "DUPLICATE_BUFFER_ROLE",
                format!("buffer role {} appears more than once", buffer.role),
            ));
        }
    }

    let max_transfer =
        parse_decimal_u64(&capabilities.max_transfer_bytes, false, "max_transfer_bytes")?;
    let mut total_transfer = 0u64;
    let mut resolved = Vec::with_capacity(command.buffers.len());

    for (role, expected_length, expected_access) in expected {
        let buffer = roles.get(role).ok_or_else(|| {
            ValidationError::new(
                "MISSING_BUFFER_ROLE",
                format!("required buffer role {role} is absent"),
            )
        })?;
        if buffer.access != expected_access {
            return Err(ValidationError::new(
                "BUFFER_ACCESS_MISMATCH",
                format!("role {role} requires {expected_access} access"),
            ));
        }
        let generation = parse_decimal_u64(&buffer.generation, false, "buffer.generation")?;
        let offset = parse_decimal_u64(&buffer.offset_bytes, true, "buffer.offset_bytes")?;
        let length = parse_decimal_u64(&buffer.length_bytes, false, "buffer.length_bytes")?;
        if length != expected_length {
            return Err(ValidationError::new(
                "BUFFER_LENGTH_MISMATCH",
                format!("role {role} must be {expected_length} bytes, got {length}"),
            ));
        }
        if offset % capabilities.required_alignment_bytes != 0 {
            return Err(ValidationError::new(
                "BUFFER_MISALIGNMENT",
                format!("role {role} offset does not satisfy device alignment"),
            ));
        }
        let end = offset.checked_add(length).ok_or_else(|| {
            ValidationError::new("BUFFER_RANGE_OVERFLOW", "offset + length overflowed u64")
        })?;
        let key = BufferKey {
            handle_id: buffer.handle_id.clone(),
            generation,
        };
        let registered = registry.entry(&key)?;
        if end > registered.byte_length {
            return Err(ValidationError::new(
                "BUFFER_RANGE_OUT_OF_BOUNDS",
                format!("role {role} range exceeds the registered handle extent"),
            ));
        }
        if !registered.permissions.permits(expected_access) {
            return Err(ValidationError::new(
                "REGISTERED_PERMISSION_MISMATCH",
                format!("registered handle for role {role} does not permit {expected_access}"),
            ));
        }
        if registered.ownership != BufferOwnership::HostOwned {
            return Err(ValidationError::new(
                "BUFFER_NOT_HOST_OWNED",
                format!("role {role} is device-owned or quarantined"),
            ));
        }
        total_transfer = total_transfer.checked_add(length).ok_or_else(|| {
            ValidationError::new("TRANSFER_SIZE_OVERFLOW", "aggregate transfer size overflowed")
        })?;
        resolved.push((buffer, key, offset, end));
    }

    if total_transfer > max_transfer {
        return Err(ValidationError::new(
            "TRANSFER_LIMIT_EXCEEDED",
            "aggregate command payload exceeds the capability transfer limit",
        ));
    }

    let output_role = output_role(&command.operation_id);
    let output = roles.get(output_role).ok_or_else(|| {
        ValidationError::new("MISSING_OUTPUT_BUFFER", "operation output role is absent")
    })?;
    let output_generation =
        parse_decimal_u64(&output.generation, false, "output.generation")?;
    let output_offset = parse_decimal_u64(&output.offset_bytes, true, "output.offset_bytes")?;
    let output_length = parse_decimal_u64(&output.length_bytes, false, "output.length_bytes")?;
    let output_end = output_offset.checked_add(output_length).ok_or_else(|| {
        ValidationError::new("OUTPUT_RANGE_OVERFLOW", "output offset + length overflowed u64")
    })?;
    let output_key = BufferKey {
        handle_id: output.handle_id.clone(),
        generation: output_generation,
    };
    for (buffer, key, start, end) in &resolved {
        if buffer.role == output_role {
            continue;
        }
        if key == &output_key && *start < output_end && output_offset < *end {
            return Err(ValidationError::new(
                "OUTPUT_INPUT_ALIAS",
                "output range overlaps an input range in the same registered buffer generation",
            ));
        }
    }
    Ok(())
}

fn expected_buffers(command: &HdcCommand) -> ValidationResult<Vec<(&'static str, u64, &'static str)>> {
    match command.operation_id.as_str() {
        "hdc.bind_xor.exact.v1" => {
            require_no_top1_parameters(&command.parameters)?;
            Ok(vec![
                ("input_a", VECTOR_BYTES, "read"),
                ("input_b", VECTOR_BYTES, "read"),
                ("output", VECTOR_BYTES, "write"),
            ])
        }
        "hdc.hamming.exact.v1" => {
            require_no_top1_parameters(&command.parameters)?;
            Ok(vec![
                ("input_a", VECTOR_BYTES, "read"),
                ("input_b", VECTOR_BYTES, "read"),
                ("result", 2, "write"),
            ])
        }
        "hdc.top1.exact.v1" => {
            let count = command.parameters.candidate_count.ok_or_else(|| {
                ValidationError::new("MISSING_CANDIDATE_COUNT", "Top-1 requires candidate_count")
            })?;
            if count == 0 || count > MAX_CANDIDATES {
                return Err(ValidationError::new(
                    "INVALID_CANDIDATE_COUNT",
                    "candidate_count must be in 1..=u32::MAX",
                ));
            }
            if command.parameters.tie_break_policy.as_deref() != Some("lowest_stable_id") {
                return Err(ValidationError::new(
                    "INVALID_TIE_BREAK_POLICY",
                    "Top-1 must break ties by lowest stable item ID",
                ));
            }
            if command.parameters.result_record_bytes != Some(RESULT_RECORD_BYTES) {
                return Err(ValidationError::new(
                    "INVALID_RESULT_RECORD_SIZE",
                    "Top-1 result record must be exactly 16 bytes",
                ));
            }
            let candidates = count.checked_mul(VECTOR_BYTES).ok_or_else(|| {
                ValidationError::new("CANDIDATE_SIZE_OVERFLOW", "candidate_count × vector bytes overflowed")
            })?;
            let ids = count.checked_mul(8).ok_or_else(|| {
                ValidationError::new("CANDIDATE_ID_SIZE_OVERFLOW", "candidate_count × 8 overflowed")
            })?;
            Ok(vec![
                ("query", VECTOR_BYTES, "read"),
                ("candidate_vectors", candidates, "read"),
                ("candidate_ids", ids, "read"),
                ("result", RESULT_RECORD_BYTES, "write"),
            ])
        }
        _ => Err(ValidationError::new(
            "UNKNOWN_OPERATION",
            "operation is not admitted by protocol v0.1",
        )),
    }
}

fn require_no_top1_parameters(parameters: &CommandParameters) -> ValidationResult<()> {
    if parameters.candidate_count.is_some()
        || parameters.tie_break_policy.is_some()
        || parameters.result_record_bytes.is_some()
    {
        return Err(ValidationError::new(
            "UNEXPECTED_OPERATION_PARAMETERS",
            "candidate_count, tie-break and result-record fields are Top-1-only",
        ));
    }
    Ok(())
}

fn expected_contract_id(operation: &str) -> ValidationResult<&'static str> {
    match operation {
        "hdc.bind_xor.exact.v1" => Ok("symthaea.binaryhv.bind-xor-exact"),
        "hdc.hamming.exact.v1" => Ok("symthaea.binaryhv.hamming-exact"),
        "hdc.top1.exact.v1" => Ok("symthaea.binaryhv.top1-exact"),
        _ => Err(ValidationError::new("UNKNOWN_OPERATION", "no semantic contract is registered")),
    }
}

/// SHA-256 digests of the normative canonical descriptors in
/// COMPUTE_EXEC_001_HDC_DEVICE_CONTRACT_V0_1.md. The exact descriptor bytes use
/// UTF-8, LF line endings and a final LF. Changing meaning requires a new
/// semantic version and digest, not a silent host-side update.
fn expected_contract_sha256(operation: &str) -> ValidationResult<&'static str> {
    match operation {
        "hdc.bind_xor.exact.v1" => {
            Ok("156f87ae7d96cdc89cf09d84723c2f65aa80f2c0fcfc193df06f8545ed332c67")
        }
        "hdc.hamming.exact.v1" => {
            Ok("c7e4cdcf7ee0ae962e1d4d033a125312a82e65a1aac7ce2f8d4f4d03b132712e")
        }
        "hdc.top1.exact.v1" => {
            Ok("04b106f50f40963f0d2f88f9d779b83d9a55bd22d2e151f45f5211fa05c3f9cd")
        }
        _ => Err(ValidationError::new("UNKNOWN_OPERATION", "no semantic digest is registered")),
    }
}

fn output_role(operation: &str) -> &'static str {
    if operation == "hdc.bind_xor.exact.v1" {
        "output"
    } else {
        "result"
    }
}

fn validate_completion_envelope(completion: &HdcCompletion) -> ValidationResult<()> {
    if completion.schema != "luminous.hdc-device-completion"
        || completion.schema_version != PROTOCOL_VERSION
    {
        return Err(ValidationError::new(
            "UNSUPPORTED_COMPLETION_SCHEMA",
            "completion schema identity/version is not supported",
        ));
    }
    validate_epoch(&completion.device_epoch)?;
    parse_decimal_u64(&completion.sequence, false, "completion.sequence")?;
    validate_sha256(&completion.implementation_sha256, "completion.implementation_sha256")?;
    validate_sha256(
        &completion.semantic_contract_sha256,
        "completion.semantic_contract_sha256",
    )?;
    if !matches!(
        completion.operation_id.as_str(),
        "hdc.bind_xor.exact.v1" | "hdc.hamming.exact.v1" | "hdc.top1.exact.v1"
    ) {
        return Err(ValidationError::new(
            "UNKNOWN_COMPLETION_OPERATION",
            "completion operation is not admitted by protocol v0.1",
        ));
    }
    if completion.command_id.is_empty() || completion.device_id.is_empty() {
        return Err(ValidationError::new(
            "INVALID_COMPLETION_IDENTITY",
            "completion command and device IDs must be non-empty",
        ));
    }
    match completion.status.as_str() {
        "completed" => {
            if completion.result.is_none()
                || completion.output_buffer.is_none()
                || completion.error_code.is_some()
            {
                return Err(ValidationError::new(
                    "MALFORMED_SUCCESS_COMPLETION",
                    "completed status requires result and output_buffer, and forbids error_code",
                ));
            }
        }
        "rejected" | "failed" | "cancelled" | "outcome_unknown" => {
            if completion.error_code.is_none()
                || completion.result.is_some()
                || completion.output_buffer.is_some()
            {
                return Err(ValidationError::new(
                    "MALFORMED_FAILURE_COMPLETION",
                    "failure status requires error_code and forbids result/output_buffer",
                ));
            }
            if completion.status == "outcome_unknown"
                && completion.error_code.as_deref() != Some("OUTCOME_UNKNOWN")
            {
                return Err(ValidationError::new(
                    "INVALID_UNKNOWN_OUTCOME_CODE",
                    "outcome_unknown must use the OUTCOME_UNKNOWN error code",
                ));
            }
        }
        _ => {
            return Err(ValidationError::new(
                "UNKNOWN_COMPLETION_STATUS",
                "completion status is not admitted by protocol v0.1",
            ));
        }
    }
    if let Some(code) = &completion.error_code {
        if code.is_empty() || code.len() > 128 {
            return Err(ValidationError::new(
                "INVALID_COMPLETION_ERROR_CODE",
                "error_code must contain 1..=128 bytes",
            ));
        }
    }
    if let Some(profile_sha256) = &completion.device_profile_sha256 {
        validate_sha256(profile_sha256, "device_profile_sha256")?;
    }
    if let Some(detail) = &completion.error_detail {
        if detail.len() > 2048 {
            return Err(ValidationError::new(
                "COMPLETION_DETAIL_TOO_LONG",
                "error_detail exceeds 2,048 bytes",
            ));
        }
    }
    Ok(())
}

fn validate_success_result(
    completion: &HdcCompletion,
    pending: &PendingCommand,
    capabilities: &DeviceCapabilities,
) -> ValidationResult<()> {
    let result = completion.result.as_ref().ok_or_else(|| {
        ValidationError::new("MISSING_COMPLETION_RESULT", "successful completion has no result")
    })?;
    validate_sha256(&result.output_sha256, "result.output_sha256")?;

    let output = completion.output_buffer.as_ref().ok_or_else(|| {
        ValidationError::new("MISSING_OUTPUT_BINDING", "successful completion must echo output binding")
    })?;
    let expected_generation =
        parse_decimal_u64(&pending.output.generation, false, "command.output.generation")?;
    let actual_generation =
        parse_decimal_u64(&output.generation, false, "completion.output_buffer.generation")?;
    let expected_offset =
        parse_decimal_u64(&pending.output.offset_bytes, true, "command.output.offset_bytes")?;
    let actual_offset =
        parse_decimal_u64(&output.offset_bytes, true, "completion.output_buffer.offset_bytes")?;
    let expected_length =
        parse_decimal_u64(&pending.output.length_bytes, false, "command.output.length_bytes")?;
    let actual_length =
        parse_decimal_u64(&output.length_bytes, false, "completion.output_buffer.length_bytes")?;

    if output.role != pending.output.role
        || output.handle_id != pending.output.handle_id
        || actual_generation != expected_generation
        || actual_offset != expected_offset
        || actual_length != expected_length
    {
        return Err(ValidationError::new(
            "COMPLETION_OUTPUT_BINDING_MISMATCH",
            "completion output handle, generation, offset, length or role differs from command",
        ));
    }

    if let Some(expected_profile) = capabilities.hardware_profile_sha256.as_deref() {
        let actual_profile = completion.device_profile_sha256.as_deref().ok_or_else(|| {
            ValidationError::new(
                "MISSING_COMPLETION_PROFILE",
                "physical completion must identify the admitted hardware profile",
            )
        })?;
        if expected_profile != actual_profile {
            return Err(ValidationError::new(
                "DEVICE_PROFILE_MISMATCH",
                "completion profile digest differs from admitted capability profile",
            ));
        }
    }

    match pending.operation_id.as_str() {
        "hdc.bind_xor.exact.v1" => {
            if result.winner_stable_id.is_some()
                || result.hamming_distance_bits.is_some()
                || result.result_record_bytes.is_some()
                || expected_length != VECTOR_BYTES
            {
                return Err(ValidationError::new(
                    "INVALID_XOR_RESULT",
                    "XOR completion must bind a 2,048-byte output and carry no scalar Top-1 fields",
                ));
            }
        }
        "hdc.hamming.exact.v1" => {
            let distance = result.hamming_distance_bits.ok_or_else(|| {
                ValidationError::new("MISSING_HAMMING_DISTANCE", "Hamming result requires exact distance")
            })?;
            if distance > DIMENSION_BITS
                || result.winner_stable_id.is_some()
                || result.result_record_bytes.is_some()
                || expected_length != 2
            {
                return Err(ValidationError::new(
                    "INVALID_HAMMING_RESULT",
                    "Hamming result must be a 16-bit exact distance in 0..=16,384",
                ));
            }
        }
        "hdc.top1.exact.v1" => {
            let stable_id = result.winner_stable_id.as_deref().ok_or_else(|| {
                ValidationError::new("MISSING_WINNER_ID", "Top-1 result requires a stable item ID")
            })?;
            parse_decimal_u64(stable_id, true, "winner_stable_id")?;
            let distance = result.hamming_distance_bits.ok_or_else(|| {
                ValidationError::new("MISSING_HAMMING_DISTANCE", "Top-1 result requires exact distance")
            })?;
            if distance > DIMENSION_BITS
                || result.result_record_bytes != Some(RESULT_RECORD_BYTES)
                || expected_length != RESULT_RECORD_BYTES
            {
                return Err(ValidationError::new(
                    "INVALID_TOP1_RESULT",
                    "Top-1 requires stable ID, distance in 0..=16,384 and a 16-byte result record",
                ));
            }
        }
        _ => {
            return Err(ValidationError::new(
                "UNKNOWN_PENDING_OPERATION",
                "pending operation is not admitted by protocol v0.1",
            ));
        }
    }
    Ok(())
}

fn collect_buffer_keys(buffers: &[BufferDescriptor]) -> ValidationResult<Vec<BufferKey>> {
    let mut keys = Vec::new();
    let mut unique = HashSet::new();
    for buffer in buffers {
        let generation = parse_decimal_u64(&buffer.generation, false, "buffer.generation")?;
        let key = BufferKey {
            handle_id: buffer.handle_id.clone(),
            generation,
        };
        if unique.insert(key.clone()) {
            keys.push(key);
        }
    }
    Ok(keys)
}

fn validate_epoch(value: &str) -> ValidationResult<()> {
    if value.len() != 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ValidationError::new(
            "INVALID_DEVICE_EPOCH",
            "device epoch must be exactly 32 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> ValidationResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ValidationError::new(
            "INVALID_SHA256",
            format!("{field} must be 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(())
}

/// Verify returned output bytes after DMA ownership has returned.
/// Checks command/completion identity, canonical semantics, output binding,
/// the device-reported SHA-256, operation-specific byte layout, and exact
/// byte-for-byte equality with an independent canonical oracle result.
pub fn verify_exact_output_bytes(
    command: &HdcCommand,
    completion: &HdcCompletion,
    actual_output: &[u8],
    oracle_output: &[u8],
) -> ValidationResult<VerifiedExactOutput> {
    validate_completion_envelope(completion)?;
    if completion.status != "completed" {
        return Err(ValidationError::new(
            "OUTPUT_NOT_COMPLETED",
            "only a completed command can enter exact output verification",
        ));
    }
    if command.schema != "luminous.hdc-device-command"
        || command.schema_version != PROTOCOL_VERSION
        || completion.command_id != command.command_id
        || completion.device_id != command.device_id
        || completion.device_epoch != command.device_epoch
        || completion.sequence != command.sequence
        || completion.operation_id != command.operation_id
        || completion.implementation_sha256 != command.implementation_sha256
        || completion.semantic_contract_sha256 != command.semantic_contract.contract_sha256
    {
        return Err(ValidationError::new(
            "OUTPUT_VERIFICATION_IDENTITY_MISMATCH",
            "completion identity does not match the original command",
        ));
    }
    if command.semantic_contract.contract_version != "1.0.0"
        || command.semantic_contract.contract_id != expected_contract_id(&command.operation_id)?
        || command.semantic_contract.contract_sha256
            != expected_contract_sha256(&command.operation_id)?
    {
        return Err(ValidationError::new(
            "OUTPUT_VERIFICATION_NONCANONICAL_SEMANTICS",
            "command does not bind the host-pinned semantic contract",
        ));
    }
    validate_sha256(&command.implementation_sha256, "command.implementation_sha256")?;

    let expected_role = output_role(&command.operation_id);
    let output_descriptor = command
        .buffers
        .iter()
        .find(|buffer| buffer.role == expected_role)
        .ok_or_else(|| ValidationError::new(
            "MISSING_COMMAND_OUTPUT",
            "command does not contain its operation-specific output buffer",
        ))?;
    if output_descriptor.access != "write" {
        return Err(ValidationError::new(
            "COMMAND_OUTPUT_NOT_WRITABLE",
            "operation output descriptor must declare write access",
        ));
    }
    let output_binding = completion.output_buffer.as_ref().ok_or_else(|| {
        ValidationError::new("MISSING_OUTPUT_BINDING", "completion has no output binding")
    })?;
    if output_binding.role != output_descriptor.role
        || output_binding.handle_id != output_descriptor.handle_id
        || parse_decimal_u64(&output_binding.generation, false, "completion.output.generation")?
            != parse_decimal_u64(&output_descriptor.generation, false, "command.output.generation")?
        || parse_decimal_u64(&output_binding.offset_bytes, true, "completion.output.offset")?
            != parse_decimal_u64(&output_descriptor.offset_bytes, true, "command.output.offset")?
        || parse_decimal_u64(&output_binding.length_bytes, false, "completion.output.length")?
            != parse_decimal_u64(&output_descriptor.length_bytes, false, "command.output.length")?
    {
        return Err(ValidationError::new(
            "OUTPUT_VERIFICATION_BUFFER_MISMATCH",
            "completion output binding differs from the submitted command",
        ));
    }

    let expected_len = match command.operation_id.as_str() {
        "hdc.bind_xor.exact.v1" => VECTOR_BYTES as usize,
        "hdc.hamming.exact.v1" => 2,
        "hdc.top1.exact.v1" => RESULT_RECORD_BYTES as usize,
        _ => return Err(ValidationError::new("UNKNOWN_OPERATION", "operation is not admitted")),
    };
    if parse_decimal_u64(&output_descriptor.length_bytes, false, "command.output.length")?
        != expected_len as u64
        || actual_output.len() != expected_len
        || oracle_output.len() != expected_len
    {
        return Err(ValidationError::new(
            "OUTPUT_LENGTH_MISMATCH",
            format!("command, actual and oracle output must each be exactly {expected_len} bytes"),
        ));
    }

    let result = completion.result.as_ref().ok_or_else(|| {
        ValidationError::new("MISSING_COMPLETION_RESULT", "completed result object is absent")
    })?;
    let actual_sha256 = format!("{:x}", Sha256::digest(actual_output));
    if actual_sha256 != result.output_sha256 {
        return Err(ValidationError::new(
            "OUTPUT_DIGEST_MISMATCH",
            "device-reported output SHA-256 differs from the actual returned bytes",
        ));
    }

    match command.operation_id.as_str() {
        "hdc.bind_xor.exact.v1" => {
            if result.winner_stable_id.is_some()
                || result.hamming_distance_bits.is_some()
                || result.result_record_bytes.is_some()
            {
                return Err(ValidationError::new(
                    "INVALID_XOR_RESULT",
                    "XOR completion must not include scalar Top-1/Hamming fields",
                ));
            }
        }
        "hdc.hamming.exact.v1" => {
            let distance = u16::from_le_bytes([actual_output[0], actual_output[1]]) as u64;
            if distance > DIMENSION_BITS
                || result.hamming_distance_bits != Some(distance)
                || result.winner_stable_id.is_some()
                || result.result_record_bytes.is_some()
            {
                return Err(ValidationError::new(
                    "INVALID_HAMMING_RESULT_BYTES",
                    "encoded Hamming output does not match the completion's exact distance",
                ));
            }
        }
        "hdc.top1.exact.v1" => {
            let winner_id = u64::from_le_bytes(actual_output[0..8].try_into().map_err(|_| {
                ValidationError::new("INVALID_TOP1_OUTPUT", "winner ID field is truncated")
            })?);
            let distance = u16::from_le_bytes([actual_output[8], actual_output[9]]) as u64;
            let reserved_is_zero = actual_output[10..16].iter().all(|byte| *byte == 0);
            let reported_id = result.winner_stable_id.as_deref()
                .ok_or_else(|| ValidationError::new("MISSING_WINNER_ID", "Top-1 result has no winner ID"))?;
            if parse_decimal_u64(reported_id, true, "winner_stable_id")? != winner_id
                || result.hamming_distance_bits != Some(distance)
                || distance > DIMENSION_BITS
                || result.result_record_bytes != Some(RESULT_RECORD_BYTES)
                || !reserved_is_zero
            {
                return Err(ValidationError::new(
                    "INVALID_TOP1_OUTPUT_BYTES",
                    "Top-1 bytes do not match result metadata or reserved bytes are non-zero",
                ));
            }
        }
        _ => return Err(ValidationError::new("UNKNOWN_OPERATION", "operation is not admitted")),
    }

    if actual_output != oracle_output {
        return Err(ValidationError::new(
            "OUTPUT_ORACLE_MISMATCH",
            "actual device output differs from the independent exact oracle bytes",
        ));
    }

    Ok(VerifiedExactOutput {
        command_id: command.command_id.clone(),
        device_id: command.device_id.clone(),
        device_epoch: command.device_epoch.clone(),
        sequence: parse_decimal_u64(&command.sequence, false, "command.sequence")?,
        operation_id: command.operation_id.clone(),
        output_handle_id: output_descriptor.handle_id.clone(),
        output_generation: parse_decimal_u64(
            &output_descriptor.generation,
            false,
            "command.output.generation",
        )?,
        output_sha256: actual_sha256,
        output_bytes: actual_output.len() as u64,
    })
}

/// Parse canonical decimal text without floating-point conversion. This
/// preserves all u64 values and rejects signs, whitespace and leading zeros.
pub fn parse_decimal_u64(
    value: &str,
    allow_zero: bool,
    field: &'static str,
) -> ValidationResult<u64> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (!allow_zero && value == "0")
    {
        return Err(ValidationError::new(
            "INVALID_DECIMAL_U64",
            format!("{field} must be canonical unsigned decimal text"),
        ));
    }
    value.parse::<u64>().map_err(|_| {
        ValidationError::new(
            "DECIMAL_U64_OVERFLOW",
            format!("{field} exceeds the u64 range"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use sha2::{Digest, Sha256};

    const SEMANTIC_CONTRACT_DOCUMENT: &str =
        include_str!("../../../../docs/research/COMPUTE_EXEC_001_HDC_DEVICE_CONTRACT_V0_1.md");

    fn descriptor_block(operation_line: &str) -> &'static str {
        let operation_position = SEMANTIC_CONTRACT_DOCUMENT
            .find(operation_line)
            .expect("operation must be present in the normative protocol document");
        let prefix = &SEMANTIC_CONTRACT_DOCUMENT[..operation_position];
        let fence = "```text\n";
        let block_start = prefix
            .rfind(fence)
            .expect("operation must be inside a fenced normative descriptor")
            + fence.len();
        let suffix = &SEMANTIC_CONTRACT_DOCUMENT[operation_position..];
        let closing_fence = suffix
            .find("\n```")
            .expect("normative descriptor must have a closing fence");
        let block_end = operation_position + closing_fence + 1; // include final LF
        &SEMANTIC_CONTRACT_DOCUMENT[block_start..block_end]
    }

    #[test]
    fn documented_semantic_descriptor_bytes_match_host_pinned_sha256() {
        let cases = [
            ("hdc.bind_xor.exact.v1", "operation=hdc.bind_xor.exact.v1"),
            ("hdc.hamming.exact.v1", "operation=hdc.hamming.exact.v1"),
            ("hdc.top1.exact.v1", "operation=hdc.top1.exact.v1"),
        ];
        for (operation_id, operation_line) in cases {
            let descriptor = descriptor_block(operation_line);
            let actual = format!("{:x}", Sha256::digest(descriptor.as_bytes()));
            let expected = expected_contract_sha256(operation_id).unwrap();
            assert_eq!(
                actual, expected,
                "protocol document descriptor bytes drifted for {operation_id}"
            );
        }
    }
    const CAPABILITIES: &str = include_str!(
        "../schemas/compute_exec_001/fixtures/simulator_capabilities_v0_1.json"
    );
    const COMMAND: &str = include_str!(
        "../schemas/compute_exec_001/fixtures/valid_hdc_top1_command_v0_1.json"
    );
    const COMPLETION: &str = include_str!(
        "../schemas/compute_exec_001/fixtures/valid_hdc_top1_completion_v0_1.json"
    );

    fn setup() -> (
        DeviceCapabilities,
        HdcCommand,
        HdcCompletion,
        BufferRegistry,
        ExecutionLedger,
    ) {
        let capabilities: DeviceCapabilities = serde_json::from_str(CAPABILITIES).unwrap();
        let command: HdcCommand = serde_json::from_str(COMMAND).unwrap();
        let completion: HdcCompletion = serde_json::from_str(COMPLETION).unwrap();

        let mut registry = BufferRegistry::default();
        registry
            .register("fixture-query", 1, 2_048, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-candidates", 1, 4_096, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-candidate-ids", 1, 16, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-result", 1, 16, BufferPermissions::WriteOnly)
            .unwrap();

        let ledger = ExecutionLedger::new(
            "sim-iverilog-hdc-tile",
            "00112233445566778899aabbccddeeff",
            4,
        )
        .unwrap();
        (capabilities, command, completion, registry, ledger)
    }

    #[test]
    fn simulator_fixture_submits_and_accepts_exact_output_binding() {
        let (capabilities, command, completion, mut registry, mut ledger) = setup();
        validate_command(&command, &capabilities, &registry).unwrap();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        assert_eq!(ledger.pending_count(), 1);
        assert_eq!(
            registry.ownership("fixture-result", 1),
            Some(&BufferOwnership::DeviceOwned {
                device_epoch: "00112233445566778899aabbccddeeff".into(),
                sequence: 1,
            })
        );
        assert_eq!(
            ledger
                .accept_completion(&completion, &capabilities, &mut registry)
                .unwrap(),
            CompletionDisposition::CompletedOutputAvailable
        );
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::OutputUnverified { .. })
        ));
        assert_eq!(registry.ownership("fixture-query", 1), Some(&BufferOwnership::HostOwned));
        assert_eq!(ledger.pending_count(), 0);

        let mut output = vec![0u8; RESULT_RECORD_BYTES as usize];
        let winner_id = completion.result.as_ref().unwrap().winner_stable_id
            .as_ref().unwrap().parse::<u64>().unwrap();
        let distance = completion.result.as_ref().unwrap().hamming_distance_bits.unwrap() as u16;
        output[0..8].copy_from_slice(&winner_id.to_le_bytes());
        output[8..10].copy_from_slice(&distance.to_le_bytes());
        completion.result.as_mut().unwrap().output_sha256 =
            format!("{:x}", Sha256::digest(&output));
        let proof = verify_exact_output_bytes(&command, &completion, &output, &output).unwrap();
        ledger.confirm_output_verified(&proof, &mut registry).unwrap();
        assert_eq!(registry.ownership("fixture-result", 1), Some(&BufferOwnership::HostOwned));
    }

    #[test]
    fn candidate_payload_multiplication_is_checked_against_registered_range() {
        let (capabilities, mut command, _, registry, _) = setup();
        command.parameters.candidate_count = Some(u32::MAX as u64);
        let error = validate_command(&command, &capabilities, &registry).unwrap_err();
        assert_eq!(error.code, "BUFFER_LENGTH_MISMATCH");
    }

    #[test]
    fn u64_decimal_parser_preserves_maximum_and_rejects_overflow() {
        assert_eq!(
            parse_decimal_u64("18446744073709551615", false, "sequence").unwrap(),
            u64::MAX
        );
        assert_eq!(
            parse_decimal_u64("18446744073709551616", false, "sequence")
                .unwrap_err()
                .code,
            "DECIMAL_U64_OVERFLOW"
        );
        assert!(parse_decimal_u64("01", false, "sequence").is_err());
    }

    #[test]
    fn checked_buffer_offset_plus_length_overflow_is_rejected() {
        let (capabilities, mut command, _, registry, _) = setup();
        let query = command
            .buffers
            .iter_mut()
            .find(|buffer| buffer.role == "query")
            .unwrap();
        // u64::MAX - 7 is correctly aligned to the fixture's 8-byte
        // requirement, so checked addition—not alignment—must reject it.
        query.offset_bytes = "18446744073709551608".into();
        let error = validate_command(&command, &capabilities, &registry).unwrap_err();
        assert_eq!(error.code, "BUFFER_RANGE_OVERFLOW");
    }

    #[test]
    fn registered_read_only_output_is_rejected_before_submission() {
        let (capabilities, command, _, _, mut ledger) = setup();
        let mut registry = BufferRegistry::default();
        registry
            .register("fixture-query", 1, 2_048, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-candidates", 1, 4_096, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-candidate-ids", 1, 16, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-result", 1, 16, BufferPermissions::ReadOnly)
            .unwrap();
        let error = ledger.submit(&command, &capabilities, &mut registry).unwrap_err();
        assert_eq!(error.code, "REGISTERED_PERMISSION_MISMATCH");
        assert_eq!(ledger.pending_count(), 0);
    }

    #[test]
    fn duplicate_sequence_and_command_id_are_rejected() {
        let (capabilities, command, _, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        let error = ledger.submit(&command, &capabilities, &mut registry).unwrap_err();
        assert_eq!(error.code, "NON_MONOTONIC_SEQUENCE");

        let mut next_sequence_same_id = command.clone();
        next_sequence_same_id.sequence = "2".into();
        let error = ledger
            .submit(&next_sequence_same_id, &capabilities, &mut registry)
            .unwrap_err();
        assert_eq!(error.code, "DUPLICATE_COMMAND_ID");
    }

    #[test]
    fn unknown_outcome_quarantines_buffers_until_explicit_quiescence() {
        let (capabilities, command, mut completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        completion.status = "outcome_unknown".into();
        completion.error_code = Some("OUTCOME_UNKNOWN".into());
        completion.result = None;
        completion.output_buffer = None;

        assert_eq!(
            ledger
                .accept_completion(&completion, &capabilities, &mut registry)
                .unwrap(),
            CompletionDisposition::Quarantined
        );
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::Quarantined { .. })
        ));
        assert_eq!(
            ledger.confirm_dma_quiescent("00112233445566778899aabbccddeeff", &mut registry),
            4
        );
        assert_eq!(registry.ownership("fixture-result", 1), Some(&BufferOwnership::HostOwned));
    }

    #[test]
    fn output_verification_rejects_digest_and_oracle_mismatches() {
        let (capabilities, command, mut completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        ledger.accept_completion(&completion, &capabilities, &mut registry).unwrap();

        let mut output = vec![0u8; RESULT_RECORD_BYTES as usize];
        output[0..8].copy_from_slice(&42u64.to_le_bytes());
        output[8..10].copy_from_slice(&7u16.to_le_bytes());
        completion.result.as_mut().unwrap().output_sha256 =
            format!("{:x}", Sha256::digest(&output));

        let mut wrong_digest = completion.clone();
        wrong_digest.result.as_mut().unwrap().output_sha256 = "a".repeat(64);
        assert_eq!(
            verify_exact_output_bytes(&command, &wrong_digest, &output, &output)
                .unwrap_err().code,
            "OUTPUT_DIGEST_MISMATCH"
        );

        let mut wrong_oracle = output.clone();
        wrong_oracle[0] ^= 1;
        assert_eq!(
            verify_exact_output_bytes(&command, &completion, &output, &wrong_oracle)
                .unwrap_err()
                .code,
            "OUTPUT_ORACLE_MISMATCH"
        );
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::OutputUnverified { .. })
        ));
    }

    #[test]
    fn output_verification_rejects_nonzero_top1_reserved_bytes() {
        let (capabilities, command, mut completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        ledger.accept_completion(&completion, &capabilities, &mut registry).unwrap();

        let mut output = vec![0u8; RESULT_RECORD_BYTES as usize];
        output[0..8].copy_from_slice(&42u64.to_le_bytes());
        output[8..10].copy_from_slice(&7u16.to_le_bytes());
        output[10] = 1;
        completion.result.as_mut().unwrap().output_sha256 =
            format!("{:x}", Sha256::digest(&output));
        assert_eq!(
            verify_exact_output_bytes(&command, &completion, &output, &output)
                .unwrap_err()
                .code,
            "INVALID_TOP1_OUTPUT_BYTES"
        );
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::OutputUnverified { .. })
        ));
    }

    #[test]
    fn reset_quarantines_outputs_that_have_not_passed_oracle_verification() {
        let (capabilities, command, completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        ledger.accept_completion(&completion, &capabilities, &mut registry).unwrap();
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::OutputUnverified { .. })
        ));

        ledger
            .reset("ffeeddccbbaa99887766554433221100", &mut registry)
            .unwrap();
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::Quarantined { .. })
        ));
        assert_eq!(
            ledger.confirm_dma_quiescent("00112233445566778899aabbccddeeff", &mut registry),
            4
        );
    }

    #[test]
    fn timeout_quarantines_buffers_and_blocks_same_epoch_submissions() {
        let (capabilities, command, _, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        ledger.mark_timeout(1, &mut registry).unwrap();

        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::Quarantined { .. })
        ));
        let mut next = command.clone();
        next.command_id = "fixture-top1-timeout-followup".into();
        next.sequence = "2".into();
        let error = ledger.submit(&next, &capabilities, &mut registry).unwrap_err();
        assert_eq!(error.code, "DEVICE_EPOCH_REQUIRES_RECONCILIATION");
    }

    #[test]
    fn unknown_outcome_blocks_the_device_epoch_until_reset() {
        let (mut capabilities, command, mut completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        completion.status = "outcome_unknown".into();
        completion.error_code = Some("OUTCOME_UNKNOWN".into());
        completion.result = None;
        completion.output_buffer = None;

        assert_eq!(
            ledger
                .accept_completion(&completion, &capabilities, &mut registry)
                .unwrap(),
            CompletionDisposition::Quarantined
        );

        let mut next = command.clone();
        next.command_id = "fixture-top1-0002".into();
        next.sequence = "2".into();
        let error = ledger.submit(&next, &capabilities, &mut registry).unwrap_err();
        assert_eq!(error.code, "DEVICE_EPOCH_REQUIRES_RECONCILIATION");

        ledger
            .reset("ffeeddccbbaa99887766554433221100", &mut registry)
            .unwrap();
        assert_eq!(ledger.current_epoch(), "ffeeddccbbaa99887766554433221100");
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::Quarantined { .. })
        ));

        let error = ledger.submit(&next, &capabilities, &mut registry).unwrap_err();
        assert_eq!(error.code, "DEVICE_EPOCH_REQUIRES_RECONCILIATION");

        assert_eq!(
            ledger.confirm_dma_quiescent("00112233445566778899aabbccddeeff", &mut registry),
            4
        );
        capabilities.device_epoch = "ffeeddccbbaa99887766554433221100".into();
        next.device_epoch = capabilities.device_epoch.clone();
        next.sequence = "1".into();
        ledger.submit(&next, &capabilities, &mut registry).unwrap();
    }

    #[test]
    fn reset_quarantines_inflight_buffers_and_rejects_late_old_epoch_completion() {
        let (capabilities, command, completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        ledger
            .reset("ffeeddccbbaa99887766554433221100", &mut registry)
            .unwrap();

        assert_eq!(ledger.pending_count(), 0);
        assert!(matches!(
            registry.ownership("fixture-query", 1),
            Some(BufferOwnership::Quarantined { .. })
        ));
        let error = ledger
            .accept_completion(&completion, &capabilities, &mut registry)
            .unwrap_err();
        assert_eq!(error.code, "STALE_OR_FOREIGN_COMPLETION");
    }

    #[test]
    fn reset_with_inflight_work_requires_old_epoch_quiescence_before_new_dispatch() {
        let (mut capabilities, mut command, _, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        ledger
            .reset("ffeeddccbbaa99887766554433221100", &mut registry)
            .unwrap();

        command.device_epoch = "ffeeddccbbaa99887766554433221100".into();
        command.sequence = "1".into();
        capabilities.device_epoch = command.device_epoch.clone();
        let error = ledger.submit(&command, &capabilities, &mut registry).unwrap_err();
        assert_eq!(error.code, "DEVICE_EPOCH_REQUIRES_RECONCILIATION");

        assert_eq!(
            ledger.confirm_dma_quiescent("00112233445566778899aabbccddeeff", &mut registry),
            4
        );
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
    }

    #[test]
    fn completion_must_echo_operation_and_implementation_identity() {
        let (capabilities, command, mut completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        completion.implementation_sha256 = "a".repeat(64);
        let error = ledger
            .accept_completion(&completion, &capabilities, &mut registry)
            .unwrap_err();
        assert_eq!(error.code, "COMPLETION_IMPLEMENTATION_MISMATCH");
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::DeviceOwned { .. })
        ));

        completion.implementation_sha256 = command.implementation_sha256.clone();
        completion.semantic_contract_sha256 = "b".repeat(64);
        let error = ledger
            .accept_completion(&completion, &capabilities, &mut registry)
            .unwrap_err();
        assert_eq!(error.code, "COMPLETION_SEMANTIC_CONTRACT_MISMATCH");

        completion.semantic_contract_sha256 =
            command.semantic_contract.contract_sha256.clone();
        completion.operation_id = "hdc.hamming.exact.v1".into();
        let error = ledger
            .accept_completion(&completion, &capabilities, &mut registry)
            .unwrap_err();
        assert_eq!(error.code, "COMPLETION_OPERATION_MISMATCH");
    }

    #[test]
    fn command_rejects_noncanonical_semantic_contract_digest() {
        let (capabilities, mut command, _, registry, _) = setup();
        command.semantic_contract.contract_sha256 = "a".repeat(64);
        let error = validate_command(&command, &capabilities, &registry).unwrap_err();
        assert_eq!(error.code, "NON_CANONICAL_SEMANTIC_DIGEST");
    }

    #[test]
    fn mismatched_completion_output_generation_never_releases_buffer() {
        let (capabilities, command, mut completion, mut registry, mut ledger) = setup();
        ledger.submit(&command, &capabilities, &mut registry).unwrap();
        completion.output_buffer.as_mut().unwrap().generation = "2".into();
        let error = ledger
            .accept_completion(&completion, &capabilities, &mut registry)
            .unwrap_err();
        assert_eq!(error.code, "COMPLETION_OUTPUT_BINDING_MISMATCH");
        assert!(matches!(
            registry.ownership("fixture-result", 1),
            Some(BufferOwnership::DeviceOwned { .. })
        ));
    }

    #[test]
    fn physical_capability_without_dma_isolation_is_rejected() {
        let (mut capabilities, _, _, _, _) = setup();
        capabilities.execution_class = "fpga_hardware".into();
        capabilities.hardware_profile_sha256 = Some("a".repeat(64));
        capabilities.firmware_sha256 = Some("b".repeat(64));
        capabilities.bitstream_sha256 = Some("c".repeat(64));
        capabilities.hardware_identity = Some(HardwareIdentity {
            manufacturer: "test".into(),
            board_model: "test".into(),
            board_revision: "A".into(),
            device_part: "test-part".into(),
            som_ordering_code: None,
        });
        capabilities.dma_isolation = "none".into();
        capabilities.cache_coherency = "coherent".into();
        assert_eq!(
            validate_capabilities(&capabilities).unwrap_err().code,
            "UNSAFE_DMA_ISOLATION"
        );
    }

    #[test]
    fn physical_asic_capability_uses_device_part_without_fpga_bitstream() {
        let (mut capabilities, _, _, _, _) = setup();
        capabilities.backend = "asic".into();
        capabilities.execution_class = "asic_hardware".into();
        capabilities.hardware_profile_sha256 = Some("a".repeat(64));
        capabilities.firmware_sha256 = Some("b".repeat(64));
        capabilities.bitstream_sha256 = None;
        capabilities.hardware_identity = Some(HardwareIdentity {
            manufacturer: "test-foundry".into(),
            board_model: "accelerator-module".into(),
            board_revision: "A".into(),
            device_part: "test-asic-rev1".into(),
            som_ordering_code: None,
        });
        capabilities.dma_isolation = "dedicated_bounded_pool".into();
        capabilities.cache_coherency = "explicit_flush_invalidate".into();

        assert!(validate_capabilities(&capabilities).is_ok());
    }

    #[test]
    fn output_alias_with_query_is_rejected() {
        let (capabilities, mut command, _, _, _) = setup();
        let output = command
            .buffers
            .iter_mut()
            .find(|buffer| buffer.role == "result")
            .unwrap();
        output.handle_id = "fixture-query".into();
        output.access = "write".into();

        let mut registry = BufferRegistry::default();
        registry
            .register("fixture-query", 1, 2_048, BufferPermissions::ReadWrite)
            .unwrap();
        registry
            .register("fixture-candidates", 1, 4_096, BufferPermissions::ReadOnly)
            .unwrap();
        registry
            .register("fixture-candidate-ids", 1, 16, BufferPermissions::ReadOnly)
            .unwrap();
        let error = validate_command(&command, &capabilities, &registry).unwrap_err();
        assert_eq!(error.code, "OUTPUT_INPUT_ALIAS");
    }
}

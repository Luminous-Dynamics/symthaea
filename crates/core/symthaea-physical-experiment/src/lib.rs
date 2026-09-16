// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen manifests and receipts for physical-computation experiments.
//!
//! This crate binds *what was intended to run* separately from *what was
//! observed*. It deliberately refuses to promote simulations/emulators into
//! physical-device evidence and keeps device-only energy distinct from
//! whole-system energy.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_physical_cognition::{
    BackendIdentity, EnergyBoundary, EvidenceLevel, ExecutionBoundary, ObservationProvenance,
};

const MANIFEST_DOMAIN: &[u8] = b"symthaea:physical-experiment:manifest:v1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea:physical-experiment:receipt:v1\0";

/// Fixed 32-byte digest used to bind workloads, manifests, and observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest32(pub [u8; 32]);

impl Digest32 {
    /// Hash arbitrary bytes with BLAKE3.
    pub fn blake3(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    /// Borrow the raw digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for Digest32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Frozen identity of the workload presented to a backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadIdentity {
    /// Canonical workload family, for example `temporal:narma10`.
    pub family: String,
    /// Workload implementation/schema version.
    pub version: String,
    /// Digest of the frozen fixture/config bytes.
    pub fixture_digest: Digest32,
}

impl WorkloadIdentity {
    /// Validate stable workload identity fields.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        validate_token("workload family", &self.family, true)?;
        validate_token("workload version", &self.version, false)?;
        Ok(())
    }
}

/// Explicit deterministic seeds used by an experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedPlan {
    /// Ordered seed list. Order is commitment-bound but receipt completeness is
    /// checked set-wise.
    pub seeds: Vec<u64>,
}

impl SeedPlan {
    /// Validate non-empty, bounded, duplicate-free seed selection.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        if self.seeds.is_empty() {
            return Err(ExperimentError::EmptySeeds);
        }
        if self.seeds.len() > 4096 {
            return Err(ExperimentError::TooManySeeds(self.seeds.len()));
        }
        let unique: BTreeSet<u64> = self.seeds.iter().copied().collect();
        if unique.len() != self.seeds.len() {
            return Err(ExperimentError::DuplicateSeed);
        }
        Ok(())
    }
}

/// Frozen temporal/sampling budget for each seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplingPlan {
    /// Frames executed but excluded from metrics.
    pub warmup_frames: u64,
    /// Frames required for the committed result.
    pub measured_frames: u64,
    /// Repeated observations/shots required per measured frame.
    pub samples_per_frame: u64,
}

impl SamplingPlan {
    /// Validate a non-zero measured/readout budget.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        if self.measured_frames == 0 {
            return Err(ExperimentError::ZeroMeasuredFrames);
        }
        if self.samples_per_frame == 0 {
            return Err(ExperimentError::ZeroSamplesPerFrame);
        }
        Ok(())
    }

    /// Expected aggregate readouts for one seed.
    pub fn expected_readouts(&self) -> Result<u64, ExperimentError> {
        self.measured_frames
            .checked_mul(self.samples_per_frame)
            .ok_or(ExperimentError::ReadoutOverflow)
    }
}

/// Required energy-accounting boundary for a committed experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyPolicy {
    /// Energy may remain unmeasured. If reported, its boundary must still be explicit.
    UnmeasuredAllowed,
    /// A device-level energy measurement is required.
    DeviceRequired,
    /// A whole-system energy measurement is required.
    WholeSystemRequired,
}

/// Exact experiment subject and protocol intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalExperimentManifest {
    /// Schema version. V1 is currently the only accepted value.
    pub schema_version: u16,
    /// Stable experiment identifier.
    pub experiment_id: String,
    /// Backend whose behavior is being measured.
    pub subject: BackendIdentity,
    /// Optional matched comparator/reference backend.
    pub comparator: Option<BackendIdentity>,
    /// Frozen workload identity.
    pub workload: WorkloadIdentity,
    /// Exact deterministic replicate seeds.
    pub seed_plan: SeedPlan,
    /// Exact warmup/measurement/readout budget.
    pub sampling: SamplingPlan,
    /// Execution boundary that must produce the receipt.
    pub expected_execution: ExecutionBoundary,
    /// Required energy accounting for this protocol.
    pub energy_policy: EnergyPolicy,
    /// Short commitment-bound protocol notes/caveats.
    pub notes: Vec<String>,
}

impl PhysicalExperimentManifest {
    /// Validate the manifest before commitment or execution.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        if self.schema_version != 1 {
            return Err(ExperimentError::UnsupportedSchema(self.schema_version));
        }
        validate_token("experiment id", &self.experiment_id, true)?;
        self.subject
            .validate()
            .map_err(|error| ExperimentError::Backend(error.to_string()))?;
        if let Some(comparator) = &self.comparator {
            comparator
                .validate()
                .map_err(|error| ExperimentError::Backend(error.to_string()))?;
            if comparator == &self.subject {
                return Err(ExperimentError::ComparatorEqualsSubject);
            }
        }
        self.workload.validate()?;
        self.seed_plan.validate()?;
        self.sampling.validate()?;
        if self.notes.len() > 64 {
            return Err(ExperimentError::TooManyNotes(self.notes.len()));
        }
        for note in &self.notes {
            if note.len() > 1024 {
                return Err(ExperimentError::NoteTooLong(note.len()));
            }
        }
        Ok(())
    }

    /// Deterministic V1 canonical bytes. This intentionally does not depend on
    /// JSON map ordering or serializer implementation details.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ExperimentError> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(MANIFEST_DOMAIN);
        push_u16(&mut out, self.schema_version);
        push_str(&mut out, &self.experiment_id)?;
        push_backend(&mut out, &self.subject)?;
        match &self.comparator {
            Some(comparator) => {
                out.push(1);
                push_backend(&mut out, comparator)?;
            }
            None => out.push(0),
        }
        push_str(&mut out, &self.workload.family)?;
        push_str(&mut out, &self.workload.version)?;
        out.extend_from_slice(self.workload.fixture_digest.as_bytes());
        push_len(&mut out, self.seed_plan.seeds.len())?;
        for seed in &self.seed_plan.seeds {
            push_u64(&mut out, *seed);
        }
        push_u64(&mut out, self.sampling.warmup_frames);
        push_u64(&mut out, self.sampling.measured_frames);
        push_u64(&mut out, self.sampling.samples_per_frame);
        out.push(execution_tag(self.expected_execution));
        out.push(energy_policy_tag(self.energy_policy));
        push_len(&mut out, self.notes.len())?;
        for note in &self.notes {
            push_str(&mut out, note)?;
        }
        Ok(out)
    }

    /// Domain-separated BLAKE3 commitment to the entire V1 manifest.
    pub fn commitment(&self) -> Result<Digest32, ExperimentError> {
        Ok(Digest32::blake3(&self.canonical_bytes()?))
    }
}

/// Receipt for one committed seed execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunReceipt {
    /// Receipt schema version.
    pub schema_version: u16,
    /// Manifest commitment this run claims to satisfy.
    pub manifest_commitment: Digest32,
    /// Backend that actually produced the observation.
    pub backend: BackendIdentity,
    /// Execution boundary actually used.
    pub execution: ExecutionBoundary,
    /// Evidence strength attached to the observation.
    pub evidence: EvidenceLevel,
    /// Seed executed by this receipt.
    pub seed: u64,
    /// Number of measured frames actually completed.
    pub completed_measured_frames: u64,
    /// Aggregate readouts/shots actually consumed.
    pub aggregate_readouts: u64,
    /// Energy accounting boundary actually supplied.
    pub energy_boundary: EnergyBoundary,
    /// Digest of the canonical/raw observation artifact retained elsewhere.
    pub observation_digest: Digest32,
    /// Receipt-local caveats. These do not change the manifest intent.
    pub caveats: Vec<String>,
}

impl RunReceipt {
    /// Validate this receipt against an exact manifest.
    pub fn validate_against(
        &self,
        manifest: &PhysicalExperimentManifest,
    ) -> Result<(), ExperimentError> {
        manifest.validate()?;
        if self.schema_version != 1 {
            return Err(ExperimentError::UnsupportedReceiptSchema(
                self.schema_version,
            ));
        }
        if self.manifest_commitment != manifest.commitment()? {
            return Err(ExperimentError::ManifestCommitmentMismatch);
        }
        if self.backend != manifest.subject {
            return Err(ExperimentError::BackendMismatch);
        }
        if self.execution != manifest.expected_execution {
            return Err(ExperimentError::ExecutionMismatch {
                expected: manifest.expected_execution,
                actual: self.execution,
            });
        }
        ObservationProvenance {
            backend: self.backend.clone(),
            execution: self.execution,
            evidence: self.evidence,
            caveats: self.caveats.clone(),
        }
        .validate()
        .map_err(|error| ExperimentError::Evidence(error.to_string()))?;
        if !manifest.seed_plan.seeds.contains(&self.seed) {
            return Err(ExperimentError::UnexpectedSeed(self.seed));
        }
        if self.completed_measured_frames != manifest.sampling.measured_frames {
            return Err(ExperimentError::MeasuredFrameMismatch {
                expected: manifest.sampling.measured_frames,
                actual: self.completed_measured_frames,
            });
        }
        let expected_readouts = manifest.sampling.expected_readouts()?;
        if self.aggregate_readouts != expected_readouts {
            return Err(ExperimentError::ReadoutMismatch {
                expected: expected_readouts,
                actual: self.aggregate_readouts,
            });
        }
        validate_energy_policy(manifest.energy_policy, self.energy_boundary)?;
        if self.caveats.len() > 64 {
            return Err(ExperimentError::TooManyNotes(self.caveats.len()));
        }
        Ok(())
    }

    /// Commitment to receipt metadata and the external observation digest.
    pub fn commitment(&self) -> Result<Digest32, ExperimentError> {
        if self.schema_version != 1 {
            return Err(ExperimentError::UnsupportedReceiptSchema(
                self.schema_version,
            ));
        }
        self.backend
            .validate()
            .map_err(|error| ExperimentError::Backend(error.to_string()))?;
        let mut out = Vec::new();
        out.extend_from_slice(RECEIPT_DOMAIN);
        push_u16(&mut out, self.schema_version);
        out.extend_from_slice(self.manifest_commitment.as_bytes());
        push_backend(&mut out, &self.backend)?;
        out.push(execution_tag(self.execution));
        out.push(evidence_tag(self.evidence));
        push_u64(&mut out, self.seed);
        push_u64(&mut out, self.completed_measured_frames);
        push_u64(&mut out, self.aggregate_readouts);
        out.push(energy_boundary_tag(self.energy_boundary));
        out.extend_from_slice(self.observation_digest.as_bytes());
        push_len(&mut out, self.caveats.len())?;
        for caveat in &self.caveats {
            push_str(&mut out, caveat)?;
        }
        Ok(Digest32::blake3(&out))
    }
}

/// Complete per-seed receipt set for a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptBundle {
    /// Exact manifest commitment shared by the bundle.
    pub manifest_commitment: Digest32,
    /// One receipt per committed seed.
    pub receipts: Vec<RunReceipt>,
}

impl ReceiptBundle {
    /// Require exactly one valid receipt for every committed seed and no others.
    pub fn validate_complete(
        &self,
        manifest: &PhysicalExperimentManifest,
    ) -> Result<(), ExperimentError> {
        let commitment = manifest.commitment()?;
        if self.manifest_commitment != commitment {
            return Err(ExperimentError::ManifestCommitmentMismatch);
        }
        if self.receipts.len() != manifest.seed_plan.seeds.len() {
            return Err(ExperimentError::ReceiptCountMismatch {
                expected: manifest.seed_plan.seeds.len(),
                actual: self.receipts.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for receipt in &self.receipts {
            receipt.validate_against(manifest)?;
            if !seen.insert(receipt.seed) {
                return Err(ExperimentError::DuplicateReceiptSeed(receipt.seed));
            }
        }
        let expected: BTreeSet<u64> = manifest.seed_plan.seeds.iter().copied().collect();
        if seen != expected {
            return Err(ExperimentError::IncompleteSeedCoverage);
        }
        Ok(())
    }
}

/// Validation failures for frozen experiment intent and receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    /// Unsupported manifest schema.
    UnsupportedSchema(u16),
    /// Unsupported receipt schema.
    UnsupportedReceiptSchema(u16),
    /// Invalid canonical token.
    InvalidToken { field: &'static str, value: String },
    /// Backend identity validation failed.
    Backend(String),
    /// Evidence boundary validation failed.
    Evidence(String),
    /// Comparator duplicates the subject backend.
    ComparatorEqualsSubject,
    /// Seed list is empty.
    EmptySeeds,
    /// Seed list is unreasonably large.
    TooManySeeds(usize),
    /// Seed list contains duplicates.
    DuplicateSeed,
    /// No measured frames were requested.
    ZeroMeasuredFrames,
    /// Samples/readouts per frame were zero.
    ZeroSamplesPerFrame,
    /// Readout multiplication overflowed.
    ReadoutOverflow,
    /// Too many notes/caveats were supplied.
    TooManyNotes(usize),
    /// A note exceeded its bounded size.
    NoteTooLong(usize),
    /// Collection/string length cannot be represented canonically.
    CanonicalLengthOverflow,
    /// Receipt does not bind the supplied manifest.
    ManifestCommitmentMismatch,
    /// Receipt backend differs from manifest subject.
    BackendMismatch,
    /// Receipt execution boundary differs from the preregistered one.
    ExecutionMismatch {
        /// Manifest boundary.
        expected: ExecutionBoundary,
        /// Receipt boundary.
        actual: ExecutionBoundary,
    },
    /// Receipt seed was not preregistered.
    UnexpectedSeed(u64),
    /// Measured frame count differs from the frozen protocol.
    MeasuredFrameMismatch { expected: u64, actual: u64 },
    /// Readout count differs from the frozen protocol.
    ReadoutMismatch { expected: u64, actual: u64 },
    /// Receipt energy boundary violates the frozen protocol.
    EnergyPolicyMismatch,
    /// Bundle contains the wrong number of receipts.
    ReceiptCountMismatch { expected: usize, actual: usize },
    /// Bundle contains two receipts for one seed.
    DuplicateReceiptSeed(u64),
    /// Bundle does not cover the exact committed seed set.
    IncompleteSeedCoverage,
}

impl std::fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ExperimentError {}

fn validate_energy_policy(
    policy: EnergyPolicy,
    actual: EnergyBoundary,
) -> Result<(), ExperimentError> {
    let valid = match policy {
        EnergyPolicy::UnmeasuredAllowed => true,
        EnergyPolicy::DeviceRequired => actual == EnergyBoundary::Device,
        EnergyPolicy::WholeSystemRequired => actual == EnergyBoundary::WholeSystem,
    };
    if valid {
        Ok(())
    } else {
        Err(ExperimentError::EnergyPolicyMismatch)
    }
}

fn validate_token(
    field: &'static str,
    value: &str,
    allow_colon: bool,
) -> Result<(), ExperimentError> {
    let valid = !value.is_empty()
        && value.len() <= 96
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.')
                || (allow_colon && byte == b':')
        });
    if valid {
        Ok(())
    } else {
        Err(ExperimentError::InvalidToken {
            field,
            value: value.to_owned(),
        })
    }
}

fn push_backend(out: &mut Vec<u8>, backend: &BackendIdentity) -> Result<(), ExperimentError> {
    backend
        .validate()
        .map_err(|error| ExperimentError::Backend(error.to_string()))?;
    push_str(out, &backend.family)?;
    push_str(out, &backend.version)?;
    push_str(out, &backend.implementation)?;
    Ok(())
}

fn push_str(out: &mut Vec<u8>, value: &str) -> Result<(), ExperimentError> {
    push_len(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_len(out: &mut Vec<u8>, value: usize) -> Result<(), ExperimentError> {
    let value = u32::try_from(value).map_err(|_| ExperimentError::CanonicalLengthOverflow)?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn execution_tag(value: ExecutionBoundary) -> u8 {
    match value {
        ExecutionBoundary::ClassicalReference => 0,
        ExecutionBoundary::Simulation => 1,
        ExecutionBoundary::Emulator => 2,
        ExecutionBoundary::PhysicalDevice => 3,
    }
}

fn evidence_tag(value: EvidenceLevel) -> u8 {
    match value {
        EvidenceLevel::Theoretical => 0,
        EvidenceLevel::Simulated => 1,
        EvidenceLevel::Emulated => 2,
        EvidenceLevel::Measured => 3,
        EvidenceLevel::Replicated => 4,
    }
}

fn energy_boundary_tag(value: EnergyBoundary) -> u8 {
    match value {
        EnergyBoundary::Unmeasured => 0,
        EnergyBoundary::Device => 1,
        EnergyBoundary::WholeSystem => 2,
    }
}

fn energy_policy_tag(value: EnergyPolicy) -> u8 {
    match value {
        EnergyPolicy::UnmeasuredAllowed => 0,
        EnergyPolicy::DeviceRequired => 1,
        EnergyPolicy::WholeSystemRequired => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> PhysicalExperimentManifest {
        PhysicalExperimentManifest {
            schema_version: 1,
            experiment_id: "aurum:narma10:v1".to_string(),
            subject: BackendIdentity::new(
                "physical:gold-nanojunction",
                "v1",
                "aurum-effective-network-sim",
            )
            .unwrap(),
            comparator: Some(
                BackendIdentity::new("classical:hysteretic", "v1", "matched-control").unwrap(),
            ),
            workload: WorkloadIdentity {
                family: "temporal:narma10".to_string(),
                version: "v1".to_string(),
                fixture_digest: Digest32::blake3(b"fixture"),
            },
            seed_plan: SeedPlan {
                seeds: vec![7, 11],
            },
            sampling: SamplingPlan {
                warmup_frames: 100,
                measured_frames: 500,
                samples_per_frame: 1,
            },
            expected_execution: ExecutionBoundary::Simulation,
            energy_policy: EnergyPolicy::UnmeasuredAllowed,
            notes: vec!["simulation-only".to_string()],
        }
    }

    fn valid_receipt(manifest: &PhysicalExperimentManifest, seed: u64) -> RunReceipt {
        RunReceipt {
            schema_version: 1,
            manifest_commitment: manifest.commitment().unwrap(),
            backend: manifest.subject.clone(),
            execution: ExecutionBoundary::Simulation,
            evidence: EvidenceLevel::Simulated,
            seed,
            completed_measured_frames: manifest.sampling.measured_frames,
            aggregate_readouts: manifest.sampling.expected_readouts().unwrap(),
            energy_boundary: EnergyBoundary::Unmeasured,
            observation_digest: Digest32::blake3(&seed.to_le_bytes()),
            caveats: vec!["not a hardware observation".to_string()],
        }
    }

    #[test]
    fn manifest_commitment_changes_when_seed_plan_changes() {
        let a = manifest();
        let mut b = a.clone();
        b.seed_plan.seeds[1] = 13;
        assert_ne!(a.commitment().unwrap(), b.commitment().unwrap());
    }

    #[test]
    fn simulation_receipt_cannot_claim_measured_evidence() {
        let manifest = manifest();
        let mut receipt = valid_receipt(&manifest, 7);
        receipt.evidence = EvidenceLevel::Measured;
        assert!(matches!(
            receipt.validate_against(&manifest),
            Err(ExperimentError::Evidence(_))
        ));
    }

    #[test]
    fn changed_protocol_invalidates_receipt() {
        let manifest = manifest();
        let receipt = valid_receipt(&manifest, 7);
        let mut changed = manifest.clone();
        changed.sampling.measured_frames += 1;
        assert_eq!(
            receipt.validate_against(&changed),
            Err(ExperimentError::ManifestCommitmentMismatch)
        );
    }

    #[test]
    fn complete_bundle_requires_exact_seed_coverage() {
        let manifest = manifest();
        let one = valid_receipt(&manifest, 7);
        let incomplete = ReceiptBundle {
            manifest_commitment: manifest.commitment().unwrap(),
            receipts: vec![one],
        };
        assert!(matches!(
            incomplete.validate_complete(&manifest),
            Err(ExperimentError::ReceiptCountMismatch { .. })
        ));

        let complete = ReceiptBundle {
            manifest_commitment: manifest.commitment().unwrap(),
            receipts: vec![valid_receipt(&manifest, 7), valid_receipt(&manifest, 11)],
        };
        complete.validate_complete(&manifest).unwrap();
    }

    #[test]
    fn whole_system_energy_policy_rejects_device_only_receipt() {
        let mut manifest = manifest();
        manifest.energy_policy = EnergyPolicy::WholeSystemRequired;
        let mut receipt = valid_receipt(&manifest, 7);
        receipt.manifest_commitment = manifest.commitment().unwrap();
        receipt.energy_boundary = EnergyBoundary::Device;
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(ExperimentError::EnergyPolicyMismatch)
        );
    }

    #[test]
    fn receipt_commitment_binds_observation_digest() {
        let manifest = manifest();
        let a = valid_receipt(&manifest, 7);
        let mut b = a.clone();
        b.observation_digest = Digest32::blake3(b"different");
        assert_ne!(a.commitment().unwrap(), b.commitment().unwrap());
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only distributed currentness for deterministic finalized fabrication upgrades.
//!
//! A local `ClockGovernedFinalizedUpgradeV1` is terminal authority only for its exact qualified
//! transition. This bridge proves that exact finalization was published after execution into a
//! strict append-only descendant log and is the only non-conflicting finalization for the handoff in
//! one exact authenticated registry/containment checkpoint view.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_upgrade_finalized_state::{
    ClockGovernedFinalizedUpgradeIdV1, ClockGovernedFinalizedUpgradeV1,
};
use symthaea_fabrication_witness_registry_containment_bound::{
    ContainmentCurrentWitnessRegistryHeadIdV1, ContainmentCurrentWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const FINALIZED_UPGRADE_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.finalized-upgrade-head-publication.v1";
pub const CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA: &str =
    "symthaea.fabrication.current-finalized-upgrade-head.v1";
pub const FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX: &str = "finalized-upgrade-head-v1:";
pub const MAX_FINALIZED_UPGRADE_HEAD_CLOCK_HOPS: usize = 4096;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.finalized-upgrade-head-publication.v1\0";
const LOG_KIND_DOMAIN: &[u8] = b"symthaea.fabrication.finalized-upgrade-head-kind.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.finalized-upgrade-head-clock-lineage.v1\0";
const CURRENT_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.current-finalized-upgrade-head.v1\0";

/// Portable publication bytes. Live currentness comes only from the opaque head capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalizedUpgradeHeadPublicationV1 {
    pub schema_version: String,
    pub finalized_upgrade_id: String,
    pub record_digest: Sha256Digest,
    pub finalization_sequence: u64,
    pub handoff_id: String,
    pub handoff_plan_digest: Sha256Digest,
    pub predecessor_upgrade_state_digest: Sha256Digest,
    pub predecessor_upgrade_state_generation: u64,
    pub finalized_upgrade_state_generation: u64,
    pub successor_source_tree_digest: Sha256Digest,
    pub successor_executable_digest: Sha256Digest,
    pub successor_durable_state_digest: Sha256Digest,
    pub successor_replay_contract_digest: Sha256Digest,
    pub authorization_id: String,
    pub execution_permit_id: String,
    pub state_binding_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentFinalizedUpgradeHeadIdV1(Sha256Digest);

impl CurrentFinalizedUpgradeHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact deterministic finalization is the non-conflicting finalized head for
/// its handoff inside one exact authenticated append-only checkpoint view.
#[derive(Debug, Clone)]
#[must_use]
pub struct CurrentFinalizedUpgradeHeadV1 {
    id: CurrentFinalizedUpgradeHeadIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    finalized_upgrade_id: ClockGovernedFinalizedUpgradeIdV1,
    record_digest: Sha256Digest,
    handoff_plan_digest: Sha256Digest,
    finalization_sequence: u64,
    predecessor_upgrade_state_digest: Sha256Digest,
    predecessor_upgrade_state_generation: u64,
    finalized_upgrade_state_generation: u64,
    successor_durable_state_digest: Sha256Digest,
    publication_digest: Sha256Digest,
    publication_count: usize,
    latest_publication_entry_sequence: u64,
    execution_transparency_log_digest: Sha256Digest,
    execution_log_size: usize,
    current_transparency_log_digest: Sha256Digest,
    current_log_size: usize,
    appended_entry_count: usize,
    execution_checkpoint_digest: Sha256Digest,
    current_checkpoint_digest: Sha256Digest,
    execution_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    execution_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
}

impl CurrentFinalizedUpgradeHeadV1 {
    pub fn id(&self) -> CurrentFinalizedUpgradeHeadIdV1 {
        self.id
    }

    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 {
        self.governance_view_id
    }

    pub fn finalized_upgrade_id(&self) -> ClockGovernedFinalizedUpgradeIdV1 {
        self.finalized_upgrade_id
    }

    pub fn record_digest(&self) -> Sha256Digest {
        self.record_digest
    }

    pub fn handoff_plan_digest(&self) -> Sha256Digest {
        self.handoff_plan_digest
    }

    pub fn finalization_sequence(&self) -> u64 {
        self.finalization_sequence
    }

    pub fn predecessor_upgrade_state_digest(&self) -> Sha256Digest {
        self.predecessor_upgrade_state_digest
    }

    pub fn predecessor_upgrade_state_generation(&self) -> u64 {
        self.predecessor_upgrade_state_generation
    }

    pub fn finalized_upgrade_state_generation(&self) -> u64 {
        self.finalized_upgrade_state_generation
    }

    pub fn successor_durable_state_digest(&self) -> Sha256Digest {
        self.successor_durable_state_digest
    }

    pub fn publication_digest(&self) -> Sha256Digest {
        self.publication_digest
    }

    pub fn publication_count(&self) -> usize {
        self.publication_count
    }

    pub fn latest_publication_entry_sequence(&self) -> u64 {
        self.latest_publication_entry_sequence
    }

    pub fn current_transparency_log_digest(&self) -> Sha256Digest {
        self.current_transparency_log_digest
    }

    pub fn current_checkpoint_digest(&self) -> Sha256Digest {
        self.current_checkpoint_digest
    }

    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.current_clock_envelope_id
    }

    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.current_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalizedUpgradeHeadError {
    InvalidPublication,
    PublicationMismatch,
    ExecutionLogInvalid(String),
    CurrentLogInvalid(String),
    ExecutionLogMismatch,
    CurrentLogMismatch,
    LogNotStrictExtension,
    GovernanceObservationBasisMismatch,
    GovernanceObservationEnvelopeMismatch,
    ExecutionBasisMismatch,
    ExecutionEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    PublicationClockNotDefinitelyLater {
        execution_upper_unix_ms: u64,
        publication_lower_unix_ms: u64,
    },
    CheckpointNotAdvanced,
    PublicationNotFound,
    ConflictingFinalizationPublished { entry_sequence: u64, subject_digest: Sha256Digest },
    PublicationBeforeExecution { entry_sequence: u64 },
    PublicationMayBeFuture { entry_sequence: u64 },
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment {
    execution_basis_id: String,
    bridge_basis_ids: Vec<String>,
    observation_basis_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct CurrentHeadCommitment {
    schema: &'static str,
    governance_view_id: String,
    finalized_upgrade_id: String,
    record_digest: String,
    handoff_plan_digest: String,
    finalization_sequence: u64,
    predecessor_upgrade_state_digest: String,
    predecessor_upgrade_state_generation: u64,
    finalized_upgrade_state_generation: u64,
    successor_durable_state_digest: String,
    publication_digest: String,
    publication_count: usize,
    latest_publication_entry_sequence: u64,
    execution_transparency_log_digest: String,
    execution_log_size: usize,
    current_transparency_log_digest: String,
    current_log_size: usize,
    appended_entry_count: usize,
    execution_checkpoint_digest: String,
    current_checkpoint_digest: String,
    execution_clock_envelope_id: String,
    current_clock_envelope_id: String,
    execution_operational_basis_id: String,
    current_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
}

pub fn build_finalized_upgrade_head_publication_v1(
    finalized: &ClockGovernedFinalizedUpgradeV1,
) -> FinalizedUpgradeHeadPublicationV1 {
    let record = finalized.record();
    FinalizedUpgradeHeadPublicationV1 {
        schema_version: FINALIZED_UPGRADE_HEAD_PUBLICATION_SCHEMA.into(),
        finalized_upgrade_id: finalized.id().to_hex(),
        record_digest: finalized.record_digest(),
        finalization_sequence: record.finalization_sequence,
        handoff_id: record.handoff_id.clone(),
        handoff_plan_digest: record.handoff_plan_digest,
        predecessor_upgrade_state_digest: record.predecessor_upgrade_state_digest,
        predecessor_upgrade_state_generation: record.predecessor_upgrade_state_generation,
        finalized_upgrade_state_generation: record.finalized_upgrade_state_generation,
        successor_source_tree_digest: record.successor_source_tree_digest,
        successor_executable_digest: record.successor_executable_digest,
        successor_durable_state_digest: record.successor_durable_state_digest,
        successor_replay_contract_digest: record.successor_replay_contract_digest,
        authorization_id: record.authorization_id.clone(),
        execution_permit_id: record.execution_permit_id.clone(),
        state_binding_id: record.state_binding_id.clone(),
    }
}

pub fn digest_finalized_upgrade_head_publication_v1(
    publication: &FinalizedUpgradeHeadPublicationV1,
) -> Result<Sha256Digest, FinalizedUpgradeHeadError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn finalized_upgrade_head_log_kind(
    handoff_plan_digest: Sha256Digest,
) -> Result<String, FinalizedUpgradeHeadError> {
    if handoff_plan_digest.0 == [0; 32] {
        return Err(FinalizedUpgradeHeadError::InvalidPublication);
    }
    let mut hasher = Sha256::new();
    hasher.update(LOG_KIND_DOMAIN);
    hasher.update(&handoff_plan_digest.0);
    Ok(format!(
        "{FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX}{}",
        hasher.finalize().to_hex()
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_current_finalized_upgrade_head_v1(
    finalized: &ClockGovernedFinalizedUpgradeV1,
    publication: &FinalizedUpgradeHeadPublicationV1,
    execution_log: &TransparencyLog,
    current_log: &TransparencyLog,
    execution_basis: &OperationalClockBasisV1,
    execution_to_observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
) -> Result<CurrentFinalizedUpgradeHeadV1, Vec<FinalizedUpgradeHeadError>> {
    let mut violations = Vec::new();
    let record = finalized.record();
    let expected_publication = build_finalized_upgrade_head_publication_v1(finalized);
    if publication != &expected_publication {
        violations.push(FinalizedUpgradeHeadError::PublicationMismatch);
    }
    let publication_digest = match digest_finalized_upgrade_head_publication_v1(publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };

    if let Err(error) = execution_log.validate() {
        violations.push(FinalizedUpgradeHeadError::ExecutionLogInvalid(format!(
            "{error:?}"
        )));
    }
    let execution_log_digest = match digest_transparency_log(execution_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(FinalizedUpgradeHeadError::ExecutionLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if execution_log_digest != record.fresh_transparency_log_digest {
        violations.push(FinalizedUpgradeHeadError::ExecutionLogMismatch);
    }

    if let Err(error) = current_log.validate() {
        violations.push(FinalizedUpgradeHeadError::CurrentLogInvalid(format!("{error:?}")));
    }
    let current_log_digest = match digest_transparency_log(current_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(FinalizedUpgradeHeadError::CurrentLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if current_log_digest != governance_view.transparency_log_digest() {
        violations.push(FinalizedUpgradeHeadError::CurrentLogMismatch);
    }
    if current_log.entries.len() <= execution_log.entries.len()
        || current_log.verify_successor_of(execution_log).is_err()
    {
        violations.push(FinalizedUpgradeHeadError::LogNotStrictExtension);
    }
    if governance_view.checkpoint_digest() == record.fresh_checkpoint_digest {
        violations.push(FinalizedUpgradeHeadError::CheckpointNotAdvanced);
    }

    if execution_basis.id().to_hex() != record.fresh_operational_basis_id {
        violations.push(FinalizedUpgradeHeadError::ExecutionBasisMismatch);
    }
    let execution_clock = match derive_clock_governance_evaluation_envelope_v1(execution_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(FinalizedUpgradeHeadError::Clock(error));
            return Err(violations);
        }
    };
    if execution_clock.id().to_hex() != record.fresh_clock_envelope_id {
        violations.push(FinalizedUpgradeHeadError::ExecutionEnvelopeMismatch);
    }

    if observation_basis.id() != governance_view.observation_operational_basis_id() {
        violations.push(FinalizedUpgradeHeadError::GovernanceObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(FinalizedUpgradeHeadError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != governance_view.observation_clock_envelope_id() {
        violations.push(FinalizedUpgradeHeadError::GovernanceObservationEnvelopeMismatch);
    }

    if execution_to_observation_clock_bridge.len() > MAX_FINALIZED_UPGRADE_HEAD_CLOCK_HOPS {
        violations.push(FinalizedUpgradeHeadError::TooManyClockHops {
            actual: execution_to_observation_clock_bridge.len(),
            maximum: MAX_FINALIZED_UPGRADE_HEAD_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        execution_basis.id(),
        execution_to_observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }
    if observation_clock.lower_unix_ms() <= execution_clock.upper_unix_ms() {
        violations.push(FinalizedUpgradeHeadError::PublicationClockNotDefinitelyLater {
            execution_upper_unix_ms: execution_clock.upper_unix_ms(),
            publication_lower_unix_ms: observation_clock.lower_unix_ms(),
        });
    }

    let log_kind = match finalized_upgrade_head_log_kind(record.handoff_plan_digest) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    let matching_entries = current_log
        .entries
        .iter()
        .filter(|entry| entry.kind == log_kind)
        .collect::<Vec<_>>();
    if matching_entries.is_empty() {
        violations.push(FinalizedUpgradeHeadError::PublicationNotFound);
        return Err(violations);
    }

    for entry in &matching_entries {
        if entry.subject_digest != publication_digest {
            violations.push(FinalizedUpgradeHeadError::ConflictingFinalizationPublished {
                entry_sequence: entry.sequence,
                subject_digest: entry.subject_digest,
            });
            continue;
        }
        let recorded_at_unix_ms = match seconds_to_millis(entry.recorded_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if recorded_at_unix_ms < execution_clock.upper_unix_ms() {
            violations.push(FinalizedUpgradeHeadError::PublicationBeforeExecution {
                entry_sequence: entry.sequence,
            });
        }
        if recorded_at_unix_ms > observation_clock.lower_unix_ms() {
            violations.push(FinalizedUpgradeHeadError::PublicationMayBeFuture {
                entry_sequence: entry.sequence,
            });
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let latest_entry = matching_entries
        .last()
        .copied()
        .ok_or_else(|| vec![FinalizedUpgradeHeadError::PublicationNotFound])?;
    let appended_entry_count = current_log
        .entries
        .len()
        .checked_sub(execution_log.entries.len())
        .ok_or_else(|| vec![FinalizedUpgradeHeadError::LogNotStrictExtension])?;
    let clock_lineage_digest = hash_serializable(
        CLOCK_LINEAGE_DOMAIN,
        &ClockLineageCommitment {
            execution_basis_id: execution_basis.id().to_hex(),
            bridge_basis_ids: execution_to_observation_clock_bridge
                .iter()
                .map(|basis| basis.id().to_hex())
                .collect(),
            observation_basis_id: observation_basis.id().to_hex(),
        },
    )
    .map_err(|error| vec![error])?;

    let commitment = CurrentHeadCommitment {
        schema: CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA,
        governance_view_id: governance_view.id().to_hex(),
        finalized_upgrade_id: finalized.id().to_hex(),
        record_digest: finalized.record_digest().to_hex(),
        handoff_plan_digest: record.handoff_plan_digest.to_hex(),
        finalization_sequence: record.finalization_sequence,
        predecessor_upgrade_state_digest: record.predecessor_upgrade_state_digest.to_hex(),
        predecessor_upgrade_state_generation: record.predecessor_upgrade_state_generation,
        finalized_upgrade_state_generation: record.finalized_upgrade_state_generation,
        successor_durable_state_digest: record.successor_durable_state_digest.to_hex(),
        publication_digest: publication_digest.to_hex(),
        publication_count: matching_entries.len(),
        latest_publication_entry_sequence: latest_entry.sequence,
        execution_transparency_log_digest: execution_log_digest.to_hex(),
        execution_log_size: execution_log.entries.len(),
        current_transparency_log_digest: current_log_digest.to_hex(),
        current_log_size: current_log.entries.len(),
        appended_entry_count,
        execution_checkpoint_digest: record.fresh_checkpoint_digest.to_hex(),
        current_checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        execution_clock_envelope_id: execution_clock.id().to_hex(),
        current_clock_envelope_id: observation_clock.id().to_hex(),
        execution_operational_basis_id: execution_basis.id().to_hex(),
        current_operational_basis_id: observation_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count: execution_to_observation_clock_bridge.len(),
    };
    let id = CurrentFinalizedUpgradeHeadIdV1(
        hash_serializable(CURRENT_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(CurrentFinalizedUpgradeHeadV1 {
        id,
        governance_view_id: governance_view.id(),
        finalized_upgrade_id: finalized.id(),
        record_digest: finalized.record_digest(),
        handoff_plan_digest: record.handoff_plan_digest,
        finalization_sequence: record.finalization_sequence,
        predecessor_upgrade_state_digest: record.predecessor_upgrade_state_digest,
        predecessor_upgrade_state_generation: record.predecessor_upgrade_state_generation,
        finalized_upgrade_state_generation: record.finalized_upgrade_state_generation,
        successor_durable_state_digest: record.successor_durable_state_digest,
        publication_digest,
        publication_count: matching_entries.len(),
        latest_publication_entry_sequence: latest_entry.sequence,
        execution_transparency_log_digest: execution_log_digest,
        execution_log_size: execution_log.entries.len(),
        current_transparency_log_digest: current_log_digest,
        current_log_size: current_log.entries.len(),
        appended_entry_count,
        execution_checkpoint_digest: record.fresh_checkpoint_digest,
        current_checkpoint_digest: governance_view.checkpoint_digest(),
        execution_clock_envelope_id: execution_clock.id(),
        current_clock_envelope_id: observation_clock.id(),
        execution_operational_basis_id: execution_basis.id(),
        current_operational_basis_id: observation_basis.id(),
        clock_lineage_digest,
        clock_hop_count: execution_to_observation_clock_bridge.len(),
    })
}

fn validate_publication(
    publication: &FinalizedUpgradeHeadPublicationV1,
) -> Result<(), FinalizedUpgradeHeadError> {
    if publication.schema_version != FINALIZED_UPGRADE_HEAD_PUBLICATION_SCHEMA
        || publication.finalization_sequence == 0
        || publication.finalized_upgrade_id.trim().is_empty()
        || publication.handoff_id.trim().is_empty()
        || publication.authorization_id.trim().is_empty()
        || publication.execution_permit_id.trim().is_empty()
        || publication.state_binding_id.trim().is_empty()
        || publication.record_digest.0 == [0; 32]
        || publication.handoff_plan_digest.0 == [0; 32]
        || publication.predecessor_upgrade_state_digest.0 == [0; 32]
        || publication.predecessor_upgrade_state_generation == 0
        || publication.finalized_upgrade_state_generation
            != publication
                .predecessor_upgrade_state_generation
                .checked_add(1)
                .ok_or(FinalizedUpgradeHeadError::InvalidPublication)?
    {
        return Err(FinalizedUpgradeHeadError::InvalidPublication);
    }
    for digest in [
        publication.successor_source_tree_digest,
        publication.successor_executable_digest,
        publication.successor_durable_state_digest,
        publication.successor_replay_contract_digest,
    ] {
        if digest.0 == [0; 32] {
            return Err(FinalizedUpgradeHeadError::InvalidPublication);
        }
    }
    Ok(())
}

fn verify_clock_lineage(
    execution_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
) -> Result<(), FinalizedUpgradeHeadError> {
    let mut expected = execution_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(FinalizedUpgradeHeadError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = observation_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(FinalizedUpgradeHeadError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn seconds_to_millis(seconds: u64) -> Result<u64, FinalizedUpgradeHeadError> {
    seconds
        .checked_mul(1_000)
        .ok_or(FinalizedUpgradeHeadError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, FinalizedUpgradeHeadError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| FinalizedUpgradeHeadError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

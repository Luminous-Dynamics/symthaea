// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-checkpoint, full-lineage negative rollback authority for hardened fabrication upgrades.
//!
//! A single valid operational state is not enough to prove rollback absence: an older state may
//! omit a rollback that a later state durably records. This bridge consumes the exact authenticated
//! registry/containment checkpoint view and requires the complete operational publication chain for
//! one handoff, from generation 1 through the latest publication in that same log. Every state is
//! hash-linked with the kernel successor theorem and every publication is rebuilt from its state.
//! Because rollback digests are monotonic/non-removable in that lineage, a complete rollback-free
//! chain provides a substantially stronger negative capability than caller omission or a stale head.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_evidence_retention_head::{
    CurrentEvidenceRetentionHeadIdV1, CurrentEvidenceRetentionHeadV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_kernel::upgrade_operational_state::{
    FabricationUpgradeOperationalState, digest_upgrade_operational_state,
    verify_upgrade_operational_state_successor,
};
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_operational_head::{
    UpgradeOperationalHeadPublicationV1, build_upgrade_operational_head_publication_v1,
    digest_upgrade_operational_head_publication_v1, upgrade_operational_head_log_kind,
};
use symthaea_fabrication_upgrade_runtime::{
    ClockGovernedUpgradeActivationPermitIdV1, ClockGovernedUpgradeActivationPermitV1,
};
use symthaea_fabrication_witness_registry_containment_bound::{
    ContainmentCurrentWitnessRegistryHeadIdV1, ContainmentCurrentWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CURRENT_NO_ROLLBACK_UPGRADE_AUTHORITY_SCHEMA: &str =
    "symthaea.fabrication.current-no-rollback-upgrade-authority.v1";
pub const MAX_CURRENT_NO_ROLLBACK_STATES: usize = 65_536;
pub const MAX_CURRENT_NO_ROLLBACK_CLOCK_HOPS: usize = 4096;

const OPERATIONAL_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.current-no-rollback-operational-lineage.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.current-no-rollback-clock-lineage.v1\0";
const CURRENT_NO_ROLLBACK_DOMAIN: &[u8] =
    b"symthaea.fabrication.current-no-rollback-upgrade-authority.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentNoRollbackUpgradeAuthorityIdV1(Sha256Digest);

impl CurrentNoRollbackUpgradeAuthorityIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that the complete operational chain for one exact handoff is rollback-free and is
/// the latest chain published inside the exact governance checkpoint view consumed here.
#[derive(Debug, Clone)]
#[must_use]
pub struct CurrentNoRollbackUpgradeAuthorityV1 {
    id: CurrentNoRollbackUpgradeAuthorityIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    governance_checkpoint_digest: Sha256Digest,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    handoff_plan_digest: Sha256Digest,
    activation_permit_id: ClockGovernedUpgradeActivationPermitIdV1,
    state_digest: Sha256Digest,
    state_generation: u64,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    operational_lineage_digest: Sha256Digest,
    operational_state_count: usize,
    probation_clearance_digest: Option<Sha256Digest>,
    probation_sequence: Option<u64>,
    reauthorized_machine_count: u64,
    retention_policy_digest: Sha256Digest,
    retention_policy_sequence: u64,
    key_snapshot_sequence: u64,
    clock_epoch: u64,
    activation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
}

impl CurrentNoRollbackUpgradeAuthorityV1 {
    pub fn id(&self) -> CurrentNoRollbackUpgradeAuthorityIdV1 {
        self.id
    }

    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 {
        self.governance_view_id
    }

    pub fn retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 {
        self.retention_head_id
    }

    pub fn governance_checkpoint_digest(&self) -> Sha256Digest {
        self.governance_checkpoint_digest
    }

    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }

    pub fn handoff_plan_digest(&self) -> Sha256Digest {
        self.handoff_plan_digest
    }

    pub fn activation_permit_id(&self) -> ClockGovernedUpgradeActivationPermitIdV1 {
        self.activation_permit_id
    }

    pub fn state_digest(&self) -> Sha256Digest {
        self.state_digest
    }

    pub fn state_generation(&self) -> u64 {
        self.state_generation
    }

    pub fn publication_digest(&self) -> Sha256Digest {
        self.publication_digest
    }

    pub fn publication_entry_sequence(&self) -> u64 {
        self.publication_entry_sequence
    }

    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }

    pub fn operational_lineage_digest(&self) -> Sha256Digest {
        self.operational_lineage_digest
    }

    pub fn operational_state_count(&self) -> usize {
        self.operational_state_count
    }

    pub fn probation_clearance_digest(&self) -> Option<Sha256Digest> {
        self.probation_clearance_digest
    }

    pub fn probation_sequence(&self) -> Option<u64> {
        self.probation_sequence
    }

    pub fn reauthorized_machine_count(&self) -> u64 {
        self.reauthorized_machine_count
    }

    pub fn retention_policy_digest(&self) -> Sha256Digest {
        self.retention_policy_digest
    }

    pub fn retention_policy_sequence(&self) -> u64 {
        self.retention_policy_sequence
    }

    pub fn key_snapshot_sequence(&self) -> u64 {
        self.key_snapshot_sequence
    }

    pub fn clock_epoch(&self) -> u64 {
        self.clock_epoch
    }

    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.observation_clock_envelope_id
    }

    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.observation_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurrentNoRollbackUpgradeError {
    GovernanceRetentionMismatch,
    HandoffActivationMismatch,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    ObservationBasisMismatch,
    ObservationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    FinalizationMayBeClosed,
    EmptyOperationalLineage,
    TooManyOperationalStates { actual: usize, maximum: usize },
    OperationalInputCountMismatch { states: usize, publications: usize, log_entries: usize },
    OperationalStateInvalid { index: usize, reason: String },
    OperationalGenesisInvalid,
    HandoffDigestMismatch { index: usize },
    StateBeforeActivation { index: usize },
    OperationalSuccessorInvalid { index: usize, reason: String },
    RollbackObserved { generation: u64, digest: Sha256Digest },
    PublicationMismatch { index: usize },
    PublicationDigestMismatch { index: usize },
    PublicationBeforeStateCommit { index: usize },
    PublicationMayBeFuture { index: usize },
    TransparencyLogInvalid(String),
    TransparencyLogMismatch,
    RetentionStateMismatch,
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct OperationalLineageEntryCommitment {
    generation: u64,
    state_digest: String,
    publication_digest: String,
    publication_entry_sequence: u64,
    publication_recorded_at_unix_s: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CurrentNoRollbackCommitment {
    schema: &'static str,
    governance_view_id: String,
    retention_head_id: String,
    governance_checkpoint_digest: String,
    handoff_id: String,
    handoff_plan_digest: String,
    activation_permit_id: String,
    state_digest: String,
    state_generation: u64,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    operational_lineage_digest: String,
    operational_state_count: usize,
    probation_clearance_digest: Option<String>,
    probation_sequence: Option<u64>,
    reauthorized_machine_count: u64,
    retention_policy_digest: String,
    retention_policy_sequence: u64,
    key_snapshot_sequence: u64,
    clock_epoch: u64,
    activation_clock_envelope_id: String,
    activation_operational_basis_id: String,
    observation_clock_envelope_id: String,
    observation_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn derive_current_no_rollback_upgrade_authority_v1(
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    retention_head: &CurrentEvidenceRetentionHeadV1,
    handoff: &ClockGovernedUpgradeHandoffV1,
    activation: &ClockGovernedUpgradeActivationPermitV1,
    activation_basis: &OperationalClockBasisV1,
    activation_to_observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    states: &[FabricationUpgradeOperationalState],
    publications: &[UpgradeOperationalHeadPublicationV1],
    log: &TransparencyLog,
) -> Result<CurrentNoRollbackUpgradeAuthorityV1, Vec<CurrentNoRollbackUpgradeError>> {
    let mut violations = Vec::new();

    if retention_head.governance_view_id() != governance_view.id()
        || retention_head.governance_checkpoint_digest() != governance_view.checkpoint_digest()
        || retention_head.transparency_log_digest() != governance_view.transparency_log_digest()
        || retention_head.observation_operational_basis_id()
            != governance_view.observation_operational_basis_id()
        || retention_head.observation_clock_envelope_id()
            != governance_view.observation_clock_envelope_id()
    {
        violations.push(CurrentNoRollbackUpgradeError::GovernanceRetentionMismatch);
    }

    if activation.handoff_id() != handoff.id()
        || activation.handoff_plan_digest() != handoff.plan_digest()
        || activation.activates_at_unix_ms() != handoff.plan().activates_at_unix_ms
        || activation.finalization_deadline_unix_ms()
            != handoff.plan().finalization_deadline_unix_ms
    {
        violations.push(CurrentNoRollbackUpgradeError::HandoffActivationMismatch);
    }

    if activation_basis.id() != activation.current_operational_basis_id() {
        violations.push(CurrentNoRollbackUpgradeError::ActivationBasisMismatch);
    }
    let activation_clock = match derive_clock_governance_evaluation_envelope_v1(activation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(CurrentNoRollbackUpgradeError::Clock(error));
            return Err(violations);
        }
    };
    if activation_clock.id() != activation.current_clock_envelope_id() {
        violations.push(CurrentNoRollbackUpgradeError::ActivationEnvelopeMismatch);
    }

    if observation_basis.id() != governance_view.observation_operational_basis_id() {
        violations.push(CurrentNoRollbackUpgradeError::ObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(CurrentNoRollbackUpgradeError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != governance_view.observation_clock_envelope_id() {
        violations.push(CurrentNoRollbackUpgradeError::ObservationEnvelopeMismatch);
    }

    if activation_to_observation_clock_bridge.len() > MAX_CURRENT_NO_ROLLBACK_CLOCK_HOPS {
        violations.push(CurrentNoRollbackUpgradeError::TooManyClockHops {
            actual: activation_to_observation_clock_bridge.len(),
            maximum: MAX_CURRENT_NO_ROLLBACK_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        activation_basis.id(),
        activation_to_observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }

    if observation_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms {
        violations.push(CurrentNoRollbackUpgradeError::FinalizationMayBeClosed);
    }

    if states.is_empty() {
        violations.push(CurrentNoRollbackUpgradeError::EmptyOperationalLineage);
        return Err(violations);
    }
    if states.len() > MAX_CURRENT_NO_ROLLBACK_STATES {
        violations.push(CurrentNoRollbackUpgradeError::TooManyOperationalStates {
            actual: states.len(),
            maximum: MAX_CURRENT_NO_ROLLBACK_STATES,
        });
        return Err(violations);
    }

    if let Err(error) = log.validate() {
        violations.push(CurrentNoRollbackUpgradeError::TransparencyLogInvalid(format!(
            "{error:?}"
        )));
    }
    let transparency_log_digest = match digest_transparency_log(log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(CurrentNoRollbackUpgradeError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if transparency_log_digest != governance_view.transparency_log_digest()
        || transparency_log_digest != retention_head.transparency_log_digest()
    {
        violations.push(CurrentNoRollbackUpgradeError::TransparencyLogMismatch);
    }

    let log_kind = match upgrade_operational_head_log_kind(handoff.plan_digest()) {
        Ok(value) => value,
        Err(error) => {
            violations.push(CurrentNoRollbackUpgradeError::Encoding(format!("{error:?}")));
            return Err(violations);
        }
    };
    let matching_entries = log
        .entries
        .iter()
        .filter(|entry| entry.kind == log_kind)
        .collect::<Vec<_>>();
    if states.len() != publications.len() || states.len() != matching_entries.len() {
        violations.push(CurrentNoRollbackUpgradeError::OperationalInputCountMismatch {
            states: states.len(),
            publications: publications.len(),
            log_entries: matching_entries.len(),
        });
        return Err(violations);
    }

    let mut lineage_commitments = Vec::with_capacity(states.len());
    let mut state_digests = Vec::with_capacity(states.len());
    let mut publication_digests = Vec::with_capacity(states.len());

    for (index, ((state, publication), entry)) in states
        .iter()
        .zip(publications.iter())
        .zip(matching_entries.iter())
        .enumerate()
    {
        if let Err(error) = state.validate_shape() {
            violations.push(CurrentNoRollbackUpgradeError::OperationalStateInvalid {
                index,
                reason: format!("{error:?}"),
            });
            continue;
        }
        if index == 0 && (state.generation != 1 || state.previous_state_digest.is_some()) {
            violations.push(CurrentNoRollbackUpgradeError::OperationalGenesisInvalid);
        }
        if state.handoff_digest != handoff.plan_digest() {
            violations.push(CurrentNoRollbackUpgradeError::HandoffDigestMismatch { index });
        }
        if state.committed_at_unix_ms < activation.activates_at_unix_ms() {
            violations.push(CurrentNoRollbackUpgradeError::StateBeforeActivation { index });
        }
        if index > 0 {
            if let Err(error) = verify_upgrade_operational_state_successor(&states[index - 1], state) {
                violations.push(CurrentNoRollbackUpgradeError::OperationalSuccessorInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
            }
        }
        if let Some(digest) = state.evidence.automatic_rollback_digest {
            violations.push(CurrentNoRollbackUpgradeError::RollbackObserved {
                generation: state.generation,
                digest,
            });
        }

        let state_digest = match digest_upgrade_operational_state(state) {
            Ok(value) => value,
            Err(error) => {
                violations.push(CurrentNoRollbackUpgradeError::OperationalStateInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        let expected_publication = match build_upgrade_operational_head_publication_v1(state) {
            Ok(value) => value,
            Err(error) => {
                violations.push(CurrentNoRollbackUpgradeError::Encoding(format!("{error:?}")));
                continue;
            }
        };
        if publication != &expected_publication {
            violations.push(CurrentNoRollbackUpgradeError::PublicationMismatch { index });
        }
        let publication_digest = match digest_upgrade_operational_head_publication_v1(publication) {
            Ok(value) => value,
            Err(error) => {
                violations.push(CurrentNoRollbackUpgradeError::Encoding(format!("{error:?}")));
                continue;
            }
        };
        if entry.subject_digest != publication_digest {
            violations.push(CurrentNoRollbackUpgradeError::PublicationDigestMismatch { index });
        }
        let publication_recorded_at_ms = match seconds_to_millis(entry.recorded_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if publication_recorded_at_ms < state.committed_at_unix_ms {
            violations.push(CurrentNoRollbackUpgradeError::PublicationBeforeStateCommit { index });
        }
        if publication_recorded_at_ms > observation_clock.lower_unix_ms() {
            violations.push(CurrentNoRollbackUpgradeError::PublicationMayBeFuture { index });
        }

        state_digests.push(state_digest);
        publication_digests.push(publication_digest);
        lineage_commitments.push(OperationalLineageEntryCommitment {
            generation: state.generation,
            state_digest: state_digest.to_hex(),
            publication_digest: publication_digest.to_hex(),
            publication_entry_sequence: entry.sequence,
            publication_recorded_at_unix_s: entry.recorded_at_unix_s,
        });
    }

    let candidate = states.last().expect("non-empty checked above");
    if candidate.evidence.retention_policy_sequence != retention_head.sequence()
        || candidate.evidence.retention_policy_digest != retention_head.policy_digest()
    {
        violations.push(CurrentNoRollbackUpgradeError::RetentionStateMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let operational_lineage_digest = hash_serializable(
        OPERATIONAL_LINEAGE_DOMAIN,
        &lineage_commitments,
    )
    .map_err(|error| vec![error])?;
    let (clock_lineage_digest, clock_hop_count) = digest_clock_lineage(
        activation_basis,
        activation_to_observation_clock_bridge,
        observation_basis,
    )
    .map_err(|error| vec![error])?;

    let state_digest = *state_digests.last().expect("one digest per valid state");
    let publication_digest = *publication_digests
        .last()
        .expect("one digest per valid publication");
    let publication_entry_sequence = matching_entries
        .last()
        .expect("matching count checked")
        .sequence;

    let commitment = CurrentNoRollbackCommitment {
        schema: CURRENT_NO_ROLLBACK_UPGRADE_AUTHORITY_SCHEMA,
        governance_view_id: governance_view.id().to_hex(),
        retention_head_id: retention_head.id().to_hex(),
        governance_checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        handoff_id: handoff.id().to_hex(),
        handoff_plan_digest: handoff.plan_digest().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        state_digest: state_digest.to_hex(),
        state_generation: candidate.generation,
        publication_digest: publication_digest.to_hex(),
        publication_entry_sequence,
        transparency_log_digest: transparency_log_digest.to_hex(),
        operational_lineage_digest: operational_lineage_digest.to_hex(),
        operational_state_count: states.len(),
        probation_clearance_digest: candidate
            .evidence
            .probation_clearance_digest
            .map(|value| value.to_hex()),
        probation_sequence: candidate.evidence.probation_sequence,
        reauthorized_machine_count: candidate.evidence.reauthorized_machine_count,
        retention_policy_digest: candidate.evidence.retention_policy_digest.to_hex(),
        retention_policy_sequence: candidate.evidence.retention_policy_sequence,
        key_snapshot_sequence: candidate.evidence.key_snapshot_sequence,
        clock_epoch: candidate.evidence.clock_epoch,
        activation_clock_envelope_id: activation_clock.id().to_hex(),
        activation_operational_basis_id: activation_basis.id().to_hex(),
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
    };
    let id = CurrentNoRollbackUpgradeAuthorityIdV1(
        hash_serializable(CURRENT_NO_ROLLBACK_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(CurrentNoRollbackUpgradeAuthorityV1 {
        id,
        governance_view_id: governance_view.id(),
        retention_head_id: retention_head.id(),
        governance_checkpoint_digest: governance_view.checkpoint_digest(),
        handoff_id: handoff.id(),
        handoff_plan_digest: handoff.plan_digest(),
        activation_permit_id: activation.id(),
        state_digest,
        state_generation: candidate.generation,
        publication_digest,
        publication_entry_sequence,
        transparency_log_digest,
        operational_lineage_digest,
        operational_state_count: states.len(),
        probation_clearance_digest: candidate.evidence.probation_clearance_digest,
        probation_sequence: candidate.evidence.probation_sequence,
        reauthorized_machine_count: candidate.evidence.reauthorized_machine_count,
        retention_policy_digest: candidate.evidence.retention_policy_digest,
        retention_policy_sequence: candidate.evidence.retention_policy_sequence,
        key_snapshot_sequence: candidate.evidence.key_snapshot_sequence,
        clock_epoch: candidate.evidence.clock_epoch,
        activation_clock_envelope_id: activation_clock.id(),
        activation_operational_basis_id: activation_basis.id(),
        observation_clock_envelope_id: observation_clock.id(),
        observation_operational_basis_id: observation_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), CurrentNoRollbackUpgradeError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(CurrentNoRollbackUpgradeError::BrokenClockLineage {
            hop: 1,
            expected_predecessor: prior_basis_id.to_hex(),
            actual_predecessor: bridge[0]
                .predecessor_operational_basis_id()
                .map(|value| value.to_hex()),
        });
    }

    let mut expected = prior_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(CurrentNoRollbackUpgradeError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }

    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(CurrentNoRollbackUpgradeError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_clock_lineage(
    start: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    end: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), CurrentNoRollbackUpgradeError> {
    let mut ids = Vec::with_capacity(bridge.len() + 2);
    ids.push(start.id().to_hex());
    ids.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if end.id() != start.id() {
        ids.push(end.id().to_hex());
    }
    let digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &ids)?;
    let hops = if end.id() == start.id() {
        0
    } else {
        bridge.len() + 1
    };
    Ok((digest, hops))
}

fn seconds_to_millis(value: u64) -> Result<u64, CurrentNoRollbackUpgradeError> {
    value
        .checked_mul(1_000)
        .ok_or(CurrentNoRollbackUpgradeError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, CurrentNoRollbackUpgradeError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CurrentNoRollbackUpgradeError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

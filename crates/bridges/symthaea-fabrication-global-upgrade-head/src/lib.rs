// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-view global upgrade-chain currentness and hardened predecessor roots.
//!
//! `CurrentFinalizedUpgradeHeadV1` is intentionally scoped to one handoff. That is sufficient to
//! prove one deterministic finalization is non-conflicting in its handoff namespace, but it is not
//! sufficient to choose the predecessor for a later upgrade: an older finalized handoff remains
//! valid in its own namespace after a newer handoff finalizes.
//!
//! This bridge closes that integration boundary in three steps:
//! 1. verify the complete concrete `FabricationUpgradeState` lineage ending at the exact activated
//!    state consumed by the deterministic finalization;
//! 2. scan every finalized-head publication in the exact authenticated log and require the candidate
//!    to have the highest non-equivocating finalization sequence;
//! 3. reconstruct the next-upgrade predecessor endpoint only from that global head, the exact
//!    deterministic terminal authority and the exact handoff that produced it.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeMap;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_kernel::upgrade_handoff::{UpgradeEndpoint, digest_upgrade_endpoint};
use symthaea_fabrication_kernel::upgrade_state::{
    FabricationUpgradeState, digest_upgrade_state, verify_upgrade_state_successor,
};
use symthaea_fabrication_kernel::upgrade_tracker::UpgradeStage;
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_finalized_head::{
    FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX, CurrentFinalizedUpgradeHeadIdV1,
    CurrentFinalizedUpgradeHeadV1, FinalizedUpgradeHeadPublicationV1,
    build_finalized_upgrade_head_publication_v1, digest_finalized_upgrade_head_publication_v1,
    finalized_upgrade_head_log_kind,
};
use symthaea_fabrication_upgrade_finalized_state::{
    ClockGovernedFinalizedUpgradeIdV1, ClockGovernedFinalizedUpgradeV1,
};
use symthaea_trust_kernel::{ClockGovernanceEvaluationEnvelopeIdV1, OperationalClockBasisIdV1};

pub const FINALIZED_UPGRADE_STATE_LINEAGE_SCHEMA: &str =
    "symthaea.fabrication.finalized-upgrade-state-lineage.v1";
pub const GLOBAL_CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA: &str =
    "symthaea.fabrication.global-current-finalized-upgrade-head.v1";
pub const FINALIZED_UPGRADE_PREDECESSOR_ROOT_SCHEMA: &str =
    "symthaea.fabrication.finalized-upgrade-predecessor-root.v1";
pub const MAX_FINALIZED_UPGRADE_STATE_LINEAGE: usize = 1_000_000;
pub const MAX_GLOBAL_FINALIZED_PUBLICATIONS: usize = 1_000_000;

const STATE_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.finalized-upgrade-state-lineage.v1\0";
const GLOBAL_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.global-current-finalized-upgrade-head.v1\0";
const PREDECESSOR_ROOT_DOMAIN: &[u8] =
    b"symthaea.fabrication.finalized-upgrade-predecessor-root.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FinalizedUpgradeStateLineageIdV1(Sha256Digest);

impl FinalizedUpgradeStateLineageIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct FinalizedUpgradeStateLineageV1 {
    id: FinalizedUpgradeStateLineageIdV1,
    finalized_upgrade_id: ClockGovernedFinalizedUpgradeIdV1,
    finalization_record_digest: Sha256Digest,
    state_count: usize,
    genesis_state_digest: Sha256Digest,
    activated_state_digest: Sha256Digest,
    activated_state_generation: u64,
    finalization_sequence: u64,
    lineage_digest: Sha256Digest,
}

impl FinalizedUpgradeStateLineageV1 {
    pub fn id(&self) -> FinalizedUpgradeStateLineageIdV1 {
        self.id
    }
    pub fn finalized_upgrade_id(&self) -> ClockGovernedFinalizedUpgradeIdV1 {
        self.finalized_upgrade_id
    }
    pub fn finalization_record_digest(&self) -> Sha256Digest {
        self.finalization_record_digest
    }
    pub fn state_count(&self) -> usize {
        self.state_count
    }
    pub fn genesis_state_digest(&self) -> Sha256Digest {
        self.genesis_state_digest
    }
    pub fn activated_state_digest(&self) -> Sha256Digest {
        self.activated_state_digest
    }
    pub fn activated_state_generation(&self) -> u64 {
        self.activated_state_generation
    }
    pub fn finalization_sequence(&self) -> u64 {
        self.finalization_sequence
    }
    pub fn lineage_digest(&self) -> Sha256Digest {
        self.lineage_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalCurrentFinalizedUpgradeHeadIdV1(Sha256Digest);

impl GlobalCurrentFinalizedUpgradeHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Highest non-equivocating finalized sequence published anywhere in one exact authenticated log.
/// This remains checkpoint-relative currentness; later checkpoints must be re-observed.
#[derive(Debug, Clone)]
#[must_use]
pub struct GlobalCurrentFinalizedUpgradeHeadV1 {
    id: GlobalCurrentFinalizedUpgradeHeadIdV1,
    current_head_id: CurrentFinalizedUpgradeHeadIdV1,
    finalized_upgrade_id: ClockGovernedFinalizedUpgradeIdV1,
    state_lineage_id: FinalizedUpgradeStateLineageIdV1,
    finalization_sequence: u64,
    record_digest: Sha256Digest,
    candidate_publication_digest: Sha256Digest,
    candidate_publication_count: usize,
    finalized_publication_count: usize,
    unique_finalized_sequence_count: usize,
    highest_finalization_sequence: u64,
    transparency_log_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl GlobalCurrentFinalizedUpgradeHeadV1 {
    pub fn id(&self) -> GlobalCurrentFinalizedUpgradeHeadIdV1 {
        self.id
    }
    pub fn current_head_id(&self) -> CurrentFinalizedUpgradeHeadIdV1 {
        self.current_head_id
    }
    pub fn finalized_upgrade_id(&self) -> ClockGovernedFinalizedUpgradeIdV1 {
        self.finalized_upgrade_id
    }
    pub fn state_lineage_id(&self) -> FinalizedUpgradeStateLineageIdV1 {
        self.state_lineage_id
    }
    pub fn finalization_sequence(&self) -> u64 {
        self.finalization_sequence
    }
    pub fn record_digest(&self) -> Sha256Digest {
        self.record_digest
    }
    pub fn candidate_publication_digest(&self) -> Sha256Digest {
        self.candidate_publication_digest
    }
    pub fn candidate_publication_count(&self) -> usize {
        self.candidate_publication_count
    }
    pub fn finalized_publication_count(&self) -> usize {
        self.finalized_publication_count
    }
    pub fn unique_finalized_sequence_count(&self) -> usize {
        self.unique_finalized_sequence_count
    }
    pub fn highest_finalization_sequence(&self) -> u64 {
        self.highest_finalization_sequence
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FinalizedUpgradePredecessorRootIdV1(Sha256Digest);

impl FinalizedUpgradePredecessorRootIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque predecessor root for the next hardened handoff. The endpoint is reconstructed from the
/// exact handoff whose deterministic finalization is globally current in the authenticated view.
#[derive(Debug, Clone)]
#[must_use]
pub struct FinalizedUpgradePredecessorRootV1 {
    id: FinalizedUpgradePredecessorRootIdV1,
    global_head_id: GlobalCurrentFinalizedUpgradeHeadIdV1,
    current_head_id: CurrentFinalizedUpgradeHeadIdV1,
    finalized_upgrade_id: ClockGovernedFinalizedUpgradeIdV1,
    producing_handoff_id: ClockGovernedUpgradeHandoffIdV1,
    producing_handoff_plan_digest: Sha256Digest,
    finalization_sequence: u64,
    endpoint: UpgradeEndpoint,
    endpoint_digest: Sha256Digest,
    rollback_target_digest: Sha256Digest,
    evidence_checkpoint_digest: Sha256Digest,
    transparency_log_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl FinalizedUpgradePredecessorRootV1 {
    pub fn id(&self) -> FinalizedUpgradePredecessorRootIdV1 {
        self.id
    }
    pub fn global_head_id(&self) -> GlobalCurrentFinalizedUpgradeHeadIdV1 {
        self.global_head_id
    }
    pub fn current_head_id(&self) -> CurrentFinalizedUpgradeHeadIdV1 {
        self.current_head_id
    }
    pub fn finalized_upgrade_id(&self) -> ClockGovernedFinalizedUpgradeIdV1 {
        self.finalized_upgrade_id
    }
    pub fn producing_handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.producing_handoff_id
    }
    pub fn producing_handoff_plan_digest(&self) -> Sha256Digest {
        self.producing_handoff_plan_digest
    }
    pub fn finalization_sequence(&self) -> u64 {
        self.finalization_sequence
    }
    pub fn endpoint(&self) -> &UpgradeEndpoint {
        &self.endpoint
    }
    pub fn endpoint_digest(&self) -> Sha256Digest {
        self.endpoint_digest
    }
    pub fn rollback_target_digest(&self) -> Sha256Digest {
        self.rollback_target_digest
    }
    pub fn evidence_checkpoint_digest(&self) -> Sha256Digest {
        self.evidence_checkpoint_digest
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalFinalizedUpgradeHeadError {
    EmptyStateLineage,
    StateLineageTooLarge { actual: usize, maximum: usize },
    StateInvalid { index: usize, reason: String },
    InvalidStateGenesis,
    StateSuccessorInvalid { index: usize, reason: String },
    ActivatedStateMismatch,
    FinalizationSequenceMismatch,
    FinalizedGenerationMismatch,
    HandoffDigestMismatch,
    TransparencyLogInvalid(String),
    TransparencyLogMismatch,
    PublicationCountTooLarge { actual: usize, maximum: usize },
    PublicationInputCountMismatch { log_entries: usize, publications: usize },
    PublicationInvalid { index: usize, reason: String },
    PublicationKindMismatch { index: usize },
    PublicationDigestMismatch { index: usize },
    FinalizedSequenceEquivocation { sequence: u64 },
    CandidateHeadMismatch,
    CandidatePublicationMissing,
    HigherFinalizedSequencePublished { candidate: u64, highest: u64 },
    FinalizedAuthorityMismatch,
    ProducingHandoffMismatch,
    ProducingEndpointMismatch,
    EndpointInvalid(String),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct StateLineageEntryCommitment {
    generation: u64,
    handoff_sequence: u64,
    stage: UpgradeStage,
    state_digest: String,
}

#[derive(Debug, Clone, Serialize)]
struct StateLineageCommitment {
    schema: &'static str,
    finalized_upgrade_id: String,
    finalization_record_digest: String,
    states: Vec<StateLineageEntryCommitment>,
}

#[derive(Debug, Clone, Serialize)]
struct GlobalHeadCommitment {
    schema: &'static str,
    current_head_id: String,
    finalized_upgrade_id: String,
    state_lineage_id: String,
    finalization_sequence: u64,
    record_digest: String,
    candidate_publication_digest: String,
    candidate_publication_count: usize,
    finalized_publication_count: usize,
    unique_finalized_sequence_count: usize,
    highest_finalization_sequence: u64,
    transparency_log_digest: String,
    checkpoint_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct PredecessorRootCommitment {
    schema: &'static str,
    global_head_id: String,
    current_head_id: String,
    finalized_upgrade_id: String,
    producing_handoff_id: String,
    producing_handoff_plan_digest: String,
    finalization_sequence: u64,
    endpoint_digest: String,
    rollback_target_digest: String,
    evidence_checkpoint_digest: String,
    transparency_log_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
}

pub fn verify_finalized_upgrade_state_lineage_v1(
    finalized: &ClockGovernedFinalizedUpgradeV1,
    states: &[FabricationUpgradeState],
) -> Result<FinalizedUpgradeStateLineageV1, Vec<GlobalFinalizedUpgradeHeadError>> {
    let mut violations = Vec::new();
    if states.is_empty() {
        return Err(vec![GlobalFinalizedUpgradeHeadError::EmptyStateLineage]);
    }
    if states.len() > MAX_FINALIZED_UPGRADE_STATE_LINEAGE {
        return Err(vec![GlobalFinalizedUpgradeHeadError::StateLineageTooLarge {
            actual: states.len(),
            maximum: MAX_FINALIZED_UPGRADE_STATE_LINEAGE,
        }]);
    }

    let mut commitments = Vec::with_capacity(states.len());
    for (index, state) in states.iter().enumerate() {
        if let Err(error) = state.validate_shape() {
            violations.push(GlobalFinalizedUpgradeHeadError::StateInvalid {
                index,
                reason: format!("{error:?}"),
            });
            continue;
        }
        let digest = match digest_upgrade_state(state) {
            Ok(value) => value,
            Err(error) => {
                violations.push(GlobalFinalizedUpgradeHeadError::StateInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        commitments.push(StateLineageEntryCommitment {
            generation: state.generation,
            handoff_sequence: state.handoff_sequence,
            stage: state.active_stage,
            state_digest: digest.to_hex(),
        });
        if index > 0 {
            if let Err(error) = verify_upgrade_state_successor(&states[index - 1], state) {
                violations.push(GlobalFinalizedUpgradeHeadError::StateSuccessorInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
            }
        }
    }

    let genesis = &states[0];
    if genesis.generation != 1
        || genesis.handoff_sequence != 1
        || genesis.previous_state_digest.is_some()
        || genesis.active_stage != UpgradeStage::Prepared
    {
        violations.push(GlobalFinalizedUpgradeHeadError::InvalidStateGenesis);
    }

    let record = finalized.record();
    let Some(activated) = states.last() else {
        return Err(vec![GlobalFinalizedUpgradeHeadError::EmptyStateLineage]);
    };
    let activated_digest = match digest_upgrade_state(activated) {
        Ok(value) => value,
        Err(error) => {
            violations.push(GlobalFinalizedUpgradeHeadError::StateInvalid {
                index: states.len() - 1,
                reason: format!("{error:?}"),
            });
            Sha256Digest([0; 32])
        }
    };
    if activated.active_stage != UpgradeStage::Activated
        || activated_digest != record.predecessor_upgrade_state_digest
        || activated.generation != record.predecessor_upgrade_state_generation
    {
        violations.push(GlobalFinalizedUpgradeHeadError::ActivatedStateMismatch);
    }
    if activated.handoff_sequence != record.finalization_sequence {
        violations.push(GlobalFinalizedUpgradeHeadError::FinalizationSequenceMismatch);
    }
    match activated.generation.checked_add(1) {
        Some(expected) if expected == record.finalized_upgrade_state_generation => {}
        _ => violations.push(GlobalFinalizedUpgradeHeadError::FinalizedGenerationMismatch),
    }
    if activated.evidence.handoff_digest != record.handoff_plan_digest {
        violations.push(GlobalFinalizedUpgradeHeadError::HandoffDigestMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let commitment = StateLineageCommitment {
        schema: FINALIZED_UPGRADE_STATE_LINEAGE_SCHEMA,
        finalized_upgrade_id: finalized.id().to_hex(),
        finalization_record_digest: finalized.record_digest().to_hex(),
        states: commitments,
    };
    let lineage_digest =
        hash_serializable(STATE_LINEAGE_DOMAIN, &commitment).map_err(|error| vec![error])?;
    let id = FinalizedUpgradeStateLineageIdV1(lineage_digest);
    let genesis_state_digest = digest_upgrade_state(genesis).map_err(|error| {
        vec![GlobalFinalizedUpgradeHeadError::StateInvalid {
            index: 0,
            reason: format!("{error:?}"),
        }]
    })?;

    Ok(FinalizedUpgradeStateLineageV1 {
        id,
        finalized_upgrade_id: finalized.id(),
        finalization_record_digest: finalized.record_digest(),
        state_count: states.len(),
        genesis_state_digest,
        activated_state_digest: activated_digest,
        activated_state_generation: activated.generation,
        finalization_sequence: activated.handoff_sequence,
        lineage_digest,
    })
}

pub fn qualify_global_current_finalized_upgrade_head_v1(
    current_head: &CurrentFinalizedUpgradeHeadV1,
    finalized: &ClockGovernedFinalizedUpgradeV1,
    state_lineage: &FinalizedUpgradeStateLineageV1,
    current_log: &TransparencyLog,
    publications: &[FinalizedUpgradeHeadPublicationV1],
) -> Result<GlobalCurrentFinalizedUpgradeHeadV1, Vec<GlobalFinalizedUpgradeHeadError>> {
    let mut violations = Vec::new();
    if current_head.finalized_upgrade_id() != finalized.id()
        || current_head.record_digest() != finalized.record_digest()
        || current_head.handoff_plan_digest() != finalized.record().handoff_plan_digest
        || current_head.finalization_sequence() != finalized.record().finalization_sequence
        || state_lineage.finalized_upgrade_id() != finalized.id()
        || state_lineage.finalization_record_digest() != finalized.record_digest()
        || state_lineage.finalization_sequence() != finalized.record().finalization_sequence
    {
        violations.push(GlobalFinalizedUpgradeHeadError::CandidateHeadMismatch);
    }

    if let Err(error) = current_log.validate() {
        violations.push(GlobalFinalizedUpgradeHeadError::TransparencyLogInvalid(format!(
            "{error:?}"
        )));
    }
    let log_digest = match digest_transparency_log(current_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(GlobalFinalizedUpgradeHeadError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if log_digest != current_head.current_transparency_log_digest() {
        violations.push(GlobalFinalizedUpgradeHeadError::TransparencyLogMismatch);
    }

    let matching_entries = current_log
        .entries
        .iter()
        .filter(|entry| entry.kind.starts_with(FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX))
        .collect::<Vec<_>>();
    if matching_entries.len() > MAX_GLOBAL_FINALIZED_PUBLICATIONS {
        violations.push(GlobalFinalizedUpgradeHeadError::PublicationCountTooLarge {
            actual: matching_entries.len(),
            maximum: MAX_GLOBAL_FINALIZED_PUBLICATIONS,
        });
        return Err(violations);
    }
    if matching_entries.len() != publications.len() {
        violations.push(GlobalFinalizedUpgradeHeadError::PublicationInputCountMismatch {
            log_entries: matching_entries.len(),
            publications: publications.len(),
        });
        return Err(violations);
    }

    let candidate_publication = build_finalized_upgrade_head_publication_v1(finalized);
    let candidate_digest = match digest_finalized_upgrade_head_publication_v1(&candidate_publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(GlobalFinalizedUpgradeHeadError::PublicationInvalid {
                index: 0,
                reason: format!("{error:?}"),
            });
            Sha256Digest([0; 32])
        }
    };
    if candidate_digest != current_head.publication_digest() {
        violations.push(GlobalFinalizedUpgradeHeadError::CandidateHeadMismatch);
    }

    let mut sequence_digests = BTreeMap::<u64, Sha256Digest>::new();
    let mut candidate_publication_count = 0usize;
    for (index, (entry, publication)) in matching_entries.iter().zip(publications).enumerate() {
        let publication_digest = match digest_finalized_upgrade_head_publication_v1(publication) {
            Ok(value) => value,
            Err(error) => {
                violations.push(GlobalFinalizedUpgradeHeadError::PublicationInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        let expected_kind = match finalized_upgrade_head_log_kind(publication.handoff_plan_digest) {
            Ok(value) => value,
            Err(error) => {
                violations.push(GlobalFinalizedUpgradeHeadError::PublicationInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        if entry.kind != expected_kind {
            violations.push(GlobalFinalizedUpgradeHeadError::PublicationKindMismatch { index });
        }
        if entry.subject_digest != publication_digest {
            violations.push(GlobalFinalizedUpgradeHeadError::PublicationDigestMismatch { index });
        }
        if let Some(previous) =
            sequence_digests.insert(publication.finalization_sequence, publication_digest)
        {
            if previous != publication_digest {
                violations.push(GlobalFinalizedUpgradeHeadError::FinalizedSequenceEquivocation {
                    sequence: publication.finalization_sequence,
                });
            }
        }
        if publication_digest == candidate_digest {
            candidate_publication_count = candidate_publication_count.saturating_add(1);
        }
    }

    if candidate_publication_count == 0 {
        violations.push(GlobalFinalizedUpgradeHeadError::CandidatePublicationMissing);
    }
    let highest_sequence = sequence_digests.keys().next_back().copied().unwrap_or(0);
    if highest_sequence > finalized.record().finalization_sequence {
        violations.push(GlobalFinalizedUpgradeHeadError::HigherFinalizedSequencePublished {
            candidate: finalized.record().finalization_sequence,
            highest: highest_sequence,
        });
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let commitment = GlobalHeadCommitment {
        schema: GLOBAL_CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA,
        current_head_id: current_head.id().to_hex(),
        finalized_upgrade_id: finalized.id().to_hex(),
        state_lineage_id: state_lineage.id().to_hex(),
        finalization_sequence: finalized.record().finalization_sequence,
        record_digest: finalized.record_digest().to_hex(),
        candidate_publication_digest: candidate_digest.to_hex(),
        candidate_publication_count,
        finalized_publication_count: matching_entries.len(),
        unique_finalized_sequence_count: sequence_digests.len(),
        highest_finalization_sequence: highest_sequence,
        transparency_log_digest: log_digest.to_hex(),
        checkpoint_digest: current_head.current_checkpoint_digest().to_hex(),
        clock_envelope_id: current_head.current_clock_envelope_id().to_hex(),
        operational_basis_id: current_head.current_operational_basis_id().to_hex(),
    };
    let id = GlobalCurrentFinalizedUpgradeHeadIdV1(
        hash_serializable(GLOBAL_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(GlobalCurrentFinalizedUpgradeHeadV1 {
        id,
        current_head_id: current_head.id(),
        finalized_upgrade_id: finalized.id(),
        state_lineage_id: state_lineage.id(),
        finalization_sequence: finalized.record().finalization_sequence,
        record_digest: finalized.record_digest(),
        candidate_publication_digest: candidate_digest,
        candidate_publication_count,
        finalized_publication_count: matching_entries.len(),
        unique_finalized_sequence_count: sequence_digests.len(),
        highest_finalization_sequence: highest_sequence,
        transparency_log_digest: log_digest,
        checkpoint_digest: current_head.current_checkpoint_digest(),
        clock_envelope_id: current_head.current_clock_envelope_id(),
        operational_basis_id: current_head.current_operational_basis_id(),
    })
}

pub fn derive_finalized_upgrade_predecessor_root_v1(
    global_head: &GlobalCurrentFinalizedUpgradeHeadV1,
    current_head: &CurrentFinalizedUpgradeHeadV1,
    finalized: &ClockGovernedFinalizedUpgradeV1,
    producing_handoff: &ClockGovernedUpgradeHandoffV1,
) -> Result<FinalizedUpgradePredecessorRootV1, GlobalFinalizedUpgradeHeadError> {
    if global_head.current_head_id() != current_head.id()
        || global_head.finalized_upgrade_id() != finalized.id()
        || global_head.record_digest() != finalized.record_digest()
        || global_head.finalization_sequence() != finalized.record().finalization_sequence
    {
        return Err(GlobalFinalizedUpgradeHeadError::FinalizedAuthorityMismatch);
    }
    if finalized.handoff_id() != producing_handoff.id()
        || finalized.record().handoff_id != producing_handoff.id().to_hex()
        || finalized.record().handoff_plan_digest != producing_handoff.plan_digest()
    {
        return Err(GlobalFinalizedUpgradeHeadError::ProducingHandoffMismatch);
    }

    let endpoint = producing_handoff.plan().successor.clone();
    endpoint
        .validate()
        .map_err(|error| GlobalFinalizedUpgradeHeadError::EndpointInvalid(format!("{error:?}")))?;
    let record = finalized.record();
    if endpoint.source_tree_digest != record.successor_source_tree_digest
        || endpoint.executable_digest != record.successor_executable_digest
        || endpoint.durable_state_digest != record.successor_durable_state_digest
        || endpoint.replay_contract_digest != record.successor_replay_contract_digest
        || endpoint.durable_state_digest != current_head.successor_durable_state_digest()
    {
        return Err(GlobalFinalizedUpgradeHeadError::ProducingEndpointMismatch);
    }
    let endpoint_digest = digest_upgrade_endpoint(&endpoint)
        .map_err(|error| GlobalFinalizedUpgradeHeadError::EndpointInvalid(format!("{error:?}")))?;

    let commitment = PredecessorRootCommitment {
        schema: FINALIZED_UPGRADE_PREDECESSOR_ROOT_SCHEMA,
        global_head_id: global_head.id().to_hex(),
        current_head_id: current_head.id().to_hex(),
        finalized_upgrade_id: finalized.id().to_hex(),
        producing_handoff_id: producing_handoff.id().to_hex(),
        producing_handoff_plan_digest: producing_handoff.plan_digest().to_hex(),
        finalization_sequence: global_head.finalization_sequence(),
        endpoint_digest: endpoint_digest.to_hex(),
        rollback_target_digest: endpoint.durable_state_digest.to_hex(),
        evidence_checkpoint_digest: global_head.checkpoint_digest().to_hex(),
        transparency_log_digest: global_head.transparency_log_digest().to_hex(),
        clock_envelope_id: global_head.clock_envelope_id().to_hex(),
        operational_basis_id: global_head.operational_basis_id().to_hex(),
    };
    let id = FinalizedUpgradePredecessorRootIdV1(hash_serializable(
        PREDECESSOR_ROOT_DOMAIN,
        &commitment,
    )?);

    Ok(FinalizedUpgradePredecessorRootV1 {
        id,
        global_head_id: global_head.id(),
        current_head_id: current_head.id(),
        finalized_upgrade_id: finalized.id(),
        producing_handoff_id: producing_handoff.id(),
        producing_handoff_plan_digest: producing_handoff.plan_digest(),
        finalization_sequence: global_head.finalization_sequence(),
        rollback_target_digest: endpoint.durable_state_digest,
        endpoint,
        endpoint_digest,
        evidence_checkpoint_digest: global_head.checkpoint_digest(),
        transparency_log_digest: global_head.transparency_log_digest(),
        clock_envelope_id: global_head.clock_envelope_id(),
        operational_basis_id: global_head.operational_basis_id(),
    })
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, GlobalFinalizedUpgradeHeadError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| GlobalFinalizedUpgradeHeadError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_upgrade_finalized_head::CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA;

    #[test]
    fn public_live_types_are_not_deserializable_by_construction() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<FinalizedUpgradeStateLineageV1>();
        assert_clone::<GlobalCurrentFinalizedUpgradeHeadV1>();
        assert_clone::<FinalizedUpgradePredecessorRootV1>();
    }

    #[test]
    fn finalized_head_schema_stays_distinct_from_global_head_schema() {
        assert_ne!(
            CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA,
            GLOBAL_CURRENT_FINALIZED_UPGRADE_HEAD_SCHEMA
        );
    }
}

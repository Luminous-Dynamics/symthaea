// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global finalized-sequence currentness and predecessor roots for lineage-bound fabrication upgrades.
//!
//! A handoff-scoped current finalized head is not enough to choose the predecessor of a later
//! upgrade: an older handoff remains current in its own namespace after a newer sequence finalizes.
//! This bridge verifies the complete concrete upgrade-state lineage for the candidate, scans every
//! lineage-finalized publication in one exact authenticated log, rejects sequence equivocation or
//! gaps, requires the candidate to be the highest sequence, and derives the next predecessor endpoint
//! directly from the candidate terminal record.

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
use symthaea_fabrication_lineage_bound_finalized_head::{
    LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX,
    CurrentLineageBoundFinalizedUpgradeHeadIdV1, CurrentLineageBoundFinalizedUpgradeHeadV1,
    LineageBoundFinalizedUpgradeHeadPublicationV1,
    build_lineage_bound_finalized_upgrade_head_publication_v1,
    digest_lineage_bound_finalized_upgrade_head_publication_v1,
    lineage_bound_finalized_upgrade_head_log_kind,
};
use symthaea_fabrication_lineage_bound_finalized_state::{
    LineageBoundFinalizedUpgradeIdV1, LineageBoundFinalizedUpgradeV1,
};
use symthaea_trust_kernel::{ClockGovernanceEvaluationEnvelopeIdV1, OperationalClockBasisIdV1};

pub const LINEAGE_BOUND_FINALIZED_STATE_LINEAGE_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalized-state-lineage.v1";
pub const GLOBAL_CURRENT_LINEAGE_BOUND_FINALIZED_HEAD_SCHEMA: &str =
    "symthaea.fabrication.global-current-lineage-bound-finalized-head.v1";
pub const LINEAGE_BOUND_FINALIZED_PREDECESSOR_ROOT_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalized-predecessor-root.v1";
pub const MAX_LINEAGE_BOUND_FINALIZED_STATE_LINEAGE: usize = 1_000_000;
pub const MAX_GLOBAL_LINEAGE_FINALIZED_PUBLICATIONS: usize = 1_000_000;

const STATE_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalized-state-lineage.v1\0";
const GLOBAL_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.global-current-lineage-bound-finalized-head.v1\0";
const PREDECESSOR_ROOT_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalized-predecessor-root.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundFinalizedStateLineageIdV1(Sha256Digest);

impl LineageBoundFinalizedStateLineageIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundFinalizedStateLineageV1 {
    id: LineageBoundFinalizedStateLineageIdV1,
    finalized_upgrade_id: LineageBoundFinalizedUpgradeIdV1,
    finalization_record_digest: Sha256Digest,
    state_count: usize,
    genesis_state_digest: Sha256Digest,
    activated_state_digest: Sha256Digest,
    activated_state_generation: u64,
    finalization_sequence: u64,
    lineage_digest: Sha256Digest,
}

impl LineageBoundFinalizedStateLineageV1 {
    pub fn id(&self) -> LineageBoundFinalizedStateLineageIdV1 { self.id }
    pub fn finalized_upgrade_id(&self) -> LineageBoundFinalizedUpgradeIdV1 { self.finalized_upgrade_id }
    pub fn finalization_record_digest(&self) -> Sha256Digest { self.finalization_record_digest }
    pub fn state_count(&self) -> usize { self.state_count }
    pub fn genesis_state_digest(&self) -> Sha256Digest { self.genesis_state_digest }
    pub fn activated_state_digest(&self) -> Sha256Digest { self.activated_state_digest }
    pub fn activated_state_generation(&self) -> u64 { self.activated_state_generation }
    pub fn finalization_sequence(&self) -> u64 { self.finalization_sequence }
    pub fn lineage_digest(&self) -> Sha256Digest { self.lineage_digest }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalCurrentLineageBoundFinalizedHeadIdV1(Sha256Digest);

impl GlobalCurrentLineageBoundFinalizedHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

/// Highest non-equivocating contiguous lineage-finalized sequence visible in one exact authenticated
/// log. This remains checkpoint-relative; a later checkpoint must be re-observed.
#[derive(Debug, Clone)]
#[must_use]
pub struct GlobalCurrentLineageBoundFinalizedHeadV1 {
    id: GlobalCurrentLineageBoundFinalizedHeadIdV1,
    current_head_id: CurrentLineageBoundFinalizedUpgradeHeadIdV1,
    finalized_upgrade_id: LineageBoundFinalizedUpgradeIdV1,
    state_lineage_id: LineageBoundFinalizedStateLineageIdV1,
    finalization_sequence: u64,
    record_digest: Sha256Digest,
    successor_endpoint_digest: Sha256Digest,
    candidate_publication_digest: Sha256Digest,
    candidate_publication_count: usize,
    lineage_publication_count: usize,
    unique_finalization_sequence_count: usize,
    lowest_finalization_sequence: u64,
    highest_finalization_sequence: u64,
    transparency_log_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl GlobalCurrentLineageBoundFinalizedHeadV1 {
    pub fn id(&self) -> GlobalCurrentLineageBoundFinalizedHeadIdV1 { self.id }
    pub fn current_head_id(&self) -> CurrentLineageBoundFinalizedUpgradeHeadIdV1 { self.current_head_id }
    pub fn finalized_upgrade_id(&self) -> LineageBoundFinalizedUpgradeIdV1 { self.finalized_upgrade_id }
    pub fn state_lineage_id(&self) -> LineageBoundFinalizedStateLineageIdV1 { self.state_lineage_id }
    pub fn finalization_sequence(&self) -> u64 { self.finalization_sequence }
    pub fn record_digest(&self) -> Sha256Digest { self.record_digest }
    pub fn successor_endpoint_digest(&self) -> Sha256Digest { self.successor_endpoint_digest }
    pub fn candidate_publication_digest(&self) -> Sha256Digest { self.candidate_publication_digest }
    pub fn candidate_publication_count(&self) -> usize { self.candidate_publication_count }
    pub fn lineage_publication_count(&self) -> usize { self.lineage_publication_count }
    pub fn unique_finalization_sequence_count(&self) -> usize { self.unique_finalization_sequence_count }
    pub fn lowest_finalization_sequence(&self) -> u64 { self.lowest_finalization_sequence }
    pub fn highest_finalization_sequence(&self) -> u64 { self.highest_finalization_sequence }
    pub fn transparency_log_digest(&self) -> Sha256Digest { self.transparency_log_digest }
    pub fn checkpoint_digest(&self) -> Sha256Digest { self.checkpoint_digest }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundFinalizedPredecessorRootIdV1(Sha256Digest);

impl LineageBoundFinalizedPredecessorRootIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

/// Opaque predecessor root for the next lineage-bound handoff. Unlike the older compatibility root,
/// the complete endpoint is reconstructed directly from the canonical terminal record.
#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundFinalizedPredecessorRootV1 {
    id: LineageBoundFinalizedPredecessorRootIdV1,
    global_head_id: GlobalCurrentLineageBoundFinalizedHeadIdV1,
    current_head_id: CurrentLineageBoundFinalizedUpgradeHeadIdV1,
    finalized_upgrade_id: LineageBoundFinalizedUpgradeIdV1,
    record_digest: Sha256Digest,
    prior_predecessor_root_digest: Sha256Digest,
    finalization_sequence: u64,
    endpoint: UpgradeEndpoint,
    endpoint_digest: Sha256Digest,
    rollback_target_digest: Sha256Digest,
    evidence_checkpoint_digest: Sha256Digest,
    transparency_log_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl LineageBoundFinalizedPredecessorRootV1 {
    pub fn id(&self) -> LineageBoundFinalizedPredecessorRootIdV1 { self.id }
    pub fn global_head_id(&self) -> GlobalCurrentLineageBoundFinalizedHeadIdV1 { self.global_head_id }
    pub fn current_head_id(&self) -> CurrentLineageBoundFinalizedUpgradeHeadIdV1 { self.current_head_id }
    pub fn finalized_upgrade_id(&self) -> LineageBoundFinalizedUpgradeIdV1 { self.finalized_upgrade_id }
    pub fn record_digest(&self) -> Sha256Digest { self.record_digest }
    pub fn prior_predecessor_root_digest(&self) -> Sha256Digest { self.prior_predecessor_root_digest }
    pub fn finalization_sequence(&self) -> u64 { self.finalization_sequence }
    pub fn endpoint(&self) -> &UpgradeEndpoint { &self.endpoint }
    pub fn endpoint_digest(&self) -> Sha256Digest { self.endpoint_digest }
    pub fn rollback_target_digest(&self) -> Sha256Digest { self.rollback_target_digest }
    pub fn evidence_checkpoint_digest(&self) -> Sha256Digest { self.evidence_checkpoint_digest }
    pub fn transparency_log_digest(&self) -> Sha256Digest { self.transparency_log_digest }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundGlobalUpgradeHeadError {
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
    FinalizedSequenceOverflow { sequence: u64 },
    FinalizedSequenceGap { previous: u64, next: u64 },
    CandidateHeadMismatch,
    CandidatePublicationMissing,
    HigherFinalizedSequencePublished { candidate: u64, highest: u64 },
    CountOverflow,
    FinalizedAuthorityMismatch,
    EndpointInvalid(String),
    EndpointDigestMismatch,
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
    successor_endpoint_digest: String,
    candidate_publication_digest: String,
    candidate_publication_count: usize,
    lineage_publication_count: usize,
    unique_finalization_sequence_count: usize,
    lowest_finalization_sequence: u64,
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
    record_digest: String,
    prior_predecessor_root_digest: String,
    finalization_sequence: u64,
    endpoint_digest: String,
    rollback_target_digest: String,
    evidence_checkpoint_digest: String,
    transparency_log_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
}

pub fn verify_lineage_bound_finalized_state_lineage_v1(
    finalized: &LineageBoundFinalizedUpgradeV1,
    states: &[FabricationUpgradeState],
) -> Result<LineageBoundFinalizedStateLineageV1, Vec<LineageBoundGlobalUpgradeHeadError>> {
    if states.is_empty() {
        return Err(vec![LineageBoundGlobalUpgradeHeadError::EmptyStateLineage]);
    }
    if states.len() > MAX_LINEAGE_BOUND_FINALIZED_STATE_LINEAGE {
        return Err(vec![LineageBoundGlobalUpgradeHeadError::StateLineageTooLarge {
            actual: states.len(),
            maximum: MAX_LINEAGE_BOUND_FINALIZED_STATE_LINEAGE,
        }]);
    }

    let mut violations = Vec::new();
    let mut commitments = Vec::with_capacity(states.len());
    let mut state_digests = Vec::with_capacity(states.len());
    for (index, state) in states.iter().enumerate() {
        if let Err(error) = state.validate_shape() {
            violations.push(LineageBoundGlobalUpgradeHeadError::StateInvalid {
                index,
                reason: format!("{error:?}"),
            });
            continue;
        }
        let digest = match digest_upgrade_state(state) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundGlobalUpgradeHeadError::StateInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        state_digests.push(digest);
        commitments.push(StateLineageEntryCommitment {
            generation: state.generation,
            handoff_sequence: state.handoff_sequence,
            stage: state.active_stage,
            state_digest: digest.to_hex(),
        });
        if index > 0 {
            if let Err(error) = verify_upgrade_state_successor(&states[index - 1], state) {
                violations.push(LineageBoundGlobalUpgradeHeadError::StateSuccessorInvalid {
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
        violations.push(LineageBoundGlobalUpgradeHeadError::InvalidStateGenesis);
    }

    let Some(activated) = states.last() else {
        return Err(vec![LineageBoundGlobalUpgradeHeadError::EmptyStateLineage]);
    };
    let activated_digest = match digest_upgrade_state(activated) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundGlobalUpgradeHeadError::StateInvalid {
                index: states.len() - 1,
                reason: format!("{error:?}"),
            });
            Sha256Digest([0; 32])
        }
    };
    let record = finalized.record();
    if activated.active_stage != UpgradeStage::Activated
        || activated_digest != record.predecessor_upgrade_state_digest
        || activated.generation != record.predecessor_upgrade_state_generation
    {
        violations.push(LineageBoundGlobalUpgradeHeadError::ActivatedStateMismatch);
    }
    if activated.handoff_sequence != record.finalization_sequence {
        violations.push(LineageBoundGlobalUpgradeHeadError::FinalizationSequenceMismatch);
    }
    match activated.generation.checked_add(1) {
        Some(expected) if expected == record.finalized_upgrade_state_generation => {}
        _ => violations.push(LineageBoundGlobalUpgradeHeadError::FinalizedGenerationMismatch),
    }
    if activated.evidence.handoff_digest != record.handoff_plan_digest {
        violations.push(LineageBoundGlobalUpgradeHeadError::HandoffDigestMismatch);
    }

    if state_digests.len() != states.len() || !violations.is_empty() {
        return Err(violations);
    }

    let lineage_commitment = StateLineageCommitment {
        schema: LINEAGE_BOUND_FINALIZED_STATE_LINEAGE_SCHEMA,
        finalized_upgrade_id: finalized.id().to_hex(),
        finalization_record_digest: finalized.record_digest().to_hex(),
        states: commitments,
    };
    let lineage_digest = hash_serializable(STATE_LINEAGE_DOMAIN, &lineage_commitment)
        .map_err(|error| vec![error])?;
    let genesis_state_digest = state_digests[0];
    let id = LineageBoundFinalizedStateLineageIdV1(lineage_digest);

    Ok(LineageBoundFinalizedStateLineageV1 {
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

pub fn qualify_global_current_lineage_bound_finalized_head_v1(
    current_head: &CurrentLineageBoundFinalizedUpgradeHeadV1,
    finalized: &LineageBoundFinalizedUpgradeV1,
    state_lineage: &LineageBoundFinalizedStateLineageV1,
    current_log: &TransparencyLog,
    publications: &[LineageBoundFinalizedUpgradeHeadPublicationV1],
) -> Result<GlobalCurrentLineageBoundFinalizedHeadV1, Vec<LineageBoundGlobalUpgradeHeadError>> {
    let mut violations = Vec::new();
    let record = finalized.record();
    if current_head.finalized_upgrade_id() != finalized.id()
        || current_head.record_digest() != finalized.record_digest()
        || current_head.handoff_plan_digest() != record.handoff_plan_digest
        || current_head.finalization_sequence() != record.finalization_sequence
        || current_head.successor_endpoint_digest() != record.successor_endpoint_digest
        || state_lineage.finalized_upgrade_id() != finalized.id()
        || state_lineage.finalization_record_digest() != finalized.record_digest()
        || state_lineage.finalization_sequence() != record.finalization_sequence
    {
        violations.push(LineageBoundGlobalUpgradeHeadError::CandidateHeadMismatch);
    }

    if let Err(error) = current_log.validate() {
        violations.push(LineageBoundGlobalUpgradeHeadError::TransparencyLogInvalid(format!(
            "{error:?}"
        )));
    }
    let log_digest = match digest_transparency_log(current_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundGlobalUpgradeHeadError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if log_digest != current_head.current_transparency_log_digest() {
        violations.push(LineageBoundGlobalUpgradeHeadError::TransparencyLogMismatch);
    }

    let matching_entries = current_log
        .entries
        .iter()
        .filter(|entry| entry.kind.starts_with(LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX))
        .collect::<Vec<_>>();
    if matching_entries.len() > MAX_GLOBAL_LINEAGE_FINALIZED_PUBLICATIONS {
        violations.push(LineageBoundGlobalUpgradeHeadError::PublicationCountTooLarge {
            actual: matching_entries.len(),
            maximum: MAX_GLOBAL_LINEAGE_FINALIZED_PUBLICATIONS,
        });
        return Err(violations);
    }
    if matching_entries.len() != publications.len() {
        violations.push(LineageBoundGlobalUpgradeHeadError::PublicationInputCountMismatch {
            log_entries: matching_entries.len(),
            publications: publications.len(),
        });
        return Err(violations);
    }

    let candidate_publication = build_lineage_bound_finalized_upgrade_head_publication_v1(finalized);
    let candidate_digest = match digest_lineage_bound_finalized_upgrade_head_publication_v1(&candidate_publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundGlobalUpgradeHeadError::PublicationInvalid {
                index: 0,
                reason: format!("{error:?}"),
            });
            Sha256Digest([0; 32])
        }
    };
    if candidate_digest != current_head.publication_digest() {
        violations.push(LineageBoundGlobalUpgradeHeadError::CandidateHeadMismatch);
    }

    let mut sequence_digests = BTreeMap::<u64, Sha256Digest>::new();
    let mut candidate_publication_count = 0usize;
    for (index, (entry, publication)) in matching_entries.iter().zip(publications).enumerate() {
        let publication_digest = match digest_lineage_bound_finalized_upgrade_head_publication_v1(publication) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundGlobalUpgradeHeadError::PublicationInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        let expected_kind = match lineage_bound_finalized_upgrade_head_log_kind(publication.handoff_plan_digest) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundGlobalUpgradeHeadError::PublicationInvalid {
                    index,
                    reason: format!("{error:?}"),
                });
                continue;
            }
        };
        if entry.kind != expected_kind {
            violations.push(LineageBoundGlobalUpgradeHeadError::PublicationKindMismatch { index });
        }
        if entry.subject_digest != publication_digest {
            violations.push(LineageBoundGlobalUpgradeHeadError::PublicationDigestMismatch { index });
        }
        if let Some(previous) = sequence_digests.insert(publication.finalization_sequence, publication_digest) {
            if previous != publication_digest {
                violations.push(LineageBoundGlobalUpgradeHeadError::FinalizedSequenceEquivocation {
                    sequence: publication.finalization_sequence,
                });
            }
        }
        if publication_digest == candidate_digest {
            candidate_publication_count = match candidate_publication_count.checked_add(1) {
                Some(value) => value,
                None => {
                    violations.push(LineageBoundGlobalUpgradeHeadError::CountOverflow);
                    candidate_publication_count
                }
            };
        }
    }

    let sequences = sequence_digests.keys().copied().collect::<Vec<_>>();
    for pair in sequences.windows(2) {
        let expected = match pair[0].checked_add(1) {
            Some(value) => value,
            None => {
                violations.push(LineageBoundGlobalUpgradeHeadError::FinalizedSequenceOverflow {
                    sequence: pair[0],
                });
                continue;
            }
        };
        if pair[1] != expected {
            violations.push(LineageBoundGlobalUpgradeHeadError::FinalizedSequenceGap {
                previous: pair[0],
                next: pair[1],
            });
        }
    }

    if candidate_publication_count == 0 {
        violations.push(LineageBoundGlobalUpgradeHeadError::CandidatePublicationMissing);
    }
    let Some(lowest_sequence) = sequences.first().copied() else {
        violations.push(LineageBoundGlobalUpgradeHeadError::CandidatePublicationMissing);
        return Err(violations);
    };
    let Some(highest_sequence) = sequences.last().copied() else {
        violations.push(LineageBoundGlobalUpgradeHeadError::CandidatePublicationMissing);
        return Err(violations);
    };
    if highest_sequence > record.finalization_sequence {
        violations.push(LineageBoundGlobalUpgradeHeadError::HigherFinalizedSequencePublished {
            candidate: record.finalization_sequence,
            highest: highest_sequence,
        });
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let commitment = GlobalHeadCommitment {
        schema: GLOBAL_CURRENT_LINEAGE_BOUND_FINALIZED_HEAD_SCHEMA,
        current_head_id: current_head.id().to_hex(),
        finalized_upgrade_id: finalized.id().to_hex(),
        state_lineage_id: state_lineage.id().to_hex(),
        finalization_sequence: record.finalization_sequence,
        record_digest: finalized.record_digest().to_hex(),
        successor_endpoint_digest: record.successor_endpoint_digest.to_hex(),
        candidate_publication_digest: candidate_digest.to_hex(),
        candidate_publication_count,
        lineage_publication_count: matching_entries.len(),
        unique_finalization_sequence_count: sequence_digests.len(),
        lowest_finalization_sequence: lowest_sequence,
        highest_finalization_sequence: highest_sequence,
        transparency_log_digest: log_digest.to_hex(),
        checkpoint_digest: current_head.current_checkpoint_digest().to_hex(),
        clock_envelope_id: current_head.current_clock_envelope_id().to_hex(),
        operational_basis_id: current_head.current_operational_basis_id().to_hex(),
    };
    let id = GlobalCurrentLineageBoundFinalizedHeadIdV1(
        hash_serializable(GLOBAL_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(GlobalCurrentLineageBoundFinalizedHeadV1 {
        id,
        current_head_id: current_head.id(),
        finalized_upgrade_id: finalized.id(),
        state_lineage_id: state_lineage.id(),
        finalization_sequence: record.finalization_sequence,
        record_digest: finalized.record_digest(),
        successor_endpoint_digest: record.successor_endpoint_digest,
        candidate_publication_digest: candidate_digest,
        candidate_publication_count,
        lineage_publication_count: matching_entries.len(),
        unique_finalization_sequence_count: sequence_digests.len(),
        lowest_finalization_sequence: lowest_sequence,
        highest_finalization_sequence: highest_sequence,
        transparency_log_digest: log_digest,
        checkpoint_digest: current_head.current_checkpoint_digest(),
        clock_envelope_id: current_head.current_clock_envelope_id(),
        operational_basis_id: current_head.current_operational_basis_id(),
    })
}

pub fn derive_lineage_bound_finalized_predecessor_root_v1(
    global_head: &GlobalCurrentLineageBoundFinalizedHeadV1,
    current_head: &CurrentLineageBoundFinalizedUpgradeHeadV1,
    finalized: &LineageBoundFinalizedUpgradeV1,
) -> Result<LineageBoundFinalizedPredecessorRootV1, LineageBoundGlobalUpgradeHeadError> {
    let record = finalized.record();
    if global_head.current_head_id() != current_head.id()
        || global_head.finalized_upgrade_id() != finalized.id()
        || global_head.record_digest() != finalized.record_digest()
        || global_head.finalization_sequence() != record.finalization_sequence
        || global_head.successor_endpoint_digest() != record.successor_endpoint_digest
        || current_head.finalized_upgrade_id() != finalized.id()
        || current_head.record_digest() != finalized.record_digest()
        || current_head.finalization_sequence() != record.finalization_sequence
        || current_head.successor_endpoint_digest() != record.successor_endpoint_digest
    {
        return Err(LineageBoundGlobalUpgradeHeadError::FinalizedAuthorityMismatch);
    }

    let endpoint = record.successor_endpoint.clone();
    endpoint
        .validate()
        .map_err(|error| LineageBoundGlobalUpgradeHeadError::EndpointInvalid(format!("{error:?}")))?;
    let endpoint_digest = digest_upgrade_endpoint(&endpoint)
        .map_err(|error| LineageBoundGlobalUpgradeHeadError::EndpointInvalid(format!("{error:?}")))?;
    if endpoint_digest != record.successor_endpoint_digest {
        return Err(LineageBoundGlobalUpgradeHeadError::EndpointDigestMismatch);
    }

    let commitment = PredecessorRootCommitment {
        schema: LINEAGE_BOUND_FINALIZED_PREDECESSOR_ROOT_SCHEMA,
        global_head_id: global_head.id().to_hex(),
        current_head_id: current_head.id().to_hex(),
        finalized_upgrade_id: finalized.id().to_hex(),
        record_digest: finalized.record_digest().to_hex(),
        prior_predecessor_root_digest: record.predecessor_root_digest.to_hex(),
        finalization_sequence: record.finalization_sequence,
        endpoint_digest: endpoint_digest.to_hex(),
        rollback_target_digest: endpoint.durable_state_digest.to_hex(),
        evidence_checkpoint_digest: global_head.checkpoint_digest().to_hex(),
        transparency_log_digest: global_head.transparency_log_digest().to_hex(),
        clock_envelope_id: global_head.clock_envelope_id().to_hex(),
        operational_basis_id: global_head.operational_basis_id().to_hex(),
    };
    let id = LineageBoundFinalizedPredecessorRootIdV1(hash_serializable(
        PREDECESSOR_ROOT_DOMAIN,
        &commitment,
    )?);

    Ok(LineageBoundFinalizedPredecessorRootV1 {
        id,
        global_head_id: global_head.id(),
        current_head_id: current_head.id(),
        finalized_upgrade_id: finalized.id(),
        record_digest: finalized.record_digest(),
        prior_predecessor_root_digest: record.predecessor_root_digest,
        finalization_sequence: record.finalization_sequence,
        endpoint,
        endpoint_digest,
        rollback_target_digest: record.successor_endpoint.durable_state_digest,
        evidence_checkpoint_digest: global_head.checkpoint_digest(),
        transparency_log_digest: global_head.transparency_log_digest(),
        clock_envelope_id: global_head.clock_envelope_id(),
        operational_basis_id: global_head.operational_basis_id(),
    })
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundGlobalUpgradeHeadError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundGlobalUpgradeHeadError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

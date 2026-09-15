// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lineage-bound upgrade handoff authority rooted in authenticated finalized predecessor evidence.
//!
//! Bootstrap authority remains byte-domain compatible with the original handoff path. After the
//! lineage-native terminal namespace begins, this bridge can independently requalify the complete
//! finalized-publication census inside the exact current governance log and mint an opaque native
//! predecessor capability. Both paths converge into the same downstream live handoff type without
//! introducing an upward Cargo dependency or a caller-implementable authority interface.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_fabrication_global_upgrade_head::FinalizedUpgradePredecessorRootV1;
use symthaea_fabrication_kernel::containment_state::{FabricationContainmentState, digest_containment_state};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::ThresholdCeremonyPolicy;
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_kernel::trust::{TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_kernel::upgrade_handoff::{UpgradeEndpoint, UpgradeHandoffPolicy, digest_upgrade_endpoint};
use symthaea_fabrication_trust_bridge::{ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1};
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffError, ClockGovernedUpgradeHandoffIdV1,
    ClockGovernedUpgradeHandoffPlanV1, ClockGovernedUpgradeHandoffV1,
    PreparedClockGovernedUpgradeHandoffIdV1, PreparedClockGovernedUpgradeHandoffV1,
    UpgradePolicyAuthorityInputV1, authorize_clock_governed_upgrade_handoff_v1,
    build_clock_governed_upgrade_handoff_plan_v1, prepare_clock_governed_upgrade_handoff_v1,
};
use symthaea_fabrication_upgrade_finalized_head::CurrentFinalizedUpgradeHeadV1;
use symthaea_fabrication_witness_registry_containment_bound::{
    ContainmentCurrentWitnessRegistryHeadIdV1, ContainmentCurrentWitnessRegistryHeadV1,
};
use symthaea_fabrication_witness_registry_head::{QuorumObservedWitnessRegistryHeadIdV1, QuorumObservedWitnessRegistryHeadV1};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const PREPARED_LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA: &str = "symthaea.fabrication.prepared-lineage-bound-upgrade-handoff.v1";
pub const LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA: &str = "symthaea.fabrication.lineage-bound-upgrade-handoff.v1";
pub const PREPARED_LINEAGE_NATIVE_UPGRADE_HANDOFF_SCHEMA: &str = "symthaea.fabrication.prepared-lineage-native-upgrade-handoff.v1";
pub const LINEAGE_NATIVE_UPGRADE_HANDOFF_SCHEMA: &str = "symthaea.fabrication.lineage-native-upgrade-handoff.v1";
pub const LINEAGE_NATIVE_FINALIZED_HEAD_PUBLICATION_SCHEMA: &str = "symthaea.fabrication.lineage-bound-finalized-upgrade-head-publication.v1";
pub const LINEAGE_NATIVE_FINALIZED_HEAD_LOG_KIND_PREFIX: &str = "lineage-finalized-upgrade-head-v1:";
pub const LINEAGE_NATIVE_PREDECESSOR_AUTHORITY_SCHEMA: &str = "symthaea.fabrication.lineage-native-predecessor-authority.v1";
pub const MAX_LINEAGE_NATIVE_FINALIZED_PUBLICATIONS: usize = 1_000_000;

const PREPARED_DOMAIN: &[u8] = b"symthaea.fabrication.prepared-lineage-bound-upgrade-handoff.v1\0";
const AUTHORIZED_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-bound-upgrade-handoff.v1\0";
const PREPARED_LINEAGE_NATIVE_DOMAIN: &[u8] = b"symthaea.fabrication.prepared-lineage-native-upgrade-handoff.v1\0";
const AUTHORIZED_LINEAGE_NATIVE_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-native-upgrade-handoff.v1\0";
const LINEAGE_NATIVE_PUBLICATION_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-bound-finalized-upgrade-head-publication.v1\0";
const LINEAGE_NATIVE_LOG_KIND_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-bound-finalized-upgrade-head-kind.v1\0";
const LINEAGE_NATIVE_PREDECESSOR_AUTHORITY_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-native-predecessor-authority.v1\0";
const LINEAGE_NATIVE_CURRENT_HEAD_REF_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-native-predecessor-current-head-ref.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineageBoundPredecessorAuthorityKindV1 { BootstrapV1, LineageNativeV1 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundPredecessorRootRefIdV1(Sha256Digest);
impl LineageBoundPredecessorRootRefIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundCurrentHeadRefIdV1(Sha256Digest);
impl LineageBoundCurrentHeadRefIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageNativeFinalizedHeadPublicationV1 {
    pub schema_version: String,
    pub finalized_upgrade_id: String,
    pub record_digest: Sha256Digest,
    pub finalization_sequence: u64,
    pub predecessor_finalization_sequence: u64,
    pub predecessor_root_digest: Sha256Digest,
    pub predecessor_current_head_digest: Sha256Digest,
    pub lineage_handoff_id: String,
    pub handoff_plan_digest: Sha256Digest,
    pub predecessor_upgrade_state_digest: Sha256Digest,
    pub predecessor_upgrade_state_generation: u64,
    pub finalized_upgrade_state_generation: u64,
    pub operational_state_digest: Sha256Digest,
    pub operational_lineage_digest: Sha256Digest,
    pub successor_endpoint_digest: Sha256Digest,
    pub authorization_id: String,
    pub execution_permit_id: String,
    pub state_binding_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageNativePredecessorAuthorityIdV1(Sha256Digest);
impl LineageNativePredecessorAuthorityIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageNativePredecessorAuthorityV1 {
    id: LineageNativePredecessorAuthorityIdV1,
    current_head_ref_digest: Sha256Digest,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    endpoint: UpgradeEndpoint,
    endpoint_digest: Sha256Digest,
    rollback_target_digest: Sha256Digest,
    finalization_sequence: u64,
    candidate_publication_digest: Sha256Digest,
    candidate_publication_count: usize,
    lineage_publication_count: usize,
    lowest_finalization_sequence: u64,
    highest_finalization_sequence: u64,
    checkpoint_digest: Sha256Digest,
    transparency_log_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}
impl LineageNativePredecessorAuthorityV1 {
    pub fn id(&self) -> LineageNativePredecessorAuthorityIdV1 { self.id }
    pub fn current_head_ref_digest(&self) -> Sha256Digest { self.current_head_ref_digest }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn endpoint(&self) -> &UpgradeEndpoint { &self.endpoint }
    pub fn endpoint_digest(&self) -> Sha256Digest { self.endpoint_digest }
    pub fn rollback_target_digest(&self) -> Sha256Digest { self.rollback_target_digest }
    pub fn finalization_sequence(&self) -> u64 { self.finalization_sequence }
    pub fn candidate_publication_digest(&self) -> Sha256Digest { self.candidate_publication_digest }
    pub fn candidate_publication_count(&self) -> usize { self.candidate_publication_count }
    pub fn lineage_publication_count(&self) -> usize { self.lineage_publication_count }
    pub fn lowest_finalization_sequence(&self) -> u64 { self.lowest_finalization_sequence }
    pub fn highest_finalization_sequence(&self) -> u64 { self.highest_finalization_sequence }
    pub fn checkpoint_digest(&self) -> Sha256Digest { self.checkpoint_digest }
    pub fn transparency_log_digest(&self) -> Sha256Digest { self.transparency_log_digest }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedLineageBoundUpgradeHandoffIdV1(Sha256Digest);
impl PreparedLineageBoundUpgradeHandoffIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundClockGovernedUpgradeHandoffIdV1(Sha256Digest);
impl LineageBoundClockGovernedUpgradeHandoffIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedLineageBoundUpgradeHandoffV1 {
    id: PreparedLineageBoundUpgradeHandoffIdV1,
    predecessor_authority_kind: LineageBoundPredecessorAuthorityKindV1,
    predecessor_root_id: LineageBoundPredecessorRootRefIdV1,
    inner_prepared: PreparedClockGovernedUpgradeHandoffV1,
    current_head_id: LineageBoundCurrentHeadRefIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    predecessor_endpoint_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: Sha256Digest,
    predecessor_transparency_log_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}
impl PreparedLineageBoundUpgradeHandoffV1 {
    pub fn id(&self) -> PreparedLineageBoundUpgradeHandoffIdV1 { self.id }
    pub fn predecessor_authority_kind(&self) -> LineageBoundPredecessorAuthorityKindV1 { self.predecessor_authority_kind }
    pub fn predecessor_root_id(&self) -> LineageBoundPredecessorRootRefIdV1 { self.predecessor_root_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_id.as_digest() }
    pub fn inner_prepared_id(&self) -> PreparedClockGovernedUpgradeHandoffIdV1 { self.inner_prepared.id() }
    pub fn signing_payload_digest(&self) -> Sha256Digest { self.inner_prepared.signing_payload_digest() }
    pub fn plan(&self) -> &ClockGovernedUpgradeHandoffPlanV1 { self.inner_prepared.plan() }
    pub fn plan_digest(&self) -> Sha256Digest { self.inner_prepared.plan_digest() }
    pub fn current_head_id(&self) -> LineageBoundCurrentHeadRefIdV1 { self.current_head_id }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_id.as_digest() }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 { self.registry_head_id }
    pub fn predecessor_endpoint_digest(&self) -> Sha256Digest { self.predecessor_endpoint_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn predecessor_checkpoint_digest(&self) -> Sha256Digest { self.predecessor_checkpoint_digest }
    pub fn predecessor_transparency_log_digest(&self) -> Sha256Digest { self.predecessor_transparency_log_digest }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn containment_state_digest(&self) -> Sha256Digest { self.containment_state_digest }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest { self.compromise_tracker_digest }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundClockGovernedUpgradeHandoffV1 {
    id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    prepared_id: PreparedLineageBoundUpgradeHandoffIdV1,
    predecessor_authority_kind: LineageBoundPredecessorAuthorityKindV1,
    predecessor_root_id: LineageBoundPredecessorRootRefIdV1,
    inner_handoff: ClockGovernedUpgradeHandoffV1,
    current_head_id: LineageBoundCurrentHeadRefIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    predecessor_endpoint_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: Sha256Digest,
    predecessor_transparency_log_digest: Sha256Digest,
    threshold_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}
impl LineageBoundClockGovernedUpgradeHandoffV1 {
    pub fn id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.id }
    pub fn prepared_id(&self) -> PreparedLineageBoundUpgradeHandoffIdV1 { self.prepared_id }
    pub fn predecessor_authority_kind(&self) -> LineageBoundPredecessorAuthorityKindV1 { self.predecessor_authority_kind }
    pub fn predecessor_root_id(&self) -> LineageBoundPredecessorRootRefIdV1 { self.predecessor_root_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_id.as_digest() }
    pub fn inner_handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 { self.inner_handoff.id() }
    pub fn plan(&self) -> &ClockGovernedUpgradeHandoffPlanV1 { self.inner_handoff.plan() }
    pub fn plan_digest(&self) -> Sha256Digest { self.inner_handoff.plan_digest() }
    pub fn current_head_id(&self) -> LineageBoundCurrentHeadRefIdV1 { self.current_head_id }
    pub fn current_head_digest(&self) -> Sha256Digest { self.current_head_id.as_digest() }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 { self.registry_head_id }
    pub fn predecessor_endpoint_digest(&self) -> Sha256Digest { self.predecessor_endpoint_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn predecessor_checkpoint_digest(&self) -> Sha256Digest { self.predecessor_checkpoint_digest }
    pub fn predecessor_transparency_log_digest(&self) -> Sha256Digest { self.predecessor_transparency_log_digest }
    pub fn threshold_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 { self.threshold_ceremony_id }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.clock_envelope_id }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.operational_basis_id }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundUpgradeHandoffError {
    PredecessorMismatch, RollbackTargetMismatch, EvidenceCheckpointMismatch, CurrentHeadMismatch,
    GovernanceViewMismatch, RegistryHeadMismatch, TrustSnapshotInvalid(String), TrustSnapshotMismatch,
    ContainmentStateInvalid(String), ContainmentStateMismatch, CompromiseTrackerMismatch,
    ContainmentGenerationMismatch, OperationalBasisMismatch, Clock(ClockGovernanceTimeError),
    ClockEnvelopeMismatch, NativeLogInvalid(String), NativeLogMismatch, NativeObservationBasisMismatch,
    NativeObservationEnvelopeMismatch, NativePublicationCountTooLarge { actual: usize, maximum: usize },
    NativePublicationInputCountMismatch { log_entries: usize, publications: usize },
    NativePublicationInvalid { index: usize }, NativePublicationKindMismatch { index: usize },
    NativePublicationDigestMismatch { index: usize }, NativePublicationMayBeFuture { entry_sequence: u64 },
    NativeSequenceEquivocation { sequence: u64 }, NativeSequenceOverflow { sequence: u64 },
    NativeSequenceGap { previous: u64, next: u64 }, NativeCandidateMissing,
    NativeEndpointInvalid(String), NativeTimeScaleOverflow, InnerBuild(String),
    InnerPreparation(Vec<ClockGovernedUpgradeHandoffError>), InnerAuthorization(String), Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct NativePredecessorCommitment {
    schema: &'static str, governance_view_id: String, endpoint_digest: String,
    rollback_target_digest: String, finalization_sequence: u64, candidate_publication_digest: String,
    candidate_publication_count: usize, lineage_publication_count: usize,
    lowest_finalization_sequence: u64, highest_finalization_sequence: u64,
    checkpoint_digest: String, transparency_log_digest: String, clock_envelope_id: String,
    operational_basis_id: String,
}
#[derive(Debug, Clone, Serialize)]
struct NativeCurrentHeadRefCommitment {
    predecessor_authority_id: String, candidate_publication_digest: String, checkpoint_digest: String,
    transparency_log_digest: String, finalization_sequence: u64,
}
#[derive(Debug, Clone, Serialize)]
struct PreparedCommitment {
    schema: &'static str, predecessor_root_id: String, inner_prepared_id: String, plan_digest: String,
    current_head_id: String, governance_view_id: String, registry_head_id: String,
    predecessor_endpoint_digest: String, predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: String, predecessor_transparency_log_digest: String,
    trust_snapshot_digest: String, containment_state_digest: String, compromise_tracker_digest: String,
    clock_envelope_id: String, operational_basis_id: String,
}
#[derive(Debug, Clone, Serialize)]
struct AuthorizedCommitment {
    schema: &'static str, prepared_id: String, predecessor_root_id: String, inner_handoff_id: String,
    plan_digest: String, current_head_id: String, governance_view_id: String, registry_head_id: String,
    predecessor_endpoint_digest: String, predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: String, predecessor_transparency_log_digest: String,
    threshold_ceremony_id: String, clock_envelope_id: String, operational_basis_id: String,
}

fn prepared_schema(kind: LineageBoundPredecessorAuthorityKindV1) -> &'static str {
    match kind { LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => PREPARED_LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA, LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => PREPARED_LINEAGE_NATIVE_UPGRADE_HANDOFF_SCHEMA }
}
fn authorized_schema(kind: LineageBoundPredecessorAuthorityKindV1) -> &'static str {
    match kind { LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA, LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => LINEAGE_NATIVE_UPGRADE_HANDOFF_SCHEMA }
}
fn prepared_domain(kind: LineageBoundPredecessorAuthorityKindV1) -> &'static [u8] {
    match kind { LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => PREPARED_DOMAIN, LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => PREPARED_LINEAGE_NATIVE_DOMAIN }
}
fn authorized_domain(kind: LineageBoundPredecessorAuthorityKindV1) -> &'static [u8] {
    match kind { LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => AUTHORIZED_DOMAIN, LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => AUTHORIZED_LINEAGE_NATIVE_DOMAIN }
}

pub fn qualify_lineage_native_predecessor_authority_v1(
    candidate_endpoint: UpgradeEndpoint,
    current_log: &TransparencyLog,
    publications: &[LineageNativeFinalizedHeadPublicationV1],
    observation_basis: &OperationalClockBasisV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
) -> Result<LineageNativePredecessorAuthorityV1, Vec<LineageBoundUpgradeHandoffError>> {
    let mut violations = Vec::new();
    if let Err(error) = candidate_endpoint.validate() { violations.push(LineageBoundUpgradeHandoffError::NativeEndpointInvalid(format!("{error:?}"))); }
    let endpoint_digest = match digest_upgrade_endpoint(&candidate_endpoint) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::NativeEndpointInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    if let Err(error) = current_log.validate() { violations.push(LineageBoundUpgradeHandoffError::NativeLogInvalid(format!("{error:?}"))); }
    let log_digest = match digest_transparency_log(current_log) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::NativeLogInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    if log_digest != governance_view.transparency_log_digest() { violations.push(LineageBoundUpgradeHandoffError::NativeLogMismatch); }
    if observation_basis.id() != governance_view.observation_operational_basis_id() { violations.push(LineageBoundUpgradeHandoffError::NativeObservationBasisMismatch); }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::Clock(error)); return Err(violations); }
    };
    if observation_clock.id() != governance_view.observation_clock_envelope_id() { violations.push(LineageBoundUpgradeHandoffError::NativeObservationEnvelopeMismatch); }

    let matching_entries = current_log.entries.iter().filter(|entry| entry.kind.starts_with(LINEAGE_NATIVE_FINALIZED_HEAD_LOG_KIND_PREFIX)).collect::<Vec<_>>();
    if matching_entries.len() > MAX_LINEAGE_NATIVE_FINALIZED_PUBLICATIONS {
        violations.push(LineageBoundUpgradeHandoffError::NativePublicationCountTooLarge { actual: matching_entries.len(), maximum: MAX_LINEAGE_NATIVE_FINALIZED_PUBLICATIONS });
        return Err(violations);
    }
    if matching_entries.len() != publications.len() {
        violations.push(LineageBoundUpgradeHandoffError::NativePublicationInputCountMismatch { log_entries: matching_entries.len(), publications: publications.len() });
        return Err(violations);
    }
    let mut sequence_digests = BTreeMap::<u64, Sha256Digest>::new();
    for (index, (entry, publication)) in matching_entries.iter().zip(publications).enumerate() {
        if !valid_lineage_native_publication(publication) { violations.push(LineageBoundUpgradeHandoffError::NativePublicationInvalid { index }); continue; }
        let publication_digest = match hash_serializable(LINEAGE_NATIVE_PUBLICATION_DOMAIN, publication) { Ok(value) => value, Err(error) => { violations.push(error); continue; } };
        let expected_kind = match lineage_native_log_kind(publication.handoff_plan_digest) { Ok(value) => value, Err(error) => { violations.push(error); continue; } };
        if entry.kind != expected_kind { violations.push(LineageBoundUpgradeHandoffError::NativePublicationKindMismatch { index }); }
        if entry.subject_digest != publication_digest { violations.push(LineageBoundUpgradeHandoffError::NativePublicationDigestMismatch { index }); }
        let recorded_at_unix_ms = match entry.recorded_at_unix_s.checked_mul(1_000) {
            Some(value) => value,
            None => { violations.push(LineageBoundUpgradeHandoffError::NativeTimeScaleOverflow); continue; }
        };
        if recorded_at_unix_ms > observation_clock.lower_unix_ms() { violations.push(LineageBoundUpgradeHandoffError::NativePublicationMayBeFuture { entry_sequence: entry.sequence }); }
        if let Some(previous) = sequence_digests.insert(publication.finalization_sequence, publication_digest) {
            if previous != publication_digest { violations.push(LineageBoundUpgradeHandoffError::NativeSequenceEquivocation { sequence: publication.finalization_sequence }); }
        }
    }
    let sequences = sequence_digests.keys().copied().collect::<Vec<_>>();
    for pair in sequences.windows(2) {
        let expected = match pair[0].checked_add(1) { Some(value) => value, None => { violations.push(LineageBoundUpgradeHandoffError::NativeSequenceOverflow { sequence: pair[0] }); continue; } };
        if pair[1] != expected { violations.push(LineageBoundUpgradeHandoffError::NativeSequenceGap { previous: pair[0], next: pair[1] }); }
    }
    let Some(lowest_sequence) = sequences.first().copied() else { violations.push(LineageBoundUpgradeHandoffError::NativeCandidateMissing); return Err(violations); };
    let Some(highest_sequence) = sequences.last().copied() else { violations.push(LineageBoundUpgradeHandoffError::NativeCandidateMissing); return Err(violations); };
    let candidate_publication_count = publications.iter().filter(|publication| publication.finalization_sequence == highest_sequence && publication.successor_endpoint_digest == endpoint_digest).count();
    if candidate_publication_count == 0 { violations.push(LineageBoundUpgradeHandoffError::NativeCandidateMissing); }
    let candidate_publication_digest = match sequence_digests.get(&highest_sequence).copied() { Some(value) => value, None => { violations.push(LineageBoundUpgradeHandoffError::NativeCandidateMissing); return Err(violations); } };
    if !violations.is_empty() { return Err(violations); }
    let rollback_target_digest = candidate_endpoint.durable_state_digest;
    let commitment = NativePredecessorCommitment {
        schema: LINEAGE_NATIVE_PREDECESSOR_AUTHORITY_SCHEMA, governance_view_id: governance_view.id().to_hex(), endpoint_digest: endpoint_digest.to_hex(),
        rollback_target_digest: rollback_target_digest.to_hex(), finalization_sequence: highest_sequence, candidate_publication_digest: candidate_publication_digest.to_hex(),
        candidate_publication_count, lineage_publication_count: matching_entries.len(), lowest_finalization_sequence: lowest_sequence, highest_finalization_sequence: highest_sequence,
        checkpoint_digest: governance_view.checkpoint_digest().to_hex(), transparency_log_digest: log_digest.to_hex(), clock_envelope_id: observation_clock.id().to_hex(), operational_basis_id: observation_basis.id().to_hex(),
    };
    let id = LineageNativePredecessorAuthorityIdV1(hash_serializable(LINEAGE_NATIVE_PREDECESSOR_AUTHORITY_DOMAIN, &commitment).map_err(|error| vec![error])?);
    let current_head_ref_digest = hash_serializable(LINEAGE_NATIVE_CURRENT_HEAD_REF_DOMAIN, &NativeCurrentHeadRefCommitment {
        predecessor_authority_id: id.to_hex(), candidate_publication_digest: candidate_publication_digest.to_hex(), checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        transparency_log_digest: log_digest.to_hex(), finalization_sequence: highest_sequence,
    }).map_err(|error| vec![error])?;
    Ok(LineageNativePredecessorAuthorityV1 {
        id, current_head_ref_digest, governance_view_id: governance_view.id(), endpoint: candidate_endpoint, endpoint_digest, rollback_target_digest,
        finalization_sequence: highest_sequence, candidate_publication_digest, candidate_publication_count, lineage_publication_count: matching_entries.len(),
        lowest_finalization_sequence: lowest_sequence, highest_finalization_sequence: highest_sequence, checkpoint_digest: governance_view.checkpoint_digest(),
        transparency_log_digest: log_digest, clock_envelope_id: observation_clock.id(), operational_basis_id: observation_basis.id(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_lineage_bound_upgrade_handoff_plan_v1(predecessor_root: &FinalizedUpgradePredecessorRootV1, successor: UpgradeEndpoint, activates_at_unix_ms: u64, finalization_deadline_unix_ms: u64, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], recovery_key_set_digest: Sha256Digest, reason: impl Into<String>) -> Result<ClockGovernedUpgradeHandoffPlanV1, LineageBoundUpgradeHandoffError> {
    build_clock_governed_upgrade_handoff_plan_v1(predecessor_root.endpoint().clone(), successor, activates_at_unix_ms, finalization_deadline_unix_ms, predecessor_root.rollback_target_digest(), policy_authorities, predecessor_root.evidence_checkpoint_digest(), recovery_key_set_digest, reason).map_err(|error| LineageBoundUpgradeHandoffError::InnerBuild(format!("{error:?}")))
}
#[allow(clippy::too_many_arguments)]
pub fn build_lineage_native_upgrade_handoff_plan_v1(predecessor: &LineageNativePredecessorAuthorityV1, successor: UpgradeEndpoint, activates_at_unix_ms: u64, finalization_deadline_unix_ms: u64, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], recovery_key_set_digest: Sha256Digest, reason: impl Into<String>) -> Result<ClockGovernedUpgradeHandoffPlanV1, LineageBoundUpgradeHandoffError> {
    build_clock_governed_upgrade_handoff_plan_v1(predecessor.endpoint().clone(), successor, activates_at_unix_ms, finalization_deadline_unix_ms, predecessor.rollback_target_digest(), policy_authorities, predecessor.checkpoint_digest(), recovery_key_set_digest, reason).map_err(|error| LineageBoundUpgradeHandoffError::InnerBuild(format!("{error:?}")))
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_lineage_bound_upgrade_handoff_v1(predecessor_root: &FinalizedUpgradePredecessorRootV1, current_head: &CurrentFinalizedUpgradeHeadV1, governance_view: &ContainmentCurrentWitnessRegistryHeadV1, registry_head: &QuorumObservedWitnessRegistryHeadV1, plan: ClockGovernedUpgradeHandoffPlanV1, handoff_policy: &UpgradeHandoffPolicy, threshold_policy: &ThresholdCeremonyPolicy, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState, operational_basis: &OperationalClockBasisV1) -> Result<PreparedLineageBoundUpgradeHandoffV1, Vec<LineageBoundUpgradeHandoffError>> {
    let mut violations = validate_common_plan(&plan, predecessor_root.endpoint(), predecessor_root.rollback_target_digest(), predecessor_root.evidence_checkpoint_digest());
    if predecessor_root.current_head_id() != current_head.id() || predecessor_root.evidence_checkpoint_digest() != current_head.current_checkpoint_digest() || predecessor_root.transparency_log_digest() != current_head.current_transparency_log_digest() || predecessor_root.operational_basis_id() != current_head.current_operational_basis_id() || predecessor_root.clock_envelope_id() != current_head.current_clock_envelope_id() { violations.push(LineageBoundUpgradeHandoffError::CurrentHeadMismatch); }
    if current_head.governance_view_id() != governance_view.id() { violations.push(LineageBoundUpgradeHandoffError::GovernanceViewMismatch); }
    let context = qualify_common_governance_context(governance_view, registry_head, predecessor_root.evidence_checkpoint_digest(), predecessor_root.transparency_log_digest(), predecessor_root.operational_basis_id(), predecessor_root.clock_envelope_id(), trust_snapshot, containment_state, operational_basis, &mut violations)?;
    if !violations.is_empty() { return Err(violations); }
    prepare_inner_handoff(LineageBoundPredecessorAuthorityKindV1::BootstrapV1, LineageBoundPredecessorRootRefIdV1(predecessor_root.id().as_digest()), LineageBoundCurrentHeadRefIdV1(current_head.id().as_digest()), predecessor_root.endpoint_digest(), predecessor_root.finalization_sequence(), predecessor_root.evidence_checkpoint_digest(), predecessor_root.transparency_log_digest(), governance_view, registry_head, plan, handoff_policy, threshold_policy, policy_authorities, trust_snapshot, containment_state, operational_basis, context)
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_lineage_native_upgrade_handoff_v1(predecessor: &LineageNativePredecessorAuthorityV1, governance_view: &ContainmentCurrentWitnessRegistryHeadV1, registry_head: &QuorumObservedWitnessRegistryHeadV1, plan: ClockGovernedUpgradeHandoffPlanV1, handoff_policy: &UpgradeHandoffPolicy, threshold_policy: &ThresholdCeremonyPolicy, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState, operational_basis: &OperationalClockBasisV1) -> Result<PreparedLineageBoundUpgradeHandoffV1, Vec<LineageBoundUpgradeHandoffError>> {
    let mut violations = validate_common_plan(&plan, predecessor.endpoint(), predecessor.rollback_target_digest(), predecessor.checkpoint_digest());
    if predecessor.governance_view_id() != governance_view.id() { violations.push(LineageBoundUpgradeHandoffError::GovernanceViewMismatch); }
    let context = qualify_common_governance_context(governance_view, registry_head, predecessor.checkpoint_digest(), predecessor.transparency_log_digest(), predecessor.operational_basis_id(), predecessor.clock_envelope_id(), trust_snapshot, containment_state, operational_basis, &mut violations)?;
    if !violations.is_empty() { return Err(violations); }
    prepare_inner_handoff(LineageBoundPredecessorAuthorityKindV1::LineageNativeV1, LineageBoundPredecessorRootRefIdV1(predecessor.id().as_digest()), LineageBoundCurrentHeadRefIdV1(predecessor.current_head_ref_digest()), predecessor.endpoint_digest(), predecessor.finalization_sequence(), predecessor.checkpoint_digest(), predecessor.transparency_log_digest(), governance_view, registry_head, plan, handoff_policy, threshold_policy, policy_authorities, trust_snapshot, containment_state, operational_basis, context)
}

#[derive(Debug, Clone, Copy)]
struct QualifiedCommonContext { trust_snapshot_digest: Sha256Digest, containment_state_digest: Sha256Digest, compromise_tracker_digest: Sha256Digest, clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1 }
fn validate_common_plan(plan: &ClockGovernedUpgradeHandoffPlanV1, predecessor: &UpgradeEndpoint, rollback_target: Sha256Digest, evidence_checkpoint: Sha256Digest) -> Vec<LineageBoundUpgradeHandoffError> {
    let mut violations = Vec::new();
    if plan.predecessor != *predecessor { violations.push(LineageBoundUpgradeHandoffError::PredecessorMismatch); }
    if plan.rollback_target_digest != rollback_target { violations.push(LineageBoundUpgradeHandoffError::RollbackTargetMismatch); }
    if plan.evidence_checkpoint_digest != evidence_checkpoint { violations.push(LineageBoundUpgradeHandoffError::EvidenceCheckpointMismatch); }
    violations
}
#[allow(clippy::too_many_arguments)]
fn qualify_common_governance_context(governance_view: &ContainmentCurrentWitnessRegistryHeadV1, registry_head: &QuorumObservedWitnessRegistryHeadV1, expected_checkpoint: Sha256Digest, expected_log: Sha256Digest, expected_basis: OperationalClockBasisIdV1, expected_envelope: ClockGovernanceEvaluationEnvelopeIdV1, trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState, operational_basis: &OperationalClockBasisV1, violations: &mut Vec<LineageBoundUpgradeHandoffError>) -> Result<QualifiedCommonContext, Vec<LineageBoundUpgradeHandoffError>> {
    if governance_view.checkpoint_digest() != expected_checkpoint || governance_view.transparency_log_digest() != expected_log || governance_view.observation_operational_basis_id() != expected_basis || governance_view.observation_clock_envelope_id() != expected_envelope { violations.push(LineageBoundUpgradeHandoffError::GovernanceViewMismatch); }
    if governance_view.registry_head_id() != registry_head.id() || governance_view.registry_digest() != registry_head.registry_digest() || governance_view.registry_sequence() != registry_head.sequence() { violations.push(LineageBoundUpgradeHandoffError::RegistryHeadMismatch); }
    if let Err(error) = trust_snapshot.validate() { violations.push(LineageBoundUpgradeHandoffError::TrustSnapshotInvalid(format!("{error:?}"))); }
    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) { Ok(value) => value, Err(error) => { violations.push(LineageBoundUpgradeHandoffError::TrustSnapshotInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) } };
    if trust_snapshot_digest != registry_head.trust_snapshot_digest() { violations.push(LineageBoundUpgradeHandoffError::TrustSnapshotMismatch); }
    if let Err(error) = containment_state.validate() { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateInvalid(format!("{error:?}"))); }
    let containment_state_digest = match digest_containment_state(containment_state) { Ok(value) => value, Err(error) => { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) } };
    let compromise_tracker_digest = match digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker) { Ok(value) => value, Err(error) => { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) } };
    if containment_state_digest != governance_view.containment_state_digest() { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateMismatch); }
    if compromise_tracker_digest != governance_view.compromise_tracker_digest() { violations.push(LineageBoundUpgradeHandoffError::CompromiseTrackerMismatch); }
    if containment_state.generation != governance_view.containment_generation() { violations.push(LineageBoundUpgradeHandoffError::ContainmentGenerationMismatch); }
    if operational_basis.id() != expected_basis || operational_basis.id() != governance_view.observation_operational_basis_id() { violations.push(LineageBoundUpgradeHandoffError::OperationalBasisMismatch); }
    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) { Ok(value) => value, Err(error) => { violations.push(LineageBoundUpgradeHandoffError::Clock(error)); return Err(violations.clone()); } };
    if clock.id() != expected_envelope || clock.id() != governance_view.observation_clock_envelope_id() { violations.push(LineageBoundUpgradeHandoffError::ClockEnvelopeMismatch); }
    Ok(QualifiedCommonContext { trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, clock_envelope_id: clock.id() })
}

#[allow(clippy::too_many_arguments)]
fn prepare_inner_handoff(kind: LineageBoundPredecessorAuthorityKindV1, predecessor_root_id: LineageBoundPredecessorRootRefIdV1, current_head_id: LineageBoundCurrentHeadRefIdV1, predecessor_endpoint_digest: Sha256Digest, predecessor_finalization_sequence: u64, predecessor_checkpoint_digest: Sha256Digest, predecessor_transparency_log_digest: Sha256Digest, governance_view: &ContainmentCurrentWitnessRegistryHeadV1, registry_head: &QuorumObservedWitnessRegistryHeadV1, plan: ClockGovernedUpgradeHandoffPlanV1, handoff_policy: &UpgradeHandoffPolicy, threshold_policy: &ThresholdCeremonyPolicy, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState, operational_basis: &OperationalClockBasisV1, context: QualifiedCommonContext) -> Result<PreparedLineageBoundUpgradeHandoffV1, Vec<LineageBoundUpgradeHandoffError>> {
    let inner_prepared = prepare_clock_governed_upgrade_handoff_v1(plan, handoff_policy, threshold_policy, policy_authorities, trust_snapshot, containment_state, operational_basis).map_err(|errors| vec![LineageBoundUpgradeHandoffError::InnerPreparation(errors)])?;
    let commitment = PreparedCommitment { schema: prepared_schema(kind), predecessor_root_id: predecessor_root_id.to_hex(), inner_prepared_id: inner_prepared.id().to_hex(), plan_digest: inner_prepared.plan_digest().to_hex(), current_head_id: current_head_id.to_hex(), governance_view_id: governance_view.id().to_hex(), registry_head_id: registry_head.id().to_hex(), predecessor_endpoint_digest: predecessor_endpoint_digest.to_hex(), predecessor_finalization_sequence, predecessor_checkpoint_digest: predecessor_checkpoint_digest.to_hex(), predecessor_transparency_log_digest: predecessor_transparency_log_digest.to_hex(), trust_snapshot_digest: context.trust_snapshot_digest.to_hex(), containment_state_digest: context.containment_state_digest.to_hex(), compromise_tracker_digest: context.compromise_tracker_digest.to_hex(), clock_envelope_id: context.clock_envelope_id.to_hex(), operational_basis_id: operational_basis.id().to_hex() };
    let id = PreparedLineageBoundUpgradeHandoffIdV1(hash_serializable(prepared_domain(kind), &commitment).map_err(|error| vec![error])?);
    Ok(PreparedLineageBoundUpgradeHandoffV1 { id, predecessor_authority_kind: kind, predecessor_root_id, inner_prepared, current_head_id, governance_view_id: governance_view.id(), registry_head_id: registry_head.id(), predecessor_endpoint_digest, predecessor_finalization_sequence, predecessor_checkpoint_digest, predecessor_transparency_log_digest, trust_snapshot_digest: context.trust_snapshot_digest, containment_state_digest: context.containment_state_digest, compromise_tracker_digest: context.compromise_tracker_digest, clock_envelope_id: context.clock_envelope_id, operational_basis_id: operational_basis.id() })
}

pub fn authorize_lineage_bound_upgrade_handoff_v1(prepared: PreparedLineageBoundUpgradeHandoffV1, ceremony: &ClockGovernedThresholdCeremonyV1) -> Result<LineageBoundClockGovernedUpgradeHandoffV1, LineageBoundUpgradeHandoffError> {
    let prepared_id = prepared.id; let kind = prepared.predecessor_authority_kind; let predecessor_root_id = prepared.predecessor_root_id; let current_head_id = prepared.current_head_id; let governance_view_id = prepared.governance_view_id; let registry_head_id = prepared.registry_head_id; let predecessor_endpoint_digest = prepared.predecessor_endpoint_digest; let predecessor_finalization_sequence = prepared.predecessor_finalization_sequence; let predecessor_checkpoint_digest = prepared.predecessor_checkpoint_digest; let predecessor_transparency_log_digest = prepared.predecessor_transparency_log_digest; let clock_envelope_id = prepared.clock_envelope_id; let operational_basis_id = prepared.operational_basis_id;
    let inner_handoff = authorize_clock_governed_upgrade_handoff_v1(prepared.inner_prepared, ceremony).map_err(|error| LineageBoundUpgradeHandoffError::InnerAuthorization(format!("{error:?}")))?;
    let commitment = AuthorizedCommitment { schema: authorized_schema(kind), prepared_id: prepared_id.to_hex(), predecessor_root_id: predecessor_root_id.to_hex(), inner_handoff_id: inner_handoff.id().to_hex(), plan_digest: inner_handoff.plan_digest().to_hex(), current_head_id: current_head_id.to_hex(), governance_view_id: governance_view_id.to_hex(), registry_head_id: registry_head_id.to_hex(), predecessor_endpoint_digest: predecessor_endpoint_digest.to_hex(), predecessor_finalization_sequence, predecessor_checkpoint_digest: predecessor_checkpoint_digest.to_hex(), predecessor_transparency_log_digest: predecessor_transparency_log_digest.to_hex(), threshold_ceremony_id: ceremony.id().to_hex(), clock_envelope_id: clock_envelope_id.to_hex(), operational_basis_id: operational_basis_id.to_hex() };
    let id = LineageBoundClockGovernedUpgradeHandoffIdV1(hash_serializable(authorized_domain(kind), &commitment)?);
    Ok(LineageBoundClockGovernedUpgradeHandoffV1 { id, prepared_id, predecessor_authority_kind: kind, predecessor_root_id, inner_handoff, current_head_id, governance_view_id, registry_head_id, predecessor_endpoint_digest, predecessor_finalization_sequence, predecessor_checkpoint_digest, predecessor_transparency_log_digest, threshold_ceremony_id: ceremony.id(), clock_envelope_id, operational_basis_id })
}

fn valid_lineage_native_publication(publication: &LineageNativeFinalizedHeadPublicationV1) -> bool {
    if publication.schema_version != LINEAGE_NATIVE_FINALIZED_HEAD_PUBLICATION_SCHEMA || !canonical_hex_id(&publication.finalized_upgrade_id) || !canonical_hex_id(&publication.lineage_handoff_id) || !canonical_hex_id(&publication.authorization_id) || !canonical_hex_id(&publication.execution_permit_id) || !canonical_hex_id(&publication.state_binding_id) || publication.finalization_sequence == 0 || publication.predecessor_finalization_sequence == 0 || publication.predecessor_finalization_sequence.checked_add(1) != Some(publication.finalization_sequence) || publication.predecessor_upgrade_state_generation == 0 || publication.predecessor_upgrade_state_generation.checked_add(1) != Some(publication.finalized_upgrade_state_generation) { return false; }
    for digest in [publication.record_digest, publication.predecessor_root_digest, publication.predecessor_current_head_digest, publication.handoff_plan_digest, publication.predecessor_upgrade_state_digest, publication.operational_state_digest, publication.operational_lineage_digest, publication.successor_endpoint_digest] { if digest.0 == [0; 32] { return false; } }
    true
}
fn lineage_native_log_kind(handoff_plan_digest: Sha256Digest) -> Result<String, LineageBoundUpgradeHandoffError> {
    if handoff_plan_digest.0 == [0; 32] { return Err(LineageBoundUpgradeHandoffError::NativePublicationInvalid { index: 0 }); }
    let mut hasher = Sha256::new(); hasher.update(LINEAGE_NATIVE_LOG_KIND_DOMAIN); hasher.update(&handoff_plan_digest.0); Ok(format!("{LINEAGE_NATIVE_FINALIZED_HEAD_LOG_KIND_PREFIX}{}", hasher.finalize().to_hex()))
}
fn canonical_hex_id(value: &str) -> bool { value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) }
fn hash_serializable<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<Sha256Digest, LineageBoundUpgradeHandoffError> { let bytes = serde_json::to_vec(value).map_err(|error| LineageBoundUpgradeHandoffError::Encoding(error.to_string()))?; let mut hasher = Sha256::new(); hasher.update(domain); hasher.update(&bytes); Ok(hasher.finalize()) }

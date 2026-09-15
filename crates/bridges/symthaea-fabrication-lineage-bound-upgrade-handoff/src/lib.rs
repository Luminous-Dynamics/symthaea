// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lineage-bound upgrade handoff authority rooted in authenticated finalized predecessors.
//!
//! Bootstrap deployments may still consume the older opaque global predecessor root directly.
//! Subsequent lineage-native cycles cross the crate-DAG boundary through a portable predecessor
//! certificate that is separately threshold-certified under the exact same policy, trust,
//! containment, and clock context as the future handoff preparation. Both routes converge on the
//! same opaque live handoff type; neither route accepts caller-supplied predecessor-critical facts.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_global_upgrade_head::{
    FinalizedUpgradePredecessorRootV1,
};
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::ThresholdCeremonyPolicy;
use symthaea_fabrication_kernel::trust::{TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_kernel::upgrade_handoff::{UpgradeEndpoint, UpgradeHandoffPolicy};
use symthaea_fabrication_lineage_predecessor_certificate::{
    LineageFinalizedPredecessorCertificateV1, VerifiedLineageFinalizedPredecessorCertificateIdV1,
    digest_lineage_finalized_predecessor_certificate_v1,
    verify_lineage_finalized_predecessor_certificate_v1,
};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
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
use symthaea_fabrication_witness_registry_head::{
    QuorumObservedWitnessRegistryHeadIdV1, QuorumObservedWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const PREPARED_LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA: &str =
    "symthaea.fabrication.prepared-lineage-bound-upgrade-handoff.v1";
pub const LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-upgrade-handoff.v1";
pub const PREPARED_CERTIFIED_LINEAGE_UPGRADE_HANDOFF_SCHEMA: &str =
    "symthaea.fabrication.prepared-certified-lineage-upgrade-handoff.v1";
pub const CERTIFIED_LINEAGE_UPGRADE_HANDOFF_SCHEMA: &str =
    "symthaea.fabrication.certified-lineage-upgrade-handoff.v1";

const PREPARED_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-lineage-bound-upgrade-handoff.v1\0";
const AUTHORIZED_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-bound-upgrade-handoff.v1\0";
const PREPARED_CERTIFIED_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-certified-lineage-upgrade-handoff.v1\0";
const AUTHORIZED_CERTIFIED_DOMAIN: &[u8] =
    b"symthaea.fabrication.certified-lineage-upgrade-handoff.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineageBoundPredecessorAuthorityKindV1 { BootstrapV1, CertifiedLineageV1 }

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
    predecessor_certificate_digest: Option<Sha256Digest>,
    verified_predecessor_certificate_id: Option<VerifiedLineageFinalizedPredecessorCertificateIdV1>,
    predecessor_certificate_ceremony_id: Option<ClockGovernedThresholdCeremonyIdV1>,
    predecessor_certificate_ceremony_digest: Option<Sha256Digest>,
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
    pub fn predecessor_certificate_digest(&self) -> Option<Sha256Digest> { self.predecessor_certificate_digest }
    pub fn verified_predecessor_certificate_id(&self) -> Option<VerifiedLineageFinalizedPredecessorCertificateIdV1> { self.verified_predecessor_certificate_id }
    pub fn predecessor_certificate_ceremony_id(&self) -> Option<ClockGovernedThresholdCeremonyIdV1> { self.predecessor_certificate_ceremony_id }
    pub fn predecessor_certificate_ceremony_digest(&self) -> Option<Sha256Digest> { self.predecessor_certificate_ceremony_digest }
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
    predecessor_certificate_digest: Option<Sha256Digest>,
    verified_predecessor_certificate_id: Option<VerifiedLineageFinalizedPredecessorCertificateIdV1>,
    predecessor_certificate_ceremony_id: Option<ClockGovernedThresholdCeremonyIdV1>,
    predecessor_certificate_ceremony_digest: Option<Sha256Digest>,
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
    pub fn predecessor_certificate_digest(&self) -> Option<Sha256Digest> { self.predecessor_certificate_digest }
    pub fn verified_predecessor_certificate_id(&self) -> Option<VerifiedLineageFinalizedPredecessorCertificateIdV1> { self.verified_predecessor_certificate_id }
    pub fn predecessor_certificate_ceremony_id(&self) -> Option<ClockGovernedThresholdCeremonyIdV1> { self.predecessor_certificate_ceremony_id }
    pub fn predecessor_certificate_ceremony_digest(&self) -> Option<Sha256Digest> { self.predecessor_certificate_ceremony_digest }
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
    ClockEnvelopeMismatch, PredecessorCertificate(String), MissingPredecessorCertificate,
    UnexpectedPredecessorCertificate, InnerBuild(String), InnerPreparation(Vec<ClockGovernedUpgradeHandoffError>),
    InnerAuthorization(String), Encoding(String),
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
#[derive(Debug, Clone, Serialize)]
struct CertifiedPreparedCommitment {
    schema: &'static str, predecessor_root_id: String, inner_prepared_id: String, plan_digest: String,
    current_head_id: String, governance_view_id: String, registry_head_id: String,
    predecessor_endpoint_digest: String, predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: String, predecessor_transparency_log_digest: String,
    predecessor_certificate_digest: String, verified_predecessor_certificate_id: String,
    predecessor_certificate_ceremony_id: String, predecessor_certificate_ceremony_digest: String,
    trust_snapshot_digest: String, containment_state_digest: String, compromise_tracker_digest: String,
    clock_envelope_id: String, operational_basis_id: String,
}
#[derive(Debug, Clone, Serialize)]
struct CertifiedAuthorizedCommitment {
    schema: &'static str, prepared_id: String, predecessor_root_id: String, inner_handoff_id: String,
    plan_digest: String, current_head_id: String, governance_view_id: String, registry_head_id: String,
    predecessor_endpoint_digest: String, predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: String, predecessor_transparency_log_digest: String,
    predecessor_certificate_digest: String, verified_predecessor_certificate_id: String,
    predecessor_certificate_ceremony_id: String, predecessor_certificate_ceremony_digest: String,
    threshold_ceremony_id: String, clock_envelope_id: String, operational_basis_id: String,
}

#[allow(clippy::too_many_arguments)]
pub fn build_lineage_bound_upgrade_handoff_plan_v1(
    predecessor_root: &FinalizedUpgradePredecessorRootV1, successor: UpgradeEndpoint,
    activates_at_unix_ms: u64, finalization_deadline_unix_ms: u64,
    policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], recovery_key_set_digest: Sha256Digest,
    reason: impl Into<String>,
) -> Result<ClockGovernedUpgradeHandoffPlanV1, LineageBoundUpgradeHandoffError> {
    build_clock_governed_upgrade_handoff_plan_v1(
        predecessor_root.endpoint().clone(), successor, activates_at_unix_ms, finalization_deadline_unix_ms,
        predecessor_root.rollback_target_digest(), policy_authorities, predecessor_root.evidence_checkpoint_digest(),
        recovery_key_set_digest, reason,
    ).map_err(|error| LineageBoundUpgradeHandoffError::InnerBuild(format!("{error:?}")))
}

#[allow(clippy::too_many_arguments)]
pub fn build_certified_lineage_upgrade_handoff_plan_v1(
    certificate: &LineageFinalizedPredecessorCertificateV1, successor: UpgradeEndpoint,
    activates_at_unix_ms: u64, finalization_deadline_unix_ms: u64,
    policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>], recovery_key_set_digest: Sha256Digest,
    reason: impl Into<String>,
) -> Result<ClockGovernedUpgradeHandoffPlanV1, LineageBoundUpgradeHandoffError> {
    digest_lineage_finalized_predecessor_certificate_v1(certificate)
        .map_err(|error| LineageBoundUpgradeHandoffError::PredecessorCertificate(format!("{error:?}")))?;
    build_clock_governed_upgrade_handoff_plan_v1(
        certificate.endpoint.clone(), successor, activates_at_unix_ms, finalization_deadline_unix_ms,
        certificate.rollback_target_digest, policy_authorities, certificate.evidence_checkpoint_digest,
        recovery_key_set_digest, reason,
    ).map_err(|error| LineageBoundUpgradeHandoffError::InnerBuild(format!("{error:?}")))
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_lineage_bound_upgrade_handoff_v1(
    predecessor_root: &FinalizedUpgradePredecessorRootV1, current_head: &CurrentFinalizedUpgradeHeadV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1, registry_head: &QuorumObservedWitnessRegistryHeadV1,
    plan: ClockGovernedUpgradeHandoffPlanV1, handoff_policy: &UpgradeHandoffPolicy,
    threshold_policy: &ThresholdCeremonyPolicy, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>],
    trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedLineageBoundUpgradeHandoffV1, Vec<LineageBoundUpgradeHandoffError>> {
    let mut violations = Vec::new();
    if plan.predecessor != *predecessor_root.endpoint() { violations.push(LineageBoundUpgradeHandoffError::PredecessorMismatch); }
    if plan.rollback_target_digest != predecessor_root.rollback_target_digest() { violations.push(LineageBoundUpgradeHandoffError::RollbackTargetMismatch); }
    if plan.evidence_checkpoint_digest != predecessor_root.evidence_checkpoint_digest() { violations.push(LineageBoundUpgradeHandoffError::EvidenceCheckpointMismatch); }
    if predecessor_root.current_head_id() != current_head.id()
        || predecessor_root.evidence_checkpoint_digest() != current_head.current_checkpoint_digest()
        || predecessor_root.transparency_log_digest() != current_head.current_transparency_log_digest()
        || predecessor_root.operational_basis_id() != current_head.current_operational_basis_id()
        || predecessor_root.clock_envelope_id() != current_head.current_clock_envelope_id()
    { violations.push(LineageBoundUpgradeHandoffError::CurrentHeadMismatch); }
    if current_head.governance_view_id() != governance_view.id()
        || governance_view.checkpoint_digest() != predecessor_root.evidence_checkpoint_digest()
        || governance_view.transparency_log_digest() != predecessor_root.transparency_log_digest()
        || governance_view.observation_operational_basis_id() != predecessor_root.operational_basis_id()
        || governance_view.observation_clock_envelope_id() != predecessor_root.clock_envelope_id()
    { violations.push(LineageBoundUpgradeHandoffError::GovernanceViewMismatch); }
    let (trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, clock) =
        validate_common_governance_context(registry_head, governance_view, trust_snapshot, containment_state, operational_basis, &mut violations)?;
    if operational_basis.id() != predecessor_root.operational_basis_id() || operational_basis.id() != current_head.current_operational_basis_id() {
        violations.push(LineageBoundUpgradeHandoffError::OperationalBasisMismatch);
    }
    if clock.id() != predecessor_root.clock_envelope_id() || clock.id() != current_head.current_clock_envelope_id() {
        violations.push(LineageBoundUpgradeHandoffError::ClockEnvelopeMismatch);
    }
    if !violations.is_empty() { return Err(violations); }
    let inner_prepared = prepare_clock_governed_upgrade_handoff_v1(
        plan, handoff_policy, threshold_policy, policy_authorities, trust_snapshot, containment_state, operational_basis,
    ).map_err(|errors| vec![LineageBoundUpgradeHandoffError::InnerPreparation(errors)])?;
    let commitment = PreparedCommitment {
        schema: PREPARED_LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA, predecessor_root_id: predecessor_root.id().to_hex(),
        inner_prepared_id: inner_prepared.id().to_hex(), plan_digest: inner_prepared.plan_digest().to_hex(),
        current_head_id: current_head.id().to_hex(), governance_view_id: governance_view.id().to_hex(), registry_head_id: registry_head.id().to_hex(),
        predecessor_endpoint_digest: predecessor_root.endpoint_digest().to_hex(), predecessor_finalization_sequence: predecessor_root.finalization_sequence(),
        predecessor_checkpoint_digest: predecessor_root.evidence_checkpoint_digest().to_hex(), predecessor_transparency_log_digest: predecessor_root.transparency_log_digest().to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(), containment_state_digest: containment_state_digest.to_hex(), compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        clock_envelope_id: clock.id().to_hex(), operational_basis_id: operational_basis.id().to_hex(),
    };
    let id = PreparedLineageBoundUpgradeHandoffIdV1(hash_serializable(PREPARED_DOMAIN, &commitment).map_err(|error| vec![error])?);
    Ok(PreparedLineageBoundUpgradeHandoffV1 {
        id, predecessor_authority_kind: LineageBoundPredecessorAuthorityKindV1::BootstrapV1,
        predecessor_root_id: LineageBoundPredecessorRootRefIdV1(predecessor_root.id().as_digest()), inner_prepared,
        current_head_id: LineageBoundCurrentHeadRefIdV1(current_head.id().as_digest()), governance_view_id: governance_view.id(), registry_head_id: registry_head.id(),
        predecessor_endpoint_digest: predecessor_root.endpoint_digest(), predecessor_finalization_sequence: predecessor_root.finalization_sequence(),
        predecessor_checkpoint_digest: predecessor_root.evidence_checkpoint_digest(), predecessor_transparency_log_digest: predecessor_root.transparency_log_digest(),
        predecessor_certificate_digest: None, verified_predecessor_certificate_id: None, predecessor_certificate_ceremony_id: None,
        predecessor_certificate_ceremony_digest: None, trust_snapshot_digest, containment_state_digest, compromise_tracker_digest,
        clock_envelope_id: clock.id(), operational_basis_id: operational_basis.id(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_certified_lineage_upgrade_handoff_v1(
    certificate: &LineageFinalizedPredecessorCertificateV1, certificate_ceremony: &ClockGovernedThresholdCeremonyV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1, registry_head: &QuorumObservedWitnessRegistryHeadV1,
    plan: ClockGovernedUpgradeHandoffPlanV1, handoff_policy: &UpgradeHandoffPolicy,
    threshold_policy: &ThresholdCeremonyPolicy, policy_authorities: &[UpgradePolicyAuthorityInputV1<'_>],
    trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedLineageBoundUpgradeHandoffV1, Vec<LineageBoundUpgradeHandoffError>> {
    let mut violations = Vec::new();
    let certificate_digest = match digest_lineage_finalized_predecessor_certificate_v1(certificate) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::PredecessorCertificate(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    if plan.predecessor != certificate.endpoint { violations.push(LineageBoundUpgradeHandoffError::PredecessorMismatch); }
    if plan.rollback_target_digest != certificate.rollback_target_digest { violations.push(LineageBoundUpgradeHandoffError::RollbackTargetMismatch); }
    if plan.evidence_checkpoint_digest != certificate.evidence_checkpoint_digest { violations.push(LineageBoundUpgradeHandoffError::EvidenceCheckpointMismatch); }
    if governance_view.id().to_hex() != certificate.governance_view_id
        || registry_head.id().to_hex() != certificate.registry_head_id
        || registry_head.registry_digest() != certificate.registry_digest
        || registry_head.sequence() != certificate.registry_sequence
        || registry_head.trust_snapshot_digest() != certificate.trust_snapshot_digest
        || governance_view.checkpoint_digest() != certificate.evidence_checkpoint_digest
        || governance_view.transparency_log_digest() != certificate.transparency_log_digest
        || governance_view.containment_state_digest() != certificate.containment_state_digest
        || governance_view.compromise_tracker_digest() != certificate.compromise_tracker_digest
        || governance_view.containment_generation() != certificate.containment_generation
        || governance_view.observation_operational_basis_id().to_hex() != certificate.operational_basis_id
        || governance_view.observation_clock_envelope_id().to_hex() != certificate.clock_envelope_id
    { violations.push(LineageBoundUpgradeHandoffError::GovernanceViewMismatch); }
    let (trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, clock) =
        validate_common_governance_context(registry_head, governance_view, trust_snapshot, containment_state, operational_basis, &mut violations)?;
    if operational_basis.id().to_hex() != certificate.operational_basis_id { violations.push(LineageBoundUpgradeHandoffError::OperationalBasisMismatch); }
    if clock.id().to_hex() != certificate.clock_envelope_id { violations.push(LineageBoundUpgradeHandoffError::ClockEnvelopeMismatch); }
    if !violations.is_empty() { return Err(violations); }
    let inner_prepared = prepare_clock_governed_upgrade_handoff_v1(
        plan, handoff_policy, threshold_policy, policy_authorities, trust_snapshot, containment_state, operational_basis,
    ).map_err(|errors| vec![LineageBoundUpgradeHandoffError::InnerPreparation(errors)])?;
    let verified_certificate = verify_lineage_finalized_predecessor_certificate_v1(certificate, certificate_ceremony, &inner_prepared)
        .map_err(|error| vec![LineageBoundUpgradeHandoffError::PredecessorCertificate(format!("{error:?}"))])?;
    let commitment = CertifiedPreparedCommitment {
        schema: PREPARED_CERTIFIED_LINEAGE_UPGRADE_HANDOFF_SCHEMA, predecessor_root_id: certificate.predecessor_root_id.clone(),
        inner_prepared_id: inner_prepared.id().to_hex(), plan_digest: inner_prepared.plan_digest().to_hex(), current_head_id: certificate.current_head_id.clone(),
        governance_view_id: governance_view.id().to_hex(), registry_head_id: registry_head.id().to_hex(), predecessor_endpoint_digest: certificate.endpoint_digest.to_hex(),
        predecessor_finalization_sequence: certificate.finalization_sequence, predecessor_checkpoint_digest: certificate.evidence_checkpoint_digest.to_hex(),
        predecessor_transparency_log_digest: certificate.transparency_log_digest.to_hex(), predecessor_certificate_digest: certificate_digest.to_hex(),
        verified_predecessor_certificate_id: verified_certificate.id().to_hex(), predecessor_certificate_ceremony_id: verified_certificate.certificate_ceremony_id().to_hex(),
        predecessor_certificate_ceremony_digest: verified_certificate.certificate_ceremony_digest().to_hex(), trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(), compromise_tracker_digest: compromise_tracker_digest.to_hex(), clock_envelope_id: clock.id().to_hex(),
        operational_basis_id: operational_basis.id().to_hex(),
    };
    let id = PreparedLineageBoundUpgradeHandoffIdV1(hash_serializable(PREPARED_CERTIFIED_DOMAIN, &commitment).map_err(|error| vec![error])?);
    Ok(PreparedLineageBoundUpgradeHandoffV1 {
        id, predecessor_authority_kind: LineageBoundPredecessorAuthorityKindV1::CertifiedLineageV1,
        predecessor_root_id: LineageBoundPredecessorRootRefIdV1(verified_certificate.predecessor_root_digest()), inner_prepared,
        current_head_id: LineageBoundCurrentHeadRefIdV1(verified_certificate.current_head_digest()), governance_view_id: governance_view.id(), registry_head_id: registry_head.id(),
        predecessor_endpoint_digest: certificate.endpoint_digest, predecessor_finalization_sequence: certificate.finalization_sequence,
        predecessor_checkpoint_digest: certificate.evidence_checkpoint_digest, predecessor_transparency_log_digest: certificate.transparency_log_digest,
        predecessor_certificate_digest: Some(certificate_digest), verified_predecessor_certificate_id: Some(verified_certificate.id()),
        predecessor_certificate_ceremony_id: Some(verified_certificate.certificate_ceremony_id()), predecessor_certificate_ceremony_digest: Some(verified_certificate.certificate_ceremony_digest()),
        trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, clock_envelope_id: clock.id(), operational_basis_id: operational_basis.id(),
    })
}

pub fn authorize_lineage_bound_upgrade_handoff_v1(
    prepared: PreparedLineageBoundUpgradeHandoffV1, ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<LineageBoundClockGovernedUpgradeHandoffV1, LineageBoundUpgradeHandoffError> {
    let prepared_id = prepared.id;
    let predecessor_authority_kind = prepared.predecessor_authority_kind;
    let predecessor_root_id = prepared.predecessor_root_id;
    let current_head_id = prepared.current_head_id;
    let governance_view_id = prepared.governance_view_id;
    let registry_head_id = prepared.registry_head_id;
    let predecessor_endpoint_digest = prepared.predecessor_endpoint_digest;
    let predecessor_finalization_sequence = prepared.predecessor_finalization_sequence;
    let predecessor_checkpoint_digest = prepared.predecessor_checkpoint_digest;
    let predecessor_transparency_log_digest = prepared.predecessor_transparency_log_digest;
    let predecessor_certificate_digest = prepared.predecessor_certificate_digest;
    let verified_predecessor_certificate_id = prepared.verified_predecessor_certificate_id;
    let predecessor_certificate_ceremony_id = prepared.predecessor_certificate_ceremony_id;
    let predecessor_certificate_ceremony_digest = prepared.predecessor_certificate_ceremony_digest;
    let clock_envelope_id = prepared.clock_envelope_id;
    let operational_basis_id = prepared.operational_basis_id;
    let inner_handoff = authorize_clock_governed_upgrade_handoff_v1(prepared.inner_prepared, ceremony)
        .map_err(|error| LineageBoundUpgradeHandoffError::InnerAuthorization(format!("{error:?}")))?;
    let id_digest = match predecessor_authority_kind {
        LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => {
            if predecessor_certificate_digest.is_some() || verified_predecessor_certificate_id.is_some()
                || predecessor_certificate_ceremony_id.is_some() || predecessor_certificate_ceremony_digest.is_some()
            { return Err(LineageBoundUpgradeHandoffError::UnexpectedPredecessorCertificate); }
            let commitment = AuthorizedCommitment {
                schema: LINEAGE_BOUND_UPGRADE_HANDOFF_SCHEMA, prepared_id: prepared_id.to_hex(), predecessor_root_id: predecessor_root_id.to_hex(),
                inner_handoff_id: inner_handoff.id().to_hex(), plan_digest: inner_handoff.plan_digest().to_hex(), current_head_id: current_head_id.to_hex(),
                governance_view_id: governance_view_id.to_hex(), registry_head_id: registry_head_id.to_hex(), predecessor_endpoint_digest: predecessor_endpoint_digest.to_hex(),
                predecessor_finalization_sequence, predecessor_checkpoint_digest: predecessor_checkpoint_digest.to_hex(), predecessor_transparency_log_digest: predecessor_transparency_log_digest.to_hex(),
                threshold_ceremony_id: ceremony.id().to_hex(), clock_envelope_id: clock_envelope_id.to_hex(), operational_basis_id: operational_basis_id.to_hex(),
            };
            hash_serializable(AUTHORIZED_DOMAIN, &commitment)?
        }
        LineageBoundPredecessorAuthorityKindV1::CertifiedLineageV1 => {
            let certificate_digest = predecessor_certificate_digest.ok_or(LineageBoundUpgradeHandoffError::MissingPredecessorCertificate)?;
            let certificate_id = verified_predecessor_certificate_id.ok_or(LineageBoundUpgradeHandoffError::MissingPredecessorCertificate)?;
            let certificate_ceremony_id = predecessor_certificate_ceremony_id.ok_or(LineageBoundUpgradeHandoffError::MissingPredecessorCertificate)?;
            let certificate_ceremony_digest = predecessor_certificate_ceremony_digest.ok_or(LineageBoundUpgradeHandoffError::MissingPredecessorCertificate)?;
            let commitment = CertifiedAuthorizedCommitment {
                schema: CERTIFIED_LINEAGE_UPGRADE_HANDOFF_SCHEMA, prepared_id: prepared_id.to_hex(), predecessor_root_id: predecessor_root_id.to_hex(),
                inner_handoff_id: inner_handoff.id().to_hex(), plan_digest: inner_handoff.plan_digest().to_hex(), current_head_id: current_head_id.to_hex(),
                governance_view_id: governance_view_id.to_hex(), registry_head_id: registry_head_id.to_hex(), predecessor_endpoint_digest: predecessor_endpoint_digest.to_hex(),
                predecessor_finalization_sequence, predecessor_checkpoint_digest: predecessor_checkpoint_digest.to_hex(), predecessor_transparency_log_digest: predecessor_transparency_log_digest.to_hex(),
                predecessor_certificate_digest: certificate_digest.to_hex(), verified_predecessor_certificate_id: certificate_id.to_hex(),
                predecessor_certificate_ceremony_id: certificate_ceremony_id.to_hex(), predecessor_certificate_ceremony_digest: certificate_ceremony_digest.to_hex(),
                threshold_ceremony_id: ceremony.id().to_hex(), clock_envelope_id: clock_envelope_id.to_hex(), operational_basis_id: operational_basis_id.to_hex(),
            };
            hash_serializable(AUTHORIZED_CERTIFIED_DOMAIN, &commitment)?
        }
    };
    Ok(LineageBoundClockGovernedUpgradeHandoffV1 {
        id: LineageBoundClockGovernedUpgradeHandoffIdV1(id_digest), prepared_id, predecessor_authority_kind,
        predecessor_root_id, inner_handoff, current_head_id, governance_view_id, registry_head_id,
        predecessor_endpoint_digest, predecessor_finalization_sequence, predecessor_checkpoint_digest,
        predecessor_transparency_log_digest, predecessor_certificate_digest, verified_predecessor_certificate_id,
        predecessor_certificate_ceremony_id, predecessor_certificate_ceremony_digest, threshold_ceremony_id: ceremony.id(),
        clock_envelope_id, operational_basis_id,
    })
}

fn validate_common_governance_context(
    registry_head: &QuorumObservedWitnessRegistryHeadV1, governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    trust_snapshot: &TrustSnapshot, containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1, violations: &mut Vec<LineageBoundUpgradeHandoffError>,
) -> Result<(Sha256Digest, Sha256Digest, Sha256Digest, symthaea_trust_kernel::ClockGovernanceEvaluationEnvelopeV1), Vec<LineageBoundUpgradeHandoffError>> {
    if governance_view.registry_head_id() != registry_head.id() || governance_view.registry_digest() != registry_head.registry_digest()
        || governance_view.registry_sequence() != registry_head.sequence()
    { violations.push(LineageBoundUpgradeHandoffError::RegistryHeadMismatch); }
    if let Err(error) = trust_snapshot.validate() { violations.push(LineageBoundUpgradeHandoffError::TrustSnapshotInvalid(format!("{error:?}"))); }
    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::TrustSnapshotInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    if trust_snapshot_digest != registry_head.trust_snapshot_digest() { violations.push(LineageBoundUpgradeHandoffError::TrustSnapshotMismatch); }
    if let Err(error) = containment_state.validate() { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateInvalid(format!("{error:?}"))); }
    let containment_state_digest = match digest_containment_state(containment_state) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    let compromise_tracker_digest = match digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateInvalid(format!("{error:?}"))); Sha256Digest([0; 32]) }
    };
    if containment_state_digest != governance_view.containment_state_digest() { violations.push(LineageBoundUpgradeHandoffError::ContainmentStateMismatch); }
    if compromise_tracker_digest != governance_view.compromise_tracker_digest() { violations.push(LineageBoundUpgradeHandoffError::CompromiseTrackerMismatch); }
    if containment_state.generation != governance_view.containment_generation() { violations.push(LineageBoundUpgradeHandoffError::ContainmentGenerationMismatch); }
    if operational_basis.id() != governance_view.observation_operational_basis_id() { violations.push(LineageBoundUpgradeHandoffError::OperationalBasisMismatch); }
    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => { violations.push(LineageBoundUpgradeHandoffError::Clock(error)); return Err(violations.clone()); }
    };
    if clock.id() != governance_view.observation_clock_envelope_id() { violations.push(LineageBoundUpgradeHandoffError::ClockEnvelopeMismatch); }
    Ok((trust_snapshot_digest, containment_state_digest, compromise_tracker_digest, clock))
}

fn hash_serializable<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<Sha256Digest, LineageBoundUpgradeHandoffError> {
    let bytes = serde_json::to_vec(value).map_err(|error| LineageBoundUpgradeHandoffError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

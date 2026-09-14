// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-view currentness for interval-safe evidence-retention authority.
//!
//! The retention authority itself proves one policy was valid, effective and threshold-authorized,
//! but not that no later sequence exists. This bridge requires that authority to have been created
//! under the exact trust snapshot and highest opaque containment state already established by the
//! composite registry/containment view, then publishes its sequence into that same transparency log.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_containment_state_authority::{
    ClockGovernedContainmentStateIdV1, ClockGovernedContainmentStateV1,
};
use symthaea_fabrication_evidence_retention_authority::{
    CLOCK_GOVERNED_EVIDENCE_RETENTION_PURPOSE, ClockGovernedEvidenceRetentionPolicyIdV1,
    ClockGovernedEvidenceRetentionPolicyV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_kernel::trust::{TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_trust_bridge::ClockGovernedThresholdCeremonyV1;
use symthaea_fabrication_witness_registry_containment_bound::{
    ContainmentCurrentWitnessRegistryHeadIdV1, ContainmentCurrentWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const EVIDENCE_RETENTION_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.evidence-retention-head-publication.v1";
pub const CURRENT_EVIDENCE_RETENTION_HEAD_SCHEMA: &str =
    "symthaea.fabrication.current-evidence-retention-head.v1";
pub const EVIDENCE_RETENTION_HEAD_LOG_KIND_PREFIX: &str = "evidence-retention-head-v1:";
pub const MAX_EVIDENCE_RETENTION_HEAD_CLOCK_HOPS: usize = 4096;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.evidence-retention-head-publication.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.evidence-retention-head-clock-lineage.v1\0";
const CURRENT_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.current-evidence-retention-head.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionHeadPublicationV1 {
    pub schema_version: String,
    pub retention_authority_id: String,
    pub policy_digest: Sha256Digest,
    pub sequence: u64,
    pub effective_at_unix_s: u64,
    pub prepared_id: String,
    pub threshold_ceremony_id: String,
    pub threshold_ceremony_digest: Sha256Digest,
    pub threshold_policy_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub containment_state_digest: Sha256Digest,
    pub compromise_tracker_digest: Sha256Digest,
    pub containment_generation: u64,
    pub authorization_clock_envelope_id: String,
    pub authorization_operational_basis_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentEvidenceRetentionHeadIdV1(Sha256Digest);

impl CurrentEvidenceRetentionHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct CurrentEvidenceRetentionHeadV1 {
    id: CurrentEvidenceRetentionHeadIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    retention_authority_id: ClockGovernedEvidenceRetentionPolicyIdV1,
    policy_digest: Sha256Digest,
    sequence: u64,
    effective_at_unix_s: u64,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    threshold_ceremony_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_authority_id: ClockGovernedContainmentStateIdV1,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
}

impl CurrentEvidenceRetentionHeadV1 {
    pub fn id(&self) -> CurrentEvidenceRetentionHeadIdV1 {
        self.id
    }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 {
        self.governance_view_id
    }
    pub fn retention_authority_id(&self) -> ClockGovernedEvidenceRetentionPolicyIdV1 {
        self.retention_authority_id
    }
    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn containment_authority_id(&self) -> ClockGovernedContainmentStateIdV1 {
        self.containment_authority_id
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.observation_clock_envelope_id
    }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.observation_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceRetentionHeadError {
    InvalidPublication,
    PublicationMismatch,
    GovernanceContainmentMismatch,
    RetentionContainmentMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotMismatch,
    CeremonyMismatch,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    CeremonyPolicyMismatch,
    CeremonyTrustMismatch,
    CeremonyCompromiseMismatch,
    CeremonyClockMismatch,
    AuthorizationBasisMismatch,
    AuthorizationEnvelopeMismatch,
    ObservationBasisMismatch,
    ObservationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    PolicyEffectiveTimeOverflow,
    PolicyMayNotBeEffective,
    TransparencyLogInvalid(String),
    TransparencyLogMismatch,
    MalformedRetentionHeadKind(String),
    RetentionSequenceRegressed { previous: u64, current: u64 },
    DuplicateRetentionSequence(u64),
    PublicationNotFound,
    HigherRetentionSequencePublished { candidate: u64, latest: u64 },
    PublicationDigestMismatch,
    PublicationBeforeAuthorization,
    PublicationMayBeFuture,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct CurrentRetentionCommitment {
    schema: &'static str,
    governance_view_id: String,
    retention_authority_id: String,
    policy_digest: String,
    sequence: u64,
    effective_at_unix_s: u64,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    threshold_ceremony_digest: String,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_authority_id: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    authorization_clock_envelope_id: String,
    authorization_operational_basis_id: String,
    observation_clock_envelope_id: String,
    observation_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
}

pub fn build_evidence_retention_head_publication_v1(
    retention: &ClockGovernedEvidenceRetentionPolicyV1,
) -> Result<EvidenceRetentionHeadPublicationV1, EvidenceRetentionHeadError> {
    let publication = EvidenceRetentionHeadPublicationV1 {
        schema_version: EVIDENCE_RETENTION_HEAD_PUBLICATION_SCHEMA.into(),
        retention_authority_id: retention.id().to_hex(),
        policy_digest: retention.policy_digest(),
        sequence: retention.policy().sequence,
        effective_at_unix_s: retention.policy().effective_at_unix_s,
        prepared_id: retention.prepared_id().to_hex(),
        threshold_ceremony_id: retention.threshold_ceremony_id().to_hex(),
        threshold_ceremony_digest: retention.threshold_ceremony_digest(),
        threshold_policy_digest: retention.threshold_policy_digest(),
        trust_snapshot_digest: retention.trust_snapshot_digest(),
        containment_state_digest: retention.containment_state_digest(),
        compromise_tracker_digest: retention.compromise_tracker_digest(),
        containment_generation: retention.containment_generation(),
        authorization_clock_envelope_id: retention.clock_envelope_id().to_hex(),
        authorization_operational_basis_id: retention.operational_basis_id().to_hex(),
    };
    validate_publication(&publication)?;
    Ok(publication)
}

pub fn digest_evidence_retention_head_publication_v1(
    publication: &EvidenceRetentionHeadPublicationV1,
) -> Result<Sha256Digest, EvidenceRetentionHeadError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn evidence_retention_head_log_kind(sequence: u64) -> Result<String, EvidenceRetentionHeadError> {
    if sequence == 0 {
        return Err(EvidenceRetentionHeadError::InvalidPublication);
    }
    Ok(format!("{EVIDENCE_RETENTION_HEAD_LOG_KIND_PREFIX}{sequence}"))
}

#[allow(clippy::too_many_arguments)]
pub fn bind_current_evidence_retention_head_v1(
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    retention: &ClockGovernedEvidenceRetentionPolicyV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
    current_containment_authority: &ClockGovernedContainmentStateV1,
    trust_snapshot: &TrustSnapshot,
    authorization_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    publication: &EvidenceRetentionHeadPublicationV1,
    log: &TransparencyLog,
) -> Result<CurrentEvidenceRetentionHeadV1, Vec<EvidenceRetentionHeadError>> {
    let mut violations = Vec::new();

    if current_containment_authority.id() != governance_view.containment_authority_id()
        || current_containment_authority.state_digest() != governance_view.containment_state_digest()
        || current_containment_authority.generation() != governance_view.containment_generation()
        || current_containment_authority.compromise_tracker_digest()
            != governance_view.compromise_tracker_digest()
    {
        violations.push(EvidenceRetentionHeadError::GovernanceContainmentMismatch);
    }
    if retention.containment_state_digest() != current_containment_authority.state_digest()
        || retention.containment_generation() != current_containment_authority.generation()
        || retention.compromise_tracker_digest()
            != current_containment_authority.compromise_tracker_digest()
    {
        violations.push(EvidenceRetentionHeadError::RetentionContainmentMismatch);
    }

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(EvidenceRetentionHeadError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(EvidenceRetentionHeadError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if trust_snapshot_digest != retention.trust_snapshot_digest()
        || trust_snapshot_digest != governance_view.trust_snapshot_digest()
    {
        violations.push(EvidenceRetentionHeadError::TrustSnapshotMismatch);
    }

    if ceremony.id() != retention.threshold_ceremony_id()
        || ceremony.ceremony_digest() != retention.threshold_ceremony_digest()
    {
        violations.push(EvidenceRetentionHeadError::CeremonyMismatch);
    }
    if ceremony.purpose() != CLOCK_GOVERNED_EVIDENCE_RETENTION_PURPOSE {
        violations.push(EvidenceRetentionHeadError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != retention.prepared_id().as_digest() {
        violations.push(EvidenceRetentionHeadError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != retention.threshold_policy_digest() {
        violations.push(EvidenceRetentionHeadError::CeremonyPolicyMismatch);
    }
    if ceremony.trust_snapshot_digest() != retention.trust_snapshot_digest() {
        violations.push(EvidenceRetentionHeadError::CeremonyTrustMismatch);
    }
    if ceremony.compromise_tracker_digest() != retention.compromise_tracker_digest() {
        violations.push(EvidenceRetentionHeadError::CeremonyCompromiseMismatch);
    }
    if ceremony.clock_envelope_id() != retention.clock_envelope_id() {
        violations.push(EvidenceRetentionHeadError::CeremonyClockMismatch);
    }

    if authorization_basis.id() != retention.operational_basis_id() {
        violations.push(EvidenceRetentionHeadError::AuthorizationBasisMismatch);
    }
    let authorization_clock = match derive_clock_governance_evaluation_envelope_v1(authorization_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(EvidenceRetentionHeadError::Clock(error));
            return Err(violations);
        }
    };
    if authorization_clock.id() != retention.clock_envelope_id() {
        violations.push(EvidenceRetentionHeadError::AuthorizationEnvelopeMismatch);
    }
    if observation_basis.id() != governance_view.observation_operational_basis_id() {
        violations.push(EvidenceRetentionHeadError::ObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(EvidenceRetentionHeadError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != governance_view.observation_clock_envelope_id() {
        violations.push(EvidenceRetentionHeadError::ObservationEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_EVIDENCE_RETENTION_HEAD_CLOCK_HOPS {
        violations.push(EvidenceRetentionHeadError::TooManyClockHops {
            actual: clock_bridge.len(),
            maximum: MAX_EVIDENCE_RETENTION_HEAD_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        authorization_basis.id(),
        clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }

    let effective_ms = match retention.policy().effective_at_unix_s.checked_mul(1_000) {
        Some(value) => value,
        None => {
            violations.push(EvidenceRetentionHeadError::PolicyEffectiveTimeOverflow);
            0
        }
    };
    if effective_ms > observation_clock.lower_unix_ms() {
        violations.push(EvidenceRetentionHeadError::PolicyMayNotBeEffective);
    }

    let expected_publication = match build_evidence_retention_head_publication_v1(retention) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if publication != &expected_publication {
        violations.push(EvidenceRetentionHeadError::PublicationMismatch);
    }
    let publication_digest = match digest_evidence_retention_head_publication_v1(publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    let transparency_log_digest = match digest_transparency_log(log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(EvidenceRetentionHeadError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if transparency_log_digest != governance_view.transparency_log_digest() {
        violations.push(EvidenceRetentionHeadError::TransparencyLogMismatch);
    }
    let (latest_sequence, publication_entry) = match inspect_retention_log(
        log,
        retention.policy().sequence,
    ) {
        Ok(value) => value,
        Err(errors) => {
            violations.extend(errors);
            (None, None)
        }
    };
    if let Some(latest) = latest_sequence {
        if latest > retention.policy().sequence {
            violations.push(EvidenceRetentionHeadError::HigherRetentionSequencePublished {
                candidate: retention.policy().sequence,
                latest,
            });
        }
    }
    let Some(publication_entry) = publication_entry else {
        violations.push(EvidenceRetentionHeadError::PublicationNotFound);
        return Err(violations);
    };
    if publication_entry.1 != publication_digest {
        violations.push(EvidenceRetentionHeadError::PublicationDigestMismatch);
    }
    let publication_recorded_at_ms = match seconds_to_millis(publication_entry.2) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if publication_recorded_at_ms < authorization_clock.upper_unix_ms() {
        violations.push(EvidenceRetentionHeadError::PublicationBeforeAuthorization);
    }
    if publication_recorded_at_ms > observation_clock.lower_unix_ms() {
        violations.push(EvidenceRetentionHeadError::PublicationMayBeFuture);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let (clock_lineage_digest, clock_hop_count) = digest_clock_lineage(
        authorization_basis,
        clock_bridge,
        observation_basis,
    )
    .map_err(|error| vec![error])?;
    let commitment = CurrentRetentionCommitment {
        schema: CURRENT_EVIDENCE_RETENTION_HEAD_SCHEMA,
        governance_view_id: governance_view.id().to_hex(),
        retention_authority_id: retention.id().to_hex(),
        policy_digest: retention.policy_digest().to_hex(),
        sequence: retention.policy().sequence,
        effective_at_unix_s: retention.policy().effective_at_unix_s,
        publication_digest: publication_digest.to_hex(),
        publication_entry_sequence: publication_entry.0,
        transparency_log_digest: transparency_log_digest.to_hex(),
        threshold_ceremony_digest: retention.threshold_ceremony_digest().to_hex(),
        threshold_policy_digest: retention.threshold_policy_digest().to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        containment_authority_id: current_containment_authority.id().to_hex(),
        containment_state_digest: current_containment_authority.state_digest().to_hex(),
        compromise_tracker_digest: current_containment_authority.compromise_tracker_digest().to_hex(),
        containment_generation: current_containment_authority.generation(),
        authorization_clock_envelope_id: authorization_clock.id().to_hex(),
        authorization_operational_basis_id: authorization_basis.id().to_hex(),
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
    };
    let id = CurrentEvidenceRetentionHeadIdV1(
        hash_serializable(CURRENT_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(CurrentEvidenceRetentionHeadV1 {
        id,
        governance_view_id: governance_view.id(),
        retention_authority_id: retention.id(),
        policy_digest: retention.policy_digest(),
        sequence: retention.policy().sequence,
        effective_at_unix_s: retention.policy().effective_at_unix_s,
        publication_digest,
        publication_entry_sequence: publication_entry.0,
        transparency_log_digest,
        threshold_ceremony_digest: retention.threshold_ceremony_digest(),
        threshold_policy_digest: retention.threshold_policy_digest(),
        trust_snapshot_digest,
        containment_authority_id: current_containment_authority.id(),
        containment_state_digest: current_containment_authority.state_digest(),
        compromise_tracker_digest: current_containment_authority.compromise_tracker_digest(),
        containment_generation: current_containment_authority.generation(),
        authorization_clock_envelope_id: authorization_clock.id(),
        authorization_operational_basis_id: authorization_basis.id(),
        observation_clock_envelope_id: observation_clock.id(),
        observation_operational_basis_id: observation_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
    })
}

fn validate_publication(
    publication: &EvidenceRetentionHeadPublicationV1,
) -> Result<(), EvidenceRetentionHeadError> {
    if publication.schema_version != EVIDENCE_RETENTION_HEAD_PUBLICATION_SCHEMA
        || publication.retention_authority_id.len() != 64
        || publication.policy_digest == Sha256Digest([0; 32])
        || publication.sequence == 0
        || publication.prepared_id.len() != 64
        || publication.threshold_ceremony_id.len() != 64
        || publication.threshold_ceremony_digest == Sha256Digest([0; 32])
        || publication.threshold_policy_digest == Sha256Digest([0; 32])
        || publication.trust_snapshot_digest == Sha256Digest([0; 32])
        || publication.containment_state_digest == Sha256Digest([0; 32])
        || publication.compromise_tracker_digest == Sha256Digest([0; 32])
        || publication.containment_generation == 0
        || publication.authorization_clock_envelope_id.len() != 64
        || publication.authorization_operational_basis_id.len() != 64
    {
        return Err(EvidenceRetentionHeadError::InvalidPublication);
    }
    Ok(())
}

fn inspect_retention_log(
    log: &TransparencyLog,
    candidate_sequence: u64,
) -> Result<
    (Option<u64>, Option<(u64, Sha256Digest, u64)>),
    Vec<EvidenceRetentionHeadError>,
> {
    let mut violations = Vec::new();
    let mut previous_sequence = None;
    let mut latest_sequence = None;
    let mut candidate_entry = None;
    for entry in &log.entries {
        let sequence = match parse_retention_sequence(&entry.kind) {
            Ok(Some(value)) => value,
            Ok(None) => continue,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if let Some(previous) = previous_sequence {
            if sequence <= previous {
                if sequence == previous {
                    violations.push(EvidenceRetentionHeadError::DuplicateRetentionSequence(sequence));
                } else {
                    violations.push(EvidenceRetentionHeadError::RetentionSequenceRegressed {
                        previous,
                        current: sequence,
                    });
                }
            }
        }
        previous_sequence = Some(sequence);
        latest_sequence = Some(latest_sequence.map_or(sequence, |latest: u64| latest.max(sequence)));
        if sequence == candidate_sequence {
            if candidate_entry.is_some() {
                violations.push(EvidenceRetentionHeadError::DuplicateRetentionSequence(sequence));
            } else {
                candidate_entry = Some((
                    entry.sequence,
                    entry.subject_digest,
                    entry.recorded_at_unix_s,
                ));
            }
        }
    }
    if violations.is_empty() {
        Ok((latest_sequence, candidate_entry))
    } else {
        Err(violations)
    }
}

fn parse_retention_sequence(kind: &str) -> Result<Option<u64>, EvidenceRetentionHeadError> {
    let Some(suffix) = kind.strip_prefix(EVIDENCE_RETENTION_HEAD_LOG_KIND_PREFIX) else {
        return Ok(None);
    };
    let sequence = suffix.parse::<u64>().map_err(|_| {
        EvidenceRetentionHeadError::MalformedRetentionHeadKind(kind.to_string())
    })?;
    if sequence == 0 || suffix != sequence.to_string() {
        return Err(EvidenceRetentionHeadError::MalformedRetentionHeadKind(
            kind.to_string(),
        ));
    }
    Ok(Some(sequence))
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), EvidenceRetentionHeadError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(EvidenceRetentionHeadError::BrokenClockLineage {
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
            return Err(EvidenceRetentionHeadError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(EvidenceRetentionHeadError::BrokenClockLineage {
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
) -> Result<(Sha256Digest, usize), EvidenceRetentionHeadError> {
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

fn seconds_to_millis(value: u64) -> Result<u64, EvidenceRetentionHeadError> {
    value
        .checked_mul(1_000)
        .ok_or(EvidenceRetentionHeadError::PolicyEffectiveTimeOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, EvidenceRetentionHeadError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| EvidenceRetentionHeadError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::crypto_digest::sha256;

    #[test]
    fn retention_log_rejects_sequence_regression() {
        let mut log = TransparencyLog::default();
        log.append(100, evidence_retention_head_log_kind(3).unwrap(), sha256(b"three"))
            .unwrap();
        log.append(101, evidence_retention_head_log_kind(2).unwrap(), sha256(b"two"))
            .unwrap();
        let errors = inspect_retention_log(&log, 3).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            EvidenceRetentionHeadError::RetentionSequenceRegressed {
                previous: 3,
                current: 2
            }
        )));
    }

    #[test]
    fn retention_log_rejects_duplicate_sequence() {
        let mut log = TransparencyLog::default();
        log.append(100, evidence_retention_head_log_kind(2).unwrap(), sha256(b"a"))
            .unwrap();
        log.append(101, evidence_retention_head_log_kind(2).unwrap(), sha256(b"b"))
            .unwrap();
        let errors = inspect_retention_log(&log, 2).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            EvidenceRetentionHeadError::DuplicateRetentionSequence(2)
        )));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governed signer-to-organization/failure-domain identities for transparency witnesses.
//!
//! The base transparency witness format cryptographically binds organization and region strings,
//! but those strings are declared by each witness. This crate adds a threshold-authorized registry
//! that binds an exact witness signing key to one exact organization and failure domain, then lets
//! a quorum-observed policy head be upgraded only when every signed witness statement matches that
//! governed profile.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::transparency_witness::{
    MAX_TRANSPARENCY_WITNESSES, SignedTransparencyWitness,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_policy_head_observation::{
    QuorumObservedPolicyHeadIdV1, QuorumObservedPolicyHeadV1,
};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const WITNESS_AUTHORITY_REGISTRY_GENESIS_PURPOSE: &str =
    "transparency-witness-authority-registry-genesis-v1";
pub const PREPARED_WITNESS_AUTHORITY_REGISTRY_GENESIS_SCHEMA: &str =
    "symthaea.fabrication.prepared-witness-authority-registry-genesis.v1";
pub const WITNESS_AUTHORITY_REGISTRY_SCHEMA: &str =
    "symthaea.fabrication.witness-authority-registry.v1";
pub const REGISTRY_BOUND_POLICY_HEAD_SCHEMA: &str =
    "symthaea.fabrication.registry-bound-policy-head.v1";
pub const MAX_WITNESS_AUTHORITY_PROFILES: usize = MAX_TRANSPARENCY_WITNESSES;

const WITNESS_REGISTRY_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry-digest.v1\0";
const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_REGISTRY_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-witness-authority-registry-genesis.v1\0";
const AUTHORIZED_REGISTRY_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-witness-set.v1\0";
const REGISTRY_BOUND_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-bound-policy-head.v1\0";

/// Portable identity profile. It is data, not authority; authority comes from the opaque registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessAuthorityProfileV1 {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub organization: String,
    pub failure_domain: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedWitnessAuthorityRegistryGenesisIdV1(Sha256Digest);

impl PreparedWitnessAuthorityRegistryGenesisIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessAuthorityRegistryIdV1(Sha256Digest);

impl WitnessAuthorityRegistryIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegistryBoundPolicyHeadIdV1(Sha256Digest);

impl RegistryBoundPolicyHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedWitnessAuthorityRegistryGenesisV1 {
    id: PreparedWitnessAuthorityRegistryGenesisIdV1,
    profiles: Vec<WitnessAuthorityProfileV1>,
    registry_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl PreparedWitnessAuthorityRegistryGenesisV1 {
    pub fn id(&self) -> PreparedWitnessAuthorityRegistryGenesisIdV1 {
        self.id
    }
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn profiles(&self) -> &[WitnessAuthorityProfileV1] {
        &self.profiles
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
    }
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

/// Opaque threshold-authorized identity registry.
#[derive(Debug, Clone)]
#[must_use]
pub struct WitnessAuthorityRegistryV1 {
    id: WitnessAuthorityRegistryIdV1,
    sequence: u64,
    profiles: Vec<WitnessAuthorityProfileV1>,
    registry_digest: Sha256Digest,
    genesis_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl WitnessAuthorityRegistryV1 {
    pub fn id(&self) -> WitnessAuthorityRegistryIdV1 {
        self.id
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn profiles(&self) -> &[WitnessAuthorityProfileV1] {
        &self.profiles
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
    }
    pub fn genesis_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.genesis_ceremony_id
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
    pub fn profile(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
    ) -> Option<&WitnessAuthorityProfileV1> {
        self.profiles
            .iter()
            .find(|profile| &profile.algorithm == algorithm && profile.key_id == key_id)
    }
}

/// Opaque upgrade of #2829's observed head: every exact signed witness is now also bound to a
/// threshold-authorized organization/failure-domain profile.
#[derive(Debug, Clone)]
#[must_use]
pub struct RegistryBoundPolicyHeadV1 {
    id: RegistryBoundPolicyHeadIdV1,
    observed_head_id: QuorumObservedPolicyHeadIdV1,
    registry_id: WitnessAuthorityRegistryIdV1,
    registry_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
}

impl RegistryBoundPolicyHeadV1 {
    pub fn id(&self) -> RegistryBoundPolicyHeadIdV1 {
        self.id
    }
    pub fn observed_head_id(&self) -> QuorumObservedPolicyHeadIdV1 {
        self.observed_head_id
    }
    pub fn registry_id(&self) -> WitnessAuthorityRegistryIdV1 {
        self.registry_id
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
    }
    pub fn witness_set_evidence_digest(&self) -> Sha256Digest {
        self.witness_set_evidence_digest
    }
    pub fn witness_count(&self) -> usize {
        self.witness_count
    }
    pub fn organization_count(&self) -> usize {
        self.organization_count
    }
    pub fn failure_domain_count(&self) -> usize {
        self.failure_domain_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessAuthorityError {
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    EmptyRegistry,
    TooManyProfiles { actual: usize, maximum: usize },
    InvalidProfile(String),
    DuplicateSigner(String),
    TrustSnapshotInvalid(String),
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    SignerCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    Clock(ClockGovernanceTimeError),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    WitnessSetMismatch,
    WitnessNotRegistered(String),
    WitnessOrganizationMismatch(String),
    WitnessFailureDomainMismatch(String),
    DuplicateObservedWitness(String),
    Encoding(String),
}

pub fn prepare_witness_authority_registry_genesis_v1(
    mut profiles: Vec<WitnessAuthorityProfileV1>,
    threshold_policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedWitnessAuthorityRegistryGenesisV1, Vec<WitnessAuthorityError>> {
    let mut violations = Vec::new();
    if !valid_threshold_policy(threshold_policy) {
        violations.push(WitnessAuthorityError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony {
        violations.push(WitnessAuthorityError::ThresholdPolicyUsageMismatch);
    }
    if profiles.is_empty() {
        violations.push(WitnessAuthorityError::EmptyRegistry);
    }
    if profiles.len() > MAX_WITNESS_AUTHORITY_PROFILES {
        violations.push(WitnessAuthorityError::TooManyProfiles {
            actual: profiles.len(),
            maximum: MAX_WITNESS_AUTHORITY_PROFILES,
        });
    }
    if let Err(error) = trust_snapshot.validate() {
        violations.push(WitnessAuthorityError::TrustSnapshotInvalid(format!("{error:?}")));
    }
    if let Err(error) = containment_state.validate() {
        violations.push(WitnessAuthorityError::ContainmentStateInvalid(format!("{error:?}")));
    }
    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessAuthorityError::Clock(error));
            return Err(violations);
        }
    };
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(WitnessAuthorityError::TrustSnapshotNotValidAcrossEnvelope(reason));
    }

    profiles.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
    });
    let mut seen = BTreeSet::new();
    for profile in &profiles {
        if !profile.algorithm.is_canonical()
            || invalid_identifier(&profile.key_id)
            || invalid_identifier(&profile.organization)
            || invalid_identifier(&profile.failure_domain)
        {
            violations.push(WitnessAuthorityError::InvalidProfile(profile.key_id.clone()));
            continue;
        }
        if !seen.insert((profile.algorithm.clone(), profile.key_id.clone())) {
            violations.push(WitnessAuthorityError::DuplicateSigner(profile.key_id.clone()));
            continue;
        }
        requalify_witness_signer(
            &profile.algorithm,
            &profile.key_id,
            trust_snapshot,
            containment_state,
            &clock,
            &mut violations,
        );
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let registry_digest = digest_witness_authority_profiles(&profiles).map_err(|error| vec![error])?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy).map_err(|error| vec![error])?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| vec![WitnessAuthorityError::TrustSnapshotInvalid(format!("{error:?}"))])?;
    let containment_state_digest = digest_containment_state(containment_state)
        .map_err(|error| vec![WitnessAuthorityError::ContainmentStateInvalid(format!("{error:?}"))])?;
    let compromise_tracker_digest = digest_signer_compromise_tracker(
        &containment_state.signer_compromise_tracker,
    )
    .map_err(|error| vec![WitnessAuthorityError::ContainmentStateInvalid(format!("{error:?}"))])?;
    let id = PreparedWitnessAuthorityRegistryGenesisIdV1(
        digest_prepared_registry(
            registry_digest,
            threshold_policy_digest,
            trust_snapshot_digest,
            containment_state_digest,
            compromise_tracker_digest,
            clock.id(),
            operational_basis.id(),
        )
        .map_err(|error| vec![error])?,
    );

    Ok(PreparedWitnessAuthorityRegistryGenesisV1 {
        id,
        profiles,
        registry_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
    })
}

pub fn authorize_witness_authority_registry_genesis_v1(
    prepared: PreparedWitnessAuthorityRegistryGenesisV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<WitnessAuthorityRegistryV1, WitnessAuthorityError> {
    if ceremony.purpose() != WITNESS_AUTHORITY_REGISTRY_GENESIS_PURPOSE {
        return Err(WitnessAuthorityError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(WitnessAuthorityError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(WitnessAuthorityError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(WitnessAuthorityError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(WitnessAuthorityError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(WitnessAuthorityError::ClockEnvelopeMismatch);
    }
    let id = WitnessAuthorityRegistryIdV1(digest_authorized_registry(
        prepared.id,
        ceremony.id(),
        ceremony.ceremony_digest(),
    )?);
    Ok(WitnessAuthorityRegistryV1 {
        id,
        sequence: 1,
        profiles: prepared.profiles,
        registry_digest: prepared.registry_digest,
        genesis_ceremony_id: ceremony.id(),
        clock_envelope_id: prepared.clock_envelope_id,
        operational_basis_id: prepared.operational_basis_id,
    })
}

pub fn bind_quorum_observed_policy_head_to_witness_registry_v1(
    observed: &QuorumObservedPolicyHeadV1,
    signed_witnesses: &[SignedTransparencyWitness],
    registry: &WitnessAuthorityRegistryV1,
) -> Result<RegistryBoundPolicyHeadV1, Vec<WitnessAuthorityError>> {
    let mut violations = Vec::new();
    let witness_set_evidence_digest = match digest_exact_witness_set(signed_witnesses) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if witness_set_evidence_digest != observed.witness_set_evidence_digest() {
        violations.push(WitnessAuthorityError::WitnessSetMismatch);
    }

    let mut seen = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut failure_domains = BTreeSet::new();
    for witness in signed_witnesses {
        let identity = (
            witness.signature.algorithm.clone(),
            witness.signature.key_id.clone(),
        );
        if !seen.insert(identity.clone()) {
            violations.push(WitnessAuthorityError::DuplicateObservedWitness(
                witness.signature.key_id.clone(),
            ));
            continue;
        }
        let Some(profile) = registry.profile(&identity.0, &identity.1) else {
            violations.push(WitnessAuthorityError::WitnessNotRegistered(identity.1));
            continue;
        };
        if witness.statement.witness_organization != profile.organization {
            violations.push(WitnessAuthorityError::WitnessOrganizationMismatch(
                profile.key_id.clone(),
            ));
        }
        if witness.statement.witness_region != profile.failure_domain {
            violations.push(WitnessAuthorityError::WitnessFailureDomainMismatch(
                profile.key_id.clone(),
            ));
        }
        organizations.insert(profile.organization.clone());
        failure_domains.insert(profile.failure_domain.clone());
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let id = RegistryBoundPolicyHeadIdV1(
        digest_registry_bound_head(
            observed.id(),
            registry.id(),
            registry.registry_digest(),
            witness_set_evidence_digest,
            signed_witnesses.len(),
            organizations.len(),
            failure_domains.len(),
        )
        .map_err(|error| vec![error])?,
    );
    Ok(RegistryBoundPolicyHeadV1 {
        id,
        observed_head_id: observed.id(),
        registry_id: registry.id(),
        registry_digest: registry.registry_digest(),
        witness_set_evidence_digest,
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
    })
}

fn requalify_witness_signer(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    clock: &symthaea_trust_kernel::ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<WitnessAuthorityError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(WitnessAuthorityError::SignerUnknown(key_id.into()));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(WitnessAuthorityError::SignerNotActive(key_id.into()));
    }
    if !record.usages.contains(&KeyUsage::TransparencyWitness) {
        violations.push(WitnessAuthorityError::SignerUsageNotAllowed(key_id.into()));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(WitnessAuthorityError::SignerNotValidAcrossEnvelope {
            key_id: key_id.into(),
            reason,
        });
    }
    for compromise in containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|compromise| {
            &compromise.signer.algorithm == algorithm
                && compromise.signer.key_id == key_id
                && compromise
                    .affected_usages
                    .contains(&KeyUsage::TransparencyWitness)
        })
    {
        match clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s) {
            Ok(()) => {}
            Err(ClockGovernanceTimeError::EventMayAlreadyBeEffective) => violations.push(
                WitnessAuthorityError::SignerCompromisedAcrossEnvelope(key_id.into()),
            ),
            Err(reason) => violations.push(WitnessAuthorityError::CompromiseTimeInvalid {
                key_id: key_id.into(),
                reason,
            }),
        }
    }
}

fn valid_threshold_policy(policy: &ThresholdCeremonyPolicy) -> bool {
    policy.minimum_distinct_signers > 0
        && policy.maximum_approvals > 0
        && policy.minimum_distinct_signers <= policy.maximum_approvals
        && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS
        && policy.required_algorithms.iter().all(SignatureAlgorithm::is_canonical)
        && policy.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|id| {
                !invalid_identifier(id) && id.len() <= MAX_THRESHOLD_KEY_ID_BYTES
            })
        })
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
}

#[derive(Serialize)]
struct ThresholdPolicyCommitment<'a> {
    minimum_distinct_signers: usize,
    maximum_approvals: usize,
    require_algorithm_diversity: bool,
    required_algorithms: &'a BTreeSet<SignatureAlgorithm>,
    allowed_key_ids: &'a Option<BTreeSet<String>>,
    key_usage: KeyUsage,
}

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, WitnessAuthorityError> {
    hash_serializable(
        THRESHOLD_POLICY_DOMAIN,
        &ThresholdPolicyCommitment {
            minimum_distinct_signers: policy.minimum_distinct_signers,
            maximum_approvals: policy.maximum_approvals,
            require_algorithm_diversity: policy.require_algorithm_diversity,
            required_algorithms: &policy.required_algorithms,
            allowed_key_ids: &policy.allowed_key_ids,
            key_usage: policy.key_usage,
        },
    )
}

fn digest_witness_authority_profiles(
    profiles: &[WitnessAuthorityProfileV1],
) -> Result<Sha256Digest, WitnessAuthorityError> {
    hash_serializable(WITNESS_REGISTRY_DOMAIN, &profiles)
}

#[derive(Serialize)]
struct PreparedRegistryCommitment {
    schema: &'static str,
    registry_digest: String,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_prepared_registry(
    registry_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
) -> Result<Sha256Digest, WitnessAuthorityError> {
    hash_serializable(
        PREPARED_REGISTRY_DOMAIN,
        &PreparedRegistryCommitment {
            schema: PREPARED_WITNESS_AUTHORITY_REGISTRY_GENESIS_SCHEMA,
            registry_digest: registry_digest.to_hex(),
            threshold_policy_digest: threshold_policy_digest.to_hex(),
            trust_snapshot_digest: trust_snapshot_digest.to_hex(),
            containment_state_digest: containment_state_digest.to_hex(),
            compromise_tracker_digest: compromise_tracker_digest.to_hex(),
            clock_envelope_id: clock_envelope_id.to_hex(),
            operational_basis_id: operational_basis_id.to_hex(),
        },
    )
}

#[derive(Serialize)]
struct AuthorizedRegistryCommitment {
    schema: &'static str,
    prepared_id: String,
    ceremony_id: String,
    ceremony_digest: String,
}

fn digest_authorized_registry(
    prepared_id: PreparedWitnessAuthorityRegistryGenesisIdV1,
    ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    ceremony_digest: Sha256Digest,
) -> Result<Sha256Digest, WitnessAuthorityError> {
    hash_serializable(
        AUTHORIZED_REGISTRY_DOMAIN,
        &AuthorizedRegistryCommitment {
            schema: WITNESS_AUTHORITY_REGISTRY_SCHEMA,
            prepared_id: prepared_id.to_hex(),
            ceremony_id: ceremony_id.to_hex(),
            ceremony_digest: ceremony_digest.to_hex(),
        },
    )
}

fn digest_exact_witness_set(
    signed_witnesses: &[SignedTransparencyWitness],
) -> Result<Sha256Digest, WitnessAuthorityError> {
    let mut evidence_digests = signed_witnesses
        .iter()
        .map(|witness| hash_serializable(SIGNED_WITNESS_EVIDENCE_DOMAIN, witness))
        .collect::<Result<Vec<_>, _>>()?;
    evidence_digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(WITNESS_SET_EVIDENCE_DOMAIN);
    hasher.update(&(evidence_digests.len() as u64).to_le_bytes());
    for digest in evidence_digests {
        hasher.update(&digest.0);
    }
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct RegistryBoundHeadCommitment {
    schema: &'static str,
    observed_head_id: String,
    registry_id: String,
    registry_digest: String,
    witness_set_evidence_digest: String,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
}

fn digest_registry_bound_head(
    observed_head_id: QuorumObservedPolicyHeadIdV1,
    registry_id: WitnessAuthorityRegistryIdV1,
    registry_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
) -> Result<Sha256Digest, WitnessAuthorityError> {
    hash_serializable(
        REGISTRY_BOUND_HEAD_DOMAIN,
        &RegistryBoundHeadCommitment {
            schema: REGISTRY_BOUND_POLICY_HEAD_SCHEMA,
            observed_head_id: observed_head_id.to_hex(),
            registry_id: registry_id.to_hex(),
            registry_digest: registry_digest.to_hex(),
            witness_set_evidence_digest: witness_set_evidence_digest.to_hex(),
            witness_count,
            organization_count,
            failure_domain_count,
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, WitnessAuthorityError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WitnessAuthorityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

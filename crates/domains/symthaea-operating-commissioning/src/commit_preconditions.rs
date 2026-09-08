// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for commissioning authorization.
//!
//! Commissioning admission proves that one canonical commissioning transition is
//! locally admissible at one observation. Detached signature verification may take
//! time. During that interval the commissioning root, profile-authority root,
//! commissioning lineage, profile lifecycle, trusted clock, or installed
//! configuration can change. This module captures those exact preconditions and
//! requires them to remain current immediately before persistence.
//!
//! This module performs no persistence and proves no signature. A production store
//! must additionally require verifier-owned cryptographic proof over
//! [`CommissioningAuthorizationCommitPreconditions::canonical_transition_bytes`]
//! under the exact captured commissioning root before atomically committing the
//! candidate commissioning head and canonical commissioning record.

use crate::admission::{
    CommissioningAuthorityRootSnapshot, CommissioningAuthorizationHead,
    PolicyCheckedCommissioningAuthorization,
};
use crate::authorization::transition::{
    CommissioningAuthorizationTransitionDigest, CommissioningAuthorizationTransitionError,
};
use crate::identity::CommissioningRecordDigest;
use crate::ConfigurationDigest;
use symthaea_safety_profile::commit_preconditions::ProfileAuthorityRootSnapshot;
use symthaea_safety_profile::lifecycle::SafetyProfileAuthorizationLifecycleState;
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Result of rechecking one atomic commissioning-store observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommissioningAuthorizationCommitState {
    /// Every captured precondition remains current. The caller may commit only
    /// after matching verifier-owned cryptographic proof as well.
    ReadyToCommit,
    /// The exact candidate transition already has the exact canonical record marker
    /// in durable history. This is historical acknowledgement only; it does not
    /// claim that the profile remains active, the configuration remains installed,
    /// or the commissioning action would still be admissible now.
    AlreadyCommitted,
}

/// Exact state that must remain true while an admitted commissioning authorization
/// is cryptographically verified and durably committed.
#[derive(Debug, Clone, PartialEq)]
pub struct CommissioningAuthorizationCommitPreconditions {
    canonical_transition_bytes: Vec<u8>,
    candidate_transition_digest: CommissioningAuthorizationTransitionDigest,
    expected_commissioning_root: CommissioningAuthorityRootSnapshot,
    expected_profile_root: ProfileAuthorityRootSnapshot,
    expected_predecessor_head: CommissioningAuthorizationHead,
    candidate_head: CommissioningAuthorizationHead,
    expected_profile_lifecycle: SafetyProfileAuthorizationLifecycleState,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    profile_valid_from_unix_ms: i64,
    profile_valid_until_unix_ms: i64,
    commissioning_valid_from_unix_ms: i64,
    commissioning_valid_until_unix_ms: i64,
    commissioning_record_digest: CommissioningRecordDigest,
    configuration_digest: ConfigurationDigest,
}

impl CommissioningAuthorizationCommitPreconditions {
    /// Capture the exact policy-checked commissioning state before detached crypto
    /// verification begins.
    ///
    /// The profile-root snapshot is supplied explicitly because the durable profile
    /// lifecycle binds root ID/digest but intentionally does not carry the local
    /// provisioning epoch. Capturing that epoch here makes same-key reprovisioning a
    /// TOCTOU-invalidating event as well.
    pub fn from_policy_checked(
        checked: &PolicyCheckedCommissioningAuthorization,
        expected_profile_root: ProfileAuthorityRootSnapshot,
    ) -> Result<Self, CommissioningAuthorizationCommitError> {
        if !checked.expected_profile_lifecycle().is_active() {
            return Err(CommissioningAuthorizationCommitError::ProfileLifecycleNotActive);
        }

        let profile_head = checked
            .expected_profile_lifecycle()
            .authorization_head()
            .identity()
            .ok_or(CommissioningAuthorizationCommitError::ProfileLifecycleHeadNotCurrent)?;
        if profile_head.authority_root_id() != expected_profile_root.root_id()
            || profile_head.authority_root_digest() != expected_profile_root.root_digest()
        {
            return Err(CommissioningAuthorizationCommitError::ProfileRootSnapshotMismatch);
        }

        let profile_subject = checked.profile_authorization().subject();
        if profile_subject.authority_root_id() != expected_profile_root.root_id()
            || profile_subject.authority_root_digest() != expected_profile_root.root_digest()
        {
            return Err(CommissioningAuthorizationCommitError::ProfileTransitionRootMismatch);
        }

        let candidate_identity = checked
            .candidate_head()
            .identity()
            .ok_or(CommissioningAuthorizationCommitError::CandidateHeadNotCurrent)?;
        let candidate_transition_digest = checked.transition().transition_digest()?;
        if candidate_identity.transition_digest() != candidate_transition_digest {
            return Err(CommissioningAuthorizationCommitError::CandidateHeadDigestMismatch);
        }
        if candidate_identity.authority_root_id() != checked.expected_root().root_id()
            || candidate_identity.authority_root_digest() != checked.expected_root().root_digest()
        {
            return Err(CommissioningAuthorizationCommitError::CandidateCommissioningRootMismatch);
        }

        let commissioning_subject = checked.transition().subject();
        Ok(Self {
            canonical_transition_bytes: checked.canonical_transition_bytes().to_vec(),
            candidate_transition_digest,
            expected_commissioning_root: checked.expected_root().clone(),
            expected_profile_root,
            expected_predecessor_head: checked.expected_predecessor_head().clone(),
            candidate_head: checked.candidate_head().clone(),
            expected_profile_lifecycle: checked.expected_profile_lifecycle().clone(),
            expected_clock_source_id: checked.expected_clock_source_id().to_owned(),
            expected_clock_epoch: checked.expected_clock_epoch(),
            profile_valid_from_unix_ms: profile_subject.valid_from_unix_ms(),
            profile_valid_until_unix_ms: profile_subject.valid_until_unix_ms(),
            commissioning_valid_from_unix_ms: commissioning_subject.valid_from_unix_ms(),
            commissioning_valid_until_unix_ms: commissioning_subject.valid_until_unix_ms(),
            commissioning_record_digest: checked.commissioning_record_digest(),
            configuration_digest: checked.configuration_digest(),
        })
    }

    pub fn canonical_transition_bytes(&self) -> &[u8] {
        &self.canonical_transition_bytes
    }

    pub fn candidate_transition_digest(&self) -> CommissioningAuthorizationTransitionDigest {
        self.candidate_transition_digest
    }

    pub fn expected_commissioning_root(&self) -> &CommissioningAuthorityRootSnapshot {
        &self.expected_commissioning_root
    }

    pub fn expected_profile_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_profile_root
    }

    pub fn expected_predecessor_head(&self) -> &CommissioningAuthorizationHead {
        &self.expected_predecessor_head
    }

    pub fn candidate_head(&self) -> &CommissioningAuthorizationHead {
        &self.candidate_head
    }

    pub fn expected_profile_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.expected_profile_lifecycle
    }

    pub fn commissioning_record_digest(&self) -> CommissioningRecordDigest {
        self.commissioning_record_digest
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    /// Recheck one atomic store observation immediately before commissioning.
    ///
    /// `candidate_record_digest_if_committed` is a durable history marker keyed by
    /// this exact candidate commissioning-transition digest. The marker must be
    /// written atomically with the candidate head + canonical commissioning record.
    /// Its presence therefore allows safe retry acknowledgement even if a later
    /// commissioning generation has since advanced the current head.
    #[allow(clippy::too_many_arguments)]
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_commissioning_root: &CommissioningAuthorityRootSnapshot,
        current_profile_root: &ProfileAuthorityRootSnapshot,
        current_commissioning_head: &CommissioningAuthorizationHead,
        current_profile_lifecycle: &SafetyProfileAuthorizationLifecycleState,
        current_configuration_digest: ConfigurationDigest,
        candidate_record_digest_if_committed: Option<CommissioningRecordDigest>,
    ) -> Result<CommissioningAuthorizationCommitState, CommissioningAuthorizationCommitError> {
        // Exact historical retry acknowledgement is intentionally checked first.
        // It does not imply current admissibility or current operational authority.
        if let Some(observed) = candidate_record_digest_if_committed {
            return if observed == self.commissioning_record_digest {
                Ok(CommissioningAuthorizationCommitState::AlreadyCommitted)
            } else {
                Err(CommissioningAuthorizationCommitError::CandidateCommitMarkerConflict {
                    expected: self.commissioning_record_digest,
                    observed,
                })
            };
        }

        // If the candidate head is current but no atomically paired record marker
        // exists, durable state is inconsistent. Never reconstruct the missing
        // commissioning identity from caller input.
        if current_commissioning_head == &self.candidate_head {
            return Err(CommissioningAuthorizationCommitError::CandidateHeadMissingRecordMarker);
        }

        if current_commissioning_root != &self.expected_commissioning_root {
            return Err(CommissioningAuthorizationCommitError::CommissioningRootSnapshotChanged);
        }
        if current_profile_root != &self.expected_profile_root {
            return Err(CommissioningAuthorizationCommitError::ProfileRootSnapshotChanged);
        }
        if current_commissioning_head != &self.expected_predecessor_head {
            return Err(CommissioningAuthorizationCommitError::CommissioningHeadChanged);
        }
        if current_profile_lifecycle != &self.expected_profile_lifecycle {
            return Err(CommissioningAuthorizationCommitError::ProfileLifecycleChanged);
        }
        if !current_profile_lifecycle.is_active() {
            return Err(CommissioningAuthorizationCommitError::ProfileLifecycleNotActive);
        }
        if current_configuration_digest != self.configuration_digest {
            return Err(CommissioningAuthorizationCommitError::ConfigurationChanged {
                expected: self.configuration_digest,
                observed: current_configuration_digest,
            });
        }
        if current_clock.source_id() != self.expected_clock_source_id
            || current_clock.epoch() != self.expected_clock_epoch
        {
            return Err(CommissioningAuthorizationCommitError::ClockLineageChanged {
                expected_source_id: self.expected_clock_source_id.clone(),
                expected_epoch: self.expected_clock_epoch,
                observed_source_id: current_clock.source_id().to_owned(),
                observed_epoch: current_clock.epoch(),
            });
        }

        validate_entire_interval(
            ValidityScope::ProfileAuthorization,
            self.profile_valid_from_unix_ms,
            self.profile_valid_until_unix_ms,
            current_clock,
        )?;
        validate_entire_interval(
            ValidityScope::CommissioningAction,
            self.commissioning_valid_from_unix_ms,
            self.commissioning_valid_until_unix_ms,
            current_clock,
        )?;

        Ok(CommissioningAuthorizationCommitState::ReadyToCommit)
    }
}

#[derive(Debug, Clone, Copy)]
enum ValidityScope {
    ProfileAuthorization,
    CommissioningAction,
}

fn validate_entire_interval(
    scope: ValidityScope,
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
    clock: &TrustedAuthorizationClockObservation,
) -> Result<(), CommissioningAuthorizationCommitError> {
    if clock.latest_unix_ms() < valid_from_unix_ms {
        return Err(match scope {
            ValidityScope::ProfileAuthorization => {
                CommissioningAuthorizationCommitError::ProfileDefinitelyNotYetValid {
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                }
            }
            ValidityScope::CommissioningAction => {
                CommissioningAuthorizationCommitError::CommissioningDefinitelyNotYetValid {
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                }
            }
        });
    }
    if clock.earliest_unix_ms() >= valid_until_unix_ms {
        return Err(match scope {
            ValidityScope::ProfileAuthorization => {
                CommissioningAuthorizationCommitError::ProfileDefinitelyExpired {
                    earliest_unix_ms: clock.earliest_unix_ms(),
                    valid_until_unix_ms,
                }
            }
            ValidityScope::CommissioningAction => {
                CommissioningAuthorizationCommitError::CommissioningDefinitelyExpired {
                    earliest_unix_ms: clock.earliest_unix_ms(),
                    valid_until_unix_ms,
                }
            }
        });
    }
    if clock.earliest_unix_ms() < valid_from_unix_ms
        || clock.latest_unix_ms() >= valid_until_unix_ms
    {
        return Err(match scope {
            ValidityScope::ProfileAuthorization => {
                CommissioningAuthorizationCommitError::ProfileUncertaintyCrossesValidityBoundary {
                    earliest_unix_ms: clock.earliest_unix_ms(),
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                    valid_until_unix_ms,
                }
            }
            ValidityScope::CommissioningAction => {
                CommissioningAuthorizationCommitError::CommissioningUncertaintyCrossesValidityBoundary {
                    earliest_unix_ms: clock.earliest_unix_ms(),
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                    valid_until_unix_ms,
                }
            }
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommissioningAuthorizationCommitError {
    #[error(transparent)]
    Transition(#[from] CommissioningAuthorizationTransitionError),
    #[error("policy-checked commissioning profile lifecycle is not active")]
    ProfileLifecycleNotActive,
    #[error("active profile lifecycle unexpectedly has no current authorization head")]
    ProfileLifecycleHeadNotCurrent,
    #[error("profile lifecycle does not belong to the captured profile-root snapshot")]
    ProfileRootSnapshotMismatch,
    #[error("supplied active profile transition does not belong to the captured profile-root snapshot")]
    ProfileTransitionRootMismatch,
    #[error("candidate commissioning head unexpectedly has no current identity")]
    CandidateHeadNotCurrent,
    #[error("candidate commissioning head digest does not match exact transition digest")]
    CandidateHeadDigestMismatch,
    #[error("candidate commissioning head does not belong to the captured commissioning root")]
    CandidateCommissioningRootMismatch,
    #[error("commissioning-authority root snapshot changed before commissioning commit")]
    CommissioningRootSnapshotChanged,
    #[error("profile-authority root snapshot changed before commissioning commit")]
    ProfileRootSnapshotChanged,
    #[error("commissioning authorization head changed before commissioning commit")]
    CommissioningHeadChanged,
    #[error("safety-profile lifecycle changed before commissioning commit")]
    ProfileLifecycleChanged,
    #[error("installed configuration changed before commissioning commit")]
    ConfigurationChanged {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
    #[error("trusted clock lineage changed before commissioning commit: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("candidate commissioning head is current without its atomically paired canonical record marker")]
    CandidateHeadMissingRecordMarker,
    #[error("a different canonical commissioning record is already marked for this exact candidate transition")]
    CandidateCommitMarkerConflict {
        expected: CommissioningRecordDigest,
        observed: CommissioningRecordDigest,
    },
    #[error("active profile authorization is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    ProfileDefinitelyNotYetValid {
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("active profile authorization is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    ProfileDefinitelyExpired {
        earliest_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses active profile validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    ProfileUncertaintyCrossesValidityBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("commissioning action is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    CommissioningDefinitelyNotYetValid {
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("commissioning action is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    CommissioningDefinitelyExpired {
        earliest_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses commissioning validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    CommissioningUncertaintyCrossesValidityBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{
        CommissioningAuthorizationAdmissionPolicy, CommissioningAuthorizationHead,
    };
    use crate::authorization::transition::CommissioningAuthorizationTransition;
    use crate::authorization::{
        CommissioningAuthorizationSubject, CommissioningAuthorityRootDigest,
    };
    use crate::{
        CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningRecord,
    };
    use std::collections::BTreeSet;
    use symthaea_operating_autonomy::LocalSafetyEnvelope;
    use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
    use symthaea_resource_hierarchy::{NodeScale, ResourceHierarchy, ResourceNode};
    use symthaea_resource_model::{
        ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit,
    };
    use symthaea_safety_configuration::{
        ConfigurationComponent, SafetyConfigurationManifest, SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::admission::SafetyProfileAuthorizationHead;
    use symthaea_safety_profile::authorization::{
        ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject,
    };
    use symthaea_safety_profile::commit_preconditions::ProfileAuthorityRootSnapshot;
    use symthaea_safety_profile::lifecycle::SafetyProfileAuthorizationLifecycleState;
    use symthaea_safety_profile::revocation::SafetyProfileAuthorizationRevocationDigest;
    use symthaea_safety_profile::transition::SafetyProfileAuthorizationTransition;
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };
    use symthaea_safety_qualification::{
        qualify_safety_configuration, QualifiedSafetyConfiguration,
    };

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn commissioning_root(byte: u8) -> CommissioningAuthorityRootDigest {
        CommissioningAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn profile_root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn power_key() -> symthaea_resource_model::ResourceKey {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack",
                    "Rack",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn local() -> LocalSafetyEnvelope {
        let mut modes = BTreeSet::new();
        modes.insert(OperatingMode::Normal);
        modes.insert(OperatingMode::Islanded);
        LocalSafetyEnvelope {
            id: "rack-survival-v1".into(),
            subject_node_id: "rack".into(),
            allowed_modes: modes,
            resource_constraints: vec![ResourceConstraint {
                key: power_key(),
                metric: BoundaryMetric::Import,
                minimum: None,
                maximum: Some(100.0),
            }],
            critical_service_floor: 0.70,
            max_shed_fraction: 0.30,
        }
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn manifest(profile: &SafetyConfigurationProfile) -> SafetyConfigurationManifest {
        SafetyConfigurationManifest {
            schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
            node_id: "rack".to_owned(),
            profile_id: profile.profile_id.clone(),
            profile_digest: profile.digest().unwrap(),
            hardware_inventory: ConfigurationComponent::Digest(digest(0x01)),
            firmware: ConfigurationComponent::Digest(digest(0x02)),
            software_closure: ConfigurationComponent::Digest(digest(0x03)),
            electrical_topology: ConfigurationComponent::Digest(digest(0x04)),
            thermal_topology: ConfigurationComponent::Digest(digest(0x05)),
            protection_settings: ConfigurationComponent::Digest(digest(0x06)),
            sensor_map: ConfigurationComponent::Digest(digest(0x07)),
            actuator_map: ConfigurationComponent::Digest(digest(0x08)),
            calibration: ConfigurationComponent::Digest(digest(0x09)),
            network_topology: ConfigurationComponent::Digest(digest(0x0a)),
        }
    }

    fn qualified() -> QualifiedSafetyConfiguration {
        let profile = profile();
        qualify_safety_configuration(&profile, manifest(&profile)).unwrap()
    }

    fn profile_transition() -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        SafetyProfileAuthorizationTransition::bootstrap(
            SafetyProfileAuthorizationSubject::new(
                "profile-auth-1",
                "profile-root-v1",
                profile_root(0x44),
                "rack",
                1,
                1_000,
                3_000,
                &profile,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn profile_lifecycle(
        transition: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationLifecycleState {
        SafetyProfileAuthorizationLifecycleState::active(
            SafetyProfileAuthorizationHead::from_transition(transition).unwrap(),
        )
        .unwrap()
    }

    fn commissioning_record() -> CommissioningRecord {
        let qualified = qualified();
        CommissioningRecord::new(
            1,
            CommissionedLocalSafetyEnvelope::new(
                local(),
                CommissioningBinding::new(
                    qualified.configuration_digest(),
                    "commissioning-evidence-1",
                )
                .unwrap(),
            ),
        )
        .unwrap()
    }

    fn commissioning_transition(
        record: &CommissioningRecord,
        profile_transition: &SafetyProfileAuthorizationTransition,
    ) -> CommissioningAuthorizationTransition {
        let hierarchy = hierarchy();
        let qualified = qualified();
        CommissioningAuthorizationTransition::bootstrap(
            CommissioningAuthorizationSubject::new(
                "commission-auth-1",
                "commission-root-v1",
                commissioning_root(0x55),
                1_200,
                2_500,
                record,
                &hierarchy,
                &qualified,
                profile_transition,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn commissioning_root_snapshot() -> CommissioningAuthorityRootSnapshot {
        CommissioningAuthorityRootSnapshot::new(
            "commission-root-v1",
            commissioning_root(0x55),
            4,
        )
        .unwrap()
    }

    fn profile_root_snapshot(epoch: u64) -> ProfileAuthorityRootSnapshot {
        ProfileAuthorityRootSnapshot::new("profile-root-v1", profile_root(0x44), epoch).unwrap()
    }

    fn clock(source: &str, epoch: u64, earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new(source, epoch, earliest, latest).unwrap()
    }

    fn checked() -> PolicyCheckedCommissioningAuthorization {
        let profile_transition = profile_transition();
        let record = commissioning_record();
        let commissioning_transition = commissioning_transition(&record, &profile_transition);
        let hierarchy = hierarchy();
        let qualified = qualified();
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            commissioning_root_snapshot(),
            CommissioningAuthorizationHead::Uninitialized,
            profile_lifecycle(&profile_transition),
        )
        .unwrap();
        policy
            .check(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                &commissioning_transition,
                &record,
                &hierarchy,
                &qualified,
                &profile_transition,
            )
            .unwrap()
    }

    fn preconditions() -> CommissioningAuthorizationCommitPreconditions {
        CommissioningAuthorizationCommitPreconditions::from_policy_checked(
            &checked(),
            profile_root_snapshot(3),
        )
        .unwrap()
    }

    #[test]
    fn exact_state_is_ready_to_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_600, 1_700),
                    &commissioning_root_snapshot(),
                    &profile_root_snapshot(3),
                    preconditions.expected_predecessor_head(),
                    preconditions.expected_profile_lifecycle(),
                    preconditions.configuration_digest(),
                    None,
                )
                .unwrap(),
            CommissioningAuthorizationCommitState::ReadyToCommit
        );
    }

    #[test]
    fn raced_profile_revocation_invalidates_commissioning() {
        let preconditions = preconditions();
        let head = preconditions
            .expected_profile_lifecycle()
            .authorization_head()
            .clone();
        let revoked = SafetyProfileAuthorizationLifecycleState::revoked(
            head,
            SafetyProfileAuthorizationRevocationDigest::Blake3_256([0x99; 32]),
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_700),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(3),
                preconditions.expected_predecessor_head(),
                &revoked,
                preconditions.configuration_digest(),
                None,
            ),
            Err(CommissioningAuthorizationCommitError::ProfileLifecycleChanged)
        );
    }

    #[test]
    fn same_key_profile_root_reprovisioning_invalidates_inflight_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_700),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(4),
                preconditions.expected_predecessor_head(),
                preconditions.expected_profile_lifecycle(),
                preconditions.configuration_digest(),
                None,
            ),
            Err(CommissioningAuthorizationCommitError::ProfileRootSnapshotChanged)
        );
    }

    #[test]
    fn installed_configuration_drift_invalidates_inflight_commit() {
        let preconditions = preconditions();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_700),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(3),
                preconditions.expected_predecessor_head(),
                preconditions.expected_profile_lifecycle(),
                digest(0xee),
                None,
            ),
            Err(CommissioningAuthorizationCommitError::ConfigurationChanged { .. })
        ));
    }

    #[test]
    fn commissioning_head_race_invalidates_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_700),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(3),
                preconditions.candidate_head(),
                preconditions.expected_profile_lifecycle(),
                preconditions.configuration_digest(),
                None,
            ),
            Err(CommissioningAuthorizationCommitError::CandidateHeadMissingRecordMarker)
        );
    }

    #[test]
    fn exact_candidate_history_marker_is_idempotent_acknowledgement() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("different-clock", 99, 9_000, 9_100),
                    &CommissioningAuthorityRootSnapshot::new(
                        "different-commission-root",
                        commissioning_root(0xaa),
                        99,
                    )
                    .unwrap(),
                    &ProfileAuthorityRootSnapshot::new(
                        "different-profile-root",
                        profile_root(0xbb),
                        99,
                    )
                    .unwrap(),
                    preconditions.candidate_head(),
                    preconditions.expected_profile_lifecycle(),
                    digest(0xee),
                    Some(preconditions.commissioning_record_digest()),
                )
                .unwrap(),
            CommissioningAuthorizationCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn conflicting_candidate_history_marker_fails_closed() {
        let preconditions = preconditions();
        let observed = CommissioningRecordDigest::Blake3_256([0xee; 32]);
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_600, 1_700),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(3),
                preconditions.expected_predecessor_head(),
                preconditions.expected_profile_lifecycle(),
                preconditions.configuration_digest(),
                Some(observed),
            ),
            Err(CommissioningAuthorizationCommitError::CandidateCommitMarkerConflict {
                expected: preconditions.commissioning_record_digest(),
                observed,
            })
        );
    }

    #[test]
    fn clock_lineage_change_invalidates_commit() {
        let preconditions = preconditions();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 8, 1_600, 1_700),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(3),
                preconditions.expected_predecessor_head(),
                preconditions.expected_profile_lifecycle(),
                preconditions.configuration_digest(),
                None,
            ),
            Err(CommissioningAuthorizationCommitError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn validity_boundary_is_rechecked_at_commit() {
        let preconditions = preconditions();
        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 2_499, 2_500),
                &commissioning_root_snapshot(),
                &profile_root_snapshot(3),
                preconditions.expected_predecessor_head(),
                preconditions.expected_profile_lifecycle(),
                preconditions.configuration_digest(),
                None,
            ),
            Err(CommissioningAuthorizationCommitError::CommissioningUncertaintyCrossesValidityBoundary { .. })
        ));
    }
}

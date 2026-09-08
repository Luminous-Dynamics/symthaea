// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed local admission for commissioning authorization.
//!
//! A canonical commissioning-authorization transition is still only evidence. This
//! module admits one transition against the exact local commissioning-authority
//! lineage, exact active safety-profile lifecycle, exact qualified configuration,
//! exact commissioning record, resource hierarchy, and uncertainty-aware trusted
//! time. Successful admission remains non-cryptographic and cannot mutate runtime
//! state.
//!
//! The complete active profile lifecycle is an input rather than only its head. A
//! profile revocation changes `Active(N)` to `Revoked(N, R)` without changing head N;
//! a future commissioning CAS must therefore compare the whole lifecycle state so a
//! revocation racing commissioning fails closed.

use crate::authorization::transition::{
    CommissioningAuthorizationPredecessor, CommissioningAuthorizationTransition,
    CommissioningAuthorizationTransitionDigest, CommissioningAuthorizationTransitionError,
};
use crate::authorization::{CommissioningAuthorityRootDigest, CommissioningAuthorizationError};
use crate::identity::{
    CommissioningIdentityError, CommissioningRecordDigest, commissioning_record_digest,
};
use crate::{CommissioningRecord, ConfigurationDigest};
use symthaea_resource_hierarchy::ResourceHierarchy;
use symthaea_safety_profile::admission::SafetyProfileAuthorizationHead;
use symthaea_safety_profile::lifecycle::SafetyProfileAuthorizationLifecycleState;
use symthaea_safety_profile::transition::{
    SafetyProfileAuthorizationTransition, SafetyProfileAuthorizationTransitionError,
};
use symthaea_safety_profile::trusted_time::TrustedAuthorizationClockObservation;
use symthaea_safety_qualification::QualifiedSafetyConfiguration;
use thiserror::Error;

/// Exact local commissioning-authority root provisioned for one node.
///
/// `epoch` changes whenever the local root provisioning is replaced or continuity
/// is lost, even if the same key bytes are provisioned again. A later commit layer
/// can therefore invalidate an in-flight authorization on root reprovisioning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommissioningAuthorityRootSnapshot {
    root_id: String,
    root_digest: CommissioningAuthorityRootDigest,
    epoch: u64,
}

impl CommissioningAuthorityRootSnapshot {
    pub fn new(
        root_id: impl Into<String>,
        root_digest: CommissioningAuthorityRootDigest,
        epoch: u64,
    ) -> Result<Self, CommissioningAuthorizationAdmissionError> {
        let root_id = root_id.into();
        if root_id.trim().is_empty() {
            return Err(CommissioningAuthorizationAdmissionError::EmptyExpectedAuthorityRootId);
        }
        if epoch == 0 {
            return Err(CommissioningAuthorizationAdmissionError::ZeroAuthorityRootEpoch);
        }
        Ok(Self {
            root_id,
            root_digest,
            epoch,
        })
    }

    pub fn root_id(&self) -> &str {
        &self.root_id
    }

    pub fn root_digest(&self) -> CommissioningAuthorityRootDigest {
        self.root_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Complete identity of one committed commissioning-authorization transition.
///
/// Fields are private and are derived from one validated transition so generation,
/// transition digest, node, and root identity cannot be assembled independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommissioningAuthorizationHeadIdentity {
    generation: u64,
    transition_digest: CommissioningAuthorizationTransitionDigest,
    subject_node_id: String,
    authority_root_id: String,
    authority_root_digest: CommissioningAuthorityRootDigest,
}

impl CommissioningAuthorizationHeadIdentity {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn transition_digest(&self) -> CommissioningAuthorizationTransitionDigest {
        self.transition_digest
    }

    pub fn subject_node_id(&self) -> &str {
        &self.subject_node_id
    }

    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }

    pub fn authority_root_digest(&self) -> CommissioningAuthorityRootDigest {
        self.authority_root_digest
    }
}

/// Exact persisted head of one commissioning-authorization lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommissioningAuthorizationHead {
    Uninitialized,
    Current(CommissioningAuthorizationHeadIdentity),
}

impl CommissioningAuthorizationHead {
    pub fn from_transition(
        transition: &CommissioningAuthorizationTransition,
    ) -> Result<Self, CommissioningAuthorizationAdmissionError> {
        transition.validate()?;
        let subject = transition.subject();
        Ok(Self::Current(CommissioningAuthorizationHeadIdentity {
            generation: transition.generation(),
            transition_digest: transition.transition_digest()?,
            subject_node_id: subject.subject_node_id().to_owned(),
            authority_root_id: subject.authority_root_id().to_owned(),
            authority_root_digest: subject.authority_root_digest(),
        }))
    }

    pub fn identity(&self) -> Option<&CommissioningAuthorizationHeadIdentity> {
        match self {
            Self::Uninitialized => None,
            Self::Current(identity) => Some(identity),
        }
    }
}

/// Trusted local policy inputs for admitting one commissioning authorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommissioningAuthorizationAdmissionPolicy {
    expected_subject_node_id: String,
    expected_root: CommissioningAuthorityRootSnapshot,
    current_head: CommissioningAuthorizationHead,
    current_profile_lifecycle: SafetyProfileAuthorizationLifecycleState,
}

impl CommissioningAuthorizationAdmissionPolicy {
    pub fn new(
        expected_subject_node_id: impl Into<String>,
        expected_root: CommissioningAuthorityRootSnapshot,
        current_head: CommissioningAuthorizationHead,
        current_profile_lifecycle: SafetyProfileAuthorizationLifecycleState,
    ) -> Result<Self, CommissioningAuthorizationAdmissionError> {
        let expected_subject_node_id = expected_subject_node_id.into();
        if expected_subject_node_id.trim().is_empty() {
            return Err(CommissioningAuthorizationAdmissionError::EmptyExpectedSubjectNodeId);
        }

        if let Some(identity) = current_head.identity() {
            if identity.subject_node_id() != expected_subject_node_id {
                return Err(
                    CommissioningAuthorizationAdmissionError::PersistedCommissioningHeadNodeMismatch {
                        expected: expected_subject_node_id,
                        observed: identity.subject_node_id().to_owned(),
                    },
                );
            }
            if identity.authority_root_id() != expected_root.root_id() {
                return Err(
                    CommissioningAuthorizationAdmissionError::PersistedCommissioningHeadRootIdMismatch {
                        expected: expected_root.root_id().to_owned(),
                        observed: identity.authority_root_id().to_owned(),
                    },
                );
            }
            if identity.authority_root_digest() != expected_root.root_digest() {
                return Err(
                    CommissioningAuthorizationAdmissionError::PersistedCommissioningHeadRootDigestMismatch,
                );
            }
        }

        if !current_profile_lifecycle.is_active() {
            return Err(CommissioningAuthorizationAdmissionError::ProfileLifecycleNotActive);
        }
        let profile_identity = current_profile_lifecycle
            .authorization_head()
            .identity()
            .ok_or(CommissioningAuthorizationAdmissionError::ProfileLifecycleHeadNotCurrent)?;
        if profile_identity.subject_node_id() != expected_subject_node_id {
            return Err(
                CommissioningAuthorizationAdmissionError::ProfileLifecycleNodeMismatch {
                    expected: expected_subject_node_id,
                    observed: profile_identity.subject_node_id().to_owned(),
                },
            );
        }

        Ok(Self {
            expected_subject_node_id,
            expected_root,
            current_head,
            current_profile_lifecycle,
        })
    }

    pub fn expected_subject_node_id(&self) -> &str {
        &self.expected_subject_node_id
    }

    pub fn expected_root(&self) -> &CommissioningAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn current_head(&self) -> &CommissioningAuthorizationHead {
        &self.current_head
    }

    pub fn current_profile_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.current_profile_lifecycle
    }

    /// Check one commissioning-authorization transition against exact local state.
    ///
    /// Success is deliberately not executable authority. The returned opaque policy
    /// value still requires Xenia detached-signature verification over its exact
    /// canonical transition bytes before a persistence adapter may commit it.
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        transition: &CommissioningAuthorizationTransition,
        record: &CommissioningRecord,
        hierarchy: &ResourceHierarchy,
        qualified: &QualifiedSafetyConfiguration,
        profile_authorization: &SafetyProfileAuthorizationTransition,
    ) -> Result<PolicyCheckedCommissioningAuthorization, CommissioningAuthorizationAdmissionError>
    {
        transition.validate()?;
        profile_authorization.validate()?;

        let subject = transition.subject();
        if subject.subject_node_id() != self.expected_subject_node_id {
            return Err(CommissioningAuthorizationAdmissionError::SubjectNodeMismatch {
                expected: self.expected_subject_node_id.clone(),
                observed: subject.subject_node_id().to_owned(),
            });
        }
        if subject.authority_root_id() != self.expected_root.root_id() {
            return Err(CommissioningAuthorizationAdmissionError::AuthorityRootIdMismatch {
                expected: self.expected_root.root_id().to_owned(),
                observed: subject.authority_root_id().to_owned(),
            });
        }
        if subject.authority_root_digest() != self.expected_root.root_digest() {
            return Err(CommissioningAuthorizationAdmissionError::AuthorityRootDigestMismatch);
        }

        check_commissioning_lineage(&self.current_head, transition)?;

        if record.subject_node_id() != self.expected_subject_node_id {
            return Err(CommissioningAuthorizationAdmissionError::RecordNodeMismatch {
                expected: self.expected_subject_node_id.clone(),
                observed: record.subject_node_id().to_owned(),
            });
        }
        if qualified.node_id() != self.expected_subject_node_id {
            return Err(CommissioningAuthorizationAdmissionError::QualificationNodeMismatch {
                expected: self.expected_subject_node_id.clone(),
                observed: qualified.node_id().to_owned(),
            });
        }
        if record.generation() != subject.commissioning_generation() {
            return Err(CommissioningAuthorizationAdmissionError::RecordGenerationMismatch {
                expected: subject.commissioning_generation(),
                observed: record.generation(),
            });
        }

        let record_digest = commissioning_record_digest(record, hierarchy)?;
        if record_digest != subject.commissioning_record_digest() {
            return Err(CommissioningAuthorizationAdmissionError::RecordDigestMismatch {
                expected: subject.commissioning_record_digest(),
                observed: record_digest,
            });
        }

        let record_configuration = record.commissioned().binding().configuration_digest();
        if record_configuration != qualified.configuration_digest() {
            return Err(
                CommissioningAuthorizationAdmissionError::RecordQualificationConfigurationMismatch,
            );
        }
        if subject.configuration_digest() != qualified.configuration_digest() {
            return Err(CommissioningAuthorizationAdmissionError::ConfigurationDigestMismatch {
                expected: subject.configuration_digest(),
                observed: qualified.configuration_digest(),
            });
        }
        if subject.profile_id() != qualified.profile_id() {
            return Err(CommissioningAuthorizationAdmissionError::ProfileIdMismatch {
                expected: subject.profile_id().to_owned(),
                observed: qualified.profile_id().to_owned(),
            });
        }
        if subject.profile_digest() != qualified.profile_digest() {
            return Err(CommissioningAuthorizationAdmissionError::ProfileDigestMismatch {
                expected: subject.profile_digest(),
                observed: qualified.profile_digest(),
            });
        }

        // The exact active profile transition must agree with both durable lifecycle
        // state and the provenance copied into the commissioning authorization.
        let profile_head_identity = self
            .current_profile_lifecycle
            .authorization_head()
            .identity()
            .ok_or(CommissioningAuthorizationAdmissionError::ProfileLifecycleHeadNotCurrent)?;
        let profile_transition_digest = profile_authorization.transition_digest()?;
        if profile_head_identity.generation() != profile_authorization.generation()
            || profile_head_identity.transition_digest() != profile_transition_digest
        {
            return Err(
                CommissioningAuthorizationAdmissionError::ProfileAuthorizationLifecycleMismatch,
            );
        }
        if subject.profile_authorization_generation() != profile_authorization.generation()
            || subject.profile_authorization_transition_digest() != profile_transition_digest
        {
            return Err(
                CommissioningAuthorizationAdmissionError::ProfileAuthorizationProvenanceMismatch,
            );
        }

        let profile_subject = profile_authorization.subject();
        if profile_subject.subject_node_id() != qualified.node_id()
            || profile_subject.profile_id() != qualified.profile_id()
            || profile_subject.profile_digest() != qualified.profile_digest()
        {
            return Err(
                CommissioningAuthorizationAdmissionError::ProfileAuthorizationQualificationMismatch,
            );
        }

        validate_entire_interval(
            ValidityScope::ProfileAuthorization,
            profile_subject.valid_from_unix_ms(),
            profile_subject.valid_until_unix_ms(),
            current_clock,
        )?;
        validate_entire_interval(
            ValidityScope::CommissioningAction,
            subject.valid_from_unix_ms(),
            subject.valid_until_unix_ms(),
            current_clock,
        )?;

        let candidate_head = CommissioningAuthorizationHead::from_transition(transition)?;

        Ok(PolicyCheckedCommissioningAuthorization {
            transition: transition.clone(),
            profile_authorization: profile_authorization.clone(),
            canonical_transition_bytes: transition.canonical_signing_bytes()?,
            expected_root: self.expected_root.clone(),
            expected_predecessor_head: self.current_head.clone(),
            candidate_head,
            expected_profile_lifecycle: self.current_profile_lifecycle.clone(),
            expected_clock_source_id: current_clock.source_id().to_owned(),
            expected_clock_epoch: current_clock.epoch(),
            commissioning_record_digest: record_digest,
            configuration_digest: qualified.configuration_digest(),
        })
    }
}

fn check_commissioning_lineage(
    current_head: &CommissioningAuthorizationHead,
    transition: &CommissioningAuthorizationTransition,
) -> Result<(), CommissioningAuthorizationAdmissionError> {
    match current_head {
        CommissioningAuthorizationHead::Uninitialized => {
            if transition.generation() != 1
                || transition.predecessor() != CommissioningAuthorizationPredecessor::Bootstrap
            {
                return Err(CommissioningAuthorizationAdmissionError::ExpectedBootstrap {
                    observed_generation: transition.generation(),
                    observed_predecessor: transition.predecessor(),
                });
            }
        }
        CommissioningAuthorizationHead::Current(identity) => {
            let expected_generation = identity.generation().checked_add(1).ok_or(
                CommissioningAuthorizationAdmissionError::GenerationExhausted {
                    current: identity.generation(),
                },
            )?;
            if transition.generation() != expected_generation {
                return Err(CommissioningAuthorizationAdmissionError::GenerationNotSuccessor {
                    current: identity.generation(),
                    expected: expected_generation,
                    observed: transition.generation(),
                });
            }
            let expected_predecessor = CommissioningAuthorizationPredecessor::Previous(
                identity.transition_digest(),
            );
            if transition.predecessor() != expected_predecessor {
                return Err(CommissioningAuthorizationAdmissionError::PredecessorHeadMismatch {
                    expected: expected_predecessor,
                    observed: transition.predecessor(),
                });
            }
        }
    }
    Ok(())
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
) -> Result<(), CommissioningAuthorizationAdmissionError> {
    if clock.latest_unix_ms() < valid_from_unix_ms {
        return Err(match scope {
            ValidityScope::ProfileAuthorization => {
                CommissioningAuthorizationAdmissionError::ProfileAuthorizationDefinitelyNotYetValid {
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                }
            }
            ValidityScope::CommissioningAction => {
                CommissioningAuthorizationAdmissionError::CommissioningDefinitelyNotYetValid {
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                }
            }
        });
    }
    if clock.earliest_unix_ms() >= valid_until_unix_ms {
        return Err(match scope {
            ValidityScope::ProfileAuthorization => {
                CommissioningAuthorizationAdmissionError::ProfileAuthorizationDefinitelyExpired {
                    earliest_unix_ms: clock.earliest_unix_ms(),
                    valid_until_unix_ms,
                }
            }
            ValidityScope::CommissioningAction => {
                CommissioningAuthorizationAdmissionError::CommissioningDefinitelyExpired {
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
                CommissioningAuthorizationAdmissionError::ProfileAuthorizationUncertaintyCrossesValidityBoundary {
                    earliest_unix_ms: clock.earliest_unix_ms(),
                    latest_unix_ms: clock.latest_unix_ms(),
                    valid_from_unix_ms,
                    valid_until_unix_ms,
                }
            }
            ValidityScope::CommissioningAction => {
                CommissioningAuthorizationAdmissionError::CommissioningUncertaintyCrossesValidityBoundary {
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

/// Non-serializable result of exact local commissioning-admission checks.
///
/// This type proves no signature. A future Xenia adapter must authenticate
/// [`Self::canonical_transition_bytes`] under [`Self::expected_root`] before any
/// commissioning head or commissioning record may be persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyCheckedCommissioningAuthorization {
    transition: CommissioningAuthorizationTransition,
    profile_authorization: SafetyProfileAuthorizationTransition,
    canonical_transition_bytes: Vec<u8>,
    expected_root: CommissioningAuthorityRootSnapshot,
    expected_predecessor_head: CommissioningAuthorizationHead,
    candidate_head: CommissioningAuthorizationHead,
    expected_profile_lifecycle: SafetyProfileAuthorizationLifecycleState,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    commissioning_record_digest: CommissioningRecordDigest,
    configuration_digest: ConfigurationDigest,
}

impl PolicyCheckedCommissioningAuthorization {
    pub fn transition(&self) -> &CommissioningAuthorizationTransition {
        &self.transition
    }

    pub fn profile_authorization(&self) -> &SafetyProfileAuthorizationTransition {
        &self.profile_authorization
    }

    pub fn canonical_transition_bytes(&self) -> &[u8] {
        &self.canonical_transition_bytes
    }

    pub fn expected_root(&self) -> &CommissioningAuthorityRootSnapshot {
        &self.expected_root
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

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    pub fn commissioning_record_digest(&self) -> CommissioningRecordDigest {
        self.commissioning_record_digest
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommissioningAuthorizationAdmissionError {
    #[error(transparent)]
    Authorization(#[from] CommissioningAuthorizationError),
    #[error(transparent)]
    Transition(#[from] CommissioningAuthorizationTransitionError),
    #[error(transparent)]
    Identity(#[from] CommissioningIdentityError),
    #[error(transparent)]
    ProfileAuthorization(#[from] SafetyProfileAuthorizationTransitionError),
    #[error("expected commissioning subject node id must not be empty")]
    EmptyExpectedSubjectNodeId,
    #[error("expected commissioning-authority root id must not be empty")]
    EmptyExpectedAuthorityRootId,
    #[error("commissioning-authority root epoch must be greater than zero")]
    ZeroAuthorityRootEpoch,
    #[error("persisted commissioning head belongs to node {observed}, expected {expected}")]
    PersistedCommissioningHeadNodeMismatch { expected: String, observed: String },
    #[error("persisted commissioning head root id mismatch: expected {expected}, observed {observed}")]
    PersistedCommissioningHeadRootIdMismatch { expected: String, observed: String },
    #[error("persisted commissioning head root digest does not match the provisioned root")]
    PersistedCommissioningHeadRootDigestMismatch,
    #[error("current safety-profile lifecycle is not active")]
    ProfileLifecycleNotActive,
    #[error("active safety-profile lifecycle unexpectedly has no current head")]
    ProfileLifecycleHeadNotCurrent,
    #[error("active safety-profile lifecycle belongs to node {observed}, expected {expected}")]
    ProfileLifecycleNodeMismatch { expected: String, observed: String },
    #[error("commissioning authorization subject node mismatch: expected {expected}, observed {observed}")]
    SubjectNodeMismatch { expected: String, observed: String },
    #[error("commissioning authority-root id mismatch: expected {expected}, observed {observed}")]
    AuthorityRootIdMismatch { expected: String, observed: String },
    #[error("commissioning authority-root digest does not match the provisioned root")]
    AuthorityRootDigestMismatch,
    #[error("expected commissioning bootstrap generation 1 with Bootstrap predecessor; observed generation {observed_generation} / {observed_predecessor:?}")]
    ExpectedBootstrap {
        observed_generation: u64,
        observed_predecessor: CommissioningAuthorizationPredecessor,
    },
    #[error("commissioning generation space exhausted at {current}")]
    GenerationExhausted { current: u64 },
    #[error("commissioning authorization generation must be exact successor of {current}: expected {expected}, observed {observed}")]
    GenerationNotSuccessor {
        current: u64,
        expected: u64,
        observed: u64,
    },
    #[error("commissioning authorization predecessor does not match exact persisted head")]
    PredecessorHeadMismatch {
        expected: CommissioningAuthorizationPredecessor,
        observed: CommissioningAuthorizationPredecessor,
    },
    #[error("commissioning record node mismatch: expected {expected}, observed {observed}")]
    RecordNodeMismatch { expected: String, observed: String },
    #[error("qualified configuration node mismatch: expected {expected}, observed {observed}")]
    QualificationNodeMismatch { expected: String, observed: String },
    #[error("commissioning record generation mismatch: expected {expected}, observed {observed}")]
    RecordGenerationMismatch { expected: u64, observed: u64 },
    #[error("commissioning record digest does not match the signed authorization")]
    RecordDigestMismatch {
        expected: CommissioningRecordDigest,
        observed: CommissioningRecordDigest,
    },
    #[error("commissioning record configuration does not match qualified configuration")]
    RecordQualificationConfigurationMismatch,
    #[error("commissioning authorization configuration digest does not match qualification")]
    ConfigurationDigestMismatch {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
    #[error("commissioning authorization profile id mismatch: expected {expected}, observed {observed}")]
    ProfileIdMismatch { expected: String, observed: String },
    #[error("commissioning authorization profile digest does not match qualification")]
    ProfileDigestMismatch {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
    #[error("exact active profile-authorization transition does not match durable profile lifecycle head")]
    ProfileAuthorizationLifecycleMismatch,
    #[error("commissioning authorization does not bind the exact active profile-authorization lineage")]
    ProfileAuthorizationProvenanceMismatch,
    #[error("active profile authorization does not match the qualified configuration profile")]
    ProfileAuthorizationQualificationMismatch,
    #[error("active profile authorization is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    ProfileAuthorizationDefinitelyNotYetValid {
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("active profile authorization is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    ProfileAuthorizationDefinitelyExpired {
        earliest_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses active profile authorization validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    ProfileAuthorizationUncertaintyCrossesValidityBoundary {
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
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses commissioning action validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
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
    use crate::authorization::{CommissioningAuthorizationSubject, CommissioningAuthorityRootDigest};
    use crate::{CommissionedLocalSafetyEnvelope, CommissioningBinding};
    use std::collections::BTreeSet;
    use symthaea_operating_autonomy::LocalSafetyEnvelope;
    use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
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
    use symthaea_safety_profile::revocation::SafetyProfileAuthorizationRevocationDigest;
    use symthaea_safety_profile::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };
    use symthaea_safety_qualification::qualify_safety_configuration;

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

    fn profile_transition(
        authorization_id: &str,
        valid_from: i64,
        valid_until: i64,
    ) -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        SafetyProfileAuthorizationTransition::bootstrap(
            SafetyProfileAuthorizationSubject::new(
                authorization_id,
                "profile-root-v1",
                profile_root(0x44),
                "rack",
                1,
                valid_from,
                valid_until,
                &profile,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn active_profile_lifecycle(
        transition: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationLifecycleState {
        SafetyProfileAuthorizationLifecycleState::active(
            SafetyProfileAuthorizationHead::from_transition(transition).unwrap(),
        )
        .unwrap()
    }

    fn record(generation: u64, evidence: &str) -> CommissioningRecord {
        let qualified = qualified();
        CommissioningRecord::new(
            generation,
            CommissionedLocalSafetyEnvelope::new(
                local(),
                CommissioningBinding::new(qualified.configuration_digest(), evidence).unwrap(),
            ),
        )
        .unwrap()
    }

    fn commissioning_transition(
        record: &CommissioningRecord,
        profile_authorization: &SafetyProfileAuthorizationTransition,
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
                profile_authorization,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn root_snapshot() -> CommissioningAuthorityRootSnapshot {
        CommissioningAuthorityRootSnapshot::new(
            "commission-root-v1",
            commissioning_root(0x55),
            4,
        )
        .unwrap()
    }

    fn clock(earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new("secure-rtc-v1", 7, earliest, latest).unwrap()
    }

    #[test]
    fn exact_bootstrap_with_active_profile_is_admitted() {
        let profile_authorization = profile_transition("profile-auth-1", 1_000, 3_000);
        let record = record(1, "commissioning-evidence-1");
        let transition = commissioning_transition(&record, &profile_authorization);
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            root_snapshot(),
            CommissioningAuthorizationHead::Uninitialized,
            active_profile_lifecycle(&profile_authorization),
        )
        .unwrap();
        let hierarchy = hierarchy();
        let qualified = qualified();

        let checked = policy
            .check(
                &clock(1_500, 1_600),
                &transition,
                &record,
                &hierarchy,
                &qualified,
                &profile_authorization,
            )
            .unwrap();

        assert_eq!(checked.commissioning_record_digest(), commissioning_record_digest(&record, &hierarchy).unwrap());
        assert_eq!(checked.configuration_digest(), qualified.configuration_digest());
        assert_eq!(checked.expected_profile_lifecycle(), policy.current_profile_lifecycle());
        assert_eq!(checked.expected_clock_source_id(), "secure-rtc-v1");
        assert_eq!(checked.expected_clock_epoch(), 7);
        assert!(checked.candidate_head().identity().is_some());
    }

    #[test]
    fn revoked_profile_lifecycle_cannot_prepare_commissioning() {
        let profile_authorization = profile_transition("profile-auth-1", 1_000, 3_000);
        let head = SafetyProfileAuthorizationHead::from_transition(&profile_authorization).unwrap();
        let revoked = SafetyProfileAuthorizationLifecycleState::revoked(
            head,
            SafetyProfileAuthorizationRevocationDigest::Blake3_256([0x99; 32]),
        )
        .unwrap();

        assert_eq!(
            CommissioningAuthorizationAdmissionPolicy::new(
                "rack",
                root_snapshot(),
                CommissioningAuthorizationHead::Uninitialized,
                revoked,
            ),
            Err(CommissioningAuthorizationAdmissionError::ProfileLifecycleNotActive)
        );
    }

    #[test]
    fn exact_active_profile_transition_must_match_lifecycle_head() {
        let active_transition = profile_transition("profile-auth-a", 1_000, 3_000);
        let different_transition = profile_transition("profile-auth-b", 1_000, 3_000);
        let record = record(1, "commissioning-evidence-1");
        let transition = commissioning_transition(&record, &different_transition);
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            root_snapshot(),
            CommissioningAuthorizationHead::Uninitialized,
            active_profile_lifecycle(&active_transition),
        )
        .unwrap();
        let hierarchy = hierarchy();
        let qualified = qualified();

        assert_eq!(
            policy.check(
                &clock(1_500, 1_600),
                &transition,
                &record,
                &hierarchy,
                &qualified,
                &different_transition,
            ),
            Err(CommissioningAuthorizationAdmissionError::ProfileAuthorizationLifecycleMismatch)
        );
    }

    #[test]
    fn expired_active_profile_cannot_authorize_new_commissioning() {
        let profile_authorization = profile_transition("profile-auth-1", 1_000, 1_400);
        let record = record(1, "commissioning-evidence-1");
        let transition = commissioning_transition(&record, &profile_authorization);
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            root_snapshot(),
            CommissioningAuthorizationHead::Uninitialized,
            active_profile_lifecycle(&profile_authorization),
        )
        .unwrap();
        let hierarchy = hierarchy();
        let qualified = qualified();

        assert!(matches!(
            policy.check(
                &clock(1_500, 1_600),
                &transition,
                &record,
                &hierarchy,
                &qualified,
                &profile_authorization,
            ),
            Err(CommissioningAuthorizationAdmissionError::ProfileAuthorizationDefinitelyExpired { .. })
        ));
    }

    #[test]
    fn changed_commissioning_record_is_rejected() {
        let profile_authorization = profile_transition("profile-auth-1", 1_000, 3_000);
        let signed_record = record(1, "commissioning-evidence-1");
        let observed_record = record(1, "different-evidence");
        let transition = commissioning_transition(&signed_record, &profile_authorization);
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            root_snapshot(),
            CommissioningAuthorizationHead::Uninitialized,
            active_profile_lifecycle(&profile_authorization),
        )
        .unwrap();
        let hierarchy = hierarchy();
        let qualified = qualified();

        assert!(matches!(
            policy.check(
                &clock(1_500, 1_600),
                &transition,
                &observed_record,
                &hierarchy,
                &qualified,
                &profile_authorization,
            ),
            Err(CommissioningAuthorizationAdmissionError::RecordDigestMismatch { .. })
        ));
    }

    #[test]
    fn commissioning_root_alias_is_rejected() {
        let profile_authorization = profile_transition("profile-auth-1", 1_000, 3_000);
        let record = record(1, "commissioning-evidence-1");
        let transition = commissioning_transition(&record, &profile_authorization);
        let policy = CommissioningAuthorizationAdmissionPolicy::new(
            "rack",
            CommissioningAuthorityRootSnapshot::new(
                "commission-root-alias",
                commissioning_root(0x55),
                4,
            )
            .unwrap(),
            CommissioningAuthorizationHead::Uninitialized,
            active_profile_lifecycle(&profile_authorization),
        )
        .unwrap();
        let hierarchy = hierarchy();
        let qualified = qualified();

        assert!(matches!(
            policy.check(
                &clock(1_500, 1_600),
                &transition,
                &record,
                &hierarchy,
                &qualified,
                &profile_authorization,
            ),
            Err(CommissioningAuthorizationAdmissionError::AuthorityRootIdMismatch { .. })
        ));
    }
}

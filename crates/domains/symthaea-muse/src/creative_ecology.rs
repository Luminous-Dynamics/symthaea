// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Functional self-maintenance and sympoietic co-creation primitives.
//!
//! This module makes two deliberately bounded engineering translations:
//!
//! - CreativeOrganizationV1 describes a recurrent network of internal
//!   production processes and the components they maintain. Its assessment is
//!   a graph-structure diagnostic, not a proof of biological autopoiesis.
//! - CoCreationSessionV1 records consent-governed, append-only contributions
//!   with exact artifact lineage. It preserves branches and disagreement;
//!   there is no consciousness-, confidence-, or status-weighted voting.
//!
//! External resources (audio, instruments, compute, participants, datasets,
//! and networks) are explicit dependencies. They are not mislabeled as
//! internally produced components. None of these types establishes life,
//! phenomenal consciousness, universal alignment, or a conscious collective.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

fn non_empty(name: &'static str, value: &str) -> Result<(), CreativeEcologyError> {
    if value.trim().is_empty() {
        return Err(CreativeEcologyError::EmptyIdentifier(name));
    }
    Ok(())
}

fn sha256_hex(name: &'static str, value: &str) -> Result<(), CreativeEcologyError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CreativeEcologyError::InvalidDigest(name));
    }
    Ok(())
}

fn unique_strings(
    name: &'static str,
    values: &[String],
) -> Result<(), CreativeEcologyError> {
    let mut seen = BTreeSet::new();
    for value in values {
        non_empty(name, value)?;
        if !seen.insert(value.as_str()) {
            return Err(CreativeEcologyError::DuplicateIdentifier(name));
        }
    }
    Ok(())
}

// ─── Functional self-maintenance ────────────────────────────────────────────

/// A component in Melothaea's declared internal production organization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrganizationComponentV1 {
    pub component_id: String,
    /// Components required for the declared creative organization to operate.
    pub required_for_operation: bool,
    /// A component that defines or enforces the self/environment interface.
    pub boundary_component: bool,
}

/// A process that consumes and regenerates internal components.
///
/// External dependencies are declared separately and treated as perturbations
/// or resources supplied by the environment, not as self-produced components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductionProcessV1 {
    pub process_id: String,
    pub consumes: Vec<String>,
    pub produces: Vec<String>,
    pub external_inputs: Vec<String>,
}

/// A versioned, inspectable functional organization.
///
/// Components and processes describe internal recurrence. Audio devices,
/// compute, people, datasets, network access, and other environmental
/// resources belong in external_dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreativeOrganizationV1 {
    pub organization_id: String,
    pub revision: u64,
    pub components: Vec<OrganizationComponentV1>,
    pub processes: Vec<ProductionProcessV1>,
    pub external_dependencies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionalRecurrenceAssessmentV1 {
    pub organization_id: String,
    pub revision: u64,
    pub required_components_without_producer: Vec<String>,
    pub required_components_without_consumer: Vec<String>,
    pub boundary_components_without_producer: Vec<String>,
    pub boundary_components_without_consumer: Vec<String>,
    pub process_network_strongly_connected: bool,
    pub internal_feedback_edge_count: usize,
    /// Engineering diagnostic only; never a biological-life or consciousness claim.
    pub functionally_recurrent: bool,
}

impl CreativeOrganizationV1 {
    pub fn validate(&self) -> Result<(), CreativeEcologyError> {
        non_empty("organization_id", &self.organization_id)?;
        if self.revision == 0 {
            return Err(CreativeEcologyError::InvalidRevision);
        }
        let component_ids: Vec<String> = self
            .components
            .iter()
            .map(|component| component.component_id.clone())
            .collect();
        unique_strings("component_id", &component_ids)?;
        let process_ids: Vec<String> = self
            .processes
            .iter()
            .map(|process| process.process_id.clone())
            .collect();
        unique_strings("process_id", &process_ids)?;
        unique_strings("external_dependency", &self.external_dependencies)?;

        let component_set: BTreeSet<&str> =
            component_ids.iter().map(String::as_str).collect();
        let external_set: BTreeSet<&str> = self
            .external_dependencies
            .iter()
            .map(String::as_str)
            .collect();

        for process in &self.processes {
            non_empty("process_id", &process.process_id)?;
            if process.consumes.is_empty() {
                return Err(CreativeEcologyError::ProcessHasNoInputs(
                    process.process_id.clone(),
                ));
            }
            if process.produces.is_empty() {
                return Err(CreativeEcologyError::ProcessHasNoOutputs(
                    process.process_id.clone(),
                ));
            }
            unique_strings("process_input", &process.consumes)?;
            unique_strings("process_output", &process.produces)?;
            unique_strings("process_external_input", &process.external_inputs)?;

            for component_id in process.consumes.iter().chain(process.produces.iter()) {
                if !component_set.contains(component_id.as_str()) {
                    return Err(CreativeEcologyError::UnknownComponent {
                        process_id: process.process_id.clone(),
                        component_id: component_id.clone(),
                    });
                }
            }
            for dependency in &process.external_inputs {
                if !external_set.contains(dependency.as_str()) {
                    return Err(CreativeEcologyError::UndeclaredExternalDependency {
                        process_id: process.process_id.clone(),
                        dependency_id: dependency.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Assess whether critical components are regenerated by a strongly
    /// connected network of internal processes.
    ///
    /// This is deliberately stricter than a weighted coherence heuristic:
    /// missing producers, missing boundary production, and a disconnected
    /// process graph remain separately inspectable.
    pub fn assess_functional_recurrence(
        &self,
    ) -> Result<FunctionalRecurrenceAssessmentV1, CreativeEcologyError> {
        self.validate()?;

        let mut required_components_without_producer = Vec::new();
        let mut required_components_without_consumer = Vec::new();
        let mut boundary_components_without_producer = Vec::new();
        let mut boundary_components_without_consumer = Vec::new();
        for component in &self.components {
            let has_producer = self.processes.iter().any(|process| {
                process
                    .produces
                    .iter()
                    .any(|produced| produced == &component.component_id)
            });
            let has_consumer = self.processes.iter().any(|process| {
                process
                    .consumes
                    .iter()
                    .any(|consumed| consumed == &component.component_id)
            });
            if component.required_for_operation && !has_producer {
                required_components_without_producer.push(component.component_id.clone());
            }
            if component.required_for_operation && !has_consumer {
                required_components_without_consumer.push(component.component_id.clone());
            }
            if component.boundary_component && !has_producer {
                boundary_components_without_producer.push(component.component_id.clone());
            }
            if component.boundary_component && !has_consumer {
                boundary_components_without_consumer.push(component.component_id.clone());
            }
        }

        let (process_network_strongly_connected, internal_feedback_edge_count) =
            process_recurrence(&self.processes);
        let functionally_recurrent = !self.processes.is_empty()
            && required_components_without_producer.is_empty()
            && required_components_without_consumer.is_empty()
            && boundary_components_without_producer.is_empty()
            && boundary_components_without_consumer.is_empty()
            && process_network_strongly_connected;

        Ok(FunctionalRecurrenceAssessmentV1 {
            organization_id: self.organization_id.clone(),
            revision: self.revision,
            required_components_without_producer,
            required_components_without_consumer,
            boundary_components_without_producer,
            boundary_components_without_consumer,
            process_network_strongly_connected,
            internal_feedback_edge_count,
            functionally_recurrent,
        })
    }
}

/// Return (strongly connected, internal producer-consumer edge count).
fn process_recurrence(processes: &[ProductionProcessV1]) -> (bool, usize) {
    if processes.is_empty() {
        return (false, 0);
    }

    let n = processes.len();
    let mut edges = vec![Vec::<usize>::new(); n];
    let mut reverse = vec![Vec::<usize>::new(); n];
    let mut edge_count = 0usize;

    for (source_index, source) in processes.iter().enumerate() {
        for (target_index, target) in processes.iter().enumerate() {
            let shares_component = source.produces.iter().any(|component| {
                target.consumes.iter().any(|consumed| consumed == component)
            });
            if shares_component {
                edges[source_index].push(target_index);
                reverse[target_index].push(source_index);
                edge_count = edge_count.saturating_add(1);
            }
        }
    }

    // A one-process organization must regenerate something it consumes.
    if n == 1 {
        return (edges[0].contains(&0), edge_count);
    }

    let forward = reachable_indices(0, &edges);
    let backward = reachable_indices(0, &reverse);
    let strongly_connected = forward.len() == n && backward.len() == n;
    (strongly_connected, edge_count)
}

fn reachable_indices(start: usize, graph: &[Vec<usize>]) -> BTreeSet<usize> {
    let mut visited = BTreeSet::new();
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            if let Some(neighbors) = graph.get(node) {
                stack.extend(neighbors.iter().copied());
            }
        }
    }
    visited
}

// ─── Sympoietic session and consent ledger ──────────────────────────────────

pub const MAX_CO_CREATION_PARTICIPANTS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributorKindV1 {
    Human,
    Melothaea,
    OtherAgent,
    InstrumentOrPerformanceSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoCreationParticipantV1 {
    pub participant_id: String,
    pub kind: ContributorKindV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConsentPurposeV1 {
    Contribute,
    Review,
    Transform,
    StoreInPersonalMemory,
    TrainTasteModel,
    Publish,
    Federate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsentGrantV1 {
    pub grant_id: String,
    pub participant_id: String,
    pub purposes: Vec<ConsentPurposeV1>,
    pub granted_at_sequence: u64,
    /// Inclusive. None means no time/sequence expiry; it is still revocable.
    pub valid_through_sequence: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsentRevocationV1 {
    pub revocation_id: String,
    pub grant_id: String,
    /// The next event sequence at the moment revocation is recorded.
    /// No event at or after this sequence may use this grant.
    pub effective_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoCreativeActionV1 {
    Propose {
        parent_artifact_sha256: Option<String>,
        result_artifact_sha256: String,
        influence_event_ids: Vec<String>,
    },
    Accept {
        proposal_event_id: String,
    },
    Adapt {
        proposal_event_id: String,
        result_artifact_sha256: String,
        influence_event_ids: Vec<String>,
    },
    Reject {
        proposal_event_id: String,
        reason_code: String,
    },
    Abstain {
        proposal_event_id: String,
        reason_code: String,
    },
    Branch {
        proposal_event_id: String,
        result_artifact_sha256: String,
        influence_event_ids: Vec<String>,
    },
}

impl CoCreativeActionV1 {
    fn required_purpose(&self) -> ConsentPurposeV1 {
        match self {
            Self::Propose { .. } => ConsentPurposeV1::Contribute,
            Self::Accept { .. } | Self::Reject { .. } | Self::Abstain { .. } => {
                ConsentPurposeV1::Review
            }
            Self::Adapt { .. } | Self::Branch { .. } => ConsentPurposeV1::Transform,
        }
    }

    fn referenced_events(&self) -> Vec<&str> {
        match self {
            Self::Propose {
                influence_event_ids, ..
            }
            | Self::Adapt {
                influence_event_ids, ..
            }
            | Self::Branch {
                influence_event_ids, ..
            } => influence_event_ids.iter().map(String::as_str).collect(),
            Self::Accept { proposal_event_id }
            | Self::Reject {
                proposal_event_id, ..
            }
            | Self::Abstain {
                proposal_event_id, ..
            } => vec![proposal_event_id.as_str()],
        }
    }

    fn validate_artifacts(&self) -> Result<(), CreativeEcologyError> {
        match self {
            Self::Propose {
                parent_artifact_sha256,
                result_artifact_sha256,
                ..
            } => {
                if let Some(parent) = parent_artifact_sha256 {
                    sha256_hex("parent_artifact_sha256", parent)?;
                }
                sha256_hex("result_artifact_sha256", result_artifact_sha256)
            }
            Self::Adapt {
                result_artifact_sha256,
                ..
            }
            | Self::Branch {
                result_artifact_sha256,
                ..
            } => sha256_hex("result_artifact_sha256", result_artifact_sha256),
            Self::Accept { .. } => Ok(()),
            Self::Reject { reason_code, .. } | Self::Abstain { reason_code, .. } => {
                non_empty("reason_code", reason_code)
            }
        }
    }

    fn produced_artifact_sha256(&self) -> Option<&str> {
        match self {
            Self::Propose {
                result_artifact_sha256,
                ..
            }
            | Self::Adapt {
                result_artifact_sha256,
                ..
            }
            | Self::Branch {
                result_artifact_sha256,
                ..
            } => Some(result_artifact_sha256.as_str()),
            Self::Accept { .. } | Self::Reject { .. } | Self::Abstain { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoCreationEventV1 {
    pub event_id: String,
    pub sequence: u64,
    pub participant_id: String,
    pub grant_id: String,
    pub purpose: ConsentPurposeV1,
    pub action: CoCreativeActionV1,
}

/// Append-only session ledger. Private collections are mutated only through
/// methods that validate participant membership, consent, ordering and lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoCreationSessionV1 {
    pub session_id: String,
    participants: Vec<CoCreationParticipantV1>,
    grants: Vec<ConsentGrantV1>,
    revocations: Vec<ConsentRevocationV1>,
    events: Vec<CoCreationEventV1>,
}

impl CoCreationSessionV1 {
    pub fn new(session_id: impl Into<String>) -> Result<Self, CreativeEcologyError> {
        let session_id = session_id.into();
        non_empty("session_id", &session_id)?;
        Ok(Self {
            session_id,
            participants: Vec::new(),
            grants: Vec::new(),
            revocations: Vec::new(),
            events: Vec::new(),
        })
    }

    pub fn register_participant(
        &mut self,
        participant: CoCreationParticipantV1,
    ) -> Result<(), CreativeEcologyError> {
        self.validate_snapshot()?;
        non_empty("participant_id", &participant.participant_id)?;
        if self
            .participants
            .iter()
            .any(|existing| existing.participant_id == participant.participant_id)
        {
            return Err(CreativeEcologyError::DuplicateIdentifier("participant_id"));
        }
        if self.participants.len() >= MAX_CO_CREATION_PARTICIPANTS {
            return Err(CreativeEcologyError::ParticipantLimitReached);
        }
        self.participants.push(participant);
        Ok(())
    }

    pub fn grant_consent(
        &mut self,
        grant_id: impl Into<String>,
        participant_id: &str,
        purposes: Vec<ConsentPurposeV1>,
        valid_through_sequence: Option<u64>,
    ) -> Result<ConsentGrantV1, CreativeEcologyError> {
        self.validate_snapshot()?;
        let grant_id = grant_id.into();
        non_empty("grant_id", &grant_id)?;
        non_empty("participant_id", participant_id)?;
        if !self
            .participants
            .iter()
            .any(|participant| participant.participant_id == participant_id)
        {
            return Err(CreativeEcologyError::UnknownParticipant);
        }
        if self.grants.iter().any(|grant| grant.grant_id == grant_id) {
            return Err(CreativeEcologyError::DuplicateIdentifier("grant_id"));
        }
        if purposes.is_empty() {
            return Err(CreativeEcologyError::EmptyConsentPurposes);
        }
        let purpose_set: BTreeSet<ConsentPurposeV1> = purposes.iter().copied().collect();
        if purpose_set.len() != purposes.len() {
            return Err(CreativeEcologyError::DuplicateConsentPurpose);
        }
        let next_sequence = self.next_sequence();
        if valid_through_sequence.is_some_and(|end| end < next_sequence) {
            return Err(CreativeEcologyError::ConsentAlreadyExpired);
        }

        let grant = ConsentGrantV1 {
            grant_id,
            participant_id: participant_id.to_string(),
            purposes,
            granted_at_sequence: next_sequence,
            valid_through_sequence,
        };
        self.grants.push(grant.clone());
        Ok(grant)
    }

    /// Revoke future use of a grant. Earlier events remain in the immutable
    /// lineage; revocation cannot retract an already disclosed derivative.
    pub fn revoke_consent(
        &mut self,
        revocation_id: impl Into<String>,
        participant_id: &str,
        grant_id: &str,
    ) -> Result<ConsentRevocationV1, CreativeEcologyError> {
        self.validate_snapshot()?;
        let revocation_id = revocation_id.into();
        non_empty("revocation_id", &revocation_id)?;
        non_empty("participant_id", participant_id)?;
        non_empty("grant_id", grant_id)?;
        let grant = self
            .grants
            .iter()
            .find(|grant| grant.grant_id == grant_id)
            .ok_or(CreativeEcologyError::UnknownConsentGrant)?;
        if grant.participant_id != participant_id {
            return Err(CreativeEcologyError::ConsentOwnerMismatch);
        }
        if self.revocations.iter().any(|item| item.grant_id == grant_id) {
            return Err(CreativeEcologyError::ConsentAlreadyRevoked);
        }
        if self
            .revocations
            .iter()
            .any(|item| item.revocation_id == revocation_id)
        {
            return Err(CreativeEcologyError::DuplicateIdentifier("revocation_id"));
        }
        let revocation = ConsentRevocationV1 {
            revocation_id,
            grant_id: grant_id.to_string(),
            effective_sequence: self.next_sequence(),
        };
        self.revocations.push(revocation.clone());
        Ok(revocation)
    }

    /// Append an action after consent and artifact-lineage checks. Sequence and
    /// event identity are assigned by the ledger, never trusted from a caller.
    pub fn append_action(
        &mut self,
        participant_id: &str,
        grant_id: &str,
        action: CoCreativeActionV1,
    ) -> Result<CoCreationEventV1, CreativeEcologyError> {
        self.validate_snapshot()?;
        non_empty("participant_id", participant_id)?;
        non_empty("grant_id", grant_id)?;
        action.validate_artifacts()?;

        let sequence = self.next_sequence();
        if !self
            .participants
            .iter()
            .any(|participant| participant.participant_id == participant_id)
        {
            return Err(CreativeEcologyError::UnknownParticipant);
        }
        let grant = self
            .grants
            .iter()
            .find(|grant| grant.grant_id == grant_id)
            .ok_or(CreativeEcologyError::UnknownConsentGrant)?;
        if grant.participant_id != participant_id {
            return Err(CreativeEcologyError::ConsentOwnerMismatch);
        }
        if sequence < grant.granted_at_sequence
            || grant
                .valid_through_sequence
                .is_some_and(|end| sequence > end)
            || self.revocations.iter().any(|revocation| {
                revocation.grant_id == grant_id && sequence >= revocation.effective_sequence
            })
        {
            return Err(CreativeEcologyError::ConsentNotActive);
        }

        let purpose = action.required_purpose();
        if !grant.purposes.contains(&purpose) {
            return Err(CreativeEcologyError::ConsentPurposeNotGranted(purpose));
        }

        for referenced in action.referenced_events() {
            let Some(previous) = self.events.iter().find(|event| event.event_id == referenced)
            else {
                return Err(CreativeEcologyError::UnknownLineageEvent(
                    referenced.to_string(),
                ));
            };
            if previous.sequence >= sequence {
                return Err(CreativeEcologyError::ForwardLineageReference);
            }
        }
        self.validate_action_targets(&action)?;

        let event = CoCreationEventV1 {
            event_id: format!("{}:{}", self.session_id, sequence),
            sequence,
            participant_id: participant_id.to_string(),
            grant_id: grant_id.to_string(),
            purpose,
            action,
        };
        self.events.push(event.clone());
        Ok(event)
    }

    fn validate_action_targets(
        &self,
        action: &CoCreativeActionV1,
    ) -> Result<(), CreativeEcologyError> {
        let target_id = match action {
            CoCreativeActionV1::Propose { .. } => return Ok(()),
            CoCreativeActionV1::Accept { proposal_event_id }
            | CoCreativeActionV1::Adapt {
                proposal_event_id, ..
            }
            | CoCreativeActionV1::Reject {
                proposal_event_id, ..
            }
            | CoCreativeActionV1::Abstain {
                proposal_event_id, ..
            }
            | CoCreativeActionV1::Branch {
                proposal_event_id, ..
            } => proposal_event_id,
        };
        let target = self
            .events
            .iter()
            .find(|event| &event.event_id == target_id)
            .ok_or_else(|| CreativeEcologyError::UnknownLineageEvent(target_id.clone()))?;
        if target.action.produced_artifact_sha256().is_none() {
            return Err(CreativeEcologyError::TargetHasNoArtifact);
        }
        Ok(())
    }

    pub fn validate_snapshot(&self) -> Result<(), CreativeEcologyError> {
        non_empty("session_id", &self.session_id)?;
        if self.participants.len() > MAX_CO_CREATION_PARTICIPANTS {
            return Err(CreativeEcologyError::ParticipantLimitReached);
        }
        let participant_ids: Vec<String> = self
            .participants
            .iter()
            .map(|participant| participant.participant_id.clone())
            .collect();
        unique_strings("participant_id", &participant_ids)?;
        let grant_ids: Vec<String> = self
            .grants
            .iter()
            .map(|grant| grant.grant_id.clone())
            .collect();
        unique_strings("grant_id", &grant_ids)?;
        let revocation_ids: Vec<String> = self
            .revocations
            .iter()
            .map(|item| item.revocation_id.clone())
            .collect();
        unique_strings("revocation_id", &revocation_ids)?;

        for grant in &self.grants {
            if !self
                .participants
                .iter()
                .any(|participant| participant.participant_id == grant.participant_id)
            {
                return Err(CreativeEcologyError::UnknownParticipant);
            }
            if grant.purposes.is_empty() {
                return Err(CreativeEcologyError::EmptyConsentPurposes);
            }
            let purpose_set: BTreeSet<ConsentPurposeV1> =
                grant.purposes.iter().copied().collect();
            if purpose_set.len() != grant.purposes.len() {
                return Err(CreativeEcologyError::DuplicateConsentPurpose);
            }
            if grant.granted_at_sequence == 0
                || grant
                    .valid_through_sequence
                    .is_some_and(|end| end < grant.granted_at_sequence)
            {
                return Err(CreativeEcologyError::ConsentAlreadyExpired);
            }
        }

        let mut seen_revoked_grants = BTreeSet::new();
        for revocation in &self.revocations {
            if !self
                .grants
                .iter()
                .any(|grant| grant.grant_id == revocation.grant_id)
            {
                return Err(CreativeEcologyError::UnknownConsentGrant);
            }
            if !seen_revoked_grants.insert(revocation.grant_id.as_str()) {
                return Err(CreativeEcologyError::ConsentAlreadyRevoked);
            }
            if revocation.effective_sequence == 0
                || revocation.effective_sequence > self.next_sequence()
            {
                return Err(CreativeEcologyError::InvalidSequence);
            }
        }

        let mut prior_events = Vec::<CoCreationEventV1>::new();
        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = index as u64 + 1;
            if event.sequence != expected_sequence
                || event.event_id != format!("{}:{}", self.session_id, expected_sequence)
            {
                return Err(CreativeEcologyError::InvalidSequence);
            }
            if !self
                .participants
                .iter()
                .any(|participant| participant.participant_id == event.participant_id)
            {
                return Err(CreativeEcologyError::UnknownParticipant);
            }
            let grant = self
                .grants
                .iter()
                .find(|grant| grant.grant_id == event.grant_id)
                .ok_or(CreativeEcologyError::UnknownConsentGrant)?;
            if grant.participant_id != event.participant_id {
                return Err(CreativeEcologyError::ConsentOwnerMismatch);
            }
            if expected_sequence < grant.granted_at_sequence
                || grant
                    .valid_through_sequence
                    .is_some_and(|end| expected_sequence > end)
                || self.revocations.iter().any(|revocation| {
                    revocation.grant_id == event.grant_id
                        && expected_sequence >= revocation.effective_sequence
                })
            {
                return Err(CreativeEcologyError::ConsentNotActive);
            }
            let purpose = event.action.required_purpose();
            if event.purpose != purpose || !grant.purposes.contains(&purpose) {
                return Err(CreativeEcologyError::ConsentPurposeNotGranted(purpose));
            }
            event.action.validate_artifacts()?;
            for referenced in event.action.referenced_events() {
                if !prior_events.iter().any(|previous| previous.event_id == referenced) {
                    return Err(CreativeEcologyError::ForwardLineageReference);
                }
            }
            self.validate_action_targets_against(&event.action, &prior_events)?;
            prior_events.push(event.clone());
        }
        Ok(())
    }

    fn validate_action_targets_against(
        &self,
        action: &CoCreativeActionV1,
        prior_events: &[CoCreationEventV1],
    ) -> Result<(), CreativeEcologyError> {
        let target_id = match action {
            CoCreativeActionV1::Propose { .. } => return Ok(()),
            CoCreativeActionV1::Accept { proposal_event_id }
            | CoCreativeActionV1::Adapt {
                proposal_event_id, ..
            }
            | CoCreativeActionV1::Reject {
                proposal_event_id, ..
            }
            | CoCreativeActionV1::Abstain {
                proposal_event_id, ..
            }
            | CoCreativeActionV1::Branch {
                proposal_event_id, ..
            } => proposal_event_id,
        };
        let target = prior_events
            .iter()
            .find(|event| &event.event_id == target_id)
            .ok_or_else(|| CreativeEcologyError::UnknownLineageEvent(target_id.clone()))?;
        if target.action.produced_artifact_sha256().is_none() {
            return Err(CreativeEcologyError::TargetHasNoArtifact);
        }
        Ok(())
    }

    fn next_sequence(&self) -> u64 {
        self.events.len() as u64 + 1
    }

    pub fn participants(&self) -> &[CoCreationParticipantV1] {
        &self.participants
    }

    pub fn grants(&self) -> &[ConsentGrantV1] {
        &self.grants
    }

    pub fn revocations(&self) -> &[ConsentRevocationV1] {
        &self.revocations
    }

    pub fn events(&self) -> &[CoCreationEventV1] {
        &self.events
    }
}

// ─── Creative identity lineage ──────────────────────────────────────────────

/// One version of Melothaea's creative identity profile.
///
/// The profile content is stored elsewhere; this record binds it to an exact
/// digest, parent revision, and the co-creation events that influenced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreativeIdentityRevisionV1 {
    pub revision_id: String,
    pub revision: u64,
    pub parent_revision_id: Option<String>,
    pub profile_sha256: String,
    pub source_session_id: Option<String>,
    pub influence_event_ids: Vec<String>,
}

impl CreativeIdentityRevisionV1 {
    pub fn validate(&self) -> Result<(), CreativeEcologyError> {
        non_empty("revision_id", &self.revision_id)?;
        if self.revision == 0 {
            return Err(CreativeEcologyError::InvalidRevision);
        }
        sha256_hex("profile_sha256", &self.profile_sha256)?;
        match &self.parent_revision_id {
            Some(parent) => {
                non_empty("parent_revision_id", parent)?;
                if parent == &self.revision_id || self.revision <= 1 {
                    return Err(CreativeEcologyError::InvalidIdentityLineage);
                }
            }
            None if self.revision != 1 => {
                return Err(CreativeEcologyError::InvalidIdentityLineage);
            }
            None => {}
        }
        if let Some(session_id) = &self.source_session_id {
            non_empty("source_session_id", session_id)?;
            if self.influence_event_ids.is_empty() {
                return Err(CreativeEcologyError::MissingInfluenceEvidence);
            }
        }
        unique_strings("influence_event_id", &self.influence_event_ids)?;
        Ok(())
    }

    /// Validate that every claimed influence belongs to the exact source
    /// session, rather than merely checking that the event identifiers exist.
    pub fn validate_against_session(
        &self,
        session: &CoCreationSessionV1,
    ) -> Result<(), CreativeEcologyError> {
        self.validate()?;
        session.validate_snapshot()?;
        if self.source_session_id.as_deref() != Some(session.session_id.as_str()) {
            return Err(CreativeEcologyError::IdentitySessionMismatch);
        }
        for event_id in &self.influence_event_ids {
            if !session.events.iter().any(|event| &event.event_id == event_id) {
                return Err(CreativeEcologyError::UnknownLineageEvent(event_id.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreativeEcologyError {
    EmptyIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidRevision,
    InvalidSequence,
    ProcessHasNoInputs(String),
    ProcessHasNoOutputs(String),
    UnknownComponent {
        process_id: String,
        component_id: String,
    },
    UndeclaredExternalDependency {
        process_id: String,
        dependency_id: String,
    },
    UnknownParticipant,
    ParticipantLimitReached,
    EmptyConsentPurposes,
    DuplicateConsentPurpose,
    ConsentAlreadyExpired,
    UnknownConsentGrant,
    ConsentAlreadyRevoked,
    ConsentOwnerMismatch,
    ConsentNotActive,
    ConsentPurposeNotGranted(ConsentPurposeV1),
    UnknownLineageEvent(String),
    ForwardLineageReference,
    TargetHasNoArtifact,
    InvalidIdentityLineage,
    MissingInfluenceEvidence,
    IdentitySessionMismatch,
}

impl fmt::Display for CreativeEcologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(name) => write!(f, "{name} must not be empty"),
            Self::DuplicateIdentifier(name) => write!(f, "duplicate {name}"),
            Self::InvalidDigest(name) => {
                write!(f, "{name} must be a 64-character SHA-256 hex digest")
            }
            Self::InvalidRevision => write!(f, "revision must be greater than zero"),
            Self::InvalidSequence => write!(f, "event sequence or identifier is invalid"),
            Self::ProcessHasNoInputs(id) => write!(f, "process {id} has no declared inputs"),
            Self::ProcessHasNoOutputs(id) => write!(f, "process {id} has no declared outputs"),
            Self::UnknownComponent {
                process_id,
                component_id,
            } => write!(
                f,
                "process {process_id} references unknown component {component_id}"
            ),
            Self::UndeclaredExternalDependency {
                process_id,
                dependency_id,
            } => write!(
                f,
                "process {process_id} references undeclared external dependency {dependency_id}"
            ),
            Self::UnknownParticipant => {
                write!(f, "participant is not registered in this session")
            }
            Self::ParticipantLimitReached => {
                write!(f, "co-creation participant limit reached")
            }
            Self::EmptyConsentPurposes => {
                write!(f, "consent must grant at least one explicit purpose")
            }
            Self::DuplicateConsentPurpose => {
                write!(f, "consent contains a duplicate purpose")
            }
            Self::ConsentAlreadyExpired => write!(f, "consent grant is already expired"),
            Self::UnknownConsentGrant => write!(f, "consent grant does not exist"),
            Self::ConsentAlreadyRevoked => write!(f, "consent grant was already revoked"),
            Self::ConsentOwnerMismatch => {
                write!(f, "consent grant belongs to a different participant")
            }
            Self::ConsentNotActive => {
                write!(f, "consent is not active for this event sequence")
            }
            Self::ConsentPurposeNotGranted(purpose) => {
                write!(f, "consent does not grant purpose {purpose:?}")
            }
            Self::UnknownLineageEvent(id) => {
                write!(f, "lineage event {id} does not exist")
            }
            Self::ForwardLineageReference => {
                write!(f, "lineage must reference an earlier event")
            }
            Self::TargetHasNoArtifact => {
                write!(f, "target event does not create an artifact")
            }
            Self::InvalidIdentityLineage => {
                write!(f, "identity revision requires a valid parent/revision relationship")
            }
            Self::MissingInfluenceEvidence => {
                write!(f, "session-derived identity revision requires influence evidence")
            }
            Self::IdentitySessionMismatch => {
                write!(f, "identity revision references a different co-creation session")
            }
        }
    }
}

impl std::error::Error for CreativeEcologyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn organization() -> CreativeOrganizationV1 {
        let components = [
            "perceptual_state",
            "appraisal_state",
            "memory_index",
            "identity_profile",
            "boundary_policy",
            "action_plan",
            "attention_policy",
            "composition_artifact",
        ]
        .into_iter()
        .map(|component_id| OrganizationComponentV1 {
            component_id: component_id.into(),
            required_for_operation: true,
            boundary_component: component_id == "boundary_policy",
        })
        .collect();

        let processes = vec![
            ProductionProcessV1 {
                process_id: "perception".into(),
                consumes: vec!["attention_policy".into(), "memory_index".into()],
                produces: vec!["perceptual_state".into()],
                external_inputs: vec!["audio_stream".into()],
            },
            ProductionProcessV1 {
                process_id: "appraisal".into(),
                consumes: vec!["perceptual_state".into(), "identity_profile".into()],
                produces: vec!["appraisal_state".into()],
                external_inputs: vec![],
            },
            ProductionProcessV1 {
                process_id: "memory_consolidation".into(),
                consumes: vec!["appraisal_state".into(), "composition_artifact".into()],
                produces: vec!["memory_index".into()],
                external_inputs: vec![],
            },
            ProductionProcessV1 {
                process_id: "identity_maintenance".into(),
                consumes: vec!["memory_index".into()],
                produces: vec!["identity_profile".into()],
                external_inputs: vec![],
            },
            ProductionProcessV1 {
                process_id: "governance".into(),
                consumes: vec!["identity_profile".into()],
                produces: vec!["boundary_policy".into()],
                external_inputs: vec!["consent_registry".into()],
            },
            ProductionProcessV1 {
                process_id: "policy".into(),
                consumes: vec![
                    "identity_profile".into(),
                    "appraisal_state".into(),
                    "boundary_policy".into(),
                ],
                produces: vec!["action_plan".into(), "attention_policy".into()],
                external_inputs: vec![],
            },
            ProductionProcessV1 {
                process_id: "composition".into(),
                consumes: vec!["action_plan".into()],
                produces: vec!["composition_artifact".into()],
                external_inputs: vec![],
            },
        ];

        CreativeOrganizationV1 {
            organization_id: "melothaea-musical-agent".into(),
            revision: 1,
            components,
            processes,
            external_dependencies: vec!["audio_stream".into(), "consent_registry".into()],
        }
    }

    fn session_with_participants() -> CoCreationSessionV1 {
        let mut session = CoCreationSessionV1::new("session-1").unwrap();
        session
            .register_participant(CoCreationParticipantV1 {
                participant_id: "artist".into(),
                kind: ContributorKindV1::Human,
            })
            .unwrap();
        session
            .register_participant(CoCreationParticipantV1 {
                participant_id: "melothaea".into(),
                kind: ContributorKindV1::Melothaea,
            })
            .unwrap();
        session
    }

    #[test]
    fn complete_internal_process_cycle_is_functionally_recurrent() {
        let report = organization().assess_functional_recurrence().unwrap();
        assert!(report.functionally_recurrent);
        assert!(report.process_network_strongly_connected);
        assert!(report.required_components_without_producer.is_empty());
        assert!(report.required_components_without_consumer.is_empty());
        assert!(report.boundary_components_without_producer.is_empty());
        assert!(report.boundary_components_without_consumer.is_empty());
        assert!(report.internal_feedback_edge_count >= 7);
    }

    #[test]
    fn missing_boundary_producer_fails_closed() {
        let mut organization = organization();
        organization
            .processes
            .retain(|process| process.process_id != "governance");
        let report = organization.assess_functional_recurrence().unwrap();
        assert!(!report.functionally_recurrent);
        assert!(report
            .boundary_components_without_producer
            .contains(&"boundary_policy".to_string()));
    }

    #[test]
    fn disconnected_maintenance_process_is_detected() {
        let mut organization = organization();
        organization.components.push(OrganizationComponentV1 {
            component_id: "isolated_audit".into(),
            required_for_operation: true,
            boundary_component: false,
        });
        organization.processes.push(ProductionProcessV1 {
            process_id: "isolated_audit_process".into(),
            consumes: vec!["perceptual_state".into()],
            produces: vec!["isolated_audit".into()],
            external_inputs: vec![],
        });
        let report = organization.assess_functional_recurrence().unwrap();
        assert!(!report.process_network_strongly_connected);
        assert!(!report.functionally_recurrent);
        assert!(report.required_components_without_producer.is_empty());
    }

    #[test]
    fn required_component_without_consumer_fails_closed() {
        let mut organization = organization();
        organization.components.push(OrganizationComponentV1 {
            component_id: "unused_generated_state".into(),
            required_for_operation: true,
            boundary_component: false,
        });
        organization.processes.push(ProductionProcessV1 {
            process_id: "unused_state_producer".into(),
            consumes: vec!["perceptual_state".into()],
            produces: vec!["unused_generated_state".into()],
            external_inputs: vec![],
        });
        let report = organization.assess_functional_recurrence().unwrap();
        assert!(report
            .required_components_without_consumer
            .contains(&"unused_generated_state".to_string()));
        assert!(!report.functionally_recurrent);
    }

    #[test]
    fn undeclared_external_dependency_is_rejected() {
        let mut organization = organization();
        organization.external_dependencies.clear();
        assert!(matches!(
            organization.validate(),
            Err(CreativeEcologyError::UndeclaredExternalDependency { .. })
        ));
    }

    #[test]
    fn propose_requires_explicit_contribution_consent_and_binds_artifacts() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-contribute",
                "artist",
                vec![ConsentPurposeV1::Contribute],
                None,
            )
            .unwrap();
        let event = session
            .append_action(
                "artist",
                "artist-contribute",
                CoCreativeActionV1::Propose {
                    parent_artifact_sha256: None,
                    result_artifact_sha256: "a".repeat(64),
                    influence_event_ids: vec![],
                },
            )
            .unwrap();
        assert_eq!(event.sequence, 1);
        assert_eq!(event.event_id, "session-1:1");
        assert_eq!(session.events().len(), 1);
        assert!(session.validate_snapshot().is_ok());
    }

    #[test]
    fn missing_consent_or_wrong_owner_is_rejected() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-review",
                "artist",
                vec![ConsentPurposeV1::Review],
                None,
            )
            .unwrap();
        let result = session.append_action(
            "artist",
            "artist-review",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: "a".repeat(64),
                influence_event_ids: vec![],
            },
        );
        assert_eq!(
            result,
            Err(CreativeEcologyError::ConsentPurposeNotGranted(
                ConsentPurposeV1::Contribute
            ))
        );

        let result = session.append_action(
            "melothaea",
            "artist-review",
            CoCreativeActionV1::Accept {
                proposal_event_id: "session-1:1".into(),
            },
        );
        assert_eq!(result, Err(CreativeEcologyError::ConsentOwnerMismatch));
    }

    #[test]
    fn adaptation_and_branch_keep_parent_lineage_separate() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-actions",
                "artist",
                vec![
                    ConsentPurposeV1::Contribute,
                    ConsentPurposeV1::Transform,
                    ConsentPurposeV1::Review,
                ],
                None,
            )
            .unwrap();
        let proposal = session
            .append_action(
                "artist",
                "artist-actions",
                CoCreativeActionV1::Propose {
                    parent_artifact_sha256: None,
                    result_artifact_sha256: "a".repeat(64),
                    influence_event_ids: vec![],
                },
            )
            .unwrap();
        let branch = session
            .append_action(
                "artist",
                "artist-actions",
                CoCreativeActionV1::Branch {
                    proposal_event_id: proposal.event_id.clone(),
                    result_artifact_sha256: "b".repeat(64),
                    influence_event_ids: vec![proposal.event_id.clone()],
                },
            )
            .unwrap();
        let adaptation = session
            .append_action(
                "artist",
                "artist-actions",
                CoCreativeActionV1::Adapt {
                    proposal_event_id: proposal.event_id.clone(),
                    result_artifact_sha256: "c".repeat(64),
                    influence_event_ids: vec![branch.event_id.clone()],
                },
            )
            .unwrap();
        assert_eq!(branch.sequence, 2);
        assert_eq!(adaptation.sequence, 3);
        assert_eq!(session.events().len(), 3);
        assert_eq!(session.events()[1].event_id, branch.event_id);
        assert_eq!(session.events()[2].event_id, adaptation.event_id);
        assert!(session.validate_snapshot().is_ok());
    }

    #[test]
    fn consent_revocation_blocks_future_use_but_retains_history() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-contribute",
                "artist",
                vec![ConsentPurposeV1::Contribute],
                None,
            )
            .unwrap();
        session
            .append_action(
                "artist",
                "artist-contribute",
                CoCreativeActionV1::Propose {
                    parent_artifact_sha256: None,
                    result_artifact_sha256: "a".repeat(64),
                    influence_event_ids: vec![],
                },
            )
            .unwrap();
        let revocation = session
            .revoke_consent("revoke-1", "artist", "artist-contribute")
            .unwrap();
        assert_eq!(revocation.effective_sequence, 2);
        let result = session.append_action(
            "artist",
            "artist-contribute",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: Some("a".repeat(64)),
                result_artifact_sha256: "b".repeat(64),
                influence_event_ids: vec!["session-1:1".into()],
            },
        );
        assert_eq!(result, Err(CreativeEcologyError::ConsentNotActive));
        assert_eq!(session.events().len(), 1);
        assert_eq!(session.revocations().len(), 1);
        assert!(session.validate_snapshot().is_ok());
    }

    #[test]
    fn only_the_grant_owner_can_revoke_consent() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-contribute",
                "artist",
                vec![ConsentPurposeV1::Contribute],
                None,
            )
            .unwrap();
        assert_eq!(
            session.revoke_consent("revoke-1", "melothaea", "artist-contribute"),
            Err(CreativeEcologyError::ConsentOwnerMismatch)
        );
        assert!(session.events().is_empty());
        assert!(session.revocations().is_empty());
    }

    #[test]
    fn influence_must_reference_an_existing_event() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-contribute",
                "artist",
                vec![ConsentPurposeV1::Contribute],
                None,
            )
            .unwrap();
        let result = session.append_action(
            "artist",
            "artist-contribute",
            CoCreativeActionV1::Propose {
                parent_artifact_sha256: None,
                result_artifact_sha256: "a".repeat(64),
                influence_event_ids: vec!["invented-event".into()],
            },
        );
        assert_eq!(
            result,
            Err(CreativeEcologyError::UnknownLineageEvent(
                "invented-event".into()
            ))
        );
    }

    #[test]
    fn identity_revision_binds_influence_to_exact_session() {
        let mut session = session_with_participants();
        session
            .grant_consent(
                "artist-contribute",
                "artist",
                vec![ConsentPurposeV1::Contribute],
                None,
            )
            .unwrap();
        let event = session
            .append_action(
                "artist",
                "artist-contribute",
                CoCreativeActionV1::Propose {
                    parent_artifact_sha256: None,
                    result_artifact_sha256: "a".repeat(64),
                    influence_event_ids: vec![],
                },
            )
            .unwrap();
        let revision = CreativeIdentityRevisionV1 {
            revision_id: "identity-2".into(),
            revision: 2,
            parent_revision_id: Some("identity-1".into()),
            profile_sha256: "d".repeat(64),
            source_session_id: Some(session.session_id.clone()),
            influence_event_ids: vec![event.event_id],
        };
        assert!(revision.validate_against_session(&session).is_ok());

        let mut wrong_session = revision.clone();
        wrong_session.source_session_id = Some("other-session".into());
        assert_eq!(
            wrong_session.validate_against_session(&session),
            Err(CreativeEcologyError::IdentitySessionMismatch)
        );

        let mut unknown_event = revision;
        unknown_event.influence_event_ids = vec!["other-session:1".into()];
        assert_eq!(
            unknown_event.validate_against_session(&session),
            Err(CreativeEcologyError::UnknownLineageEvent(
                "other-session:1".into()
            ))
        );
    }

    #[test]
    fn malformed_identity_revision_is_rejected() {
        let revision = CreativeIdentityRevisionV1 {
            revision_id: "identity-2".into(),
            revision: 2,
            parent_revision_id: Some("identity-1".into()),
            profile_sha256: "d".repeat(64),
            source_session_id: Some("session-1".into()),
            influence_event_ids: vec!["session-1:1".into()],
        };
        assert!(revision.validate().is_ok());

        let mut missing_influence = revision;
        missing_influence.influence_event_ids.clear();
        assert_eq!(
            missing_influence.validate(),
            Err(CreativeEcologyError::MissingInfluenceEvidence)
        );
    }
}

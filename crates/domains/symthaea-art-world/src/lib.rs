// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Host-neutral creative-world contract for Symthaea.
//!
//! This crate deliberately does **not** define what art is, what is beautiful,
//! or which intervention is preferred. It defines the boundary between an
//! artistic cognitive system and a host such as a raster canvas, Bevy, or
//! Blender: observations, affordances, proposals, counterfactual previews,
//! explicit authority, revision-bound commits, and append-only receipts.
//!
//! The central invariant is simple: **perception, proposal, and authority are
//! separate capabilities**. A host adapter must never turn "Symthaea can see
//! this" into "Symthaea may mutate this" by implication.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Versioned semantic contract shared by all studio hosts.
pub const ART_WORLD_SCHEMA_V1: &str = "symthaea.art-world.v1";

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }
    };
}

id_type!(WorldId);
id_type!(RevisionId);
id_type!(EntityId);
id_type!(ActionId);
id_type!(ProposalId);
id_type!(IntentId);
id_type!(BranchId);
id_type!(EventId);

/// Concrete host implementing the creative-world contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostKind {
    Canvas,
    Bevy,
    Blender,
    Other(String),
}

/// Mutating authority granted to the artistic system in the current session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityMode {
    /// Perceive and critique only. No proposals or mutations.
    Observe,
    /// Create reversible proposals/previews. Commit requires explicit acceptance.
    Propose,
    /// Mutate an explicitly designated autonomous workspace. Commits still emit receipts.
    Author,
}

/// Evidence establishing that a proposal is allowed to become a committed mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitAuthority {
    /// A named actor explicitly accepted this proposal.
    ExplicitAcceptance { actor: String },
    /// An autonomous-author policy granted mutation authority for this workspace.
    AutonomousAuthor { policy: String },
    /// A preregistered experiment explicitly granted mutation authority.
    ExperimentPermit { protocol: String },
}

/// Immutable identity of one host revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldRevision {
    pub world_id: WorldId,
    pub revision_id: RevisionId,
    /// Monotonic host-local sequence number.
    pub sequence: u64,
    /// Host-defined digest of the committed artifact/scene state.
    pub content_hash: String,
}

/// Small, host-neutral entity description. Rich host-native data remains in the adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySummary {
    pub entity_id: EntityId,
    pub kind: String,
    pub label: Option<String>,
    pub parent: Option<EntityId>,
    pub visible: bool,
    pub metadata: BTreeMap<String, String>,
}

/// Read-only scene/artifact observation at one exact committed revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub schema: String,
    pub host: HostKind,
    pub revision: WorldRevision,
    pub selected_entities: Vec<EntityId>,
    pub entities: Vec<EntitySummary>,
    pub metadata: BTreeMap<String, String>,
}

impl WorldSnapshot {
    pub fn new(host: HostKind, revision: WorldRevision) -> Self {
        Self {
            schema: ART_WORLD_SCHEMA_V1.to_string(),
            host,
            revision,
            selected_entities: Vec::new(),
            entities: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

/// Artistic operation vocabulary. These are semantic actions, not raw host API calls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtOperation {
    ImportArtifact,
    CreateForm,
    TransformForm,
    RemoveForm,
    JoinForms,
    SeparateForms,
    ApplyMaterial,
    AlterSurface,
    PlaceLight,
    MoveCamera,
    CreateStroke,
    EraseStroke,
    Deform,
    Repeat,
    InterruptPattern,
    Reveal,
    Occlude,
    /// A first-class artistic action: deliberately preserve the current state.
    Abstain,
}

/// Typed parameter payload. Host adapters decide which parameter names are valid
/// for each declared affordance; arbitrary executable code is intentionally absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Bool(bool),
    Text(String),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    ColorRgba([f32; 4]),
}

/// One operation the host is willing to expose in the current context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Affordance {
    pub name: String,
    pub operation: ArtOperation,
    pub target_kinds: Vec<String>,
    pub parameter_schema: BTreeMap<String, String>,
    /// Short host-generated explanation of constraints/costs.
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsequenceDirection {
    Increase,
    Decrease,
    Preserve,
    Unknown,
}

/// A falsifiable prediction about what an intervention is expected to change.
/// It is evidence about a consequence, not a beauty score or preference claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictedConsequence {
    pub dimension: String,
    pub direction: ConsequenceDirection,
    pub expected_delta: Option<f64>,
    pub confidence: Option<f64>,
    pub evidence_refs: Vec<String>,
}

/// One semantic artistic intervention bound to the exact revision it was conceived against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtisticAction {
    pub action_id: ActionId,
    pub parent_revision: RevisionId,
    pub operation: ArtOperation,
    pub targets: Vec<EntityId>,
    pub parameters: BTreeMap<String, ParameterValue>,
    pub intent_id: Option<IntentId>,
    pub rationale: Option<String>,
    pub predicted_consequences: Vec<PredictedConsequence>,
}

impl ArtisticAction {
    pub fn abstain(action_id: impl Into<ActionId>, revision: impl Into<RevisionId>) -> Self {
        Self {
            action_id: action_id.into(),
            parent_revision: revision.into(),
            operation: ArtOperation::Abstain,
            targets: Vec::new(),
            parameters: BTreeMap::new(),
            intent_id: None,
            rationale: None,
            predicted_consequences: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalState {
    Proposed,
    Accepted,
    Rejected,
    Expired,
    Applied,
}

/// Reversible proposal. A proposal is not a committed mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionProposal {
    pub proposal_id: ProposalId,
    pub action: ArtisticAction,
    pub state: ProposalState,
    pub created_at_sequence: u64,
    pub decision_actor: Option<String>,
    pub decision_reason: Option<String>,
}

impl ActionProposal {
    pub fn new(
        proposal_id: impl Into<ProposalId>,
        action: ArtisticAction,
        created_at_sequence: u64,
    ) -> Self {
        Self {
            proposal_id: proposal_id.into(),
            action,
            state: ProposalState::Proposed,
            created_at_sequence,
            decision_actor: None,
            decision_reason: None,
        }
    }

    pub fn accept(&mut self, actor: impl Into<String>) -> Result<(), ProposalError> {
        self.require_proposed()?;
        self.state = ProposalState::Accepted;
        self.decision_actor = Some(actor.into());
        Ok(())
    }

    pub fn reject(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), ProposalError> {
        self.require_proposed()?;
        self.state = ProposalState::Rejected;
        self.decision_actor = Some(actor.into());
        self.decision_reason = Some(reason.into());
        Ok(())
    }

    pub fn expire(&mut self, reason: impl Into<String>) -> Result<(), ProposalError> {
        self.require_proposed()?;
        self.state = ProposalState::Expired;
        self.decision_reason = Some(reason.into());
        Ok(())
    }

    pub fn mark_applied(&mut self) -> Result<(), ProposalError> {
        if self.state != ProposalState::Accepted {
            return Err(ProposalError::InvalidTransition {
                from: self.state,
                to: ProposalState::Applied,
            });
        }
        self.state = ProposalState::Applied;
        Ok(())
    }

    fn require_proposed(&self) -> Result<(), ProposalError> {
        if self.state == ProposalState::Proposed {
            Ok(())
        } else {
            Err(ProposalError::InvalidTransition {
                from: self.state,
                to: ProposalState::Proposed,
            })
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProposalError {
    #[error("invalid proposal transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: ProposalState,
        to: ProposalState,
    },
}

/// Preview/render produced from a non-committed branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub media_type: String,
    pub locator: String,
    pub digest: Option<String>,
}

/// Counterfactual branch evaluated without mutating the committed base revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualBranch {
    pub branch_id: BranchId,
    pub base_revision: RevisionId,
    pub proposals: Vec<ProposalId>,
    pub previews: Vec<ArtifactRef>,
    pub observation_summary: BTreeMap<String, ParameterValue>,
}

/// Append-only causal/audit record emitted by host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldEventKind {
    Observed,
    ProposalCreated,
    PreviewCreated,
    ProposalAccepted,
    ProposalRejected,
    ProposalExpired,
    CommitApplied,
    CommitFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldEvent {
    pub event_id: EventId,
    pub sequence: u64,
    pub kind: WorldEventKind,
    pub revision: RevisionId,
    pub proposal_id: Option<ProposalId>,
    pub actor: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

/// Central authority check shared by host adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityGate {
    mode: AuthorityMode,
}

impl AuthorityGate {
    pub fn new(mode: AuthorityMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> AuthorityMode {
        self.mode
    }

    pub fn validate_proposal(&self) -> Result<(), AuthorityError> {
        match self.mode {
            AuthorityMode::Observe => Err(AuthorityError::ProposalNotPermitted),
            AuthorityMode::Propose | AuthorityMode::Author => Ok(()),
        }
    }

    pub fn validate_commit(&self, permit: &CommitAuthority) -> Result<(), AuthorityError> {
        match self.mode {
            AuthorityMode::Observe => Err(AuthorityError::CommitNotPermitted),
            AuthorityMode::Propose => match permit {
                CommitAuthority::ExplicitAcceptance { .. } => Ok(()),
                _ => Err(AuthorityError::ExplicitAcceptanceRequired),
            },
            AuthorityMode::Author => Ok(()),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AuthorityError {
    #[error("observe-only sessions cannot create proposals")]
    ProposalNotPermitted,
    #[error("this session cannot commit mutations")]
    CommitNotPermitted,
    #[error("proposal mode requires explicit acceptance before commit")]
    ExplicitAcceptanceRequired,
}

#[derive(Debug, Error)]
pub enum HostError {
    #[error(transparent)]
    Authority(#[from] AuthorityError),
    #[error(transparent)]
    Proposal(#[from] ProposalError),
    #[error("proposal was created against revision {expected:?}, host is now at {actual:?}")]
    WrongRevision {
        expected: RevisionId,
        actual: RevisionId,
    },
    #[error("host does not support operation {0:?} in the current context")]
    UnsupportedOperation(ArtOperation),
    #[error("host error: {0}")]
    Host(String),
}

/// Minimum contract for any environment Symthaea can inhabit as an artist.
///
/// Implementations should be deterministic where the host permits it and must
/// bind previews/commits to exact revisions. `preview` must never mutate the
/// committed revision.
pub trait CreativeWorldHost {
    fn authority(&self) -> AuthorityGate;
    fn snapshot(&mut self) -> Result<WorldSnapshot, HostError>;
    fn affordances(&mut self, snapshot: &WorldSnapshot) -> Result<Vec<Affordance>, HostError>;
    fn propose(&mut self, action: ArtisticAction) -> Result<ActionProposal, HostError>;
    fn preview(&mut self, proposal: &ActionProposal) -> Result<CounterfactualBranch, HostError>;
    fn commit(
        &mut self,
        proposal: &ActionProposal,
        permit: CommitAuthority,
    ) -> Result<WorldRevision, HostError>;
    fn reject(
        &mut self,
        proposal: &ActionProposal,
        actor: &str,
        reason: &str,
    ) -> Result<(), HostError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_mode_cannot_propose_or_commit() {
        let gate = AuthorityGate::new(AuthorityMode::Observe);
        assert_eq!(
            gate.validate_proposal(),
            Err(AuthorityError::ProposalNotPermitted)
        );
        assert_eq!(
            gate.validate_commit(&CommitAuthority::ExplicitAcceptance {
                actor: "human".into(),
            }),
            Err(AuthorityError::CommitNotPermitted)
        );
    }

    #[test]
    fn proposal_mode_requires_explicit_acceptance() {
        let gate = AuthorityGate::new(AuthorityMode::Propose);
        assert!(gate.validate_proposal().is_ok());
        assert!(
            gate.validate_commit(&CommitAuthority::ExplicitAcceptance {
                actor: "collaborator".into(),
            })
            .is_ok()
        );
        assert_eq!(
            gate.validate_commit(&CommitAuthority::AutonomousAuthor {
                policy: "studio".into(),
            }),
            Err(AuthorityError::ExplicitAcceptanceRequired)
        );
    }

    #[test]
    fn author_mode_still_requires_an_explicit_permit_record() {
        let gate = AuthorityGate::new(AuthorityMode::Author);
        assert!(
            gate.validate_commit(&CommitAuthority::AutonomousAuthor {
                policy: "autonomous-study-v1".into(),
            })
            .is_ok()
        );
    }

    #[test]
    fn proposal_lifecycle_is_monotonic() {
        let action = ArtisticAction::abstain("a1", "r1");
        let mut proposal = ActionProposal::new("p1", action, 7);
        proposal.accept("artist").unwrap();
        proposal.mark_applied().unwrap();
        assert_eq!(proposal.state, ProposalState::Applied);
        assert!(proposal.reject("artist", "late reversal").is_err());
    }

    #[test]
    fn abstention_is_a_first_class_action() {
        let action = ArtisticAction::abstain("a1", "r1");
        assert_eq!(action.operation, ArtOperation::Abstain);
        assert!(action.targets.is_empty());
    }

    #[test]
    fn snapshots_carry_schema_and_revision_identity() {
        let revision = WorldRevision {
            world_id: WorldId::from("studio"),
            revision_id: RevisionId::from("r42"),
            sequence: 42,
            content_hash: "abc".into(),
        };
        let snapshot = WorldSnapshot::new(HostKind::Blender, revision.clone());
        assert_eq!(snapshot.schema, ART_WORLD_SCHEMA_V1);
        assert_eq!(snapshot.revision, revision);
    }
}

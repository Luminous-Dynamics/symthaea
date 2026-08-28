// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transport-neutral Blender side of the Symthaea Studio Runtime.
//!
//! This crate defines a narrow, versioned protocol between the Rust artistic
//! runtime and a thin Blender add-on. It intentionally exposes semantic art
//! operations rather than arbitrary Python, `bpy` expressions, shell commands,
//! or eval strings. The add-on is expected to translate the closed operation
//! vocabulary into concrete Blender API calls.
//!
//! Transport (Unix socket, stdio, WebSocket, etc.) is deliberately left out of
//! v1 so the authority/revision contract can be qualified independently.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_art_world::{
    ActionProposal, Affordance, ArtOperation, ArtifactRef, ArtisticAction, AuthorityError,
    AuthorityGate, AuthorityMode, CommitAuthority, CounterfactualBranch, ProposalId, RevisionId,
    WorldRevision, WorldSnapshot,
};
use thiserror::Error;

pub const BLENDER_PROTOCOL_V1: &str = "symthaea.blender-bridge.v1";

/// Host capabilities returned during handshake. Capability discovery is part
/// of the protocol so artistic cognition does not assume every Blender build
/// exposes the same tools or add-ons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlenderCapabilityManifest {
    pub protocol_version: String,
    pub blender_version: String,
    pub supported_operations: Vec<ArtOperation>,
    pub supports_preview_render: bool,
    pub supports_grease_pencil: bool,
    pub supports_geometry_nodes: bool,
}

impl BlenderCapabilityManifest {
    pub fn minimal(blender_version: impl Into<String>) -> Self {
        Self {
            protocol_version: BLENDER_PROTOCOL_V1.to_string(),
            blender_version: blender_version.into(),
            supported_operations: vec![
                ArtOperation::CreateForm,
                ArtOperation::TransformForm,
                ArtOperation::RemoveForm,
                ArtOperation::ApplyMaterial,
                ArtOperation::PlaceLight,
                ArtOperation::MoveCamera,
                ArtOperation::CreateStroke,
                ArtOperation::Deform,
                ArtOperation::Abstain,
            ],
            supports_preview_render: true,
            supports_grease_pencil: true,
            supports_geometry_nodes: false,
        }
    }
}

/// Requests sent from Symthaea to the Blender adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlenderRequest {
    Handshake {
        protocol_version: String,
        client_name: String,
    },
    Observe {
        include_preview_render: bool,
    },
    ListAffordances {
        snapshot_revision: RevisionId,
    },
    Propose {
        action: ArtisticAction,
    },
    Preview {
        proposal_id: ProposalId,
    },
    Commit {
        proposal_id: ProposalId,
        permit: CommitAuthority,
    },
    Reject {
        proposal_id: ProposalId,
        actor: String,
        reason: String,
    },
}

/// Responses returned by the Blender adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlenderResponse {
    HandshakeAck {
        capabilities: BlenderCapabilityManifest,
    },
    Snapshot {
        snapshot: WorldSnapshot,
        render: Option<ArtifactRef>,
    },
    Affordances {
        revision: RevisionId,
        affordances: Vec<Affordance>,
    },
    ProposalCreated {
        proposal: ActionProposal,
    },
    PreviewCreated {
        branch: CounterfactualBranch,
    },
    CommitApplied {
        revision: WorldRevision,
    },
    Rejected {
        proposal_id: ProposalId,
    },
    Error {
        code: String,
        message: String,
    },
}

/// Closed symbolic mapping understood by a Blender add-on. These strings are
/// protocol keys, never executable source code.
pub fn blender_operator_key(operation: &ArtOperation) -> Option<&'static str> {
    match operation {
        ArtOperation::ImportArtifact => Some("artifact.import"),
        ArtOperation::CreateForm => Some("form.create"),
        ArtOperation::TransformForm => Some("form.transform"),
        ArtOperation::RemoveForm => Some("form.remove"),
        ArtOperation::JoinForms => Some("form.join"),
        ArtOperation::SeparateForms => Some("form.separate"),
        ArtOperation::ApplyMaterial => Some("material.apply"),
        ArtOperation::AlterSurface => Some("surface.alter"),
        ArtOperation::PlaceLight => Some("light.place"),
        ArtOperation::MoveCamera => Some("camera.move"),
        ArtOperation::CreateStroke => Some("stroke.create"),
        ArtOperation::EraseStroke => Some("stroke.erase"),
        ArtOperation::Deform => Some("form.deform"),
        ArtOperation::Repeat => Some("pattern.repeat"),
        ArtOperation::InterruptPattern => Some("pattern.interrupt"),
        ArtOperation::Reveal => Some("visibility.reveal"),
        ArtOperation::Occlude => Some("visibility.occlude"),
        ArtOperation::Abstain => Some("world.abstain"),
    }
}

/// Validate mutation-bearing requests before they reach Blender.
///
/// Observe/list/preview requests remain read-only. `Propose` requires at least
/// proposal authority. `Commit` is checked against the session's authority
/// mode and its explicit permit.
pub fn validate_request_authority(
    mode: AuthorityMode,
    request: &BlenderRequest,
) -> Result<(), AuthorityError> {
    let gate = AuthorityGate::new(mode);
    match request {
        BlenderRequest::Propose { .. } => gate.validate_proposal(),
        BlenderRequest::Commit { permit, .. } => gate.validate_commit(permit),
        _ => Ok(()),
    }
}

/// Newline-delimited JSON codec suitable for stdio or framed socket transports.
pub struct JsonLineCodec;

impl JsonLineCodec {
    pub fn encode_request(request: &BlenderRequest) -> Result<String, BridgeError> {
        encode_line(request)
    }

    pub fn decode_request(line: &str) -> Result<BlenderRequest, BridgeError> {
        decode_line(line)
    }

    pub fn encode_response(response: &BlenderResponse) -> Result<String, BridgeError> {
        encode_line(response)
    }

    pub fn decode_response(line: &str) -> Result<BlenderResponse, BridgeError> {
        decode_line(line)
    }
}

fn encode_line<T: Serialize>(value: &T) -> Result<String, BridgeError> {
    let mut encoded = serde_json::to_string(value)?;
    encoded.push('\n');
    Ok(encoded)
}

fn decode_line<T: for<'de> Deserialize<'de>>(line: &str) -> Result<T, BridgeError> {
    let trimmed = line.strip_suffix('\n').unwrap_or(line);
    if trimmed.contains('\n') || trimmed.contains('\r') {
        return Err(BridgeError::MultipleFrames);
    }
    Ok(serde_json::from_str(trimmed)?)
}

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("one JSON-line frame may not contain embedded newlines")]
    MultipleFrames,
    #[error("unsupported protocol version: {0}")]
    UnsupportedProtocol(String),
}

/// Reject incompatible clients before any host state is observed or changed.
pub fn validate_protocol(version: &str) -> Result<(), BridgeError> {
    if version == BLENDER_PROTOCOL_V1 {
        Ok(())
    } else {
        Err(BridgeError::UnsupportedProtocol(version.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_art_world::{ActionId, ArtisticAction};

    fn action() -> ArtisticAction {
        ArtisticAction {
            action_id: ActionId::from("a1"),
            parent_revision: RevisionId::from("r1"),
            operation: ArtOperation::Deform,
            targets: vec![],
            parameters: BTreeMap::new(),
            intent_id: None,
            rationale: Some("break a too-perfect symmetry".into()),
            predicted_consequences: vec![],
        }
    }

    #[test]
    fn request_round_trip_is_single_frame() {
        let request = BlenderRequest::Propose { action: action() };
        let encoded = JsonLineCodec::encode_request(&request).unwrap();
        assert!(encoded.ends_with('\n'));
        assert_eq!(encoded.matches('\n').count(), 1);
        assert_eq!(JsonLineCodec::decode_request(&encoded).unwrap(), request);
    }

    #[test]
    fn observe_mode_rejects_proposals() {
        let request = BlenderRequest::Propose { action: action() };
        assert_eq!(
            validate_request_authority(AuthorityMode::Observe, &request),
            Err(AuthorityError::ProposalNotPermitted)
        );
    }

    #[test]
    fn propose_mode_rejects_autonomous_commit() {
        let request = BlenderRequest::Commit {
            proposal_id: ProposalId::from("p1"),
            permit: CommitAuthority::AutonomousAuthor {
                policy: "studio".into(),
            },
        };
        assert_eq!(
            validate_request_authority(AuthorityMode::Propose, &request),
            Err(AuthorityError::ExplicitAcceptanceRequired)
        );
    }

    #[test]
    fn operator_mapping_is_closed_and_symbolic() {
        for op in BlenderCapabilityManifest::minimal("4.x").supported_operations {
            let key = blender_operator_key(&op).expect("advertised operation must map");
            assert!(!key.contains('('));
            assert!(!key.contains(';'));
            assert!(!key.contains('\n'));
        }
    }

    #[test]
    fn protocol_is_exactly_versioned() {
        assert!(validate_protocol(BLENDER_PROTOCOL_V1).is_ok());
        assert!(validate_protocol("symthaea.blender-bridge.v0").is_err());
    }
}

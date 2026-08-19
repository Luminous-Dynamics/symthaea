// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability negotiation for heterogeneous SCIP peers.

use crate::{
    GroundedHdcCodec, HdcProfile, InterchangeError, InterchangeRepresentation, ProtocolVersion,
    SCIP_V1,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerCapabilities {
    pub versions: Vec<ProtocolVersion>,
    pub representations: Vec<InterchangeRepresentation>,
    pub hdc_profiles: Vec<HdcProfile>,
    pub sparse_hdc_deltas: bool,
    pub semantic_references: bool,
}

impl PeerCapabilities {
    pub fn symthaea_default() -> Self {
        Self {
            versions: vec![SCIP_V1],
            representations: vec![
                InterchangeRepresentation::Hdc,
                InterchangeRepresentation::GroundedGraph,
                InterchangeRepresentation::StructuredJson,
                InterchangeRepresentation::HumanText,
            ],
            hdc_profiles: vec![GroundedHdcCodec::standard().profile().clone()],
            sparse_hdc_deltas: true,
            semantic_references: true,
        }
    }

    pub fn structured_only() -> Self {
        Self {
            versions: vec![SCIP_V1],
            representations: vec![
                InterchangeRepresentation::GroundedGraph,
                InterchangeRepresentation::StructuredJson,
                InterchangeRepresentation::HumanText,
            ],
            hdc_profiles: vec![],
            sparse_hdc_deltas: false,
            semantic_references: true,
        }
    }

    pub fn text_only() -> Self {
        Self {
            versions: vec![SCIP_V1],
            representations: vec![InterchangeRepresentation::HumanText],
            hdc_profiles: vec![],
            sparse_hdc_deltas: false,
            semantic_references: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegotiatedSession {
    pub version: ProtocolVersion,
    pub representation: InterchangeRepresentation,
    pub hdc_profile: Option<HdcProfile>,
    pub sparse_hdc_deltas: bool,
    pub semantic_references: bool,
}

pub fn negotiate(
    local: &PeerCapabilities,
    remote: &PeerCapabilities,
) -> Result<NegotiatedSession, InterchangeError> {
    let version = local
        .versions
        .iter()
        .filter(|version| remote.versions.contains(version))
        .copied()
        .max()
        .ok_or(InterchangeError::NegotiationFailed)?;

    let supports = |representation: &InterchangeRepresentation| {
        local.representations.contains(representation)
            && remote.representations.contains(representation)
    };

    if supports(&InterchangeRepresentation::Hdc) {
        if let Some(profile) = local.hdc_profiles.iter().find(|local_profile| {
            remote.hdc_profiles.iter().any(|remote_profile| {
                remote_profile.codebook_fingerprint == local_profile.codebook_fingerprint
                    && remote_profile.dimension == local_profile.dimension
                    && remote_profile.algebra == local_profile.algebra
                    && remote_profile.atom_derivation == local_profile.atom_derivation
            })
        }) {
            return Ok(NegotiatedSession {
                version,
                representation: InterchangeRepresentation::Hdc,
                hdc_profile: Some(profile.clone()),
                sparse_hdc_deltas: local.sparse_hdc_deltas && remote.sparse_hdc_deltas,
                semantic_references: local.semantic_references && remote.semantic_references,
            });
        }
    }

    for representation in [
        InterchangeRepresentation::GroundedGraph,
        InterchangeRepresentation::StructuredJson,
        InterchangeRepresentation::HumanText,
    ] {
        if supports(&representation) {
            return Ok(NegotiatedSession {
                version,
                representation,
                hdc_profile: None,
                sparse_hdc_deltas: false,
                semantic_references: local.semantic_references && remote.semantic_references,
            });
        }
    }

    Err(InterchangeError::NegotiationFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_symthaea_peers_choose_hdc() {
        let caps = PeerCapabilities::symthaea_default();
        let session = negotiate(&caps, &caps).unwrap();
        assert_eq!(session.representation, InterchangeRepresentation::Hdc);
        assert!(session.hdc_profile.is_some());
        assert!(session.sparse_hdc_deltas);
    }

    #[test]
    fn profile_mismatch_falls_back_without_guessing() {
        let local = PeerCapabilities::symthaea_default();
        let mut remote = PeerCapabilities::symthaea_default();
        remote.hdc_profiles[0].codebook_fingerprint = "different".into();

        let session = negotiate(&local, &remote).unwrap();
        assert_eq!(
            session.representation,
            InterchangeRepresentation::GroundedGraph
        );
        assert!(session.hdc_profile.is_none());
    }

    #[test]
    fn text_peer_gets_text_fallback() {
        let session = negotiate(
            &PeerCapabilities::symthaea_default(),
            &PeerCapabilities::text_only(),
        )
        .unwrap();
        assert_eq!(session.representation, InterchangeRepresentation::HumanText);
    }
}

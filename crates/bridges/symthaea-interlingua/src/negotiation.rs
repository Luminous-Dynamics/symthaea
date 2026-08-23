// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability negotiation for heterogeneous SCIP peers.

use crate::protocol::require_content_hash;
use crate::{
    GroundedHdcCodec, HdcProfile, InterchangeError, InterchangeRepresentation, ProtocolVersion,
    SCIP_V1, ScipLimits,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_ADVERTISED_VERSIONS: usize = 32;
const MAX_ADVERTISED_REPRESENTATIONS: usize = 64;
const MAX_ADVERTISED_HDC_PROFILES: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerCapabilities {
    pub versions: Vec<ProtocolVersion>,
    pub representations: Vec<InterchangeRepresentation>,
    pub hdc_profiles: Vec<HdcProfile>,
    pub sparse_hdc_deltas: bool,
    /// Support exact `GraphDelta` synchronization when an exact grounded
    /// representation is also shared by the session.
    ///
    /// `serde(default)` keeps older SCIP v1 capability advertisements readable;
    /// absence means the peer did not advertise this optional capability.
    #[serde(default)]
    pub exact_graph_deltas: bool,
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
            exact_graph_deltas: true,
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
            exact_graph_deltas: true,
            semantic_references: true,
        }
    }

    pub fn text_only() -> Self {
        Self {
            versions: vec![SCIP_V1],
            representations: vec![InterchangeRepresentation::HumanText],
            hdc_profiles: vec![],
            sparse_hdc_deltas: false,
            exact_graph_deltas: false,
            semantic_references: false,
        }
    }

    pub fn validate(&self) -> Result<(), InterchangeError> {
        self.validate_with_limits(&ScipLimits::default())
    }

    pub fn validate_with_limits(&self, limits: &ScipLimits) -> Result<(), InterchangeError> {
        if self.versions.is_empty() || self.versions.len() > MAX_ADVERTISED_VERSIONS {
            return Err(InterchangeError::NegotiationFailed);
        }
        if self.representations.is_empty()
            || self.representations.len() > MAX_ADVERTISED_REPRESENTATIONS
        {
            return Err(InterchangeError::NegotiationFailed);
        }
        if self.hdc_profiles.len() > MAX_ADVERTISED_HDC_PROFILES {
            return Err(InterchangeError::NegotiationFailed);
        }

        let mut versions = BTreeSet::new();
        if self.versions.iter().any(|version| !versions.insert(*version)) {
            return Err(InterchangeError::NegotiationFailed);
        }

        let mut representations = BTreeSet::new();
        for representation in &self.representations {
            let key = representation_key(representation);
            if !representations.insert(key) {
                return Err(InterchangeError::NegotiationFailed);
            }
            if let InterchangeRepresentation::Custom(value) = representation
                && (value.trim().is_empty() || value.len() > limits.max_identifier_bytes)
            {
                return Err(InterchangeError::NegotiationFailed);
            }
        }

        let advertises_hdc = self.representations.contains(&InterchangeRepresentation::Hdc);
        if advertises_hdc == self.hdc_profiles.is_empty() {
            return Err(InterchangeError::NegotiationFailed);
        }

        let mut fingerprints = BTreeSet::new();
        for profile in &self.hdc_profiles {
            if profile.dimension == 0 || profile.dimension > limits.max_hdc_dimension {
                return Err(InterchangeError::NegotiationFailed);
            }
            for value in [&profile.algebra, &profile.atom_derivation, &profile.namespace] {
                if value.trim().is_empty() || value.len() > limits.max_identifier_bytes {
                    return Err(InterchangeError::NegotiationFailed);
                }
            }
            require_content_hash(&profile.codebook_fingerprint, "HDC codebook fingerprint")?;
            if !fingerprints.insert(profile.codebook_fingerprint.as_str()) {
                return Err(InterchangeError::NegotiationFailed);
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegotiationPolicy {
    /// Fail instead of silently selecting a non-HDC representation when both
    /// peers advertise HDC but no exact profile is shared.
    pub require_hdc: bool,
    /// Permit presentation-only text fallback when no grounded representation
    /// is shared.
    pub allow_human_text_fallback: bool,
}

impl Default for NegotiationPolicy {
    fn default() -> Self {
        Self {
            require_hdc: false,
            allow_human_text_fallback: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegotiatedSession {
    pub version: ProtocolVersion,
    /// Backward-compatible preferred representation. Per-message transfer should
    /// use `shared_representations` rather than assuming this is mandatory.
    pub representation: InterchangeRepresentation,
    pub hdc_profile: Option<HdcProfile>,
    pub sparse_hdc_deltas: bool,
    /// Exact canonical graph deltas are available only when both peers advertise
    /// them and the session shares at least one exact grounded representation.
    #[serde(default)]
    pub exact_graph_deltas: bool,
    pub semantic_references: bool,
    /// Complete common representation set, ordered by SCIP preference.
    pub shared_representations: Vec<InterchangeRepresentation>,
    /// True when both peers advertised HDC but no exact HDC profile was shared.
    pub hdc_downgraded: bool,
}

pub fn negotiate(
    local: &PeerCapabilities,
    remote: &PeerCapabilities,
) -> Result<NegotiatedSession, InterchangeError> {
    negotiate_with_policy(local, remote, NegotiationPolicy::default())
}

pub fn negotiate_with_policy(
    local: &PeerCapabilities,
    remote: &PeerCapabilities,
    policy: NegotiationPolicy,
) -> Result<NegotiatedSession, InterchangeError> {
    local.validate()?;
    remote.validate()?;

    // This implementation speaks SCIP 1.0 only. Never negotiate a version just
    // because two peers happen to advertise the same future value.
    if !local.versions.contains(&SCIP_V1) || !remote.versions.contains(&SCIP_V1) {
        return Err(InterchangeError::NegotiationFailed);
    }
    let version = SCIP_V1;

    let supports = |representation: &InterchangeRepresentation| {
        local.representations.contains(representation)
            && remote.representations.contains(representation)
    };

    let both_advertise_hdc = supports(&InterchangeRepresentation::Hdc);
    let shared_hdc_profile = if both_advertise_hdc {
        local.hdc_profiles.iter().find(|local_profile| {
            remote
                .hdc_profiles
                .iter()
                .any(|remote_profile| profiles_match(local_profile, remote_profile))
        })
    } else {
        None
    };

    if policy.require_hdc && shared_hdc_profile.is_none() {
        return Err(InterchangeError::NegotiationFailed);
    }

    let mut shared_representations = Vec::new();
    if shared_hdc_profile.is_some() {
        shared_representations.push(InterchangeRepresentation::Hdc);
    }
    for representation in [
        InterchangeRepresentation::GroundedGraph,
        InterchangeRepresentation::StructuredJson,
    ] {
        if supports(&representation) {
            shared_representations.push(representation);
        }
    }
    if policy.allow_human_text_fallback && supports(&InterchangeRepresentation::HumanText) {
        shared_representations.push(InterchangeRepresentation::HumanText);
    }

    // Preserve mutually supported custom representations without letting them
    // outrank the standardized SCIP representations.
    for representation in &local.representations {
        if matches!(representation, InterchangeRepresentation::Custom(_))
            && remote.representations.contains(representation)
        {
            shared_representations.push(representation.clone());
        }
    }

    let representation = shared_representations
        .first()
        .cloned()
        .ok_or(InterchangeError::NegotiationFailed)?;
    let hdc_profile = shared_hdc_profile.cloned();
    let hdc_downgraded = both_advertise_hdc && hdc_profile.is_none();
    let shares_exact_grounded_representation = shared_representations.iter().any(|representation| {
        matches!(
            representation,
            InterchangeRepresentation::GroundedGraph | InterchangeRepresentation::StructuredJson
        )
    });

    Ok(NegotiatedSession {
        version,
        representation,
        hdc_profile,
        sparse_hdc_deltas: shared_hdc_profile.is_some()
            && local.sparse_hdc_deltas
            && remote.sparse_hdc_deltas,
        exact_graph_deltas: shares_exact_grounded_representation
            && local.exact_graph_deltas
            && remote.exact_graph_deltas,
        semantic_references: local.semantic_references && remote.semantic_references,
        shared_representations,
        hdc_downgraded,
    })
}

fn profiles_match(left: &HdcProfile, right: &HdcProfile) -> bool {
    left.codebook_fingerprint == right.codebook_fingerprint
        && left.dimension == right.dimension
        && left.algebra == right.algebra
        && left.atom_derivation == right.atom_derivation
        && left.namespace == right.namespace
}

fn representation_key(representation: &InterchangeRepresentation) -> String {
    match representation {
        InterchangeRepresentation::GroundedGraph => "grounded-graph".into(),
        InterchangeRepresentation::Hdc => "hdc".into(),
        InterchangeRepresentation::StructuredJson => "structured-json".into(),
        InterchangeRepresentation::HumanText => "human-text".into(),
        InterchangeRepresentation::Custom(value) => format!("custom:{value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_symthaea_peers_choose_hdc_and_exact_graph_delta_support() {
        let caps = PeerCapabilities::symthaea_default();
        let session = negotiate(&caps, &caps).unwrap();
        assert_eq!(session.representation, InterchangeRepresentation::Hdc);
        assert!(session.hdc_profile.is_some());
        assert!(session.sparse_hdc_deltas);
        assert!(session.exact_graph_deltas);
        assert!(!session.hdc_downgraded);
        assert!(
            session
                .shared_representations
                .contains(&InterchangeRepresentation::GroundedGraph)
        );
    }

    #[test]
    fn structured_peers_negotiate_exact_graph_deltas_without_hdc_deltas() {
        let caps = PeerCapabilities::structured_only();
        let session = negotiate(&caps, &caps).unwrap();
        assert_eq!(
            session.representation,
            InterchangeRepresentation::GroundedGraph
        );
        assert!(!session.sparse_hdc_deltas);
        assert!(session.exact_graph_deltas);
    }

    #[test]
    fn exact_graph_deltas_require_an_exact_grounded_shared_representation() {
        let mut local = PeerCapabilities::symthaea_default();
        local.representations = vec![InterchangeRepresentation::Hdc];
        let mut remote = local.clone();
        remote.exact_graph_deltas = true;

        let session = negotiate(&local, &remote).unwrap();
        assert_eq!(session.representation, InterchangeRepresentation::Hdc);
        assert!(session.sparse_hdc_deltas);
        assert!(!session.exact_graph_deltas);
    }

    #[test]
    fn exact_graph_deltas_require_bilateral_advertisement() {
        let local = PeerCapabilities::structured_only();
        let mut remote = PeerCapabilities::structured_only();
        remote.exact_graph_deltas = false;

        let session = negotiate(&local, &remote).unwrap();
        assert!(!session.exact_graph_deltas);
    }

    #[test]
    fn missing_exact_graph_delta_field_defaults_to_not_advertised() {
        let mut encoded = serde_json::to_value(PeerCapabilities::structured_only()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .remove("exact_graph_deltas");

        let decoded: PeerCapabilities = serde_json::from_value(encoded).unwrap();
        assert!(!decoded.exact_graph_deltas);
    }

    #[test]
    fn profile_mismatch_falls_back_and_records_downgrade() {
        let local = PeerCapabilities::symthaea_default();
        let mut remote = PeerCapabilities::symthaea_default();
        remote.hdc_profiles[0].codebook_fingerprint = "f".repeat(64);

        let session = negotiate(&local, &remote).unwrap();
        assert_eq!(
            session.representation,
            InterchangeRepresentation::GroundedGraph
        );
        assert!(session.hdc_profile.is_none());
        assert!(session.hdc_downgraded);
        assert!(session.exact_graph_deltas);
    }

    #[test]
    fn strict_hdc_policy_fails_closed_on_profile_mismatch() {
        let local = PeerCapabilities::symthaea_default();
        let mut remote = PeerCapabilities::symthaea_default();
        remote.hdc_profiles[0].codebook_fingerprint = "f".repeat(64);

        assert!(
            negotiate_with_policy(
                &local,
                &remote,
                NegotiationPolicy {
                    require_hdc: true,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn namespace_mismatch_never_negotiates_hdc() {
        let local = PeerCapabilities::symthaea_default();
        let mut remote = PeerCapabilities::symthaea_default();
        remote.hdc_profiles[0].namespace = "different.namespace".into();
        // A malicious or buggy peer might retain the old fingerprint. Explicit
        // namespace comparison still prevents accepting the profile.

        let session = negotiate(&local, &remote).unwrap();
        assert_ne!(session.representation, InterchangeRepresentation::Hdc);
        assert!(session.hdc_downgraded);
    }

    #[test]
    fn text_peer_gets_text_fallback_only_when_policy_allows_it() {
        let local = PeerCapabilities::symthaea_default();
        let remote = PeerCapabilities::text_only();
        let session = negotiate(&local, &remote).unwrap();
        assert_eq!(session.representation, InterchangeRepresentation::HumanText);
        assert!(!session.exact_graph_deltas);

        assert!(
            negotiate_with_policy(
                &local,
                &remote,
                NegotiationPolicy {
                    allow_human_text_fallback: false,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn future_version_is_not_accidentally_negotiated() {
        let mut local = PeerCapabilities::structured_only();
        let mut remote = PeerCapabilities::structured_only();
        let future = ProtocolVersion { major: 9, minor: 0 };
        local.versions = vec![future];
        remote.versions = vec![future];
        assert!(negotiate(&local, &remote).is_err());
    }

    #[test]
    fn malformed_capability_advertisement_is_rejected() {
        let mut caps = PeerCapabilities::symthaea_default();
        caps.representations.push(InterchangeRepresentation::Hdc);
        assert!(caps.validate().is_err());
    }
}

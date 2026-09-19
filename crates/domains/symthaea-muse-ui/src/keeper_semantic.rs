// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable semantic-timeline evidence for persisted keepers.
//!
//! A saved keeper can be auditioned from its WAV without any semantic timeline.
//! Synchronized musical A/B is a stronger claim: it requires a persisted bundle
//! that is bound to the same score/recipe/audio commitments already verified
//! through the keeper genealogy path. This module performs that cross-check and
//! refuses to reconstruct missing historical evidence with the current engine.

use std::fmt;

use gloo_net::http::Request;
use symthaea_muse_protocol::{
    ArtifactIdentity, LISTEN_COMPOSITION_BUNDLE_VERSION,
    keeper_semantic::{KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION, KeeperSemanticBundleV1},
};

use crate::comparison::MusicalComparisonAnchor;
use crate::comparison_capability::ComparisonTimelineAuthority;
use crate::comparison_timeline::{ComparisonAnchorResolutionError, resolve_musical_anchor};

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedKeeperSemanticBundle {
    pub sidecar: KeeperSemanticBundleV1,
    pub authority: ComparisonTimelineAuthority,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeeperSemanticEvidenceError {
    InvalidArtifactKey,
    RequestFailed(String),
    HttpStatus(u16),
    ParseFailed(String),
    UnsupportedSchemaVersion(u32),
    UnsupportedListenBundleVersion(u32),
    MalformedCommitment(&'static str),
    EvidenceMismatch(&'static str),
    InvalidTimeline(ComparisonAnchorResolutionError),
}

impl fmt::Display for KeeperSemanticEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArtifactKey => write!(f, "keeper audio key is not path-safe"),
            Self::RequestFailed(error) => {
                write!(f, "keeper semantic-bundle request failed: {error}")
            }
            Self::HttpStatus(status) => {
                write!(f, "keeper semantic-bundle endpoint returned HTTP {status}")
            }
            Self::ParseFailed(error) => write!(f, "failed to parse keeper semantic bundle: {error}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported keeper semantic schema version {version}")
            }
            Self::UnsupportedListenBundleVersion(version) => {
                write!(f, "unsupported Listen composition bundle version {version}")
            }
            Self::MalformedCommitment(label) => {
                write!(f, "keeper semantic {label} commitment is not SHA-256 hex")
            }
            Self::EvidenceMismatch(label) => {
                write!(f, "keeper semantic {label} evidence conflicts with verified identity")
            }
            Self::InvalidTimeline(error) => {
                write!(f, "keeper semantic timeline is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for KeeperSemanticEvidenceError {}

/// Fetch and verify the durable semantic sidecar for one keeper.
///
/// `Ok(None)` has exactly one meaning: the server reports 404 because no stored
/// sidecar exists (for example, a legacy keeper). We do not recompose or infer a
/// replacement. Any present-but-invalid sidecar is an error, never downgraded to
/// the legacy path.
pub async fn fetch_verified_keeper_semantic_bundle(
    backend: &str,
    audio_key: &str,
    verified_artifact: &ArtifactIdentity,
) -> Result<Option<VerifiedKeeperSemanticBundle>, KeeperSemanticEvidenceError> {
    if !valid_artifact_key(audio_key) {
        return Err(KeeperSemanticEvidenceError::InvalidArtifactKey);
    }

    let url = format!(
        "{}/api/keeper-semantic-bundle/{audio_key}",
        backend.trim_end_matches('/')
    );
    let response = Request::get(&url)
        .send()
        .await
        .map_err(|error| KeeperSemanticEvidenceError::RequestFailed(error.to_string()))?;

    if response.status() == 404 {
        return Ok(None);
    }
    if !response.ok() {
        return Err(KeeperSemanticEvidenceError::HttpStatus(response.status()));
    }

    let sidecar = response
        .json::<KeeperSemanticBundleV1>()
        .await
        .map_err(|error| KeeperSemanticEvidenceError::ParseFailed(error.to_string()))?;

    validate_keeper_semantic_bundle(audio_key, verified_artifact, sidecar).map(Some)
}

fn validate_keeper_semantic_bundle(
    expected_audio_key: &str,
    verified_artifact: &ArtifactIdentity,
    sidecar: KeeperSemanticBundleV1,
) -> Result<VerifiedKeeperSemanticBundle, KeeperSemanticEvidenceError> {
    let authority = validate_binding(
        expected_audio_key,
        verified_artifact,
        SidecarBinding {
            schema_version: sidecar.schema_version,
            audio_key: sidecar.audio_key.as_str(),
            listen_bundle_version: sidecar.listen_bundle_version,
            score_sha256: sidecar.score_sha256.as_str(),
            recipe_sha256: sidecar.recipe_sha256.as_str(),
            audio_sha256: sidecar.audio_sha256.as_str(),
        },
    )?;

    // Running the resolver at the first musical coordinate validates the whole
    // duration/tempo/meter authority boundary before semantic sync is granted.
    // The resolver itself checks every tempo/meter point, not only the selected
    // point, and rejects malformed/discontinuous maps or zero-length pieces.
    let origin = MusicalComparisonAnchor::new(0, 0.0)
        .expect("zero bar/beat is a valid context-free comparison anchor");
    resolve_musical_anchor(&sidecar.payload, origin)
        .map_err(KeeperSemanticEvidenceError::InvalidTimeline)?;

    Ok(VerifiedKeeperSemanticBundle { sidecar, authority })
}

#[derive(Clone, Copy)]
struct SidecarBinding<'a> {
    schema_version: u32,
    audio_key: &'a str,
    listen_bundle_version: u32,
    score_sha256: &'a str,
    recipe_sha256: &'a str,
    audio_sha256: &'a str,
}

fn validate_binding(
    expected_audio_key: &str,
    verified_artifact: &ArtifactIdentity,
    binding: SidecarBinding<'_>,
) -> Result<ComparisonTimelineAuthority, KeeperSemanticEvidenceError> {
    if binding.schema_version != KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION {
        return Err(KeeperSemanticEvidenceError::UnsupportedSchemaVersion(
            binding.schema_version,
        ));
    }
    if binding.listen_bundle_version != LISTEN_COMPOSITION_BUNDLE_VERSION {
        return Err(
            KeeperSemanticEvidenceError::UnsupportedListenBundleVersion(
                binding.listen_bundle_version,
            ),
        );
    }
    if binding.audio_key != expected_audio_key {
        return Err(KeeperSemanticEvidenceError::EvidenceMismatch("audio-key"));
    }

    for (label, hash) in [
        ("score", binding.score_sha256),
        ("recipe", binding.recipe_sha256),
        ("audio", binding.audio_sha256),
    ] {
        if !is_sha256_hex(hash) {
            return Err(KeeperSemanticEvidenceError::MalformedCommitment(label));
        }
    }

    if binding.score_sha256 != verified_artifact.score_content.0 {
        return Err(KeeperSemanticEvidenceError::EvidenceMismatch("score"));
    }
    if binding.recipe_sha256 != verified_artifact.composition.0 {
        return Err(KeeperSemanticEvidenceError::EvidenceMismatch("recipe"));
    }
    if binding.audio_sha256 != verified_artifact.rendition.0 {
        return Err(KeeperSemanticEvidenceError::EvidenceMismatch("audio"));
    }

    Ok(ComparisonTimelineAuthority::persisted_keeper(
        binding.listen_bundle_version,
    ))
}

fn valid_artifact_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 80
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::{
        CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    fn verified_artifact() -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId("a".repeat(64)),
            composition: CompositionArtifactId("b".repeat(64)),
            rendition: RenditionArtifactId("c".repeat(64)),
        }
    }

    fn binding<'a>(
        audio_key: &'a str,
        score_sha256: &'a str,
        recipe_sha256: &'a str,
        audio_sha256: &'a str,
    ) -> SidecarBinding<'a> {
        SidecarBinding {
            schema_version: KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION,
            audio_key,
            listen_bundle_version: LISTEN_COMPOSITION_BUNDLE_VERSION,
            score_sha256,
            recipe_sha256,
            audio_sha256,
        }
    }

    #[test]
    fn exact_sidecar_binding_grants_persisted_timeline_authority() {
        let score = "a".repeat(64);
        let recipe = "b".repeat(64);
        let audio = "c".repeat(64);
        let authority = validate_binding(
            "keeper-a",
            &verified_artifact(),
            binding("keeper-a", &score, &recipe, &audio),
        )
        .unwrap();

        assert_eq!(
            authority,
            ComparisonTimelineAuthority::persisted_keeper(LISTEN_COMPOSITION_BUNDLE_VERSION)
        );
    }

    #[test]
    fn sidecar_identity_mismatches_fail_closed() {
        let verified = verified_artifact();
        let good_score = "a".repeat(64);
        let good_recipe = "b".repeat(64);
        let good_audio = "c".repeat(64);
        let other = "e".repeat(64);

        for (label, candidate) in [
            (
                "audio-key",
                binding("other", &good_score, &good_recipe, &good_audio),
            ),
            (
                "score",
                binding("keeper-a", &other, &good_recipe, &good_audio),
            ),
            (
                "recipe",
                binding("keeper-a", &good_score, &other, &good_audio),
            ),
            (
                "audio",
                binding("keeper-a", &good_score, &good_recipe, &other),
            ),
        ] {
            assert_eq!(
                validate_binding("keeper-a", &verified, candidate),
                Err(KeeperSemanticEvidenceError::EvidenceMismatch(label))
            );
        }
    }

    #[test]
    fn malformed_commitment_is_not_downgraded_to_legacy_unavailable() {
        let recipe = "b".repeat(64);
        let audio = "c".repeat(64);
        assert_eq!(
            validate_binding(
                "keeper-a",
                &verified_artifact(),
                binding("keeper-a", "not-a-hash", &recipe, &audio),
            ),
            Err(KeeperSemanticEvidenceError::MalformedCommitment("score"))
        );
    }

    #[test]
    fn unsupported_versions_fail_before_semantic_authority_is_granted() {
        let score = "a".repeat(64);
        let recipe = "b".repeat(64);
        let audio = "c".repeat(64);

        let mut wrong_schema = binding("keeper-a", &score, &recipe, &audio);
        wrong_schema.schema_version = 99;
        assert_eq!(
            validate_binding("keeper-a", &verified_artifact(), wrong_schema),
            Err(KeeperSemanticEvidenceError::UnsupportedSchemaVersion(99))
        );

        let mut wrong_bundle = binding("keeper-a", &score, &recipe, &audio);
        wrong_bundle.listen_bundle_version = LISTEN_COMPOSITION_BUNDLE_VERSION + 1;
        assert_eq!(
            validate_binding("keeper-a", &verified_artifact(), wrong_bundle),
            Err(KeeperSemanticEvidenceError::UnsupportedListenBundleVersion(
                LISTEN_COMPOSITION_BUNDLE_VERSION + 1,
            ))
        );
    }

    #[test]
    fn keeper_artifact_key_boundary_matches_server_path_safety_contract() {
        assert!(valid_artifact_key("19a_keeper-17"));
        assert!(!valid_artifact_key(""));
        assert!(!valid_artifact_key("../keeper"));
        assert!(!valid_artifact_key("keeper/child"));
        assert!(!valid_artifact_key(&"a".repeat(81)));
    }
}

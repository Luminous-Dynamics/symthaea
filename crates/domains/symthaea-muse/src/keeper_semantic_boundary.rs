// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Construction and HTTP authority for durable keeper semantic evidence.
//!
//! This boundary deliberately separates two operations that must remain honest:
//!
//! 1. **stage time** recomputes score/recipe/WAV commitments from the exact
//!    objects and bytes being published, builds the shared V1 wire object, and
//!    delegates create-once persistence to `keeper_semantic_store`;
//! 2. **read time** reads only the stored sidecar. It never consults an in-memory
//!    candidate and never reconstructs evidence with the current engine.
//!
//! The production `muse_studio` wiring should therefore be very small: build its
//! already-authoritative `ListenCompositionBundle`, call
//! [`stage_keeper_semantic_evidence`] before the staging-directory rename, and
//! route GET requests to [`keeper_semantic_bundle`].

use std::fmt;
use std::path::Path;

use axum::Json;
use axum::extract::Path as AxPath;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_muse_protocol::keeper_semantic::{
    KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION, KeeperSemanticBundleV1,
};
use symthaea_muse_protocol::{
    ArtifactIdentity, BundleWarning, LISTEN_COMPOSITION_BUNDLE_VERSION, ListenCompositionBundle,
};

use crate::keeper_semantic_store::{
    KeeperSemanticStoreError, read_persisted_semantic_bundle, write_semantic_bundle_to_staging,
};

pub const KEEPER_AUDIO_ROOT: &str = "data/taste/audio";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeeperSemanticBoundaryError {
    ScoreSerialization(String),
    RecipeSerialization(String),
    IdentityMismatch(&'static str),
    Store(KeeperSemanticStoreError),
}

impl fmt::Display for KeeperSemanticBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScoreSerialization(error) => {
                write!(f, "keeper score serialization failed: {error}")
            }
            Self::RecipeSerialization(error) => {
                write!(f, "keeper recipe serialization failed: {error}")
            }
            Self::IdentityMismatch(label) => {
                write!(f, "keeper {label} commitment conflicts with candidate identity")
            }
            Self::Store(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for KeeperSemanticBoundaryError {}

impl From<KeeperSemanticStoreError> for KeeperSemanticBoundaryError {
    fn from(value: KeeperSemanticStoreError) -> Self {
        Self::Store(value)
    }
}

/// Build the exact durable semantic object for one keeper.
///
/// Hashes are recomputed here from the exact score/recipe objects and rendered
/// WAV bytes being staged; callers do not supply hash strings that could drift
/// from those artifacts. `expected_identity` is optional only to support callers
/// that predate `ArtifactIdentity`. When supplied, every recomputed commitment
/// must match it before any storage mutation is attempted.
pub fn build_keeper_semantic_bundle<S: Serialize, R: Serialize>(
    audio_key: &str,
    score: &S,
    recipe: &R,
    wav: &[u8],
    expected_identity: Option<&ArtifactIdentity>,
    payload: ListenCompositionBundle,
    warnings: Vec<BundleWarning>,
) -> Result<KeeperSemanticBundleV1, KeeperSemanticBoundaryError> {
    let score_bytes = serde_json::to_vec(score)
        .map_err(|error| KeeperSemanticBoundaryError::ScoreSerialization(error.to_string()))?;
    let recipe_bytes = serde_json::to_vec(recipe)
        .map_err(|error| KeeperSemanticBoundaryError::RecipeSerialization(error.to_string()))?;

    let score_sha256 = sha256_hex(&score_bytes);
    let recipe_sha256 = sha256_hex(&recipe_bytes);
    let audio_sha256 = sha256_hex(wav);

    if let Some(identity) = expected_identity {
        if identity.score_content.0 != score_sha256 {
            return Err(KeeperSemanticBoundaryError::IdentityMismatch("score"));
        }
        if identity.composition.0 != recipe_sha256 {
            return Err(KeeperSemanticBoundaryError::IdentityMismatch("recipe"));
        }
        if identity.rendition.0 != audio_sha256 {
            return Err(KeeperSemanticBoundaryError::IdentityMismatch("audio"));
        }
    }

    Ok(KeeperSemanticBundleV1 {
        schema_version: KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION,
        audio_key: audio_key.to_string(),
        listen_bundle_version: LISTEN_COMPOSITION_BUNDLE_VERSION,
        score_sha256,
        recipe_sha256,
        audio_sha256,
        warnings,
        payload,
    })
}

/// Build and persist the semantic sidecar in the caller-owned keeper staging
/// directory. Publication remains the caller's responsibility and must happen
/// only after this succeeds.
#[allow(clippy::too_many_arguments)]
pub fn stage_keeper_semantic_evidence<S: Serialize, R: Serialize>(
    staging_dir: &Path,
    audio_key: &str,
    score: &S,
    recipe: &R,
    wav: &[u8],
    expected_identity: Option<&ArtifactIdentity>,
    payload: ListenCompositionBundle,
    warnings: Vec<BundleWarning>,
) -> Result<KeeperSemanticBundleV1, KeeperSemanticBoundaryError> {
    let bundle = build_keeper_semantic_bundle(
        audio_key,
        score,
        recipe,
        wav,
        expected_identity,
        payload,
        warnings,
    )?;
    write_semantic_bundle_to_staging(staging_dir, audio_key, &bundle)?;
    Ok(bundle)
}

/// Pure HTTP-status mapping for the disk-only read authority.
///
/// * 400: caller supplied an invalid/path-unsafe key;
/// * 404: valid keeper key but no persisted semantic sidecar (legacy/absent);
/// * 500: stored evidence exists but is malformed, contradictory, symlinked,
///   or otherwise unreadable;
/// * success: exact shared DTO loaded from disk.
pub fn load_keeper_semantic_for_http(
    root: &Path,
    audio_key: &str,
) -> Result<KeeperSemanticBundleV1, StatusCode> {
    match read_persisted_semantic_bundle(root, audio_key) {
        Ok(Some(bundle)) => Ok(bundle),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(KeeperSemanticStoreError::InvalidArtifactKey) => Err(StatusCode::BAD_REQUEST),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Production Axum handler. It has no `Studio` state parameter by design: a
/// successful response can only come from durable keeper storage, so a process
/// restart cannot silently change the evidence source.
pub async fn keeper_semantic_bundle(AxPath(audio_key): AxPath<String>) -> Response {
    match load_keeper_semantic_for_http(Path::new(KEEPER_AUDIO_ROOT), &audio_key) {
        Ok(bundle) => Json(bundle).into_response(),
        Err(status) => status.into_response(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_muse_protocol::{
        CompositionArtifactId, MeterPoint, MusicalTime, RenditionArtifactId,
        ScoreContentArtifactId, TempoPoint,
    };

    #[derive(Serialize)]
    struct ScoreFixture {
        notes: Vec<u8>,
    }

    #[derive(Serialize)]
    struct RecipeFixture {
        seed: u64,
    }

    fn test_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "melothaea-keeper-semantic-boundary-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn zero() -> MusicalTime {
        MusicalTime {
            tick: 0,
            beats: 0.0,
            seconds: 0.0,
        }
    }

    fn payload() -> ListenCompositionBundle {
        ListenCompositionBundle {
            ticks_per_beat: 960,
            duration_ticks: 3840,
            duration_beats: 4.0,
            duration_seconds: 2.0,
            form_kind: "Ternary".to_string(),
            tempo_map: vec![TempoPoint {
                at: zero(),
                bpm: 120.0,
            }],
            meter_map: vec![MeterPoint {
                at: zero(),
                numerator: 4,
                denominator: 4,
            }],
            sections: Vec::new(),
            phrases: Vec::new(),
            notes: Vec::new(),
            motif_definitions: Vec::new(),
            motif_occurrences: Vec::new(),
            cadences: Vec::new(),
            sonorities: Vec::new(),
            orchestration: Vec::new(),
            resonance: None,
        }
    }

    fn fixture() -> (ScoreFixture, RecipeFixture, Vec<u8>) {
        (
            ScoreFixture {
                notes: vec![60, 64, 67],
            },
            RecipeFixture { seed: 42 },
            b"exact rendered wav bytes".to_vec(),
        )
    }

    #[test]
    fn construction_hashes_exact_objects_and_wav_bytes() {
        let (score, recipe, wav) = fixture();
        let bundle = build_keeper_semantic_bundle(
            "keeper-a",
            &score,
            &recipe,
            &wav,
            None,
            payload(),
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            bundle.score_sha256,
            sha256_hex(&serde_json::to_vec(&score).unwrap())
        );
        assert_eq!(
            bundle.recipe_sha256,
            sha256_hex(&serde_json::to_vec(&recipe).unwrap())
        );
        assert_eq!(bundle.audio_sha256, sha256_hex(&wav));
        assert_eq!(bundle.schema_version, KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.listen_bundle_version, LISTEN_COMPOSITION_BUNDLE_VERSION);
    }

    #[test]
    fn candidate_identity_is_cross_checked_before_storage() {
        let (score, recipe, wav) = fixture();
        let score_hash = sha256_hex(&serde_json::to_vec(&score).unwrap());
        let recipe_hash = sha256_hex(&serde_json::to_vec(&recipe).unwrap());
        let audio_hash = sha256_hex(&wav);
        let exact = ArtifactIdentity {
            score_content: ScoreContentArtifactId(score_hash),
            composition: CompositionArtifactId(recipe_hash),
            rendition: RenditionArtifactId(audio_hash),
        };
        assert!(
            build_keeper_semantic_bundle(
                "keeper-a",
                &score,
                &recipe,
                &wav,
                Some(&exact),
                payload(),
                Vec::new(),
            )
            .is_ok()
        );

        let mut wrong = exact.clone();
        wrong.rendition.0 = "0".repeat(64);
        assert_eq!(
            build_keeper_semantic_bundle(
                "keeper-a",
                &score,
                &recipe,
                &wav,
                Some(&wrong),
                payload(),
                Vec::new(),
            ),
            Err(KeeperSemanticBoundaryError::IdentityMismatch("audio"))
        );
    }

    #[test]
    fn stage_then_publish_survives_without_any_candidate_state() {
        let root = test_root("restart");
        std::fs::create_dir_all(&root).unwrap();
        let staging = root.join(".tmp-keeper-a");
        std::fs::create_dir(&staging).unwrap();
        let (score, recipe, wav) = fixture();
        let expected = stage_keeper_semantic_evidence(
            &staging,
            "keeper-a",
            &score,
            &recipe,
            &wav,
            None,
            payload(),
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            load_keeper_semantic_for_http(&root, "keeper-a"),
            Err(StatusCode::NOT_FOUND),
            "staging evidence must not be externally readable before publication"
        );
        std::fs::rename(&staging, root.join("keeper-a")).unwrap();
        assert_eq!(
            load_keeper_semantic_for_http(&root, "keeper-a").unwrap(),
            expected
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn http_mapping_distinguishes_bad_key_legacy_and_corrupt_storage() {
        let root = test_root("http-map");
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(
            load_keeper_semantic_for_http(&root, "../keeper"),
            Err(StatusCode::BAD_REQUEST)
        );
        std::fs::create_dir(root.join("legacy")).unwrap();
        assert_eq!(
            load_keeper_semantic_for_http(&root, "legacy"),
            Err(StatusCode::NOT_FOUND)
        );
        let corrupt = root.join("corrupt");
        std::fs::create_dir(&corrupt).unwrap();
        std::fs::write(
            corrupt.join(crate::keeper_semantic_store::KEEPER_SEMANTIC_FILENAME),
            b"not-json",
        )
        .unwrap();
        assert_eq!(
            load_keeper_semantic_for_http(&root, "corrupt"),
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
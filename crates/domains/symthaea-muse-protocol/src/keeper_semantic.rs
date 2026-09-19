// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared wire contract for durable semantic evidence attached to persisted keepers.
//!
//! This type is consumed by both the native Muse Studio server and the wasm UI.
//! It deliberately binds durable keeper identity (`audio_key` plus content hashes)
//! to the realized Listen composition bundle instead of reusing the live
//! `BundleEnvelope`'s process-scoped numeric candidate id.

use serde::{Deserialize, Serialize};

use crate::{BundleWarning, ListenCompositionBundle};

/// First durable keeper semantic-sidecar schema.
pub const KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Persisted realized semantic evidence for one keeper rendition.
///
/// The three SHA-256 commitments are the same score / recipe / rendered-audio
/// axes carried by `ArtifactIdentity`. Consumers must still cross-check them
/// against independently verified artifact identity before granting authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeeperSemanticBundleV1 {
    pub schema_version: u32,
    pub audio_key: String,
    pub listen_bundle_version: u32,
    pub score_sha256: String,
    pub recipe_sha256: String,
    pub audio_sha256: String,
    #[serde(default)]
    pub warnings: Vec<BundleWarning>,
    pub payload: ListenCompositionBundle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeterPoint, TempoPoint};

    #[test]
    fn wire_contract_round_trips_without_process_scoped_piece_identity() {
        let bundle = KeeperSemanticBundleV1 {
            schema_version: KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION,
            audio_key: "keeper-a".into(),
            listen_bundle_version: crate::LISTEN_COMPOSITION_BUNDLE_VERSION,
            score_sha256: "a".repeat(64),
            recipe_sha256: "b".repeat(64),
            audio_sha256: "c".repeat(64),
            warnings: Vec::new(),
            payload: ListenCompositionBundle {
                duration_beats: 4.0,
                duration_seconds: 2.0,
                tempo_map: vec![TempoPoint {
                    beat: 0.0,
                    seconds: 0.0,
                    bpm: 120.0,
                }],
                meter_map: vec![MeterPoint {
                    beat: 0.0,
                    numerator: 4,
                    denominator: 4,
                }],
                ..ListenCompositionBundle::default()
            },
        };

        let json = serde_json::to_string(&bundle).unwrap();
        assert!(!json.contains("piece_id"));
        assert!(!json.contains("candidate_id"));
        let decoded: KeeperSemanticBundleV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, bundle);
    }
}

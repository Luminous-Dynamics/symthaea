// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Library-specific projection and identity resolution for persisted keepers.
//!
//! `symthaea_muse_protocol::KeeperEntry` intentionally mirrors the older display
//! subset of each JSONL row. Newer server rows already persist two additional
//! identity handles (`genealogy_id`, `score_sha256`). This adapter extracts those
//! fields without changing the shared wire struct or making old log rows invalid.
//! It never invents missing hashes: legacy rows simply expose `None`.
//!
//! A keeper is promoted to a full `ArtifactIdentity` only after its genealogy
//! manifest is fetched and cross-checked against those stored handles. The
//! manifest's score/recipe/audio hashes are the same three content-addressed axes
//! used by live candidates. Any disagreement fails closed instead of silently
//! attributing one persisted artifact's identity to another.

use gloo_net::http::Request;
use serde_json::Value;
use symthaea_muse_protocol::{
    ArtifactIdentity, CompositionArtifactId, GenealogyManifest, KeeperEntry, RenditionArtifactId,
    ScoreContentArtifactId,
};

#[derive(Clone, Debug)]
pub struct KeeperRecord {
    pub entry: KeeperEntry,
    pub genealogy_id: Option<i64>,
    pub score_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedKeeperIdentity {
    pub genealogy_id: i64,
    pub artifact: ArtifactIdentity,
    pub manifest_sha256: String,
}

pub async fn fetch_keeper_records(backend: &str) -> Result<Vec<KeeperRecord>, String> {
    let url = format!("{}/api/keepers", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    let value = resp
        .json::<Value>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;
    parse_keeper_records(value)
}

/// Resolve the full content-addressed identity of one persisted keeper when the
/// keeper row provides enough handles to cross-check the genealogy manifest.
/// Legacy rows return `Ok(None)`; contradictory or malformed identity evidence
/// is an error rather than a fallback identity.
pub async fn fetch_verified_keeper_identity(
    backend: &str,
    record: &KeeperRecord,
) -> Result<Option<VerifiedKeeperIdentity>, String> {
    let (Some(genealogy_id), Some(score_sha256)) =
        (record.genealogy_id, record.score_sha256.as_deref())
    else {
        return Ok(None);
    };

    let url = format!(
        "{}/api/genealogy/{genealogy_id}",
        backend.trim_end_matches('/')
    );
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("genealogy request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!(
            "genealogy {genealogy_id} returned HTTP {}",
            resp.status()
        ));
    }
    let manifest = resp
        .json::<GenealogyManifest>()
        .await
        .map_err(|e| format!("failed to parse genealogy {genealogy_id}: {e}"))?;
    validate_keeper_manifest(record, score_sha256, manifest).map(Some)
}

fn parse_keeper_records(value: Value) -> Result<Vec<KeeperRecord>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "keeper response was not an array".to_string())?;

    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let entry = serde_json::from_value::<KeeperEntry>(row.clone())
                .map_err(|e| format!("failed to parse keeper row {index}: {e}"))?;
            let genealogy_id = row.get("genealogy_id").and_then(Value::as_i64);
            let score_sha256 = row
                .get("score_sha256")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .filter(|hash| !hash.is_empty());
            Ok(KeeperRecord {
                entry,
                genealogy_id,
                score_sha256,
            })
        })
        .collect()
}

fn validate_keeper_manifest(
    record: &KeeperRecord,
    expected_score_sha256: &str,
    manifest: GenealogyManifest,
) -> Result<VerifiedKeeperIdentity, String> {
    let expected_genealogy_id = record
        .genealogy_id
        .ok_or_else(|| "keeper has no genealogy id".to_string())?;
    if manifest.id != expected_genealogy_id {
        return Err(format!(
            "genealogy id mismatch: keeper references {expected_genealogy_id}, manifest reports {}",
            manifest.id
        ));
    }
    if manifest.audio_key != record.entry.audio_key {
        return Err(format!(
            "genealogy audio-key mismatch for keeper {}",
            record.entry.audio_key
        ));
    }
    if manifest.score_sha256.as_deref() != Some(expected_score_sha256) {
        return Err(format!(
            "genealogy score hash mismatch for keeper {}",
            record.entry.audio_key
        ));
    }

    for (label, hash) in [
        ("score", expected_score_sha256),
        ("recipe", manifest.recipe_sha256.as_str()),
        ("audio", manifest.audio_sha256.as_str()),
        ("manifest", manifest.manifest_sha256.as_str()),
    ] {
        if !is_sha256_hex(hash) {
            return Err(format!(
                "genealogy {label} hash is not a 64-character SHA-256 hex digest"
            ));
        }
    }

    Ok(VerifiedKeeperIdentity {
        genealogy_id: manifest.id,
        artifact: ArtifactIdentity {
            score_content: ScoreContentArtifactId(expected_score_sha256.to_string()),
            composition: CompositionArtifactId(manifest.recipe_sha256),
            rendition: RenditionArtifactId(manifest.audio_sha256),
        },
        manifest_sha256: manifest.manifest_sha256,
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use symthaea_muse_protocol::{GenealogyOrigin, GenealogyRelation};

    fn identity_record() -> KeeperRecord {
        KeeperRecord {
            entry: KeeperEntry {
                audio_key: "keeper-a".into(),
                ..KeeperEntry::default()
            },
            genealogy_id: Some(17),
            score_sha256: Some("a".repeat(64)),
        }
    }

    fn matching_manifest() -> GenealogyManifest {
        GenealogyManifest {
            id: 17,
            family_id: 17,
            parent_id: None,
            namespace: "C".into(),
            relation: GenealogyRelation::Root,
            origin: GenealogyOrigin::MuseGenerated {
                seed: 42,
                style_name: "Classical".into(),
            },
            audio_key: "keeper-a".into(),
            recipe_sha256: "b".repeat(64),
            score_sha256: Some("a".repeat(64)),
            audio_sha256: "c".repeat(64),
            manifest_sha256: "d".repeat(64),
            created_at_unix_ms: 1,
        }
    }

    #[test]
    fn newer_keeper_rows_preserve_existing_identity_handles() {
        let expected_score = "a".repeat(64);
        let records = parse_keeper_records(json!([{
            "ts": 1,
            "seed": 42,
            "spec": "Classical",
            "title": "Copper Lantern",
            "audio_key": "keeper-a",
            "midi_available": true,
            "genealogy_id": 17,
            "score_sha256": expected_score
        }]))
        .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.audio_key, "keeper-a");
        assert_eq!(records[0].genealogy_id, Some(17));
        assert_eq!(records[0].score_sha256.as_deref(), Some("a".repeat(64).as_str()));
    }

    #[test]
    fn legacy_keeper_rows_remain_valid_without_manufactured_identity() {
        let records = parse_keeper_records(json!([{
            "ts": 1,
            "seed": 42,
            "spec": "Classical",
            "audio_key": "legacy-a"
        }]))
        .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entry.audio_key, "legacy-a");
        assert_eq!(records[0].genealogy_id, None);
        assert_eq!(records[0].score_sha256, None);
    }

    #[test]
    fn empty_score_hash_is_treated_as_unavailable_not_identity() {
        let records = parse_keeper_records(json!([{
            "audio_key": "keeper-a",
            "score_sha256": ""
        }]))
        .unwrap();
        assert_eq!(records[0].score_sha256, None);
    }

    #[test]
    fn non_array_response_fails_closed() {
        assert!(parse_keeper_records(json!({"audio_key": "not-a-list"})).is_err());
    }

    #[test]
    fn matching_manifest_promotes_to_exact_artifact_identity() {
        let record = identity_record();
        let verified = validate_keeper_manifest(
            &record,
            record.score_sha256.as_deref().unwrap(),
            matching_manifest(),
        )
        .unwrap();
        assert_eq!(verified.genealogy_id, 17);
        assert_eq!(verified.artifact.score_content.0, "a".repeat(64));
        assert_eq!(verified.artifact.composition.0, "b".repeat(64));
        assert_eq!(verified.artifact.rendition.0, "c".repeat(64));
        assert_eq!(verified.manifest_sha256, "d".repeat(64));
    }

    #[test]
    fn genealogy_audio_key_mismatch_fails_closed() {
        let record = identity_record();
        let mut manifest = matching_manifest();
        manifest.audio_key = "other-keeper".into();
        assert!(validate_keeper_manifest(
            &record,
            record.score_sha256.as_deref().unwrap(),
            manifest,
        )
        .is_err());
    }

    #[test]
    fn genealogy_score_hash_mismatch_fails_closed() {
        let record = identity_record();
        let mut manifest = matching_manifest();
        manifest.score_sha256 = Some("e".repeat(64));
        assert!(validate_keeper_manifest(
            &record,
            record.score_sha256.as_deref().unwrap(),
            manifest,
        )
        .is_err());
    }

    #[test]
    fn malformed_genealogy_hash_fails_closed() {
        let record = identity_record();
        let mut manifest = matching_manifest();
        manifest.audio_sha256 = "not-a-hash".into();
        assert!(validate_keeper_manifest(
            &record,
            record.score_sha256.as_deref().unwrap(),
            manifest,
        )
        .is_err());
    }
}

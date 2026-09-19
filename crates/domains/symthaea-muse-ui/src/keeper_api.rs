// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Library-specific projection of the append-only keeper log.
//!
//! `symthaea_muse_protocol::KeeperEntry` intentionally mirrors the older display
//! subset of each JSONL row. Newer server rows already persist two additional
//! identity handles (`genealogy_id`, `score_sha256`). This adapter extracts those
//! fields without changing the shared wire struct or making old log rows invalid.
//! It never invents missing hashes: legacy rows simply expose `None`.

use gloo_net::http::Request;
use serde_json::Value;
use symthaea_muse_protocol::KeeperEntry;

#[derive(Clone, Debug)]
pub struct KeeperRecord {
    pub entry: KeeperEntry,
    pub genealogy_id: Option<i64>,
    pub score_sha256: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn newer_keeper_rows_preserve_existing_identity_handles() {
        let records = parse_keeper_records(json!([{
            "ts": 1,
            "seed": 42,
            "spec": "Classical",
            "title": "Copper Lantern",
            "audio_key": "keeper-a",
            "midi_available": true,
            "genealogy_id": 17,
            "score_sha256": "a".repeat(64)
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
}

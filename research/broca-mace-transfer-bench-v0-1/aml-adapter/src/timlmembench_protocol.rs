// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rust-native TimlMemBench JSONL protocol smoke adapter.
//!
//! This is a wiring and contract-conformance baseline only. It is not the MACE
//! governed-memory implementation, does not qualify evidence, and is not an
//! official benchmark result. Its lexical retrieval intentionally mirrors the
//! shape of the upstream offline smoke adapter so the public harness can be
//! connected without a model API.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

const TOP_K: usize = 5;

#[derive(Clone, Debug)]
struct LexicalDocV1 {
    memory_id: String,
    text: String,
}

#[derive(Clone, Debug, Default)]
pub struct TimlMemBenchLexicalAdapterV1 {
    sample_id: Option<String>,
    language: Option<String>,
    docs: Vec<LexicalDocV1>,
    closed: bool,
}

impl TimlMemBenchLexicalAdapterV1 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle one official-protocol request and return exactly one response object.
    /// Errors are returned as JSON error objects so the JSONL process can remain
    /// line-synchronous; the official runner treats any error response as fatal.
    pub fn handle(&mut self, request: &Value) -> Value {
        match self.try_handle(request) {
            Ok(response) => response,
            Err(error) => json!({ "error": error }),
        }
    }

    fn try_handle(&mut self, request: &Value) -> Result<Value, String> {
        if self.closed {
            return Err("adapter is closed".to_owned());
        }
        let object = request.as_object()
            .ok_or_else(|| "request must be a JSON object".to_owned())?;
        let request_type = required_string(object, "request_type")?;

        match request_type.as_str() {
            "reset" => self.reset(object),
            "ingest_session" => self.ingest_session(object),
            "query" => self.query(object),
            "close" => {
                require_exact_keys(object, &["request_type"])?;
                self.docs.clear();
                self.sample_id = None;
                self.language = None;
                self.closed = true;
                Ok(json!({ "ok": true }))
            }
            other => Err(format!("unknown request_type: {other}")),
        }
    }

    fn reset(&mut self, object: &Map<String, Value>) -> Result<Value, String> {
        require_exact_keys(object, &["request_type", "sample_id", "language"])?;
        let sample_id = required_string(object, "sample_id")?;
        let language = required_string(object, "language")?;
        self.sample_id = Some(sample_id);
        self.language = Some(language);
        self.docs.clear();
        Ok(json!({ "ok": true }))
    }

    fn ingest_session(&mut self, object: &Map<String, Value>) -> Result<Value, String> {
        require_exact_keys(object, &["request_type", "sample_id", "language", "session"])?;
        self.require_sample_and_language(
            required_string(object, "sample_id")?.as_str(),
            required_string(object, "language")?.as_str(),
        )?;
        let session = object.get("session")
            .and_then(Value::as_object)
            .ok_or_else(|| "session must be a JSON object".to_owned())?;
        let session_id = required_string(session, "session_id")?;
        let transcript = session_to_transcript(session)?;
        if transcript.trim().is_empty() {
            return Ok(json!({ "ok": true, "metadata": { "stored": false } }));
        }
        if self.docs.iter().any(|doc| doc.memory_id == session_id) {
            return Err(format!("duplicate session_id in sample: {session_id}"));
        }
        self.docs.push(LexicalDocV1 {
            memory_id: session_id,
            text: transcript,
        });
        Ok(json!({ "ok": true }))
    }

    fn query(&self, object: &Map<String, Value>) -> Result<Value, String> {
        require_query_keys(object)?;
        let mode = required_string(object, "mode")?;
        let sample_id = required_string(object, "sample_id")?;
        let language = required_string(object, "language")?;
        self.require_sample_and_language(&sample_id, &language)?;
        let task_id = required_string(object, "task_id")?;
        let query = required_string(object, "query")?;

        let docs = match mode.as_str() {
            "raw_native_cases" => self.docs.clone(),
            "canonical_qa" => canonical_docs(object)?,
            other => return Err(format!("unsupported query mode: {other}")),
        };
        let hits = lexical_rank(&query, &docs);
        let retrieved_texts = hits.iter().map(|hit| hit.text.clone()).collect::<Vec<_>>();
        let retrieved_memory_ids = hits.iter().map(|hit| hit.memory_id.clone()).collect::<Vec<_>>();
        let answer = retrieved_texts.first()
            .cloned()
            .unwrap_or_else(|| "<ABSTAIN>".to_owned());
        let score_total = hits.iter().map(|hit| hit.score).sum::<u64>();

        Ok(json!({
            "answer": answer,
            "retrieved_memory_ids": retrieved_memory_ids,
            "retrieved_texts": retrieved_texts,
            "metadata": {
                "adapter": "mace_rust_lexical_protocol_smoke_v1",
                "answer_mode": "extractive_top_hit",
                "mode": mode,
                "sample_id": sample_id,
                "task_id": task_id,
                "hit_count": hits.len(),
                "lexical_overlap_total": score_total
            },
            "usage": {
                "query_terms": token_counts(&query).values().sum::<usize>(),
                "documents_scanned": docs.len()
            }
        }))
    }

    fn require_sample_and_language(&self, sample_id: &str, language: &str) -> Result<(), String> {
        if self.sample_id.as_deref() != Some(sample_id) {
            return Err("sample_id does not match the active reset boundary".to_owned());
        }
        if self.language.as_deref() != Some(language) {
            return Err("language does not match the active reset boundary".to_owned());
        }
        Ok(())
    }
}

fn require_query_keys(object: &Map<String, Value>) -> Result<(), String> {
    const ALLOWED: &[&str] = &[
        "request_type", "mode", "baseline", "language", "sample_id", "task_id",
        "task_type", "case_type", "metrics", "query", "answer_policy", "memory_items",
    ];
    for key in object.keys() {
        if !ALLOWED.contains(&key.as_str()) {
            return Err(format!(
                "query field is not part of the public adapter contract: {key}"
            ));
        }
    }
    let mode = required_string(object, "mode")?;
    match mode.as_str() {
        "canonical_qa" => {
            if !object.contains_key("memory_items") {
                return Err("canonical_qa requires memory_items".to_owned());
            }
        }
        "raw_native_cases" => {
            if !object.contains_key("case_type") || !object.contains_key("metrics") {
                return Err("raw_native_cases requires case_type and metrics".to_owned());
            }
        }
        other => return Err(format!("unsupported query mode: {other}")),
    }
    Ok(())
}

fn require_exact_keys(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unexpected protocol field: {key}"));
        }
    }
    for key in allowed {
        if !object.contains_key(*key) {
            return Err(format!("missing protocol field: {key}"));
        }
    }
    Ok(())
}

fn required_string(object: &Map<String, Value>, key: &str) -> Result<String, String> {
    object.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn optional_string(object: &Map<String, Value>, key: &str) -> String {
    object.get(key).and_then(Value::as_str).unwrap_or("").to_owned()
}

fn session_to_transcript(session: &Map<String, Value>) -> Result<String, String> {
    let turns = session.get("turns")
        .and_then(Value::as_array)
        .ok_or_else(|| "session.turns must be an array".to_owned())?;
    let mut lines = vec![
        format!("session_id: {}", optional_string(session, "session_id")),
        format!("surface: {}", optional_string(session, "surface")),
        format!("started_at: {}", optional_string(session, "started_at")),
        String::new(),
    ];
    for turn in turns {
        let turn = turn.as_object()
            .ok_or_else(|| "each session turn must be an object".to_owned())?;
        let speaker = turn.get("speaker_id")
            .and_then(Value::as_str)
            .or_else(|| turn.get("speaker_role").and_then(Value::as_str))
            .unwrap_or("speaker");
        let timestamp = turn.get("timestamp").map(value_to_string).unwrap_or_default();
        let text = turn.get("text").and_then(Value::as_str).unwrap_or("").trim();
        if !text.is_empty() {
            lines.push(format!("{speaker} [{timestamp}]: {text}"));
        }
    }
    Ok(lines.join("\n"))
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn canonical_docs(object: &Map<String, Value>) -> Result<Vec<LexicalDocV1>, String> {
    let items = object.get("memory_items")
        .and_then(Value::as_array)
        .ok_or_else(|| "memory_items must be an array".to_owned())?;
    let allowed = [
        "memory_id", "kind", "content", "summary", "status",
        "scope_type", "scope_id", "source_turn_ids",
    ];
    let mut docs = Vec::new();
    for item in items {
        let item = item.as_object()
            .ok_or_else(|| "each memory item must be an object".to_owned())?;
        for key in item.keys() {
            if !allowed.contains(&key.as_str()) {
                return Err(format!("unexpected canonical memory field: {key}"));
            }
        }
        let text = item.get("summary").and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| item.get("content").and_then(Value::as_str))
            .unwrap_or("")
            .trim()
            .to_owned();
        if text.is_empty() {
            continue;
        }
        let memory_id = item.get("memory_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| "canonical memory item with content must have a memory_id".to_owned())?;
        docs.push(LexicalDocV1 { memory_id, text });
    }
    Ok(docs)
}

#[derive(Clone, Debug)]
struct LexicalHitV1 {
    memory_id: String,
    text: String,
    score: u64,
}

fn lexical_rank(query: &str, docs: &[LexicalDocV1]) -> Vec<LexicalHitV1> {
    let query_terms = token_counts(query);
    let mut hits = docs.iter().filter_map(|doc| {
        let doc_terms = token_counts(&doc.text);
        let score = query_terms.iter().map(|(term, count)| {
            (*count).min(*doc_terms.get(term).unwrap_or(&0)) as u64
        }).sum::<u64>();
        (score > 0).then(|| LexicalHitV1 {
            memory_id: doc.memory_id.clone(),
            text: doc.text.clone(),
            score,
        })
    }).collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right.score.cmp(&left.score)
            .then_with(|| left.memory_id.cmp(&right.memory_id))
            .then_with(|| left.text.cmp(&right.text))
    });
    hits.truncate(TOP_K);
    hits
}

fn token_counts(text: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut ascii_token = String::new();
    let flush_ascii = |token: &mut String, counts: &mut BTreeMap<String, usize>| {
        if !token.is_empty() {
            *counts.entry(token.to_ascii_lowercase()).or_default() += 1;
            token.clear();
        }
    };
    for character in text.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            ascii_token.push(character);
        } else {
            flush_ascii(&mut ascii_token, &mut counts);
            if ('\u{4e00}'..='\u{9fff}').contains(&character) {
                *counts.entry(character.to_string()).or_default() += 1;
            }
        }
    }
    flush_ascii(&mut ascii_token, &mut counts);
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset(adapter: &mut TimlMemBenchLexicalAdapterV1, sample: &str) {
        let response = adapter.handle(&json!({
            "request_type": "reset",
            "sample_id": sample,
            "language": "en"
        }));
        assert_eq!(response["ok"], true);
    }

    fn session(sample: &str, text: &str) -> Value {
        json!({
            "request_type": "ingest_session",
            "sample_id": sample,
            "language": "en",
            "session": {
                "session_id": format!("{sample}__s001"),
                "surface": "private_chat",
                "started_at": "2026-01-01T00:00:00Z",
                "turns": [{
                    "speaker_role": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "text": text
                }]
            }
        })
    }

    #[test]
    fn official_reset_ingest_raw_query_round_trip() {
        let mut adapter = TimlMemBenchLexicalAdapterV1::new();
        reset(&mut adapter, "sample_001");
        assert_eq!(adapter.handle(&session("sample_001", "The user prefers green tea."))["ok"], true);
        let response = adapter.handle(&json!({
            "request_type": "query",
            "mode": "raw_native_cases",
            "baseline": "mace_rust_smoke",
            "language": "en",
            "sample_id": "sample_001",
            "task_id": "case_001",
            "task_type": "raw_native_cases",
            "case_type": "activation",
            "metrics": ["Activation"],
            "query": "green tea preference"
        }));
        assert!(response["answer"].as_str().unwrap().contains("green tea"));
        assert_eq!(response["retrieved_memory_ids"][0], "sample_001__s001");
    }

    #[test]
    fn reset_isolates_samples_and_canonical_pool_is_used_as_supplied() {
        let mut adapter = TimlMemBenchLexicalAdapterV1::new();
        reset(&mut adapter, "sample_001");
        assert_eq!(adapter.handle(&session("sample_001", "private old memory"))["ok"], true);
        reset(&mut adapter, "sample_002");
        let response = adapter.handle(&json!({
            "request_type": "query",
            "mode": "canonical_qa",
            "baseline": "mace_rust_smoke",
            "language": "en",
            "sample_id": "sample_002",
            "task_id": "question_001",
            "query": "current preference",
            "answer_policy": "answer from evidence only",
            "memory_items": [{
                "memory_id": "canonical-1",
                "kind": "fact",
                "content": "The current preference is rooibos tea.",
                "summary": null,
                "status": "active",
                "scope_type": "user",
                "scope_id": "synthetic-user",
                "source_turn_ids": []
            }]
        }));
        assert!(response["answer"].as_str().unwrap().contains("rooibos"));
        assert_eq!(response["retrieved_memory_ids"][0], "canonical-1");
    }

    #[test]
    fn query_rejects_gold_label_or_unknown_field_injection() {
        let mut adapter = TimlMemBenchLexicalAdapterV1::new();
        reset(&mut adapter, "sample_001");
        let response = adapter.handle(&json!({
            "request_type": "query",
            "mode": "raw_native_cases",
            "language": "en",
            "sample_id": "sample_001",
            "task_id": "case_001",
            "case_type": "activation",
            "metrics": ["Activation"],
            "query": "test",
            "expected_behavior": "LEAKED GOLD LABEL"
        }));
        assert!(response["error"].as_str().unwrap().contains("expected_behavior"));
    }

    #[test]
    fn cross_sample_ingestion_is_rejected() {
        let mut adapter = TimlMemBenchLexicalAdapterV1::new();
        reset(&mut adapter, "sample_001");
        let response = adapter.handle(&session("sample_002", "must not enter this sample"));
        assert!(response["error"].as_str().unwrap().contains("sample_id"));
        assert!(adapter.docs.is_empty());
    }

    #[test]
    fn close_clears_state_and_prevents_further_requests() {
        let mut adapter = TimlMemBenchLexicalAdapterV1::new();
        reset(&mut adapter, "sample_001");
        let response = adapter.handle(&json!({ "request_type": "close" }));
        assert_eq!(response["ok"], true);
        assert!(adapter.docs.is_empty());
        let after_close = adapter.handle(&json!({ "request_type": "reset", "sample_id": "x", "language": "en" }));
        assert!(after_close["error"].as_str().unwrap().contains("closed"));
    }

    #[test]
    fn lexical_ties_are_deterministic() {
        let docs = vec![
            LexicalDocV1 { memory_id: "b".into(), text: "tea preference".into() },
            LexicalDocV1 { memory_id: "a".into(), text: "tea preference".into() },
        ];
        let hits = lexical_rank("tea preference", &docs);
        assert_eq!(hits[0].memory_id, "a");
    }

    #[test]
    fn duplicate_raw_session_id_is_rejected_within_the_active_sample() {
        let mut adapter = TimlMemBenchLexicalAdapterV1::new();
        reset(&mut adapter, "sample_001");
        assert_eq!(adapter.handle(&session("sample_001", "first record"))["ok"], true);
        let duplicate = adapter.handle(&session("sample_001", "duplicate record"));
        assert!(duplicate["error"].as_str().unwrap().contains("duplicate session_id"));
        assert_eq!(adapter.docs.len(), 1);
    }

}

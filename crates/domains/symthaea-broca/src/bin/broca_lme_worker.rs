// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Persistent JSONL worker for the LongMemEval-V2 lexical evidence baseline.
//!
//! This worker intentionally implements a transparent lexical baseline only.
//! It does not claim semantic HDC retrieval or visual query understanding.
//!
//! Protocol (one JSON object per line):
//!   {"request_id":"1","op":"init","scope_id":"web","bm25":{"k1":1.2,"b":0.75,"max_items":5}}
//!   {"request_id":"2","op":"insert","trajectory":{...official trajectory...}}
//!   {"request_id":"3","op":"query","query":"...","query_image":null}
//!   {"request_id":"4","op":"health"}
//!
//! Each response is one JSON line with the same request_id and an explicit ok
//! bit. Input line length is bounded. Inserted evidence is source-pointer-bound
//! to canonical JSON bytes, excludes outcome/answer/question metadata, and is
//! inserted atomically per trajectory.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use symthaea_broca::memory_evidence::{
    Bm25Config, EvidenceIndex, EvidenceLifecycle, EvidencePayload, EvidenceRecord,
    EvidenceSource,
};

const MAX_JSONL_LINE_BYTES: usize = 32 * 1024 * 1024;
const MAX_TEXT_FIELD_BYTES: usize = 4 * 1024 * 1024;
const MAX_EVIDENCE_ITEMS_PER_TRAJECTORY: usize = 50_000;

#[derive(Default)]
struct Worker {
    initialized: bool,
    scope_id: String,
    now_unix_ms: u64,
    bm25: Bm25Config,
    index: EvidenceIndex,
    inserted_trajectory_ids: BTreeSet<String>,
}

impl Worker {
    fn new() -> Self {
        Self {
            bm25: Bm25Config::default(),
            ..Self::default()
        }
    }

    fn handle(&mut self, request: Value) -> Value {
        let request_id = request
            .get("request_id")
            .cloned()
            .unwrap_or(Value::Null);
        let operation = request
            .get("op")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty());

        let result = if request_id.as_str().is_none_or(|id| id.trim().is_empty()) {
            Err("request_id must be a non-empty string".to_string())
        } else if let Some(operation) = operation {
            self.dispatch(operation, &request)
        } else {
            Err("op must be a non-empty string".to_string())
        };

        match result {
            Ok(result) => json!({
                "request_id": request_id,
                "ok": true,
                "result": result
            }),
            Err(error) => json!({
                "request_id": request_id,
                "ok": false,
                "error": error
            }),
        }
    }

    fn dispatch(&mut self, operation: &str, request: &Value) -> Result<Value, String> {
        match operation {
            "init" => self.initialize(request),
            "insert" => self.insert_trajectory(request),
            "query" => self.query(request),
            "health" => self.health(),
            other => Err(format!("unsupported op: {other}")),
        }
    }

    fn initialize(&mut self, request: &Value) -> Result<Value, String> {
        if self.initialized {
            return Err("worker is already initialized".into());
        }
        let scope = request
            .get("scope_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "init requires non-empty scope_id".to_string())?;
        if scope != "web" && scope != "enterprise" {
            return Err("scope_id must be exactly 'web' or 'enterprise' for LongMemEval-V2".into());
        }

        let bm25_obj = request.get("bm25").and_then(Value::as_object);
        let config = Bm25Config {
            k1: bm25_obj
                .and_then(|obj| obj.get("k1"))
                .and_then(Value::as_f64)
                .unwrap_or(1.2),
            b: bm25_obj
                .and_then(|obj| obj.get("b"))
                .and_then(Value::as_f64)
                .unwrap_or(0.75),
            max_items: bm25_obj
                .and_then(|obj| obj.get("max_items"))
                .and_then(Value::as_u64)
                .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
                .unwrap_or(5),
        };
        let now_unix_ms = request
            .get("now_unix_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);

        // search_bm25 validates the full config before worker state is changed.
        self.index
            .search_bm25("", scope, now_unix_ms, Path::new("."), config)?;

        self.scope_id = scope.to_string();
        self.now_unix_ms = now_unix_ms;
        self.bm25 = config;
        self.initialized = true;

        Ok(json!({
            "scope_id": self.scope_id,
            "bm25": {
                "k1": self.bm25.k1,
                "b": self.bm25.b,
                "max_items": self.bm25.max_items
            },
            "capabilities": {
                "text_query": true,
                "visual_query": false,
                "retrieval_type": "lexical_bm25",
                "semantic_encoder": false
            }
        }))
    }

    fn insert_trajectory(&mut self, request: &Value) -> Result<Value, String> {
        self.require_initialized()?;
        let trajectory = request
            .get("trajectory")
            .filter(|value| value.is_object())
            .ok_or_else(|| "insert requires a trajectory object".to_string())?;

        let trajectory_id = trajectory
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "trajectory.id must be a non-empty string".to_string())?
            .to_string();
        if self.inserted_trajectory_ids.contains(&trajectory_id) {
            return Err(format!("duplicate trajectory insert attempted: {trajectory_id}"));
        }

        // Use project-defined sorted-key JSON encoding so evidence digests and
        // pointers are stable even if serde_json's Map implementation is built
        // with insertion-order preservation enabled.
        let source_bytes = stable_json_bytes(trajectory)?;
        if source_bytes.len() > MAX_JSONL_LINE_BYTES {
            return Err("canonical trajectory exceeds configured JSONL input bound".into());
        }
        let source_sha256 = sha256_hex(&source_bytes);
        let records = collect_trajectory_evidence(
            trajectory,
            &trajectory_id,
            &self.scope_id,
            &source_sha256,
        )?;
        if records.is_empty() {
            return Err("trajectory yielded no eligible evidence strings".into());
        }
        if records.len() > MAX_EVIDENCE_ITEMS_PER_TRAJECTORY {
            return Err("trajectory exceeds MAX_EVIDENCE_ITEMS_PER_TRAJECTORY".into());
        }

        // No partial index mutation if any pointer/digest/ID is invalid.
        let inserted = self.index.insert_batch_atomic(records, &source_bytes)?;
        self.inserted_trajectory_ids.insert(trajectory_id.clone());

        Ok(json!({
            "trajectory_id": trajectory_id,
            "inserted_evidence_items": inserted,
            "source_sha256": source_sha256
        }))
    }

    fn query(&self, request: &Value) -> Result<Value, String> {
        self.require_initialized()?;
        let query = request
            .get("query")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "query requires non-empty query text".to_string())?;
        let hits = self.index.search_bm25(
            query,
            &self.scope_id,
            self.now_unix_ms,
            Path::new("."),
            self.bm25,
        )?;

        let context: Vec<Value> = hits
            .iter()
            .map(|hit| serde_json::to_value(&hit.context).map_err(|error| error.to_string()))
            .collect::<Result<_, _>>()?;
        let audit: Vec<Value> = hits
            .iter()
            .map(|hit| {
                json!({
                    "hdc_id": hit.hdc_id,
                    "evidence_id": hit.evidence_id,
                    "trajectory_id": hit.source.trajectory_id,
                    "step_id": hit.source.step_id,
                    "step_index": hit.source.step_index,
                    "source_sha256": hit.source.source_sha256,
                    "source_pointer": hit.source.source_pointer,
                    "content_sha256": hit.content_sha256,
                    "bm25_score": hit.bm25_score
                })
            })
            .collect();

        let image_query_received = request
            .get("query_image")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());

        Ok(json!({
            "context": context,
            "audit": audit,
            "image_query_received_but_not_used": image_query_received,
            "capabilities": {
                "text_query": true,
                "visual_query": false,
                "retrieval_type": "lexical_bm25",
                "semantic_encoder": false
            }
        }))
    }

    fn health(&self) -> Result<Value, String> {
        Ok(json!({
            "initialized": self.initialized,
            "scope_id": if self.initialized { Some(self.scope_id.as_str()) } else { None },
            "inserted_trajectory_count": self.inserted_trajectory_ids.len(),
            "capabilities": {
                "text_query": true,
                "visual_query": false,
                "retrieval_type": "lexical_bm25",
                "semantic_encoder": false
            }
        }))
    }

    fn require_initialized(&self) -> Result<(), String> {
        if self.initialized {
            Ok(())
        } else {
            Err("worker must receive init before insert/query".into())
        }
    }
}


/// Serialize JSON with recursively sorted object keys and preserved array order.
/// This is the worker's pinned stable encoding (not a claim of full RFC 8785).
fn stable_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    fn write_value(value: &Value, output: &mut Vec<u8>) -> Result<(), String> {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(true) => output.extend_from_slice(b"true"),
            Value::Bool(false) => output.extend_from_slice(b"false"),
            Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
            Value::String(_) => {
                let escaped = serde_json::to_vec(value)
                    .map_err(|error| format!("cannot encode JSON string: {error}"))?;
                output.extend_from_slice(&escaped);
            }
            Value::Array(items) => {
                output.push(b'[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        output.push(b',');
                    }
                    write_value(item, output)?;
                }
                output.push(b']');
            }
            Value::Object(object) => {
                output.push(b'{');
                let mut entries: Vec<_> = object.iter().collect();
                entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                for (index, (key, item)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        output.push(b',');
                    }
                    let escaped_key = serde_json::to_vec(key)
                        .map_err(|error| format!("cannot encode JSON object key: {error}"))?;
                    output.extend_from_slice(&escaped_key);
                    output.push(b':');
                    write_value(item, output)?;
                }
                output.push(b'}');
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    write_value(value, &mut output)?;
    Ok(output)
}

fn collect_trajectory_evidence(
    trajectory: &Value,
    trajectory_id: &str,
    scope_id: &str,
    source_sha256: &str,
) -> Result<Vec<EvidenceRecord>, String> {
    let mut fields: Vec<(String, String, Option<usize>)> = Vec::new();

    for key in ["goal", "start_url"] {
        if let Some(value) = trajectory.get(key).and_then(Value::as_str) {
            if !value.trim().is_empty() {
                fields.push((format!("/{key}"), value.to_string(), None));
            }
        }
    }

    if let Some(states) = trajectory.get("states").and_then(Value::as_array) {
        for (state_index, state) in states.iter().enumerate() {
            if !state.is_object() {
                return Err(format!("trajectory state {state_index} must be an object"));
            }
            for key in ["url", "action"] {
                if let Some(value) = state.get(key).and_then(Value::as_str) {
                    if !value.trim().is_empty() {
                        fields.push((
                            format!("/states/{state_index}/{key}"),
                            value.to_string(),
                            Some(state_index),
                        ));
                    }
                }
            }
            let thought_key = if state.get("thoughts").and_then(Value::as_str).is_some() {
                "thoughts"
            } else {
                "thought"
            };
            if let Some(value) = state.get(thought_key).and_then(Value::as_str) {
                if !value.trim().is_empty() {
                    fields.push((
                        format!("/states/{state_index}/{thought_key}"),
                        value.to_string(),
                        Some(state_index),
                    ));
                }
            }
            let text_key = if state.get("accessibility_tree").and_then(Value::as_str).is_some() {
                "accessibility_tree"
            } else {
                "text"
            };
            if let Some(value) = state.get(text_key).and_then(Value::as_str) {
                if !value.trim().is_empty() {
                    fields.push((
                        format!("/states/{state_index}/{text_key}"),
                        value.to_string(),
                        Some(state_index),
                    ));
                }
            }
        }
    }

    if fields.iter().any(|(_, text, _)| text.len() > MAX_TEXT_FIELD_BYTES) {
        return Err("evidence text field exceeds MAX_TEXT_FIELD_BYTES".into());
    }

    let mut records = Vec::with_capacity(fields.len());
    let mut seen_ids = BTreeSet::new();
    for (source_pointer, text, step_index) in fields {
        let identity = format!("{source_sha256}\n{source_pointer}");
        let identity_digest = Sha256::digest(identity.as_bytes());
        let evidence_id = hex::encode(identity_digest);
        let hdc_id = u64::from_be_bytes(
            identity_digest[0..8]
                .try_into()
                .map_err(|_| "internal evidence ID digest length error".to_string())?,
        );
        if !seen_ids.insert(hdc_id) {
            return Err("HDC ID collision within trajectory; refusing partial or ambiguous insert".into());
        }

        records.push(EvidenceRecord {
            schema_version: "BROCA-EVIDENCE-ITEM-V1".to_string(),
            evidence_id,
            hdc_id,
            scope_id: scope_id.to_string(),
            source: EvidenceSource {
                trajectory_id: trajectory_id.to_string(),
                step_id: step_index.map(|index| format!("state-{index}")),
                step_index: step_index.and_then(|index| u64::try_from(index).ok()),
                source_sha256: source_sha256.to_string(),
                source_pointer,
            },
            payload: EvidencePayload::Text { value: text.clone() },
            content_sha256: sha256_hex(text.as_bytes()),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: None,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            superseded_by: None,
        });
    }

    // Source bytes and every pointer are verified once, atomically, by
    // EvidenceIndex::insert_batch_atomic before any mutation is committed.
    Ok(records)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Read a single newline-delimited message without allowing an unbounded line
/// allocation. Oversized requests terminate the worker fail-closed.
fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    limit: usize,
) -> io::Result<Option<Vec<u8>>> {
    let mut output = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if output.is_empty() {
                Ok(None)
            } else {
                Ok(Some(output))
            };
        }
        let newline_position = available.iter().position(|byte| *byte == b'\n');
        let take = newline_position.map_or(available.len(), |position| position + 1);
        if output.len().saturating_add(take) > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSONL request exceeds {limit} byte limit"),
            ));
        }
        output.extend_from_slice(&available[..take]);
        reader.consume(take);

        if newline_position.is_some() {
            if output.last() == Some(&b'\n') {
                output.pop();
            }
            if output.last() == Some(&b'\r') {
                output.pop();
            }
            return Ok(Some(output));
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    let mut worker = Worker::new();

    loop {
        let line = match read_bounded_line(&mut reader, MAX_JSONL_LINE_BYTES) {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(error) => {
                eprintln!("broca-lme-worker: {error}");
                return Err(Box::new(error));
            }
        };
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }

        let response = match serde_json::from_slice::<Value>(&line) {
            Ok(request) => worker.handle(request),
            Err(error) => json!({
                "request_id": null,
                "ok": false,
                "error": format!("malformed JSON request: {error}")
            }),
        };
        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn stable_source_encoding_ignores_object_key_order() {
        let first: Value = serde_json::from_str(
            r#"{"z":{"b":2,"a":1},"a":[true,"text"]}"#
        ).unwrap();
        let second: Value = serde_json::from_str(
            r#"{"a":[true,"text"],"z":{"a":1,"b":2}}"#
        ).unwrap();

        assert_eq!(stable_json_bytes(&first).unwrap(), stable_json_bytes(&second).unwrap());
    }

    #[test]
    fn init_is_required_and_scope_is_pinned() {
        let mut worker = Worker::new();
        let before_init = worker.handle(json!({
            "request_id": "q1",
            "op": "query",
            "query": "billing"
        }));
        assert_eq!(before_init["ok"], false);

        let bad_scope = worker.handle(json!({
            "request_id": "q2",
            "op": "init",
            "scope_id": "question-provided"
        }));
        assert_eq!(bad_scope["ok"], false);

        let good_scope = worker.handle(json!({
            "request_id": "q3",
            "op": "init",
            "scope_id": "web"
        }));
        assert_eq!(good_scope["ok"], true);
        assert_eq!(good_scope["result"]["capabilities"]["semantic_encoder"], false);
    }

    #[test]
    fn ingestion_is_source_bound_and_excludes_outcome_and_question_metadata() {
        let trajectory = json!({
            "id": "trajectory-17",
            "goal": "Open billing configuration",
            "start_url": "https://example.invalid/",
            "outcome": "GOLD ANSWER MUST NOT ENTER MEMORY",
            "question_id": "evaluation-only-id",
            "question": "evaluator-only question",
            "states": [{
                "url": "https://example.invalid/billing",
                "action": "click Billing Settings",
                "thoughts": "Look for the payment panel",
                "accessibility_tree": "button Billing Settings textbox Account ID",
                "screenshot": "state-0.png"
            }]
        });
        let source = stable_json_bytes(&trajectory).unwrap();
        let source_hash = sha256_hex(&source);
        let records = collect_trajectory_evidence(
            &trajectory,
            "trajectory-17",
            "web",
            &source_hash,
        ).unwrap();

        assert_eq!(records.len(), 6);
        for record in &records {
            record.verify_source_bytes(&source).unwrap();
            assert!(!match &record.payload {
                EvidencePayload::Text { value } => value.contains("GOLD ANSWER") || value.contains("evaluator-only"),
                EvidencePayload::Image { .. } => false,
            });
        }
        assert!(!records.iter().any(|record| record.source.source_pointer.contains("outcome")));
        assert!(!records.iter().any(|record| record.source.source_pointer.contains("question")));
    }

    #[test]
    fn trajectory_identity_is_deterministic_and_query_returns_context_plus_separate_audit() {
        let trajectory = json!({
            "id": "trajectory-18",
            "goal": "Find the invoice tab",
            "states": [{
                "url": "https://example.invalid/invoices",
                "action": "open Invoices",
                "text": "Invoices heading New Invoice button"
            }]
        });
        let bytes = stable_json_bytes(&trajectory).unwrap();
        let digest = sha256_hex(&bytes);
        let a = collect_trajectory_evidence(&trajectory, "trajectory-18", "web", &digest).unwrap();
        let b = collect_trajectory_evidence(&trajectory, "trajectory-18", "web", &digest).unwrap();
        assert_eq!(a, b);

        let mut worker = Worker::new();
        assert_eq!(
            worker.handle(json!({"request_id":"1","op":"init","scope_id":"web"}))["ok"],
            true
        );
        assert_eq!(
            worker.handle(json!({"request_id":"2","op":"insert","trajectory":trajectory}))["ok"],
            true
        );
        let response = worker.handle(json!({
            "request_id":"3",
            "op":"query",
            "query":"invoice heading"
        }));
        assert_eq!(response["ok"], true);
        assert_eq!(response["result"]["context"][0]["type"], "text");
        assert!(response["result"]["audit"][0]["source_pointer"].is_string());
        assert_eq!(response["result"]["capabilities"]["visual_query"], false);
    }

    #[test]
    fn duplicate_trajectory_and_cross_scope_configuration_fail_closed() {
        let trajectory = json!({
            "id": "trajectory-19",
            "goal": "Open account settings",
            "states": [{"url":"https://example.invalid/account","text":"Account Settings"}]
        });
        let mut worker = Worker::new();
        worker.handle(json!({"request_id":"1","op":"init","scope_id":"web"}));
        assert_eq!(
            worker.handle(json!({"request_id":"2","op":"insert","trajectory":trajectory.clone()}))["ok"],
            true
        );
        assert_eq!(
            worker.handle(json!({"request_id":"3","op":"insert","trajectory":trajectory}))["ok"],
            false
        );
        assert_eq!(
            worker.handle(json!({"request_id":"4","op":"init","scope_id":"enterprise"}))["ok"],
            false
        );
    }

    #[test]
    fn bounded_reader_rejects_oversized_line() {
        let mut reader = io::Cursor::new(b"{\"op\":\"query\"}\n".to_vec());
        let error = read_bounded_line(&mut reader, 4).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}

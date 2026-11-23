// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Persistent JSONL sidecar for the LongMemEval-V2 official Memory API shim.
//!
//! The first arm is deliberately text-retrieval-only: query_image is accepted
//! at the protocol boundary but does not affect BM25 ranking. Retrieved
//! trajectory windows may still return their original screenshot evidence.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use broca_aml_adapter::longmemeval_v2::{
    LongMemEvalV2Bm25IndexV1, LongMemEvalV2ContextKind,
    LongMemEvalV2ContextItemV1, LongMemEvalV2HitV1,
    LongMemEvalV2TrajectoryV1, resolve_screenshot_reference,
};
use broca_aml_adapter::raw_session::Bm25ProfileV1;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

struct Config {
    domain: String,
    dataset_root: PathBuf,
    top_k: usize,
    slice_radius: usize,
    neighbor_radius: usize,
    profile: Bm25ProfileV1,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let domain = env::var("MACE_LMEV2_DOMAIN")
            .map_err(|_| "MACE_LMEV2_DOMAIN is required".to_string())?;
        if domain != "web" && domain != "enterprise" {
            return Err("MACE_LMEV2_DOMAIN must be web or enterprise".into());
        }
        let root = env::var("MACE_LMEV2_DATASET_ROOT")
            .map_err(|_| "MACE_LMEV2_DATASET_ROOT is required".to_string())?;
        let dataset_root = fs::canonicalize(root)
            .map_err(|error| format!("canonicalize dataset root: {error}"))?;
        if !dataset_root.is_dir() {
            return Err("MACE_LMEV2_DATASET_ROOT must be a directory".into());
        }
        let parse_usize = |name: &str, default: usize| -> Result<usize, String> {
            env::var(name)
                .ok()
                .map(|value| value.parse::<usize>().map_err(|error| format!("{name}: {error}")))
                .transpose()
                .map(|value| value.unwrap_or(default))
        };
        let parse_f64 = |name: &str, default: f64| -> Result<f64, String> {
            env::var(name)
                .ok()
                .map(|value| value.parse::<f64>().map_err(|error| format!("{name}: {error}")))
                .transpose()
                .map(|value| value.unwrap_or(default))
        };
        let top_k = parse_usize("MACE_LMEV2_TOP_K", 8)?;
        let slice_radius = parse_usize("MACE_LMEV2_SLICE_RADIUS", 1)?;
        let neighbor_radius = parse_usize("MACE_LMEV2_NEIGHBOR_RADIUS", 1)?;
        if top_k == 0 || top_k > 100 {
            return Err("MACE_LMEV2_TOP_K must be in 1..=100".into());
        }
        if slice_radius > 32 || neighbor_radius > 32 {
            return Err("MACE_LMEV2 slice and neighbor radii must be in 0..=32".into());
        }
        let profile = Bm25ProfileV1 {
            k1: parse_f64("MACE_LMEV2_BM25_K1", 1.2)?,
            b: parse_f64("MACE_LMEV2_BM25_B", 0.75)?,
            tokenizer_version: "unicode_alphanumeric_v1",
        };
        if !profile.k1.is_finite() || profile.k1 <= 0.0
            || !profile.b.is_finite() || !(0.0..=1.0).contains(&profile.b)
        {
            return Err("invalid BM25 k1/b profile".into());
        }
        Ok(Self {
            domain, dataset_root, top_k, slice_radius, neighbor_radius, profile,
        })
    }
}


fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn context_json(item: LongMemEvalV2ContextItemV1) -> Value {
    let kind = match item.kind {
        LongMemEvalV2ContextKind::Text => "text",
        LongMemEvalV2ContextKind::Image => "image",
    };
    json!({"type": kind, "value": item.value})
}

fn append_hit(
    items: &mut Vec<Value>,
    seen_images: &mut BTreeSet<String>,
    hit: &LongMemEvalV2HitV1,
    root: &Path,
) -> Result<(), String> {
    let resolved = hit
        .resolved_context_items(root)
        .map_err(|error| format!("resolve hit {} screenshot context: {error:?}", hit.chunk_id))?;
    for item in resolved {
        if item.kind == LongMemEvalV2ContextKind::Image
            && !seen_images.insert(item.value.clone())
        {
            continue;
        }
        items.push(context_json(item));
    }
    Ok(())
}

fn handle_request(
    request: Value,
    config: &Config,
    trajectories: &mut Vec<LongMemEvalV2TrajectoryV1>,
    index: &mut Option<LongMemEvalV2Bm25IndexV1>,
) -> Result<Value, String> {
    let operation = request
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| "request is missing string op".to_string())?;

    match operation {
        "insert" => {
            let source = request
                .get("trajectory")
                .ok_or_else(|| "insert request is missing trajectory".to_string())?;
            let trajectory_json = serde_json::to_string(source)
                .map_err(|error| format!("serialize trajectory: {error}"))?;
            let trajectory = LongMemEvalV2TrajectoryV1::parse_jsonl_record(&trajectory_json)
                .map_err(|error| format!("parse trajectory: {error:?}"))?;
            if trajectory.domain != config.domain {
                return Err(format!(
                    "trajectory domain {:?} does not match configured domain {:?}",
                    trajectory.domain, config.domain
                ));
            }
            if trajectories.iter().any(|item| item.id == trajectory.id) {
                return Err(format!("duplicate trajectory insert: {}", trajectory.id));
            }
            trajectories.push(trajectory);
            *index = None;
            Ok(json!({"ok": true, "inserted": 1}))
        }
        "query" => {
            let query = request
                .get("query")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "query request requires non-empty query".to_string())?;
            // Accepted for API compatibility. The lexical baseline does not
            // inspect or rank the question image; this is not multimodal search.
            let _query_image = request.get("query_image").and_then(Value::as_str);

            let build_started = Instant::now();
            let needs_build = index.is_none();
            if needs_build {
                let built = LongMemEvalV2Bm25IndexV1::build(
                    config.domain.clone(),
                    trajectories,
                    config.slice_radius,
                    config.profile,
                )
                .map_err(|error| format!("build trajectory BM25 index: {error:?}"))?;
                *index = Some(built);
            }
            let index_build_ns = if needs_build {
                u64::try_from(build_started.elapsed().as_nanos()).unwrap_or(u64::MAX)
            } else {
                0
            };
            let index_ref = index.as_ref().ok_or("index unavailable after build")?;
            let result = index_ref
                .search(&config.domain, query, config.top_k, config.neighbor_radius)
                .map_err(|error| format!("trajectory BM25 query: {error:?}"))?;

            let mut items = Vec::new();
            let mut seen_images = BTreeSet::new();
            for hit in &result.hits {
                append_hit(&mut items, &mut seen_images, hit, &config.dataset_root)?;
            }
            for neighbor in &result.neighbor_context {
                items.push(json!({
                    "type": "text",
                    "value": neighbor.chunk.evidence_text()
                }));
                for state in &neighbor.chunk.state_window {
                    let resolved = resolve_screenshot_reference(
                        &config.dataset_root,
                        &state.screenshot_reference,
                    )
                    .map_err(|error| format!("resolve neighbor screenshot: {error:?}"))?;
                    if let Some(expected_digest) = &state.screenshot_sha256 {
                        let bytes = fs::read(&resolved)
                            .map_err(|_| "neighbor screenshot is missing after resolution")?;
                        let actual_digest = sha256_hex(&bytes);
                        if actual_digest != *expected_digest {
                            return Err("neighbor screenshot digest mismatch".into());
                        }
                    }
                    let value = resolved.to_string_lossy().into_owned();
                    if seen_images.insert(value.clone()) {
                        items.push(json!({"type":"image","value":value}));
                    }
                }
            }
            Ok(json!({
                "ok": true,
                "items": items,
                "hit_count": result.hits.len(),
                "neighbor_count": result.neighbor_context.len(),
                "index_build_ns": index_build_ns,
                "search_latency_ns": result.search_latency_ns,
                "query_image_used": false,
                "retrieval_mode": "text_bm25_with_original_image_evidence"
            }))
        }
        other => Err(format!("unsupported sidecar operation: {other}")),
    }
}

fn main() {
    let config = Config::from_env().unwrap_or_else(|error| {
        eprintln!("mace_lmev2_stdio configuration error: {error}");
        std::process::exit(2);
    });
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    let mut trajectories = Vec::<LongMemEvalV2TrajectoryV1>::new();
    let mut index = None;

    for line in stdin.lock().lines() {
        let response = match line {
            Ok(line) => match serde_json::from_str::<Value>(&line) {
                Ok(request) => handle_request(request, &config, &mut trajectories, &mut index)
                    .unwrap_or_else(|error| json!({"ok": false, "error": error})),
                Err(error) => json!({"ok": false, "error": format!("invalid request JSON: {error}")}),
            },
            Err(error) => json!({"ok": false, "error": format!("read request: {error}")}),
        };
        if serde_json::to_writer(&mut writer, &response).is_err()
            || writer.write_all(b"\n").is_err()
            || writer.flush().is_err()
        {
            eprintln!("mace_lmev2_stdio: failed writing response");
            std::process::exit(1);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "mace-lmev2-sidecar-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("fixture root");
        root
    }

    fn fixture_config(root: PathBuf) -> Config {
        Config {
            domain: "web".to_owned(),
            dataset_root: root,
            top_k: 4,
            slice_radius: 0,
            neighbor_radius: 0,
            profile: Bm25ProfileV1 {
                k1: 1.2,
                b: 0.75,
                tokenizer_version: "unicode_alphanumeric_v1",
            },
        }
    }

    fn fixture_trajectory(root: &Path, image_bytes: &[u8]) -> Value {
        let trajectory_dir = root.join("trajectory-1");
        fs::create_dir_all(&trajectory_dir).expect("trajectory fixture directory");
        fs::write(trajectory_dir.join("state.png"), image_bytes).expect("image fixture");
        json!({
            "id": "trajectory-1",
            "domain": "web",
            "goal": "Create an order for Ada",
            "start_url": "https://shop.example/",
            "outcome": "success",
            "states": [{
                "url": "https://shop.example/orders/42",
                "action": "submit order",
                "thoughts": "Order submitted",
                "accessibility_tree": "Order details. Customer Ada. Status pending.",
                "screenshot": "trajectory-1/state.png",
                "state_index": 42,
                "step": 1,
                "mace_screenshot_sha256": sha256_hex(image_bytes)
            }]
        })
    }

    #[test]
    fn official_memory_api_request_returns_text_and_verified_image_context_items() {
        let root = fixture_root("context");
        let image_bytes = b"verified screenshot bytes";
        let trajectory = fixture_trajectory(&root, image_bytes);
        let expected_image = fs::canonicalize(root.join("trajectory-1/state.png"))
            .expect("canonical fixture image");
        let config = fixture_config(root.clone());
        let mut trajectories = Vec::new();
        let mut index = None;

        let inserted = handle_request(
            json!({"op": "insert", "trajectory": trajectory}),
            &config,
            &mut trajectories,
            &mut index,
        ).expect("insert");
        assert_eq!(inserted["ok"], true);
        assert_eq!(inserted["inserted"], 1);

        let response = handle_request(
            json!({
                "op": "query",
                "query": "customer Ada pending order",
                "query_image": "/private/query-image.png"
            }),
            &config,
            &mut trajectories,
            &mut index,
        ).expect("query");
        assert_eq!(response["ok"], true);
        assert_eq!(response["query_image_used"], false);
        assert_eq!(response["retrieval_mode"], "text_bm25_with_original_image_evidence");
        let items = response["items"].as_array().expect("context items");
        assert!(items.iter().any(|item| {
            item["type"] == "text" && item["value"].as_str()
                .is_some_and(|value| value.contains("Customer Ada"))
        }));
        assert!(items.iter().any(|item| {
            item["type"] == "image" && item["value"].as_str()
                == expected_image.to_str()
        }));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn duplicate_insert_is_rejected_before_reindexing() {
        let root = fixture_root("duplicate");
        let trajectory = fixture_trajectory(&root, b"image bytes");
        let config = fixture_config(root.clone());
        let mut trajectories = Vec::new();
        let mut index = None;

        handle_request(
            json!({"op": "insert", "trajectory": trajectory.clone()}),
            &config,
            &mut trajectories,
            &mut index,
        ).expect("first insert");
        let result = handle_request(
            json!({"op": "insert", "trajectory": trajectory}),
            &config,
            &mut trajectories,
            &mut index,
        );
        assert!(result.is_err());
        assert_eq!(trajectories.len(), 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn screenshot_byte_replacement_fails_before_image_context_is_returned() {
        let root = fixture_root("digest-mismatch");
        let trajectory = fixture_trajectory(&root, b"original bytes");
        let config = fixture_config(root.clone());
        let image_path = root.join("trajectory-1/state.png");
        let mut trajectories = Vec::new();
        let mut index = None;

        handle_request(
            json!({"op": "insert", "trajectory": trajectory}),
            &config,
            &mut trajectories,
            &mut index,
        ).expect("insert");
        fs::write(&image_path, b"replaced bytes").expect("tamper image bytes");

        let result = handle_request(
            json!({"op": "query", "query": "customer Ada pending order"}),
            &config,
            &mut trajectories,
            &mut index,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ScreenshotDigestMismatch"));
        fs::remove_dir_all(root).expect("cleanup");
    }
}

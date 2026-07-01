// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HuggingFace Hub model download helper.
//!
//! Downloads model files (safetensors, tokenizer) from HuggingFace Hub
//! using the `hf-hub` crate. Files are cached at `~/.cache/huggingface/hub/`.
//!
//! Supports both single-file models (`model.safetensors`) and multi-shard
//! models (`model.safetensors.index.json` + per-shard files).

use anyhow::Result;
use std::collections::HashSet;

/// Check if a model path looks like an HF Hub repo ID (e.g. "Qwen/Qwen3-Embedding-0.6B")
/// rather than a local filesystem path.
pub fn is_repo_id(path: &str) -> bool {
    // Must contain a slash (org/model), must not start with / or . (local paths),
    // and must not exist as a local directory
    path.contains('/')
        && !path.starts_with('/')
        && !path.starts_with('.')
        && !std::path::Path::new(path).exists()
}

/// Download model files from HuggingFace Hub, returning the local cache directory.
///
/// Handles two layouts:
/// 1. **Single shard**: downloads `model.safetensors`
/// 2. **Multi-shard**: downloads `model.safetensors.index.json`, parses its
///    `weight_map` to discover shard filenames, then downloads each shard.
///
/// Always downloads `tokenizer.json`. Files are cached at
/// `~/.cache/huggingface/hub/` and reused on subsequent calls.
pub fn ensure_model(repo_id: &str) -> Result<String> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HF Hub API: {e}"))?;
    let repo = api.model(repo_id.to_string());

    // Try multi-shard first: look for the index file
    let model_dir = match repo.get("model.safetensors.index.json") {
        Ok(index_path) => {
            // Parse weight_map → unique shard filenames
            let index_bytes = std::fs::read(&index_path)
                .map_err(|e| anyhow::anyhow!("Failed to read index file: {e}"))?;
            let index: serde_json::Value = serde_json::from_slice(&index_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to parse index JSON: {e}"))?;

            let weight_map = index
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow::anyhow!("Index file missing weight_map object"))?;

            let shard_files: HashSet<&str> =
                weight_map.values().filter_map(|v| v.as_str()).collect();

            // Download each unique shard file
            for shard in &shard_files {
                repo.get(shard)
                    .map_err(|e| anyhow::anyhow!("Failed to download shard {shard}: {e}"))?;
            }

            // Return directory from the index file path
            index_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("No parent directory for index file"))?
                .to_path_buf()
        }
        Err(_) => {
            // Fall back to single-shard model
            let safetensors_path = repo
                .get("model.safetensors")
                .map_err(|e| anyhow::anyhow!("Failed to download model.safetensors: {e}"))?;

            safetensors_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("No parent directory for downloaded model"))?
                .to_path_buf()
        }
    };

    // Always download tokenizer
    repo.get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json: {e}"))?;

    Ok(model_dir.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_repo_id() {
        assert!(is_repo_id("Qwen/Qwen3-Embedding-0.6B"));
        assert!(is_repo_id("sentence-transformers/all-MiniLM-L6-v2"));
        assert!(!is_repo_id("/absolute/path/to/model"));
        assert!(!is_repo_id("./relative/path"));
        assert!(!is_repo_id("no-slash-here"));
    }

    #[test]
    fn test_parse_shard_index() {
        // Simulate parsing a model.safetensors.index.json
        let index_json = r#"{
            "metadata": {"total_size": 3000000000},
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.14.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors"
            }
        }"#;

        let index: serde_json::Value = serde_json::from_str(index_json).unwrap();
        let weight_map = index.get("weight_map").unwrap().as_object().unwrap();
        let shard_files: HashSet<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();

        assert_eq!(shard_files.len(), 2);
        assert!(shard_files.contains("model-00001-of-00002.safetensors"));
        assert!(shard_files.contains("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn test_parse_shard_index_missing_weight_map() {
        let bad_json = r#"{"metadata": {}}"#;
        let index: serde_json::Value = serde_json::from_str(bad_json).unwrap();
        assert!(
            index
                .get("weight_map")
                .and_then(|v| v.as_object())
                .is_none()
        );
    }
}

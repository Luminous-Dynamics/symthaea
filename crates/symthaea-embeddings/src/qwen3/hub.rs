//! HuggingFace Hub model download helper.
//!
//! Downloads model files (safetensors, tokenizer) from HuggingFace Hub
//! using the `hf-hub` crate. Files are cached at `~/.cache/huggingface/hub/`.

use anyhow::Result;

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
/// Downloads `model.safetensors` and `tokenizer.json` (both required for inference).
/// Files are cached at `~/.cache/huggingface/hub/` and reused on subsequent calls.
pub fn ensure_model(repo_id: &str) -> Result<String> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HF Hub API: {e}"))?;
    let repo = api.model(repo_id.to_string());

    // Download required files (cached automatically)
    let safetensors_path = repo
        .get("model.safetensors")
        .map_err(|e| anyhow::anyhow!("Failed to download model.safetensors: {e}"))?;
    let _tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json: {e}"))?;

    // Return the directory containing the downloaded files
    let dir = safetensors_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("No parent directory for downloaded model"))?;
    Ok(dir.to_string_lossy().to_string())
}

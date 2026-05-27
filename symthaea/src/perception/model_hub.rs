// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Model Hub - HuggingFace Model Download Utility
//!
//! Provides automatic downloading and caching of ONNX models from HuggingFace Hub.
//!
//! ## Supported Models
//!
//! | Model | HuggingFace ID | Use Case |
//! |-------|----------------|----------|
//! | Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B | Text embeddings |
//! | SigLIP-SO400M | google/siglip-so400m-patch14-384 | Image embeddings |
//! | Moondream2 | vikhyatk/moondream2 | Image captioning |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::model_hub::{ModelHub, ModelSpec};
//!
//! let hub = ModelHub::new()?;
//!
//! // Download Qwen3 embedding model
//! let qwen3_path = hub.ensure_model(ModelSpec::Qwen3Embedding)?;
//!
//! // Download SigLIP vision model
//! let siglip_path = hub.ensure_model(ModelSpec::SigLIP)?;
//! ```

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Model specifications for download
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSpec {
    /// Qwen3-Embedding-0.6B for text embeddings
    Qwen3Embedding,

    /// SigLIP-SO400M for image embeddings
    SigLIP,

    /// Moondream2 for image captioning
    Moondream,

    /// BGE-base-en-v1.5 for text embeddings (legacy)
    BGE,
}

impl ModelSpec {
    /// Get the HuggingFace repository ID
    pub fn repo_id(&self) -> &'static str {
        match self {
            ModelSpec::Qwen3Embedding => "Qwen/Qwen3-Embedding-0.6B",
            ModelSpec::SigLIP => "google/siglip-so400m-patch14-384",
            ModelSpec::Moondream => "vikhyatk/moondream2",
            ModelSpec::BGE => "BAAI/bge-base-en-v1.5",
        }
    }

    /// Get the local directory name for this model
    pub fn local_name(&self) -> &'static str {
        match self {
            ModelSpec::Qwen3Embedding => "qwen3-embedding-0.6b",
            ModelSpec::SigLIP => "siglip-so400m",
            ModelSpec::Moondream => "moondream2",
            ModelSpec::BGE => "bge-base-en-v1.5",
        }
    }

    /// Get the required files for this model
    pub fn required_files(&self) -> &'static [&'static str] {
        match self {
            ModelSpec::Qwen3Embedding | ModelSpec::BGE => &["model.onnx", "tokenizer.json"],
            ModelSpec::SigLIP => &["model.onnx"],
            ModelSpec::Moondream => &["model.onnx", "tokenizer.json"],
        }
    }

    /// Get the embedding dimension for this model
    pub fn embedding_dim(&self) -> usize {
        match self {
            ModelSpec::Qwen3Embedding => 1024,
            ModelSpec::SigLIP => 768,
            ModelSpec::Moondream => 0, // VLM, not embedding
            ModelSpec::BGE => 768,
        }
    }
}

/// Model Hub for downloading and managing models
pub struct ModelHub {
    /// Base directory for model storage
    models_dir: PathBuf,
}

impl Default for ModelHub {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("models"),
        }
    }
}

impl ModelHub {
    /// Create a new ModelHub with default models directory
    pub fn new() -> Result<Self> {
        let hub = Self::default();
        std::fs::create_dir_all(&hub.models_dir).context("Failed to create models directory")?;
        Ok(hub)
    }

    /// Create a ModelHub with custom models directory
    pub fn with_dir(models_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&models_dir).context("Failed to create models directory")?;
        Ok(Self { models_dir })
    }

    /// Get the path where a model would be stored
    pub fn model_path(&self, spec: ModelSpec) -> PathBuf {
        self.models_dir.join(spec.local_name())
    }

    /// Check if a model is already downloaded
    pub fn is_downloaded(&self, spec: ModelSpec) -> bool {
        let model_dir = self.model_path(spec);
        if !model_dir.exists() {
            return false;
        }

        // Check all required files exist (also check onnx/ subfolder)
        spec.required_files().iter().all(|file| {
            model_dir.join(file).exists()
                || model_dir.join("onnx").join(file).exists()
                // For SigLIP, vision_model.onnx is preferred over model.onnx
                || (spec == ModelSpec::SigLIP && model_dir.join("onnx/vision_model.onnx").exists())
        })
    }

    /// Ensure a model is available, downloading if necessary
    ///
    /// Returns the path to the model directory.
    #[cfg(feature = "vision")]
    pub fn ensure_model(&self, spec: ModelSpec) -> Result<PathBuf> {
        let model_dir = self.model_path(spec);

        if self.is_downloaded(spec) {
            tracing::info!("Model {:?} already available at {:?}", spec, model_dir);
            return Ok(model_dir);
        }

        tracing::info!("Downloading model {:?} from HuggingFace...", spec);
        self.download_model(spec)?;

        Ok(model_dir)
    }

    /// Ensure a model is available (stub version without vision feature)
    #[cfg(not(feature = "vision"))]
    pub fn ensure_model(&self, spec: ModelSpec) -> Result<PathBuf> {
        let model_dir = self.model_path(spec);

        if self.is_downloaded(spec) {
            return Ok(model_dir);
        }

        // Without vision feature, we can't download - return instructions
        anyhow::bail!(
            "Model {:?} not found at {:?}. \n\
            To download, run:\n\
            pip install optimum[exporters] transformers\n\
            optimum-cli export onnx --model {} {:?}",
            spec,
            model_dir,
            spec.repo_id(),
            model_dir
        )
    }

    /// Download a model from HuggingFace Hub
    #[cfg(feature = "vision")]
    fn download_model(&self, spec: ModelSpec) -> Result<()> {
        use hf_hub::api::sync::Api;

        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.model(spec.repo_id().to_string());
        let model_dir = self.model_path(spec);

        std::fs::create_dir_all(&model_dir).context("Failed to create model directory")?;

        // Download each required file
        for file in spec.required_files() {
            tracing::info!("Downloading {}...", file);

            let downloaded_path = repo
                .get(file)
                .context(format!("Failed to download {}", file))?;

            // Copy to our models directory
            let dest_path = model_dir.join(file);
            std::fs::copy(&downloaded_path, &dest_path)
                .context(format!("Failed to copy {} to {:?}", file, dest_path))?;
        }

        tracing::info!("Model {:?} downloaded to {:?}", spec, model_dir);
        Ok(())
    }

    /// Try to ensure a model is available, returning typed error instead of panicking.
    ///
    /// This wraps [`ensure_model`] and converts the `anyhow::Error` into a
    /// [`ModelLoadError`](crate::perception::model_status::ModelLoadError)
    /// so callers can match on the error kind without string parsing.
    pub fn try_ensure_model(
        &self,
        spec: ModelSpec,
    ) -> Result<PathBuf, crate::perception::model_status::ModelLoadError> {
        self.ensure_model(spec).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("Not found") {
                crate::perception::model_status::ModelLoadError::FileNotFound(msg)
            } else if msg.contains("feature") || msg.contains("Feature") {
                crate::perception::model_status::ModelLoadError::FeatureDisabled(msg)
            } else if msg.contains("network") || msg.contains("download") {
                crate::perception::model_status::ModelLoadError::NetworkError(msg)
            } else {
                crate::perception::model_status::ModelLoadError::Other(msg)
            }
        })
    }

    /// List all available models
    pub fn list_available(&self) -> Vec<(ModelSpec, bool)> {
        vec![
            (
                ModelSpec::Qwen3Embedding,
                self.is_downloaded(ModelSpec::Qwen3Embedding),
            ),
            (ModelSpec::SigLIP, self.is_downloaded(ModelSpec::SigLIP)),
            (
                ModelSpec::Moondream,
                self.is_downloaded(ModelSpec::Moondream),
            ),
            (ModelSpec::BGE, self.is_downloaded(ModelSpec::BGE)),
        ]
    }

    /// Get the models directory
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Print download instructions for all models
    pub fn print_download_instructions() {
        println!("=== Model Download Instructions ===\n");

        println!("Install dependencies:");
        println!("  pip install optimum[exporters] transformers\n");

        println!("Download models:\n");

        for spec in [ModelSpec::Qwen3Embedding, ModelSpec::SigLIP, ModelSpec::BGE] {
            println!("  # {}", spec.local_name());
            println!(
                "  optimum-cli export onnx --model {} models/{}/\n",
                spec.repo_id(),
                spec.local_name()
            );
        }

        println!("Or use the automatic downloader (requires 'vision' feature):");
        println!("  cargo run --features vision --example download_models");
    }
}

/// Convenience function to get model path, downloading if needed
pub fn ensure_model(spec: ModelSpec) -> Result<PathBuf> {
    let hub = ModelHub::new()?;
    hub.ensure_model(spec)
}

/// Check if a model is available without downloading
pub fn is_model_available(spec: ModelSpec) -> bool {
    ModelHub::default().is_downloaded(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_spec_repo_id() {
        assert_eq!(
            ModelSpec::Qwen3Embedding.repo_id(),
            "Qwen/Qwen3-Embedding-0.6B"
        );
        assert_eq!(
            ModelSpec::SigLIP.repo_id(),
            "google/siglip-so400m-patch14-384"
        );
    }

    #[test]
    fn test_model_spec_local_name() {
        assert_eq!(
            ModelSpec::Qwen3Embedding.local_name(),
            "qwen3-embedding-0.6b"
        );
        assert_eq!(ModelSpec::SigLIP.local_name(), "siglip-so400m");
    }

    #[test]
    fn test_model_spec_embedding_dim() {
        assert_eq!(ModelSpec::Qwen3Embedding.embedding_dim(), 1024);
        assert_eq!(ModelSpec::SigLIP.embedding_dim(), 768);
        assert_eq!(ModelSpec::BGE.embedding_dim(), 768);
    }

    #[test]
    fn test_model_hub_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let hub = ModelHub::with_dir(temp_dir.path().to_path_buf()).unwrap();
        assert!(hub.models_dir().exists());
    }

    #[test]
    fn test_model_path() {
        let hub = ModelHub::default();
        let path = hub.model_path(ModelSpec::Qwen3Embedding);
        assert!(path.ends_with("qwen3-embedding-0.6b"));
    }
}
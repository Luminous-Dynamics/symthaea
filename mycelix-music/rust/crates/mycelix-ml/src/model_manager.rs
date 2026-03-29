// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Model management - downloading, caching, and loading ONNX models

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::{MlError, MlResult, ModelType};

/// Model metadata from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,
    /// Model type
    pub model_type: ModelType,
    /// Model version
    pub version: String,
    /// Download URL
    pub url: Option<String>,
    /// Expected file size in bytes
    pub size_bytes: Option<u64>,
    /// SHA256 checksum
    pub checksum: Option<String>,
    /// Model description
    pub description: String,
    /// Input dimension (e.g., mel bins)
    pub input_dim: usize,
    /// Output labels if classification model
    pub output_labels: Option<Vec<String>>,
}

/// Model state in the manager
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    NotDownloaded,
    Downloading,
    Downloaded,
    Loaded,
    Failed(String),
}

/// Manages model lifecycle - discovery, download, caching, and loading
pub struct ModelManager {
    /// Base directory for model storage
    cache_dir: PathBuf,
    /// Available models registry
    registry: HashMap<ModelType, ModelInfo>,
    /// Model states
    states: RwLock<HashMap<ModelType, ModelState>>,
    /// Loaded model paths
    paths: RwLock<HashMap<ModelType, PathBuf>>,
}

impl ModelManager {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        let cache_dir = cache_dir.into();

        // Ensure cache directory exists
        if let Err(e) = fs::create_dir_all(&cache_dir) {
            warn!("Failed to create model cache directory: {}", e);
        }

        Self {
            cache_dir,
            registry: Self::default_registry(),
            states: RwLock::new(HashMap::new()),
            paths: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default cache location
    pub fn with_default_cache() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mycelix")
            .join("models");
        Self::new(cache_dir)
    }

    /// Get the default model registry
    fn default_registry() -> HashMap<ModelType, ModelInfo> {
        let mut registry = HashMap::new();

        // Genre classifier
        registry.insert(ModelType::GenreClassifier, ModelInfo {
            id: "genre-classifier-v1".to_string(),
            model_type: ModelType::GenreClassifier,
            version: "1.0.0".to_string(),
            url: None, // Would be filled with actual model URL
            size_bytes: None,
            checksum: None,
            description: "Multi-genre music classifier trained on GTZAN and FMA".to_string(),
            input_dim: 128,
            output_labels: Some(vec![
                "rock", "pop", "hip-hop", "electronic", "classical",
                "jazz", "r&b", "country", "metal", "folk",
                "blues", "reggae", "latin", "punk", "indie",
            ].into_iter().map(String::from).collect()),
        });

        // Mood detector
        registry.insert(ModelType::MoodDetector, ModelInfo {
            id: "mood-detector-v1".to_string(),
            model_type: ModelType::MoodDetector,
            version: "1.0.0".to_string(),
            url: None,
            size_bytes: None,
            checksum: None,
            description: "Mood/emotion predictor for valence, energy, danceability".to_string(),
            input_dim: 128,
            output_labels: Some(vec![
                "valence", "energy", "danceability", "acousticness",
                "instrumentalness", "speechiness", "liveness",
            ].into_iter().map(String::from).collect()),
        });

        // Instrument recognizer
        registry.insert(ModelType::InstrumentRecognizer, ModelInfo {
            id: "instrument-recognizer-v1".to_string(),
            model_type: ModelType::InstrumentRecognizer,
            version: "1.0.0".to_string(),
            url: None,
            size_bytes: None,
            checksum: None,
            description: "Multi-label instrument detection model".to_string(),
            input_dim: 128,
            output_labels: Some(vec![
                "drums", "bass", "guitar", "piano", "synthesizer",
                "strings", "brass", "woodwind", "vocals", "percussion",
            ].into_iter().map(String::from).collect()),
        });

        // Embedding extractor
        registry.insert(ModelType::EmbeddingExtractor, ModelInfo {
            id: "audio-embedding-v1".to_string(),
            model_type: ModelType::EmbeddingExtractor,
            version: "1.0.0".to_string(),
            url: None,
            size_bytes: None,
            checksum: None,
            description: "Audio embedding model for similarity search".to_string(),
            input_dim: 128,
            output_labels: None,
        });

        // Vocal detector
        registry.insert(ModelType::VocalDetector, ModelInfo {
            id: "vocal-detector-v1".to_string(),
            model_type: ModelType::VocalDetector,
            version: "1.0.0".to_string(),
            url: None,
            size_bytes: None,
            checksum: None,
            description: "Binary vocal/instrumental classifier".to_string(),
            input_dim: 128,
            output_labels: Some(vec!["instrumental".to_string(), "vocal".to_string()]),
        });

        // BPM estimator
        registry.insert(ModelType::BpmEstimator, ModelInfo {
            id: "bpm-estimator-v1".to_string(),
            model_type: ModelType::BpmEstimator,
            version: "1.0.0".to_string(),
            url: None,
            size_bytes: None,
            checksum: None,
            description: "Tempo estimation model".to_string(),
            input_dim: 128,
            output_labels: None,
        });

        // Key detector
        registry.insert(ModelType::KeyDetector, ModelInfo {
            id: "key-detector-v1".to_string(),
            model_type: ModelType::KeyDetector,
            version: "1.0.0".to_string(),
            url: None,
            size_bytes: None,
            checksum: None,
            description: "Musical key detection model".to_string(),
            input_dim: 12, // Chromagram
            output_labels: Some(vec![
                "C major", "C# major", "D major", "D# major", "E major", "F major",
                "F# major", "G major", "G# major", "A major", "A# major", "B major",
                "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
                "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
            ].into_iter().map(String::from).collect()),
        });

        registry
    }

    /// Get model info
    pub fn model_info(&self, model_type: ModelType) -> Option<&ModelInfo> {
        self.registry.get(&model_type)
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.registry.values().collect()
    }

    /// Get model state
    pub fn model_state(&self, model_type: ModelType) -> ModelState {
        self.states.read().get(&model_type).cloned()
            .unwrap_or(ModelState::NotDownloaded)
    }

    /// Get path for a model
    pub fn model_path(&self, model_type: ModelType) -> Option<PathBuf> {
        self.paths.read().get(&model_type).cloned()
    }

    /// Check if model file exists locally
    pub fn is_cached(&self, model_type: ModelType) -> bool {
        let filename = format!("{}.onnx", self.model_id(model_type));
        self.cache_dir.join(&filename).exists()
    }

    /// Get model ID for a type
    fn model_id(&self, model_type: ModelType) -> &str {
        self.registry.get(&model_type)
            .map(|info| info.id.as_str())
            .unwrap_or("unknown")
    }

    /// Register a custom model
    pub fn register_model(&mut self, model_type: ModelType, info: ModelInfo) {
        self.registry.insert(model_type, info);
    }

    /// Load a model from a local path
    pub fn load_from_path(&self, model_type: ModelType, path: impl AsRef<Path>) -> MlResult<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(MlError::ModelNotFound(path.display().to_string()));
        }

        self.paths.write().insert(model_type, path.to_path_buf());
        self.states.write().insert(model_type, ModelState::Downloaded);

        info!("Registered model {:?} from {}", model_type, path.display());
        Ok(())
    }

    /// Load a model from the cache directory
    pub fn load_from_cache(&self, model_type: ModelType) -> MlResult<PathBuf> {
        let filename = format!("{}.onnx", self.model_id(model_type));
        let path = self.cache_dir.join(&filename);

        if !path.exists() {
            return Err(MlError::ModelNotFound(format!(
                "Model {:?} not found in cache at {}",
                model_type,
                path.display()
            )));
        }

        self.paths.write().insert(model_type, path.clone());
        self.states.write().insert(model_type, ModelState::Downloaded);

        info!("Loaded model {:?} from cache", model_type);
        Ok(path)
    }

    /// Check which models are available locally
    pub fn scan_local_models(&self) -> Vec<ModelType> {
        let mut available = Vec::new();

        for model_type in self.registry.keys() {
            if self.is_cached(*model_type) {
                available.push(*model_type);
                self.states.write().insert(*model_type, ModelState::Downloaded);
            }
        }

        available
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::with_default_cache()
    }
}

/// Model bundle - a collection of models for a specific use case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    pub name: String,
    pub description: String,
    pub models: Vec<ModelType>,
}

impl ModelBundle {
    /// Core analysis bundle - genre, mood, instruments
    pub fn core_analysis() -> Self {
        Self {
            name: "Core Analysis".to_string(),
            description: "Essential models for music analysis".to_string(),
            models: vec![
                ModelType::GenreClassifier,
                ModelType::MoodDetector,
                ModelType::InstrumentRecognizer,
            ],
        }
    }

    /// Full analysis bundle - all models
    pub fn full_analysis() -> Self {
        Self {
            name: "Full Analysis".to_string(),
            description: "Complete set of analysis models".to_string(),
            models: vec![
                ModelType::GenreClassifier,
                ModelType::MoodDetector,
                ModelType::InstrumentRecognizer,
                ModelType::VocalDetector,
                ModelType::EmbeddingExtractor,
                ModelType::BpmEstimator,
                ModelType::KeyDetector,
            ],
        }
    }

    /// Similarity search bundle
    pub fn similarity() -> Self {
        Self {
            name: "Similarity Search".to_string(),
            description: "Models for audio similarity and recommendations".to_string(),
            models: vec![
                ModelType::EmbeddingExtractor,
                ModelType::GenreClassifier,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_model_manager_creation() {
        let temp_dir = env::temp_dir().join("mycelix-test-models");
        let manager = ModelManager::new(&temp_dir);

        assert_eq!(manager.cache_dir(), temp_dir);
        assert!(!manager.list_models().is_empty());
    }

    #[test]
    fn test_model_registry() {
        let manager = ModelManager::with_default_cache();

        // Check that core models are registered
        assert!(manager.model_info(ModelType::GenreClassifier).is_some());
        assert!(manager.model_info(ModelType::MoodDetector).is_some());
        assert!(manager.model_info(ModelType::EmbeddingExtractor).is_some());
    }

    #[test]
    fn test_model_bundles() {
        let core = ModelBundle::core_analysis();
        assert!(core.models.contains(&ModelType::GenreClassifier));

        let full = ModelBundle::full_analysis();
        assert!(full.models.len() > core.models.len());
    }
}

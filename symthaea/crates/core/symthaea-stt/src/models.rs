// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pre-trained Model Packaging and Distribution
//!
//! Handles loading, saving, and distributing Symthaea STT models.
//!
//! # Model Package Format
//!
//! Models are stored as compressed JSON bundles containing:
//! - Phoneme prototypes (HDC vectors)
//! - Language model parameters
//! - Configuration metadata
//!
//! ```text
//! symthaea-model-v1.json
//! ├── metadata
//! │   ├── version
//! │   ├── created
//! │   └── description
//! ├── prototypes
//! │   └── [phoneme -> HV16 data]
//! ├── language_model
//! │   └── [n-gram parameters]
//! └── config
//!     ├── audio
//!     └── decoder
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::audio::AudioConfig;
use crate::bootstrap::TrainedPrototypes;
use crate::hdc::HV16;
use crate::lm::BeamConfig;
use crate::lm::NgramLM;

/// Current model format version
pub const MODEL_VERSION: &str = "1.0.0";

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model format version
    pub version: String,
    /// Creation timestamp (ISO 8601)
    pub created: String,
    /// Model description
    pub description: String,
    /// Training corpus info
    pub training_info: Option<String>,
    /// Number of phoneme prototypes
    pub num_prototypes: usize,
    /// Language model order
    pub lm_order: Option<usize>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            version: MODEL_VERSION.to_string(),
            created: chrono_lite_now(),
            description: "Symthaea STT Model".to_string(),
            training_info: None,
            num_prototypes: 0,
            lm_order: None,
            extra: HashMap::new(),
        }
    }
}

/// Serializable prototype data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrototypeData {
    label: String,
    /// HV16 data as u128 words (16 words for 2048 bits)
    words: Vec<u128>,
}

/// Complete model package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackage {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Phoneme prototypes (serialized)
    prototypes: Vec<PrototypeData>,
    /// Language model (optional)
    language_model: Option<NgramLM>,
    /// Audio configuration
    pub audio_config: AudioConfig,
    /// Decoder configuration
    pub decoder_config: BeamConfig,
}

impl ModelPackage {
    /// Create a new empty model package
    pub fn new() -> Self {
        Self {
            metadata: ModelMetadata::default(),
            prototypes: Vec::new(),
            language_model: None,
            audio_config: AudioConfig::default(),
            decoder_config: BeamConfig::default(),
        }
    }

    /// Create from trained prototypes
    pub fn from_prototypes(prototypes: &TrainedPrototypes) -> Self {
        let pairs = prototypes.as_pairs();
        let proto_data: Vec<PrototypeData> = pairs
            .into_iter()
            .map(|(label, hv)| PrototypeData {
                label,
                words: hv.words.to_vec(),
            })
            .collect();

        let mut pkg = Self::new();
        pkg.metadata.num_prototypes = proto_data.len();
        pkg.prototypes = proto_data;
        pkg
    }

    /// Set model description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.metadata.description = desc.to_string();
        self
    }

    /// Set training info
    pub fn with_training_info(mut self, info: &str) -> Self {
        self.metadata.training_info = Some(info.to_string());
        self
    }

    /// Set language model
    pub fn with_language_model(mut self, lm: NgramLM) -> Self {
        self.metadata.lm_order = Some(lm.order);
        self.language_model = Some(lm);
        self
    }

    /// Set audio configuration
    pub fn with_audio_config(mut self, config: AudioConfig) -> Self {
        self.audio_config = config;
        self
    }

    /// Set decoder configuration
    pub fn with_decoder_config(mut self, config: BeamConfig) -> Self {
        self.decoder_config = config;
        self
    }

    /// Add metadata field
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata
            .extra
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Get prototypes as (label, HV16) pairs
    pub fn get_prototypes(&self) -> Vec<(String, HV16)> {
        use crate::hdc::HDC_WORDS;

        self.prototypes
            .iter()
            .map(|p| {
                let mut words = [0u128; HDC_WORDS];
                for (i, &v) in p.words.iter().enumerate().take(HDC_WORDS) {
                    words[i] = v;
                }
                (p.label.clone(), HV16 { words })
            })
            .collect()
    }

    /// Get language model reference
    pub fn get_language_model(&self) -> Option<&NgramLM> {
        self.language_model.as_ref()
    }

    /// Get number of prototypes
    pub fn num_prototypes(&self) -> usize {
        self.prototypes.len()
    }

    /// Check if model has language model
    pub fn has_language_model(&self) -> bool {
        self.language_model.is_some()
    }

    /// Save model to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(std::io::Error::other)
    }

    /// Save model to compressed binary format
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_vec(self).map_err(std::io::Error::other)?;

        // Simple compression: just store as-is for now
        // In production, could use flate2 or similar
        let mut file = File::create(path)?;
        file.write_all(&json)?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(std::io::Error::other)
    }

    /// Load model from binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(std::io::Error::other)
    }

    /// Get model summary as string
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Symthaea STT Model v{}\n", self.metadata.version));
        s.push_str(&format!("Description: {}\n", self.metadata.description));
        s.push_str(&format!("Created: {}\n", self.metadata.created));
        s.push_str(&format!("Prototypes: {}\n", self.num_prototypes()));

        if let Some(order) = self.metadata.lm_order {
            s.push_str(&format!("Language Model: {order}-gram\n"));
        }

        if let Some(ref info) = self.metadata.training_info {
            s.push_str(&format!("Training: {info}\n"));
        }

        s
    }
}

impl Default for ModelPackage {
    fn default() -> Self {
        Self::new()
    }
}

/// Model registry for managing multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// Available models
    models: HashMap<String, ModelInfo>,
}

/// Information about a registered model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Path to model file
    pub path: String,
    /// Model size in bytes
    pub size: u64,
    /// Model checksum (BLAKE3)
    pub checksum: Option<String>,
    /// Whether model is default
    pub is_default: bool,
}

impl ModelRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a model
    pub fn register(&mut self, name: &str, info: ModelInfo) {
        self.models.insert(name.to_string(), info);
    }

    /// Get model info by name
    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    /// Get default model
    pub fn get_default(&self) -> Option<&ModelInfo> {
        self.models.values().find(|m| m.is_default)
    }

    /// List all available models
    pub fn list(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Save registry to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(std::io::Error::other)
    }

    /// Load registry from file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(std::io::Error::other)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-built model configurations
pub mod presets {
    use super::*;

    /// Create a minimal English phoneme model
    ///
    /// Contains 39 ARPABET phonemes with random prototypes.
    /// Useful for testing and as a starting point.
    pub fn minimal_english() -> ModelPackage {
        let phonemes = [
            "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
            "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
            "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
        ];

        let protos: Vec<PrototypeData> = phonemes
            .iter()
            .map(|&p| PrototypeData {
                label: p.to_string(),
                words: HV16::random(p).words.to_vec(),
            })
            .collect();

        ModelPackage {
            metadata: ModelMetadata {
                version: MODEL_VERSION.to_string(),
                created: chrono_lite_now(),
                description: "Minimal English phoneme model (39 ARPABET phones)".to_string(),
                training_info: Some("Random initialization".to_string()),
                num_prototypes: 39,
                lm_order: None,
                extra: HashMap::new(),
            },
            prototypes: protos,
            language_model: None,
            audio_config: AudioConfig::default(),
            decoder_config: BeamConfig::default(),
        }
    }

    /// Create a model optimized for edge deployment
    ///
    /// Uses smaller LTC hidden size and beam width.
    pub fn edge_optimized() -> ModelPackage {
        let mut pkg = minimal_english();

        pkg.metadata.description = "Edge-optimized model for low-latency inference".to_string();
        pkg.decoder_config = BeamConfig {
            beam_width: 5,
            lm_weight: 0.2,
            word_bonus: 0.0,
            min_prob: -15.0,
            length_norm: true,
        };

        pkg
    }

    /// Create a high-accuracy model configuration
    ///
    /// Uses larger beam width and stronger LM weight.
    pub fn high_accuracy() -> ModelPackage {
        let mut pkg = minimal_english();

        pkg.metadata.description = "High-accuracy model for batch transcription".to_string();
        pkg.decoder_config = BeamConfig {
            beam_width: 20,
            lm_weight: 0.4,
            word_bonus: 0.5,
            min_prob: -20.0,
            length_norm: true,
        };

        pkg
    }
}

/// Simple timestamp function (no external dependencies)
fn chrono_lite_now() -> String {
    // Use std::time for basic timestamp
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();

    // Simple conversion to ISO-8601 format
    let days_since_1970 = secs / 86400;
    let mut year = 1970;
    let mut days = days_since_1970 as i64;

    // Calculate year
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    // Calculate month and day
    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &dim in &days_in_months {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }
    let day = days + 1;

    // Time of day
    let secs_today = secs % 86400;
    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_package_creation() {
        let pkg = ModelPackage::new();
        assert_eq!(pkg.metadata.version, MODEL_VERSION);
        assert_eq!(pkg.num_prototypes(), 0);
    }

    #[test]
    fn test_minimal_english_preset() {
        let pkg = presets::minimal_english();
        assert_eq!(pkg.num_prototypes(), 39);
        assert!(pkg.metadata.description.contains("English"));
    }

    #[test]
    fn test_prototype_roundtrip() {
        let pkg = presets::minimal_english();
        let protos = pkg.get_prototypes();

        assert_eq!(protos.len(), 39);

        // Verify we can look up a phoneme
        let aa = protos.iter().find(|(l, _)| l == "AA");
        assert!(aa.is_some());
    }

    #[test]
    fn test_model_with_options() {
        let pkg = presets::minimal_english()
            .with_description("Test model")
            .with_training_info("Unit tests")
            .with_metadata("test_key", "test_value");

        assert_eq!(pkg.metadata.description, "Test model");
        assert_eq!(pkg.metadata.training_info, Some("Unit tests".to_string()));
        assert_eq!(
            pkg.metadata.extra.get("test_key"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_model_summary() {
        let pkg = presets::minimal_english();
        let summary = pkg.summary();

        assert!(summary.contains("Symthaea"));
        assert!(summary.contains("39"));
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();

        registry.register(
            "test",
            ModelInfo {
                name: "test".to_string(),
                description: "Test model".to_string(),
                path: "/path/to/model".to_string(),
                size: 1024,
                checksum: None,
                is_default: true,
            },
        );

        assert_eq!(registry.list().len(), 1);
        assert!(registry.get("test").is_some());
        assert!(registry.get_default().is_some());
    }

    #[test]
    fn test_edge_optimized_preset() {
        let pkg = presets::edge_optimized();
        assert_eq!(pkg.decoder_config.beam_width, 5);
    }

    #[test]
    fn test_high_accuracy_preset() {
        let pkg = presets::high_accuracy();
        assert_eq!(pkg.decoder_config.beam_width, 20);
    }
}

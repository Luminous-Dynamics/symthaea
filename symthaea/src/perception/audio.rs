// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Audio Perception - Speech Recognition Integration
//!
//! Integrates symthaea-stt (HDC + LTC speech recognition) with the
//! multi-modal perception pipeline.
//!
//! ## Architecture
//!
//! ```text
//! Audio (WAV/FLAC)
//!       │
//!       ▼
//! ┌─────────────────┐
//! │ AudioProjector  │ ← LTC temporal dynamics
//! │ (symthaea-stt)  │
//! └────────┬────────┘
//!          │ BinaryHV (2048-bit)
//!          ▼
//! ┌─────────────────┐
//! │ PhonemeDecoder  │ ← Hopfield associative memory
//! │                 │
//! └────────┬────────┘
//!          │ Phoneme sequence
//!          ▼
//! ┌─────────────────┐
//! │ Bridge to Core  │ ← BinaryHV → ContinuousHV conversion
//! │                 │
//! └────────┬────────┘
//!          │
//!          ▼
//! PerceptionInput(Auditory)
//! ```

#[cfg(feature = "voice-stt")]
use anyhow::Context as _;
use anyhow::Result;
#[cfg(not(feature = "voice-stt"))]
use anyhow::anyhow;
use std::path::Path;

// Re-define types locally when full_perception is not available
// This allows the audio module to work standalone

/// Types of sensory modalities (local copy for standalone use)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalityType {
    /// Visual/image input
    Visual,
    /// Auditory/sound input
    Auditory,
    /// Textual input
    Textual,
}

/// Perception input (simplified for audio-only use)
#[derive(Debug, Clone)]
pub struct AudioInput {
    /// Input identifier
    pub id: String,
    /// Modality type
    pub modality: ModalityType,
    /// Embedded representation (as f32 vector)
    pub embedding: Option<Vec<f32>>,
    /// Confidence score
    pub confidence: f32,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl AudioInput {
    /// Create a new audio input
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            modality: ModalityType::Auditory,
            embedding: None,
            confidence: 1.0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Audio perception configuration
#[derive(Debug, Clone)]
pub struct AudioPerceptionConfig {
    /// Path to trained phoneme prototypes
    pub prototypes_path: Option<String>,
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Enable prosody analysis
    pub enable_prosody: bool,
    /// Confidence threshold for phoneme detection
    pub confidence_threshold: f32,
}

impl Default for AudioPerceptionConfig {
    fn default() -> Self {
        Self {
            prototypes_path: None,
            sample_rate: 16000,
            enable_prosody: true,
            confidence_threshold: 0.3,
        }
    }
}

/// Audio perception result
#[derive(Debug, Clone)]
pub struct AudioPerceptionResult {
    /// Decoded phonemes
    pub phonemes: Vec<String>,
    /// Per-frame confidence scores
    pub confidences: Vec<f32>,
    /// Prosody features (pitch, energy contour)
    pub prosody: Option<ProsodyFeatures>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Number of frames processed
    pub num_frames: usize,
}

/// Prosody features extracted from audio
#[derive(Debug, Clone)]
pub struct ProsodyFeatures {
    /// Average pitch (Hz)
    pub avg_pitch: f32,
    /// Pitch variance
    pub pitch_variance: f32,
    /// Average energy
    pub avg_energy: f32,
    /// Speaking rate (phonemes per second)
    pub speaking_rate: f32,
}

/// Audio perception system - bridges symthaea-stt to perception pipeline
#[cfg(feature = "voice-stt")]
pub struct AudioPerception {
    config: AudioPerceptionConfig,
    projector: symthaea_stt::AudioProjector,
    decoder: symthaea_stt::PhonemeDecoder,
    has_prototypes: bool,
}

#[cfg(feature = "voice-stt")]
impl AudioPerception {
    /// Create a new audio perception system
    pub fn new(config: AudioPerceptionConfig) -> Result<Self> {
        let projector = symthaea_stt::AudioProjector::default_config();
        let mut decoder = symthaea_stt::PhonemeDecoder::new();
        let mut has_prototypes = false;

        // Load prototypes if available
        if let Some(ref path) = config.prototypes_path {
            if Path::new(path).exists() {
                let prototypes = symthaea_stt::TrainedPrototypes::load(path)
                    .context("Failed to load phoneme prototypes")?;
                decoder.load_prototypes(&prototypes.as_pairs());
                has_prototypes = true;
            }
        }

        Ok(Self {
            config,
            projector,
            decoder,
            has_prototypes,
        })
    }

    /// Process audio file and return perception input
    pub fn process_file(&mut self, path: &Path) -> Result<AudioInput> {
        let start = std::time::Instant::now();

        // Load audio
        let (samples, _sample_rate) =
            symthaea_stt::AudioFrontend::load_wav(path).context("Failed to load audio file")?;

        // Project through LTC
        let hvs = self.projector.project(&samples);

        // Decode phonemes (if prototypes loaded)
        let (phonemes, confidences) = if self.has_prototypes {
            let mut ph = Vec::new();
            let mut cf = Vec::new();
            for hv in &hvs {
                if let Some((phoneme, confidence)) = self.decoder.decode(hv) {
                    ph.push(phoneme);
                    cf.push(confidence);
                }
            }
            (ph, cf)
        } else {
            (Vec::new(), Vec::new())
        };

        let processing_time = start.elapsed().as_secs_f64() * 1000.0;

        // Create combined HV from all frames (bundled representation)
        let combined_hv = symthaea_stt::bundle(&hvs);

        // Convert BinaryHV to continuous f32 embedding for perception pipeline
        // Use core dimension (16,384) for compatibility with symthaea-core
        let embedding = combined_hv.to_core_continuous();

        // Build audio input
        let mut input = AudioInput::new(
            path.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "audio".to_string()),
        )
        .with_embedding(embedding);

        // Add metadata
        input = input
            .with_metadata("phonemes", phonemes.join(" "))
            .with_metadata("num_frames", hvs.len().to_string())
            .with_metadata("processing_time_ms", format!("{:.2}", processing_time));

        // Set confidence based on decoder average
        if !confidences.is_empty() {
            let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
            input.confidence = avg_confidence;
        }

        Ok(input)
    }

    /// Process raw audio samples
    pub fn process_samples(&mut self, samples: &[f32]) -> Result<AudioPerceptionResult> {
        let start = std::time::Instant::now();

        // Project through LTC
        let hvs = self.projector.project(samples);

        // Decode phonemes
        let (phonemes, confidences) = if self.has_prototypes {
            let mut ph = Vec::new();
            let mut cf = Vec::new();
            for hv in &hvs {
                if let Some((phoneme, confidence)) = self.decoder.decode(hv) {
                    ph.push(phoneme);
                    cf.push(confidence);
                }
            }
            (ph, cf)
        } else {
            (Vec::new(), Vec::new())
        };

        let processing_time = start.elapsed().as_secs_f64() * 1000.0;

        // Extract prosody if enabled
        let prosody = if self.config.enable_prosody {
            Some(self.extract_prosody(samples, &hvs))
        } else {
            None
        };

        Ok(AudioPerceptionResult {
            phonemes,
            confidences,
            prosody,
            processing_time_ms: processing_time,
            num_frames: hvs.len(),
        })
    }

    /// Extract prosody features from audio
    fn extract_prosody(&self, samples: &[f32], _hvs: &[symthaea_stt::HV16]) -> ProsodyFeatures {
        // Simple prosody extraction
        if samples.is_empty() {
            return ProsodyFeatures {
                avg_pitch: 0.0,
                pitch_variance: 0.0,
                avg_energy: 0.0,
                speaking_rate: 0.0,
            };
        }
        let energy: f32 = samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32;

        // Count zero crossings as rough pitch proxy
        let mut zero_crossings = 0u32;
        for i in 1..samples.len() {
            if (samples[i - 1] > 0.0) != (samples[i] > 0.0) {
                zero_crossings += 1;
            }
        }
        let zc_rate = zero_crossings as f32 / samples.len() as f32;
        let rough_pitch = zc_rate * self.config.sample_rate as f32 / 2.0;

        ProsodyFeatures {
            avg_pitch: rough_pitch,
            pitch_variance: 0.0, // Would need more sophisticated analysis
            avg_energy: energy,
            speaking_rate: 0.0, // Would need phoneme timing
        }
    }

    /// Check if prototypes are loaded
    pub fn has_prototypes(&self) -> bool {
        self.has_prototypes
    }
}

/// Stub implementation when voice-stt feature is not enabled
#[cfg(not(feature = "voice-stt"))]
#[allow(dead_code)] // Config reserved for feature-enabled mode
pub struct AudioPerception {
    config: AudioPerceptionConfig,
}

#[cfg(not(feature = "voice-stt"))]
impl AudioPerception {
    pub fn new(config: AudioPerceptionConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn process_file(&mut self, _path: &Path) -> Result<AudioInput> {
        Err(anyhow!("Audio perception requires 'voice-stt' feature"))
    }

    pub fn process_samples(&mut self, _samples: &[f32]) -> Result<AudioPerceptionResult> {
        Err(anyhow!("Audio perception requires 'voice-stt' feature"))
    }

    pub fn has_prototypes(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // AudioPerceptionConfig Tests
    // =========================================================================

    #[test]
    fn test_audio_perception_config_default() {
        let config = AudioPerceptionConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert!(config.enable_prosody);
        assert!(config.prototypes_path.is_none());
        assert!((config.confidence_threshold - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_audio_perception_config_custom() {
        let config = AudioPerceptionConfig {
            prototypes_path: Some("/path/to/prototypes".to_string()),
            sample_rate: 22050,
            enable_prosody: false,
            confidence_threshold: 0.5,
        };
        assert_eq!(config.sample_rate, 22050);
        assert!(!config.enable_prosody);
        assert!(config.prototypes_path.is_some());
    }

    // =========================================================================
    // AudioInput Tests
    // =========================================================================

    #[test]
    fn test_audio_input_creation() {
        let input = AudioInput::new("test");
        assert_eq!(input.modality, ModalityType::Auditory);
        assert_eq!(input.id, "test");
        assert!(input.embedding.is_none());
        assert!((input.confidence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_input_with_embedding() {
        let embedding = vec![0.1, 0.2, 0.3];
        let input = AudioInput::new("test").with_embedding(embedding.clone());
        assert!(input.embedding.is_some());
        assert_eq!(input.embedding.unwrap(), embedding);
    }

    #[test]
    fn test_audio_input_with_metadata() {
        let input = AudioInput::new("test")
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");
        assert_eq!(input.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(input.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_audio_input_chained_builders() {
        let input = AudioInput::new("audio_file")
            .with_embedding(vec![0.5; 10])
            .with_metadata("format", "wav")
            .with_metadata("duration", "5.0");

        assert_eq!(input.id, "audio_file");
        assert!(input.embedding.is_some());
        assert_eq!(input.metadata.len(), 2);
    }

    // =========================================================================
    // ModalityType Tests
    // =========================================================================

    #[test]
    fn test_modality_type_equality() {
        assert_eq!(ModalityType::Auditory, ModalityType::Auditory);
        assert_eq!(ModalityType::Visual, ModalityType::Visual);
        assert_ne!(ModalityType::Auditory, ModalityType::Visual);
        assert_ne!(ModalityType::Textual, ModalityType::Auditory);
    }

    #[test]
    fn test_modality_type_clone() {
        let modality = ModalityType::Auditory;
        let cloned = modality;
        assert_eq!(modality, cloned);
    }

    // =========================================================================
    // AudioPerceptionResult Tests
    // =========================================================================

    #[test]
    fn test_audio_perception_result_creation() {
        let result = AudioPerceptionResult {
            phonemes: vec!["a".to_string(), "b".to_string()],
            confidences: vec![0.9, 0.85],
            prosody: None,
            processing_time_ms: 10.5,
            num_frames: 100,
        };
        assert_eq!(result.phonemes.len(), 2);
        assert_eq!(result.confidences.len(), 2);
        assert!(result.prosody.is_none());
        assert_eq!(result.num_frames, 100);
    }

    // =========================================================================
    // ProsodyFeatures Tests
    // =========================================================================

    #[test]
    fn test_prosody_features_creation() {
        let prosody = ProsodyFeatures {
            avg_pitch: 150.0,
            pitch_variance: 25.0,
            avg_energy: 0.5,
            speaking_rate: 4.5,
        };
        assert!((prosody.avg_pitch - 150.0).abs() < 0.01);
        assert!((prosody.pitch_variance - 25.0).abs() < 0.01);
        assert!((prosody.avg_energy - 0.5).abs() < 0.01);
        assert!((prosody.speaking_rate - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_prosody_features_clone() {
        let prosody = ProsodyFeatures {
            avg_pitch: 200.0,
            pitch_variance: 30.0,
            avg_energy: 0.7,
            speaking_rate: 5.0,
        };
        let cloned = prosody.clone();
        assert!((prosody.avg_pitch - cloned.avg_pitch).abs() < 0.01);
    }

    // =========================================================================
    // AudioPerception Stub Tests (when voice-stt is not enabled)
    // =========================================================================

    #[test]
    fn test_audio_perception_stub_creation() {
        let config = AudioPerceptionConfig::default();
        let result = AudioPerception::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_audio_perception_stub_has_no_prototypes() {
        let config = AudioPerceptionConfig::default();
        let perception = AudioPerception::new(config).unwrap();
        assert!(!perception.has_prototypes());
    }

    #[cfg(not(feature = "voice-stt"))]
    #[test]
    fn test_audio_perception_stub_process_file_errors() {
        use std::path::Path;
        let config = AudioPerceptionConfig::default();
        let mut perception = AudioPerception::new(config).unwrap();
        let result = perception.process_file(Path::new("/nonexistent/file.wav"));
        assert!(result.is_err());
    }

    #[cfg(not(feature = "voice-stt"))]
    #[test]
    fn test_audio_perception_stub_process_samples_errors() {
        let config = AudioPerceptionConfig::default();
        let mut perception = AudioPerception::new(config).unwrap();
        let result = perception.process_samples(&[0.1, 0.2, 0.3]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_audio_input_empty_id() {
        let input = AudioInput::new("");
        assert_eq!(input.id, "");
        assert_eq!(input.modality, ModalityType::Auditory);
    }

    #[test]
    fn test_audio_input_empty_embedding() {
        let input = AudioInput::new("test").with_embedding(vec![]);
        assert!(input.embedding.is_some());
        assert!(input.embedding.unwrap().is_empty());
    }

    #[test]
    fn test_audio_input_unicode_metadata() {
        let input = AudioInput::new("test")
            .with_metadata("language", "日本語")
            .with_metadata("emoji", "🎵");
        assert_eq!(input.metadata.get("language"), Some(&"日本語".to_string()));
        assert_eq!(input.metadata.get("emoji"), Some(&"🎵".to_string()));
    }
}

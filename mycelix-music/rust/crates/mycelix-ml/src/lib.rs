// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix ML Inference Engine
//!
//! High-performance machine learning inference for audio analysis:
//! - ONNX model loading and execution via tract-onnx
//! - Genre classification
//! - Mood detection
//! - Instrument recognition
//! - Vocal detection
//! - Audio embeddings for similarity
//!
//! # Example
//!
//! ```ignore
//! use mycelix_ml::{InferenceEngine, ModelRegistry, ModelManager};
//! use std::sync::Arc;
//!
//! // Create engine with model registry
//! let registry = Arc::new(ModelRegistry::new());
//! let engine = InferenceEngine::new(registry);
//!
//! // Analyze audio
//! let genre = engine.classify_genre(&audio_samples, 44100).await?;
//! let mood = engine.analyze_mood(&audio_samples, 44100).await?;
//! ```

pub mod model_manager;
pub mod downloader;
pub mod similarity;
pub mod stems;

pub use model_manager::{ModelBundle, ModelInfo, ModelManager, ModelState};
pub use similarity::{
    SimilarityIndex, SharedSimilarityIndex, TrackFeatures, SimilarityResult,
    SearchQuery, SimilarityError, MOOD_LABELS, GENRE_LABELS,
    mood_from_label, genre_from_label,
};
pub use downloader::{
    download_model, download_bundle, get_model_sources, get_model_source,
    is_model_downloaded, get_model_path, delete_model, get_total_download_size,
    ModelSource, DownloadProgress, DownloadError,
};
pub use stems::{
    StemType, Stem, StemSeparator, StemMixer, SharedStemSeparator,
    SeparationConfig, SeparationResult, StemError,
};

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tract_onnx::prelude::*;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum MlError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidInputShape { expected: String, actual: String },

    #[error("Preprocessing failed: {0}")]
    PreprocessingError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type MlResult<T> = Result<T, MlError>;

// ============================================================================
// Core Types
// ============================================================================

/// Model type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    GenreClassifier,
    MoodDetector,
    InstrumentRecognizer,
    VocalDetector,
    EmbeddingExtractor,
    BpmEstimator,
    KeyDetector,
    VoiceActivityDetector,
}

/// Genre classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreResult {
    pub primary: String,
    pub confidence: f32,
    pub all_genres: Vec<(String, f32)>,
}

/// Mood analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodResult {
    pub valence: f32,
    pub energy: f32,
    pub danceability: f32,
    pub acousticness: f32,
    pub instrumentalness: f32,
    pub speechiness: f32,
    pub liveness: f32,
}

/// Instrument detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentResult {
    pub instruments: Vec<(String, f32)>,
    pub has_vocals: bool,
    pub vocal_confidence: f32,
}

/// Audio embedding for similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedding {
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub model_version: String,
}

impl AudioEmbedding {
    pub fn cosine_similarity(&self, other: &AudioEmbedding) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let dot: f32 = self.vector.iter().zip(&other.vector).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    pub fn euclidean_distance(&self, other: &AudioEmbedding) -> f32 {
        if self.dimension != other.dimension {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ============================================================================
// Audio Preprocessor
// ============================================================================

/// Preprocessor for converting audio to model input format
pub struct AudioPreprocessor {
    sample_rate: u32,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    target_length: usize,
}

impl AudioPreprocessor {
    pub fn new(sample_rate: u32, n_mels: usize) -> Self {
        Self {
            sample_rate,
            n_mels,
            n_fft: 2048,
            hop_length: 512,
            target_length: 128, // ~3 seconds at typical hop
        }
    }

    /// Convert audio samples to mel spectrogram
    pub fn to_mel_spectrogram(&self, samples: &[f32]) -> MlResult<Array2<f32>> {
        let num_frames = (samples.len() / self.hop_length).saturating_sub(1);
        let num_frames = num_frames.min(self.target_length);

        if num_frames == 0 {
            return Err(MlError::PreprocessingError("Audio too short".to_string()));
        }

        let mut mel_spec = Array2::zeros((self.n_mels, num_frames));

        // Simplified mel spectrogram computation
        // In production, would use proper FFT and mel filterbank
        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(samples.len());
            let frame = &samples[start..end];

            // Compute power spectrum (simplified)
            let power: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;

            // Distribute across mel bins (simplified)
            for mel_idx in 0..self.n_mels {
                let freq_ratio = mel_idx as f32 / self.n_mels as f32;
                let bin_power = power * (1.0 - (freq_ratio - 0.5).abs() * 2.0).max(0.0);
                mel_spec[[mel_idx, frame_idx]] = (bin_power + 1e-10).log10();
            }
        }

        // Normalize
        let mean = mel_spec.mean().unwrap_or(0.0);
        let std = mel_spec.std(0.0);
        if std > 0.0 {
            mel_spec.mapv_inplace(|x| (x - mean) / std);
        }

        Ok(mel_spec)
    }

    /// Compute chromagram for key detection
    pub fn to_chromagram(&self, samples: &[f32]) -> MlResult<Array2<f32>> {
        let num_frames = (samples.len() / self.hop_length).saturating_sub(1);
        let num_frames = num_frames.min(self.target_length);

        if num_frames == 0 {
            return Err(MlError::PreprocessingError("Audio too short".to_string()));
        }

        let mut chroma = Array2::zeros((12, num_frames));

        // Simplified chroma computation
        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(samples.len());
            let frame = &samples[start..end];

            // Simple energy in each pitch class (very simplified)
            for (i, sample) in frame.iter().enumerate().take(12) {
                let pitch_class = i % 12;
                chroma[[pitch_class, frame_idx]] += sample.abs();
            }
        }

        // Normalize columns
        for mut col in chroma.columns_mut() {
            let max = col.iter().cloned().fold(0.0f32, f32::max);
            if max > 0.0 {
                col.mapv_inplace(|x| x / max);
            }
        }

        Ok(chroma)
    }
}

impl Default for AudioPreprocessor {
    fn default() -> Self {
        Self::new(22050, 128)
    }
}

// ============================================================================
// Model Registry
// ============================================================================

/// Registry for managing loaded models
pub struct ModelRegistry {
    models: RwLock<HashMap<ModelType, LoadedModel>>,
    model_paths: HashMap<ModelType, String>,
}

struct LoadedModel {
    model: TypedRunnableModel<TypedModel>,
    input_shape: Vec<usize>,
    output_labels: Option<Vec<String>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            model_paths: HashMap::new(),
        }
    }

    /// Register a model path
    pub fn register_model(&mut self, model_type: ModelType, path: String) {
        self.model_paths.insert(model_type, path);
    }

    /// Load a model from disk
    pub fn load_model(&self, model_type: ModelType) -> MlResult<()> {
        let path = self.model_paths.get(&model_type)
            .ok_or_else(|| MlError::ModelNotFound(format!("{:?}", model_type)))?;

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| MlError::ModelLoadError(e.to_string()))?
            .into_optimized()
            .map_err(|e| MlError::ModelLoadError(e.to_string()))?
            .into_runnable()
            .map_err(|e| MlError::ModelLoadError(e.to_string()))?;

        // Get input shape
        let input_shape = model.model().input_fact(0)
            .map_err(|e| MlError::ModelLoadError(e.to_string()))?
            .shape.as_concrete()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let loaded = LoadedModel {
            model,
            input_shape,
            output_labels: None,
        };

        self.models.write().insert(model_type, loaded);
        Ok(())
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self, model_type: ModelType) -> bool {
        self.models.read().contains_key(&model_type)
    }

    /// Get loaded model count
    pub fn loaded_count(&self) -> usize {
        self.models.read().len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Inference Engine
// ============================================================================

/// Main ML inference engine
pub struct InferenceEngine {
    registry: Arc<ModelRegistry>,
    preprocessor: AudioPreprocessor,
    genre_labels: Vec<String>,
    instrument_labels: Vec<String>,
}

impl InferenceEngine {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self {
            registry,
            preprocessor: AudioPreprocessor::default(),
            genre_labels: vec![
                "rock", "pop", "hip-hop", "electronic", "classical",
                "jazz", "r&b", "country", "metal", "folk",
                "blues", "reggae", "latin", "punk", "indie",
            ].into_iter().map(String::from).collect(),
            instrument_labels: vec![
                "drums", "bass", "guitar", "piano", "synthesizer",
                "strings", "brass", "woodwind", "vocals", "percussion",
            ].into_iter().map(String::from).collect(),
        }
    }

    /// Classify genre from audio samples
    pub async fn classify_genre(&self, samples: &[f32], sample_rate: u32) -> MlResult<GenreResult> {
        // Resample if needed
        let samples = if sample_rate != self.preprocessor.sample_rate {
            self.resample(samples, sample_rate, self.preprocessor.sample_rate)
        } else {
            samples.to_vec()
        };

        // Extract features
        let mel_spec = self.preprocessor.to_mel_spectrogram(&samples)?;

        // Run inference (using heuristics if model not loaded)
        let probabilities = self.classify_with_model_or_heuristic(&mel_spec, ModelType::GenreClassifier);

        // Find top genres
        let mut genre_scores: Vec<(String, f32)> = self.genre_labels
            .iter()
            .cloned()
            .zip(probabilities.iter().cloned())
            .collect();

        genre_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (primary, confidence) = genre_scores.first()
            .cloned()
            .unwrap_or(("unknown".to_string(), 0.0));

        Ok(GenreResult {
            primary,
            confidence,
            all_genres: genre_scores,
        })
    }

    /// Analyze mood from audio samples
    pub async fn analyze_mood(&self, samples: &[f32], sample_rate: u32) -> MlResult<MoodResult> {
        let samples = if sample_rate != self.preprocessor.sample_rate {
            self.resample(samples, sample_rate, self.preprocessor.sample_rate)
        } else {
            samples.to_vec()
        };

        let mel_spec = self.preprocessor.to_mel_spectrogram(&samples)?;

        // Extract mood features (heuristic-based)
        let energy = self.compute_energy(&samples);
        let spectral_centroid = self.compute_spectral_centroid(&mel_spec);
        let spectral_flatness = self.compute_spectral_flatness(&mel_spec);
        let zero_crossing_rate = self.compute_zcr(&samples);

        // Map to mood dimensions
        let valence = (spectral_centroid * 0.5 + (1.0 - spectral_flatness) * 0.5).clamp(0.0, 1.0);
        let energy_score = energy.clamp(0.0, 1.0);
        let danceability = ((energy + (1.0 - spectral_flatness)) / 2.0).clamp(0.0, 1.0);
        let acousticness = spectral_flatness.clamp(0.0, 1.0);
        let instrumentalness = (1.0 - self.estimate_vocal_presence(&samples)).clamp(0.0, 1.0);
        let speechiness = self.estimate_speech_content(&samples, zero_crossing_rate);
        let liveness = self.estimate_liveness(&samples);

        Ok(MoodResult {
            valence,
            energy: energy_score,
            danceability,
            acousticness,
            instrumentalness,
            speechiness,
            liveness,
        })
    }

    /// Detect instruments in audio
    pub async fn detect_instruments(&self, samples: &[f32], sample_rate: u32) -> MlResult<InstrumentResult> {
        let samples = if sample_rate != self.preprocessor.sample_rate {
            self.resample(samples, sample_rate, self.preprocessor.sample_rate)
        } else {
            samples.to_vec()
        };

        let mel_spec = self.preprocessor.to_mel_spectrogram(&samples)?;

        // Heuristic instrument detection
        let spectral_centroid = self.compute_spectral_centroid(&mel_spec);
        let spectral_flatness = self.compute_spectral_flatness(&mel_spec);
        let energy = self.compute_energy(&samples);
        let zcr = self.compute_zcr(&samples);

        let mut instruments = Vec::new();

        // Bass detection (low spectral centroid)
        if spectral_centroid < 0.3 {
            instruments.push(("bass".to_string(), 0.7 - spectral_centroid));
        }

        // Drums detection (high ZCR, percussive)
        if zcr > 0.1 {
            instruments.push(("drums".to_string(), (zcr * 5.0).min(0.9)));
        }

        // Guitar detection (mid-range)
        if spectral_centroid > 0.2 && spectral_centroid < 0.5 {
            instruments.push(("guitar".to_string(), 0.5));
        }

        // Piano detection (clear harmonics, low flatness)
        if spectral_flatness < 0.3 && spectral_centroid > 0.3 {
            instruments.push(("piano".to_string(), 0.5));
        }

        // Synthesizer detection (high flatness)
        if spectral_flatness > 0.5 {
            instruments.push(("synthesizer".to_string(), spectral_flatness));
        }

        // Vocal detection
        let vocal_confidence = self.estimate_vocal_presence(&samples);
        let has_vocals = vocal_confidence > 0.5;

        if has_vocals {
            instruments.push(("vocals".to_string(), vocal_confidence));
        }

        instruments.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(InstrumentResult {
            instruments,
            has_vocals,
            vocal_confidence,
        })
    }

    /// Extract audio embedding for similarity search
    pub async fn extract_embedding(&self, samples: &[f32], sample_rate: u32) -> MlResult<AudioEmbedding> {
        let samples = if sample_rate != self.preprocessor.sample_rate {
            self.resample(samples, sample_rate, self.preprocessor.sample_rate)
        } else {
            samples.to_vec()
        };

        let mel_spec = self.preprocessor.to_mel_spectrogram(&samples)?;
        let chroma = self.preprocessor.to_chromagram(&samples)?;

        // Create embedding from features
        let mut embedding = Vec::with_capacity(256);

        // Mel statistics (64 dims)
        for row in mel_spec.rows() {
            embedding.push(row.mean().unwrap_or(0.0));
            embedding.push(row.std(0.0));
        }

        // Ensure we have exactly 256 dimensions
        embedding.truncate(256);
        while embedding.len() < 256 {
            embedding.push(0.0);
        }

        // Normalize embedding
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(AudioEmbedding {
            vector: embedding,
            dimension: 256,
            model_version: "heuristic-v1".to_string(),
        })
    }

    /// Estimate BPM using ML or heuristics
    pub async fn estimate_bpm(&self, samples: &[f32], sample_rate: u32) -> MlResult<f32> {
        // Onset detection
        let hop_size = sample_rate as usize / 50; // 20ms hops
        let mut onset_envelope = Vec::new();
        let mut prev_energy = 0.0f32;

        for chunk in samples.chunks(hop_size) {
            let energy: f32 = chunk.iter().map(|x| x * x).sum();
            let diff = (energy - prev_energy).max(0.0);
            onset_envelope.push(diff);
            prev_energy = energy;
        }

        // Autocorrelation for tempo
        let min_lag = (60.0 / 200.0 * 50.0) as usize; // 200 BPM max
        let max_lag = (60.0 / 60.0 * 50.0) as usize;  // 60 BPM min

        let mut best_lag = min_lag;
        let mut best_corr = 0.0f32;

        for lag in min_lag..max_lag.min(onset_envelope.len() / 2) {
            let mut corr = 0.0f32;
            for i in 0..onset_envelope.len() - lag {
                corr += onset_envelope[i] * onset_envelope[i + lag];
            }
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        let bpm = 60.0 * 50.0 / best_lag as f32;
        Ok(bpm.clamp(60.0, 200.0))
    }

    /// Detect musical key
    pub async fn detect_key(&self, samples: &[f32], sample_rate: u32) -> MlResult<(String, f32)> {
        let samples = if sample_rate != self.preprocessor.sample_rate {
            self.resample(samples, sample_rate, self.preprocessor.sample_rate)
        } else {
            samples.to_vec()
        };

        let chroma = self.preprocessor.to_chromagram(&samples)?;

        // Krumhansl-Kessler key profiles
        let major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
        let minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

        let key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

        // Average chroma across time
        let avg_chroma: Vec<f32> = chroma.rows()
            .into_iter()
            .map(|row| row.mean().unwrap_or(0.0))
            .collect();

        let mut best_key = "C".to_string();
        let mut best_corr = f32::NEG_INFINITY;
        let mut is_minor = false;

        // Correlate with all possible keys
        for rotation in 0..12 {
            // Rotate chroma
            let rotated: Vec<f32> = (0..12)
                .map(|i| avg_chroma[(i + rotation) % 12])
                .collect();

            // Correlate with major
            let major_corr: f32 = rotated.iter()
                .zip(major_profile.iter())
                .map(|(a, b)| a * (*b as f32))
                .sum();

            if major_corr > best_corr {
                best_corr = major_corr;
                best_key = format!("{} major", key_names[rotation]);
                is_minor = false;
            }

            // Correlate with minor
            let minor_corr: f32 = rotated.iter()
                .zip(minor_profile.iter())
                .map(|(a, b)| a * (*b as f32))
                .sum();

            if minor_corr > best_corr {
                best_corr = minor_corr;
                best_key = format!("{} minor", key_names[rotation]);
                is_minor = true;
            }
        }

        // Normalize confidence
        let confidence = (best_corr / 50.0).clamp(0.0, 1.0);

        Ok((best_key, confidence))
    }

    // Helper methods

    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx = src_idx as usize;
            let frac = src_idx - idx as f64;

            let sample = if idx + 1 < samples.len() {
                samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
            } else if idx < samples.len() {
                samples[idx]
            } else {
                0.0
            };
            resampled.push(sample);
        }

        resampled
    }

    fn classify_with_model_or_heuristic(&self, mel_spec: &Array2<f32>, _model_type: ModelType) -> Vec<f32> {
        // Heuristic classification based on spectral features
        let spectral_centroid = self.compute_spectral_centroid(mel_spec);
        let spectral_flatness = self.compute_spectral_flatness(mel_spec);

        let mut probs = vec![0.0f32; self.genre_labels.len()];

        // Simple heuristic mapping
        if spectral_centroid > 0.6 && spectral_flatness > 0.4 {
            probs[3] = 0.7; // electronic
        } else if spectral_centroid < 0.3 {
            probs[0] = 0.5; // rock
            probs[8] = 0.4; // metal
        } else if spectral_flatness < 0.2 {
            probs[4] = 0.6; // classical
            probs[5] = 0.3; // jazz
        } else {
            probs[1] = 0.5; // pop
            probs[2] = 0.3; // hip-hop
        }

        // Normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        probs
    }

    fn compute_energy(&self, samples: &[f32]) -> f32 {
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        (rms * 5.0).min(1.0)
    }

    fn compute_spectral_centroid(&self, mel_spec: &Array2<f32>) -> f32 {
        let mut weighted_sum = 0.0f32;
        let mut total = 0.0f32;

        for (i, row) in mel_spec.rows().into_iter().enumerate() {
            let energy: f32 = row.iter().map(|x| x.abs()).sum();
            weighted_sum += energy * i as f32;
            total += energy;
        }

        if total > 0.0 {
            weighted_sum / total / mel_spec.nrows() as f32
        } else {
            0.5
        }
    }

    fn compute_spectral_flatness(&self, mel_spec: &Array2<f32>) -> f32 {
        let eps = 1e-10f32;

        let mut flatness_sum = 0.0f32;
        let mut count = 0;

        for col in mel_spec.columns() {
            let col_abs: Vec<f32> = col.iter().map(|x| x.abs() + eps).collect();
            let geo_mean = col_abs.iter().map(|x| x.ln()).sum::<f32>() / col_abs.len() as f32;
            let geo_mean = geo_mean.exp();
            let arith_mean = col_abs.iter().sum::<f32>() / col_abs.len() as f32;

            if arith_mean > eps {
                flatness_sum += geo_mean / arith_mean;
                count += 1;
            }
        }

        if count > 0 {
            flatness_sum / count as f32
        } else {
            0.5
        }
    }

    fn compute_zcr(&self, samples: &[f32]) -> f32 {
        let mut crossings = 0;
        for window in samples.windows(2) {
            if (window[0] >= 0.0) != (window[1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / samples.len() as f32
    }

    fn estimate_vocal_presence(&self, samples: &[f32]) -> f32 {
        // Simple vocal presence estimation based on harmonic structure
        let zcr = self.compute_zcr(samples);
        let energy = self.compute_energy(samples);

        // Vocals typically have mid-range ZCR and moderate energy
        let vocal_score = if zcr > 0.05 && zcr < 0.2 && energy > 0.1 {
            0.6 + (1.0 - (zcr - 0.1).abs() * 10.0).max(0.0) * 0.3
        } else {
            0.2
        };

        vocal_score.clamp(0.0, 1.0)
    }

    fn estimate_speech_content(&self, samples: &[f32], zcr: f32) -> f32 {
        // Speech typically has higher ZCR than singing
        if zcr > 0.15 {
            ((zcr - 0.15) * 5.0).min(0.8)
        } else {
            0.1
        }
    }

    fn estimate_liveness(&self, samples: &[f32]) -> f32 {
        // Estimate if recording is live (more dynamic range, audience noise)
        let chunks: Vec<f32> = samples.chunks(4410)
            .map(|c| c.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        if chunks.is_empty() {
            return 0.5;
        }

        let mean: f32 = chunks.iter().sum::<f32>() / chunks.len() as f32;
        let variance: f32 = chunks.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / chunks.len() as f32;

        // Higher variance suggests live recording
        (variance * 10.0).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Batch Processing
// ============================================================================

/// Batch processor for multiple audio files
pub struct BatchProcessor {
    engine: Arc<InferenceEngine>,
    max_concurrent: usize,
}

impl BatchProcessor {
    pub fn new(engine: Arc<InferenceEngine>, max_concurrent: usize) -> Self {
        Self {
            engine,
            max_concurrent,
        }
    }

    /// Process multiple audio samples in parallel
    pub async fn process_batch<F, T>(
        &self,
        items: Vec<(Vec<f32>, u32)>,
        processor: F,
    ) -> Vec<MlResult<T>>
    where
        F: Fn(&InferenceEngine, &[f32], u32) -> MlResult<T> + Send + Sync + Clone,
        T: Send,
    {
        use futures::stream::{self, StreamExt};

        let engine = self.engine.clone();

        stream::iter(items)
            .map(|(samples, sample_rate)| {
                let engine = engine.clone();
                let processor = processor.clone();
                async move {
                    processor(&engine, &samples, sample_rate)
                }
            })
            .buffer_unordered(self.max_concurrent)
            .collect()
            .await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_audio(duration_sec: f32, frequency: f32, sample_rate: u32) -> Vec<f32> {
        let num_samples = (duration_sec * sample_rate as f32) as usize;
        (0..num_samples)
            .map(|i| (i as f32 / sample_rate as f32 * frequency * 2.0 * std::f32::consts::PI).sin() * 0.5)
            .collect()
    }

    #[tokio::test]
    async fn test_genre_classification() {
        let registry = Arc::new(ModelRegistry::new());
        let engine = InferenceEngine::new(registry);

        let audio = generate_test_audio(5.0, 440.0, 22050);
        let result = engine.classify_genre(&audio, 22050).await.unwrap();

        assert!(!result.primary.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_mood_analysis() {
        let registry = Arc::new(ModelRegistry::new());
        let engine = InferenceEngine::new(registry);

        let audio = generate_test_audio(5.0, 440.0, 22050);
        let result = engine.analyze_mood(&audio, 22050).await.unwrap();

        assert!(result.valence >= 0.0 && result.valence <= 1.0);
        assert!(result.energy >= 0.0 && result.energy <= 1.0);
    }

    #[tokio::test]
    async fn test_embedding_extraction() {
        let registry = Arc::new(ModelRegistry::new());
        let engine = InferenceEngine::new(registry);

        let audio = generate_test_audio(5.0, 440.0, 22050);
        let embedding = engine.extract_embedding(&audio, 22050).await.unwrap();

        assert_eq!(embedding.dimension, 256);
        assert_eq!(embedding.vector.len(), 256);
    }

    #[test]
    fn test_embedding_similarity() {
        let emb1 = AudioEmbedding {
            vector: vec![1.0, 0.0, 0.0],
            dimension: 3,
            model_version: "test".to_string(),
        };

        let emb2 = AudioEmbedding {
            vector: vec![1.0, 0.0, 0.0],
            dimension: 3,
            model_version: "test".to_string(),
        };

        let similarity = emb1.cosine_similarity(&emb2);
        assert!((similarity - 1.0).abs() < 0.001);
    }
}

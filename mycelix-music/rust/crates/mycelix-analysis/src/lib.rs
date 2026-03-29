// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Music Analysis Engine
//!
//! Advanced music analysis including:
//! - Genre classification
//! - Mood detection
//! - Music similarity and fingerprinting
//! - Instrument recognition
//! - Vocal detection
//! - Music structure analysis (verse, chorus, bridge)

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use async_trait::async_trait;
use rayon::prelude::*;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum AnalysisError {
    #[error("Audio data too short for analysis: need {required} samples, got {actual}")]
    AudioTooShort { required: usize, actual: usize },

    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, AnalysisError>;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreClassification {
    pub primary_genre: String,
    pub confidence: f32,
    pub secondary_genres: Vec<(String, f32)>,
    pub subgenres: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodAnalysis {
    pub valence: f32,     // Happy vs Sad (0-1)
    pub energy: f32,      // Calm vs Energetic (0-1)
    pub danceability: f32, // Low vs High (0-1)
    pub intensity: f32,   // Soft vs Intense (0-1)
    pub mood_tags: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFingerprint {
    pub hash: String,
    pub duration_ms: u64,
    pub landmarks: Vec<FingerprintLandmark>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintLandmark {
    pub time_ms: u64,
    pub frequency_bin: u32,
    pub hash: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentDetection {
    pub instruments: Vec<(String, f32)>,
    pub has_vocals: bool,
    pub vocal_gender: Option<VocalGender>,
    pub instrumentation_complexity: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VocalGender {
    Male,
    Female,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicStructure {
    pub sections: Vec<StructureSection>,
    pub estimated_key_changes: Vec<KeyChange>,
    pub tempo_changes: Vec<TempoChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureSection {
    pub section_type: SectionType,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: f32,
    pub repetition_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SectionType {
    Intro,
    Verse,
    PreChorus,
    Chorus,
    Bridge,
    Instrumental,
    Solo,
    Outro,
    Breakdown,
    Drop,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyChange {
    pub time_ms: u64,
    pub from_key: String,
    pub to_key: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoChange {
    pub time_ms: u64,
    pub from_bpm: f32,
    pub to_bpm: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScore {
    pub overall: f32,
    pub melodic: f32,
    pub rhythmic: f32,
    pub timbral: f32,
    pub harmonic: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAnalysis {
    pub genre: GenreClassification,
    pub mood: MoodAnalysis,
    pub fingerprint: AudioFingerprint,
    pub instruments: InstrumentDetection,
    pub structure: MusicStructure,
    pub duration_ms: u64,
    pub analysis_version: String,
}

// ============================================================================
// Analysis Traits
// ============================================================================

#[async_trait]
pub trait GenreClassifier: Send + Sync {
    async fn classify(&self, audio_data: &[f32], sample_rate: u32) -> Result<GenreClassification>;
}

#[async_trait]
pub trait MoodDetector: Send + Sync {
    async fn detect(&self, audio_data: &[f32], sample_rate: u32) -> Result<MoodAnalysis>;
}

#[async_trait]
pub trait Fingerprinter: Send + Sync {
    async fn fingerprint(&self, audio_data: &[f32], sample_rate: u32) -> Result<AudioFingerprint>;
    async fn compare(&self, fp1: &AudioFingerprint, fp2: &AudioFingerprint) -> Result<f32>;
}

#[async_trait]
pub trait InstrumentDetector: Send + Sync {
    async fn detect(&self, audio_data: &[f32], sample_rate: u32) -> Result<InstrumentDetection>;
}

#[async_trait]
pub trait StructureAnalyzer: Send + Sync {
    async fn analyze(&self, audio_data: &[f32], sample_rate: u32) -> Result<MusicStructure>;
}

// ============================================================================
// Heuristic-Based Implementations
// ============================================================================

pub struct HeuristicGenreClassifier {
    genre_profiles: HashMap<String, GenreProfile>,
}

#[derive(Debug, Clone)]
struct GenreProfile {
    bpm_range: (f32, f32),
    energy_range: (f32, f32),
    spectral_centroid_range: (f32, f32),
    zero_crossing_range: (f32, f32),
}

impl HeuristicGenreClassifier {
    pub fn new() -> Self {
        let mut profiles = HashMap::new();

        profiles.insert("electronic".to_string(), GenreProfile {
            bpm_range: (120.0, 150.0),
            energy_range: (0.6, 1.0),
            spectral_centroid_range: (2000.0, 5000.0),
            zero_crossing_range: (0.1, 0.3),
        });

        profiles.insert("rock".to_string(), GenreProfile {
            bpm_range: (100.0, 140.0),
            energy_range: (0.5, 0.9),
            spectral_centroid_range: (1500.0, 4000.0),
            zero_crossing_range: (0.08, 0.2),
        });

        profiles.insert("hip-hop".to_string(), GenreProfile {
            bpm_range: (80.0, 115.0),
            energy_range: (0.4, 0.8),
            spectral_centroid_range: (1000.0, 3000.0),
            zero_crossing_range: (0.05, 0.15),
        });

        profiles.insert("classical".to_string(), GenreProfile {
            bpm_range: (40.0, 120.0),
            energy_range: (0.1, 0.6),
            spectral_centroid_range: (500.0, 2500.0),
            zero_crossing_range: (0.02, 0.1),
        });

        profiles.insert("jazz".to_string(), GenreProfile {
            bpm_range: (80.0, 180.0),
            energy_range: (0.3, 0.7),
            spectral_centroid_range: (1000.0, 3500.0),
            zero_crossing_range: (0.05, 0.15),
        });

        profiles.insert("pop".to_string(), GenreProfile {
            bpm_range: (100.0, 130.0),
            energy_range: (0.5, 0.8),
            spectral_centroid_range: (1500.0, 3500.0),
            zero_crossing_range: (0.06, 0.18),
        });

        Self { genre_profiles: profiles }
    }

    fn extract_features(&self, audio_data: &[f32], sample_rate: u32) -> AudioFeatures {
        AudioFeatures {
            bpm: estimate_bpm(audio_data, sample_rate),
            energy: calculate_energy(audio_data),
            spectral_centroid: calculate_spectral_centroid(audio_data, sample_rate),
            zero_crossing_rate: calculate_zcr(audio_data, sample_rate),
        }
    }
}

#[derive(Debug)]
struct AudioFeatures {
    bpm: f32,
    energy: f32,
    spectral_centroid: f32,
    zero_crossing_rate: f32,
}

#[async_trait]
impl GenreClassifier for HeuristicGenreClassifier {
    async fn classify(&self, audio_data: &[f32], sample_rate: u32) -> Result<GenreClassification> {
        if audio_data.len() < sample_rate as usize * 5 {
            return Err(AnalysisError::AudioTooShort {
                required: sample_rate as usize * 5,
                actual: audio_data.len(),
            });
        }

        let features = self.extract_features(audio_data, sample_rate);
        let mut scores: Vec<(String, f32)> = Vec::new();

        for (genre, profile) in &self.genre_profiles {
            let mut score = 0.0f32;

            // BPM match
            if features.bpm >= profile.bpm_range.0 && features.bpm <= profile.bpm_range.1 {
                score += 0.25;
            }

            // Energy match
            if features.energy >= profile.energy_range.0 && features.energy <= profile.energy_range.1 {
                score += 0.25;
            }

            // Spectral centroid match
            if features.spectral_centroid >= profile.spectral_centroid_range.0
                && features.spectral_centroid <= profile.spectral_centroid_range.1
            {
                score += 0.25;
            }

            // ZCR match
            if features.zero_crossing_rate >= profile.zero_crossing_range.0
                && features.zero_crossing_rate <= profile.zero_crossing_range.1
            {
                score += 0.25;
            }

            scores.push((genre.clone(), score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (primary, confidence) = scores.first().cloned().unwrap_or(("unknown".to_string(), 0.0));
        let secondary: Vec<(String, f32)> = scores.into_iter().skip(1).take(3).collect();

        Ok(GenreClassification {
            primary_genre: primary,
            confidence,
            secondary_genres: secondary,
            subgenres: Vec::new(),
        })
    }
}

impl Default for HeuristicGenreClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Mood Detector Implementation
// ============================================================================

pub struct HeuristicMoodDetector;

impl HeuristicMoodDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HeuristicMoodDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MoodDetector for HeuristicMoodDetector {
    async fn detect(&self, audio_data: &[f32], sample_rate: u32) -> Result<MoodAnalysis> {
        let energy = calculate_energy(audio_data);
        let spectral_centroid = calculate_spectral_centroid(audio_data, sample_rate);
        let zcr = calculate_zcr(audio_data, sample_rate);
        let bpm = estimate_bpm(audio_data, sample_rate);

        // Heuristic mood estimation
        let valence = (spectral_centroid / 5000.0).min(1.0) * 0.5 + (1.0 - calculate_dynamic_range(audio_data) / 20.0).max(0.0) * 0.5;
        let energy_mood = energy;
        let danceability: f32 = if bpm >= 100.0 && bpm <= 130.0 { 0.8 } else if bpm >= 80.0 && bpm <= 150.0 { 0.6 } else { 0.3 };
        let intensity = energy * 0.6 + zcr * 2.0;

        let mut mood_tags = Vec::new();
        if valence > 0.6 && energy_mood > 0.6 {
            mood_tags.push(("uplifting".to_string(), 0.8));
            mood_tags.push(("happy".to_string(), 0.7));
        } else if valence < 0.4 && energy_mood < 0.4 {
            mood_tags.push(("melancholic".to_string(), 0.8));
            mood_tags.push(("sad".to_string(), 0.6));
        } else if energy_mood > 0.7 {
            mood_tags.push(("energetic".to_string(), 0.8));
            mood_tags.push(("powerful".to_string(), 0.6));
        } else if energy_mood < 0.3 {
            mood_tags.push(("calm".to_string(), 0.8));
            mood_tags.push(("relaxing".to_string(), 0.7));
        } else {
            mood_tags.push(("neutral".to_string(), 0.6));
        }

        Ok(MoodAnalysis {
            valence: valence.clamp(0.0, 1.0),
            energy: energy_mood.clamp(0.0, 1.0),
            danceability: danceability.clamp(0.0, 1.0),
            intensity: intensity.clamp(0.0, 1.0),
            mood_tags,
        })
    }
}

// ============================================================================
// Audio Fingerprinting
// ============================================================================

pub struct SpectralFingerprinter {
    fft_size: usize,
    hop_size: usize,
}

impl SpectralFingerprinter {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        Self {
            fft_size: fft_size.next_power_of_two(),
            hop_size,
        }
    }
}

impl Default for SpectralFingerprinter {
    fn default() -> Self {
        Self::new(2048, 512)
    }
}

#[async_trait]
impl Fingerprinter for SpectralFingerprinter {
    async fn fingerprint(&self, audio_data: &[f32], sample_rate: u32) -> Result<AudioFingerprint> {
        let duration_ms = (audio_data.len() as u64 * 1000) / sample_rate as u64;
        let mut landmarks = Vec::new();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Generate spectral peaks as landmarks
        for (frame_idx, chunk) in audio_data.chunks(self.hop_size).enumerate() {
            let time_ms = (frame_idx * self.hop_size) as u64 * 1000 / sample_rate as u64;

            // Find peak in this frame
            let (peak_idx, peak_val) = chunk.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .unwrap_or((0, &0.0));

            if peak_val.abs() > 0.01 {
                use std::hash::{Hash, Hasher};
                let freq_bin = peak_idx as u32;
                (frame_idx, freq_bin).hash(&mut hasher);

                landmarks.push(FingerprintLandmark {
                    time_ms,
                    frequency_bin: freq_bin,
                    hash: hasher.finish() as u32,
                });
            }
        }

        use std::hash::{Hash, Hasher};
        audio_data.len().hash(&mut hasher);
        let hash = format!("{:016x}", hasher.finish());

        Ok(AudioFingerprint {
            hash,
            duration_ms,
            landmarks,
        })
    }

    async fn compare(&self, fp1: &AudioFingerprint, fp2: &AudioFingerprint) -> Result<f32> {
        if fp1.landmarks.is_empty() || fp2.landmarks.is_empty() {
            return Ok(0.0);
        }

        let mut matches = 0;
        for l1 in &fp1.landmarks {
            for l2 in &fp2.landmarks {
                if l1.hash == l2.hash {
                    matches += 1;
                    break;
                }
            }
        }

        let similarity = matches as f32 / fp1.landmarks.len().max(fp2.landmarks.len()) as f32;
        Ok(similarity)
    }
}

// ============================================================================
// Instrument Detection
// ============================================================================

pub struct HeuristicInstrumentDetector;

impl HeuristicInstrumentDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HeuristicInstrumentDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InstrumentDetector for HeuristicInstrumentDetector {
    async fn detect(&self, audio_data: &[f32], sample_rate: u32) -> Result<InstrumentDetection> {
        let spectral_centroid = calculate_spectral_centroid(audio_data, sample_rate);
        let zcr = calculate_zcr(audio_data, sample_rate);
        let energy = calculate_energy(audio_data);

        let mut instruments = Vec::new();

        // Very heuristic instrument detection based on spectral characteristics
        if spectral_centroid < 1000.0 {
            instruments.push(("bass".to_string(), 0.7));
        }
        if spectral_centroid > 2000.0 && spectral_centroid < 4000.0 {
            instruments.push(("guitar".to_string(), 0.5));
        }
        if zcr > 0.1 {
            instruments.push(("drums".to_string(), 0.6));
        }
        if spectral_centroid > 3000.0 {
            instruments.push(("synthesizer".to_string(), 0.5));
        }

        // Vocal detection heuristic
        let has_vocals = spectral_centroid > 1000.0 && spectral_centroid < 4000.0 && energy > 0.2;
        let instrumentation_complexity = (instruments.len() as f32 / 5.0).min(1.0);

        Ok(InstrumentDetection {
            instruments,
            has_vocals,
            vocal_gender: if has_vocals { Some(VocalGender::Unknown) } else { None },
            instrumentation_complexity,
        })
    }
}

// ============================================================================
// Structure Analysis
// ============================================================================

pub struct SegmentationStructureAnalyzer {
    min_section_duration_ms: u64,
}

impl SegmentationStructureAnalyzer {
    pub fn new(min_section_duration_ms: u64) -> Self {
        Self { min_section_duration_ms }
    }
}

impl Default for SegmentationStructureAnalyzer {
    fn default() -> Self {
        Self::new(10000) // 10 seconds minimum section
    }
}

#[async_trait]
impl StructureAnalyzer for SegmentationStructureAnalyzer {
    async fn analyze(&self, audio_data: &[f32], sample_rate: u32) -> Result<MusicStructure> {
        let duration_ms = (audio_data.len() as u64 * 1000) / sample_rate as u64;
        let segment_size = (sample_rate as usize * self.min_section_duration_ms as usize) / 1000;

        let mut sections = Vec::new();
        let mut prev_energy = 0.0f32;

        for (i, chunk) in audio_data.chunks(segment_size).enumerate() {
            let start_ms = i as u64 * self.min_section_duration_ms;
            let end_ms = ((i + 1) as u64 * self.min_section_duration_ms).min(duration_ms);
            let energy = calculate_energy(chunk);

            // Heuristic section type detection
            let section_type = if i == 0 {
                SectionType::Intro
            } else if end_ms >= duration_ms - self.min_section_duration_ms {
                SectionType::Outro
            } else if energy > prev_energy * 1.3 && energy > 0.5 {
                SectionType::Chorus
            } else if energy < prev_energy * 0.7 {
                SectionType::Verse
            } else if energy > 0.7 {
                SectionType::Drop
            } else {
                SectionType::Verse
            };

            sections.push(StructureSection {
                section_type,
                start_ms,
                end_ms,
                confidence: 0.5, // Heuristic confidence
                repetition_index: None,
            });

            prev_energy = energy;
        }

        Ok(MusicStructure {
            sections,
            estimated_key_changes: Vec::new(),
            tempo_changes: Vec::new(),
        })
    }
}

// ============================================================================
// Comprehensive Analyzer
// ============================================================================

pub struct MusicAnalyzer {
    genre_classifier: Box<dyn GenreClassifier>,
    mood_detector: Box<dyn MoodDetector>,
    fingerprinter: Box<dyn Fingerprinter>,
    instrument_detector: Box<dyn InstrumentDetector>,
    structure_analyzer: Box<dyn StructureAnalyzer>,
}

impl MusicAnalyzer {
    pub fn new() -> Self {
        Self {
            genre_classifier: Box::new(HeuristicGenreClassifier::new()),
            mood_detector: Box::new(HeuristicMoodDetector::new()),
            fingerprinter: Box::new(SpectralFingerprinter::default()),
            instrument_detector: Box::new(HeuristicInstrumentDetector::new()),
            structure_analyzer: Box::new(SegmentationStructureAnalyzer::default()),
        }
    }

    pub async fn analyze(&self, audio_data: &[f32], sample_rate: u32) -> Result<ComprehensiveAnalysis> {
        let duration_ms = (audio_data.len() as u64 * 1000) / sample_rate as u64;

        // Run analyses concurrently
        let (genre, mood, fingerprint, instruments, structure) = tokio::join!(
            self.genre_classifier.classify(audio_data, sample_rate),
            self.mood_detector.detect(audio_data, sample_rate),
            self.fingerprinter.fingerprint(audio_data, sample_rate),
            self.instrument_detector.detect(audio_data, sample_rate),
            self.structure_analyzer.analyze(audio_data, sample_rate),
        );

        Ok(ComprehensiveAnalysis {
            genre: genre?,
            mood: mood?,
            fingerprint: fingerprint?,
            instruments: instruments?,
            structure: structure?,
            duration_ms,
            analysis_version: "1.0.0-heuristic".to_string(),
        })
    }

    pub async fn classify_genre(&self, audio_data: &[f32], sample_rate: u32) -> Result<GenreClassification> {
        self.genre_classifier.classify(audio_data, sample_rate).await
    }

    pub async fn detect_mood(&self, audio_data: &[f32], sample_rate: u32) -> Result<MoodAnalysis> {
        self.mood_detector.detect(audio_data, sample_rate).await
    }

    pub async fn fingerprint(&self, audio_data: &[f32], sample_rate: u32) -> Result<AudioFingerprint> {
        self.fingerprinter.fingerprint(audio_data, sample_rate).await
    }

    pub async fn compare_fingerprints(&self, fp1: &AudioFingerprint, fp2: &AudioFingerprint) -> Result<f32> {
        self.fingerprinter.compare(fp1, fp2).await
    }
}

impl Default for MusicAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

fn estimate_bpm(audio_data: &[f32], sample_rate: u32) -> f32 {
    let hop_size = sample_rate as usize / 20;
    let mut onsets = Vec::new();
    let mut prev_energy = 0.0f32;

    for (i, chunk) in audio_data.chunks(hop_size).enumerate() {
        let energy: f32 = chunk.iter().map(|s| s * s).sum();
        if energy > prev_energy * 1.5 && energy > 0.001 {
            onsets.push(i);
        }
        prev_energy = energy;
    }

    if onsets.len() < 4 {
        return 120.0; // Default BPM
    }

    let intervals: Vec<usize> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
    let avg_interval = intervals.iter().sum::<usize>() as f32 / intervals.len() as f32;
    let seconds_per_beat = avg_interval * hop_size as f32 / sample_rate as f32;
    let bpm = 60.0 / seconds_per_beat;

    bpm.clamp(60.0, 200.0)
}

fn calculate_energy(audio_data: &[f32]) -> f32 {
    let rms = (audio_data.iter().map(|s| s * s).sum::<f32>() / audio_data.len() as f32).sqrt();
    (rms * 3.0).min(1.0) // Scale RMS to roughly 0-1 range
}

fn calculate_spectral_centroid(audio_data: &[f32], sample_rate: u32) -> f32 {
    let n = 2048.min(audio_data.len());
    let mut magnitude_sum = 0.0f32;
    let mut weighted_sum = 0.0f32;

    for k in 1..n/2 {
        let freq = k as f32 * sample_rate as f32 / n as f32;
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        for (i, sample) in audio_data.iter().take(n).enumerate() {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }

        let magnitude = (real * real + imag * imag).sqrt();
        magnitude_sum += magnitude;
        weighted_sum += freq * magnitude;
    }

    if magnitude_sum > 0.0 {
        weighted_sum / magnitude_sum
    } else {
        0.0
    }
}

fn calculate_zcr(audio_data: &[f32], sample_rate: u32) -> f32 {
    let mut crossings = 0;
    for window in audio_data.windows(2) {
        if (window[0] >= 0.0) != (window[1] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f32 / (audio_data.len() as f32 / sample_rate as f32)
}

fn calculate_dynamic_range(audio_data: &[f32]) -> f32 {
    let window_size = 4410;
    let mut loudness_values: Vec<f32> = Vec::new();

    for chunk in audio_data.chunks(window_size) {
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if rms > 0.0001 {
            loudness_values.push(20.0 * rms.log10());
        }
    }

    if loudness_values.len() < 2 {
        return 0.0;
    }

    loudness_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10 = loudness_values[loudness_values.len() / 10];
    let p90 = loudness_values[loudness_values.len() * 9 / 10];
    p90 - p10
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
        let classifier = HeuristicGenreClassifier::new();
        let audio = generate_test_audio(10.0, 440.0, 44100);
        let result = classifier.classify(&audio, 44100).await.unwrap();
        assert!(!result.primary_genre.is_empty());
    }

    #[tokio::test]
    async fn test_mood_detection() {
        let detector = HeuristicMoodDetector::new();
        let audio = generate_test_audio(10.0, 440.0, 44100);
        let result = detector.detect(&audio, 44100).await.unwrap();
        assert!(result.valence >= 0.0 && result.valence <= 1.0);
        assert!(result.energy >= 0.0 && result.energy <= 1.0);
    }

    #[tokio::test]
    async fn test_fingerprinting() {
        let fingerprinter = SpectralFingerprinter::default();
        let audio = generate_test_audio(10.0, 440.0, 44100);
        let fp = fingerprinter.fingerprint(&audio, 44100).await.unwrap();
        assert!(!fp.hash.is_empty());
        assert!(fp.duration_ms > 0);
    }

    #[tokio::test]
    async fn test_comprehensive_analysis() {
        let analyzer = MusicAnalyzer::new();
        let audio = generate_test_audio(10.0, 440.0, 44100);
        let result = analyzer.analyze(&audio, 44100).await.unwrap();
        assert!(!result.genre.primary_genre.is_empty());
        assert!(result.duration_ms > 0);
    }

    #[test]
    fn test_utility_functions() {
        let audio = generate_test_audio(1.0, 440.0, 44100);
        assert!(calculate_energy(&audio) > 0.0);
        assert!(calculate_spectral_centroid(&audio, 44100) > 0.0);
        assert!(calculate_zcr(&audio, 44100) > 0.0);
    }
}

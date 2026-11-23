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

/// Schema version for frame-aligned acoustic prosody evidence.
pub const ACOUSTIC_PROSODY_SCHEMA_VERSION: u32 = 1;

/// Energy normalization applied by the current formant-track extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsodyEnergyNormalization {
    /// Frame RMS is normalized against the peak RMS of this utterance.
    UtterancePeak,
}

/// Extractor output class; this is an acoustic estimate, not a phonetic label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticProsodyFrameStatusV1 {
    /// The extractor found periodicity above its voicing threshold.
    Voiced,
    /// The frame had signal energy but no valid periodic F0 estimate.
    Unvoiced,
    /// The extractor's relative energy floor classified the frame as silent.
    Silent,
}

/// Fully disclosed parameter set for the current acoustic feature extractor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProsodyExtractorConfigV1 {
    /// Internal analysis sample rate after the extractor's resampling step.
    pub analysis_sample_rate_hz: u32,
    /// Analysis window duration in seconds.
    pub window_seconds: f32,
    /// Analysis frame rate in Hz.
    pub frame_rate_hz: f32,
    /// LPC order used by the current combined formant/F0 extractor.
    pub lpc_order: usize,
    /// Pre-emphasis coefficient used by the formant branch.
    pub preemphasis: f32,
    /// Lower F0 search limit.
    pub f0_min_hz: f32,
    /// Upper F0 search limit.
    pub f0_max_hz: f32,
    /// Periodicity threshold used for the voiced estimate.
    pub voicing_threshold: f32,
    /// Relative energy floor used for the silent estimate.
    pub silence_relative_energy: f32,
}

/// One frame of measurable acoustic prosody evidence.
///
/// The F0 field is absent for unvoiced/low-periodicity frames. The periodicity
/// field is an estimator score, not a calibrated probability of voicing.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticProsodyFrameV1 {
    /// Stream segment identity supplied by the upstream sample owner.
    pub stream_segment_index: u64,
    pub frame_index: u64,
    /// Estimator classification; not a phoneme or emotion label.
    pub status: AcousticProsodyFrameStatusV1,
    /// Half-open sample interval in the original input stream.
    pub source_start_sample: u64,
    pub source_end_sample: u64,
    /// Frame start relative to this utterance, in seconds.
    pub time_seconds: f32,
    /// Estimated F0 only when the voicing/periodicity threshold is met.
    pub f0_hz: Option<f32>,
    /// Normalized-autocorrelation periodicity score in [0, 1].
    pub periodicity: f32,
    /// RMS energy normalized by the utterance peak RMS (not calibrated SPL).
    pub energy_utterance_peak_normalized: f32,
}

/// Frame-aligned acoustic prosody features.
///
/// Summary fields are optional because silence or unvoiced input may provide no
/// valid F0 observations. Speaking rate remains unavailable until qualified
/// phoneme/word timing is supplied; it is never replaced with a fabricated 0.
#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyFeatures {
    pub schema_version: u32,
    pub extractor_id: &'static str,
    pub source_sample_rate: u32,
    /// Absolute source sample support of this processed block.
    pub source_start_sample: u64,
    pub source_end_sample: u64,
    /// Segment index is incremented by the stream owner after discontinuity.
    pub stream_segment_index: u64,
    pub extractor_config: ProsodyExtractorConfigV1,
    pub frame_rate_hz: f32,
    pub energy_normalization: ProsodyEnergyNormalization,
    pub frames: Vec<AcousticProsodyFrameV1>,
    /// Mean of voiced, valid frame F0 estimates.
    pub avg_pitch_hz: Option<f32>,
    /// Population variance of voiced, valid frame F0 estimates (Hz²).
    pub pitch_variance_hz2: Option<f32>,
    /// Mean utterance-peak-normalized energy.
    pub avg_energy_utterance_peak_normalized: Option<f32>,
    /// Phonemes/s from qualified alignment; None when no alignment is supplied.
    pub speaking_rate_phonemes_per_second: Option<f32>,
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
        if config.sample_rate == 0 {
            return Err(anyhow::anyhow!(
                "audio perception sample rate must be greater than zero"
            ));
        }
        let projector = symthaea_stt::AudioProjector::new(
            symthaea_stt::AudioConfig {
                sample_rate: config.sample_rate,
                ..Default::default()
            },
            symthaea_stt::LtcConfig::default(),
        );
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
        let (samples, sample_rate) =
            symthaea_stt::AudioFrontend::load_wav(path).context("Failed to load audio file")?;
        self.validate_sample_rate(sample_rate)?;

        // Project through the configured acoustic encoder
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

    /// Process mono PCM samples at `AudioPerceptionConfig::sample_rate`.
    ///
    /// This convenience method assumes the sample block begins at source
    /// sample zero and segment zero. Streaming callers should use
    /// process_samples_at_source_position() so overruns/resets cannot erase
    /// absolute source continuity.
    pub fn process_samples(&mut self, samples: &[f32]) -> Result<AudioPerceptionResult> {
        let sample_rate = self.config.sample_rate;
        self.process_samples_at_source_position(samples, sample_rate, 0, 0)
    }

    /// Reject PCM whose source sample rate does not match the configured HDC frontend.
    fn validate_sample_rate(&self, sample_rate: u32) -> Result<()> {
        if sample_rate == self.config.sample_rate && sample_rate > 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "audio sample-rate mismatch: received {sample_rate} Hz, perception is configured for {} Hz; resample explicitly before processing",
                self.config.sample_rate
            ))
        }
    }

    /// Validate rate, source position and sample finiteness, then process mono PCM.
    ///
    /// `source_start_sample` is the absolute first sample in the immutable
    /// source clock. `stream_segment_index` must advance when the caller
    /// declares a discontinuity; features never bridge segments.
    pub fn process_samples_at_source_position(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        source_start_sample: u64,
        stream_segment_index: u64,
    ) -> Result<AudioPerceptionResult> {
        self.validate_sample_rate(sample_rate)?;
        if let Some(invalid_index) = samples.iter().position(|sample| !sample.is_finite()) {
            return Err(anyhow::anyhow!(
                "audio sample at block offset {invalid_index} is not finite"
            ));
        }
        let sample_count = u64::try_from(samples.len())
            .map_err(|_| anyhow::anyhow!("audio sample count exceeds source index range"))?;
        let source_end_sample = source_start_sample
            .checked_add(sample_count)
            .ok_or_else(|| anyhow::anyhow!("audio source sample interval overflows u64"))?;

        let start = std::time::Instant::now();

        // Project through LTC.
        let hvs = self.projector.project(samples);

        // Decode phonemes.
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

        // Extract prosody if enabled.
        let prosody = if self.config.enable_prosody {
            Some(self.extract_prosody(
                samples,
                sample_rate,
                source_start_sample,
                source_end_sample,
                stream_segment_index,
            ))
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

    /// Validate source rate when known, then process mono PCM starting at sample zero.
    pub fn process_samples_at_rate(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AudioPerceptionResult> {
        self.process_samples_at_source_position(samples, sample_rate, 0, 0)
    }

    /// Extract frame-aligned F0, periodicity and normalized RMS evidence.
    ///
    /// The existing formant extractor internally resamples to its configured
    /// analysis rate. Its F1-F3 fallback peaks are deliberately not surfaced
    /// here as measured formants.
    fn extract_prosody(
        &self,
        samples: &[f32],
        sample_rate: u32,
        source_start_sample: u64,
        source_end_sample: u64,
        stream_segment_index: u64,
    ) -> ProsodyFeatures {
        use symthaea_vocal_tract::formant_extraction::{ExtractionConfig, extract_formant_frames};
        use symthaea_vocal_tract::types::SourceType;

        const EXTRACTOR_ID: &str = "symthaea-vocal-tract.lpc-autocorrelation.v1";
        let extraction_config = ExtractionConfig::default();
        let extracted = extract_formant_frames(samples, sample_rate, &extraction_config);
        let window_samples = (extraction_config.window_secs * sample_rate as f32).round() as u64;
        let extractor_config = ProsodyExtractorConfigV1 {
            analysis_sample_rate_hz: extraction_config.analysis_sr,
            window_seconds: extraction_config.window_secs,
            frame_rate_hz: extraction_config.frame_rate,
            lpc_order: extraction_config.lpc_order,
            preemphasis: extraction_config.preemphasis,
            f0_min_hz: extraction_config.f0_min,
            f0_max_hz: extraction_config.f0_max,
            voicing_threshold: extraction_config.voicing_threshold,
            silence_relative_energy: extraction_config.silence_rel_energy,
        };

        let frames: Vec<AcousticProsodyFrameV1> = extracted
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                let relative_start = (frame.time.max(0.0) * sample_rate as f32).round() as u64;
                let frame_source_start = source_start_sample.saturating_add(relative_start);
                let frame_source_end = frame_source_start
                    .saturating_add(window_samples)
                    .min(source_end_sample);
                let f0_hz =
                    if frame.f0 > 0.0 && frame.voicing >= extraction_config.voicing_threshold {
                        Some(frame.f0)
                    } else {
                        None
                    };
                let status = match frame.source_type {
                    SourceType::Silent => AcousticProsodyFrameStatusV1::Silent,
                    SourceType::Vowel if f0_hz.is_some() => AcousticProsodyFrameStatusV1::Voiced,
                    _ => AcousticProsodyFrameStatusV1::Unvoiced,
                };

                AcousticProsodyFrameV1 {
                    stream_segment_index,
                    frame_index: index as u64,
                    status,
                    source_start_sample: frame_source_start,
                    source_end_sample: frame_source_end,
                    time_seconds: frame.time,
                    f0_hz,
                    periodicity: frame.voicing.clamp(0.0, 1.0),
                    energy_utterance_peak_normalized: frame.energy.clamp(0.0, 1.0),
                }
            })
            .collect();

        let voiced_f0: Vec<f32> = frames.iter().filter_map(|frame| frame.f0_hz).collect();
        let avg_pitch_hz = if voiced_f0.is_empty() {
            None
        } else {
            Some(voiced_f0.iter().sum::<f32>() / voiced_f0.len() as f32)
        };
        let pitch_variance_hz2 = if voiced_f0.len() < 2 {
            None
        } else {
            let mean = avg_pitch_hz.unwrap_or(0.0);
            Some(
                voiced_f0
                    .iter()
                    .map(|pitch| (pitch - mean).powi(2))
                    .sum::<f32>()
                    / voiced_f0.len() as f32,
            )
        };
        let avg_energy_utterance_peak_normalized = if frames.is_empty() {
            None
        } else {
            Some(
                frames
                    .iter()
                    .map(|frame| frame.energy_utterance_peak_normalized)
                    .sum::<f32>()
                    / frames.len() as f32,
            )
        };

        ProsodyFeatures {
            schema_version: ACOUSTIC_PROSODY_SCHEMA_VERSION,
            extractor_id: EXTRACTOR_ID,
            source_sample_rate: sample_rate,
            source_start_sample,
            source_end_sample,
            stream_segment_index,
            extractor_config,
            frame_rate_hz: extraction_config.frame_rate,
            energy_normalization: ProsodyEnergyNormalization::UtterancePeak,
            frames,
            avg_pitch_hz,
            pitch_variance_hz2,
            avg_energy_utterance_peak_normalized,
            // Speaking rate requires word/phone timing from the aligned ASR
            // contract; waveform periodicity is not a valid substitute.
            speaking_rate_phonemes_per_second: None,
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

    pub fn process_samples_at_rate(
        &mut self,
        _samples: &[f32],
        _sample_rate: u32,
    ) -> Result<AudioPerceptionResult> {
        Err(anyhow!("Audio perception requires 'voice-stt' feature"))
    }

    pub fn process_samples_at_source_position(
        &mut self,
        _samples: &[f32],
        _sample_rate: u32,
        _source_start_sample: u64,
        _stream_segment_index: u64,
    ) -> Result<AudioPerceptionResult> {
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

    #[cfg(feature = "voice-stt")]
    #[test]
    fn audio_perception_rejects_zero_sample_rate_configuration() {
        let config = AudioPerceptionConfig {
            sample_rate: 0,
            ..AudioPerceptionConfig::default()
        };
        assert!(AudioPerception::new(config).is_err());
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
    // Frame-aligned acoustic prosody tests
    // =========================================================================

    #[cfg(feature = "voice-stt")]
    fn sine_wave(sample_rate: u32, frequency_hz: f32, seconds: f32) -> Vec<f32> {
        let sample_count = (sample_rate as f32 * seconds).round() as usize;
        (0..sample_count)
            .map(|index| {
                let time = index as f32 / sample_rate as f32;
                0.4 * (std::f32::consts::TAU * frequency_hz * time).sin()
            })
            .collect()
    }

    #[cfg(feature = "voice-stt")]
    #[test]
    fn prosody_extracts_frame_aligned_f0_and_preserves_source_support() {
        let perception = AudioPerception::new(AudioPerceptionConfig::default()).unwrap();
        let samples = sine_wave(16_000, 200.0, 1.5);
        let prosody = perception.extract_prosody(&samples, 16_000);

        assert_eq!(prosody.schema_version, ACOUSTIC_PROSODY_SCHEMA_VERSION);
        assert_eq!(
            prosody.extractor_id,
            "symthaea-vocal-tract.lpc-autocorrelation.v1"
        );
        assert_eq!(prosody.source_sample_rate, 16_000);
        assert_eq!(
            prosody.energy_normalization,
            ProsodyEnergyNormalization::UtterancePeak
        );
        assert!(prosody.frames.len() > 250);
        let mean_f0 = prosody.avg_pitch_hz.expect("voiced tone should yield F0");
        assert!(
            (180.0..220.0).contains(&mean_f0),
            "estimated F0 {mean_f0} Hz should be close to 200 Hz"
        );
        assert!(prosody.pitch_variance_hz2.is_some());
        assert!(prosody.speaking_rate_phonemes_per_second.is_none());
        assert!(
            prosody
                .frames
                .iter()
                .all(|frame| frame.energy_utterance_peak_normalized <= 0.851)
        );

        for pair in prosody.frames.windows(2) {
            assert!(pair[1].time_seconds > pair[0].time_seconds);
            assert!(pair[1].source_start_sample > pair[0].source_start_sample);
            assert!(pair[0].source_end_sample > pair[0].source_start_sample);
            assert!(pair[0].f0_hz.is_some());
        }
    }

    #[cfg(feature = "voice-stt")]
    #[test]
    fn prosody_marks_silence_as_unvoiced_instead_of_zero_pitch_observation() {
        let perception = AudioPerception::new(AudioPerceptionConfig::default()).unwrap();
        let silence = vec![0.0; 16_000];
        let prosody = perception.extract_prosody(&silence, 16_000);

        assert!(!prosody.frames.is_empty());
        assert!(prosody.avg_pitch_hz.is_none());
        assert!(prosody.pitch_variance_hz2.is_none());
        assert_eq!(prosody.avg_energy_utterance_peak_normalized, Some(0.0));
        assert!(prosody.frames.iter().all(|frame| frame.f0_hz.is_none()));
        assert!(prosody.frames.iter().all(|frame| frame.periodicity == 0.0));
    }

    #[cfg(feature = "voice-stt")]
    #[test]
    fn prosody_frames_preserve_absolute_source_offset_and_segment() {
        let mut perception = AudioPerception::new(AudioPerceptionConfig::default()).unwrap();
        let samples = sine_wave(16_000, 200.0, 0.5);
        let source_start = 48_000_u64;
        let segment_index = 7_u64;
        let source_end = source_start + samples.len() as u64;

        let result = perception
            .process_samples_at_source_position(&samples, 16_000, source_start, segment_index)
            .unwrap();
        let prosody = result.prosody.expect("prosody is enabled by default");
        assert_eq!(prosody.source_start_sample, source_start);
        assert_eq!(prosody.source_end_sample, source_end);
        assert_eq!(prosody.stream_segment_index, segment_index);
        assert!(!prosody.frames.is_empty());
        assert!(prosody.frames.iter().all(|frame| {
            frame.stream_segment_index == segment_index
                && frame.source_start_sample >= source_start
                && frame.source_end_sample <= source_end
                && frame.source_end_sample > frame.source_start_sample
        }));
    }

    #[cfg(feature = "voice-stt")]
    #[test]
    fn non_finite_pcm_and_source_index_overflow_are_rejected() {
        let mut perception = AudioPerception::new(AudioPerceptionConfig::default()).unwrap();
        assert!(
            perception
                .process_samples_at_source_position(&[0.0, f32::NAN, 0.0], 16_000, 0, 0,)
                .is_err()
        );
        assert!(
            perception
                .process_samples_at_source_position(&[0.0], 16_000, u64::MAX, 0)
                .is_err()
        );
    }

    #[cfg(feature = "voice-stt")]
    #[test]
    fn rate_aware_pcm_entry_point_rejects_unresampled_audio() {
        let mut perception = AudioPerception::new(AudioPerceptionConfig::default()).unwrap();
        assert!(
            perception
                .process_samples_at_rate(&[0.0; 160], 48_000)
                .is_err()
        );
    }

    #[test]
    fn prosody_features_clone_preserves_optional_measurements_and_frames() {
        let prosody = ProsodyFeatures {
            schema_version: ACOUSTIC_PROSODY_SCHEMA_VERSION,
            extractor_id: "test-extractor",
            source_sample_rate: 16_000,
            source_start_sample: 10_000,
            source_end_sample: 10_400,
            stream_segment_index: 2,
            extractor_config: ProsodyExtractorConfigV1 {
                analysis_sample_rate_hz: 10_000,
                window_seconds: 0.025,
                frame_rate_hz: 200.0,
                lpc_order: 16,
                preemphasis: 0.97,
                f0_min_hz: 60.0,
                f0_max_hz: 400.0,
                voicing_threshold: 0.3,
                silence_relative_energy: 0.02,
            },
            frame_rate_hz: 200.0,
            energy_normalization: ProsodyEnergyNormalization::UtterancePeak,
            frames: vec![AcousticProsodyFrameV1 {
                stream_segment_index: 2,
                frame_index: 0,
                status: AcousticProsodyFrameStatusV1::Voiced,
                source_start_sample: 10_000,
                source_end_sample: 10_400,
                time_seconds: 0.0,
                f0_hz: Some(200.0),
                periodicity: 0.9,
                energy_utterance_peak_normalized: 0.7,
            }],
            avg_pitch_hz: Some(200.0),
            pitch_variance_hz2: None,
            avg_energy_utterance_peak_normalized: Some(0.7),
            speaking_rate_phonemes_per_second: None,
        };
        let cloned = prosody.clone();
        assert_eq!(prosody, cloned);
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

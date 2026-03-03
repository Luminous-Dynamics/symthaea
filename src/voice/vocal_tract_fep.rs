//! FEP active inference agent for vocal tract control — re-exported from `symthaea-vocal-tract`.
//!
//! Core types (`VocalTractFepAgent`, `VocalTractPipeline`, `ProsodyContext`, etc.) come
//! from the `symthaea-vocal-tract` sub-crate. This module adds:
//!
//! - `VocalTractObservation` ↔ `VoiceOutputMetrics` conversion (`From` impl)
//! - `StreamingVocalTract` (wraps pipeline + `FormantVocoder` from main crate)

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS from sub-crate
// ═══════════════════════════════════════════════════════════════════════════════

pub use symthaea_vocal_tract::fep::{
    VocalAction, VocalTractFepAgent, VocalTractFepResult, VocalTractObservation,
};

#[cfg(feature = "vocal-tract")]
pub use symthaea_vocal_tract::pipeline::{
    predict_duration, Intonation, PitchAccent, ProsodyContext, VocalTractPipeline,
};

// ═══════════════════════════════════════════════════════════════════════════════
// VOICEOUTPUTMETRICS → VOCALTRACTOBSERVATION CONVERSION
// ═══════════════════════════════════════════════════════════════════════════════

use crate::voice::voice_feedback::VoiceOutputMetrics;

impl From<&VoiceOutputMetrics> for VocalTractObservation {
    fn from(m: &VoiceOutputMetrics) -> Self {
        Self {
            articulation_score: m.articulation_score as f64,
            formant_accuracy: m.formant_accuracy as f64,
            pitch_stability: m.pitch_stability as f64,
            coarticulation_smoothness: m.coarticulation_smoothness as f64,
            duration_accuracy: m.duration_accuracy as f64,
            energy_consistency: m.energy_consistency as f64,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STREAMING VOCAL TRACT (stays in main crate — depends on FormantVocoder)
// ═══════════════════════════════════════════════════════════════════════════════

/// Real-time streaming vocal tract: pipeline + per-frame vocoder.
///
/// Wraps `VocalTractPipeline` and `FormantVocoder` for interactive use.
/// Each `tick()` call produces a small audio chunk (e.g., 120 samples at 24kHz/200Hz).
///
/// When the `neural-vocoder` feature is enabled and a BigVGAN ONNX model is available,
/// formant frames are converted to mel spectrograms and fed to a background ONNX thread
/// for ultra-realistic waveform generation. The DSP vocoder serves as immediate fallback
/// during neural vocoder startup latency and whenever the neural path is unavailable.
#[cfg(feature = "vocal-tract")]
pub struct StreamingVocalTract {
    /// The vocal tract pipeline (encoder → controller → FEP).
    pub pipeline: VocalTractPipeline,
    /// The formant vocoder (formant frames → audio samples).
    pub vocoder: super::vocoder::FormantVocoder,
    /// Audio samples per motor frame (sample_rate / frame_rate).
    samples_per_frame: usize,

    // ── Neural vocoder fields (feature-gated) ───────────────────────
    /// Background neural vocoder channel (None = DSP-only mode).
    #[cfg(feature = "neural-vocoder")]
    neural_channel: Option<super::neural_vocoder::NeuralVocoderChannel>,
    /// Formant-to-mel converter.
    #[cfg(feature = "neural-vocoder")]
    mel_converter: Option<symthaea_vocal_tract::formant_to_mel::FormantToMelConverter>,
    /// Accumulated mel frames waiting to be submitted as a chunk.
    #[cfg(feature = "neural-vocoder")]
    mel_buffer: Vec<Vec<f32>>,
    /// How many mel frames to buffer before submitting to neural vocoder.
    #[cfg(feature = "neural-vocoder")]
    mel_buffer_target: usize,
    /// Pending neural audio response receiver.
    #[cfg(feature = "neural-vocoder")]
    pending_neural_audio: Option<std::sync::mpsc::Receiver<super::neural_vocoder::VocoderResponse>>,
    /// Cache of neural audio samples ready to be returned.
    #[cfg(feature = "neural-vocoder")]
    neural_audio_cache: std::collections::VecDeque<f32>,
}

#[cfg(feature = "vocal-tract")]
impl StreamingVocalTract {
    /// Create a new streaming vocal tract (DSP-only mode).
    ///
    /// - `genesis`: seed for deterministic initialization
    /// - `sample_rate`: audio sample rate (e.g., 24000)
    /// - `frame_rate`: motor frame rate in Hz (typically 200)
    pub fn new(
        genesis: &symthaea_core::genesis::GenesisSeed,
        sample_rate: u32,
        frame_rate: u32,
    ) -> Self {
        let vocoder_config = super::vocoder::VocoderConfig {
            sample_rate,
            ..Default::default()
        };
        let mut pipeline = VocalTractPipeline::new(genesis);
        populate_manner_map(&mut pipeline);
        Self {
            pipeline,
            vocoder: super::vocoder::FormantVocoder::with_config(vocoder_config),
            samples_per_frame: (sample_rate / frame_rate) as usize,
            #[cfg(feature = "neural-vocoder")]
            neural_channel: None,
            #[cfg(feature = "neural-vocoder")]
            mel_converter: None,
            #[cfg(feature = "neural-vocoder")]
            mel_buffer: Vec::new(),
            #[cfg(feature = "neural-vocoder")]
            mel_buffer_target: 32,
            #[cfg(feature = "neural-vocoder")]
            pending_neural_audio: None,
            #[cfg(feature = "neural-vocoder")]
            neural_audio_cache: std::collections::VecDeque::new(),
        }
    }

    /// Create a streaming vocal tract with neural vocoder support.
    ///
    /// Attempts to load the BigVGAN ONNX model on a background thread.
    /// If the model is unavailable, falls back to DSP-only mode transparently.
    #[cfg(feature = "neural-vocoder")]
    pub fn with_neural_vocoder(
        genesis: &symthaea_core::genesis::GenesisSeed,
        sample_rate: u32,
        frame_rate: u32,
        neural_config: super::neural_vocoder::NeuralVocoderConfig,
    ) -> Self {
        let mel_buffer_target = neural_config.mel_buffer_size;
        let n_mels = neural_config.n_mels;

        let mel_config = symthaea_vocal_tract::formant_to_mel::FormantToMelConfig {
            sample_rate,
            n_mels,
            motor_frame_rate: frame_rate,
            ..Default::default()
        };

        let channel = super::neural_vocoder::NeuralVocoderChannel::spawn(neural_config);
        let has_neural = channel.is_some();

        let mut svt = Self::new(genesis, sample_rate, frame_rate);
        svt.neural_channel = channel;
        svt.mel_converter = if has_neural {
            Some(symthaea_vocal_tract::formant_to_mel::FormantToMelConverter::new(mel_config))
        } else {
            None
        };
        svt.mel_buffer_target = mel_buffer_target;
        svt
    }

    /// Whether the neural vocoder is active (model loaded, thread running).
    #[cfg(feature = "neural-vocoder")]
    pub fn has_neural_vocoder(&self) -> bool {
        self.neural_channel.is_some()
    }

    /// Run one motor frame and produce audio samples.
    ///
    /// Returns a chunk of audio samples (length = sample_rate / frame_rate).
    /// When neural vocoder is active, formant frames are converted to mel and
    /// submitted for ONNX inference; DSP gap-fills until neural audio is ready.
    pub fn tick(
        &mut self,
        cognitive_state: &super::vocal_tract_encoder::VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
    ) -> Vec<f32> {
        let frame = self
            .pipeline
            .tick_phoneme(cognitive_state, metrics, dt, phoneme);

        #[cfg(feature = "neural-vocoder")]
        if self.neural_channel.is_some() {
            return self.tick_neural(&frame, cognitive_state);
        }

        self.vocoder
            .synthesize_frame(&frame, self.samples_per_frame)
    }

    /// Run one motor frame with prosody and produce audio samples.
    pub fn tick_with_prosody(
        &mut self,
        cognitive_state: &super::vocal_tract_encoder::VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
        prosody: &ProsodyContext,
    ) -> Vec<f32> {
        let frame = self
            .pipeline
            .tick_with_prosody(cognitive_state, metrics, dt, phoneme, prosody);

        #[cfg(feature = "neural-vocoder")]
        if self.neural_channel.is_some() {
            return self.tick_neural(&frame, cognitive_state);
        }

        self.vocoder
            .synthesize_frame(&frame, self.samples_per_frame)
    }

    /// Neural vocoder tick: mel conversion → buffer → submit → collect → gap-fill.
    #[cfg(feature = "neural-vocoder")]
    fn tick_neural(
        &mut self,
        frame: &symthaea_vocal_tract::types::FormantFrame,
        cognitive_state: &super::vocal_tract_encoder::VoiceCognitiveState,
    ) -> Vec<f32> {
        use symthaea_vocal_tract::formant_to_mel::MelVoiceQuality;

        // 1. Convert cognitive state to voice quality for breathiness modulation
        let vq = MelVoiceQuality {
            rd: 1.0 + cognitive_state.emotional_valence,
            arousal: cognitive_state.emotional_arousal,
        };

        // 2. Push formant frame through mel converter
        if let Some(ref mut converter) = self.mel_converter {
            let mel_frames = converter.push_frame(frame, &vq);
            self.mel_buffer.extend(mel_frames);
        }

        // 3. Submit mel buffer when full
        if self.mel_buffer.len() >= self.mel_buffer_target {
            if let Some(ref channel) = self.neural_channel {
                let chunk: Vec<Vec<f32>> = self.mel_buffer.drain(..).collect();
                match channel.submit(chunk) {
                    Ok(rx) => {
                        self.pending_neural_audio = Some(rx);
                    }
                    Err(_) => {
                        // Channel full (backpressure) — drop this chunk, DSP will gap-fill
                        tracing::debug!("Neural vocoder backpressure, DSP gap-fill");
                    }
                }
            }
        }

        // 4. Try to collect completed neural audio
        if let Some(ref rx) = self.pending_neural_audio {
            if let Ok(response) = rx.try_recv() {
                self.neural_audio_cache.extend(response.audio.iter());
                self.pending_neural_audio = None;
            }
        }

        // 5. Return neural audio if available, else DSP gap-fill
        if self.neural_audio_cache.len() >= self.samples_per_frame {
            let samples: Vec<f32> = self
                .neural_audio_cache
                .drain(..self.samples_per_frame)
                .collect();
            samples
        } else {
            // DSP gap-fill
            self.vocoder
                .synthesize_frame(frame, self.samples_per_frame)
        }
    }

    /// Get the number of audio samples per motor frame.
    pub fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }

    /// Reset the entire streaming system.
    pub fn reset(&mut self) {
        self.pipeline.reset();
        self.vocoder.reset();
        #[cfg(feature = "neural-vocoder")]
        {
            if let Some(ref mut converter) = self.mel_converter {
                converter.reset();
            }
            self.mel_buffer.clear();
            self.pending_neural_audio = None;
            self.neural_audio_cache.clear();
        }
    }
}

/// Populate a pipeline's manner and voicing maps from the ARPABET formant database.
///
/// This wires phoneme names to their manner of articulation so `tick_phoneme()`
/// can set `source_type` on output frames for proper vocoder excitation, and
/// also populates the voicing map for manner-aware energy/voicing overrides.
#[cfg(feature = "vocal-tract")]
pub fn populate_manner_map(pipeline: &mut VocalTractPipeline) {
    let db = super::formant_targets::get_formant_database();
    let manner_map: std::collections::HashMap<String, _> = db
        .iter()
        .map(|(name, target)| (name.clone(), target.manner))
        .collect();
    let voicing_map: std::collections::HashMap<String, bool> = db
        .iter()
        .map(|(name, target)| (name.clone(), target.is_voiced))
        .collect();
    pipeline.set_manner_map(manner_map);
    pipeline.set_voicing_map(voicing_map);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_metrics_to_observation_conversion() {
        let metrics = VoiceOutputMetrics {
            articulation_score: 0.8,
            formant_accuracy: 0.7,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.85,
            duration_accuracy: 0.75,
            energy_consistency: 0.6,
            ..Default::default()
        };

        let obs: VocalTractObservation = (&metrics).into();
        assert!((obs.articulation_score - 0.8).abs() < 1e-6);
        assert!((obs.formant_accuracy - 0.7).abs() < 1e-6);
        assert!((obs.pitch_stability - 0.9).abs() < 1e-6);
        assert!((obs.coarticulation_smoothness - 0.85).abs() < 1e-6);
        assert!((obs.duration_accuracy - 0.75).abs() < 1e-6);
        assert!((obs.energy_consistency - 0.6).abs() < 1e-6);
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_streaming_frame_sample_count() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-streaming");
        let mut streaming = StreamingVocalTract::new(&genesis, 24000, 200);
        let state = VoiceCognitiveState::default();

        let samples = streaming.tick(&state, None, 0.005, None);

        // 24000 / 200 = 120 samples per frame
        assert_eq!(
            samples.len(),
            120,
            "Should produce exactly samples_per_frame samples"
        );
        assert_eq!(streaming.samples_per_frame(), 120);
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_streaming_100_frames_no_clicks() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-streaming-smooth");
        let mut streaming = StreamingVocalTract::new(&genesis, 24000, 200);
        let state = VoiceCognitiveState::default();

        let mut all_samples = Vec::new();
        for _ in 0..100 {
            let chunk = streaming.tick(&state, None, 0.005, Some("AH"));
            all_samples.extend_from_slice(&chunk);
        }

        // 100 frames × 120 samples = 12000 samples
        assert_eq!(all_samples.len(), 12000);

        // No NaN/Inf
        assert!(
            all_samples.iter().all(|s| s.is_finite()),
            "All samples should be finite"
        );

        // Check for clicks: max sample-to-sample delta should be reasonable.
        let max_delta: f32 = all_samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta < 0.8,
            "Max sample delta too high (click detected): {max_delta:.3}"
        );
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_streaming_end_to_end() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-streaming-e2e");
        let mut streaming = StreamingVocalTract::new(&genesis, 24000, 200);
        let state = VoiceCognitiveState {
            emotional_arousal: 0.7,
            emotional_valence: 0.3,
            ..Default::default()
        };

        // Simulate a short utterance: 10 frames of /AH/ then 10 of /IY/
        let mut audio = Vec::new();
        for _ in 0..10 {
            audio.extend(streaming.tick(&state, None, 0.005, Some("AH")));
        }
        for _ in 0..10 {
            audio.extend(streaming.tick(&state, None, 0.005, Some("IY")));
        }

        assert_eq!(audio.len(), 2400); // 20 frames × 120 samples
                                       // Should have non-trivial audio content
        let rms: f32 = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        assert!(
            rms > 1e-6,
            "Streaming output should have non-trivial content: rms={rms}"
        );
    }
}

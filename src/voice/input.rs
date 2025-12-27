//! Voice Input Module - Speech-to-Text using whisper-rs
//!
//! Provides real-time speech recognition using whisper.cpp bindings.
//! Designed for low-latency conversational use.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{VoiceError, VoiceResult};
use super::models::{ModelManager, WhisperModel};

/// Configuration for voice input
#[derive(Debug, Clone)]
pub struct VoiceInputConfig {
    /// Whisper model to use
    pub model: WhisperModel,
    /// Language (None for auto-detect)
    pub language: Option<String>,
    /// Sample rate for audio (default: 16000)
    pub sample_rate: u32,
    /// Channels (default: 1 for mono)
    pub channels: u16,
    /// Silence threshold for VAD
    pub silence_threshold: f32,
    /// Minimum speech duration in ms
    pub min_speech_ms: u32,
    /// Maximum audio buffer in seconds
    pub max_buffer_sec: f32,
}

impl Default for VoiceInputConfig {
    fn default() -> Self {
        Self {
            model: WhisperModel::Base,
            language: Some("en".to_string()),
            sample_rate: 16000,
            channels: 1,
            silence_threshold: 0.02,
            min_speech_ms: 500,
            max_buffer_sec: 30.0,
        }
    }
}

/// Result of a transcription
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// The transcribed text
    pub text: String,
    /// Detected language (if auto-detect enabled)
    pub language: Option<String>,
    /// Audio duration in milliseconds
    pub duration_ms: u64,
    /// Transcription confidence (0.0 - 1.0)
    pub confidence: f32,
}

/// Voice input handler for speech recognition
pub struct VoiceInput {
    config: VoiceInputConfig,
    #[cfg(feature = "voice-stt")]
    ctx: whisper_rs::WhisperContext,
    #[cfg(feature = "voice-stt")]
    state: whisper_rs::WhisperState<'static>,
    is_listening: Arc<AtomicBool>,
}

impl VoiceInput {
    /// Create a new voice input handler
    #[cfg(feature = "voice-stt")]
    pub fn new(config: VoiceInputConfig) -> VoiceResult<Self> {
        let manager = ModelManager::new()?;
        let model_path = manager.whisper_path(config.model)?;
        Self::from_model_path(&model_path, config)
    }

    /// Create from a specific model path
    #[cfg(feature = "voice-stt")]
    pub fn from_model_path(path: &Path, config: VoiceInputConfig) -> VoiceResult<Self> {
        use whisper_rs::{WhisperContext, WhisperContextParameters};

        let ctx = WhisperContext::new_with_params(
            path.to_str().ok_or_else(|| VoiceError::ModelLoad("Invalid path".into()))?,
            WhisperContextParameters::default(),
        ).map_err(|e| VoiceError::ModelLoad(format!("Failed to load Whisper: {}", e)))?;

        // Create state from context - requires unsafe for 'static lifetime
        let state = ctx.create_state()
            .map_err(|e| VoiceError::ModelLoad(format!("Failed to create state: {}", e)))?;

        Ok(Self {
            config,
            ctx,
            state: unsafe { std::mem::transmute(state) },
            is_listening: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Transcribe audio data (16kHz, mono, f32 samples in [-1, 1])
    #[cfg(feature = "voice-stt")]
    pub fn transcribe(&mut self, audio: &[f32]) -> VoiceResult<TranscriptionResult> {
        use whisper_rs::{FullParams, SamplingStrategy};

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure language
        if let Some(ref lang) = self.config.language {
            params.set_language(Some(lang));
        } else {
            params.set_detect_language(true);
        }

        // Disable timestamps for faster processing
        params.set_print_timestamps(false);
        params.set_print_realtime(false);
        params.set_print_progress(false);

        // Performance tuning
        params.set_n_threads(4);
        params.set_single_segment(true);

        // Run transcription
        self.state.full(params, audio)
            .map_err(|e| VoiceError::Transcription(format!("Whisper error: {}", e)))?;

        // Collect results
        let num_segments = self.state.full_n_segments()
            .map_err(|e| VoiceError::Transcription(format!("Segment error: {}", e)))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = self.state.full_get_segment_text(i) {
                text.push_str(&segment);
                text.push(' ');
            }
        }

        let duration_ms = (audio.len() as f64 / self.config.sample_rate as f64 * 1000.0) as u64;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            language: self.config.language.clone(),
            duration_ms,
            confidence: 0.9,  // Whisper doesn't provide per-segment confidence easily
        })
    }

    /// Start listening on the default input device
    #[cfg(all(feature = "voice-stt", feature = "cpal"))]
    pub fn start_listening(&self) -> VoiceResult<AudioStream> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| VoiceError::AudioDevice("No input device found".into()))?;

        let config = cpal::StreamConfig {
            channels: self.config.channels,
            sample_rate: cpal::SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let samples = Arc::new(std::sync::Mutex::new(Vec::new()));
        let samples_clone = Arc::clone(&samples);
        let is_listening = Arc::clone(&self.is_listening);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if is_listening.load(Ordering::SeqCst) {
                    if let Ok(mut guard) = samples_clone.lock() {
                        guard.extend_from_slice(data);
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        ).map_err(|e| VoiceError::AudioDevice(format!("Stream error: {}", e)))?;

        stream.play()
            .map_err(|e| VoiceError::AudioDevice(format!("Play error: {}", e)))?;

        self.is_listening.store(true, Ordering::SeqCst);

        Ok(AudioStream {
            _stream: stream,
            samples,
            is_listening: Arc::clone(&self.is_listening),
        })
    }

    /// Check if currently listening
    pub fn is_listening(&self) -> bool {
        self.is_listening.load(Ordering::SeqCst)
    }

    /// Stop listening
    pub fn stop_listening(&self) {
        self.is_listening.store(false, Ordering::SeqCst);
    }
}

/// Handle to an active audio stream
#[cfg(all(feature = "voice-stt", feature = "cpal"))]
pub struct AudioStream {
    _stream: cpal::Stream,
    samples: Arc<std::sync::Mutex<Vec<f32>>>,
    is_listening: Arc<AtomicBool>,
}

#[cfg(all(feature = "voice-stt", feature = "cpal"))]
impl AudioStream {
    /// Get collected samples and clear buffer
    pub fn take_samples(&self) -> Vec<f32> {
        if let Ok(mut guard) = self.samples.lock() {
            std::mem::take(&mut *guard)
        } else {
            Vec::new()
        }
    }

    /// Stop the stream
    pub fn stop(&self) {
        self.is_listening.store(false, Ordering::SeqCst);
    }
}

/// Load audio from a WAV file
#[cfg(feature = "voice-stt")]
pub fn load_wav(path: &Path) -> VoiceResult<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let samples = if spec.channels == 2 {
        samples.chunks(2)
            .map(|pair| (pair[0] + pair.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Simple voice activity detection
pub fn detect_speech(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }

    let energy: f32 = samples.iter().map(|s| s.abs()).sum::<f32>() / samples.len() as f32;
    energy > threshold
}

/// Find speech segments in audio
pub fn find_speech_segments(
    samples: &[f32],
    sample_rate: u32,
    threshold: f32,
    min_duration_ms: u32,
) -> Vec<(usize, usize)> {
    let window_samples = (sample_rate as f32 * 0.025) as usize;  // 25ms windows
    let min_samples = (sample_rate as f32 * min_duration_ms as f32 / 1000.0) as usize;

    let mut segments = Vec::new();
    let mut start: Option<usize> = None;

    for (i, chunk) in samples.chunks(window_samples).enumerate() {
        let is_speech = detect_speech(chunk, threshold);
        let pos = i * window_samples;

        match (is_speech, start) {
            (true, None) => start = Some(pos),
            (false, Some(s)) if pos - s >= min_samples => {
                segments.push((s, pos));
                start = None;
            }
            (false, Some(_)) => start = None,
            _ => {}
        }
    }

    // Handle trailing speech
    if let Some(s) = start {
        if samples.len() - s >= min_samples {
            segments.push((s, samples.len()));
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_input_config_default() {
        let config = VoiceInputConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.model, WhisperModel::Base);
    }

    #[test]
    fn test_transcription_result() {
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            language: Some("en".to_string()),
            duration_ms: 1500,
            confidence: 0.95,
        };
        assert_eq!(result.text, "Hello world");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_detect_speech_silence() {
        let silence = vec![0.001, 0.002, -0.001, 0.0];
        assert!(!detect_speech(&silence, 0.02));
    }

    #[test]
    fn test_detect_speech_active() {
        let speech = vec![0.5, -0.3, 0.6, -0.4, 0.5];
        assert!(detect_speech(&speech, 0.02));
    }

    #[test]
    fn test_find_speech_segments_empty() {
        let silence = vec![0.001; 16000];  // 1 second of silence
        let segments = find_speech_segments(&silence, 16000, 0.02, 500);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_find_speech_segments_with_speech() {
        let mut audio = vec![0.001; 8000];  // 0.5s silence
        audio.extend(vec![0.5; 16000]);     // 1s speech
        audio.extend(vec![0.001; 8000]);    // 0.5s silence

        let segments = find_speech_segments(&audio, 16000, 0.02, 500);
        assert!(!segments.is_empty());
    }
}

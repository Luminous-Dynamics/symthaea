//! Voice Output Module - Text-to-Speech using Kokoro TTS
//!
//! Provides natural speech synthesis using Kokoro-82M via ONNX runtime.
//! Supports LTC-aware speech pacing for natural conversation flow.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::io::Read;

use super::{VoiceError, VoiceResult, LTCPacing};
use super::models::{ModelManager, KokoroModel};

/// Configuration for voice output
#[derive(Debug, Clone)]
pub struct VoiceOutputConfig {
    /// Kokoro model to use
    pub model: KokoroModel,
    /// Output sample rate (default: 24000)
    pub sample_rate: u32,
    /// Base speech rate multiplier
    pub base_rate: f32,
    /// Enable LTC-aware pacing
    pub ltc_pacing: bool,
    /// Voice file name (e.g., "af_bella" - looks for af_bella.bin in voices dir)
    pub voice_name: String,
}

impl Default for VoiceOutputConfig {
    fn default() -> Self {
        Self {
            model: KokoroModel::V019,
            sample_rate: 24000,
            base_rate: 1.0,
            ltc_pacing: true,
            voice_name: "af_bella".to_string(),  // Default voice
        }
    }
}

/// Result of speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Audio samples (f32 in [-1, 1])
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

impl SynthesisResult {
    /// Save to WAV file
    #[cfg(feature = "voice-tts")]
    pub fn save_wav(&self, path: &Path) -> VoiceResult<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        for sample in &self.samples {
            writer.write_sample(*sample)
                .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        }

        writer.finalize()
            .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(())
    }
}

/// Voice output handler for speech synthesis
#[cfg(feature = "voice-tts")]
pub struct VoiceOutput {
    config: VoiceOutputConfig,
    session: ort::session::Session,
    /// Cached style vector (256 dimensions)
    style_vector: Vec<f32>,
}

#[cfg(not(feature = "voice-tts"))]
pub struct VoiceOutput {
    config: VoiceOutputConfig,
}

/// Execution provider used for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CUDA GPU acceleration
    Cuda,
    /// TensorRT (NVIDIA optimized)
    TensorRT,
    /// CPU with multi-threading
    Cpu,
}

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionProvider::Cuda => write!(f, "CUDA"),
            ExecutionProvider::TensorRT => write!(f, "TensorRT"),
            ExecutionProvider::Cpu => write!(f, "CPU"),
        }
    }
}

#[cfg(feature = "voice-tts")]
impl VoiceOutput {
    /// Create a new voice output handler
    pub fn new(config: VoiceOutputConfig) -> VoiceResult<Self> {
        let manager = ModelManager::new()?;
        let model_path = manager.kokoro_path(config.model)?;

        // Voice files are in the voices subdirectory next to the model
        let voices_dir = model_path.parent()
            .ok_or_else(|| VoiceError::ModelLoad("Invalid model path".into()))?
            .join("voices");
        let voice_path = voices_dir.join(format!("{}.bin", config.voice_name));

        Self::from_paths(&model_path, &voice_path, config)
    }

    /// Create from specific model and voice file paths
    pub fn from_paths(model_path: &Path, voice_path: &Path, config: VoiceOutputConfig) -> VoiceResult<Self> {
        Self::from_paths_with_provider(model_path, voice_path, config, None)
    }

    /// Create with explicit execution provider selection
    /// If provider is None, auto-detects best available (CUDA > TensorRT > CPU)
    pub fn from_paths_with_provider(
        model_path: &Path,
        voice_path: &Path,
        config: VoiceOutputConfig,
        preferred_provider: Option<ExecutionProvider>,
    ) -> VoiceResult<Self> {
        use ort::session::builder::GraphOptimizationLevel;

        // Try to use GPU if available, fall back to CPU
        let (session, provider) = Self::create_session_with_best_provider(
            model_path,
            preferred_provider
        )?;

        tracing::info!("Loaded Kokoro TTS model using {} execution provider", provider);

        // Load style vector from voice file (256 floats = 1024 bytes)
        let style_vector = Self::load_style_vector(voice_path)?;

        Ok(Self { config, session, style_vector })
    }

    /// Create session with best available execution provider
    fn create_session_with_best_provider(
        model_path: &Path,
        preferred: Option<ExecutionProvider>,
    ) -> VoiceResult<(ort::session::Session, ExecutionProvider)> {
        use ort::session::builder::GraphOptimizationLevel;

        // Determine provider order based on preference
        let providers = match preferred {
            Some(p) => vec![p],
            None => {
                // Auto-detect: try CUDA first, then TensorRT, then CPU
                #[cfg(feature = "voice-tts-cuda")]
                {
                    vec![ExecutionProvider::Cuda, ExecutionProvider::TensorRT, ExecutionProvider::Cpu]
                }
                #[cfg(not(feature = "voice-tts-cuda"))]
                {
                    vec![ExecutionProvider::Cpu]
                }
            }
        };

        for provider in &providers {
            match Self::try_create_session(model_path, *provider) {
                Ok(session) => return Ok((session, *provider)),
                Err(e) => {
                    tracing::warn!("Failed to create session with {}: {}", provider, e);
                    continue;
                }
            }
        }

        // All providers failed, give detailed error
        Err(VoiceError::ModelLoad(
            "Failed to create ONNX session with any execution provider".into()
        ))
    }

    /// Try to create a session with a specific execution provider
    fn try_create_session(
        model_path: &Path,
        provider: ExecutionProvider,
    ) -> VoiceResult<ort::session::Session> {
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = ort::session::Session::builder()
            .map_err(|e| VoiceError::ModelLoad(format!("ONNX builder error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VoiceError::ModelLoad(format!("Optimization error: {}", e)))?;

        // Configure based on provider
        match provider {
            ExecutionProvider::Cuda => {
                #[cfg(feature = "voice-tts-cuda")]
                {
                    // CUDA execution provider - use GPU device 0
                    builder = builder
                        .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])
                        .map_err(|e| VoiceError::ModelLoad(format!("CUDA provider error: {}", e)))?;
                    tracing::info!("Initializing CUDA execution provider for GPU acceleration");
                }
                #[cfg(not(feature = "voice-tts-cuda"))]
                {
                    return Err(VoiceError::ModelLoad("CUDA feature not enabled".into()));
                }
            }
            ExecutionProvider::TensorRT => {
                #[cfg(feature = "voice-tts-cuda")]
                {
                    // TensorRT for NVIDIA-optimized inference
                    builder = builder
                        .with_execution_providers([ort::execution_providers::TensorRTExecutionProvider::default().build()])
                        .map_err(|e| VoiceError::ModelLoad(format!("TensorRT provider error: {}", e)))?;
                    tracing::info!("Initializing TensorRT execution provider for optimized GPU inference");
                }
                #[cfg(not(feature = "voice-tts-cuda"))]
                {
                    return Err(VoiceError::ModelLoad("TensorRT requires CUDA feature".into()));
                }
            }
            ExecutionProvider::Cpu => {
                // CPU with multi-threading
                builder = builder
                    .with_intra_threads(12)
                    .map_err(|e| VoiceError::ModelLoad(format!("Thread config error: {}", e)))?;
                tracing::info!("Using CPU execution provider with 12 threads");
            }
        }

        builder
            .commit_from_file(model_path)
            .map_err(|e| VoiceError::ModelLoad(format!("Failed to load model: {}", e)))
    }

    /// Create from a specific model path (uses default voice)
    pub fn from_model_path(path: &Path, config: VoiceOutputConfig) -> VoiceResult<Self> {
        let voices_dir = path.parent()
            .ok_or_else(|| VoiceError::ModelLoad("Invalid model path".into()))?
            .join("voices");
        let voice_path = voices_dir.join(format!("{}.bin", config.voice_name));
        Self::from_paths(path, &voice_path, config)
    }

    /// Load style vector from a voice binary file
    /// Voice files contain 256-dimensional f32 style vectors
    fn load_style_vector(voice_path: &Path) -> VoiceResult<Vec<f32>> {
        let mut file = std::fs::File::open(voice_path)
            .map_err(|e| VoiceError::ModelLoad(format!("Failed to open voice file {:?}: {}", voice_path, e)))?;

        // Read first 256 floats (1024 bytes) for the style vector
        let mut buffer = vec![0u8; 256 * 4];
        file.read_exact(&mut buffer)
            .map_err(|e| VoiceError::ModelLoad(format!("Failed to read voice file: {}", e)))?;

        // Convert bytes to f32 (little-endian)
        let style_vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        tracing::info!("Loaded style vector with {} dimensions from {:?}", style_vector.len(), voice_path);

        Ok(style_vector)
    }

    /// Synthesize speech from text
    pub fn synthesize(&mut self, text: &str) -> VoiceResult<SynthesisResult> {
        self.synthesize_with_pacing(text, LTCPacing::default())
    }

    /// Synthesize speech with LTC-aware pacing
    pub fn synthesize_with_pacing(&mut self, text: &str, pacing: LTCPacing) -> VoiceResult<SynthesisResult> {
        use ort::value::Value;

        // Prepare text tokens using Misaki tokenizer via socket
        let tokens = self.text_to_tokens(text);
        let token_len = tokens.len();

        // Speed adjustment based on LTC pacing
        let rate = self.config.base_rate * pacing.speech_rate;
        let speed = vec![rate];

        // Style vector is already loaded (256 dimensions)
        let style = self.style_vector.clone();
        let style_len = style.len();

        // Convert to ort Values using (shape, data) tuple format
        // Kokoro expects: input_ids (1, seq_len), style (1, 256), speed (1,)
        let input_ids_value = Value::from_array(([1usize, token_len], tokens))
            .map_err(|e| VoiceError::Synthesis(format!("input_ids tensor error: {}", e)))?;
        let style_value = Value::from_array(([1usize, style_len], style))
            .map_err(|e| VoiceError::Synthesis(format!("style tensor error: {}", e)))?;
        let speed_value = Value::from_array(([1usize], speed))
            .map_err(|e| VoiceError::Synthesis(format!("speed tensor error: {}", e)))?;

        // Build inputs using ort 2.0 API with correct Kokoro input names
        let inputs = ort::inputs![
            "input_ids" => input_ids_value,
            "style" => style_value,
            "speed" => speed_value,
        ];

        // Run inference
        let outputs = self.session.run(inputs)
            .map_err(|e| VoiceError::Synthesis(format!("Inference error: {}", e)))?;

        // Extract audio - ort 2.0: iterate outputs to get (name, value_ref) tuples
        let (_, audio_ref) = outputs.iter().next()
            .ok_or_else(|| VoiceError::Synthesis("No audio output".into()))?;

        // Extract tensor - try_extract_tensor returns (&Shape, &[T]) tuple
        let (_, audio_data) = audio_ref
            .try_extract_tensor::<f32>()
            .map_err(|e| VoiceError::Synthesis(format!("Extract error: {}", e)))?;

        let samples: Vec<f32> = audio_data.to_vec();
        let duration_ms = (samples.len() as f64 / self.config.sample_rate as f64 * 1000.0) as u64;

        Ok(SynthesisResult {
            samples,
            sample_rate: self.config.sample_rate,
            duration_ms,
        })
    }

    /// Convert text to token IDs using Misaki (official Kokoro G2P)
    fn text_to_tokens(&self, text: &str) -> Vec<i64> {
        // Use Misaki tokenizer for proper phoneme conversion
        match super::tokenizer::tokenize(text) {
            Ok(tokens) => tokens,
            Err(e) => {
                tracing::error!("Tokenization failed: {}, using fallback", e);
                // Minimal fallback - just return BOS/EOS
                vec![0i64, 0]
            }
        }
    }

    /// Play audio through default output device
    #[cfg(feature = "audio")]
    pub fn play(&self, result: &SynthesisResult) -> VoiceResult<()> {
        use rodio::{OutputStream, Sink};

        let (_stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| VoiceError::AudioDevice(format!("Output device error: {}", e)))?;

        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| VoiceError::AudioDevice(format!("Sink error: {}", e)))?;

        // Create audio source from samples
        let source = SamplesSource {
            samples: result.samples.clone(),
            sample_rate: result.sample_rate,
            position: 0,
        };

        sink.append(source);
        sink.sleep_until_end();

        Ok(())
    }

    /// Play audio with LTC-aware pauses
    #[cfg(feature = "audio")]
    pub fn play_with_pacing(&self, result: &SynthesisResult, pacing: LTCPacing) -> VoiceResult<()> {
        // Play the audio
        self.play(result)?;

        // Add post-speech pause based on LTC state
        std::thread::sleep(std::time::Duration::from_millis(pacing.pause_ms as u64));

        Ok(())
    }

    /// Synthesize and play in streaming mode - lower latency by playing while generating
    /// Splits text into sentences and plays each as it's synthesized
    #[cfg(feature = "audio")]
    pub fn stream_speak(&mut self, text: &str, pacing: LTCPacing) -> VoiceResult<()> {
        use rodio::{OutputStream, Sink};

        let (_stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| VoiceError::AudioDevice(format!("Output device error: {}", e)))?;

        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| VoiceError::AudioDevice(format!("Sink error: {}", e)))?;

        // Split into sentences for streaming
        let sentences = split_sentences(text);

        for sentence in sentences {
            if sentence.trim().is_empty() {
                continue;
            }

            // Synthesize this sentence
            let result = self.synthesize_with_pacing(sentence, pacing.clone())?;

            // Create audio source and append to sink (plays asynchronously)
            let source = SamplesSource {
                samples: result.samples,
                sample_rate: result.sample_rate,
                position: 0,
            };

            sink.append(source);
        }

        // Wait for all audio to finish
        sink.sleep_until_end();

        // Add final pause
        std::thread::sleep(std::time::Duration::from_millis(pacing.pause_ms as u64));

        Ok(())
    }

    /// Synthesize multiple texts in parallel using a thread pool
    /// Returns results in the same order as inputs
    pub fn synthesize_batch(&mut self, texts: &[&str], pacing: LTCPacing) -> VoiceResult<Vec<SynthesisResult>> {
        // For now, synthesize sequentially but could be parallelized with model cloning
        texts.iter()
            .map(|text| self.synthesize_with_pacing(text, pacing.clone()))
            .collect()
    }

    /// Get synthesis latency estimate in milliseconds for given text length
    pub fn estimate_latency_ms(&self, text_chars: usize) -> u64 {
        // Empirical: ~10ms tokenization + ~50ms per 100 chars on CPU
        // GPU is ~5x faster
        let base_ms = 10u64;
        let per_100_chars = 50u64;
        base_ms + (text_chars as u64 * per_100_chars / 100)
    }
}

/// Audio source adapter for rodio
#[cfg(feature = "audio")]
struct SamplesSource {
    samples: Vec<f32>,
    sample_rate: u32,
    position: usize,
}

#[cfg(feature = "audio")]
impl Iterator for SamplesSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.samples.len() {
            let sample = self.samples[self.position];
            self.position += 1;
            Some(sample)
        } else {
            None
        }
    }
}

#[cfg(feature = "audio")]
impl rodio::Source for SamplesSource {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.samples.len() - self.position)
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        let secs = self.samples.len() as f64 / self.sample_rate as f64;
        Some(std::time::Duration::from_secs_f64(secs))
    }
}

/// Split text into sentences for natural pacing
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;

    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            let end = i + c.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end;
        }
    }

    // Handle remaining text
    let remaining = text[start..].trim();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

/// Add natural pauses to text for better TTS
pub fn add_pauses(text: &str) -> String {
    text.replace(", ", ", ... ")
        .replace("; ", "; ... ")
        .replace(": ", ": ... ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_output_config_default() {
        let config = VoiceOutputConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.base_rate, 1.0);
        assert!(config.ltc_pacing);
    }

    #[test]
    fn test_synthesis_result() {
        let result = SynthesisResult {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 24000,
            duration_ms: 125,
        };
        assert_eq!(result.samples.len(), 3);
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello! How are you? I am fine.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello!");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine.");
    }

    #[test]
    fn test_split_sentences_no_punctuation() {
        let text = "Hello world";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_add_pauses() {
        let text = "First, second; third: fourth.";
        let with_pauses = add_pauses(text);
        assert!(with_pauses.contains(", ..."));
        assert!(with_pauses.contains("; ..."));
        assert!(with_pauses.contains(": ..."));
    }

    #[test]
    fn test_ltc_pacing_affects_rate() {
        let fast_pacing = LTCPacing::from_ltc(0.8, 0.05);
        let slow_pacing = LTCPacing::from_ltc(0.2, -0.05);

        assert!(fast_pacing.speech_rate > slow_pacing.speech_rate);
        assert!(fast_pacing.pause_ms < slow_pacing.pause_ms);
    }
}

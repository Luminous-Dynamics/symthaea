// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Audio Processing Core
//!
//! High-performance audio processing library providing:
//! - Audio decoding and encoding (MP3, AAC, FLAC, Opus, WAV)
//! - Sample rate conversion and resampling
//! - Audio normalization and loudness analysis (EBU R128)
//! - Waveform generation for visualization
//! - Audio fingerprinting
//! - Real-time DSP effects

pub mod codec;
pub mod dsp;
pub mod analysis;
pub mod effects;
pub mod fingerprint;
pub mod waveform;

use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Decoding error: {0}")]
    DecodingError(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    #[error("Buffer underrun")]
    BufferUnderrun,

    #[error("Buffer overflow")]
    BufferOverflow,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

pub type AudioResult<T> = Result<T, AudioError>;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Supported audio formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    Mp3,
    Aac,
    Flac,
    Opus,
    Wav,
    Ogg,
    Webm,
}

impl AudioFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Some(Self::Mp3),
            "aac" | "m4a" => Some(Self::Aac),
            "flac" => Some(Self::Flac),
            "opus" => Some(Self::Opus),
            "wav" => Some(Self::Wav),
            "ogg" => Some(Self::Ogg),
            "webm" => Some(Self::Webm),
            _ => None,
        }
    }

    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Opus => "audio/opus",
            Self::Wav => "audio/wav",
            Self::Ogg => "audio/ogg",
            Self::Webm => "audio/webm",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Aac => "m4a",
            Self::Flac => "flac",
            Self::Opus => "opus",
            Self::Wav => "wav",
            Self::Ogg => "ogg",
            Self::Webm => "webm",
        }
    }
}

/// Audio quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Low quality (96 kbps)
    Low,
    /// Medium quality (160 kbps)
    Medium,
    /// High quality (320 kbps)
    High,
    /// Lossless quality
    Lossless,
}

impl Quality {
    pub fn bitrate(&self) -> u32 {
        match self {
            Self::Low => 96_000,
            Self::Medium => 160_000,
            Self::High => 320_000,
            Self::Lossless => 0, // Variable for lossless
        }
    }
}

/// Audio channel layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Surround51,
    Surround71,
    Ambisonic1stOrder,
    Ambisonic2ndOrder,
    Ambisonic3rdOrder,
}

impl ChannelLayout {
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Ambisonic1stOrder => 4,
            Self::Ambisonic2ndOrder => 9,
            Self::Ambisonic3rdOrder => 16,
        }
    }
}

/// Audio buffer containing interleaved samples
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved audio samples (f32, normalized -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: usize,
    /// Channel layout
    pub layout: ChannelLayout,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(sample_rate: u32, channels: usize, layout: ChannelLayout) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
            channels,
            layout,
        }
    }

    /// Create from raw samples
    pub fn from_samples(
        samples: Vec<f32>,
        sample_rate: u32,
        channels: usize,
        layout: ChannelLayout,
    ) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            layout,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    /// Get number of frames (samples per channel)
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels
    }

    /// Get a specific channel's samples
    pub fn channel(&self, index: usize) -> Vec<f32> {
        if index >= self.channels {
            return Vec::new();
        }

        self.samples
            .iter()
            .skip(index)
            .step_by(self.channels)
            .copied()
            .collect()
    }

    /// Mix down to mono
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let frame_count = self.frame_count();
        let mut mono_samples = Vec::with_capacity(frame_count);

        for frame in 0..frame_count {
            let mut sum = 0.0f32;
            for ch in 0..self.channels {
                sum += self.samples[frame * self.channels + ch];
            }
            mono_samples.push(sum / self.channels as f32);
        }

        Self {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
            layout: ChannelLayout::Mono,
        }
    }

    /// Convert to stereo (upmix or downmix)
    pub fn to_stereo(&self) -> Self {
        match self.channels {
            1 => {
                // Mono to stereo: duplicate channel
                let mut stereo = Vec::with_capacity(self.samples.len() * 2);
                for sample in &self.samples {
                    stereo.push(*sample);
                    stereo.push(*sample);
                }
                Self {
                    samples: stereo,
                    sample_rate: self.sample_rate,
                    channels: 2,
                    layout: ChannelLayout::Stereo,
                }
            }
            2 => self.clone(),
            _ => {
                // Downmix to stereo
                let frame_count = self.frame_count();
                let mut stereo = Vec::with_capacity(frame_count * 2);

                for frame in 0..frame_count {
                    let base = frame * self.channels;
                    // Simple downmix: average left channels and right channels
                    let left = (0..self.channels / 2)
                        .map(|i| self.samples[base + i])
                        .sum::<f32>()
                        / (self.channels / 2) as f32;
                    let right = (self.channels / 2..self.channels)
                        .map(|i| self.samples[base + i])
                        .sum::<f32>()
                        / (self.channels - self.channels / 2) as f32;

                    stereo.push(left);
                    stereo.push(right);
                }

                Self {
                    samples: stereo,
                    sample_rate: self.sample_rate,
                    channels: 2,
                    layout: ChannelLayout::Stereo,
                }
            }
        }
    }

    /// Append another buffer (must have same format)
    pub fn append(&mut self, other: &AudioBuffer) -> AudioResult<()> {
        if self.sample_rate != other.sample_rate || self.channels != other.channels {
            return Err(AudioError::ProcessingError(
                "Cannot append buffers with different formats".to_string(),
            ));
        }
        self.samples.extend(&other.samples);
        Ok(())
    }

    /// Split at a specific frame
    pub fn split_at(&self, frame: usize) -> (Self, Self) {
        let sample_index = frame * self.channels;
        let (left, right) = self.samples.split_at(sample_index.min(self.samples.len()));

        (
            Self {
                samples: left.to_vec(),
                sample_rate: self.sample_rate,
                channels: self.channels,
                layout: self.layout,
            },
            Self {
                samples: right.to_vec(),
                sample_rate: self.sample_rate,
                channels: self.channels,
                layout: self.layout,
            },
        )
    }

    /// Get RMS (root mean square) level
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum_squares / self.samples.len() as f32).sqrt()
    }

    /// Get peak level
    pub fn peak(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

// ============================================================================
// AUDIO METADATA
// ============================================================================

/// Audio file metadata
#[derive(Debug, Clone, Default)]
pub struct AudioMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub album_artist: Option<String>,
    pub track_number: Option<u32>,
    pub disc_number: Option<u32>,
    pub year: Option<u32>,
    pub genre: Option<String>,
    pub duration_ms: u64,
    pub bitrate: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bit_depth: Option<u8>,
    pub format: Option<AudioFormat>,
    pub encoder: Option<String>,
    pub isrc: Option<String>,
    pub cover_art: Option<Vec<u8>>,
}

// ============================================================================
// TRANSCODING
// ============================================================================

/// Transcoding job configuration
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    pub id: Uuid,
    pub source_format: AudioFormat,
    pub target_format: AudioFormat,
    pub target_quality: Quality,
    pub target_sample_rate: Option<u32>,
    pub target_channels: Option<usize>,
    pub normalize: bool,
    pub target_loudness: Option<f32>, // LUFS
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            source_format: AudioFormat::Wav,
            target_format: AudioFormat::Mp3,
            target_quality: Quality::High,
            target_sample_rate: None,
            target_channels: None,
            normalize: true,
            target_loudness: Some(-14.0), // Spotify standard
        }
    }
}

/// Transcoding result
#[derive(Debug)]
pub struct TranscodeResult {
    pub id: Uuid,
    pub output_data: Vec<u8>,
    pub output_format: AudioFormat,
    pub duration_ms: u64,
    pub output_bitrate: u32,
    pub loudness_lufs: Option<f32>,
    pub peak_db: f32,
    pub processing_time_ms: u64,
}

/// Audio transcoder with support for multiple formats
pub struct Transcoder {
    config: TranscodeConfig,
}

impl Transcoder {
    pub fn new(config: TranscodeConfig) -> Self {
        Self { config }
    }

    /// Transcode audio data
    pub async fn transcode(&self, input: &[u8]) -> AudioResult<TranscodeResult> {
        let start = std::time::Instant::now();

        // Decode input
        let decoded = self.decode(input)?;

        // Process (resample, normalize, etc.)
        let processed = self.process(decoded)?;

        // Encode to target format
        let encoded = self.encode(&processed)?;

        let processing_time = start.elapsed().as_millis() as u64;

        Ok(TranscodeResult {
            id: self.config.id,
            output_data: encoded,
            output_format: self.config.target_format,
            duration_ms: (processed.duration() * 1000.0) as u64,
            output_bitrate: self.config.target_quality.bitrate(),
            loudness_lufs: None, // Would be calculated during processing
            peak_db: 20.0 * processed.peak().log10(),
            processing_time_ms: processing_time,
        })
    }

    fn decode(&self, input: &[u8]) -> AudioResult<AudioBuffer> {
        // Use symphonia for decoding
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::probe::Hint;

        let cursor = std::io::Cursor::new(input.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let mut hint = Hint::new();
        hint.with_extension(self.config.source_format.extension());

        let format_opts = Default::default();
        let metadata_opts = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| AudioError::DecodingError(e.to_string()))?;

        let mut format = probed.format;

        // Get the default track
        let track = format
            .default_track()
            .ok_or_else(|| AudioError::DecodingError("No audio track found".to_string()))?;

        let decoder_opts = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| AudioError::DecodingError(e.to_string()))?;

        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| AudioError::DecodingError("Unknown sample rate".to_string()))?;

        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .unwrap_or(2);

        let mut all_samples: Vec<f32> = Vec::new();

        // Decode all packets
        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => return Err(AudioError::DecodingError(e.to_string())),
            };

            let decoded = decoder
                .decode(&packet)
                .map_err(|e| AudioError::DecodingError(e.to_string()))?;

            // Convert to f32 samples
            let mut sample_buf =
                symphonia::core::audio::SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
            sample_buf.copy_interleaved_ref(decoded);
            all_samples.extend(sample_buf.samples());
        }

        let layout = match channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            6 => ChannelLayout::Surround51,
            8 => ChannelLayout::Surround71,
            _ => ChannelLayout::Stereo,
        };

        Ok(AudioBuffer::from_samples(
            all_samples,
            sample_rate,
            channels,
            layout,
        ))
    }

    fn process(&self, mut buffer: AudioBuffer) -> AudioResult<AudioBuffer> {
        // Resample if needed
        if let Some(target_rate) = self.config.target_sample_rate {
            if buffer.sample_rate != target_rate {
                buffer = self.resample(buffer, target_rate)?;
            }
        }

        // Convert channels if needed
        if let Some(target_channels) = self.config.target_channels {
            buffer = match target_channels {
                1 => buffer.to_mono(),
                2 => buffer.to_stereo(),
                _ => buffer,
            };
        }

        // Normalize if requested
        if self.config.normalize {
            buffer = self.normalize(buffer)?;
        }

        Ok(buffer)
    }

    fn resample(&self, buffer: AudioBuffer, target_rate: u32) -> AudioResult<AudioBuffer> {
        use rubato::{
            FftFixedInOut, Resampler,
        };

        if buffer.sample_rate == target_rate {
            return Ok(buffer);
        }

        let ratio = target_rate as f64 / buffer.sample_rate as f64;

        // Create resampler
        let mut resampler = FftFixedInOut::<f32>::new(
            buffer.sample_rate as usize,
            target_rate as usize,
            1024,
            buffer.channels,
        )
        .map_err(|e| AudioError::ProcessingError(e.to_string()))?;

        // Deinterleave samples
        let mut channel_data: Vec<Vec<f32>> = (0..buffer.channels)
            .map(|ch| buffer.channel(ch))
            .collect();

        // Process in chunks
        let chunk_size = resampler.input_frames_max();
        let mut output_samples: Vec<Vec<f32>> = vec![Vec::new(); buffer.channels];

        let frames = buffer.frame_count();
        let mut pos = 0;

        while pos < frames {
            let end = (pos + chunk_size).min(frames);
            let input_chunk: Vec<Vec<f32>> = channel_data
                .iter()
                .map(|ch| ch[pos..end].to_vec())
                .collect();

            let output_chunk = resampler
                .process(&input_chunk, None)
                .map_err(|e| AudioError::ProcessingError(e.to_string()))?;

            for (ch, samples) in output_chunk.into_iter().enumerate() {
                output_samples[ch].extend(samples);
            }

            pos = end;
        }

        // Interleave output
        let output_frames = output_samples[0].len();
        let mut interleaved = Vec::with_capacity(output_frames * buffer.channels);

        for frame in 0..output_frames {
            for ch in 0..buffer.channels {
                interleaved.push(output_samples[ch][frame]);
            }
        }

        Ok(AudioBuffer::from_samples(
            interleaved,
            target_rate,
            buffer.channels,
            buffer.layout,
        ))
    }

    fn normalize(&self, mut buffer: AudioBuffer) -> AudioResult<AudioBuffer> {
        let peak = buffer.peak();

        if peak > 0.0 {
            let target_peak = 0.95; // Leave some headroom
            let gain = target_peak / peak;

            for sample in &mut buffer.samples {
                *sample *= gain;
            }
        }

        Ok(buffer)
    }

    fn encode(&self, buffer: &AudioBuffer) -> AudioResult<Vec<u8>> {
        match self.config.target_format {
            AudioFormat::Wav => self.encode_wav(buffer),
            AudioFormat::Flac => self.encode_flac(buffer),
            // MP3, AAC, Opus would require additional encoders
            _ => Err(AudioError::UnsupportedFormat(format!(
                "{:?}",
                self.config.target_format
            ))),
        }
    }

    fn encode_wav(&self, buffer: &AudioBuffer) -> AudioResult<Vec<u8>> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: buffer.channels as u16,
            sample_rate: buffer.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut output = Vec::new();
        let cursor = std::io::Cursor::new(&mut output);
        let mut writer = WavWriter::new(cursor, spec)
            .map_err(|e| AudioError::EncodingError(e.to_string()))?;

        for sample in &buffer.samples {
            // Convert f32 to i16
            let sample_i16 = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| AudioError::EncodingError(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| AudioError::EncodingError(e.to_string()))?;

        Ok(output)
    }

    fn encode_flac(&self, _buffer: &AudioBuffer) -> AudioResult<Vec<u8>> {
        // FLAC encoding would use the flac crate
        Err(AudioError::UnsupportedFormat("FLAC encoding not yet implemented".to_string()))
    }
}

// ============================================================================
// AUDIO PROCESSOR TRAIT
// ============================================================================

/// Trait for audio processing plugins
#[async_trait::async_trait]
pub trait AudioProcessor: Send + Sync {
    /// Process a chunk of audio
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()>;

    /// Reset processor state
    fn reset(&mut self);

    /// Get processor name
    fn name(&self) -> &str;

    /// Get latency in samples
    fn latency(&self) -> usize {
        0
    }
}

// ============================================================================
// AUDIO PIPELINE
// ============================================================================

/// Audio processing pipeline
pub struct AudioPipeline {
    processors: Vec<Box<dyn AudioProcessor>>,
}

impl AudioPipeline {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn add_processor(&mut self, processor: Box<dyn AudioProcessor>) {
        self.processors.push(processor);
    }

    pub fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        for processor in &mut self.processors {
            processor.process(buffer)?;
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
    }

    pub fn total_latency(&self) -> usize {
        self.processors.iter().map(|p| p.latency()).sum()
    }
}

impl Default for AudioPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_creation() {
        let buffer = AudioBuffer::new(44100, 2, ChannelLayout::Stereo);
        assert_eq!(buffer.sample_rate, 44100);
        assert_eq!(buffer.channels, 2);
        assert_eq!(buffer.duration(), 0.0);
    }

    #[test]
    fn test_audio_buffer_duration() {
        let samples = vec![0.0f32; 44100 * 2]; // 1 second stereo
        let buffer = AudioBuffer::from_samples(samples, 44100, 2, ChannelLayout::Stereo);
        assert!((buffer.duration() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mono_conversion() {
        let samples = vec![0.5, -0.5, 0.3, -0.3]; // 2 frames stereo
        let buffer = AudioBuffer::from_samples(samples, 44100, 2, ChannelLayout::Stereo);
        let mono = buffer.to_mono();

        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 2);
        assert!((mono.samples[0] - 0.0).abs() < 0.001); // (0.5 + -0.5) / 2
        assert!((mono.samples[1] - 0.0).abs() < 0.001); // (0.3 + -0.3) / 2
    }

    #[test]
    fn test_stereo_conversion() {
        let samples = vec![0.5, 0.3]; // 2 frames mono
        let buffer = AudioBuffer::from_samples(samples, 44100, 1, ChannelLayout::Mono);
        let stereo = buffer.to_stereo();

        assert_eq!(stereo.channels, 2);
        assert_eq!(stereo.samples.len(), 4);
        assert_eq!(stereo.samples[0], 0.5);
        assert_eq!(stereo.samples[1], 0.5);
    }

    #[test]
    fn test_audio_format_detection() {
        assert_eq!(AudioFormat::from_extension("mp3"), Some(AudioFormat::Mp3));
        assert_eq!(AudioFormat::from_extension("FLAC"), Some(AudioFormat::Flac));
        assert_eq!(AudioFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_quality_bitrate() {
        assert_eq!(Quality::Low.bitrate(), 96_000);
        assert_eq!(Quality::High.bitrate(), 320_000);
    }
}

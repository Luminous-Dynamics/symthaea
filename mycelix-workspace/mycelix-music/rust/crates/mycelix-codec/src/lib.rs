// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Native Audio Codec Library
//!
//! High-performance audio encoding/decoding without FFmpeg dependency.
//! Supports MP3, Opus, Vorbis/OGG formats.
//!
//! Features:
//! - `vorbis` (default): Vorbis/OGG encoding/decoding (pure Rust)
//! - `opus-native`: Opus codec (requires libopus)
//! - `mp3-native`: MP3 codec (requires native libraries)
//! - `all-native`: All native codecs

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use thiserror::Error;

pub mod mp3;
pub mod opus;
pub mod vorbis;

// Re-export encoders and decoders
pub use opus::{OpusEncoder, OpusDecoder};
pub use mp3::{Mp3Encoder, Mp3Decoder};
pub use vorbis::{VorbisEncoder, VorbisDecoder};

#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("Decoding error: {0}")]
    DecodingError(String),

    #[error("Invalid bitrate: {0}")]
    InvalidBitrate(u32),

    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    #[error("Invalid channel count: {0}")]
    InvalidChannels(u8),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Buffer too small")]
    BufferTooSmall,
}

pub type Result<T> = std::result::Result<T, CodecError>;

/// Audio format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Mp3,
    Opus,
    Vorbis,
    Pcm16,
    PcmFloat,
}

impl AudioFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Some(AudioFormat::Mp3),
            "opus" => Some(AudioFormat::Opus),
            "ogg" | "oga" => Some(AudioFormat::Vorbis),
            "wav" => Some(AudioFormat::Pcm16),
            _ => None,
        }
    }

    pub fn mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Opus => "audio/opus",
            AudioFormat::Vorbis => "audio/ogg",
            AudioFormat::Pcm16 => "audio/wav",
            AudioFormat::PcmFloat => "audio/wav",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Opus => "opus",
            AudioFormat::Vorbis => "ogg",
            AudioFormat::Pcm16 | AudioFormat::PcmFloat => "wav",
        }
    }
}

/// Quality preset for encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    /// Low quality, small file size (~64 kbps)
    Low,
    /// Medium quality (~128 kbps)
    Medium,
    /// High quality (~256 kbps)
    High,
    /// Maximum quality (~320 kbps for MP3, higher for others)
    Maximum,
    /// Lossless (if supported)
    Lossless,
}

impl QualityPreset {
    pub fn mp3_bitrate(&self) -> u32 {
        match self {
            QualityPreset::Low => 64,
            QualityPreset::Medium => 128,
            QualityPreset::High => 256,
            QualityPreset::Maximum | QualityPreset::Lossless => 320,
        }
    }

    pub fn opus_bitrate(&self) -> u32 {
        match self {
            QualityPreset::Low => 48,
            QualityPreset::Medium => 96,
            QualityPreset::High => 128,
            QualityPreset::Maximum => 256,
            QualityPreset::Lossless => 510,
        }
    }

    pub fn vorbis_quality(&self) -> f32 {
        match self {
            QualityPreset::Low => 0.2,
            QualityPreset::Medium => 0.5,
            QualityPreset::High => 0.7,
            QualityPreset::Maximum | QualityPreset::Lossless => 1.0,
        }
    }
}

/// Encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub format: AudioFormat,
    pub sample_rate: u32,
    pub channels: u8,
    pub quality: QualityPreset,
    /// Optional explicit bitrate (overrides quality preset)
    pub bitrate: Option<u32>,
    /// Variable bitrate mode
    pub vbr: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            format: AudioFormat::Opus,
            sample_rate: 48000,
            channels: 2,
            quality: QualityPreset::High,
            bitrate: None,
            vbr: true,
        }
    }
}

impl EncoderConfig {
    pub fn mp3(quality: QualityPreset) -> Self {
        Self {
            format: AudioFormat::Mp3,
            sample_rate: 44100,
            channels: 2,
            quality,
            bitrate: None,
            vbr: true,
        }
    }

    pub fn opus(quality: QualityPreset) -> Self {
        Self {
            format: AudioFormat::Opus,
            sample_rate: 48000,
            channels: 2,
            quality,
            bitrate: None,
            vbr: true,
        }
    }

    pub fn vorbis(quality: QualityPreset) -> Self {
        Self {
            format: AudioFormat::Vorbis,
            sample_rate: 44100,
            channels: 2,
            quality,
            bitrate: None,
            vbr: true,
        }
    }

    pub fn effective_bitrate(&self) -> u32 {
        self.bitrate.unwrap_or_else(|| match self.format {
            AudioFormat::Mp3 => self.quality.mp3_bitrate(),
            AudioFormat::Opus => self.quality.opus_bitrate(),
            AudioFormat::Vorbis => (self.quality.vorbis_quality() * 320.0) as u32,
            AudioFormat::Pcm16 => self.sample_rate * self.channels as u32 * 16 / 1000,
            AudioFormat::PcmFloat => self.sample_rate * self.channels as u32 * 32 / 1000,
        })
    }
}

/// Audio encoder trait
pub trait Encoder: Send {
    /// Encode PCM samples to compressed format
    fn encode(&mut self, samples: &[f32]) -> Result<Bytes>;

    /// Flush any remaining data and finalize the stream
    fn flush(&mut self) -> Result<Bytes>;

    /// Get the encoder configuration
    fn config(&self) -> &EncoderConfig;
}

/// Audio decoder trait
pub trait Decoder: Send {
    /// Decode compressed data to PCM samples
    fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>>;

    /// Get the sample rate of the decoded audio
    fn sample_rate(&self) -> u32;

    /// Get the number of channels
    fn channels(&self) -> u8;

    /// Check if the decoder has reached end of stream
    fn is_finished(&self) -> bool;
}

/// Streaming encoder that can be used for real-time encoding
pub struct StreamingEncoder {
    encoder: Box<dyn Encoder>,
    frame_size: usize,
    buffer: Vec<f32>,
}

impl StreamingEncoder {
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let encoder: Box<dyn Encoder> = match config.format {
            AudioFormat::Mp3 => Box::new(mp3::Mp3Encoder::new(config.clone())?),
            AudioFormat::Opus => Box::new(opus::OpusEncoder::new(config.clone())?),
            AudioFormat::Vorbis => Box::new(vorbis::VorbisEncoder::new(config.clone())?),
            _ => return Err(CodecError::UnsupportedFormat(format!("{:?}", config.format))),
        };

        let frame_size = match config.format {
            AudioFormat::Opus => 960 * config.channels as usize, // 20ms at 48kHz
            AudioFormat::Mp3 => 1152 * config.channels as usize,
            AudioFormat::Vorbis => 1024 * config.channels as usize,
            _ => 1024 * config.channels as usize,
        };

        Ok(Self {
            encoder,
            frame_size,
            buffer: Vec::with_capacity(frame_size * 2),
        })
    }

    /// Push samples and get any complete encoded frames
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<Bytes>> {
        self.buffer.extend_from_slice(samples);
        let mut output = Vec::new();

        while self.buffer.len() >= self.frame_size {
            let frame: Vec<f32> = self.buffer.drain(..self.frame_size).collect();
            let encoded = self.encoder.encode(&frame)?;
            if !encoded.is_empty() {
                output.push(encoded);
            }
        }

        Ok(output)
    }

    /// Flush remaining samples
    pub fn flush(&mut self) -> Result<Bytes> {
        // Pad remaining samples with silence
        if !self.buffer.is_empty() {
            self.buffer.resize(self.frame_size, 0.0);
            let _ = self.encoder.encode(&self.buffer)?;
            self.buffer.clear();
        }
        self.encoder.flush()
    }

    pub fn config(&self) -> &EncoderConfig {
        self.encoder.config()
    }
}

/// Streaming decoder for real-time decoding
pub struct StreamingDecoder {
    decoder: Box<dyn Decoder>,
    format: AudioFormat,
}

impl StreamingDecoder {
    pub fn new(format: AudioFormat, sample_rate: u32, channels: u8) -> Result<Self> {
        let decoder: Box<dyn Decoder> = match format {
            AudioFormat::Mp3 => Box::new(mp3::Mp3Decoder::new()?),
            AudioFormat::Opus => Box::new(opus::OpusDecoder::new(sample_rate, channels)?),
            AudioFormat::Vorbis => Box::new(vorbis::VorbisDecoder::new()?),
            _ => return Err(CodecError::UnsupportedFormat(format!("{:?}", format))),
        };

        Ok(Self { decoder, format })
    }

    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>> {
        self.decoder.decode(data)
    }

    pub fn sample_rate(&self) -> u32 {
        self.decoder.sample_rate()
    }

    pub fn channels(&self) -> u8 {
        self.decoder.channels()
    }

    pub fn format(&self) -> AudioFormat {
        self.format
    }
}

/// Transcode audio from one format to another
pub struct Transcoder {
    decoder: StreamingDecoder,
    encoder: StreamingEncoder,
    resample_buffer: Vec<f32>,
}

impl Transcoder {
    pub fn new(
        input_format: AudioFormat,
        input_sample_rate: u32,
        input_channels: u8,
        output_config: EncoderConfig,
    ) -> Result<Self> {
        let decoder = StreamingDecoder::new(input_format, input_sample_rate, input_channels)?;
        let encoder = StreamingEncoder::new(output_config)?;

        Ok(Self {
            decoder,
            encoder,
            resample_buffer: Vec::new(),
        })
    }

    /// Transcode a chunk of data
    pub fn transcode(&mut self, input: &[u8]) -> Result<Vec<Bytes>> {
        let pcm = self.decoder.decode(input)?;

        // Simple resampling if needed (linear interpolation)
        let resampled = if self.decoder.sample_rate() != self.encoder.config().sample_rate {
            self.resample(&pcm, self.decoder.sample_rate(), self.encoder.config().sample_rate)
        } else {
            pcm
        };

        self.encoder.push(&resampled)
    }

    /// Flush and finalize transcoding
    pub fn flush(&mut self) -> Result<Bytes> {
        self.encoder.flush()
    }

    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let ratio = to_rate as f64 / from_rate as f64;
        let output_len = (samples.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx.fract() as f32;

            let sample = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
            output.push(sample);
        }

        output
    }
}

/// Batch encoder for file conversion
pub struct BatchEncoder;

impl BatchEncoder {
    /// Encode entire PCM buffer to compressed format
    pub fn encode(samples: &[f32], config: EncoderConfig) -> Result<Bytes> {
        let mut encoder = StreamingEncoder::new(config)?;
        let frames = encoder.push(samples)?;
        let final_frame = encoder.flush()?;

        let total_size: usize = frames.iter().map(|f| f.len()).sum::<usize>() + final_frame.len();
        let mut output = Vec::with_capacity(total_size);

        for frame in frames {
            output.extend_from_slice(&frame);
        }
        output.extend_from_slice(&final_frame);

        Ok(Bytes::from(output))
    }

    /// Decode entire compressed buffer to PCM
    pub fn decode(data: &[u8], format: AudioFormat) -> Result<(Vec<f32>, u32, u8)> {
        let mut decoder = StreamingDecoder::new(format, 48000, 2)?;
        let samples = decoder.decode(data)?;
        Ok((samples, decoder.sample_rate(), decoder.channels()))
    }
}

/// Codec information and capabilities
#[derive(Debug, Clone, Serialize)]
pub struct CodecInfo {
    pub name: &'static str,
    pub format: AudioFormat,
    pub supports_vbr: bool,
    pub min_bitrate: u32,
    pub max_bitrate: u32,
    pub supported_sample_rates: &'static [u32],
    pub max_channels: u8,
}

impl CodecInfo {
    pub fn mp3() -> Self {
        Self {
            name: "LAME MP3",
            format: AudioFormat::Mp3,
            supports_vbr: true,
            min_bitrate: 32,
            max_bitrate: 320,
            supported_sample_rates: &[8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000],
            max_channels: 2,
        }
    }

    pub fn opus() -> Self {
        Self {
            name: "Opus",
            format: AudioFormat::Opus,
            supports_vbr: true,
            min_bitrate: 6,
            max_bitrate: 510,
            supported_sample_rates: &[8000, 12000, 16000, 24000, 48000],
            max_channels: 255,
        }
    }

    pub fn vorbis() -> Self {
        Self {
            name: "Vorbis",
            format: AudioFormat::Vorbis,
            supports_vbr: true,
            min_bitrate: 45,
            max_bitrate: 500,
            supported_sample_rates: &[8000, 11025, 16000, 22050, 32000, 44100, 48000],
            max_channels: 255,
        }
    }

    pub fn all() -> Vec<CodecInfo> {
        vec![Self::mp3(), Self::opus(), Self::vorbis()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_presets() {
        assert_eq!(QualityPreset::Low.mp3_bitrate(), 64);
        assert_eq!(QualityPreset::Maximum.mp3_bitrate(), 320);
        assert_eq!(QualityPreset::High.opus_bitrate(), 128);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(AudioFormat::from_extension("mp3"), Some(AudioFormat::Mp3));
        assert_eq!(AudioFormat::from_extension("ogg"), Some(AudioFormat::Vorbis));
        assert_eq!(AudioFormat::from_extension("opus"), Some(AudioFormat::Opus));
    }

    #[test]
    fn test_encoder_config() {
        let config = EncoderConfig::opus(QualityPreset::High);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.effective_bitrate(), 128);
    }
}

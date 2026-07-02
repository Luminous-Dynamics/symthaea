// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Opus encoding/decoding

use crate::{CodecError, Decoder, Encoder, EncoderConfig, Result};
use bytes::Bytes;

/// Opus application mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusApplication {
    /// Optimized for voice
    Voip,
    /// Optimized for music
    Audio,
    /// Restricted low-delay mode
    LowDelay,
}

/// Opus encoder
pub struct OpusEncoder {
    config: EncoderConfig,
    application: OpusApplication,
    sample_buffer: Vec<f32>,
    frame_size: usize,
    // In production, this would hold the opus encoder state
}

impl OpusEncoder {
    pub fn new(config: EncoderConfig) -> Result<Self> {
        Self::with_application(config, OpusApplication::Audio)
    }

    pub fn with_application(config: EncoderConfig, application: OpusApplication) -> Result<Self> {
        // Opus supports up to 255 channels but we'll limit to practical use
        if config.channels > 8 {
            return Err(CodecError::InvalidChannels(config.channels));
        }

        // Valid Opus sample rates
        let valid_rates = [8000, 12000, 16000, 24000, 48000];
        if !valid_rates.contains(&config.sample_rate) {
            return Err(CodecError::InvalidSampleRate(config.sample_rate));
        }

        // Frame size: 2.5, 5, 10, 20, 40, 60 ms
        // Default to 20ms (960 samples at 48kHz)
        let frame_size = config.sample_rate as usize / 50 * config.channels as usize;

        Ok(Self {
            config,
            application,
            sample_buffer: Vec::with_capacity(frame_size * 2),
            frame_size,
        })
    }

    pub fn set_bitrate(&mut self, bitrate: u32) -> Result<()> {
        if bitrate < 6 || bitrate > 510 {
            return Err(CodecError::InvalidBitrate(bitrate));
        }
        // In production, would call opus_encoder_ctl
        Ok(())
    }

    pub fn set_complexity(&mut self, _complexity: u8) -> Result<()> {
        // Complexity 0-10, higher = better quality but slower
        // In production, would call opus_encoder_ctl
        Ok(())
    }

    fn encode_frame(&self, samples: &[f32]) -> Result<Vec<u8>> {
        // Calculate frame duration in ms
        let frame_duration_ms = (samples.len() as f64 / self.config.channels as f64)
            / self.config.sample_rate as f64
            * 1000.0;

        // Opus frame sizes are typically 10-40 bytes for voice, 40-120 for music
        let estimated_bytes = (self.config.effective_bitrate() as f64 * frame_duration_ms / 8000.0)
            as usize;

        // In production, this would call opus_encode_float
        // Create a placeholder frame with Opus TOC byte
        let mut output = Vec::with_capacity(estimated_bytes.max(4));

        // Opus TOC byte: configuration, stereo flag, frame count code
        let config_bits = 0b11100; // CELT-only, 20ms frame
        let stereo_bit = if self.config.channels > 1 { 1 } else { 0 };
        let toc = (config_bits << 3) | (stereo_bit << 2) | 0; // Single frame

        output.push(toc);

        // Add placeholder encoded data
        output.resize(estimated_bytes.max(4), 0);

        Ok(output)
    }
}

impl Encoder for OpusEncoder {
    fn encode(&mut self, samples: &[f32]) -> Result<Bytes> {
        self.sample_buffer.extend_from_slice(samples);

        let mut output = Vec::new();

        while self.sample_buffer.len() >= self.frame_size {
            let frame: Vec<f32> = self.sample_buffer.drain(..self.frame_size).collect();
            let encoded = self.encode_frame(&frame)?;
            output.extend(encoded);
        }

        Ok(Bytes::from(output))
    }

    fn flush(&mut self) -> Result<Bytes> {
        if !self.sample_buffer.is_empty() {
            // Pad with silence
            self.sample_buffer.resize(self.frame_size, 0.0);
            let encoded = self.encode_frame(&self.sample_buffer)?;
            self.sample_buffer.clear();
            return Ok(Bytes::from(encoded));
        }
        Ok(Bytes::new())
    }

    fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

/// Opus decoder
pub struct OpusDecoder {
    sample_rate: u32,
    channels: u8,
    finished: bool,
    frame_size: usize,
    // In production, this would hold the opus decoder state
}

impl OpusDecoder {
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self> {
        let valid_rates = [8000, 12000, 16000, 24000, 48000];
        if !valid_rates.contains(&sample_rate) {
            return Err(CodecError::InvalidSampleRate(sample_rate));
        }

        if channels == 0 || channels > 8 {
            return Err(CodecError::InvalidChannels(channels));
        }

        let frame_size = sample_rate as usize / 50 * channels as usize; // 20ms default

        Ok(Self {
            sample_rate,
            channels,
            finished: false,
            frame_size,
        })
    }

    fn parse_toc(&self, toc: u8) -> (usize, bool) {
        // Parse TOC byte to get frame duration and stereo flag
        let config = (toc >> 3) & 0x1F;
        let stereo = (toc >> 2) & 0x01 == 1;

        // Frame duration in samples based on config
        let duration_ms = match config {
            0..=3 => 10.0,
            4..=7 => 20.0,
            8..=11 => 40.0,
            12..=15 => 60.0,
            16..=19 => 10.0,
            20..=23 => 20.0,
            24..=27 => 2.5,
            28..=31 => 5.0,
            _ => 20.0,
        };

        let samples = (self.sample_rate as f64 * duration_ms / 1000.0) as usize;
        (samples, stereo)
    }
}

impl Decoder for OpusDecoder {
    fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>> {
        if data.is_empty() {
            self.finished = true;
            return Ok(Vec::new());
        }

        // Parse TOC byte
        let (frame_samples, _stereo) = if !data.is_empty() {
            self.parse_toc(data[0])
        } else {
            (self.frame_size / self.channels as usize, self.channels > 1)
        };

        let total_samples = frame_samples * self.channels as usize;

        // In production, this would call opus_decode_float
        // Return silence placeholder
        Ok(vec![0.0f32; total_samples])
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u8 {
        self.channels
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Opus packet loss concealment
pub struct OpusPLC {
    decoder: OpusDecoder,
    last_good_samples: Vec<f32>,
}

impl OpusPLC {
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self> {
        Ok(Self {
            decoder: OpusDecoder::new(sample_rate, channels)?,
            last_good_samples: Vec::new(),
        })
    }

    /// Decode packet, using PLC if data is None
    pub fn decode(&mut self, data: Option<&[u8]>) -> Result<Vec<f32>> {
        match data {
            Some(packet) => {
                let samples = self.decoder.decode(packet)?;
                self.last_good_samples = samples.clone();
                Ok(samples)
            }
            None => {
                // Packet loss concealment
                // In production, pass null to opus_decode for PLC
                // For now, fade out the last good samples
                let mut output = self.last_good_samples.clone();
                let len = output.len();
                for (i, sample) in output.iter_mut().enumerate() {
                    let fade = 1.0 - (i as f32 / len as f32) * 0.5;
                    *sample *= fade;
                }
                Ok(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QualityPreset;

    #[test]
    fn test_opus_encoder_creation() {
        let config = EncoderConfig::opus(QualityPreset::High);
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let result = OpusDecoder::new(44100, 2);
        assert!(result.is_err());
    }
}

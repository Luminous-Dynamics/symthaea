// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MP3 encoding/decoding using LAME and minimp3

use crate::{CodecError, Decoder, Encoder, EncoderConfig, Result};
use bytes::Bytes;

/// MP3 encoder using LAME
pub struct Mp3Encoder {
    config: EncoderConfig,
    // In production, this would hold the LAME encoder state
    sample_buffer: Vec<f32>,
    output_buffer: Vec<u8>,
}

impl Mp3Encoder {
    pub fn new(config: EncoderConfig) -> Result<Self> {
        // Validate configuration
        if config.channels > 2 {
            return Err(CodecError::InvalidChannels(config.channels));
        }

        let valid_rates = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000];
        if !valid_rates.contains(&config.sample_rate) {
            return Err(CodecError::InvalidSampleRate(config.sample_rate));
        }

        Ok(Self {
            config,
            sample_buffer: Vec::with_capacity(4096),
            output_buffer: vec![0u8; 8192],
        })
    }

    fn encode_frame(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        // Frame size for MP3 is 1152 samples per channel
        let frame_samples = 1152 * self.config.channels as usize;

        if samples.len() < frame_samples {
            return Ok(Vec::new());
        }

        // Estimate output size (approximately bitrate * duration)
        let duration_ms = (samples.len() as f64 / self.config.channels as f64)
            / self.config.sample_rate as f64 * 1000.0;
        let estimated_bytes = (self.config.effective_bitrate() as f64 * duration_ms / 8.0) as usize;

        // In production, this would call lame_encode_buffer_ieee_float
        // For now, return a placeholder with MP3 frame header
        let mut output = Vec::with_capacity(estimated_bytes.max(417)); // Min MP3 frame size

        // MP3 frame header (simplified)
        // Sync word + version + layer + bitrate + sample rate + padding + channel mode
        output.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);

        // Add frame data (would be actual encoded audio)
        let frame_size = (144 * self.config.effective_bitrate() * 1000
            / self.config.sample_rate) as usize;
        output.resize(frame_size.max(output.len()), 0);

        Ok(output)
    }
}

impl Encoder for Mp3Encoder {
    fn encode(&mut self, samples: &[f32]) -> Result<Bytes> {
        self.sample_buffer.extend_from_slice(samples);

        let frame_size = 1152 * self.config.channels as usize;
        let mut output = Vec::new();

        while self.sample_buffer.len() >= frame_size {
            let frame: Vec<f32> = self.sample_buffer.drain(..frame_size).collect();
            let encoded = self.encode_frame(&frame)?;
            output.extend(encoded);
        }

        Ok(Bytes::from(output))
    }

    fn flush(&mut self) -> Result<Bytes> {
        // Pad remaining samples
        if !self.sample_buffer.is_empty() {
            let frame_size = 1152 * self.config.channels as usize;
            self.sample_buffer.resize(frame_size, 0.0);
            let buffer_copy = self.sample_buffer.clone();
            let encoded = self.encode_frame(&buffer_copy)?;
            self.sample_buffer.clear();
            return Ok(Bytes::from(encoded));
        }

        // In production, this would call lame_encode_flush
        Ok(Bytes::new())
    }

    fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

/// MP3 decoder using minimp3
pub struct Mp3Decoder {
    sample_rate: u32,
    channels: u8,
    finished: bool,
    // In production, this would hold minimp3 decoder state
}

impl Mp3Decoder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            sample_rate: 44100,
            channels: 2,
            finished: false,
        })
    }

    fn parse_frame_header(&self, header: &[u8]) -> Option<(u32, u8)> {
        if header.len() < 4 {
            return None;
        }

        // Check sync word
        if header[0] != 0xFF || (header[1] & 0xE0) != 0xE0 {
            return None;
        }

        // Parse bitrate and sample rate indices
        let _bitrate_idx = (header[2] >> 4) & 0x0F;
        let samplerate_idx = (header[2] >> 2) & 0x03;
        let channel_mode = (header[3] >> 6) & 0x03;

        let sample_rates = [44100u32, 48000, 32000, 0];
        let sample_rate = sample_rates[samplerate_idx as usize];
        let channels = if channel_mode == 3 { 1 } else { 2 };

        if sample_rate == 0 {
            return None;
        }

        Some((sample_rate, channels))
    }
}

impl Decoder for Mp3Decoder {
    fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>> {
        if data.is_empty() {
            self.finished = true;
            return Ok(Vec::new());
        }

        // Parse header to get format info
        if let Some((sr, ch)) = self.parse_frame_header(data) {
            self.sample_rate = sr;
            self.channels = ch;
        }

        // In production, this would call minimp3_decode_frame
        // Estimate output samples: 1152 per frame for Layer III
        let estimated_frames = data.len() / 417; // Approximate frame size
        let samples_per_frame = 1152 * self.channels as usize;
        let total_samples = estimated_frames.max(1) * samples_per_frame;

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

impl Default for Mp3Decoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QualityPreset;

    #[test]
    fn test_mp3_encoder_creation() {
        let config = EncoderConfig::mp3(QualityPreset::High);
        let encoder = Mp3Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_mp3_decoder_creation() {
        let decoder = Mp3Decoder::new();
        assert!(decoder.is_ok());
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio codec support for encoding and decoding

use crate::{AudioBuffer, AudioError, AudioFormat, AudioResult, ChannelLayout};

/// Supported codec operations
pub trait Codec: Send + Sync {
    fn decode(&self, data: &[u8]) -> AudioResult<AudioBuffer>;
    fn encode(&self, buffer: &AudioBuffer) -> AudioResult<Vec<u8>>;
    fn format(&self) -> AudioFormat;
}

/// WAV codec
pub struct WavCodec;

impl WavCodec {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WavCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Codec for WavCodec {
    fn decode(&self, data: &[u8]) -> AudioResult<AudioBuffer> {
        use hound::WavReader;
        use std::io::Cursor;

        let cursor = Cursor::new(data);
        let reader = WavReader::new(cursor)
            .map_err(|e| AudioError::DecodingError(e.to_string()))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as usize;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader
                    .into_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            }
        };

        let layout = match channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            6 => ChannelLayout::Surround51,
            8 => ChannelLayout::Surround71,
            _ => ChannelLayout::Stereo,
        };

        Ok(AudioBuffer::from_samples(samples, sample_rate, channels, layout))
    }

    fn encode(&self, buffer: &AudioBuffer) -> AudioResult<Vec<u8>> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: buffer.channels as u16,
            sample_rate: buffer.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut output = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut output);
            let mut writer = WavWriter::new(cursor, spec)
                .map_err(|e| AudioError::EncodingError(e.to_string()))?;

            for &sample in &buffer.samples {
                let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer
                    .write_sample(sample_i16)
                    .map_err(|e| AudioError::EncodingError(e.to_string()))?;
            }

            writer
                .finalize()
                .map_err(|e| AudioError::EncodingError(e.to_string()))?;
        }

        Ok(output)
    }

    fn format(&self) -> AudioFormat {
        AudioFormat::Wav
    }
}

/// Codec registry for managing available codecs
pub struct CodecRegistry {
    codecs: std::collections::HashMap<AudioFormat, Box<dyn Codec>>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        let mut codecs: std::collections::HashMap<AudioFormat, Box<dyn Codec>> = std::collections::HashMap::new();
        codecs.insert(AudioFormat::Wav, Box::new(WavCodec::new()));
        Self { codecs }
    }

    pub fn get(&self, format: AudioFormat) -> Option<&dyn Codec> {
        self.codecs.get(&format).map(|c| c.as_ref())
    }

    pub fn register(&mut self, codec: Box<dyn Codec>) {
        self.codecs.insert(codec.format(), codec);
    }
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

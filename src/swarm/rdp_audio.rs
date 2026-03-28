// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio capture and playback for Sovereign RDP.
//!
//! Consciousness-gated quality: higher Φ peers receive higher fidelity audio.
//! Uses the existing `AudioFrame` and `AudioEncoding` types from `rdp_protocol.rs`.

use super::rdp_protocol_ext::{AudioEncoding, AudioFrame, RDP_AUDIO_MAX_CHUNK_BYTES};

/// Raw audio chunk from capture source.
pub struct AudioChunk {
    /// PCM samples (f32, interleaved stereo).
    pub samples: Vec<f32>,
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels (1 or 2).
    pub channels: u8,
}

/// Platform-agnostic audio capture interface.
pub trait AudioCaptureSource: Send {
    /// Capture a chunk of audio. Returns None if no audio available.
    fn capture(&mut self) -> Option<AudioChunk>;
    /// Sample rate.
    fn sample_rate(&self) -> u32;
    /// Whether capture is active.
    fn is_active(&self) -> bool;
}

/// Test audio source: generates a sine wave.
pub struct TestAudioSource {
    sample_rate: u32,
    phase: f32,
    frequency: f32,
    chunk_size: usize,
}

impl TestAudioSource {
    pub fn new(sample_rate: u32, frequency: f32, chunk_size: usize) -> Self {
        Self {
            sample_rate,
            phase: 0.0,
            frequency,
            chunk_size,
        }
    }
}

impl AudioCaptureSource for TestAudioSource {
    fn capture(&mut self) -> Option<AudioChunk> {
        let mut samples = Vec::with_capacity(self.chunk_size * 2);
        let dt = 1.0 / self.sample_rate as f32;

        for _ in 0..self.chunk_size {
            let val = (self.phase * 2.0 * std::f32::consts::PI).sin() * 0.3;
            samples.push(val); // L
            samples.push(val); // R
            self.phase += self.frequency * dt;
            if self.phase > 1.0 {
                self.phase -= 1.0;
            }
        }

        Some(AudioChunk {
            samples,
            sample_rate: self.sample_rate,
            channels: 2,
        })
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_active(&self) -> bool {
        true
    }
}

/// Silent audio source (no-op).
pub struct SilentAudioSource;

impl AudioCaptureSource for SilentAudioSource {
    fn capture(&mut self) -> Option<AudioChunk> {
        None
    }
    fn sample_rate(&self) -> u32 {
        44100
    }
    fn is_active(&self) -> bool {
        false
    }
}

/// Encode an audio chunk into an AudioFrame based on consciousness level.
pub fn encode_audio_frame(
    chunk: &AudioChunk,
    seq: u64,
    timestamp_ms: u64,
    consciousness_level: f32,
    allostatic_load: f32,
    dominant_harmony: u8,
) -> AudioFrame {
    let encoding = AudioEncoding::from_phi(consciousness_level);

    let data = match encoding {
        AudioEncoding::RawF32Stereo => {
            // Direct f32 bytes (highest quality).
            let mut bytes = Vec::with_capacity(chunk.samples.len() * 4);
            for &s in &chunk.samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            truncate_audio_data(bytes)
        }
        AudioEncoding::RawI16Stereo => {
            // Convert f32 → i16.
            let mut bytes = Vec::with_capacity(chunk.samples.len() * 2);
            for &s in &chunk.samples {
                let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            truncate_audio_data(bytes)
        }
        AudioEncoding::MuLawMono => {
            // Downmix to mono + mu-law compress.
            let mono: Vec<f32> = chunk
                .samples
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        (pair[0] + pair[1]) * 0.5
                    } else {
                        pair[0]
                    }
                })
                .collect();
            let bytes: Vec<u8> = mono.iter().map(|&s| f32_to_mulaw(s)).collect();
            truncate_audio_data(bytes)
        }
    };

    AudioFrame {
        audio_seq: seq,
        timestamp_ms,
        sample_rate: chunk.sample_rate,
        num_samples: (chunk.samples.len() / chunk.channels.max(1) as usize) as u16,
        data,
        encoding,
        consciousness_level,
        allostatic_load,
        dominant_harmony,
    }
}

/// Decode an AudioFrame back to f32 PCM stereo samples.
pub fn decode_audio_frame(frame: &AudioFrame) -> Vec<f32> {
    match frame.encoding {
        AudioEncoding::RawF32Stereo => frame
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        AudioEncoding::RawI16Stereo => frame
            .data
            .chunks_exact(2)
            .map(|b| {
                let i = i16::from_le_bytes([b[0], b[1]]);
                i as f32 / 32767.0
            })
            .collect(),
        AudioEncoding::MuLawMono => {
            // Decode mu-law mono → stereo.
            frame
                .data
                .iter()
                .flat_map(|&b| {
                    let s = mulaw_to_f32(b);
                    [s, s] // Mono → stereo
                })
                .collect()
        }
    }
}

/// Truncate audio data to max chunk size.
fn truncate_audio_data(mut data: Vec<u8>) -> Vec<u8> {
    if data.len() > RDP_AUDIO_MAX_CHUNK_BYTES {
        data.truncate(RDP_AUDIO_MAX_CHUNK_BYTES);
    }
    data
}

/// Mu-law encode: f32 [-1,1] → u8.
fn f32_to_mulaw(sample: f32) -> u8 {
    let mu = 255.0f32;
    let s = sample.clamp(-1.0, 1.0);
    let sign = if s >= 0.0 { 0u8 } else { 0x80 };
    let magnitude = s.abs();
    let compressed = ((1.0 + mu * magnitude).ln() / (1.0 + mu).ln() * 127.0) as u8;
    sign | compressed
}

/// Mu-law decode: u8 → f32 [-1,1].
fn mulaw_to_f32(byte: u8) -> f32 {
    let mu = 255.0f32;
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let magnitude = (byte & 0x7F) as f32 / 127.0;
    let expanded = ((1.0 + mu).powf(magnitude) - 1.0) / mu;
    sign * expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_f32_stereo() {
        let chunk = AudioChunk {
            samples: vec![0.5, -0.3, 0.1, 0.7],
            sample_rate: 44100,
            channels: 2,
        };
        let frame = encode_audio_frame(&chunk, 1, 0, 0.8, 0.0, 0);
        assert!(matches!(frame.encoding, AudioEncoding::RawF32Stereo));

        let decoded = decode_audio_frame(&frame);
        assert_eq!(decoded.len(), 4);
        assert!((decoded[0] - 0.5).abs() < 0.001);
        assert!((decoded[1] - (-0.3)).abs() < 0.001);
    }

    #[test]
    fn test_encode_decode_i16_stereo() {
        let chunk = AudioChunk {
            samples: vec![0.5, -0.5],
            sample_rate: 44100,
            channels: 2,
        };
        let frame = encode_audio_frame(&chunk, 1, 0, 0.5, 0.0, 0);
        assert!(matches!(frame.encoding, AudioEncoding::RawI16Stereo));

        let decoded = decode_audio_frame(&frame);
        assert!((decoded[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_encode_decode_mulaw_mono() {
        let chunk = AudioChunk {
            samples: vec![0.3, 0.3], // Stereo pair
            sample_rate: 44100,
            channels: 2,
        };
        let frame = encode_audio_frame(&chunk, 1, 0, 0.2, 0.0, 0);
        assert!(matches!(frame.encoding, AudioEncoding::MuLawMono));

        let decoded = decode_audio_frame(&frame);
        assert_eq!(decoded.len(), 2); // Mono expanded to stereo
        assert!((decoded[0] - 0.3).abs() < 0.05); // Mu-law is lossy
    }

    #[test]
    fn test_mulaw_roundtrip() {
        for s in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let encoded = f32_to_mulaw(s);
            let decoded = mulaw_to_f32(encoded);
            assert!(
                (decoded - s).abs() < 0.05,
                "Mu-law roundtrip failed: {} → {} → {}",
                s,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_test_audio_source() {
        let mut src = TestAudioSource::new(44100, 440.0, 128);
        let chunk = src.capture().unwrap();
        assert_eq!(chunk.samples.len(), 256); // 128 frames × 2 channels
        assert_eq!(chunk.sample_rate, 44100);
        assert!(src.is_active());
    }
}

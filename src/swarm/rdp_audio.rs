// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio capture and playback for Sovereign RDP.
//!
//! Uses the existing `AudioFrame` and `AudioEncoding` types from `rdp_protocol.rs`.
//! Oversized chunks are rejected before encoding; payload bytes are never silently
//! truncated while retaining the original sample count.

use super::rdp_protocol::{AudioEncoding, AudioFrame, RDP_AUDIO_MAX_CHUNK_BYTES};

/// Raw audio chunk from capture source.
///
/// Samples are interleaved by source channel. One- and two-channel input is
/// accepted; the encoded raw-PCM formats are stereo, so mono input is duplicated.
pub struct AudioChunk {
    /// PCM samples (f32, interleaved by channel).
    pub samples: Vec<f32>,
    /// Sample rate in frames per second.
    pub sample_rate: u32,
    /// Number of interleaved source channels (1 or 2).
    pub channels: u8,
}

/// Errors returned when an audio chunk cannot be represented by one RDP frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioFrameEncodeError {
    ZeroSampleRate,
    EmptyChunk,
    UnsupportedChannelCount(u8),
    IncompleteChannelFrame {
        sample_values: usize,
        channels: u8,
    },
    TooManySampleFrames {
        sample_frames: usize,
        max: usize,
    },
    PayloadTooLarge {
        encoded_bytes: usize,
        max_bytes: usize,
    },
}

impl std::fmt::Display for AudioFrameEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "audio sample rate must be non-zero"),
            Self::EmptyChunk => write!(f, "audio chunk must contain at least one sample frame"),
            Self::UnsupportedChannelCount(channels) => {
                write!(f, "unsupported source channel count: {channels}")
            }
            Self::IncompleteChannelFrame {
                sample_values,
                channels,
            } => write!(
                f,
                "{sample_values} sample values do not form complete {channels}-channel frames"
            ),
            Self::TooManySampleFrames { sample_frames, max } => {
                write!(
                    f,
                    "audio chunk has {sample_frames} sample frames; maximum is {max}"
                )
            }
            Self::PayloadTooLarge {
                encoded_bytes,
                max_bytes,
            } => write!(
                f,
                "encoded audio payload is {encoded_bytes} bytes; maximum is {max_bytes}"
            ),
        }
    }
}

impl std::error::Error for AudioFrameEncodeError {}

/// Errors returned when an encoded audio frame is structurally inconsistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioFrameDecodeError {
    ZeroSampleRate,
    ZeroSampleCount,
    PayloadTooLarge {
        payload_bytes: usize,
        max_bytes: usize,
    },
    PayloadLengthMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
}

impl std::fmt::Display for AudioFrameDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "audio frame sample rate must be non-zero"),
            Self::ZeroSampleCount => {
                write!(f, "audio frame must contain at least one sample frame")
            }
            Self::PayloadTooLarge {
                payload_bytes,
                max_bytes,
            } => write!(
                f,
                "audio frame payload is {payload_bytes} bytes; maximum is {max_bytes}"
            ),
            Self::PayloadLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "audio frame payload has {actual_bytes} bytes; expected {expected_bytes}"
            ),
        }
    }
}

impl std::error::Error for AudioFrameDecodeError {}

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

/// Return the source-channel pair for one sample frame, duplicating mono input.
fn stereo_pair(chunk: &AudioChunk, frame_index: usize) -> (f32, f32) {
    let channels = usize::from(chunk.channels);
    if channels == 1 {
        let sample = chunk.samples[frame_index];
        (sample, sample)
    } else {
        let offset = frame_index * 2;
        (chunk.samples[offset], chunk.samples[offset + 1])
    }
}

/// Encode one audio chunk into one bounded AudioFrame.
///
/// `num_samples` is the number of sample frames per encoded channel, not the
/// number of scalar interleaved values. Raw PCM encodings carry stereo; μ-law
/// carries mono. Mono input is duplicated for raw stereo and retained as mono
/// for μ-law. An oversized chunk returns an error before allocating or emitting
/// a partial frame; callers that need longer audio must segment it explicitly.
pub fn encode_audio_frame(
    chunk: &AudioChunk,
    seq: u64,
    timestamp_ms: u64,
    consciousness_level: f32,
    allostatic_load: f32,
    dominant_harmony: u8,
) -> Result<AudioFrame, AudioFrameEncodeError> {
    if chunk.sample_rate == 0 {
        return Err(AudioFrameEncodeError::ZeroSampleRate);
    }
    if chunk.channels != 1 && chunk.channels != 2 {
        return Err(AudioFrameEncodeError::UnsupportedChannelCount(
            chunk.channels,
        ));
    }
    if chunk.samples.is_empty() {
        return Err(AudioFrameEncodeError::EmptyChunk);
    }

    let channels = usize::from(chunk.channels);
    if chunk.samples.len() % channels != 0 {
        return Err(AudioFrameEncodeError::IncompleteChannelFrame {
            sample_values: chunk.samples.len(),
            channels: chunk.channels,
        });
    }

    let sample_frames = chunk.samples.len() / channels;
    if sample_frames > u16::MAX as usize {
        return Err(AudioFrameEncodeError::TooManySampleFrames {
            sample_frames,
            max: u16::MAX as usize,
        });
    }

    let encoding = AudioEncoding::from_phi(consciousness_level);
    let encoded_bytes_per_frame = match encoding {
        AudioEncoding::RawF32Stereo => 2 * std::mem::size_of::<f32>(),
        AudioEncoding::RawI16Stereo => 2 * std::mem::size_of::<i16>(),
        AudioEncoding::MuLawMono => std::mem::size_of::<u8>(),
    };
    let encoded_len = sample_frames.checked_mul(encoded_bytes_per_frame).ok_or(
        AudioFrameEncodeError::PayloadTooLarge {
            encoded_bytes: usize::MAX,
            max_bytes: RDP_AUDIO_MAX_CHUNK_BYTES,
        },
    )?;
    if encoded_len > RDP_AUDIO_MAX_CHUNK_BYTES {
        return Err(AudioFrameEncodeError::PayloadTooLarge {
            encoded_bytes: encoded_len,
            max_bytes: RDP_AUDIO_MAX_CHUNK_BYTES,
        });
    }

    let mut data = Vec::with_capacity(encoded_len);
    match encoding {
        AudioEncoding::RawF32Stereo => {
            for frame_index in 0..sample_frames {
                let (left, right) = stereo_pair(chunk, frame_index);
                data.extend_from_slice(&left.to_le_bytes());
                data.extend_from_slice(&right.to_le_bytes());
            }
        }
        AudioEncoding::RawI16Stereo => {
            for frame_index in 0..sample_frames {
                let (left, right) = stereo_pair(chunk, frame_index);
                let left = (left * 32767.0).clamp(-32768.0, 32767.0) as i16;
                let right = (right * 32767.0).clamp(-32768.0, 32767.0) as i16;
                data.extend_from_slice(&left.to_le_bytes());
                data.extend_from_slice(&right.to_le_bytes());
            }
        }
        AudioEncoding::MuLawMono => {
            for frame_index in 0..sample_frames {
                let mono = if channels == 1 {
                    chunk.samples[frame_index]
                } else {
                    let (left, right) = stereo_pair(chunk, frame_index);
                    (left + right) * 0.5
                };
                data.push(f32_to_mulaw(mono));
            }
        }
    }

    debug_assert_eq!(data.len(), encoded_len);
    Ok(AudioFrame {
        audio_seq: seq,
        timestamp_ms,
        sample_rate: chunk.sample_rate,
        num_samples: sample_frames as u16,
        data,
        encoding,
        consciousness_level,
        allostatic_load,
        dominant_harmony,
    })
}

/// Decode an AudioFrame back to f32 PCM stereo samples.
///
/// The encoded channel count is defined by the encoding variant. Raw PCM is
/// stereo; μ-law is mono and is duplicated into stereo output. Payload length
/// must exactly match `num_samples` and the encoding, so trailing bytes and
/// truncated channel frames are rejected rather than silently discarded.
pub fn decode_audio_frame(frame: &AudioFrame) -> Result<Vec<f32>, AudioFrameDecodeError> {
    if frame.sample_rate == 0 {
        return Err(AudioFrameDecodeError::ZeroSampleRate);
    }

    if frame.num_samples == 0 {
        return Err(AudioFrameDecodeError::ZeroSampleCount);
    }

    let sample_frames = usize::from(frame.num_samples);
    let expected_len = match frame.encoding {
        AudioEncoding::RawF32Stereo => sample_frames * 2 * std::mem::size_of::<f32>(),
        AudioEncoding::RawI16Stereo => sample_frames * 2 * std::mem::size_of::<i16>(),
        AudioEncoding::MuLawMono => sample_frames * std::mem::size_of::<u8>(),
    };
    if expected_len > RDP_AUDIO_MAX_CHUNK_BYTES {
        return Err(AudioFrameDecodeError::PayloadTooLarge {
            payload_bytes: expected_len,
            max_bytes: RDP_AUDIO_MAX_CHUNK_BYTES,
        });
    }
    if frame.data.len() != expected_len {
        return Err(AudioFrameDecodeError::PayloadLengthMismatch {
            expected_bytes: expected_len,
            actual_bytes: frame.data.len(),
        });
    }

    let decoded = match frame.encoding {
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
        AudioEncoding::MuLawMono => frame
            .data
            .iter()
            .flat_map(|&b| {
                let sample = mulaw_to_f32(b);
                [sample, sample]
            })
            .collect(),
    };
    Ok(decoded)
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
        let frame = encode_audio_frame(&chunk, 1, 0, 0.8, 0.0, 0).unwrap();
        assert!(matches!(frame.encoding, AudioEncoding::RawF32Stereo));

        let decoded = decode_audio_frame(&frame).unwrap();
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
        let frame = encode_audio_frame(&chunk, 1, 0, 0.5, 0.0, 0).unwrap();
        assert!(matches!(frame.encoding, AudioEncoding::RawI16Stereo));

        let decoded = decode_audio_frame(&frame).unwrap();
        assert!((decoded[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_encode_decode_mulaw_mono() {
        let chunk = AudioChunk {
            samples: vec![0.3, 0.3], // Stereo pair
            sample_rate: 44100,
            channels: 2,
        };
        let frame = encode_audio_frame(&chunk, 1, 0, 0.2, 0.0, 0).unwrap();
        assert!(matches!(frame.encoding, AudioEncoding::MuLawMono));

        let decoded = decode_audio_frame(&frame).unwrap();
        assert_eq!(decoded.len(), 2); // Mono expanded to stereo
        assert!((decoded[0] - 0.3).abs() < 0.05); // Mu-law is lossy
    }

    #[test]
    fn test_mono_input_is_explicitly_upmixed_for_raw_stereo() {
        let chunk = AudioChunk {
            samples: vec![0.25, -0.25],
            sample_rate: 48_000,
            channels: 1,
        };
        let frame = encode_audio_frame(&chunk, 2, 10, 0.8, 0.0, 0).unwrap();
        assert_eq!(frame.num_samples, 2);
        assert_eq!(
            decode_audio_frame(&frame).unwrap(),
            vec![0.25, 0.25, -0.25, -0.25]
        );
    }

    #[test]
    fn test_mulaw_roundtrip() {
        for sample in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let encoded = f32_to_mulaw(sample);
            let decoded = mulaw_to_f32(encoded);
            assert!(
                (decoded - sample).abs() < 0.05,
                "Mu-law roundtrip failed: {} → {} → {}",
                sample,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_test_audio_source() {
        let mut source = TestAudioSource::new(44100, 440.0, 128);
        let chunk = source.capture().unwrap();
        assert_eq!(chunk.samples.len(), 256); // 128 frames × 2 channels
        assert_eq!(chunk.sample_rate, 44100);
        assert!(source.is_active());
    }

    #[test]
    fn encoded_frame_uses_the_rdp_wire_protocol_type() {
        use super::super::rdp_protocol::RdpFrame;

        let chunk = AudioChunk {
            samples: vec![0.1, -0.1],
            sample_rate: 48_000,
            channels: 2,
        };
        let frame = encode_audio_frame(&chunk, 1, 0, 0.8, 0.0, 0).unwrap();
        let _wire_frame = RdpFrame::Audio(frame);
    }

    fn stereo_chunk(sample_frames: usize) -> AudioChunk {
        AudioChunk {
            samples: vec![0.25; sample_frames * 2],
            sample_rate: 48_000,
            channels: 2,
        }
    }

    #[test]
    fn every_encoding_accepts_exact_payload_limit_and_rejects_one_frame_over() {
        let cases = [
            (0.8, 2_048_usize),  // f32 stereo: 8 bytes per sample frame
            (0.5, 4_096_usize),  // i16 stereo: 4 bytes per sample frame
            (0.2, 16_384_usize), // μ-law mono: 1 byte per sample frame
        ];

        for (phi, max_frames) in cases {
            let exact = encode_audio_frame(&stereo_chunk(max_frames), 7, 100, phi, 0.0, 0)
                .expect("exact payload limit must be accepted");
            assert_eq!(exact.data.len(), RDP_AUDIO_MAX_CHUNK_BYTES);
            assert_eq!(usize::from(exact.num_samples), max_frames);
            assert_eq!(decode_audio_frame(&exact).unwrap().len(), max_frames * 2);

            let oversized = stereo_chunk(max_frames + 1);
            assert!(matches!(
                encode_audio_frame(&oversized, 8, 120, phi, 0.0, 0),
                Err(AudioFrameEncodeError::PayloadTooLarge { .. })
            ));
        }
    }

    #[test]
    fn rejects_empty_invalid_rate_channel_and_incomplete_frames() {
        let empty = AudioChunk {
            samples: Vec::new(),
            sample_rate: 48_000,
            channels: 2,
        };
        assert!(matches!(
            encode_audio_frame(&empty, 1, 0, 0.8, 0.0, 0),
            Err(AudioFrameEncodeError::EmptyChunk)
        ));

        let zero_rate = AudioChunk {
            samples: vec![0.1, 0.2],
            sample_rate: 0,
            channels: 2,
        };
        assert!(matches!(
            encode_audio_frame(&zero_rate, 1, 0, 0.8, 0.0, 0),
            Err(AudioFrameEncodeError::ZeroSampleRate)
        ));

        let unsupported_channels = AudioChunk {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 48_000,
            channels: 3,
        };
        assert!(matches!(
            encode_audio_frame(&unsupported_channels, 1, 0, 0.8, 0.0, 0),
            Err(AudioFrameEncodeError::UnsupportedChannelCount(3))
        ));

        let incomplete_stereo_frame = AudioChunk {
            samples: vec![0.1],
            sample_rate: 48_000,
            channels: 2,
        };
        assert!(matches!(
            encode_audio_frame(&incomplete_stereo_frame, 1, 0, 0.8, 0.0, 0),
            Err(AudioFrameEncodeError::IncompleteChannelFrame { .. })
        ));
    }

    #[test]
    fn rejects_sample_count_over_u16_limit() {
        let chunk = stereo_chunk(u16::MAX as usize + 1);
        assert!(matches!(
            encode_audio_frame(&chunk, 1, 0, 0.2, 0.0, 0),
            Err(AudioFrameEncodeError::TooManySampleFrames { .. })
        ));
    }

    #[test]
    fn decoder_rejects_zero_sample_and_oversized_metadata() {
        let empty = AudioFrame {
            audio_seq: 1,
            timestamp_ms: 0,
            sample_rate: 48_000,
            num_samples: 0,
            data: Vec::new(),
            encoding: AudioEncoding::MuLawMono,
            consciousness_level: 0.2,
            allostatic_load: 0.0,
            dominant_harmony: 0,
        };
        assert!(matches!(
            decode_audio_frame(&empty),
            Err(AudioFrameDecodeError::ZeroSampleCount)
        ));

        let oversized = AudioFrame {
            audio_seq: 2,
            timestamp_ms: 20,
            sample_rate: 48_000,
            num_samples: 2_049,
            data: vec![0; 2_049 * 8],
            encoding: AudioEncoding::RawF32Stereo,
            consciousness_level: 0.8,
            allostatic_load: 0.0,
            dominant_harmony: 0,
        };
        assert!(matches!(
            decode_audio_frame(&oversized),
            Err(AudioFrameDecodeError::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn decoder_rejects_truncated_or_trailing_payload_bytes() {
        let mut frame = AudioFrame {
            audio_seq: 1,
            timestamp_ms: 0,
            sample_rate: 48_000,
            num_samples: 1,
            data: vec![0; 3],
            encoding: AudioEncoding::RawI16Stereo,
            consciousness_level: 0.5,
            allostatic_load: 0.0,
            dominant_harmony: 0,
        };
        assert!(matches!(
            decode_audio_frame(&frame),
            Err(AudioFrameDecodeError::PayloadLengthMismatch {
                expected_bytes: 4,
                actual_bytes: 3
            })
        ));

        frame.data.push(0);
        frame.data.push(0);
        assert!(matches!(
            decode_audio_frame(&frame),
            Err(AudioFrameDecodeError::PayloadLengthMismatch {
                expected_bytes: 4,
                actual_bytes: 5
            })
        ));
    }
}

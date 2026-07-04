// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Streaming Audio Processing
//!
//! Real-time audio processing for live transcription:
//! - Ring buffer for continuous audio
//! - Frame-by-frame processing
//! - Incremental HDC projection
//! - Low-latency phoneme decoding
//!
//! # Architecture
//!
//! ```text
//! Audio In ─► Ring Buffer ─► Frame Extractor ─► LTC Projector ─► Decoder ─► Text Out
//!                  ▲                                   │
//!                  └───────── Feedback Loop ───────────┘
//! ```

use crate::audio::{AudioConfig, AudioFrontend};
use crate::hdc::HV16;
use crate::ltc::{LtcCell, LtcConfig};
use crate::phoneme::PhonemeResonator;
use std::collections::VecDeque;

/// Streaming processor configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Frame duration (seconds)
    pub frame_duration: f32,
    /// Hop duration (seconds) - determines latency
    pub hop_duration: f32,
    /// Ring buffer duration (seconds)
    pub buffer_duration: f32,
    /// Number of mel filterbank channels
    pub n_mels: usize,
    /// Minimum frames before emitting result
    pub min_frames: usize,
    /// Smoothing window for output
    pub smoothing_window: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_duration: 0.025, // 25ms frames
            hop_duration: 0.010,   // 10ms hop -> 10ms latency
            buffer_duration: 2.0,  // 2 second buffer
            n_mels: 40,
            min_frames: 3,
            smoothing_window: 3,
        }
    }
}

impl StreamConfig {
    /// Low-latency configuration (faster but less accurate)
    pub fn low_latency() -> Self {
        Self {
            hop_duration: 0.005, // 5ms hop
            min_frames: 2,
            smoothing_window: 2,
            ..Default::default()
        }
    }

    /// High-quality configuration (more accurate but higher latency)
    pub fn high_quality() -> Self {
        Self {
            frame_duration: 0.030, // 30ms frames
            hop_duration: 0.015,   // 15ms hop
            min_frames: 5,
            smoothing_window: 5,
            ..Default::default()
        }
    }
}

/// Ring buffer for streaming audio
pub struct AudioRingBuffer {
    buffer: VecDeque<f32>,
    capacity: usize,
    write_pos: usize,
}

impl AudioRingBuffer {
    /// Create a new ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            write_pos: 0,
        }
    }

    /// Write samples to the buffer
    pub fn write(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
        self.write_pos += samples.len();
    }

    /// Read a frame of samples at a given position
    pub fn read_frame(&self, start: usize, length: usize) -> Option<Vec<f32>> {
        if start + length > self.buffer.len() {
            return None;
        }

        Some(
            self.buffer
                .iter()
                .skip(start)
                .take(length)
                .copied()
                .collect(),
        )
    }

    /// Number of available samples
    pub fn available(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
    }

    /// Total samples written
    pub fn total_written(&self) -> usize {
        self.write_pos
    }
}

/// Streaming frame result
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Frame index
    pub frame_idx: usize,
    /// Timestamp (seconds)
    pub timestamp: f32,
    /// Acoustic hypervector
    pub hv: HV16,
    /// Salience score
    pub salience: f32,
    /// Best phoneme guess
    pub phoneme: Option<String>,
    /// Confidence score
    pub confidence: f32,
    /// Is this a potential word boundary?
    pub is_boundary: bool,
}

/// Streaming audio processor
pub struct StreamProcessor {
    config: StreamConfig,
    /// Ring buffer for audio
    buffer: AudioRingBuffer,
    /// LTC cell for temporal processing
    ltc: LtcCell,
    /// Previous LTC state (for salience)
    prev_state: Vec<f32>,
    /// Audio frontend for feature extraction (reserved for mel-spectrogram processing)
    #[allow(dead_code)]
    frontend: AudioFrontend,
    /// Phoneme resonator for decoding
    resonator: PhonemeResonator,
    /// Frame counter
    frame_count: usize,
    /// Samples per frame
    samples_per_frame: usize,
    /// Samples per hop
    samples_per_hop: usize,
    /// Next sample position to process
    next_sample: usize,
    /// Recent frames for smoothing
    recent_frames: VecDeque<StreamFrame>,
    /// Salience threshold for boundary detection
    salience_threshold: f32,
}

impl StreamProcessor {
    /// Create a new streaming processor
    pub fn new(config: StreamConfig) -> Self {
        let buffer_samples = (config.sample_rate as f32 * config.buffer_duration) as usize;
        let buffer = AudioRingBuffer::new(buffer_samples);

        let ltc_config = LtcConfig {
            hidden_size: 64,
            ..Default::default()
        };
        let ltc = LtcCell::new(config.n_mels, ltc_config);

        let audio_config = AudioConfig {
            sample_rate: config.sample_rate,
            frame_duration: config.frame_duration,
            hop_duration: config.hop_duration,
            n_mels: config.n_mels,
            ..Default::default()
        };
        let frontend = AudioFrontend::new(audio_config);

        let samples_per_frame = (config.sample_rate as f32 * config.frame_duration) as usize;
        let samples_per_hop = (config.sample_rate as f32 * config.hop_duration) as usize;

        Self {
            config,
            buffer,
            ltc,
            prev_state: vec![0.0; 64],
            frontend,
            resonator: PhonemeResonator::new(),
            frame_count: 0,
            samples_per_frame,
            samples_per_hop,
            next_sample: 0,
            recent_frames: VecDeque::with_capacity(10),
            salience_threshold: 0.5,
        }
    }

    /// Load phoneme prototypes
    pub fn load_prototypes(&mut self, prototypes: &[(String, HV16)]) {
        self.resonator.clear();
        for (label, hv) in prototypes {
            self.resonator.store(label, *hv);
        }
    }

    /// Push audio samples into the processor
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.buffer.write(samples);
    }

    /// Process available frames and return results
    pub fn process(&mut self) -> Vec<StreamFrame> {
        let mut results = Vec::new();

        // Process all available complete frames
        while self.next_sample + self.samples_per_frame <= self.buffer.available() {
            if let Some(frame) = self.process_frame() {
                results.push(frame);
            }
            self.next_sample += self.samples_per_hop;
        }

        results
    }

    /// Process a single frame
    fn process_frame(&mut self) -> Option<StreamFrame> {
        // Get audio frame
        let audio = self
            .buffer
            .read_frame(self.next_sample, self.samples_per_frame)?;

        // Extract features (simplified - would use mel spectrogram)
        let features = self.extract_simple_features(&audio);

        // Process through LTC
        let state = self
            .ltc
            .forward(&features, self.config.hop_duration)
            .to_vec();
        let salience = self.ltc.compute_salience(&self.prev_state);

        // Convert state to hypervector
        let hv = Self::state_to_hv(&state);

        // Decode phoneme
        let (phoneme, confidence) = if !self.resonator.is_empty() {
            let results = self.resonator.query(&hv, 1);
            results
                .into_iter()
                .next()
                .map(|(p, c)| (Some(p), c))
                .unwrap_or((None, 0.0))
        } else {
            (None, 0.0)
        };

        // Detect word boundary
        let is_boundary = salience > self.salience_threshold;

        let frame = StreamFrame {
            frame_idx: self.frame_count,
            timestamp: self.frame_count as f32 * self.config.hop_duration,
            hv,
            salience,
            phoneme,
            confidence,
            is_boundary,
        };

        self.prev_state = state;
        self.frame_count += 1;

        // Add to recent frames for smoothing
        self.recent_frames.push_back(frame.clone());
        if self.recent_frames.len() > self.config.smoothing_window {
            self.recent_frames.pop_front();
        }

        Some(frame)
    }

    /// Simple feature extraction (energy-based)
    fn extract_simple_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = vec![0.0; self.config.n_mels];

        // Energy
        let energy: f32 = audio.iter().map(|x| x * x).sum::<f32>().sqrt();
        features[0] = energy;

        // Zero crossing rate
        let zcr = audio
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count() as f32
            / audio.len() as f32;
        features[1] = zcr;

        // Simple spectral features (band energies)
        let frame_len = audio.len();
        for band in 2..self.config.n_mels.min(10) {
            let start = band * frame_len / 10;
            let end = (band + 1) * frame_len / 10;
            features[band] =
                audio[start..end].iter().map(|x| x.abs()).sum::<f32>() / (end - start) as f32;
        }

        features
    }

    /// Convert state to hypervector
    fn state_to_hv(state: &[f32]) -> HV16 {
        let mut data = Vec::with_capacity(state.len() * 4);
        for &s in state {
            data.extend_from_slice(&s.to_le_bytes());
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(&data);
        let mut output = [0u8; 256];
        hasher.finalize_xof().fill(&mut output);

        HV16::from_bytes(&output)
    }

    /// Get smoothed phoneme output
    pub fn get_smoothed_phoneme(&self) -> Option<String> {
        if self.recent_frames.len() < self.config.min_frames {
            return None;
        }

        // Vote among recent frames
        let mut votes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for frame in &self.recent_frames {
            if let Some(ref phoneme) = frame.phoneme {
                *votes.entry(phoneme.clone()).or_insert(0) += 1;
            }
        }

        votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(phoneme, _)| phoneme)
    }

    /// Get accumulated text from recent processing
    pub fn get_partial_text(&self) -> String {
        let mut phonemes = Vec::new();
        let mut last_phoneme: Option<String> = None;

        for frame in &self.recent_frames {
            if let Some(ref phoneme) = frame.phoneme
                && last_phoneme.as_ref() != Some(phoneme)
            {
                phonemes.push(phoneme.clone());
                last_phoneme = Some(phoneme.clone());
            }
        }

        if phonemes.is_empty() {
            String::new()
        } else {
            format!("/{}/", phonemes.join(" "))
        }
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.ltc.reset();
        self.prev_state = vec![0.0; 64];
        self.frame_count = 0;
        self.next_sample = 0;
        self.recent_frames.clear();
    }

    /// Get current latency in milliseconds
    pub fn latency_ms(&self) -> f32 {
        (self.samples_per_frame as f32 / self.config.sample_rate as f32
            + self.config.min_frames as f32 * self.config.hop_duration)
            * 1000.0
    }

    /// Get current frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get buffer fill level (0.0 - 1.0)
    pub fn buffer_fill(&self) -> f32 {
        self.buffer.available() as f32 / self.buffer.capacity as f32
    }
}

/// Streaming transcription session
pub struct StreamSession {
    processor: StreamProcessor,
    /// Accumulated phonemes
    phonemes: Vec<String>,
    /// Word boundaries
    boundaries: Vec<usize>,
    /// Is session active
    active: bool,
}

impl StreamSession {
    /// Create a new streaming session
    pub fn new(config: StreamConfig) -> Self {
        Self {
            processor: StreamProcessor::new(config),
            phonemes: Vec::new(),
            boundaries: Vec::new(),
            active: false,
        }
    }

    /// Load prototypes
    pub fn load_prototypes(&mut self, prototypes: &[(String, HV16)]) {
        self.processor.load_prototypes(prototypes);
    }

    /// Start the session
    pub fn start(&mut self) {
        self.active = true;
        self.processor.reset();
        self.phonemes.clear();
        self.boundaries.clear();
    }

    /// Stop the session
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Push audio and get incremental results
    pub fn push(&mut self, samples: &[f32]) -> Vec<String> {
        if !self.active {
            return Vec::new();
        }

        self.processor.push_audio(samples);
        let frames = self.processor.process();

        let mut new_phonemes = Vec::new();

        for frame in frames {
            if let Some(phoneme) = frame.phoneme {
                // Collapse repeated phonemes
                if self.phonemes.last() != Some(&phoneme) {
                    self.phonemes.push(phoneme.clone());
                    new_phonemes.push(phoneme);
                }
            }

            if frame.is_boundary {
                self.boundaries.push(self.phonemes.len());
            }
        }

        new_phonemes
    }

    /// Get full transcription so far
    pub fn get_transcription(&self) -> String {
        if self.phonemes.is_empty() {
            return String::new();
        }

        // Insert spaces at word boundaries
        let mut result = String::new();
        let mut last_boundary = 0;

        for &boundary in &self.boundaries {
            if boundary > last_boundary && boundary <= self.phonemes.len() {
                result.push_str(&self.phonemes[last_boundary..boundary].join(" "));
                result.push_str(" | ");
                last_boundary = boundary;
            }
        }

        if last_boundary < self.phonemes.len() {
            result.push_str(&self.phonemes[last_boundary..].join(" "));
        }

        format!("/{result}/")
    }

    /// Is session active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> f32 {
        self.processor.latency_ms()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer() {
        let mut buffer = AudioRingBuffer::new(100);

        buffer.write(&[1.0, 2.0, 3.0]);
        assert_eq!(buffer.available(), 3);

        let frame = buffer.read_frame(0, 3).unwrap();
        assert_eq!(frame, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_stream_processor() {
        let config = StreamConfig::default();
        let mut processor = StreamProcessor::new(config);

        // Generate 100ms of sine wave
        let samples: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        processor.push_audio(&samples);
        let frames = processor.process();

        // Should produce several frames
        assert!(!frames.is_empty());
        assert!(frames.len() >= 5); // 100ms / 10ms hop = 10 frames
    }

    #[test]
    fn test_stream_session() {
        let config = StreamConfig::default();
        let mut session = StreamSession::new(config);

        session.start();
        assert!(session.is_active());

        session.stop();
        assert!(!session.is_active());
    }

    #[test]
    fn test_latency() {
        let low_latency = StreamConfig::low_latency();
        let high_quality = StreamConfig::high_quality();

        let low_processor = StreamProcessor::new(low_latency);
        let high_processor = StreamProcessor::new(high_quality);

        // Low latency should have lower latency than high quality
        assert!(low_processor.latency_ms() < high_processor.latency_ms());
    }
}

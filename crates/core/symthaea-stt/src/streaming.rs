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

use crate::audio::{AudioConfig, AudioFrontend, AudioProjector};
use crate::bootstrap::{PrototypeEncoderMetadata, TrainedPrototypes};
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
    /// Absolute sample index of the oldest retained sample.
    base_sample_pos: u64,
    /// Absolute, exclusive sample index after the newest written sample.
    write_pos: u64,
}

impl AudioRingBuffer {
    /// Create a new ring buffer.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            base_sample_pos: 0,
            write_pos: 0,
        }
    }

    /// Append samples while preserving their absolute position in the stream.
    pub fn write(&mut self, samples: &[f32]) -> Result<(), String> {
        if samples.is_empty() {
            return Ok(());
        }

        let incoming = u64::try_from(samples.len())
            .map_err(|_| "audio chunk length exceeds u64 sample indexing".to_string())?;
        let next_write_pos = self
            .write_pos
            .checked_add(incoming)
            .ok_or_else(|| "absolute audio sample position overflow".to_string())?;

        if samples.len() >= self.capacity {
            // The new chunk alone fills or exceeds the buffer.
            let skipped = u64::try_from(samples.len() - self.capacity)
                .map_err(|_| "audio overflow length exceeds u64".to_string())?;
            self.base_sample_pos = self
                .write_pos
                .checked_add(skipped)
                .ok_or_else(|| "audio buffer base position overflow".to_string())?;
            self.buffer.clear();
            self.buffer
                .extend(samples[samples.len() - self.capacity..].iter().copied());
        } else {
            let overflow = self
                .buffer
                .len()
                .saturating_add(samples.len())
                .saturating_sub(self.capacity);
            for _ in 0..overflow {
                self.buffer.pop_front();
            }
            let overflow = u64::try_from(overflow)
                .map_err(|_| "audio overflow length exceeds u64".to_string())?;
            self.base_sample_pos = self
                .base_sample_pos
                .checked_add(overflow)
                .ok_or_else(|| "audio buffer base position overflow".to_string())?;
            self.buffer.extend(samples.iter().copied());
        }

        self.write_pos = next_write_pos;
        debug_assert_eq!(
            self.base_sample_pos + self.buffer.len() as u64,
            self.write_pos,
            "retained audio range must match absolute write position"
        );
        Ok(())
    }

    /// Read a frame by its absolute source-sample position.
    pub fn read_frame(&self, start_sample: u64, length: usize) -> Option<Vec<f32>> {
        let length_u64 = u64::try_from(length).ok()?;
        let end_sample = start_sample.checked_add(length_u64)?;
        if start_sample < self.base_sample_pos || end_sample > self.write_pos {
            return None;
        }

        let relative_start =
            usize::try_from(start_sample.checked_sub(self.base_sample_pos)?).ok()?;
        let relative_end = relative_start.checked_add(length)?;
        if relative_end > self.buffer.len() {
            return None;
        }

        Some(
            self.buffer
                .iter()
                .skip(relative_start)
                .take(length)
                .copied()
                .collect(),
        )
    }

    /// Absolute index of the oldest sample still retained.
    pub fn base_sample_pos(&self) -> u64 {
        self.base_sample_pos
    }

    /// Number of available samples
    pub fn available(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.base_sample_pos = 0;
        self.write_pos = 0;
    }

    /// Total samples written since the last buffer reset.
    pub fn total_written(&self) -> u64 {
        self.write_pos
    }
}

/// A source-sample interval lost before the processor could consume it.
///
/// The interval is half-open: [lost_start_sample, lost_end_sample).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamDiscontinuity {
    /// Monotonic segment number after this discontinuity.
    pub segment_index: u64,
    /// First sample that the processor still needed but could no longer read.
    pub lost_start_sample: u64,
    /// First retained sample after the lost interval.
    pub lost_end_sample: u64,
}

impl StreamDiscontinuity {
    /// Exact number of source samples skipped by this discontinuity.
    pub fn lost_samples(&self) -> u64 {
        self.lost_end_sample.saturating_sub(self.lost_start_sample)
    }
}

/// Streaming frame result.
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Monotonic output frame index within the current processor session.
    pub frame_idx: usize,
    /// Source start time in seconds, derived from the absolute sample index.
    pub timestamp: f32,
    /// Absolute half-open source sample interval represented by this frame.
    pub source_start_sample: u64,
    pub source_end_sample: u64,
    /// Temporal segment; changes after an unrecoverable source gap.
    pub segment_index: u64,
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
    /// Shared log-mel frontend used by offline training and live inference.
    frontend: AudioFrontend,
    /// Previous mel frame for the shared locality-preserving delta encoder.
    prev_mel_features: Option<Vec<f32>>,
    /// Phoneme resonator for decoding
    resonator: PhonemeResonator,
    /// Frame counter
    frame_count: usize,
    /// Samples per frame
    samples_per_frame: usize,
    /// Samples per hop
    samples_per_hop: usize,
    /// Next absolute source-sample position to process.
    next_sample: u64,
    /// Monotonic temporal segment, advanced after a source gap.
    segment_index: u64,
    /// Total source samples skipped due to buffer overrun.
    dropped_sample_count: u64,
    /// Estimated frame starts skipped with source samples that became unavailable.
    skipped_frame_count: u64,
    /// Most recent discontinuity, consumed with take_discontinuity().
    last_discontinuity: Option<StreamDiscontinuity>,
    /// Recent frames for smoothing
    recent_frames: VecDeque<StreamFrame>,
    /// Salience threshold for boundary detection
    salience_threshold: f32,
}

impl StreamProcessor {
    /// Create a new streaming processor
    pub fn new(config: StreamConfig) -> Self {
        let buffer_samples = ((config.sample_rate as f32 * config.buffer_duration) as usize).max(1);
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

        let samples_per_frame =
            ((config.sample_rate as f32 * config.frame_duration) as usize).max(1);
        let samples_per_hop = ((config.sample_rate as f32 * config.hop_duration) as usize).max(1);

        Self {
            config,
            buffer,
            ltc,
            prev_state: vec![0.0; 64],
            frontend,
            prev_mel_features: None,
            resonator: PhonemeResonator::new(),
            frame_count: 0,
            samples_per_frame,
            samples_per_hop,
            next_sample: 0,
            segment_index: 0,
            dropped_sample_count: 0,
            skipped_frame_count: 0,
            last_discontinuity: None,
            recent_frames: VecDeque::with_capacity(10),
            salience_threshold: 0.5,
        }
    }

    /// Load untyped phoneme vectors.
    ///
    /// This low-level API cannot verify encoder compatibility. Prefer
    /// load_trained_prototypes() for persisted model artifacts.
    pub fn load_prototypes(&mut self, prototypes: &[(String, HV16)]) {
        self.resonator.clear();
        for (label, hv) in prototypes {
            self.resonator.store(label, *hv);
        }
    }

    /// Validate and load an artifact trained in the live raw-frame mel-HDC space.
    pub fn load_trained_prototypes(
        &mut self,
        prototypes: &TrainedPrototypes,
    ) -> Result<(), String> {
        let expected =
            PrototypeEncoderMetadata::direct_mel_frame_v1(self.frontend.config().clone());
        prototypes.validate_encoder(&expected)?;
        self.load_prototypes(&prototypes.as_pairs());
        Ok(())
    }

    /// Push audio samples into the processor.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<(), String> {
        self.buffer.write(samples)
    }

    /// Process available frames and return results.
    ///
    /// A consumer that falls behind the bounded buffer resumes in a new
    /// temporal segment. Missing source samples are recorded exactly, and
    /// temporal/smoothing state is cleared rather than bridged over the gap.
    pub fn process(&mut self) -> Vec<StreamFrame> {
        let mut results = Vec::new();

        let retained_base = self.buffer.base_sample_pos();
        if self.next_sample < retained_base {
            let lost_start_sample = self.next_sample;
            let lost_end_sample = retained_base;
            let lost_samples = lost_end_sample - lost_start_sample;

            self.segment_index = self.segment_index.saturating_add(1);
            self.dropped_sample_count = self.dropped_sample_count.saturating_add(lost_samples);
            let hop_samples = u64::try_from(self.samples_per_hop)
                .unwrap_or(u64::MAX)
                .max(1);
            let skipped_frames =
                lost_samples.saturating_add(hop_samples.saturating_sub(1)) / hop_samples;
            self.skipped_frame_count = self.skipped_frame_count.saturating_add(skipped_frames);
            self.last_discontinuity = Some(StreamDiscontinuity {
                segment_index: self.segment_index,
                lost_start_sample,
                lost_end_sample,
            });

            // Start at the first retained sample and establish a new hop grid.
            self.next_sample = retained_base;
            self.ltc.reset();
            self.prev_state.fill(0.0);
            self.prev_mel_features = None;
            self.recent_frames.clear();
        }

        loop {
            let frame_samples = match u64::try_from(self.samples_per_frame) {
                Ok(value) => value,
                Err(_) => break,
            };
            let Some(frame_end) = self.next_sample.checked_add(frame_samples) else {
                break;
            };
            if frame_end > self.buffer.total_written() {
                break;
            }

            let Some(frame) = self.process_frame() else {
                break;
            };
            results.push(frame);

            let hop_samples = match u64::try_from(self.samples_per_hop) {
                Ok(value) => value,
                Err(_) => break,
            };
            let Some(next_sample) = self.next_sample.checked_add(hop_samples) else {
                break;
            };
            self.next_sample = next_sample;
        }

        results
    }

    /// Take the most recent input discontinuity, if one was detected.
    pub fn take_discontinuity(&mut self) -> Option<StreamDiscontinuity> {
        self.last_discontinuity.take()
    }

    /// Total number of source samples skipped after the processor fell behind.
    pub fn dropped_sample_count(&self) -> u64 {
        self.dropped_sample_count
    }

    /// Number of frame starts skipped because their source samples were lost.
    pub fn skipped_frame_count(&self) -> u64 {
        self.skipped_frame_count
    }

    /// Current temporal segment number.
    pub fn segment_index(&self) -> u64 {
        self.segment_index
    }

    /// Process a single frame
    fn process_frame(&mut self) -> Option<StreamFrame> {
        // Read the exact absolute source interval for this frame.
        let source_start_sample = self.next_sample;
        let source_end_sample =
            source_start_sample.checked_add(u64::try_from(self.samples_per_frame).ok()?)?;
        let audio = self
            .buffer
            .read_frame(source_start_sample, self.samples_per_frame)?;

        // Use the same log-mel frontend and locality-preserving HDC encoder
        // as AudioProjector::project_raw, which is the space used to train
        // phoneme prototypes. One source frame must produce exactly one mel
        // frame; otherwise the configured frame geometry is inconsistent.
        let mut mel_frames = self.frontend.extract_features(&audio);
        if mel_frames.len() != 1 {
            return None;
        }
        let mel_features = mel_frames.pop()?;

        // The HDC vector preserves mel-neighborhood locality and uses the
        // same previous-frame delta encoding as offline training. LTC remains
        // an independent temporal/salience pathway, not a cryptographic hash
        // masquerading as a similarity embedding.
        let hv = AudioProjector::direct_mel_hv(&mel_features, self.prev_mel_features.as_deref());

        // Process log-mel features through LTC for temporal state/salience.
        let state = self
            .ltc
            .forward(&mel_features, self.config.hop_duration)
            .to_vec();
        let salience = self.ltc.compute_salience(&self.prev_state);

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
            timestamp: source_start_sample as f32 / self.config.sample_rate.max(1) as f32,
            source_start_sample,
            source_end_sample,
            segment_index: self.segment_index,
            hv,
            salience,
            phoneme,
            confidence,
            is_boundary,
        };

        self.prev_state = state;
        self.prev_mel_features = Some(mel_features);
        self.frame_count += 1;

        // Add to recent frames for smoothing
        self.recent_frames.push_back(frame.clone());
        if self.recent_frames.len() > self.config.smoothing_window {
            self.recent_frames.pop_front();
        }

        Some(frame)
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
        self.prev_mel_features = None;
        self.frame_count = 0;
        self.next_sample = 0;
        self.segment_index = 0;
        self.dropped_sample_count = 0;
        self.skipped_frame_count = 0;
        self.last_discontinuity = None;
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
    /// Word boundaries (phoneme offsets).
    boundaries: Vec<usize>,
    /// Phoneme offsets where an audio discontinuity starts a new segment.
    segment_boundaries: Vec<usize>,
    /// Most recent discontinuity retained for the session caller.
    last_discontinuity: Option<StreamDiscontinuity>,
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
            segment_boundaries: Vec::new(),
            last_discontinuity: None,
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
        self.segment_boundaries.clear();
        self.last_discontinuity = None;
    }

    /// Stop the session
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Push audio and get incremental results
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<String>, String> {
        if !self.active {
            return Ok(Vec::new());
        }

        self.processor.push_audio(samples)?;
        let frames = self.processor.process();

        if let Some(gap) = self.processor.take_discontinuity() {
            if !self.phonemes.is_empty() {
                self.segment_boundaries.push(self.phonemes.len());
            }
            self.last_discontinuity = Some(gap);
        }

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

        Ok(new_phonemes)
    }

    /// Format one contiguous phoneme segment, preserving word-boundary markers.
    fn format_segment(&self, start: usize, end: usize) -> String {
        let mut pieces = Vec::new();
        let mut cursor = start;

        for &boundary in &self.boundaries {
            if boundary > cursor && boundary <= end {
                pieces.push(self.phonemes[cursor..boundary].join(" "));
                cursor = boundary;
            }
        }

        if cursor < end {
            pieces.push(self.phonemes[cursor..end].join(" "));
        }

        pieces.join(" | ")
    }

    /// Get full transcription so far. Audio gaps are rendered explicitly;
    /// phonemes on opposite sides of a discontinuity are not joined silently.
    pub fn get_transcription(&self) -> String {
        if self.phonemes.is_empty() {
            return String::new();
        }

        let mut segments = Vec::new();
        let mut start = 0usize;

        for &end in &self.segment_boundaries {
            if end > start && end <= self.phonemes.len() {
                let segment = self.format_segment(start, end);
                if !segment.is_empty() {
                    segments.push(segment);
                }
                start = end;
            }
        }

        if start < self.phonemes.len() {
            let segment = self.format_segment(start, self.phonemes.len());
            if !segment.is_empty() {
                segments.push(segment);
            }
        }

        format!("/{}/", segments.join(" [audio gap] "))
    }

    /// Is session active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Take the most recent input discontinuity, if one occurred.
    pub fn take_discontinuity(&mut self) -> Option<StreamDiscontinuity> {
        self.last_discontinuity.take()
    }

    /// Number of source samples skipped by input overruns.
    pub fn dropped_sample_count(&self) -> u64 {
        self.processor.dropped_sample_count()
    }

    /// Number of frame starts skipped because their source samples were lost.
    pub fn skipped_frame_count(&self) -> u64 {
        self.processor.skipped_frame_count()
    }

    /// Current temporal segment number.
    pub fn segment_index(&self) -> u64 {
        self.processor.segment_index()
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

        buffer.write(&[1.0, 2.0, 3.0]).unwrap();
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

        processor.push_audio(&samples).unwrap();
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
    fn ring_buffer_uses_absolute_sample_positions_after_overwrite() {
        let mut buffer = AudioRingBuffer::new(5);
        buffer.write(&[0.0, 1.0, 2.0, 3.0]).unwrap();
        buffer.write(&[4.0, 5.0, 6.0]).unwrap();

        assert_eq!(buffer.base_sample_pos(), 2);
        assert_eq!(buffer.total_written(), 7);
        assert_eq!(buffer.read_frame(0, 2), None);
        assert_eq!(
            buffer.read_frame(2, 5).unwrap(),
            vec![2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn oversized_chunk_retains_its_tail_and_exact_origin() {
        let mut buffer = AudioRingBuffer::new(4);
        let samples: Vec<f32> = (0..9).map(|value| value as f32).collect();
        buffer.write(&samples).unwrap();

        assert_eq!(buffer.base_sample_pos(), 5);
        assert_eq!(buffer.total_written(), 9);
        assert_eq!(buffer.read_frame(5, 4).unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(buffer.read_frame(4, 4), None);
    }

    #[test]
    fn ring_buffer_absolute_reads_match_reference_across_chunk_shapes() {
        // Deterministic reference-model stress test: vary capacities, empty
        // writes, partial writes, exact-capacity writes, and multi-capacity
        // overruns. Every readable absolute interval must match the retained
        // source samples exactly; every unavailable interval must return None.
        for capacity in 1usize..=8 {
            let mut buffer = AudioRingBuffer::new(capacity);
            let mut reference = VecDeque::<f32>::new();
            let mut total_written = 0u64;

            for round in 0usize..48 {
                let chunk_len = (round * 11 + capacity * 7) % (capacity * 3 + 2);
                let samples: Vec<f32> = (0..chunk_len)
                    .map(|offset| (total_written + offset as u64) as f32)
                    .collect();
                buffer.write(&samples).unwrap();
                total_written += chunk_len as u64;

                for sample in samples {
                    if reference.len() == capacity {
                        reference.pop_front();
                    }
                    reference.push_back(sample);
                }

                let retained: Vec<f32> = reference.iter().copied().collect();
                let expected_base = total_written - retained.len() as u64;
                assert_eq!(buffer.base_sample_pos(), expected_base);
                assert_eq!(buffer.total_written(), total_written);
                assert_eq!(buffer.available(), retained.len());

                for start in expected_base.saturating_sub(1)..=total_written {
                    for length in 0..=capacity + 1 {
                        let end = start + length as u64;
                        let expected = if start >= expected_base && end <= total_written {
                            let relative_start = (start - expected_base) as usize;
                            Some(retained[relative_start..relative_start + length].to_vec())
                        } else {
                            None
                        };
                        assert_eq!(
                            buffer.read_frame(start, length),
                            expected,
                            "capacity={capacity}, round={round}, start={start}, length={length}"
                        );
                    }
                }
            }
        }
    }

    fn small_stream_config() -> StreamConfig {
        StreamConfig {
            // Use the production 16 kHz / 40-bin mel profile so the regression
            // exercises a physically plausible frontend, not a 1 kHz test-only
            // configuration whose mel filterbank exceeds Nyquist.
            sample_rate: 16_000,
            frame_duration: 0.025,
            hop_duration: 0.010,
            buffer_duration: 0.1,
            n_mels: 40,
            min_frames: 1,
            smoothing_window: 2,
        }
    }

    #[test]
    fn processor_continues_for_more_than_ten_buffer_durations() {
        let mut processor = StreamProcessor::new(small_stream_config());
        let mut total_frames = 0usize;

        // 1.2 seconds through a 100 ms buffer, processing after each irregular
        // 137-sample chunk. This must not stall as the absolute origin advances.
        let samples: Vec<f32> = (0..19_200)
            .map(|index| (index as f32 * 0.00071).sin())
            .collect();
        for chunk in samples.chunks(137) {
            processor.push_audio(chunk).unwrap();
            total_frames += processor.process().len();
        }

        assert!(total_frames > 100, "frame count must keep advancing");
        assert_eq!(processor.frame_count(), total_frames);
        assert_eq!(processor.dropped_sample_count(), 0);
        assert_eq!(processor.skipped_frame_count(), 0);
        assert_eq!(processor.segment_index(), 0);
    }

    #[test]
    fn overrun_emits_exact_gap_and_starts_a_new_segment() {
        let mut processor = StreamProcessor::new(small_stream_config());

        // Consume three 25 ms windows at a 10 ms hop. Their starts are
        // 0, 160 and 320; after the third window the consumer cursor is 480.
        // Then block it while 4,000 samples arrive into the 1,600-sample buffer.
        let initial: Vec<f32> = (0..720)
            .map(|value| (value as f32 * 0.001).sin())
            .collect();
        processor.push_audio(&initial).unwrap();
        let first = processor.process();
        assert_eq!(first.len(), 3);
        assert_eq!(first[0].source_start_sample, 0);
        assert_eq!(first[2].source_end_sample, 720);
        assert_eq!(processor.frame_count(), 3);

        let overrun: Vec<f32> = (0..4_000)
            .map(|value| (value as f32 * 0.0013).sin())
            .collect();
        processor.push_audio(&overrun).unwrap();
        let resumed = processor.process();

        let gap = processor
            .take_discontinuity()
            .expect("overrun must be observable");
        assert_eq!(gap.segment_index, 1);
        assert_eq!(gap.lost_start_sample, 480);
        assert_eq!(gap.lost_end_sample, 3_120);
        assert_eq!(gap.lost_samples(), 2_640);
        assert_eq!(processor.dropped_sample_count(), 2_640);
        assert_eq!(processor.skipped_frame_count(), 17);
        assert_eq!(processor.segment_index(), 1);

        assert!(!resumed.is_empty());
        assert_eq!(resumed[0].source_start_sample, 3_120);
        assert_eq!(resumed[0].segment_index, 1);
        assert_eq!(resumed[0].frame_idx, 3, "frame IDs remain monotonic");
        assert!(resumed.iter().all(|frame| frame.source_start_sample >= 3_120));
        assert!(resumed.iter().all(|frame| frame.segment_index == 1));
        assert!(resumed.windows(2).all(|pair| {
            pair[0].source_start_sample < pair[1].source_start_sample
                && pair[0].frame_idx < pair[1].frame_idx
        }));
    }

    #[test]
    fn reset_starts_a_new_origin_and_clears_gap_counters() {
        let mut processor = StreamProcessor::new(small_stream_config());
        let initial: Vec<f32> = (0..720)
            .map(|value| (value as f32 * 0.001).sin())
            .collect();
        let overrun: Vec<f32> = (0..4_000)
            .map(|value| (value as f32 * 0.0013).sin())
            .collect();
        processor.push_audio(&initial).unwrap();
        processor.process();
        processor.push_audio(&overrun).unwrap();
        processor.process();
        assert!(processor.take_discontinuity().is_some());

        processor.reset();
        assert_eq!(processor.frame_count(), 0);
        assert_eq!(processor.segment_index(), 0);
        assert_eq!(processor.dropped_sample_count(), 0);
        assert_eq!(processor.skipped_frame_count(), 0);
        assert!(processor.take_discontinuity().is_none());

        let fresh: Vec<f32> = (0..800)
            .map(|value| (value as f32 * 0.002).sin())
            .collect();
        processor.push_audio(&fresh).unwrap();
        let frames = processor.process();
        assert_eq!(frames[0].source_start_sample, 0);
        assert_eq!(frames[0].segment_index, 0);
    }

    fn collect_live_hvs(audio: &[f32], chunk_samples: usize) -> Vec<HV16> {
        let mut processor = StreamProcessor::new(StreamConfig::default());
        let mut result = Vec::new();

        for chunk in audio.chunks(chunk_samples) {
            processor.push_audio(chunk).unwrap();
            result.extend(processor.process().into_iter().map(|frame| frame.hv));
        }

        assert_eq!(processor.dropped_sample_count(), 0);
        assert_eq!(processor.segment_index(), 0);
        result
    }

    fn deterministic_audio_fixture() -> Vec<f32> {
        (0..16_000)
            .map(|index| {
                let t = index as f32 / 16_000.0;
                0.55 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.20 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn stream_processor_accepts_only_exact_raw_frame_encoder_identity() {
        let mut processor = StreamProcessor::new(StreamConfig::default());
        let audio_config = processor.frontend.config().clone();
        let raw = TrainedPrototypes::new_with_encoder(
            crate::bootstrap::BootstrapConfig::default(),
            PrototypeEncoderMetadata::direct_mel_frame_v1(audio_config.clone()),
        );
        assert!(processor.load_trained_prototypes(&raw).is_ok());

        let temporal = TrainedPrototypes::new_with_encoder(
            crate::bootstrap::BootstrapConfig::default(),
            PrototypeEncoderMetadata::direct_mel_temporal_window7_v1(audio_config),
        );
        assert!(processor.load_trained_prototypes(&temporal).is_err());
    }

    #[test]
    fn stream_processor_rejects_raw_prototypes_with_different_hop_cadence() {
        let projector = AudioProjector::default_config();
        let raw = TrainedPrototypes::new_with_encoder(
            crate::bootstrap::BootstrapConfig::default(),
            PrototypeEncoderMetadata::direct_mel_frame_v1(projector.config().clone()),
        );
        let mut low_latency = StreamProcessor::new(StreamConfig::low_latency());

        assert!(low_latency.load_trained_prototypes(&raw).is_err());
    }

    #[test]
    fn live_stream_hvs_match_offline_training_encoder() {
        let audio = deterministic_audio_fixture();
        let mut projector = AudioProjector::default_config();
        let offline = projector.project_raw(&audio);
        let live = collect_live_hvs(&audio, audio.len());

        assert_eq!(live.len(), offline.len());
        assert_eq!(
            live, offline,
            "each live frame must use the same mel-HDC code as project_raw"
        );
    }

    #[test]
    fn live_encoder_is_invariant_to_input_chunk_boundaries() {
        let audio = deterministic_audio_fixture();
        let mut projector = AudioProjector::default_config();
        let offline = projector.project_raw(&audio);

        for chunk_samples in [80usize, 160, 592, 137] {
            let live = collect_live_hvs(&audio, chunk_samples);
            assert_eq!(
                live, offline,
                "live HDC sequence changed for chunk size {chunk_samples}"
            );
        }
    }

    #[test]
    fn stream_session_transcription_marks_audio_gaps() {
        let mut session = StreamSession::new(small_stream_config());
        session.phonemes = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        session.boundaries = vec![1, 3];
        session.segment_boundaries = vec![2];

        assert_eq!(session.get_transcription(), "/a | b [audio gap] c | d/");
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

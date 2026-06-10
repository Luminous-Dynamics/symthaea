// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid-HDC Fusion Scorer - The Neuro-Symbolic Bridge
//!
//! Fuses CfC (Closed-form Continuous-time) neural dynamics with
//! HDC (Hyperdimensional Computing) symbolic binding.
//!
//! # The Architecture
//!
//! ```text
//!                        ┌─────────────────────────────────────┐
//!                        │         LIQUID-HDC FUSION           │
//!                        │                                     │
//!   Raw Audio ──────────►│  ┌─────────┐    ┌──────────────┐   │
//!                        │  │  CfC    │    │  Multi-Scale │   │
//!                        │  │  Bank   │───►│  HDC Scorer  │───┼──► Grammar Score
//!                        │  │ (Ear)   │    │   (Brain)    │   │
//!                        │  └─────────┘    └──────────────┘   │
//!                        │       │                ▲           │
//!                        │       │    ┌───────────┘           │
//!                        │       ▼    │                       │
//!                        │  ┌─────────┴───┐                   │
//!                        │  │ Time-Aware  │                   │
//!                        │  │   Binder    │                   │
//!                        │  │ (Φ⊗Δt⊗I)   │                   │
//!                        │  └─────────────┘                   │
//!                        └─────────────────────────────────────┘
//! ```
//!
//! # Key Innovation
//!
//! The CfC neurons adapt their time constants (τ) based on input tempo.
//! This means:
//! - Fast whale clicks → Fast τ → Quick response
//! - Slow whale songs → Slow τ → Integrated response
//!
//! The temporal binding (Φ⊗Δt⊗I) means:
//! - A "short loud click" ≠ "long quiet click"
//! - Duration and intensity become part of the symbolic representation

use crate::hdc::{BundleAccumulator, HDC_DIM, HV16};
use crate::liquid::{LiquidBank, LiquidEvent, LiquidEventType, LiquidFeatures, TimeAwareBinder};
use std::collections::HashMap;

/// Liquid-HDC Fusion Scorer
///
/// Combines adaptive CfC feature extraction with multi-scale HDC grammar.
pub struct LiquidHDC {
    /// CfC-based liquid feature bank
    liquid_bank: LiquidBank,

    /// Time-aware hypervector binder
    binder: TimeAwareBinder,

    /// Event type basis vectors
    event_basis: HashMap<LiquidEventType, HV16>,

    /// Grammar memory (trained patterns)
    grammar_memory: HV16,

    /// Context state (rolling history)
    context_state: HV16,

    /// Training count
    training_count: usize,

    /// Sample rate
    sample_rate: f32,

    /// Frame size
    frame_size: usize,

    /// Recent events buffer (for sequence scoring)
    recent_events: Vec<LiquidEvent>,

    /// Maximum events to keep
    max_events: usize,
}

impl LiquidHDC {
    /// Create a new Liquid-HDC scorer
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        // Initialize event basis vectors
        let mut event_basis = HashMap::new();
        event_basis.insert(LiquidEventType::Click, HV16::random("liquid_hdc_click"));
        event_basis.insert(LiquidEventType::Whistle, HV16::random("liquid_hdc_whistle"));
        event_basis.insert(LiquidEventType::Burst, HV16::random("liquid_hdc_burst"));
        event_basis.insert(LiquidEventType::Silence, HV16::random("liquid_hdc_silence"));

        Self {
            liquid_bank: LiquidBank::new(sample_rate, frame_size),
            binder: TimeAwareBinder::new(8, 4),
            event_basis,
            grammar_memory: HV16::zero(),
            context_state: HV16::zero(),
            training_count: 0,
            sample_rate,
            frame_size,
            recent_events: Vec::new(),
            max_events: 100,
        }
    }

    /// Create with cetacean-optimized parameters
    pub fn cetacean_optimized() -> Self {
        Self::new(44100.0, 2048) // Standard cetacean recording params
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.liquid_bank.reset();
        self.context_state = HV16::zero();
        self.recent_events.clear();
    }

    /// Get basis HV for an event type
    fn get_event_basis(&self, event_type: &LiquidEventType) -> HV16 {
        self.event_basis
            .get(event_type)
            .copied()
            .unwrap_or_else(|| HV16::random("unknown_event"))
    }

    /// Create time-aware HV for an event
    fn event_to_hv(&self, event: &LiquidEvent) -> HV16 {
        let basis = self.get_event_basis(&event.event_type);
        self.binder.bind(&basis, event.duration, event.intensity)
    }

    /// Process audio frame, returning liquid features and optional event
    pub fn process_frame(&mut self, samples: &[f32]) -> (LiquidFeatures, Option<LiquidEvent>) {
        let features = self.liquid_bank.process_frame(samples);

        // Detect events based on features
        let event = self.detect_event(&features);

        if let Some(ref e) = event {
            // Update context with event
            let event_hv = self.event_to_hv(e);
            let rotated = self.context_state.rotate(1);
            self.context_state = rotated.bind(&event_hv);

            // Store in recent events
            self.recent_events.push(e.clone());
            if self.recent_events.len() > self.max_events {
                self.recent_events.remove(0);
            }
        }

        (features, event)
    }

    /// Detect event from liquid features
    fn detect_event(&self, features: &LiquidFeatures) -> Option<LiquidEvent> {
        // Thresholds
        let click_thresh = 0.6;
        let whistle_thresh = 0.5;
        let burst_thresh = 0.4;
        let silence_thresh = 0.01;

        // Classify
        let event_type = if features.energy < silence_thresh {
            return None; // No event for silence
        } else if features.click > click_thresh
            && features.click > features.whistle
            && features.click > features.burst
        {
            LiquidEventType::Click
        } else if features.whistle > whistle_thresh && features.whistle > features.burst {
            LiquidEventType::Whistle
        } else if features.burst > burst_thresh {
            LiquidEventType::Burst
        } else {
            return None;
        };

        // Create event (duration estimated from frame)
        let frame_duration = self.frame_size as f32 / self.sample_rate;
        let event_hv = self.binder.bind(
            &self.get_event_basis(&event_type),
            features.duration_factor, // Use adaptive τ as duration proxy
            features.energy,
        );

        Some(LiquidEvent {
            event_type,
            start_time: 0.0, // Relative
            duration: frame_duration * features.duration_factor,
            intensity: features.energy,
            hv: event_hv,
        })
    }

    /// Score current context against grammar memory
    pub fn score(&self) -> f32 {
        self.grammar_memory.similarity(&self.context_state)
    }

    /// Train on current context
    pub fn train(&mut self) {
        if self.training_count == 0 {
            self.grammar_memory = self.context_state;
        } else {
            // Weighted bundle
            let mut accum = BundleAccumulator::new();
            accum.add_weighted(&self.grammar_memory, self.training_count as f32);
            accum.add(&self.context_state);
            self.grammar_memory = accum.finalize();
        }
        self.training_count += 1;
    }

    /// Process full audio buffer and return sequence of events
    pub fn process_audio(&mut self, audio: &[f32]) -> Vec<LiquidEvent> {
        self.reset();
        let mut events = Vec::new();

        for chunk in audio.chunks(self.frame_size) {
            if chunk.len() == self.frame_size {
                let (_, event) = self.process_frame(chunk);
                if let Some(e) = event {
                    events.push(e);
                }
            }
        }

        events
    }

    /// Score a sequence of events
    pub fn score_events(&mut self, events: &[LiquidEvent]) -> f32 {
        self.context_state = HV16::zero();

        for event in events {
            let event_hv = self.event_to_hv(event);
            let rotated = self.context_state.rotate(1);
            self.context_state = rotated.bind(&event_hv);
        }

        self.score()
    }

    /// Train on a sequence of events
    pub fn train_events(&mut self, events: &[LiquidEvent]) {
        self.context_state = HV16::zero();

        for event in events {
            let event_hv = self.event_to_hv(event);
            let rotated = self.context_state.rotate(1);
            self.context_state = rotated.bind(&event_hv);
        }

        self.train();
    }

    /// Get statistics
    pub fn stats(&self) -> LiquidHDCStats {
        LiquidHDCStats {
            training_count: self.training_count,
            grammar_density: self.grammar_memory.popcount() as f32 / HDC_DIM as f32,
            recent_events: self.recent_events.len(),
        }
    }
}

/// Statistics for Liquid-HDC scorer
#[derive(Debug)]
pub struct LiquidHDCStats {
    /// Number of training examples
    pub training_count: usize,
    /// Grammar memory density
    pub grammar_density: f32,
    /// Number of recent events in buffer
    pub recent_events: usize,
}

/// High-level Liquid-HDC Pipeline
///
/// Complete pipeline from raw audio to grammar score.
pub struct LiquidHDCPipeline {
    /// Core scorer
    scorer: LiquidHDC,

    /// Minimum event gap for segmentation (seconds)
    min_gap: f32,

    /// History of processed sequences
    sequence_history: Vec<Vec<LiquidEvent>>,
}

impl LiquidHDCPipeline {
    /// Create a new pipeline
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        Self {
            scorer: LiquidHDC::new(sample_rate, frame_size),
            min_gap: 0.5, // 500ms gap = new sequence
            sequence_history: Vec::new(),
        }
    }

    /// Create cetacean-optimized pipeline
    pub fn cetacean() -> Self {
        Self::new(44100.0, 2048)
    }

    /// Process audio file and extract sequences
    pub fn process_and_segment(&mut self, audio: &[f32]) -> Vec<Vec<LiquidEvent>> {
        let events = self.scorer.process_audio(audio);

        // Segment by gaps
        let mut sequences: Vec<Vec<LiquidEvent>> = Vec::new();
        let mut current_seq: Vec<LiquidEvent> = Vec::new();
        let mut last_end = 0.0f32;

        for event in events {
            if !current_seq.is_empty() && event.start_time - last_end > self.min_gap {
                // Gap detected, start new sequence
                sequences.push(std::mem::take(&mut current_seq));
            }

            last_end = event.start_time + event.duration;
            current_seq.push(event);
        }

        if !current_seq.is_empty() {
            sequences.push(current_seq);
        }

        self.sequence_history.extend(sequences.clone());
        sequences
    }

    /// Train on all discovered sequences
    pub fn train_all(&mut self) {
        for seq in &self.sequence_history {
            self.scorer.train_events(seq);
        }
    }

    /// Score a new sequence
    pub fn score_sequence(&mut self, events: &[LiquidEvent]) -> f32 {
        self.scorer.score_events(events)
    }

    /// Discriminate between two sequences
    pub fn discriminate(&mut self, seq1: &[LiquidEvent], seq2: &[LiquidEvent]) -> f32 {
        let score1 = self.scorer.score_events(seq1);
        let score2 = self.scorer.score_events(seq2);
        score1 - score2
    }

    /// Get scorer stats
    pub fn stats(&self) -> LiquidHDCStats {
        self.scorer.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_click_audio(sample_rate: f32, duration: f32) -> Vec<f32> {
        let n_samples = (sample_rate * duration) as usize;
        let mut audio = vec![0.0f32; n_samples];

        // Add impulses (clicks)
        let click_interval = (sample_rate * 0.1) as usize; // Every 100ms
        for i in (0..n_samples).step_by(click_interval) {
            if i < n_samples {
                audio[i] = 1.0;
                if i + 1 < n_samples {
                    audio[i + 1] = -0.5;
                }
            }
        }

        audio
    }

    fn generate_whistle_audio(sample_rate: f32, duration: f32, freq: f32) -> Vec<f32> {
        let n_samples = (sample_rate * duration) as usize;
        (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_liquid_hdc_creation() {
        let scorer = LiquidHDC::cetacean_optimized();
        let stats = scorer.stats();

        assert_eq!(stats.training_count, 0);
        assert_eq!(stats.recent_events, 0);
    }

    #[test]
    fn test_liquid_hdc_process_click() {
        let mut scorer = LiquidHDC::new(44100.0, 1024);

        // Generate click audio
        let audio = generate_click_audio(44100.0, 0.5);

        let events = scorer.process_audio(&audio);

        println!("Click audio: detected {} events", events.len());
        for event in &events {
            println!(
                "  {:?}: dur={:.3}s, int={:.4}",
                event.event_type, event.duration, event.intensity
            );
        }

        // Should detect some events (clicks or bursts)
        // Note: May not classify as "Click" without tuning
    }

    #[test]
    fn test_liquid_hdc_process_whistle() {
        let mut scorer = LiquidHDC::new(44100.0, 1024);

        // Generate whistle audio
        let audio = generate_whistle_audio(44100.0, 0.5, 1000.0);

        let events = scorer.process_audio(&audio);

        println!("Whistle audio: detected {} events", events.len());
        for event in &events {
            println!(
                "  {:?}: dur={:.3}s, int={:.4}",
                event.event_type, event.duration, event.intensity
            );
        }
    }

    #[test]
    fn test_liquid_hdc_training() {
        let mut scorer = LiquidHDC::new(44100.0, 1024);

        // Create synthetic events
        let trained_events: Vec<LiquidEvent> = vec![
            LiquidEvent {
                event_type: LiquidEventType::Click,
                start_time: 0.0,
                duration: 0.05,
                intensity: 0.8,
                hv: HV16::random("e1"),
            },
            LiquidEvent {
                event_type: LiquidEventType::Whistle,
                start_time: 0.1,
                duration: 0.2,
                intensity: 0.6,
                hv: HV16::random("e2"),
            },
            LiquidEvent {
                event_type: LiquidEventType::Click,
                start_time: 0.4,
                duration: 0.05,
                intensity: 0.9,
                hv: HV16::random("e3"),
            },
        ];

        // Train
        for _ in 0..20 {
            scorer.train_events(&trained_events);
        }

        let stats = scorer.stats();
        assert_eq!(stats.training_count, 20);

        // Score trained sequence
        let trained_score = scorer.score_events(&trained_events);

        // Score different sequence
        let different_events: Vec<LiquidEvent> = vec![
            LiquidEvent {
                event_type: LiquidEventType::Burst,
                start_time: 0.0,
                duration: 0.3,
                intensity: 0.4,
                hv: HV16::random("d1"),
            },
            LiquidEvent {
                event_type: LiquidEventType::Burst,
                start_time: 0.4,
                duration: 0.3,
                intensity: 0.5,
                hv: HV16::random("d2"),
            },
        ];

        let different_score = scorer.score_events(&different_events);

        println!("Trained sequence score: {:.4}", trained_score);
        println!("Different sequence score: {:.4}", different_score);
        println!("Discrimination: {:+.4}", trained_score - different_score);

        // Trained should score higher
        assert!(
            trained_score > different_score,
            "Trained should beat different: {:.4} vs {:.4}",
            trained_score,
            different_score
        );
    }

    #[test]
    fn test_time_aware_discrimination() {
        let mut scorer = LiquidHDC::new(44100.0, 1024);

        // Train on SHORT LOUD clicks
        let short_loud: Vec<LiquidEvent> = (0..5)
            .map(|i| LiquidEvent {
                event_type: LiquidEventType::Click,
                start_time: i as f32 * 0.1,
                duration: 0.02, // SHORT
                intensity: 0.9, // LOUD
                hv: HV16::random(&format!("sl{}", i)),
            })
            .collect();

        for _ in 0..30 {
            scorer.train_events(&short_loud);
        }

        // Test: short loud (should match)
        let test_short_loud: Vec<LiquidEvent> = (0..3)
            .map(|i| LiquidEvent {
                event_type: LiquidEventType::Click,
                start_time: i as f32 * 0.1,
                duration: 0.02,
                intensity: 0.85,
                hv: HV16::random(&format!("tsl{}", i)),
            })
            .collect();

        // Test: LONG QUIET (should not match as well)
        let test_long_quiet: Vec<LiquidEvent> = (0..3)
            .map(|i| LiquidEvent {
                event_type: LiquidEventType::Click,
                start_time: i as f32 * 0.5,
                duration: 0.5,  // LONG
                intensity: 0.2, // QUIET
                hv: HV16::random(&format!("tlq{}", i)),
            })
            .collect();

        let score_sl = scorer.score_events(&test_short_loud);
        let score_lq = scorer.score_events(&test_long_quiet);

        println!("Short-Loud clicks: {:.4}", score_sl);
        println!("Long-Quiet clicks: {:.4}", score_lq);
        println!("Time-Aware Discrimination: {:+.4}", score_sl - score_lq);

        // Time-aware binding should distinguish
        // Note: Both are "Click" type, but duration/intensity differ
        assert!(
            score_sl > score_lq,
            "Time-aware should distinguish: {:.4} vs {:.4}",
            score_sl,
            score_lq
        );
    }

    #[test]
    fn test_pipeline() {
        let mut pipeline = LiquidHDCPipeline::cetacean();

        // Generate test audio with click pattern
        let audio = generate_click_audio(44100.0, 1.0);

        // Process and segment
        let sequences = pipeline.process_and_segment(&audio);

        println!("Pipeline detected {} sequences", sequences.len());
        for (i, seq) in sequences.iter().enumerate() {
            println!("  Sequence {}: {} events", i, seq.len());
        }

        // Train on discovered patterns
        pipeline.train_all();

        let stats = pipeline.stats();
        println!(
            "After training: {} patterns, density {:.3}",
            stats.training_count, stats.grammar_density
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Beat Synchronization for Rap Synthesis
//!
//! This module provides beat-aware timing for rap/rhythmic speech synthesis.
//! It synchronizes phoneme timing with musical beats, creating rhythmic flow.
//!
//! ## Key Concepts
//!
//! - **BPM**: Beats per minute (tempo)
//! - **Beat Grid**: Regular time intervals for beat alignment
//! - **Swing**: Off-beat timing for groove feel
//! - **Flow Patterns**: Rhythmic templates for syllable placement

use serde::{Deserialize, Serialize};

/// Beat grid position
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatPosition {
    /// Bar number (0-indexed)
    pub bar: u32,
    /// Beat within bar (0-3 for 4/4 time)
    pub beat: u8,
    /// Sub-beat division (0-3 for 16th notes)
    pub subdivision: u8,
    /// Absolute time in seconds
    pub time: f32,
}

impl BeatPosition {
    /// Create a beat position from absolute time
    pub fn from_time(time: f32, bpm: f32, beats_per_bar: u8) -> Self {
        let beat_duration = 60.0 / bpm;
        let total_beats = time / beat_duration;

        let bar = (total_beats / beats_per_bar as f32).floor() as u32;
        let beat_in_bar = (total_beats % beats_per_bar as f32).floor() as u8;
        let subdivision = ((total_beats.fract() * 4.0).floor() as u8).min(3);

        Self {
            bar,
            beat: beat_in_bar,
            subdivision,
            time,
        }
    }

    /// Convert to absolute time
    pub fn to_time(&self, bpm: f32, beats_per_bar: u8) -> f32 {
        let beat_duration = 60.0 / bpm;
        let total_beats = self.bar as f32 * beats_per_bar as f32
            + self.beat as f32
            + self.subdivision as f32 / 4.0;
        total_beats * beat_duration
    }

    /// Get the next beat position
    pub fn next_subdivision(&self, beats_per_bar: u8) -> Self {
        let mut new = *self;
        new.subdivision += 1;
        if new.subdivision >= 4 {
            new.subdivision = 0;
            new.beat += 1;
            if new.beat >= beats_per_bar {
                new.beat = 0;
                new.bar += 1;
            }
        }
        new
    }

    /// Check if this is on a strong beat (1 or 3 in 4/4)
    pub fn is_strong_beat(&self) -> bool {
        self.subdivision == 0 && (self.beat == 0 || self.beat == 2)
    }

    /// Check if this is on any beat (not subdivision)
    pub fn is_on_beat(&self) -> bool {
        self.subdivision == 0
    }
}

/// Swing feel configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SwingConfig {
    /// Swing amount (0.0 = straight, 0.5 = full triplet swing)
    pub amount: f32,
    /// Which subdivisions to swing (typically off-beats)
    pub swing_odd: bool,
}

impl Default for SwingConfig {
    fn default() -> Self {
        Self {
            amount: 0.0,
            swing_odd: true,
        }
    }
}

impl SwingConfig {
    /// Hip-hop style swing
    pub fn hip_hop() -> Self {
        Self {
            amount: 0.15,
            swing_odd: true,
        }
    }

    /// Jazz/blues triplet swing
    pub fn triplet() -> Self {
        Self {
            amount: 0.33,
            swing_odd: true,
        }
    }
}

/// Flow pattern - defines syllable placement within a bar
#[derive(Debug, Clone)]
pub struct FlowPattern {
    /// Name of the pattern
    pub name: String,
    /// Syllable positions as fractions of a bar (0.0 to 1.0)
    pub positions: Vec<f32>,
    /// Stress levels for each position (0.0 to 1.0)
    pub stresses: Vec<f32>,
    /// Duration multipliers for each position
    pub durations: Vec<f32>,
}

impl FlowPattern {
    /// Create a pattern that hits every 16th note
    pub fn sixteenth_notes() -> Self {
        let positions: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let stresses = vec![
            1.0, 0.3, 0.5, 0.3, // Beat 1
            0.8, 0.3, 0.5, 0.3, // Beat 2
            1.0, 0.3, 0.5, 0.3, // Beat 3
            0.8, 0.3, 0.5, 0.3, // Beat 4
        ];
        let durations = vec![1.0; 16];

        Self {
            name: "16th Notes".to_string(),
            positions,
            stresses,
            durations,
        }
    }

    /// Classic hip-hop flow pattern
    pub fn boom_bap() -> Self {
        Self {
            name: "Boom Bap".to_string(),
            positions: vec![0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
            stresses: vec![1.0, 0.4, 0.7, 0.4, 0.9, 0.4, 0.7, 0.4],
            durations: vec![1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
        }
    }

    /// Triplet flow (like Migos)
    pub fn triplet_flow() -> Self {
        let mut positions = Vec::new();
        let mut stresses = Vec::new();
        let mut durations = Vec::new();

        // 4 beats, 3 notes per beat = 12 positions
        for beat in 0..4 {
            for triplet in 0..3 {
                let pos = beat as f32 / 4.0 + triplet as f32 / 12.0;
                positions.push(pos);
                stresses.push(if triplet == 0 { 1.0 } else { 0.5 });
                durations.push(1.0);
            }
        }

        Self {
            name: "Triplet Flow".to_string(),
            positions,
            stresses,
            durations,
        }
    }

    /// Double-time flow
    pub fn double_time() -> Self {
        let positions: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        let stresses: Vec<f32> = (0..32)
            .map(|i| {
                if i % 8 == 0 {
                    1.0
                } else if i % 4 == 0 {
                    0.7
                } else if i % 2 == 0 {
                    0.5
                } else {
                    0.3
                }
            })
            .collect();
        let durations = vec![0.8; 32];

        Self {
            name: "Double Time".to_string(),
            positions,
            stresses,
            durations,
        }
    }

    /// Laid-back/lazy flow
    pub fn laid_back() -> Self {
        Self {
            name: "Laid Back".to_string(),
            positions: vec![0.05, 0.28, 0.55, 0.78], // Slightly behind the beat
            stresses: vec![0.9, 0.6, 0.8, 0.5],
            durations: vec![1.2, 1.0, 1.2, 1.0],
        }
    }

    /// Get number of syllables this pattern can hold
    pub fn syllable_count(&self) -> usize {
        self.positions.len()
    }
}

/// Beat synchronization engine
#[derive(Debug, Clone)]
pub struct BeatSync {
    /// Tempo in BPM
    pub bpm: f32,
    /// Beats per bar (typically 4)
    pub beats_per_bar: u8,
    /// Swing configuration
    pub swing: SwingConfig,
    /// Current beat position
    current_position: BeatPosition,
}

impl BeatSync {
    /// Create a new beat sync engine
    pub fn new(bpm: f32) -> Self {
        Self {
            bpm,
            beats_per_bar: 4,
            swing: SwingConfig::default(),
            current_position: BeatPosition {
                bar: 0,
                beat: 0,
                subdivision: 0,
                time: 0.0,
            },
        }
    }

    /// Create with hip-hop settings (90 BPM with swing)
    pub fn hip_hop() -> Self {
        Self {
            bpm: 90.0,
            beats_per_bar: 4,
            swing: SwingConfig::hip_hop(),
            current_position: BeatPosition {
                bar: 0,
                beat: 0,
                subdivision: 0,
                time: 0.0,
            },
        }
    }

    /// Create with trap settings (140 BPM, no swing)
    pub fn trap() -> Self {
        Self {
            bpm: 140.0,
            beats_per_bar: 4,
            swing: SwingConfig::default(),
            current_position: BeatPosition {
                bar: 0,
                beat: 0,
                subdivision: 0,
                time: 0.0,
            },
        }
    }

    /// Get duration of one beat in seconds
    pub fn beat_duration(&self) -> f32 {
        60.0 / self.bpm
    }

    /// Get duration of one bar in seconds
    pub fn bar_duration(&self) -> f32 {
        self.beat_duration() * self.beats_per_bar as f32
    }

    /// Get duration of one 16th note in seconds
    pub fn sixteenth_duration(&self) -> f32 {
        self.beat_duration() / 4.0
    }

    /// Apply swing to a time value
    pub fn apply_swing(&self, time: f32) -> f32 {
        if self.swing.amount == 0.0 {
            return time;
        }

        let pos = BeatPosition::from_time(time, self.bpm, self.beats_per_bar);

        // Swing applies only to odd subdivisions (1, 3) when swing_odd is true
        // These are the off-beat 16th notes that get delayed for groove
        if self.swing.swing_odd && pos.subdivision % 2 == 1 {
            let swing_delay = self.sixteenth_duration() * self.swing.amount;
            time + swing_delay
        } else {
            time
        }
    }

    /// Quantize a time to the nearest beat grid position
    pub fn quantize(&self, time: f32, grid_size: u8) -> f32 {
        let grid_duration = self.beat_duration() / grid_size as f32;
        let grid_position = (time / grid_duration).round();
        grid_position * grid_duration
    }

    /// Quantize to 16th note grid
    pub fn quantize_16th(&self, time: f32) -> f32 {
        self.quantize(time, 4)
    }

    /// Quantize to 8th note grid
    pub fn quantize_8th(&self, time: f32) -> f32 {
        self.quantize(time, 2)
    }

    /// Map syllables to beat positions using a flow pattern
    pub fn map_syllables(
        &self,
        syllables: &[String],
        pattern: &FlowPattern,
        start_bar: u32,
    ) -> Vec<SyllableTiming> {
        let mut timings = Vec::new();
        let pattern_len = pattern.positions.len();

        for (i, syllable) in syllables.iter().enumerate() {
            let pattern_idx = i % pattern_len;
            let bar_offset = i / pattern_len;

            let bar_start = (start_bar + bar_offset as u32) as f32 * self.bar_duration();
            let position_in_bar = pattern.positions[pattern_idx] * self.bar_duration();
            let raw_time = bar_start + position_in_bar;

            // Apply swing
            let time = self.apply_swing(raw_time);

            // Calculate duration (time until next syllable or end of bar)
            let next_time = if i + 1 < syllables.len() {
                let next_pattern_idx = (i + 1) % pattern_len;
                let next_bar_offset = (i + 1) / pattern_len;
                let next_bar_start =
                    (start_bar + next_bar_offset as u32) as f32 * self.bar_duration();
                let next_position = pattern.positions[next_pattern_idx] * self.bar_duration();
                self.apply_swing(next_bar_start + next_position)
            } else {
                time + self.sixteenth_duration() * 2.0
            };

            let base_duration = (next_time - time) * pattern.durations[pattern_idx];

            timings.push(SyllableTiming {
                syllable: syllable.clone(),
                start_time: time,
                duration: base_duration.max(0.05), // Minimum duration
                stress: pattern.stresses[pattern_idx],
                beat_position: BeatPosition::from_time(time, self.bpm, self.beats_per_bar),
            });
        }

        timings
    }

    /// Get current position
    pub fn position(&self) -> BeatPosition {
        self.current_position
    }

    /// Advance to a specific time
    pub fn seek(&mut self, time: f32) {
        self.current_position = BeatPosition::from_time(time, self.bpm, self.beats_per_bar);
        self.current_position.time = time;
    }

    /// Advance by delta time
    pub fn advance(&mut self, delta: f32) {
        let new_time = self.current_position.time + delta;
        self.seek(new_time);
    }
}

impl Default for BeatSync {
    fn default() -> Self {
        Self::new(120.0) // Default 120 BPM
    }
}

/// Timing information for a syllable
#[derive(Debug, Clone)]
pub struct SyllableTiming {
    /// The syllable text
    pub syllable: String,
    /// Start time in seconds
    pub start_time: f32,
    /// Duration in seconds
    pub duration: f32,
    /// Stress level (0.0 to 1.0)
    pub stress: f32,
    /// Beat position
    pub beat_position: BeatPosition,
}

/// Breath/pause marker for natural flow
#[derive(Debug, Clone)]
pub struct BreathMarker {
    /// Time position
    pub time: f32,
    /// Duration of breath/pause
    pub duration: f32,
    /// Type of pause
    pub pause_type: PauseType,
}

/// Types of pauses in rap flow
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PauseType {
    /// Quick breath between phrases
    Breath,
    /// Dramatic pause for emphasis
    Dramatic,
    /// Bar transition pause
    BarEnd,
    /// Verse transition
    VerseEnd,
}

impl BreathMarker {
    pub fn breath(time: f32) -> Self {
        Self {
            time,
            duration: 0.15,
            pause_type: PauseType::Breath,
        }
    }

    pub fn dramatic(time: f32) -> Self {
        Self {
            time,
            duration: 0.4,
            pause_type: PauseType::Dramatic,
        }
    }

    pub fn bar_end(time: f32) -> Self {
        Self {
            time,
            duration: 0.25,
            pause_type: PauseType::BarEnd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_position_from_time() {
        let pos = BeatPosition::from_time(2.5, 120.0, 4);
        // At 120 BPM, one beat = 0.5s, one bar = 2.0s
        // 2.5s = 1 bar + 1 beat
        assert_eq!(pos.bar, 1);
        assert_eq!(pos.beat, 1);
    }

    #[test]
    fn test_beat_position_to_time() {
        let pos = BeatPosition {
            bar: 1,
            beat: 2,
            subdivision: 0,
            time: 0.0,
        };
        let time = pos.to_time(120.0, 4);
        // 1 bar (2s) + 2 beats (1s) = 3s
        assert!((time - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_beat_sync_quantize() {
        let sync = BeatSync::new(120.0);
        // At 120 BPM, 16th note = 0.125s
        let quantized = sync.quantize_16th(0.13);
        assert!((quantized - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_flow_pattern_boom_bap() {
        let pattern = FlowPattern::boom_bap();
        assert_eq!(pattern.syllable_count(), 8);
        assert!(pattern.stresses[0] > pattern.stresses[1]); // First beat is strongest
    }

    #[test]
    fn test_syllable_mapping() {
        let sync = BeatSync::new(120.0);
        let pattern = FlowPattern::boom_bap();
        let syllables: Vec<String> = vec!["yo", "check", "the", "mic"]
            .into_iter()
            .map(String::from)
            .collect();

        let timings = sync.map_syllables(&syllables, &pattern, 0);

        assert_eq!(timings.len(), 4);
        assert!(timings[0].start_time < timings[1].start_time);
        assert!(timings[0].stress > timings[1].stress); // First syllable is on strong beat
    }

    #[test]
    fn test_swing() {
        let mut sync = BeatSync::new(120.0);
        sync.swing = SwingConfig::hip_hop();

        let straight = 0.125; // Off-beat 16th note
        let swung = sync.apply_swing(straight);

        assert!(swung > straight); // Swing delays off-beats
    }

    // ============================================================
    // BeatPosition::next_subdivision() tests
    // ============================================================

    #[test]
    fn test_beat_position_next_subdivision_within_beat() {
        // Test advancing within the same beat
        let pos = BeatPosition {
            bar: 0,
            beat: 0,
            subdivision: 0,
            time: 0.0,
        };
        let next = pos.next_subdivision(4);
        assert_eq!(next.bar, 0);
        assert_eq!(next.beat, 0);
        assert_eq!(next.subdivision, 1);
    }

    #[test]
    fn test_beat_position_next_subdivision_beat_boundary() {
        // Test advancing from subdivision 3 to next beat
        let pos = BeatPosition {
            bar: 0,
            beat: 1,
            subdivision: 3,
            time: 0.0,
        };
        let next = pos.next_subdivision(4);
        assert_eq!(next.bar, 0);
        assert_eq!(next.beat, 2);
        assert_eq!(next.subdivision, 0);
    }

    #[test]
    fn test_beat_position_next_subdivision_bar_boundary() {
        // Test advancing from last subdivision of last beat to next bar
        let pos = BeatPosition {
            bar: 0,
            beat: 3,
            subdivision: 3,
            time: 0.0,
        };
        let next = pos.next_subdivision(4);
        assert_eq!(next.bar, 1);
        assert_eq!(next.beat, 0);
        assert_eq!(next.subdivision, 0);
    }

    #[test]
    fn test_beat_position_next_subdivision_3_4_time() {
        // Test with 3/4 time signature (3 beats per bar)
        let pos = BeatPosition {
            bar: 0,
            beat: 2,
            subdivision: 3,
            time: 0.0,
        };
        let next = pos.next_subdivision(3);
        assert_eq!(next.bar, 1);
        assert_eq!(next.beat, 0);
        assert_eq!(next.subdivision, 0);
    }

    #[test]
    fn test_beat_position_next_subdivision_chain() {
        // Test chaining multiple next_subdivision calls
        let mut pos = BeatPosition {
            bar: 0,
            beat: 0,
            subdivision: 0,
            time: 0.0,
        };

        // Advance through an entire bar (4 beats * 4 subdivisions = 16 advances)
        for _ in 0..16 {
            pos = pos.next_subdivision(4);
        }

        assert_eq!(pos.bar, 1);
        assert_eq!(pos.beat, 0);
        assert_eq!(pos.subdivision, 0);
    }

    // ============================================================
    // BeatPosition::is_strong_beat() and is_on_beat() tests
    // ============================================================

    #[test]
    fn test_beat_position_is_strong_beat_beat_one() {
        // Beat 1 (beat 0) is a strong beat
        let pos = BeatPosition {
            bar: 0,
            beat: 0,
            subdivision: 0,
            time: 0.0,
        };
        assert!(pos.is_strong_beat());
        assert!(pos.is_on_beat());
    }

    #[test]
    fn test_beat_position_is_strong_beat_beat_three() {
        // Beat 3 (beat 2) is a strong beat in 4/4
        let pos = BeatPosition {
            bar: 0,
            beat: 2,
            subdivision: 0,
            time: 0.0,
        };
        assert!(pos.is_strong_beat());
        assert!(pos.is_on_beat());
    }

    #[test]
    fn test_beat_position_weak_beats() {
        // Beats 2 and 4 (beats 1 and 3) are weak beats
        let pos_beat2 = BeatPosition {
            bar: 0,
            beat: 1,
            subdivision: 0,
            time: 0.0,
        };
        let pos_beat4 = BeatPosition {
            bar: 0,
            beat: 3,
            subdivision: 0,
            time: 0.0,
        };

        assert!(!pos_beat2.is_strong_beat());
        assert!(pos_beat2.is_on_beat());
        assert!(!pos_beat4.is_strong_beat());
        assert!(pos_beat4.is_on_beat());
    }

    #[test]
    fn test_beat_position_is_on_beat_subdivisions() {
        // Subdivisions are not on the beat
        for sub in 1..4 {
            let pos = BeatPosition {
                bar: 0,
                beat: 0,
                subdivision: sub,
                time: 0.0,
            };
            assert!(!pos.is_on_beat());
            assert!(!pos.is_strong_beat());
        }
    }

    #[test]
    fn test_beat_position_strong_beat_with_subdivision() {
        // Strong beat position but with subdivision != 0 is not strong
        let pos = BeatPosition {
            bar: 0,
            beat: 0,
            subdivision: 2,
            time: 0.0,
        };
        assert!(!pos.is_strong_beat());
    }

    // ============================================================
    // FlowPattern variant tests
    // ============================================================

    #[test]
    fn test_flow_pattern_boom_bap_structure() {
        let pattern = FlowPattern::boom_bap();

        // Boom bap has 8 positions (8th notes)
        assert_eq!(pattern.positions.len(), 8);
        assert_eq!(pattern.stresses.len(), 8);
        assert_eq!(pattern.durations.len(), 8);

        // Positions should be evenly spaced 8th notes
        for (i, &pos) in pattern.positions.iter().enumerate() {
            let expected = i as f32 * 0.125;
            assert!((pos - expected).abs() < 0.001, "Position {} mismatch", i);
        }

        // First beat (0.0) should have highest stress
        assert_eq!(pattern.stresses[0], 1.0);
        // Downbeat of beat 3 should have high stress
        assert_eq!(pattern.stresses[4], 0.9);
    }

    #[test]
    fn test_flow_pattern_triplet_flow_structure() {
        let pattern = FlowPattern::triplet_flow();

        // Triplet flow: 4 beats * 3 triplets = 12 positions
        assert_eq!(pattern.positions.len(), 12);
        assert_eq!(pattern.stresses.len(), 12);
        assert_eq!(pattern.durations.len(), 12);

        // First note of each beat triplet should have stress 1.0
        for beat in 0..4 {
            let idx = beat * 3;
            assert_eq!(
                pattern.stresses[idx], 1.0,
                "Beat {} first triplet stress",
                beat
            );
        }

        // Other triplet notes should have stress 0.5
        for beat in 0..4 {
            assert_eq!(pattern.stresses[beat * 3 + 1], 0.5);
            assert_eq!(pattern.stresses[beat * 3 + 2], 0.5);
        }
    }

    #[test]
    fn test_flow_pattern_triplet_flow_positions() {
        let pattern = FlowPattern::triplet_flow();

        // Check triplet timing math
        // Each beat is 0.25 of a bar, triplets divide that into 3
        let tolerance = 0.001;

        // First beat triplets: 0.0, 1/12, 2/12
        assert!((pattern.positions[0] - 0.0).abs() < tolerance);
        assert!((pattern.positions[1] - 1.0 / 12.0).abs() < tolerance);
        assert!((pattern.positions[2] - 2.0 / 12.0).abs() < tolerance);

        // Second beat starts at 0.25
        assert!((pattern.positions[3] - 0.25).abs() < tolerance);
    }

    #[test]
    fn test_flow_pattern_double_time_structure() {
        let pattern = FlowPattern::double_time();

        // Double time has 32 positions (32nd notes)
        assert_eq!(pattern.positions.len(), 32);
        assert_eq!(pattern.stresses.len(), 32);
        assert_eq!(pattern.durations.len(), 32);

        // All durations should be 0.8 (shorter for faster delivery)
        for &dur in &pattern.durations {
            assert!((dur - 0.8).abs() < 0.001);
        }
    }

    #[test]
    fn test_flow_pattern_double_time_stress_hierarchy() {
        let pattern = FlowPattern::double_time();

        // Every 8th position (downbeats) should have stress 1.0
        assert_eq!(pattern.stresses[0], 1.0);
        assert_eq!(pattern.stresses[8], 1.0);
        assert_eq!(pattern.stresses[16], 1.0);
        assert_eq!(pattern.stresses[24], 1.0);

        // Every 4th position (but not 8th) should have stress 0.7
        assert_eq!(pattern.stresses[4], 0.7);
        assert_eq!(pattern.stresses[12], 0.7);

        // Every 2nd position (but not 4th) should have stress 0.5
        assert_eq!(pattern.stresses[2], 0.5);
        assert_eq!(pattern.stresses[6], 0.5);

        // Odd positions should have stress 0.3
        assert_eq!(pattern.stresses[1], 0.3);
        assert_eq!(pattern.stresses[3], 0.3);
    }

    #[test]
    fn test_flow_pattern_laid_back_behind_beat() {
        let pattern = FlowPattern::laid_back();

        // Laid back has 4 positions, all slightly behind the grid
        assert_eq!(pattern.positions.len(), 4);

        // Position 0 should be slightly after 0.0
        assert!(pattern.positions[0] > 0.0);
        assert!(pattern.positions[0] < 0.125);

        // Position 1 should be slightly after 0.25
        assert!(pattern.positions[1] > 0.25);
        assert!(pattern.positions[1] < 0.375);

        // Position 2 should be slightly after 0.5
        assert!(pattern.positions[2] > 0.5);
        assert!(pattern.positions[2] < 0.625);

        // Position 3 should be slightly after 0.75
        assert!(pattern.positions[3] > 0.75);
        assert!(pattern.positions[3] < 0.875);
    }

    #[test]
    fn test_flow_pattern_laid_back_durations() {
        let pattern = FlowPattern::laid_back();

        // Laid back has longer durations for a lazy feel
        assert!(pattern.durations[0] > 1.0); // 1.2
        assert!(pattern.durations[2] > 1.0); // 1.2
    }

    #[test]
    fn test_flow_pattern_sixteenth_notes() {
        let pattern = FlowPattern::sixteenth_notes();

        // 16th notes pattern has 16 positions
        assert_eq!(pattern.positions.len(), 16);

        // Positions should be evenly spaced
        for (i, &pos) in pattern.positions.iter().enumerate() {
            let expected = i as f32 / 16.0;
            assert!((pos - expected).abs() < 0.001);
        }
    }

    // ============================================================
    // SwingConfig tests
    // ============================================================

    #[test]
    fn test_swing_config_zero_swing() {
        let sync = BeatSync::new(120.0);
        // Default swing is 0.0

        // With zero swing, times should be unchanged
        let times = [0.0, 0.125, 0.25, 0.375, 0.5];
        for &time in &times {
            let swung = sync.apply_swing(time);
            assert!(
                (swung - time).abs() < 0.001,
                "Time {} should be unchanged with zero swing",
                time
            );
        }
    }

    #[test]
    fn test_swing_config_full_triplet_swing() {
        let mut sync = BeatSync::new(120.0);
        sync.swing = SwingConfig {
            amount: 0.5, // Full triplet swing
            swing_odd: true,
        };

        // At 120 BPM, 16th note = 0.125s
        // Swing delay = 0.125 * 0.5 = 0.0625s

        // Off-beat (subdivision 1) should be delayed
        let off_beat_time = 0.125; // Second 16th note
        let swung = sync.apply_swing(off_beat_time);
        let expected_delay = 0.125 * 0.5;
        assert!((swung - (off_beat_time + expected_delay)).abs() < 0.001);
    }

    #[test]
    fn test_swing_config_only_affects_odd_subdivisions() {
        let mut sync = BeatSync::new(120.0);
        sync.swing = SwingConfig::hip_hop();

        // Even subdivisions (on-beats) should not be swung
        let on_beat_times = [0.0, 0.25, 0.5, 0.75]; // Subdivisions 0 and 2
        for &time in &on_beat_times {
            let swung = sync.apply_swing(time);
            assert!(
                (swung - time).abs() < 0.001,
                "On-beat time {} should not be swung",
                time
            );
        }
    }

    #[test]
    fn test_swing_config_hip_hop_preset() {
        let config = SwingConfig::hip_hop();
        assert!((config.amount - 0.15).abs() < 0.001);
        assert!(config.swing_odd);
    }

    #[test]
    fn test_swing_config_triplet_preset() {
        let config = SwingConfig::triplet();
        assert!((config.amount - 0.33).abs() < 0.001);
        assert!(config.swing_odd);
    }

    #[test]
    fn test_swing_interaction_with_flow_pattern() {
        let mut sync = BeatSync::new(120.0);
        sync.swing = SwingConfig::hip_hop();

        let pattern = FlowPattern::boom_bap();
        let syllables: Vec<String> = vec![
            "one", "two", "three", "four", "five", "six", "seven", "eight",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let timings = sync.map_syllables(&syllables, &pattern, 0);

        // All syllables should be in ascending time order
        assert!(timings[1].start_time > timings[0].start_time);

        // Boom-bap positions at 120 BPM all land on even 16th-note
        // subdivisions, so hip-hop swing (which delays odd subdivisions)
        // should not alter their timing.
        let bar_duration = sync.bar_duration();
        for (i, timing) in timings.iter().enumerate() {
            let pattern_idx = i % pattern.positions.len();
            let bar_offset = (i / pattern.positions.len()) as f32;
            let straight =
                bar_offset * bar_duration + pattern.positions[pattern_idx] * bar_duration;
            assert!(
                (timing.start_time - straight).abs() < 1e-4,
                "Position {} should not be swung, got {} vs straight {}",
                i,
                timing.start_time,
                straight,
            );
        }
    }

    // ============================================================
    // BeatSync timing tests
    // ============================================================

    #[test]
    fn test_beat_sync_tempo_accuracy_120bpm() {
        let sync = BeatSync::new(120.0);

        // At 120 BPM: beat = 0.5s, bar = 2.0s, 16th = 0.125s
        assert!((sync.beat_duration() - 0.5).abs() < 0.001);
        assert!((sync.bar_duration() - 2.0).abs() < 0.001);
        assert!((sync.sixteenth_duration() - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_beat_sync_tempo_accuracy_90bpm() {
        let sync = BeatSync::new(90.0);

        // At 90 BPM: beat = 0.667s, bar = 2.667s, 16th = 0.167s
        let expected_beat = 60.0 / 90.0;
        assert!((sync.beat_duration() - expected_beat).abs() < 0.001);
        assert!((sync.bar_duration() - expected_beat * 4.0).abs() < 0.001);
        assert!((sync.sixteenth_duration() - expected_beat / 4.0).abs() < 0.001);
    }

    #[test]
    fn test_beat_sync_tempo_accuracy_140bpm() {
        let sync = BeatSync::new(140.0);

        // At 140 BPM: beat = 0.4286s
        let expected_beat = 60.0 / 140.0;
        assert!((sync.beat_duration() - expected_beat).abs() < 0.001);
    }

    #[test]
    fn test_beat_sync_seek_and_position() {
        let mut sync = BeatSync::new(120.0);

        // Seek to 2.5 seconds
        sync.seek(2.5);
        let pos = sync.position();

        // At 120 BPM, 2.5s = 1 bar + 1 beat
        assert_eq!(pos.bar, 1);
        assert_eq!(pos.beat, 1);
        assert!((pos.time - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_beat_sync_advance() {
        let mut sync = BeatSync::new(120.0);

        // Advance by one beat (0.5s at 120 BPM)
        sync.advance(0.5);
        let pos = sync.position();

        assert_eq!(pos.bar, 0);
        assert_eq!(pos.beat, 1);
        assert!((pos.time - 0.5).abs() < 0.001);

        // Advance again
        sync.advance(0.5);
        let pos = sync.position();
        assert_eq!(pos.beat, 2);
    }

    #[test]
    fn test_beat_sync_quantize_8th_notes() {
        let sync = BeatSync::new(120.0);

        // 8th note at 120 BPM = 0.25s, midpoint is 0.125s
        // Values below midpoint round down, values above round up
        let quantized = sync.quantize_8th(0.1);
        assert!((quantized - 0.0).abs() < 0.001); // 0.1 < 0.125, rounds down to 0

        let quantized = sync.quantize_8th(0.2);
        assert!((quantized - 0.25).abs() < 0.001); // 0.2 > 0.125, rounds up to 0.25
    }

    #[test]
    fn test_beat_sync_hip_hop_preset() {
        let sync = BeatSync::hip_hop();

        assert!((sync.bpm - 90.0).abs() < 0.001);
        assert_eq!(sync.beats_per_bar, 4);
        assert!((sync.swing.amount - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_beat_sync_trap_preset() {
        let sync = BeatSync::trap();

        assert!((sync.bpm - 140.0).abs() < 0.001);
        assert_eq!(sync.beats_per_bar, 4);
        assert!((sync.swing.amount - 0.0).abs() < 0.001); // No swing in trap
    }

    #[test]
    fn test_beat_sync_default() {
        let sync = BeatSync::default();

        assert!((sync.bpm - 120.0).abs() < 0.001);
        assert_eq!(sync.beats_per_bar, 4);
    }

    #[test]
    fn test_syllable_mapping_across_bars() {
        let sync = BeatSync::new(120.0);
        let pattern = FlowPattern::laid_back(); // 4 syllables per bar

        // 8 syllables should span 2 bars
        let syllables: Vec<String> = vec![
            "one", "two", "three", "four", "five", "six", "seven", "eight",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let timings = sync.map_syllables(&syllables, &pattern, 0);

        // Syllables 5-8 should be in bar 1
        assert_eq!(timings[4].beat_position.bar, 1);
        assert_eq!(timings[5].beat_position.bar, 1);
        assert_eq!(timings[6].beat_position.bar, 1);
        assert_eq!(timings[7].beat_position.bar, 1);
    }

    #[test]
    fn test_syllable_mapping_minimum_duration() {
        let sync = BeatSync::new(120.0);
        let pattern = FlowPattern::double_time(); // Fast pattern with short durations

        let syllables: Vec<String> = vec!["a", "b"].into_iter().map(String::from).collect();

        let timings = sync.map_syllables(&syllables, &pattern, 0);

        // All durations should be at least 0.05s
        for timing in &timings {
            assert!(
                timing.duration >= 0.05,
                "Duration {} is below minimum",
                timing.duration
            );
        }
    }

    // ============================================================
    // BreathMarker and PauseType tests
    // ============================================================

    #[test]
    fn test_breath_marker_breath() {
        let marker = BreathMarker::breath(1.5);

        assert!((marker.time - 1.5).abs() < 0.001);
        assert!((marker.duration - 0.15).abs() < 0.001);
        assert_eq!(marker.pause_type, PauseType::Breath);
    }

    #[test]
    fn test_breath_marker_dramatic() {
        let marker = BreathMarker::dramatic(2.0);

        assert!((marker.time - 2.0).abs() < 0.001);
        assert!((marker.duration - 0.4).abs() < 0.001);
        assert_eq!(marker.pause_type, PauseType::Dramatic);
    }

    #[test]
    fn test_breath_marker_bar_end() {
        let marker = BreathMarker::bar_end(4.0);

        assert!((marker.time - 4.0).abs() < 0.001);
        assert!((marker.duration - 0.25).abs() < 0.001);
        assert_eq!(marker.pause_type, PauseType::BarEnd);
    }

    #[test]
    fn test_pause_type_equality() {
        assert_eq!(PauseType::Breath, PauseType::Breath);
        assert_ne!(PauseType::Breath, PauseType::Dramatic);
        assert_ne!(PauseType::BarEnd, PauseType::VerseEnd);
    }
}

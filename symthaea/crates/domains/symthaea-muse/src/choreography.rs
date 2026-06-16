// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dance choreography for a 21-joint humanoid body.
//!
//! Maps musical note sequences into joint-target keyframes using Laban
//! Movement Analysis effort qualities. Frequency determines body region
//! (low pitch = legs, high pitch = arms), and the cognitive state selects
//! a dance style that shapes amplitude and contour.

use serde::{Deserialize, Serialize};

use crate::{MusicalState, Note};

/// Number of joints in the humanoid skeleton.
pub const NUM_JOINTS: usize = 21;

/// Frequencies below this threshold primarily drive leg movement.
pub const FREQ_LOW_THRESHOLD: f32 = 250.0;

/// Frequencies above this threshold primarily drive arm movement.
pub const FREQ_HIGH_THRESHOLD: f32 = 400.0;

/// Laban effort weight quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementWeight {
    /// Delicate, buoyant movement.
    Light,
    /// Powerful, impactful movement.
    Strong,
    /// Slow, continuous movement.
    Sustained,
    /// Sharp, sudden movement.
    Quick,
}

/// Overall style of the dance phrase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DanceStyle {
    /// Slow, inward, meditative movement.
    Contemplative,
    /// Wide, dynamic, emotional movement.
    Expressive,
    /// Percussive, beat-locked movement.
    Rhythmic,
    /// Upward-reaching, progressive movement.
    Ascending,
}

/// A single keyframe of joint targets at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DanceKeyframe {
    /// Time in seconds from the start of the phrase.
    pub time: f32,
    /// Target position for each of the 21 joints, in [-1, 1].
    pub joint_targets: [f32; NUM_JOINTS],
    /// Velocity profile [0, 1] for interpolation speed.
    pub velocity_profile: f32,
    /// Effort weight quality for this keyframe.
    pub weight: MovementWeight,
}

/// A complete dance phrase: a sequence of keyframes with style metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DancePhrase {
    /// Ordered keyframes.
    pub keyframes: Vec<DanceKeyframe>,
    /// Dance style of this phrase.
    pub style: DanceStyle,
    /// Beat alignment times (seconds) for synchronization.
    pub beat_alignment: Vec<f32>,
    /// Total duration of the phrase in seconds.
    pub duration_secs: f32,
}

/// Select a dance style from cognitive state.
fn select_style(state: &MusicalState) -> DanceStyle {
    let stillness = state.harmony_activations[7];
    let progression = state.harmony_activations[6];
    let play = state.harmony_activations[3];

    if stillness > 0.5 {
        DanceStyle::Contemplative
    } else if progression > 0.5 {
        DanceStyle::Ascending
    } else if play > 0.5 || state.arousal > 0.6 {
        DanceStyle::Rhythmic
    } else {
        DanceStyle::Expressive
    }
}

/// Amplitude multiplier for each dance style.
fn style_amplitude(style: DanceStyle) -> f32 {
    match style {
        DanceStyle::Contemplative => 0.3,
        DanceStyle::Expressive => 0.6,
        DanceStyle::Rhythmic => 0.8,
        DanceStyle::Ascending => 0.5,
    }
}

/// Weight quality from velocity.
fn weight_from_velocity(velocity: f32) -> MovementWeight {
    if velocity > 0.7 {
        MovementWeight::Strong
    } else if velocity > 0.4 {
        MovementWeight::Quick
    } else if velocity > 0.2 {
        MovementWeight::Sustained
    } else {
        MovementWeight::Light
    }
}

/// Generate a dance phrase from a note sequence and cognitive state.
///
/// Each note onset produces a keyframe. Frequency determines body region:
/// - Below 250 Hz: primarily legs (joint indices 3-14)
/// - Above 400 Hz: primarily arms (joint indices 15-20)
/// - Between: full body
///
/// Lead arm alternates by note index (not frequency).
pub fn choreograph(notes: &[Note], state: &MusicalState, duration_secs: f32) -> DancePhrase {
    let style = select_style(state);
    let amp = style_amplitude(style);
    let mut keyframes = Vec::with_capacity(notes.len());
    let mut beat_alignment = Vec::with_capacity(notes.len());

    for (note_idx, note) in notes.iter().enumerate() {
        beat_alignment.push(note.start_time);

        let mut targets = [0.0f32; NUM_JOINTS];
        let base = note.velocity * amp;

        // Determine which side leads (alternating by note index)
        let lead_sign: f32 = if note_idx % 2 == 0 { 1.0 } else { -1.0 };

        if note.frequency < FREQ_LOW_THRESHOLD {
            // Low pitch: legs (indices 3-14)
            for j in 3..=14 {
                let offset = if j % 2 == 0 { lead_sign } else { -lead_sign };
                targets[j] = base * offset;
            }
        } else if note.frequency > FREQ_HIGH_THRESHOLD {
            // High pitch: arms (indices 15-20)
            for j in 15..=20 {
                let offset = if j % 2 == 0 { lead_sign } else { -lead_sign };
                targets[j] = base * offset;
            }
        } else {
            // Mid-range: full body
            for j in 0..NUM_JOINTS {
                let offset = if j % 2 == 0 { lead_sign } else { -lead_sign };
                targets[j] = base * offset * 0.5;
            }
        }

        // Ascending style: progressive arm lift
        if style == DanceStyle::Ascending && duration_secs > 0.0 {
            let progress = note.start_time / duration_secs;
            for j in 15..=20 {
                targets[j] += 0.1 * progress;
            }
        }

        // Clamp all targets to [-1, 1]
        for t in targets.iter_mut() {
            *t = t.clamp(-1.0, 1.0);
        }

        keyframes.push(DanceKeyframe {
            time: note.start_time,
            joint_targets: targets,
            velocity_profile: note.velocity,
            weight: weight_from_velocity(note.velocity),
        });
    }

    DancePhrase {
        keyframes,
        style,
        beat_alignment,
        duration_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_notes() -> Vec<Note> {
        vec![
            Note {
                frequency: 100.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 300.0,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.6,
            },
            Note {
                frequency: 500.0,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 200.0,
                start_time: 1.5,
                duration: 0.5,
                velocity: 0.5,
            },
        ]
    }

    #[test]
    fn keyframes_per_note() {
        let state = MusicalState::default();
        let phrase = choreograph(&test_notes(), &state, 2.0);
        assert_eq!(phrase.keyframes.len(), 4);
    }

    #[test]
    fn beat_alignment_matches_notes() {
        let state = MusicalState::default();
        let notes = test_notes();
        let phrase = choreograph(&notes, &state, 2.0);
        assert_eq!(phrase.beat_alignment.len(), notes.len());
        for (align, note) in phrase.beat_alignment.iter().zip(notes.iter()) {
            assert!((align - note.start_time).abs() < 1e-6);
        }
    }

    #[test]
    fn joint_bounds() {
        let state = MusicalState {
            arousal: 0.9,
            ..Default::default()
        };
        let phrase = choreograph(&test_notes(), &state, 2.0);
        for kf in &phrase.keyframes {
            for &t in &kf.joint_targets {
                assert!(t >= -1.0 && t <= 1.0, "joint target {t} out of bounds");
            }
        }
    }

    #[test]
    fn low_pitch_drives_legs() {
        let notes = vec![Note {
            frequency: 100.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        }];
        let state = MusicalState::default();
        let phrase = choreograph(&notes, &state, 1.0);
        let kf = &phrase.keyframes[0];
        // Legs (3-14) should have non-zero targets
        let leg_energy: f32 = kf.joint_targets[3..=14].iter().map(|t| t.abs()).sum();
        // Arms (15-20) should be near zero
        let arm_energy: f32 = kf.joint_targets[15..=20].iter().map(|t| t.abs()).sum();
        assert!(leg_energy > 0.0, "legs should move on low pitch");
        assert!(arm_energy < 1e-6, "arms should be still on low pitch");
    }

    #[test]
    fn high_pitch_drives_arms() {
        let notes = vec![Note {
            frequency: 500.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        }];
        let state = MusicalState::default();
        let phrase = choreograph(&notes, &state, 1.0);
        let kf = &phrase.keyframes[0];
        let arm_energy: f32 = kf.joint_targets[15..=20].iter().map(|t| t.abs()).sum();
        let leg_energy: f32 = kf.joint_targets[3..=14].iter().map(|t| t.abs()).sum();
        assert!(arm_energy > 0.0, "arms should move on high pitch");
        assert!(leg_energy < 1e-6, "legs should be still on high pitch");
    }

    #[test]
    fn ascending_style_lifts_arms() {
        let state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            ..Default::default()
        };
        let notes = vec![
            Note {
                frequency: 300.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.5,
            },
            Note {
                frequency: 300.0,
                start_time: 1.5,
                duration: 0.5,
                velocity: 0.5,
            },
        ];
        let phrase = choreograph(&notes, &state, 2.0);
        assert_eq!(phrase.style, DanceStyle::Ascending);
        // Later note should have higher arm values due to progressive lift
        let early_arm: f32 = phrase.keyframes[0].joint_targets[15..=20].iter().sum();
        let late_arm: f32 = phrase.keyframes[1].joint_targets[15..=20].iter().sum();
        assert!(
            late_arm > early_arm,
            "ascending should lift arms over time: early={early_arm}, late={late_arm}"
        );
    }
}

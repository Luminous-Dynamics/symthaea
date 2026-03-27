// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dance choreography: music-driven movement for embodied consciousness.
//!
//! Generates keyframe sequences for a 21-joint humanoid body, driven by
//! musical notes and cognitive state. Movement quality follows Laban's
//! effort framework (1950): Weight × Time × Space × Flow.
//!
//! # Music-to-Dance Mapping
//!
//! | Music Property | Dance Property | Basis |
//! |----------------|----------------|-------|
//! | Note onset | Keyframe trigger | Rhythmic entrainment |
//! | Frequency | Joint selection | Low=legs, high=arms |
//! | Velocity | Movement weight | Laban effort qualities |
//! | Rest (gap) | Sustained pose | Recovery/breath |
//! | Section type | Dance style | Cognitive context |
//!
//! # Joint Groups (dm_control humanoid)
//!
//! | Group | Joints (indices) | Musical Register |
//! |-------|------------------|------------------|
//! | Core | abdomen 0-2 | All (always active) |
//! | Legs | hips/knees/ankles 3-14 | Low pitch (<250 Hz) |
//! | Arms | shoulders/elbows 15-20 | High pitch (>400 Hz) |

use crate::{MusicalState, Note};
use serde::{Deserialize, Serialize};

/// Number of joints in the humanoid body (dm_control standard).
pub const NUM_JOINTS: usize = 21;

/// Laban movement weight — effort quality of the movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementWeight {
    /// Light: flowing, effortless (high serotonin).
    Light,
    /// Strong: powerful, grounded (high noradrenaline).
    Strong,
    /// Sustained: controlled, held (high consciousness).
    Sustained,
    /// Quick: fast, darting (high arousal).
    Quick,
}

/// Dance style derived from dominant Eight Harmony.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DanceStyle {
    /// SacredStillness: slow, grounded, tai-chi-like.
    Contemplative,
    /// PanSentientFlourishing: flowing, lyrical.
    Expressive,
    /// InfinitePlay: sharp, syncopated.
    Rhythmic,
    /// EvolutionaryProgression: reaching, growing, ascending.
    Ascending,
}

/// A single keyframe in a dance phrase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DanceKeyframe {
    /// Time in seconds (aligned to musical beat).
    pub time: f32,
    /// Target joint angles [-1, 1] for all 21 joints.
    pub joint_targets: [f32; NUM_JOINTS],
    /// Velocity profile [0, 1]: 0=slow/sustained, 1=sharp/staccato.
    pub velocity_profile: f32,
    /// Movement effort quality.
    pub weight: MovementWeight,
}

/// A complete dance phrase: sequence of keyframes with style metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DancePhrase {
    /// Ordered keyframes.
    pub keyframes: Vec<DanceKeyframe>,
    /// Overall dance style.
    pub style: DanceStyle,
    /// Beat times that keyframes align to.
    pub beat_alignment: Vec<f32>,
    /// Duration in seconds.
    pub duration_secs: f32,
}

/// Determine dance style from cognitive state.
pub fn determine_style(state: &MusicalState) -> DanceStyle {
    let stillness = state.harmony_activations[7];
    let flourishing = state.harmony_activations[1];
    let play = state.harmony_activations[3];
    let progression = state.harmony_activations[6];

    if stillness > 0.5 && stillness >= play && stillness >= progression {
        DanceStyle::Contemplative
    } else if play > 0.5 && play >= flourishing {
        DanceStyle::Rhythmic
    } else if progression > 0.5 {
        DanceStyle::Ascending
    } else {
        DanceStyle::Expressive
    }
}

/// Determine movement weight from cognitive state.
fn determine_weight(state: &MusicalState) -> MovementWeight {
    let arousal = state.arousal;
    let ne = state.noradrenaline;
    let serotonin = state.serotonin;
    let psi = state.consciousness_level;

    if arousal > 0.7 {
        MovementWeight::Quick
    } else if ne > 0.6 {
        MovementWeight::Strong
    } else if psi > 0.6 {
        MovementWeight::Sustained
    } else if serotonin > 0.5 {
        MovementWeight::Light
    } else {
        MovementWeight::Light
    }
}

/// Generate a dance phrase from a musical note sequence and cognitive state.
///
/// Each note onset triggers a keyframe. The note's frequency determines which
/// joint groups are activated (low pitch → legs, high pitch → arms).
pub fn choreograph(
    notes: &[Note],
    state: &MusicalState,
    duration_secs: f32,
) -> DancePhrase {
    let style = determine_style(state);
    let weight = determine_weight(state);

    let mut keyframes = Vec::new();
    let mut beat_alignment = Vec::new();

    // Amplitude scaling by style
    let amplitude = match style {
        DanceStyle::Contemplative => 0.3,
        DanceStyle::Expressive => 0.6,
        DanceStyle::Rhythmic => 0.8,
        DanceStyle::Ascending => 0.5,
    };

    let velocity_profile = match weight {
        MovementWeight::Light => 0.3,
        MovementWeight::Strong => 0.8,
        MovementWeight::Sustained => 0.2,
        MovementWeight::Quick => 0.9,
    };

    for note in notes {
        if note.start_time > duration_secs {
            break;
        }

        let mut targets = [0.0f32; NUM_JOINTS];
        let freq = note.frequency;
        let vel = note.velocity;

        // Map frequency to joint groups
        if freq < 250.0 {
            // Low pitch → legs (ground-based movement)
            activate_legs(&mut targets, vel * amplitude, note);
        } else if freq > 400.0 {
            // High pitch → arms (elevated gesture)
            activate_arms(&mut targets, vel * amplitude, note);
        } else {
            // Mid pitch → full body (core engagement)
            activate_core(&mut targets, vel * amplitude * 0.5);
            activate_legs(&mut targets, vel * amplitude * 0.3, note);
            activate_arms(&mut targets, vel * amplitude * 0.3, note);
        }

        // Core always engages slightly
        targets[0] += vel * amplitude * 0.15; // abdomen_y
        targets[1] += vel * amplitude * 0.1;  // abdomen_z

        // Ascending style: progressive upward bias
        if style == DanceStyle::Ascending {
            for t in targets[15..=20].iter_mut() {
                *t += 0.1 * note.start_time / duration_secs.max(0.1);
            }
        }

        // Clamp all targets
        for t in targets.iter_mut() {
            *t = t.clamp(-1.0, 1.0);
        }

        keyframes.push(DanceKeyframe {
            time: note.start_time,
            joint_targets: targets,
            velocity_profile,
            weight,
        });
        beat_alignment.push(note.start_time);
    }

    DancePhrase {
        keyframes,
        style,
        beat_alignment,
        duration_secs,
    }
}

/// Activate leg joints based on a note.
fn activate_legs(targets: &mut [f32; NUM_JOINTS], amplitude: f32, note: &Note) {
    // Use duration as a proxy for gesture scale (longer = bigger movement)
    let scale = (note.duration * 2.0).min(1.0);
    // Symmetric leg movement for grounding
    // Right leg: hips 3-5, knee 6, ankles 7-8
    targets[3] += amplitude * 0.4 * scale;  // right_hip_x
    targets[6] += amplitude * 0.6 * scale;  // right_knee (bend)
    targets[8] += amplitude * 0.3 * scale;  // right_ankle_y
    // Left leg: hips 9-11, knee 12, ankles 13-14
    targets[9] += amplitude * 0.4 * scale;  // left_hip_x
    targets[12] += amplitude * 0.6 * scale; // left_knee
    targets[14] += amplitude * 0.3 * scale; // left_ankle_y
}

/// Activate arm joints based on a note.
fn activate_arms(targets: &mut [f32; NUM_JOINTS], amplitude: f32, note: &Note) {
    let scale = (note.duration * 2.0).min(1.0);
    // Asymmetric arm gestures (leading arm based on frequency parity)
    let lead_right = (note.frequency as u32) % 2 == 0;
    let (lead_sh1, lead_sh2, lead_el) = if lead_right {
        (15, 16, 17) // right shoulder1, shoulder2, elbow
    } else {
        (18, 19, 20) // left shoulder1, shoulder2, elbow
    };
    targets[lead_sh1] += amplitude * 0.7 * scale;
    targets[lead_sh2] += amplitude * 0.5 * scale;
    targets[lead_el] += amplitude * 0.4 * scale;

    // Follow arm: smaller mirror
    let (follow_sh1, follow_el) = if lead_right { (18, 20) } else { (15, 17) };
    targets[follow_sh1] += amplitude * 0.3 * scale;
    targets[follow_el] += amplitude * 0.2 * scale;
}

/// Activate core/torso joints.
fn activate_core(targets: &mut [f32; NUM_JOINTS], amplitude: f32) {
    targets[0] += amplitude * 0.5; // abdomen_y (lateral bend)
    targets[1] += amplitude * 0.3; // abdomen_z (rotation)
    targets[2] += amplitude * 0.2; // abdomen_x (flexion)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_notes() -> Vec<Note> {
        vec![
            Note { frequency: 130.81, start_time: 0.0, duration: 0.5, velocity: 0.8 },  // C3 (low)
            Note { frequency: 261.63, start_time: 0.5, duration: 0.5, velocity: 0.7 },  // C4 (mid)
            Note { frequency: 523.25, start_time: 1.0, duration: 0.5, velocity: 0.9 },  // C5 (high)
            Note { frequency: 392.00, start_time: 1.5, duration: 0.5, velocity: 0.6 },  // G4 (mid)
        ]
    }

    #[test]
    fn generates_keyframes_per_note() {
        let state = MusicalState::default();
        let phrase = choreograph(&test_notes(), &state, 4.0);
        assert_eq!(phrase.keyframes.len(), 4);
    }

    #[test]
    fn keyframes_aligned_to_beats() {
        let state = MusicalState::default();
        let notes = test_notes();
        let phrase = choreograph(&notes, &state, 4.0);
        for (kf, note) in phrase.keyframes.iter().zip(notes.iter()) {
            assert!(
                (kf.time - note.start_time).abs() < 0.001,
                "keyframe at {} should align with note at {}",
                kf.time,
                note.start_time
            );
        }
    }

    #[test]
    fn joint_targets_bounded() {
        let state = MusicalState {
            consciousness_level: 0.9,
            arousal: 0.9,
            ..Default::default()
        };
        let phrase = choreograph(&test_notes(), &state, 4.0);
        for kf in &phrase.keyframes {
            for (i, &t) in kf.joint_targets.iter().enumerate() {
                assert!(
                    t >= -1.0 && t <= 1.0,
                    "joint {} target {} out of [-1, 1]",
                    i, t
                );
            }
        }
    }

    #[test]
    fn low_pitch_activates_legs() {
        let state = MusicalState::default();
        let low_notes = vec![Note {
            frequency: 130.0, // very low
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        }];
        let phrase = choreograph(&low_notes, &state, 2.0);
        let kf = &phrase.keyframes[0];
        // Legs (indices 3-14) should have more activation than arms (15-20)
        let leg_energy: f32 = kf.joint_targets[3..15].iter().map(|t| t.abs()).sum();
        let arm_energy: f32 = kf.joint_targets[15..21].iter().map(|t| t.abs()).sum();
        assert!(
            leg_energy > arm_energy,
            "low pitch: legs ({leg_energy}) should dominate arms ({arm_energy})"
        );
    }

    #[test]
    fn high_pitch_activates_arms() {
        let state = MusicalState::default();
        let high_notes = vec![Note {
            frequency: 800.0, // high
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        }];
        let phrase = choreograph(&high_notes, &state, 2.0);
        let kf = &phrase.keyframes[0];
        let leg_energy: f32 = kf.joint_targets[3..15].iter().map(|t| t.abs()).sum();
        let arm_energy: f32 = kf.joint_targets[15..21].iter().map(|t| t.abs()).sum();
        assert!(
            arm_energy > leg_energy,
            "high pitch: arms ({arm_energy}) should dominate legs ({leg_energy})"
        );
    }

    #[test]
    fn stillness_gives_contemplative() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.1, 0.8],
            ..Default::default()
        };
        assert_eq!(determine_style(&state), DanceStyle::Contemplative);
    }

    #[test]
    fn play_gives_rhythmic() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.8, 0.3, 0.3, 0.1, 0.1],
            ..Default::default()
        };
        assert_eq!(determine_style(&state), DanceStyle::Rhythmic);
    }

    #[test]
    fn high_arousal_gives_quick_weight() {
        let state = MusicalState {
            arousal: 0.9,
            ..Default::default()
        };
        assert_eq!(determine_weight(&state), MovementWeight::Quick);
    }

    #[test]
    fn empty_notes_empty_phrase() {
        let state = MusicalState::default();
        let phrase = choreograph(&[], &state, 4.0);
        assert!(phrase.keyframes.is_empty());
    }

    #[test]
    fn respects_duration_limit() {
        let state = MusicalState::default();
        let notes = vec![
            Note { frequency: 440.0, start_time: 0.0, duration: 0.5, velocity: 0.8 },
            Note { frequency: 440.0, start_time: 5.0, duration: 0.5, velocity: 0.8 }, // beyond duration
        ];
        let phrase = choreograph(&notes, &state, 2.0);
        assert_eq!(phrase.keyframes.len(), 1, "should skip notes beyond duration");
    }

    #[test]
    fn ascending_style_progressive_lift() {
        let state = MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.8, 0.1],
            ..Default::default()
        };
        let notes = vec![
            Note { frequency: 500.0, start_time: 0.0, duration: 0.5, velocity: 0.7 },
            Note { frequency: 500.0, start_time: 1.0, duration: 0.5, velocity: 0.7 },
            Note { frequency: 500.0, start_time: 2.0, duration: 0.5, velocity: 0.7 },
        ];
        let phrase = choreograph(&notes, &state, 3.0);
        assert_eq!(phrase.style, DanceStyle::Ascending);
        // Later keyframes should have higher arm targets due to progressive lift
        let arm_early: f32 = phrase.keyframes[0].joint_targets[15..21].iter().sum();
        let arm_late: f32 = phrase.keyframes[2].joint_targets[15..21].iter().sum();
        assert!(
            arm_late > arm_early,
            "ascending: late arms ({arm_late}) should exceed early ({arm_early})"
        );
    }
}

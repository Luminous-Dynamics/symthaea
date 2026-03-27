// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Animation timeline: temporal visual composition for consciousness art.
//!
//! Produces sequences of SVG frames from keyframed `CognitiveSnapshot` states,
//! with configurable interpolation modes. The CfC-dynamics mode creates organic
//! transitions where the art "thinks" its way between states rather than
//! mechanically tweening.
//!
//! # Interpolation Modes
//!
//! | Mode | Character | Basis |
//! |------|-----------|-------|
//! | Linear | Smooth, mechanical | LERP between snapshots |
//! | Spring | Bouncy, physical | Damped harmonic oscillator |
//! | Ease | Natural, S-curve | Smooth-step easing |

use serde::{Deserialize, Serialize};
use symthaea_canvas::CognitiveSnapshot;

/// Easing curve for transitions between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EasingCurve {
    /// Linear interpolation (constant velocity).
    Linear,
    /// Smooth ease-in-out (cubic Hermite).
    EaseInOut,
    /// Quick start, slow end.
    EaseOut,
    /// Slow start, quick end.
    EaseIn,
}

/// Interpolation mode for the animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMode {
    /// Linear LERP between snapshot keyframes.
    Linear,
    /// Damped spring oscillator (bouncy, physical transitions).
    Spring,
    /// Smooth easing with configurable curve.
    Ease,
}

/// A visual keyframe: a cognitive snapshot at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualKeyframe {
    /// Time in seconds.
    pub time: f32,
    /// Cognitive state to render at this moment.
    pub snapshot: CognitiveSnapshot,
    /// Easing curve for transition TO this keyframe.
    pub easing: EasingCurve,
}

/// An animation timeline: ordered keyframes with duration and interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationTimeline {
    /// Total duration in seconds.
    pub duration_secs: f32,
    /// Ordered keyframes (must be sorted by time).
    pub keyframes: Vec<VisualKeyframe>,
    /// Interpolation mode for between-keyframe states.
    pub interpolation: InterpolationMode,
    /// Frames per second for rendering.
    pub fps: f32,
}

impl Default for AnimationTimeline {
    fn default() -> Self {
        Self {
            duration_secs: 4.0,
            keyframes: vec![],
            interpolation: InterpolationMode::Ease,
            fps: 24.0,
        }
    }
}

/// Apply easing function to a t [0,1] parameter.
fn apply_easing(t: f32, curve: EasingCurve) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        EasingCurve::Linear => t,
        EasingCurve::EaseInOut => {
            // Smooth-step: 3t² - 2t³
            t * t * (3.0 - 2.0 * t)
        }
        EasingCurve::EaseOut => {
            // Decelerate: 1 - (1-t)²
            1.0 - (1.0 - t) * (1.0 - t)
        }
        EasingCurve::EaseIn => {
            // Accelerate: t²
            t * t
        }
    }
}

/// LERP between two f32 values.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// LERP between two f64 values.
fn lerp64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Spring interpolation: damped harmonic oscillator.
///
/// Creates a bouncy transition that overshoots then settles.
fn spring_ease(t: f32) -> f32 {
    if t >= 1.0 {
        return 1.0;
    }
    let omega = 8.0; // frequency
    let zeta = 0.4; // damping ratio (underdamped)
    let t = t.clamp(0.0, 1.0);
    1.0 - (-zeta * omega * t).exp() * ((1.0 - zeta * zeta).sqrt() * omega * t).cos()
}

/// Interpolate between two `CognitiveSnapshot`s.
///
/// Blends all numeric fields. Non-numeric fields (vecs) use the closer keyframe.
fn interpolate_snapshots(
    a: &CognitiveSnapshot,
    b: &CognitiveSnapshot,
    raw_t: f32,
    easing: EasingCurve,
    mode: InterpolationMode,
) -> CognitiveSnapshot {
    let t = match mode {
        InterpolationMode::Linear => raw_t.clamp(0.0, 1.0),
        InterpolationMode::Spring => spring_ease(raw_t),
        InterpolationMode::Ease => apply_easing(raw_t, easing),
    };
    let t64 = t as f64;

    let mut result = if t < 0.5 { a.clone() } else { b.clone() };

    // Interpolate scalar fields
    result.consciousness_level = lerp64(a.consciousness_level, b.consciousness_level, t64);
    result.prediction_error = lerp(a.prediction_error, b.prediction_error, t);
    result.living_mind_vitality = lerp64(a.living_mind_vitality, b.living_mind_vitality, t64);
    result.living_mind_coherence = lerp64(a.living_mind_coherence, b.living_mind_coherence, t64);

    // Neuromodulators
    result.dopamine = lerp(a.dopamine, b.dopamine, t);
    result.noradrenaline = lerp(a.noradrenaline, b.noradrenaline, t);
    result.serotonin = lerp(a.serotonin, b.serotonin, t);
    result.acetylcholine = lerp(a.acetylcholine, b.acetylcholine, t);
    result.oxytocin = lerp(a.oxytocin, b.oxytocin, t);
    result.gaba = lerp(a.gaba, b.gaba, t);

    // Affect
    result.valence = lerp(a.valence, b.valence, t);
    result.arousal = lerp(a.arousal, b.arousal, t);

    // Cantor depth
    result.cantor_metacognitive_depth =
        lerp(a.cantor_metacognitive_depth, b.cantor_metacognitive_depth, t);

    // Harmony activations
    for i in 0..8 {
        result.harmony_activations[i] =
            lerp(a.harmony_activations[i], b.harmony_activations[i], t);
    }

    // Thought vector (element-wise LERP if same length, else use closer)
    if a.thought_vector.len() == b.thought_vector.len() {
        result.thought_vector = a
            .thought_vector
            .iter()
            .zip(b.thought_vector.iter())
            .map(|(&va, &vb)| lerp(va, vb, t))
            .collect();
    }

    result
}

/// Render an animation timeline into a sequence of SVG frame strings.
///
/// Returns one SVG per frame. Frame count = ceil(duration * fps).
pub fn render_animation(
    timeline: &AnimationTimeline,
    config: &super::AtelierConfig,
) -> Vec<String> {
    if timeline.keyframes.is_empty() {
        return vec![];
    }

    let frame_count = (timeline.duration_secs * timeline.fps).ceil() as usize;
    let frame_duration = 1.0 / timeline.fps;
    let mut frames = Vec::with_capacity(frame_count);

    for frame_idx in 0..frame_count {
        let time = frame_idx as f32 * frame_duration;

        // Find bracketing keyframes
        let snapshot = sample_at_time(timeline, time);

        // Generate artwork for this frame's snapshot
        let artwork = super::create_artwork(config, &snapshot, frame_idx as u64);
        frames.push(artwork.svg);
    }

    frames
}

/// Sample the interpolated snapshot at a given time.
pub fn sample_at_time(timeline: &AnimationTimeline, time: f32) -> CognitiveSnapshot {
    let kfs = &timeline.keyframes;
    if kfs.is_empty() {
        return CognitiveSnapshot::dormant();
    }
    if kfs.len() == 1 || time <= kfs[0].time {
        return kfs[0].snapshot.clone();
    }
    if time >= kfs.last().unwrap().time {
        return kfs.last().unwrap().snapshot.clone();
    }

    // Find the two bracketing keyframes
    let mut upper_idx = 1;
    while upper_idx < kfs.len() && kfs[upper_idx].time < time {
        upper_idx += 1;
    }
    let lower_idx = upper_idx - 1;

    let a = &kfs[lower_idx];
    let b = &kfs[upper_idx];

    // Compute local t [0, 1] within this segment
    let segment_duration = (b.time - a.time).max(0.001);
    let local_t = (time - a.time) / segment_duration;

    interpolate_snapshots(
        &a.snapshot,
        &b.snapshot,
        local_t,
        b.easing,
        timeline.interpolation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap_low() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.2,
            dopamine: 0.1,
            valence: -0.5,
            arousal: 0.1,
            harmony_activations: [0.1; 8],
            ..CognitiveSnapshot::dormant()
        }
    }

    fn snap_high() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.9,
            dopamine: 0.8,
            valence: 0.7,
            arousal: 0.8,
            harmony_activations: [0.8; 8],
            ..CognitiveSnapshot::dormant()
        }
    }

    fn test_timeline() -> AnimationTimeline {
        AnimationTimeline {
            duration_secs: 2.0,
            keyframes: vec![
                VisualKeyframe {
                    time: 0.0,
                    snapshot: snap_low(),
                    easing: EasingCurve::Linear,
                },
                VisualKeyframe {
                    time: 2.0,
                    snapshot: snap_high(),
                    easing: EasingCurve::EaseInOut,
                },
            ],
            interpolation: InterpolationMode::Ease,
            fps: 10.0,
        }
    }

    #[test]
    fn frame_count_matches_duration_times_fps() {
        let tl = test_timeline();
        let config = super::super::AtelierConfig::default();
        let frames = render_animation(&tl, &config);
        let expected = (tl.duration_secs * tl.fps).ceil() as usize;
        assert_eq!(frames.len(), expected, "should produce {} frames", expected);
    }

    #[test]
    fn interpolation_midpoint_is_between() {
        let tl = test_timeline();
        let mid = sample_at_time(&tl, 1.0);
        assert!(
            mid.consciousness_level > 0.2 && mid.consciousness_level < 0.9,
            "midpoint consciousness {} should be between 0.2 and 0.9",
            mid.consciousness_level
        );
        assert!(
            mid.dopamine > 0.1 && mid.dopamine < 0.8,
            "midpoint dopamine {} should be between",
            mid.dopamine
        );
    }

    #[test]
    fn interpolation_at_start_matches_first_keyframe() {
        let tl = test_timeline();
        let start = sample_at_time(&tl, 0.0);
        assert!(
            (start.consciousness_level - 0.2).abs() < 0.01,
            "start should match first keyframe"
        );
    }

    #[test]
    fn interpolation_at_end_matches_last_keyframe() {
        let tl = test_timeline();
        let end = sample_at_time(&tl, 2.0);
        assert!(
            (end.consciousness_level - 0.9).abs() < 0.01,
            "end should match last keyframe"
        );
    }

    #[test]
    fn harmony_interpolates() {
        let tl = test_timeline();
        let mid = sample_at_time(&tl, 1.0);
        for &h in &mid.harmony_activations {
            assert!(h > 0.1 && h < 0.8, "harmony {} should be between", h);
        }
    }

    #[test]
    fn empty_timeline_empty_frames() {
        let tl = AnimationTimeline::default();
        let config = super::super::AtelierConfig::default();
        let frames = render_animation(&tl, &config);
        assert!(frames.is_empty());
    }

    #[test]
    fn single_keyframe_constant() {
        let tl = AnimationTimeline {
            duration_secs: 1.0,
            keyframes: vec![VisualKeyframe {
                time: 0.0,
                snapshot: snap_high(),
                easing: EasingCurve::Linear,
            }],
            interpolation: InterpolationMode::Linear,
            fps: 5.0,
        };
        let s1 = sample_at_time(&tl, 0.0);
        let s2 = sample_at_time(&tl, 0.5);
        assert!(
            (s1.consciousness_level - s2.consciousness_level).abs() < 0.01,
            "single keyframe should be constant"
        );
    }

    #[test]
    fn easing_functions_bounded() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for curve in [
                EasingCurve::Linear,
                EasingCurve::EaseInOut,
                EasingCurve::EaseIn,
                EasingCurve::EaseOut,
            ] {
                let e = apply_easing(t, curve);
                assert!(
                    e >= 0.0 && e <= 1.0,
                    "easing {:?} at t={} = {} out of bounds",
                    curve, t, e
                );
            }
        }
    }

    #[test]
    fn spring_converges() {
        let v0 = spring_ease(0.0);
        let v1 = spring_ease(1.0);
        assert!(v0.abs() < 0.1, "spring at t=0 should be near 0");
        assert!((v1 - 1.0).abs() < 0.01, "spring at t=1 should be near 1");
    }

    #[test]
    fn all_frames_are_valid_svg() {
        let tl = AnimationTimeline {
            duration_secs: 0.5,
            keyframes: vec![
                VisualKeyframe {
                    time: 0.0,
                    snapshot: snap_low(),
                    easing: EasingCurve::Linear,
                },
                VisualKeyframe {
                    time: 0.5,
                    snapshot: snap_high(),
                    easing: EasingCurve::EaseInOut,
                },
            ],
            interpolation: InterpolationMode::Linear,
            fps: 4.0,
        };
        let config = super::super::AtelierConfig {
            width: 200.0,
            height: 200.0,
            max_elements: 50,
            ..Default::default()
        };
        let frames = render_animation(&tl, &config);
        assert_eq!(frames.len(), 2);
        for (i, svg) in frames.iter().enumerate() {
            assert!(
                svg.contains("<svg"),
                "frame {} should contain SVG tag",
                i
            );
        }
    }
}

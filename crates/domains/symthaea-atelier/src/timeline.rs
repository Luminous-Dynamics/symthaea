// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Animation timeline with keyframed CognitiveSnapshot interpolation.
//!
//! Generates frame-by-frame SVG sequences from consciousness state trajectories,
//! enabling time-based visual art that reflects cognitive evolution.

use crate::AtelierConfig;
use symthaea_canvas::CognitiveSnapshot;

/// Easing curve for temporal interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingCurve {
    /// Constant-speed interpolation.
    Linear,
    /// Smooth start and end (Hermite smoothstep).
    EaseInOut,
    /// Smooth end, sharp start (quadratic deceleration).
    EaseOut,
    /// Sharp end, smooth start (quadratic acceleration).
    EaseIn,
}

/// Interpolation mode for keyframe transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear interpolation (LERP).
    Linear,
    /// Damped harmonic spring (overshoots, then settles).
    Spring,
    /// Easing-curve-based interpolation.
    Ease,
}

/// A keyframe binding a cognitive snapshot to a point in time.
#[derive(Debug, Clone)]
pub struct VisualKeyframe {
    /// Time in seconds from the start of the animation.
    pub time: f32,
    /// Cognitive state at this keyframe.
    pub snapshot: CognitiveSnapshot,
    /// Easing curve to use when transitioning FROM this keyframe.
    pub easing: EasingCurve,
}

/// An animation timeline composed of keyframes.
#[derive(Debug, Clone)]
pub struct AnimationTimeline {
    /// Total duration in seconds.
    pub duration_secs: f32,
    /// Keyframes, sorted by time.
    pub keyframes: Vec<VisualKeyframe>,
    /// Interpolation mode.
    pub interpolation: InterpolationMode,
    /// Target frames per second.
    pub fps: f32,
}

impl AnimationTimeline {
    /// Create a new empty timeline.
    pub fn new(duration_secs: f32, fps: f32) -> Self {
        Self {
            duration_secs,
            keyframes: Vec::new(),
            interpolation: InterpolationMode::Ease,
            fps,
        }
    }

    /// Add a keyframe. Keyframes must be added in chronological order.
    pub fn add_keyframe(&mut self, kf: VisualKeyframe) {
        self.keyframes.push(kf);
    }

    /// Total number of frames in the animation.
    pub fn frame_count(&self) -> usize {
        (self.duration_secs * self.fps).ceil() as usize
    }
}

/// Apply an easing curve to a normalized time value t in [0, 1].
pub fn apply_easing(t: f32, curve: EasingCurve) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        EasingCurve::Linear => t,
        EasingCurve::EaseInOut => {
            // Hermite smoothstep: 3t^2 - 2t^3
            t * t * (3.0 - 2.0 * t)
        }
        EasingCurve::EaseIn => {
            // Quadratic acceleration
            t * t
        }
        EasingCurve::EaseOut => {
            // Quadratic deceleration
            1.0 - (1.0 - t) * (1.0 - t)
        }
    }
}

/// Damped harmonic spring easing.
///
/// Models a critically-underdamped oscillator: overshoots target then settles.
/// zeta=0.4 (underdamped), omega=8.0.
pub fn spring_ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let zeta = 0.4_f32;
    let omega = 8.0_f32;
    let damped = (-zeta * omega * t).exp();
    let oscillation = ((1.0 - zeta * zeta).sqrt() * omega * t).cos();
    1.0 - damped * oscillation
}

/// Interpolate between two cognitive snapshots.
///
/// All scalar fields are linearly interpolated (LERP). Betti numbers are
/// rounded after interpolation. Thought vectors are element-wise interpolated.
pub fn interpolate_snapshots(
    a: &CognitiveSnapshot,
    b: &CognitiveSnapshot,
    t: f32,
    easing: EasingCurve,
    mode: InterpolationMode,
) -> CognitiveSnapshot {
    let s = match mode {
        InterpolationMode::Linear => t.clamp(0.0, 1.0),
        InterpolationMode::Spring => spring_ease(t),
        InterpolationMode::Ease => apply_easing(t, easing),
    };

    let lerp_f32 = |a: f32, b: f32| a + (b - a) * s;
    let lerp_f64 = |a: f64, b: f64| a + (b - a) * s as f64;

    // Interpolate harmony activations
    let mut harmony_activations = [0.0f32; 8];
    for i in 0..8 {
        harmony_activations[i] = lerp_f32(a.harmony_activations[i], b.harmony_activations[i]);
    }

    // Interpolate thought vector (element-wise, matching shorter length)
    let tv_len = a.thought_vector.len().min(b.thought_vector.len());
    let mut thought_vector = Vec::with_capacity(tv_len);
    for i in 0..tv_len {
        thought_vector.push(lerp_f32(a.thought_vector[i], b.thought_vector[i]));
    }

    // Interpolate Betti numbers (round to nearest integer)
    let betti_0 = (a.betti_0 as f32 + (b.betti_0 as f32 - a.betti_0 as f32) * s).round() as usize;
    let betti_1 = (a.betti_1 as f32 + (b.betti_1 as f32 - a.betti_1 as f32) * s).round() as usize;
    let betti_2 = (a.betti_2 as f32 + (b.betti_2 as f32 - a.betti_2 as f32) * s).round() as usize;

    CognitiveSnapshot {
        consciousness_level: lerp_f64(a.consciousness_level, b.consciousness_level),
        prediction_error: lerp_f32(a.prediction_error, b.prediction_error),
        living_mind_vitality: lerp_f64(a.living_mind_vitality, b.living_mind_vitality),
        living_mind_coherence: lerp_f64(a.living_mind_coherence, b.living_mind_coherence),
        dopamine: lerp_f32(a.dopamine, b.dopamine),
        noradrenaline: lerp_f32(a.noradrenaline, b.noradrenaline),
        serotonin: lerp_f32(a.serotonin, b.serotonin),
        acetylcholine: lerp_f32(a.acetylcholine, b.acetylcholine),
        oxytocin: lerp_f32(a.oxytocin, b.oxytocin),
        gaba: lerp_f32(a.gaba, b.gaba),
        allostatic_load: lerp_f32(a.allostatic_load, b.allostatic_load),
        betti_0,
        betti_1,
        betti_2,
        persistence_components: a.persistence_components.clone(),
        persistence_cycles: a.persistence_cycles.clone(),
        cantor_metacognitive_depth: lerp_f32(
            a.cantor_metacognitive_depth,
            b.cantor_metacognitive_depth,
        ),
        cantor_last_depth: if s < 0.5 {
            a.cantor_last_depth
        } else {
            b.cantor_last_depth
        },
        valence: lerp_f32(a.valence, b.valence),
        arousal: lerp_f32(a.arousal, b.arousal),
        harmony_activations,
        thought_vector,
        cycle_count: if s < 0.5 {
            a.cycle_count
        } else {
            b.cycle_count
        },
    }
}

/// Sample the timeline at a specific time, producing an interpolated snapshot.
///
/// If time is before the first keyframe, returns the first snapshot.
/// If time is after the last keyframe, returns the last snapshot.
pub fn sample_at_time(timeline: &AnimationTimeline, time: f32) -> CognitiveSnapshot {
    debug_assert!(
        timeline
            .keyframes
            .windows(2)
            .all(|w| w[0].time <= w[1].time),
        "keyframes must be sorted by time"
    );

    if timeline.keyframes.is_empty() {
        return CognitiveSnapshot::dormant();
    }
    if timeline.keyframes.len() == 1 {
        return timeline.keyframes[0].snapshot.clone();
    }

    // Before first keyframe
    if time <= timeline.keyframes[0].time {
        return timeline.keyframes[0].snapshot.clone();
    }
    // After last keyframe (safe: len >= 2 guaranteed by guards above)
    let last_kf = &timeline.keyframes[timeline.keyframes.len() - 1];
    if time >= last_kf.time {
        return last_kf.snapshot.clone();
    }

    // Find bracketing keyframes
    for i in 0..timeline.keyframes.len() - 1 {
        let kf_a = &timeline.keyframes[i];
        let kf_b = &timeline.keyframes[i + 1];
        if time >= kf_a.time && time <= kf_b.time {
            let span = kf_b.time - kf_a.time;
            let t = if span > 0.0 {
                (time - kf_a.time) / span
            } else {
                0.0
            };
            return interpolate_snapshots(
                &kf_a.snapshot,
                &kf_b.snapshot,
                t,
                kf_a.easing,
                timeline.interpolation,
            );
        }
    }

    last_kf.snapshot.clone()
}

/// Render the full animation timeline as a sequence of SVG strings.
///
/// Produces one SVG per frame at the timeline's configured FPS.
pub fn render_animation(timeline: &AnimationTimeline, config: &AtelierConfig) -> Vec<String> {
    let frame_count = timeline.frame_count();
    let mut svgs = Vec::with_capacity(frame_count);

    for frame_idx in 0..frame_count {
        let time = frame_idx as f32 / timeline.fps;
        let snapshot = sample_at_time(timeline, time);
        let artwork = crate::create_artwork(config, &snapshot, frame_idx as u64);
        svgs.push(artwork.svg);
    }

    svgs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timeline() -> AnimationTimeline {
        let mut tl = AnimationTimeline::new(2.0, 10.0);
        let mut snap_a = CognitiveSnapshot::dormant();
        snap_a.consciousness_level = 0.1;
        snap_a.harmony_activations = [0.0; 8];

        let mut snap_b = CognitiveSnapshot::dormant();
        snap_b.consciousness_level = 0.9;
        snap_b.harmony_activations = [1.0; 8];

        tl.add_keyframe(VisualKeyframe {
            time: 0.0,
            snapshot: snap_a,
            easing: EasingCurve::EaseInOut,
        });
        tl.add_keyframe(VisualKeyframe {
            time: 2.0,
            snapshot: snap_b,
            easing: EasingCurve::Linear,
        });
        tl
    }

    #[test]
    fn frame_count_correct() {
        let tl = make_timeline();
        assert_eq!(tl.frame_count(), 20);
    }

    #[test]
    fn midpoint_interpolation() {
        let tl = make_timeline();
        let mid = sample_at_time(&tl, 1.0);
        // EaseInOut at t=0.5: smoothstep(0.5) = 0.5
        assert!(
            mid.consciousness_level > 0.3 && mid.consciousness_level < 0.7,
            "midpoint should be near 0.5, got {}",
            mid.consciousness_level
        );
    }

    #[test]
    fn start_matches_first_keyframe() {
        let tl = make_timeline();
        let start = sample_at_time(&tl, 0.0);
        assert!(
            (start.consciousness_level - 0.1).abs() < 0.001,
            "start should match first keyframe"
        );
    }

    #[test]
    fn end_matches_last_keyframe() {
        let tl = make_timeline();
        let end = sample_at_time(&tl, 2.0);
        assert!(
            (end.consciousness_level - 0.9).abs() < 0.001,
            "end should match last keyframe"
        );
    }

    #[test]
    fn harmony_interpolation() {
        let tl = make_timeline();
        let mid = sample_at_time(&tl, 1.0);
        for &h in &mid.harmony_activations {
            assert!(h > 0.3 && h < 0.7, "harmony should be near 0.5, got {h}");
        }
    }

    #[test]
    fn empty_timeline() {
        let tl = AnimationTimeline::new(1.0, 10.0);
        let snap = sample_at_time(&tl, 0.5);
        assert!(
            (snap.consciousness_level - 0.05).abs() < 0.01,
            "empty timeline returns dormant"
        );
    }

    #[test]
    fn single_keyframe() {
        let mut tl = AnimationTimeline::new(1.0, 10.0);
        let mut snap = CognitiveSnapshot::dormant();
        snap.consciousness_level = 0.77;
        tl.add_keyframe(VisualKeyframe {
            time: 0.0,
            snapshot: snap,
            easing: EasingCurve::Linear,
        });
        let result = sample_at_time(&tl, 0.5);
        assert!((result.consciousness_level - 0.77).abs() < 0.001);
    }

    #[test]
    fn easing_bounds() {
        for curve in [
            EasingCurve::Linear,
            EasingCurve::EaseInOut,
            EasingCurve::EaseIn,
            EasingCurve::EaseOut,
        ] {
            assert!((apply_easing(0.0, curve)).abs() < 0.001, "{curve:?} at 0");
            assert!(
                (apply_easing(1.0, curve) - 1.0).abs() < 0.001,
                "{curve:?} at 1"
            );
            let mid = apply_easing(0.5, curve);
            assert!(mid >= 0.0 && mid <= 1.0, "{curve:?} mid={mid}");
        }
    }

    #[test]
    fn spring_convergence() {
        let at_end = spring_ease(1.0);
        assert!(
            (at_end - 1.0).abs() < 0.1,
            "spring should converge near 1.0 at t=1, got {at_end}"
        );
        // Spring may overshoot
        let at_mid = spring_ease(0.3);
        assert!(at_mid > 0.0, "spring should be positive at t=0.3");
    }

    #[test]
    fn render_produces_valid_svgs() {
        let tl = make_timeline();
        let config = AtelierConfig {
            style: crate::AtelierStyle::ParametricCurve,
            iteration_budget: 1,
            ..Default::default()
        };
        let svgs = render_animation(&tl, &config);
        assert_eq!(svgs.len(), 20);
        for svg in &svgs {
            assert!(svg.contains("<svg"), "each frame should be valid SVG");
        }
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! State smoother: low-pass filter on all MusicalState parameters.
//!
//! Prevents abrupt changes in consciousness state from producing
//! jarring audio transitions. All parameters change gradually over
//! a configurable time constant (default ~1 second).

use crate::MusicalState;

/// Low-pass filter for MusicalState transitions.
pub struct StateSmoother {
    /// Current smoothed state.
    current: MusicalState,
    /// Smoothing coefficient per chunk (lower = smoother).
    alpha: f32,
    /// Whether the smoother has been initialized.
    initialized: bool,
}

impl StateSmoother {
    /// Create a smoother with given time constant.
    ///
    /// `time_constant_chunks`: number of chunks to reach ~63% of target.
    /// At 32ms/chunk: 30 chunks ≈ 1 second.
    pub fn new(time_constant_chunks: f32) -> Self {
        Self {
            current: MusicalState::default(),
            alpha: (1.0 / time_constant_chunks.max(1.0)).clamp(0.01, 1.0),
            initialized: false,
        }
    }

    /// Default: ~1 second time constant (30 chunks at 32ms).
    pub fn default_smooth() -> Self {
        Self::new(30.0)
    }

    /// Smooth a target state, returning the interpolated result.
    pub fn smooth(&mut self, target: &MusicalState) -> MusicalState {
        if !self.initialized {
            self.current = target.clone();
            self.initialized = true;
            return self.current.clone();
        }

        let a = self.alpha;
        self.current.consciousness_level +=
            a * (target.consciousness_level - self.current.consciousness_level);
        self.current.arousal += a * (target.arousal - self.current.arousal);
        self.current.valence += a * (target.valence - self.current.valence);
        self.current.dopamine += a * (target.dopamine - self.current.dopamine);
        self.current.serotonin += a * (target.serotonin - self.current.serotonin);
        self.current.noradrenaline += a * (target.noradrenaline - self.current.noradrenaline);
        self.current.prediction_error +=
            a * (target.prediction_error - self.current.prediction_error);

        for i in 0..8 {
            self.current.harmony_activations[i] +=
                a * (target.harmony_activations[i] - self.current.harmony_activations[i]);
        }

        self.current.clone()
    }

    /// Get current smoothed state.
    pub fn current(&self) -> &MusicalState {
        &self.current
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_is_passthrough() {
        let mut sm = StateSmoother::default_smooth();
        let target = MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        };
        let result = sm.smooth(&target);
        assert_eq!(result.consciousness_level, 0.8);
    }

    #[test]
    fn subsequent_calls_interpolate() {
        let mut sm = StateSmoother::default_smooth();
        let low = MusicalState {
            consciousness_level: 0.2,
            ..Default::default()
        };
        sm.smooth(&low); // initialize

        let high = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        let result = sm.smooth(&high);

        // Should be between 0.2 and 0.9 (not jump to 0.9)
        assert!(result.consciousness_level > 0.2, "should increase");
        assert!(
            result.consciousness_level < 0.9,
            "should not jump: {}",
            result.consciousness_level
        );
    }

    #[test]
    fn converges_over_time() {
        let mut sm = StateSmoother::default_smooth();
        let start = MusicalState {
            arousal: 0.1,
            ..Default::default()
        };
        sm.smooth(&start);

        let target = MusicalState {
            arousal: 0.8,
            ..Default::default()
        };
        let mut last = 0.1;
        for _ in 0..100 {
            let result = sm.smooth(&target);
            assert!(
                result.arousal >= last - 0.001,
                "should monotonically approach target"
            );
            last = result.arousal;
        }
        assert!(
            (last - 0.8).abs() < 0.05,
            "should converge to target: {last}"
        );
    }

    #[test]
    fn harmonies_smooth() {
        let mut sm = StateSmoother::default_smooth();
        sm.smooth(&MusicalState {
            harmony_activations: [0.0; 8],
            ..Default::default()
        });

        let target = MusicalState {
            harmony_activations: [1.0; 8],
            ..Default::default()
        };
        let result = sm.smooth(&target);

        assert!(
            result.harmony_activations[0] > 0.0 && result.harmony_activations[0] < 1.0,
            "harmonies should interpolate: {}",
            result.harmony_activations[0]
        );
    }
}

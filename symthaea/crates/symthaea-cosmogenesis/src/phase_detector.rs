// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::CognitiveCosmogenesisMetrics;

pub struct PhaseTransitionDetector;

impl PhaseTransitionDetector {
    /// Detects if a phase transition occurred based on the second derivative of separation.
    /// A "phase transition" is defined as the step with the maximum acceleration in separation improvement.
    pub fn find_critical_point(history: &[CognitiveCosmogenesisMetrics]) -> Option<usize> {
        if history.len() < 3 {
            return None;
        }

        let mut max_accel = 0.0;
        let mut critical_step = None;

        for i in 1..history.len() - 1 {
            let v1 = history[i].separation_proxy - history[i - 1].separation_proxy;
            let v2 = history[i + 1].separation_proxy - history[i].separation_proxy;
            let accel = (v2 - v1).abs();

            if accel > max_accel {
                max_accel = accel;
                critical_step = Some(history[i].current_step);
            }
        }
        critical_step
    }
}

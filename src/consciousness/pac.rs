// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Phase-Amplitude Coupling (PAC) - The Binding Mechanism
======================================================

Implements rigorous Cross-Frequency Coupling (CFC) to measure how
low-frequency Global Workspace oscillations (\(\theta\) or \(\alpha\)) modulate
high-frequency Feature Binding oscillations (\(\gamma\)).

# Scientific Role in Symthaea
In the Master Equation $C = \min(\Phi, B, W, A, R)$, the "Binding" ($B$)
and "Workspace" ($W$) components are often correlated. PAC provides the
causal link: $W \to B$.

If $PAC(W, B)$ is high, it means the Global Workspace is successfully
orchestrating local feature binding (Top-down control).

# Algorithm
Uses the **Modulation Index (MI)** based on the Kullback-Leibler divergence
of the amplitude distribution from a uniform distribution.

1. Filter signal A into low-freq band (Phase).
2. Filter signal B into high-freq band (Amplitude).
3. Bin amplitudes of B according to phase of A.
4. Calculate deviation from uniform distribution.
*/

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Tracks signals for Phase-Amplitude Coupling computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacTracker {
    /// Name of the low-frequency driver (e.g., "Workspace")
    pub low_freq_name: String,

    /// Name of the high-frequency responder (e.g., "Binding")
    pub high_freq_name: String,

    /// Phase history of low-freq signal
    pub phase_history: VecDeque<f64>,

    /// Amplitude history of high-freq signal
    pub amplitude_history: VecDeque<f64>,

    /// Window size for computation
    pub window_size: usize,

    /// Number of phase bins (default 18 for 20-degree bins)
    pub num_bins: usize,
}

impl PacTracker {
    pub fn new(low_name: &str, high_name: &str, window: usize) -> Self {
        Self {
            low_freq_name: low_name.to_string(),
            high_freq_name: high_name.to_string(),
            phase_history: VecDeque::with_capacity(window),
            amplitude_history: VecDeque::with_capacity(window),
            window_size: window,
            num_bins: 18,
        }
    }

    /// Add a new data point
    ///
    /// * `low_freq_val`: Raw value of low-freq signal (will be Hilbert-transformed or proxy-phased)
    /// * `high_freq_val`: Raw value of high-freq signal (amplitude)
    pub fn observe(&mut self, low_freq_phase: f64, high_freq_amp: f64) {
        if self.phase_history.len() >= self.window_size {
            self.phase_history.pop_front();
            self.amplitude_history.pop_front();
        }

        // Normalize phase to [0, 2π]
        let normalized_phase = low_freq_phase.rem_euclid(2.0 * PI);

        self.phase_history.push_back(normalized_phase);
        self.amplitude_history.push_back(high_freq_amp);
    }

    /// Compute Modulation Index (MI)
    ///
    /// Returns value in [0, 1] where:
    /// - 0.0: Uniform distribution (No coupling)
    /// - 1.0: Dirac distribution (Perfect coupling)
    pub fn compute_modulation_index(&self) -> f64 {
        if self.phase_history.len() < self.num_bins {
            return 0.0;
        }

        // 1. Bin Amplitudes by Phase
        let mut bins = vec![0.0; self.num_bins];
        let mut counts = vec![0; self.num_bins];
        let bin_width = 2.0 * PI / self.num_bins as f64;

        for (phase, amp) in self.phase_history.iter().zip(self.amplitude_history.iter()) {
            let bin_idx = (phase / bin_width).floor() as usize;
            let idx = bin_idx.clamp(0, self.num_bins - 1);

            bins[idx] += amp;
            counts[idx] += 1;
        }

        // 2. Normalize bins (Mean amplitude per phase)
        let mut mean_amplitudes = vec![0.0; self.num_bins];
        let mut total_amp = 0.0;

        for i in 0..self.num_bins {
            if counts[i] > 0 {
                mean_amplitudes[i] = bins[i] / counts[i] as f64;
            } else {
                mean_amplitudes[i] = 0.0;
            }
            total_amp += mean_amplitudes[i];
        }

        // 3. Convert to probability distribution (P)
        if total_amp < 1e-9 {
            return 0.0;
        }

        let p: Vec<f64> = mean_amplitudes.iter().map(|&a| a / total_amp).collect();

        // 4. Compute Shannon Entropy (H)
        let mut h = 0.0;
        for &prob in &p {
            if prob > 1e-9 {
                h -= prob * prob.ln();
            }
        }

        // 5. Compute Modulation Index
        // MI = (H_max - H) / H_max
        // H_max = ln(N) (Uniform distribution)
        let h_max = (self.num_bins as f64).ln();

        ((h_max - h) / h_max).clamp(0.0, 1.0)
    }
}

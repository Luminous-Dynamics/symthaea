// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Phase space tracker for the 9-transmitter neuromodulator bath.
///
/// Records state vectors over a sliding window and computes entropy,
/// centroid, variance, and attractor detection.
pub struct BathPhaseTracker {
    pub(crate) history: VecDeque<[f32; 9]>,
    capacity: usize,
    /// Total number of state vectors recorded (including those evicted from the window).
    pub total_recorded: usize,
}

impl Default for BathPhaseTracker {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(200),
            capacity: 200,
            total_recorded: 0,
        }
    }
}

impl BathPhaseTracker {
    /// Record a state vector.
    pub fn record(&mut self, state: [f32; 9]) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(state);
        self.total_recorded += 1;
    }

    /// Shannon entropy averaged across all 9 dimensions (10-bin histogram per dimension).
    pub fn entropy(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let n = self.history.len() as f32;
        let mut total_entropy = 0.0_f32;
        for dim in 0..9 {
            let mut bins = [0u32; 10];
            for state in &self.history {
                let idx = ((state[dim].clamp(0.0, 1.999)) * 5.0) as usize;
                bins[idx.min(9)] += 1;
            }
            let mut dim_entropy = 0.0_f32;
            for &count in &bins {
                if count > 0 {
                    let p = count as f32 / n;
                    dim_entropy -= p * p.ln();
                }
            }
            total_entropy += dim_entropy;
        }
        total_entropy / 9.0
    }

    /// Arithmetic mean of recorded state vectors.
    pub fn centroid(&self) -> [f32; 9] {
        if self.history.is_empty() {
            return [0.0; 9];
        }
        let n = self.history.len() as f32;
        let mut sum = [0.0_f32; 9];
        for state in &self.history {
            for (i, &v) in state.iter().enumerate() {
                sum[i] += v;
            }
        }
        for v in &mut sum {
            *v /= n;
        }
        sum
    }

    /// Per-dimension variance of recorded state vectors.
    pub fn variance(&self) -> [f32; 9] {
        if self.history.len() < 2 {
            return [0.0; 9];
        }
        let centroid = self.centroid();
        let n = self.history.len() as f32;
        let mut var = [0.0_f32; 9];
        for state in &self.history {
            for (i, &v) in state.iter().enumerate() {
                let diff = v - centroid[i];
                var[i] += diff * diff;
            }
        }
        for v in &mut var {
            *v /= n;
        }
        var
    }

    /// Detect attractor: returns Some(centroid) if total variance < 0.05 and >= 50 samples.
    pub fn detect_attractor(&self) -> Option<[f32; 9]> {
        if self.history.len() < 50 {
            return None;
        }
        let var = self.variance();
        let total_var: f32 = var.iter().sum();
        if total_var < 0.05 {
            Some(self.centroid())
        } else {
            None
        }
    }

    /// Return the last N state vectors from history as a trajectory.
    pub fn trajectory(&self, n: usize) -> Vec<[f32; 9]> {
        let len = self.history.len();
        let start = len.saturating_sub(n);
        self.history.iter().skip(start).copied().collect()
    }

    /// Current number of recorded samples.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the tracker has no recorded samples.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Export the full trajectory as a serializable timeline with summary statistics.
    pub fn to_timeline(&self, phase_label: &str) -> BathTimeline {
        let entries: Vec<BathTimelineEntry> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, &state)| {
                let cycle = self.total_recorded.saturating_sub(self.history.len()) + i;
                BathTimelineEntry { cycle, state }
            })
            .collect();
        BathTimeline {
            entries,
            centroid: self.centroid(),
            variance: self.variance(),
            entropy: self.entropy(),
            phase_label: phase_label.to_string(),
        }
    }
}

/// A single entry in a bath timeline export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BathTimelineEntry {
    pub cycle: usize,
    pub state: [f32; 9],
}

/// Serializable timeline of full bath trajectory for offline analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BathTimeline {
    pub entries: Vec<BathTimelineEntry>,
    pub centroid: [f32; 9],
    pub variance: [f32; 9],
    pub entropy: f32,
    pub phase_label: String,
}

/// A confirmed phase transition event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    pub from: String,
    pub to: String,
    pub cycle: usize,
}

/// Hysteresis-based phase transition detector preventing oscillatory flicker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionDetector {
    current_phase: String,
    pending_phase: Option<String>,
    pending_count: u32,
    hysteresis_threshold: u32,
    transitions: VecDeque<PhaseTransition>,
    max_history: usize,
    cycle_counter: usize,
}

impl PhaseTransitionDetector {
    pub fn new(hysteresis_threshold: u32) -> Self {
        Self {
            current_phase: "balanced".to_string(),
            pending_phase: None,
            pending_count: 0,
            hysteresis_threshold,
            transitions: VecDeque::with_capacity(50),
            max_history: 50,
            cycle_counter: 0,
        }
    }

    pub fn update(&mut self, label: &str) -> Option<PhaseTransition> {
        self.cycle_counter += 1;
        if label == self.current_phase {
            self.pending_phase = None;
            self.pending_count = 0;
            return None;
        }
        if let Some(ref pending) = self.pending_phase {
            if pending == label {
                self.pending_count += 1;
                if self.pending_count >= self.hysteresis_threshold {
                    let transition = PhaseTransition {
                        from: self.current_phase.clone(),
                        to: label.to_string(),
                        cycle: self.cycle_counter,
                    };
                    self.current_phase = label.to_string();
                    self.pending_phase = None;
                    self.pending_count = 0;
                    if self.transitions.len() >= self.max_history {
                        self.transitions.pop_front();
                    }
                    self.transitions.push_back(transition.clone());
                    return Some(transition);
                }
            } else {
                self.pending_phase = Some(label.to_string());
                self.pending_count = 1;
            }
        } else {
            self.pending_phase = Some(label.to_string());
            self.pending_count = 1;
        }
        None
    }

    pub fn current_phase(&self) -> &str {
        &self.current_phase
    }

    pub fn transitions(&self) -> &VecDeque<PhaseTransition> {
        &self.transitions
    }

    pub fn reset(&mut self) {
        self.current_phase = "balanced".to_string();
        self.pending_phase = None;
        self.pending_count = 0;
        self.transitions.clear();
        self.cycle_counter = 0;
    }
}

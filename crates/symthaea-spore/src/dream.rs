// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual dream engine for the Spore WASM consciousness kernel.
//!
//! Records high-surprise moments during cognitive cycles, then "dreams" by
//! generating counterfactual actions and simulating alternative outcomes.
//! Wisdom entries capture lessons where a counterfactual would have produced
//! higher Phi (consciousness integration proxy).
//!
//! Fully WASM-compatible: no std::time, no blake3, no anyhow.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tuning knobs for the dream engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Minimum surprise to record an event (0.0–1.0).
    pub surprise_threshold: f32,
    /// Number of counterfactual actions generated per dream cycle.
    pub counterfactual_count: usize,
    /// Minimum Phi improvement to promote a counterfactual to wisdom.
    pub wisdom_threshold: f32,
    /// Maximum stored dream events before oldest are evicted.
    pub max_memory_size: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.1,
            counterfactual_count: 5,
            wisdom_threshold: 0.01,
            max_memory_size: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A recorded high-surprise moment suitable for counterfactual replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEvent {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub actual_outcome: Vec<f32>,
    pub surprise: f32,
    pub cycle: u64,
}

/// A lesson learned: a counterfactual action that would have improved Phi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wisdom {
    pub context_state: Vec<f32>,
    pub better_action: Vec<f32>,
    pub phi_improvement: f32,
    pub confidence: f32,
}

/// Output of a single dream cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamResult {
    pub insights: usize,
    pub events_processed: usize,
    pub simulations_run: usize,
    pub best_phi_improvement: f32,
}

/// Cumulative dream statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamStats {
    pub events_recorded: u64,
    pub dream_cycles: u64,
    pub total_insights: u64,
    pub total_simulations: u64,
    pub wisdom_count: usize,
}

// ---------------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG (deterministic, no-std, WASM-safe).
fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut mag_a: f64 = 0.0;
    let mut mag_b: f64 = 0.0;
    for i in 0..len {
        let va = a[i] as f64;
        let vb = b[i] as f64;
        dot += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// Simple Phi proxy: mean of absolute values.
fn phi_proxy(outcome: &[f32]) -> f32 {
    if outcome.is_empty() {
        return 0.0;
    }
    let sum: f32 = outcome.iter().map(|v| v.abs()).sum();
    sum / outcome.len() as f32
}

// ---------------------------------------------------------------------------
// Dream engine
// ---------------------------------------------------------------------------

/// Counterfactual dream engine.
///
/// Records high-surprise cognitive moments, then replays them with
/// perturbed actions to discover better strategies.
pub struct DreamEngine {
    config: DreamConfig,
    events: Vec<DreamEvent>,
    wisdoms: Vec<Wisdom>,
    rng_state: u64,
    stats: InternalStats,
}

#[derive(Default)]
struct InternalStats {
    events_recorded: u64,
    dream_cycles: u64,
    total_insights: u64,
    total_simulations: u64,
}

impl DreamEngine {
    /// Create a dream engine with the given configuration.
    pub fn new(config: DreamConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            wisdoms: Vec::new(),
            rng_state: 0xDEAD_BEEF_CAFE_1234,
            stats: InternalStats::default(),
        }
    }

    /// Create a dream engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DreamConfig::default())
    }

    /// Record a cognitive moment if its surprise exceeds the threshold.
    pub fn record(
        &mut self,
        state: &[f32],
        action: &[f32],
        outcome: &[f32],
        surprise: f32,
        cycle: u64,
    ) {
        if surprise < self.config.surprise_threshold {
            return;
        }

        self.events.push(DreamEvent {
            state: state.to_vec(),
            action: action.to_vec(),
            actual_outcome: outcome.to_vec(),
            surprise,
            cycle,
        });
        self.stats.events_recorded += 1;

        // Evict oldest events when over capacity.
        while self.events.len() > self.config.max_memory_size {
            self.events.remove(0);
        }
    }

    /// Run one dream cycle: pick the most surprising event, generate
    /// counterfactual actions, simulate outcomes, and consolidate wisdom.
    ///
    /// Returns `None` if no events are available.
    pub fn dream(&mut self) -> Option<DreamResult> {
        if self.events.is_empty() {
            return None;
        }

        self.stats.dream_cycles += 1;

        // Pick the most surprising event.
        let idx = self
            .events
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.surprise
                    .partial_cmp(&b.surprise)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let event = self.events.remove(idx);
        let actual_phi = phi_proxy(&event.actual_outcome);

        let mut insights = 0usize;
        let mut best_phi_improvement: f32 = 0.0;
        let simulations = self.config.counterfactual_count;

        for _ in 0..simulations {
            // Generate a counterfactual action by perturbing the original.
            let cf_action = self.perturb(&event.action);

            // Simulate the counterfactual outcome.
            let cf_outcome = self.simulate(&event.state, &cf_action, &event.actual_outcome);

            let cf_phi = phi_proxy(&cf_outcome);
            let improvement = cf_phi - actual_phi;

            if improvement > self.config.wisdom_threshold {
                // Confidence based on similarity between counterfactual and
                // original action — small perturbations yield higher confidence.
                let sim = cosine_similarity(&cf_action, &event.action);
                let confidence = sim.max(0.0).min(1.0);

                self.wisdoms.push(Wisdom {
                    context_state: event.state.clone(),
                    better_action: cf_action,
                    phi_improvement: improvement,
                    confidence,
                });
                insights += 1;
                self.stats.total_insights += 1;

                if improvement > best_phi_improvement {
                    best_phi_improvement = improvement;
                }
            }
        }

        self.stats.total_simulations += simulations as u64;

        Some(DreamResult {
            insights,
            events_processed: 1,
            simulations_run: simulations,
            best_phi_improvement,
        })
    }

    /// Run multiple dream cycles, returning results for each.
    pub fn dream_session(&mut self, cycles: usize) -> Vec<DreamResult> {
        let mut results = Vec::with_capacity(cycles);
        for _ in 0..cycles {
            match self.dream() {
                Some(r) => results.push(r),
                None => break,
            }
        }
        results
    }

    /// Access accumulated wisdom.
    pub fn wisdom(&self) -> &[Wisdom] {
        &self.wisdoms
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> DreamStats {
        DreamStats {
            events_recorded: self.stats.events_recorded,
            dream_cycles: self.stats.dream_cycles,
            total_insights: self.stats.total_insights,
            total_simulations: self.stats.total_simulations,
            wisdom_count: self.wisdoms.len(),
        }
    }

    /// Number of dream events currently held.
    pub fn memory_size(&self) -> usize {
        self.events.len()
    }

    /// Discard all stored events and wisdom.
    pub fn clear_memory(&mut self) {
        self.events.clear();
        self.wisdoms.clear();
    }

    // -- private helpers --------------------------------------------------

    /// Perturb a vector with small random noise (±0.2 per component).
    fn perturb(&mut self, action: &[f32]) -> Vec<f32> {
        action
            .iter()
            .map(|v| {
                let noise = (xorshift(&mut self.rng_state) as f32 - 0.5) * 0.4;
                v + noise
            })
            .collect()
    }

    /// Simulate a counterfactual outcome: 60 % actual outcome + 40 % heuristic.
    ///
    /// The heuristic component blends the perturbed action with the state
    /// to produce a plausible alternative.
    fn simulate(&mut self, state: &[f32], cf_action: &[f32], actual_outcome: &[f32]) -> Vec<f32> {
        let len = actual_outcome.len();
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let actual = actual_outcome[i];
            // Heuristic: element-wise blend of state and counterfactual action.
            let s = if i < state.len() { state[i] } else { 0.0 };
            let a = if i < cf_action.len() {
                cf_action[i]
            } else {
                0.0
            };
            let heuristic = 0.5 * s + 0.5 * a;
            // 60/40 blend.
            let noise = (xorshift(&mut self.rng_state) as f32 - 0.5) * 0.05;
            result.push(0.6 * actual + 0.4 * heuristic + noise);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_threshold_not_recorded() {
        let mut engine = DreamEngine::with_defaults();
        engine.record(&[1.0], &[0.5], &[0.3], 0.05, 1);
        assert_eq!(engine.memory_size(), 0);
    }

    #[test]
    fn test_record_and_dream() {
        let mut engine = DreamEngine::with_defaults();
        engine.record(&[1.0, 0.5], &[0.2, 0.3], &[0.1, 0.1], 0.8, 10);
        assert_eq!(engine.memory_size(), 1);

        let result = engine.dream().expect("should dream");
        assert_eq!(result.events_processed, 1);
        assert_eq!(result.simulations_run, 5);
        assert_eq!(engine.memory_size(), 0);
    }

    #[test]
    fn test_max_memory_eviction() {
        let config = DreamConfig {
            max_memory_size: 3,
            surprise_threshold: 0.0,
            ..DreamConfig::default()
        };
        let mut engine = DreamEngine::new(config);
        for i in 0..5 {
            engine.record(&[i as f32], &[0.0], &[0.0], 0.5, i);
        }
        assert_eq!(engine.memory_size(), 3);
        // Oldest events should have been evicted; newest cycle ids remain.
        let cycles: Vec<u64> = engine.events.iter().map(|e| e.cycle).collect();
        assert_eq!(cycles, vec![2, 3, 4]);
    }

    #[test]
    fn test_dream_session_drains_events() {
        let mut engine = DreamEngine::with_defaults();
        for i in 0..4 {
            engine.record(
                &[0.5, 0.5],
                &[0.1, 0.2],
                &[0.3, 0.4],
                0.5 + i as f32 * 0.1,
                i,
            );
        }
        let results = engine.dream_session(10);
        // Should process all 4 events, then stop.
        assert_eq!(results.len(), 4);
        assert_eq!(engine.memory_size(), 0);

        let stats = engine.stats();
        assert_eq!(stats.dream_cycles, 4);
        assert_eq!(stats.events_recorded, 4);
        assert_eq!(stats.total_simulations, 20);
    }

    #[test]
    fn test_cosine_similarity_identity() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity should be 1.0");
    }
}

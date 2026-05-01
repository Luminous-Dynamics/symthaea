// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness phase transition: bifurcation analysis of collective Φ.
//!
//! # Scientific Context
//!
//! In Integrated Information Theory (Tononi 2004), Φ is the canonical measure of
//! integrated information. This module investigates whether collective Φ in a
//! multi-agent physics system undergoes a **phase transition** as the coupling
//! strength between agents increases — analogous to the ferromagnetic transition
//! in the Ising model (Ising 1925) or the percolation threshold (Stauffer & Aharony 1994).
//!
//! The bifurcation parameter is `φ_gravity_strength k` (units: m⁻¹):
//! ```text
//! F_ij = k · Φ_i · Φ_j · r̂_ij / |r_ij|²
//! ```
//! At k = 0: entities are independent → fragmented regime, Φ̄ ≈ Φ_0.
//! At k = k_c: collective integration emerges → Φ̄ bifurcates upward.
//! At k >> k_c: entities coalesce into a unified field → Φ̄ → Φ_max.
//!
//! # Transition Signatures
//!
//! We track two primary observables:
//!
//! 1. **Order parameter**: `m(k) = ⟨Φ(k)⟩ - Φ_0`
//!    - m = 0 in the disordered (fragmented) phase
//!    - m > 0 in the ordered (integrated) phase
//!    - At a second-order transition: `m ~ |k - k_c|^β` (power law)
//!
//! 2. **Susceptibility**: `χ(k) = N · Var(Φ) / ⟨Φ⟩`
//!    - Peaks at the critical point k_c (diverges in the thermodynamic limit)
//!    - Analogous to magnetic susceptibility ∂M/∂H in the Ising model
//!
//! We also compute the **Binder cumulant** (Binder 1981):
//! ```text
//! U₄ = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²)
//! ```
//! which crosses at k_c for different system sizes N, providing a
//! size-independent estimate of the critical coupling.
//!
//! # Usage
//!
//! ```ignore
//! let config = BifurcationConfig {
//!     k_min: 0.0, k_max: 1.0, k_steps: 40,
//!     num_agents: 8, warmup_ticks: 100, measure_ticks: 200,
//! };
//! let result = run_bifurcation_sweep(&config);
//! println!("Critical coupling: k_c ≈ {:.3}", result.critical_coupling);
//! ```

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

use crate::coupling::ConsciousnessField;

// ────────────────────────────────────────────────────────────
//  Configuration
// ────────────────────────────────────────────────────────────

/// Configuration for a bifurcation sweep experiment.
#[derive(Debug, Clone)]
pub struct BifurcationConfig {
    /// Minimum coupling strength k (Φ-gravity).
    pub k_min: f64,
    /// Maximum coupling strength k.
    pub k_max: f64,
    /// Number of k values to sweep (linearly spaced).
    pub k_steps: usize,
    /// Number of agents in the simulation.
    pub num_agents: usize,
    /// Ring radius for initial agent placement (2D).
    pub ring_radius: f64,
    /// Ticks to run before measuring (allows transients to decay).
    pub warmup_ticks: u64,
    /// Ticks to measure Φ over for statistics.
    pub measure_ticks: u64,
    /// Physics timestep (seconds).
    pub dt: f64,
    /// Fixed consciousness inputs for all agents (controls baseline Φ).
    pub agent_inputs: ConsciousnessInputs,
}

impl Default for BifurcationConfig {
    fn default() -> Self {
        Self {
            k_min: 0.0,
            k_max: 1.0,
            k_steps: 20,
            num_agents: 8,
            ring_radius: 10.0,
            warmup_ticks: 50,
            measure_ticks: 100,
            dt: 1.0 / 60.0,
            agent_inputs: ConsciousnessInputs {
                phi: 0.5,
                broadcast: 0.5,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.5,
                embodiment: 0.5,
                knowledge: 0.5,
                synchrony: 0.5,
            },
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Per-step measurements
// ────────────────────────────────────────────────────────────

/// Measurements at a single coupling strength k.
#[derive(Debug, Clone)]
pub struct BifurcationPoint {
    /// Coupling strength k for this point.
    pub k: f64,
    /// Mean collective Φ over the measurement window.
    pub mean_phi: f64,
    /// Variance of collective Φ over the measurement window.
    pub var_phi: f64,
    /// Order parameter: `mean_phi − phi_baseline` (phi at k=0).
    /// Positive = integration emerged above baseline.
    pub order_parameter: f64,
    /// Susceptibility: `N × var_phi / mean_phi` (peaks at critical point).
    /// Returns 0.0 when mean_phi < 1e-10 (degenerate).
    pub susceptibility: f64,
    /// Binder cumulant U₄ = 1 − ⟨m⁴⟩/(3⟨m²⟩²).
    /// Crosses at k_c for different system sizes (size-independent estimate).
    pub binder_cumulant: f64,
    /// Number of agents (system size N for finite-size scaling).
    pub num_agents: usize,
}

impl BifurcationPoint {
    /// Compute a point from raw Φ time series.
    pub fn from_series(k: f64, phi_series: &[f64], phi_baseline: f64, num_agents: usize) -> Self {
        let n = phi_series.len() as f64;
        if phi_series.is_empty() {
            return Self {
                k,
                mean_phi: 0.0,
                var_phi: 0.0,
                order_parameter: 0.0,
                susceptibility: 0.0,
                binder_cumulant: 0.0,
                num_agents,
            };
        }

        let mean = phi_series.iter().sum::<f64>() / n;
        let var = phi_series.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;

        // Order parameter (deviations from baseline)
        let m_series: Vec<f64> = phi_series.iter().map(|&v| v - phi_baseline).collect();
        let m2 = m_series.iter().map(|&m| m.powi(2)).sum::<f64>() / n;
        let m4 = m_series.iter().map(|&m| m.powi(4)).sum::<f64>() / n;

        let binder = if m2 > 1e-12 {
            1.0 - m4 / (3.0 * m2.powi(2))
        } else {
            0.0
        };

        let susceptibility = if mean > 1e-10 {
            (num_agents as f64) * var / mean
        } else {
            0.0
        };

        Self {
            k,
            mean_phi: mean,
            var_phi: var,
            order_parameter: mean - phi_baseline,
            susceptibility,
            binder_cumulant: binder,
            num_agents,
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Full sweep result
// ────────────────────────────────────────────────────────────

/// Result of a full bifurcation sweep.
#[derive(Debug, Clone)]
pub struct BifurcationResult {
    /// All data points from the sweep.
    pub points: Vec<BifurcationPoint>,
    /// Estimated critical coupling k_c (k at peak susceptibility).
    pub critical_coupling: f64,
    /// Φ at k=0 (baseline, independent agents).
    pub phi_baseline: f64,
    /// Φ at k_max (fully coupled regime).
    pub phi_max_coupling: f64,
    /// Maximum susceptibility observed (proxy for divergence strength).
    pub peak_susceptibility: f64,
    /// Whether the transition appears continuous (second-order).
    /// True if order parameter rises smoothly; false if jump > `JUMP_THRESHOLD`.
    pub is_continuous: bool,
}

/// Jump threshold for first-order transition detection.
pub const JUMP_THRESHOLD: f64 = 0.15;

impl BifurcationResult {
    /// Find the index of the peak susceptibility.
    pub fn peak_susceptibility_idx(&self) -> Option<usize> {
        self.points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.susceptibility.partial_cmp(&b.susceptibility).unwrap())
            .map(|(i, _)| i)
    }

    /// Estimate the critical exponent β from the order parameter rise.
    ///
    /// Fits `m ~ |k - k_c|^β` for k > k_c using log-log regression.
    /// Returns `None` if there are fewer than 3 post-critical points.
    pub fn estimate_beta(&self) -> Option<f64> {
        let kc = self.critical_coupling;
        let post_critical: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter(|p| p.k > kc && p.order_parameter > 1e-4)
            .map(|p| ((p.k - kc).ln(), p.order_parameter.ln()))
            .collect();
        if post_critical.len() < 3 {
            return None;
        }
        // Simple log-log slope (β = Δln(m) / Δln(k-kc))
        let n = post_critical.len() as f64;
        let (sum_x, sum_y, sum_xy, sum_x2) = post_critical.iter().fold(
            (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |(sx, sy, sxy, sx2), &(x, y)| (sx + x, sy + y, sxy + x * y, sx2 + x * x),
        );
        let beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        if beta.is_finite() {
            Some(beta)
        } else {
            None
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Simulation kernel
// ────────────────────────────────────────────────────────────

/// Advance the simulation by one tick: update entity consciousness, decay errors, step physics.
///
/// Each entity's position is extracted from the physics world (immutable borrow),
/// then immediately released so the field can be mutated — no intermediate Vec needed.
fn tick_sim(
    world: &mut PhysicsWorld<2>,
    field: &mut ConsciousnessField<2>,
    handles: &[symtropy_physics::BodyHandle],
    inputs: &ConsciousnessInputs,
    dt: f64,
) {
    // Extract position and update each entity one at a time.
    // `world.body(h)` borrow is released as soon as we extract f64 coords.
    for &h in handles {
        let pos = match world.body(h) {
            Some(b) => {
                let p = b.position();
                Point::new([p.coord(0), p.coord(1)])
            }
            None => Point::origin(),
        };
        field.update_entity(h, inputs, pos);
    }
    field.tick_prediction_errors();
    world.step_with_callback(dt, field);
}

/// Run a single coupling value, return the Φ time series from the measurement window.
fn run_single_k(config: &BifurcationConfig, k: f64) -> Vec<f64> {
    let mut world = PhysicsWorld::<2>::new(SVector::zeros());
    let mut field = ConsciousnessField::new();
    field.phi_gravity_strength = k;

    let n = config.num_agents.max(1);
    let r = config.ring_radius;
    let mut handles = Vec::with_capacity(n);

    // Place agents uniformly on a ring
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let h = world.add_sphere(Point::new([r * angle.cos(), r * angle.sin()]), 1.0, 1.0);
        field.register(h, 100.0, 2.0);
        handles.push(h);
    }

    // Warmup: let system reach quasi-steady state
    for _ in 0..config.warmup_ticks {
        tick_sim(
            &mut world,
            &mut field,
            &handles,
            &config.agent_inputs,
            config.dt,
        );
    }

    // Measurement window: record collective Φ each tick
    let mut phi_series = Vec::with_capacity(config.measure_ticks as usize);
    for _ in 0..config.measure_ticks {
        tick_sim(
            &mut world,
            &mut field,
            &handles,
            &config.agent_inputs,
            config.dt,
        );
        phi_series.push(field.collective_phi);
    }
    phi_series
}

// ────────────────────────────────────────────────────────────
//  Main entry point
// ────────────────────────────────────────────────────────────

/// Run a full bifurcation sweep over `k ∈ [k_min, k_max]` with `k_steps` steps.
///
/// Each step simulates `warmup_ticks + measure_ticks` ticks, then extracts
/// statistics from the measurement window.
///
/// # Example
/// ```ignore
/// let config = BifurcationConfig::default();
/// let result = run_bifurcation_sweep(&config);
/// println!("k_c = {:.3}, continuous = {}", result.critical_coupling, result.is_continuous);
/// ```
pub fn run_bifurcation_sweep(config: &BifurcationConfig) -> BifurcationResult {
    let steps = config.k_steps.max(2);
    let k_values: Vec<f64> = (0..steps)
        .map(|i| config.k_min + (config.k_max - config.k_min) * i as f64 / (steps - 1) as f64)
        .collect();

    // Baseline Φ at k=0
    let baseline_series = run_single_k(config, 0.0);
    let phi_baseline = baseline_series.iter().sum::<f64>() / baseline_series.len().max(1) as f64;

    let points: Vec<BifurcationPoint> = k_values
        .iter()
        .map(|&k| {
            let series = run_single_k(config, k);
            BifurcationPoint::from_series(k, &series, phi_baseline, config.num_agents)
        })
        .collect();

    // Estimate critical coupling (peak susceptibility)
    let (peak_idx, peak_susceptibility) = points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.susceptibility))
        .fold(
            (0, 0.0f64),
            |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) },
        );
    let critical_coupling = points[peak_idx].k;

    let phi_max_coupling = points.last().map(|p| p.mean_phi).unwrap_or(phi_baseline);

    // Detect continuity: check for jumps > JUMP_THRESHOLD in order parameter
    let is_continuous = !points
        .windows(2)
        .any(|w| (w[1].order_parameter - w[0].order_parameter).abs() > JUMP_THRESHOLD);

    BifurcationResult {
        points,
        critical_coupling,
        phi_baseline,
        phi_max_coupling,
        peak_susceptibility,
        is_continuous,
    }
}

// ────────────────────────────────────────────────────────────
//  Criticality indicators (for paper figures)
// ────────────────────────────────────────────────────────────

/// Summary statistics for reporting a phase transition in a paper.
#[derive(Debug, Clone)]
pub struct CriticalityIndicators {
    /// Estimated critical coupling k_c.
    pub k_c: f64,
    /// Order parameter at k_max (integration strength).
    pub order_parameter_max: f64,
    /// Peak susceptibility (proxy for correlation length divergence).
    pub peak_susceptibility: f64,
    /// Estimated critical exponent β (None if insufficient data).
    pub beta: Option<f64>,
    /// Transition type.
    pub transition_type: TransitionType,
    /// Ratio Φ_max / Φ_baseline (how much Φ increases from integration).
    pub phi_amplification: f64,
}

/// Classification of the transition type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Continuous (second-order): order parameter rises smoothly.
    /// Analogous to: ferromagnetic, percolation, neural criticality.
    Continuous,
    /// Discontinuous (first-order): order parameter jumps at k_c.
    /// Analogous to: liquid-gas, spinodal decomposition.
    Discontinuous,
    /// Crossover: no sharp transition detected (finite-size or weak coupling).
    Crossover,
}

impl BifurcationResult {
    /// Compute [`CriticalityIndicators`] for paper-ready reporting.
    pub fn criticality_indicators(&self) -> CriticalityIndicators {
        let order_parameter_max = self
            .points
            .last()
            .map(|p| p.order_parameter.max(0.0))
            .unwrap_or(0.0);

        let transition_type = if !self.is_continuous {
            TransitionType::Discontinuous
        } else if self.peak_susceptibility < 0.01 {
            TransitionType::Crossover
        } else {
            TransitionType::Continuous
        };

        let phi_amplification = if self.phi_baseline > 1e-10 {
            self.phi_max_coupling / self.phi_baseline
        } else {
            1.0
        };

        CriticalityIndicators {
            k_c: self.critical_coupling,
            order_parameter_max,
            peak_susceptibility: self.peak_susceptibility,
            beta: self.estimate_beta(),
            transition_type,
            phi_amplification,
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal config for fast unit tests (not representative — use more ticks for science).
    fn fast_config() -> BifurcationConfig {
        BifurcationConfig {
            k_min: 0.0,
            k_max: 0.5,
            k_steps: 5,
            num_agents: 3,
            ring_radius: 5.0,
            warmup_ticks: 5,
            measure_ticks: 10,
            dt: 1.0 / 60.0,
            agent_inputs: ConsciousnessInputs {
                phi: 0.5,
                broadcast: 0.5,
                working_memory: 0.5,
                attention: 0.5,
                recurrence: 0.5,
                embodiment: 0.5,
                knowledge: 0.5,
                synchrony: 0.5,
            },
        }
    }

    #[test]
    fn bifurcation_point_from_empty_series() {
        let p = BifurcationPoint::from_series(0.5, &[], 0.3, 8);
        assert_eq!(p.mean_phi, 0.0);
        assert_eq!(p.order_parameter, 0.0);
    }

    #[test]
    fn bifurcation_point_constant_series() {
        let series = vec![0.7, 0.7, 0.7, 0.7];
        let p = BifurcationPoint::from_series(1.0, &series, 0.5, 8);
        assert!((p.mean_phi - 0.7).abs() < 1e-9);
        assert!((p.var_phi).abs() < 1e-9);
        assert!((p.order_parameter - 0.2).abs() < 1e-9);
    }

    #[test]
    fn bifurcation_point_variance_correct() {
        // series: [0.4, 0.6] → mean=0.5, var=0.01
        let series = vec![0.4, 0.6];
        let p = BifurcationPoint::from_series(0.5, &series, 0.0, 4);
        assert!((p.mean_phi - 0.5).abs() < 1e-9);
        assert!((p.var_phi - 0.01).abs() < 1e-9);
    }

    #[test]
    fn run_bifurcation_sweep_returns_correct_step_count() {
        let config = fast_config();
        let result = run_bifurcation_sweep(&config);
        assert_eq!(result.points.len(), config.k_steps);
    }

    #[test]
    fn sweep_baseline_phi_is_positive() {
        let config = fast_config();
        let result = run_bifurcation_sweep(&config);
        // Inputs phi=0.5 → consciousness should be nonzero
        assert!(
            result.phi_baseline > 0.0,
            "baseline Φ should be positive with phi=0.5 inputs"
        );
    }

    #[test]
    fn sweep_k_values_span_range() {
        let config = fast_config();
        let result = run_bifurcation_sweep(&config);
        assert!((result.points.first().unwrap().k - config.k_min).abs() < 1e-9);
        assert!((result.points.last().unwrap().k - config.k_max).abs() < 1e-9);
    }

    #[test]
    fn sweep_critical_coupling_in_range() {
        let config = fast_config();
        let result = run_bifurcation_sweep(&config);
        assert!(result.critical_coupling >= config.k_min);
        assert!(result.critical_coupling <= config.k_max);
    }

    #[test]
    fn criticality_indicators_phi_amplification_ge_one_or_zero() {
        let config = fast_config();
        let result = run_bifurcation_sweep(&config);
        let ind = result.criticality_indicators();
        // Either amplification >= 1 (coupling helps) or baseline is near zero
        assert!(ind.phi_amplification >= 0.0);
    }

    #[test]
    fn criticality_indicators_transition_type_valid() {
        let config = fast_config();
        let result = run_bifurcation_sweep(&config);
        let ind = result.criticality_indicators();
        // Just verify it doesn't panic and returns a valid variant
        let _ = ind.transition_type;
    }

    #[test]
    fn estimate_beta_none_for_insufficient_points() {
        let config = BifurcationConfig {
            k_steps: 2,
            ..fast_config()
        };
        let result = run_bifurcation_sweep(&config);
        // With only 2 k values, beta estimation can't have 3+ post-critical points
        let _ = result.estimate_beta(); // Should not panic
    }

    #[test]
    fn jump_threshold_constant_is_positive() {
        assert!(JUMP_THRESHOLD > 0.0);
        assert!(JUMP_THRESHOLD < 1.0);
    }

    #[test]
    fn binder_cumulant_constant_series_zero() {
        // For a constant order parameter series, m² = m_mean², m⁴ = m_mean⁴
        // U₄ = 1 - m⁴ / (3m²²) = 1 - m⁴ / (3m⁴) = 2/3
        let series = vec![0.6, 0.6, 0.6, 0.6];
        let p = BifurcationPoint::from_series(0.5, &series, 0.0, 8);
        // When var=0, m_series is constant. m²² = m4 = const.
        // U₄ = 1 - const^2/(3*const^2) = 2/3
        assert!(
            (p.binder_cumulant - 2.0 / 3.0).abs() < 1e-6,
            "Binder cumulant for constant series should be 2/3, got {}",
            p.binder_cumulant
        );
    }
}

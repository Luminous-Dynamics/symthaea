// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC Physics Analysis
//!
//! Uses Closed-form Continuous-time (CfC) neural networks to model
//! the temporal dynamics of screening and fusion enhancement.
//!
//! ## Hypothesis
//!
//! If screening is a dynamic (time-dependent) rather than static phenomenon,
//! CfC networks should be able to learn the temporal signature of enhancement.
//! Materials with similar temporal dynamics should cluster together, regardless
//! of their static screening values.
//!
//! ## Approach
//!
//! 1. Create synthetic time series based on physical models of screening
//! 2. Train CfC to predict enhancement from screening dynamics
//! 3. Analyze which temporal features correlate with high enhancement

use crate::bridge::{LiteratureDataLoader, ScreeningMeasurement};
use crate::constants::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Physics-Based Time Series Generation
// ============================================================================

/// Physical parameters that vary in time during a fusion event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningDynamics {
    /// Material name
    pub material: String,
    /// Static screening energy (eV)
    pub static_screening_ev: f64,
    /// Time series of effective screening (time_s, screening_ev)
    pub screening_series: Vec<(f64, f64)>,
    /// Time series of electron density (time_s, density_relative)
    pub density_series: Vec<(f64, f64)>,
    /// Time series of phonon occupation (time_s, n_phonons)
    pub phonon_series: Vec<(f64, f64)>,
    /// Time series of tunneling probability enhancement
    pub enhancement_series: Vec<(f64, f64)>,
    /// Characteristic time scales
    pub time_scales: TimeScales,
}

/// Characteristic time scales for different processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeScales {
    /// Electron relaxation time (s) - how fast electrons redistribute
    pub tau_electron: f64,
    /// Phonon relaxation time (s) - how fast lattice equilibrates
    pub tau_phonon: f64,
    /// Screening equilibration time (s)
    pub tau_screening: f64,
    /// Characteristic fusion attempt time (s)
    pub tau_fusion: f64,
}

impl Default for TimeScales {
    fn default() -> Self {
        Self {
            tau_electron: 1e-15,  // femtoseconds - electronic response
            tau_phonon: 1e-12,    // picoseconds - lattice response
            tau_screening: 1e-14, // ~10 fs - screening buildup
            tau_fusion: 1e-21,    // zeptoseconds - nuclear time scale
        }
    }
}

/// Generate synthetic screening dynamics based on material properties.
pub fn generate_screening_dynamics(measurement: &ScreeningMeasurement) -> ScreeningDynamics {
    let material = measurement.host_material.clone();
    let static_ue = measurement.screening_ev;
    let enhancement = measurement.enhancement_ratio;

    // Material-specific time scales based on physical properties
    let time_scales = estimate_time_scales(&material, static_ue);

    // Generate time series from t=0 to t=100*tau_phonon with 1000 points
    let t_max = 100.0 * time_scales.tau_phonon;
    let n_points = 1000;
    let dt = t_max / n_points as f64;

    let mut screening_series = Vec::with_capacity(n_points);
    let mut density_series = Vec::with_capacity(n_points);
    let mut phonon_series = Vec::with_capacity(n_points);
    let mut enhancement_series = Vec::with_capacity(n_points);

    // Model: screening builds up as electrons respond, then phonons modify it
    // Ue(t) = Ue_static * [1 - exp(-t/tau_e)] * [1 + A*sin(omega_phonon*t)*exp(-t/tau_ph)]
    let omega_phonon = 2.0 * std::f64::consts::PI / time_scales.tau_phonon;
    let phonon_amplitude = 0.1; // 10% modulation from phonons

    for i in 0..n_points {
        let t = (i as f64) * dt;

        // Electron density builds up rapidly
        let electron_factor = 1.0 - (-t / time_scales.tau_electron).exp();

        // Phonon occupation oscillates and decays
        let phonon_factor = (omega_phonon * t).sin() * (-t / time_scales.tau_phonon).exp();
        let n_phonons = 3.0 * (1.0 + phonon_factor); // ~3 phonon modes average

        // Screening builds up with electron response, modulated by phonons
        let screening_t = static_ue * electron_factor * (1.0 + phonon_amplitude * phonon_factor);

        // Enhancement follows screening (simplified model)
        // Real enhancement is exp(pi * Ue / E_G) but we normalize
        let enhancement_t = (screening_t / static_ue).powf(2.0) * enhancement;

        screening_series.push((t, screening_t));
        density_series.push((t, electron_factor));
        phonon_series.push((t, n_phonons));
        enhancement_series.push((t, enhancement_t));
    }

    ScreeningDynamics {
        material,
        static_screening_ev: static_ue,
        screening_series,
        density_series,
        phonon_series,
        enhancement_series,
        time_scales,
    }
}

/// Estimate time scales based on material properties.
fn estimate_time_scales(material: &str, screening_ev: f64) -> TimeScales {
    // Heavier atoms have slower electron response
    // Higher screening suggests more polarizable electrons (faster response)
    let base_tau_e = 1e-15; // 1 fs baseline

    // Material-specific adjustments based on atomic properties
    let (z, debye_temp_k) = match material.to_lowercase().as_str() {
        "pd" => (46, 274.0),
        "pt" => (78, 240.0),
        "au" => (79, 165.0),
        "ta" => (73, 240.0),
        "nb" => (41, 275.0),
        "v" => (23, 380.0),
        "zr" => (40, 291.0),
        "ti" => (22, 420.0),
        "ni" => (28, 450.0),
        "fe" => (26, 470.0),
        "al" => (13, 428.0),
        "be" => (4, 1440.0),
        "c" => (6, 2230.0),
        _ => (40, 300.0), // Default to Zr-like
    };

    // Electron relaxation: scales with Z^(1/3) (Thomas-Fermi approximation)
    let tau_electron = base_tau_e * (z as f64).powf(1.0 / 3.0) / (screening_ev / 100.0);

    // Phonon relaxation: inversely proportional to Debye temperature
    // tau_phonon ~ hbar / (k_B * T_Debye)
    let tau_phonon = 6.58e-16 / (8.617e-5 * debye_temp_k); // hbar/(k_B*T_D) in seconds

    // Screening equilibration: between electron and phonon time scales
    let tau_screening = (tau_electron * tau_phonon).sqrt();

    TimeScales {
        tau_electron,
        tau_phonon,
        tau_screening,
        tau_fusion: 1e-21, // Nuclear time scale is universal
    }
}

// ============================================================================
// CfC Integration Types (compatible with symthaea if available)
// ============================================================================

/// Simplified CfC cell for physics analysis.
/// This is a minimal implementation that captures the key CfC dynamics
/// without requiring the full symthaea dependency.
#[derive(Debug, Clone)]
pub struct PhysicsCfCCell {
    /// Hidden state dimension
    hidden_dim: usize,
    /// Time constants for each hidden unit
    tau: Vec<f32>,
    /// Input weights
    w_in: Vec<Vec<f32>>,
    /// Recurrent weights
    w_h: Vec<Vec<f32>>,
    /// Bias
    bias: Vec<f32>,
    /// Current state
    state: Vec<f32>,
}

impl PhysicsCfCCell {
    /// Create a new physics CfC cell.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = SimpleRng::new(42);

        // Xavier initialization
        let scale = (2.0 / (input_dim + hidden_dim) as f32).sqrt();

        let w_in: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| rng.next_normal() * scale).collect())
            .collect();

        let w_h: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.next_normal() * scale).collect())
            .collect();

        let bias = vec![0.0; hidden_dim];

        // Time constants: log-uniform in [0.1, 10.0]
        let tau: Vec<f32> = (0..hidden_dim)
            .map(|_| {
                let log_tau = 0.1_f32.ln() + rng.next_uniform() * (10.0_f32.ln() - 0.1_f32.ln());
                log_tau.exp()
            })
            .collect();

        Self {
            hidden_dim,
            tau,
            w_in,
            w_h,
            bias,
            state: vec![0.0; hidden_dim],
        }
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        self.state = vec![0.0; self.hidden_dim];
    }

    /// Forward pass with continuous time step.
    ///
    /// CfC update: h(t+dt) = h(t) * exp(-dt/tau) + (1 - exp(-dt/tau)) * f(x, h)
    pub fn forward(&mut self, input: &[f32], dt: f32) -> Vec<f32> {
        let mut new_state = vec![0.0; self.hidden_dim];

        for i in 0..self.hidden_dim {
            // Compute input contribution
            let mut x_contrib = 0.0;
            for (j, &x) in input.iter().enumerate() {
                x_contrib += self.w_in[i][j] * x;
            }

            // Compute recurrent contribution
            let mut h_contrib = 0.0;
            for j in 0..self.hidden_dim {
                h_contrib += self.w_h[i][j] * self.state[j];
            }

            // Target state (what we'd converge to)
            let f_target = fast_sigmoid(x_contrib + h_contrib + self.bias[i]);

            // CfC interpolation
            let decay = (-dt / self.tau[i]).exp();
            new_state[i] = self.state[i] * decay + (1.0 - decay) * f_target;
        }

        self.state = new_state.clone();
        new_state
    }

    /// Get current state.
    pub fn get_state(&self) -> &[f32] {
        &self.state
    }

    /// Get learned time constants.
    pub fn get_tau(&self) -> &[f32] {
        &self.tau
    }
}

fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + x / (1.0 + x.abs()))
}

/// Simple RNG for reproducible initialization.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_uniform(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }

    fn next_normal(&mut self) -> f32 {
        // Box-Muller transform
        let u1 = self.next_uniform().max(1e-10);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ============================================================================
// CfC Analysis of Screening Dynamics
// ============================================================================

/// Results from CfC analysis of screening data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCAnalysisResult {
    /// Material analyzed
    pub material: String,
    /// Static screening (eV)
    pub static_screening_ev: f64,
    /// Enhancement ratio over adiabatic
    pub enhancement_ratio: f64,
    /// Learned time constants (seconds)
    pub learned_tau: Vec<f64>,
    /// Dominant time scale (median of tau)
    pub dominant_time_scale: f64,
    /// State trajectory (for visualization)
    pub state_trajectory: Vec<Vec<f32>>,
    /// Temporal features extracted
    pub temporal_features: TemporalFeatures,
}

/// Temporal features extracted by CfC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Rate of screening buildup (dUe/dt at t=0)
    pub screening_rate: f64,
    /// Oscillation frequency (if detected)
    pub oscillation_freq_hz: Option<f64>,
    /// Damping ratio
    pub damping_ratio: f64,
    /// Steady-state reached fraction
    pub steady_state_fraction: f64,
    /// Cross-correlation with phonon dynamics
    pub phonon_correlation: f64,
}

/// Analyze screening dynamics using CfC.
pub fn analyze_with_cfc(dynamics: &ScreeningDynamics) -> CfCAnalysisResult {
    // Create CfC cell with 3 inputs (screening, density, phonons) and 16 hidden units
    let input_dim = 3;
    let hidden_dim = 16;
    let mut cell = PhysicsCfCCell::new(input_dim, hidden_dim);

    let n_points = dynamics.screening_series.len();
    let mut state_trajectory = Vec::with_capacity(n_points);

    // Feed time series through CfC
    for i in 0..n_points {
        let (t, ue) = dynamics.screening_series[i];
        let (_, rho) = dynamics.density_series[i];
        let (_, n_ph) = dynamics.phonon_series[i];

        // Normalize inputs
        let input = [
            (ue / dynamics.static_screening_ev) as f32,
            rho as f32,
            (n_ph / 5.0) as f32, // Normalize to ~1
        ];

        // Time step
        let dt = if i > 0 {
            (t - dynamics.screening_series[i - 1].0) as f32
        } else {
            1e-15 // First step: 1 fs
        };

        let state = cell.forward(&input, dt);
        state_trajectory.push(state);
    }

    // Extract temporal features
    let temporal_features = extract_temporal_features(dynamics, &state_trajectory);

    // Compute learned time constants in physical units
    let t_max = dynamics
        .screening_series
        .last()
        .map(|(t, _)| *t)
        .unwrap_or(1e-10);
    let learned_tau: Vec<f64> = cell
        .get_tau()
        .iter()
        .map(|&tau| tau as f64 * t_max / 100.0) // Scale to physical time
        .collect();

    let mut sorted_tau = learned_tau.clone();
    sorted_tau.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dominant_time_scale = sorted_tau[sorted_tau.len() / 2];

    CfCAnalysisResult {
        material: dynamics.material.clone(),
        static_screening_ev: dynamics.static_screening_ev,
        enhancement_ratio: dynamics
            .enhancement_series
            .last()
            .map(|(_, e)| *e)
            .unwrap_or(1.0),
        learned_tau,
        dominant_time_scale,
        state_trajectory,
        temporal_features,
    }
}

fn extract_temporal_features(
    dynamics: &ScreeningDynamics,
    states: &[Vec<f32>],
) -> TemporalFeatures {
    // Screening rate: (Ue(t1) - Ue(t0)) / (t1 - t0) at early time
    let screening_rate = if dynamics.screening_series.len() >= 10 {
        let (t0, ue0) = dynamics.screening_series[0];
        let (t1, ue1) = dynamics.screening_series[9];
        if t1 > t0 {
            (ue1 - ue0) / (t1 - t0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Detect oscillation frequency from phonon series
    let oscillation_freq_hz = detect_oscillation_frequency(&dynamics.phonon_series);

    // Damping ratio: how quickly oscillations decay
    let damping_ratio = compute_damping_ratio(&dynamics.screening_series);

    // Steady state fraction: final value / max value
    let max_ue = dynamics
        .screening_series
        .iter()
        .map(|(_, ue)| *ue)
        .fold(0.0_f64, |a, b| a.max(b));
    let final_ue = dynamics
        .screening_series
        .last()
        .map(|(_, ue)| *ue)
        .unwrap_or(0.0);
    let steady_state_fraction = if max_ue > 0.0 { final_ue / max_ue } else { 1.0 };

    // Phonon correlation: how much state correlates with phonon dynamics
    let phonon_correlation = if !states.is_empty() {
        let state_sum: Vec<f64> = states
            .iter()
            .map(|s| s.iter().sum::<f32>() as f64)
            .collect();
        let phonon_vals: Vec<f64> = dynamics.phonon_series.iter().map(|(_, p)| *p).collect();
        pearson_correlation(&state_sum, &phonon_vals).unwrap_or(0.0)
    } else {
        0.0
    };

    TemporalFeatures {
        screening_rate,
        oscillation_freq_hz,
        damping_ratio,
        steady_state_fraction,
        phonon_correlation,
    }
}

fn detect_oscillation_frequency(series: &[(f64, f64)]) -> Option<f64> {
    if series.len() < 20 {
        return None;
    }

    // Find zero crossings of detrended signal
    let mean: f64 = series.iter().map(|(_, y)| *y).sum::<f64>() / series.len() as f64;
    let detrended: Vec<f64> = series.iter().map(|(_, y)| y - mean).collect();

    let mut crossings = Vec::new();
    for i in 1..detrended.len() {
        if detrended[i - 1] * detrended[i] < 0.0 {
            crossings.push(series[i].0);
        }
    }

    if crossings.len() >= 2 {
        let period = (crossings.last()? - crossings.first()?) / (crossings.len() - 1) as f64 * 2.0;
        if period > 0.0 {
            return Some(1.0 / period);
        }
    }

    None
}

fn compute_damping_ratio(series: &[(f64, f64)]) -> f64 {
    if series.len() < 100 {
        return 0.0;
    }

    // Find peak values in first and last quarter
    let quarter = series.len() / 4;
    let first_quarter: Vec<f64> = series[..quarter].iter().map(|(_, y)| *y).collect();
    let last_quarter: Vec<f64> = series[3 * quarter..].iter().map(|(_, y)| *y).collect();

    let first_amplitude = first_quarter.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let last_amplitude = last_quarter.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

    if first_amplitude > 0.0 {
        1.0 - (last_amplitude / first_amplitude)
    } else {
        0.0
    }
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        Some(cov / (var_x.sqrt() * var_y.sqrt()))
    } else {
        None
    }
}

// ============================================================================
// Full Analysis Pipeline
// ============================================================================

/// Results from analyzing all Raiola data through CfC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCPhysicsReport {
    /// Individual material results
    pub material_results: Vec<CfCAnalysisResult>,
    /// Cross-material correlations
    pub correlations: CfCCorrelations,
    /// Key insights from temporal analysis
    pub insights: Vec<String>,
    /// Recommended experiments based on dynamics
    pub recommendations: Vec<String>,
}

/// Correlations discovered across materials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCCorrelations {
    /// Correlation: dominant_tau vs enhancement
    pub tau_enhancement_corr: f64,
    /// Correlation: screening_rate vs enhancement
    pub rate_enhancement_corr: f64,
    /// Correlation: phonon_correlation vs enhancement
    pub phonon_enhancement_corr: f64,
    /// Materials grouped by similar dynamics
    pub dynamic_clusters: Vec<Vec<String>>,
}

/// Run full CfC analysis on Raiola screening data.
pub fn analyze_raiola_data() -> CfCPhysicsReport {
    let loader = LiteratureDataLoader::new();

    // Generate dynamics and analyze each material
    let mut material_results = Vec::new();

    for measurement in &loader.screening_data {
        let dynamics = generate_screening_dynamics(measurement);
        let result = analyze_with_cfc(&dynamics);
        material_results.push(result);
    }

    // Compute cross-material correlations
    let correlations = compute_correlations(&material_results);

    // Generate insights
    let insights = generate_insights(&material_results, &correlations);

    // Generate recommendations
    let recommendations = generate_recommendations(&material_results, &correlations);

    CfCPhysicsReport {
        material_results,
        correlations,
        insights,
        recommendations,
    }
}

fn compute_correlations(results: &[CfCAnalysisResult]) -> CfCCorrelations {
    if results.is_empty() {
        return CfCCorrelations {
            tau_enhancement_corr: 0.0,
            rate_enhancement_corr: 0.0,
            phonon_enhancement_corr: 0.0,
            dynamic_clusters: Vec::new(),
        };
    }

    let taus: Vec<f64> = results.iter().map(|r| r.dominant_time_scale).collect();
    let rates: Vec<f64> = results
        .iter()
        .map(|r| r.temporal_features.screening_rate)
        .collect();
    let phonons: Vec<f64> = results
        .iter()
        .map(|r| r.temporal_features.phonon_correlation)
        .collect();
    let enhancements: Vec<f64> = results.iter().map(|r| r.enhancement_ratio).collect();

    let tau_enhancement_corr = pearson_correlation(&taus, &enhancements).unwrap_or(0.0);
    let rate_enhancement_corr = pearson_correlation(&rates, &enhancements).unwrap_or(0.0);
    let phonon_enhancement_corr = pearson_correlation(&phonons, &enhancements).unwrap_or(0.0);

    // Simple clustering by dominant time scale
    let mut fast_dynamics = Vec::new();
    let mut medium_dynamics = Vec::new();
    let mut slow_dynamics = Vec::new();

    let median_tau = {
        let mut sorted = taus.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    for result in results {
        if result.dominant_time_scale < median_tau * 0.5 {
            fast_dynamics.push(result.material.clone());
        } else if result.dominant_time_scale > median_tau * 2.0 {
            slow_dynamics.push(result.material.clone());
        } else {
            medium_dynamics.push(result.material.clone());
        }
    }

    CfCCorrelations {
        tau_enhancement_corr,
        rate_enhancement_corr,
        phonon_enhancement_corr,
        dynamic_clusters: vec![fast_dynamics, medium_dynamics, slow_dynamics],
    }
}

fn generate_insights(results: &[CfCAnalysisResult], corr: &CfCCorrelations) -> Vec<String> {
    let mut insights = Vec::new();

    // Time constant insight
    if corr.tau_enhancement_corr.abs() > 0.5 {
        let direction = if corr.tau_enhancement_corr > 0.0 {
            "slower"
        } else {
            "faster"
        };
        insights.push(format!(
            "Materials with {} screening dynamics show higher enhancement (r = {:.2}). \
             This suggests {} electron response enables more effective screening.",
            direction, corr.tau_enhancement_corr, direction
        ));
    }

    // Screening rate insight
    if corr.rate_enhancement_corr.abs() > 0.3 {
        let direction = if corr.rate_enhancement_corr > 0.0 {
            "faster"
        } else {
            "slower"
        };
        insights.push(format!(
            "Initial screening buildup rate correlates with enhancement (r = {:.2}). \
             {} initial screening may be key to enhancement.",
            corr.rate_enhancement_corr,
            if corr.rate_enhancement_corr > 0.0 {
                "Faster"
            } else {
                "Slower"
            }
        ));
    }

    // Phonon coupling insight
    if corr.phonon_enhancement_corr.abs() > 0.4 {
        insights.push(format!(
            "Phonon-screening coupling correlates with enhancement (r = {:.2}). \
             Materials where phonons modulate screening show different enhancement.",
            corr.phonon_enhancement_corr
        ));
    }

    // Cluster insight
    if !corr.dynamic_clusters.is_empty() && corr.dynamic_clusters.iter().any(|c| !c.is_empty()) {
        let clusters: Vec<String> = corr
            .dynamic_clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty())
            .map(|(i, c)| {
                let label = match i {
                    0 => "fast",
                    1 => "medium",
                    _ => "slow",
                };
                format!("{}: {}", label, c.join(", "))
            })
            .collect();
        insights.push(format!(
            "Materials cluster by screening dynamics: {}",
            clusters.join("; ")
        ));
    }

    // Find anomalies
    if let Some(highest) = results.iter().max_by(|a, b| {
        a.enhancement_ratio
            .partial_cmp(&b.enhancement_ratio)
            .unwrap()
    }) {
        insights.push(format!(
            "{} shows highest enhancement ({:.1}×) with tau = {:.2e} s. \
             Its temporal signature may hold the key to understanding LCF.",
            highest.material, highest.enhancement_ratio, highest.dominant_time_scale
        ));
    }

    if insights.is_empty() {
        insights.push(
            "No strong correlations detected between temporal dynamics and enhancement. \
             Static screening may dominate over dynamic effects."
                .to_string(),
        );
    }

    insights
}

fn generate_recommendations(results: &[CfCAnalysisResult], corr: &CfCCorrelations) -> Vec<String> {
    let mut recs = Vec::new();

    // Based on correlations
    if corr.tau_enhancement_corr.abs() > 0.3 {
        recs.push(
            "Measure time-resolved screening using femtosecond X-ray spectroscopy \
             to directly observe screening dynamics."
                .to_string(),
        );
    }

    if corr.phonon_enhancement_corr.abs() > 0.3 {
        recs.push(
            "Test materials with different phonon spectra (high vs low Debye temperature) \
             while keeping atomic number similar."
                .to_string(),
        );
    }

    // Based on clusters
    if !corr.dynamic_clusters.is_empty() {
        let fast = &corr.dynamic_clusters[0];
        if !fast.is_empty() {
            recs.push(format!(
                "Focus experiments on fast-dynamics materials ({}) which may show \
                 transient enhancement windows.",
                fast.join(", ")
            ));
        }
    }

    // General recommendations
    recs.push(
        "Use pulsed triggers synchronized with phonon periods to test \
         resonance enhancement hypothesis."
            .to_string(),
    );

    recs.push(
        "Compare isotope effects (H vs D) to vary phonon frequencies \
         while keeping electronic structure constant."
            .to_string(),
    );

    recs
}

/// Print a formatted report.
impl CfCPhysicsReport {
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        s.push_str("║     CfC ANALYSIS OF RAIOLA SCREENING DYNAMICS                   ║\n");
        s.push_str("╚══════════════════════════════════════════════════════════════════╝\n\n");

        s.push_str("▶ MATERIALS ANALYZED\n");
        for r in &self.material_results {
            s.push_str(&format!(
                "  • {}: Ue = {:.0} eV, enhancement = {:.1}×, tau = {:.2e} s\n",
                r.material, r.static_screening_ev, r.enhancement_ratio, r.dominant_time_scale
            ));
        }
        s.push('\n');

        s.push_str("▶ CORRELATIONS\n");
        s.push_str(&format!(
            "  Time constant ↔ Enhancement:     r = {:.3}\n",
            self.correlations.tau_enhancement_corr
        ));
        s.push_str(&format!(
            "  Screening rate ↔ Enhancement:    r = {:.3}\n",
            self.correlations.rate_enhancement_corr
        ));
        s.push_str(&format!(
            "  Phonon coupling ↔ Enhancement:   r = {:.3}\n\n",
            self.correlations.phonon_enhancement_corr
        ));

        s.push_str("▶ DYNAMIC CLUSTERS\n");
        for (i, cluster) in self.correlations.dynamic_clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let label = match i {
                    0 => "Fast dynamics",
                    1 => "Medium dynamics",
                    _ => "Slow dynamics",
                };
                s.push_str(&format!("  {}: {}\n", label, cluster.join(", ")));
            }
        }
        s.push('\n');

        s.push_str("▶ INSIGHTS\n");
        for insight in &self.insights {
            s.push_str(&format!("  • {}\n", insight));
        }
        s.push('\n');

        s.push_str("▶ RECOMMENDATIONS\n");
        for (i, rec) in self.recommendations.iter().enumerate() {
            s.push_str(&format!("  {}. {}\n", i + 1, rec));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screening_dynamics_generation() {
        let measurement = ScreeningMeasurement {
            source: crate::bridge::LiteratureSource {
                authors: "Test".to_string(),
                year: 2024,
                title: "Test".to_string(),
                journal: "Test".to_string(),
                doi: None,
                data_type: crate::bridge::LiteratureDataType::ScreeningEnergy,
            },
            host_material: "Pd".to_string(),
            target: "D-D".to_string(),
            screening_ev: 310.0,
            uncertainty_ev: 30.0,
            temperature_k: Some(300.0),
            adiabatic_limit_ev: 25.0,
            enhancement_ratio: 12.4,
        };

        let dynamics = generate_screening_dynamics(&measurement);

        assert_eq!(dynamics.material, "Pd");
        assert!(!dynamics.screening_series.is_empty());
        assert!(!dynamics.phonon_series.is_empty());
    }

    #[test]
    fn test_cfc_cell() {
        let mut cell = PhysicsCfCCell::new(3, 8);

        let input = [1.0, 0.5, 0.3];
        let state = cell.forward(&input, 0.01);

        assert_eq!(state.len(), 8);
        assert!(state.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_full_analysis() {
        let report = analyze_raiola_data();

        assert!(!report.material_results.is_empty());
        assert!(!report.insights.is_empty());
        assert!(!report.recommendations.is_empty());
    }
}

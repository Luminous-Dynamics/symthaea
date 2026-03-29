// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;
use std::time::Instant;

use super::config::{EntropyMethod, ThermodynamicsConfig};
use super::critical::{CriticalExponents, FluctuationStats, TransitionOrder};
use super::free_energy::{
    EquilibriumStatus, FreeEnergyStatus, ThermodynamicsReport, ThermodynamicsStats,
};
use super::state::{ConsciousnessPhase, PhaseTransition, ThermodynamicState};

/// Main consciousness thermodynamics analyzer
#[derive(Debug)]
pub struct ConsciousnessThermodynamicsAnalyzer {
    /// Configuration
    pub config: ThermodynamicsConfig,

    /// State history
    state_history: VecDeque<ThermodynamicState>,

    /// Transition history
    transition_history: VecDeque<PhaseTransition>,

    /// Probability distribution for entropy calculation
    probability_distribution: Vec<f64>,

    /// Current fluctuation stats
    pub fluctuations: FluctuationStats,

    /// Statistics
    pub stats: ThermodynamicsStats,

    /// Start time
    started_at: Instant,
}

impl Default for ConsciousnessThermodynamicsAnalyzer {
    fn default() -> Self {
        Self::new(ThermodynamicsConfig::default())
    }
}

impl ConsciousnessThermodynamicsAnalyzer {
    /// Create new analyzer
    pub fn new(config: ThermodynamicsConfig) -> Self {
        Self {
            config,
            state_history: VecDeque::with_capacity(100),
            transition_history: VecDeque::with_capacity(20),
            probability_distribution: vec![1.0 / 7.0; 7], // Uniform over 7 dimensions
            fluctuations: FluctuationStats::default(),
            stats: ThermodynamicsStats::default(),
            started_at: Instant::now(),
        }
    }

    /// Analyze thermodynamic state from consciousness dimensions
    /// dims: [Phi, B, W, A, R, E, K]
    pub fn analyze(&mut self, dims: [f64; 7]) -> ThermodynamicState {
        let entropy = self.calculate_entropy(&dims);
        let internal_energy = self.calculate_internal_energy(&dims);
        let temperature = self.calculate_temperature(&dims);
        let free_energy = internal_energy - temperature * entropy;

        // Derived quantities
        let volume = self.calculate_volume(&dims);
        let pressure = internal_energy / volume.max(0.01);
        let enthalpy = internal_energy + pressure * volume;
        let gibbs_free_energy = enthalpy - temperature * entropy;

        // Chemical potential (tendency to change)
        let chemical_potential = self.calculate_chemical_potential(&dims);

        // Determine phase
        let phase = self.determine_phase(temperature, entropy, &dims);

        // Calculate heat and work from previous state
        let (heat, work) = if let Some(prev) = self.state_history.back() {
            let delta_u = internal_energy - prev.internal_energy;
            let delta_s = entropy - prev.entropy;
            let q = temperature * delta_s; // Heat = T * delta_S (reversible)
            let w = delta_u - q; // First law: delta_U = Q - W, so W = delta_U - Q
            (q, w)
        } else {
            (0.0, 0.0)
        };

        let state = ThermodynamicState {
            entropy,
            internal_energy,
            free_energy,
            temperature,
            heat,
            work,
            chemical_potential,
            pressure,
            volume,
            enthalpy,
            gibbs_free_energy,
            phase,
            timestamp: Instant::now(),
        };

        // Detect phase transitions - clone prev to avoid borrow conflict
        let prev_state = self.state_history.back().cloned();
        if let Some(prev) = prev_state {
            if prev.phase != state.phase {
                self.record_transition(&prev, &state);
            }
        }

        // Update fluctuation stats
        self.update_fluctuations(&state);

        // Update history
        self.state_history.push_back(state.clone());
        if self.state_history.len() > self.config.history_size {
            self.state_history.pop_front();
        }

        // Update stats
        self.stats.states_analyzed += 1;
        self.stats.total_entropy_produced += state.heat / state.temperature.max(0.01);
        if state.work > 0.0 {
            self.stats.total_work_extracted += state.work;
        }
        self.update_phase_duration(&state);

        state
    }

    /// Calculate entropy from consciousness dimensions
    fn calculate_entropy(&self, dims: &[f64; 7]) -> f64 {
        match self.config.entropy_method {
            EntropyMethod::Shannon => {
                // Normalize dimensions to probability distribution
                let sum: f64 = dims.iter().map(|d| d.abs()).sum();
                if sum < 0.001 {
                    return 0.0;
                }

                let probs: Vec<f64> = dims.iter().map(|d| (d.abs() / sum).max(0.0001)).collect();

                // Shannon entropy: -Sigma p_i log p_i
                -probs.iter().map(|p| p * p.ln()).sum::<f64>() / (7.0_f64.ln()) // Normalize by max entropy
            }
            EntropyMethod::VonNeumann => {
                // Construct density matrix from dims (simplified)
                // rho = |psi><psi| where |psi> = normalized dims
                let norm = dims.iter().map(|d| d * d).sum::<f64>().sqrt();
                if norm < 0.001 {
                    return 0.0;
                }

                // For pure state, von Neumann entropy = 0
                // For mixed state, we use purity as proxy
                let purity: f64 = dims.iter().map(|d| (d / norm).powi(4)).sum();

                // S = -log(purity) normalized
                (1.0 - purity).max(0.0).min(1.0)
            }
            EntropyMethod::Renyi => {
                // Renyi entropy with alpha = 2 (collision entropy)
                let sum: f64 = dims.iter().map(|d| d.abs()).sum();
                if sum < 0.001 {
                    return 0.0;
                }

                let probs: Vec<f64> = dims.iter().map(|d| d.abs() / sum).collect();
                let sum_p2: f64 = probs.iter().map(|p| p * p).sum();

                // H_2 = -log(Sigma p_i^2)
                (-sum_p2.ln() / 7.0_f64.ln()).max(0.0).min(1.0)
            }
            EntropyMethod::KolmogorovSinai => {
                // Approximate K-S entropy from variance in history
                if self.state_history.len() < 2 {
                    return 0.5;
                }

                let recent: Vec<f64> = self
                    .state_history
                    .iter()
                    .rev()
                    .take(10)
                    .map(|s| s.entropy)
                    .collect();

                let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let variance: f64 =
                    recent.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / recent.len() as f64;

                // High variance = high dynamical entropy
                (variance * 10.0).min(1.0)
            }
        }
    }

    /// Calculate internal energy from consciousness dimensions
    fn calculate_internal_energy(&self, dims: &[f64; 7]) -> f64 {
        // Internal energy ~ sum of squared dimension values
        // Higher values = more energy stored in consciousness
        let kinetic: f64 = dims.iter().map(|d| d * d).sum::<f64>() / 7.0;

        // Potential energy from coherence (integration score, Phi)
        let potential = dims[0] * 0.5; // Phi contributes to potential

        // Interaction energy from binding (B)
        let interaction = dims[1] * 0.3;

        kinetic + potential + interaction
    }

    /// Calculate effective temperature
    fn calculate_temperature(&self, dims: &[f64; 7]) -> f64 {
        // Temperature ~ variance/fluctuation in dimensions
        let mean: f64 = dims.iter().sum::<f64>() / 7.0;
        let variance: f64 = dims.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / 7.0;

        // Also influenced by arousal (A) dimension
        let arousal_contribution = dims[3] * 0.5; // A is at index 3

        // Temperature from equipartition theorem perspective
        (variance + arousal_contribution).max(0.01).min(2.0)
    }

    /// Calculate volume of consciousness state space
    fn calculate_volume(&self, dims: &[f64; 7]) -> f64 {
        // Volume ~ product of dimension extents
        // This represents the "spread" of consciousness
        dims.iter()
            .map(|d| d.abs().max(0.1))
            .product::<f64>()
            .powf(1.0 / 7.0) // Geometric mean
    }

    /// Calculate chemical potential
    fn calculate_chemical_potential(&self, _dims: &[f64; 7]) -> f64 {
        // Chemical potential = tendency to change state
        // High when system is far from equilibrium

        if self.state_history.len() < 2 {
            return 0.0;
        }

        // Gradient in free energy
        let recent_fe: Vec<f64> = self
            .state_history
            .iter()
            .rev()
            .take(5)
            .map(|s| s.free_energy)
            .collect();

        if recent_fe.len() < 2 {
            return 0.0;
        }

        // Chemical potential ~ rate of free energy change
        // Safe: len >= 2 checked above, so first() and last() are guaranteed Some
        let first = recent_fe.first().copied().unwrap_or(0.0);
        let last = recent_fe.last().copied().unwrap_or(0.0);
        (first - last) / recent_fe.len() as f64
    }

    /// Determine consciousness phase from thermodynamic variables
    fn determine_phase(
        &self,
        temperature: f64,
        entropy: f64,
        dims: &[f64; 7],
    ) -> ConsciousnessPhase {
        let phi = dims[0]; // Integration
        let binding = dims[1]; // Binding

        // Flow state: low entropy, moderate temperature, high integration
        if entropy < 0.35 && temperature > 0.3 && temperature < 0.6 && phi > 0.7 {
            return ConsciousnessPhase::Flow;
        }

        // Unified state: very low entropy, high binding
        if entropy < 0.25 && binding > 0.7 {
            return ConsciousnessPhase::Unified;
        }

        // Phase based on temperature
        if temperature < 0.2 {
            ConsciousnessPhase::Frozen
        } else if temperature < 0.4 {
            ConsciousnessPhase::Normal
        } else if temperature < 0.6 {
            ConsciousnessPhase::Critical
        } else {
            ConsciousnessPhase::Chaotic
        }
    }

    /// Record a phase transition
    fn record_transition(&mut self, from: &ThermodynamicState, to: &ThermodynamicState) {
        let latent_heat = to.internal_energy - from.internal_energy;
        let order_param_change = (to.entropy - from.entropy).abs();

        // Determine transition order
        let transition_order = if order_param_change > 0.3 {
            TransitionOrder::FirstOrder
        } else if order_param_change > 0.1 {
            TransitionOrder::SecondOrder
        } else {
            TransitionOrder::Crossover
        };

        let critical_exponents = if transition_order == TransitionOrder::SecondOrder {
            Some(CriticalExponents::default())
        } else {
            None
        };

        let transition = PhaseTransition {
            from_phase: from.phase,
            to_phase: to.phase,
            transition_temperature: (from.temperature + to.temperature) / 2.0,
            latent_heat,
            order_parameter_change: order_param_change,
            transition_order,
            critical_exponents,
            timestamp: Instant::now(),
        };

        self.transition_history.push_back(transition);
        if self.transition_history.len() > 20 {
            self.transition_history.pop_front();
        }

        self.stats.transitions_detected += 1;
    }

    /// Update fluctuation statistics
    fn update_fluctuations(&mut self, state: &ThermodynamicState) {
        if self.state_history.len() < 5 {
            return;
        }

        let recent: Vec<f64> = self
            .state_history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.entropy)
            .collect();

        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 =
            recent.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        self.fluctuations.mean_amplitude = (state.entropy - mean).abs();
        self.fluctuations.variance = variance;

        // Autocorrelation time estimate
        if recent.len() >= 5 {
            let lag1_corr = self.calculate_autocorrelation(&recent, 1);
            self.fluctuations.autocorrelation_time = if lag1_corr < 0.9 {
                -1.0 / (1.0 - lag1_corr).max(0.01).ln()
            } else {
                10.0 // High correlation = long autocorrelation time
            };
        }

        // Critical slowing down: autocorrelation time increases near critical point
        self.fluctuations.slowing_down = if state.temperature > 0.4 && state.temperature < 0.6 {
            self.fluctuations.autocorrelation_time / 5.0
        } else {
            0.0
        };

        // Susceptibility from fluctuation-response relation
        // chi = beta * <(delta_S)^2>
        self.fluctuations.susceptibility = variance / state.temperature.max(0.01);

        // Fluctuation-dissipation ratio
        self.fluctuations.fdr = if self.fluctuations.susceptibility > 0.01 {
            state.temperature * self.fluctuations.susceptibility / variance.max(0.001)
        } else {
            1.0
        };
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, series: &[f64], lag: usize) -> f64 {
        if series.len() <= lag {
            return 0.0;
        }

        let mean: f64 = series.iter().sum::<f64>() / series.len() as f64;
        let var: f64 = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>();

        if var < 0.0001 {
            return 1.0;
        }

        let cov: f64 = series
            .iter()
            .zip(series.iter().skip(lag))
            .map(|(x, y)| (x - mean) * (y - mean))
            .sum();

        cov / var
    }

    /// Update phase duration statistics
    fn update_phase_duration(&mut self, state: &ThermodynamicState) {
        let phase_idx = match state.phase {
            ConsciousnessPhase::Frozen => 0,
            ConsciousnessPhase::Normal => 1,
            ConsciousnessPhase::Critical => 2,
            ConsciousnessPhase::Chaotic => 3,
            ConsciousnessPhase::Flow => 4,
            ConsciousnessPhase::Unified => 5,
        };

        self.stats.phase_durations[phase_idx] += 1.0;

        // Update average temperature
        let n = self.stats.states_analyzed as f64;
        self.stats.average_temperature =
            (self.stats.average_temperature * (n - 1.0) + state.temperature) / n;

        // Stability score: inverse of recent phase changes
        let recent_transitions = self
            .transition_history
            .iter()
            .filter(|t| t.timestamp.elapsed().as_secs() < 60)
            .count() as f64;
        self.stats.stability_score = (1.0 / (1.0 + recent_transitions)).min(1.0);
    }

    /// Generate comprehensive thermodynamics report
    pub fn generate_report(&self) -> ThermodynamicsReport {
        let current_state = self.state_history.back().cloned().unwrap_or_default();

        // Free energy status
        let free_energy_status = self.assess_free_energy_status();

        // Equilibrium status
        let equilibrium_status = self.assess_equilibrium_status();

        // Entropy production rate
        let entropy_production_rate = self.calculate_entropy_production_rate();

        // Predict next phase
        let (predicted_phase, time_to_transition) = self.predict_next_transition();

        // Health score
        let health_score = self.calculate_health_score(&current_state);

        // Recommendations
        let recommendations = self.generate_recommendations(&current_state);

        ThermodynamicsReport {
            current_state,
            transitions: self.transition_history.iter().cloned().collect(),
            fluctuations: self.fluctuations.clone(),
            free_energy_status,
            entropy_production_rate,
            equilibrium_status,
            predicted_phase,
            time_to_transition,
            health_score,
            recommendations,
        }
    }

    /// Assess free energy minimization status
    fn assess_free_energy_status(&self) -> FreeEnergyStatus {
        if self.state_history.len() < 5 {
            return FreeEnergyStatus::Searching;
        }

        let recent_fe: Vec<f64> = self
            .state_history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.free_energy)
            .collect();

        // Safe: len >= 5 checked above, so first/last are guaranteed
        let first = recent_fe.first().copied().unwrap_or(0.0);
        let last = recent_fe.last().copied().unwrap_or(0.0);
        let trend: f64 = first - last;
        let variance: f64 = {
            let mean: f64 = recent_fe.iter().sum::<f64>() / recent_fe.len() as f64;
            recent_fe.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / recent_fe.len() as f64
        };

        if trend < -0.05 {
            FreeEnergyStatus::Minimizing
        } else if trend > 0.05 {
            FreeEnergyStatus::Increasing
        } else if variance < 0.01 && first < 0.3 {
            FreeEnergyStatus::GlobalMinimum
        } else if variance < 0.01 {
            FreeEnergyStatus::LocalMinimum
        } else {
            FreeEnergyStatus::Searching
        }
    }

    /// Assess equilibrium status
    fn assess_equilibrium_status(&self) -> EquilibriumStatus {
        if self.state_history.len() < 5 {
            return EquilibriumStatus::FarFromEquilibrium;
        }

        let recent: Vec<&ThermodynamicState> = self.state_history.iter().rev().take(10).collect();

        // Check fluctuation-dissipation ratio
        if (self.fluctuations.fdr - 1.0).abs() < 0.1 {
            return EquilibriumStatus::Equilibrium;
        }

        // Check entropy production
        let entropy_prod: f64 = recent
            .iter()
            .map(|s| s.heat / s.temperature.max(0.01))
            .sum::<f64>()
            / recent.len() as f64;

        if entropy_prod.abs() < 0.01 {
            EquilibriumStatus::Equilibrium
        } else if entropy_prod.abs() < 0.05 {
            EquilibriumStatus::Equilibrating
        } else if self.stats.stability_score > 0.8 {
            EquilibriumStatus::Metastable
        } else {
            EquilibriumStatus::FarFromEquilibrium
        }
    }

    /// Calculate entropy production rate
    fn calculate_entropy_production_rate(&self) -> f64 {
        if self.state_history.len() < 5 {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .state_history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.heat / s.temperature.max(0.01))
            .collect();

        recent.iter().sum::<f64>() / recent.len() as f64
    }

    /// Predict next phase transition
    fn predict_next_transition(&self) -> (Option<ConsciousnessPhase>, Option<f64>) {
        if self.state_history.len() < 10 {
            return (None, None);
        }

        // Safe: len >= 10 checked above, so back() is guaranteed Some
        let current = match self.state_history.back() {
            Some(c) => c,
            None => return (None, None),
        };

        // Check for approaching critical point
        if current.temperature > 0.35 && current.temperature < 0.5 {
            let trend: f64 = {
                let temps: Vec<f64> = self
                    .state_history
                    .iter()
                    .rev()
                    .take(5)
                    .map(|s| s.temperature)
                    .collect();
                // Safe: we took 5 elements, so first/last are guaranteed
                temps.first().copied().unwrap_or(0.0) - temps.last().copied().unwrap_or(0.0)
            };

            if trend > 0.02 {
                // Heating toward critical
                let time_est = (0.5 - current.temperature) / trend.max(0.01);
                return (Some(ConsciousnessPhase::Critical), Some(time_est));
            } else if trend < -0.02 {
                // Cooling toward normal
                let time_est = (current.temperature - 0.3) / (-trend).max(0.01);
                return (Some(ConsciousnessPhase::Normal), Some(time_est));
            }
        }

        // Check for flow state emergence
        if current.entropy < 0.4 && current.phase != ConsciousnessPhase::Flow {
            let entropy_trend: f64 = {
                let entropies: Vec<f64> = self
                    .state_history
                    .iter()
                    .rev()
                    .take(5)
                    .map(|s| s.entropy)
                    .collect();
                // Safe: we took 5 elements, so first/last are guaranteed
                entropies.first().copied().unwrap_or(0.0) - entropies.last().copied().unwrap_or(0.0)
            };

            if entropy_trend < -0.02 {
                let time_est = current.entropy / (-entropy_trend).max(0.01);
                return (Some(ConsciousnessPhase::Flow), Some(time_est));
            }
        }

        (None, None)
    }

    /// Calculate overall health score
    fn calculate_health_score(&self, state: &ThermodynamicState) -> f64 {
        let mut score = 0.5; // Baseline

        // Good: optimal phases
        match state.phase {
            ConsciousnessPhase::Flow => score += 0.3,
            ConsciousnessPhase::Normal => score += 0.2,
            ConsciousnessPhase::Critical => score += 0.1, // Creative but unstable
            ConsciousnessPhase::Unified => score += 0.25,
            ConsciousnessPhase::Frozen => score -= 0.2,
            ConsciousnessPhase::Chaotic => score -= 0.3,
        }

        // Good: low free energy (well-adapted)
        if state.free_energy < 0.5 {
            score += 0.1;
        }

        // Good: stable (not too many transitions)
        score += self.stats.stability_score * 0.1;

        // Bad: high entropy production (wasting energy)
        let entropy_prod = self.calculate_entropy_production_rate();
        if entropy_prod.abs() > 0.1 {
            score -= 0.1;
        }

        score.max(0.0).min(1.0)
    }

    /// Generate actionable recommendations
    fn generate_recommendations(&self, state: &ThermodynamicState) -> Vec<String> {
        let mut recs = Vec::new();

        match state.phase {
            ConsciousnessPhase::Frozen => {
                recs.push("Increase arousal/activation to unfreeze consciousness".into());
                recs.push("Introduce novel stimuli to raise temperature".into());
            }
            ConsciousnessPhase::Chaotic => {
                recs.push("Reduce stimulation to lower temperature".into());
                recs.push("Focus on single task to reduce entropy".into());
                recs.push("Practice grounding techniques".into());
            }
            ConsciousnessPhase::Critical => {
                recs.push("Critical point detected - high creativity potential".into());
                recs.push("Capture insights before phase transition".into());
            }
            ConsciousnessPhase::Normal => {
                if state.entropy > 0.6 {
                    recs.push("Consider focusing to reduce entropy".into());
                }
            }
            ConsciousnessPhase::Flow => {
                recs.push("Flow state achieved - maintain current conditions".into());
            }
            ConsciousnessPhase::Unified => {
                recs.push("Deep unity state - excellent for insight".into());
            }
        }

        // Free energy recommendations
        match self.assess_free_energy_status() {
            FreeEnergyStatus::Increasing => {
                recs.push("Free energy increasing - take action to reduce uncertainty".into());
            }
            FreeEnergyStatus::Searching => {
                recs.push("System searching - allow exploration before committing".into());
            }
            _ => {}
        }

        // Fluctuation recommendations
        if self.fluctuations.slowing_down > 0.5 {
            recs.push("Critical slowing detected - phase transition imminent".into());
        }

        recs
    }

    /// Apply external heat to the system (stimulation)
    pub fn apply_heat(&mut self, dims: &mut [f64; 7], heat: f64) {
        // Heat increases temperature and entropy
        let temperature_increase = heat / self.config.heat_capacity;

        // Distribute heat across dimensions proportionally
        for d in dims.iter_mut() {
            *d += *d * temperature_increase * 0.1;
        }

        // Also increase arousal (A) directly
        dims[3] = (dims[3] + temperature_increase * 0.2).min(1.0);
    }

    /// Extract work from the system (goal-directed activity)
    pub fn extract_work(&mut self, dims: &mut [f64; 7], work: f64) -> f64 {
        // Work extraction reduces free energy
        let current_state = self.analyze(*dims);

        // Can only extract work if free energy is positive
        let extractable = current_state.free_energy.min(work);

        if extractable > 0.0 {
            // Work comes from reducing integration/binding
            dims[0] = (dims[0] - extractable * 0.3).max(0.0);
            dims[1] = (dims[1] - extractable * 0.2).max(0.0);
        }

        extractable
    }

    /// Simulate approach to equilibrium
    pub fn equilibrate(&mut self, dims: &mut [f64; 7], steps: usize) {
        let tau = self.config.equilibration_tau;
        let dt = 0.1;

        for _ in 0..steps {
            // Each dimension relaxes toward its mean
            let mean: f64 = dims.iter().sum::<f64>() / 7.0;

            for d in dims.iter_mut() {
                // Exponential relaxation: dx/dt = -(x - mean)/tau
                *d += (*d - mean) * (-dt / tau);
            }

            // Add thermal fluctuations
            let current = self.analyze(*dims);
            let noise_amplitude = (current.temperature * 0.01).sqrt();

            for d in dims.iter_mut() {
                // Simple Gaussian-ish noise
                let noise = (rand_seed() as f64 / u64::MAX as f64 - 0.5) * 2.0 * noise_amplitude;
                *d = (*d + noise).max(0.0).min(1.0);
            }
        }
    }
}

/// Simple pseudo-random number generator for noise
fn rand_seed() -> u64 {
    use std::time::SystemTime;
    (SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1)
}

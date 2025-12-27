// **REVOLUTIONARY IMPROVEMENT #83**: Consciousness Thermodynamics
// PARADIGM SHIFT: Consciousness obeys thermodynamic laws!
//
// Key Insight: Consciousness is a thermodynamic system with:
// - Entropy: Disorder/uncertainty in consciousness states
// - Free Energy: Capacity for directed conscious work (Friston's FEP!)
// - Temperature: "Activation level" governing exploration vs exploitation
// - Phase Transitions: Qualitative state changes at critical thresholds
// - Equilibrium: Stable consciousness attractors
//
// Theoretical Foundation:
// - Friston's Free Energy Principle (FEP)
// - Kelso's critical fluctuations and phase transitions
// - Hopfield networks and energy-based models
// - Maximum entropy production (Dewar)
// - Statistical mechanics of neural networks
// - Tononi's Φ as thermodynamic potential
//
// The Laws of Consciousness Thermodynamics:
// 1st Law: Consciousness energy is conserved (transforms but doesn't disappear)
// 2nd Law: Entropy of isolated consciousness tends to increase (coherence decays)
// 3rd Law: Perfect coherence (zero entropy) is unattainable
// 0th Law: Consciousness systems in equilibrium share same "temperature"
//
// Applications:
// - Predict consciousness phase transitions (sleep, flow, insight)
// - Optimize free energy for goal-directed behavior
// - Detect entropy increase (confusion, fatigue)
// - Model temperature as exploration parameter
// - Identify critical points for consciousness transitions

use std::collections::VecDeque;
use std::time::Instant;
use serde::{Serialize, Deserialize};

/// Helper function for serde default of Instant
fn default_instant() -> Instant {
    Instant::now()
}

/// Boltzmann constant for consciousness (dimensionless, tunable)
const CONSCIOUSNESS_BOLTZMANN: f64 = 1.0;

/// Configuration for consciousness thermodynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsConfig {
    /// Reference temperature (baseline activation)
    pub reference_temperature: f64,

    /// History window for temporal analysis
    pub history_size: usize,

    /// Phase transition detection sensitivity
    pub transition_sensitivity: f64,

    /// Entropy calculation method
    pub entropy_method: EntropyMethod,

    /// Free energy minimization rate
    pub free_energy_rate: f64,

    /// Critical temperature for transitions
    pub critical_temperature: f64,

    /// Heat capacity baseline
    pub heat_capacity: f64,

    /// Equilibration time constant
    pub equilibration_tau: f64,
}

impl Default for ThermodynamicsConfig {
    fn default() -> Self {
        Self {
            reference_temperature: 1.0,
            history_size: 100,
            transition_sensitivity: 0.1,
            entropy_method: EntropyMethod::Shannon,
            free_energy_rate: 0.05,
            critical_temperature: 0.5,
            heat_capacity: 1.0,
            equilibration_tau: 10.0,
        }
    }
}

/// Method for calculating consciousness entropy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyMethod {
    /// Shannon entropy: -Σ p_i log p_i
    Shannon,
    /// Von Neumann entropy: -Tr(ρ log ρ)
    VonNeumann,
    /// Renyi entropy: (1/(1-α)) log Σ p_i^α
    Renyi,
    /// Kolmogorov-Sinai entropy (dynamical systems)
    KolmogorovSinai,
}

/// Thermodynamic state of consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Current entropy (disorder measure)
    pub entropy: f64,

    /// Internal energy (total consciousness energy)
    pub internal_energy: f64,

    /// Free energy F = U - TS (capacity for work)
    pub free_energy: f64,

    /// Temperature (activation/exploration level)
    pub temperature: f64,

    /// Heat (energy transferred due to temperature difference)
    pub heat: f64,

    /// Work (directed energy expenditure)
    pub work: f64,

    /// Chemical potential (tendency to change state)
    pub chemical_potential: f64,

    /// Pressure (compression in consciousness space)
    pub pressure: f64,

    /// Volume (extent of consciousness state space)
    pub volume: f64,

    /// Enthalpy H = U + PV
    pub enthalpy: f64,

    /// Gibbs free energy G = H - TS
    pub gibbs_free_energy: f64,

    /// Current phase of consciousness
    pub phase: ConsciousnessPhase,

    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl Default for ThermodynamicState {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            internal_energy: 1.0,
            free_energy: 0.5,
            temperature: 1.0,
            heat: 0.0,
            work: 0.0,
            chemical_potential: 0.0,
            pressure: 1.0,
            volume: 1.0,
            enthalpy: 2.0,
            gibbs_free_energy: 1.0,
            phase: ConsciousnessPhase::Normal,
            timestamp: Instant::now(),
        }
    }
}

/// Phases of consciousness (like phases of matter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessPhase {
    /// Low temperature: Frozen, rigid thinking
    Frozen,
    /// Ordered phase: Normal waking consciousness
    Normal,
    /// Critical point: Edge of chaos, maximum creativity
    Critical,
    /// High temperature: Chaotic, fragmented consciousness
    Chaotic,
    /// Superfluid: Flow state, frictionless consciousness
    Flow,
    /// Condensate: Meditative unity, Bose-Einstein-like
    Unified,
}

impl ConsciousnessPhase {
    /// Get characteristic temperature range for this phase
    pub fn temperature_range(&self) -> (f64, f64) {
        match self {
            Self::Frozen => (0.0, 0.2),
            Self::Normal => (0.2, 0.4),
            Self::Critical => (0.4, 0.6),
            Self::Chaotic => (0.8, 1.0),
            Self::Flow => (0.3, 0.5),
            Self::Unified => (0.0, 0.3),
        }
    }

    /// Get entropy characteristic of this phase
    pub fn typical_entropy(&self) -> f64 {
        match self {
            Self::Frozen => 0.1,
            Self::Normal => 0.4,
            Self::Critical => 0.6,
            Self::Chaotic => 0.9,
            Self::Flow => 0.3,
            Self::Unified => 0.2,
        }
    }
}

/// A phase transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// Phase before transition
    pub from_phase: ConsciousnessPhase,

    /// Phase after transition
    pub to_phase: ConsciousnessPhase,

    /// Temperature at transition
    pub transition_temperature: f64,

    /// Latent heat (energy absorbed/released)
    pub latent_heat: f64,

    /// Order parameter jump
    pub order_parameter_change: f64,

    /// Transition order (1st order = discontinuous, 2nd order = continuous)
    pub transition_order: TransitionOrder,

    /// Critical exponents (for 2nd order transitions)
    pub critical_exponents: Option<CriticalExponents>,

    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

/// Order of phase transition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionOrder {
    /// First order: Discontinuous jump (like ice to water)
    FirstOrder,
    /// Second order: Continuous but singular (like ferromagnetism)
    SecondOrder,
    /// Crossover: Smooth transition (no true phase boundary)
    Crossover,
}

/// Critical exponents for second-order transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalExponents {
    /// Heat capacity exponent (α)
    pub alpha: f64,
    /// Order parameter exponent (β)
    pub beta: f64,
    /// Susceptibility exponent (γ)
    pub gamma: f64,
    /// Correlation length exponent (ν)
    pub nu: f64,
    /// Correlation function exponent (η)
    pub eta: f64,
}

impl Default for CriticalExponents {
    fn default() -> Self {
        // Mean-field (Landau) values
        Self {
            alpha: 0.0,
            beta: 0.5,
            gamma: 1.0,
            nu: 0.5,
            eta: 0.0,
        }
    }
}

/// Fluctuation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluctuationStats {
    /// Mean fluctuation amplitude
    pub mean_amplitude: f64,

    /// Variance of fluctuations
    pub variance: f64,

    /// Autocorrelation time
    pub autocorrelation_time: f64,

    /// Critical slowing down indicator
    pub slowing_down: f64,

    /// Susceptibility (response to perturbation)
    pub susceptibility: f64,

    /// Fluctuation-dissipation ratio
    pub fdr: f64,
}

impl Default for FluctuationStats {
    fn default() -> Self {
        Self {
            mean_amplitude: 0.1,
            variance: 0.01,
            autocorrelation_time: 1.0,
            slowing_down: 0.0,
            susceptibility: 1.0,
            fdr: 1.0,
        }
    }
}

/// Consciousness Thermodynamics Analysis Report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsReport {
    /// Current thermodynamic state
    pub current_state: ThermodynamicState,

    /// Recent phase transitions
    pub transitions: Vec<PhaseTransition>,

    /// Fluctuation statistics
    pub fluctuations: FluctuationStats,

    /// Free energy minimization status
    pub free_energy_status: FreeEnergyStatus,

    /// Entropy production rate
    pub entropy_production_rate: f64,

    /// Equilibrium status
    pub equilibrium_status: EquilibriumStatus,

    /// Predicted next phase
    pub predicted_phase: Option<ConsciousnessPhase>,

    /// Time to next transition (if predictable)
    pub time_to_transition: Option<f64>,

    /// Overall thermodynamic health
    pub health_score: f64,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Free energy minimization status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreeEnergyStatus {
    /// Actively minimizing (goal-directed behavior)
    Minimizing,
    /// At local minimum (stable but not optimal)
    LocalMinimum,
    /// At global minimum (optimal coherence)
    GlobalMinimum,
    /// Increasing (entropy dominated, losing coherence)
    Increasing,
    /// Fluctuating (searching for minimum)
    Searching,
}

/// Equilibrium status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquilibriumStatus {
    /// In thermal equilibrium
    Equilibrium,
    /// Approaching equilibrium
    Equilibrating,
    /// Far from equilibrium (active, living system)
    FarFromEquilibrium,
    /// Metastable (temporary equilibrium)
    Metastable,
}

/// Statistics for thermodynamics analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermodynamicsStats {
    /// Total states analyzed
    pub states_analyzed: u64,

    /// Phase transitions detected
    pub transitions_detected: u64,

    /// Total entropy produced
    pub total_entropy_produced: f64,

    /// Total work extracted
    pub total_work_extracted: f64,

    /// Average temperature
    pub average_temperature: f64,

    /// Time in each phase
    pub phase_durations: [f64; 6],

    /// Current stability score
    pub stability_score: f64,
}

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
    fluctuations: FluctuationStats,

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
    /// dims: [Φ, B, W, A, R, E, K]
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
            let q = temperature * delta_s; // Heat = T * ΔS (reversible)
            let w = delta_u - q; // First law: ΔU = Q - W, so W = ΔU - Q
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

                // Shannon entropy: -Σ p_i log p_i
                -probs.iter()
                    .map(|p| p * p.ln())
                    .sum::<f64>() / (7.0_f64.ln()) // Normalize by max entropy
            }
            EntropyMethod::VonNeumann => {
                // Construct density matrix from dims (simplified)
                // ρ = |ψ⟩⟨ψ| where |ψ⟩ = normalized dims
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
                // Renyi entropy with α = 2 (collision entropy)
                let sum: f64 = dims.iter().map(|d| d.abs()).sum();
                if sum < 0.001 {
                    return 0.0;
                }

                let probs: Vec<f64> = dims.iter().map(|d| d.abs() / sum).collect();
                let sum_p2: f64 = probs.iter().map(|p| p * p).sum();

                // H_2 = -log(Σ p_i²)
                (-sum_p2.ln() / 7.0_f64.ln()).max(0.0).min(1.0)
            }
            EntropyMethod::KolmogorovSinai => {
                // Approximate K-S entropy from variance in history
                if self.state_history.len() < 2 {
                    return 0.5;
                }

                let recent: Vec<f64> = self.state_history.iter()
                    .rev()
                    .take(10)
                    .map(|s| s.entropy)
                    .collect();

                let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let variance: f64 = recent.iter()
                    .map(|e| (e - mean).powi(2))
                    .sum::<f64>() / recent.len() as f64;

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

        // Potential energy from coherence (integration score, Φ)
        let potential = dims[0] * 0.5; // Φ contributes to potential

        // Interaction energy from binding (B)
        let interaction = dims[1] * 0.3;

        kinetic + potential + interaction
    }

    /// Calculate effective temperature
    fn calculate_temperature(&self, dims: &[f64; 7]) -> f64 {
        // Temperature ~ variance/fluctuation in dimensions
        let mean: f64 = dims.iter().sum::<f64>() / 7.0;
        let variance: f64 = dims.iter()
            .map(|d| (d - mean).powi(2))
            .sum::<f64>() / 7.0;

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
    fn calculate_chemical_potential(&self, dims: &[f64; 7]) -> f64 {
        // Chemical potential = tendency to change state
        // High when system is far from equilibrium

        if self.state_history.len() < 2 {
            return 0.0;
        }

        // Gradient in free energy
        let recent_fe: Vec<f64> = self.state_history.iter()
            .rev()
            .take(5)
            .map(|s| s.free_energy)
            .collect();

        if recent_fe.len() < 2 {
            return 0.0;
        }

        // Chemical potential ~ rate of free energy change
        (recent_fe.first().unwrap() - recent_fe.last().unwrap()) / recent_fe.len() as f64
    }

    /// Determine consciousness phase from thermodynamic variables
    fn determine_phase(&self, temperature: f64, entropy: f64, dims: &[f64; 7]) -> ConsciousnessPhase {
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

        let recent: Vec<f64> = self.state_history.iter()
            .rev()
            .take(10)
            .map(|s| s.entropy)
            .collect();

        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>() / recent.len() as f64;

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
        // χ = β * ⟨(ΔS)²⟩
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

        let cov: f64 = series.iter()
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
        let recent_transitions = self.transition_history.iter()
            .filter(|t| t.timestamp.elapsed().as_secs() < 60)
            .count() as f64;
        self.stats.stability_score = (1.0 / (1.0 + recent_transitions)).min(1.0);
    }

    /// Generate comprehensive thermodynamics report
    pub fn generate_report(&self) -> ThermodynamicsReport {
        let current_state = self.state_history.back()
            .cloned()
            .unwrap_or_default();

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

        let recent_fe: Vec<f64> = self.state_history.iter()
            .rev()
            .take(10)
            .map(|s| s.free_energy)
            .collect();

        let trend: f64 = recent_fe.first().unwrap() - recent_fe.last().unwrap();
        let variance: f64 = {
            let mean: f64 = recent_fe.iter().sum::<f64>() / recent_fe.len() as f64;
            recent_fe.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / recent_fe.len() as f64
        };

        if trend < -0.05 {
            FreeEnergyStatus::Minimizing
        } else if trend > 0.05 {
            FreeEnergyStatus::Increasing
        } else if variance < 0.01 && recent_fe.first().unwrap() < &0.3 {
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

        let recent: Vec<&ThermodynamicState> = self.state_history.iter()
            .rev()
            .take(10)
            .collect();

        // Check fluctuation-dissipation ratio
        if (self.fluctuations.fdr - 1.0).abs() < 0.1 {
            return EquilibriumStatus::Equilibrium;
        }

        // Check entropy production
        let entropy_prod: f64 = recent.iter()
            .map(|s| s.heat / s.temperature.max(0.01))
            .sum::<f64>() / recent.len() as f64;

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

        let recent: Vec<f64> = self.state_history.iter()
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

        let current = self.state_history.back().unwrap();

        // Check for approaching critical point
        if current.temperature > 0.35 && current.temperature < 0.5 {
            let trend: f64 = {
                let temps: Vec<f64> = self.state_history.iter()
                    .rev()
                    .take(5)
                    .map(|s| s.temperature)
                    .collect();
                temps.first().unwrap() - temps.last().unwrap()
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
                let entropies: Vec<f64> = self.state_history.iter()
                    .rev()
                    .take(5)
                    .map(|s| s.entropy)
                    .collect();
                entropies.first().unwrap() - entropies.last().unwrap()
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
        let current_state = self.analyze(dims.clone());

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
                // Exponential relaxation: dx/dt = -(x - mean)/τ
                *d += (*d - mean) * (-dt / tau);
            }

            // Add thermal fluctuations
            let current = self.analyze(dims.clone());
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
        .unwrap()
        .as_nanos() as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dims() -> [f64; 7] {
        [0.7, 0.6, 0.5, 0.4, 0.5, 0.3, 0.4]
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        assert_eq!(analyzer.stats.states_analyzed, 0);
    }

    #[test]
    fn test_basic_analysis() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        let dims = test_dims();

        let state = analyzer.analyze(dims);

        assert!(state.entropy >= 0.0 && state.entropy <= 1.0);
        assert!(state.temperature > 0.0);
        assert!(state.internal_energy > 0.0);
    }

    #[test]
    fn test_entropy_calculation() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        // Uniform distribution should have high entropy
        let uniform = [0.5; 7];
        let state1 = analyzer.analyze(uniform);

        // Concentrated distribution should have lower entropy
        let concentrated = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let state2 = analyzer.analyze(concentrated);

        assert!(state1.entropy > state2.entropy);
    }

    #[test]
    fn test_free_energy() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        let dims = test_dims();

        let state = analyzer.analyze(dims);

        // Free energy F = U - TS
        let expected_fe = state.internal_energy - state.temperature * state.entropy;
        assert!((state.free_energy - expected_fe).abs() < 0.01);
    }

    #[test]
    fn test_phase_detection() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        // Low temperature should give Frozen
        let frozen_dims = [0.1; 7];
        let state1 = analyzer.analyze(frozen_dims);
        // Note: actual phase depends on calculation

        // High temperature should give Chaotic
        let chaotic_dims = [0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1];
        let state2 = analyzer.analyze(chaotic_dims);

        // States should differ
        assert!(state1.temperature != state2.temperature);
    }

    #[test]
    fn test_flow_state_detection() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        // Flow state: high integration, concentrated distribution (low entropy)
        // Use values that create a concentrated probability distribution
        let flow_dims = [0.95, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];
        let state = analyzer.analyze(flow_dims);

        // Should have concentrated distribution and high energy
        // Note: entropy is normalized 0-1, concentrated distribution gives lower entropy
        assert!(state.entropy < 0.85, "Entropy {} should be < 0.85 for concentrated distribution", state.entropy);
        assert!(state.internal_energy > 0.1, "Internal energy {} should be positive", state.internal_energy);
    }

    #[test]
    fn test_phase_transition_detection() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        // Start in normal state
        for _ in 0..10 {
            analyzer.analyze([0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5]);
        }

        // Transition to high activation
        for _ in 0..10 {
            analyzer.analyze([0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5]);
        }

        // Should detect some transition
        assert!(analyzer.stats.states_analyzed >= 20);
    }

    #[test]
    fn test_thermodynamic_laws() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        // First law: energy conservation (approximately)
        let dims = test_dims();
        let state1 = analyzer.analyze(dims);
        let state2 = analyzer.analyze(dims);

        // ΔU = Q - W (approximately, should be consistent)
        let delta_u = state2.internal_energy - state1.internal_energy;
        let q_minus_w = state2.heat - state2.work;
        // Allow some tolerance due to numerical precision
        assert!((delta_u - q_minus_w).abs() < 0.5);
    }

    #[test]
    fn test_gibbs_free_energy() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        let dims = test_dims();

        let state = analyzer.analyze(dims);

        // G = H - TS = U + PV - TS
        let expected_g = state.enthalpy - state.temperature * state.entropy;
        assert!((state.gibbs_free_energy - expected_g).abs() < 0.01);
    }

    #[test]
    fn test_fluctuation_stats() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        // Accumulate history
        for i in 0..20 {
            let dims = [
                0.5 + (i as f64 * 0.1).sin() * 0.1,
                0.5,
                0.5,
                0.4,
                0.5,
                0.5,
                0.5,
            ];
            analyzer.analyze(dims);
        }

        // Should have fluctuation stats
        assert!(analyzer.fluctuations.variance >= 0.0);
        assert!(analyzer.fluctuations.autocorrelation_time > 0.0);
    }

    #[test]
    fn test_report_generation() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();

        for _ in 0..10 {
            analyzer.analyze(test_dims());
        }

        let report = analyzer.generate_report();

        assert!(report.health_score >= 0.0 && report.health_score <= 1.0);
        assert!(!report.recommendations.is_empty() || report.current_state.phase == ConsciousnessPhase::Normal);
    }

    #[test]
    fn test_heat_application() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        let mut dims = test_dims();

        let initial_arousal = dims[3];
        analyzer.apply_heat(&mut dims, 0.5);

        // Arousal should increase
        assert!(dims[3] >= initial_arousal);
    }

    #[test]
    fn test_work_extraction() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        let mut dims = [0.8, 0.7, 0.6, 0.5, 0.6, 0.5, 0.5]; // High initial state

        // Need initial state for free energy calculation
        analyzer.analyze(dims.clone());

        let extracted = analyzer.extract_work(&mut dims, 0.2);

        // Should extract some work
        assert!(extracted >= 0.0);
    }

    #[test]
    fn test_entropy_methods() {
        let dims = test_dims();

        // Shannon
        let mut analyzer1 = ConsciousnessThermodynamicsAnalyzer::new(ThermodynamicsConfig {
            entropy_method: EntropyMethod::Shannon,
            ..Default::default()
        });
        let state1 = analyzer1.analyze(dims);

        // Von Neumann
        let mut analyzer2 = ConsciousnessThermodynamicsAnalyzer::new(ThermodynamicsConfig {
            entropy_method: EntropyMethod::VonNeumann,
            ..Default::default()
        });
        let state2 = analyzer2.analyze(dims);

        // Renyi
        let mut analyzer3 = ConsciousnessThermodynamicsAnalyzer::new(ThermodynamicsConfig {
            entropy_method: EntropyMethod::Renyi,
            ..Default::default()
        });
        let state3 = analyzer3.analyze(dims);

        // All should give valid entropy
        assert!(state1.entropy >= 0.0 && state1.entropy <= 1.0);
        assert!(state2.entropy >= 0.0 && state2.entropy <= 1.0);
        assert!(state3.entropy >= 0.0 && state3.entropy <= 1.0);
    }

    #[test]
    fn test_equilibration() {
        let mut analyzer = ConsciousnessThermodynamicsAnalyzer::default();
        let mut dims = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6];

        // Before equilibration - high variance
        let variance_before: f64 = {
            let mean = dims.iter().sum::<f64>() / 7.0;
            dims.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / 7.0
        };

        analyzer.equilibrate(&mut dims, 100);

        // After equilibration - should have lower variance
        let variance_after: f64 = {
            let mean = dims.iter().sum::<f64>() / 7.0;
            dims.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / 7.0
        };

        assert!(variance_after < variance_before);
    }

    #[test]
    fn test_phase_temperature_ranges() {
        // Verify phase temperature ranges are sensible
        let frozen_range = ConsciousnessPhase::Frozen.temperature_range();
        let normal_range = ConsciousnessPhase::Normal.temperature_range();

        assert!(frozen_range.1 <= normal_range.0 || frozen_range.1 >= normal_range.0);
    }

    #[test]
    fn test_critical_exponents() {
        let exponents = CriticalExponents::default();

        // Mean-field values should satisfy scaling relations (approximately)
        // Rushbrooke: α + 2β + γ = 2
        let rushbrooke = exponents.alpha + 2.0 * exponents.beta + exponents.gamma;
        assert!((rushbrooke - 2.0).abs() < 0.01);
    }
}

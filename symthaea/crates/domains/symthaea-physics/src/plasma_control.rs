// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Plasma Control Feedback System
//!
//! Connects integrated information (Phi) monitoring to plasma control actions for
//! tokamak-style fusion reactor operation. This module provides real-time feedback
//! control based on consciousness metrics (Phi), enabling proactive disruption avoidance.
//!
//! ## Overview
//!
//! The plasma control system monitors Phi levels and their dynamics to:
//! 1. Assess plasma stability in real-time
//! 2. Recommend control actions based on Phi thresholds
//! 3. Execute soft landing sequences when disruption is imminent
//! 4. Learn optimal control policies via temporal difference learning
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                     PLASMA CONTROL FEEDBACK LOOP                           │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │                                                                            │
//! │   ┌─────────────┐      ┌─────────────────┐      ┌──────────────────┐      │
//! │   │ Φ Monitor   │─────▶│ PlasmaControl   │─────▶│ Control Action   │      │
//! │   │ (PhiMonitor)│      │ Loop            │      │ Selection        │      │
//! │   └─────────────┘      └─────────────────┘      └────────┬─────────┘      │
//! │          │                     │                          │               │
//! │          │                     ▼                          │               │
//! │          │            ┌─────────────────┐                 │               │
//! │          │            │ Stability       │                 │               │
//! │          └───────────▶│ Assessment      │◀────────────────┘               │
//! │                       └────────┬────────┘                                 │
//! │                                │                                          │
//! │                                ▼                                          │
//! │                       ┌─────────────────┐      ┌──────────────────┐      │
//! │                       │ FEP Bridge      │─────▶│ TD Learning      │      │
//! │                       │ (MotorCommand)  │      │ (Policy Update)  │      │
//! │                       └─────────────────┘      └──────────────────┘      │
//! │                                                                            │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Phi Thresholds
//!
//! | Threshold | Value | Meaning |
//! |-----------|-------|---------|
//! | Stable    | ≥0.6  | Normal operation, maintain state |
//! | Warning   | 0.4-0.6 | Elevated risk, consider power reduction |
//! | Critical  | 0.25-0.4 | Imminent disruption, soft landing |
//! | Emergency | <0.25 | Emergency shutdown required |
//!
//! ## Integration with FEP
//!
//! The plasma control actions map to FEP `MotorCommandType`:
//! - `MaintainState` -> `NoOp`
//! - `ReducePower` -> `LearningRateAdjust` (reduce system energy)
//! - `AdjustMagneticField` -> `AttentionShift` (redirect configuration)
//! - `InjectGas` -> `ExplorationTrigger` (introduce new dynamics)
//! - `TriggerSoftLanding` -> `ReflectionInitiate` (controlled wind-down)
//! - `EmergencyShutdown` -> `ExpectationReset` (immediate state reset)

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, MotorCommandType, Observation,
    TemporalDifferenceLearningConfig,
};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the plasma control feedback system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaControlConfig {
    /// Phi threshold for normal/stable operation
    pub phi_stable_threshold: f32,

    /// Phi threshold for warning state (elevated disruption risk)
    pub phi_warning_threshold: f32,

    /// Phi threshold for critical state (imminent disruption)
    pub phi_critical_threshold: f32,

    /// Volatility threshold indicating instability
    pub volatility_threshold: f32,

    /// Rate of change threshold for rapid Phi drops (per second)
    pub rate_of_change_threshold: f32,

    /// Maximum allowed control loop latency in milliseconds
    pub control_latency_ms: u64,

    /// Enable TD learning for policy improvement
    pub enable_td_learning: bool,

    /// Soft landing duration in milliseconds
    pub soft_landing_duration_ms: u64,

    /// History window size for trend analysis
    pub history_window: usize,

    /// Minimum confidence required for action execution
    pub min_action_confidence: f32,
}

impl Default for PlasmaControlConfig {
    fn default() -> Self {
        Self {
            phi_stable_threshold: 0.6,
            phi_warning_threshold: 0.4,
            phi_critical_threshold: 0.25,
            volatility_threshold: 0.15,
            rate_of_change_threshold: -0.1,
            control_latency_ms: 10,
            enable_td_learning: true,
            soft_landing_duration_ms: 1000,
            history_window: 50,
            min_action_confidence: 0.3,
        }
    }
}

// =============================================================================
// GAS SPECIES
// =============================================================================

/// Gas species for injection control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GasSpecies {
    /// Deuterium - primary fuel
    Deuterium,
    /// Tritium - fusion fuel
    Tritium,
    /// Helium - ash/dilution
    Helium,
    /// Neon - radiative cooling
    Neon,
    /// Argon - radiative cooling (stronger)
    Argon,
    /// Nitrogen - impurity control
    Nitrogen,
}

impl GasSpecies {
    /// Get the cooling effectiveness of this gas species
    pub fn cooling_factor(&self) -> f32 {
        match self {
            GasSpecies::Deuterium => 0.1,
            GasSpecies::Tritium => 0.1,
            GasSpecies::Helium => 0.2,
            GasSpecies::Neon => 0.6,
            GasSpecies::Argon => 0.9,
            GasSpecies::Nitrogen => 0.4,
        }
    }

    /// Get the radiation efficiency (Z-effective contribution)
    pub fn z_effective(&self) -> f32 {
        match self {
            GasSpecies::Deuterium => 1.0,
            GasSpecies::Tritium => 1.0,
            GasSpecies::Helium => 2.0,
            GasSpecies::Neon => 10.0,
            GasSpecies::Argon => 18.0,
            GasSpecies::Nitrogen => 7.0,
        }
    }
}

// =============================================================================
// TREND DIRECTION
// =============================================================================

/// Direction of Phi trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Phi is increasing
    Rising,
    /// Phi is stable
    Stable,
    /// Phi is decreasing
    Falling,
    /// Phi is rapidly decreasing (emergency)
    RapidFall,
}

// TrendDirection was previously convertible from PhiTrend (language::phi_monitor).
// Now that physics is extracted, TrendDirection is used directly.

// =============================================================================
// PLASMA CONTROL ACTION
// =============================================================================

/// Control actions for plasma stabilization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlasmaControlAction {
    /// Normal operation - maintain current state
    MaintainState,

    /// Reduce heating power by a percentage (0.0 - 1.0)
    ReducePower(f32),

    /// Adjust magnetic field configuration
    AdjustMagneticField {
        /// Change in toroidal field (Tesla)
        delta_bt: f32,
        /// Change in poloidal field (Tesla)
        delta_bp: f32,
    },

    /// Inject gas for density/cooling control
    InjectGas {
        /// Gas species to inject
        species: GasSpecies,
        /// Amount to inject (arbitrary units, 0.0 - 1.0)
        amount: f32,
    },

    /// Trigger controlled shutdown sequence
    TriggerSoftLanding,

    /// Immediate emergency termination
    EmergencyShutdown,
}

impl PlasmaControlAction {
    /// Convert to FEP MotorCommandType for integration
    pub fn to_motor_command_type(&self) -> MotorCommandType {
        match self {
            PlasmaControlAction::MaintainState => MotorCommandType::NoOp,
            PlasmaControlAction::ReducePower(_) => MotorCommandType::LearningRateAdjust,
            PlasmaControlAction::AdjustMagneticField { .. } => MotorCommandType::AttentionShift,
            PlasmaControlAction::InjectGas { .. } => MotorCommandType::ExplorationTrigger,
            PlasmaControlAction::TriggerSoftLanding => MotorCommandType::ReflectionInitiate,
            PlasmaControlAction::EmergencyShutdown => MotorCommandType::ExpectationReset,
        }
    }

    /// Get action index for TD learning
    pub fn to_action_index(&self) -> usize {
        match self {
            PlasmaControlAction::MaintainState => 0,
            PlasmaControlAction::ReducePower(_) => 1,
            PlasmaControlAction::AdjustMagneticField { .. } => 2,
            PlasmaControlAction::InjectGas { .. } => 3,
            PlasmaControlAction::TriggerSoftLanding => 4,
            PlasmaControlAction::EmergencyShutdown => 5,
        }
    }

    /// Create action from index (for TD learning)
    pub fn from_action_index(index: usize, intensity: f32) -> Self {
        match index {
            0 => PlasmaControlAction::MaintainState,
            1 => PlasmaControlAction::ReducePower(intensity),
            2 => PlasmaControlAction::AdjustMagneticField {
                delta_bt: intensity * 0.1,
                delta_bp: intensity * 0.05,
            },
            3 => PlasmaControlAction::InjectGas {
                species: if intensity > 0.7 {
                    GasSpecies::Argon
                } else {
                    GasSpecies::Neon
                },
                amount: intensity,
            },
            4 => PlasmaControlAction::TriggerSoftLanding,
            _ => PlasmaControlAction::EmergencyShutdown,
        }
    }

    /// Get the severity/urgency of this action (0.0 - 1.0)
    pub fn severity(&self) -> f32 {
        match self {
            PlasmaControlAction::MaintainState => 0.0,
            PlasmaControlAction::ReducePower(pct) => 0.2 + pct * 0.3,
            PlasmaControlAction::AdjustMagneticField { .. } => 0.3,
            PlasmaControlAction::InjectGas { amount, .. } => 0.4 + amount * 0.2,
            PlasmaControlAction::TriggerSoftLanding => 0.8,
            PlasmaControlAction::EmergencyShutdown => 1.0,
        }
    }
}

// =============================================================================
// STABILITY REGIME
// =============================================================================

/// Current stability regime based on Phi level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilityRegime {
    /// Normal operation (Phi >= stable threshold)
    Stable,
    /// Elevated risk (warning <= Phi < stable)
    Warning,
    /// Imminent disruption (critical <= Phi < warning)
    Critical,
    /// Emergency state (Phi < critical)
    Emergency,
}

impl StabilityRegime {
    /// Determine regime from Phi value and config
    pub fn from_phi(phi: f64, config: &PlasmaControlConfig) -> Self {
        let phi = phi as f32;
        if phi >= config.phi_stable_threshold {
            StabilityRegime::Stable
        } else if phi >= config.phi_warning_threshold {
            StabilityRegime::Warning
        } else if phi >= config.phi_critical_threshold {
            StabilityRegime::Critical
        } else {
            StabilityRegime::Emergency
        }
    }

    /// Get the urgency level (0.0 - 1.0)
    pub fn urgency(&self) -> f32 {
        match self {
            StabilityRegime::Stable => 0.0,
            StabilityRegime::Warning => 0.3,
            StabilityRegime::Critical => 0.7,
            StabilityRegime::Emergency => 1.0,
        }
    }
}

// =============================================================================
// PLASMA CONTROL STATE
// =============================================================================

/// Current state of the plasma system for control purposes.
///
/// This is a simplified representation focused on control-relevant parameters.
/// For full plasma state with HDC encoding, see [`super::plasma_hdc_encoder::PlasmaState`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaControlState {
    /// Current integrated information (Phi)
    pub phi: f64,

    /// Toroidal magnetic field (Tesla)
    pub bt: f32,

    /// Poloidal magnetic field (Tesla)
    pub bp: f32,

    /// Plasma current (MA)
    pub ip: f32,

    /// Electron density (10^19 m^-3)
    pub ne: f32,

    /// Electron temperature (keV)
    pub te: f32,

    /// Heating power (MW)
    pub p_heat: f32,

    /// Current timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

impl PlasmaControlState {
    /// Create a new plasma control state
    pub fn new(phi: f64, timestamp_ns: u64) -> Self {
        Self {
            phi,
            bt: 5.3, // Typical ITER value
            bp: 0.3,
            ip: 15.0,     // 15 MA
            ne: 10.0,     // 10^20 m^-3
            te: 10.0,     // 10 keV
            p_heat: 50.0, // 50 MW
            timestamp_ns,
        }
    }

    /// Create from observation vector (for FEP integration)
    pub fn from_observation(obs: &Observation, timestamp_ns: u64) -> Self {
        Self {
            phi: obs.values.first().copied().unwrap_or(0.5),
            bt: 5.3,
            bp: 0.3,
            ip: 15.0,
            ne: obs.values.get(1).map(|v| *v as f32 * 20.0).unwrap_or(10.0),
            te: obs.values.get(2).map(|v| *v as f32 * 20.0).unwrap_or(10.0),
            p_heat: obs.values.get(3).map(|v| *v as f32 * 100.0).unwrap_or(50.0),
            timestamp_ns,
        }
    }

    /// Convert to observation for FEP
    pub fn to_observation(&self) -> Observation {
        Observation::new(
            vec![
                self.phi,
                (self.ne / 20.0) as f64,
                (self.te / 20.0) as f64,
                (self.p_heat / 100.0) as f64,
            ],
            1.0,
            "plasma",
        )
    }

    /// Compute confinement quality (simplified)
    pub fn confinement_quality(&self) -> f32 {
        // Simplified H-mode factor proxy
        let tau_e = self.ne * self.te / self.p_heat.max(0.1);
        (tau_e / 3.0).min(2.0)
    }

    /// Compute MHD stability margin (simplified)
    pub fn mhd_stability(&self) -> f32 {
        // Simplified beta limit proxy
        let beta = self.ne * self.te / (self.bt * self.bt);
        1.0 - (beta / 3.0).min(1.0)
    }
}

// =============================================================================
// STABILITY ASSESSMENT
// =============================================================================

/// Assessment of plasma stability from Phi monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaStabilityAssessment {
    /// Current Phi value
    pub current_phi: f64,

    /// Current trend direction
    pub phi_trend: TrendDirection,

    /// Volatility measure (standard deviation of recent Phi values)
    pub volatility: f64,

    /// Estimated time to critical threshold (if trending down)
    pub time_to_critical_estimate_ms: Option<u64>,

    /// Current stability regime
    pub regime: StabilityRegime,

    /// Recommended control action
    pub recommended_action: PlasmaControlAction,

    /// Confidence in the recommendation (0.0 - 1.0)
    pub confidence: f32,

    /// Assessment timestamp
    pub timestamp_ns: u64,

    /// Rate of Phi change (per second)
    pub phi_rate: f64,
}

impl PlasmaStabilityAssessment {
    /// Check if intervention is required
    pub fn requires_intervention(&self) -> bool {
        !matches!(self.recommended_action, PlasmaControlAction::MaintainState)
    }

    /// Check if this is an emergency situation
    pub fn is_emergency(&self) -> bool {
        matches!(self.regime, StabilityRegime::Emergency)
            || matches!(
                self.recommended_action,
                PlasmaControlAction::EmergencyShutdown
            )
    }

    /// Get the urgency level (0.0 - 1.0)
    pub fn urgency(&self) -> f32 {
        self.regime
            .urgency()
            .max(self.recommended_action.severity())
    }
}

// =============================================================================
// PHI HISTORY ENTRY
// =============================================================================

/// A single entry in the Phi history
#[derive(Debug, Clone, Copy)]
struct PhiHistoryEntry {
    phi: f64,
    timestamp_ns: u64,
}

// =============================================================================
// PLASMA CONTROL LOOP
// =============================================================================

/// Main plasma control feedback loop
pub struct PlasmaControlLoop {
    /// Configuration
    config: PlasmaControlConfig,

    /// FEP active inference agent for learning optimal control
    fep_agent: Option<ActiveInferenceAgent>,

    /// History of Phi measurements
    phi_history: VecDeque<PhiHistoryEntry>,

    /// Current assessment
    current_assessment: Option<PlasmaStabilityAssessment>,

    /// Soft landing state (if active)
    soft_landing_state: Option<SoftLandingState>,

    /// Last control action
    last_action: Option<PlasmaControlAction>,

    /// Timing for latency monitoring
    last_cycle_time: Option<Instant>,

    /// Statistics
    stats: PlasmaControlStats,
}

/// State tracking for soft landing sequence
#[derive(Debug, Clone)]
struct SoftLandingState {
    start_time: Instant,
    /// Reserved for soft landing trajectory calculation
    _initial_phi: f64,
    /// Reserved for soft landing trajectory calculation
    _target_phi: f64,
    phase: SoftLandingPhase,
}

/// Phases of soft landing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoftLandingPhase {
    PowerRampDown,
    DensityControl,
    FieldRelaxation,
    Complete,
}

/// Statistics for plasma control
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlasmaControlStats {
    /// Total control cycles
    pub total_cycles: u64,
    /// Cycles in each regime
    pub cycles_stable: u64,
    pub cycles_warning: u64,
    pub cycles_critical: u64,
    pub cycles_emergency: u64,
    /// Actions taken
    pub actions_maintain: u64,
    pub actions_reduce_power: u64,
    pub actions_adjust_field: u64,
    pub actions_inject_gas: u64,
    pub actions_soft_landing: u64,
    pub actions_emergency: u64,
    /// Latency statistics (microseconds)
    pub avg_latency_us: f64,
    pub max_latency_us: u64,
    /// TD learning statistics
    pub avg_td_error: f64,
    pub total_td_updates: u64,
}

impl PlasmaControlLoop {
    /// Create a new plasma control loop
    pub fn new(config: PlasmaControlConfig) -> Self {
        let fep_agent = if config.enable_td_learning {
            let fep_config = ActiveInferenceAgentConfig {
                state_dim: 8,
                obs_dim: 4,
                num_actions: 6, // Number of control action types
                enable_td_learning: true,
                td_config: TemporalDifferenceLearningConfig {
                    initial_learning_rate: 0.05,
                    gamma: 0.95,
                    lambda: 0.8,
                    ..Default::default()
                },
                ..Default::default()
            };
            Some(ActiveInferenceAgent::new(fep_config))
        } else {
            None
        };

        Self {
            config,
            fep_agent,
            phi_history: VecDeque::with_capacity(1000),
            current_assessment: None,
            soft_landing_state: None,
            last_action: None,
            last_cycle_time: None,
            stats: PlasmaControlStats::default(),
        }
    }

    /// Assess plasma stability from current state
    pub fn assess(&self, plasma_state: &PlasmaControlState) -> PlasmaStabilityAssessment {
        let phi = plasma_state.phi;
        let regime = StabilityRegime::from_phi(phi, &self.config);

        // Calculate trend and rate
        let (trend, phi_rate) = self.calculate_trend_and_rate();

        // Calculate volatility
        let volatility = self.calculate_volatility();

        // Estimate time to critical
        let time_to_critical = self.estimate_time_to_critical(phi, phi_rate);

        // Determine recommended action
        let (recommended_action, confidence) =
            self.determine_action(phi, regime, trend, volatility, phi_rate, plasma_state);

        PlasmaStabilityAssessment {
            current_phi: phi,
            phi_trend: trend,
            volatility,
            time_to_critical_estimate_ms: time_to_critical,
            regime,
            recommended_action,
            confidence,
            timestamp_ns: plasma_state.timestamp_ns,
            phi_rate,
        }
    }

    /// Update internal state with new Phi measurement
    pub fn update(&mut self, new_phi: f64, timestamp_ns: u64) {
        let cycle_start = Instant::now();

        // Record Phi history
        if self.phi_history.len() >= 1000 {
            self.phi_history.pop_front();
        }
        self.phi_history.push_back(PhiHistoryEntry {
            phi: new_phi,
            timestamp_ns,
        });

        // Update stats
        self.stats.total_cycles += 1;

        // Track latency
        if let Some(last_time) = self.last_cycle_time {
            let latency = last_time.elapsed().as_micros() as u64;
            let n = self.stats.total_cycles as f64;
            self.stats.avg_latency_us =
                (self.stats.avg_latency_us * (n - 1.0) + latency as f64) / n;
            self.stats.max_latency_us = self.stats.max_latency_us.max(latency);
        }
        self.last_cycle_time = Some(cycle_start);

        // Update soft landing if active
        if let Some(sl_state) = self.soft_landing_state.clone() {
            self.update_soft_landing(sl_state, new_phi);
        }
    }

    /// Get the current recommended control action
    pub fn get_control_action(&self) -> PlasmaControlAction {
        self.current_assessment
            .as_ref()
            .map(|a| a.recommended_action.clone())
            .unwrap_or(PlasmaControlAction::MaintainState)
    }

    /// Execute a control cycle: assess, update, and return action
    pub fn control_cycle(
        &mut self,
        plasma_state: &PlasmaControlState,
    ) -> PlasmaStabilityAssessment {
        // Update internal state
        self.update(plasma_state.phi, plasma_state.timestamp_ns);

        // Perform assessment
        let assessment = self.assess(plasma_state);

        // Update regime statistics
        match assessment.regime {
            StabilityRegime::Stable => self.stats.cycles_stable += 1,
            StabilityRegime::Warning => self.stats.cycles_warning += 1,
            StabilityRegime::Critical => self.stats.cycles_critical += 1,
            StabilityRegime::Emergency => self.stats.cycles_emergency += 1,
        }

        // Update action statistics
        match &assessment.recommended_action {
            PlasmaControlAction::MaintainState => self.stats.actions_maintain += 1,
            PlasmaControlAction::ReducePower(_) => self.stats.actions_reduce_power += 1,
            PlasmaControlAction::AdjustMagneticField { .. } => self.stats.actions_adjust_field += 1,
            PlasmaControlAction::InjectGas { .. } => self.stats.actions_inject_gas += 1,
            PlasmaControlAction::TriggerSoftLanding => self.stats.actions_soft_landing += 1,
            PlasmaControlAction::EmergencyShutdown => self.stats.actions_emergency += 1,
        }

        // TD learning update if enabled
        if let Some(ref mut agent) = self.fep_agent {
            let obs = plasma_state.to_observation();
            let perception = agent.perceive(&obs);
            let action_result = agent.select_action();

            // Record TD stats
            if let Some(td_stats) = agent.td_stats() {
                self.stats.avg_td_error = td_stats.avg_td_error;
                self.stats.total_td_updates = td_stats.total_updates;
            }

            // Use FEP action if confidence is high enough
            if action_result
                .action_probabilities
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .copied()
                .unwrap_or(0.0)
                > self.config.min_action_confidence as f64
            {
                // FEP agent has high confidence - could use its action
                // For now we still use rule-based but log the FEP recommendation
                let _ = perception; // Suppress warning
            }
        }

        // Store assessment
        self.current_assessment = Some(assessment.clone());
        self.last_action = Some(assessment.recommended_action.clone());

        assessment
    }

    /// Start soft landing sequence
    pub fn initiate_soft_landing(&mut self, current_phi: f64) {
        self.soft_landing_state = Some(SoftLandingState {
            start_time: Instant::now(),
            _initial_phi: current_phi,
            _target_phi: 0.0,
            phase: SoftLandingPhase::PowerRampDown,
        });
    }

    /// Check if soft landing is in progress
    pub fn is_soft_landing_active(&self) -> bool {
        self.soft_landing_state.is_some()
    }

    /// Get current statistics
    pub fn stats(&self) -> &PlasmaControlStats {
        &self.stats
    }

    /// Get current config
    pub fn config(&self) -> &PlasmaControlConfig {
        &self.config
    }

    /// Reset the control loop
    pub fn reset(&mut self) {
        self.phi_history.clear();
        self.current_assessment = None;
        self.soft_landing_state = None;
        self.last_action = None;
        self.last_cycle_time = None;
        self.stats = PlasmaControlStats::default();

        if let Some(ref mut agent) = self.fep_agent {
            agent.reset();
        }
    }

    // --- Private helper methods ---

    fn calculate_trend_and_rate(&self) -> (TrendDirection, f64) {
        if self.phi_history.len() < 3 {
            return (TrendDirection::Stable, 0.0);
        }

        let recent: Vec<_> = self.phi_history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return (TrendDirection::Stable, 0.0);
        }

        // Calculate rate (Phi per second)
        let newest = recent
            .first()
            .expect("recent has at least 2 elements (checked above)");
        let oldest = recent
            .last()
            .expect("recent has at least 2 elements (checked above)");
        let dt_ns = newest.timestamp_ns.saturating_sub(oldest.timestamp_ns);
        let dt_s = dt_ns as f64 / 1_000_000_000.0;

        let rate = if dt_s > 0.0 {
            (newest.phi - oldest.phi) / dt_s
        } else {
            0.0
        };

        // Determine trend
        let trend = if rate > 0.05 {
            TrendDirection::Rising
        } else if rate < self.config.rate_of_change_threshold as f64 * 2.0 {
            TrendDirection::RapidFall
        } else if rate < self.config.rate_of_change_threshold as f64 {
            TrendDirection::Falling
        } else {
            TrendDirection::Stable
        };

        (trend, rate)
    }

    fn calculate_volatility(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        let values: Vec<f64> = self
            .phi_history
            .iter()
            .rev()
            .take(self.config.history_window)
            .map(|e| e.phi)
            .collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }

    fn estimate_time_to_critical(&self, current_phi: f64, rate: f64) -> Option<u64> {
        if rate >= 0.0 || current_phi <= self.config.phi_critical_threshold as f64 {
            return None;
        }

        let delta_phi = current_phi - self.config.phi_critical_threshold as f64;
        let time_s = delta_phi / (-rate);

        if time_s > 0.0 && time_s < 60.0 {
            Some((time_s * 1000.0) as u64)
        } else {
            None
        }
    }

    fn determine_action(
        &self,
        phi: f64,
        regime: StabilityRegime,
        trend: TrendDirection,
        volatility: f64,
        phi_rate: f64,
        plasma_state: &PlasmaControlState,
    ) -> (PlasmaControlAction, f32) {
        // Emergency conditions override everything
        if regime == StabilityRegime::Emergency {
            return (PlasmaControlAction::EmergencyShutdown, 1.0);
        }

        // Soft landing if critical with falling trend
        if regime == StabilityRegime::Critical {
            if matches!(trend, TrendDirection::Falling | TrendDirection::RapidFall) {
                return (PlasmaControlAction::TriggerSoftLanding, 0.95);
            }

            // Try aggressive intervention first
            return (
                PlasmaControlAction::InjectGas {
                    species: GasSpecies::Argon,
                    amount: 0.8,
                },
                0.85,
            );
        }

        // Warning regime - moderate intervention
        if regime == StabilityRegime::Warning {
            // High volatility - inject cooling gas
            if volatility > self.config.volatility_threshold as f64 {
                return (
                    PlasmaControlAction::InjectGas {
                        species: GasSpecies::Neon,
                        amount: 0.5,
                    },
                    0.7,
                );
            }

            // Falling trend - reduce power
            if matches!(trend, TrendDirection::Falling | TrendDirection::RapidFall) {
                let reduction = (-phi_rate as f32 * 2.0).clamp(0.1, 0.5);
                return (PlasmaControlAction::ReducePower(reduction), 0.75);
            }

            // Otherwise try field adjustment
            return (
                PlasmaControlAction::AdjustMagneticField {
                    delta_bt: 0.05,
                    delta_bp: 0.02,
                },
                0.6,
            );
        }

        // Stable regime - maintain or fine-tune
        if regime == StabilityRegime::Stable {
            // Still watch for rapid drops
            if trend == TrendDirection::RapidFall {
                return (PlasmaControlAction::ReducePower(0.1), 0.5);
            }

            // High volatility even in stable regime
            if volatility > self.config.volatility_threshold as f64 * 0.8 {
                return (
                    PlasmaControlAction::AdjustMagneticField {
                        delta_bt: 0.02,
                        delta_bp: 0.01,
                    },
                    0.4,
                );
            }

            // Suppress unused variable warnings
            let _ = phi;
            let _ = plasma_state;
        }

        // Default: maintain state
        (PlasmaControlAction::MaintainState, 0.9)
    }

    fn update_soft_landing(&mut self, mut state: SoftLandingState, _current_phi: f64) {
        let elapsed = state.start_time.elapsed();
        let duration = Duration::from_millis(self.config.soft_landing_duration_ms);

        // Progress through phases
        let phase_duration = duration / 3;
        state.phase = if elapsed < phase_duration {
            SoftLandingPhase::PowerRampDown
        } else if elapsed < phase_duration * 2 {
            SoftLandingPhase::DensityControl
        } else if elapsed < duration {
            SoftLandingPhase::FieldRelaxation
        } else {
            SoftLandingPhase::Complete
        };

        if state.phase == SoftLandingPhase::Complete {
            self.soft_landing_state = None;
        } else {
            self.soft_landing_state = Some(state);
        }
    }
}

// =============================================================================
// PLASMA SIMULATOR
// =============================================================================

/// Simulator for testing plasma control without real reactor
pub struct PlasmaSimulator {
    /// Current simulated state
    state: PlasmaControlState,

    /// Random seed for reproducibility
    seed: u64,

    /// Simulation time step (ns)
    dt_ns: u64,

    /// Current simulation time
    current_time_ns: u64,

    /// Disruption scenario (if any)
    scenario: Option<DisruptionScenario>,
}

/// Pre-defined disruption scenarios for testing
#[derive(Debug, Clone)]
pub enum DisruptionScenario {
    /// Gradual density limit approach
    DensityLimit { start_time_ns: u64, rate: f64 },
    /// Rapid MHD event
    MhdEvent { trigger_time_ns: u64, severity: f64 },
    /// Oscillating instability
    Oscillation { period_ns: u64, amplitude: f64 },
    /// Edge localized mode (ELM)
    Elm { period_ns: u64, drop_magnitude: f64 },
}

impl PlasmaSimulator {
    /// Create a new simulator
    pub fn new(seed: u64) -> Self {
        Self {
            state: PlasmaControlState::new(0.7, 0),
            seed,
            dt_ns: 1_000_000, // 1ms default
            current_time_ns: 0,
            scenario: None,
        }
    }

    /// Set simulation time step
    pub fn with_timestep_ns(mut self, dt_ns: u64) -> Self {
        self.dt_ns = dt_ns;
        self
    }

    /// Set disruption scenario
    pub fn with_scenario(mut self, scenario: DisruptionScenario) -> Self {
        self.scenario = Some(scenario);
        self
    }

    /// Get current state
    pub fn state(&self) -> &PlasmaControlState {
        &self.state
    }

    /// Advance simulation by one timestep
    pub fn step(&mut self) -> &PlasmaControlState {
        self.current_time_ns += self.dt_ns;
        self.state.timestamp_ns = self.current_time_ns;

        // Apply scenario effects
        if let Some(ref scenario) = self.scenario {
            self.apply_scenario(scenario.clone());
        }

        // Add noise
        let noise = self.pseudo_random() * 0.02 - 0.01;
        self.state.phi = (self.state.phi + noise).clamp(0.0, 1.0);

        &self.state
    }

    /// Apply control action to simulator
    pub fn apply_action(&mut self, action: &PlasmaControlAction) {
        match action {
            PlasmaControlAction::MaintainState => {}
            PlasmaControlAction::ReducePower(pct) => {
                self.state.p_heat *= 1.0 - pct;
                // Power reduction improves Phi slightly
                self.state.phi = (self.state.phi + 0.05 * *pct as f64).min(1.0);
            }
            PlasmaControlAction::AdjustMagneticField { delta_bt, delta_bp } => {
                self.state.bt += delta_bt;
                self.state.bp += delta_bp;
                // Field adjustment can help stability
                self.state.phi = (self.state.phi + 0.03).min(1.0);
            }
            PlasmaControlAction::InjectGas { species, amount } => {
                // Gas injection affects density and temperature
                self.state.ne += amount * 2.0;
                self.state.te *= 1.0 - amount * 0.1;
                // Cooling effect improves Phi for instabilities
                self.state.phi = (self.state.phi
                    + species.cooling_factor() as f64 * *amount as f64 * 0.1)
                    .min(1.0);
            }
            PlasmaControlAction::TriggerSoftLanding => {
                // Soft landing ramps down everything
                self.state.p_heat *= 0.5;
                self.state.phi = (self.state.phi - 0.1).max(0.0);
            }
            PlasmaControlAction::EmergencyShutdown => {
                // Immediate shutdown
                self.state.p_heat = 0.0;
                self.state.ip = 0.0;
                self.state.phi = 0.0;
            }
        }
    }

    /// Reset to initial conditions
    pub fn reset(&mut self) {
        self.state = PlasmaControlState::new(0.7, 0);
        self.current_time_ns = 0;
    }

    fn apply_scenario(&mut self, scenario: DisruptionScenario) {
        match scenario {
            DisruptionScenario::DensityLimit {
                start_time_ns,
                rate,
            } => {
                if self.current_time_ns >= start_time_ns {
                    let dt = (self.current_time_ns - start_time_ns) as f64 / 1_000_000_000.0;
                    self.state.phi = (0.8 - rate * dt).max(0.0);
                }
            }
            DisruptionScenario::MhdEvent {
                trigger_time_ns,
                severity,
            } => {
                if self.current_time_ns >= trigger_time_ns
                    && self.current_time_ns < trigger_time_ns + 10_000_000
                {
                    self.state.phi = (self.state.phi - severity * 0.1).max(0.0);
                }
            }
            DisruptionScenario::Oscillation {
                period_ns,
                amplitude,
            } => {
                let phase =
                    (self.current_time_ns as f64 / period_ns as f64) * std::f64::consts::TAU;
                self.state.phi = 0.5 + amplitude * phase.sin();
            }
            DisruptionScenario::Elm {
                period_ns,
                drop_magnitude,
            } => {
                let cycle_position = self.current_time_ns % period_ns;
                if cycle_position < period_ns / 10 {
                    self.state.phi = (self.state.phi - drop_magnitude).max(0.1);
                } else {
                    // Recovery
                    self.state.phi = (self.state.phi + 0.02).min(0.8);
                }
            }
        }
    }

    fn pseudo_random(&mut self) -> f64 {
        // Simple LCG for reproducible randomness
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.seed >> 16) & 0x7FFF) as f64 / 32767.0
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = PlasmaControlConfig::default();
        assert_eq!(config.phi_stable_threshold, 0.6);
        assert_eq!(config.phi_warning_threshold, 0.4);
        assert_eq!(config.phi_critical_threshold, 0.25);
        assert_eq!(config.control_latency_ms, 10);
    }

    #[test]
    fn test_stability_regime_from_phi() {
        let config = PlasmaControlConfig::default();

        assert_eq!(
            StabilityRegime::from_phi(0.7, &config),
            StabilityRegime::Stable
        );
        assert_eq!(
            StabilityRegime::from_phi(0.5, &config),
            StabilityRegime::Warning
        );
        assert_eq!(
            StabilityRegime::from_phi(0.3, &config),
            StabilityRegime::Critical
        );
        assert_eq!(
            StabilityRegime::from_phi(0.1, &config),
            StabilityRegime::Emergency
        );
    }

    #[test]
    fn test_action_selection_stable() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Stable state
        let state = PlasmaControlState::new(0.7, 0);
        let assessment = control_loop.control_cycle(&state);

        assert_eq!(assessment.regime, StabilityRegime::Stable);
        assert!(matches!(
            assessment.recommended_action,
            PlasmaControlAction::MaintainState
        ));
    }

    #[test]
    fn test_action_selection_warning() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Warning state
        let state = PlasmaControlState::new(0.45, 0);
        let assessment = control_loop.control_cycle(&state);

        assert_eq!(assessment.regime, StabilityRegime::Warning);
        // Should recommend some intervention
        assert!(assessment.requires_intervention());
    }

    #[test]
    fn test_action_selection_critical() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Critical state
        let state = PlasmaControlState::new(0.28, 0);
        let assessment = control_loop.control_cycle(&state);

        assert_eq!(assessment.regime, StabilityRegime::Critical);
        assert!(assessment.requires_intervention());
        assert!(assessment.urgency() > 0.5);
    }

    #[test]
    fn test_action_selection_emergency() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Emergency state
        let state = PlasmaControlState::new(0.1, 0);
        let assessment = control_loop.control_cycle(&state);

        assert_eq!(assessment.regime, StabilityRegime::Emergency);
        assert!(matches!(
            assessment.recommended_action,
            PlasmaControlAction::EmergencyShutdown
        ));
        assert!(assessment.is_emergency());
    }

    #[test]
    fn test_regime_transitions() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Start stable
        let mut phi = 0.7;
        let mut timestamp = 0u64;

        for i in 0..20 {
            let state = PlasmaControlState::new(phi, timestamp);
            let assessment = control_loop.control_cycle(&state);

            // After a few cycles in warning, should recommend action
            if i > 5 && assessment.regime == StabilityRegime::Warning {
                assert!(assessment.requires_intervention());
            }

            // Decrease phi gradually
            phi -= 0.03;
            timestamp += 10_000_000; // 10ms
        }

        // Should have reached critical/emergency by now
        let final_state = PlasmaControlState::new(phi, timestamp);
        let final_assessment = control_loop.control_cycle(&final_state);
        assert!(final_assessment.urgency() > 0.5);
    }

    #[test]
    fn test_timing_constraint() {
        let config = PlasmaControlConfig {
            control_latency_ms: 10,
            ..Default::default()
        };
        let mut control_loop = PlasmaControlLoop::new(config);

        let start = Instant::now();

        // Run 100 control cycles
        for i in 0..100 {
            let state = PlasmaControlState::new(0.5 + (i as f64 * 0.001), i * 1_000_000);
            let _ = control_loop.control_cycle(&state);
        }

        let elapsed = start.elapsed();
        let avg_per_cycle = elapsed / 100;

        // Each cycle should complete well under 10ms
        assert!(
            avg_per_cycle < Duration::from_millis(10),
            "Average cycle time {:?} exceeds 10ms budget",
            avg_per_cycle
        );
    }

    #[test]
    fn test_soft_landing_activation() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Build up history with falling trend
        for i in 0..15 {
            let phi = 0.5 - (i as f64 * 0.02);
            let timestamp = i * 10_000_000;
            let state = PlasmaControlState::new(phi.max(0.26), timestamp);
            let _ = control_loop.control_cycle(&state);
        }

        // Critical state with falling trend should trigger soft landing
        let critical_state = PlasmaControlState::new(0.27, 15 * 10_000_000);
        let assessment = control_loop.control_cycle(&critical_state);

        assert!(matches!(
            assessment.recommended_action,
            PlasmaControlAction::TriggerSoftLanding | PlasmaControlAction::InjectGas { .. }
        ));
    }

    #[test]
    fn test_simulator_basic() {
        let mut sim = PlasmaSimulator::new(42);

        // Initial state should be stable
        assert!(sim.state().phi > 0.6);

        // Step forward
        for _ in 0..100 {
            sim.step();
        }

        // Should still be roughly stable without scenario
        assert!(sim.state().phi > 0.5);
    }

    #[test]
    fn test_simulator_density_limit() {
        let mut sim = PlasmaSimulator::new(42).with_scenario(DisruptionScenario::DensityLimit {
            start_time_ns: 10_000_000,
            rate: 1.0, // 1.0 per second => drops ~0.09 per 90ms of simulation
        });

        // Step 200 times at 1ms each = 200ms total
        // After start_time (10ms), 190ms of decay at rate 1.0/s => phi = 0.8 - 1.0*0.19 = 0.61
        // With more steps we get further drop
        for _ in 0..500 {
            sim.step();
        }

        // 500ms total, 490ms after start => phi = 0.8 - 1.0*0.49 = 0.31 (plus noise override)
        // The scenario directly sets phi = (0.8 - rate * dt), so it should be well below 0.5
        assert!(
            sim.state().phi < 0.5,
            "Phi should have dropped below 0.5 due to density limit, got {}",
            sim.state().phi
        );
    }

    #[test]
    fn test_simulator_action_application() {
        let mut sim = PlasmaSimulator::new(42);
        sim.state = PlasmaControlState::new(0.4, 0); // Start in warning regime

        // Apply power reduction
        sim.apply_action(&PlasmaControlAction::ReducePower(0.3));

        // Phi should improve slightly
        assert!(sim.state().phi > 0.4);
        // Power should be reduced
        assert!(sim.state().p_heat < 50.0);
    }

    #[test]
    fn test_control_action_to_motor_command() {
        assert_eq!(
            PlasmaControlAction::MaintainState.to_motor_command_type(),
            MotorCommandType::NoOp
        );
        assert_eq!(
            PlasmaControlAction::ReducePower(0.5).to_motor_command_type(),
            MotorCommandType::LearningRateAdjust
        );
        assert_eq!(
            PlasmaControlAction::TriggerSoftLanding.to_motor_command_type(),
            MotorCommandType::ReflectionInitiate
        );
        assert_eq!(
            PlasmaControlAction::EmergencyShutdown.to_motor_command_type(),
            MotorCommandType::ExpectationReset
        );
    }

    #[test]
    fn test_gas_species_properties() {
        assert!(GasSpecies::Argon.cooling_factor() > GasSpecies::Neon.cooling_factor());
        assert!(GasSpecies::Argon.z_effective() > GasSpecies::Neon.z_effective());
        assert_eq!(GasSpecies::Deuterium.z_effective(), 1.0);
    }

    #[test]
    fn test_plasma_state_conversions() {
        let state = PlasmaControlState::new(0.65, 1000);
        let obs = state.to_observation();

        assert_eq!(obs.values.len(), 4);
        assert!((obs.values[0] - 0.65).abs() < 0.001);

        let reconstructed = PlasmaControlState::from_observation(&obs, 1000);
        assert!((reconstructed.phi - state.phi).abs() < 0.001);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        // Run various scenarios
        for phi in [0.7, 0.5, 0.3, 0.1] {
            let state = PlasmaControlState::new(phi, 0);
            let _ = control_loop.control_cycle(&state);
        }

        let stats = control_loop.stats();
        assert_eq!(stats.total_cycles, 4);
        assert!(stats.cycles_stable >= 1);
        assert!(stats.cycles_emergency >= 1);
    }

    #[test]
    fn test_integrated_control_simulation() {
        let config = PlasmaControlConfig::default();
        let mut control_loop = PlasmaControlLoop::new(config);

        let mut sim = PlasmaSimulator::new(12345)
            .with_timestep_ns(1_000_000)
            .with_scenario(DisruptionScenario::DensityLimit {
                start_time_ns: 50_000_000,
                rate: 0.05,
            });

        let mut actions_taken = Vec::new();

        // Run simulation with control
        for _ in 0..200 {
            let state = sim.step().clone();
            let assessment = control_loop.control_cycle(&state);

            if assessment.requires_intervention() {
                actions_taken.push(assessment.recommended_action.clone());
                sim.apply_action(&assessment.recommended_action);
            }
        }

        // Control should have intervened
        assert!(
            !actions_taken.is_empty(),
            "Control should have taken actions"
        );

        // Final state should not be in emergency (control prevented disruption)
        // or we caught it and shut down safely
        let final_assessment = control_loop.current_assessment.as_ref().unwrap();
        assert!(
            final_assessment.regime != StabilityRegime::Emergency
                || actions_taken.iter().any(|a| matches!(
                    a,
                    PlasmaControlAction::EmergencyShutdown
                        | PlasmaControlAction::TriggerSoftLanding
                )),
            "Either avoided emergency or shut down safely"
        );
    }
}

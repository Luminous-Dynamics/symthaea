//! Shared types for all cognitive routers
//!
//! This module contains common types used across all router implementations,
//! enabling consistent interfaces and interoperability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export from parent types module
pub use super::super::types::{ComponentId, ImprovementType, AccuracyMetric, BottleneckType};

// Import from core (these will eventually be moved here)
pub use super::super::core::{
    ConsciousnessAction,
    CognitiveResourceType,
    LatentConsciousnessState,
    ConsciousnessWorldModel,
    WorldModelConfig,
    MetaCognitiveController,
    MetaCognitiveConfig,
    SelfModel,
    SelfModelConfig,
    SubsystemId,
};

// ═══════════════════════════════════════════════════════════════════════════
// PROCESSING MODE
// ═══════════════════════════════════════════════════════════════════════════

/// Processing mode for consciousness operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ProcessingMode {
    /// Serial processing with high attention
    #[default]
    Serial,
    /// Parallel processing across subsystems
    Parallel,
    /// Oscillatory phase-locked processing
    Oscillatory,
    /// Neural binding operations
    Binding,
    /// Information integration
    Integration,
    /// Gathering sensory input
    InputGathering,
    /// Focused attention processing
    Attention,
    /// Generating responses/outputs
    OutputGeneration,
    /// Memory consolidation
    Consolidation,
    /// System reset/cleanup
    Reset,
    /// Background maintenance
    Maintenance,
}

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY PHASE (Enum for phase categories)
// ═══════════════════════════════════════════════════════════════════════════

/// Oscillatory phase category
///
/// Represents the four phases of a consciousness oscillation cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OscillatoryPhase {
    /// Peak phase (0 radians) - maximum integration
    #[default]
    Peak,
    /// Falling phase (π/2 radians) - output generation
    Falling,
    /// Trough phase (π radians) - reset/consolidation
    Trough,
    /// Rising phase (3π/2 radians) - input gathering
    Rising,
}

impl OscillatoryPhase {
    /// Get processing profile for this phase
    pub fn processing_profile(&self) -> ProcessingProfile {
        match self {
            OscillatoryPhase::Peak => ProcessingProfile {
                optimal_for: vec![ProcessingMode::Binding, ProcessingMode::Integration],
                integration_capacity: 1.0,
                description: "Maximum integration and binding".to_string(),
            },
            OscillatoryPhase::Rising => ProcessingProfile {
                optimal_for: vec![ProcessingMode::InputGathering, ProcessingMode::Attention],
                integration_capacity: 0.7,
                description: "Input gathering and attention focus".to_string(),
            },
            OscillatoryPhase::Falling => ProcessingProfile {
                optimal_for: vec![ProcessingMode::OutputGeneration, ProcessingMode::Consolidation],
                integration_capacity: 0.6,
                description: "Output generation and consolidation".to_string(),
            },
            OscillatoryPhase::Trough => ProcessingProfile {
                optimal_for: vec![ProcessingMode::Reset, ProcessingMode::Maintenance],
                integration_capacity: 0.3,
                description: "System reset and maintenance".to_string(),
            },
        }
    }

    /// Calculate time until reaching this phase from current phase
    pub fn time_until(&self, current_phase: f64, frequency: f64) -> f64 {
        use std::f64::consts::PI;
        let target_angle = match self {
            OscillatoryPhase::Peak => 0.0,
            OscillatoryPhase::Rising => 3.0 * PI / 2.0,
            OscillatoryPhase::Falling => PI / 2.0,
            OscillatoryPhase::Trough => PI,
        };

        let mut diff = target_angle - current_phase;
        if diff < 0.0 {
            diff += 2.0 * PI;
        }

        // time = angle / (2π * frequency)
        diff / (2.0 * PI * frequency)
    }

    /// Determine phase category from angle
    pub fn from_angle(angle: f64) -> Self {
        use std::f64::consts::PI;
        let normalized = ((angle % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);

        if normalized < PI / 4.0 || normalized >= 7.0 * PI / 4.0 {
            OscillatoryPhase::Peak
        } else if normalized < 3.0 * PI / 4.0 {
            OscillatoryPhase::Falling
        } else if normalized < 5.0 * PI / 4.0 {
            OscillatoryPhase::Trough
        } else {
            OscillatoryPhase::Rising
        }
    }
}

/// Processing profile for a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingProfile {
    /// Processing modes optimal for this phase
    pub optimal_for: Vec<ProcessingMode>,
    /// Integration capacity [0, 1]
    pub integration_capacity: f64,
    /// Description of this phase's function
    pub description: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY STATE (Full struct for oscillatory processing)
// ═══════════════════════════════════════════════════════════════════════════

/// State of oscillatory processing
///
/// Tracks the full oscillatory dynamics including phase, frequency, and coherence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryState {
    /// Current phase angle in radians [0, 2π)
    pub phase: f64,
    /// Oscillation frequency in Hz
    pub frequency: f64,
    /// Integrated information (Φ)
    pub phi: f64,
    /// Phase coherence [0, 1]
    pub coherence: f64,
    /// Categorized phase
    pub phase_category: OscillatoryPhase,
}

impl Default for OscillatoryState {
    fn default() -> Self {
        Self {
            phase: 0.0,
            frequency: 40.0, // Gamma band
            phi: 0.5,
            coherence: 1.0,
            phase_category: OscillatoryPhase::Peak,
        }
    }
}

impl OscillatoryState {
    /// Create new oscillatory state
    pub fn new(phase: f64, frequency: f64, phi: f64, coherence: f64) -> Self {
        Self {
            phase,
            frequency,
            phi,
            coherence,
            phase_category: OscillatoryPhase::from_angle(phase),
        }
    }

    /// Advance phase by time step
    pub fn advance(&mut self, dt: f64) {
        use std::f64::consts::PI;
        self.phase += 2.0 * PI * self.frequency * dt;
        self.phase = self.phase % (2.0 * PI);
        self.phase_category = OscillatoryPhase::from_angle(self.phase);
    }

    /// Get effective Φ (modulated by phase)
    pub fn effective_phi(&self) -> f64 {
        let phase_factor = self.phase_category.processing_profile().integration_capacity;
        self.phi * self.coherence * phase_factor
    }

    /// Predict future state
    pub fn predict(&self, dt: f64) -> OscillatoryState {
        let mut predicted = self.clone();
        predicted.advance(dt);
        predicted
    }
}

/// Sync state for oscillatory processing (simple enum version)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OscillatorySyncState {
    /// Synchronizing phase
    #[default]
    Synchronizing,
    /// Phase locked and processing
    PhaseLocked,
    /// Desynchronizing
    Desynchronizing,
}

// ═══════════════════════════════════════════════════════════════════════════
// ROUTING STRATEGY
// ═══════════════════════════════════════════════════════════════════════════

/// Routing strategy based on consciousness level (Φ)
///
/// Each strategy represents a different processing mode optimized for
/// the current level of integrated information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RoutingStrategy {
    /// Φ > 0.8: Full deliberation with all resources
    FullDeliberation,
    /// Φ ∈ [0.6, 0.8]: Standard processing with moderate resources
    #[default]
    StandardProcessing,
    /// Φ ∈ [0.4, 0.6]: Heuristic-guided with light resources
    HeuristicGuided,
    /// Φ ∈ [0.2, 0.4]: Fast patterns with minimal resources
    FastPatterns,
    /// Φ < 0.2: Reflexive emergency processing
    Reflexive,
    /// Uncertain state: Use ensemble strategies
    Ensemble,
    /// Preparing for higher consciousness state
    Preparatory,
}

impl RoutingStrategy {
    /// Get strategy from phi value
    pub fn from_phi(phi: f64) -> Self {
        if phi > 0.8 {
            RoutingStrategy::FullDeliberation
        } else if phi > 0.6 {
            RoutingStrategy::StandardProcessing
        } else if phi > 0.4 {
            RoutingStrategy::HeuristicGuided
        } else if phi > 0.2 {
            RoutingStrategy::FastPatterns
        } else {
            RoutingStrategy::Reflexive
        }
    }

    /// Get resource allocation factor [0.0, 1.0]
    pub fn resource_factor(&self) -> f64 {
        match self {
            RoutingStrategy::FullDeliberation => 1.0,
            RoutingStrategy::StandardProcessing => 0.7,
            RoutingStrategy::HeuristicGuided => 0.4,
            RoutingStrategy::FastPatterns => 0.2,
            RoutingStrategy::Reflexive => 0.1,
            RoutingStrategy::Ensemble => 0.8,
            RoutingStrategy::Preparatory => 0.5,
        }
    }

    /// Get expected latency in milliseconds
    pub fn expected_latency_ms(&self) -> u32 {
        match self {
            RoutingStrategy::FullDeliberation => 500,
            RoutingStrategy::StandardProcessing => 200,
            RoutingStrategy::HeuristicGuided => 100,
            RoutingStrategy::FastPatterns => 50,
            RoutingStrategy::Reflexive => 10,
            RoutingStrategy::Ensemble => 300,
            RoutingStrategy::Preparatory => 150,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ROUTING PLANS AND PREDICTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Predicted future routing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedRoute {
    /// Predicted phi at this step
    pub predicted_phi: f64,
    /// Recommended strategy
    pub strategy: RoutingStrategy,
    /// Confidence in prediction [0.0, 1.0]
    pub confidence: f64,
    /// Steps in the future
    pub steps_ahead: usize,
    /// Actions that led to this prediction
    pub actions: Vec<ConsciousnessAction>,
    /// Expected reward
    pub expected_reward: f64,
}

/// A routing plan with multiple future steps
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingPlan {
    /// Current state
    pub current_phi: f64,
    /// Current strategy
    pub current_strategy: RoutingStrategy,
    /// Predicted future routes
    pub predictions: Vec<PredictedRoute>,
    /// Recommended pre-allocation
    pub recommended_preallocation: HashMap<CognitiveResourceType, f64>,
    /// Transition warning: true if major strategy change expected
    pub transition_warning: bool,
    /// Expected phi trajectory
    pub phi_trajectory: Vec<f64>,
}

/// Outcome of a routing decision for learning
#[derive(Debug, Clone)]
pub struct RoutingOutcome {
    /// What we predicted
    pub predicted_phi: f64,
    /// What actually happened
    pub actual_phi: f64,
    /// Strategy we used
    pub strategy_used: RoutingStrategy,
    /// Whether prediction was accurate
    pub prediction_accurate: bool,
    /// Resources consumed
    pub resources_consumed: f64,
    /// Time taken in ms
    pub latency_ms: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY ROUTING TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Combined routing strategy (magnitude + phase)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CombinedRoutingStrategy {
    /// Magnitude-based strategy (from PredictiveRouter)
    pub magnitude_strategy: RoutingStrategy,
    /// Phase-based optimal mode
    pub phase_mode: ProcessingMode,
    /// Combined resource factor
    pub resource_factor: f64,
    /// Combined confidence
    pub confidence: f64,
}

/// Phase-locked routing plan
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaseLockedPlan {
    /// Underlying magnitude plan
    pub magnitude_plan: RoutingPlan,
    /// Current oscillatory phase (radians)
    pub current_phase: f64,
    /// Predicted oscillatory states for upcoming cycles
    pub predicted_states: Vec<OscillatoryState>,
    /// Optimal execution windows (phase, time_until)
    pub execution_windows: Vec<(OscillatoryPhase, f64)>,
    /// Combined routing strategy (magnitude + phase)
    pub combined_strategy: CombinedRoutingStrategy,
    /// Phase coherence measure [0, 1]
    pub phase_coherence: f64,
}

/// Scheduled operation for phase-locked execution
#[derive(Debug, Clone)]
pub struct ScheduledOperation {
    /// Operation ID
    pub id: u64,
    /// Target phase for execution
    pub target_phase: OscillatoryPhase,
    /// Time until target phase (seconds)
    pub time_until: f64,
    /// Operation type
    pub operation_type: ScheduledOpType,
    /// Priority [0.0, 1.0]
    pub priority: f64,
}

/// Type of scheduled operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledOpType {
    /// High-integration computation
    Integration,
    /// Information broadcast
    Broadcast,
    /// Memory consolidation
    Consolidation,
    /// Exploration/learning
    Exploration,
    /// Resource recovery
    Recovery,
}

// ═══════════════════════════════════════════════════════════════════════════
// ROUTER TRAIT
// ═══════════════════════════════════════════════════════════════════════════

/// Common trait for all cognitive routers
pub trait Router: Send + Sync {
    /// Get the name of this router
    fn name(&self) -> &'static str;

    /// Get the current routing strategy based on state
    fn current_strategy(&self, phi: f64) -> RoutingStrategy;

    /// Plan routing for the given state
    fn plan(&mut self, state: &LatentConsciousnessState) -> RoutingPlan;

    /// Execute routing and return the selected strategy
    fn execute(&mut self, state: &LatentConsciousnessState, action: ConsciousnessAction) -> RoutingStrategy;

    /// Record outcome for learning
    fn record_outcome(&mut self, outcome: RoutingOutcome);

    /// Get router-specific statistics
    fn stats(&self) -> RouterStats;
}

/// Generic router statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterStats {
    /// Total routing decisions made
    pub decisions_made: u64,
    /// Accurate predictions
    pub accurate_predictions: u64,
    /// Strategy transitions
    pub transitions: u64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Average phi error
    pub avg_phi_error: f64,
    /// Router-specific metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl RouterStats {
    /// Get prediction accuracy
    pub fn accuracy(&self) -> f64 {
        if self.decisions_made == 0 {
            0.0
        } else {
            self.accurate_predictions as f64 / self.decisions_made as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY ROUTER STUBS
// NOTE: OscillatoryRouter, OscillatoryRouterConfig, OscillatoryRouterStats, OscillatoryRouterSummary
// are now defined in oscillatory.rs module with full implementation (Phase 5G complete)

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_strategy_from_phi() {
        assert_eq!(RoutingStrategy::from_phi(0.9), RoutingStrategy::FullDeliberation);
        assert_eq!(RoutingStrategy::from_phi(0.7), RoutingStrategy::StandardProcessing);
        assert_eq!(RoutingStrategy::from_phi(0.5), RoutingStrategy::HeuristicGuided);
        assert_eq!(RoutingStrategy::from_phi(0.3), RoutingStrategy::FastPatterns);
        assert_eq!(RoutingStrategy::from_phi(0.1), RoutingStrategy::Reflexive);
    }

    #[test]
    fn test_routing_strategy_resource_factor() {
        assert_eq!(RoutingStrategy::FullDeliberation.resource_factor(), 1.0);
        assert_eq!(RoutingStrategy::Reflexive.resource_factor(), 0.1);
    }

    #[test]
    fn test_router_stats_accuracy() {
        let mut stats = RouterStats::default();
        assert_eq!(stats.accuracy(), 0.0);

        stats.decisions_made = 10;
        stats.accurate_predictions = 8;
        assert!((stats.accuracy() - 0.8).abs() < 0.001);
    }
}

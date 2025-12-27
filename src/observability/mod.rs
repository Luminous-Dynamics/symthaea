/*!
 * Observability & Telemetry for Symthaea HLB
 *
 * This module provides visibility into Symthaea's internal dynamics:
 * - Router selection decisions (with UCB1 bandit stats)
 * - GWT workspace ignitions (coalition formation)
 * - Φ (Integrated Information) measurements over time
 * - Primitive activation patterns
 * - Security authorization decisions
 * - Language understanding pipeline
 *
 * ## Philosophy
 *
 * **Observability-First AI**: You cannot trust what you cannot see.
 *
 * Traditional ML systems are black boxes. Symthaea is designed for transparency:
 * - Every consciousness event is traceable
 * - Every decision is explainable
 * - Every Φ measurement is recordable
 *
 * ## Architecture
 *
 * ```
 * SymthaeaObserver (trait)
 *     ↓
 * TraceObserver (JSON export)
 * TelemetryObserver (real-time metrics)
 * ConsoleObserver (debug logging)
 * NullObserver (no-op for production)
 * ```
 *
 * ## Usage
 *
 * ```rust
 * use symthaea::observability::{TraceObserver, create_shared_observer};
 * use symthaea::SymthaeaHLB;
 *
 * // Create shared observer
 * let observer = create_shared_observer(TraceObserver::new("trace.json")?);
 *
 * // Create Symthaea with observability
 * let mut brain = SymthaeaHLB::with_observer(16_384, 1_024, observer).await?;
 *
 * // Events are automatically recorded
 * brain.process("install nginx").await?;
 *
 * // Finalize trace (flush + summary)
 * brain.finalize_trace()?;
 * ```
 */

pub mod types;
pub mod trace_observer;
pub mod telemetry_observer;
pub mod console_observer;
pub mod null_observer;
pub mod correlation;       // Phase 3: Causal correlation tracking
pub mod causal_graph;      // Phase 3: Causal graph analysis
pub mod trace_analyzer;    // Phase 3: High-level trace analysis utilities
pub mod streaming_causal;  // Revolutionary Enhancement #1: Streaming causal analysis
pub mod pattern_library;   // Revolutionary Enhancement #2: Causal pattern recognition
pub mod probabilistic_inference;  // Revolutionary Enhancement #3: Probabilistic inference
pub mod causal_intervention;  // Revolutionary Enhancement #4 Phase 1: Causal intervention
pub mod counterfactual_reasoning;  // Revolutionary Enhancement #4 Phase 2: Counterfactual reasoning
pub mod action_planning;  // Revolutionary Enhancement #4 Phase 3: Action planning
pub mod causal_explanation;  // Revolutionary Enhancement #4 Phase 4: Causal explanations
pub mod byzantine_defense;   // Revolutionary Enhancement #5 Phase 1: Meta-Learning Byzantine Defense (FIXED API)
pub mod predictive_byzantine_defense;  // Revolutionary Enhancement #5 Phase 2: Real-time Predictive Defense
pub mod ml_explainability;   // Revolutionary Enhancement #6: Universal Causal Explainability for ML Models (FIXED: Import structure + all API errors)

pub use types::*;
pub use trace_observer::TraceObserver;
pub use telemetry_observer::TelemetryObserver;
pub use console_observer::ConsoleObserver;
pub use null_observer::NullObserver;
pub use correlation::{CorrelationContext, EventMetadata, ScopedParent};
pub use causal_graph::{CausalGraph, CausalNode, CausalEdge, EdgeType, CausalAnswer};
pub use trace_analyzer::{TraceAnalyzer, PerformanceSummary, CorrelationAnalysis, StatisticalSummary};
pub use streaming_causal::{
    StreamingCausalAnalyzer, StreamingConfig, StreamingStats,
    CausalInsight, AlertSeverity, AlertConfig,
};
pub use pattern_library::{
    MotifLibrary, CausalMotif, MotifMatch, MotifSeverity, MotifStats,
};
pub use probabilistic_inference::{
    ProbabilisticCausalGraph, ProbabilisticEdge, ProbabilisticConfig,
    ProbabilisticResult, ProbabilisticPrediction, UncertaintySource,
    BayesianInference,
};
pub use causal_intervention::{
    CausalInterventionEngine, InterventionSpec, InterventionType,
    InterventionResult,
};
pub use counterfactual_reasoning::{
    CounterfactualEngine, CounterfactualQuery, CounterfactualResult,
    Evidence, HiddenState,
};
pub use action_planning::{
    ActionPlanner, ActionPlan, PlannedIntervention,
    Goal, GoalDirection, PlannerConfig,
};
pub use causal_explanation::{
    ExplanationGenerator, CausalExplanation, ExplanationType,
    ExplanationLevel, VisualHints,
};
pub use byzantine_defense::{
    AttackModel, AttackType, SystemState, AttackPreconditions,
    AttackPattern, AttackSimulation, Countermeasure,
};
pub use predictive_byzantine_defense::{
    PredictiveDefender, PredictiveDefenseConfig, AttackWarning,
    CountermeasureDeployment, PredictiveDefenseStats,
};
pub use ml_explainability::{
    MLModelObserver, MLObserverConfig, ModelObservation, ObservationMetadata, MLObserverStats,
    CausalModelLearner, LearningStats,
    InteractiveExplainer, ExplainQuery, ExplanationResult,
    CounterfactualExplanation, ExplainerStats,
};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Core observability trait
///
/// Implement this to create custom observers for Symthaea.
/// All methods have default no-op implementations.
pub trait SymthaeaObserver: Send + Sync {
    /// Record router selection event
    fn record_router_selection(&mut self, event: RouterSelectionEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record workspace ignition (GWT activation)
    fn record_workspace_ignition(&mut self, event: WorkspaceIgnitionEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record Φ (integrated information) measurement
    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record primitive activation
    fn record_primitive_activation(&mut self, event: PrimitiveActivationEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record response generation
    fn record_response_generated(&mut self, event: ResponseGeneratedEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record security check
    fn record_security_check(&mut self, event: SecurityCheckEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record error occurrence
    fn record_error(&mut self, event: ErrorEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record language understanding pipeline step
    fn record_language_step(&mut self, event: LanguageStepEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record Narrative Self state (Revolutionary Improvement #73)
    fn record_narrative_self(&mut self, event: NarrativeSelfEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record Cross-Modal Binding event (Revolutionary Improvement #72)
    fn record_cross_modal_binding(&mut self, event: CrossModalBindingEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Record GWT Integration event (Revolutionary Improvement #73)
    fn record_gwt_integration(&mut self, event: GWTIntegrationEvent) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// Finalize observation (close files, compute summaries, etc.)
    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get session statistics
    fn get_stats(&self) -> Result<ObserverStats> {
        Ok(ObserverStats::default())
    }
}

/// Thread-safe observer wrapper
pub type SharedObserver = Arc<RwLock<Box<dyn SymthaeaObserver>>>;

/// Create a shared observer from any observer implementation
pub fn create_shared_observer<O: SymthaeaObserver + 'static>(observer: O) -> SharedObserver {
    Arc::new(RwLock::new(Box::new(observer)))
}

/// Observer statistics
#[derive(Debug, Clone, Default)]
pub struct ObserverStats {
    pub total_events: usize,
    pub router_selections: usize,
    pub workspace_ignitions: usize,
    pub phi_measurements: usize,
    pub primitive_activations: usize,
    pub responses_generated: usize,
    pub security_checks: usize,
    pub errors: usize,
    pub language_steps: usize,
    /// **NEW**: Narrative Self events (Revolutionary Improvement #73)
    pub narrative_self_events: usize,
    /// **NEW**: Cross-Modal binding events (Revolutionary Improvement #72)
    pub cross_modal_bindings: usize,
    /// **NEW**: GWT integration events (Revolutionary Improvement #73)
    pub gwt_integrations: usize,
}

impl ObserverStats {
    pub fn increment_event(&mut self, event_type: &str) {
        self.total_events += 1;
        match event_type {
            "router_selection" => self.router_selections += 1,
            "workspace_ignition" => self.workspace_ignitions += 1,
            "phi_measurement" => self.phi_measurements += 1,
            "primitive_activation" => self.primitive_activations += 1,
            "response_generated" => self.responses_generated += 1,
            "security_check" => self.security_checks += 1,
            "error" => self.errors += 1,
            "language_step" => self.language_steps += 1,
            // Revolutionary Improvement #72 & #73
            "narrative_self" => self.narrative_self_events += 1,
            "cross_modal_binding" => self.cross_modal_bindings += 1,
            "gwt_integration" => self.gwt_integrations += 1,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_observer() {
        let mut observer = NullObserver::new();

        // Should all succeed as no-ops
        assert!(observer.record_router_selection(RouterSelectionEvent::default()).is_ok());
        assert!(observer.record_workspace_ignition(WorkspaceIgnitionEvent::default()).is_ok());
        assert!(observer.finalize().is_ok());
    }

    #[test]
    fn test_shared_observer() {
        let observer = NullObserver::new();
        let shared = create_shared_observer(observer);

        // Should be able to clone and use
        let _cloned = shared.clone();
    }

    #[test]
    fn test_observer_stats() {
        let mut stats = ObserverStats::default();

        stats.increment_event("router_selection");
        stats.increment_event("phi_measurement");
        stats.increment_event("phi_measurement");

        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.router_selections, 1);
        assert_eq!(stats.phi_measurements, 2);
    }
}

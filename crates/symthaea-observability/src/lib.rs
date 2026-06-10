// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observability Module - Telemetry and Monitoring
//!
//! Comprehensive observability framework for consciousness-aware systems:
//! - Event recording and tracing
//! - Phi measurement monitoring
//! - Router selection observability
//! - Causal tracing with parent-child relationships
//! - Performance metrics and statistics
//! - Counterfactual reasoning with uncertainty propagation

#![deny(unsafe_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

pub mod causal_observer;
pub mod counterfactual_reasoning;

pub use causal_observer::CausalTraceObserver;
pub use counterfactual_reasoning::{
    CausalLink, CausalNode, CounterfactualConfig, CounterfactualEngine, CounterfactualResult,
    CounterfactualUncertainty,
};

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVER TRAIT AND CORE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Core observer trait for Symthaea observability
pub trait SymthaeaObserver: Send + Sync {
    /// Record a Phi measurement event
    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<()>;

    /// Record a router selection event
    fn record_router_selection(&mut self, event: RouterSelectionEvent) -> Result<()>;

    /// Record a workspace ignition event
    fn record_workspace_ignition(&mut self, event: WorkspaceIgnitionEvent) -> Result<()>;

    /// Record a primitive activation event
    fn record_primitive_activation(&mut self, event: PrimitiveActivationEvent) -> Result<()>;

    /// Record a response generated event
    fn record_response_generated(&mut self, event: ResponseGeneratedEvent) -> Result<()>;

    /// Record a security check event
    fn record_security_check(&mut self, event: SecurityCheckEvent) -> Result<()>;

    /// Record an error event
    fn record_error(&mut self, event: ErrorEvent) -> Result<()>;

    /// Record a language processing step
    fn record_language_step(&mut self, event: LanguageStepEvent) -> Result<()>;

    /// Record a narrative self event
    fn record_narrative_self(&mut self, event: NarrativeSelfEvent) -> Result<()>;

    /// Record a cross-modal binding event
    fn record_cross_modal_binding(&mut self, event: CrossModalBindingEvent) -> Result<()>;

    /// Record a GWT integration event
    fn record_gwt_integration(&mut self, event: GWTIntegrationEvent) -> Result<()>;

    /// Record a Broca pipeline event (Reason-then-Generate observability)
    fn record_broca_pipeline(&mut self, event: BrocaPipelineEvent) -> Result<()>;

    /// Flush any buffered events
    fn flush(&mut self) -> Result<()>;

    /// Get observer statistics
    fn stats(&self) -> ObserverStats;
}

/// Statistics for an observer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObserverStats {
    /// Total events recorded
    pub total_events: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Total bytes written
    pub bytes_written: u64,
    /// Average event latency
    pub avg_latency_us: f64,
    /// Error count
    pub error_count: u64,
    /// Last event timestamp
    pub last_event_time: Option<u64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORRELATION CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/// Context for tracking causal correlations between events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationContext {
    /// Correlation ID for this context
    pub correlation_id: String,
    /// Stack of parent event IDs
    parent_stack: Vec<String>,
    /// Additional tags
    pub tags: HashMap<String, String>,
    /// Start time
    pub start_time: u64,
}

impl CorrelationContext {
    /// Create new correlation context
    pub fn new(correlation_id: impl Into<String>) -> Self {
        Self {
            correlation_id: correlation_id.into(),
            parent_stack: Vec::new(),
            tags: HashMap::new(),
            start_time: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Push a parent onto the stack
    pub fn push_parent(&mut self, parent_id: impl Into<String>) {
        self.parent_stack.push(parent_id.into());
    }

    /// Pop the current parent
    pub fn pop_parent(&mut self) -> Option<String> {
        self.parent_stack.pop()
    }

    /// Get current parent ID
    pub fn current_parent(&self) -> Option<&String> {
        self.parent_stack.last()
    }

    /// Create event metadata with current context
    pub fn create_event_metadata(&self) -> EventMetadata {
        EventMetadata {
            correlation_id: self.correlation_id.clone(),
            parent_id: self.current_parent().cloned(),
            tags: self.tags.clone(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }
}

/// Metadata attached to events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Correlation ID linking related events
    pub correlation_id: String,
    /// Parent event ID (for causal linking)
    pub parent_id: Option<String>,
    /// Additional tags
    pub tags: HashMap<String, String>,
    /// Event timestamp (microseconds)
    pub timestamp: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi measurement event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurementEvent {
    /// Event ID
    pub id: String,
    /// Phi value
    pub phi: f64,
    /// Method used
    pub method: String,
    /// System size
    pub system_size: usize,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Additional metadata
    pub metadata: EventMetadata,
}

/// Router selection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterSelectionEvent {
    /// Event ID
    pub id: String,
    /// Selected router
    pub router: String,
    /// Input summary
    pub input_summary: String,
    /// Selection scores
    pub scores: HashMap<String, f64>,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Workspace ignition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceIgnitionEvent {
    /// Event ID
    pub id: String,
    /// Workspace name
    pub workspace: String,
    /// Components initialized
    pub components: Vec<String>,
    /// Ignition duration
    pub duration_us: u64,
    /// Success flag
    pub success: bool,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Primitive activation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveActivationEvent {
    /// Event ID
    pub id: String,
    /// Primitive ID
    pub primitive_id: String,
    /// Activation strength
    pub activation: f64,
    /// Input pattern hash
    pub input_hash: String,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Response generated event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseGeneratedEvent {
    /// Event ID
    pub id: String,
    /// Response type
    pub response_type: String,
    /// Token count
    pub token_count: usize,
    /// Generation time
    pub duration_us: u64,
    /// Quality score
    pub quality_score: f64,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Security check event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCheckEvent {
    /// Event ID
    pub id: String,
    /// Check type
    pub check_type: String,
    /// Result (passed/failed)
    pub passed: bool,
    /// Details
    pub details: String,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// Event ID
    pub id: String,
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Language processing step event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStepEvent {
    /// Event ID
    pub id: String,
    /// Step type
    pub step: String,
    /// Input text hash
    pub input_hash: String,
    /// Processing time
    pub duration_us: u64,
    /// Output summary
    pub output_summary: String,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Narrative self event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSelfEvent {
    /// Event ID
    pub id: String,
    /// Narrative component
    pub component: String,
    /// Coherence score
    pub coherence: f64,
    /// Update type
    pub update_type: String,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Cross-modal binding event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalBindingEvent {
    /// Event ID
    pub id: String,
    /// Modalities bound
    pub modalities: Vec<String>,
    /// Binding strength
    pub strength: f64,
    /// Integration phi
    pub integration_phi: f64,
    /// Metadata
    pub metadata: EventMetadata,
}

/// Global Workspace Theory integration event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GWTIntegrationEvent {
    /// Event ID
    pub id: String,
    /// Broadcast message
    pub broadcast: String,
    /// Participating modules
    pub modules: Vec<String>,
    /// Global workspace activation
    pub activation: f64,
    /// Metadata
    pub metadata: EventMetadata,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BROCA PIPELINE OBSERVABILITY (Reason-then-Generate)
// ═══════════════════════════════════════════════════════════════════════════════

/// Broca Pipeline event for observability of the Reason-then-Generate architecture.
///
/// Tracks each phase of the HDC+LTC → Structured Thought → LLM Translation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrocaPipelineEvent {
    /// Event ID
    pub id: String,
    /// Correlation ID for tracing related events
    pub correlation_id: String,
    /// Phase of the pipeline
    pub phase: BrocaPhase,
    /// Input hash (for correlation, not raw content)
    pub input_hash: String,
    /// Processing duration in microseconds
    pub duration_us: u64,
    /// Epistemic status determined by HDC classification
    pub epistemic_status: String,
    /// Semantic intent classification
    pub semantic_intent: String,
    /// HDC familiarity score (0.0-1.0)
    pub familiarity: f32,
    /// HDC novelty score (0.0-1.0)
    pub novelty: f32,
    /// Consciousness level (phi)
    pub phi: f64,
    /// Whether translation fidelity was verified
    pub fidelity_verified: bool,
    /// Response type classification
    pub response_type: String,
    /// Relationship stage
    pub relationship_stage: String,
    /// Trust level
    pub trust: f32,
    /// Additional metadata
    pub metadata: EventMetadata,
}

/// Phases of the Broca pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrocaPhase {
    /// Phase 1: Perception (input encoding)
    Perception,
    /// Phase 2: Cognition (mind tick)
    Cognition,
    /// Phase 3: Extraction (structured thought)
    Extraction,
    /// Phase 4: Relational enrichment
    RelationalEnrichment,
    /// Phase 5: Translation (LLM Broca's Area)
    Translation,
    /// Phase 6: Fidelity verification
    FidelityVerification,
    /// Phase 7: Partnership update
    PartnershipUpdate,
    /// Complete pipeline
    Complete,
}

impl std::fmt::Display for BrocaPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrocaPhase::Perception => write!(f, "perception"),
            BrocaPhase::Cognition => write!(f, "cognition"),
            BrocaPhase::Extraction => write!(f, "extraction"),
            BrocaPhase::RelationalEnrichment => write!(f, "relational_enrichment"),
            BrocaPhase::Translation => write!(f, "translation"),
            BrocaPhase::FidelityVerification => write!(f, "fidelity_verification"),
            BrocaPhase::PartnershipUpdate => write!(f, "partnership_update"),
            BrocaPhase::Complete => write!(f, "complete"),
        }
    }
}

/// Aggregate statistics for Broca pipeline observability
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrocaPipelineStats {
    /// Total queries processed
    pub total_queries: u64,
    /// Queries by epistemic status
    pub epistemic_distribution: HashMap<String, u64>,
    /// Queries by semantic intent
    pub intent_distribution: HashMap<String, u64>,
    /// Fidelity verification pass rate
    pub fidelity_pass_count: u64,
    /// Fidelity verification fail count
    pub fidelity_fail_count: u64,
    /// Average familiarity score
    pub avg_familiarity: f64,
    /// Average novelty score
    pub avg_novelty: f64,
    /// Average processing time (microseconds)
    pub avg_duration_us: f64,
    /// High novelty queries (potential hallucination triggers)
    pub high_novelty_count: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// NO-OP OBSERVER
// ═══════════════════════════════════════════════════════════════════════════════

/// No-op observer that discards all events
#[derive(Debug, Clone, Default)]
pub struct NoOpObserver {
    stats: ObserverStats,
}

impl NoOpObserver {
    /// Create new no-op observer
    pub fn new() -> Self {
        Self::default()
    }
}

impl SymthaeaObserver for NoOpObserver {
    fn record_phi_measurement(&mut self, _event: PhiMeasurementEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_router_selection(&mut self, _event: RouterSelectionEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_workspace_ignition(&mut self, _event: WorkspaceIgnitionEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_primitive_activation(&mut self, _event: PrimitiveActivationEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_response_generated(&mut self, _event: ResponseGeneratedEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_security_check(&mut self, _event: SecurityCheckEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_error(&mut self, _event: ErrorEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_language_step(&mut self, _event: LanguageStepEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_narrative_self(&mut self, _event: NarrativeSelfEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_cross_modal_binding(&mut self, _event: CrossModalBindingEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_gwt_integration(&mut self, _event: GWTIntegrationEvent) -> Result<()> {
        self.stats.total_events += 1;
        Ok(())
    }

    fn record_broca_pipeline(&mut self, _event: BrocaPipelineEvent) -> Result<()> {
        self.stats.total_events += 1;
        *self
            .stats
            .events_by_type
            .entry("broca_pipeline".to_string())
            .or_insert(0) += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn stats(&self) -> ObserverStats {
        self.stats.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED OBSERVER
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::{Arc, RwLock};

/// Thread-safe shared observer wrapper
pub type SharedObserver<O> = Arc<RwLock<O>>;

/// Create a shared no-op observer
pub fn no_op_observer() -> SharedObserver<NoOpObserver> {
    Arc::new(RwLock::new(NoOpObserver::new()))
}

/// Null observer (alias for NoOpObserver for API compatibility)
pub type NullObserver = NoOpObserver;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI COMPONENTS (for compatibility with symthaea-core)
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi measurement components (local definition for compatibility)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhiComponents {
    /// Raw phi value
    pub phi: f64,
    /// Integrated information
    pub integrated_info: f64,
    /// System entropy
    pub entropy: f64,
    /// Number of elements
    pub element_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn default_meta() -> EventMetadata {
        EventMetadata::default()
    }

    fn phi_event(id: &str) -> PhiMeasurementEvent {
        PhiMeasurementEvent {
            id: id.to_string(),
            phi: 0.5,
            method: "IIT".to_string(),
            system_size: 10,
            duration_us: 100,
            metadata: default_meta(),
        }
    }

    fn router_event(id: &str) -> RouterSelectionEvent {
        RouterSelectionEvent {
            id: id.to_string(),
            router: "thalamic".to_string(),
            input_summary: "test".to_string(),
            scores: HashMap::new(),
            metadata: default_meta(),
        }
    }

    fn workspace_event(id: &str) -> WorkspaceIgnitionEvent {
        WorkspaceIgnitionEvent {
            id: id.to_string(),
            workspace: "default".to_string(),
            components: vec!["core".to_string()],
            duration_us: 50,
            success: true,
            metadata: default_meta(),
        }
    }

    fn primitive_event(id: &str) -> PrimitiveActivationEvent {
        PrimitiveActivationEvent {
            id: id.to_string(),
            primitive_id: "prim_1".to_string(),
            activation: 0.75,
            input_hash: "abc123".to_string(),
            metadata: default_meta(),
        }
    }

    fn response_event(id: &str) -> ResponseGeneratedEvent {
        ResponseGeneratedEvent {
            id: id.to_string(),
            response_type: "text".to_string(),
            token_count: 42,
            duration_us: 200,
            quality_score: 0.9,
            metadata: default_meta(),
        }
    }

    fn security_event(id: &str) -> SecurityCheckEvent {
        SecurityCheckEvent {
            id: id.to_string(),
            check_type: "sandbox".to_string(),
            passed: true,
            details: "all clear".to_string(),
            metadata: default_meta(),
        }
    }

    fn error_event(id: &str) -> ErrorEvent {
        ErrorEvent {
            id: id.to_string(),
            code: "E001".to_string(),
            message: "test error".to_string(),
            stack_trace: None,
            severity: ErrorSeverity::Error,
            metadata: default_meta(),
        }
    }

    fn language_event(id: &str) -> LanguageStepEvent {
        LanguageStepEvent {
            id: id.to_string(),
            step: "tokenize".to_string(),
            input_hash: "hash1".to_string(),
            duration_us: 30,
            output_summary: "tokenized ok".to_string(),
            metadata: default_meta(),
        }
    }

    fn narrative_event(id: &str) -> NarrativeSelfEvent {
        NarrativeSelfEvent {
            id: id.to_string(),
            component: "autobiography".to_string(),
            coherence: 0.85,
            update_type: "append".to_string(),
            metadata: default_meta(),
        }
    }

    fn cross_modal_event(id: &str) -> CrossModalBindingEvent {
        CrossModalBindingEvent {
            id: id.to_string(),
            modalities: vec!["visual".to_string(), "auditory".to_string()],
            strength: 0.7,
            integration_phi: 0.4,
            metadata: default_meta(),
        }
    }

    fn gwt_event(id: &str) -> GWTIntegrationEvent {
        GWTIntegrationEvent {
            id: id.to_string(),
            broadcast: "alert".to_string(),
            modules: vec!["prefrontal".to_string(), "hippocampal".to_string()],
            activation: 0.6,
            metadata: default_meta(),
        }
    }

    fn broca_event(id: &str) -> BrocaPipelineEvent {
        BrocaPipelineEvent {
            id: id.to_string(),
            correlation_id: "broca_corr".to_string(),
            phase: BrocaPhase::Perception,
            input_hash: "input_hash".to_string(),
            duration_us: 120,
            epistemic_status: "known".to_string(),
            semantic_intent: "query".to_string(),
            familiarity: 0.8,
            novelty: 0.2,
            phi: 0.45,
            fidelity_verified: true,
            response_type: "answer".to_string(),
            relationship_stage: "trust".to_string(),
            trust: 0.9,
            metadata: default_meta(),
        }
    }

    // ── Existing tests (preserved) ──────────────────────────────────────

    #[test]
    fn test_correlation_context() {
        let mut ctx = CorrelationContext::new("test-correlation");
        ctx.push_parent("parent-1");
        assert_eq!(ctx.current_parent(), Some(&"parent-1".to_string()));

        ctx.push_parent("parent-2");
        assert_eq!(ctx.current_parent(), Some(&"parent-2".to_string()));

        ctx.pop_parent();
        assert_eq!(ctx.current_parent(), Some(&"parent-1".to_string()));
    }

    #[test]
    fn test_noop_observer() {
        let mut obs = NoOpObserver::new();
        let event = PhiMeasurementEvent {
            id: "test".to_string(),
            phi: 0.5,
            method: "IIT".to_string(),
            system_size: 10,
            duration_us: 100,
            metadata: EventMetadata::default(),
        };
        obs.record_phi_measurement(event).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    // ── NoOpObserver: each event type increments total_events ────────────

    #[test]
    fn test_noop_record_phi_measurement() {
        let mut obs = NoOpObserver::new();
        obs.record_phi_measurement(phi_event("p1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_router_selection() {
        let mut obs = NoOpObserver::new();
        obs.record_router_selection(router_event("r1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_workspace_ignition() {
        let mut obs = NoOpObserver::new();
        obs.record_workspace_ignition(workspace_event("w1"))
            .unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_primitive_activation() {
        let mut obs = NoOpObserver::new();
        obs.record_primitive_activation(primitive_event("pa1"))
            .unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_response_generated() {
        let mut obs = NoOpObserver::new();
        obs.record_response_generated(response_event("rg1"))
            .unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_security_check() {
        let mut obs = NoOpObserver::new();
        obs.record_security_check(security_event("sc1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_error() {
        let mut obs = NoOpObserver::new();
        obs.record_error(error_event("e1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_language_step() {
        let mut obs = NoOpObserver::new();
        obs.record_language_step(language_event("ls1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_narrative_self() {
        let mut obs = NoOpObserver::new();
        obs.record_narrative_self(narrative_event("ns1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_cross_modal_binding() {
        let mut obs = NoOpObserver::new();
        obs.record_cross_modal_binding(cross_modal_event("cm1"))
            .unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_gwt_integration() {
        let mut obs = NoOpObserver::new();
        obs.record_gwt_integration(gwt_event("g1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
    }

    #[test]
    fn test_noop_record_broca_pipeline() {
        let mut obs = NoOpObserver::new();
        obs.record_broca_pipeline(broca_event("bp1")).unwrap();
        assert_eq!(obs.stats().total_events, 1);
        // broca_pipeline also tracks by-type
        assert_eq!(
            *obs.stats()
                .events_by_type
                .get("broca_pipeline")
                .unwrap_or(&0),
            1
        );
    }

    // ── Stats accumulation across many observe() calls ──────────────────

    #[test]
    fn test_stats_accumulation_across_mixed_events() {
        let mut obs = NoOpObserver::new();

        obs.record_phi_measurement(phi_event("p1")).unwrap();
        obs.record_router_selection(router_event("r1")).unwrap();
        obs.record_error(error_event("e1")).unwrap();
        obs.record_workspace_ignition(workspace_event("w1"))
            .unwrap();
        obs.record_broca_pipeline(broca_event("bp1")).unwrap();
        obs.record_broca_pipeline(broca_event("bp2")).unwrap();

        let stats = obs.stats();
        assert_eq!(stats.total_events, 6);
        // broca_pipeline is the only type tracked by NoOpObserver
        assert_eq!(*stats.events_by_type.get("broca_pipeline").unwrap_or(&0), 2);
    }

    // ── Flush is no-op but succeeds ─────────────────────────────────────

    #[test]
    fn test_noop_flush() {
        let mut obs = NoOpObserver::new();
        assert!(obs.flush().is_ok());
    }

    // ── EventMetadata tag operations and parent stack ────────────────────

    #[test]
    fn test_event_metadata_from_context_with_tags() {
        let mut ctx = CorrelationContext::new("ctx-1");
        ctx.add_tag("env", "test");
        ctx.add_tag("region", "us-east");

        let meta = ctx.create_event_metadata();
        assert_eq!(meta.correlation_id, "ctx-1");
        assert_eq!(meta.tags.get("env").map(|s| s.as_str()), Some("test"));
        assert_eq!(meta.tags.get("region").map(|s| s.as_str()), Some("us-east"));
        assert!(meta.parent_id.is_none()); // no parents pushed
        assert!(meta.timestamp > 0);
    }

    #[test]
    fn test_event_metadata_parent_from_stack() {
        let mut ctx = CorrelationContext::new("ctx-2");
        ctx.push_parent("scope-a");
        ctx.push_parent("scope-b");

        let meta = ctx.create_event_metadata();
        // parent_id should be the top of the stack
        assert_eq!(meta.parent_id, Some("scope-b".to_string()));
    }

    #[test]
    fn test_correlation_context_pop_empty_returns_none() {
        let mut ctx = CorrelationContext::new("empty");
        assert!(ctx.pop_parent().is_none());
        assert!(ctx.current_parent().is_none());
    }

    #[test]
    fn test_correlation_context_tag_overwrite() {
        let mut ctx = CorrelationContext::new("ctx-3");
        ctx.add_tag("key", "value1");
        ctx.add_tag("key", "value2");
        assert_eq!(ctx.tags.get("key").map(|s| s.as_str()), Some("value2"));
    }

    // ── SharedObserver / no_op_observer ──────────────────────────────────

    #[test]
    fn test_shared_no_op_observer() {
        let shared = no_op_observer();
        {
            let mut obs = shared.write().unwrap();
            obs.record_phi_measurement(phi_event("shared_p1")).unwrap();
            obs.record_error(error_event("shared_e1")).unwrap();
        }
        let stats = shared.read().unwrap().stats();
        assert_eq!(stats.total_events, 2);
    }

    // ── PhiComponents default ───────────────────────────────────────────

    #[test]
    fn test_phi_components_default() {
        let pc = PhiComponents::default();
        assert_eq!(pc.phi, 0.0);
        assert_eq!(pc.integrated_info, 0.0);
        assert_eq!(pc.entropy, 0.0);
        assert_eq!(pc.element_count, 0);
    }

    // ── BrocaPhase Display ──────────────────────────────────────────────

    #[test]
    fn test_broca_phase_display() {
        assert_eq!(format!("{}", BrocaPhase::Perception), "perception");
        assert_eq!(format!("{}", BrocaPhase::Cognition), "cognition");
        assert_eq!(format!("{}", BrocaPhase::Extraction), "extraction");
        assert_eq!(
            format!("{}", BrocaPhase::RelationalEnrichment),
            "relational_enrichment"
        );
        assert_eq!(format!("{}", BrocaPhase::Translation), "translation");
        assert_eq!(
            format!("{}", BrocaPhase::FidelityVerification),
            "fidelity_verification"
        );
        assert_eq!(
            format!("{}", BrocaPhase::PartnershipUpdate),
            "partnership_update"
        );
        assert_eq!(format!("{}", BrocaPhase::Complete), "complete");
    }

    // ── ErrorSeverity variants ──────────────────────────────────────────

    #[test]
    fn test_error_severity_equality() {
        assert_eq!(ErrorSeverity::Info, ErrorSeverity::Info);
        assert_eq!(ErrorSeverity::Warning, ErrorSeverity::Warning);
        assert_eq!(ErrorSeverity::Error, ErrorSeverity::Error);
        assert_eq!(ErrorSeverity::Critical, ErrorSeverity::Critical);
        assert_ne!(ErrorSeverity::Info, ErrorSeverity::Critical);
    }

    // ── ObserverStats default ───────────────────────────────────────────

    #[test]
    fn test_observer_stats_default() {
        let stats = ObserverStats::default();
        assert_eq!(stats.total_events, 0);
        assert!(stats.events_by_type.is_empty());
        assert_eq!(stats.bytes_written, 0);
        assert_eq!(stats.avg_latency_us, 0.0);
        assert_eq!(stats.error_count, 0);
        assert!(stats.last_event_time.is_none());
    }
}

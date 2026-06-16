// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
 * Causal Trace Observer (Phase 4)
 *
 * Wrapper observer that automatically injects causal correlation metadata
 * into events, enabling the CausalGraph to build parent-child relationships.
 *
 * ## Usage
 *
 * ```rust,ignore
 * use symthaea_observability::{
 *     CausalTraceObserver, CorrelationContext,
 * };
 *
 * // Create base observer
 * let base = NullObserver::new();
 *
 * // Wrap with causal tracking
 * let correlation_id = "req_user_query_42";
 * let mut observer = CausalTraceObserver::new(base, correlation_id);
 *
 * // Events automatically get metadata with parent-child relationships
 * observer.begin_scope("phi_measurement"); // Push parent
 * observer.record_phi_measurement(event)?;
 *
 * observer.begin_scope("routing"); // Nested scope
 * observer.record_router_selection(event)?; // Has phi_measurement as parent
 * observer.end_scope(); // Pop routing
 *
 * observer.end_scope(); // Pop phi_measurement
 * ```
 */

use anyhow::Result;

use crate::{
    BrocaPipelineEvent, CorrelationContext, CrossModalBindingEvent, ErrorEvent, EventMetadata,
    GWTIntegrationEvent, LanguageStepEvent, NarrativeSelfEvent, ObserverStats, PhiMeasurementEvent,
    PrimitiveActivationEvent, ResponseGeneratedEvent, RouterSelectionEvent, SecurityCheckEvent,
    SymthaeaObserver, WorkspaceIgnitionEvent,
};

/// Observer wrapper that injects causal correlation metadata
pub struct CausalTraceObserver<O: SymthaeaObserver> {
    /// Inner observer to delegate to
    inner: O,
    /// Correlation context for tracking parent-child relationships
    context: CorrelationContext,
    /// Last event ID (for automatic parent linking)
    last_event_id: Option<String>,
    /// Whether to automatically link sequential events
    auto_link: bool,
    /// Event count for tracking
    event_count: u64,
}

impl<O: SymthaeaObserver> CausalTraceObserver<O> {
    /// Create new causal observer wrapping an inner observer
    pub fn new(inner: O, correlation_id: impl Into<String>) -> Self {
        Self {
            inner,
            context: CorrelationContext::new(correlation_id),
            last_event_id: None,
            auto_link: false,
            event_count: 0,
        }
    }

    /// Enable auto-linking (each event becomes parent of the next)
    pub fn with_auto_link(mut self) -> Self {
        self.auto_link = true;
        self
    }

    /// Begin a causal scope (events in this scope have current parent)
    pub fn begin_scope(&mut self, event_id: impl Into<String>) {
        self.context.push_parent(event_id);
    }

    /// End current causal scope
    pub fn end_scope(&mut self) {
        self.context.pop_parent();
    }

    /// Get reference to the correlation context
    pub fn context(&self) -> &CorrelationContext {
        &self.context
    }

    /// Get mutable reference to the correlation context
    pub fn context_mut(&mut self) -> &mut CorrelationContext {
        &mut self.context
    }

    /// Get the inner observer (consumes self)
    pub fn into_inner(self) -> O {
        self.inner
    }

    /// Get event count
    pub fn event_count(&self) -> u64 {
        self.event_count
    }

    /// Create event metadata with proper parent linking
    fn create_metadata(&mut self) -> EventMetadata {
        let metadata = self.context.create_event_metadata();
        self.event_count += 1;

        if self.auto_link {
            // After creating, push this as parent for next event
            if self.last_event_id.is_some() {
                self.context.pop_parent(); // Remove old auto-parent
            }
            // Use correlation_id as the event identifier
            self.last_event_id = Some(metadata.correlation_id.clone());
            self.context.push_parent(&metadata.correlation_id);
        }

        metadata
    }
}

impl<O: SymthaeaObserver> SymthaeaObserver for CausalTraceObserver<O> {
    fn record_router_selection(&mut self, mut event: RouterSelectionEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_router_selection(event)
    }

    fn record_workspace_ignition(&mut self, mut event: WorkspaceIgnitionEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_workspace_ignition(event)
    }

    fn record_phi_measurement(&mut self, mut event: PhiMeasurementEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_phi_measurement(event)
    }

    fn record_primitive_activation(&mut self, mut event: PrimitiveActivationEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_primitive_activation(event)
    }

    fn record_response_generated(&mut self, mut event: ResponseGeneratedEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_response_generated(event)
    }

    fn record_security_check(&mut self, mut event: SecurityCheckEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_security_check(event)
    }

    fn record_error(&mut self, mut event: ErrorEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_error(event)
    }

    fn record_language_step(&mut self, mut event: LanguageStepEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_language_step(event)
    }

    fn record_narrative_self(&mut self, mut event: NarrativeSelfEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_narrative_self(event)
    }

    fn record_cross_modal_binding(&mut self, mut event: CrossModalBindingEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_cross_modal_binding(event)
    }

    fn record_gwt_integration(&mut self, mut event: GWTIntegrationEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_gwt_integration(event)
    }

    fn record_broca_pipeline(&mut self, mut event: BrocaPipelineEvent) -> Result<()> {
        event.metadata = self.create_metadata();
        self.inner.record_broca_pipeline(event)
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn stats(&self) -> ObserverStats {
        self.inner.stats()
    }
}

/// Convenience trait for events that can have metadata injected
pub trait WithCausalMetadata {
    /// Set metadata on this event
    fn set_metadata(&mut self, metadata: EventMetadata);

    /// Get metadata from this event
    fn get_metadata(&self) -> &EventMetadata;
}

// Implement for all event types
impl WithCausalMetadata for RouterSelectionEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for WorkspaceIgnitionEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for PhiMeasurementEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for PrimitiveActivationEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for ResponseGeneratedEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for SecurityCheckEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for ErrorEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for LanguageStepEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for NarrativeSelfEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for CrossModalBindingEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for GWTIntegrationEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

impl WithCausalMetadata for BrocaPipelineEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) {
        self.metadata = metadata;
    }
    fn get_metadata(&self) -> &EventMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BrocaPhase, BrocaPipelineEvent, CrossModalBindingEvent, ErrorEvent, ErrorSeverity,
        GWTIntegrationEvent, LanguageStepEvent, NarrativeSelfEvent, NullObserver,
        PrimitiveActivationEvent, ResponseGeneratedEvent, SecurityCheckEvent,
    };

    fn make_phi_event() -> PhiMeasurementEvent {
        PhiMeasurementEvent {
            id: "test_phi".to_string(),
            phi: 0.5,
            method: "IIT".to_string(),
            system_size: 10,
            duration_us: 100,
            metadata: EventMetadata::default(),
        }
    }

    fn make_router_event() -> RouterSelectionEvent {
        RouterSelectionEvent {
            id: "test_router".to_string(),
            router: "thalamic".to_string(),
            input_summary: "test input".to_string(),
            scores: std::collections::HashMap::new(),
            metadata: EventMetadata::default(),
        }
    }

    fn make_workspace_event() -> WorkspaceIgnitionEvent {
        WorkspaceIgnitionEvent {
            id: "test_workspace".to_string(),
            workspace: "default".to_string(),
            components: vec!["core".to_string()],
            duration_us: 50,
            success: true,
            metadata: EventMetadata::default(),
        }
    }

    fn make_error_event() -> ErrorEvent {
        ErrorEvent {
            id: "test_error".to_string(),
            code: "E001".to_string(),
            message: "something broke".to_string(),
            stack_trace: None,
            severity: ErrorSeverity::Error,
            metadata: EventMetadata::default(),
        }
    }

    fn make_primitive_event() -> PrimitiveActivationEvent {
        PrimitiveActivationEvent {
            id: "test_prim".to_string(),
            primitive_id: "prim_1".to_string(),
            activation: 0.8,
            input_hash: "h1".to_string(),
            metadata: EventMetadata::default(),
        }
    }

    fn make_response_event() -> ResponseGeneratedEvent {
        ResponseGeneratedEvent {
            id: "test_resp".to_string(),
            response_type: "text".to_string(),
            token_count: 100,
            duration_us: 500,
            quality_score: 0.95,
            metadata: EventMetadata::default(),
        }
    }

    fn make_security_event() -> SecurityCheckEvent {
        SecurityCheckEvent {
            id: "test_sec".to_string(),
            check_type: "sandbox".to_string(),
            passed: true,
            details: "ok".to_string(),
            metadata: EventMetadata::default(),
        }
    }

    fn make_language_event() -> LanguageStepEvent {
        LanguageStepEvent {
            id: "test_lang".to_string(),
            step: "tokenize".to_string(),
            input_hash: "h2".to_string(),
            duration_us: 40,
            output_summary: "done".to_string(),
            metadata: EventMetadata::default(),
        }
    }

    fn make_narrative_event() -> NarrativeSelfEvent {
        NarrativeSelfEvent {
            id: "test_narr".to_string(),
            component: "memory".to_string(),
            coherence: 0.9,
            update_type: "append".to_string(),
            metadata: EventMetadata::default(),
        }
    }

    fn make_cross_modal_event() -> CrossModalBindingEvent {
        CrossModalBindingEvent {
            id: "test_cm".to_string(),
            modalities: vec!["visual".to_string()],
            strength: 0.7,
            integration_phi: 0.3,
            metadata: EventMetadata::default(),
        }
    }

    fn make_gwt_event() -> GWTIntegrationEvent {
        GWTIntegrationEvent {
            id: "test_gwt".to_string(),
            broadcast: "alert".to_string(),
            modules: vec!["mod1".to_string()],
            activation: 0.5,
            metadata: EventMetadata::default(),
        }
    }

    fn make_broca_event() -> BrocaPipelineEvent {
        BrocaPipelineEvent {
            id: "test_broca".to_string(),
            correlation_id: "bc1".to_string(),
            phase: BrocaPhase::Cognition,
            input_hash: "h3".to_string(),
            duration_us: 80,
            epistemic_status: "known".to_string(),
            semantic_intent: "query".to_string(),
            familiarity: 0.6,
            novelty: 0.4,
            phi: 0.5,
            fidelity_verified: false,
            response_type: "answer".to_string(),
            relationship_stage: "trust".to_string(),
            trust: 0.8,
            metadata: EventMetadata::default(),
        }
    }

    // ── Existing tests (preserved) ──────────────────────────────────────

    #[test]
    fn test_causal_observer_basic() {
        let base = NullObserver::new();
        let mut observer = CausalTraceObserver::new(base, "test_correlation");

        // Record an event
        let event = make_phi_event();
        assert!(observer.record_phi_measurement(event).is_ok());

        // Check state
        assert_eq!(observer.context().correlation_id, "test_correlation");
        assert_eq!(observer.event_count(), 1);
    }

    #[test]
    fn test_causal_observer_scopes() {
        let base = NullObserver::new();
        let mut observer = CausalTraceObserver::new(base, "test");

        // Create root event
        let root_event = make_phi_event();
        observer.record_phi_measurement(root_event).unwrap();

        // Begin scope with this event as parent
        observer.begin_scope("root_event");

        // Child events should have parent
        let child_event = make_router_event();
        observer.record_router_selection(child_event).unwrap();

        // End scope
        observer.end_scope();

        // Verify count
        assert_eq!(observer.event_count(), 2);
    }

    #[test]
    fn test_causal_observer_auto_link() {
        let base = NullObserver::new();
        let mut observer = CausalTraceObserver::new(base, "test").with_auto_link();

        // First event - no parent
        observer.record_phi_measurement(make_phi_event()).unwrap();

        // Second event - should have first as parent (via auto_link context)
        observer
            .record_router_selection(make_router_event())
            .unwrap();

        // Third event - should have second as parent
        observer
            .record_workspace_ignition(make_workspace_event())
            .unwrap();

        assert_eq!(observer.event_count(), 3);
    }

    // ── Nested begin_scope / end_scope ──────────────────────────────────

    #[test]
    fn test_nested_scopes_three_deep() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "nested");

        obs.begin_scope("level-1");
        assert_eq!(obs.context().current_parent(), Some(&"level-1".to_string()));

        obs.begin_scope("level-2");
        assert_eq!(obs.context().current_parent(), Some(&"level-2".to_string()));

        obs.begin_scope("level-3");
        assert_eq!(obs.context().current_parent(), Some(&"level-3".to_string()));

        // Pop back up
        obs.end_scope();
        assert_eq!(obs.context().current_parent(), Some(&"level-2".to_string()));

        obs.end_scope();
        assert_eq!(obs.context().current_parent(), Some(&"level-1".to_string()));

        obs.end_scope();
        assert!(obs.context().current_parent().is_none());
    }

    #[test]
    fn test_scope_events_get_correct_parent() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "scope_parent");

        obs.begin_scope("parent-A");

        // Record event inside scope — the metadata should capture parent-A
        // (We cannot inspect the forwarded metadata directly because NoOpObserver
        //  discards it, but the context state is verifiable.)
        obs.record_phi_measurement(make_phi_event()).unwrap();
        assert_eq!(obs.event_count(), 1);

        // Nest deeper
        obs.begin_scope("parent-B");
        obs.record_router_selection(make_router_event()).unwrap();
        assert_eq!(obs.event_count(), 2);

        obs.end_scope(); // pop parent-B
        obs.end_scope(); // pop parent-A

        assert!(obs.context().current_parent().is_none());
    }

    // ── end_scope on empty stack is harmless ─────────────────────────────

    #[test]
    fn test_end_scope_on_empty_stack_is_harmless() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "empty_end");

        // Should not panic
        obs.end_scope();
        obs.end_scope();
        assert!(obs.context().current_parent().is_none());
    }

    // ── into_inner() recovery ───────────────────────────────────────────

    #[test]
    fn test_into_inner_recovery() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "recover");

        obs.record_phi_measurement(make_phi_event()).unwrap();
        obs.record_error(make_error_event()).unwrap();
        obs.record_router_selection(make_router_event()).unwrap();

        assert_eq!(obs.event_count(), 3);

        // Recover the inner observer and verify its state
        let inner = obs.into_inner();
        let stats = inner.stats();
        assert_eq!(stats.total_events, 3);
    }

    #[test]
    fn test_into_inner_preserves_broca_type_counts() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "broca_inner");

        obs.record_broca_pipeline(make_broca_event()).unwrap();
        obs.record_broca_pipeline(make_broca_event()).unwrap();

        let inner = obs.into_inner();
        let stats = inner.stats();
        assert_eq!(stats.total_events, 2);
        assert_eq!(*stats.events_by_type.get("broca_pipeline").unwrap_or(&0), 2);
    }

    // ── Composite observer pattern (CausalTraceObserver wrapping CausalTraceObserver) ──

    #[test]
    fn test_composite_observer_double_wrap() {
        let base = NullObserver::new();
        let inner_causal = CausalTraceObserver::new(base, "inner_corr");
        let mut outer_causal = CausalTraceObserver::new(inner_causal, "outer_corr");

        outer_causal.begin_scope("outer_scope");
        outer_causal
            .record_phi_measurement(make_phi_event())
            .unwrap();
        outer_causal
            .record_workspace_ignition(make_workspace_event())
            .unwrap();
        outer_causal.end_scope();

        assert_eq!(outer_causal.event_count(), 2);
        assert_eq!(outer_causal.context().correlation_id, "outer_corr");

        // Unwrap one layer
        let inner = outer_causal.into_inner();
        assert_eq!(inner.context().correlation_id, "inner_corr");
        // Inner also recorded the forwarded events
        assert_eq!(inner.event_count(), 2);

        // Unwrap to base
        let base_recovered = inner.into_inner();
        assert_eq!(base_recovered.stats().total_events, 2);
    }

    // ── flush delegates to inner ────────────────────────────────────────

    #[test]
    fn test_flush_delegates() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "flush_test");
        assert!(obs.flush().is_ok());
    }

    // ── stats delegates to inner ────────────────────────────────────────

    #[test]
    fn test_stats_delegates_to_inner() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "stats_test");

        obs.record_phi_measurement(make_phi_event()).unwrap();
        let stats = obs.stats();
        // stats() delegates to inner (NullObserver), which counted 1 event
        assert_eq!(stats.total_events, 1);
    }

    // ── context_mut allows tag injection ─────────────────────────────────

    #[test]
    fn test_context_mut_tag_injection() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "tag_test");

        obs.context_mut().add_tag("request_id", "req-42");
        assert_eq!(
            obs.context().tags.get("request_id").map(|s| s.as_str()),
            Some("req-42")
        );
    }

    // ── All 12 event types through CausalTraceObserver ──────────────────

    #[test]
    fn test_all_event_types_through_causal_observer() {
        let base = NullObserver::new();
        let mut obs = CausalTraceObserver::new(base, "all_types");

        obs.record_phi_measurement(make_phi_event()).unwrap();
        obs.record_router_selection(make_router_event()).unwrap();
        obs.record_workspace_ignition(make_workspace_event())
            .unwrap();
        obs.record_primitive_activation(make_primitive_event())
            .unwrap();
        obs.record_response_generated(make_response_event())
            .unwrap();
        obs.record_security_check(make_security_event()).unwrap();
        obs.record_error(make_error_event()).unwrap();
        obs.record_language_step(make_language_event()).unwrap();
        obs.record_narrative_self(make_narrative_event()).unwrap();
        obs.record_cross_modal_binding(make_cross_modal_event())
            .unwrap();
        obs.record_gwt_integration(make_gwt_event()).unwrap();
        obs.record_broca_pipeline(make_broca_event()).unwrap();

        assert_eq!(obs.event_count(), 12);
        assert_eq!(obs.stats().total_events, 12);
    }

    // ── WithCausalMetadata trait ─────────────────────────────────────────

    #[test]
    fn test_with_causal_metadata_roundtrip() {
        let meta = EventMetadata {
            correlation_id: "corr-1".to_string(),
            parent_id: Some("parent-1".to_string()),
            tags: std::collections::HashMap::new(),
            timestamp: 12345,
        };

        let mut event = make_phi_event();
        event.set_metadata(meta.clone());
        assert_eq!(event.get_metadata().correlation_id, "corr-1");
        assert_eq!(event.get_metadata().parent_id, Some("parent-1".to_string()));
        assert_eq!(event.get_metadata().timestamp, 12345);
    }
}

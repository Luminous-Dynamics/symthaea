/*!
 * Causal Trace Observer (Phase 4)
 *
 * Wrapper observer that automatically injects causal correlation metadata
 * into events, enabling the CausalGraph to build parent-child relationships.
 *
 * ## Usage
 *
 * ```rust,ignore
 * use symthaea::observability::{
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

use super::{
    SymthaeaObserver, ObserverStats,
    CorrelationContext, EventMetadata,
    RouterSelectionEvent, WorkspaceIgnitionEvent, PhiMeasurementEvent,
    PrimitiveActivationEvent, ResponseGeneratedEvent, SecurityCheckEvent,
    ErrorEvent, LanguageStepEvent, NarrativeSelfEvent, CrossModalBindingEvent,
    GWTIntegrationEvent, BrocaPipelineEvent,
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
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for WorkspaceIgnitionEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for PhiMeasurementEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for PrimitiveActivationEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for ResponseGeneratedEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for SecurityCheckEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for ErrorEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for LanguageStepEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for NarrativeSelfEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for CrossModalBindingEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for GWTIntegrationEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

impl WithCausalMetadata for BrocaPipelineEvent {
    fn set_metadata(&mut self, metadata: EventMetadata) { self.metadata = metadata; }
    fn get_metadata(&self) -> &EventMetadata { &self.metadata }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::NullObserver;

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
        observer.record_router_selection(make_router_event()).unwrap();

        // Third event - should have second as parent
        observer.record_workspace_ignition(make_workspace_event()).unwrap();

        assert_eq!(observer.event_count(), 3);
    }
}

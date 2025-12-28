/*!
 * Trace Observer - Exports complete execution traces to JSON
 *
 * Compatible with symthaea-inspect tool for replay and analysis.
 */

use super::{SymthaeaObserver, types::*, ObserverStats};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Observer that exports complete execution traces to JSON
///
/// Creates trace files compatible with the Inspector tool.
/// Trace format matches tools/trace-schema-v1.json
pub struct TraceObserver {
    trace_path: PathBuf,
    trace: Trace,
    stats: ObserverStats,
    auto_flush: bool,
}

impl TraceObserver {
    /// Create new trace observer
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session_id = Uuid::new_v4().to_string();

        Ok(Self {
            trace_path: path.as_ref().to_path_buf(),
            trace: Trace::new(session_id),
            stats: ObserverStats::default(),
            auto_flush: false,
        })
    }

    /// Create with custom session ID
    pub fn with_session_id<P: AsRef<Path>>(path: P, session_id: String) -> Result<Self> {
        Ok(Self {
            trace_path: path.as_ref().to_path_buf(),
            trace: Trace::new(session_id),
            stats: ObserverStats::default(),
            auto_flush: false,
        })
    }

    /// Enable auto-flush after each event (for live monitoring)
    pub fn with_auto_flush(mut self, auto_flush: bool) -> Self {
        self.auto_flush = auto_flush;
        self
    }

    /// Add event to trace
    fn add_event(&mut self, event_type: &str, data: impl serde::Serialize) -> Result<()> {
        let event = Event {
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            data: serde_json::to_value(data)?,
        };

        self.trace.events.push(event);
        self.stats.increment_event(event_type);

        if self.auto_flush {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush trace to disk (without finalizing)
    pub fn flush(&self) -> Result<()> {
        let file = File::create(&self.trace_path)
            .context("Failed to create trace file")?;

        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &self.trace)
            .context("Failed to write trace")?;

        Ok(())
    }

    /// Compute session summary
    fn compute_summary(&self) -> SessionSummary {
        let mut summary = SessionSummary::default();

        summary.total_events = self.trace.events.len();

        let mut phi_measurements = Vec::new();
        let mut router_distribution = std::collections::HashMap::new();

        // Revolutionary Improvement #72 & #73: Self + Cross-Modal Φ tracking
        let mut self_phi_measurements = Vec::new();
        let mut self_coherence_measurements = Vec::new();
        let mut cross_modal_phi_measurements = Vec::new();

        for event in &self.trace.events {
            match event.event_type.as_str() {
                "router_selection" => {
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(router) = data.get("selected_router").and_then(|r| r.as_str()) {
                            *router_distribution.entry(router.to_string()).or_insert(0) += 1;
                        }
                    }
                }
                "workspace_ignition" => {
                    summary.ignition_count += 1;
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(phi) = data.get("phi").and_then(|p| p.as_f64()) {
                            phi_measurements.push(phi);
                        }
                    }
                }
                "phi_measurement" => {
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(phi) = data.get("phi").and_then(|p| p.as_f64()) {
                            phi_measurements.push(phi);
                        }
                    }
                }
                "error" => {
                    summary.errors += 1;
                }
                "security_check" => {
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(decision) = data.get("decision").and_then(|d| d.as_str()) {
                            if decision == "denied" {
                                summary.security_denials += 1;
                            }
                        }
                    }
                }
                // Revolutionary Improvement #73: Narrative Self tracking
                "narrative_self" => {
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(self_phi) = data.get("self_phi").and_then(|p| p.as_f64()) {
                            self_phi_measurements.push(self_phi);
                        }
                        if let Some(coherence) = data.get("coherence").and_then(|c| c.as_f64()) {
                            self_coherence_measurements.push(coherence);
                        }
                        if data.get("veto").is_some() {
                            summary.veto_count += 1;
                        }
                    }
                }
                // Revolutionary Improvement #72: Cross-Modal Φ tracking
                "cross_modal_binding" => {
                    summary.binding_events += 1;
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(cm_phi) = data.get("cross_modal_phi").and_then(|p| p.as_f64()) {
                            cross_modal_phi_measurements.push(cm_phi);
                        }
                    }
                }
                // Revolutionary Improvement #73: GWT Integration tracking
                "gwt_integration" => {
                    if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.data.clone()) {
                        if let Some(vetoes) = data.get("vetoes_issued").and_then(|v| v.as_u64()) {
                            summary.veto_count += vetoes as usize;
                        }
                    }
                }
                _ => {}
            }
        }

        // Compute Φ statistics
        if !phi_measurements.is_empty() {
            summary.average_phi = phi_measurements.iter().sum::<f64>() / phi_measurements.len() as f64;
            summary.max_phi = phi_measurements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            summary.min_phi = phi_measurements.iter().cloned().fold(f64::INFINITY, f64::min);
        }

        // Compute Self-Φ statistics (Revolutionary Improvement #73)
        if !self_phi_measurements.is_empty() {
            summary.self_phi_average = self_phi_measurements.iter().sum::<f64>()
                / self_phi_measurements.len() as f64;
        }
        if !self_coherence_measurements.is_empty() {
            summary.self_coherence_average = self_coherence_measurements.iter().sum::<f64>()
                / self_coherence_measurements.len() as f64;
        }

        // Compute Cross-Modal Φ statistics (Revolutionary Improvement #72)
        if !cross_modal_phi_measurements.is_empty() {
            summary.cross_modal_phi_average = cross_modal_phi_measurements.iter().sum::<f64>()
                / cross_modal_phi_measurements.len() as f64;
        }

        summary.router_distribution = router_distribution;

        // Compute duration
        if let Some(end) = self.trace.timestamp_end {
            summary.duration_ms = (end - self.trace.timestamp_start).num_milliseconds() as u64;
        }

        summary
    }
}

impl SymthaeaObserver for TraceObserver {
    fn record_router_selection(&mut self, event: RouterSelectionEvent) -> Result<()> {
        self.add_event("router_selection", event)
    }

    fn record_workspace_ignition(&mut self, event: WorkspaceIgnitionEvent) -> Result<()> {
        self.add_event("workspace_ignition", event)
    }

    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<()> {
        self.add_event("phi_measurement", event)
    }

    fn record_primitive_activation(&mut self, event: PrimitiveActivationEvent) -> Result<()> {
        self.add_event("primitive_activation", event)
    }

    fn record_response_generated(&mut self, event: ResponseGeneratedEvent) -> Result<()> {
        self.add_event("response_generated", event)
    }

    fn record_security_check(&mut self, event: SecurityCheckEvent) -> Result<()> {
        self.add_event("security_check", event)
    }

    fn record_error(&mut self, event: ErrorEvent) -> Result<()> {
        self.add_event("error", event)
    }

    fn record_language_step(&mut self, event: LanguageStepEvent) -> Result<()> {
        self.add_event("language_step", event)
    }

    // Revolutionary Improvement #72 & #73: Self + Cross-Modal Φ tracing

    fn record_narrative_self(&mut self, event: NarrativeSelfEvent) -> Result<()> {
        self.add_event("narrative_self", event)
    }

    fn record_cross_modal_binding(&mut self, event: CrossModalBindingEvent) -> Result<()> {
        self.add_event("cross_modal_binding", event)
    }

    fn record_gwt_integration(&mut self, event: GWTIntegrationEvent) -> Result<()> {
        self.add_event("gwt_integration", event)
    }

    fn finalize(&mut self) -> Result<()> {
        // Set end timestamp
        self.trace.timestamp_end = Some(chrono::Utc::now());

        // Compute summary
        self.trace.summary = Some(self.compute_summary());

        // Write final trace
        self.flush()?;

        Ok(())
    }

    fn get_stats(&self) -> Result<ObserverStats> {
        Ok(self.stats.clone())
    }
}

impl Drop for TraceObserver {
    fn drop(&mut self) {
        // Try to finalize on drop (best effort)
        let _ = self.finalize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_trace_observer_creation() -> Result<()> {
        let temp = NamedTempFile::new()?;
        let observer = TraceObserver::new(temp.path())?;

        assert_eq!(observer.trace.version, "1.0");
        assert_eq!(observer.trace.events.len(), 0);

        Ok(())
    }

    #[test]
    fn test_trace_observer_events() -> Result<()> {
        let temp = NamedTempFile::new()?;
        let mut observer = TraceObserver::new(temp.path())?;

        // Record some events
        observer.record_router_selection(RouterSelectionEvent {
            selected_router: "TestRouter".to_string(),
            confidence: 0.9,
            ..Default::default()
        })?;

        observer.record_phi_measurement(PhiMeasurementEvent {
            phi: 0.75,
            ..Default::default()
        })?;

        assert_eq!(observer.trace.events.len(), 2);
        assert_eq!(observer.stats.router_selections, 1);
        assert_eq!(observer.stats.phi_measurements, 1);

        Ok(())
    }

    #[test]
    fn test_trace_observer_finalize() -> Result<()> {
        let temp = NamedTempFile::new()?;
        let mut observer = TraceObserver::new(temp.path())?;

        observer.record_router_selection(RouterSelectionEvent::default())?;
        observer.finalize()?;

        // Check file was written
        assert!(temp.path().exists());

        // Check can be read back
        let content = std::fs::read_to_string(temp.path())?;
        let trace: Trace = serde_json::from_str(&content)?;

        assert_eq!(trace.version, "1.0");
        assert_eq!(trace.events.len(), 1);
        assert!(trace.timestamp_end.is_some());
        assert!(trace.summary.is_some());

        Ok(())
    }
}

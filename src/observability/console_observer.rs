/*!
 * Console Observer - Debug logging to stdout/stderr
 *
 * Useful for development and debugging.
 */

use super::{SymthaeaObserver, types::*, ObserverStats};
use anyhow::Result;

/// Observer that prints events to console
///
/// Use for development and debugging.
/// Can filter by event type and verbosity level.
#[derive(Debug)]
pub struct ConsoleObserver {
    verbosity: Verbosity,
    show_router_selection: bool,
    show_workspace_ignition: bool,
    show_phi_measurement: bool,
    show_primitive_activation: bool,
    show_response: bool,
    show_security: bool,
    show_errors: bool,
    show_language_steps: bool,
    stats: ObserverStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Minimal,  // Only errors and critical events
    Normal,   // Important events
    Detailed, // All events with details
    Full,     // Everything including traces
}

impl Default for ConsoleObserver {
    fn default() -> Self {
        Self::new(Verbosity::Normal)
    }
}

impl ConsoleObserver {
    pub fn new(verbosity: Verbosity) -> Self {
        Self {
            verbosity,
            show_router_selection: true,
            show_workspace_ignition: true,
            show_phi_measurement: true,
            show_primitive_activation: false, // Too verbose by default
            show_response: true,
            show_security: true,
            show_errors: true,
            show_language_steps: false, // Too verbose by default
            stats: ObserverStats::default(),
        }
    }

    pub fn with_router_selection(mut self, show: bool) -> Self {
        self.show_router_selection = show;
        self
    }

    pub fn with_workspace_ignition(mut self, show: bool) -> Self {
        self.show_workspace_ignition = show;
        self
    }

    pub fn with_phi_measurement(mut self, show: bool) -> Self {
        self.show_phi_measurement = show;
        self
    }

    pub fn with_primitive_activation(mut self, show: bool) -> Self {
        self.show_primitive_activation = show;
        self
    }

    pub fn with_language_steps(mut self, show: bool) -> Self {
        self.show_language_steps = show;
        self
    }

    fn print_minimal(&self, msg: &str) {
        if matches!(self.verbosity, Verbosity::Minimal | Verbosity::Normal | Verbosity::Detailed | Verbosity::Full) {
            println!("[{}] {}", chrono::Utc::now().format("%H:%M:%S"), msg);
        }
    }

    fn print_normal(&self, msg: &str) {
        if matches!(self.verbosity, Verbosity::Normal | Verbosity::Detailed | Verbosity::Full) {
            println!("[{}] {}", chrono::Utc::now().format("%H:%M:%S"), msg);
        }
    }

    fn print_detailed(&self, msg: &str) {
        if matches!(self.verbosity, Verbosity::Detailed | Verbosity::Full) {
            println!("[{}] {}", chrono::Utc::now().format("%H:%M:%S"), msg);
        }
    }
}

impl SymthaeaObserver for ConsoleObserver {
    fn record_router_selection(&mut self, event: RouterSelectionEvent) -> Result<()> {
        self.stats.increment_event("router_selection");

        if !self.show_router_selection {
            return Ok(());
        }

        self.print_normal(&format!(
            "🔀 Router: {} (confidence: {:.1}%)",
            event.selected_router,
            event.confidence * 100.0
        ));

        if matches!(self.verbosity, Verbosity::Full) {
            self.print_detailed(&format!("   Input: {}", event.input));
            for alt in &event.alternatives {
                self.print_detailed(&format!("   Alternative: {} ({:.1}%)", alt.router, alt.score * 100.0));
            }
        }

        Ok(())
    }

    fn record_workspace_ignition(&mut self, event: WorkspaceIgnitionEvent) -> Result<()> {
        self.stats.increment_event("workspace_ignition");

        if !self.show_workspace_ignition {
            return Ok(());
        }

        self.print_normal(&format!(
            "🔥 Workspace Ignition: Φ={:.3}, coalition_size={}, free_energy={:.2}",
            event.phi, event.coalition_size, event.free_energy
        ));

        if matches!(self.verbosity, Verbosity::Detailed | Verbosity::Full) {
            self.print_detailed(&format!("   Active primitives: {:?}", event.active_primitives));
        }

        Ok(())
    }

    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<()> {
        self.stats.increment_event("phi_measurement");

        if !self.show_phi_measurement {
            return Ok(())
        }

        self.print_detailed(&format!("📊 Φ: {:.3}", event.phi));

        if matches!(self.verbosity, Verbosity::Full) {
            self.print_detailed(&format!("   Integration: {:.3}", event.components.integration));
            self.print_detailed(&format!("   Binding: {:.3}", event.components.binding));
            self.print_detailed(&format!("   Workspace: {:.3}", event.components.workspace));
        }

        Ok(())
    }

    fn record_primitive_activation(&mut self, event: PrimitiveActivationEvent) -> Result<()> {
        self.stats.increment_event("primitive_activation");

        if !self.show_primitive_activation {
            return Ok(());
        }

        self.print_detailed(&format!(
            "⚡ Primitive: {} (tier={}, strength={:.2})",
            event.primitive_name, event.tier, event.activation_strength
        ));

        Ok(())
    }

    fn record_response_generated(&mut self, event: ResponseGeneratedEvent) -> Result<()> {
        self.stats.increment_event("response_generated");

        if !self.show_response {
            return Ok(());
        }

        let safety_icon = if event.safety_verified { "✅" } else { "⚠️" };
        let confirm_icon = if event.requires_confirmation { "🔒" } else { "" };

        self.print_normal(&format!(
            "{}{} Response: {} (confidence: {:.1}%)",
            safety_icon,
            confirm_icon,
            event.content.chars().take(60).collect::<String>(),
            event.confidence * 100.0
        ));

        Ok(())
    }

    fn record_security_check(&mut self, event: SecurityCheckEvent) -> Result<()> {
        self.stats.increment_event("security_check");

        if !self.show_security {
            return Ok(());
        }

        let icon = match event.decision {
            SecurityDecision::Allowed => "✅",
            SecurityDecision::Denied => "❌",
            SecurityDecision::RequiresConfirmation => "⚠️",
        };

        self.print_normal(&format!(
            "{} Security: {} ({:?})",
            icon, event.operation, event.decision
        ));

        if let Some(reason) = &event.reason {
            self.print_detailed(&format!("   Reason: {}", reason));
        }

        Ok(())
    }

    fn record_error(&mut self, event: ErrorEvent) -> Result<()> {
        self.stats.increment_event("error");

        if !self.show_errors {
            return Ok(());
        }

        let icon = if event.recoverable { "⚠️" } else { "❌" };

        self.print_minimal(&format!(
            "{} Error: {} - {}",
            icon, event.error_type, event.message
        ));

        Ok(())
    }

    fn record_language_step(&mut self, event: LanguageStepEvent) -> Result<()> {
        self.stats.increment_event("language_step");

        if !self.show_language_steps {
            return Ok(());
        }

        self.print_detailed(&format!(
            "💬 Language: {:?} (confidence: {:.1}%, {}ms)",
            event.step_type, event.confidence * 100.0, event.duration_ms
        ));

        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        self.print_minimal("📊 Session Summary:");
        self.print_minimal(&format!("   Total events: {}", self.stats.total_events));
        self.print_minimal(&format!("   Router selections: {}", self.stats.router_selections));
        self.print_minimal(&format!("   Workspace ignitions: {}", self.stats.workspace_ignitions));
        self.print_minimal(&format!("   Φ measurements: {}", self.stats.phi_measurements));
        self.print_minimal(&format!("   Errors: {}", self.stats.errors));

        Ok(())
    }

    fn get_stats(&self) -> Result<ObserverStats> {
        Ok(self.stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_console_observer() {
        let mut observer = ConsoleObserver::new(Verbosity::Normal);

        // Should not panic
        observer.record_router_selection(RouterSelectionEvent {
            selected_router: "TestRouter".to_string(),
            confidence: 0.9,
            ..Default::default()
        }).unwrap();

        observer.finalize().unwrap();

        let stats = observer.get_stats().unwrap();
        assert_eq!(stats.router_selections, 1);
    }
}

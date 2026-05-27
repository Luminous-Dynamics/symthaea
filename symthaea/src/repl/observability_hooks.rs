// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observability Hooks - Telemetry and Tracing for REPL
//!
//! Provides hooks for observing REPL events and consciousness state.
//! Integrates with the main observability module for causal tracing.

use std::collections::HashMap;
use std::time::{Instant, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, info, warn};

use crate::cognitive_loop::ConsciousnessSnapshot;

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVABILITY HOOK TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for observing REPL events
///
/// Implementors must be Send to allow async processing in different threads.
pub trait ObservabilityHook: Send {
    /// Called when input is received
    fn on_input(&mut self, input: &str);

    /// Called when output is generated
    fn on_output(&mut self, output: &str, consciousness: &ConsciousnessSnapshot);

    /// Called when an action is executed
    fn on_action(&mut self, command: &str, executed: bool);

    /// Called when state is reset
    fn on_reset(&mut self);

    /// Called on each cognitive cycle
    fn on_cycle(&mut self, cycle_num: u64, consciousness: &ConsciousnessSnapshot);

    /// Called when flow state changes
    fn on_flow_change(&mut self, in_flow: bool, intensity: f32);

    /// Get accumulated statistics
    fn stats(&self) -> ObservabilityStats;
}

/// Observability statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservabilityStats {
    /// Total events observed
    pub total_events: u64,
    /// Input events
    pub input_events: u64,
    /// Output events
    pub output_events: u64,
    /// Action events
    pub action_events: u64,
    /// Cycle events
    pub cycle_events: u64,
    /// Flow state changes
    pub flow_changes: u64,
    /// Average phi during session
    pub avg_phi: f32,
    /// Peak phi during session
    pub peak_phi: f32,
    /// Time in flow (seconds)
    pub total_flow_time_secs: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS TRACER
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness tracer - logs consciousness state over time
pub struct ConsciousnessTracer {
    /// Session ID for correlation
    session_id: String,

    /// Trace buffer
    traces: Vec<ConsciousnessTrace>,

    /// Maximum traces to keep
    max_traces: usize,

    /// Statistics
    stats: ObservabilityStats,

    /// Last flow state
    last_in_flow: bool,

    /// Flow start time
    flow_start: Option<Instant>,

    /// Phi accumulator for average
    phi_sum: f32,
    phi_count: u64,
}

/// Single consciousness trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessTrace {
    /// Timestamp (relative to session start)
    pub timestamp_ms: u64,
    /// Event type
    pub event_type: TraceEventType,
    /// Phi at this moment
    pub phi: f32,
    /// Coherence at this moment
    pub coherence: f32,
    /// Pattern detected
    pub pattern: String,
    /// Cognitive depth
    pub depth: String,
    /// In flow state
    pub in_flow: bool,
    /// Valence
    pub valence: f32,
    /// Arousal
    pub arousal: f32,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Trace event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceEventType {
    Input,
    Output,
    Action,
    Cycle,
    FlowEnter,
    FlowExit,
    Reset,
}

impl ConsciousnessTracer {
    /// Create a new tracer
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            traces: Vec::with_capacity(1000),
            max_traces: 10000,
            stats: ObservabilityStats::default(),
            last_in_flow: false,
            flow_start: None,
            phi_sum: 0.0,
            phi_count: 0,
        }
    }

    /// Record a trace
    fn record(
        &mut self,
        event_type: TraceEventType,
        consciousness: &ConsciousnessSnapshot,
        context: HashMap<String, String>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let trace = ConsciousnessTrace {
            timestamp_ms: timestamp,
            event_type,
            phi: consciousness.unified_psi,
            coherence: consciousness.temporal_coherence,
            pattern: format!("{:?}", consciousness.pattern),
            depth: format!("{:?}", consciousness.cognitive_depth),
            in_flow: consciousness.in_flow,
            valence: consciousness.unified_valence,
            arousal: consciousness.unified_arousal,
            context,
        };

        // Update statistics
        self.stats.total_events += 1;
        self.phi_sum += consciousness.unified_psi;
        self.phi_count += 1;
        self.stats.avg_phi = self.phi_sum / self.phi_count as f32;

        if consciousness.unified_psi > self.stats.peak_phi {
            self.stats.peak_phi = consciousness.unified_psi;
        }

        // Check flow state change
        if consciousness.in_flow && !self.last_in_flow {
            self.flow_start = Some(Instant::now());
            self.stats.flow_changes += 1;
        } else if !consciousness.in_flow && self.last_in_flow {
            if let Some(start) = self.flow_start.take() {
                self.stats.total_flow_time_secs += start.elapsed().as_secs_f32();
            }
            self.stats.flow_changes += 1;
        }
        self.last_in_flow = consciousness.in_flow;

        // Add trace (with rotation)
        self.traces.push(trace);
        if self.traces.len() > self.max_traces {
            self.traces.remove(0);
        }

        // Log trace
        debug!(
            session = %self.session_id,
            event = ?event_type,
            phi = %consciousness.unified_psi,
            coherence = %consciousness.temporal_coherence,
            in_flow = %consciousness.in_flow,
            "Consciousness trace"
        );
    }

    /// Get all traces
    pub fn traces(&self) -> &[ConsciousnessTrace] {
        &self.traces
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Export traces to JSON
    pub fn export_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.traces)?)
    }
}

impl ObservabilityHook for ConsciousnessTracer {
    fn on_input(&mut self, input: &str) {
        self.stats.input_events += 1;

        // Create minimal consciousness snapshot for input events
        // In real use, we'd have access to the actual consciousness state
        let mut context = HashMap::new();
        context.insert("input_length".to_string(), input.len().to_string());
        context.insert(
            "input_preview".to_string(),
            input.chars().take(50).collect::<String>(),
        );

        // Can't record without consciousness state - log instead
        info!(
            session = %self.session_id,
            input_len = %input.len(),
            "Input received"
        );
    }

    fn on_output(&mut self, output: &str, consciousness: &ConsciousnessSnapshot) {
        self.stats.output_events += 1;

        let mut context = HashMap::new();
        context.insert("output_length".to_string(), output.len().to_string());

        self.record(TraceEventType::Output, consciousness, context);
    }

    fn on_action(&mut self, command: &str, executed: bool) {
        self.stats.action_events += 1;

        info!(
            session = %self.session_id,
            command = %command,
            executed = %executed,
            "Action event"
        );
    }

    fn on_reset(&mut self) {
        info!(
            session = %self.session_id,
            total_traces = %self.traces.len(),
            "Session reset"
        );
    }

    fn on_cycle(&mut self, cycle_num: u64, consciousness: &ConsciousnessSnapshot) {
        self.stats.cycle_events += 1;

        let mut context = HashMap::new();
        context.insert("cycle_num".to_string(), cycle_num.to_string());

        self.record(TraceEventType::Cycle, consciousness, context);
    }

    fn on_flow_change(&mut self, in_flow: bool, intensity: f32) {
        info!(
            session = %self.session_id,
            in_flow = %in_flow,
            intensity = %intensity,
            "Flow state change"
        );
    }

    fn stats(&self) -> ObservabilityStats {
        self.stats.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOGGING HOOK
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple logging hook using tracing
pub struct LoggingHook {
    /// Session ID
    session_id: String,
    /// Statistics
    stats: ObservabilityStats,
    /// Log level
    level: Level,
}

impl LoggingHook {
    /// Create a new logging hook
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            stats: ObservabilityStats::default(),
            level: Level::INFO,
        }
    }

    /// Set log level
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }
}

impl ObservabilityHook for LoggingHook {
    fn on_input(&mut self, input: &str) {
        self.stats.input_events += 1;
        self.stats.total_events += 1;

        info!(
            session = %self.session_id,
            input_len = %input.len(),
            "REPL input"
        );
    }

    fn on_output(&mut self, output: &str, consciousness: &ConsciousnessSnapshot) {
        self.stats.output_events += 1;
        self.stats.total_events += 1;

        info!(
            session = %self.session_id,
            phi = %consciousness.unified_psi,
            coherence = %consciousness.temporal_coherence,
            in_flow = %consciousness.in_flow,
            output_len = %output.len(),
            "REPL output"
        );
    }

    fn on_action(&mut self, command: &str, executed: bool) {
        self.stats.action_events += 1;
        self.stats.total_events += 1;

        if executed {
            info!(
                session = %self.session_id,
                command = %command,
                "Action executed"
            );
        } else {
            warn!(
                session = %self.session_id,
                command = %command,
                "Action blocked"
            );
        }
    }

    fn on_reset(&mut self) {
        self.stats.total_events += 1;

        info!(
            session = %self.session_id,
            "Session reset"
        );
    }

    fn on_cycle(&mut self, cycle_num: u64, consciousness: &ConsciousnessSnapshot) {
        self.stats.cycle_events += 1;
        self.stats.total_events += 1;

        debug!(
            session = %self.session_id,
            cycle = %cycle_num,
            phi = %consciousness.unified_psi,
            "Cognitive cycle"
        );
    }

    fn on_flow_change(&mut self, in_flow: bool, intensity: f32) {
        self.stats.flow_changes += 1;
        self.stats.total_events += 1;

        if in_flow {
            info!(
                session = %self.session_id,
                intensity = %intensity,
                "Entered flow state"
            );
        } else {
            info!(
                session = %self.session_id,
                "Exited flow state"
            );
        }
    }

    fn stats(&self) -> ObservabilityStats {
        self.stats.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS AGGREGATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Aggregates metrics over time for dashboard display
pub struct MetricsAggregator {
    /// Session ID
    #[allow(dead_code)] // RESERVED(future): per-session telemetry identifier
    session_id: String,
    /// Statistics
    stats: ObservabilityStats,
    /// Phi history (for sparkline)
    phi_history: Vec<f32>,
    /// Coherence history
    coherence_history: Vec<f32>,
    /// Max history length
    max_history: usize,
    /// Histogram buckets for phi
    phi_histogram: [u32; 10],
    /// Histogram buckets for coherence
    coherence_histogram: [u32; 10],
}

impl MetricsAggregator {
    /// Create a new aggregator
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            stats: ObservabilityStats::default(),
            phi_history: Vec::with_capacity(100),
            coherence_history: Vec::with_capacity(100),
            max_history: 100,
            phi_histogram: [0; 10],
            coherence_histogram: [0; 10],
        }
    }

    /// Get phi history for sparkline
    pub fn phi_history(&self) -> &[f32] {
        &self.phi_history
    }

    /// Get coherence history for sparkline
    pub fn coherence_history(&self) -> &[f32] {
        &self.coherence_history
    }

    /// Get phi histogram
    pub fn phi_histogram(&self) -> &[u32; 10] {
        &self.phi_histogram
    }

    /// Get coherence histogram
    pub fn coherence_histogram(&self) -> &[u32; 10] {
        &self.coherence_histogram
    }

    /// Update histogram
    fn update_histogram(histogram: &mut [u32; 10], value: f32) {
        let bucket = (value.clamp(0.0, 0.999) * 10.0) as usize;
        histogram[bucket] += 1;
    }
}

impl ObservabilityHook for MetricsAggregator {
    fn on_input(&mut self, _input: &str) {
        self.stats.input_events += 1;
        self.stats.total_events += 1;
    }

    fn on_output(&mut self, _output: &str, consciousness: &ConsciousnessSnapshot) {
        self.stats.output_events += 1;
        self.stats.total_events += 1;

        // Update histories
        self.phi_history.push(consciousness.unified_psi);
        self.coherence_history
            .push(consciousness.temporal_coherence);

        if self.phi_history.len() > self.max_history {
            self.phi_history.remove(0);
        }
        if self.coherence_history.len() > self.max_history {
            self.coherence_history.remove(0);
        }

        // Update histograms
        Self::update_histogram(&mut self.phi_histogram, consciousness.unified_psi);
        Self::update_histogram(
            &mut self.coherence_histogram,
            consciousness.temporal_coherence,
        );

        // Update stats
        if consciousness.unified_psi > self.stats.peak_phi {
            self.stats.peak_phi = consciousness.unified_psi;
        }
    }

    fn on_action(&mut self, _command: &str, _executed: bool) {
        self.stats.action_events += 1;
        self.stats.total_events += 1;
    }

    fn on_reset(&mut self) {
        // Keep histograms but clear histories
        self.phi_history.clear();
        self.coherence_history.clear();
    }

    fn on_cycle(&mut self, _cycle_num: u64, consciousness: &ConsciousnessSnapshot) {
        self.stats.cycle_events += 1;
        self.stats.total_events += 1;

        // Update avg phi (rolling)
        let alpha = 0.05;
        self.stats.avg_phi = self.stats.avg_phi * (1.0 - alpha) + consciousness.unified_psi * alpha;
    }

    fn on_flow_change(&mut self, _in_flow: bool, _intensity: f32) {
        self.stats.flow_changes += 1;
        self.stats.total_events += 1;
    }

    fn stats(&self) -> ObservabilityStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require CognitiveLoopService which is complex to mock
    // These are integration tests that verify the hook interface

    #[test]
    fn test_tracer_creation() {
        let tracer = ConsciousnessTracer::new("test-session");
        assert_eq!(tracer.session_id(), "test-session");
        assert!(tracer.traces().is_empty());
    }

    #[test]
    fn test_logging_hook_creation() {
        let hook = LoggingHook::new("test-session");
        let stats = hook.stats();
        assert_eq!(stats.total_events, 0);
    }

    #[test]
    fn test_metrics_aggregator_creation() {
        let agg = MetricsAggregator::new("test-session");
        assert!(agg.phi_history().is_empty());
        assert!(agg.coherence_history().is_empty());
    }

    #[test]
    fn test_observability_stats_default() {
        let stats = ObservabilityStats::default();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.avg_phi, 0.0);
    }
}

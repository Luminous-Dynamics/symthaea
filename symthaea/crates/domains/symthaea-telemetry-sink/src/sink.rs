// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`TelemetrySink`] — the live-wiring adapter from Symthaea subsystems to the WaterfallBuffer.
//!
//! ## Usage Pattern
//!
//! ```rust,no_run
//! use symthaea_telemetry_sink::TelemetrySink;
//! use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, types::Observation};
//! use symthaea_phi_oracle::{IntegrationOracle, OracleConfig};
//! use symthaea_workspace::GlobalWorkspace;
//!
//! let mut sink = TelemetrySink::new();
//! let mut fep = ActiveInferenceAgent::new(ActiveInferenceAgentConfig::default());
//! let mut oracle = IntegrationOracle::new_simple(4, OracleConfig::default()).unwrap();
//! let mut workspace = GlobalWorkspace::new();
//!
//! // Each tick: observe → update → sample → push to waterfall
//! let obs = Observation::from_consciousness_state(3.2, 0.8, 0.9, 0.7);
//! fep.perceive(&obs);
//! oracle.observe(&obs.values).ok();
//!
//! if let Some(snap) = sink.sample(&fep, &oracle, &workspace) {
//!     sink.push_to_waterfall(snap);
//! }
//! ```

use symthaea_fep::ActiveInferenceAgent;
use symthaea_phi_oracle::IntegrationOracle;
use symthaea_projection::{
    frame::ProjectionMode,
    waterfall::{WaterfallBuffer, WaterfallConfig},
};
use symthaea_workspace::GlobalWorkspace;

use crate::{frame_builder::FrameBuilder, metric_snapshot::MetricSnapshot};

/// The live telemetry sink — bridges Symthaea subsystems to the WaterfallBuffer.
pub struct TelemetrySink {
    /// The 64-frame waterfall history buffer (Time-Waterfall mode).
    pub waterfall: WaterfallBuffer,
    /// Frame builder for converting snapshots to projection frames.
    builder: FrameBuilder,
    /// Monotonic frame counter.
    frame_id: u64,
    /// History of the last N phi values for trend analysis.
    phi_history: Vec<f64>,
    /// History of the last N prediction_error values for anomaly computation.
    pred_err_history: Vec<f64>,
}

impl TelemetrySink {
    /// Create a new sink with default waterfall configuration.
    pub fn new() -> Self {
        Self {
            waterfall: WaterfallBuffer::new(WaterfallConfig::default()),
            builder: FrameBuilder::new(ProjectionMode::TimeWaterfall),
            frame_id: 0,
            phi_history: Vec::with_capacity(32),
            pred_err_history: Vec::with_capacity(32),
        }
    }

    /// Create with a custom waterfall config.
    pub fn with_config(config: WaterfallConfig) -> Self {
        Self {
            waterfall: WaterfallBuffer::new(config),
            builder: FrameBuilder::new(ProjectionMode::TimeWaterfall),
            frame_id: 0,
            phi_history: Vec::with_capacity(32),
            pred_err_history: Vec::with_capacity(32),
        }
    }

    /// Sample current state from all three subsystems and produce a [`MetricSnapshot`].
    ///
    /// Returns `None` if the subsystems are not yet ready (insufficient history).
    pub fn sample(
        &mut self,
        fep: &ActiveInferenceAgent,
        oracle: &IntegrationOracle,
        workspace: &GlobalWorkspace,
    ) -> Option<MetricSnapshot> {
        self.frame_id += 1;

        // ── Metric 1: Phi (IIT integration index) ──────────────────────────
        let phi = oracle.measure().map(|r| r.integration_index).unwrap_or(0.0);

        // ── Metric 2: FEP prediction error ─────────────────────────────────
        // Accessed through the public last_fe_components field
        let fep_prediction_error = fep
            .last_fe_components
            .as_ref()
            .map(|c| c.prediction_error)
            .unwrap_or(0.0);

        // ── Metric 3: Workspace activation ─────────────────────────────────
        let workspace_activation = workspace
            .current_focus
            .as_ref()
            .map(|f| f.magnitude)
            .unwrap_or(0.0);

        // ── Metric 4: HOT confidence (belief self-model confidence) ─────────
        // From the public belief field on the agent
        let hot_confidence = fep.belief.confidence();

        // ── Metric 6: Memory pressure (total belief uncertainty) ───────────
        let memory_pressure = fep.belief.total_uncertainty();

        // ── Metric 7: MIP instability (1 − normalized phi oracle index) ────
        let mip_instability = oracle
            .measure()
            .map(|r| 1.0 - r.normalized_index.clamp(0.0, 1.0))
            .unwrap_or(0.0);

        // Update trend histories
        self.phi_history.push(phi);
        self.pred_err_history.push(fep_prediction_error);
        if self.phi_history.len() > 32 {
            self.phi_history.remove(0);
        }
        if self.pred_err_history.len() > 32 {
            self.pred_err_history.remove(0);
        }

        Some(MetricSnapshot::new(
            self.frame_id,
            phi,
            fep_prediction_error,
            workspace_activation,
            hot_confidence,
            memory_pressure,
            mip_instability,
        ))
    }

    /// Sample from raw values (for testing or when subsystems are unavailable).
    pub fn sample_raw(
        &mut self,
        phi: f64,
        fep_prediction_error: f64,
        workspace_activation: f64,
        hot_confidence: f64,
        memory_pressure: f64,
        mip_instability: f64,
    ) -> MetricSnapshot {
        self.frame_id += 1;
        MetricSnapshot::new(
            self.frame_id,
            phi,
            fep_prediction_error,
            workspace_activation,
            hot_confidence,
            memory_pressure,
            mip_instability,
        )
    }

    /// Push a [`MetricSnapshot`] into the waterfall buffer.
    pub fn push_to_waterfall(&mut self, snapshot: MetricSnapshot) {
        let anomalies = self.waterfall.detect_shape_anomalies();
        for anomaly in &anomalies {
            tracing::warn!("Waterfall shape anomaly: {:?}", anomaly);
        }
        let frame = self.builder.build(&snapshot);
        self.waterfall.push(frame);
    }

    /// Run the demo scenario from the design doc (for testing and demos).
    ///
    /// Produces 64 frames in sequence:
    /// 1. Stable baseline (20 frames)
    /// 2. Pump contradiction event (10 frames)
    /// 3. False-green detection (5 frames)
    /// 4. Repair action (5 frames)
    /// 5. Recovery (14 frames)
    /// 6. Chronicle event (10 frames)
    pub fn run_demo_scenario(&mut self) {
        // Phase 1: Stable baseline
        for i in 0..20u64 {
            let snap = self.sample_raw(
                3.2 + (i as f64 * 0.01).sin() * 0.1,   // phi — stable
                0.15 + (i as f64 * 0.05).cos() * 0.05, // pred_error — low
                0.6 + (i as f64 * 0.03).sin() * 0.05,  // workspace — moderate
                0.85,                                  // hot_confidence — high
                0.8,                                   // memory_pressure — normal
                0.1,                                   // mip_instability — low
            );
            self.push_to_waterfall(snap);
        }

        // Phase 2: Pump contradiction event — FEP spike, phi drops
        for i in 0..10u64 {
            let t = i as f64 / 10.0;
            let snap = self
                .sample_raw(
                    3.2 - t * 2.0,  // phi collapsing
                    0.15 + t * 3.0, // prediction error spiking
                    0.6 + t * 0.3,  // workspace activating in response
                    0.85 - t * 0.2, // hot_confidence falling
                    0.8 + t * 1.2,  // memory_pressure rising
                    0.1 + t * 0.6,  // mip_instability rising
                )
                .with_event("pump_sensor_contradiction");
            self.push_to_waterfall(snap);
        }

        // Phase 3: False-green detection — machine says OK, ecology disagrees
        for _ in 0..5u64 {
            // Suspiciously uniform machine-reported metrics
            let snap = self.sample_raw(1.0, 0.001, 0.5, 0.999, 0.001, 0.001);
            self.push_to_waterfall(snap);
        }

        // Phase 4: Repair action — perturbation, then phi starts recovering
        for i in 0..5u64 {
            let t = i as f64 / 5.0;
            let snap = self
                .sample_raw(
                    1.2 + t * 0.8, // phi starting to recover
                    2.5 - t * 2.0, // prediction error falling
                    0.8,
                    0.65 + t * 0.1,
                    1.5 - t * 0.5,
                    0.6 - t * 0.3,
                )
                .with_event("repair_action_initiated");
            self.push_to_waterfall(snap);
        }

        // Phase 5: Recovery — phi stabilizing back toward baseline
        for i in 0..14u64 {
            let t = i as f64 / 14.0;
            let snap = self.sample_raw(
                2.0 + t * 1.2 + (i as f64 * 0.2).sin() * 0.1,
                0.5 - t * 0.35,
                0.7,
                0.75 + t * 0.1,
                1.0 - t * 0.2,
                0.3 - t * 0.2,
            );
            self.push_to_waterfall(snap);
        }

        // Phase 6: Chronicle event — durable record formed
        for i in 0..10u64 {
            let snap = self.sample_raw(
                3.1 + (i as f64 * 0.1).sin() * 0.05,
                0.15,
                0.6,
                0.88,
                0.75,
                0.08,
            );
            // High anomaly_score → will generate Chronicle ref
            let mut snap = snap;
            if i == 0 {
                snap = snap.with_event("chronicle_durability_threshold_reached");
            }
            self.push_to_waterfall(snap);
        }
    }

    /// Current number of frames in the waterfall.
    pub fn frame_count(&self) -> usize {
        self.waterfall.len()
    }
}

impl Default for TelemetrySink {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_scenario_fills_64_frames() {
        let mut sink = TelemetrySink::new();
        sink.run_demo_scenario();
        // 20 + 10 + 5 + 5 + 14 + 10 = 64
        assert_eq!(sink.frame_count(), 64);
    }

    #[test]
    fn raw_sample_push_cycle() {
        let mut sink = TelemetrySink::new();
        let snap = sink.sample_raw(3.0, 0.2, 0.5, 0.9, 0.8, 0.1);
        sink.push_to_waterfall(snap);
        assert_eq!(sink.frame_count(), 1);
    }

    #[test]
    fn waterfall_most_recent_frame_has_phi() {
        let mut sink = TelemetrySink::new();
        let snap = sink.sample_raw(4.5, 0.1, 0.7, 0.95, 0.6, 0.05);
        sink.push_to_waterfall(snap);
        let frame = sink.waterfall.get_by_age(0).unwrap();
        let phi = frame.scalar("phi").unwrap();
        assert!((phi - 4.5).abs() < 1e-9);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified Cognitive Loop — Vision → Broca → Geodesic
//!
//! This is the top-level orchestration layer for the full Symthaea architecture.

use crate::encoder::ThoughtChannels;
#[cfg(feature = "mamba-cpu")]
use crate::liquid_mamba::{LiquidMambaGenerator, MonologueTrainingConfig};
use crate::thought_chunk::{ProgramNode, ThoughtChunkSequence};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, instrument, warn};

#[cfg(feature = "code-sheaf-eval")]
use symthaea_geodesic::synthesis::GeodesicSynthesizer;
#[cfg(feature = "code-sheaf-eval")]
use symthaea_geodesic::tri_oracle::TriOracle;
use symthaea_vision_manifold::manifold::VisionManifold;
use symthaea_vision_manifold::types::VisionTelemetry;

/// Output of one full cognitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveOutput {
    pub vision: VisionTelemetry,
    pub channels: ThoughtChannels,
    pub monologue: ThoughtChunkSequence,
    pub program_nodes: Vec<ProgramNode>,
    pub synthesis: Option<String>, // Final synthesized plan/code
    pub cycle_time_ms: u64,
    pub mean_psi: f32,
    pub mean_confidence: f32,
}

/// Telemetry and metrics for the cognitive loop.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitiveMetrics {
    pub total_steps: usize,
    pub total_cycles: usize,
    pub avg_cycle_time_ms: f32,
    pub avg_psi: f32,
    pub avg_confidence: f32,
    pub total_nodes_generated: usize,
    pub total_synthesis_calls: usize,
    pub last_cycle_time_ms: u64,
    pub last_mean_psi: f32,
    pub last_mean_confidence: f32,
}

impl CognitiveMetrics {
    pub fn update(&mut self, output: &CognitiveOutput) {
        self.total_steps += 1;
        self.total_cycles += 1;
        self.last_cycle_time_ms = output.cycle_time_ms;
        self.last_mean_psi = output.mean_psi;
        self.last_mean_confidence = output.mean_confidence;

        // Running averages
        let n = self.total_cycles as f32;
        self.avg_cycle_time_ms =
            (self.avg_cycle_time_ms * (n - 1.0) + output.cycle_time_ms as f32) / n;
        self.avg_psi = (self.avg_psi * (n - 1.0) + output.mean_psi) / n;
        self.avg_confidence = (self.avg_confidence * (n - 1.0) + output.mean_confidence) / n;

        self.total_nodes_generated += output.program_nodes.len();
        if output.synthesis.is_some() {
            self.total_synthesis_calls += 1;
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Configuration for the cognitive loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoopConfig {
    pub monologue_chunks: usize,
    pub enable_training: bool,
    #[cfg(feature = "mamba-cpu")]
    pub training_config: MonologueTrainingConfig,
    pub frame_width: u32,
    pub frame_height: u32,
    pub frame_channels: usize,
}

impl Default for CognitiveLoopConfig {
    fn default() -> Self {
        Self {
            monologue_chunks: 5,
            enable_training: true,
            #[cfg(feature = "mamba-cpu")]
            training_config: MonologueTrainingConfig::default(),
            frame_width: 640,
            frame_height: 480,
            frame_channels: 3,
        }
    }
}

/// The unified cognitive loop.
/// Owns Vision, Broca, and Geodesic subsystems.
pub struct CognitiveLoop {
    pub vision: VisionManifold,
    #[cfg(feature = "mamba-cpu")]
    pub broca: LiquidMambaGenerator,
    #[cfg(feature = "code-sheaf-eval")]
    pub geodesic: GeodesicSynthesizer,
    #[cfg(feature = "code-sheaf-eval")]
    pub tri_oracle: TriOracle,
    pub config: CognitiveLoopConfig,
    pub metrics: CognitiveMetrics,
    pub step_count: usize,
}

impl CognitiveLoop {
    pub fn new(
        vision: VisionManifold,
        #[cfg(feature = "mamba-cpu")] broca: LiquidMambaGenerator,
        #[cfg(feature = "code-sheaf-eval")] geodesic: GeodesicSynthesizer,
        #[cfg(feature = "code-sheaf-eval")] tri_oracle: TriOracle,
        config: CognitiveLoopConfig,
    ) -> Self {
        Self {
            vision,
            #[cfg(feature = "mamba-cpu")]
            broca,
            #[cfg(feature = "code-sheaf-eval")]
            geodesic,
            #[cfg(feature = "code-sheaf-eval")]
            tri_oracle,
            config,
            metrics: CognitiveMetrics::default(),
            step_count: 0,
        }
    }

    /// Run one full cognitive cycle: Vision → Broca → Geodesic
    #[instrument(skip(self, frame), fields(step = self.step_count))]
    pub fn cognitive_step(&mut self, frame: &[u8], dt: f32) -> Result<CognitiveOutput> {
        let start = Instant::now();

        debug!("Starting cognitive cycle {}", self.step_count);

        // 1. Vision: Perceive the world
        let vision_telemetry = self.vision.observe_frame(
            frame,
            self.config.frame_width,
            self.config.frame_height,
            self.config.frame_channels,
            dt,
        );

        // 2. Broca: Generate semantic monologue
        // Map real vision telemetry to thought channels
        let mut channels = ThoughtChannels::default();
        channels.channels[0] = vision_telemetry.manifold_coherence.clamp(0.0, 1.0);
        channels.channels[1] = (vision_telemetry.num_salient_patches as f32 / 10.0).min(1.0);
        channels.channels[9] = vision_telemetry.prediction_error.clamp(0.0, 1.0);
        // Note: In real system use from_vision() if properly mapped in encoder.rs

        #[cfg(feature = "mamba-cpu")]
        let monologue = self
            .broca
            .generate_semantic_monologue(&channels, self.config.monologue_chunks)?;
        #[cfg(not(feature = "mamba-cpu"))]
        let monologue = ThoughtChunkSequence::default();

        // 3. Convert monologue to program nodes
        let program_nodes = monologue.to_program_nodes();

        // 4. Geodesic: Synthesize structured output (plan/code/action)
        #[cfg(feature = "code-sheaf-eval")]
        let synthesis = if let Some(best_node) = program_nodes
            .iter()
            .find(|n| n.kind == crate::thought_chunk::NodeKind::Code)
        {
            let spec =
                symthaea_geodesic::synthesis::CodeSpec::new("synthesized", &best_node.content);
            let result = self.geodesic.synthesize(&spec);
            if result.success {
                Some("succeeded".to_string())
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(not(feature = "code-sheaf-eval"))]
        let synthesis = None;

        // 5. Optional self-supervised training
        #[cfg(feature = "mamba-cpu")]
        if self.config.enable_training {
            let _ = self
                .broca
                .train_on_semantic_monologue(&channels, &self.config.training_config);
        }

        let cycle_time = start.elapsed().as_millis() as u64;

        let output = CognitiveOutput {
            vision: vision_telemetry,
            channels,
            monologue,
            program_nodes,
            synthesis,
            cycle_time_ms: cycle_time,
            mean_psi: 0.0,
            mean_confidence: 0.0,
        };

        let mut final_output = output;
        final_output.mean_psi = final_output.monologue.mean_psi();
        final_output.mean_confidence = final_output.monologue.total_confidence();

        // === Update Telemetry ===
        self.metrics.update(&final_output);
        self.step_count += 1;

        info!(
            step = self.step_count,
            psi = final_output.mean_psi,
            confidence = final_output.mean_confidence,
            nodes = final_output.program_nodes.len(),
            time_ms = cycle_time,
            "Cognitive cycle completed"
        );

        Ok(final_output)
    }

    /// Get current telemetry snapshot
    pub fn get_metrics(&self) -> &CognitiveMetrics {
        &self.metrics
    }

    /// Reset all telemetry
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
        info!("Cognitive metrics reset");
    }

    /// Print a nice summary
    pub fn print_status(&self) {
        println!("=== CognitiveLoop Status ===");
        println!("Steps: {}", self.step_count);
        println!("Avg Cycle Time: {:.1}ms", self.metrics.avg_cycle_time_ms);
        println!("Avg Ψ: {:.3}", self.metrics.avg_psi);
        println!("Avg Confidence: {:.3}", self.metrics.avg_confidence);
        println!(
            "Total Nodes Generated: {}",
            self.metrics.total_nodes_generated
        );
        println!("Synthesis Calls: {}", self.metrics.total_synthesis_calls);
    }

    /// Get a summary of the last cognitive cycle
    pub fn last_cycle_summary(&self) -> String {
        format!(
            "Step {} | ψ={:.2} | conf={:.2}",
            self.step_count, self.metrics.last_mean_psi, self.metrics.last_mean_confidence
        )
    }
}

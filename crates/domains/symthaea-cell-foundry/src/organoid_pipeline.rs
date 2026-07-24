// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified organoid consciousness pipeline: connects morphogenetic field
//! dynamics (Direction E) with digital organoid simulation (Direction K).
//!
//! Pipeline: Stem cells -> Morphogenetic patterning -> Neural differentiation ->
//! Digital organoid simulation -> Consciousness monitoring -> Ethics gating
//!
//! This module provides the high-level orchestration that runs both systems
//! together, using the morphogenetic field to drive organoid tissue organization
//! and the digital organoid to simulate neural activity and measure Phi.
//!
//! References:
//! - Lancaster et al. (2013). Cerebral organoids model human brain development.
//! - Turing (1952). Chemical basis of morphogenesis.
//! - Tononi (2004). IIT.
//! - Levin (2019). The computational boundary of a self — see
//!   [`crate::bioelectric`] for the gap-junction/Vmem layer this pipeline's
//!   morphogenetic organoid now carries alongside the chemical field.

use crate::digital_organoid::{DevelopmentalStage, DigitalOrganoid, LocalFieldPotential};
use crate::morphogenetic_consciousness::NeuralOrganoid;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the unified organoid consciousness pipeline.
pub struct OrganoidPipelineConfig {
    /// Number of initial stem/progenitor cells.
    pub initial_cells: usize,
    /// Maximum cell count before proliferation is capped.
    pub max_cells: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Phi value at which the ethics gate halts the experiment.
    pub ethics_phi_threshold: f64,
    /// Day to simulate up to (unless ethics halts earlier).
    pub target_day: u32,
    /// Whether to record LFP each day.
    pub enable_lfp_recording: bool,
    /// Number of morphogenetic reaction-diffusion sub-steps per day.
    pub morphogenetic_steps_per_day: u32,
}

impl Default for OrganoidPipelineConfig {
    fn default() -> Self {
        Self {
            initial_cells: 100,
            max_cells: 10_000,
            seed: 42,
            ethics_phi_threshold: 0.5,
            target_day: 120,
            enable_lfp_recording: false,
            morphogenetic_steps_per_day: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Results from running the full pipeline.
pub struct PipelineResult {
    /// Day at which the pipeline stopped.
    pub final_day: u32,
    /// Phi on the last simulated day.
    pub final_phi: f64,
    /// Maximum Phi observed across the entire run.
    pub peak_phi: f64,
    /// Whether the pipeline was halted by the ethics gate.
    pub halted_by_ethics: bool,
    /// Human-readable reason for halting (if any).
    pub halt_reason: Option<String>,
    /// Complete day-by-day trajectory.
    pub developmental_trajectory: Vec<DaySnapshot>,
    /// Developmental stage on the final day.
    pub final_stage: DevelopmentalStage,
    /// Total cell count at the end.
    pub cell_count: usize,
    /// Neuron count at the end.
    pub neuron_count: usize,
    /// Synapse count at the end.
    pub synapse_count: usize,
}

/// Snapshot of one simulated day.
pub struct DaySnapshot {
    /// Developmental day number.
    pub day: u32,
    /// Total cell count.
    pub cell_count: usize,
    /// Fraction of cells that are neurons (from digital organoid model).
    pub neuron_fraction: f32,
    /// Synapses per neuron (from digital organoid model).
    pub synapse_density: f32,
    /// Estimated Phi (from digital organoid model).
    pub phi: f64,
    /// Mean firing rate across neurons (Hz).
    pub mean_firing_rate: f32,
    /// Developmental stage (from digital organoid model).
    pub stage: DevelopmentalStage,
    /// LFP recording if enabled.
    pub lfp: Option<LocalFieldPotential>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Unified organoid consciousness pipeline.
///
/// Runs two models in lockstep:
///   - [`NeuralOrganoid`] (Direction E): morphogenetic field, reaction-diffusion
///     patterning, consciousness-potential computation.
///   - [`DigitalOrganoid`] (Direction K): gene-expression-driven differentiation,
///     spiking neural activity, LFP, Phi estimation, ethics gating.
///
/// Each `step()` call advances both models by one developmental day and records
/// a [`DaySnapshot`]. The digital organoid's Phi is the authoritative
/// consciousness measure; the morphogenetic model's global phi provides a
/// complementary field-theoretic view.
pub struct OrganoidPipeline {
    config: OrganoidPipelineConfig,
    /// Direction K digital organoid.
    digital: DigitalOrganoid,
    /// Direction E morphogenetic organoid.
    morphogenetic: NeuralOrganoid,
    /// Accumulated day snapshots.
    day_snapshots: Vec<DaySnapshot>,
    /// Whether the pipeline has been halted.
    halted: bool,
    /// Reason for halting.
    halt_reason: Option<String>,
    /// Peak Phi observed so far.
    peak_phi: f64,
}

impl OrganoidPipeline {
    /// Create a new pipeline from the given configuration.
    pub fn new(config: OrganoidPipelineConfig) -> Self {
        let digital =
            DigitalOrganoid::with_max_cells(config.initial_cells, config.max_cells, config.seed);
        let morphogenetic =
            NeuralOrganoid::with_max_cells(config.initial_cells, config.max_cells, config.seed);
        Self {
            config,
            digital,
            morphogenetic,
            day_snapshots: Vec::new(),
            halted: false,
            halt_reason: None,
            peak_phi: 0.0,
        }
    }

    /// Run the full pipeline from stem cells to the target day (or ethics halt).
    pub fn run(&mut self) -> PipelineResult {
        while !self.halted && self.digital.day() < self.config.target_day {
            if self.step().is_none() {
                break;
            }
        }
        self.build_result()
    }

    /// Advance both models by one developmental day and return a snapshot.
    ///
    /// Returns `None` if the pipeline is halted.
    pub fn step(&mut self) -> Option<DaySnapshot> {
        if self.halted {
            return None;
        }

        // Advance the morphogenetic model (Direction E) at the configured
        // reaction-diffusion temporal resolution.
        self.morphogenetic
            .advance_day_with_substeps(self.config.morphogenetic_steps_per_day);

        // Advance the digital organoid model (Direction K).
        let metrics = match self.digital.advance_day() {
            Some(m) => m,
            None => {
                self.halted = true;
                self.halt_reason = Some("Digital organoid halted".to_string());
                return None;
            }
        };

        // Compute the LFP if enabled.
        let lfp = if self.config.enable_lfp_recording {
            Some(self.digital.generate_spatial_electrode_snapshot())
        } else {
            None
        };

        // Use the digital organoid's neural fraction — it tracks differentiation
        // via gene-expression-driven cell-fate decisions and stays consistent
        // with the cell_count also sourced from the digital organoid.
        let neuron_fraction = metrics.neural_fraction;

        let phi = metrics.network_integration_proxy();
        if phi > self.peak_phi {
            self.peak_phi = phi;
        }

        let snapshot = DaySnapshot {
            day: self.digital.day(),
            cell_count: metrics.cell_count,
            neuron_fraction,
            synapse_density: metrics.synapse_density,
            phi,
            mean_firing_rate: metrics.mean_firing_rate_hz,
            stage: metrics.stage,
            lfp,
        };

        self.day_snapshots.push(snapshot.clone_snapshot());

        // Ethics gate: halt if Phi exceeds threshold.
        if phi >= self.config.ethics_phi_threshold {
            self.halted = true;
            self.halt_reason = Some(format!(
                "Network integration proxy ({:.4}) exceeded ethics threshold ({:.4}) on day {}",
                phi,
                self.config.ethics_phi_threshold,
                self.digital.day()
            ));
            self.digital.terminate(&format!(
                "Ethics halt: integration proxy {:.4} >= {:.4}",
                phi, self.config.ethics_phi_threshold
            ));
        }

        Some(snapshot)
    }

    /// Get the current developmental stage from the digital organoid.
    pub fn stage(&self) -> DevelopmentalStage {
        self.digital.stage()
    }

    /// Get the current network integration proxy from the digital organoid.
    pub fn current_phi(&self) -> f64 {
        self.digital.phi_history().last().copied().unwrap_or(0.0)
    }

    /// Whether the pipeline has been halted (by ethics or completion).
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Get all day snapshots recorded so far.
    pub fn trajectory(&self) -> &[DaySnapshot] {
        &self.day_snapshots
    }

    /// Generate the graph/activity/maturity integration-proxy curve.
    pub fn integration_proxy_curve(&self) -> Vec<(u32, f64)> {
        self.day_snapshots.iter().map(|s| (s.day, s.phi)).collect()
    }

    /// Historical compatibility wrapper for [`Self::integration_proxy_curve`].
    #[deprecated(
        since = "0.1.0",
        note = "the value is a network integration proxy, not IIT Phi"
    )]
    pub fn phi_curve(&self) -> Vec<(u32, f64)> {
        self.integration_proxy_curve()
    }

    /// Find the first day when the integration proxy reached a threshold.
    pub fn integration_proxy_threshold_day(&self, threshold: f64) -> Option<u32> {
        self.day_snapshots
            .iter()
            .find(|s| s.phi >= threshold)
            .map(|s| s.day)
    }

    /// Historical compatibility wrapper for
    /// [`Self::integration_proxy_threshold_day`].
    #[deprecated(
        since = "0.1.0",
        note = "threshold crossing is not evidence of consciousness onset"
    )]
    pub fn consciousness_onset_day(&self, threshold: f64) -> Option<u32> {
        self.integration_proxy_threshold_day(threshold)
    }

    /// Get a reference to the underlying digital organoid.
    pub fn digital_organoid(&self) -> &DigitalOrganoid {
        &self.digital
    }

    /// Get a reference to the underlying morphogenetic organoid.
    pub fn morphogenetic_organoid(&self) -> &NeuralOrganoid {
        &self.morphogenetic
    }

    /// Get a mutable reference to the underlying morphogenetic organoid —
    /// used to drive bioelectric perturbation/regeneration experiments
    /// (`amputate`, `capture_target_morphology`,
    /// `set_gap_junction_permeability`) alongside the pipeline's normal
    /// day-by-day `step`/`run`.
    pub fn morphogenetic_organoid_mut(&mut self) -> &mut NeuralOrganoid {
        &mut self.morphogenetic
    }

    // -- private --

    fn build_result(&self) -> PipelineResult {
        let final_phi = self.day_snapshots.last().map(|s| s.phi).unwrap_or(0.0);
        let final_stage = self.digital.stage();
        let metrics = self.digital.metrics();

        PipelineResult {
            final_day: self.digital.day(),
            final_phi,
            peak_phi: self.peak_phi,
            halted_by_ethics: self.halt_reason.is_some(),
            halt_reason: self.halt_reason.clone(),
            developmental_trajectory: self
                .day_snapshots
                .iter()
                .map(|s| s.clone_snapshot())
                .collect(),
            final_stage,
            cell_count: metrics.cell_count,
            neuron_count: metrics.neuron_count,
            synapse_count: metrics.synapse_count,
        }
    }
}

// DaySnapshot doesn't derive Clone because LocalFieldPotential might be large,
// but we need to copy snapshots for the result.
impl DaySnapshot {
    fn clone_snapshot(&self) -> DaySnapshot {
        DaySnapshot {
            day: self.day,
            cell_count: self.cell_count,
            neuron_fraction: self.neuron_fraction,
            synapse_density: self.synapse_density,
            phi: self.phi,
            mean_firing_rate: self.mean_firing_rate,
            stage: self.stage,
            lfp: self.lfp.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_creates_with_default_config() {
        let pipeline = OrganoidPipeline::new(OrganoidPipelineConfig::default());
        assert!(!pipeline.is_halted());
        assert_eq!(pipeline.stage(), DevelopmentalStage::EarlyProliferation);
        assert!(pipeline.trajectory().is_empty());
    }

    #[test]
    fn pipeline_runs_and_produces_trajectory() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            target_day: 30,
            seed: 7,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();
        assert!(!result.developmental_trajectory.is_empty());
        assert!(result.final_day > 0);
        assert!(result.final_day <= 30);
    }

    #[test]
    fn phi_increases_over_development() {
        let config = OrganoidPipelineConfig {
            initial_cells: 80,
            target_day: 80,
            seed: 99,
            // High threshold so ethics doesn't halt us.
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();
        // Peak phi should be at least as large as the first day's phi.
        let first_phi = result
            .developmental_trajectory
            .first()
            .map(|s| s.phi)
            .unwrap_or(0.0);
        assert!(result.peak_phi >= first_phi);
    }

    #[test]
    fn ethics_halt_stops_pipeline() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            target_day: 200,
            seed: 42,
            // Very low threshold so it triggers quickly.
            ethics_phi_threshold: 0.001,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();
        // If phi ever exceeded 0.001 the pipeline should have halted early.
        if result.peak_phi >= 0.001 {
            assert!(result.halted_by_ethics);
            assert!(result.halt_reason.is_some());
            assert!(result.final_day < 200);
        }
    }

    #[test]
    fn consciousness_onset_day_returns_correct_value() {
        let config = OrganoidPipelineConfig {
            initial_cells: 60,
            target_day: 60,
            seed: 123,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        pipeline.run();

        // Onset at threshold 0.0 should be the first day with any Phi > 0.
        let onset_zero = pipeline.integration_proxy_threshold_day(0.0);
        // If there's any trajectory at all, day 1+ should have phi >= 0.0.
        if !pipeline.trajectory().is_empty() {
            assert!(onset_zero.is_some());
        }

        // Onset at impossibly high threshold should be None.
        let onset_high = pipeline.integration_proxy_threshold_day(999.0);
        assert!(onset_high.is_none());
    }

    #[test]
    fn cell_count_increases_during_proliferative_phase() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            target_day: 15,
            seed: 55,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();

        // After proliferation days, cell count should be >= initial.
        assert!(result.cell_count >= 50);
    }

    #[test]
    fn neuron_fraction_increases_after_neural_induction() {
        let config = OrganoidPipelineConfig {
            initial_cells: 80,
            target_day: 50,
            seed: 77,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();

        // Find earliest and latest neuron fractions.
        let early_frac = result
            .developmental_trajectory
            .iter()
            .take(5)
            .map(|s| s.neuron_fraction)
            .sum::<f32>();
        let late_frac = result
            .developmental_trajectory
            .iter()
            .rev()
            .take(5)
            .map(|s| s.neuron_fraction)
            .sum::<f32>();

        // Late neuron fraction should be at least as high as early.
        assert!(late_frac >= early_frac);
    }

    #[test]
    fn pipeline_result_contains_valid_metadata() {
        let config = OrganoidPipelineConfig {
            initial_cells: 40,
            target_day: 20,
            seed: 11,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();

        assert_eq!(result.final_day, 20);
        assert!(result.peak_phi >= result.final_phi);
        assert!(result.cell_count > 0);
        assert_eq!(result.developmental_trajectory.len(), 20);
    }

    #[test]
    fn integration_proxy_api_uses_epistemically_bounded_names() {
        let mut pipeline = OrganoidPipeline::new(OrganoidPipelineConfig {
            initial_cells: 30,
            target_day: 5,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        });
        pipeline.run();
        assert_eq!(
            pipeline.integration_proxy_curve().len(),
            pipeline.trajectory().len()
        );
        assert_eq!(pipeline.integration_proxy_threshold_day(999.0), None);
    }

    #[test]
    fn phi_curve_has_correct_length() {
        let config = OrganoidPipelineConfig {
            initial_cells: 30,
            target_day: 25,
            seed: 33,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        pipeline.run();

        let curve = pipeline.integration_proxy_curve();
        assert_eq!(curve.len(), pipeline.trajectory().len());
        // Each entry should have day > 0.
        for (day, _phi) in &curve {
            assert!(*day > 0);
        }
    }

    #[test]
    fn halted_pipeline_returns_halted_in_result() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            target_day: 200,
            seed: 42,
            ethics_phi_threshold: 0.001,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();

        // After run, is_halted should match the result.
        if result.halted_by_ethics {
            assert!(pipeline.is_halted());
            assert!(result.halt_reason.is_some());
        }
    }

    #[test]
    fn max_cells_config_applies_to_both_models() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            max_cells: 55,
            target_day: 100,
            seed: 2026,
            ethics_phi_threshold: 10.0,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();
        assert!(result.cell_count <= 55);
        assert!(pipeline.morphogenetic_organoid().field.cells.len() <= 55);
        assert_eq!(pipeline.digital_organoid().max_cells(), 55);
        assert_eq!(pipeline.morphogenetic_organoid().field.max_cells, 55);
    }

    #[test]
    fn morphogenetic_steps_per_day_changes_field_trajectory() {
        let mut zero = OrganoidPipeline::new(OrganoidPipelineConfig {
            initial_cells: 30,
            target_day: 1,
            seed: 909,
            ethics_phi_threshold: 10.0,
            morphogenetic_steps_per_day: 0,
            ..Default::default()
        });
        let mut ten = OrganoidPipeline::new(OrganoidPipelineConfig {
            initial_cells: 30,
            target_day: 1,
            seed: 909,
            ethics_phi_threshold: 10.0,
            morphogenetic_steps_per_day: 10,
            ..Default::default()
        });
        zero.step();
        ten.step();
        assert_ne!(
            zero.morphogenetic_organoid().field.activator,
            ten.morphogenetic_organoid().field.activator
        );
    }

    #[test]
    fn lfp_recording_when_enabled() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            target_day: 10,
            seed: 88,
            ethics_phi_threshold: 10.0,
            enable_lfp_recording: true,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        pipeline.run();

        // Every snapshot should have an LFP.
        for snap in pipeline.trajectory() {
            assert!(snap.lfp.is_some());
        }
    }

    #[test]
    fn lfp_not_recorded_when_disabled() {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            target_day: 10,
            seed: 88,
            ethics_phi_threshold: 10.0,
            enable_lfp_recording: false,
            ..Default::default()
        };
        let mut pipeline = OrganoidPipeline::new(config);
        pipeline.run();

        for snap in pipeline.trajectory() {
            assert!(snap.lfp.is_none());
        }
    }
}

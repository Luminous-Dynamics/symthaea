// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Biosphere Coherence Index B(t)
//!
//! Continuous coherence curve from the origin of life (3.8 Ga) to present.
//! Computed as the geometric mean of 6 normalized dimensions:
//!
//! 1. **Biodiversity** — genus richness (PBDB, taphonomically corrected)
//! 2. **Complexity** — maximum organism complexity grade [1-10]
//! 3. **Resilience** — recovery half-time after mass extinctions
//! 4. **Network integration** — trophic web density (Kleiber's Law + O₂)
//! 5. **Energy throughput** — biosphere metabolic power (GEOCARB + biomass)
//! 6. **Information capacity** — genome complexity × species richness
//!
//! ## Key innovations
//!
//! - **Taphonomic filter**: SQS-inspired Bayesian correction for preservation bias
//! - **Shared thermodynamic currency**: B(t)→K(t) bridge via energy throughput (HANPP)
//! - **Lotka-Volterra niche-filling**: multi-phase logistic recovery replaces exponential
//! - **Metabolic scaling**: Kleiber's Law derives network integration from O₂ + complexity
//! - **Geochemical HMM bounding**: δ¹³C constrains Precambrian CIs via biomass correlation
//!
//! ## Architecture
//!
//! Implements `SimulationModule` trait. Runs at `TickPhase::Viability` (phase 0)
//! with no dependencies, so it executes before the ViabilityEngine.
//! On tick 0, emits ecosystem initial conditions as feedback signals.
//! Every tick, emits the `biosphere_coherence` signal.

pub mod bridge;
pub mod consciousness_emergence;
pub mod counterfactual;
pub mod counterfactual_consciousness;
pub mod deep_time_data;
pub mod dimensions;
pub mod ensemble;
pub mod kt_engine;
pub mod mass_extinctions;
pub mod sensitivity;
pub mod temporal_bins;
pub mod uncertainty;
pub mod unified_curve;
pub mod validation;

use std::collections::HashMap;

use luminous_sim_core::module_trait::{
    ModuleAssessment, ModuleId, ModuleOutputs, ReportSection, SensitivityParam, SimulationModule,
    TickPhase,
};

use bridge::{
    biosphere_energy_adjustment, bt_to_ecosystem_health, CoherencePoint, CoherenceSource,
};
use deep_time_data::DeepTimeData;
use dimensions::{compute_bt, compute_bt_ci, compute_raw_dimensions, BiosphereMaxima};
use mass_extinctions::{canonical_mass_extinctions, extinction_multiplier, MassExtinctionEvent};
use temporal_bins::{build_deep_time_bins, MaAge, TemporalBin};

/// Top-level biosphere coherence engine.
///
/// Precomputes B(t) from 3.8 Ga to present during `build()`,
/// then provides lookup + interpolation during simulation ticks.
pub struct BiosphereCoherenceEngine {
    /// Temporal bins (sorted oldest-first).
    bins: Vec<TemporalBin>,
    /// B(t) values per bin (after extinction multiplier + energy adjustment).
    bt_values: Vec<f64>,
    /// B(t) 95% CI bounds per bin [lower, upper].
    bt_ci: Vec<(f64, f64)>,
    /// Mass extinction events.
    extinctions: Vec<MassExtinctionEvent>,
    /// Full coherence curve (for export).
    curve: Vec<CoherencePoint>,
    /// Whether tick 0 has been processed.
    initialized: bool,
}

impl BiosphereCoherenceEngine {
    /// Build the full B(t) curve from embedded data tables.
    /// Called once at initialization.
    pub fn build() -> Self {
        let data = DeepTimeData::load();
        let extinctions = canonical_mass_extinctions();
        let bins = build_deep_time_bins();
        let maxima = BiosphereMaxima::default();

        let mut bt_values = Vec::with_capacity(bins.len());
        let mut bt_ci = Vec::with_capacity(bins.len());
        let mut curve = Vec::with_capacity(bins.len());

        for bin in &bins {
            let age = bin.midpoint_ma;

            // Compute raw dimensions
            let raw = compute_raw_dimensions(bin, &data, &extinctions);

            // Get δ¹³C excursion for geochemical CI bounding
            let d13c_excursion = Some(data.interpolate_d13c_excursion(age));

            // Normalize (with real δ¹³C data for CI tightening)
            let norm = raw.normalize(&maxima, bin.resolution, age, d13c_excursion);

            // Geometric mean
            let bt_raw = compute_bt(&norm);

            // Apply extinction shock-recovery multiplier
            let ext_mult = extinction_multiplier(age, &extinctions);

            // Apply energy adjustment (HANPP for recent ages)
            let energy_adj = biosphere_energy_adjustment(age);

            let bt = (bt_raw * ext_mult * energy_adj).clamp(0.001, 1.0);

            // Confidence intervals
            let (ci_lo_raw, ci_hi_raw) = compute_bt_ci(&norm);
            let ci_lo = (ci_lo_raw * ext_mult * energy_adj).max(0.0);
            let ci_hi = (ci_hi_raw * ext_mult * energy_adj).min(1.0);

            bt_values.push(bt);
            bt_ci.push((ci_lo, ci_hi));

            curve.push(CoherencePoint {
                age_ma: age,
                value: bt,
                ci_lower: ci_lo,
                ci_upper: ci_hi,
                source: if age > bridge::BT_KT_HANDOFF_MA {
                    CoherenceSource::Biosphere
                } else if age > bridge::BT_KT_HANDOFF_MA - bridge::BT_KT_BLEND_WINDOW_MA {
                    CoherenceSource::Blended
                } else {
                    CoherenceSource::Civilization
                },
            });
        }

        Self {
            bins,
            bt_values,
            bt_ci,
            extinctions,
            curve,
            initialized: false,
        }
    }

    /// Interpolate B(t) at an arbitrary age (Ma).
    pub fn bt_at(&self, age_ma: MaAge) -> (f64, f64, f64) {
        if self.bins.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Find bracketing bins
        for i in 0..self.bins.len().saturating_sub(1) {
            let older = &self.bins[i];
            let younger = &self.bins[i + 1];
            if age_ma <= older.midpoint_ma && age_ma >= younger.midpoint_ma {
                let t = (older.midpoint_ma - age_ma) / (older.midpoint_ma - younger.midpoint_ma);
                let val = self.bt_values[i] + t * (self.bt_values[i + 1] - self.bt_values[i]);
                let lo = self.bt_ci[i].0 + t * (self.bt_ci[i + 1].0 - self.bt_ci[i].0);
                let hi = self.bt_ci[i].1 + t * (self.bt_ci[i + 1].1 - self.bt_ci[i].1);
                return (val, lo, hi);
            }
        }

        // Clamp to nearest
        if age_ma >= self.bins[0].midpoint_ma {
            (self.bt_values[0], self.bt_ci[0].0, self.bt_ci[0].1)
        } else {
            let last = self.bins.len() - 1;
            (self.bt_values[last], self.bt_ci[last].0, self.bt_ci[last].1)
        }
    }

    /// Get the full B(t) curve for export/visualization.
    pub fn full_curve(&self) -> &[CoherencePoint] {
        &self.curve
    }

    /// Number of temporal bins.
    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    /// Number of mass extinction events modeled.
    pub fn extinction_count(&self) -> usize {
        self.extinctions.len()
    }

    /// Terminal B(t) value (most recent bin).
    pub fn terminal_bt(&self) -> f64 {
        self.bt_values.last().copied().unwrap_or(0.0)
    }

    /// Export the curve as CSV string.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("age_ma,bt_value,ci_lower,ci_upper,source\n");
        for pt in &self.curve {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:?}\n",
                pt.age_ma, pt.value, pt.ci_lower, pt.ci_upper, pt.source
            ));
        }
        csv
    }
}

impl SimulationModule for BiosphereCoherenceEngine {
    fn id(&self) -> ModuleId {
        "biosphere_coherence"
    }

    fn name(&self) -> &str {
        "Deep-Time Biosphere Coherence B(t)"
    }

    fn phase(&self) -> TickPhase {
        TickPhase::Viability // Phase 0, runs first
    }

    fn tick(
        &mut self,
        _tick: u32,
        feedback: &HashMap<String, f64>,
    ) -> Result<ModuleOutputs, String> {
        let mut outputs = ModuleOutputs::default();

        if !self.initialized {
            self.initialized = true;

            // Emit initial ecosystem conditions from B(t) terminal state
            let terminal = self.terminal_bt();
            let health = bt_to_ecosystem_health(terminal, terminal);

            outputs
                .feedback_signals
                .insert("bt_biodiversity".into(), health.biodiversity);
            outputs
                .feedback_signals
                .insert("bt_forest_cover".into(), health.forest_cover);
            outputs
                .feedback_signals
                .insert("bt_soil_health".into(), health.soil_health);
            outputs
                .feedback_signals
                .insert("bt_ocean_health".into(), health.ocean_health);
            outputs
                .feedback_signals
                .insert("bt_freshwater".into(), health.freshwater);
            outputs
                .feedback_signals
                .insert("bt_terminal_value".into(), terminal);

            // Shock-recovery baseline (mean recovery time across all events)
            let mean_recovery = self
                .extinctions
                .iter()
                .map(|e| e.recovery_time_ma)
                .sum::<f64>()
                / self.extinctions.len().max(1) as f64;
            outputs
                .feedback_signals
                .insert("bt_mean_recovery_ma".into(), mean_recovery);

            // Report
            outputs.metrics.push(("bt_terminal".into(), terminal));
            outputs
                .metrics
                .push(("bt_bins".into(), self.bins.len() as f64));
            outputs
                .metrics
                .push(("bt_extinctions".into(), self.extinctions.len() as f64));
        }

        // Every tick: emit planetary context signal
        let eco_feedback = feedback
            .get("ecosystem_service_index")
            .copied()
            .unwrap_or(0.8);
        outputs
            .feedback_signals
            .insert("biosphere_coherence".into(), eco_feedback);

        Ok(outputs)
    }

    fn report_section(&self) -> Option<ReportSection> {
        let terminal = self.terminal_bt();
        let (_, ci_lo, ci_hi) = self.bt_at(0.005); // ~3000 BCE

        Some(ReportSection {
            module_id: "biosphere_coherence".into(),
            module_name: "Deep-Time Biosphere Coherence B(t)".into(),
            summary: format!(
                "B(t) computed across {} bins from 3.8 Ga to 3000 BCE. \
                 Terminal value: {:.3} (CI: {:.3}-{:.3}). \
                 {} mass extinction events modeled with Lotka-Volterra recovery. \
                 HANPP energy bridge grounds B(t)→K(t) transition.",
                self.bins.len(),
                terminal,
                ci_lo,
                ci_hi,
                self.extinctions.len(),
            ),
            metrics: vec![
                ("bt_terminal".into(), terminal),
                ("bt_ci_lower_terminal".into(), ci_lo),
                ("bt_ci_upper_terminal".into(), ci_hi),
                ("extinction_events".into(), self.extinctions.len() as f64),
                ("temporal_bins".into(), self.bins.len() as f64),
            ],
            warnings: vec![
                "Precambrian CIs are wide (±0.3-0.4) — honest epistemic uncertainty".into(),
                "Network integration and information capacity use metabolic scaling estimates"
                    .into(),
            ],
        })
    }

    fn honest_assessment(&self) -> Option<ModuleAssessment> {
        Some(ModuleAssessment {
            module_id: "biosphere_coherence".into(),
            confidence: "Medium".into(),
            limitations: vec![
                "Precambrian diversity corrected via taphonomic filter but still uncertain (CI ±0.25-0.40)".into(),
                "B(t)→K(t) bridge uses shared thermodynamic currency (HANPP) but HANPP data pre-1800 is estimated".into(),
                "Recovery model uses Lotka-Volterra niche-filling (better than exponential) but real recovery has stochastic elements".into(),
                "Network integration bounded by Kleiber's Law + O₂ but trophic webs rarely preserved in fossil record".into(),
                "Geochemical HMM bounding tightens Precambrian CIs where δ¹³C data exists, but gaps remain".into(),
                "Genus-level diversity has preservation bias despite taphonomic correction (Alroy 2010 SQS)".into(),
            ],
        })
    }

    fn sensitivity_params(&self) -> Vec<SensitivityParam> {
        vec![
            SensitivityParam {
                name: "bt_carrying_capacity_ratio".into(),
                description: "Post-extinction carrying capacity overshoot".into(),
                default_value: 1.10,
                min_value: 1.0,
                max_value: 1.30,
                citation: Some("Erwin (2001) 'Lessons from the past'".into()),
            },
            SensitivityParam {
                name: "bt_recovery_rate".into(),
                description: "Lotka-Volterra intrinsic recovery rate (per Ma)".into(),
                default_value: 0.5,
                min_value: 0.2,
                max_value: 1.0,
                citation: Some("Kirchner & Weil (2000) 'Delayed biological recovery'".into()),
            },
            SensitivityParam {
                name: "bt_taphonomic_multiplier_precambrian".into(),
                description: "Preservation correction for Precambrian soft-body bias".into(),
                default_value: 2.0,
                min_value: 1.0,
                max_value: 5.0,
                citation: Some("Alroy (2010) 'Fair sampling of taxonomic richness'".into()),
            },
            SensitivityParam {
                name: "bt_hanpp_modern".into(),
                description: "Modern human appropriation of NPP".into(),
                default_value: 0.24,
                min_value: 0.15,
                max_value: 0.35,
                citation: Some("Haberl et al. (2007) 'Quantifying HANPP'".into()),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_builds_successfully() {
        let engine = BiosphereCoherenceEngine::build();
        assert!(engine.bin_count() >= 50);
        assert!(engine.extinction_count() >= 14);
    }

    #[test]
    fn bt_values_in_valid_range() {
        let engine = BiosphereCoherenceEngine::build();
        for (i, &val) in engine.bt_values.iter().enumerate() {
            assert!(
                val >= 0.0 && val <= 1.0,
                "B(t) out of range at bin {}: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn bt_increases_over_deep_time() {
        let engine = BiosphereCoherenceEngine::build();
        // Origin of life should be lower than Cambrian
        let (bt_origin, _, _) = engine.bt_at(3500.0);
        let (bt_cambrian, _, _) = engine.bt_at(500.0);
        assert!(
            bt_origin < bt_cambrian,
            "B(t) at origin ({}) should be < Cambrian ({})",
            bt_origin,
            bt_cambrian
        );
    }

    #[test]
    fn end_permian_is_a_trough() {
        let engine = BiosphereCoherenceEngine::build();
        let (bt_pre, _, _) = engine.bt_at(260.0);
        let (bt_permian, _, _) = engine.bt_at(252.0);
        let (bt_post, _, _) = engine.bt_at(240.0);
        assert!(
            bt_permian < bt_pre,
            "End-Permian ({}) should be < pre-Permian ({})",
            bt_permian,
            bt_pre
        );
        assert!(
            bt_post > bt_permian,
            "Post-Permian recovery ({}) should be > nadir ({})",
            bt_post,
            bt_permian
        );
    }

    #[test]
    fn terminal_bt_is_reasonable() {
        let engine = BiosphereCoherenceEngine::build();
        let terminal = engine.terminal_bt();
        // Terminal is at ~3000 BCE when biosphere is near peak (pre-industrial)
        // All dimensions are near maxima, so B(t) ≈ 0.7-1.0 is expected
        assert!(
            terminal > 0.3,
            "Terminal B(t) should be >0.3, got {}",
            terminal
        );
    }

    #[test]
    fn csv_export_has_header_and_data() {
        let engine = BiosphereCoherenceEngine::build();
        let csv = engine.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].contains("age_ma"));
        assert!(lines.len() > 50, "CSV should have >50 data rows");
    }

    #[test]
    fn simulation_module_tick_emits_signals() {
        let mut engine = BiosphereCoherenceEngine::build();
        let feedback = HashMap::new();
        let outputs = engine.tick(0, &feedback).unwrap();
        assert!(outputs.feedback_signals.contains_key("bt_terminal_value"));
        assert!(outputs.feedback_signals.contains_key("bt_biodiversity"));
        assert!(outputs.feedback_signals.contains_key("biosphere_coherence"));
    }

    #[test]
    fn simulation_module_report_exists() {
        let engine = BiosphereCoherenceEngine::build();
        let report = engine.report_section().unwrap();
        assert!(report.summary.contains("3.8 Ga"));
        assert!(!report.metrics.is_empty());
    }

    #[test]
    fn honest_assessment_acknowledges_limitations() {
        let engine = BiosphereCoherenceEngine::build();
        let assessment = engine.honest_assessment().unwrap();
        assert!(assessment.limitations.len() >= 5);
        // Should mention all 5 upgraded limitations
        let text = assessment.limitations.join(" ");
        assert!(
            text.contains("taphonomic"),
            "Should mention taphonomic correction"
        );
        assert!(
            text.contains("HANPP") || text.contains("thermodynamic"),
            "Should mention energy bridge"
        );
        assert!(
            text.contains("Lotka-Volterra") || text.contains("niche"),
            "Should mention logistic recovery"
        );
        assert!(text.contains("Kleiber"), "Should mention metabolic scaling");
        assert!(
            text.contains("δ¹³C") || text.contains("Geochemical"),
            "Should mention geochemical bounding"
        );
    }
}

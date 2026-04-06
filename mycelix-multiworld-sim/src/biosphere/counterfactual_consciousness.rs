// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterfactual consciousness analysis.
//!
//! The ultimate question: **was the Chicxulub asteroid required for consciousness?**
//!
//! This module runs the consciousness emergence analysis on the Complexity-Capped
//! K-Pg counterfactual branch. If the capped biosphere's consciousness feasibility
//! never reaches THRESHOLD_REFLECTIVE (0.30), then the asteroid wasn't just helpful —
//! it was thermodynamically necessary for meta-cognition to emerge.

use super::consciousness_emergence::{
    compute_emergence_curve, ConsciousnessEmergencePoint, THRESHOLD_AWARE, THRESHOLD_MINIMAL,
    THRESHOLD_REFLECTIVE, THRESHOLD_SENTIENT,
};
use super::deep_time_data::DeepTimeData;
use super::dimensions::{compute_bt, compute_raw_dimensions, BiosphereMaxima};
use super::mass_extinctions::{canonical_mass_extinctions, MassExtinctionEvent};
use super::temporal_bins::{build_deep_time_bins, MaAge};

/// Result of the counterfactual consciousness analysis.
#[derive(Debug)]
pub struct CounterfactualConsciousnessResult {
    /// Baseline consciousness emergence (all extinctions active).
    pub baseline_milestones: Vec<(String, MaAge)>,
    /// Peak baseline feasibility.
    pub baseline_peak_feasibility: f64,

    /// K-Pg suppressed consciousness emergence (reptilian ceiling).
    pub kpg_suppressed_milestones: Vec<(String, MaAge)>,
    /// Peak K-Pg suppressed feasibility.
    pub kpg_suppressed_peak_feasibility: f64,
    /// Whether meta-cognitive reflection is ever reached without Chicxulub.
    pub meta_cognition_possible_without_asteroid: bool,
    /// Whether self-awareness is ever reached without Chicxulub.
    pub self_awareness_possible_without_asteroid: bool,

    /// The complexity cap values that lock the reptilian ceiling.
    pub reptilian_ceiling: (f64, f64, f64), // (complexity, energy, info)

    /// Full emergence curves for both branches.
    pub baseline_curve: Vec<ConsciousnessEmergencePoint>,
    pub suppressed_curve: Vec<ConsciousnessEmergencePoint>,
}

/// Compute dimension norms for the K-Pg suppressed (Complexity-Capped) branch.
///
/// All bins after the K-Pg extinction (66 Ma) have complexity, energy throughput,
/// and information capacity permanently capped at Late Cretaceous levels.
fn compute_kpg_capped_dimensions(
) -> (Vec<MaAge>, Vec<super::dimensions::BiosphereDimensionsNorm>, (f64, f64, f64)) {
    let data = DeepTimeData::load();
    let all_extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    // Find the K-Pg event
    let kpg = all_extinctions
        .iter()
        .find(|e| e.name.contains("End-Cretaceous"))
        .expect("K-Pg event must exist");

    // Filter out K-Pg from extinction list
    let filtered_extinctions: Vec<MassExtinctionEvent> = all_extinctions
        .iter()
        .filter(|e| !e.name.contains("End-Cretaceous"))
        .cloned()
        .collect();

    // Find the pre-K-Pg bin to establish the ceiling
    let pre_kpg_bin = bins
        .iter()
        .filter(|b| b.midpoint_ma > kpg.age_ma)
        .min_by(|a, b| {
            (a.midpoint_ma - kpg.age_ma)
                .abs()
                .partial_cmp(&(b.midpoint_ma - kpg.age_ma).abs())
                .unwrap()
        })
        .expect("Should find pre-K-Pg bin");

    let pre_kpg_raw = compute_raw_dimensions(pre_kpg_bin, &data, &all_extinctions);
    let pre_kpg_d13c = Some(data.interpolate_d13c_excursion(pre_kpg_bin.midpoint_ma));
    let pre_kpg_norm = pre_kpg_raw.normalize(&maxima, pre_kpg_bin.resolution, pre_kpg_bin.midpoint_ma, pre_kpg_d13c);

    let complexity_cap = pre_kpg_norm.values[1];
    let energy_cap = pre_kpg_norm.values[4];
    let info_cap = pre_kpg_norm.values[5];

    let ages: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();
    let dims: Vec<super::dimensions::BiosphereDimensionsNorm> = bins
        .iter()
        .map(|bin| {
            let raw = compute_raw_dimensions(bin, &data, &filtered_extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
            let mut norm = raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c);

            // Apply Complexity Cap for all bins after K-Pg (younger = smaller age)
            if bin.midpoint_ma < kpg.age_ma {
                norm.values[1] = norm.values[1].min(complexity_cap);
                norm.values[4] = norm.values[4].min(energy_cap);
                norm.values[5] = norm.values[5].min(info_cap);
            }

            norm
        })
        .collect();

    (ages, dims, (complexity_cap, energy_cap, info_cap))
}

/// Run the full counterfactual consciousness analysis.
pub fn run_counterfactual_consciousness() -> CounterfactualConsciousnessResult {
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    // ── Baseline: all extinctions active ──
    let baseline_ages: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();
    let baseline_dims: Vec<_> = bins
        .iter()
        .map(|bin| {
            let raw = compute_raw_dimensions(bin, &data, &extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
            raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c)
        })
        .collect();
    let baseline_curve = compute_emergence_curve(&baseline_ages, &baseline_dims);

    // ── K-Pg Suppressed: Complexity-Capped branch ──
    let (suppressed_ages, suppressed_dims, ceiling) = compute_kpg_capped_dimensions();
    let suppressed_curve = compute_emergence_curve(&suppressed_ages, &suppressed_dims);

    // Extract milestones
    let baseline_milestones = extract_milestones(&baseline_curve);
    let suppressed_milestones = extract_milestones(&suppressed_curve);

    let baseline_peak = baseline_curve
        .iter()
        .map(|p| p.feasibility)
        .fold(0.0_f64, f64::max);
    let suppressed_peak = suppressed_curve
        .iter()
        .map(|p| p.feasibility)
        .fold(0.0_f64, f64::max);

    let meta_possible = suppressed_peak >= THRESHOLD_REFLECTIVE;
    let aware_possible = suppressed_peak >= THRESHOLD_AWARE;

    CounterfactualConsciousnessResult {
        baseline_milestones,
        baseline_peak_feasibility: baseline_peak,
        kpg_suppressed_milestones: suppressed_milestones,
        kpg_suppressed_peak_feasibility: suppressed_peak,
        meta_cognition_possible_without_asteroid: meta_possible,
        self_awareness_possible_without_asteroid: aware_possible,
        reptilian_ceiling: ceiling,
        baseline_curve,
        suppressed_curve,
    }
}

fn extract_milestones(curve: &[ConsciousnessEmergencePoint]) -> Vec<(String, MaAge)> {
    curve
        .iter()
        .filter_map(|pt| {
            pt.milestone.as_ref().map(|m| (m.clone(), pt.age_ma))
        })
        .collect()
}

impl CounterfactualConsciousnessResult {
    /// Format as a markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Was the Asteroid Required for Consciousness?\n\n");

        md.push_str("## Baseline (All Extinctions Active)\n\n");
        for (milestone, age) in &self.baseline_milestones {
            md.push_str(&format!("- **{}**: ~{:.0} Ma\n", milestone, age));
        }
        md.push_str(&format!("- Peak feasibility: {:.4}\n\n", self.baseline_peak_feasibility));

        md.push_str("## K-Pg Suppressed (Reptilian Ceiling)\n\n");
        md.push_str(&format!(
            "Complexity cap: {:.2}, Energy cap: {:.2}, Information cap: {:.2}\n\n",
            self.reptilian_ceiling.0, self.reptilian_ceiling.1, self.reptilian_ceiling.2
        ));
        if self.kpg_suppressed_milestones.is_empty() {
            md.push_str("- **No consciousness milestones achieved.**\n");
        } else {
            for (milestone, age) in &self.kpg_suppressed_milestones {
                md.push_str(&format!("- **{}**: ~{:.0} Ma\n", milestone, age));
            }
        }
        md.push_str(&format!(
            "- Peak feasibility: {:.4} (threshold for meta-cognition: {:.2})\n\n",
            self.kpg_suppressed_peak_feasibility, THRESHOLD_REFLECTIVE
        ));

        md.push_str("## Verdict\n\n");
        if !self.meta_cognition_possible_without_asteroid {
            md.push_str("**The asteroid was REQUIRED for meta-cognitive reflection.**\n\n");
            md.push_str(&format!(
                "Without Chicxulub, consciousness feasibility peaks at {:.4} — permanently\n",
                self.kpg_suppressed_peak_feasibility
            ));
            md.push_str(&format!(
                "below the {:.2} threshold for meta-cognition. The reptilian biosphere\n",
                THRESHOLD_REFLECTIVE
            ));
            md.push_str("lacks the information capacity and complexity to support recursive\n");
            md.push_str("self-reflection. Mammals never radiate. Primates never emerge.\n");
            md.push_str("The universe remains conscious of itself through no known pathway.\n");
        } else {
            md.push_str("Meta-cognition was possible without the asteroid, but delayed.\n");
        }

        if !self.self_awareness_possible_without_asteroid {
            md.push_str("\n**Not even basic self-awareness would have emerged.**\n");
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counterfactual_consciousness_runs() {
        let result = run_counterfactual_consciousness();
        assert!(result.baseline_peak_feasibility > 0.0);
        assert!(result.kpg_suppressed_peak_feasibility > 0.0);
    }

    #[test]
    fn baseline_reaches_meta_cognition() {
        let result = run_counterfactual_consciousness();
        assert!(
            result.baseline_peak_feasibility >= THRESHOLD_REFLECTIVE,
            "Baseline should reach meta-cognition threshold (peak: {:.4})",
            result.baseline_peak_feasibility
        );
    }

    #[test]
    fn suppressed_peak_lower_than_baseline() {
        let result = run_counterfactual_consciousness();
        assert!(
            result.kpg_suppressed_peak_feasibility < result.baseline_peak_feasibility,
            "Suppressed peak ({:.4}) should be < baseline ({:.4})",
            result.kpg_suppressed_peak_feasibility,
            result.baseline_peak_feasibility,
        );
    }

    #[test]
    fn reptilian_ceiling_locks_at_cretaceous() {
        let result = run_counterfactual_consciousness();
        let (cap_c, _, _) = result.reptilian_ceiling;
        assert!(
            cap_c <= 0.85,
            "Complexity cap should be ≤0.85 (Late Cretaceous), got {}",
            cap_c
        );
    }

    #[test]
    fn baseline_has_milestones() {
        let result = run_counterfactual_consciousness();
        assert!(
            !result.baseline_milestones.is_empty(),
            "Baseline should have consciousness milestones"
        );
    }

    #[test]
    fn markdown_contains_verdict() {
        let result = run_counterfactual_consciousness();
        let md = result.to_markdown();
        assert!(md.contains("Verdict"));
        assert!(md.contains("Reptilian Ceiling"));
    }
}

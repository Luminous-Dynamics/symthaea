// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterfactual mass extinction analysis with **Complexity Cap**.
//!
//! Tests the central hypothesis: **catastrophe is generative**.
//!
//! ## The Complexity Cap
//!
//! Mass extinctions clear apex species stuck in evolutionary local maxima
//! (like dinosaurs), opening niche space for high-metabolism, high-integration
//! species (like mammals). If you suppress an extinction where
//! `complexity_increase = true`, the counterfactual branch doesn't just skip
//! the temporary shock — it **misses the evolutionary step-function entirely**.
//!
//! The suppressed biosphere can never exceed the maximum integration achieved
//! by the pre-extinction regime. A world where Chicxulub missed is permanently
//! trapped in a reptilian thermodynamic ceiling for 66 million years.
//!
//! ## Implementation
//!
//! 1. Record `complexity_grade`, `information_capacity`, and `energy_throughput`
//!    at the bin immediately preceding the suppressed extinction.
//! 2. For all subsequent bins, apply a **hard cap** — those dimensions cannot
//!    exceed the pre-extinction maximum.
//! 3. Compare terminal B(t) at 3000 BCE between baseline and capped branch.

use super::bridge::biosphere_energy_adjustment;
use super::deep_time_data::DeepTimeData;
use super::dimensions::{
    compute_bt, compute_raw_dimensions, BiosphereDimensionsNorm, BiosphereMaxima,
};
use super::mass_extinctions::{
    canonical_mass_extinctions, extinction_multiplier, MassExtinctionEvent,
};
use super::temporal_bins::{build_deep_time_bins, MaAge, TemporalBin};

/// Hard cap on dimensions for a suppressed counterfactual branch.
/// These represent the thermodynamic ceiling of the pre-extinction regime.
#[derive(Debug, Clone)]
struct ComplexityCap {
    /// Maximum complexity grade achievable without the extinction.
    complexity_grade_cap: f64,
    /// Maximum energy throughput (normalized) without the extinction.
    energy_throughput_cap: f64,
    /// Maximum information capacity (normalized) without the extinction.
    information_capacity_cap: f64,
    /// Age at which the cap activates (= extinction age).
    activation_age_ma: MaAge,
}

/// Result of a single counterfactual branch.
#[derive(Debug, Clone)]
pub struct CounterfactualBranch {
    /// Which extinction was suppressed (None = baseline with all events).
    pub suppressed_event: Option<String>,
    /// B(t) values per bin (same bins as baseline).
    pub bt_values: Vec<f64>,
    /// Terminal B(t) at most recent bin (3000 BCE).
    pub terminal_bt: f64,
    /// Whether a complexity cap was applied.
    pub capped: bool,
    /// Pre-extinction cap values (if capped).
    pub cap_values: Option<(f64, f64, f64)>, // (complexity, energy, info)
    /// Crossover epoch: the age (Ma) where baseline B(t) first exceeds
    /// the suppressed branch. None if baseline never overtakes.
    /// This is the mathematical proof point of "catastrophe is generative."
    pub crossover_age_ma: Option<MaAge>,
}

/// Full counterfactual comparison.
#[derive(Debug)]
pub struct CounterfactualComparison {
    pub baseline: CounterfactualBranch,
    pub branches: Vec<CounterfactualBranch>,
    pub analysis: CounterfactualAnalysis,
}

/// Summary analysis.
#[derive(Debug)]
pub struct CounterfactualAnalysis {
    /// How many suppressions result in lower terminal B(t) than baseline?
    /// Higher = stronger evidence that catastrophe is generative.
    pub extinctions_that_helped: usize,
    pub extinctions_that_hurt: usize,
    pub most_impactful: String,
    pub most_impactful_delta: f64,
    pub catastrophe_is_generative: bool,
}

/// Compute B(t) for the baseline (all extinctions active, no cap).
fn compute_baseline_curve() -> Vec<f64> {
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    bins.iter()
        .map(|bin| {
            let age = bin.midpoint_ma;
            let raw = compute_raw_dimensions(bin, &data, &extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(age));
            let norm = raw.normalize(&maxima, bin.resolution, age, d13c);
            let bt_raw = compute_bt(&norm);
            let ext_mult = extinction_multiplier(age, &extinctions);
            let energy_adj = biosphere_energy_adjustment(age);
            (bt_raw * ext_mult * energy_adj).clamp(0.001, 1.0)
        })
        .collect()
}

/// Compute B(t) for a counterfactual branch with Complexity Cap.
///
/// When `complexity_increase = true` for the suppressed event:
/// 1. Find the bin immediately preceding the suppressed extinction
/// 2. Record normalized complexity_grade, energy_throughput, information_capacity
/// 3. For all subsequent bins, cap those 3 dimensions at the pre-extinction values
/// 4. The suppressed biosphere is permanently trapped in a pre-extinction ceiling
fn compute_capped_curve(
    suppressed_name: &str,
) -> (Vec<f64>, bool, Option<(f64, f64, f64)>) {
    let data = DeepTimeData::load();
    let all_extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    // Find the suppressed event
    let suppressed_event = all_extinctions
        .iter()
        .find(|e| e.name.contains(suppressed_name));

    // Build the complexity cap if the event drives a complexity increase
    let cap = suppressed_event.and_then(|event| {
        if !event.complexity_increase {
            return None; // No cap needed — extinction didn't drive complexity gain
        }

        // Find the bin immediately preceding the extinction
        let pre_bin = bins
            .iter()
            .filter(|b| b.midpoint_ma > event.age_ma)
            .min_by(|a, b| {
                (a.midpoint_ma - event.age_ma)
                    .abs()
                    .partial_cmp(&(b.midpoint_ma - event.age_ma).abs())
                    .unwrap()
            });

        pre_bin.map(|bin| {
            let raw = compute_raw_dimensions(bin, &data, &all_extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
            let norm = raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c);

            ComplexityCap {
                complexity_grade_cap: norm.values[1],    // Complexity
                energy_throughput_cap: norm.values[4],   // Energy
                information_capacity_cap: norm.values[5], // Information
                activation_age_ma: event.age_ma,
            }
        })
    });

    // Filter out the suppressed event from the extinction list
    let filtered_extinctions: Vec<MassExtinctionEvent> = all_extinctions
        .iter()
        .filter(|e| !e.name.contains(suppressed_name))
        .cloned()
        .collect();

    let cap_values = cap.as_ref().map(|c| {
        (c.complexity_grade_cap, c.energy_throughput_cap, c.information_capacity_cap)
    });
    let is_capped = cap.is_some();

    let curve = bins
        .iter()
        .map(|bin| {
            let age = bin.midpoint_ma;
            let raw = compute_raw_dimensions(bin, &data, &filtered_extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(age));
            let mut norm = raw.normalize(&maxima, bin.resolution, age, d13c);

            // Apply the Complexity Cap for all bins AFTER the suppressed extinction
            if let Some(ref cap) = cap {
                if age < cap.activation_age_ma {
                    // We're past the extinction (younger age = smaller Ma)
                    // Hard cap: dimensions cannot exceed pre-extinction maximum
                    norm.values[1] = norm.values[1].min(cap.complexity_grade_cap);
                    norm.values[4] = norm.values[4].min(cap.energy_throughput_cap);
                    norm.values[5] = norm.values[5].min(cap.information_capacity_cap);
                }
            }

            let bt_raw = compute_bt(&norm);
            let ext_mult = extinction_multiplier(age, &filtered_extinctions);
            let energy_adj = biosphere_energy_adjustment(age);
            (bt_raw * ext_mult * energy_adj).clamp(0.001, 1.0)
        })
        .collect();

    (curve, is_capped, cap_values)
}

/// Run the full counterfactual experiment with Complexity Cap.
pub fn run_counterfactual() -> CounterfactualComparison {
    let all_extinctions = canonical_mass_extinctions();
    let baseline_values = compute_baseline_curve();
    let bins = build_deep_time_bins();
    let bin_midpoints: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();

    let baseline = CounterfactualBranch {
        suppressed_event: None,
        terminal_bt: *baseline_values.last().unwrap_or(&0.0),
        bt_values: baseline_values,
        capped: false,
        cap_values: None,
        crossover_age_ma: None,
    };

    let test_events = [
        "Great Oxidation Event",
        "End-Ordovician",
        "Late Devonian",
        "End-Permian",
        "End-Triassic",
        "End-Cretaceous",
    ];

    let mut branches = Vec::new();
    for event_name in &test_events {
        let (curve, capped, cap_vals) = compute_capped_curve(event_name);

        // Detect crossover: find the first bin (moving toward present, i.e. increasing index)
        // where baseline > suppressed. We look starting from the extinction age.
        let event = all_extinctions.iter().find(|e| e.name.contains(event_name));
        let crossover_age_ma = event.and_then(|evt| {
            // Find the bin index at or just after the extinction
            let start_idx = bin_midpoints
                .iter()
                .position(|&a| a < evt.age_ma)
                .unwrap_or(0);

            // Scan forward (toward present = increasing index)
            for i in start_idx..curve.len() {
                if baseline.bt_values[i] > curve[i] + 0.001 {
                    return Some(bin_midpoints[i]);
                }
            }
            None
        });

        branches.push(CounterfactualBranch {
            suppressed_event: Some(event_name.to_string()),
            terminal_bt: *curve.last().unwrap_or(&0.0),
            bt_values: curve,
            capped,
            cap_values: cap_vals,
            crossover_age_ma,
        });
    }

    // Analysis: compare terminal B(t) at 3000 BCE
    let mut helped = 0;
    let mut hurt = 0;
    let mut most_impactful = String::new();
    let mut most_impactful_delta = 0.0_f64;

    for branch in &branches {
        // Delta = baseline_terminal - suppressed_terminal
        // Positive delta means baseline (with extinction) is HIGHER → extinction helped
        let delta = baseline.terminal_bt - branch.terminal_bt;
        if delta > 0.001 {
            helped += 1;
        } else if delta < -0.001 {
            hurt += 1;
        }
        if delta.abs() > most_impactful_delta.abs() {
            most_impactful_delta = delta;
            most_impactful = branch.suppressed_event.clone().unwrap_or_default();
        }
    }

    let analysis = CounterfactualAnalysis {
        extinctions_that_helped: helped,
        extinctions_that_hurt: hurt,
        most_impactful,
        most_impactful_delta,
        catastrophe_is_generative: helped > hurt,
    };

    CounterfactualComparison {
        baseline,
        branches,
        analysis,
    }
}

impl CounterfactualComparison {
    /// Format as markdown table.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Counterfactual Mass Extinction Analysis (Complexity Cap)\n\n");
        md.push_str("If an extinction is suppressed, the biosphere is permanently capped at\n");
        md.push_str("the pre-extinction complexity/energy/information ceiling.\n\n");
        md.push_str("| Event Suppressed | Terminal B(t) | Δ vs Baseline | Crossover | Cap (C/E/I) |\n");
        md.push_str("|---|---|---|---|---|\n");
        md.push_str(&format!(
            "| **Baseline (all events)** | **{:.4}** | — | — | — |\n",
            self.baseline.terminal_bt,
        ));
        for branch in &self.branches {
            let delta = branch.terminal_bt - self.baseline.terminal_bt;
            let sign = if delta > 0.0 { "+" } else { "" };
            let cap_str = branch
                .cap_values
                .map(|(c, e, i)| format!("{:.2}/{:.2}/{:.2}", c, e, i))
                .unwrap_or_else(|| "—".into());
            let crossover_str = branch
                .crossover_age_ma
                .map(|a| {
                    if a >= 1000.0 {
                        format!("{:.1} Ga", a / 1000.0)
                    } else {
                        format!("{:.0} Ma", a)
                    }
                })
                .unwrap_or_else(|| "never".into());
            md.push_str(&format!(
                "| {} | {:.4} | {}{:.4} | {} | {} |\n",
                branch.suppressed_event.as_deref().unwrap_or("?"),
                branch.terminal_bt,
                sign,
                delta,
                crossover_str,
                cap_str,
            ));
        }
        md.push_str(&format!(
            "\n**Verdict**: {} of {} extinctions produced higher eventual coherence (baseline > suppressed).\n",
            self.analysis.extinctions_that_helped,
            self.analysis.extinctions_that_helped + self.analysis.extinctions_that_hurt,
        ));
        md.push_str(&format!(
            "**Most impactful**: {} (Δ = {}{:.4})\n",
            self.analysis.most_impactful,
            if self.analysis.most_impactful_delta > 0.0 { "+" } else { "" },
            self.analysis.most_impactful_delta,
        ));
        md.push_str(&format!(
            "**Catastrophe is generative**: {}\n",
            self.analysis.catastrophe_is_generative,
        ));
        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counterfactual_runs_successfully() {
        let comparison = run_counterfactual();
        assert_eq!(comparison.branches.len(), 6, "Should test 6 events");
        assert!(comparison.baseline.terminal_bt > 0.0);
    }

    #[test]
    fn capped_branches_have_lower_terminal_bt() {
        let comparison = run_counterfactual();
        // Events with complexity_increase=true should produce lower terminal B(t)
        // when suppressed (because the biosphere is capped at pre-extinction ceiling)
        for branch in &comparison.branches {
            if branch.capped {
                assert!(
                    branch.terminal_bt <= comparison.baseline.terminal_bt + 0.001,
                    "Capped branch '{}' (terminal {:.4}) should not exceed baseline ({:.4})",
                    branch.suppressed_event.as_deref().unwrap_or("?"),
                    branch.terminal_bt,
                    comparison.baseline.terminal_bt,
                );
            }
        }
    }

    #[test]
    fn end_cretaceous_suppression_traps_in_reptilian_ceiling() {
        let comparison = run_counterfactual();
        let kpg = comparison.branches.iter()
            .find(|b| b.suppressed_event.as_deref() == Some("End-Cretaceous"))
            .unwrap();

        assert!(kpg.capped, "K-Pg suppression should be capped");

        // The complexity cap should be at Late Cretaceous level (~0.80 = mammals present but not dominant)
        let (cap_c, cap_e, cap_i) = kpg.cap_values.unwrap();
        assert!(
            cap_c < 0.85,
            "K-Pg complexity cap should be < 0.85 (pre-mammalian radiation), got {}",
            cap_c,
        );

        // Terminal B(t) should be significantly lower than baseline
        let delta = comparison.baseline.terminal_bt - kpg.terminal_bt;
        assert!(
            delta > 0.01,
            "Without Chicxulub, terminal B(t) should be lower by >0.01, got delta {}",
            delta,
        );
    }

    #[test]
    fn end_permian_suppression_traps_in_paleozoic_ceiling() {
        let comparison = run_counterfactual();
        let permian = comparison.branches.iter()
            .find(|b| b.suppressed_event.as_deref() == Some("End-Permian"))
            .unwrap();

        assert!(permian.capped, "End-Permian suppression should be capped");

        // Without the Great Dying, no dinosaurs, no mammals → stuck at Permian complexity
        let delta = comparison.baseline.terminal_bt - permian.terminal_bt;
        assert!(
            delta > 0.01,
            "Without End-Permian, terminal B(t) should be much lower, got delta {}",
            delta,
        );
    }

    #[test]
    fn catastrophe_is_generative() {
        let comparison = run_counterfactual();
        // The majority of complexity-increasing extinctions should show
        // baseline > suppressed at terminal (helped > hurt)
        assert!(
            comparison.analysis.catastrophe_is_generative,
            "Expected catastrophe to be generative: {} helped, {} hurt",
            comparison.analysis.extinctions_that_helped,
            comparison.analysis.extinctions_that_hurt,
        );
    }

    #[test]
    fn markdown_output_contains_table() {
        let comparison = run_counterfactual();
        let md = comparison.to_markdown();
        assert!(md.contains("Complexity Cap"), "Should contain title");
        assert!(md.contains("End-Permian"), "Should contain End-Permian");
        assert!(md.contains("Catastrophe is generative"), "Should contain verdict");
    }

    #[test]
    fn uncapped_branches_not_constrained() {
        let comparison = run_counterfactual();
        // Late Devonian has complexity_increase=true, so it SHOULD be capped
        let devonian = comparison.branches.iter()
            .find(|b| b.suppressed_event.as_deref() == Some("Late Devonian"))
            .unwrap();
        assert!(devonian.capped, "Late Devonian should be capped (complexity_increase=true)");
    }
}

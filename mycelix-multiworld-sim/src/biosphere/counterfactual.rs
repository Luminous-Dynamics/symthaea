// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterfactual mass extinction analysis with **multiple cap regimes**.
//!
//! Tests the hypothesis: catastrophe may be generative. The strength of
//! this claim depends critically on the counterfactual assumption about
//! what happens to the suppressed branch post-extinction. We implement
//! four regimes to honestly assess how sensitive the result is:
//!
//! - **HardCap**: Permanent lock at pre-extinction ceiling (strongest assumption)
//! - **SoftCap**: Slow leakage — suppressed branch gains complexity at a fraction
//!   of the baseline rate (models troodontid/cephalopod innovation pathways)
//! - **DelayedEscape**: Hard cap holds for N Ma, then lifts entirely
//! - **NoCap**: Just suppress the extinction multiplier (naive approach)
//!
//! The verdict's robustness is measured by how many regimes preserve it.

use super::bridge::biosphere_energy_adjustment;
use super::deep_time_data::DeepTimeData;
use super::dimensions::{compute_bt, compute_raw_dimensions, BiosphereMaxima};
use super::mass_extinctions::{
    canonical_mass_extinctions, extinction_multiplier, MassExtinctionEvent,
};
use super::temporal_bins::{build_deep_time_bins, MaAge};

/// How the suppressed branch's dimensions evolve post-extinction.
/// This is the most contestable assumption in the framework.
#[derive(Debug, Clone)]
pub enum CapRegime {
    /// Permanent lock at pre-extinction ceiling.
    /// Strongest assumption: without extinction clearing apex taxa,
    /// no pathway exists to higher integration. Encodes the conclusion
    /// most strongly — must be tested against weaker regimes.
    HardCap,
    /// Slow leakage: suppressed branch gains complexity at `leakage_rate`
    /// fraction of the empirical rate per Ma. Models alternative innovation
    /// pathways (troodontid intelligence, cephalopod complexity).
    /// leakage_rate = 0.10 means 10% of baseline innovation speed.
    SoftCap { leakage_rate: f64 },
    /// Hard cap holds for `hold_ma` million years, then lifts entirely.
    /// Models delayed but eventual innovation without extinction.
    DelayedEscape { hold_ma: f64 },
    /// No cap at all — just suppress the extinction multiplier.
    /// The naive approach. Shows what happens without any counterfactual
    /// modeling of evolutionary lock-in.
    NoCap,
}

impl std::fmt::Display for CapRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HardCap => write!(f, "HardCap"),
            Self::SoftCap { leakage_rate } => write!(f, "SoftCap({:.0}%)", leakage_rate * 100.0),
            Self::DelayedEscape { hold_ma } => write!(f, "DelayedEscape({:.0}Ma)", hold_ma),
            Self::NoCap => write!(f, "NoCap"),
        }
    }
}

/// Pre-extinction dimension ceiling values.
#[derive(Debug, Clone)]
struct DimensionCeiling {
    complexity: f64,
    energy: f64,
    information: f64,
    activation_age_ma: MaAge,
}

/// Result of a single counterfactual branch under a specific regime.
#[derive(Debug, Clone)]
pub struct CounterfactualBranch {
    pub suppressed_event: Option<String>,
    pub regime: String,
    pub bt_values: Vec<f64>,
    pub terminal_bt: f64,
    pub cap_values: Option<(f64, f64, f64)>,
    /// Crossover epoch where baseline overtakes this branch.
    pub crossover_age_ma: Option<MaAge>,
}

/// Full comparison across events and regimes.
#[derive(Debug)]
pub struct CounterfactualComparison {
    pub baseline: CounterfactualBranch,
    pub branches: Vec<CounterfactualBranch>,
    pub analysis: CounterfactualAnalysis,
}

#[derive(Debug)]
pub struct CounterfactualAnalysis {
    pub extinctions_that_helped: usize,
    pub extinctions_that_hurt: usize,
    pub most_impactful: String,
    pub most_impactful_delta: f64,
    pub catastrophe_is_generative: bool,
}

/// Result of running all four regimes on a single event.
#[derive(Debug)]
pub struct MultiRegimeResult {
    pub event_name: String,
    pub baseline_terminal: f64,
    pub hard_cap_terminal: f64,
    pub soft_cap_terminal: f64,
    pub delayed_escape_terminal: f64,
    pub no_cap_terminal: f64,
    /// How many of the 4 regimes show baseline > suppressed.
    pub regimes_supporting_generative: usize,
}

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

/// Compute B(t) for a suppressed branch under a given cap regime.
fn compute_regime_curve(
    suppressed_name: &str,
    regime: &CapRegime,
) -> (Vec<f64>, Option<(f64, f64, f64)>) {
    let data = DeepTimeData::load();
    let all_extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    let suppressed_event = all_extinctions
        .iter()
        .find(|e| e.name.contains(suppressed_name));

    // Find pre-extinction ceiling (needed for all cap regimes except NoCap)
    let ceiling = suppressed_event.and_then(|event| {
        if !event.complexity_increase {
            return None;
        }
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
            DimensionCeiling {
                complexity: norm.values[1],
                energy: norm.values[4],
                information: norm.values[5],
                activation_age_ma: event.age_ma,
            }
        })
    });

    let cap_values = ceiling
        .as_ref()
        .map(|c| (c.complexity, c.energy, c.information));

    let filtered: Vec<MassExtinctionEvent> = all_extinctions
        .iter()
        .filter(|e| !e.name.contains(suppressed_name))
        .cloned()
        .collect();

    let curve = bins
        .iter()
        .map(|bin| {
            let age = bin.midpoint_ma;
            let raw = compute_raw_dimensions(bin, &data, &filtered);
            let d13c = Some(data.interpolate_d13c_excursion(age));
            let mut norm = raw.normalize(&maxima, bin.resolution, age, d13c);

            // Apply cap regime for post-extinction bins
            if let Some(ref ceil) = ceiling {
                if age < ceil.activation_age_ma {
                    let time_since = ceil.activation_age_ma - age; // Ma since extinction

                    match regime {
                        CapRegime::HardCap => {
                            norm.values[1] = norm.values[1].min(ceil.complexity);
                            norm.values[4] = norm.values[4].min(ceil.energy);
                            norm.values[5] = norm.values[5].min(ceil.information);
                        }
                        CapRegime::SoftCap { leakage_rate } => {
                            // Cap grows slowly: cap(t) = ceil + leakage × (empirical - ceil) × time
                            // This lets the suppressed branch slowly approach empirical values
                            let leak_factor = (leakage_rate * time_since / 100.0).min(1.0);
                            let eff_cap_c = ceil.complexity
                                + leak_factor * (norm.values[1] - ceil.complexity).max(0.0);
                            let eff_cap_e =
                                ceil.energy + leak_factor * (norm.values[4] - ceil.energy).max(0.0);
                            let eff_cap_i = ceil.information
                                + leak_factor * (norm.values[5] - ceil.information).max(0.0);
                            norm.values[1] = norm.values[1].min(eff_cap_c);
                            norm.values[4] = norm.values[4].min(eff_cap_e);
                            norm.values[5] = norm.values[5].min(eff_cap_i);
                        }
                        CapRegime::DelayedEscape { hold_ma } => {
                            if time_since < *hold_ma {
                                // Hard cap during hold period
                                norm.values[1] = norm.values[1].min(ceil.complexity);
                                norm.values[4] = norm.values[4].min(ceil.energy);
                                norm.values[5] = norm.values[5].min(ceil.information);
                            }
                            // After hold_ma: cap lifts, empirical values used directly
                        }
                        CapRegime::NoCap => {
                            // No cap — empirical values pass through unchanged
                        }
                    }
                }
            }

            let bt_raw = compute_bt(&norm);
            let ext_mult = extinction_multiplier(age, &filtered);
            let energy_adj = biosphere_energy_adjustment(age);
            (bt_raw * ext_mult * energy_adj).clamp(0.001, 1.0)
        })
        .collect();

    (curve, cap_values)
}

/// Run the standard counterfactual with HardCap (for backward compatibility).
pub fn run_counterfactual() -> CounterfactualComparison {
    run_counterfactual_with_regime(&CapRegime::HardCap)
}

/// Run counterfactual with a specific cap regime.
pub fn run_counterfactual_with_regime(regime: &CapRegime) -> CounterfactualComparison {
    let all_extinctions = canonical_mass_extinctions();
    let baseline_values = compute_baseline_curve();
    let bins = build_deep_time_bins();
    let bin_midpoints: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();

    let baseline = CounterfactualBranch {
        suppressed_event: None,
        regime: "baseline".into(),
        terminal_bt: *baseline_values.last().unwrap_or(&0.0),
        bt_values: baseline_values,
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
        let (curve, cap_vals) = compute_regime_curve(event_name, regime);

        let event = all_extinctions.iter().find(|e| e.name.contains(event_name));
        let crossover_age_ma = event.and_then(|evt| {
            let start_idx = bin_midpoints
                .iter()
                .position(|&a| a < evt.age_ma)
                .unwrap_or(0);
            for i in start_idx..curve.len() {
                if baseline.bt_values[i] > curve[i] + 0.001 {
                    return Some(bin_midpoints[i]);
                }
            }
            None
        });

        branches.push(CounterfactualBranch {
            suppressed_event: Some(event_name.to_string()),
            regime: format!("{}", regime),
            terminal_bt: *curve.last().unwrap_or(&0.0),
            bt_values: curve,
            cap_values: cap_vals,
            crossover_age_ma,
        });
    }

    let mut helped = 0;
    let mut hurt = 0;
    let mut most_impactful = String::new();
    let mut most_impactful_delta = 0.0_f64;

    for branch in &branches {
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

    CounterfactualComparison {
        baseline,
        branches,
        analysis: CounterfactualAnalysis {
            extinctions_that_helped: helped,
            extinctions_that_hurt: hurt,
            most_impactful,
            most_impactful_delta,
            catastrophe_is_generative: helped > hurt,
        },
    }
}

/// Run all four regimes on the K-Pg event and report results.
pub fn run_multi_regime_kpg() -> Vec<MultiRegimeResult> {
    let baseline_values = compute_baseline_curve();
    let baseline_terminal = *baseline_values.last().unwrap_or(&0.0);

    let events = ["End-Cretaceous", "End-Permian", "Great Oxidation Event"];
    let regimes = [
        CapRegime::HardCap,
        CapRegime::SoftCap { leakage_rate: 0.10 },
        CapRegime::DelayedEscape { hold_ma: 30.0 },
        CapRegime::NoCap,
    ];

    events
        .iter()
        .map(|event_name| {
            let terminals: Vec<f64> = regimes
                .iter()
                .map(|regime| {
                    let (curve, _) = compute_regime_curve(event_name, regime);
                    *curve.last().unwrap_or(&0.0)
                })
                .collect();

            let supporting = terminals
                .iter()
                .filter(|&&t| baseline_terminal > t + 0.001)
                .count();

            MultiRegimeResult {
                event_name: event_name.to_string(),
                baseline_terminal,
                hard_cap_terminal: terminals[0],
                soft_cap_terminal: terminals[1],
                delayed_escape_terminal: terminals[2],
                no_cap_terminal: terminals[3],
                regimes_supporting_generative: supporting,
            }
        })
        .collect()
}

impl CounterfactualComparison {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        let regime = self
            .branches
            .first()
            .map(|b| b.regime.as_str())
            .unwrap_or("?");
        md.push_str(&format!("# Counterfactual Analysis ({})\n\n", regime));
        md.push_str("| Event | Terminal B(t) | Δ | Crossover | Cap (C/E/I) |\n");
        md.push_str("|---|---|---|---|---|\n");
        md.push_str(&format!(
            "| **Baseline** | **{:.4}** | — | — | — |\n",
            self.baseline.terminal_bt
        ));
        for branch in &self.branches {
            let delta = branch.terminal_bt - self.baseline.terminal_bt;
            let sign = if delta > 0.0 { "+" } else { "" };
            let cap_str = branch
                .cap_values
                .map(|(c, e, i)| format!("{:.2}/{:.2}/{:.2}", c, e, i))
                .unwrap_or_else(|| "—".into());
            let cross_str = branch
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
                cross_str,
                cap_str,
            ));
        }
        md.push_str(&format!(
            "\n**Generative**: {}/{} events ({})\n",
            self.analysis.extinctions_that_helped,
            self.analysis.extinctions_that_helped + self.analysis.extinctions_that_hurt,
            regime,
        ));
        md
    }
}

/// Format multi-regime results as markdown.
pub fn multi_regime_markdown(results: &[MultiRegimeResult]) -> String {
    let mut md = String::new();
    md.push_str("# Multi-Regime Counterfactual Comparison\n\n");
    md.push_str("How robust is \"catastrophe is generative\" across different assumptions?\n\n");
    md.push_str("| Event | Baseline | HardCap | SoftCap(10%) | Escape(30Ma) | NoCap | Regimes Supporting |\n");
    md.push_str("|---|---|---|---|---|---|---|\n");
    for r in results {
        md.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {}/4 |\n",
            r.event_name,
            r.baseline_terminal,
            r.hard_cap_terminal,
            r.soft_cap_terminal,
            r.delayed_escape_terminal,
            r.no_cap_terminal,
            r.regimes_supporting_generative,
        ));
    }
    md.push_str("\n**Interpretation**:\n");
    md.push_str("- 4/4: Verdict robust across all assumptions\n");
    md.push_str("- 3/4: Robust under most assumptions\n");
    md.push_str("- 2/4: Depends on counterfactual model\n");
    md.push_str("- 1/4: Only holds under strongest assumption (HardCap)\n");
    md.push_str("- 0/4: Extinction was NOT generative\n");
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_cap_runs() {
        let comparison = run_counterfactual_with_regime(&CapRegime::HardCap);
        assert_eq!(comparison.branches.len(), 6);
        assert!(comparison.baseline.terminal_bt > 0.0);
    }

    #[test]
    fn soft_cap_runs() {
        let comparison = run_counterfactual_with_regime(&CapRegime::SoftCap { leakage_rate: 0.10 });
        assert_eq!(comparison.branches.len(), 6);
    }

    #[test]
    fn delayed_escape_runs() {
        let comparison =
            run_counterfactual_with_regime(&CapRegime::DelayedEscape { hold_ma: 30.0 });
        assert_eq!(comparison.branches.len(), 6);
    }

    #[test]
    fn no_cap_runs() {
        let comparison = run_counterfactual_with_regime(&CapRegime::NoCap);
        assert_eq!(comparison.branches.len(), 6);
    }

    #[test]
    fn hard_cap_more_restrictive_than_soft() {
        let hard = run_counterfactual_with_regime(&CapRegime::HardCap);
        let soft = run_counterfactual_with_regime(&CapRegime::SoftCap { leakage_rate: 0.10 });
        // Under soft cap, suppressed branches should have equal or higher terminal B(t)
        for (h, s) in hard.branches.iter().zip(soft.branches.iter()) {
            assert!(
                s.terminal_bt >= h.terminal_bt - 0.001,
                "{}: soft ({:.4}) should >= hard ({:.4})",
                h.suppressed_event.as_deref().unwrap_or("?"),
                s.terminal_bt,
                h.terminal_bt,
            );
        }
    }

    #[test]
    fn no_cap_highest_terminal() {
        let hard = run_counterfactual_with_regime(&CapRegime::HardCap);
        let none = run_counterfactual_with_regime(&CapRegime::NoCap);
        for (h, n) in hard.branches.iter().zip(none.branches.iter()) {
            assert!(
                n.terminal_bt >= h.terminal_bt - 0.001,
                "{}: no-cap ({:.4}) should >= hard ({:.4})",
                h.suppressed_event.as_deref().unwrap_or("?"),
                n.terminal_bt,
                h.terminal_bt,
            );
        }
    }

    #[test]
    fn multi_regime_kpg_runs() {
        let results = run_multi_regime_kpg();
        assert_eq!(results.len(), 3); // K-Pg, Permian, GOE
        for r in &results {
            assert!(r.baseline_terminal > 0.0);
        }
    }

    #[test]
    fn regime_ordering_correct() {
        // HardCap ≤ SoftCap ≤ DelayedEscape ≤ NoCap (for terminal B(t))
        let results = run_multi_regime_kpg();
        for r in &results {
            assert!(
                r.hard_cap_terminal <= r.soft_cap_terminal + 0.01,
                "{}: hard ({:.4}) > soft ({:.4})",
                r.event_name,
                r.hard_cap_terminal,
                r.soft_cap_terminal,
            );
            assert!(
                r.soft_cap_terminal <= r.no_cap_terminal + 0.01,
                "{}: soft ({:.4}) > no-cap ({:.4})",
                r.event_name,
                r.soft_cap_terminal,
                r.no_cap_terminal,
            );
        }
    }

    #[test]
    fn markdown_contains_regime() {
        let comparison = run_counterfactual_with_regime(&CapRegime::SoftCap { leakage_rate: 0.10 });
        let md = comparison.to_markdown();
        assert!(md.contains("SoftCap"));
    }
}

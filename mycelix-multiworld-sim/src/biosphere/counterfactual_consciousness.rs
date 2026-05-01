// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterfactual consciousness analysis across multiple cap regimes.
//!
//! Asks: under which assumptions about evolutionary lock-in was the
//! Chicxulub asteroid necessary for meta-cognitive consciousness?

use super::consciousness_emergence::{
    compute_emergence_curve, ConsciousnessEmergencePoint, THRESHOLD_AWARE, THRESHOLD_REFLECTIVE,
};
use super::counterfactual::CapRegime;
use super::deep_time_data::DeepTimeData;
use super::dimensions::{compute_raw_dimensions, BiosphereMaxima};
use super::mass_extinctions::{canonical_mass_extinctions, MassExtinctionEvent};
use super::temporal_bins::{build_deep_time_bins, MaAge};

/// Result for a single cap regime.
#[derive(Debug)]
pub struct RegimeConsciousnessResult {
    pub regime: String,
    pub peak_feasibility: f64,
    pub meta_cognition_reached: bool,
    pub self_awareness_reached: bool,
    pub milestones: Vec<(String, MaAge)>,
}

/// Full multi-regime consciousness analysis.
#[derive(Debug)]
pub struct CounterfactualConsciousnessResult {
    /// Baseline (all extinctions active).
    pub baseline_milestones: Vec<(String, MaAge)>,
    pub baseline_peak_feasibility: f64,
    /// Results per cap regime.
    pub regime_results: Vec<RegimeConsciousnessResult>,
    /// The complexity cap values (from pre-K-Pg bin).
    pub reptilian_ceiling: (f64, f64, f64),
    /// How many of 4 regimes require the asteroid for meta-cognition.
    pub regimes_requiring_asteroid: usize,
}

/// Compute K-Pg suppressed dimensions under a given cap regime.
fn compute_kpg_suppressed_dims(
    regime: &CapRegime,
) -> (
    Vec<MaAge>,
    Vec<super::dimensions::BiosphereDimensionsNorm>,
    (f64, f64, f64),
) {
    let data = DeepTimeData::load();
    let all_extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    let kpg = all_extinctions
        .iter()
        .find(|e| e.name.contains("End-Cretaceous"))
        .unwrap();

    let filtered: Vec<MassExtinctionEvent> = all_extinctions
        .iter()
        .filter(|e| !e.name.contains("End-Cretaceous"))
        .cloned()
        .collect();

    let pre_kpg_bin = bins
        .iter()
        .filter(|b| b.midpoint_ma > kpg.age_ma)
        .min_by(|a, b| {
            (a.midpoint_ma - kpg.age_ma)
                .abs()
                .partial_cmp(&(b.midpoint_ma - kpg.age_ma).abs())
                .unwrap()
        })
        .unwrap();

    let pre_raw = compute_raw_dimensions(pre_kpg_bin, &data, &all_extinctions);
    let pre_d13c = Some(data.interpolate_d13c_excursion(pre_kpg_bin.midpoint_ma));
    let pre_norm = pre_raw.normalize(
        &maxima,
        pre_kpg_bin.resolution,
        pre_kpg_bin.midpoint_ma,
        pre_d13c,
    );

    let ceil_c = pre_norm.values[1];
    let ceil_e = pre_norm.values[4];
    let ceil_i = pre_norm.values[5];

    let ages: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();
    let dims: Vec<_> = bins
        .iter()
        .map(|bin| {
            let raw = compute_raw_dimensions(bin, &data, &filtered);
            let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
            let mut norm = raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c);

            if bin.midpoint_ma < kpg.age_ma {
                let time_since = kpg.age_ma - bin.midpoint_ma;

                match regime {
                    CapRegime::HardCap => {
                        norm.values[1] = norm.values[1].min(ceil_c);
                        norm.values[4] = norm.values[4].min(ceil_e);
                        norm.values[5] = norm.values[5].min(ceil_i);
                    }
                    CapRegime::SoftCap { leakage_rate } => {
                        let leak = (leakage_rate * time_since / 100.0).min(1.0);
                        norm.values[1] =
                            norm.values[1].min(ceil_c + leak * (norm.values[1] - ceil_c).max(0.0));
                        norm.values[4] =
                            norm.values[4].min(ceil_e + leak * (norm.values[4] - ceil_e).max(0.0));
                        norm.values[5] =
                            norm.values[5].min(ceil_i + leak * (norm.values[5] - ceil_i).max(0.0));
                    }
                    CapRegime::DelayedEscape { hold_ma } => {
                        if time_since < *hold_ma {
                            norm.values[1] = norm.values[1].min(ceil_c);
                            norm.values[4] = norm.values[4].min(ceil_e);
                            norm.values[5] = norm.values[5].min(ceil_i);
                        }
                    }
                    CapRegime::NoCap => {}
                }
            }

            norm
        })
        .collect();

    (ages, dims, (ceil_c, ceil_e, ceil_i))
}

/// Run the full multi-regime counterfactual consciousness analysis.
pub fn run_counterfactual_consciousness() -> CounterfactualConsciousnessResult {
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    // Baseline
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
    let baseline_milestones = extract_milestones(&baseline_curve);
    let baseline_peak = baseline_curve
        .iter()
        .map(|p| p.feasibility)
        .fold(0.0_f64, f64::max);

    // Run all 4 regimes
    let regimes = [
        CapRegime::HardCap,
        CapRegime::SoftCap { leakage_rate: 0.10 },
        CapRegime::DelayedEscape { hold_ma: 30.0 },
        CapRegime::NoCap,
    ];

    let mut ceiling = (0.0, 0.0, 0.0);
    let mut regime_results = Vec::new();

    for regime in &regimes {
        let (ages, dims, ceil) = compute_kpg_suppressed_dims(regime);
        ceiling = ceil;

        let curve = compute_emergence_curve(&ages, &dims);
        let peak = curve.iter().map(|p| p.feasibility).fold(0.0_f64, f64::max);
        let milestones = extract_milestones(&curve);

        regime_results.push(RegimeConsciousnessResult {
            regime: format!("{}", regime),
            peak_feasibility: peak,
            meta_cognition_reached: peak >= THRESHOLD_REFLECTIVE,
            self_awareness_reached: peak >= THRESHOLD_AWARE,
            milestones,
        });
    }

    let requiring = regime_results
        .iter()
        .filter(|r| !r.meta_cognition_reached)
        .count();

    CounterfactualConsciousnessResult {
        baseline_milestones,
        baseline_peak_feasibility: baseline_peak,
        regime_results,
        reptilian_ceiling: ceiling,
        regimes_requiring_asteroid: requiring,
    }
}

fn extract_milestones(curve: &[ConsciousnessEmergencePoint]) -> Vec<(String, MaAge)> {
    curve
        .iter()
        .filter_map(|pt| pt.milestone.as_ref().map(|m| (m.clone(), pt.age_ma)))
        .collect()
}

impl CounterfactualConsciousnessResult {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Was the Asteroid Required for Consciousness?\n\n");
        md.push_str("## Baseline (All Extinctions Active)\n\n");
        for (m, age) in &self.baseline_milestones {
            md.push_str(&format!("- **{}**: ~{:.0} Ma\n", m, age));
        }
        md.push_str(&format!(
            "- Peak feasibility: {:.4}\n\n",
            self.baseline_peak_feasibility
        ));

        md.push_str(&format!(
            "## K-Pg Suppressed (Ceiling: C={:.2}, E={:.2}, I={:.2})\n\n",
            self.reptilian_ceiling.0, self.reptilian_ceiling.1, self.reptilian_ceiling.2
        ));

        md.push_str("| Regime | Peak Feasibility | Meta-Cognition? | Self-Awareness? |\n");
        md.push_str("|---|---|---|---|\n");
        for r in &self.regime_results {
            md.push_str(&format!(
                "| {} | {:.4} | {} | {} |\n",
                r.regime,
                r.peak_feasibility,
                if r.meta_cognition_reached {
                    "YES"
                } else {
                    "NO"
                },
                if r.self_awareness_reached {
                    "YES"
                } else {
                    "NO"
                },
            ));
        }

        md.push_str(&format!(
            "\n**Regimes requiring asteroid**: {}/4\n\n",
            self.regimes_requiring_asteroid
        ));

        md.push_str("## Verdict\n\n");
        let verdict = match self.regimes_requiring_asteroid {
            4 => "Asteroid required under ALL assumptions. Robust result.",
            3 => "Asteroid required under 3/4 assumptions. Strong but not universal.",
            2 => "Asteroid required under 2/4 assumptions. Depends on model.",
            1 => "Asteroid required only under strongest assumption (HardCap). Fragile.",
            0 => "Asteroid NOT required under any assumption. Meta-cognition was inevitable.",
            _ => "Error in regime count.",
        };
        md.push_str(&format!("**{}**\n\n", verdict));

        if self.regimes_requiring_asteroid > 0 && self.regimes_requiring_asteroid < 4 {
            md.push_str("The result is model-dependent: the answer to \"was the asteroid\n");
            md.push_str("necessary?\" depends on whether evolution can innovate past an\n");
            md.push_str("incumbent apex regime without catastrophic clearing. This is itself\n");
            md.push_str("an open question in paleobiology.\n");
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_regime_analysis_runs() {
        let result = run_counterfactual_consciousness();
        assert_eq!(result.regime_results.len(), 4);
        assert!(result.baseline_peak_feasibility > 0.0);
    }

    #[test]
    fn baseline_reaches_meta_cognition() {
        let result = run_counterfactual_consciousness();
        assert!(result.baseline_peak_feasibility >= THRESHOLD_REFLECTIVE);
    }

    #[test]
    fn hard_cap_most_restrictive() {
        let result = run_counterfactual_consciousness();
        let hard = result
            .regime_results
            .iter()
            .find(|r| r.regime.contains("HardCap"))
            .unwrap();
        let no_cap = result
            .regime_results
            .iter()
            .find(|r| r.regime.contains("NoCap"))
            .unwrap();
        assert!(hard.peak_feasibility <= no_cap.peak_feasibility + 0.001);
    }

    #[test]
    fn regime_results_ordered() {
        let result = run_counterfactual_consciousness();
        // HardCap ≤ SoftCap ≤ NoCap for peak feasibility
        let peaks: Vec<f64> = result
            .regime_results
            .iter()
            .map(|r| r.peak_feasibility)
            .collect();
        assert!(
            peaks[0] <= peaks[1] + 0.01,
            "HardCap ({}) > SoftCap ({})",
            peaks[0],
            peaks[1]
        );
        assert!(
            peaks[1] <= peaks[3] + 0.01,
            "SoftCap ({}) > NoCap ({})",
            peaks[1],
            peaks[3]
        );
    }

    #[test]
    fn markdown_contains_regime_table() {
        let result = run_counterfactual_consciousness();
        let md = result.to_markdown();
        assert!(md.contains("HardCap"));
        assert!(md.contains("SoftCap"));
        assert!(md.contains("NoCap"));
        assert!(md.contains("Verdict"));
    }
}

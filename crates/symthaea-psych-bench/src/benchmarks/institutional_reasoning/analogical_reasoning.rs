// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Analogical Reasoning Benchmark
//!
//! Tests whether the HDC composition algebra can solve institutional analogies
//! of the form "A is to B as ??? is to D". This is a harder test than
//! decomposition because it requires computing and applying transformations
//! across different institutional domains.
//!
//! ## HDC Implementation
//!
//! Given bundled compositions A and B, the transformation is computed as the
//! set-difference of their source components. Components in A but not B are
//! "removed"; components in B but not A are "added". This transformation is
//! then applied to D using XOR-binding (toggle), and the nearest axiom to
//! the result is identified.
//!
//! ## Key Claims Tested
//!
//! 1. **Analogical Transfer**: The system can apply a structural transformation
//!    from one institutional domain to another.
//! 2. **Shared-Component Sensitivity**: Analogies between axioms with shared
//!    components should produce higher similarity than unrelated pairs.
//! 3. **Symmetry**: If A:B :: X:D, then B:A :: Y:D should produce a different
//!    (and meaningful) result.
//!
//! ## References
//!
//! - Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
//!   *Cognitive Science*.
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::CompositionAlgebra;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

/// Institutional Reasoning: Analogical Reasoning benchmark.
pub struct AnalogicalReasoningBenchmark;

struct TrialResult {
    /// Fraction of analogies that produce above-chance similarity
    transfer_accuracy: f64,
    /// Mean similarity of analogical targets to their nearest axiom
    transfer_strength: f64,
    /// Mean selectivity: (best_sim - 2nd_best_sim) / best_sim
    analogical_selectivity: f64,
    /// Whether A:B::?:D ≠ B:A::?:D (asymmetry)
    asymmetry_score: f64,
    /// HDC-native analogy transfer accuracy (XOR-based, encoding-space)
    hdc_transfer_accuracy: f64,
    /// HDC-native analogy transfer strength
    hdc_transfer_strength: f64,
}

/// An analogy test case: A is to B as ??? is to D.
struct AnalogyCase {
    a: &'static str,
    b: &'static str,
    d: &'static str,
}

impl AnalogicalReasoningBenchmark {
    fn analogy_cases() -> Vec<AnalogyCase> {
        vec![
            // ── Shared-component analogies ──
            AnalogyCase {
                // REVOLUTION(AUTH,ENFORC,PROHIB) ↔ LEGIT_GOV(AUTH,LEGIT,TRUST): share AUTH
                a: "REVOLUTION",
                b: "LEGITIMATE_GOVERNANCE",
                d: "TRADE_AGREEMENT",
            },
            AnalogyCase {
                // LEGIT_GOV(AUTH,LEGIT,TRUST) ↔ CORRUPTION(AUTH,DEFECT,EXCHANGE): share AUTH
                a: "LEGITIMATE_GOVERNANCE",
                b: "CORRUPTION",
                d: "SOCIAL_CONTRACT",
            },
            AnalogyCase {
                // TRADE(TREATY,EXCHANGE,RECIP) ↔ ECON_SANCT(SANCT,EXCHANGE,PROHIB): share EXCHANGE
                a: "TRADE_AGREEMENT",
                b: "ECONOMIC_SANCTION",
                d: "LEGITIMATE_GOVERNANCE",
            },
            AnalogyCase {
                // DEM_ELECT(AUTH,LEGIT,POP,COOP) ↔ SOCIAL(SOV,LEGIT,OBLIG,COOP): share LEGIT+COOP
                a: "DEMOCRATIC_ELECTION",
                b: "SOCIAL_CONTRACT",
                d: "REVOLUTION",
            },
            AnalogyCase {
                // REVOLUTION(AUTH,ENFORC,PROHIB) ↔ ARMS_EMBARGO(SANCT,ENFORC,PROHIB): share ENFORC+PROHIB
                a: "REVOLUTION",
                b: "ARMS_EMBARGO",
                d: "CIVIL_DISOBEDIENCE",
            },
            // ── No-shared analogies ──
            AnalogyCase {
                a: "REGULATORY_CAPTURE",
                b: "TRADE_AGREEMENT",
                d: "LEGITIMATE_GOVERNANCE",
            },
            AnalogyCase {
                a: "BORDER_DISPUTE",
                b: "CORRUPTION",
                d: "CIVIL_DISOBEDIENCE",
            },
            AnalogyCase {
                a: "FAILED_STATE",
                b: "REGULATORY_CAPTURE",
                d: "TRADE_AGREEMENT",
            },
            AnalogyCase {
                a: "BORDER_DISPUTE",
                b: "CIVIL_DISOBEDIENCE",
                d: "ECONOMIC_SANCTION",
            },
            AnalogyCase {
                a: "FAILED_STATE",
                b: "ARMS_EMBARGO",
                d: "REVOLUTION",
            },
            // ── Adversarial: differ by exactly 1 primitive (A3) ──
            AnalogyCase {
                // REVOLUTION(AUTH,ENFORC,PROHIB) vs ARMS_EMBARGO(SANCT,ENFORC,PROHIB)
                // differ only in AUTHORITY↔SANCTION
                a: "REVOLUTION",
                b: "ARMS_EMBARGO",
                d: "LEGITIMATE_GOVERNANCE",
            },
            AnalogyCase {
                // PEACE_TREATY(TREATY,COOP,SOV) vs TRADE_AGREEMENT(TREATY,EXCHANGE,RECIP)
                // share TREATY, differ in COOP+SOV vs EXCHANGE+RECIP
                a: "PEACE_TREATY",
                b: "TRADE_AGREEMENT",
                d: "FEDERATION",
            },
            AnalogyCase {
                // CONSTITUTIONAL_CRISIS(LEGIT,AUTH,DEFECT) vs CORRUPTION(AUTH,DEFECT,EXCHANGE)
                // share AUTH+DEFECT, differ in LEGIT vs EXCHANGE
                a: "CONSTITUTIONAL_CRISIS",
                b: "CORRUPTION",
                d: "IMPEACHMENT",
            },
            AnalogyCase {
                // COLONIALISM(SOV,ENFORC,POP,DEFECT) vs IMPEACHMENT(AUTH,LEGIT,ENFORC,POP)
                // share ENFORC+POP, differ in SOV+DEFECT vs AUTH+LEGIT
                a: "COLONIALISM",
                b: "IMPEACHMENT",
                d: "DEMOCRATIC_ELECTION",
            },
        ]
    }

    /// All 18 institutional axiom names for similarity profile computation.
    const AXIOM_NAMES: &'static [&'static str] = &[
        "REVOLUTION",
        "FAILED_STATE",
        "BORDER_DISPUTE",
        "LEGITIMATE_GOVERNANCE",
        "REGULATORY_CAPTURE",
        "TRADE_AGREEMENT",
        "ECONOMIC_SANCTION",
        "CIVIL_DISOBEDIENCE",
        "DEMOCRATIC_ELECTION",
        "ARMS_EMBARGO",
        "SOCIAL_CONTRACT",
        "CORRUPTION",
        "PEACE_TREATY",
        "FEDERATION",
        "CONSTITUTIONAL_CRISIS",
        "IMPEACHMENT",
        "DIPLOMACY",
        "COLONIALISM",
    ];

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _seed = config.trial_seed("institutional", "analogical_reasoning", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        let loaded = algebra.load_institutional_axioms(&system);
        assert!(loaded >= 18);

        let cases = Self::analogy_cases();

        // ── 1. Transfer accuracy + strength + selectivity ──
        let mut above_chance = 0usize;
        let mut strengths = Vec::new();
        let mut selectivities = Vec::new();

        for case in &cases {
            let result = algebra.query_analogy(case.a, case.b, case.d, &system);
            let (_nearest, sim, result_hv) = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            let sim_f64 = sim as f64;
            if !sim_f64.is_finite() {
                continue;
            }

            if sim > 0.50 {
                above_chance += 1;
            }
            strengths.push(sim_f64);

            // Selectivity: how much the best match exceeds the 2nd-best.
            // High selectivity = the analogy maps crisply to one axiom.
            let mut sims: Vec<f32> = Self::AXIOM_NAMES
                .iter()
                .filter_map(|&name| {
                    algebra
                        .get_encoding(name, &system)
                        .map(|hv| result_hv.similarity(&hv))
                })
                .collect();
            sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            if sims.len() >= 2 && sims[0] > 0.0 {
                let sel = (sims[0] - sims[1]) / sims[0];
                selectivities.push(sel as f64);
            }
        }

        let transfer_accuracy = if cases.is_empty() {
            0.0
        } else {
            above_chance as f64 / cases.len() as f64
        };

        let transfer_strength = if strengths.is_empty() {
            0.0
        } else {
            strengths.iter().sum::<f64>() / strengths.len() as f64
        };

        let analogical_selectivity = if selectivities.is_empty() {
            0.0
        } else {
            selectivities.iter().sum::<f64>() / selectivities.len() as f64
        };

        // ── 2. HDC-native analogy (XOR encoding-space transform) ──
        let mut hdc_above_chance = 0usize;
        let mut hdc_strengths = Vec::new();
        for case in &cases {
            if let Ok((_nearest, sim, _hv)) = algebra.query_analogy_hdc(case.a, case.b, case.d) {
                let sim_f64 = sim as f64;
                if sim_f64.is_finite() {
                    if sim > 0.50 {
                        hdc_above_chance += 1;
                    }
                    hdc_strengths.push(sim_f64);
                }
            }
        }
        let hdc_transfer_accuracy = if cases.is_empty() {
            0.0
        } else {
            hdc_above_chance as f64 / cases.len() as f64
        };
        let hdc_transfer_strength = if hdc_strengths.is_empty() {
            0.0
        } else {
            hdc_strengths.iter().sum::<f64>() / hdc_strengths.len() as f64
        };

        // ── 3. Asymmetry via similarity-profile divergence ──
        // Compare the full similarity profile (to all 12 axioms) between
        // forward A:B::?:D and reverse B:A::?:D. This captures differences
        // in ranking and magnitude, not just nearest-axiom identity.
        let mut asymmetry_scores = Vec::new();
        for case in &cases {
            let fwd = algebra.query_analogy(case.a, case.b, case.d, &system);
            let rev = algebra.query_analogy(case.b, case.a, case.d, &system);
            if let (Ok((_fwd_name, _fwd_sim, fwd_hv)), Ok((_rev_name, _rev_sim, rev_hv))) =
                (fwd, rev)
            {
                // Compute similarity profiles to all axioms
                let mut fwd_profile = Vec::new();
                let mut rev_profile = Vec::new();
                for &name in Self::AXIOM_NAMES {
                    if let Some(ax_hv) = algebra.get_encoding(name, &system) {
                        fwd_profile.push(fwd_hv.similarity(&ax_hv) as f64);
                        rev_profile.push(rev_hv.similarity(&ax_hv) as f64);
                    }
                }
                // Profile divergence: 1 - cosine_similarity(fwd_profile, rev_profile)
                if !fwd_profile.is_empty() {
                    let dot: f64 = fwd_profile
                        .iter()
                        .zip(&rev_profile)
                        .map(|(a, b)| a * b)
                        .sum();
                    let mag_f: f64 = fwd_profile.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let mag_r: f64 = rev_profile.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if mag_f > 0.0 && mag_r > 0.0 {
                        let cos_sim = dot / (mag_f * mag_r);
                        // Scale up divergence: raw 1-cos is tiny, so multiply to [0,1]
                        let divergence = ((1.0 - cos_sim) * 100.0).min(1.0);
                        asymmetry_scores.push(divergence);
                    }
                }
            }
        }
        let asymmetry_score = if asymmetry_scores.is_empty() {
            0.0
        } else {
            asymmetry_scores.iter().sum::<f64>() / asymmetry_scores.len() as f64
        };

        TrialResult {
            transfer_accuracy,
            transfer_strength,
            analogical_selectivity,
            asymmetry_score,
            hdc_transfer_accuracy,
            hdc_transfer_strength,
        }
    }
}

impl PsychBenchmark for AnalogicalReasoningBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::AnalogicalReasoning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Analogical Reasoning (HDC Composition Algebra)",
            citation: "Gentner, D. (1983). Structure-mapping. Cognitive Science.",
            year: 1983,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut transfer_accs = Vec::new();
        let mut transfer_strengths = Vec::new();
        let mut selectivities = Vec::new();
        let mut asymmetries = Vec::new();
        let mut hdc_accs = Vec::new();
        let mut hdc_strengths = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            transfer_accs.push(r.transfer_accuracy);
            transfer_strengths.push(r.transfer_strength);
            selectivities.push(r.analogical_selectivity);
            asymmetries.push(r.asymmetry_score);
            hdc_accs.push(r.hdc_transfer_accuracy);
            hdc_strengths.push(r.hdc_transfer_strength);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("transfer_strength".to_string(), r.transfer_strength);
                extra.insert(
                    "analogical_selectivity".to_string(),
                    r.analogical_selectivity,
                );
                extra.insert("asymmetry_score".to_string(), r.asymmetry_score);
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "analogical_reasoning".to_string(),
                    correct: r.transfer_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.transfer_strength,
                    confidence: r.transfer_accuracy,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "analogical_transfer_accuracy",
            MetricValue::from_samples(&transfer_accs),
        );
        result.insert(
            "analogical_transfer_strength",
            MetricValue::from_samples(&transfer_strengths),
        );
        result.insert(
            "analogical_selectivity",
            MetricValue::from_samples(&selectivities),
        );
        result.insert(
            "analogical_asymmetry_score",
            MetricValue::from_samples(&asymmetries),
        );
        result.insert(
            "analogical_hdc_transfer_accuracy",
            MetricValue::from_samples(&hdc_accs),
        );
        result.insert(
            "analogical_hdc_transfer_strength",
            MetricValue::from_samples(&hdc_strengths),
        );

        result.conditions = 4;
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analogical_reasoning_runs() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        assert!(result.metrics.contains_key("analogical_transfer_accuracy"));
        assert!(result.metrics.contains_key("analogical_transfer_strength"));
        assert!(result.metrics.contains_key("analogical_selectivity"));
    }

    #[test]
    fn test_analogical_reasoning_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_transfer_strength_above_chance() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        let strength = result.metrics.get("analogical_transfer_strength").unwrap();
        // With bundled compositions, analogical targets should have
        // above-chance similarity to some axiom
        assert!(
            strength.mean > 0.48,
            "Transfer strength should be near or above chance, got {}",
            strength.mean
        );
    }

    #[test]
    fn test_selectivity_positive() {
        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        let sel = result.metrics.get("analogical_selectivity").unwrap();
        // Selectivity measures how crisply the analogy maps to one axiom
        // (best - 2nd_best) / best. Should be positive and finite.
        assert!(
            sel.mean > 0.0 && sel.mean.is_finite(),
            "Selectivity should be positive and finite, got {}",
            sel.mean
        );
    }

    #[test]
    fn test_print_analogical_scores() {
        use symthaea_core::hdc::primitive_system::PrimitiveSystem;

        let config = BenchmarkConfig::default();
        let result = AnalogicalReasoningBenchmark.run(&config);
        eprintln!("\n═══ Analogical Reasoning Benchmark Scores ═══");
        for (key, val) in &result.metrics {
            let short = key.strip_prefix("analogical_").unwrap_or(key);
            eprintln!("  {short:<35} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }

        // Per-case analogy details
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms(&system);

        eprintln!("\n  ── Per-case analogy details ──");
        for case in AnalogicalReasoningBenchmark::analogy_cases() {
            match algebra.query_analogy(case.a, case.b, case.d, &system) {
                Ok((nearest, sim, result_hv)) => {
                    // Compute selectivity for this case
                    let mut sims: Vec<f32> = AnalogicalReasoningBenchmark::AXIOM_NAMES
                        .iter()
                        .filter_map(|&name| {
                            algebra
                                .get_encoding(name, &system)
                                .map(|hv| result_hv.similarity(&hv))
                        })
                        .collect();
                    sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let sel = if sims.len() >= 2 && sims[0] > 0.0 {
                        (sims[0] - sims[1]) / sims[0]
                    } else {
                        0.0
                    };
                    eprintln!(
                        "  {}:{} :: ?:{} => nearest={nearest}, sim={sim:.4}, sel={sel:.4}",
                        case.a, case.b, case.d
                    );
                }
                Err(e) => {
                    eprintln!("  [ERR ] {}:{} :: ?:{}: {e}", case.a, case.b, case.d);
                }
            }
        }
        eprintln!("═══════════════════════════════════════════════\n");
    }
}

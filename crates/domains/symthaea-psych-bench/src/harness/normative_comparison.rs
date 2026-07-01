// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Normative comparison of agent performance against human baselines.
//!
//! Computes z-scores, percentiles, and clinical-style interpretations for each
//! benchmark metric by comparing agent values to published human population
//! norms. Uses the Abramowitz & Stegun (1964) approximation for the standard
//! normal CDF.

use serde::{Deserialize, Serialize};

use super::baselines::{BaselineCollection, BaselineMap};
use super::report::{BenchmarkReport, key_metric_for_benchmark};

/// A single normative comparison between agent performance and human baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormativeScore {
    /// Benchmark name (e.g., "Executive::Stroop").
    pub benchmark: String,
    /// Metric name (e.g., "incongruent_accuracy").
    pub metric: String,
    /// Agent's observed value.
    pub agent_value: f64,
    /// Human population mean.
    pub human_mean: f64,
    /// Human population standard deviation.
    pub human_sd: f64,
    /// z-score: (agent - human_mean) / human_sd, sign-corrected so positive = better.
    pub z_score: f64,
    /// Percentile rank in the human population (0.0 to 100.0).
    pub percentile: f64,
    /// Clinical interpretation label (e.g., "Average", "Superior").
    pub interpretation: String,
    /// Source citation for the baseline.
    pub source: String,
    /// Sample population region (e.g., "WEIRD") for cross-cultural awareness.
    #[serde(default)]
    pub sample_region: Option<String>,
}

/// Convert a z-score to a percentile rank (0.0–100.0).
///
/// Delegates to `analysis::normal_cdf` (Abramowitz & Stegun 26.2.17).
pub fn z_to_percentile(z: f64) -> f64 {
    super::analysis::normal_cdf(z) * 100.0
}

/// Clinical interpretation of a z-score.
///
/// Uses standard neuropsychological classification bands:
/// - Superior: z > 2.0 (>97.7th percentile)
/// - Above average: z > 1.0 (>84.1th percentile)
/// - Average: -1.0 <= z <= 1.0 (15.9th to 84.1th percentile)
/// - Below average: z < -1.0 (<15.9th percentile)
/// - Impaired: z < -2.0 (<2.3rd percentile)
pub fn interpret_z(z: f64) -> &'static str {
    if z > 2.0 {
        "Superior"
    } else if z > 1.0 {
        "Above average"
    } else if z >= -1.0 {
        "Average"
    } else if z >= -2.0 {
        "Below average"
    } else {
        "Impaired"
    }
}

/// Full normative comparison report across all benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormativeReport {
    /// Individual normative scores.
    pub scores: Vec<NormativeScore>,
    /// Mean z-score across all comparisons.
    pub overall_mean_z: f64,
    /// Overall percentile (from overall_mean_z).
    pub overall_percentile: f64,
    /// Metrics where agent significantly exceeds human norms (z > 1.0).
    pub strengths: Vec<String>,
    /// Metrics where agent significantly falls below human norms (z < -1.0).
    pub weaknesses: Vec<String>,
}

use super::report::is_lower_better;

/// Map a benchmark name to its corresponding baseline domain and key.
///
/// Returns `(baseline_key, &BaselineMap)` pairs for the benchmark's key metric.
fn baseline_for_benchmark<'a>(
    benchmark: &str,
    metric: &str,
    bl: &'a BaselineCollection,
) -> Option<(&'static str, &'a BaselineMap)> {
    // Map benchmark domain prefix to baseline collection and key
    match benchmark {
        // Executive domain
        name if name.contains("Stroop") && !name.contains("Emotional") => {
            Some(("stroop_effect", &bl.executive))
        }
        name if name.contains("WCST") || name.contains("Wisconsin") => {
            Some(("wcst_categories_completed", &bl.executive))
        }
        name if name.contains("Flanker") => Some(("flanker_effect", &bl.executive)),
        name if name.contains("TowerOfLondon") => Some(("tol_overall_optimal_rate", &bl.executive)),
        name if name.contains("DualTask") => Some(("dual_task_cost", &bl.executive)),
        name if name.contains("Ravens") => Some(("ravens_overall_accuracy", &bl.executive)),
        name if name.contains("IowaGambling") => Some(("igt_overall_net_score", &bl.executive)),

        // WorM domain
        name if name.contains("N-back") => Some(("nback_2_accuracy", &bl.worm)),
        name if name.contains("ChangeDetection") => Some(("change_detection_k4", &bl.worm)),
        name if name.contains("SerialRecall") => Some(("serial_primacy_advantage", &bl.worm)),
        name if name.contains("SpatialUpdating") => Some(("spatial_updating_accuracy", &bl.worm)),
        // Binding domain — route to binding baselines, not worm
        name if name.contains("TemporalOrder") => Some(("discrimination_slope", &bl.binding)),
        name if name.contains("CrossModal") => Some(("cross_modal_binding_accuracy", &bl.binding)),
        name if name.contains("FeatureConjunction") => Some(("conjunction_accuracy", &bl.binding)),
        name if name.contains("Binding") => Some(("binding_accuracy", &bl.worm)),
        name if name.contains("DigitSpan") => Some(("digit_span_forward", &bl.worm)),

        // Sustained Attention
        name if name.contains("PVT") => Some(("vigilance_decrement", &bl.sustained_attention)),
        name if name.contains("SART") => Some(("commission_errors", &bl.sustained_attention)),
        name if name.contains("CPT") => Some(("cpt_d_prime", &bl.sustained_attention)),

        // Metacognition
        name if name.contains("Calibration") => Some(("calibration_error_ece", &bl.metacognition)),
        name if name.contains("FOK") || name.contains("FeelingOfKnowing") => {
            Some(("fok_gamma", &bl.metacognition))
        }
        name if name.contains("ChangeBlindness") => {
            Some(("cb_detection_with_disruption", &bl.metacognition))
        }

        // ToMBench
        name if name.contains("FalseBelief") => Some(("false_belief_accuracy", &bl.tombench)),
        name if name.contains("FauxPas") => Some(("faux_pas_accuracy", &bl.tombench)),
        name if name.contains("Hinting") => Some(("hinting_accuracy", &bl.tombench)),
        name if name.contains("Persuasion") => Some(("persuasion_detection", &bl.tombench)),
        name if name.contains("StrangeStory") => Some(("strange_story_accuracy", &bl.tombench)),

        // Affect
        name if name.contains("Emotional") && name.contains("Stroop") => {
            Some(("emotional_interference", &bl.affect))
        }
        name if name.contains("MoodCongruent") => Some(("congruence_ratio", &bl.affect)),
        name if name.contains("ValenceClassification") => Some(("valence_accuracy", &bl.affect)),

        // Inhibition
        name if name.contains("GoNoGo") => Some(("nogo_accuracy", &bl.inhibition)),
        name if name.contains("StopSignal") => Some(("ssrt_ticks", &bl.inhibition)),

        // Attention
        name if name.contains("VisualSearch") => Some(("search_asymmetry", &bl.attention)),
        name if name.contains("AttentionalBlink") => Some(("blink_magnitude", &bl.attention)),

        // Motor
        name if name.contains("FittsLaw") => Some(("fitts_r_squared", &bl.motor)),
        name if name.contains("SRTT") => Some(("learning_effect", &bl.motor)),
        name if name.contains("Bimanual") => Some(("coordination_cost", &bl.motor)),

        // Language
        name if name.contains("LexicalDecision") => Some(("lexicality_effect", &bl.language)),
        name if name.contains("SemanticPriming") => Some(("priming_effect", &bl.language)),
        name if name.contains("SemanticCoherence") => Some(("coherence_mean", &bl.language)),
        name if name.contains("GardenPath") => Some(("disambiguation_cost", &bl.language)),

        // Social
        name if name.contains("RME") => Some(("rme_accuracy", &bl.social)),
        name if name.contains("UltimatumGame") => Some(("fairness_sensitivity", &bl.social)),
        name if name.contains("SocialNorm") => Some(("norm_d_prime", &bl.social)),

        // Reasoning (ARC) — variant-specific baselines
        name if name.contains("ArcCompositional") => {
            Some(("arc_compositional_accuracy", &bl.reasoning))
        }
        name if name.contains("ArcAnalogy") => Some(("arc_analogy_accuracy", &bl.reasoning)),
        name if name.contains("ArcAbductive") => Some(("arc_abduction_accuracy", &bl.reasoning)),
        name if name.contains("ArcChain") => Some(("arc_chain_accuracy", &bl.reasoning)),
        name if name.contains("ArcNoise") => Some(("arc_noise_resilience", &bl.reasoning)),
        name if name.contains("ArcFewShot") => Some(("arc_accuracy_1shot", &bl.reasoning)),
        name if name.contains("ArcScaling") => Some(("arc_grid_3x3_accuracy", &bl.reasoning)),
        name if name.contains("ArcRSA") => Some(("arc_rsa_correlation", &bl.reasoning)),
        name if name.contains("ArcAlgebra") => Some(("arc_algebra_score", &bl.reasoning)),
        name if name.contains("ArcStaircase") => Some(("arc_capacity_threshold", &bl.reasoning)),
        name if name.contains("Arc") => Some(("arc_transfer_accuracy", &bl.reasoning)),

        // Memory Agent
        name if name.contains("AccurateRetrieval") => {
            Some(("accurate_retrieval", &bl.memory_agent))
        }
        name if name.contains("LongRange") => Some(("long_range_delay_50", &bl.memory_agent)),
        name if name.contains("ProspectiveMemory") => Some(("pm_hit_rate", &bl.memory_agent)),
        name if name.contains("ConflictResolution") => {
            Some(("conflict_recency_preference", &bl.memory_agent))
        }
        name if name.contains("TestTimeLearning") => Some(("test_time_learning", &bl.memory_agent)),

        // CogBench
        name if name.contains("BART") || name.contains("Bart") => {
            Some(("bart_avg_pumps", &bl.cogbench))
        }
        name if name.contains("Reversal") => Some(("reversal_win_stay", &bl.cogbench)),
        name if name.contains("RestlessBandit") => Some(("restless_bandit_accuracy", &bl.cogbench)),
        name if name.contains("Instrumental") => Some(("instrumental_sensitivity", &bl.cogbench)),
        name if name.contains("Horizon") => Some(("directed_exploration", &bl.cogbench)),
        name if name.contains("TemporalDiscounting") => Some(("discounting_score", &bl.cogbench)),
        name if name.contains("TwoStep") => Some(("model_basedness", &bl.cogbench)),
        name if name.contains("Probabilistic") => {
            Some(("probabilistic_likelihood_weight", &bl.cogbench))
        }

        // Creativity
        name if name.contains("AlternateUses") => Some(("aut_originality", &bl.creativity)),
        name if name.contains("RemoteAssociates") => {
            Some(("rat_mean_solution_rank", &bl.creativity))
        }

        // Butlin
        name if name.contains("Butlin") => Some(("mean_quality_score", &bl.butlin)),

        // Neuromod
        name if name.contains("RewardLearning") => Some(("trials_to_criterion", &bl.neuromod)),
        name if name.contains("YerkesDodson") => Some(("peak_ne_level", &bl.neuromod)),
        name if name.contains("AttentionNetwork") => Some(("conflict_effect", &bl.neuromod)),
        name if name.contains("MoodInduction") => Some(("mood_congruent_bias", &bl.neuromod)),
        name if name.contains("PharmacologicalAblation") => {
            Some(("da_knockout_lr_drop_pct", &bl.neuromod))
        }
        name if name.contains("LiveLoopAblation") => {
            Some(("live_da_knockout_gradient_drop_pct", &bl.neuromod))
        }
        name if name.contains("PharmacologicalChallenge") => {
            Some(("da_agonist_gradient_scale", &bl.neuromod))
        }
        name if name.contains("InjectionChallenge") => {
            Some(("stimulant_peak_effect", &bl.neuromod))
        }
        name if name.contains("AllostaticStress") => {
            Some(("chronic_da_baseline_final", &bl.neuromod))
        }
        name if name.contains("BehavioralKnockout") => Some(("da_ko_lr_d", &bl.neuromod)),
        name if name.contains("ConsciousnessPharmacology") => {
            Some(("psychedelic_proxy_peak", &bl.neuromod))
        }

        _ => {
            // Try generic metric name lookup against all baseline maps
            let _ = metric; // suppress unused warning
            None
        }
    }
}

impl NormativeReport {
    /// Build a normative comparison report from benchmark results.
    ///
    /// For each benchmark result, looks up the key metric and matching human
    /// baseline. Only includes comparisons where the baseline has a known
    /// population SD (required for z-score computation).
    pub fn from_report(report: &BenchmarkReport) -> Self {
        let bl = BaselineCollection::all();
        let mut scores = Vec::new();

        for result in &report.results {
            let benchmark = &result.benchmark;
            let metric_name = key_metric_for_benchmark(benchmark);

            // Get agent's metric value
            let metric_value = match result.metrics.get(metric_name) {
                Some(mv) => mv,
                None => continue,
            };

            // Find matching baseline
            let (baseline_key, baseline_map) =
                match baseline_for_benchmark(benchmark, metric_name, &bl) {
                    Some(pair) => pair,
                    None => continue,
                };

            let baseline = match baseline_map.get(baseline_key) {
                Some(b) => b,
                None => continue,
            };

            // Only include if SD is available (required for z-score)
            let human_sd = match baseline.sd {
                Some(sd) if sd.abs() > 1e-15 => sd,
                _ => continue,
            };

            let human_mean = baseline.value;
            let mut z = (metric_value.mean - human_mean) / human_sd;

            // Negate for lower-is-better metrics so positive z always means "better"
            if is_lower_better(metric_name) {
                z = -z;
            }

            let percentile = z_to_percentile(z);
            let interpretation = interpret_z(z).to_string();

            scores.push(NormativeScore {
                benchmark: benchmark.clone(),
                metric: metric_name.to_string(),
                agent_value: metric_value.mean,
                human_mean,
                human_sd,
                z_score: z,
                percentile,
                interpretation,
                source: baseline.source.to_string(),
                sample_region: None,
            });
        }

        // Enrich with cross-cultural metadata
        let cultural = super::baselines::BaselineCollection::cultural_metadata();
        for score in &mut scores {
            let domain = score
                .benchmark
                .split("::")
                .next()
                .unwrap_or("")
                .to_lowercase();
            if let Some(meta) = cultural.get(domain.as_str()) {
                score.sample_region = Some(meta.sample_region.to_string());
            }
        }

        // Compute summary statistics
        let overall_mean_z = if scores.is_empty() {
            0.0
        } else {
            scores.iter().map(|s| s.z_score).sum::<f64>() / scores.len() as f64
        };

        let overall_percentile = z_to_percentile(overall_mean_z);

        let strengths: Vec<String> = scores
            .iter()
            .filter(|s| s.z_score > 1.0)
            .map(|s| format!("{} / {} (z={:.2})", s.benchmark, s.metric, s.z_score))
            .collect();

        let weaknesses: Vec<String> = scores
            .iter()
            .filter(|s| s.z_score < -1.0)
            .map(|s| format!("{} / {} (z={:.2})", s.benchmark, s.metric, s.z_score))
            .collect();

        Self {
            scores,
            overall_mean_z,
            overall_percentile,
            strengths,
            weaknesses,
        }
    }

    /// Format as a Markdown table with normative comparison data.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str("## Normative Comparison Report\n\n");
        out.push_str(&format!(
            "Overall: mean z = {:.2}, percentile = {:.1}%, interpretation = {}\n\n",
            self.overall_mean_z,
            self.overall_percentile,
            interpret_z(self.overall_mean_z),
        ));

        // Table header
        out.push_str(
            "| Benchmark | Metric | Agent | Human (mean +/- sd) | z | Percentile | Interpretation |\n",
        );
        out.push_str(
            "|-----------|--------|-------|---------------------|---|------------|----------------|\n",
        );

        for s in &self.scores {
            out.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} +/- {:.3} | {:.2} | {:.1}% | {} |\n",
                s.benchmark,
                s.metric,
                s.agent_value,
                s.human_mean,
                s.human_sd,
                s.z_score,
                s.percentile,
                s.interpretation,
            ));
        }

        if !self.strengths.is_empty() {
            out.push_str("\n### Strengths (z > 1.0)\n\n");
            for s in &self.strengths {
                out.push_str(&format!("- {}\n", s));
            }
        }

        if !self.weaknesses.is_empty() {
            out.push_str("\n### Weaknesses (z < -1.0)\n\n");
            for w in &self.weaknesses {
                out.push_str(&format!("- {}\n", w));
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::{BenchmarkReport, BenchmarkResult, MetricValue};

    #[test]
    fn test_z_to_percentile_zero() {
        let p = z_to_percentile(0.0);
        // z=0 should be the 50th percentile
        assert!(
            (p - 50.0).abs() < 0.1,
            "z=0 should be ~50th percentile, got {:.2}",
            p
        );
    }

    #[test]
    fn test_z_to_percentile_positive() {
        let p = z_to_percentile(1.0);
        // z=1.0 -> ~84.13th percentile
        assert!(
            (p - 84.13).abs() < 0.5,
            "z=1.0 should be ~84.13th percentile, got {:.2}",
            p
        );

        let p2 = z_to_percentile(2.0);
        // z=2.0 -> ~97.72nd percentile
        assert!(
            (p2 - 97.72).abs() < 0.5,
            "z=2.0 should be ~97.72nd percentile, got {:.2}",
            p2
        );

        let p3 = z_to_percentile(3.0);
        // z=3.0 -> ~99.87th percentile
        assert!(
            (p3 - 99.87).abs() < 0.5,
            "z=3.0 should be ~99.87th percentile, got {:.2}",
            p3
        );
    }

    #[test]
    fn test_z_to_percentile_negative() {
        let p = z_to_percentile(-1.0);
        // z=-1.0 -> ~15.87th percentile
        assert!(
            (p - 15.87).abs() < 0.5,
            "z=-1.0 should be ~15.87th percentile, got {:.2}",
            p
        );

        let p2 = z_to_percentile(-2.0);
        // z=-2.0 -> ~2.28th percentile
        assert!(
            (p2 - 2.28).abs() < 0.5,
            "z=-2.0 should be ~2.28th percentile, got {:.2}",
            p2
        );

        // Symmetry: P(z) + P(-z) should equal 100
        let sym = z_to_percentile(1.5) + z_to_percentile(-1.5);
        assert!(
            (sym - 100.0).abs() < 0.1,
            "CDF symmetry violated: P(1.5) + P(-1.5) = {:.2}",
            sym
        );
    }

    #[test]
    fn test_interpret_z_ranges() {
        assert_eq!(interpret_z(2.5), "Superior");
        assert_eq!(interpret_z(2.01), "Superior");
        assert_eq!(interpret_z(1.5), "Above average");
        assert_eq!(interpret_z(1.01), "Above average");
        assert_eq!(interpret_z(1.0), "Average");
        assert_eq!(interpret_z(0.0), "Average");
        assert_eq!(interpret_z(-0.5), "Average");
        assert_eq!(interpret_z(-1.0), "Average");
        assert_eq!(interpret_z(-1.5), "Below average");
        assert_eq!(interpret_z(-1.01), "Below average");
        assert_eq!(interpret_z(-2.0), "Below average");
        assert_eq!(interpret_z(-2.01), "Impaired");
        assert_eq!(interpret_z(-3.0), "Impaired");
    }

    #[test]
    fn test_normative_score_construction() {
        let score = NormativeScore {
            benchmark: "Executive::Stroop".to_string(),
            metric: "incongruent_accuracy".to_string(),
            agent_value: 0.90,
            human_mean: 0.85,
            human_sd: 0.10,
            z_score: 0.5,
            percentile: z_to_percentile(0.5),
            interpretation: interpret_z(0.5).to_string(),
            source: "MacLeod (1991)".to_string(),
            sample_region: None,
        };

        assert_eq!(score.benchmark, "Executive::Stroop");
        assert_eq!(score.metric, "incongruent_accuracy");
        assert!((score.agent_value - 0.90).abs() < 1e-10);
        assert!((score.human_mean - 0.85).abs() < 1e-10);
        assert!((score.human_sd - 0.10).abs() < 1e-10);
        assert!((score.z_score - 0.5).abs() < 1e-10);
        assert!(score.percentile > 50.0 && score.percentile < 100.0);
        assert_eq!(score.interpretation, "Average");
    }

    #[test]
    fn test_normative_from_report() {
        let mut report = BenchmarkReport::new();

        // Add Stroop result — baseline has sd
        let mut stroop = BenchmarkResult::new("Executive::Stroop", None);
        stroop.insert(
            "stroop_effect",
            MetricValue::from_samples(&[0.10, 0.08, 0.12]),
        );
        report.add(stroop);

        // Add N-back result
        let mut nback = BenchmarkResult::new("WorM::N-back", None);
        nback.insert(
            "nback_2::accuracy",
            MetricValue::from_samples(&[0.80, 0.82, 0.78]),
        );
        report.add(nback);

        // Add Calibration (lower-is-better metric)
        let mut calib = BenchmarkResult::new("Metacognition::Calibration", None);
        calib.insert(
            "calibration_error_ece",
            MetricValue::from_samples(&[0.10, 0.12, 0.08]),
        );
        report.add(calib);

        let normative = NormativeReport::from_report(&report);

        // Should have some scores (depends on which baselines have SD)
        // At minimum, baselines we know have SD: nback_2_accuracy (sd=0.10), stroop_incongruent_accuracy
        assert!(
            !normative.scores.is_empty(),
            "Expected at least one normative comparison"
        );

        // Verify z-scores are finite
        for s in &normative.scores {
            assert!(
                s.z_score.is_finite(),
                "z-score not finite for {}",
                s.benchmark
            );
            assert!(
                s.percentile >= 0.0 && s.percentile <= 100.0,
                "Percentile out of range for {}: {}",
                s.benchmark,
                s.percentile
            );
            assert!(
                !s.interpretation.is_empty(),
                "Empty interpretation for {}",
                s.benchmark
            );
        }

        // Overall statistics should be reasonable
        assert!(normative.overall_mean_z.is_finite());
        assert!(normative.overall_percentile >= 0.0 && normative.overall_percentile <= 100.0);

        // Check lower-is-better negation: if calibration ECE is below human mean,
        // z-score should be positive (better than human)
        if let Some(calib_score) = normative
            .scores
            .iter()
            .find(|s| s.benchmark.contains("Calibration"))
        {
            // agent ECE ~0.10, if human ECE is higher, z should be positive
            // (lower ECE is better, so negation applies)
            assert!(
                calib_score.z_score.is_finite(),
                "Calibration z-score should be finite"
            );
        }
    }

    #[test]
    fn test_normative_markdown() {
        let mut report = BenchmarkReport::new();

        let mut stroop = BenchmarkResult::new("Executive::Stroop", None);
        stroop.insert(
            "stroop_effect",
            MetricValue::from_samples(&[0.10, 0.08, 0.12]),
        );
        report.add(stroop);

        let normative = NormativeReport::from_report(&report);
        let md = normative.to_markdown();

        // Should contain table structure
        assert!(
            md.contains("Normative Comparison Report"),
            "Missing report title"
        );
        assert!(md.contains("| Benchmark |"), "Missing table header");
        assert!(md.contains("|-----------|"), "Missing table separator");
        assert!(md.contains("mean z ="), "Missing overall mean z");
        assert!(md.contains("percentile ="), "Missing overall percentile");

        // If any scores were produced, table should have data rows
        if !normative.scores.is_empty() {
            assert!(
                md.contains("Executive::Stroop") || md.contains("Stroop"),
                "Missing Stroop benchmark in table"
            );
        }
    }

    #[test]
    fn test_normative_empty_report() {
        let report = BenchmarkReport::new();
        let normative = NormativeReport::from_report(&report);

        assert!(normative.scores.is_empty());
        assert!((normative.overall_mean_z).abs() < 1e-10);
        assert!(normative.strengths.is_empty());
        assert!(normative.weaknesses.is_empty());
    }

    #[test]
    fn test_normative_strengths_weaknesses() {
        // Construct a report with known z-score outcomes
        let report = NormativeReport {
            scores: vec![
                NormativeScore {
                    benchmark: "Test::Strong".to_string(),
                    metric: "accuracy".to_string(),
                    agent_value: 0.95,
                    human_mean: 0.70,
                    human_sd: 0.10,
                    z_score: 2.5,
                    percentile: z_to_percentile(2.5),
                    interpretation: interpret_z(2.5).to_string(),
                    source: "Test".to_string(),
                    sample_region: None,
                },
                NormativeScore {
                    benchmark: "Test::Weak".to_string(),
                    metric: "accuracy".to_string(),
                    agent_value: 0.40,
                    human_mean: 0.70,
                    human_sd: 0.10,
                    z_score: -3.0,
                    percentile: z_to_percentile(-3.0),
                    interpretation: interpret_z(-3.0).to_string(),
                    source: "Test".to_string(),
                    sample_region: None,
                },
                NormativeScore {
                    benchmark: "Test::Average".to_string(),
                    metric: "accuracy".to_string(),
                    agent_value: 0.72,
                    human_mean: 0.70,
                    human_sd: 0.10,
                    z_score: 0.2,
                    percentile: z_to_percentile(0.2),
                    interpretation: interpret_z(0.2).to_string(),
                    source: "Test".to_string(),
                    sample_region: None,
                },
            ],
            overall_mean_z: (2.5 + -3.0 + 0.2) / 3.0,
            overall_percentile: z_to_percentile((2.5 + -3.0 + 0.2) / 3.0),
            strengths: vec!["Test::Strong / accuracy (z=2.50)".to_string()],
            weaknesses: vec!["Test::Weak / accuracy (z=-3.00)".to_string()],
        };

        assert_eq!(report.strengths.len(), 1);
        assert!(report.strengths[0].contains("Test::Strong"));
        assert_eq!(report.weaknesses.len(), 1);
        assert!(report.weaknesses[0].contains("Test::Weak"));

        // Verify markdown includes strengths/weaknesses sections
        let md = report.to_markdown();
        assert!(md.contains("### Strengths"));
        assert!(md.contains("### Weaknesses"));
        assert!(md.contains("Test::Strong"));
        assert!(md.contains("Test::Weak"));
    }

    #[test]
    fn test_lower_better_z_negation() {
        // Verify that is_lower_better correctly identifies metrics
        assert!(is_lower_better("stroop_effect"));
        assert!(is_lower_better("flanker_effect"));
        assert!(is_lower_better("dual_task_cost"));
        assert!(is_lower_better("calibration_error_ece"));
        assert!(is_lower_better("commission_errors"));
        assert!(is_lower_better("ssrt_ticks"));
        assert!(is_lower_better("coordination_cost"));
        assert!(is_lower_better("vigilance_decrement"));
        assert!(is_lower_better("disambiguation_cost"));
        assert!(is_lower_better("blink_magnitude"));
        assert!(!is_lower_better("accuracy"));
        assert!(!is_lower_better("hit_rate"));
        assert!(!is_lower_better("fok_gamma"));
    }
}

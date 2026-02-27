//! Composite cognitive profile: aggregate 67 benchmarks into ~8 domain scores.
//!
//! Produces a "radar chart" data structure mapping cognitive domains to
//! normalized performance scores, enabling quick comparison of strengths
//! and weaknesses across cognitive faculties.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::report::BenchmarkReport;

/// A cognitive domain with its constituent benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainScore {
    /// Domain name.
    pub domain: String,
    /// Mean score across benchmarks (0.0–1.0 scale).
    pub score: f64,
    /// Number of benchmarks contributing.
    pub n_benchmarks: usize,
    /// Individual benchmark scores within this domain.
    pub benchmark_scores: Vec<(String, f64)>,
    /// Interpretation label.
    pub interpretation: String,
}

/// The 8 cognitive domains and their benchmark assignments.
pub fn domain_assignments() -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut map = BTreeMap::new();

    map.insert("Executive", vec![
        "Executive::Stroop",
        "Executive::WCST",
        "Executive::Flanker",
        "Executive::TowerOfLondon",
        "Executive::DualTask",
        "Executive::Ravens",
        "Executive::IowaGambling",
    ]);

    map.insert("Memory", vec![
        "WorM::N-back",
        "WorM::ChangeDetection",
        "WorM::SerialRecall",
        "WorM::SpatialUpdating",
        "WorM::Binding",
        "WorM::DigitSpan",
        "MemoryAgent::AccurateRetrieval",
        "MemoryAgent::LongRange",
        "MemoryAgent::ProspectiveMemory",
        "MemoryAgent::ConflictResolution",
        "MemoryAgent::TestTimeLearning",
    ]);

    map.insert("Attention", vec![
        "Attention::VisualSearch",
        "Attention::AttentionalBlink",
        "SustainedAttention::PVT",
        "SustainedAttention::CPT",
        "SustainedAttention::SART",
    ]);

    map.insert("Social", vec![
        "Social::RME",
        "Social::SocialNorm",
        "Social::UltimatumGame",
        "ToMBench::FalseBelief",
        "ToMBench::FauxPas",
        "ToMBench::Persuasion",
        "ToMBench::StrangeStory",
        "ToMBench::Hinting",
    ]);

    map.insert("Motor", vec![
        "Motor::FittsLaw",
        "Motor::Bimanual",
        "Motor::SRTT",
    ]);

    map.insert("Language", vec![
        "Language::LexicalDecision",
        "Language::SemanticPriming",
        "Language::SemanticCoherence",
        "Language::GardenPath",
    ]);

    map.insert("Metacognition", vec![
        "Metacognition::Calibration",
        "Metacognition::FOK",
    ]);

    map.insert("Reasoning", vec![
        "Reasoning::ArcFluid",
        "Reasoning::ArcCompositional",
        "Reasoning::ArcAnalogy",
        "Reasoning::ArcAbductive",
        "Reasoning::ArcAlgebra",
        "Reasoning::ArcChain",
        "Reasoning::ArcFewShot",
        "Reasoning::ArcNoise",
        "Reasoning::ArcRsa",
        "Reasoning::ArcScaling",
        "Reasoning::ArcStaircase",
    ]);

    map
}

/// Key metric to use for each benchmark's score.
fn key_metric(benchmark_name: &str) -> &'static str {
    match benchmark_name {
        name if name.contains("Stroop") && name.contains("Emotional") => "emotional_interference",
        name if name.contains("Stroop") => "incongruent_accuracy",
        name if name.contains("WCST") => "accuracy",
        name if name.contains("Flanker") => "incongruent_accuracy",
        name if name.contains("TowerOfLondon") => "accuracy",
        name if name.contains("DualTask") => "dual_cost",
        name if name.contains("Ravens") => "accuracy",
        name if name.contains("IowaGambling") => "net_score",
        name if name.contains("N-back") => "hit_rate",
        name if name.contains("ChangeDetection") => "accuracy",
        name if name.contains("SerialRecall") => "accuracy",
        name if name.contains("SpatialUpdating") => "accuracy",
        name if name.contains("Binding") => "accuracy",
        name if name.contains("DigitSpan") => "max_span",
        name if name.contains("PVT") => "mean_rt_ticks",
        name if name.contains("CPT") => "hit_rate",
        name if name.contains("SART") => "accuracy",
        name if name.contains("FittsLaw") => "fitts_r_squared",
        name if name.contains("Bimanual") => "accuracy",
        name if name.contains("SRTT") => "accuracy",
        name if name.contains("GoNoGo") => "overall_accuracy",
        name if name.contains("StopSignal") => "accuracy",
        name if name.contains("VisualSearch") => "accuracy",
        name if name.contains("AttentionalBlink") => "t1_accuracy",
        name if name.contains("RME") => "accuracy",
        name if name.contains("SocialNorm") => "accuracy",
        name if name.contains("UltimatumGame") => "acceptance_rate",
        name if name.contains("FalseBelief") => "accuracy",
        name if name.contains("FauxPas") => "accuracy",
        name if name.contains("Persuasion") => "accuracy",
        name if name.contains("StrangeStory") => "accuracy",
        name if name.contains("Hinting") => "accuracy",
        name if name.contains("LexicalDecision") => "word_accuracy",
        name if name.contains("SemanticPriming") => "priming_effect",
        name if name.contains("SemanticCoherence") => "accuracy",
        name if name.contains("GardenPath") => "accuracy",
        name if name.contains("Calibration") => "calibration_ece",
        name if name.contains("FOK") => "gamma",
        name if name.contains("Arc") => "transfer_accuracy",
        name if name.contains("AccurateRetrieval") => "accuracy",
        name if name.contains("LongRange") => "accuracy",
        name if name.contains("ProspectiveMemory") => "pm_accuracy",
        name if name.contains("ConflictResolution") => "accuracy",
        name if name.contains("TestTimeLearning") => "accuracy",
        _ => "accuracy",
    }
}

/// Whether a metric is "lower is better" (needs inversion for profile scoring).
fn is_lower_better(metric: &str) -> bool {
    matches!(metric,
        "mean_rt_ticks" | "dual_cost" | "calibration_ece" | "stroop_effect"
    )
}

/// Full cognitive profile from a benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveProfile {
    /// Per-domain scores.
    pub domains: Vec<DomainScore>,
    /// Overall composite score (mean across domains).
    pub overall: f64,
    /// Strongest domain.
    pub strongest: String,
    /// Weakest domain.
    pub weakest: String,
}

impl CognitiveProfile {
    /// Compute a cognitive profile from a benchmark report.
    pub fn from_report(report: &BenchmarkReport) -> Self {
        let assignments = domain_assignments();
        let mut domains = Vec::new();

        for (domain, benchmark_names) in &assignments {
            let mut scores = Vec::new();

            for &bench_name in benchmark_names {
                if let Some(result) = report.results.iter().find(|r| r.benchmark == bench_name) {
                    let metric = key_metric(bench_name);
                    if let Some(mv) = result.metrics.get(metric) {
                        let mut score = mv.mean;

                        // Normalize: invert if lower-is-better
                        if is_lower_better(metric) {
                            // Map to 0-1 where higher = better
                            // Use sigmoid-like transform: score = 1 / (1 + value/scale)
                            score = 1.0 / (1.0 + score.abs() / 5.0);
                        }

                        // Clamp to [0, 1]
                        score = score.clamp(0.0, 1.0);
                        scores.push((bench_name.to_string(), score));
                    }
                }
            }

            if !scores.is_empty() {
                let mean_score = scores.iter().map(|(_, s)| s).sum::<f64>() / scores.len() as f64;
                let interpretation = interpret_score(mean_score).to_string();

                domains.push(DomainScore {
                    domain: domain.to_string(),
                    score: mean_score,
                    n_benchmarks: scores.len(),
                    benchmark_scores: scores,
                    interpretation,
                });
            }
        }

        domains.sort_by(|a, b| a.domain.cmp(&b.domain));

        let overall = if domains.is_empty() {
            0.0
        } else {
            domains.iter().map(|d| d.score).sum::<f64>() / domains.len() as f64
        };

        let strongest = domains
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .map(|d| d.domain.clone())
            .unwrap_or_default();

        let weakest = domains
            .iter()
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .map(|d| d.domain.clone())
            .unwrap_or_default();

        Self {
            domains,
            overall,
            strongest,
            weakest,
        }
    }

    /// Format as ASCII radar chart.
    pub fn to_radar_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("## Cognitive Profile\n\n");
        out.push_str(&format!(
            "Overall: {:.1}% | Strongest: {} | Weakest: {}\n\n",
            self.overall * 100.0,
            self.strongest,
            self.weakest
        ));

        // Bar chart representation
        let max_width = 40;
        for d in &self.domains {
            let bar_len = (d.score * max_width as f64).round() as usize;
            let bar: String = "#".repeat(bar_len);
            let pad: String = " ".repeat(max_width - bar_len);
            out.push_str(&format!(
                "{:<14} |{}{}| {:.1}% ({} benchmarks) {}\n",
                d.domain,
                bar,
                pad,
                d.score * 100.0,
                d.n_benchmarks,
                d.interpretation
            ));
        }

        out.push_str("\n### Domain Details\n\n");
        for d in &self.domains {
            out.push_str(&format!("**{}** ({:.1}%)\n", d.domain, d.score * 100.0));
            for (bench, score) in &d.benchmark_scores {
                let short = bench.split("::").last().unwrap_or(bench);
                out.push_str(&format!("  - {}: {:.1}%\n", short, score * 100.0));
            }
            out.push('\n');
        }

        out
    }
}

/// Interpret a domain score.
fn interpret_score(score: f64) -> &'static str {
    if score >= 0.90 {
        "Exceptional"
    } else if score >= 0.75 {
        "Strong"
    } else if score >= 0.50 {
        "Moderate"
    } else if score >= 0.25 {
        "Below average"
    } else {
        "Impaired"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::{BenchmarkResult, MetricValue};

    fn make_report() -> BenchmarkReport {
        let mut report = BenchmarkReport::new();

        // Add a few results across domains
        let mut stroop = BenchmarkResult::new("Executive::Stroop", None);
        stroop.insert("incongruent_accuracy", MetricValue::from_samples(&[0.85]));
        report.add(stroop);

        let mut nback = BenchmarkResult::new("WorM::N-back", None);
        nback.insert("hit_rate", MetricValue::from_samples(&[0.70]));
        report.add(nback);

        let mut pvt = BenchmarkResult::new("SustainedAttention::PVT", None);
        pvt.insert("mean_rt_ticks", MetricValue::from_samples(&[4.0]));
        report.add(pvt);

        let mut rme = BenchmarkResult::new("Social::RME", None);
        rme.insert("accuracy", MetricValue::from_samples(&[0.60]));
        report.add(rme);

        let mut fitts = BenchmarkResult::new("Motor::FittsLaw", None);
        fitts.insert("fitts_r_squared", MetricValue::from_samples(&[0.95]));
        report.add(fitts);

        let mut lex = BenchmarkResult::new("Language::LexicalDecision", None);
        lex.insert("word_accuracy", MetricValue::from_samples(&[0.90]));
        report.add(lex);

        let mut calib = BenchmarkResult::new("Metacognition::Calibration", None);
        calib.insert("calibration_ece", MetricValue::from_samples(&[0.15]));
        report.add(calib);

        let mut arc = BenchmarkResult::new("Reasoning::ArcFluid", None);
        arc.insert("transfer_accuracy", MetricValue::from_samples(&[0.55]));
        report.add(arc);

        report
    }

    #[test]
    fn test_domain_assignments_coverage() {
        let assignments = domain_assignments();
        assert_eq!(assignments.len(), 8);
        assert!(assignments.contains_key("Executive"));
        assert!(assignments.contains_key("Memory"));
        assert!(assignments.contains_key("Attention"));
        assert!(assignments.contains_key("Social"));
        assert!(assignments.contains_key("Motor"));
        assert!(assignments.contains_key("Language"));
        assert!(assignments.contains_key("Metacognition"));
        assert!(assignments.contains_key("Reasoning"));
    }

    #[test]
    fn test_cognitive_profile_from_report() {
        let report = make_report();
        let profile = CognitiveProfile::from_report(&report);

        assert!(!profile.domains.is_empty());
        assert!(profile.overall >= 0.0 && profile.overall <= 1.0);
        assert!(!profile.strongest.is_empty());
        assert!(!profile.weakest.is_empty());
    }

    #[test]
    fn test_domain_scores_bounded() {
        let report = make_report();
        let profile = CognitiveProfile::from_report(&report);

        for d in &profile.domains {
            assert!(
                d.score >= 0.0 && d.score <= 1.0,
                "{} score out of bounds: {}",
                d.domain,
                d.score
            );
        }
    }

    #[test]
    fn test_lower_better_inversion() {
        // PVT mean_rt_ticks=4.0 should be inverted to a 0-1 score
        let report = make_report();
        let profile = CognitiveProfile::from_report(&report);

        let attention = profile.domains.iter().find(|d| d.domain == "Attention");
        assert!(attention.is_some());
        let score = attention.unwrap().score;
        // RT of 4.0 → 1/(1+4/5) = 1/1.8 ≈ 0.556
        assert!(score > 0.3 && score < 0.8, "PVT score should be moderate: {}", score);
    }

    #[test]
    fn test_radar_markdown() {
        let report = make_report();
        let profile = CognitiveProfile::from_report(&report);
        let md = profile.to_radar_markdown();

        assert!(md.contains("Cognitive Profile"));
        assert!(md.contains("Overall:"));
        assert!(md.contains("Strongest:"));
        assert!(md.contains("#")); // bar chart
    }

    #[test]
    fn test_interpret_score() {
        assert_eq!(interpret_score(0.95), "Exceptional");
        assert_eq!(interpret_score(0.80), "Strong");
        assert_eq!(interpret_score(0.60), "Moderate");
        assert_eq!(interpret_score(0.30), "Below average");
        assert_eq!(interpret_score(0.10), "Impaired");
    }

    #[test]
    fn test_empty_report() {
        let report = BenchmarkReport::new();
        let profile = CognitiveProfile::from_report(&report);
        assert!(profile.domains.is_empty());
        assert!((profile.overall).abs() < 1e-10);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark harness: traits, configuration, reporting, and baselines.

pub mod analysis;
pub mod baselines;
pub mod cognitive_profile;
pub mod config;
pub mod cross_domain_prediction;
pub mod difficulty;
pub mod html;
pub mod live_runner;
pub mod neuromod_correlation;
pub mod neuromod_profiles;
pub mod normative_comparison;
pub mod psychometric_report;
pub mod reliability_analysis;
pub mod report;
pub mod sat_curves;
pub mod snapshot;
pub mod staircase;
pub mod transfer;
pub mod trial_analysis;

pub use cognitive_profile::CognitiveProfile;
pub use config::{AblationConfig, AblationPreset, BenchmarkConfig};
pub use cross_domain_prediction::{
    CrossDomainMatrix, DomainCorrelation, PredictionModel, SharedMechanism,
};
pub use difficulty::{DifficultyModel, DifficultyModelType, difficulty_model_for};
pub use neuromod_correlation::NeuromodCorrelationMatrix;
pub use normative_comparison::{NormativeReport, NormativeScore};
pub use psychometric_report::{BenchmarkDetail, PsychometricReport, ReportSummary};
pub use reliability_analysis::{
    PracticeDirection, PracticeEffect, ReliabilityBattery, ReliabilityClass, TestRetestResult,
    compute_icc, compute_sem, pearson_r,
};
pub use report::{
    BaselineComparison, BenchmarkReport, BenchmarkResult, CompositeScore, ForestPlotRow,
    LearningCurveRow, MetricValue, RtSummary, key_metric_for_benchmark, provenance_table,
};
pub use sat_curves::{SatBattery, SatCurve, SatFit, SatPoint, run_sat_curve};
pub use snapshot::{
    RegressionReport, RegressionResult, RegressionSeverity, RegressionSnapshot, RegressionSummary,
    SNAPSHOT_SCHEMA_VERSION,
};
pub use staircase::{StaircaseConfig, StaircaseResult, StaircaseRule, run_staircase};
pub use trial_analysis::{
    CalibrationResult, ErrorBurst, SpeedAccuracyResult, StrategyShift, TrialAnalysis,
    TrialBlockRow, TrialOutcome,
};

/// Provenance metadata for a psychological benchmark — citation, paradigm, year.
#[derive(Debug, Clone)]
pub struct BenchmarkProvenance {
    /// Experimental paradigm name (e.g., "Stroop Color-Word", "Wisconsin Card Sort").
    pub paradigm: &'static str,
    /// Primary citation in APA-ish format.
    pub citation: &'static str,
    /// Publication year of the primary citation.
    pub year: u16,
    /// DOI if available.
    pub doi: Option<&'static str>,
}

/// A runnable psychological benchmark.
pub trait PsychBenchmark {
    /// Human-readable benchmark name.
    fn name(&self) -> &str;

    /// Run the benchmark with given configuration.
    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult;

    /// Run ablation study with multiple configurations.
    fn run_ablation(&self, configs: &[AblationConfig]) -> Vec<BenchmarkResult> {
        configs
            .iter()
            .map(|ac| {
                let mut bc = ac.base.clone();
                bc.label = Some(ac.name.clone());
                self.run(&bc)
            })
            .collect()
    }

    /// Provenance metadata for this benchmark (citation, paradigm, year).
    ///
    /// Returns `None` by default; benchmarks override to provide citations.
    fn provenance(&self) -> Option<BenchmarkProvenance> {
        None
    }
}

/// Adaptive trial runner: increases trial count until precision target is met.
pub struct AdaptiveRunner;

impl AdaptiveRunner {
    /// Run a benchmark with adaptive trial counts.
    ///
    /// If `config.adaptive_trials` is false, simply delegates to `bench.run()`.
    /// Otherwise, starts with `config.min_trials`, checks the key metric's CI
    /// half-width, and doubles trials until `precision_target` is met or
    /// `max_trials` is reached.
    pub fn run_adaptive(
        bench: &dyn PsychBenchmark,
        config: &BenchmarkConfig,
        key_metric: &str,
    ) -> BenchmarkResult {
        if !config.adaptive_trials {
            return bench.run(config);
        }

        let mut trials = config.min_trials;
        loop {
            let mut cfg = config.clone();
            cfg.trials_per_condition = trials;
            let result = bench.run(&cfg);

            if trials >= config.max_trials {
                return result;
            }

            // Check precision: CI half-width / |mean| < precision_target
            if let Some(metric) = result.metrics.get(key_metric) {
                let half_width = (metric.ci_upper - metric.ci_lower) / 2.0;
                let target = config.precision_target * metric.mean.abs();
                if target > 0.0 && half_width <= target {
                    return result;
                }
            } else {
                // Key metric not found — return what we have
                return result;
            }

            // Double trials (capped at max)
            trials = (trials * 2).min(config.max_trials);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal benchmark with no provenance (tests default).
    struct NoProv;
    impl PsychBenchmark for NoProv {
        fn name(&self) -> &str {
            "NoProv"
        }
        fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
            BenchmarkResult::new("NoProv", None)
        }
    }

    /// Benchmark with provenance populated.
    struct WithProv;
    impl PsychBenchmark for WithProv {
        fn name(&self) -> &str {
            "WithProv"
        }
        fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
            BenchmarkResult::new("WithProv", None)
        }
        fn provenance(&self) -> Option<BenchmarkProvenance> {
            Some(BenchmarkProvenance {
                paradigm: "Test Paradigm",
                citation: "Test (2024)",
                year: 2024,
                doi: Some("10.1234/test"),
            })
        }
    }

    #[test]
    fn test_provenance_default_none() {
        let b = NoProv;
        assert!(b.provenance().is_none());
    }

    #[test]
    fn test_provenance_populated() {
        let b = WithProv;
        let p = b.provenance().unwrap();
        assert_eq!(p.paradigm, "Test Paradigm");
        assert_eq!(p.citation, "Test (2024)");
        assert_eq!(p.year, 2024);
        assert_eq!(p.doi, Some("10.1234/test"));
    }

    #[test]
    fn test_all_benchmarks_have_provenance() {
        // Instantiate all 67 benchmarks and verify each has provenance
        use crate::benchmarks::affect::*;
        use crate::benchmarks::attention::*;
        use crate::benchmarks::binding::*;
        use crate::benchmarks::butlin::*;
        use crate::benchmarks::causal_reasoning::*;
        use crate::benchmarks::clinical::*;
        use crate::benchmarks::coding::*;
        use crate::benchmarks::cogbench::*;
        use crate::benchmarks::consciousness::*;
        use crate::benchmarks::creativity::*;
        use crate::benchmarks::executive::*;
        use crate::benchmarks::inhibition::*;
        use crate::benchmarks::institutional_reasoning;
        use crate::benchmarks::language::*;
        use crate::benchmarks::memory_agent::*;
        use crate::benchmarks::metacognition::*;
        use crate::benchmarks::motor::*;
        use crate::benchmarks::reasoning::*;
        use crate::benchmarks::security::*;
        use crate::benchmarks::social::*;
        use crate::benchmarks::spatial::*;
        use crate::benchmarks::speech::*;
        use crate::benchmarks::substrate::*;
        use crate::benchmarks::sustained_attention::*;
        use crate::benchmarks::tombench::*;
        use crate::benchmarks::worm::*;

        let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
            // WorM
            Box::new(BindingBenchmark),
            Box::new(ChangeDetectionBenchmark),
            Box::new(DigitSpanBenchmark),
            Box::new(NBackBenchmark),
            Box::new(SerialRecallBenchmark),
            Box::new(SpatialUpdatingBenchmark),
            // CogBench
            Box::new(BartBenchmark),
            Box::new(HorizonBenchmark),
            Box::new(InstrumentalLearningBenchmark),
            Box::new(ProbabilisticReasoningBenchmark),
            Box::new(RestlessBanditBenchmark),
            Box::new(ReversalLearningBenchmark),
            Box::new(TemporalDiscountingBenchmark),
            Box::new(TwoStepBenchmark),
            // Executive
            Box::new(StroopBenchmark),
            Box::new(FlankerBenchmark),
            Box::new(WisconsinCardSortingBenchmark),
            Box::new(IowaGamblingBenchmark),
            Box::new(TowerOfLondonBenchmark),
            Box::new(RavensProgressiveMatricesBenchmark),
            Box::new(DualTaskBenchmark),
            // ToMBench
            Box::new(FalseBeliefBenchmark),
            Box::new(FauxPasBenchmark),
            Box::new(HintingBenchmark),
            Box::new(PersuasionBenchmark),
            Box::new(StrangeStoryBenchmark),
            // MemoryAgent
            Box::new(AccurateRetrievalBenchmark),
            Box::new(ConflictResolutionBenchmark),
            Box::new(LongRangeBenchmark),
            Box::new(ProspectiveMemoryBenchmark),
            Box::new(TestTimeLearningBenchmark),
            // Metacognition
            Box::new(MetacognitiveCalibrationBenchmark),
            Box::new(FeelingOfKnowingBenchmark),
            Box::new(ChangeBlindnessBenchmark),
            // Affect
            Box::new(EmotionalStroopBenchmark),
            Box::new(MoodCongruentRecallBenchmark),
            Box::new(ValenceClassificationBenchmark),
            // Creativity
            Box::new(AlternateUsesBenchmark),
            Box::new(RemoteAssociatesBenchmark),
            Box::new(DivergentThinkingBenchmark),
            Box::new(ConceptualBlendingBenchmark),
            // Butlin
            Box::new(ButlinIndicatorSuite),
            // Inhibition
            Box::new(GoNoGoBenchmark),
            Box::new(StopSignalBenchmark),
            // Attention
            Box::new(AttentionalBlinkBenchmark),
            Box::new(VisualSearchBenchmark),
            Box::new(MismatchNegativityBenchmark),
            // Reasoning
            Box::new(ArcFluidBenchmark),
            Box::new(ArcCompositionalBenchmark),
            Box::new(ArcAnalogyBenchmark),
            Box::new(ArcAbductiveBenchmark),
            Box::new(ArcChainBenchmark),
            Box::new(ArcNoiseBenchmark),
            Box::new(ArcFewShotBenchmark),
            Box::new(ArcScalingBenchmark),
            Box::new(ArcRsaBenchmark),
            Box::new(ArcAlgebraBenchmark),
            Box::new(ArcStaircaseBenchmark),
            // Sustained Attention
            Box::new(SartBenchmark),
            Box::new(PvtBenchmark),
            Box::new(CptBenchmark),
            // Motor
            Box::new(SrttBenchmark),
            Box::new(FittsLawBenchmark),
            Box::new(BimanualBenchmark),
            Box::new(ProprioceptiveDriftBenchmark),
            // Language
            Box::new(GardenPathBenchmark),
            Box::new(SemanticCoherenceBenchmark),
            Box::new(LexicalDecisionBenchmark),
            Box::new(SemanticPrimingBenchmark),
            // Social
            Box::new(RmeBenchmark),
            Box::new(UltimatumGameBenchmark),
            Box::new(SocialNormBenchmark),
            Box::new(PrisonersDilemmaBenchmark),
            Box::new(PublicGoodsBenchmark),
            Box::new(DictatorGameBenchmark),
            Box::new(MachiavelliBenchmark),
            // Binding
            Box::new(TemporalOrderBenchmark),
            Box::new(CrossModalBindingBenchmark),
            // Spatial
            Box::new(MentalRotationBenchmark),
            Box::new(SpatialPathUpdatingBenchmark),
            Box::new(LandmarkBindingBenchmark),
            Box::new(PerspectiveTakingBenchmark),
            // Causal Reasoning
            Box::new(CausalChainBenchmark),
            Box::new(ConfoundDetectionBenchmark),
            Box::new(InterventionEffectBenchmark),
            // Speech
            Box::new(PhonemeDiscriminationBenchmark),
            Box::new(VotContinuumBenchmark),
            // Consciousness
            Box::new(BlindSightBenchmark),
            Box::new(BinocularRivalryBenchmark),
            // Substrate
            Box::new(SubstrateTransferBenchmark),
            Box::new(SubstrateDegradationBenchmark),
            // Institutional Reasoning
            Box::new(institutional_reasoning::InstitutionalReasoningBenchmark),
            Box::new(institutional_reasoning::AnalogicalReasoningBenchmark),
            Box::new(institutional_reasoning::CausalChainBenchmark),
            Box::new(institutional_reasoning::CounterfactualReasoningBenchmark),
            Box::new(institutional_reasoning::WeightedDecompositionBenchmark),
            Box::new(institutional_reasoning::InstitutionalStabilityBenchmark),
            Box::new(institutional_reasoning::InstitutionalIsomorphismBenchmark),
            // Clinical/Therapeutic
            Box::new(EmpathicAccuracyBenchmark),
            Box::new(TherapeuticResponseBenchmark),
            Box::new(AllianceMaintenanceBenchmark),
            Box::new(CrisisDetectionBenchmark),
            Box::new(CognitiveDistortionBenchmark),
            Box::new(MotivationalInterviewingBenchmark),
            // Security (HDC-FHE)
            Box::new(EncryptedClassificationBenchmark),
            Box::new(CollectiveAggregationBenchmark),
            Box::new(EncryptedLearningBenchmark),
            Box::new(CrossMaskPrivacyBenchmark),
            Box::new(EncryptedBindingBenchmark),
            Box::new(ScalingAnalysisBenchmark),
            // Coding
            Box::new(HumanEvalMiniBenchmark),
            Box::new(BugDetectionBenchmark),
        ];

        let mut missing = Vec::new();
        for b in &benchmarks {
            if b.provenance().is_none() {
                missing.push(b.name().to_string());
            }
        }
        assert!(
            missing.is_empty(),
            "Benchmarks missing provenance: {:?}",
            missing
        );
    }

    #[test]
    fn test_provenance_table_format() {
        let b = WithProv;
        let refs: Vec<&dyn PsychBenchmark> = vec![&b];
        let table = provenance_table(&refs);
        assert!(table.contains("| WithProv |"));
        assert!(table.contains("Test (2024)"));
        assert!(table.contains("10.1234/test"));
    }

    #[test]
    fn test_provenance_years_reasonable() {
        let b = WithProv;
        let p = b.provenance().unwrap();
        assert!(p.year >= 1850, "Year too early: {}", p.year);
        assert!(p.year <= 2026, "Year too late: {}", p.year);
    }

    #[test]
    fn test_provenance_paradigm_non_empty() {
        let b = WithProv;
        let p = b.provenance().unwrap();
        assert!(!p.paradigm.is_empty());
        assert!(!p.citation.is_empty());
    }

    #[test]
    fn test_html_includes_citations() {
        let entries = vec![(
            "TestBench",
            BenchmarkProvenance {
                paradigm: "Test Paradigm",
                citation: "Tester (2024)",
                year: 2024,
                doi: Some("10.1234/test"),
            },
        )];
        let mut html = String::new();
        html::write_provenance_section(&mut html, &entries);
        assert!(html.contains("Citations"));
        assert!(html.contains("TestBench"));
        assert!(html.contains("10.1234/test"));
    }

    // ──── Adaptive trial tests ────

    #[test]
    fn test_adaptive_defaults_false() {
        let config = BenchmarkConfig::default();
        assert!(!config.adaptive_trials);
        assert_eq!(config.min_trials, 10);
        assert_eq!(config.max_trials, 200);
        assert!((config.precision_target - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_serde_roundtrip() {
        let mut config = BenchmarkConfig::default();
        config.adaptive_trials = true;
        config.min_trials = 15;
        config.max_trials = 100;
        config.precision_target = 0.03;
        let json = serde_json::to_string(&config).unwrap();
        let loaded: BenchmarkConfig = serde_json::from_str(&json).unwrap();
        assert!(loaded.adaptive_trials);
        assert_eq!(loaded.min_trials, 15);
        assert_eq!(loaded.max_trials, 100);
        assert!((loaded.precision_target - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_backward_compat() {
        // Old JSON without adaptive fields should deserialize with defaults
        let json = r#"{
            "dimension": 512,
            "trials_per_condition": 20,
            "working_memory_capacity": 7,
            "seed": 42,
            "enable_social": true,
            "enable_fep": true,
            "planning_horizon": 3,
            "action_temperature": 1.0,
            "label": null,
            "time_pressure": 0.0
        }"#;
        let config: BenchmarkConfig = serde_json::from_str(json).unwrap();
        assert!(!config.adaptive_trials);
        assert_eq!(config.min_trials, 10);
        assert_eq!(config.max_trials, 200);
    }

    #[test]
    fn test_adaptive_disabled_passthrough() {
        let b = NoProv;
        let config = BenchmarkConfig::default(); // adaptive_trials = false
        let result = AdaptiveRunner::run_adaptive(&b, &config, "some_metric");
        assert_eq!(result.benchmark, "NoProv");
    }

    #[test]
    fn test_adaptive_increases_trials() {
        use crate::benchmarks::executive::StroopBenchmark;
        let config = BenchmarkConfig {
            adaptive_trials: true,
            min_trials: 5,
            max_trials: 20,
            precision_target: 0.001, // Very tight — should force increase
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = AdaptiveRunner::run_adaptive(&StroopBenchmark, &config, "stroop_effect");
        // Should have been re-run with more trials (metric present and finite)
        assert!(result.metrics.contains_key("stroop_effect"));
    }

    #[test]
    fn test_adaptive_respects_max() {
        use crate::benchmarks::executive::StroopBenchmark;
        let config = BenchmarkConfig {
            adaptive_trials: true,
            min_trials: 5,
            max_trials: 10,
            precision_target: 0.0001, // Impossible to achieve
            trials_per_condition: 5,
            ..Default::default()
        };
        // Should terminate (not infinite loop) at max_trials
        let result = AdaptiveRunner::run_adaptive(&StroopBenchmark, &config, "stroop_effect");
        assert!(result.metrics.contains_key("stroop_effect"));
    }

    #[test]
    fn test_adaptive_stops_at_precision() {
        use crate::benchmarks::executive::StroopBenchmark;
        let config = BenchmarkConfig {
            adaptive_trials: true,
            min_trials: 5,
            max_trials: 200,
            precision_target: 0.5, // Very loose — should stop at min_trials
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = AdaptiveRunner::run_adaptive(&StroopBenchmark, &config, "stroop_effect");
        assert!(result.metrics.contains_key("stroop_effect"));
    }

    #[test]
    fn test_adaptive_deterministic() {
        use crate::benchmarks::executive::StroopBenchmark;
        let config = BenchmarkConfig {
            adaptive_trials: true,
            min_trials: 5,
            max_trials: 20,
            precision_target: 0.1,
            trials_per_condition: 5,
            seed: 42,
            ..Default::default()
        };
        let r1 = AdaptiveRunner::run_adaptive(&StroopBenchmark, &config, "stroop_effect");
        let r2 = AdaptiveRunner::run_adaptive(&StroopBenchmark, &config, "stroop_effect");
        let m1 = r1.metrics["stroop_effect"].mean;
        let m2 = r2.metrics["stroop_effect"].mean;
        assert!(
            (m1 - m2).abs() < 1e-10,
            "same seed should produce same result: {} vs {}",
            m1,
            m2
        );
    }

    #[test]
    fn test_ssm_backend_flag_default_off() {
        let config = BenchmarkConfig::default();
        assert!(!config.ssm_backend, "ssm_backend should default to false");
    }

    #[test]
    fn test_ssm_backend_flag_enables() {
        use crate::benchmarks::memory_agent::ProspectiveMemoryBenchmark;
        // Without SSM flag, should run normally
        let config_off = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ssm_backend: false,
            ..Default::default()
        };
        let result_off = ProspectiveMemoryBenchmark.run(&config_off);
        assert!(result_off.metrics["pm_hit_rate"].mean.is_finite());

        // With SSM flag, should also run normally
        let config_on = BenchmarkConfig {
            ssm_backend: true,
            ..config_off.clone()
        };
        let result_on = ProspectiveMemoryBenchmark.run(&config_on);
        assert!(result_on.metrics["pm_hit_rate"].mean.is_finite());

        // Both should produce valid results (may differ in values)
        assert!(result_off.metrics.contains_key("pm_hit_rate"));
        assert!(result_on.metrics.contains_key("pm_hit_rate"));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: verify normative comparison produces scores for all
//! benchmarks that have matching baselines.
//!
//! Runs all benchmarks with minimal config, builds a NormativeReport,
//! and checks that the expected benchmark set produces valid z-scores.

use symthaea_psych_bench::benchmarks::affect::*;
use symthaea_psych_bench::benchmarks::attention::*;
use symthaea_psych_bench::benchmarks::butlin::*;
use symthaea_psych_bench::benchmarks::coding::*;
use symthaea_psych_bench::benchmarks::cogbench::*;
use symthaea_psych_bench::benchmarks::creativity::*;
use symthaea_psych_bench::benchmarks::executive::*;
use symthaea_psych_bench::benchmarks::inhibition::*;
use symthaea_psych_bench::benchmarks::language::*;
use symthaea_psych_bench::benchmarks::memory_agent::*;
use symthaea_psych_bench::benchmarks::metacognition::*;
use symthaea_psych_bench::benchmarks::motor::*;
use symthaea_psych_bench::benchmarks::neuromod::*;
use symthaea_psych_bench::benchmarks::reasoning::*;
use symthaea_psych_bench::benchmarks::social::*;
use symthaea_psych_bench::benchmarks::sustained_attention::*;
use symthaea_psych_bench::benchmarks::tombench::*;
use symthaea_psych_bench::benchmarks::worm::*;
use symthaea_psych_bench::harness::normative_comparison::NormativeReport;
use symthaea_psych_bench::harness::report::BenchmarkReport;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

#[test]
fn test_normative_coverage() {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };

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
        // Affect
        Box::new(EmotionalStroopBenchmark),
        Box::new(MoodCongruentRecallBenchmark),
        Box::new(ValenceClassificationBenchmark),
        // Creativity
        Box::new(AlternateUsesBenchmark),
        Box::new(RemoteAssociatesBenchmark),
        Box::new(DivergentThinkingBenchmark),
        Box::new(ConceptualBlendingBenchmark),
        Box::new(InsightProblemBenchmark),
        // Butlin
        Box::new(ButlinIndicatorSuite),
        // Inhibition
        Box::new(GoNoGoBenchmark),
        Box::new(StopSignalBenchmark),
        // Attention
        Box::new(AttentionalBlinkBenchmark),
        Box::new(VisualSearchBenchmark),
        // Reasoning
        Box::new(ArcFluidBenchmark),
        Box::new(ArcAbductiveBenchmark),
        Box::new(ArcAlgebraBenchmark),
        Box::new(ArcAnalogyBenchmark),
        Box::new(ArcChainBenchmark),
        Box::new(ArcCompositionalBenchmark),
        Box::new(ArcFewShotBenchmark),
        Box::new(ArcNoiseBenchmark),
        Box::new(ArcRsaBenchmark),
        Box::new(ArcScalingBenchmark),
        Box::new(ArcStaircaseBenchmark),
        // Sustained Attention
        Box::new(SartBenchmark),
        Box::new(PvtBenchmark),
        Box::new(CptBenchmark),
        // Motor
        Box::new(SrttBenchmark),
        Box::new(FittsLawBenchmark),
        Box::new(BimanualBenchmark),
        // Language
        Box::new(GardenPathBenchmark),
        Box::new(SemanticCoherenceBenchmark),
        Box::new(LexicalDecisionBenchmark),
        Box::new(SemanticPrimingBenchmark),
        // Social
        Box::new(RmeBenchmark),
        Box::new(UltimatumGameBenchmark),
        Box::new(SocialNormBenchmark),
        // Neuromod
        Box::new(PharmacologicalChallengeBenchmark),
        Box::new(InjectionChallengeBenchmark),
        Box::new(AllostaticStressBenchmark),
        Box::new(RewardLearningBenchmark),
        Box::new(YerkesDodsonBenchmark),
        Box::new(AttentionNetworkBenchmark),
        Box::new(MoodInductionBenchmark),
        Box::new(PharmacologicalAblationBenchmark),
        Box::new(BehavioralKnockoutBenchmark),
        Box::new(ConsciousnessPharmacologyBenchmark),
        // Coding
        Box::new(HumanEvalMiniBenchmark),
        Box::new(BugDetectionBenchmark),
    ];

    // Build a full report from all benchmarks
    let mut report = BenchmarkReport::new();
    for bench in &benchmarks {
        report.add(bench.run(&config));
    }

    let normative = NormativeReport::from_report(&report);

    // Benchmarks that MUST produce normative scores (they have baselines with SD)
    let expected_benchmarks = [
        "Executive::Stroop",
        "Executive::Flanker",
        "Executive::DualTask",
        "WorM::N-back",
        "WorM::DigitSpan",
        "WorM::ChangeDetection",
        "WorM::SerialRecall",
        "SustainedAttention::SART",
        "SustainedAttention::PVT",
        "SustainedAttention::CPT",
        "Metacognition::Calibration",
        "Metacognition::FeelingOfKnowing",
        "ToMBench::FalseBelief",
        "Inhibition::GoNoGo",
        "Attention::VisualSearch",
        "Attention::AttentionalBlink",
        "Motor::SRTT",
        "Motor::Bimanual",
        // Language (fixed baseline key mappings)
        "Language::LexicalDecision",
        "Language::SemanticPriming",
        "Language::SemanticCoherence",
        // Social (fixed baseline key mappings)
        "Social::UltimatumGame",
        "Social::SocialNorm",
        // Reasoning (ARC variants)
        "Reasoning::ArcFluid",
        "Reasoning::ArcAbductive",
        "Reasoning::ArcAlgebra",
        "Reasoning::ArcAnalogy",
        "Reasoning::ArcChain",
        "Reasoning::ArcCompositional",
        "Reasoning::ArcFewShot",
        "Reasoning::ArcNoise",
        "Reasoning::ArcRSA",
        "Reasoning::ArcScaling",
        "Reasoning::ArcStaircase",
        // Neuromod
        "Neuromod::PharmacologicalChallenge",
        "Neuromod::InjectionChallenge",
        "Neuromod::AllostaticStress",
        "Neuromod::RewardLearning",
        "Neuromod::YerkesDodson",
        "Neuromod::AttentionNetwork",
        "Neuromod::MoodInduction",
        "Neuromod::PharmacologicalAblation",
        "Neuromod::BehavioralKnockout",
        "Neuromod::ConsciousnessPharmacology",
    ];

    let covered: Vec<&str> = normative
        .scores
        .iter()
        .map(|s| s.benchmark.as_str())
        .collect();

    let mut missing = Vec::new();
    for expected in &expected_benchmarks {
        if !covered.contains(expected) {
            missing.push(*expected);
        }
    }

    assert!(
        missing.is_empty(),
        "Normative comparison missing for {} benchmarks: {:?}\nCovered: {:?}",
        missing.len(),
        missing,
        covered,
    );

    // Verify all scores have valid z-scores and percentiles
    for score in &normative.scores {
        assert!(
            score.z_score.is_finite(),
            "{}: z-score not finite: {}",
            score.benchmark,
            score.z_score
        );
        assert!(
            score.percentile >= 0.0 && score.percentile <= 100.0,
            "{}: percentile out of range: {}",
            score.benchmark,
            score.percentile
        );
        assert!(
            !score.interpretation.is_empty(),
            "{}: empty interpretation",
            score.benchmark
        );
        assert!(
            score.human_sd > 0.0,
            "{}: human SD must be positive: {}",
            score.benchmark,
            score.human_sd
        );
    }

    // Overall statistics should be finite
    assert!(normative.overall_mean_z.is_finite());
    assert!(normative.overall_percentile >= 0.0 && normative.overall_percentile <= 100.0);
}

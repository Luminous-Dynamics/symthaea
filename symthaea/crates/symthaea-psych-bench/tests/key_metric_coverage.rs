// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: verify that every benchmark's key metric actually exists
//! in the output of `bench.run()`.
//!
//! Uses minimal config (dim=128, trials=3) for speed.

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
use symthaea_psych_bench::benchmarks::reasoning::*;
use symthaea_psych_bench::benchmarks::social::*;
use symthaea_psych_bench::benchmarks::sustained_attention::*;
use symthaea_psych_bench::benchmarks::tombench::*;
use symthaea_psych_bench::benchmarks::worm::*;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark, key_metric_for_benchmark};

#[test]
fn test_key_metric_exists_in_output() {
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
        // Language
        Box::new(GardenPathBenchmark),
        Box::new(SemanticCoherenceBenchmark),
        Box::new(LexicalDecisionBenchmark),
        Box::new(SemanticPrimingBenchmark),
        // Social
        Box::new(RmeBenchmark),
        Box::new(UltimatumGameBenchmark),
        Box::new(SocialNormBenchmark),
        // Coding
        Box::new(HumanEvalMiniBenchmark),
        Box::new(BugDetectionBenchmark),
    ];

    let mut failures = Vec::new();

    for bench in &benchmarks {
        let name = bench.name();
        let result = bench.run(&config);
        let key = key_metric_for_benchmark(name);

        if !result.metrics.contains_key(key) {
            let available: Vec<&str> = result.metrics.keys().map(|k| k.as_str()).collect();
            failures.push(format!(
                "{}: expected metric '{}', available: {:?}",
                name, key, available,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Key metric missing from benchmark output:\n{}",
        failures.join("\n"),
    );
}

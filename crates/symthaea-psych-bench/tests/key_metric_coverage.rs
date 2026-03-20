//! Integration test: verify that every benchmark's key metric actually exists
//! in the output of `bench.run()`.
//!
//! Uses minimal config (dim=128, trials=3) for speed.

use symthaea_psych_bench::benchmarks::affect::*;
use symthaea_psych_bench::benchmarks::attention::*;
use symthaea_psych_bench::benchmarks::binding::*;
use symthaea_psych_bench::benchmarks::butlin::*;
use symthaea_psych_bench::benchmarks::causal_reasoning;
use symthaea_psych_bench::benchmarks::clinical::*;
use symthaea_psych_bench::benchmarks::cogbench::*;
use symthaea_psych_bench::benchmarks::consciousness::*;
use symthaea_psych_bench::benchmarks::creativity::*;
use symthaea_psych_bench::benchmarks::executive::*;
use symthaea_psych_bench::benchmarks::inhibition::*;
use symthaea_psych_bench::benchmarks::institutional_reasoning;
use symthaea_psych_bench::benchmarks::language::*;
use symthaea_psych_bench::benchmarks::mathematics::*;
use symthaea_psych_bench::benchmarks::memory_agent::*;
use symthaea_psych_bench::benchmarks::metacognition::*;
use symthaea_psych_bench::benchmarks::motor::*;
use symthaea_psych_bench::benchmarks::neuromod::*;
use symthaea_psych_bench::benchmarks::reasoning::*;
use symthaea_psych_bench::benchmarks::security::*;
use symthaea_psych_bench::benchmarks::social::*;
use symthaea_psych_bench::benchmarks::spatial::*;
use symthaea_psych_bench::benchmarks::speech::*;
use symthaea_psych_bench::benchmarks::substrate::*;
use symthaea_psych_bench::benchmarks::sustained_attention::*;
use symthaea_psych_bench::benchmarks::tombench::*;
use symthaea_psych_bench::benchmarks::worm::*;
use symthaea_psych_bench::harness::{key_metric_for_benchmark, BenchmarkConfig, PsychBenchmark};

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
        Box::new(ChangeBlindnessBenchmark),
        // Affect
        Box::new(EmotionalStroopBenchmark),
        Box::new(MoodCongruentRecallBenchmark),
        Box::new(ValenceClassificationBenchmark),
        // Creativity
        Box::new(AlternateUsesBenchmark),
        Box::new(RemoteAssociatesBenchmark),
        Box::new(DivergentThinkingBenchmark),
        // Butlin
        Box::new(ButlinIndicatorSuite),
        // Inhibition
        Box::new(GoNoGoBenchmark),
        Box::new(StopSignalBenchmark),
        Box::new(FlankerInhibitionBenchmark),
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
        Box::new(FeatureConjunctionBenchmark),
        // Spatial
        Box::new(MentalRotationBenchmark),
        Box::new(SpatialPathUpdatingBenchmark),
        Box::new(LandmarkBindingBenchmark),
        Box::new(PerspectiveTakingBenchmark),
        // Causal Reasoning
        Box::new(causal_reasoning::CausalChainBenchmark),
        Box::new(causal_reasoning::ConfoundDetectionBenchmark),
        Box::new(causal_reasoning::InterventionEffectBenchmark),
        // Speech
        Box::new(PhonemeDiscriminationBenchmark),
        Box::new(VotContinuumBenchmark),
        Box::new(CategoricalPerceptionBenchmark),
        // Consciousness
        Box::new(BlindSightBenchmark),
        Box::new(BinocularRivalryBenchmark),
        Box::new(PerceptualCrowdingBenchmark),
        // Substrate
        Box::new(SubstrateTransferBenchmark),
        Box::new(SubstrateDegradationBenchmark),
        Box::new(SubstrateLatencyBenchmark),
        // Clinical/Therapeutic
        Box::new(EmpathicAccuracyBenchmark),
        Box::new(TherapeuticResponseBenchmark),
        Box::new(AllianceMaintenanceBenchmark),
        Box::new(CrisisDetectionBenchmark),
        Box::new(CognitiveDistortionBenchmark),
        Box::new(MotivationalInterviewingBenchmark),
        // Institutional Reasoning
        Box::new(institutional_reasoning::InstitutionalReasoningBenchmark),
        Box::new(institutional_reasoning::AnalogicalReasoningBenchmark),
        Box::new(institutional_reasoning::CausalChainBenchmark),
        Box::new(institutional_reasoning::CounterfactualReasoningBenchmark),
        Box::new(institutional_reasoning::WeightedDecompositionBenchmark),
        Box::new(institutional_reasoning::InstitutionalStabilityBenchmark),
        Box::new(institutional_reasoning::InstitutionalIsomorphismBenchmark),
        // Mathematics
        Box::new(ArithmeticWordProblemsBenchmark),
        Box::new(LinearSystemSolvingBenchmark),
        Box::new(PolynomialRootsBenchmark),
        Box::new(MatrixOperationsBenchmark),
        Box::new(StatisticalInferenceBenchmark),
        Box::new(BayesianReasoningBenchmark),
        Box::new(LogicalDeductionBenchmark),
        Box::new(ConstraintPuzzlesBenchmark),
        Box::new(ProofConstructionBenchmark),
        Box::new(DefiniteIntegralsBenchmark),
        // Security (HDC-FHE)
        Box::new(EncryptedClassificationBenchmark),
        Box::new(CollectiveAggregationBenchmark),
        Box::new(EncryptedLearningBenchmark),
        Box::new(CrossMaskPrivacyBenchmark),
        Box::new(EncryptedBindingBenchmark),
        Box::new(ScalingAnalysisBenchmark),
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
        Box::new(DoseResponseBenchmark),
        Box::new(AntagonistProfilesBenchmark),
        Box::new(ToleranceWithdrawalBenchmark),
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

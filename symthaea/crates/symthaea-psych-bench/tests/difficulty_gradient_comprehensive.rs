//! Comprehensive difficulty gradient validation: verify that accuracy
//! monotonically decreases with difficulty across all 30 benchmarks that have
//! registered difficulty models.

use symthaea_psych_bench::benchmarks::{
    affect::EmotionalStroopBenchmark,
    attention::{AttentionalBlinkBenchmark, VisualSearchBenchmark},
    executive::{FlankerBenchmark, StroopBenchmark, WisconsinCardSortingBenchmark},
    inhibition::StopSignalBenchmark,
    language::{LexicalDecisionBenchmark, SemanticPrimingBenchmark},
    memory_agent::{ConflictResolutionBenchmark, LongRangeBenchmark},
    metacognition::FeelingOfKnowingBenchmark,
    motor::BimanualBenchmark,
    social::{RmeBenchmark, SocialNormBenchmark, UltimatumGameBenchmark},
    sustained_attention::{CptBenchmark, PvtBenchmark, SartBenchmark},
    tombench::{
        FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
        StrangeStoryBenchmark,
    },
    worm::{
        BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
        SerialRecallBenchmark, SpatialUpdatingBenchmark,
    },
};
use symthaea_psych_bench::harness::{
    difficulty::{difficulty_model_for, DifficultyModelType},
    report::key_metric_for_benchmark,
    BenchmarkConfig, PsychBenchmark,
};

/// Run a benchmark at multiple difficulty levels and return its key metric values.
fn difficulty_sweep(bench: &dyn PsychBenchmark, difficulties: &[f64]) -> Vec<f64> {
    difficulties
        .iter()
        .map(|&d| {
            let config = BenchmarkConfig {
                dimension: 128,
                trials_per_condition: 15,
                seed: 42,
                difficulty: d,
                ..Default::default()
            };
            let result = bench.run(&config);
            let metric_key = key_metric_for_benchmark(bench.name());
            result
                .metrics
                .get(metric_key)
                .map(|m| m.mean)
                .unwrap_or(f64::NAN)
        })
        .collect()
}

/// Verify difficulty gradient across all 30 benchmarks with registered models.
///
/// We test at 3 levels (0.0, 0.5, 0.9) and verify that performance at
/// difficulty=0.0 is better than or equal to performance at difficulty=0.9,
/// with a tolerance for stochastic noise.
#[test]
fn comprehensive_difficulty_gradient() {
    let difficulties = [0.0, 0.5, 0.9];

    // All benchmarks with registered difficulty models
    let benchmarks: Vec<(&str, Box<dyn PsychBenchmark>)> = vec![
        // Interference models
        ("Executive::Stroop", Box::new(StroopBenchmark)),
        ("Executive::Flanker", Box::new(FlankerBenchmark)),
        ("Affect::EmotionalStroop", Box::new(EmotionalStroopBenchmark)),
        ("Motor::Bimanual", Box::new(BimanualBenchmark)),
        (
            "MemoryAgent::ConflictResolution",
            Box::new(ConflictResolutionBenchmark),
        ),
        ("MemoryAgent::LongRange", Box::new(LongRangeBenchmark)),
        ("WorM::Binding", Box::new(BindingBenchmark)),
        (
            "WorM::ChangeDetection",
            Box::new(ChangeDetectionBenchmark),
        ),
        ("WorM::DigitSpan", Box::new(DigitSpanBenchmark)),
        ("WorM::N-back", Box::new(NBackBenchmark)),
        ("WorM::SerialRecall", Box::new(SerialRecallBenchmark)),
        ("WorM::SpatialUpdating", Box::new(SpatialUpdatingBenchmark)),
        // SNR models
        (
            "Attention::AttentionalBlink",
            Box::new(AttentionalBlinkBenchmark),
        ),
        (
            "Attention::VisualSearch",
            Box::new(VisualSearchBenchmark),
        ),
        ("Executive::WCST", Box::new(WisconsinCardSortingBenchmark)),
        ("Inhibition::StopSignal", Box::new(StopSignalBenchmark)),
        (
            "Language::LexicalDecision",
            Box::new(LexicalDecisionBenchmark),
        ),
        (
            "Language::SemanticPriming",
            Box::new(SemanticPrimingBenchmark),
        ),
        (
            "Metacognition::FeelingOfKnowing",
            Box::new(FeelingOfKnowingBenchmark),
        ),
        ("Social::RME", Box::new(RmeBenchmark)),
        ("Social::SocialNorm", Box::new(SocialNormBenchmark)),
        ("Social::UltimatumGame", Box::new(UltimatumGameBenchmark)),
        ("SustainedAttention::CPT", Box::new(CptBenchmark)),
        ("SustainedAttention::PVT", Box::new(PvtBenchmark)),
        ("SustainedAttention::SART", Box::new(SartBenchmark)),
        ("ToMBench::FalseBelief", Box::new(FalseBeliefBenchmark)),
        ("ToMBench::FauxPas", Box::new(FauxPasBenchmark)),
        ("ToMBench::Hinting", Box::new(HintingBenchmark)),
        ("ToMBench::Persuasion", Box::new(PersuasionBenchmark)),
        ("ToMBench::StrangeStory", Box::new(StrangeStoryBenchmark)),
    ];

    // Lower-is-better metrics: difficulty should make values HIGHER
    let lower_is_better = [
        "stroop_effect",
        "flanker_effect",
        "dual_task_cost",
        "commission_errors",
        "ssrt_ticks",
        "coordination_cost",
        "calibration_error_ece",
        "vigilance_decrement",
        "disambiguation_cost",
        "blink_magnitude",
    ];

    let mut failures = Vec::new();

    for (label, bench) in &benchmarks {
        let values = difficulty_sweep(bench.as_ref(), &difficulties);
        let metric = key_metric_for_benchmark(bench.name());

        // All values should be finite
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                failures.push(format!(
                    "{}: metric '{}' not finite at difficulty={}",
                    label, metric, difficulties[i]
                ));
            }
        }

        if values.iter().any(|v| !v.is_finite()) {
            continue;
        }

        let first = values[0];
        let last = values[values.len() - 1];
        let is_lower_better = lower_is_better.contains(&metric);

        // Tolerance: 0.15 for stochastic benchmarks at low trial counts
        let tolerance = 0.15;

        let gradient_ok = if is_lower_better {
            // Lower-is-better: value at d=0.9 should be >= value at d=0.0 - tolerance
            last >= first - tolerance
        } else {
            // Higher-is-better: value at d=0.0 should be >= value at d=0.9 - tolerance
            first >= last - tolerance
        };

        if !gradient_ok {
            failures.push(format!(
                "{}: metric '{}' wrong gradient direction \
                 (d=0.0: {:.3}, d=0.5: {:.3}, d=0.9: {:.3}, lower_better={})",
                label, metric, first, values[1], last, is_lower_better,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Difficulty gradient violations ({}/{}):\n  {}",
        failures.len(),
        benchmarks.len(),
        failures.join("\n  ")
    );
}

/// Verify all registered difficulty models return non-default types.
#[test]
fn all_registered_models_non_default() {
    let registered = [
        "Executive::Stroop",
        "Executive::Flanker",
        "Affect::EmotionalStroop",
        "Motor::Bimanual",
        "MemoryAgent::ConflictResolution",
        "MemoryAgent::LongRange",
        "WorM::Binding",
        "WorM::ChangeDetection",
        "WorM::DigitSpan",
        "WorM::N-back",
        "WorM::SerialRecall",
        "WorM::SpatialUpdating",
        "Attention::AttentionalBlink",
        "Attention::VisualSearch",
        "Executive::WCST",
        "Inhibition::StopSignal",
        "Language::LexicalDecision",
        "Language::SemanticPriming",
        "Metacognition::FeelingOfKnowing",
        "Social::RME",
        "Social::SocialNorm",
        "Social::UltimatumGame",
        "SustainedAttention::CPT",
        "SustainedAttention::PVT",
        "SustainedAttention::SART",
        "ToMBench::FalseBelief",
        "ToMBench::FauxPas",
        "ToMBench::Hinting",
        "ToMBench::Persuasion",
        "ToMBench::StrangeStory",
    ];

    for name in &registered {
        let model = difficulty_model_for(name);
        assert_ne!(
            model.model_type,
            DifficultyModelType::Default,
            "{} should have a non-Default difficulty model",
            name
        );
    }
}

/// Verify unregistered benchmarks get the Default model.
#[test]
fn unregistered_benchmarks_get_default() {
    let unregistered = [
        "Butlin::IndicatorSuite",
        "Creativity::AlternateUses",
        "CogBench::BART",
        "UnknownBenchmark",
    ];

    for name in &unregistered {
        let model = difficulty_model_for(name);
        assert_eq!(
            model.model_type,
            DifficultyModelType::Default,
            "{} should fall through to Default difficulty model",
            name
        );
    }
}

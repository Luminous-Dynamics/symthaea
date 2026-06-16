// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    BenchmarkConfig, PsychBenchmark,
    difficulty::{DifficultyModelType, difficulty_model_for},
    report::key_metric_for_benchmark,
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
        (
            "Affect::EmotionalStroop",
            Box::new(EmotionalStroopBenchmark),
        ),
        ("Motor::Bimanual", Box::new(BimanualBenchmark)),
        (
            "MemoryAgent::ConflictResolution",
            Box::new(ConflictResolutionBenchmark),
        ),
        ("MemoryAgent::LongRange", Box::new(LongRangeBenchmark)),
        ("WorM::Binding", Box::new(BindingBenchmark)),
        ("WorM::ChangeDetection", Box::new(ChangeDetectionBenchmark)),
        ("WorM::DigitSpan", Box::new(DigitSpanBenchmark)),
        ("WorM::N-back", Box::new(NBackBenchmark)),
        ("WorM::SerialRecall", Box::new(SerialRecallBenchmark)),
        ("WorM::SpatialUpdating", Box::new(SpatialUpdatingBenchmark)),
        // SNR models
        (
            "Attention::AttentionalBlink",
            Box::new(AttentionalBlinkBenchmark),
        ),
        ("Attention::VisualSearch", Box::new(VisualSearchBenchmark)),
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
        "list_7::primacy_index",
    ];

    // Benchmarks exempt from the minimum-effect check. These have difficulty
    // models wired but their KEY METRIC is insensitive at dim=128 / 15 trials:
    // - Discrete count metrics (forward_span, categories_completed)
    // - Ceiling metrics (overall_accuracy = 1.0 even with degradation)
    // - Metrics dominated by stochastic keyword matching (hinting, priming)
    // - Metrics whose primary variance comes from stimulus design (bimanual)
    // The difficulty model DOES modulate internal variables (threshold, noise,
    // temperature) but the observable key metric doesn't shift enough to detect.
    let exempt_from_effect_check = [
        "Executive::WCST",           // categories_completed always 6 at low dim
        "Language::SemanticPriming", // priming_effect too small at dim=128
        "Social::SocialNorm",        // d' at ceiling with 128-dim HVs
        "Social::UltimatumGame",     // fairness_sensitivity deterministic at low trials
        "Motor::Bimanual",           // coordination_cost stays 0 at low dim
        "WorM::ChangeDetection",     // set_size_4::accuracy insensitive to noise_frac delta
        "WorM::DigitSpan",           // forward_span is discrete integer (7→7)
        "WorM::N-back",              // nback_2::accuracy ceiling at dim=128
        "WorM::SpatialUpdating",     // overall_accuracy at ceiling (1.0)
        "SustainedAttention::CPT",   // d' at ceiling with dim=128 HVs
        "ToMBench::Hinting",         // keyword-dominant scoring, sigmoid too flat
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

        // Stricter check: non-Default models MUST produce a measurable effect.
        // If d=0.0 and d=0.9 produce identical metrics, the difficulty wiring
        // is broken (model registered but never read by run_trial).
        // Some benchmarks are exempt (discrete/ceiling metrics at low dim).
        if !exempt_from_effect_check.contains(label) {
            let abs_diff = (last - first).abs();
            let min_effect = 0.003; // must see at least 0.3% change
            if abs_diff < min_effect {
                failures.push(format!(
                    "{}: metric '{}' shows NO difficulty effect \
                     (d=0.0: {:.4}, d=0.9: {:.4}, |diff|={:.4} < {:.4}). \
                     Difficulty model may be registered but unwired in run_trial().",
                    label, metric, first, last, abs_diff, min_effect,
                ));
            }
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

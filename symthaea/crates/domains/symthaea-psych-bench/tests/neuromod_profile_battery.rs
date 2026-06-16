// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Integration Test: Neuromodulator-Driven Individual Differences
// ==================================================================================
//
// Data structure tests run without features.
// Live cognitive loop tests require `symthaea-backend` feature and are #[ignore].
// ==================================================================================

use symthaea_psych_bench::harness::neuromod_profiles::{
    DissociationMatrix, NeuromodProfile, ProfilePerformance,
};

// ── Profile Data Structures ─────────────────────────────────────────

#[test]
fn test_standard_profiles() {
    let profiles = NeuromodProfile::standard_profiles();
    assert_eq!(profiles.len(), 5);

    assert!(profiles[0].da.is_none());
    assert!(profiles[0].ne.is_none());
    assert!(profiles[0].sht.is_none());
    assert!(profiles[0].ach.is_none());

    assert_eq!(profiles[1].da, Some(0.85));
    assert!(profiles[1].ne.is_none());
    assert_eq!(profiles[2].ne, Some(0.85));
    assert_eq!(profiles[3].sht, Some(0.85));
    assert_eq!(profiles[4].ach, Some(0.85));
}

// ── Dissociation Detection ──────────────────────────────────────────

#[test]
fn test_dissociation_detection() {
    let mut matrix = DissociationMatrix::new();
    matrix.add_entry(ProfilePerformance {
        profile: "High-DA".to_string(),
        benchmark: "WCST".to_string(),
        accuracy: 0.85,
        relative_performance: 1.15,
    });
    matrix.add_entry(ProfilePerformance {
        profile: "High-DA".to_string(),
        benchmark: "CPT".to_string(),
        accuracy: 0.65,
        relative_performance: 0.85,
    });
    matrix.add_entry(ProfilePerformance {
        profile: "Balanced".to_string(),
        benchmark: "WCST".to_string(),
        accuracy: 0.75,
        relative_performance: 1.0,
    });

    matrix.find_dissociations();

    assert_eq!(matrix.dissociations.len(), 1);
    assert_eq!(matrix.dissociations[0].profile, "High-DA");
    assert_eq!(matrix.dissociations[0].enhanced, "WCST");
    assert_eq!(matrix.dissociations[0].impaired, "CPT");
}

// ── Expected Dissociation Patterns ──────────────────────────────────

#[test]
fn test_expected_dissociations() {
    let expected = symthaea_psych_bench::harness::neuromod_profiles::expected_dissociations();
    assert_eq!(expected.len(), 4);
    assert_eq!(
        expected[0],
        ("High-DA", "Executive::WCST", "SustainedAttention::CPT")
    );
    assert_eq!(
        expected[1],
        ("High-NE", "SustainedAttention::PVT", "Motor::FittsLaw")
    );
    assert_eq!(
        expected[2],
        (
            "High-5HT",
            "Metacognition::Calibration",
            "Creativity::AlternateUses"
        )
    );
    assert_eq!(
        expected[3],
        (
            "High-ACh",
            "Attention::VisualSearch",
            "Creativity::RemoteAssociates"
        )
    );
}

// ── Markdown Output ─────────────────────────────────────────────────

#[test]
fn test_dissociation_matrix_markdown() {
    let mut matrix = DissociationMatrix::new();
    matrix.add_entry(ProfilePerformance {
        profile: "High-DA".to_string(),
        benchmark: "WCST".to_string(),
        accuracy: 0.85,
        relative_performance: 1.15,
    });
    matrix.add_entry(ProfilePerformance {
        profile: "High-DA".to_string(),
        benchmark: "CPT".to_string(),
        accuracy: 0.65,
        relative_performance: 0.85,
    });
    matrix.find_dissociations();

    let md = matrix.to_markdown();
    assert!(md.contains("Profile"));
    assert!(md.contains("High-DA"));
    assert!(md.contains("WCST"));
    assert!(md.contains("Dissociation"));
}

// ── Live Profile Battery (requires symthaea-backend) ────────────────

#[cfg(feature = "symthaea-backend")]
mod live_tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;
    use symthaea_psych_bench::harness::config::BenchmarkConfig;
    use symthaea_psych_bench::harness::live_runner::{
        CognitiveLoopBenchmarkRunner, LoopDrivable, LoopStimulus,
    };

    fn bench_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    struct ProfileBenchmark {
        name: &'static str,
        seed: u64,
    }

    impl LoopDrivable for ProfileBenchmark {
        fn loop_name(&self) -> &str {
            self.name
        }
        fn generate_stimuli(&self, config: &BenchmarkConfig) -> Vec<LoopStimulus> {
            let dim = config.dimension;
            let target = ContinuousHV::random(dim, self.seed);
            let distractor = ContinuousHV::random(dim, self.seed + 1000);
            (0..15)
                .map(|i| LoopStimulus {
                    stimulus_hv: target.clone(),
                    alternatives: vec![distractor.clone(), target.clone()],
                    correct_idx: 1,
                    condition: "profile".to_string(),
                    trial_idx: i,
                })
                .collect()
        }
    }

    #[test]
    #[ignore = "requires full cognitive loop infrastructure"]
    fn test_live_profile_battery() {
        let config = bench_config();
        let profiles = NeuromodProfile::standard_profiles();
        let bench = ProfileBenchmark {
            name: "Test::Profile",
            seed: 42,
        };

        let mut matrix = DissociationMatrix::new();

        let mut runner_balanced =
            CognitiveLoopBenchmarkRunner::new("profile-balanced").expect("runner");
        let baseline = runner_balanced.run_benchmark(&bench, &config);

        for profile in &profiles {
            if profile.name == "Balanced" {
                matrix.add_entry(ProfilePerformance {
                    profile: profile.name.clone(),
                    benchmark: bench.loop_name().to_string(),
                    accuracy: baseline.accuracy,
                    relative_performance: 1.0,
                });
                continue;
            }

            let mut runner =
                CognitiveLoopBenchmarkRunner::new(&format!("profile-{}", profile.name))
                    .expect("runner");
            runner.service_mut().clamp_neuromod_levels(
                profile.da,
                profile.ne,
                profile.sht,
                profile.ach,
            );

            let result = runner.run_benchmark(&bench, &config);
            let relative = if baseline.accuracy > 0.01 {
                result.accuracy / baseline.accuracy
            } else {
                1.0
            };

            matrix.add_entry(ProfilePerformance {
                profile: profile.name.clone(),
                benchmark: bench.loop_name().to_string(),
                accuracy: result.accuracy,
                relative_performance: relative,
            });
        }

        assert_eq!(matrix.entries.len(), 5);
        for entry in &matrix.entries {
            assert!(entry.accuracy.is_finite());
            assert!(entry.relative_performance.is_finite());
        }
    }

    #[test]
    #[ignore = "requires full cognitive loop infrastructure"]
    fn test_live_multi_benchmark_sweep() {
        let config = bench_config();
        let benchmarks = [
            ProfileBenchmark {
                name: "BenchA",
                seed: 100,
            },
            ProfileBenchmark {
                name: "BenchB",
                seed: 200,
            },
        ];

        let mut matrix = DissociationMatrix::new();
        let mut baselines = Vec::new();
        for bench in &benchmarks {
            let mut runner =
                CognitiveLoopBenchmarkRunner::new(&format!("sweep-baseline-{}", bench.name))
                    .expect("runner");
            baselines.push(runner.run_benchmark(bench, &config).accuracy);
        }

        for (i, bench) in benchmarks.iter().enumerate() {
            let mut runner = CognitiveLoopBenchmarkRunner::new(&format!("sweep-da-{}", bench.name))
                .expect("runner");
            runner
                .service_mut()
                .clamp_neuromod_levels(Some(0.85), None, None, None);
            let result = runner.run_benchmark(bench, &config);
            let relative = if baselines[i] > 0.01 {
                result.accuracy / baselines[i]
            } else {
                1.0
            };
            matrix.add_entry(ProfilePerformance {
                profile: "High-DA".to_string(),
                benchmark: bench.loop_name().to_string(),
                accuracy: result.accuracy,
                relative_performance: relative,
            });
        }

        matrix.find_dissociations();
        assert_eq!(matrix.entries.len(), 2);
    }
}

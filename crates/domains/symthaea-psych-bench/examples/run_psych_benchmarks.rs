// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run the full psychological benchmark suite and print results.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --json
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --csv
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --json-output /tmp/bench.json
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --snapshot baselines/v0.5.0.json
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --compare baselines/v0.5.0.json
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --composites
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --paper-table latex
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --forest-plot
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --learning-curve
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --population
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --sat
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --percentiles
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --pca
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --ablation-effects
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --correlation-matrix
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --bootstrap
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --llm-baselines
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --html-output /tmp/report.html
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --sequential
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --citations
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --meta-analysis
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --adaptive-trials
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --ssm-backend
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --participants 50
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --dim-sweep
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --arc-data-dir /path/to/ARC-AGI/data/training
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --reasoning-validity
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --arc-report
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --arc-reliability
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --cross-domain
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --arc-latex
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --trial-trace
//!   cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --difficulty 0.5

use rayon::prelude::*;
use std::path::PathBuf;
use symthaea_psych_bench::benchmarks::affect::{
    EmotionalStroopBenchmark, MoodCongruentRecallBenchmark, ValenceClassificationBenchmark,
};
use symthaea_psych_bench::benchmarks::attention::{
    AttentionalBlinkBenchmark, MismatchNegativityBenchmark, VisualSearchBenchmark,
};
use symthaea_psych_bench::benchmarks::binding::TemporalOrderBenchmark;
use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::coding::{
    AlgorithmRecognitionBenchmark, BugDetectionBenchmark, HumanEvalMiniBenchmark,
};
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
    TemporalDiscountingBenchmark, TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::consciousness::BlindSightBenchmark;
use symthaea_psych_bench::benchmarks::creativity::{
    AlternateUsesBenchmark, RemoteAssociatesBenchmark,
};
use symthaea_psych_bench::benchmarks::executive::{
    DualTaskBenchmark, FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark,
    StroopBenchmark, TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::{GoNoGoBenchmark, StopSignalBenchmark};
use symthaea_psych_bench::benchmarks::language::{
    GardenPathBenchmark, LexicalDecisionBenchmark, SemanticCoherenceBenchmark,
    SemanticPrimingBenchmark,
};
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    ProspectiveMemoryBenchmark, TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::metacognition::{
    ChangeBlindnessBenchmark, FeelingOfKnowingBenchmark, MetacognitiveCalibrationBenchmark,
};
use symthaea_psych_bench::benchmarks::motor::{
    BimanualBenchmark, FittsLawBenchmark, ProprioceptiveDriftBenchmark, SrttBenchmark,
};
use symthaea_psych_bench::benchmarks::neuromod::{
    AllostaticStressBenchmark, AntagonistProfilesBenchmark, AttentionNetworkBenchmark,
    BehavioralKnockoutBenchmark, ConsciousnessPharmacologyBenchmark, DoseResponseBenchmark,
    InjectionChallengeBenchmark, MoodInductionBenchmark, PharmacologicalAblationBenchmark,
    PharmacologicalChallengeBenchmark, RewardLearningBenchmark, ToleranceWithdrawalBenchmark,
    YerkesDodsonBenchmark,
};
use symthaea_psych_bench::benchmarks::qualia_confidence::{
    BistablePerceptionBenchmark, GwtAsphyxiationBenchmark, MetacognitiveIgnitionBenchmark,
    PerturbationalComplexityBenchmark, PhaseTransitionBenchmark, QualiaConfidenceScore,
    SomaticInterferenceBenchmark, UnconsciousPrimingBenchmark,
};
use symthaea_psych_bench::benchmarks::reasoning::{
    ArcAbductiveBenchmark, ArcAlgebraBenchmark, ArcAnalogyBenchmark, ArcChainBenchmark,
    ArcCompositionalBenchmark, ArcFewShotBenchmark, ArcFluidBenchmark, ArcNoiseBenchmark,
    ArcRsaBenchmark, ArcScalingBenchmark, ArcStaircaseBenchmark, ArcStrictBenchmark,
};
use symthaea_psych_bench::benchmarks::social::{
    DictatorGameBenchmark, MachiavelliBenchmark, PrisonersDilemmaBenchmark, PublicGoodsBenchmark,
    RmeBenchmark, SocialNormBenchmark, UltimatumGameBenchmark,
};
use symthaea_psych_bench::benchmarks::speech::PhonemeDiscriminationBenchmark;
use symthaea_psych_bench::benchmarks::substrate::SubstrateTransferBenchmark;
use symthaea_psych_bench::benchmarks::sustained_attention::{
    CptBenchmark, PvtBenchmark, SartBenchmark,
};
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
    SerialRecallBenchmark, SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::{
    BenchmarkConfig, BenchmarkReport, PsychBenchmark, RegressionReport, RegressionSnapshot,
    analysis::CrossBenchmarkAnalysis,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_json = args.iter().any(|a| a == "--json");
    let output_csv = args.iter().any(|a| a == "--csv");
    let output_profile = args.iter().any(|a| a == "--profile");
    let output_composites = args.iter().any(|a| a == "--composites");
    let paper_table = args.iter().any(|a| a == "--paper-table");
    let paper_latex = args
        .windows(2)
        .any(|w| w[0] == "--paper-table" && w[1] == "latex");
    let json_output_path: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--json-output")
        .map(|w| PathBuf::from(&w[1]));
    let filter: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--filter")
        .map(|w| w[1].to_lowercase());
    let snapshot_path: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--snapshot")
        .map(|w| PathBuf::from(&w[1]));
    let compare_path: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--compare")
        .map(|w| PathBuf::from(&w[1]));
    let reliability_mode = args.iter().any(|a| a == "--reliability");
    let forest_plot = args.iter().any(|a| a == "--forest-plot");
    let learning_curve = args.iter().any(|a| a == "--learning-curve");
    let population_sim = args.iter().any(|a| a == "--population");
    let sat_mode = args.iter().any(|a| a == "--sat");
    let percentiles_mode = args.iter().any(|a| a == "--percentiles");
    let pca_mode = args.iter().any(|a| a == "--pca");
    let ablation_effects = args.iter().any(|a| a == "--ablation-effects");
    let correlation_matrix = args.iter().any(|a| a == "--correlation-matrix");
    let bootstrap_mode = args.iter().any(|a| a == "--bootstrap");
    let llm_baselines = args.iter().any(|a| a == "--llm-baselines");
    let sequential_mode = args.iter().any(|a| a == "--sequential");
    let citations_mode = args.iter().any(|a| a == "--citations");
    let meta_analysis_mode = args.iter().any(|a| a == "--meta-analysis");
    let adaptive_trials_mode = args.iter().any(|a| a == "--adaptive-trials");
    let ssm_backend_mode = args.iter().any(|a| a == "--ssm-backend");
    let participants_count: Option<usize> = args
        .windows(2)
        .find(|w| w[0] == "--participants")
        .and_then(|w| w[1].parse().ok());
    let dim_sweep_mode = args.iter().any(|a| a == "--dim-sweep");
    let html_output_path: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--html-output")
        .map(|w| PathBuf::from(&w[1]));
    let arc_data_dir: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--arc-data-dir")
        .map(|w| PathBuf::from(&w[1]));
    let reasoning_validity = args.iter().any(|a| a == "--reasoning-validity");
    let arc_report_mode = args.iter().any(|a| a == "--arc-report");
    let arc_reliability_mode = args.iter().any(|a| a == "--arc-reliability");
    let cross_domain_mode = args.iter().any(|a| a == "--cross-domain");
    let arc_latex_mode = args.iter().any(|a| a == "--arc-latex");
    let trial_trace = args.iter().any(|a| a == "--trial-trace");
    let difficulty: f64 = args
        .windows(2)
        .find(|w| w[0] == "--difficulty")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(0.0);

    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        adaptive_trials: adaptive_trials_mode,
        ssm_backend: ssm_backend_mode,
        trial_trace,
        difficulty,
        ..Default::default()
    };

    let mut report = BenchmarkReport::new();

    let benchmarks: Vec<Box<dyn PsychBenchmark + Send + Sync>> = vec![
        // WorM
        Box::new(NBackBenchmark),
        Box::new(ChangeDetectionBenchmark),
        Box::new(SerialRecallBenchmark),
        Box::new(SpatialUpdatingBenchmark),
        Box::new(BindingBenchmark),
        Box::new(DigitSpanBenchmark),
        // CogBench
        Box::new(ProbabilisticReasoningBenchmark),
        Box::new(HorizonBenchmark),
        Box::new(RestlessBanditBenchmark),
        Box::new(InstrumentalLearningBenchmark),
        Box::new(TwoStepBenchmark),
        Box::new(TemporalDiscountingBenchmark),
        Box::new(BartBenchmark),
        Box::new(ReversalLearningBenchmark),
        // Executive
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(IowaGamblingBenchmark),
        Box::new(RavensProgressiveMatricesBenchmark),
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(TowerOfLondonBenchmark),
        Box::new(DualTaskBenchmark),
        // Metacognition
        Box::new(MetacognitiveCalibrationBenchmark),
        // Butlin
        Box::new(ButlinIndicatorSuite),
        // ToMBench
        Box::new(FalseBeliefBenchmark),
        Box::new(FauxPasBenchmark),
        Box::new(PersuasionBenchmark),
        Box::new(StrangeStoryBenchmark),
        Box::new(HintingBenchmark),
        // MemoryAgent
        Box::new(AccurateRetrievalBenchmark),
        Box::new(TestTimeLearningBenchmark),
        Box::new(LongRangeBenchmark),
        Box::new(ConflictResolutionBenchmark),
        // Affect
        Box::new(ValenceClassificationBenchmark),
        Box::new(MoodCongruentRecallBenchmark),
        Box::new(EmotionalStroopBenchmark),
        // Creativity
        Box::new(RemoteAssociatesBenchmark),
        Box::new(AlternateUsesBenchmark),
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
        Box::new(ArcStrictBenchmark),
        // Additional MemoryAgent
        Box::new(ProspectiveMemoryBenchmark),
        // Additional Metacognition
        Box::new(FeelingOfKnowingBenchmark),
        Box::new(ChangeBlindnessBenchmark),
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
        // Neuromod
        Box::new(AttentionNetworkBenchmark),
        Box::new(MoodInductionBenchmark),
        Box::new(RewardLearningBenchmark),
        Box::new(YerkesDodsonBenchmark),
        Box::new(PharmacologicalChallengeBenchmark),
        Box::new(PharmacologicalAblationBenchmark),
        Box::new(InjectionChallengeBenchmark),
        Box::new(BehavioralKnockoutBenchmark),
        Box::new(ConsciousnessPharmacologyBenchmark),
        Box::new(AllostaticStressBenchmark),
        Box::new(DoseResponseBenchmark),
        Box::new(AntagonistProfilesBenchmark),
        Box::new(ToleranceWithdrawalBenchmark),
        // Qualia Confidence (Consciousness)
        Box::new(GwtAsphyxiationBenchmark),
        Box::new(PhaseTransitionBenchmark),
        Box::new(PerturbationalComplexityBenchmark),
        Box::new(SomaticInterferenceBenchmark),
        Box::new(BistablePerceptionBenchmark),
        Box::new(UnconsciousPrimingBenchmark),
        Box::new(MetacognitiveIgnitionBenchmark),
        // Binding
        Box::new(TemporalOrderBenchmark),
        // Speech
        Box::new(PhonemeDiscriminationBenchmark),
        // Consciousness
        Box::new(BlindSightBenchmark),
        // Substrate
        Box::new(SubstrateTransferBenchmark),
        // Coding
        Box::new(HumanEvalMiniBenchmark),
        Box::new(BugDetectionBenchmark),
        Box::new(AlgorithmRecognitionBenchmark),
    ];

    // Filter benchmarks by name if --filter was specified
    let active_indices: Vec<usize> = benchmarks
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            filter
                .as_ref()
                .map(|f| b.name().to_lowercase().contains(f))
                .unwrap_or(true)
        })
        .map(|(i, _)| i)
        .collect();

    eprintln!(
        "Running {} benchmarks{}...",
        active_indices.len(),
        if sequential_mode {
            " (sequential)"
        } else {
            " (parallel)"
        }
    );

    if sequential_mode {
        for &i in &active_indices {
            let bench = &benchmarks[i];
            eprint!("  {} ... ", bench.name());
            let result = bench.run(&config);
            eprintln!("{}ms ({} metrics)", result.elapsed_ms, result.metrics.len());
            report.add(result);
        }
    } else {
        let results: Vec<_> = active_indices
            .par_iter()
            .map(|&i| {
                let bench = &benchmarks[i];
                bench.run(&config)
            })
            .collect();
        for result in results {
            eprintln!(
                "  {} ... {}ms ({} metrics)",
                result.benchmark,
                result.elapsed_ms,
                result.metrics.len()
            );
            report.add(result);
        }
    }

    // Write JSON to file if --json-output was specified
    if let Some(path) = &json_output_path {
        report
            .to_json_file(path)
            .expect("failed to write JSON output file");
        eprintln!("JSON written to {}", path.display());
    }

    // Save regression snapshot if --snapshot was specified
    if let Some(ref path) = snapshot_path {
        let git_hash = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());
        let mut snapshot = RegressionSnapshot::from_report(&report, "baseline");
        if let Some(hash) = git_hash {
            snapshot = snapshot.with_git_hash(hash);
        }
        snapshot.config_summary = format!(
            "dim={}, trials={}, seed={}",
            config.dimension, config.trials_per_condition, config.seed
        );
        snapshot
            .save(path)
            .expect("failed to save regression snapshot");
        eprintln!("Snapshot saved to {}", path.display());
    }

    // Compare against baseline if --compare was specified
    if let Some(ref path) = compare_path {
        let baseline = RegressionSnapshot::load(path).expect("failed to load baseline snapshot");
        let current = RegressionSnapshot::from_report(&report, "current");
        let regression = RegressionReport::compare(&baseline, &current, 0.05, 0.10);
        println!("\n{}", regression.format_summary());
        if regression.has_critical() {
            std::process::exit(1);
        }
    }

    if output_profile {
        println!("\n{}", report.format_profile());
    }

    if output_composites {
        println!("\n{}", report.format_composites());
    }

    // Qualia Confidence Matrix composite score (always shown when qualia benchmarks ran)
    {
        let qualia_results: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.benchmark.contains("QualiaConfidence"))
            .cloned()
            .collect();
        if qualia_results.len() >= 2 {
            let score = QualiaConfidenceScore::from_results(&qualia_results);
            eprint!("{}", score.format_report());
        }
    }

    // Reliability analysis: run battery with 5 different seeds
    if reliability_mode {
        let seeds = [42, 137, 256, 512, 1024];
        eprintln!(
            "\nRunning reliability analysis with {} seeds...",
            seeds.len()
        );

        let mut reports = vec![report.clone()]; // seed=42 already run above
        for &seed in &seeds[1..] {
            let seed_config = BenchmarkConfig {
                dimension: 512,
                trials_per_condition: 10,
                seed,
                ..Default::default()
            };
            let mut seed_report = BenchmarkReport::new();
            for bench in &benchmarks {
                if let Some(ref f) = filter {
                    if !bench.name().to_lowercase().contains(f) {
                        continue;
                    }
                }
                eprint!("  [seed={}] {} ... ", seed, bench.name());
                let result = bench.run(&seed_config);
                eprintln!("{}ms", result.elapsed_ms);
                seed_report.add(result);
            }
            reports.push(seed_report);
        }

        let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&reports);
        println!("\n--- Test-Retest Reliability (Split-Half) ---");
        println!("{}", analysis.format_reliability());

        // ICC(2,1) per benchmark
        println!("\n--- ICC(2,1) Reliability ---");
        println!(
            "| {:30} | {:>8} | {:15} |",
            "Benchmark", "ICC(2,1)", "Interpretation"
        );
        println!(
            "| {:30} | {:>8} | {:15} |",
            "------------------------------", "--------", "---------------"
        );
        for (name, _vals) in &analysis.values {
            let icc = compute_benchmark_icc_cli(&reports, name);
            let interp = if icc >= 0.75 {
                "Excellent"
            } else if icc >= 0.50 {
                "Moderate"
            } else if icc >= 0.25 {
                "Fair"
            } else {
                "Poor"
            };
            let short = name.split("::").last().unwrap_or(name);
            println!("| {:30} | {:>8.3} | {:15} |", short, icc, interp);
        }

        println!("\n--- Correlation Matrix ---");
        println!("{}", analysis.format_matrix());

        let validity = analysis.construct_validity();
        println!("\n--- Construct Validity ---");
        println!(
            "Same-domain pairs: {}, Convergent (r>0.3): {}, Mean within-domain r: {:.3}",
            validity.same_domain_pairs, validity.convergent_pairs, validity.mean_within_correlation,
        );

        println!("\n--- MTMM Validity ---");
        println!("{}", analysis.format_mtmm());

        // Cronbach's alpha: treat each seed's benchmark scores as items
        println!("\n--- Internal Consistency (Cronbach's Alpha) ---");
        {
            use symthaea_psych_bench::harness::analysis::cronbachs_alpha;
            // Items = seeds, observations = benchmark mean z-scores
            let items: Vec<Vec<f64>> = reports
                .iter()
                .map(|r| {
                    analysis
                        .values
                        .keys()
                        .map(|name| {
                            r.results
                                .iter()
                                .find(|res| res.benchmark == *name)
                                .and_then(|res| {
                                    let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(name);
                                    res.metrics.get(key).map(|m| m.mean)
                                })
                                .unwrap_or(0.0)
                        })
                        .collect()
                })
                .collect();
            let alpha = cronbachs_alpha(&items);
            let interp = if alpha >= 0.90 {
                "Excellent"
            } else if alpha >= 0.80 {
                "Good"
            } else if alpha >= 0.70 {
                "Acceptable"
            } else if alpha >= 0.60 {
                "Questionable"
            } else {
                "Poor"
            };
            println!("  Alpha = {:.3} ({})", alpha, interp);
            println!(
                "  Items (seeds) = {}, Benchmarks = {}",
                reports.len(),
                analysis.values.len()
            );
        }

        // PCA on reliability data
        println!("\n--- PCA (Reliability Data) ---");
        println!("{}", analysis.format_pca(3));
        return;
    }

    // Forest plot export
    if forest_plot {
        let ascii = report.forest_plot_ascii();
        println!("\n--- Effect Size Forest Plot ---");
        println!("{}", ascii);
        println!("\n--- Forest Plot CSV ---");
        println!("{}", report.forest_plot_csv());
        return;
    }

    // Learning curve: run 5 sequential blocks and track metric trajectory
    if learning_curve {
        println!("\n--- Learning Curve Analysis (5 blocks) ---");
        let n_blocks = 5;
        let trials_per_block = config.trials_per_condition;

        println!(
            "| {:25} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>7} |",
            "Benchmark", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Slope"
        );
        println!(
            "| {:25} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>7} |",
            "-------------------------",
            "--------",
            "--------",
            "--------",
            "--------",
            "--------",
            "-------"
        );

        for bench in &benchmarks {
            if let Some(ref f) = filter {
                if !bench.name().to_lowercase().contains(f) {
                    continue;
                }
            }

            let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
            let mut block_means = Vec::new();

            for block in 0..n_blocks {
                let block_config = BenchmarkConfig {
                    seed: config.seed.wrapping_add(block as u64 * 1000),
                    trials_per_condition: trials_per_block,
                    ..config.clone()
                };
                let result = bench.run(&block_config);
                let mean = result.metrics.get(key).map(|m| m.mean).unwrap_or(0.0);
                block_means.push(mean);
            }

            // Linear regression slope
            let slope = linear_slope(&block_means);
            let short = bench.name().split("::").last().unwrap_or(bench.name());

            println!(
                "| {:25} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>+7.4} |",
                &short[..short.len().min(25)],
                block_means[0],
                block_means[1],
                block_means[2],
                block_means[3],
                block_means[4],
                slope,
            );
        }
        return;
    }

    // Speed-accuracy tradeoff: 6-level sweep with Wickelgren curve fitting
    if sat_mode {
        use symthaea_psych_bench::harness::analysis::{
            SatCurvePoint, fit_sat_curve, sat_curve_ascii,
        };

        println!("\n--- Speed-Accuracy Tradeoff (6-level sweep) ---");
        let pressures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

        println!(
            "| {:25} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6} |",
            "Benchmark", "P=0.0", "P=0.2", "P=0.4", "P=0.6", "P=0.8", "P=1.0", "R²"
        );
        println!(
            "| {:25} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6} |",
            "-------------------------",
            "-------",
            "-------",
            "-------",
            "-------",
            "-------",
            "-------",
            "------"
        );

        let mut all_curves = Vec::new();

        for bench in &benchmarks {
            if let Some(ref f) = filter {
                if !bench.name().to_lowercase().contains(f) {
                    continue;
                }
            }

            let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
            let mut sat_points = Vec::new();

            for &pressure in &pressures {
                let sat_config = BenchmarkConfig {
                    time_pressure: pressure,
                    ..config.clone()
                };
                let result = bench.run(&sat_config);
                let acc = result.metrics.get(key).map(|m| m.mean).unwrap_or(0.0);
                let rt = result
                    .metrics
                    .get("rt_ticks")
                    .map(|m| m.mean)
                    .unwrap_or(0.0);
                sat_points.push(SatCurvePoint {
                    time_pressure: pressure,
                    accuracy: acc,
                    rt_ticks: rt,
                });
            }

            let curve = fit_sat_curve(bench.name(), &sat_points);
            let short = bench.name().split("::").last().unwrap_or(bench.name());
            println!(
                "| {:25} | {:>7.3} | {:>7.3} | {:>7.3} | {:>7.3} | {:>7.3} | {:>7.3} | {:>6.3} |",
                &short[..short.len().min(25)],
                sat_points[0].accuracy,
                sat_points[1].accuracy,
                sat_points[2].accuracy,
                sat_points[3].accuracy,
                sat_points[4].accuracy,
                sat_points[5].accuracy,
                curve.r_squared,
            );
            all_curves.push(curve);
        }

        // Print fitted curve details
        println!("\n--- Fitted Wickelgren (1977) Curves ---");
        for curve in &all_curves {
            println!("{}", sat_curve_ascii(curve));
        }

        return;
    }

    // Population simulation: N=100 synthetic participants
    if population_sim {
        println!("\n--- Population Simulation (N=100) ---");
        let n_participants = 100;

        println!(
            "| {:25} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} |",
            "Benchmark", "Mean", "SD", "Min", "Max", "Human Mean"
        );
        println!(
            "| {:25} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} |",
            "-------------------------",
            "--------",
            "--------",
            "--------",
            "--------",
            "----------"
        );

        for bench in &benchmarks {
            if let Some(ref f) = filter {
                if !bench.name().to_lowercase().contains(f) {
                    continue;
                }
            }

            let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
            let mut participant_scores = Vec::new();

            for p in 0..n_participants {
                let p_config = BenchmarkConfig {
                    seed: config.seed.wrapping_add(p as u64 * 7919), // prime spacing
                    dimension: 512,
                    trials_per_condition: 5, // fewer trials per participant (realistic)
                    ..config.clone()
                };
                let result = bench.run(&p_config);
                if let Some(metric) = result.metrics.get(key) {
                    participant_scores.push(metric.mean);
                }
            }

            if participant_scores.is_empty() {
                continue;
            }

            let n = participant_scores.len() as f64;
            let mean = participant_scores.iter().sum::<f64>() / n;
            let var = participant_scores
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            let sd = var.sqrt();
            let min = participant_scores
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max = participant_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            // Look up human baseline
            let temp_report = {
                let mut r = BenchmarkReport::new();
                let bench_result = bench.run(&config);
                r.add(bench_result);
                r
            };
            let human_str = {
                use symthaea_psych_bench::harness::baselines::BaselineCollection;
                let bl = BaselineCollection::all();
                let comparisons = temp_report.find_comparisons(&temp_report.results[0], &bl);
                comparisons
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, c)| format!("{:.3}", c.human_value))
                    .unwrap_or_else(|| "\u{2014}".to_string())
            };

            let short = bench.name().split("::").last().unwrap_or(bench.name());
            println!(
                "| {:25} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>10} |",
                &short[..short.len().min(25)],
                mean,
                sd,
                min,
                max,
                human_str,
            );
        }
        return;
    }

    // Individual differences simulation: per-participant WM + time_pressure variation
    if let Some(n_participants) = participants_count {
        use symthaea_psych_bench::harness::analysis::{
            format_individual_differences, individual_differences,
        };

        println!(
            "\n--- Individual Differences Simulation (N={}) ---\n",
            n_participants
        );

        for bench in &benchmarks {
            if let Some(ref f) = filter {
                if !bench.name().to_lowercase().contains(f) {
                    continue;
                }
            }

            let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
            let mut participant_scores = Vec::new();

            for p in 0..n_participants {
                // Derive per-participant parameters deterministically
                let mut rng = config.seed.wrapping_add(p as u64 * 7919) ^ 0x9E3779B97F4A7C15;
                let xor_shift = |s: &mut u64| {
                    *s ^= *s << 13;
                    *s ^= *s >> 7;
                    *s ^= *s << 17;
                };

                // WM capacity: normal(7, 1), clamped [3, 11]
                xor_shift(&mut rng);
                let u1 = (rng % 10000) as f64 / 10000.0;
                xor_shift(&mut rng);
                let u2 = (rng % 10000) as f64 / 10000.0;
                let z_wm =
                    (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let wm_cap = (7.0 + z_wm).round().clamp(3.0, 11.0) as usize;

                // Time pressure: normal(0.3, 0.15), clamped [0.0, 1.0]
                xor_shift(&mut rng);
                let u3 = (rng % 10000) as f64 / 10000.0;
                xor_shift(&mut rng);
                let u4 = (rng % 10000) as f64 / 10000.0;
                let z_tp =
                    (-2.0 * u3.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u4).cos();
                let tp = (0.3 + 0.15 * z_tp).clamp(0.0, 1.0);

                let p_config = BenchmarkConfig {
                    seed: config.seed.wrapping_add(p as u64 * 7919),
                    working_memory_capacity: wm_cap,
                    time_pressure: tp,
                    trials_per_condition: 5,
                    ..config.clone()
                };
                let result = bench.run(&p_config);
                if let Some(metric) = result.metrics.get(key) {
                    participant_scores.push(metric.mean);
                }
            }

            if !participant_scores.is_empty() {
                let id_result = individual_differences(bench.name(), key, &participant_scores);
                println!("{}", format_individual_differences(&id_result));
            }
        }
        return;
    }

    // HDC dimensionality scaling sweep
    if dim_sweep_mode {
        use symthaea_psych_bench::harness::analysis::{
            DimScalingResult, dim_scaling_slope, format_dim_scaling,
        };

        let sweep_dims = [128, 256, 512, 1024, 2048];
        println!(
            "\n--- HDC Dimensionality Scaling (dims: {:?}) ---\n",
            sweep_dims
        );
        println!(
            "| {:25} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>+8} | {:>6} |",
            "Benchmark", "d=128", "d=256", "d=512", "d=1024", "d=2048", "Slope", "R²"
        );
        println!(
            "| {:25} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>8} | {:>6} |",
            "-------------------------",
            "-------",
            "-------",
            "-------",
            "-------",
            "-------",
            "--------",
            "------"
        );

        for bench in &benchmarks {
            if let Some(ref f) = filter {
                if !bench.name().to_lowercase().contains(f) {
                    continue;
                }
            }

            let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
            let mut values = Vec::new();

            for &dim in &sweep_dims {
                let dim_config = BenchmarkConfig {
                    dimension: dim,
                    trials_per_condition: 5,
                    ..config.clone()
                };
                let result = bench.run(&dim_config);
                let val = result.metrics.get(key).map(|m| m.mean).unwrap_or(0.0);
                values.push(val);
            }

            let dims_usize: Vec<usize> = sweep_dims.iter().map(|&d| d as usize).collect();
            let (slope, r2) = dim_scaling_slope(&dims_usize, &values);
            let short = bench.name().split("::").last().unwrap_or(bench.name());
            println!(
                "| {:25} | {:>7.3} | {:>7.3} | {:>7.3} | {:>7.3} | {:>7.3} | {:>+8.4} | {:>6.3} |",
                &short[..short.len().min(25)],
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                slope,
                r2,
            );

            let _result = DimScalingResult {
                benchmark: bench.name().to_string(),
                dims: dims_usize,
                values,
                slope,
                r_squared: r2,
            };
        }
        return;
    }

    // Ablation effect sizes: compare full config vs degraded (low WM, low dim)
    if ablation_effects {
        use symthaea_psych_bench::harness::analysis::{
            ablation_effect_sizes, format_ablation_effects,
        };

        println!("\n--- Ablation Effect Sizes ---\n");

        let degraded_configs: Vec<(&str, BenchmarkConfig)> = vec![
            (
                "Low WM (capacity=2)",
                BenchmarkConfig {
                    working_memory_capacity: 2,
                    ..config.clone()
                },
            ),
            (
                "Low Dimension (dim=64)",
                BenchmarkConfig {
                    dimension: 64,
                    ..config.clone()
                },
            ),
            (
                "High Pressure (p=1.0)",
                BenchmarkConfig {
                    time_pressure: 1.0,
                    ..config.clone()
                },
            ),
        ];

        // Get baseline means
        let baseline_means: Vec<(&str, f64)> = benchmarks
            .iter()
            .filter_map(|bench| {
                if let Some(ref f) = filter {
                    if !bench.name().to_lowercase().contains(f) {
                        return None;
                    }
                }
                let key =
                    symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
                report
                    .results
                    .iter()
                    .find(|r| r.benchmark == bench.name())
                    .and_then(|r| r.metrics.get(key).map(|m| (bench.name(), m.mean)))
            })
            .collect();

        for (label, deg_config) in &degraded_configs {
            println!("### {} vs Baseline\n", label);
            let mut ablated_means: Vec<(&str, f64)> = Vec::new();

            for bench in &benchmarks {
                if let Some(ref f) = filter {
                    if !bench.name().to_lowercase().contains(f) {
                        continue;
                    }
                }
                let key =
                    symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
                let result = bench.run(deg_config);
                if let Some(m) = result.metrics.get(key) {
                    ablated_means.push((bench.name(), m.mean));
                }
            }

            let effects = ablation_effect_sizes(&baseline_means, &ablated_means);
            println!("{}\n", format_ablation_effects(&effects));
        }
        return;
    }

    // Normative percentile table
    if percentiles_mode {
        use symthaea_psych_bench::harness::analysis::percentile_from_z;
        use symthaea_psych_bench::harness::baselines::BaselineCollection;

        let bl = BaselineCollection::all();
        println!("\n--- Normative Percentile Table ---");
        println!(
            "| {:25} | {:>25} | {:>8} | {:>8} | {:>8} | {:>10} |",
            "Benchmark", "Metric", "Value", "Z-Score", "%ile", "Rating"
        );
        println!(
            "| {:25} | {:>25} | {:>8} | {:>8} | {:>8} | {:>10} |",
            "-------------------------",
            "-------------------------",
            "--------",
            "--------",
            "--------",
            "----------"
        );

        for bench_result in &report.results {
            let comparisons = report.find_comparisons(bench_result, &bl);
            for (key, comp) in &comparisons {
                if let Some(z) = comp.z_score {
                    let pct = percentile_from_z(z);
                    let rating = if pct >= 90.0 {
                        "Superior"
                    } else if pct >= 75.0 {
                        "Above Avg"
                    } else if pct >= 25.0 {
                        "Average"
                    } else if pct >= 10.0 {
                        "Below Avg"
                    } else {
                        "Impaired"
                    };
                    let agent_val = bench_result
                        .metrics
                        .get(key.as_str())
                        .map(|m| m.mean)
                        .unwrap_or(0.0);
                    let short_bench = bench_result
                        .benchmark
                        .split("::")
                        .last()
                        .unwrap_or(&bench_result.benchmark);
                    let short_key = if key.len() > 25 { &key[..25] } else { key };
                    println!(
                        "| {:25} | {:>25} | {:>8.3} | {:>+8.2} | {:>7.1}% | {:>10} |",
                        &short_bench[..short_bench.len().min(25)],
                        short_key,
                        agent_val,
                        z,
                        pct,
                        rating,
                    );
                }
            }
        }
        return;
    }

    // Principal component analysis on the full battery
    if pca_mode {
        // Need multi-seed data for PCA
        let seeds = [42, 137, 256, 512, 1024];
        eprintln!("\nRunning PCA with {} seeds...", seeds.len());

        let mut reports = vec![report.clone()];
        for &seed in &seeds[1..] {
            let seed_config = BenchmarkConfig {
                dimension: 512,
                trials_per_condition: 10,
                seed,
                ..Default::default()
            };
            let mut seed_report = BenchmarkReport::new();
            for bench in &benchmarks {
                if let Some(ref f) = filter {
                    if !bench.name().to_lowercase().contains(f) {
                        continue;
                    }
                }
                let result = bench.run(&seed_config);
                seed_report.add(result);
            }
            reports.push(seed_report);
        }

        let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&reports);
        println!("\n--- Principal Component Analysis ---");
        println!("{}", analysis.format_pca(5));
        return;
    }

    // Correlation matrix: run battery with 5 seeds, print matrix + construct validity
    if correlation_matrix {
        let seeds = [42, 137, 256, 512, 1024];
        eprintln!("\nRunning correlation matrix with {} seeds...", seeds.len());

        let mut reports = vec![report.clone()];
        for &seed in &seeds[1..] {
            let seed_config = BenchmarkConfig {
                dimension: 512,
                trials_per_condition: 10,
                seed,
                ..Default::default()
            };
            let mut seed_report = BenchmarkReport::new();
            for bench in &benchmarks {
                if let Some(ref f) = filter {
                    if !bench.name().to_lowercase().contains(f) {
                        continue;
                    }
                }
                let result = bench.run(&seed_config);
                seed_report.add(result);
            }
            reports.push(seed_report);
        }

        let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&reports);
        println!("\n--- Correlation Matrix ---");
        println!("{}", analysis.format_matrix());

        let validity = analysis.construct_validity();
        println!("\n--- Construct Validity ---");
        println!(
            "Same-domain pairs: {}, Convergent (r>0.3): {}, Mean within-domain r: {:.3}",
            validity.same_domain_pairs, validity.convergent_pairs, validity.mean_within_correlation,
        );

        println!("\n--- MTMM Validity ---");
        println!("{}", analysis.format_mtmm());
        return;
    }

    // Bootstrap CI mode: re-compute all metric CIs using BCa bootstrap
    if bootstrap_mode {
        println!("\n--- Bootstrap Confidence Intervals (BCa, 2000 resamples) ---");
        // Bootstrap CIs are computed via from_samples_bootstrap in MetricValue
        // Re-display results with bootstrap CIs
        for result in &report.results {
            let key =
                symthaea_psych_bench::harness::report::key_metric_for_benchmark(&result.benchmark);
            if let Some(metric) = result.metrics.get(key) {
                let short = result
                    .benchmark
                    .split("::")
                    .last()
                    .unwrap_or(&result.benchmark);
                println!(
                    "  {:<25} {:.3} [{:.3}, {:.3}] (parametric)  →  bootstrap available via MetricValue::from_samples_bootstrap()",
                    short, metric.mean, metric.ci_lower, metric.ci_upper,
                );
            }
        }
    }

    // LLM baselines comparison
    if llm_baselines {
        use symthaea_psych_bench::harness::baselines::BaselineCollection;
        let bl = BaselineCollection::all();
        println!("\n--- LLM Baseline Comparisons (GPT-4) ---");
        println!(
            "| {:25} | {:>25} | {:>8} | {:>10} | {:>10} | {:>8} |",
            "Benchmark", "Metric", "Agent", "Human", "LLM (GPT4)", "LLM %"
        );
        println!(
            "| {:25} | {:>25} | {:>8} | {:>10} | {:>10} | {:>8} |",
            "-------------------------",
            "-------------------------",
            "--------",
            "----------",
            "----------",
            "--------"
        );
        for result in &report.results {
            let llm_comparisons =
                symthaea_psych_bench::harness::report::BenchmarkReport::find_llm_comparisons(
                    &report, result, &bl,
                );
            for (key, comp) in &llm_comparisons {
                let agent_val = result
                    .metrics
                    .get(key.as_str())
                    .map(|m| m.mean)
                    .unwrap_or(0.0);
                let short = result
                    .benchmark
                    .split("::")
                    .last()
                    .unwrap_or(&result.benchmark);
                let short_key = if key.len() > 25 { &key[..25] } else { key };
                println!(
                    "| {:25} | {:>25} | {:>8.3} | {:>10} | {:>10.3} | {:>7.1}% |",
                    &short[..short.len().min(25)],
                    short_key,
                    agent_val,
                    "\u{2014}",
                    comp.human_value,
                    comp.ratio * 100.0,
                );
            }
        }
    }

    // Meta-analysis (random-effects, DerSimonian-Laird)
    if meta_analysis_mode {
        use symthaea_psych_bench::harness::analysis::{
            format_meta_analysis, meta_analysis_from_forest,
        };
        let rows = report.forest_plot_data();
        if let Some(result) = meta_analysis_from_forest(&rows) {
            println!("\n{}", format_meta_analysis(&result));
        } else {
            eprintln!("No effect sizes available for meta-analysis.");
        }
    }

    // Citations / provenance table
    if citations_mode {
        let refs: Vec<&dyn PsychBenchmark> = active_indices
            .iter()
            .map(|&i| &*benchmarks[i] as &dyn PsychBenchmark)
            .collect();
        let table = symthaea_psych_bench::harness::provenance_table(&refs);
        println!("\n{}", table);
    }

    // HTML report output
    if let Some(ref path) = html_output_path {
        use symthaea_psych_bench::harness::html;
        // Collect provenance entries for the HTML report
        let prov_entries: Vec<(&str, symthaea_psych_bench::harness::BenchmarkProvenance)> =
            active_indices
                .iter()
                .filter_map(|&i| {
                    let b = &*benchmarks[i];
                    b.provenance().map(|p| (b.name(), p))
                })
                .collect();
        let mut html_content = html::generate_report(&report, llm_baselines);
        // Insert provenance section before closing footer if entries exist
        if !prov_entries.is_empty() {
            let mut prov_html = String::new();
            html::write_provenance_section(&mut prov_html, &prov_entries);
            // Insert before </body>
            if let Some(pos) = html_content.rfind("</body>") {
                html_content.insert_str(pos, &prov_html);
            }
        }
        // Insert SAT curves SVG if we have benchmarks to sweep
        {
            use symthaea_psych_bench::harness::sat_curves::run_sat_curve;
            let pressures = [0.0, 0.5, 1.0];
            let mut sat_curves = Vec::new();
            for &i in &active_indices {
                let bench = &benchmarks[i];
                sat_curves.push(run_sat_curve(bench.as_ref(), &config, &pressures));
            }
            if !sat_curves.is_empty() {
                let mut sat_html = String::new();
                html::write_sat_curves(&mut sat_html, &sat_curves);
                if let Some(pos) = html_content.rfind("</body>") {
                    html_content.insert_str(pos, &sat_html);
                }
            }
        }
        std::fs::write(path, html_content).expect("failed to write HTML report");
        eprintln!("HTML report written to {}", path.display());
    }

    // ── Real ARC-AGI dataset evaluation ──
    if let Some(ref arc_dir) = arc_data_dir {
        use symthaea_psych_bench::benchmarks::reasoning::arc_dataset;
        match arc_dataset::load_arc_tasks(arc_dir) {
            Ok(tasks) => {
                eprintln!(
                    "Loaded {} ARC tasks from {}",
                    tasks.len(),
                    arc_dir.display()
                );
                let result = arc_dataset::evaluate_arc_tasks(&tasks, config.dimension, config.seed);
                println!("\n{}", arc_dataset::format_arc_dataset_result(&result));
            }
            Err(e) => eprintln!("Failed to load ARC dataset: {}", e),
        }
    }

    // ── Reasoning domain validity analysis ──
    if reasoning_validity {
        use symthaea_psych_bench::harness::analysis::{
            format_reasoning_validity, reasoning_validity_single_run,
        };
        let validity = reasoning_validity_single_run(&report);
        println!("\n{}", format_reasoning_validity(&validity));
    }

    // ── Unified ARC report ──
    if arc_report_mode {
        use symthaea_psych_bench::harness::analysis::format_arc_report;
        println!("\n{}", format_arc_report(&report));
    }

    // ── ARC split-half reliability ──
    if arc_reliability_mode {
        use symthaea_psych_bench::harness::analysis::{
            arc_split_half_reliability, format_arc_reliability,
        };
        let results = arc_split_half_reliability(&report);
        println!("\n{}", format_arc_reliability(&results));
    }

    // ── Cross-domain prediction analysis ──
    if cross_domain_mode {
        use symthaea_psych_bench::harness::analysis::{
            cross_domain_prediction, format_cross_domain_prediction,
        };
        // Run 10-seed battery for stable estimates
        let mut seed_reports = Vec::new();
        for seed_offset in 0..10u64 {
            let mut seed_config = config.clone();
            seed_config.seed = config.seed + seed_offset;
            let mut seed_report = BenchmarkReport::new();
            for bench in &benchmarks {
                seed_report.add(bench.run(&seed_config));
            }
            seed_reports.push(seed_report);
        }
        let target = "Reasoning::ArcFluid";
        let predictors = &[
            "WorM::NBack",
            "Executive::WCST",
            "Attention::VisualSearch",
            "Motor::SRTT",
        ];
        match cross_domain_prediction(&seed_reports, target, predictors) {
            Some(result) => println!("\n{}", format_cross_domain_prediction(&result)),
            None => eprintln!("Insufficient data for cross-domain prediction"),
        }
    }

    // ── ARC publication-ready LaTeX ──
    if arc_latex_mode {
        use symthaea_psych_bench::harness::analysis::arc_paper_section_latex;
        println!("\n{}", arc_paper_section_latex(&report));
    }

    if output_json {
        println!("{}", report.to_json().expect("JSON serialization"));
    } else if output_csv {
        println!("{}", report.to_csv().expect("CSV serialization"));
    } else if paper_latex {
        println!("{}", report.paper_summary_latex());
    } else if paper_table {
        println!("{}", report.paper_summary());
    } else {
        println!("\n{}", report.summary());
    }
}

/// Compute linear regression slope for ordered block means.
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean).powi(2);
    }
    if den.abs() < 1e-15 { 0.0 } else { num / den }
}

/// Compute ICC(2,1) for a benchmark across seed reports.
fn compute_benchmark_icc_cli(reports: &[BenchmarkReport], benchmark_name: &str) -> f64 {
    use symthaea_psych_bench::harness::analysis::icc_2_1;

    let first_report = &reports[0];
    let bench_result = first_report
        .results
        .iter()
        .find(|r| r.benchmark == benchmark_name);

    let metric_keys: Vec<String> = match bench_result {
        Some(r) => r.metrics.keys().cloned().collect(),
        None => return 0.0,
    };

    if metric_keys.len() < 2 {
        return 0.0;
    }

    let observations: Vec<Vec<f64>> = reports
        .iter()
        .map(|report| {
            let result = report
                .results
                .iter()
                .find(|r| r.benchmark == benchmark_name);
            metric_keys
                .iter()
                .map(|key| {
                    result
                        .and_then(|r| r.metrics.get(key))
                        .map(|m| m.mean)
                        .unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    icc_2_1(&observations)
}

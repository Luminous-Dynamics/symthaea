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

use std::path::PathBuf;
use symthaea_psych_bench::benchmarks::affect::{
    EmotionalStroopBenchmark, MoodCongruentRecallBenchmark, ValenceClassificationBenchmark,
};
use symthaea_psych_bench::benchmarks::attention::AttentionalBlinkBenchmark;
use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
    TemporalDiscountingBenchmark, TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::creativity::{
    AlternateUsesBenchmark, RemoteAssociatesBenchmark,
};
use symthaea_psych_bench::benchmarks::executive::{
    FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark, StroopBenchmark,
    TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::GoNoGoBenchmark;
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    ProspectiveMemoryBenchmark, TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::metacognition::MetacognitiveCalibrationBenchmark;
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
    SerialRecallBenchmark, SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::{
    analysis::CrossBenchmarkAnalysis, BenchmarkConfig, BenchmarkReport, PsychBenchmark,
    RegressionReport, RegressionSnapshot,
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

    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };

    let mut report = BenchmarkReport::new();

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
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
        // Attention
        Box::new(AttentionalBlinkBenchmark),
        // Additional MemoryAgent
        Box::new(ProspectiveMemoryBenchmark),
    ];

    eprintln!("Running {} benchmarks...", benchmarks.len());
    for bench in &benchmarks {
        if let Some(ref f) = filter {
            if !bench.name().to_lowercase().contains(f) {
                continue;
            }
        }
        eprint!("  {} ... ", bench.name());
        let result = bench.run(&config);
        eprintln!("{}ms ({} metrics)", result.elapsed_ms, result.metrics.len());
        report.add(result);
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

    // Speed-accuracy tradeoff: run at 3 pressure levels
    if sat_mode {
        println!("\n--- Speed-Accuracy Tradeoff ---");
        let pressures = [0.0, 0.5, 1.0];
        let labels = ["None", "Medium", "High"];

        println!(
            "| {:25} | {:>12} | {:>12} | {:>12} |",
            "Benchmark", "P=0.0 (None)", "P=0.5 (Med)", "P=1.0 (High)"
        );
        println!(
            "| {:25} | {:>12} | {:>12} | {:>12} |",
            "-------------------------", "------------", "------------", "------------"
        );

        for bench in &benchmarks {
            if let Some(ref f) = filter {
                if !bench.name().to_lowercase().contains(f) {
                    continue;
                }
            }

            let key = symthaea_psych_bench::harness::report::key_metric_for_benchmark(bench.name());
            let mut level_means = Vec::new();

            for &pressure in &pressures {
                let sat_config = BenchmarkConfig {
                    time_pressure: pressure,
                    ..config.clone()
                };
                let result = bench.run(&sat_config);
                let mean = result.metrics.get(key).map(|m| m.mean).unwrap_or(0.0);
                level_means.push(mean);
            }

            let short = bench.name().split("::").last().unwrap_or(bench.name());
            println!(
                "| {:25} | {:>12.3} | {:>12.3} | {:>12.3} |",
                &short[..short.len().min(25)],
                level_means[0],
                level_means[1],
                level_means[2],
            );
        }

        // Overall SAT summary
        println!("\nPressure labels: {:?}", labels);
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
    if den.abs() < 1e-15 {
        0.0
    } else {
        num / den
    }
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

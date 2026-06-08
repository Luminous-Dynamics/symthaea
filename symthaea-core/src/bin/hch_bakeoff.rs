// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(not(feature = "cantor-hdc"))]
fn main() {
    eprintln!("hch_bakeoff requires --features cantor-hdc");
    std::process::exit(2);
}

#[cfg(feature = "cantor-hdc")]
mod app {
    use serde::Serialize;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use symthaea_core::hdc::cantor_pyramid::{
        BundleMode, CantorHdcConfig, CantorRouter, HashRouter, HypercubeRouter,
        LoadBalancedHashRouter, PrefixMaxRouter, PrototypeIndex, PrototypeRouter,
        PyramidCantorVector, RandomRouter, SmallWorldRouter,
    };
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    #[derive(Debug, Clone)]
    struct Args {
        objects: usize,
        seeds: usize,
        dim: usize,
        branching: usize,
        abstain_threshold: f32,
        shortcuts: usize,
        out: PathBuf,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                objects: 128,
                seeds: 3,
                dim: 16_384,
                branching: 16,
                abstain_threshold: 0.05,
                shortcuts: 2,
                out: PathBuf::from("reports/hch_v06.json"),
            }
        }
    }

    #[derive(Debug, Clone, Serialize)]
    struct BakeoffReport {
        architecture: &'static str,
        version: &'static str,
        objects: usize,
        seeds: usize,
        dim: usize,
        branching: usize,
        leaf_dim: usize,
        abstain_threshold: f32,
        shortcuts: usize,
        diagnosis: String,
        results: Vec<RouterSummary>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct RouterSummary {
        router: String,
        top1: f32,
        top3: f32,
        mean_margin: f32,
        abstention_rate: f32,
        answered_accuracy: f32,
        load_entropy: f32,
        max_leaf_load: usize,
        mean_leaf_load: f32,
        latency_ms: f32,
        oracle_gap_top1: f32,
        oracle_gap_margin: f32,
    }

    #[derive(Debug, Clone)]
    struct TrialResult {
        router: String,
        top1: f32,
        top3: f32,
        mean_margin: f32,
        abstention_rate: f32,
        answered_accuracy: f32,
        load_entropy: f32,
        max_leaf_load: usize,
        mean_leaf_load: f32,
        latency_ms: f32,
    }

    #[derive(Clone)]
    struct Example {
        role: ContinuousHV,
        value_idx: usize,
    }

    pub fn main() {
        let args = parse_args().unwrap_or_else(|err| {
            eprintln!("{err}");
            print_usage();
            std::process::exit(2);
        });

        let report = run_bakeoff(&args);
        write_reports(&args.out, &report).unwrap_or_else(|err| {
            eprintln!("writing reports: {err}");
            std::process::exit(1);
        });

        print_summary(&report);
    }

    fn parse_args() -> Result<Args, String> {
        let mut args = Args::default();
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--objects" => args.objects = parse_next(&mut iter, "--objects")?,
                "--seeds" => args.seeds = parse_next(&mut iter, "--seeds")?,
                "--dim" => args.dim = parse_next(&mut iter, "--dim")?,
                "--branching" => args.branching = parse_next(&mut iter, "--branching")?,
                "--abstain" => args.abstain_threshold = parse_next(&mut iter, "--abstain")?,
                "--shortcuts" => args.shortcuts = parse_next(&mut iter, "--shortcuts")?,
                "--out" => args.out = PathBuf::from(next_value(&mut iter, "--out")?),
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        if args.objects == 0 {
            return Err("--objects must be > 0".into());
        }
        if args.seeds == 0 {
            return Err("--seeds must be > 0".into());
        }
        if args.branching == 0 {
            return Err("--branching must be > 0".into());
        }
        if args.dim % args.branching != 0 {
            return Err("--dim must be divisible by --branching".into());
        }

        Ok(args)
    }

    fn parse_next<T: std::str::FromStr>(
        iter: &mut impl Iterator<Item = String>,
        flag: &str,
    ) -> Result<T, String> {
        let value = next_value(iter, flag)?;
        value
            .parse()
            .map_err(|_| format!("invalid value for {flag}: {value}"))
    }

    fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        iter.next()
            .ok_or_else(|| format!("missing value after {flag}"))
    }

    fn print_usage() {
        eprintln!(
            "usage: hch_bakeoff [--objects N] [--seeds N] [--dim N] [--branching N] \\
             [--abstain F] [--shortcuts N] [--out PATH]"
        );
    }

    fn run_bakeoff(args: &Args) -> BakeoffReport {
        let leaf_dim = args.dim / args.branching;
        let config = CantorHdcConfig {
            total_dim: args.dim,
            levels: 2,
            branching: args.branching,
            leaf_dim,
            bundle_mode: BundleMode::UnitNormalize,
        };
        let hypercube_dims = args.branching.next_power_of_two().trailing_zeros() as usize;
        let mut trials = Vec::new();

        for seed_idx in 0..args.seeds {
            let seed = seed_idx as u64;
            let codebook = build_codebook(leaf_dim, args.objects * 2, seed);
            let train = build_examples(leaf_dim, args.objects, seed, 20_000);
            let test = build_examples(leaf_dim, args.objects, seed, 40_000);
            let prototype_index = build_prototypes(&train, config, hypercube_dims, seed);

            let routers = build_routers(
                args,
                config,
                hypercube_dims,
                seed,
                &prototype_index.leaf_keys,
            );
            for (name, router) in routers {
                trials.push(run_trial(
                    name,
                    config,
                    &test,
                    &codebook,
                    router.as_ref(),
                    args.abstain_threshold,
                ));
            }
            let oracle_config = CantorHdcConfig {
                total_dim: config.leaf_dim * args.objects,
                levels: 2,
                branching: args.objects,
                leaf_dim: config.leaf_dim,
                bundle_mode: config.bundle_mode,
            };
            trials.push(run_oracle_trial(
                "OracleSameCapacity".into(),
                config,
                &test,
                &codebook,
                args.abstain_threshold,
            ));
            trials.push(run_oracle_trial(
                "OracleHighCapacity".into(),
                oracle_config,
                &test,
                &codebook,
                args.abstain_threshold,
            ));
        }

        let mut results = summarize_trials(&trials, args.seeds);
        let oracle = results
            .iter()
            .find(|result| result.router == "OracleHighCapacity")
            .cloned();
        if let Some(oracle) = oracle {
            for result in &mut results {
                result.oracle_gap_top1 = oracle.top1 - result.top1;
                result.oracle_gap_margin = oracle.mean_margin - result.mean_margin;
            }
        }

        let diagnosis = diagnose(&results);

        BakeoffReport {
            architecture: "RHN",
            version: "rhn-v0.7",
            objects: args.objects,
            seeds: args.seeds,
            dim: args.dim,
            branching: args.branching,
            leaf_dim,
            abstain_threshold: args.abstain_threshold,
            shortcuts: args.shortcuts,
            diagnosis,
            results,
        }
    }

    fn build_codebook(leaf_dim: usize, size: usize, seed: u64) -> Vec<ContinuousHV> {
        (0..size.max(100))
            .map(|idx| ContinuousHV::random(leaf_dim, seed + 10_000 + idx as u64))
            .collect()
    }

    fn build_examples(leaf_dim: usize, count: usize, seed: u64, offset: u64) -> Vec<Example> {
        (0..count)
            .map(|idx| Example {
                role: ContinuousHV::random(leaf_dim, seed + offset + idx as u64),
                value_idx: idx,
            })
            .collect()
    }

    fn build_prototypes(
        train: &[Example],
        config: CantorHdcConfig,
        hypercube_dims: usize,
        seed: u64,
    ) -> PrototypeIndex {
        let router = HypercubeRouter {
            dimensions: hypercube_dims,
            seed,
        };
        let zero = ContinuousHV::zero(config.leaf_dim);
        let assignments: Vec<(usize, ContinuousHV)> = train
            .iter()
            .map(|example| {
                (
                    router.route(&example.role, &zero, config.branching),
                    example.role.clone(),
                )
            })
            .collect();

        PrototypeIndex::from_assignments(
            &assignments,
            config.branching,
            config.leaf_dim,
            BundleMode::UnitNormalize,
        )
    }

    fn build_routers(
        args: &Args,
        config: CantorHdcConfig,
        hypercube_dims: usize,
        seed: u64,
        learned_leaf_keys: &[ContinuousHV],
    ) -> Vec<(String, Box<dyn CantorRouter>)> {
        vec![
            ("Random".into(), Box::new(RandomRouter { seed })),
            ("Hash".into(), Box::new(HashRouter)),
            (
                "Hypercube".into(),
                Box::new(HypercubeRouter {
                    dimensions: hypercube_dims,
                    seed,
                }),
            ),
            (
                "LB-Hash-2".into(),
                Box::new(LoadBalancedHashRouter::new(config.branching, 2)),
            ),
            (
                "LB-Hash-4".into(),
                Box::new(LoadBalancedHashRouter::new(config.branching, 4)),
            ),
            ("PrefixMax".into(), Box::new(PrefixMaxRouter)),
            (
                "PrototypeLearned".into(),
                Box::new(PrototypeRouter {
                    leaf_keys: learned_leaf_keys.to_vec(),
                }),
            ),
            (
                "SmallWorldLearned".into(),
                Box::new(SmallWorldRouter {
                    dimensions: hypercube_dims,
                    seed,
                    leaf_keys: learned_leaf_keys.to_vec(),
                    shortcuts: args.shortcuts,
                }),
            ),
        ]
    }

    fn run_trial(
        router_name: String,
        config: CantorHdcConfig,
        examples: &[Example],
        codebook: &[ContinuousHV],
        router: &dyn CantorRouter,
        abstain_threshold: f32,
    ) -> TrialResult {
        let start = Instant::now();
        let mut pyramid = PyramidCantorVector::new(config, None);
        let zero = ContinuousHV::zero(config.leaf_dim);
        let mut stored = Vec::with_capacity(examples.len());
        let mut leaf_counts = vec![0usize; config.branching];

        for example in examples {
            let value_idx = example.value_idx % codebook.len();
            let binding = example.role.bind(&codebook[value_idx]);
            let leaf_idx = router.route_and_record(&example.role, &zero, config.branching);
            leaf_counts[leaf_idx] += 1;
            let leaf = pyramid.find_node(1, leaf_idx).unwrap().clone();
            pyramid.bundle_at_node(&leaf, &binding);
            stored.push((example.role.clone(), value_idx, leaf_idx));
        }

        score_trial(
            router_name,
            config,
            &pyramid,
            &stored,
            codebook,
            &leaf_counts,
            abstain_threshold,
            start.elapsed().as_secs_f32() * 1000.0,
        )
    }

    fn run_oracle_trial(
        router_name: String,
        config: CantorHdcConfig,
        examples: &[Example],
        codebook: &[ContinuousHV],
        abstain_threshold: f32,
    ) -> TrialResult {
        let start = Instant::now();
        let mut pyramid = PyramidCantorVector::new(config, None);
        let mut stored = Vec::with_capacity(examples.len());
        let mut leaf_counts = vec![0usize; config.branching];

        for (idx, example) in examples.iter().enumerate() {
            let value_idx = example.value_idx % codebook.len();
            let binding = example.role.bind(&codebook[value_idx]);
            let leaf_idx = idx % config.branching;
            leaf_counts[leaf_idx] += 1;
            let leaf = pyramid.find_node(1, leaf_idx).unwrap().clone();
            pyramid.bundle_at_node(&leaf, &binding);
            stored.push((example.role.clone(), value_idx, leaf_idx));
        }

        score_trial(
            router_name,
            config,
            &pyramid,
            &stored,
            codebook,
            &leaf_counts,
            abstain_threshold,
            start.elapsed().as_secs_f32() * 1000.0,
        )
    }

    fn score_trial(
        router_name: String,
        config: CantorHdcConfig,
        pyramid: &PyramidCantorVector,
        stored: &[(ContinuousHV, usize, usize)],
        codebook: &[ContinuousHV],
        leaf_counts: &[usize],
        abstain_threshold: f32,
        latency_ms: f32,
    ) -> TrialResult {
        let mut hits1 = 0usize;
        let mut hits3 = 0usize;
        let mut total_margin = 0.0;
        let mut abstained = 0usize;
        let mut answered = 0usize;
        let mut answered_hits = 0usize;

        for (role, correct_idx, leaf_idx) in stored {
            let leaf = pyramid.find_node(1, *leaf_idx).unwrap();
            let recovered = ContinuousHV::from_slice(pyramid.node_data(leaf)).bind(&role.inverse());
            let mut sims: Vec<(usize, f32)> = codebook
                .iter()
                .enumerate()
                .map(|(idx, candidate)| (idx, recovered.similarity(candidate)))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let margin = sims[0].1 - sims[1].1;
            total_margin += margin;

            if sims[0].0 == *correct_idx {
                hits1 += 1;
            }
            if sims.iter().take(3).any(|(idx, _)| *idx == *correct_idx) {
                hits3 += 1;
            }

            if margin < abstain_threshold {
                abstained += 1;
            } else {
                answered += 1;
                if sims[0].0 == *correct_idx {
                    answered_hits += 1;
                }
            }
        }

        let n = stored.len().max(1);
        TrialResult {
            router: router_name,
            top1: hits1 as f32 / n as f32,
            top3: hits3 as f32 / n as f32,
            mean_margin: total_margin / n as f32,
            abstention_rate: abstained as f32 / n as f32,
            answered_accuracy: if answered > 0 {
                answered_hits as f32 / answered as f32
            } else {
                0.0
            },
            load_entropy: calculate_entropy(leaf_counts),
            max_leaf_load: *leaf_counts.iter().max().unwrap_or(&0),
            mean_leaf_load: n as f32 / config.branching as f32,
            latency_ms,
        }
    }

    fn calculate_entropy(counts: &[usize]) -> f32 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        counts
            .iter()
            .filter(|count| **count > 0)
            .map(|count| {
                let p = *count as f32 / total as f32;
                -(p * p.log2())
            })
            .sum()
    }

    fn summarize_trials(trials: &[TrialResult], seeds: usize) -> Vec<RouterSummary> {
        let mut router_names = Vec::new();
        for trial in trials {
            if !router_names.contains(&trial.router) {
                router_names.push(trial.router.clone());
            }
        }

        router_names
            .into_iter()
            .map(|router| {
                let matching: Vec<&TrialResult> = trials
                    .iter()
                    .filter(|trial| trial.router == router)
                    .collect();
                let denom = matching.len().max(seeds).max(1) as f32;
                RouterSummary {
                    router,
                    top1: avg(&matching, |trial| trial.top1, denom),
                    top3: avg(&matching, |trial| trial.top3, denom),
                    mean_margin: avg(&matching, |trial| trial.mean_margin, denom),
                    abstention_rate: avg(&matching, |trial| trial.abstention_rate, denom),
                    answered_accuracy: avg(&matching, |trial| trial.answered_accuracy, denom),
                    load_entropy: avg(&matching, |trial| trial.load_entropy, denom),
                    max_leaf_load: matching
                        .iter()
                        .map(|trial| trial.max_leaf_load)
                        .max()
                        .unwrap_or(0),
                    mean_leaf_load: avg(&matching, |trial| trial.mean_leaf_load, denom),
                    latency_ms: avg(&matching, |trial| trial.latency_ms, denom),
                    oracle_gap_top1: 0.0,
                    oracle_gap_margin: 0.0,
                }
            })
            .collect()
    }

    fn avg<F: Fn(&TrialResult) -> f32>(trials: &[&TrialResult], f: F, denom: f32) -> f32 {
        trials.iter().map(|trial| f(trial)).sum::<f32>() / denom
    }

    fn diagnose(results: &[RouterSummary]) -> String {
        let Some(high_capacity_oracle) = results
            .iter()
            .find(|result| result.router == "OracleHighCapacity")
        else {
            return "missing high-capacity oracle result; diagnosis unavailable".into();
        };
        let same_capacity_oracle = results
            .iter()
            .find(|result| result.router == "OracleSameCapacity");
        let best_non_oracle = results
            .iter()
            .filter(|result| !result.router.starts_with("Oracle"))
            .max_by(|a, b| a.top1.partial_cmp(&b.top1).unwrap());

        let Some(best) = best_non_oracle else {
            return "missing non-oracle routers; diagnosis unavailable".into();
        };

        let high_capacity_gap = high_capacity_oracle.top1 - best.top1;
        if let Some(same_capacity_oracle) = same_capacity_oracle {
            let same_capacity_gap = same_capacity_oracle.top1 - best.top1;
            if same_capacity_gap < 0.05 && high_capacity_gap > 0.20 {
                return format!(
                    "fixed-capacity bottleneck likely: same-capacity oracle is near best non-oracle ({}), but high-capacity oracle gap is {:.1} points",
                    best.router,
                    high_capacity_gap * 100.0
                );
            }
        }

        if high_capacity_gap > 0.20 {
            format!(
                "routing bottleneck likely: high-capacity oracle top1 exceeds best non-oracle ({}) by {:.1} points",
                best.router,
                high_capacity_gap * 100.0
            )
        } else if high_capacity_oracle.top1 < 0.30 {
            format!(
                "bundling/retrieval bottleneck likely: high-capacity oracle top1 is only {:.1}%",
                high_capacity_oracle.top1 * 100.0
            )
        } else {
            format!(
                "router gap is moderate: best non-oracle is {} with {:.1}% top1",
                best.router,
                best.top1 * 100.0
            )
        }
    }

    fn write_reports(path: &Path, report: &BakeoffReport) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let json = serde_json::to_string_pretty(report).map_err(|err| err.to_string())?;
        fs::write(path, json).map_err(|err| err.to_string())?;

        let csv_path = path.with_extension("csv");
        fs::write(csv_path, to_csv(report)).map_err(|err| err.to_string())?;
        Ok(())
    }

    fn to_csv(report: &BakeoffReport) -> String {
        let mut csv = String::from(
            "router,top1,top3,mean_margin,abstention_rate,answered_accuracy,load_entropy,max_leaf_load,mean_leaf_load,latency_ms,oracle_gap_top1,oracle_gap_margin\n",
        );
        for result in &report.results {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6}\n",
                result.router,
                result.top1,
                result.top3,
                result.mean_margin,
                result.abstention_rate,
                result.answered_accuracy,
                result.load_entropy,
                result.max_leaf_load,
                result.mean_leaf_load,
                result.latency_ms,
                result.oracle_gap_top1,
                result.oracle_gap_margin
            ));
        }
        csv
    }

    fn print_summary(report: &BakeoffReport) {
        println!("{} bakeoff {}", report.architecture, report.version);
        println!(
            "objects={} seeds={} dim={} branching={} leaf_dim={} tau={:.3}",
            report.objects,
            report.seeds,
            report.dim,
            report.branching,
            report.leaf_dim,
            report.abstain_threshold
        );
        println!("{}", report.diagnosis);
        println!(
            "{:20} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>6} {:>9}",
            "Router", "Top1", "Top3", "Margin", "Abstain", "AnsAcc", "Entropy", "MaxL", "Latency"
        );
        for result in &report.results {
            println!(
                "{:20} {:>6.1}% {:>6.1}% {:>8.4} {:>7.1}% {:>7.1}% {:>8.2} {:>6} {:>8.1}ms",
                result.router,
                result.top1 * 100.0,
                result.top3 * 100.0,
                result.mean_margin,
                result.abstention_rate * 100.0,
                result.answered_accuracy * 100.0,
                result.load_entropy,
                result.max_leaf_load,
                result.latency_ms
            );
        }
    }
}

#[cfg(feature = "cantor-hdc")]
fn main() {
    app::main();
}

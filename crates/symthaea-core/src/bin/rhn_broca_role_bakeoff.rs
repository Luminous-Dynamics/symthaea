// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(not(feature = "cantor-hdc"))]
fn main() {
    eprintln!("rhn_broca_role_bakeoff requires --features cantor-hdc");
    std::process::exit(2);
}

#[cfg(feature = "cantor-hdc")]
mod app {
    use serde::Serialize;
    use std::collections::HashSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use symthaea_core::hdc::cantor_pyramid::{
        BundleMode, CantorRouter, HashRouter, HypercubeRouter, LoadBalancedHashRouter,
        PrefixMaxRouter, PrototypeIndex, PrototypeRouter, SmallWorldRouter,
    };
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    #[derive(Debug, Clone)]
    struct Args {
        frames: usize,
        seeds: usize,
        dim: usize,
        branching: usize,
        abstain_threshold: f32,
        redundancy_k: usize,
        retrieval_fanout: usize,
        shortcuts: usize,
        out: PathBuf,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                frames: 96,
                seeds: 3,
                dim: 16_384,
                branching: 64,
                abstain_threshold: 0.03,
                redundancy_k: 1,
                retrieval_fanout: 3,
                shortcuts: 2,
                out: PathBuf::from("reports/rhn_broca_role_v011.json"),
            }
        }
    }

    #[derive(Debug, Clone, Serialize)]
    struct RoleBakeoffReport {
        architecture: &'static str,
        version: &'static str,
        task: &'static str,
        frames: usize,
        seeds: usize,
        dim: usize,
        branching: usize,
        abstain_threshold: f32,
        redundancy_k: usize,
        retrieval_fanout: usize,
        shortcuts: usize,
        diagnosis: String,
        results: Vec<RouterSummary>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct RouterSummary {
        router: String,
        top1: f32,
        top3: f32,
        role_reversal_rate: f32,
        subject_preservation: f32,
        relation_preservation: f32,
        object_preservation: f32,
        frame_preservation: f32,
        abstention_rate: f32,
        answered_accuracy: f32,
        mean_margin: f32,
        load_entropy: f32,
        max_leaf_load: usize,
        mean_leaf_load: f32,
        retrieval_fanout: usize,
        redundancy_k: usize,
        searched_nodes_mean: f32,
        searched_nodes_max: usize,
        latency_ms: f32,
        latency_per_query_ms: f32,
        top1_per_fanout: f32,
        answered_accuracy_per_latency_ms: f32,
    }

    #[derive(Debug, Clone)]
    struct TrialResult {
        router: String,
        top1: f32,
        top3: f32,
        role_reversal_rate: f32,
        subject_preservation: f32,
        relation_preservation: f32,
        object_preservation: f32,
        frame_preservation: f32,
        abstention_rate: f32,
        answered_accuracy: f32,
        mean_margin: f32,
        load_entropy: f32,
        max_leaf_load: usize,
        mean_leaf_load: f32,
        retrieval_fanout: usize,
        redundancy_k: usize,
        searched_nodes_mean: f32,
        searched_nodes_max: usize,
        latency_ms: f32,
        latency_per_query_ms: f32,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RoleFrameSpec {
        subject: &'static str,
        relation: &'static str,
        object: &'static str,
    }

    #[derive(Clone)]
    struct RoleFrame {
        spec: RoleFrameSpec,
        vector: ContinuousHV,
        context: ContinuousHV,
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
        parse_args_from(std::env::args().skip(1), &mut args)?;
        Ok(args)
    }

    fn parse_args_from<I>(args_iter: I, args: &mut Args) -> Result<(), String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut iter = args_iter.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--frames" => args.frames = parse_next(&mut iter, "--frames")?,
                "--seeds" => args.seeds = parse_next(&mut iter, "--seeds")?,
                "--dim" => args.dim = parse_next(&mut iter, "--dim")?,
                "--branching" => args.branching = parse_next(&mut iter, "--branching")?,
                "--abstain" => args.abstain_threshold = parse_next(&mut iter, "--abstain")?,
                "--redundancy-k" => args.redundancy_k = parse_next(&mut iter, "--redundancy-k")?,
                "--retrieval-fanout" => {
                    args.retrieval_fanout = parse_next(&mut iter, "--retrieval-fanout")?
                }
                "--shortcuts" => args.shortcuts = parse_next(&mut iter, "--shortcuts")?,
                "--out" => args.out = PathBuf::from(next_value(&mut iter, "--out")?),
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }
        validate_args(args)
    }

    fn validate_args(args: &Args) -> Result<(), String> {
        if args.frames == 0 {
            return Err("--frames must be > 0".into());
        }
        if args.seeds == 0 {
            return Err("--seeds must be > 0".into());
        }
        if args.dim == 0 {
            return Err("--dim must be > 0".into());
        }
        if args.branching == 0 {
            return Err("--branching must be > 0".into());
        }
        if args.redundancy_k == 0 {
            return Err("--redundancy-k must be > 0".into());
        }
        if args.retrieval_fanout == 0 {
            return Err("--retrieval-fanout must be > 0".into());
        }
        Ok(())
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
            "usage: rhn_broca_role_bakeoff [--frames N] [--seeds N] [--dim N] \\
             [--branching N] [--abstain F] [--redundancy-k N] \\
             [--retrieval-fanout N] [--shortcuts N] [--out PATH]"
        );
    }

    fn run_bakeoff(args: &Args) -> RoleBakeoffReport {
        let mut trials = Vec::new();
        for seed_idx in 0..args.seeds {
            let seed = seed_idx as u64;
            let frames = build_role_frames(args.frames, args.dim, seed);
            let hypercube_dims = args.branching.next_power_of_two().trailing_zeros() as usize;
            let prototype_keys =
                build_prototype_keys(&frames, args.branching, hypercube_dims, seed);
            let routers = build_routers(args, hypercube_dims, seed, &prototype_keys);

            for (name, router) in routers {
                trials.push(run_trial(name, args, &frames, router.as_ref()));
            }
        }

        let results = summarize_trials(&trials, args.seeds);
        let diagnosis = diagnose(&results);
        RoleBakeoffReport {
            architecture: "RHN",
            version: "rhn-v0.11",
            task: "broca_semantic_role_preservation",
            frames: args.frames,
            seeds: args.seeds,
            dim: args.dim,
            branching: args.branching,
            abstain_threshold: args.abstain_threshold,
            redundancy_k: args.redundancy_k,
            retrieval_fanout: args.retrieval_fanout,
            shortcuts: args.shortcuts,
            diagnosis,
            results,
        }
    }

    fn build_role_frames(count: usize, dim: usize, seed: u64) -> Vec<RoleFrame> {
        role_pairs()
            .into_iter()
            .cycle()
            .take(count)
            .enumerate()
            .map(|(idx, spec)| {
                let subject_role = symbol_hv(dim, seed, "role:subject");
                let relation_role = symbol_hv(dim, seed, "role:relation");
                let object_role = symbol_hv(dim, seed, "role:object");
                let discourse_role = symbol_hv(dim, seed, "role:discourse");
                let subject = symbol_hv(dim, seed, spec.subject);
                let relation = symbol_hv(dim, seed, spec.relation);
                let object = symbol_hv(dim, seed, spec.object);
                let vector = ContinuousHV::bundle_owned(&[
                    subject_role.bind(&subject),
                    relation_role.bind(&relation),
                    object_role.bind(&object),
                    ContinuousHV::random(dim, seed + 70_000 + idx as u64).scale(0.03),
                ]);
                let context = discourse_role.bind(&relation);
                RoleFrame {
                    spec,
                    vector,
                    context,
                }
            })
            .collect()
    }

    fn role_pairs() -> Vec<RoleFrameSpec> {
        vec![
            pair("alice", "helps", "bob"),
            pair("bob", "helps", "alice"),
            pair("doctor", "warns", "patient"),
            pair("patient", "warns", "doctor"),
            pair("child", "gives_book_to", "teacher"),
            pair("teacher", "gives_book_to", "child"),
            pair("engineer", "repairs", "bridge"),
            pair("bridge", "supports", "engineer"),
            pair("mentor", "questions", "student"),
            pair("student", "questions", "mentor"),
            pair("agent", "refuses", "unsafe_request"),
            pair("unsafe_request", "pressures", "agent"),
            pair("caretaker", "comforts", "resident"),
            pair("resident", "trusts", "caretaker"),
            pair("reviewer", "critiques", "patch"),
            pair("patch", "changes", "module"),
        ]
    }

    fn pair(subject: &'static str, relation: &'static str, object: &'static str) -> RoleFrameSpec {
        RoleFrameSpec {
            subject,
            relation,
            object,
        }
    }

    fn symbol_hv(dim: usize, seed: u64, symbol: &str) -> ContinuousHV {
        ContinuousHV::random(dim, seed ^ stable_hash(symbol))
    }

    fn stable_hash(value: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in value.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn build_prototype_keys(
        frames: &[RoleFrame],
        branching: usize,
        hypercube_dims: usize,
        seed: u64,
    ) -> Vec<ContinuousHV> {
        let router = HypercubeRouter {
            dimensions: hypercube_dims,
            seed,
        };
        let assignments: Vec<(usize, ContinuousHV)> = frames
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % 2 == 0)
            .map(|(_, frame)| {
                (
                    router.route(&frame.vector, &frame.context, branching),
                    frame.vector.clone(),
                )
            })
            .collect();
        PrototypeIndex::from_assignments(
            &assignments,
            branching,
            frames
                .first()
                .map_or(0, |frame| frame.vector.as_slice().len()),
            BundleMode::UnitNormalize,
        )
        .leaf_keys
    }

    fn build_routers(
        args: &Args,
        hypercube_dims: usize,
        seed: u64,
        prototype_keys: &[ContinuousHV],
    ) -> Vec<(String, Box<dyn CantorRouter>)> {
        vec![
            ("Hash".into(), Box::new(HashRouter)),
            (
                "LB-Hash-2".into(),
                Box::new(LoadBalancedHashRouter::new(args.branching, 2)),
            ),
            (
                "Hypercube".into(),
                Box::new(HypercubeRouter {
                    dimensions: hypercube_dims,
                    seed,
                }),
            ),
            ("PrefixMax".into(), Box::new(PrefixMaxRouter)),
            (
                "PrototypeLearned".into(),
                Box::new(PrototypeRouter {
                    leaf_keys: prototype_keys.to_vec(),
                }),
            ),
            (
                "SmallWorldLearned".into(),
                Box::new(SmallWorldRouter {
                    dimensions: hypercube_dims,
                    seed,
                    leaf_keys: prototype_keys.to_vec(),
                    shortcuts: args.shortcuts,
                }),
            ),
        ]
    }

    fn run_trial(
        router_name: String,
        args: &Args,
        frames: &[RoleFrame],
        router: &dyn CantorRouter,
    ) -> TrialResult {
        let start = Instant::now();
        let mut buckets = vec![Vec::<usize>::new(); args.branching];
        let mut leaf_counts = vec![0usize; args.branching];

        for (idx, frame) in frames.iter().enumerate() {
            let primary = router.route_and_record(&frame.vector, &frame.context, args.branching);
            for leaf in candidate_leaves(primary, args.branching, args.redundancy_k) {
                buckets[leaf].push(idx);
            }
            leaf_counts[primary] += 1;
        }

        let mut correct = 0usize;
        let mut top3 = 0usize;
        let mut reversals = 0usize;
        let mut subject_ok = 0usize;
        let mut relation_ok = 0usize;
        let mut object_ok = 0usize;
        let mut frame_ok = 0usize;
        let mut abstained = 0usize;
        let mut answered = 0usize;
        let mut answered_correct = 0usize;
        let mut margin_sum = 0.0f32;
        let mut searched_sum = 0usize;
        let mut searched_max = 0usize;

        for (expected_idx, frame) in frames.iter().enumerate() {
            let primary = router.route(&frame.vector, &frame.context, args.branching);
            let leaves = candidate_leaves(primary, args.branching, args.retrieval_fanout);
            searched_sum += leaves.len();
            searched_max = searched_max.max(leaves.len());

            let mut seen = HashSet::new();
            let mut scored = Vec::new();
            for leaf in leaves {
                for &candidate_idx in &buckets[leaf] {
                    if seen.insert(candidate_idx) {
                        let score = frame.vector.similarity(&frames[candidate_idx].vector);
                        scored.push((candidate_idx, score));
                    }
                }
            }
            if scored.is_empty() {
                abstained += 1;
                continue;
            }

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let best_idx = scored[0].0;
            let second_score = scored.get(1).map_or(-1.0, |(_, score)| *score);
            let margin = scored[0].1 - second_score;
            margin_sum += margin;

            if margin < args.abstain_threshold {
                abstained += 1;
            } else {
                answered += 1;
                if best_idx == expected_idx {
                    answered_correct += 1;
                }
            }

            if best_idx == expected_idx {
                correct += 1;
            }
            if scored.iter().take(3).any(|(idx, _)| *idx == expected_idx) {
                top3 += 1;
            }

            let predicted = &frames[best_idx].spec;
            let expected = &frame.spec;
            if predicted.subject == expected.subject {
                subject_ok += 1;
            }
            if predicted.relation == expected.relation {
                relation_ok += 1;
            }
            if predicted.object == expected.object {
                object_ok += 1;
            }
            if predicted == expected {
                frame_ok += 1;
            }
            if is_role_reversal(expected, predicted) {
                reversals += 1;
            }
        }

        let total = frames.len().max(1) as f32;
        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        TrialResult {
            router: router_name,
            top1: correct as f32 / total,
            top3: top3 as f32 / total,
            role_reversal_rate: reversals as f32 / total,
            subject_preservation: subject_ok as f32 / total,
            relation_preservation: relation_ok as f32 / total,
            object_preservation: object_ok as f32 / total,
            frame_preservation: frame_ok as f32 / total,
            abstention_rate: abstained as f32 / total,
            answered_accuracy: if answered == 0 {
                0.0
            } else {
                answered_correct as f32 / answered as f32
            },
            mean_margin: margin_sum / total,
            load_entropy: entropy(&leaf_counts),
            max_leaf_load: leaf_counts.iter().copied().max().unwrap_or(0),
            mean_leaf_load: leaf_counts.iter().sum::<usize>() as f32 / args.branching as f32,
            retrieval_fanout: args.retrieval_fanout,
            redundancy_k: args.redundancy_k,
            searched_nodes_mean: searched_sum as f32 / total,
            searched_nodes_max: searched_max,
            latency_ms,
            latency_per_query_ms: latency_ms / total,
        }
    }

    fn candidate_leaves(primary: usize, branching: usize, limit: usize) -> Vec<usize> {
        let mut leaves = Vec::new();
        push_unique(&mut leaves, primary % branching);

        if branching.is_power_of_two() {
            let dimensions = branching.trailing_zeros() as usize;
            for neighbor in HypercubeRouter::hamming_neighbors(primary % branching, dimensions) {
                push_unique(&mut leaves, neighbor % branching);
                if leaves.len() >= limit {
                    return leaves;
                }
            }
        }

        let mut offset = 1usize;
        while leaves.len() < limit {
            push_unique(&mut leaves, (primary + offset) % branching);
            offset += 1;
        }
        leaves
    }

    fn push_unique(values: &mut Vec<usize>, value: usize) {
        if !values.contains(&value) {
            values.push(value);
        }
    }

    fn is_role_reversal(expected: &RoleFrameSpec, predicted: &RoleFrameSpec) -> bool {
        predicted.subject == expected.object
            && predicted.object == expected.subject
            && predicted.relation == expected.relation
    }

    fn entropy(counts: &[usize]) -> f32 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let entropy = counts.iter().fold(0.0, |acc, count| {
            if *count == 0 {
                acc
            } else {
                let p = *count as f32 / total as f32;
                acc - p * p.log2()
            }
        });
        let max_entropy = (counts.len().max(1) as f32).log2();
        if max_entropy <= f32::EPSILON {
            0.0
        } else {
            entropy / max_entropy
        }
    }

    fn summarize_trials(trials: &[TrialResult], seeds: usize) -> Vec<RouterSummary> {
        let mut names: Vec<String> = trials.iter().map(|trial| trial.router.clone()).collect();
        names.sort();
        names.dedup();

        names
            .into_iter()
            .map(|name| {
                let matching: Vec<&TrialResult> =
                    trials.iter().filter(|trial| trial.router == name).collect();
                let divisor = seeds.max(1) as f32;
                let avg = |f: fn(&TrialResult) -> f32| {
                    matching.iter().map(|trial| f(trial)).sum::<f32>() / divisor
                };
                let max_leaf_load = matching
                    .iter()
                    .map(|trial| trial.max_leaf_load)
                    .max()
                    .unwrap_or(0);
                let searched_nodes_max = matching
                    .iter()
                    .map(|trial| trial.searched_nodes_max)
                    .max()
                    .unwrap_or(0);
                let top1 = avg(|trial| trial.top1);
                let answered_accuracy = avg(|trial| trial.answered_accuracy);
                let latency_per_query_ms = avg(|trial| trial.latency_per_query_ms);
                RouterSummary {
                    router: name,
                    top1,
                    top3: avg(|trial| trial.top3),
                    role_reversal_rate: avg(|trial| trial.role_reversal_rate),
                    subject_preservation: avg(|trial| trial.subject_preservation),
                    relation_preservation: avg(|trial| trial.relation_preservation),
                    object_preservation: avg(|trial| trial.object_preservation),
                    frame_preservation: avg(|trial| trial.frame_preservation),
                    abstention_rate: avg(|trial| trial.abstention_rate),
                    answered_accuracy,
                    mean_margin: avg(|trial| trial.mean_margin),
                    load_entropy: avg(|trial| trial.load_entropy),
                    max_leaf_load,
                    mean_leaf_load: avg(|trial| trial.mean_leaf_load),
                    retrieval_fanout: matching.first().map_or(0, |trial| trial.retrieval_fanout),
                    redundancy_k: matching.first().map_or(0, |trial| trial.redundancy_k),
                    searched_nodes_mean: avg(|trial| trial.searched_nodes_mean),
                    searched_nodes_max,
                    latency_ms: avg(|trial| trial.latency_ms),
                    latency_per_query_ms,
                    top1_per_fanout: top1
                        / matching
                            .first()
                            .map_or(1.0, |trial| trial.retrieval_fanout.max(1) as f32),
                    answered_accuracy_per_latency_ms: if latency_per_query_ms <= f32::EPSILON {
                        0.0
                    } else {
                        answered_accuracy / latency_per_query_ms
                    },
                }
            })
            .collect()
    }

    fn diagnose(results: &[RouterSummary]) -> String {
        let Some(best) = results.iter().max_by(|a, b| {
            a.frame_preservation
                .partial_cmp(&b.frame_preservation)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return "no results".into();
        };

        if best.role_reversal_rate > 0.1 {
            format!(
                "{} preserves frames best, but role reversal remains high; routing is not Broca-ready",
                best.router
            )
        } else if best.frame_preservation >= 0.8 {
            format!(
                "{} shows strong role preservation; candidate for feature-gated Broca planner tests",
                best.router
            )
        } else {
            format!(
                "{} is best, but frame preservation is still below the Broca integration threshold",
                best.router
            )
        }
    }

    fn write_reports(path: &Path, report: &RoleBakeoffReport) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(report).unwrap())?;

        let csv_path = path.with_extension("csv");
        let mut csv = String::from(
            "router,top1,top3,role_reversal_rate,subject_preservation,relation_preservation,\
             object_preservation,frame_preservation,abstention_rate,answered_accuracy,\
             mean_margin,load_entropy,max_leaf_load,mean_leaf_load,retrieval_fanout,\
             redundancy_k,searched_nodes_mean,searched_nodes_max,latency_ms,\
             latency_per_query_ms,top1_per_fanout,answered_accuracy_per_latency_ms\n",
        );
        for result in &report.results {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6}\n",
                result.router,
                result.top1,
                result.top3,
                result.role_reversal_rate,
                result.subject_preservation,
                result.relation_preservation,
                result.object_preservation,
                result.frame_preservation,
                result.abstention_rate,
                result.answered_accuracy,
                result.mean_margin,
                result.load_entropy,
                result.max_leaf_load,
                result.mean_leaf_load,
                result.retrieval_fanout,
                result.redundancy_k,
                result.searched_nodes_mean,
                result.searched_nodes_max,
                result.latency_ms,
                result.latency_per_query_ms,
                result.top1_per_fanout,
                result.answered_accuracy_per_latency_ms
            ));
        }
        fs::write(csv_path, csv)
    }

    fn print_summary(report: &RoleBakeoffReport) {
        println!(
            "RHN v0.11 Broca role bakeoff: frames={} seeds={} dim={} branching={}",
            report.frames, report.seeds, report.dim, report.branching
        );
        println!("{}", report.diagnosis);
        println!(
            "{:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "router", "top1", "top3", "rev", "subj", "obj", "lat"
        );
        for result in &report.results {
            println!(
                "{:<22} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.2}ms",
                result.router,
                result.top1 * 100.0,
                result.top3 * 100.0,
                result.role_reversal_rate * 100.0,
                result.subject_preservation * 100.0,
                result.object_preservation * 100.0,
                result.latency_ms
            );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn detects_exact_role_reversal() {
            let expected = pair("alice", "helps", "bob");
            let predicted = pair("bob", "helps", "alice");
            assert!(is_role_reversal(&expected, &predicted));
        }

        #[test]
        fn distinguishes_directional_frames() {
            let frames = build_role_frames(2, 512, 7);
            assert_ne!(frames[0].spec, frames[1].spec);
            assert!(frames[0].vector.similarity(&frames[1].vector) < 0.95);
        }

        #[test]
        fn parses_cost_arguments() {
            let mut args = Args::default();
            parse_args_from(
                [
                    "--redundancy-k",
                    "2",
                    "--retrieval-fanout",
                    "4",
                    "--frames",
                    "16",
                ]
                .into_iter()
                .map(String::from),
                &mut args,
            )
            .unwrap();
            assert_eq!(args.redundancy_k, 2);
            assert_eq!(args.retrieval_fanout, 4);
            assert_eq!(args.frames, 16);
        }
    }
}

#[cfg(feature = "cantor-hdc")]
fn main() {
    app::main();
}

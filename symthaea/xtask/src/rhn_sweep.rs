pub fn run_sweep(
    dims: Vec<usize>,
    objects: Vec<usize>,
    seeds: Vec<usize>,
    branchings: Vec<usize>,
    thresholds: Vec<usize>,
    redundancy_ks: Vec<usize>,
    fanouts: Vec<usize>,
    policies: Vec<String>,
    out: std::path::PathBuf,
) -> anyhow::Result<()> {
    use serde::{Deserialize, Serialize};
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::process::Command;

    #[derive(Deserialize, Serialize)]
    struct BakeoffResult {
        router: String,
        top1: f32,
        top3: f32,
        mean_margin: f32,
        searched_nodes_mean: f32,
        searched_nodes_max: usize,
        missing_leaf_count: usize,
        logical_storage_multiplier: f32,
        top1_per_logical_storage: f32,
        top1_per_fanout: f32,
        oracle_gap_top1: f32,
    }

    #[derive(Deserialize, Serialize)]
    struct BakeoffOutput {
        results: Vec<BakeoffResult>,
    }

    let runs_dir = out.join("runs");
    fs::create_dir_all(&runs_dir)?;
    let raw_runs = File::create(out.join("raw_runs.jsonl"))?;
    let mut raw_writer = BufWriter::new(raw_runs);
    let failed_runs = File::create(out.join("failed_runs.jsonl"))?;
    let mut failed_writer = BufWriter::new(failed_runs);

    let mut all_results = Vec::new();
    let mut inv_count = 0;
    let mut success_count = 0;
    let mut fail_count = 0;

    for &dim in &dims {
        for &obj in &objects {
            for &seed in &seeds {
                for &branching in &branchings {
                    for &threshold in &thresholds {
                        for &k in &redundancy_ks {
                            for &fanout in &fanouts {
                                for policy in &policies {
                                    inv_count += 1;
                                    let run_id = format!(
                                        "d{}_o{}_s{}_b{}_t{}_k{}_f{}_{}",
                                        dim, obj, seed, branching, threshold, k, fanout, policy
                                    );
                                    let output_path = runs_dir.join(format!("{}.json", run_id));

                                    let status = Command::new("cargo")
                                        .args([
                                            "run",
                                            "--bin",
                                            "hch_bakeoff",
                                            "--features",
                                            "cantor-hdc",
                                            "--locked",
                                            "--",
                                            "--dim",
                                            &dim.to_string(),
                                            "--objects",
                                            &obj.to_string(),
                                            "--seeds",
                                            &seed.to_string(),
                                            "--branching",
                                            &branching.to_string(),
                                            "--split-threshold",
                                            &threshold.to_string(),
                                            "--redundancy-k",
                                            &k.to_string(),
                                            "--retrieval-fanout",
                                            &fanout.to_string(),
                                            "--retrieval-policy",
                                            policy,
                                            "--out",
                                            output_path.to_str().unwrap(),
                                        ])
                                        .status()?;

                                    if status.success() {
                                        success_count += 1;
                                        let content = fs::read_to_string(&output_path)?;
                                        let output: BakeoffOutput = serde_json::from_str(&content)?;

                                        // Write JSONL entry
                                        raw_writer
                                            .write_all(content.replace('\n', "").as_bytes())?;
                                        raw_writer.write_all(b"\n")?;

                                        for r in output.results {
                                            all_results.push((
                                                r,
                                                dim,
                                                obj,
                                                seed,
                                                branching,
                                                threshold,
                                                k,
                                                fanout,
                                                policy.clone(),
                                            ));
                                        }
                                    } else {
                                        fail_count += 1;
                                        writeln!(
                                            failed_writer,
                                            "{{\"run_id\": \"{}\", \"status\": \"failed\"}}",
                                            run_id
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    raw_writer.flush()?;
    failed_writer.flush()?;

    // Generate CSV
    let mut csv = csv::Writer::from_path(out.join("aggregate.csv"))?;
    csv.write_record(&[
        "dim",
        "objects",
        "seed",
        "branching",
        "split_threshold",
        "redundancy_k",
        "fanout",
        "policy",
        "router",
        "top1",
        "top3",
        "margin",
        "searched_mean",
        "searched_max",
        "missing_leaf_count",
        "log_storage",
        "top1_per_storage",
        "top1_per_fanout",
        "oracle_gap_top1",
    ])?;
    for (r, dim, obj, seed, branching, threshold, k, fanout, policy) in &all_results {
        csv.write_record(&[
            dim,
            obj,
            seed,
            branching,
            threshold,
            k,
            fanout,
            policy,
            &r.router,
            r.top1.to_string(),
            r.top3.to_string(),
            r.mean_margin.to_string(),
            r.searched_nodes_mean.to_string(),
            r.searched_nodes_max.to_string(),
            r.missing_leaf_count.to_string(),
            r.logical_storage_multiplier.to_string(),
            r.top1_per_logical_storage.to_string(),
            r.top1_per_fanout.to_string(),
            r.oracle_gap_top1.to_string(),
        ])?;
    }

    csv.flush()?;

    // Generate Markdown summary
    let mut summary = File::create(out.join("summary.md"))?;
    writeln!(summary, "# RHN Sweep Summary\n")?;
    writeln!(summary, "Benchmark Invocations: {}", inv_count)?;
    writeln!(summary, "Successful Invocations: {}", success_count)?;
    writeln!(summary, "Failed Invocations: {}", fail_count)?;
    writeln!(summary, "Result Rows: {}", all_results.len())?;
    writeln!(
        summary,
        "Missing Leaf Count Sum: {}",
        all_results
            .iter()
            .map(|(r, _, _, _, _, _, _, _, _)| r.missing_leaf_count)
            .sum::<usize>()
    )?;

    let best_overall = all_results.iter().max_by(|a, b| {
        a.0.top1_per_fanout
            .partial_cmp(&b.0.top1_per_fanout)
            .unwrap()
    });
    let best_non_oracle = all_results
        .iter()
        .filter(|(r, _, _, _, _, _, _, _, _)| !r.router.starts_with("Oracle"))
        .max_by(|a, b| {
            a.0.top1_per_fanout
                .partial_cmp(&b.0.top1_per_fanout)
                .unwrap()
        });

    if let Some((r, _, _, _, _, _, _, _, _)) = best_overall {
        writeln!(
            summary,
            "\n### Best Overall Router (by top1_per_fanout)\n- Router: {}\n- Score: {:.4}",
            r.router, r.top1_per_fanout
        )?;
    }
    if let Some((r, _, _, _, _, _, _, _, _)) = best_non_oracle {
        writeln!(
            summary,
            "\n### Best Non-Oracle Router (by top1_per_fanout)\n- Router: {}\n- Score: {:.4}",
            r.router, r.top1_per_fanout
        )?;
    }

    Ok(())
}

pub fn run_finalize(input_dir: std::path::PathBuf, out: std::path::PathBuf) -> anyhow::Result<()> {
    use serde::{Deserialize, Serialize};
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};

    #[derive(Deserialize, Serialize)]
    struct BakeoffResult {
        router: String,
        top1: f32,
        top3: f32,
        mean_margin: f32,
        searched_nodes_mean: f32,
        searched_nodes_max: usize,
        missing_leaf_count: usize,
        logical_storage_multiplier: f32,
        top1_per_logical_storage: f32,
        top1_per_fanout: f32,
        oracle_gap_top1: f32,
    }

    #[derive(Deserialize, Serialize)]
    struct BakeoffOutput {
        results: Vec<BakeoffResult>,
    }

    fs::create_dir_all(&out)?;
    let raw_runs = File::create(out.join("raw_runs.jsonl"))?;
    let mut raw_writer = BufWriter::new(raw_runs);
    let failed_runs = File::create(out.join("failed_runs.jsonl"))?;
    let mut failed_writer = BufWriter::new(failed_runs);

    let mut all_results = Vec::new();
    let mut success_count = 0;
    let mut fail_count = 0;

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            if let Ok(output) = serde_json::from_str::<BakeoffOutput>(&content) {
                success_count += 1;
                raw_writer.write_all(content.replace('\n', "").as_bytes())?;
                raw_writer.write_all(b"\n")?;
                for r in output.results {
                    let name = path.file_stem().unwrap().to_str().unwrap();
                    let parts: Vec<String> = name.split('_').map(|s| s.to_string()).collect();
                    all_results.push((
                        r,
                        parts[0].trim_start_matches('d').to_string(),
                        parts[1].trim_start_matches('o').to_string(),
                        parts[2].trim_start_matches('s').to_string(),
                        parts[3].trim_start_matches('b').to_string(),
                        parts[4].trim_start_matches('t').to_string(),
                        parts[5].trim_start_matches('k').to_string(),
                        parts[6].trim_start_matches('f').to_string(),
                        parts[7].to_string(),
                    ));
                }
            } else {
                fail_count += 1;
                writeln!(
                    failed_writer,
                    "{{\"run\": \"{:?}\", \"status\": \"failed\"}}",
                    path
                )?;
            }
        }
    }
    raw_writer.flush()?;
    failed_writer.flush()?;

    let mut csv = csv::Writer::from_path(out.join("aggregate.csv"))?;
    csv.write_record(&[
        "dim",
        "objects",
        "seed",
        "branching",
        "split_threshold",
        "redundancy_k",
        "fanout",
        "policy",
        "router",
        "top1",
        "top3",
        "margin",
        "searched_mean",
        "searched_max",
        "missing_leaf_count",
        "log_storage",
        "top1_per_storage",
        "top1_per_fanout",
        "oracle_gap_top1",
    ])?;
    for (r, dim, obj, seed, branching, threshold, k, fanout, policy) in &all_results {
        csv.write_record(&[
            dim,
            obj,
            seed,
            branching,
            threshold,
            k,
            fanout,
            policy,
            &r.router,
            r.top1.to_string(),
            r.top3.to_string(),
            r.mean_margin.to_string(),
            r.searched_nodes_mean.to_string(),
            r.searched_nodes_max.to_string(),
            r.missing_leaf_count.to_string(),
            r.logical_storage_multiplier.to_string(),
            r.top1_per_logical_storage.to_string(),
            r.top1_per_fanout.to_string(),
            r.oracle_gap_top1.to_string(),
        ])?;
    }

    csv.flush()?;

    let mut summary = File::create(out.join("summary.md"))?;
    writeln!(summary, "# RHN Sweep Summary (Status: PARTIAL)\n")?;
    writeln!(summary, "Successful Invocations: {}", success_count)?;
    writeln!(summary, "Failed Invocations: {}", fail_count)?;
    writeln!(summary, "Result Rows: {}", all_results.len())?;
    writeln!(
        summary,
        "Missing Leaf Count Sum: {}",
        all_results
            .iter()
            .map(|(r, _, _, _, _, _, _, _, _)| r.missing_leaf_count)
            .sum::<usize>()
    )?;

    let best_overall = all_results.iter().max_by(|a, b| {
        a.0.top1_per_fanout
            .partial_cmp(&b.0.top1_per_fanout)
            .unwrap()
    });
    let best_non_oracle = all_results
        .iter()
        .filter(|(r, _, _, _, _, _, _, _, _)| !r.router.starts_with("Oracle"))
        .max_by(|a, b| {
            a.0.top1_per_fanout
                .partial_cmp(&b.0.top1_per_fanout)
                .unwrap()
        });

    if let Some((r, _, _, _, _, _, _, _, _)) = best_overall {
        writeln!(
            summary,
            "\n### Best Overall Router (by top1_per_fanout)\n- Router: {}\n- Score: {:.4}",
            r.router, r.top1_per_fanout
        )?;
    }
    if let Some((r, _, _, _, _, _, _, _, _)) = best_non_oracle {
        writeln!(
            summary,
            "\n### Best Non-Oracle Router (by top1_per_fanout)\n- Router: {}\n- Score: {:.4}",
            r.router, r.top1_per_fanout
        )?;
    }

    Ok(())
}

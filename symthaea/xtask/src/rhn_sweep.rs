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
    use serde::Deserialize;
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::process::Command;

    #[derive(Deserialize, serde::Serialize)]
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

    #[derive(Deserialize, serde::Serialize)]
    struct BakeoffOutput {
        architecture: String,
        version: String,
        results: Vec<BakeoffResult>,
    }

    let runs_dir = out.join("runs");
    fs::create_dir_all(&runs_dir)?;
    let raw_runs = File::create(out.join("raw_runs.jsonl"))?;
    let mut raw_writer = BufWriter::new(raw_runs);

    let mut all_results = Vec::new();

    for &dim in &dims {
        for &obj in &objects {
            for &seed in &seeds {
                for &branching in &branchings {
                    for &threshold in &thresholds {
                        for &k in &redundancy_ks {
                            for &fanout in &fanouts {
                                for policy in &policies {
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
                                        let content = fs::read_to_string(&output_path)?;
                                        let output: BakeoffOutput = serde_json::from_str(&content)?;

                                        // Write JSONL entry
                                        raw_writer
                                            .write_all(content.replace('\n', "").as_bytes())?;
                                        raw_writer.write_all(b"\n")?;

                                        for r in output.results {
                                            all_results.push((
                                                dim,
                                                obj,
                                                seed,
                                                branching,
                                                threshold,
                                                k,
                                                fanout,
                                                policy.clone(),
                                                r,
                                            ));
                                        }
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
    for (dim, obj, seed, branching, threshold, k, fanout, policy, r) in &all_results {
        csv.write_record(&[
            dim.to_string(),
            obj.to_string(),
            seed.to_string(),
            branching.to_string(),
            threshold.to_string(),
            k.to_string(),
            fanout.to_string(),
            policy.to_string(),
            r.router.clone(),
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
    writeln!(summary, "Total runs: {}", all_results.len())?;
    writeln!(summary, "\n| Router | Top1/Fanout | Missing Leaves |")?;
    writeln!(summary, "| --- | --- | --- |")?;
    for (dim, obj, seed, branching, threshold, k, fanout, policy, r) in &all_results {
        writeln!(
            summary,
            "| {} | {:.4} | {} |",
            r.router, r.top1_per_fanout, r.missing_leaf_count
        )?;
    }

    Ok(())
}

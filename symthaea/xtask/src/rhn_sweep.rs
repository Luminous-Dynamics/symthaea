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
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::process::Command;

    let runs_dir = out.join("runs");
    fs::create_dir_all(&runs_dir)?;
    let raw_runs = File::create(out.join("raw_runs.jsonl"))?;
    let mut raw_writer = BufWriter::new(raw_runs);

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
                                        let json = fs::read_to_string(&output_path)?;
                                        raw_writer.write_all(json.as_bytes())?;
                                        raw_writer.write_all(b"\n")?;
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
    Ok(())
}

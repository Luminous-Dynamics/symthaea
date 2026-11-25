use orbital_joint_evidence_voi_bench::run_audit;
use std::env;
use std::fs;

fn main() {
    if let Err(error) = run() {
        eprintln!("orbital_joint_evidence_voi_bench_error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let input_path = args.next().ok_or(
        "usage: orbital-joint-evidence-voi-bench <scenarios.json> --source-commit <40-hex-sha>",
    )?;
    let flag = args.next().ok_or("missing --source-commit flag")?;
    if flag != "--source-commit" {
        return Err("expected --source-commit".to_owned());
    }
    let source_commit = args.next().ok_or("missing source commit")?;
    if args.next().is_some() {
        return Err("unexpected trailing arguments".to_owned());
    }
    let bytes =
        fs::read(&input_path).map_err(|error| format!("cannot read {input_path}: {error}"))?;
    let receipt = run_audit(&bytes, &source_commit)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?
    );
    if receipt.oracle_mismatches > 0 {
        return Err(format!(
            "{} of {} frozen scenario checks mismatched",
            receipt.oracle_mismatches, receipt.scenario_count
        ));
    }
    Ok(())
}

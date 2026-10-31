use pod_meta_lab::graph::{
    encode_oracle_artifact_v1,
    encode_planner_artifact_v1,
    generate_poar_graph_suite,
    DatasetSplit,
};
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn fail(message: impl AsRef<str>) -> String {
    message.as_ref().to_string()
}

fn split_from_str(value: &str) -> Result<DatasetSplit, String> {
    match value {
        "dev" | "development" => Ok(DatasetSplit::Development),
        "cal" | "calibration" => Ok(DatasetSplit::Calibration),
        "test" => Ok(DatasetSplit::Test),
        other => Err(format!("unknown split {other:?}; expected dev, cal, or test")),
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|error| format!("failed to execute sha256sum: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "sha256sum failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("sha256sum output was not UTF-8: {error}"))?;
    let digest = stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| "sha256sum returned no digest".to_string())?;
    if digest.len() != 64
        || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        || digest.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err(format!("sha256sum returned a non-canonical digest: {digest:?}"));
    }
    Ok(digest.to_string())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err(format!(
            "usage: {} <seed> <dev|cal|test> <planner.poar> <oracle.poar.sealed> <generator-revision>",
            args.first().map_or("poar-graph-export", String::as_str)
        ));
    }

    let seed = args[1]
        .parse::<u64>()
        .map_err(|error| format!("invalid seed {:?}: {error}", args[1]))?;
    let split = split_from_str(&args[2])?;
    let planner_path = Path::new(&args[3]);
    let oracle_path = Path::new(&args[4]);
    let generator_revision = &args[5];
    if generator_revision.is_empty() {
        return Err("generator revision must not be empty".into());
    }
    if planner_path == oracle_path {
        return Err("planner and oracle output paths must be different".into());
    }

    let suite = generate_poar_graph_suite(seed, split);
    let planner_artifact = encode_planner_artifact_v1(&suite, generator_revision)?;
    fs::write(planner_path, planner_artifact.as_bytes())
        .map_err(|error| format!("failed to write planner artifact: {error}"))?;

    let planner_sha256 = sha256_file(planner_path)?;
    let oracle_artifact =
        encode_oracle_artifact_v1(&suite, generator_revision, &planner_sha256)?;
    fs::write(oracle_path, oracle_artifact.as_bytes())
        .map_err(|error| format!("failed to write sealed oracle artifact: {error}"))?;

    println!("poar_export_schema=POAR-TSV/1");
    println!("poar_seed={seed}");
    println!("poar_split={}", args[2]);
    println!("planner_sha256={planner_sha256}");
    println!("planner_bytes={}", planner_artifact.len());
    println!("oracle_bytes={}", oracle_artifact.len());
    println!("cases={}", suite.case_count());
    println!("oracle_artifact_is_sensitive=true");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("poar-graph-export: {error}");
        std::process::exit(1);
    }
}

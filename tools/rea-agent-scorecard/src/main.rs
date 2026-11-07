use rea_agent_scorecard::{MAX_ORACLE_BYTES, MAX_SUBMISSION_BYTES, evaluate_scorecard};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if size > maximum as u64 {
        return Err(format!(
            "{} is {size} bytes; limit is {maximum}",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(size as usize);
    file.take(maximum as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() > maximum {
        return Err(format!("{} exceeded the {maximum}-byte limit while reading", path.display()));
    }
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 || args[0] == "--help" || args[0] == "-h" {
        return Err(
            "usage: rea-agent-scorecard <hidden-oracle.json> <agent-submission.json>".to_owned(),
        );
    }

    let oracle = read_bounded(Path::new(&args[0]), MAX_ORACLE_BYTES)?;
    let submission = read_bounded(Path::new(&args[1]), MAX_SUBMISSION_BYTES)?;
    let scorecard = evaluate_scorecard(&oracle, &submission)
        .map_err(|error| format!("scorecard evaluation failed ({:?}): {error}", error.kind))?;
    let output = serde_json::to_string_pretty(&scorecard)
        .map_err(|error| format!("serialize scorecard: {error}"))?;
    println!("{output}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

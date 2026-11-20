//! CLI for the deterministic MUSE v1 symbolic edit evaluator.

use std::{env, fs, path::PathBuf};
use symthaea_muse::edit_evidence::evaluate_symbolic_case;

fn path_after(args: &[String], flag: &str, required: bool) -> Result<Option<PathBuf>, String> {
    match args.iter().position(|arg| arg == flag) {
        Some(index) => args.get(index + 1).map(PathBuf::from).map(Some)
            .ok_or_else(|| format!("missing path after {flag}")),
        None if required => Err(format!("missing required argument {flag}")),
        None => Ok(None),
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", concat!(
            "muse_edit_evaluator --case FILE --source-score FILE --candidate-score FILE ",
            "[--candidate-audio FILE] [--output FILE]\n",
            "Scores deterministic v1 symbolic velocity, pitch and duration edits; unsupported families abstain."
        ));
        return Ok(());
    }
    let case_path = path_after(&args, "--case", true)?.unwrap();
    let source_path = path_after(&args, "--source-score", true)?.unwrap();
    let candidate_path = path_after(&args, "--candidate-score", true)?.unwrap();
    let audio_path = path_after(&args, "--candidate-audio", false)?;
    let output_path = path_after(&args, "--output", false)?;

    let case = fs::read(case_path).map_err(|error| format!("read case: {error}"))?;
    let source = fs::read(source_path).map_err(|error| format!("read source Score: {error}"))?;
    let candidate = fs::read(candidate_path).map_err(|error| format!("read candidate Score: {error}"))?;
    let audio = audio_path.map(fs::read).transpose().map_err(|error| format!("read candidate audio: {error}"))?;
    let result = evaluate_symbolic_case(&case, &source, &candidate, audio.as_deref())?;
    let rendered = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("serialize result: {error}"))?;
    if let Some(path) = output_path {
        fs::write(path, rendered).map_err(|error| format!("write result: {error}"))?;
    } else {
        println!("{}", String::from_utf8(rendered).map_err(|error| format!("result is not UTF-8: {error}"))?);
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("muse_edit_evaluator: {error}");
        std::process::exit(2);
    }
}

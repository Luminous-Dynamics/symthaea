use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_judge_grounding_audit::transform_qualification::parse_and_qualify;

fn run() -> Result<bool, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 5 || args[0] != "audit" {
        return Err(
            "usage: mel-audio-transform-qualification audit MANIFEST.json SOURCE_AUDIO VARIANT_AUDIO REPORT.json".to_owned(),
        );
    }
    let input = fs::read_to_string(&args[1])
        .map_err(|error| format!("failed to read {}: {error}", args[1]))?;
    let report = parse_and_qualify(&input, Path::new(&args[2]), Path::new(&args[3]))?;
    let output = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    if let Some(parent) = Path::new(&args[4]).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create report directory: {error}"))?;
    }
    fs::write(&args[4], format!("{output}\n"))
        .map_err(|error| format!("failed to write {}: {error}", args[4]))?;
    println!(
        "Transform {} for task dimension {}: {}",
        report.transform_family,
        report.task_dimension,
        if report.eligible_for_declared_task_dimension {
            "eligible"
        } else {
            "ineligible"
        }
    );
    if !report.eligible_for_declared_task_dimension {
        for reason in &report.ineligibility_reasons {
            eprintln!("ineligible: {reason}");
        }
    }
    Ok(report.eligible_for_declared_task_dimension)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("mel-audio-transform-qualification: {error}");
            ExitCode::FAILURE
        }
    }
}

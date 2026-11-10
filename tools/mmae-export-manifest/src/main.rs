use mmae_export_manifest::{export_submission_manifest, OutputMappingV1};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("MMAE export preflight failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 6
        || args[0] != "--expected-ids"
        || args[2] != "--outputs"
        || args[4] != "--submission-dir"
    {
        return Err(
            "usage: mmae-export-manifest --expected-ids <sdk-track-ids.json> --outputs <outputs.jsonl> --submission-dir <directory>"
                .to_owned(),
        );
    }

    let expected_path = PathBuf::from(&args[1]);
    let output_map_path = PathBuf::from(&args[3]);
    let submission_root = PathBuf::from(&args[5]);

    let expected_bytes = fs::read(&expected_path)
        .map_err(|error| format!("cannot read expected ID artifact: {error}"))?;
    let expected_ids: Vec<String> = serde_json::from_slice(&expected_bytes)
        .map_err(|error| format!("expected ID artifact must be a JSON string array: {error}"))?;
    if expected_ids.len() != 500 || expected_ids.iter().collect::<BTreeSet<_>>().len() != 500 {
        return Err("expected ID artifact must contain exactly 500 unique strings".to_owned());
    }

    let output_text = fs::read_to_string(&output_map_path)
        .map_err(|error| format!("cannot read output mapping JSONL as UTF-8: {error}"))?;
    let mut mappings = Vec::new();
    for (line_index, line) in output_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mapping: OutputMappingV1 = serde_json::from_str(line).map_err(|error| {
            format!("output mapping line {} is invalid: {error}", line_index + 1)
        })?;
        mappings.push(mapping);
    }

    let report = export_submission_manifest(&expected_ids, &mappings, &submission_root)
        .map_err(|error| error.to_string())?;
    println!("Prepared {} official sample mappings.", report.sample_count);
    println!("Manifest bytes: {}", report.manifest_bytes);
    println!("Audio bytes verified: {}", report.total_audio_bytes);
    println!("Manifest SHA-256: {}", report.manifest_sha256);
    println!(
        "submission.jsonl: {}",
        submission_root.join("submission.jsonl").display()
    );
    println!("Next step: run the official Audio Editing Challenge SDK validator/packer.");
    println!("No WAV conversion, tar packaging, credential handling or upload was performed.");
    Ok(())
}

//! Import and score an official MusicListenBench v1.0 rendered-test submission.
//!
//! Usage:
//! mel-listenbench-import <eval.json> <eval_perturb.json> <eval_perturb.meta.jsonl>
//!   <submission.jsonl> <data/audio-root> <data/audio_perturb-root> <report.json>

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::musiclistenbench_import::import_official_submission;

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "usage: mel-listenbench-import <eval.json> <eval_perturb.json> <eval_perturb.meta.jsonl> <submission.jsonl> <audio-root> <audio-perturb-root> <report.json>".into()
        );
    }
    let clean_eval = fs::read(&args[0])
        .map_err(|error| format!("cannot read {}: {error}", args[0]))?;
    let perturb_eval = fs::read(&args[1])
        .map_err(|error| format!("cannot read {}: {error}", args[1]))?;
    let perturb_meta = fs::read(&args[2])
        .map_err(|error| format!("cannot read {}: {error}", args[2]))?;
    let submission = fs::read(&args[3])
        .map_err(|error| format!("cannot read {}: {error}", args[3]))?;

    let report = import_official_submission(
        &clean_eval,
        &perturb_eval,
        &perturb_meta,
        &submission,
        Path::new(&args[4]),
        Path::new(&args[5]),
    )?;
    let output = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize imported report: {error}"))?;
    fs::write(&args[6], output)
        .map_err(|error| format!("cannot write {}: {error}", args[6]))?;

    println!("MusicListenBench v1.0 official rendered-test import");
    println!("model: {} ({})", report.official_submission_meta.model, report.official_submission_meta.version);
    println!("pairs: {}", report.official_scoring_report.pair_count);
    println!("missing answers: {}", report.missing_answer_items);
    println!("clean accuracy: {:.4}", report.official_scoring_report.overall.clean_accuracy.unwrap_or(f64::NAN));
    println!("FLIP/STAY Acc_FS: {:.4}", report.official_scoring_report.overall.accuracy_flip_stay.unwrap_or(f64::NAN));
    println!("FLIP miss rate: {:.4}", report.official_scoring_report.overall.miss_rate.unwrap_or(f64::NAN));
    println!("STAY false-flip rate: {:.4}", report.official_scoring_report.overall.false_flip_rate.unwrap_or(f64::NAN));
    println!("full report: {}", args[6]);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mel-listenbench-import: {error}");
            ExitCode::from(2)
        }
    }
}

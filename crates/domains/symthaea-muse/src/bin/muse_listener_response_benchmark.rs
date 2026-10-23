// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible CLI for the Muse listener-response temporal-baseline benchmark.
//!
//! JSONL input has one ListenerResponseRecord per non-empty line. Use --demo for
//! a deterministic synthetic smoke fixture; its audio identities are synthetic
//! placeholders and must never be mistaken for rendered-audio evidence.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use symthaea_muse::listener_response::{
    LISTENER_RESPONSE_SCHEMA_VERSION, ListeningIntent, ListenerResponseRecord,
    SelfReportedAffect,
};
use symthaea_muse::listener_response_benchmark::{
    DEFAULT_PERSONAL_PRIOR_STRENGTH, ListenerResponseBenchmarkReport, TemporalHoldoutSplit,
    evaluate_temporal_holdout,
};

const RUNNER_VERSION: &str = "muse-listener-response-runner-v2";

#[derive(Debug, Default)]
struct Options {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    training_through_sequence: Option<u64>,
    testing_from_sequence: Option<u64>,
    prior_strength: Option<f64>,
    demo: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkArtifact {
    artifact_version: &'static str,
    runner_version: &'static str,
    /// Compile-time source revision when supplied by the build environment.
    runner_build_revision: &'static str,
    data_kind: &'static str,
    source_sha256: String,
    source_records: usize,
    /// A note remains present in the report so demo evidence cannot be mistaken
    /// for real audio hashes or real listener observations.
    evidence_boundary: &'static str,
    report: ListenerResponseBenchmarkReport,
}

fn usage() -> &'static str {
    "Muse listener-response temporal baseline benchmark\n\
\n\
USAGE:\n\
  cargo run -p symthaea-muse --bin muse_listener_response_benchmark -- \\\n\
    --input RESPONSES.jsonl --training-through N --testing-from N [OPTIONS]\n\
  cargo run -p symthaea-muse --bin muse_listener_response_benchmark -- --demo\n\
\n\
OPTIONS:\n\
  --input PATH             JSONL file, one listener-response record per line\n\
  --output PATH            Write JSON report to PATH instead of stdout\n\
  --training-through N     Last global collection_sequence allowed in training\n\
  --testing-from N         First global collection_sequence allowed in testing\n\
  --prior-strength X       Personal-prior shrinkage pseudo-count (default 3.0)\n\
  --demo                   Run a deterministic synthetic fixture (no real data)\n\
  --help                   Print this help\n\
\n\
Records in the interval (training-through, testing-from) are embargoed. No\n\
session may cross training, embargo, or test partitions. The JSON report binds\n\
the exact source JSONL bytes with SHA-256. --demo uses synthetic placeholder\n\
audio identities, not rendered audio and not human-subject evidence."
}

fn parse_options(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, String> {
    let mut options = Options {
        prior_strength: Some(DEFAULT_PERSONAL_PRIOR_STRENGTH),
        ..Options::default()
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--demo" => options.demo = true,
            "--input" => {
                options.input = Some(PathBuf::from(next_value(&mut args, "--input")?));
            }
            "--output" => {
                options.output = Some(PathBuf::from(next_value(&mut args, "--output")?));
            }
            "--training-through" => {
                options.training_through_sequence = Some(parse_u64(
                    &next_value(&mut args, "--training-through")?,
                    "--training-through",
                )?);
            }
            "--testing-from" => {
                options.testing_from_sequence = Some(parse_u64(
                    &next_value(&mut args, "--testing-from")?,
                    "--testing-from",
                )?);
            }
            "--prior-strength" => {
                options.prior_strength = Some(
                    next_value(&mut args, "--prior-strength")?
                        .parse::<f64>()
                        .map_err(|error| format!("invalid --prior-strength: {error}"))?,
                );
            }
            unknown => return Err(format!("unknown argument: {unknown}\n\n{}", usage())),
        }
    }

    if options.demo {
        if options.input.is_some() {
            return Err("--demo and --input cannot be combined".to_owned());
        }
        if options.training_through_sequence.is_none() {
            options.training_through_sequence = Some(4);
        }
        if options.testing_from_sequence.is_none() {
            options.testing_from_sequence = Some(7);
        }
    } else {
        if options.input.is_none() {
            return Err("--input is required unless --demo is used".to_owned());
        }
        if options.training_through_sequence.is_none() || options.testing_from_sequence.is_none() {
            return Err("--training-through and --testing-from are required".to_owned());
        }
    }

    if !options.prior_strength.unwrap_or(DEFAULT_PERSONAL_PRIOR_STRENGTH).is_finite()
        || options.prior_strength.unwrap_or(DEFAULT_PERSONAL_PRIOR_STRENGTH) < 0.0
    {
        return Err("--prior-strength must be finite and >= 0".to_owned());
    }
    Ok(Some(options))
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|error| format!("invalid {flag}: {error}"))
}

fn parse_jsonl(input: &str) -> Result<Vec<ListenerResponseRecord>, String> {
    let mut records = Vec::new();
    for (index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<ListenerResponseRecord>(line).map_err(|error| {
            format!("invalid JSONL record at line {}: {error}", index + 1)
        })?;
        records.push(record);
    }
    if records.is_empty() {
        return Err("input contains no JSONL records".to_owned());
    }
    Ok(records)
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn source_digest(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn paths_refer_to_same_file(input: &Path, output: &Path) -> bool {
    if input == output {
        return true;
    }
    match (fs::canonicalize(input), fs::canonicalize(output)) {
        (Ok(input), Ok(output)) => input == output,
        _ => false,
    }
}

fn demo_records() -> Vec<ListenerResponseRecord> {
    vec![
        demo_episode("demo-01", "demo-listener-a", 1, 1, 1, -0.2, 0.1),
        demo_episode("demo-02", "demo-listener-b", 1, 1, 2, 0.0, 0.1),
        demo_episode("demo-03", "demo-listener-a", 2, 1, 3, -0.1, 0.2),
        demo_episode("demo-04", "demo-listener-b", 2, 1, 4, 0.0, 0.2),
        demo_episode("demo-05", "demo-listener-a", 3, 1, 5, -0.1, 0.2),
        demo_episode("demo-06", "demo-listener-b", 3, 1, 6, 0.0, 0.1),
        demo_episode("demo-07", "demo-listener-a", 4, 1, 7, 0.0, 0.3),
        demo_episode("demo-08", "demo-listener-b", 4, 1, 8, 0.0, 0.2),
    ]
}

fn demo_episode(
    id: &str,
    participant: &str,
    session_index: u32,
    order_in_session: u32,
    collection_sequence: u64,
    valence_before: f32,
    valence_after: f32,
) -> ListenerResponseRecord {
    let mut hasher = Sha256::new();
    hasher.update(b"synthetic-placeholder-not-audio:");
    hasher.update(collection_sequence.to_be_bytes());
    let audio_sha256 = hex_lower(&hasher.finalize());

    ListenerResponseRecord {
        schema_version: LISTENER_RESPONSE_SCHEMA_VERSION.to_owned(),
        episode_id: id.to_owned(),
        participant_token: participant.to_owned(),
        presentation_token: format!("presentation-{id}"),
        audio_sha256,
        score_sha256: None,
        model_version: Some("synthetic-fixture-v1".to_owned()),
        collection_sequence,
        session_index,
        order_in_session,
        intent: ListeningIntent::Enjoy,
        target_affect: None,
        activity: None,
        familiarity_1_to_5: None,
        felt_before: Some(SelfReportedAffect {
            valence: Some(valence_before),
            arousal: Some(0.4),
            tension: Some(0.3),
            energy: Some(0.4),
        }),
        during: Vec::new(),
        felt_after: Some(SelfReportedAffect {
            valence: Some(valence_after),
            arousal: Some(0.4),
            tension: Some(0.3),
            energy: Some(0.4),
        }),
        perceived_music: None,
        enjoyment_1_to_7: Some(5),
        goal_attainment_1_to_7: None,
        playback_completed: true,
        consent_confirmed: true,
        consent_protocol_version: "synthetic-fixture-only".to_owned(),
    }
}

fn run(options: Options) -> Result<String, String> {
    let prior_strength = options.prior_strength.unwrap_or(DEFAULT_PERSONAL_PRIOR_STRENGTH);
    let (raw_bytes, records, data_kind) = if options.demo {
        let records = demo_records();
        let mut raw = String::new();
        for record in &records {
            raw.push_str(
                &serde_json::to_string(record)
                    .map_err(|error| format!("serialize demo record: {error}"))?,
            );
            raw.push('\n');
        }
        (raw.into_bytes(), records, "synthetic_fixture")
    } else {
        let input_path = options.input.as_ref().ok_or("--input is required")?;
        if options
            .output
            .as_ref()
            .is_some_and(|output| paths_refer_to_same_file(input_path, output))
        {
            return Err(
                "refusing to overwrite the source JSONL with the benchmark report".to_owned(),
            );
        }
        let raw_bytes = fs::read(input_path)
            .map_err(|error| format!("read {}: {error}", input_path.display()))?;
        let raw_text = std::str::from_utf8(&raw_bytes)
            .map_err(|error| format!("input is not UTF-8: {error}"))?;
        let records = parse_jsonl(raw_text)?;
        (raw_bytes, records, "listener_response_jsonl")
    };

    let split = TemporalHoldoutSplit {
        training_through_sequence: options
            .training_through_sequence
            .ok_or("--training-through is required")?,
        testing_from_sequence: options
            .testing_from_sequence
            .ok_or("--testing-from is required")?,
    };
    let report = evaluate_temporal_holdout(&records, split, prior_strength)
        .map_err(|issues| {
            serde_json::to_string_pretty(&issues)
                .unwrap_or_else(|_| format!("benchmark rejected input: {issues:?}"))
        })?;

    let artifact = BenchmarkArtifact {
        artifact_version: "muse-listener-response-evidence-v2",
        runner_version: RUNNER_VERSION,
        runner_build_revision: option_env!("GIT_COMMIT_SHA")
            .or(option_env!("GITHUB_SHA"))
            .unwrap_or("unbound-at-build-time"),
        data_kind,
        source_sha256: source_digest(&raw_bytes),
        source_records: records.len(),
        evidence_boundary: if options.demo {
            concat!(
                "Synthetic demonstration only; placeholder audio identities are not hashes ",
                "of audio bytes and are not human-listener evidence."
            )
        } else {
            concat!(
                "Hash identifies the exact input JSONL bytes only; record audio hashes must ",
                "separately identify the audio bytes actually presented."
            )
        },
        report,
    };
    serde_json::to_string_pretty(&artifact)
        .map_err(|error| format!("serialize benchmark report: {error}"))
}

fn main() {
    match parse_options(env::args().skip(1)) {
        Ok(None) => {
            println!("{}", usage());
        }
        Ok(Some(options)) => {
            let output_path = options.output.clone();
            match run(options) {
                Ok(output) => {
                    if let Some(path) = output_path {
                        if let Err(error) = fs::write(&path, format!("{output}\n")) {
                            eprintln!("write {}: {error}", path.display());
                            process::exit(2);
                        }
                    } else {
                        println!("{output}");
                    }
                }
                Err(error) => {
                    eprintln!("{error}");
                    process::exit(2);
                }
            }
        },
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_runs_through_the_same_evaluator() {
        let options = parse_options(["--demo".to_owned()]).unwrap().unwrap();
        let output = run(options).unwrap();
        let value: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["data_kind"], "synthetic_fixture");
        assert_eq!(value["source_records"], 8);
        assert!(value["runner_build_revision"].as_str().is_some_and(|value| !value.is_empty()));
        assert!(value["source_sha256"].as_str().unwrap().len() == 64);
        assert_eq!(value["report"]["training_records"], 4);
        assert_eq!(value["report"]["embargoed_records"], 2);
        assert_eq!(value["report"]["testing_records"], 2);
        assert_eq!(value["artifact_version"], "muse-listener-response-evidence-v2");
        assert_eq!(
            value["report"]["benchmark_version"],
            "muse-listener-response-benchmark-v2"
        );
        assert_eq!(value["report"]["channels"][0]["scored_test_observations"], 2);
        assert!(value["report"]["channels"][0]["participant_cluster_comparisons"].is_array());
    }

    #[test]
    fn demo_hash_is_deterministic() {
        let options = parse_options(["--demo".to_owned()]).unwrap().unwrap();
        let first = run(options).unwrap();
        let options = parse_options(["--demo".to_owned()]).unwrap().unwrap();
        let second = run(options).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn identical_input_and_output_paths_are_rejected() {
        assert!(paths_refer_to_same_file(
            std::path::Path::new("responses.jsonl"),
            std::path::Path::new("responses.jsonl"),
        ));
        assert!(!paths_refer_to_same_file(
            std::path::Path::new("responses.jsonl"),
            std::path::Path::new("report.json"),
        ));
    }

    #[test]
    fn missing_required_flags_are_rejected() {
        assert!(parse_options(Vec::<String>::new()).is_err());
        assert!(parse_options([
            "--demo".to_owned(),
            "--input".to_owned(),
            "x".to_owned(),
        ])
        .is_err());
    }

    #[test]
    fn jsonl_parser_round_trips_multiple_records_and_ignores_blank_lines() {
        let records = demo_records();
        let first = serde_json::to_string(&records[0]).unwrap();
        let second = serde_json::to_string(&records[1]).unwrap();
        let input = format!("{first}\n\n{second}\n");
        let parsed = parse_jsonl(&input).unwrap();
        assert_eq!(parsed.as_slice(), &records[..2]);
    }

    #[test]
    fn jsonl_parser_reports_the_bad_line_number() {
        let records = demo_records();
        let first = serde_json::to_string(&records[0]).unwrap();
        let input = format!("{first}\nnot-json\n");
        let error = parse_jsonl(&input).unwrap_err();
        assert!(error.contains("line 2"));
    }

    #[test]
    fn output_path_is_parsed_as_an_option() {
        let options = parse_options([
            "--input".to_owned(), "responses.jsonl".to_owned(),
            "--training-through".to_owned(), "10".to_owned(),
            "--testing-from".to_owned(), "15".to_owned(),
            "--output".to_owned(), "report.json".to_owned(),
        ]).unwrap().unwrap();
        assert_eq!(options.output, Some(PathBuf::from("report.json")));
    }
}

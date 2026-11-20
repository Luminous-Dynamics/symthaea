use muse_score_audio_conformance::{
    evaluate_score_audio_conformance_v1, ScoreAudioContractV1,
};
use serde_json::to_vec_pretty;
use std::{env, fs, path::PathBuf, process};

fn usage() -> &'static str {
    "muse-score-audio-conformance verify SCORE.json AUDIO.wav CONTRACT.json\n\
     The v1 observer supports one assigned, non-overlapping Score part in a mono isolated WAV stem.\n\
     It uses the frozen Score tempo and does not align, resample, source-separate, or infer instrument identity."
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help") {
        println!("{}", usage());
        return Ok(0);
    }
    if args.len() != 5 || args[1] != "verify" {
        println!("{}", usage());
        return Ok(2);
    }
    let score_path = PathBuf::from(&args[2]);
    let audio_path = PathBuf::from(&args[3]);
    let contract_path = PathBuf::from(&args[4]);
    let score = fs::read(&score_path)
        .map_err(|error| format!("read {}: {error}", score_path.display()))?;
    let audio = fs::read(&audio_path)
        .map_err(|error| format!("read {}: {error}", audio_path.display()))?;
    let contract_bytes = fs::read(&contract_path)
        .map_err(|error| format!("read {}: {error}", contract_path.display()))?;
    let contract_value: serde_json::Value = serde_json::from_slice(&contract_bytes)
        .map_err(|error| format!("contract JSON invalid: {error}"))?;
    let contract: ScoreAudioContractV1 = serde_json::from_value(contract_value)
        .map_err(|error| format!("contract wire fields invalid: {error}"))?;
    let result = evaluate_score_audio_conformance_v1(&score, &audio, &contract)?;
    let rendered = to_vec_pretty(&result)
        .map_err(|error| format!("serialize conformance result: {error}"))?;
    println!(
        "{}",
        String::from_utf8(rendered).map_err(|error| format!("result is not UTF-8: {error}"))?
    );
    match result.pointer("/outcome/score_audio_status").and_then(serde_json::Value::as_str) {
        Some("pass") => Ok(0),
        Some("fail") => Ok(2),
        Some("abstain") => Ok(3),
        _ => Err("observer result has an unsupported outcome status".into()),
    }
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("muse-score-audio-conformance: {error}");
            process::exit(4);
        }
    }
}

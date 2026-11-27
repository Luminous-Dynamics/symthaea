//! Recompute exact source, decoded PCM, frame-window and context digests.
//!
//! Usage: mel-audio-evidence-window-audit <manifest.json> <source-audio.bin> <decoded-pcm.hex> <request-body.bin>
//! The PCM fixture transport is hexadecimal text; it is decoded into raw PCM
//! bytes before any declared digest or frame offset is evaluated.

use std::env;
use std::fs;
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::audio_evidence_window::{
    AudioEvidenceVerificationReportV1, verify_audio_evidence_manifest_json,
};

fn decode_hex_file(path: &str) -> Result<Vec<u8>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed reading decoded PCM hex {}: {error}", path))?;
    let digits: Vec<u8> = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    if digits.len() % 2 != 0 {
        return Err("decoded PCM hex has an odd number of hexadecimal digits".into());
    }
    let mut output = Vec::with_capacity(digits.len() / 2);
    for (index, pair) in digits.chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])
            .ok_or_else(|| format!("invalid hexadecimal digit at byte {}", index * 2))?;
        let low = hex_nibble(pair[1])
            .ok_or_else(|| format!("invalid hexadecimal digit at byte {}", index * 2 + 1))?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn run() -> Result<String, String> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: mel-audio-evidence-window-audit <manifest.json> <source-audio.bin> <decoded-pcm.hex> <request-body.bin>"
                .into(),
        );
    }
    let manifest_path = args[0].to_string_lossy().into_owned();
    let source_path = args[1].to_string_lossy().into_owned();
    let pcm_hex_path = args[2].to_string_lossy().into_owned();
    let request_body_path = args[3].to_string_lossy().into_owned();

    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("failed reading manifest {}: {error}", manifest_path))?;
    let source_audio = fs::read(&source_path)
        .map_err(|error| format!("failed reading source audio {}: {error}", source_path))?;
    let decoded_pcm = decode_hex_file(&pcm_hex_path)?;
    let request_body = fs::read(&request_body_path)
        .map_err(|error| format!("failed reading request body {request_body_path}: {error}"))?;

    let report: AudioEvidenceVerificationReportV1 =
        verify_audio_evidence_manifest_json(&manifest, &source_audio, &decoded_pcm, &request_body)?;
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed serializing audio evidence report: {error}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mel-audio-evidence-window-audit: {error}");
            ExitCode::from(2)
        }
    }
}

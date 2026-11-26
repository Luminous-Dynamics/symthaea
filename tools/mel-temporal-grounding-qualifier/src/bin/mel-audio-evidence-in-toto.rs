//! Emit an unsigned in-toto Statement v1 payload for an audio-bound revision.
//!
//! Usage: mel-audio-evidence-in-toto <revision-case.json> <audio-manifest.json>
//!        <source-audio.bin> <decoded-pcm.hex> <request-body.bin>
//!
//! The output is an unsigned in-toto payload for the #13254 DSSE signer. It is
//! not a DSSE envelope and it carries no signer or trust-root verification.

use std::env;
use std::fs;
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::audio_evidence_in_toto::{
    build_audio_evidence_in_toto_statement, UnsignedInTotoAudioEvidenceStatementV1,
};
use mel_temporal_grounding_qualifier::audio_evidence_window::AudioEvidenceWindowManifestV1;
use mel_temporal_grounding_qualifier::temporal_revision::RevisionCaseV2;

fn decode_hex_file(path: &str) -> Result<Vec<u8>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed reading decoded PCM hex {path}: {error}"))?;
    let digits: Vec<u8> = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    if digits.len() % 2 != 0 {
        return Err("decoded PCM hex has an odd number of hexadecimal digits".into());
    }
    digits
        .chunks_exact(2)
        .enumerate()
        .map(|(index, pair)| {
            let high = hex_nibble(pair[0])
                .ok_or_else(|| format!("invalid hexadecimal digit at byte {}", index * 2))?;
            let low = hex_nibble(pair[1])
                .ok_or_else(|| format!("invalid hexadecimal digit at byte {}", index * 2 + 1))?;
            Ok((high << 4) | low)
        })
        .collect()
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
    if args.len() != 5 {
        return Err(
            "usage: mel-audio-evidence-in-toto <revision-case.json> <audio-manifest.json> <source-audio.bin> <decoded-pcm.hex> <request-body.bin>"
                .into(),
        );
    }
    let revision_path = args[0].to_string_lossy().into_owned();
    let manifest_path = args[1].to_string_lossy().into_owned();
    let source_path = args[2].to_string_lossy().into_owned();
    let pcm_path = args[3].to_string_lossy().into_owned();
    let request_body_path = args[4].to_string_lossy().into_owned();

    let revision_text = fs::read_to_string(&revision_path)
        .map_err(|error| format!("failed reading revision case {revision_path}: {error}"))?;
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("failed reading evidence manifest {manifest_path}: {error}"))?;
    let source_audio = fs::read(&source_path)
        .map_err(|error| format!("failed reading source audio {source_path}: {error}"))?;
    let decoded_pcm = decode_hex_file(&pcm_path)?;
    let request_body = fs::read(&request_body_path)
        .map_err(|error| format!("failed reading request body {request_body_path}: {error}"))?;

    let revision_case: RevisionCaseV2 = serde_json::from_str(&revision_text)
        .map_err(|error| format!("invalid temporal revision case JSON: {error}"))?;
    let evidence_manifest: AudioEvidenceWindowManifestV1 = serde_json::from_str(&manifest_text)
        .map_err(|error| format!("invalid audio evidence manifest JSON: {error}"))?;

    let payload: UnsignedInTotoAudioEvidenceStatementV1 =
        build_audio_evidence_in_toto_statement(
            &revision_case,
            &evidence_manifest,
            &source_audio,
            &decoded_pcm,
            &request_body,
        )?;
    serde_json::to_string_pretty(&payload)
        .map_err(|error| format!("failed serializing unsigned in-toto payload: {error}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mel-audio-evidence-in-toto: {error}");
            ExitCode::from(2)
        }
    }
}

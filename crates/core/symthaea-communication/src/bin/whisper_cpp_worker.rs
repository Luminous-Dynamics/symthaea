// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local, offline whisper.cpp worker for pinned single-language ASR experiments.
//!
//! This process implements the versioned JSON-lines provider protocol. It uses
//! only a configured local whisper-cli binary and model file; it never downloads
//! weights. Language identification is deliberately unavailable for a monolingual
//! recognizer and returns no candidates rather than claiming to detect Afrikaans.

use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde_json::Value;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use symthaea_communication::human::{
    HUMAN_PROVIDER_PROTOCOL_VERSION, HumanProviderEnvelope, HumanProviderRequest,
    HumanProviderResult, LanguageCandidate, PreservedText,
};
use symthaea_communication::{Modality, SignalObservation};

const MAX_REQUEST_BYTES: usize = 96 * 1024 * 1024;
const MAX_TRANSCRIPT_BYTES: u64 = 1024 * 1024;
static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Parser)]
#[command(
    name = "symthaea-whisper-cpp-worker",
    about = "Offline JSON-lines ASR worker for a preconverted local whisper.cpp model"
)]
struct Args {
    /// Exact whisper.cpp executable used for inference.
    #[arg(long)]
    whisper_cli: PathBuf,
    /// Preconverted GGML/GGUF-compatible whisper.cpp model file.
    #[arg(long)]
    model: PathBuf,
    /// Configured single-language task (e.g. af); this is not language identification.
    #[arg(long, default_value = "af")]
    language: String,
    /// Provider ID must match --provider-id passed to symthaea-comm-eval.
    #[arg(long, default_value = "buzzasr-afrikaans")]
    provider_id: String,
    /// Existing private scratch directory; temporary WAV/transcript files are removed after each request.
    #[arg(long)]
    scratch_dir: PathBuf,
    /// Reject input audio longer than this many seconds.
    #[arg(long, default_value_t = 60)]
    max_audio_seconds: u64,
    /// Kill whisper-cli if one clip exceeds this runtime, in seconds.
    #[arg(long, default_value_t = 180)]
    timeout_seconds: u64,
}

struct TemporaryRun {
    path: PathBuf,
}

impl Drop for TemporaryRun {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn require_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    if path.is_symlink() || !path.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot resolve {label}: {error}"))
}

fn make_temporary_run(root: &Path) -> Result<TemporaryRun, String> {
    if root.is_symlink() || !root.is_dir() {
        return Err("scratch_dir must be an existing real directory, not a symlink".into());
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve scratch_dir: {error}"))?;
    for _ in 0..16 {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let sequence = RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!("asr-{}-{timestamp}-{sequence}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => {
                let resolved = path
                    .canonicalize()
                    .map_err(|error| format!("cannot resolve temporary run: {error}"))?;
                if !resolved.starts_with(&root) || resolved == root {
                    let _ = fs::remove_dir_all(&resolved);
                    return Err("temporary run directory escaped scratch_dir".into());
                }
                return Ok(TemporaryRun { path: resolved });
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot create private temporary run: {error}")),
        }
    }
    Err("cannot allocate a unique temporary run directory".into())
}

fn validated_audio<'a>(
    observation: &'a SignalObservation,
    max_audio_seconds: u64,
) -> Result<(u32, &'a [f32]), String> {
    observation
        .validate_identity()
        .map_err(|error| format!("observation identity failed: {error}"))?;
    let (sample_rate_hz, channels) = match observation.modality {
        Modality::Audio {
            sample_rate_hz,
            channels,
        } => (sample_rate_hz, channels),
        _ => return Err("whisper.cpp worker only accepts audio observations".into()),
    };
    if channels != 1 {
        return Err("whisper.cpp worker requires mono audio".into());
    }
    if !(8_000..=192_000).contains(&sample_rate_hz) {
        return Err("sample rate must be between 8 kHz and 192 kHz".into());
    }
    if observation.samples.is_empty() {
        return Err("audio sample array is empty".into());
    }
    let duration = observation.samples.len() as f64 / sample_rate_hz as f64;
    if !duration.is_finite() || duration <= 0.0 || duration > max_audio_seconds as f64 {
        return Err(format!("audio duration must be in (0, {max_audio_seconds}] seconds"));
    }
    if observation.samples.iter().any(|sample| !sample.is_finite() || sample.abs() > 1.0) {
        return Err("audio samples must be finite and normalized to [-1, 1]".into());
    }
    Ok((sample_rate_hz, &observation.samples))
}

fn write_pcm16_wav(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).map_err(|error| format!("cannot create temporary WAV: {error}"))?;
    for sample in samples {
        let quantized = (*sample * i16::MAX as f32).round() as i16;
        writer
            .write_sample(quantized)
            .map_err(|error| format!("cannot write temporary WAV: {error}"))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("cannot finalize temporary WAV: {error}"))
}

fn preserved_transcript(text: String, language: &str) -> PreservedText {
    PreservedText {
        original: text.clone(),
        // An explicitly empty normalized string represents a legitimate empty
        // ASR hypothesis. It must score as deletions, not disappear as a worker error.
        normalized: text.is_empty().then(String::new),
        primary_language: Some(language.to_owned()),
        script: Some("Latn".into()),
        spans: Vec::new(),
    }
}

fn transcribe(args: &Args, model: &Path, whisper_cli: &Path, observation: &SignalObservation) -> Result<PreservedText, String> {
    let (sample_rate, samples) = validated_audio(observation, args.max_audio_seconds)?;
    let temporary = make_temporary_run(&args.scratch_dir)?;
    let wav_path = temporary.path.join("input.wav");
    let output_base = temporary.path.join("transcript");
    let output_text = temporary.path.join("transcript.txt");
    write_pcm16_wav(&wav_path, sample_rate, samples)?;

    let mut child = Command::new(whisper_cli)
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(&wav_path)
        .arg("-l")
        .arg(&args.language)
        .arg("-np")
        .arg("-otxt")
        .arg("-of")
        .arg(&output_base)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("failed to launch local whisper-cli: {error}"))?;

    let deadline = Instant::now() + Duration::from_secs(args.timeout_seconds);
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("cannot wait for whisper-cli: {error}"))?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("whisper-cli exceeded {} seconds", args.timeout_seconds));
        }
        thread::sleep(Duration::from_millis(25));
    };
    if !status.success() {
        return Err(format!("whisper-cli failed with exit status {status}"));
    }
    if output_text.is_symlink() || !output_text.is_file() {
        return Err("whisper-cli did not create a regular transcript file".into());
    }
    let metadata = output_text
        .metadata()
        .map_err(|error| format!("cannot inspect whisper-cli output: {error}"))?;
    if metadata.len() > MAX_TRANSCRIPT_BYTES {
        return Err("whisper-cli transcript exceeds the 1 MiB safety limit".into());
    }
    let transcript = fs::read_to_string(&output_text)
        .map_err(|error| format!("cannot read UTF-8 whisper-cli transcript: {error}"))?
        .trim()
        .to_owned();
    Ok(preserved_transcript(transcript, &args.language))
}

fn process_request(args: &Args, model: &Path, whisper_cli: &Path, request: HumanProviderRequest) -> (String, Result<HumanProviderResult, String>) {
    match request {
        HumanProviderRequest::Capabilities { request_id } => (
            request_id,
            Err("capability enumeration is not implemented by this ASR-only worker".into()),
        ),
        HumanProviderRequest::IdentifyLanguage { request_id, .. } => (
            request_id,
            // This worker is configured for one language. Do not manufacture a
            // language-ID result from that configuration.
            Ok(HumanProviderResult::Languages(Vec::<LanguageCandidate>::new())),
        ),
        HumanProviderRequest::Transcribe {
            request_id,
            observation,
        } => (request_id, transcribe(args, model, whisper_cli, &observation).map(HumanProviderResult::Text)),
        HumanProviderRequest::Translate { request_id, .. } => (
            request_id,
            Err("translation is not implemented by this ASR-only worker".into()),
        ),
        HumanProviderRequest::TranslateSpeech { request_id, .. } => (
            request_id,
            Err("speech translation is not implemented by this ASR-only worker".into()),
        ),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.provider_id.trim().is_empty()
        || args.language.trim().is_empty()
        || args.max_audio_seconds == 0
        || args.timeout_seconds == 0
    {
        return Err("provider id, language, and positive audio/runtime limits are required".into());
    }
    let whisper_cli = require_file(&args.whisper_cli, "whisper_cli")?;
    let model = require_file(&args.model, "model")?;
    if args.scratch_dir.is_symlink() || !args.scratch_dir.is_dir() {
        return Err("scratch_dir must be an existing real directory, not a symlink".into());
    }

    let stdin = io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut output = stdout.lock();
    loop {
        let mut bytes = Vec::new();
        let read = input.by_ref().take((MAX_REQUEST_BYTES + 1) as u64).read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err("request exceeds the 96 MiB input safety limit".into());
        }
        let line = std::str::from_utf8(&bytes)?;
        if line.trim().is_empty() {
            continue;
        }
        let request: HumanProviderRequest = serde_json::from_str(line)?;
        let (request_id, result) = process_request(&args, &model, &whisper_cli, request);
        let (result, error) = match result {
            Ok(value) => (Some(value), None),
            Err(message) => (None, Some(message)),
        };
        let envelope = HumanProviderEnvelope {
            protocol_version: HUMAN_PROVIDER_PROTOCOL_VERSION,
            provider_id: args.provider_id.clone(),
            request_id,
            result,
            error,
        };
        serde_json::to_writer(&mut output, &envelope)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_transcript_is_an_explicit_empty_hypothesis() {
        let text = preserved_transcript(String::new(), "af");
        assert_eq!(text.original, "");
        assert_eq!(text.normalized.as_deref(), Some(""));
        assert_eq!(text.primary_language.as_deref(), Some("af"));
        assert!(text.spans.is_empty());
    }

    #[test]
    fn audio_validation_rejects_out_of_range_samples() {
        let mut observation = SignalObservation {
            id: String::new(),
            modality: Modality::Audio {
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![0.0, 1.2],
            features: Default::default(),
            original_text: None,
            normalized_text: None,
            uncertain_spans: vec![],
            timing: symthaea_communication::TimeSpan { start_s: 0.0, end_s: 1.0 / 16_000.0 },
            location: None,
            calibration: Default::default(),
            source_identity: None,
            environment: Default::default(),
        };
        observation.refresh_id().unwrap();
        assert!(validated_audio(&observation, 60).is_err());
    }
}

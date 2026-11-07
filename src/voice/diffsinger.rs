// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed JSONL bridge to a locally provisioned singing-native renderer.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::singing_engine::{
    CapabilitySupport, ControlCapability, SingingBackendCapabilities, SingingControl,
    SingingVoiceEngine, VocalPerformance, VocalStem,
};

const PROTOCOL_VERSION: u32 = 1;
const MAX_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Serialize)]
struct RenderRequest<'a> {
    protocol_version: u32,
    request_id: String,
    operation: &'static str,
    performance: &'a VocalPerformance,
}

#[derive(Deserialize)]
struct RenderResult {
    samples: Vec<f32>,
    sample_rate: u32,
}

#[derive(Deserialize)]
struct WorkerResponse {
    protocol_version: u32,
    provider_id: String,
    request_id: String,
    result: Option<RenderResult>,
    error: Option<String>,
}

pub struct DiffSingerEngine {
    worker: PathBuf,
    provider_id: String,
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    request_counter: u64,
}

impl DiffSingerEngine {
    /// Start an explicitly allowlisted local worker. Provisioning and model
    /// download are deliberately separate from runtime.
    pub fn spawn(worker: &Path) -> Result<Self> {
        let worker = worker
            .canonicalize()
            .with_context(|| format!("DiffSinger worker does not exist: {}", worker.display()))?;
        let mut child = Command::new(&worker)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to start {}", worker.display()))?;
        let input = child
            .stdin
            .take()
            .context("DiffSinger worker has no stdin")?;
        let output = BufReader::new(
            child
                .stdout
                .take()
                .context("DiffSinger worker has no stdout")?,
        );
        Ok(Self {
            worker,
            provider_id: "diffsinger-local".to_string(),
            child,
            input,
            output,
            request_counter: 0,
        })
    }

    pub fn worker_path(&self) -> &Path {
        &self.worker
    }
}

fn diffsinger_capability_manifest() -> SingingBackendCapabilities {
    use CapabilitySupport::{Approximated, Conditional, Native, Unsupported};
    let control = |control, support, rationale: &str| ControlCapability {
        control,
        support,
        rationale: rationale.to_string(),
    };
    SingingBackendCapabilities::new(
        "diffsinger-local",
        vec![
            control(SingingControl::LyricText, Unsupported, "the worker receives phonemes and score events; raw lyric text is not a model input"),
            control(SingingControl::Language, Conditional, "the current worker accepts English only and requires an explicit voicebank language ID"),
            control(SingingControl::ExactPhoneSequence, Native, "explicit supported IPA phones map to required voicebank ARPAbet tokens; unknown phones fail"),
            control(SingingControl::SyllableAlignment, Approximated, "phone duration is apportioned within v2 syllables from natural-duration priors"),
            control(SingingControl::NotePitch, Native, "per-note frequency is expanded into the DiffSinger F0 frame sequence"),
            control(SingingControl::NoteDuration, Native, "note durations and phone natural-duration priors define the phoneme frame durations"),
            control(SingingControl::AbsoluteOnsetAndRests, Native, "absolute note and melisma gaps are emitted as SP durations; overlapping score events fail closed"),
            control(SingingControl::ConsonantAnticipation, Approximated, "onset consonants use an explicit advance value and borrow duration from the preceding vowel"),
            control(SingingControl::Melisma, Native, "the explicitly marked vowel is extended across melisma notes without re-articulating consonants"),
            control(SingingControl::ContinuousPitchCurve, Unsupported, "only discrete note F0 is expanded; arbitrary PitchPoint interpolation is not applied"),
            control(SingingControl::PhraseExpressionCurve, Unsupported, "phrase expression curves are not passed to the worker model"),
            control(SingingControl::PhraseBreathTiming, Unsupported, "phrase breath durations are not converted into breath events; only explicit score rests become SP tokens"),
            control(SingingControl::Breathiness, Unsupported, "the current worker asks the variance model to predict breathiness rather than applying VocalSyllable breathiness"),
            control(SingingControl::Energy, Unsupported, "VocalSyllable energy is not currently mapped to the model's velocity input"),
            control(SingingControl::Tension, Unsupported, "VocalSyllable/phrase tension is not applied to the variance model input"),
            control(SingingControl::Vibrato, Unsupported, "the current F0 sequence is constant within each note and does not encode vibrato"),
            control(SingingControl::Portamento, Unsupported, "continuous pitch transitions between note targets are not generated"),
            control(SingingControl::Articulation, Approximated, "onset consonants can be advanced by borrowing from the preceding vowel; coda consonants follow the final melisma vowel, but the articulation enum itself is not consumed"),
            control(SingingControl::VoiceIdentity, Conditional, "the current worker uses the fixed Standard voicebank embeddings; identity depends on the provisioned voicebank"),
            control(SingingControl::LocalEdits, Unsupported, "the worker renders the full performance and exposes no localized cache/edit operation"),
        ],
    )
    .expect("static DiffSinger capability manifest is complete")
}

impl SingingVoiceEngine for DiffSingerEngine {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn capabilities(&self) -> SingingBackendCapabilities {
        diffsinger_capability_manifest()
    }

    fn render(&mut self, performance: &VocalPerformance) -> Result<VocalStem> {
        performance.validate()?;
        if self.child.try_wait()?.is_some() {
            bail!("DiffSinger worker is not running");
        }
        self.request_counter += 1;
        let request_id = format!("diffsinger-{}", self.request_counter);
        let request = RenderRequest {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id.clone(),
            operation: "render",
            performance,
        };
        serde_json::to_writer(&mut self.input, &request)?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;

        let mut bytes = Vec::new();
        (&mut self.output)
            .take(MAX_RESPONSE_BYTES + 1)
            .read_until(b'\n', &mut bytes)?;
        if bytes.len() as u64 > MAX_RESPONSE_BYTES {
            bail!("DiffSinger response exceeds 64 MiB");
        }
        if bytes.is_empty() {
            bail!("DiffSinger worker disconnected");
        }
        let response: WorkerResponse =
            serde_json::from_slice(&bytes).context("invalid DiffSinger worker response")?;
        if response.protocol_version != PROTOCOL_VERSION
            || response.provider_id != self.provider_id
            || response.request_id != request_id
        {
            bail!("DiffSinger worker identity or protocol mismatch");
        }
        if let Some(error) = response.error {
            bail!("DiffSinger worker: {error}");
        }
        let result = response
            .result
            .context("DiffSinger worker returned neither result nor error")?;
        let stem = VocalStem {
            samples: result.samples,
            sample_rate: result.sample_rate,
            backend: self.provider_id.clone(),
        };
        stem.validate()?;
        Ok(stem)
    }
}

impl Drop for DiffSingerEngine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod capability_manifest_tests {
    use super::*;

    #[test]
    fn static_backend_capability_manifest_is_complete_and_valid() {
        let manifest = diffsinger_capability_manifest();
        manifest.validate().unwrap();
        assert_eq!(manifest.controls.len(), SingingControl::ALL.len());
        assert!(
            manifest
                .controls
                .iter()
                .any(|entry| entry.control == SingingControl::ConsonantAnticipation)
        );
        super::super::singing_engine::SingingRenderPolicy::score_conditioned_comparison()
            .validate_against(&manifest)
            .unwrap();
    }
}

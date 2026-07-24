// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed JSONL bridge to a locally provisioned singing-native renderer.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::singing_engine::{SingingVoiceEngine, VocalPerformance, VocalStem};

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

impl SingingVoiceEngine for DiffSingerEngine {
    fn id(&self) -> &str {
        &self.provider_id
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

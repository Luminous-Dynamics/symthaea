// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Background publisher that uploads `CompositionExport`s to the Mycelix-Music
//! Holochain DNA via its HTTP API.
//!
//! The publisher runs in a dedicated background thread. The cognitive loop
//! periodically polls `MuseManager::take_next_export()` and calls
//! `MusicPublisher::submit()` to hand off the WAV bytes without blocking.
//!
//! # Wire protocol
//!
//! ```text
//! POST http://<host>/api/upload
//! Content-Type: multipart/form-data
//!   file  — audio/wav bytes
//!   title — UTF-8 string
//!   duration_secs — f32 as string
//! ```
//!
//! The server returns JSON `{ "cid": "Qm…", "entry_hash": "…" }` on success.

use std::sync::mpsc;
use std::thread;

use crate::cognitive_loop::managers::muse_manager::CompositionExport;

// ─── Default endpoint ─────────────────────────────────────────────────────────

/// Default Mycelix-Music upload endpoint (local dev instance).
const DEFAULT_UPLOAD_URL: &str = "http://localhost:8121/api/upload";

/// Upload timeout in seconds.
const UPLOAD_TIMEOUT_SECS: u64 = 30;

// ─── Types ────────────────────────────────────────────────────────────────────

/// Outcome of a single upload attempt.
#[derive(Debug, Clone)]
pub struct UploadResult {
    pub title: String,
    pub success: bool,
    /// IPFS CID returned by the server, if any.
    pub cid: Option<String>,
    /// Error description on failure.
    pub error: Option<String>,
}

// ─── Publisher ────────────────────────────────────────────────────────────────

/// Background publisher for Mycelix-Music composition uploads.
///
/// Create once, call `submit()` from the cognitive loop thread.
/// Results accumulate in `take_results()`.
pub struct MusicPublisher {
    sender: mpsc::Sender<CompositionExport>,
    result_receiver: mpsc::Receiver<UploadResult>,
    upload_url: String,
    submitted: u64,
    succeeded: u64,
    failed: u64,
}

impl MusicPublisher {
    /// Create a publisher targeting the default local Mycelix-Music instance.
    pub fn new() -> Self {
        Self::with_url(DEFAULT_UPLOAD_URL)
    }

    /// Create a publisher targeting a custom upload endpoint.
    pub fn with_url(url: &str) -> Self {
        let (tx, rx) = mpsc::channel::<CompositionExport>();
        let (result_tx, result_rx) = mpsc::channel::<UploadResult>();
        let upload_url = url.to_string();

        let thread_url = upload_url.clone();
        thread::Builder::new()
            .name("music-publisher".to_string())
            .spawn(move || {
                run_publisher_thread(rx, result_tx, &thread_url);
            })
            .expect("music-publisher thread spawn");

        Self {
            sender: tx,
            result_receiver: result_rx,
            upload_url,
            submitted: 0,
            succeeded: 0,
            failed: 0,
        }
    }

    /// Submit a composition for background upload.
    ///
    /// Non-blocking: the export is queued and uploaded by the background thread.
    /// Returns `false` if the background thread has terminated unexpectedly.
    pub fn submit(&mut self, export: CompositionExport) -> bool {
        if self.sender.send(export).is_ok() {
            self.submitted += 1;
            true
        } else {
            false
        }
    }

    /// Drain any completed upload results (call from cognitive loop, non-blocking).
    ///
    /// Returns the results since the last call.
    pub fn drain_results(&mut self) -> Vec<UploadResult> {
        let mut results = Vec::new();
        while let Ok(r) = self.result_receiver.try_recv() {
            if r.success {
                self.succeeded += 1;
            } else {
                self.failed += 1;
            }
            results.push(r);
        }
        results
    }

    /// Snapshot telemetry (submitted / succeeded / failed counts).
    pub fn telemetry(&self) -> (u64, u64, u64) {
        (self.submitted, self.succeeded, self.failed)
    }

    /// The upload URL this publisher was configured with.
    pub fn upload_url(&self) -> &str {
        &self.upload_url
    }
}

impl Default for MusicPublisher {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Background thread ────────────────────────────────────────────────────────

fn run_publisher_thread(
    rx: mpsc::Receiver<CompositionExport>,
    result_tx: mpsc::Sender<UploadResult>,
    upload_url: &str,
) {
    // Build a blocking reqwest client (we're already on our own thread).
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(UPLOAD_TIMEOUT_SECS))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            log::error!("[music-publisher] failed to build HTTP client: {e}");
            return;
        }
    };

    while let Ok(export) = rx.recv() {
        let title = export.title.clone();
        let result = upload_composition(&client, upload_url, export);
        // Best-effort: if the cognitive loop dropped the receiver, stop.
        if result_tx.send(result).is_err() {
            break;
        }
    }
}

/// Perform a single multipart upload. Returns an `UploadResult` either way.
fn upload_composition(
    client: &reqwest::blocking::Client,
    url: &str,
    export: CompositionExport,
) -> UploadResult {
    let title = export.title.clone();
    let duration_str = export.duration_secs.to_string();
    let wav_bytes = export.wav_bytes;

    let file_part = match reqwest::blocking::multipart::Part::bytes(wav_bytes)
        .file_name("composition.wav")
        .mime_str("audio/wav")
    {
        Ok(p) => p,
        Err(e) => {
            return UploadResult {
                title,
                success: false,
                cid: None,
                error: Some(format!("MIME error: {e}")),
            };
        }
    };

    let form = reqwest::blocking::multipart::Form::new()
        .part("file", file_part)
        .text("title", export.title.clone())
        .text("duration_secs", duration_str);

    match client.post(url).multipart(form).send() {
        Ok(resp) if resp.status().is_success() => {
            // Try to extract CID from JSON body; tolerate non-JSON responses.
            let cid = resp
                .json::<serde_json::Value>()
                .ok()
                .and_then(|v| v["cid"].as_str().map(String::from));
            UploadResult {
                title,
                success: true,
                cid,
                error: None,
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            UploadResult {
                title,
                success: false,
                cid: None,
                error: Some(format!("HTTP {status}: {body}")),
            }
        }
        Err(e) => UploadResult {
            title,
            success: false,
            cid: None,
            error: Some(e.to_string()),
        },
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse::MusicalState;

    fn dummy_export(title: &str) -> CompositionExport {
        CompositionExport {
            wav_bytes: b"RIFF....WAVEdata".to_vec(),
            title: title.to_string(),
            duration_secs: 8.0,
            musical_state: MusicalState::default(),
            timestamp_secs: 42,
        }
    }

    #[test]
    fn publisher_creates_without_panic() {
        // Just verify construction and drop don't panic.
        // Actual network calls are not exercised in unit tests.
        let _pub = MusicPublisher::with_url("http://127.0.0.1:19999/unreachable");
    }

    #[test]
    fn submit_increments_count() {
        let mut pub_ = MusicPublisher::with_url("http://127.0.0.1:19999/unreachable");
        assert!(pub_.submit(dummy_export("test")));
        let (submitted, _, _) = pub_.telemetry();
        assert_eq!(submitted, 1);
    }

    #[test]
    fn drain_results_empty_initially() {
        let mut pub_ = MusicPublisher::with_url("http://127.0.0.1:19999/unreachable");
        let results = pub_.drain_results();
        assert!(results.is_empty());
    }
}

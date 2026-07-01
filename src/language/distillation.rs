// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Distillation data collector: captures (ThoughtChannels, text) pairs from LLM output.
//!
//! When `SYMTHAEA_DISTILL_PATH` is set, each successful LLM translation is recorded
//! as a JSONL line containing the ThoughtChannels encoding and the generated text.
//! This data is used to train the native Broca SSM backend.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use symthaea_broca::training::TrainingPair;

/// Collects (channels, text) pairs for Broca training.
///
/// Thread-safe: uses a `Mutex<BufWriter<File>>` for serialized writes
/// and an `AtomicUsize` for the record counter.
pub struct DistillationCollector {
    file: Mutex<BufWriter<File>>,
    count: AtomicUsize,
}

impl DistillationCollector {
    /// Create a new collector writing to the given JSONL path.
    ///
    /// Opens the file in append mode so multiple sessions can accumulate data.
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Mutex::new(BufWriter::new(file)),
            count: AtomicUsize::new(0),
        })
    }

    /// Create from the `SYMTHAEA_DISTILL_PATH` environment variable.
    ///
    /// Returns `None` if the env var is not set.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("SYMTHAEA_DISTILL_PATH").ok()?;
        if path.is_empty() {
            return None;
        }
        match Self::new(&path) {
            Ok(c) => {
                tracing::info!(
                    path = %path,
                    "Distillation collector active — recording training pairs"
                );
                Some(c)
            }
            Err(e) => {
                tracing::warn!(
                    path = %path,
                    error = %e,
                    "Failed to open distillation file"
                );
                None
            }
        }
    }

    /// Record a (channels, text) training pair as a JSONL line.
    pub fn record(&self, channels: &symthaea_broca::encoder::ThoughtChannels, text: &str) {
        if text.is_empty() {
            return;
        }

        let pair = TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: text.to_string(),
            target_ids: Vec::new(), // Will be tokenized during training
            valence: 0.0,
            arousal: 0.0,
        };

        match serde_json::to_string(&pair) {
            Ok(json) => {
                if let Ok(mut writer) = self.file.lock() {
                    if let Err(e) = writeln!(writer, "{}", json) {
                        tracing::warn!(error = %e, "Failed to write distillation record");
                    } else if let Err(e) = writer.flush() {
                        tracing::warn!(error = %e, "Failed to flush distillation record");
                    } else {
                        self.count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to serialize distillation record");
            }
        }
    }

    /// Number of pairs recorded so far.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_broca::encoder::ThoughtChannels;

    #[test]
    fn test_collector_records_to_file() {
        let dir = std::env::temp_dir().join("broca_distill_test");
        let path = dir.to_str().unwrap().to_string() + ".jsonl";

        // Clean up from previous runs
        let _ = std::fs::remove_file(&path);

        let collector = DistillationCollector::new(&path).unwrap();
        let channels = ThoughtChannels::default();
        collector.record(&channels, "hello world");
        collector.record(&channels, "test response");
        assert_eq!(collector.count(), 2);

        // Verify JSONL content
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let pair: TrainingPair = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(pair.target_text, "hello world");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_collector_skips_empty_text() {
        let dir = std::env::temp_dir().join("broca_distill_empty");
        let path = dir.to_str().unwrap().to_string() + ".jsonl";

        let _ = std::fs::remove_file(&path);

        let collector = DistillationCollector::new(&path).unwrap();
        let channels = ThoughtChannels::default();
        collector.record(&channels, "");
        assert_eq!(collector.count(), 0);

        let _ = std::fs::remove_file(&path);
    }
}

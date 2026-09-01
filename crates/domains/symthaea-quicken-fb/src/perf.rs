// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Low-overhead boot renderer performance measurement.
//!
//! Instrumentation is opt-in in the live renderer. The headless benchmark uses
//! the same summary implementation so CI and hardware receipts have comparable
//! percentile semantics.

#![forbid(unsafe_code)]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const PERFORMANCE_RECEIPT_VERSION: u16 = 1;
pub const MAX_LIVE_SAMPLES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimingSummary {
    pub count: u64,
    pub min_us: u64,
    pub mean_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
}

impl TimingSummary {
    pub const fn empty() -> Self {
        Self {
            count: 0,
            min_us: 0,
            mean_us: 0,
            p50_us: 0,
            p95_us: 0,
            p99_us: 0,
            max_us: 0,
        }
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "count": self.count,
            "min_us": self.min_us,
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "max_us": self.max_us,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct TimingSeries {
    samples_us: Vec<u64>,
}

impl TimingSeries {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples_us: Vec::with_capacity(capacity),
        }
    }

    pub fn record(&mut self, duration: Duration) {
        self.samples_us.push(saturating_micros(duration));
    }

    pub fn record_us(&mut self, micros: u64) {
        self.samples_us.push(micros);
    }

    pub fn len(&self) -> usize {
        self.samples_us.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples_us.is_empty()
    }

    pub fn summary(&self) -> TimingSummary {
        if self.samples_us.is_empty() {
            return TimingSummary::empty();
        }

        let mut sorted = self.samples_us.clone();
        sorted.sort_unstable();
        let sum: u128 = sorted.iter().map(|&value| value as u128).sum();
        let mean = (sum / sorted.len() as u128).min(u64::MAX as u128) as u64;

        TimingSummary {
            count: sorted.len() as u64,
            min_us: sorted[0],
            mean_us: mean,
            p50_us: percentile(&sorted, 50),
            p95_us: percentile(&sorted, 95),
            p99_us: percentile(&sorted, 99),
            max_us: *sorted.last().unwrap_or(&0),
        }
    }
}

/// Opt-in recorder for the real DRM renderer. No file I/O occurs per-frame.
pub struct BootPerformanceRecorder {
    process_start: Instant,
    frame_budget_us: u64,
    pub drm_open_us: Option<u64>,
    pub first_frame_us: Option<u64>,
    pub frames: u64,
    pub deadline_misses: u64,
    pub samples_dropped: u64,
    pub grow: TimingSeries,
    pub render: TimingSeries,
    pub blit: TimingSeries,
    pub frame_work: TimingSeries,
}

impl BootPerformanceRecorder {
    pub fn new(process_start: Instant, frame_budget: Duration) -> Self {
        let capacity = MAX_LIVE_SAMPLES.min(2048);
        Self {
            process_start,
            frame_budget_us: saturating_micros(frame_budget),
            drm_open_us: None,
            first_frame_us: None,
            frames: 0,
            deadline_misses: 0,
            samples_dropped: 0,
            grow: TimingSeries::with_capacity(capacity),
            render: TimingSeries::with_capacity(capacity),
            blit: TimingSeries::with_capacity(capacity),
            frame_work: TimingSeries::with_capacity(capacity),
        }
    }

    pub fn mark_drm_open(&mut self, duration: Duration) {
        self.drm_open_us = Some(saturating_micros(duration));
    }

    pub fn record_frame(
        &mut self,
        grow: Duration,
        render: Duration,
        blit: Duration,
        frame_work: Duration,
    ) {
        if self.first_frame_us.is_none() {
            self.first_frame_us = Some(saturating_micros(self.process_start.elapsed()));
        }
        self.frames = self.frames.saturating_add(1);
        if saturating_micros(frame_work) > self.frame_budget_us {
            self.deadline_misses = self.deadline_misses.saturating_add(1);
        }

        if self.frame_work.len() < MAX_LIVE_SAMPLES {
            self.grow.record(grow);
            self.render.record(render);
            self.blit.record(blit);
            self.frame_work.record(frame_work);
        } else {
            self.samples_dropped = self.samples_dropped.saturating_add(1);
        }
    }

    pub fn write_atomic(
        &self,
        path: &Path,
        width: u32,
        height: u32,
        refresh_hz: u32,
        branch_count: usize,
        release_us: u64,
    ) -> io::Result<()> {
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "performance receipt path has no parent")
        })?;
        fs::create_dir_all(parent)?;

        let value = serde_json::json!({
            "version": PERFORMANCE_RECEIPT_VERSION,
            "resolution": { "width": width, "height": height },
            "refresh_hz": refresh_hz,
            "drm_open_us": self.drm_open_us,
            "first_frame_us": self.first_frame_us,
            "frames": self.frames,
            "deadline_misses": self.deadline_misses,
            "retained_samples": self.frame_work.len(),
            "samples_dropped": self.samples_dropped,
            "branch_count": branch_count,
            "release_us": release_us,
            "grow": self.grow.summary().to_json(),
            "render": self.render.summary().to_json(),
            "blit": self.blit.summary().to_json(),
            "frame_work": self.frame_work.summary().to_json(),
        });
        let bytes = serde_json::to_vec_pretty(&value)
            .map_err(|error| io::Error::other(format!("serialize performance receipt: {error}")))?;
        let tmp = temporary_path(path);
        fs::write(&tmp, bytes)?;
        fs::rename(tmp, path)?;
        Ok(())
    }
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((1..=100).contains(&percentile));
    // Standard nearest-rank percentile: ceil(P/100 * N), converted to a
    // zero-based index. This makes 1..=100 report p50=50, p95=95, p99=99.
    let rank = sorted
        .len()
        .saturating_mul(percentile)
        .saturating_add(99)
        / 100;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(format!(".tmp-{}", std::process::id()));
    PathBuf::from(temporary)
}

fn saturating_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_percentiles_are_deterministic() {
        let mut series = TimingSeries::default();
        for value in 1..=100 {
            series.record_us(value);
        }
        let summary = series.summary();
        assert_eq!(summary.count, 100);
        assert_eq!(summary.min_us, 1);
        assert_eq!(summary.p50_us, 50);
        assert_eq!(summary.p95_us, 95);
        assert_eq!(summary.p99_us, 99);
        assert_eq!(summary.max_us, 100);
    }

    #[test]
    fn singleton_percentiles_are_the_sample() {
        let mut series = TimingSeries::default();
        series.record_us(42);
        let summary = series.summary();
        assert_eq!(summary.p50_us, 42);
        assert_eq!(summary.p95_us, 42);
        assert_eq!(summary.p99_us, 42);
    }

    #[test]
    fn live_samples_are_hard_bounded() {
        let mut recorder = BootPerformanceRecorder::new(
            Instant::now(),
            Duration::from_millis(34),
        );
        for _ in 0..(MAX_LIVE_SAMPLES + 7) {
            recorder.record_frame(
                Duration::from_micros(1),
                Duration::from_micros(2),
                Duration::from_micros(3),
                Duration::from_micros(6),
            );
        }
        assert_eq!(recorder.frame_work.len(), MAX_LIVE_SAMPLES);
        assert_eq!(recorder.samples_dropped, 7);
        assert_eq!(recorder.frames, (MAX_LIVE_SAMPLES + 7) as u64);
    }

    #[test]
    fn empty_summary_is_explicit_zero_state() {
        assert_eq!(TimingSeries::default().summary(), TimingSummary::empty());
    }
}

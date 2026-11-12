// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Optional installation/boot progress monitoring.
///
/// The FIFO is opened with `O_NONBLOCK`. A missing writer, missing pipe, invalid
/// pipe, or transient read error therefore never stalls the boot renderer.
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader};
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    DiskWrite(u64),
    DerivationComplete(String),
    PhaseChange(String),
    InstallComplete,
    IoRate(f64),
}

pub struct ProgressMonitor {
    reader: Option<BufReader<File>>,
    last_diskstats_bytes: u64,
    last_diskstats_time: Instant,
    pub io_rate: f32,
    pub complete: bool,
    line_buf: String,
}

impl ProgressMonitor {
    /// Create a monitor reading from a FIFO if one is available.
    ///
    /// Opening a FIFO read-only without `O_NONBLOCK` waits for a writer and can
    /// deadlock early boot. Keep this path fail-open and optional.
    pub fn new(pipe_path: Option<&str>) -> Self {
        let reader = pipe_path.and_then(Self::open_fifo_nonblocking);
        Self {
            reader,
            last_diskstats_bytes: Self::read_diskstats_bytes(),
            last_diskstats_time: Instant::now(),
            io_rate: 0.0,
            complete: false,
            line_buf: String::with_capacity(256),
        }
    }

    fn open_fifo_nonblocking(path: &str) -> Option<BufReader<File>> {
        let path = Path::new(path);
        let metadata = path.metadata().ok()?;
        if !metadata.file_type().is_fifo() {
            return None;
        }
        OpenOptions::new()
            .read(true)
            .custom_flags(nix::libc::O_NONBLOCK)
            .open(path)
            .ok()
            .map(BufReader::new)
    }

    /// Poll for available events without waiting for a producer.
    pub fn poll(&mut self) -> Vec<ProgressEvent> {
        let mut events = Vec::new();

        if let Some(ref mut reader) = self.reader {
            loop {
                self.line_buf.clear();
                match reader.read_line(&mut self.line_buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        let line = self.line_buf.trim();
                        if let Some(event) = Self::parse_line(line) {
                            if matches!(event, ProgressEvent::InstallComplete) {
                                self.complete = true;
                            }
                            events.push(event);
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_diskstats_time);
        if elapsed >= Duration::from_millis(250) {
            let current_bytes = Self::read_diskstats_bytes();
            let delta = current_bytes.saturating_sub(self.last_diskstats_bytes);
            let bytes_per_sec = delta as f64 / elapsed.as_secs_f64();
            self.io_rate = (bytes_per_sec / 200_000_000.0).min(1.0) as f32;

            if delta > 0 {
                events.push(ProgressEvent::IoRate(bytes_per_sec));
                events.push(ProgressEvent::DiskWrite(current_bytes));
            }

            self.last_diskstats_bytes = current_bytes;
            self.last_diskstats_time = now;
        }

        events
    }

    /// Protocol:
    /// - `DRV:<derivation-name>`
    /// - `PHASE:<phase-name>`
    /// - `COMPLETE`
    /// - `WRITE:<bytes>`
    fn parse_line(line: &str) -> Option<ProgressEvent> {
        if line.is_empty() {
            return None;
        }
        if let Some(derivation) = line.strip_prefix("DRV:") {
            return Some(ProgressEvent::DerivationComplete(
                derivation.trim().to_string(),
            ));
        }
        if let Some(phase) = line.strip_prefix("PHASE:") {
            return Some(ProgressEvent::PhaseChange(phase.trim().to_string()));
        }
        if line == "COMPLETE" {
            return Some(ProgressEvent::InstallComplete);
        }
        if let Some(bytes) = line.strip_prefix("WRITE:") {
            if let Ok(bytes) = bytes.trim().parse::<u64>() {
                return Some(ProgressEvent::DiskWrite(bytes));
            }
        }
        None
    }

    fn read_diskstats_bytes() -> u64 {
        let path = Path::new("/proc/diskstats");
        let Ok(file) = File::open(path) else {
            return 0;
        };
        let reader = BufReader::new(file);
        let mut total = 0u64;

        for line in reader.lines() {
            let Ok(line) = line else { continue };
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 10 {
                continue;
            }
            let name = fields[2];
            if !is_whole_disk(name) {
                continue;
            }
            if let Ok(sectors) = fields[9].parse::<u64>() {
                total = total.saturating_add(sectors.saturating_mul(512));
            }
        }
        total
    }
}

fn is_whole_disk(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("nvme") {
        // Whole NVMe namespaces look like nvme0n1; partitions add p1, p2, ...
        return rest.contains('n') && !rest.contains('p');
    }
    if let Some(rest) = name.strip_prefix("sd") {
        return rest.len() == 1 && rest.as_bytes()[0].is_ascii_alphabetic();
    }
    if let Some(rest) = name.strip_prefix("vd") {
        return rest.len() == 1 && rest.as_bytes()[0].is_ascii_alphabetic();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_derivation() {
        let event = ProgressMonitor::parse_line("DRV:nixos-system-luminous-24.11");
        assert!(matches!(
            event,
            Some(ProgressEvent::DerivationComplete(ref value))
                if value == "nixos-system-luminous-24.11"
        ));
    }

    #[test]
    fn parses_phase() {
        let event = ProgressMonitor::parse_line("PHASE:formatting");
        assert!(matches!(
            event,
            Some(ProgressEvent::PhaseChange(ref value)) if value == "formatting"
        ));
    }

    #[test]
    fn parses_complete() {
        assert!(matches!(
            ProgressMonitor::parse_line("COMPLETE"),
            Some(ProgressEvent::InstallComplete)
        ));
    }

    #[test]
    fn parses_write() {
        assert!(matches!(
            ProgressMonitor::parse_line("WRITE:1048576"),
            Some(ProgressEvent::DiskWrite(1_048_576))
        ));
    }

    #[test]
    fn rejects_empty_and_unknown() {
        assert!(ProgressMonitor::parse_line("").is_none());
        assert!(ProgressMonitor::parse_line("GARBAGE:stuff").is_none());
    }

    #[test]
    fn creation_without_pipe_is_immediate_and_optional() {
        let monitor = ProgressMonitor::new(None);
        assert!(!monitor.complete);
        assert!(monitor.reader.is_none());
    }

    #[test]
    fn nonexistent_pipe_fails_open() {
        let monitor = ProgressMonitor::new(Some("/nonexistent/spore-boot-progress"));
        assert!(monitor.reader.is_none());
    }

    #[test]
    fn disk_name_filter_skips_partitions() {
        assert!(is_whole_disk("sda"));
        assert!(!is_whole_disk("sda1"));
        assert!(is_whole_disk("vda"));
        assert!(!is_whole_disk("vda2"));
        assert!(is_whole_disk("nvme0n1"));
        assert!(!is_whole_disk("nvme0n1p1"));
    }
}

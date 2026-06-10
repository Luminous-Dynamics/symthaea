// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Installation progress monitoring.
///
/// Reads events from a named pipe (FIFO) written by the installer, or
/// falls back to polling /proc/diskstats for disk I/O rate estimation.
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::os::unix::fs::FileTypeExt;
use std::path::Path;
use std::time::{Duration, Instant};

/// Events emitted by the progress monitor.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Disk write observed (cumulative bytes).
    DiskWrite(u64),
    /// A Nix derivation has finished building.
    DerivationComplete(String),
    /// Installation phase change (e.g., "partitioning", "formatting", "installing", "bootloader").
    PhaseChange(String),
    /// Installation is complete — begin final animation.
    InstallComplete,
    /// I/O rate estimate (bytes per second).
    IoRate(f64),
}

/// Progress monitor that reads from a named pipe.
pub struct ProgressMonitor {
    reader: Option<BufReader<File>>,
    /// Fallback: poll /proc/diskstats
    last_diskstats_bytes: u64,
    last_diskstats_time: Instant,
    /// Normalized I/O rate (0.0 to 1.0).
    pub io_rate: f32,
    /// Whether installation is complete.
    pub complete: bool,
    line_buf: String,
}

impl ProgressMonitor {
    /// Create a monitor reading from the given named pipe path.
    /// If the pipe doesn't exist or can't be opened, falls back to diskstats polling.
    pub fn new(pipe_path: Option<&str>) -> Self {
        let reader = pipe_path.and_then(|path| {
            let p = Path::new(path);
            if !p.exists() {
                return None;
            }
            // Check it's a FIFO
            if let Ok(meta) = p.metadata() {
                if !meta.file_type().is_fifo() {
                    return None;
                }
            }
            // Open non-blocking — we'll poll
            File::open(p).ok().map(BufReader::new)
        });

        Self {
            reader,
            last_diskstats_bytes: Self::read_diskstats_bytes(),
            last_diskstats_time: Instant::now(),
            io_rate: 0.0,
            complete: false,
            line_buf: String::with_capacity(256),
        }
    }

    /// Poll for new events. Non-blocking: returns events available right now.
    pub fn poll(&mut self) -> Vec<ProgressEvent> {
        let mut events = Vec::new();

        if let Some(ref mut reader) = self.reader {
            // Try reading lines from the pipe
            loop {
                self.line_buf.clear();
                match reader.read_line(&mut self.line_buf) {
                    Ok(0) => break, // EOF / no data
                    Ok(_) => {
                        let line = self.line_buf.trim();
                        if let Some(ev) = Self::parse_line(line) {
                            if matches!(ev, ProgressEvent::InstallComplete) {
                                self.complete = true;
                            }
                            events.push(ev);
                        }
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
        }

        // Always poll diskstats for I/O rate
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_diskstats_time);
        if elapsed >= Duration::from_millis(250) {
            let current_bytes = Self::read_diskstats_bytes();
            let delta = current_bytes.saturating_sub(self.last_diskstats_bytes);
            let bytes_per_sec = delta as f64 / elapsed.as_secs_f64();

            // Normalize: 0 = idle, 1.0 = ~200MB/s (fast NVMe write)
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

    /// Parse a line from the progress pipe.
    /// Protocol:
    ///   DRV:<derivation-name>
    ///   PHASE:<phase-name>
    ///   COMPLETE
    ///   WRITE:<bytes>
    fn parse_line(line: &str) -> Option<ProgressEvent> {
        if line.is_empty() {
            return None;
        }
        if let Some(drv) = line.strip_prefix("DRV:") {
            return Some(ProgressEvent::DerivationComplete(drv.trim().to_string()));
        }
        if let Some(phase) = line.strip_prefix("PHASE:") {
            return Some(ProgressEvent::PhaseChange(phase.trim().to_string()));
        }
        if line == "COMPLETE" {
            return Some(ProgressEvent::InstallComplete);
        }
        if let Some(bytes_str) = line.strip_prefix("WRITE:") {
            if let Ok(bytes) = bytes_str.trim().parse::<u64>() {
                return Some(ProgressEvent::DiskWrite(bytes));
            }
        }
        None
    }

    /// Read total sector writes from /proc/diskstats (all block devices).
    /// Returns approximate bytes written.
    fn read_diskstats_bytes() -> u64 {
        let mut total: u64 = 0;
        let path = Path::new("/proc/diskstats");
        if !path.exists() {
            return 0;
        }
        let Ok(file) = File::open(path) else {
            return 0;
        };
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let Ok(line) = line else { continue };
            let fields: Vec<&str> = line.split_whitespace().collect();
            // Field 9 (0-indexed) is sectors written
            if fields.len() >= 10 {
                let name = fields[2];
                // Only count whole-disk devices, skip partitions
                if name.starts_with("sd") || name.starts_with("nvme") || name.starts_with("vd") {
                    if let Ok(sectors) = fields[9].parse::<u64>() {
                        total += sectors * 512; // sectors are 512 bytes
                    }
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_derivation() {
        let ev = ProgressMonitor::parse_line("DRV:nixos-system-luminous-24.11");
        assert!(
            matches!(ev, Some(ProgressEvent::DerivationComplete(ref s)) if s == "nixos-system-luminous-24.11")
        );
    }

    #[test]
    fn test_parse_phase() {
        let ev = ProgressMonitor::parse_line("PHASE:formatting");
        assert!(matches!(ev, Some(ProgressEvent::PhaseChange(ref s)) if s == "formatting"));
    }

    #[test]
    fn test_parse_complete() {
        let ev = ProgressMonitor::parse_line("COMPLETE");
        assert!(matches!(ev, Some(ProgressEvent::InstallComplete)));
    }

    #[test]
    fn test_parse_write() {
        let ev = ProgressMonitor::parse_line("WRITE:1048576");
        assert!(matches!(ev, Some(ProgressEvent::DiskWrite(1048576))));
    }

    #[test]
    fn test_parse_empty() {
        assert!(ProgressMonitor::parse_line("").is_none());
    }

    #[test]
    fn test_parse_unknown() {
        assert!(ProgressMonitor::parse_line("GARBAGE:stuff").is_none());
    }

    #[test]
    fn test_monitor_creation_no_pipe() {
        let mon = ProgressMonitor::new(None);
        assert!(!mon.complete);
        assert!(mon.reader.is_none());
    }

    #[test]
    fn test_monitor_creation_nonexistent_pipe() {
        let mon = ProgressMonitor::new(Some("/nonexistent/pipe"));
        assert!(mon.reader.is_none());
    }
}

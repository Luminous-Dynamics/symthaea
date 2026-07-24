// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Printer API abstraction layer.
//!
//! Provides a unified [`PrinterApi`] trait for controlling 3D printers,
//! with concrete implementations for OctoPrint, Moonraker, and a fully
//! functional mock for testing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Current operational status of a printer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrinterStatus {
    /// No connection established.
    Disconnected,
    /// Connected and ready to accept jobs.
    Idle,
    /// Actively executing a print job.
    Printing {
        /// Completion fraction in [0.0, 1.0].
        progress: f32,
        /// Identifier for the running job.
        job_id: String,
    },
    /// Printer is in an error state.
    Error(String),
}

/// A snapshot of the printer's thermal state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureReading {
    pub nozzle_actual: f32,
    pub nozzle_target: f32,
    pub bed_actual: f32,
    pub bed_target: f32,
}

impl TemperatureReading {
    /// Returns `true` when both nozzle and bed are within `tolerance` degrees
    /// of their respective targets.
    pub fn at_target(&self, tolerance: f32) -> bool {
        (self.nozzle_actual - self.nozzle_target).abs() <= tolerance
            && (self.bed_actual - self.bed_target).abs() <= tolerance
    }
}

/// Errors that can occur during printer communication.
#[derive(Debug, Clone)]
pub enum PrinterError {
    /// Failed to establish a connection.
    ConnectionFailed(String),
    /// A request timed out.
    Timeout,
    /// A generic API-level error.
    ApiError(String),
    /// Operation attempted while not connected.
    NotConnected,
    /// Backend is declared but no live transport implementation is compiled.
    BackendUnavailable(String),
}

impl fmt::Display for PrinterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            Self::Timeout => write!(f, "request timed out"),
            Self::ApiError(msg) => write!(f, "API error: {msg}"),
            Self::NotConnected => write!(f, "printer not connected"),
            Self::BackendUnavailable(name) => {
                write!(f, "printer backend unavailable: {name}")
            }
        }
    }
}

impl std::error::Error for PrinterError {}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Unified control surface for 3D printers.
pub trait PrinterApi {
    /// Human-readable name of the printer backend.
    fn name(&self) -> &str;

    /// Establish a connection to the printer.
    fn connect(&mut self) -> Result<(), PrinterError>;

    /// Query the printer's current status.
    fn status(&self) -> Result<PrinterStatus, PrinterError>;

    /// Submit raw G-code and return a job identifier.
    fn submit_gcode(&mut self, gcode: &str) -> Result<String, PrinterError>;

    /// Cancel a running job by its identifier.
    fn cancel_job(&mut self, job_id: &str) -> Result<(), PrinterError>;

    /// Read current temperature sensors.
    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError>;
}

// ---------------------------------------------------------------------------
// MockPrinter
// ---------------------------------------------------------------------------

/// A job tracked by the mock printer.
#[derive(Debug, Clone)]
pub struct MockJob {
    pub id: String,
    pub gcode_len: usize,
    pub progress: f32,
}

/// Fully functional in-memory printer for testing.
#[derive(Debug)]
pub struct MockPrinter {
    connected: bool,
    status: PrinterStatus,
    job_counter: u32,
    jobs: VecDeque<MockJob>,
    temperatures: TemperatureReading,
    /// When `true` the next mutable operation will fail.
    pub error_on_next: bool,
}

impl MockPrinter {
    /// Create a new disconnected mock printer.
    pub fn new() -> Self {
        Self {
            connected: false,
            status: PrinterStatus::Disconnected,
            job_counter: 0,
            jobs: VecDeque::new(),
            temperatures: TemperatureReading {
                nozzle_actual: 25.0,
                nozzle_target: 200.0,
                bed_actual: 25.0,
                bed_target: 60.0,
            },
            error_on_next: false,
        }
    }

    /// Advance the progress of the front job by `delta` (clamped to 1.0).
    /// When progress reaches 1.0, the job is finished and the printer
    /// transitions to [`PrinterStatus::Idle`].
    pub fn advance_progress(&mut self, delta: f32) {
        if let Some(job) = self.jobs.front_mut() {
            job.progress = (job.progress + delta).min(1.0);
            if job.progress >= 1.0 {
                let _ = self.jobs.pop_front();
                if self.jobs.is_empty() {
                    self.status = PrinterStatus::Idle;
                } else {
                    let next = self.jobs.front().unwrap();
                    self.status = PrinterStatus::Printing {
                        progress: next.progress,
                        job_id: next.id.clone(),
                    };
                }
            } else {
                self.status = PrinterStatus::Printing {
                    progress: job.progress,
                    job_id: job.id.clone(),
                };
            }
        }
    }

    /// Set the mock temperatures directly (for testing thermal checks).
    pub fn set_temperatures(&mut self, temps: TemperatureReading) {
        self.temperatures = temps;
    }

    fn require_connected(&self) -> Result<(), PrinterError> {
        if !self.connected {
            Err(PrinterError::NotConnected)
        } else {
            Ok(())
        }
    }

    fn check_error_flag(&mut self) -> Result<(), PrinterError> {
        if self.error_on_next {
            self.error_on_next = false;
            Err(PrinterError::ApiError("forced error".into()))
        } else {
            Ok(())
        }
    }
}

impl Default for MockPrinter {
    fn default() -> Self {
        Self::new()
    }
}

impl PrinterApi for MockPrinter {
    fn name(&self) -> &str {
        "MockPrinter"
    }

    fn connect(&mut self) -> Result<(), PrinterError> {
        self.check_error_flag()?;
        self.connected = true;
        self.status = PrinterStatus::Idle;
        Ok(())
    }

    fn status(&self) -> Result<PrinterStatus, PrinterError> {
        self.require_connected()?;
        Ok(self.status.clone())
    }

    fn submit_gcode(&mut self, gcode: &str) -> Result<String, PrinterError> {
        self.require_connected()?;
        self.check_error_flag()?;

        if gcode.is_empty() {
            return Err(PrinterError::ApiError("empty gcode".into()));
        }

        self.job_counter += 1;
        let job_id = format!("mock-job-{}", self.job_counter);

        let job = MockJob {
            id: job_id.clone(),
            gcode_len: gcode.len(),
            progress: 0.0,
        };

        let is_first = self.jobs.is_empty();
        self.jobs.push_back(job);

        if is_first {
            self.status = PrinterStatus::Printing {
                progress: 0.0,
                job_id: job_id.clone(),
            };
        }

        Ok(job_id)
    }

    fn cancel_job(&mut self, job_id: &str) -> Result<(), PrinterError> {
        self.require_connected()?;
        self.check_error_flag()?;

        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != job_id);

        if self.jobs.len() == before {
            return Err(PrinterError::ApiError(format!("job not found: {job_id}")));
        }

        // Update status.
        if self.jobs.is_empty() {
            self.status = PrinterStatus::Idle;
        } else {
            let front = self.jobs.front().unwrap();
            self.status = PrinterStatus::Printing {
                progress: front.progress,
                job_id: front.id.clone(),
            };
        }

        Ok(())
    }

    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError> {
        self.require_connected()?;
        Ok(self.temperatures.clone())
    }
}

// ---------------------------------------------------------------------------
// OctoPrintClient (stub)
// ---------------------------------------------------------------------------

/// Capability marker for a future OctoPrint REST implementation.
///
/// This type fails closed: [`PrinterApi::connect`] returns
/// [`PrinterError::BackendUnavailable`] until a real authenticated transport is
/// implemented. It must never report a successful live connection.
pub struct OctoPrintClient {
    base_url: String,
    api_key: String,
    connected: bool,
}

impl OctoPrintClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            connected: false,
        }
    }

    /// Access the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Access the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    fn stub_err() -> PrinterError {
        PrinterError::BackendUnavailable("OctoPrint transport is not implemented".into())
    }
}

impl PrinterApi for OctoPrintClient {
    fn name(&self) -> &str {
        "OctoPrint"
    }

    fn connect(&mut self) -> Result<(), PrinterError> {
        self.connected = false;
        Err(Self::stub_err())
    }

    fn status(&self) -> Result<PrinterStatus, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }

    fn submit_gcode(&mut self, _gcode: &str) -> Result<String, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }

    fn cancel_job(&mut self, _job_id: &str) -> Result<(), PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }

    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }
}

// ---------------------------------------------------------------------------
// MoonrakerClient (stub)
// ---------------------------------------------------------------------------

/// Capability marker for a future Moonraker (Klipper) JSON-RPC transport.
///
/// Connection fails closed until the transport is implemented.
pub struct MoonrakerClient {
    base_url: String,
    connected: bool,
}

impl MoonrakerClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            connected: false,
        }
    }

    /// Access the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn stub_err() -> PrinterError {
        PrinterError::BackendUnavailable("Moonraker transport is not implemented".into())
    }
}

impl PrinterApi for MoonrakerClient {
    fn name(&self) -> &str {
        "Moonraker"
    }

    fn connect(&mut self) -> Result<(), PrinterError> {
        self.connected = false;
        Err(Self::stub_err())
    }

    fn status(&self) -> Result<PrinterStatus, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }

    fn submit_gcode(&mut self, _gcode: &str) -> Result<String, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }

    fn cancel_job(&mut self, _job_id: &str) -> Result<(), PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }

    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Err(Self::stub_err())
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a printer backend from an explicit URL scheme.
///
/// Only `mock://` is operational in this crate snapshot. Declared live
/// backends fail closed instead of silently constructing clients that can never
/// complete a request.
pub fn printer_from_url(url: &str) -> Result<Box<dyn PrinterApi>, PrinterError> {
    let lower = url.to_lowercase();
    if lower.starts_with("mock://") {
        Ok(Box::new(MockPrinter::new()))
    } else if lower.contains("octoprint") {
        Err(PrinterError::BackendUnavailable(
            "OctoPrint transport is not implemented".into(),
        ))
    } else if lower.contains("moonraker") || lower.contains("klipper") {
        Err(PrinterError::BackendUnavailable(
            "Moonraker transport is not implemented".into(),
        ))
    } else {
        Err(PrinterError::ConnectionFailed(format!(
            "unsupported printer URL: {url}"
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_connect_disconnect() {
        let mut p = MockPrinter::new();
        assert!(p.status().is_err()); // not connected
        p.connect().unwrap();
        assert!(matches!(p.status().unwrap(), PrinterStatus::Idle));
    }

    #[test]
    fn mock_submit_job() {
        let mut p = MockPrinter::new();
        p.connect().unwrap();
        let id = p.submit_gcode("G28\nG1 X10").unwrap();
        assert!(id.starts_with("mock-job-"));
        match p.status().unwrap() {
            PrinterStatus::Printing { job_id, progress } => {
                assert_eq!(job_id, id);
                assert!((progress - 0.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Printing, got {:?}", other),
        }
    }

    #[test]
    fn mock_cancel_job() {
        let mut p = MockPrinter::new();
        p.connect().unwrap();
        let id = p.submit_gcode("G28").unwrap();
        p.cancel_job(&id).unwrap();
        assert!(matches!(p.status().unwrap(), PrinterStatus::Idle));
    }

    #[test]
    fn mock_progress() {
        let mut p = MockPrinter::new();
        p.connect().unwrap();
        let _id = p.submit_gcode("G28").unwrap();
        p.advance_progress(0.5);
        if let PrinterStatus::Printing { progress, .. } = p.status().unwrap() {
            assert!((progress - 0.5).abs() < f32::EPSILON);
        } else {
            panic!("expected Printing");
        }
        // Complete the job.
        p.advance_progress(0.6);
        assert!(matches!(p.status().unwrap(), PrinterStatus::Idle));
    }

    #[test]
    fn mock_temperature() {
        let mut p = MockPrinter::new();
        p.connect().unwrap();
        let t = p.get_temperatures().unwrap();
        assert!(!t.at_target(1.0)); // defaults far from target

        p.set_temperatures(TemperatureReading {
            nozzle_actual: 200.0,
            nozzle_target: 200.0,
            bed_actual: 60.0,
            bed_target: 60.0,
        });
        assert!(p.get_temperatures().unwrap().at_target(0.1));
    }

    #[test]
    fn mock_not_connected_errors() {
        let mut p = MockPrinter::new();
        assert!(p.status().is_err());
        assert!(p.submit_gcode("G28").is_err());
        assert!(p.cancel_job("x").is_err());
        assert!(p.get_temperatures().is_err());
    }

    #[test]
    fn mock_empty_gcode_rejected() {
        let mut p = MockPrinter::new();
        p.connect().unwrap();
        let err = p.submit_gcode("").unwrap_err();
        assert!(matches!(err, PrinterError::ApiError(_)));
    }

    #[test]
    fn mock_multiple_jobs() {
        let mut p = MockPrinter::new();
        p.connect().unwrap();
        let j1 = p.submit_gcode("G28").unwrap();
        let j2 = p.submit_gcode("G1 X10").unwrap();
        assert_ne!(j1, j2);
        assert_eq!(p.jobs.len(), 2);

        // Cancel front job — second becomes active.
        p.cancel_job(&j1).unwrap();
        if let PrinterStatus::Printing { job_id, .. } = p.status().unwrap() {
            assert_eq!(job_id, j2);
        } else {
            panic!("expected j2 active");
        }

        // Complete second.
        p.advance_progress(1.0);
        assert!(matches!(p.status().unwrap(), PrinterStatus::Idle));
    }

    #[test]
    fn octoprint_stub_lifecycle() {
        let mut c = OctoPrintClient::new("http://octoprint.local", "key123");
        assert_eq!(c.name(), "OctoPrint");
        assert_eq!(c.base_url(), "http://octoprint.local");
        assert_eq!(c.api_key(), "key123");

        // Not connected yet.
        assert!(c.status().is_err());

        let err = c.connect().unwrap_err();
        assert!(matches!(err, PrinterError::BackendUnavailable(_)));
        assert!(matches!(c.status(), Err(PrinterError::NotConnected)));
    }

    #[test]
    fn moonraker_stub_lifecycle() {
        let mut c = MoonrakerClient::new("http://moonraker.local");
        assert_eq!(c.name(), "Moonraker");
        assert_eq!(c.base_url(), "http://moonraker.local");

        assert!(c.get_temperatures().is_err());
        let err = c.connect().unwrap_err();
        assert!(matches!(err, PrinterError::BackendUnavailable(_)));
        assert!(matches!(
            c.get_temperatures(),
            Err(PrinterError::NotConnected)
        ));
    }

    #[test]
    fn url_factory_fails_closed_for_live_backends() {
        assert!(matches!(
            printer_from_url("http://my-octoprint:5000"),
            Err(PrinterError::BackendUnavailable(_))
        ));
        assert!(matches!(
            printer_from_url("http://moonraker.lan:7125"),
            Err(PrinterError::BackendUnavailable(_))
        ));

        let p = printer_from_url("mock://local").unwrap();
        assert_eq!(p.name(), "MockPrinter");
    }

    #[test]
    fn error_display() {
        let e = PrinterError::ConnectionFailed("refused".into());
        assert!(e.to_string().contains("refused"));

        let e = PrinterError::Timeout;
        assert!(e.to_string().contains("timed out"));

        let e = PrinterError::ApiError("bad request".into());
        assert!(e.to_string().contains("bad request"));

        let e = PrinterError::NotConnected;
        assert!(e.to_string().contains("not connected"));

        let e = PrinterError::BackendUnavailable("octoprint".into());
        assert!(e.to_string().contains("unavailable"));
    }
}

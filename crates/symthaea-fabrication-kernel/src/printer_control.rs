//! Printer API abstraction for OctoPrint / Moonraker.
//!
//! Provides a unified `PrinterApi` trait that encapsulates 3D printer
//! communication regardless of backend firmware. Includes stub
//! implementations for OctoPrint and Moonraker (real HTTP wired behind
//! `printer-control` feature), plus a fully-functional `MockPrinter`
//! for deterministic testing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors that can occur during printer communication.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrinterError {
    /// TCP/serial connection could not be established.
    ConnectionFailed(String),
    /// The printer did not respond within the deadline.
    Timeout,
    /// The printer API returned an application-level error.
    ApiError(String),
    /// The operation requires a connected printer.
    NotConnected,
}

impl fmt::Display for PrinterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrinterError::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            PrinterError::Timeout => write!(f, "printer timeout"),
            PrinterError::ApiError(msg) => write!(f, "API error: {msg}"),
            PrinterError::NotConnected => write!(f, "printer not connected"),
        }
    }
}

impl std::error::Error for PrinterError {}

// ── Status / Telemetry ──────────────────────────────────────────────────

/// Current operational status of the printer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrinterStatus {
    /// No active connection to the printer.
    Disconnected,
    /// Connected and idle — ready to accept jobs.
    Idle,
    /// Actively executing a print job.
    Printing {
        /// Completion fraction in [0.0, 1.0].
        progress: f32,
        /// Identifier of the running job.
        job_id: String,
    },
    /// The printer has entered an error state.
    Error(String),
}

/// Snapshot of thermal sensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureReading {
    /// Current nozzle temperature in degrees Celsius.
    pub nozzle_actual: f32,
    /// Target nozzle temperature in degrees Celsius.
    pub nozzle_target: f32,
    /// Current heated-bed temperature in degrees Celsius.
    pub bed_actual: f32,
    /// Target heated-bed temperature in degrees Celsius.
    pub bed_target: f32,
}

impl TemperatureReading {
    /// Returns `true` when both nozzle and bed are within `tolerance_c`
    /// degrees of their targets.
    pub fn at_target(&self, tolerance_c: f32) -> bool {
        (self.nozzle_actual - self.nozzle_target).abs() <= tolerance_c
            && (self.bed_actual - self.bed_target).abs() <= tolerance_c
    }
}

// ── Trait ────────────────────────────────────────────────────────────────

/// Unified printer control interface.
///
/// Implementors abstract over different printer firmware APIs (OctoPrint,
/// Moonraker, Klipper, etc.) so that higher-level fabrication logic
/// remains backend-agnostic.
pub trait PrinterApi {
    /// Human-readable name of this printer instance.
    fn name(&self) -> &str;

    /// Establish a connection to the printer.
    fn connect(&mut self) -> Result<(), PrinterError>;

    /// Query the printer's current operational status.
    fn status(&self) -> Result<PrinterStatus, PrinterError>;

    /// Submit raw G-code for execution and return a job identifier.
    fn submit_gcode(&mut self, gcode: &str) -> Result<String, PrinterError>;

    /// Cancel a running job by identifier.
    fn cancel_job(&mut self, job_id: &str) -> Result<(), PrinterError>;

    /// Read the current temperature sensors.
    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError>;
}

// ── OctoPrint ───────────────────────────────────────────────────────────

/// Client for the OctoPrint REST API.
///
/// When the `printer-control` Cargo feature is disabled, all methods
/// return deterministic stub responses so that the rest of the crate can
/// be tested without network access.
pub struct OctoPrintClient {
    /// Base URL of the OctoPrint instance (e.g. `http://192.168.1.50:5000`).
    base_url: String,
    /// OctoPrint API key for authentication.
    api_key: String,
    /// Whether we have an active connection.
    connected: bool,
}

impl OctoPrintClient {
    /// Create a new OctoPrint client.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            connected: false,
        }
    }

    /// Returns the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns a redacted view of the API key (first 4 chars).
    pub fn api_key_redacted(&self) -> String {
        if self.api_key.len() > 4 {
            format!("{}...", &self.api_key[..4])
        } else {
            "****".to_string()
        }
    }
}

impl PrinterApi for OctoPrintClient {
    fn name(&self) -> &str {
        "OctoPrint"
    }

    #[cfg(not(feature = "printer-control"))]
    fn connect(&mut self) -> Result<(), PrinterError> {
        // Stub: mark connected without HTTP.
        // Real impl: POST /api/connection { "command": "connect" }
        self.connected = true;
        Ok(())
    }

    #[cfg(feature = "printer-control")]
    fn connect(&mut self) -> Result<(), PrinterError> {
        // TODO: real HTTP — POST {base_url}/api/connection
        //   Headers: X-Api-Key: {api_key}
        //   Body: { "command": "connect" }
        //   On 2xx: self.connected = true
        //   On error: return ConnectionFailed
        self.connected = true;
        Ok(())
    }

    fn status(&self) -> Result<PrinterStatus, PrinterError> {
        if !self.connected {
            return Ok(PrinterStatus::Disconnected);
        }
        // Stub: always idle when connected.
        // Real impl: GET /api/job → parse state/progress
        Ok(PrinterStatus::Idle)
    }

    fn submit_gcode(&mut self, gcode: &str) -> Result<String, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        if gcode.is_empty() {
            return Err(PrinterError::ApiError("empty gcode".into()));
        }
        // Stub: return a synthetic job id.
        // Real impl: POST /api/files/local (upload) then POST /api/files/local/{filename} { "command": "select", "print": true }
        let job_id = format!("octo-job-{:08x}", gcode.len() as u32);
        Ok(job_id)
    }

    fn cancel_job(&mut self, _job_id: &str) -> Result<(), PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        // Stub: always succeed.
        // Real impl: POST /api/job { "command": "cancel" }
        Ok(())
    }

    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        // Stub: room-temperature reading.
        // Real impl: GET /api/printer → parse temperature.tool0 / temperature.bed
        Ok(TemperatureReading {
            nozzle_actual: 24.0,
            nozzle_target: 0.0,
            bed_actual: 23.0,
            bed_target: 0.0,
        })
    }
}

// ── Moonraker ───────────────────────────────────────────────────────────

/// Client for the Moonraker (Klipper) JSON-RPC/REST API.
///
/// Same stub pattern as `OctoPrintClient` — real HTTP gated behind
/// `printer-control`.
pub struct MoonrakerClient {
    /// Base URL of the Moonraker instance (e.g. `http://192.168.1.60:7125`).
    base_url: String,
    /// Whether we have an active connection.
    connected: bool,
}

impl MoonrakerClient {
    /// Create a new Moonraker client.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            connected: false,
        }
    }

    /// Returns the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl PrinterApi for MoonrakerClient {
    fn name(&self) -> &str {
        "Moonraker"
    }

    #[cfg(not(feature = "printer-control"))]
    fn connect(&mut self) -> Result<(), PrinterError> {
        // Stub: mark connected without HTTP.
        // Real impl: GET /server/info — confirm Moonraker is reachable
        self.connected = true;
        Ok(())
    }

    #[cfg(feature = "printer-control")]
    fn connect(&mut self) -> Result<(), PrinterError> {
        // TODO: real HTTP — GET {base_url}/server/info
        //   On 2xx with result.klippy_connected == true: self.connected = true
        //   On error: return ConnectionFailed
        self.connected = true;
        Ok(())
    }

    fn status(&self) -> Result<PrinterStatus, PrinterError> {
        if !self.connected {
            return Ok(PrinterStatus::Disconnected);
        }
        // Stub: always idle when connected.
        // Real impl: GET /printer/objects/query?print_stats → parse state
        Ok(PrinterStatus::Idle)
    }

    fn submit_gcode(&mut self, gcode: &str) -> Result<String, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        if gcode.is_empty() {
            return Err(PrinterError::ApiError("empty gcode".into()));
        }
        // Stub: return a synthetic job id.
        // Real impl: POST /printer/gcode/script { "script": gcode }
        //   then GET /server/files/metadata to retrieve job id
        let job_id = format!("moon-job-{:08x}", gcode.len() as u32);
        Ok(job_id)
    }

    fn cancel_job(&mut self, _job_id: &str) -> Result<(), PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        // Stub: always succeed.
        // Real impl: POST /printer/print/cancel
        Ok(())
    }

    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        // Stub: room-temperature reading.
        // Real impl: GET /printer/objects/query?extruder&heater_bed
        Ok(TemperatureReading {
            nozzle_actual: 24.5,
            nozzle_target: 0.0,
            bed_actual: 23.5,
            bed_target: 0.0,
        })
    }
}

// ── Mock Printer ────────────────────────────────────────────────────────

/// Job entry tracked by `MockPrinter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockJob {
    /// Job identifier.
    pub job_id: String,
    /// The G-code that was submitted.
    pub gcode: String,
    /// Completion fraction [0.0, 1.0].
    pub progress: f32,
    /// Whether the job is still active.
    pub active: bool,
}

/// Fully functional mock printer for deterministic testing.
///
/// Tracks connection state, job queue, and thermal readings without
/// any I/O. Suitable for unit tests and integration harnesses.
pub struct MockPrinter {
    printer_name: String,
    connected: bool,
    jobs: VecDeque<MockJob>,
    next_job_id: u64,
    nozzle_actual: f32,
    nozzle_target: f32,
    bed_actual: f32,
    bed_target: f32,
    error_state: Option<String>,
}

impl MockPrinter {
    /// Create a new mock printer with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            printer_name: name.into(),
            connected: false,
            jobs: VecDeque::new(),
            next_job_id: 1,
            nozzle_actual: 22.0,
            nozzle_target: 0.0,
            bed_actual: 21.0,
            bed_target: 0.0,
            error_state: None,
        }
    }

    /// Set the nozzle temperature readings.
    pub fn set_nozzle_temperature(&mut self, actual: f32, target: f32) {
        self.nozzle_actual = actual;
        self.nozzle_target = target;
    }

    /// Set the bed temperature readings.
    pub fn set_bed_temperature(&mut self, actual: f32, target: f32) {
        self.bed_actual = actual;
        self.bed_target = target;
    }

    /// Force the printer into an error state.
    pub fn inject_error(&mut self, message: impl Into<String>) {
        self.error_state = Some(message.into());
    }

    /// Clear any injected error state.
    pub fn clear_error(&mut self) {
        self.error_state = None;
    }

    /// Advance the active job's progress by `delta` (clamped to 1.0).
    pub fn advance_progress(&mut self, delta: f32) {
        if let Some(job) = self.jobs.front_mut() {
            if job.active {
                job.progress = (job.progress + delta).min(1.0);
                if job.progress >= 1.0 {
                    job.active = false;
                }
            }
        }
    }

    /// Returns all jobs (active and completed).
    pub fn all_jobs(&self) -> &VecDeque<MockJob> {
        &self.jobs
    }

    /// Returns only active jobs.
    pub fn active_jobs(&self) -> Vec<&MockJob> {
        self.jobs.iter().filter(|j| j.active).collect()
    }

    /// Returns the number of completed jobs.
    pub fn completed_count(&self) -> usize {
        self.jobs.iter().filter(|j| !j.active).count()
    }

    /// Whether the mock printer is in an error state.
    pub fn has_error(&self) -> bool {
        self.error_state.is_some()
    }

    /// Force disconnect (useful for testing reconnection paths).
    pub fn force_disconnect(&mut self) {
        self.connected = false;
    }
}

impl PrinterApi for MockPrinter {
    fn name(&self) -> &str {
        &self.printer_name
    }

    fn connect(&mut self) -> Result<(), PrinterError> {
        if let Some(ref err) = self.error_state {
            return Err(PrinterError::ConnectionFailed(err.clone()));
        }
        self.connected = true;
        Ok(())
    }

    fn status(&self) -> Result<PrinterStatus, PrinterError> {
        if !self.connected {
            return Ok(PrinterStatus::Disconnected);
        }
        if let Some(ref err) = self.error_state {
            return Ok(PrinterStatus::Error(err.clone()));
        }
        // Find the first active job.
        if let Some(job) = self.jobs.iter().find(|j| j.active) {
            return Ok(PrinterStatus::Printing {
                progress: job.progress,
                job_id: job.job_id.clone(),
            });
        }
        Ok(PrinterStatus::Idle)
    }

    fn submit_gcode(&mut self, gcode: &str) -> Result<String, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        if self.error_state.is_some() {
            return Err(PrinterError::ApiError("printer in error state".into()));
        }
        if gcode.is_empty() {
            return Err(PrinterError::ApiError("empty gcode".into()));
        }
        let job_id = format!("mock-job-{:04}", self.next_job_id);
        self.next_job_id += 1;
        self.jobs.push_back(MockJob {
            job_id: job_id.clone(),
            gcode: gcode.to_string(),
            progress: 0.0,
            active: true,
        });
        Ok(job_id)
    }

    fn cancel_job(&mut self, job_id: &str) -> Result<(), PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        let found = self.jobs.iter_mut().find(|j| j.job_id == job_id && j.active);
        match found {
            Some(job) => {
                job.active = false;
                Ok(())
            }
            None => Err(PrinterError::ApiError(format!(
                "no active job with id '{job_id}'"
            ))),
        }
    }

    fn get_temperatures(&self) -> Result<TemperatureReading, PrinterError> {
        if !self.connected {
            return Err(PrinterError::NotConnected);
        }
        Ok(TemperatureReading {
            nozzle_actual: self.nozzle_actual,
            nozzle_target: self.nozzle_target,
            bed_actual: self.bed_actual,
            bed_target: self.bed_target,
        })
    }
}

// ── Utility ─────────────────────────────────────────────────────────────

/// Convenience: create a `Box<dyn PrinterApi>` from a URL string.
///
/// Inspects the URL to guess the backend:
/// - Contains `:7125` or `/server` → Moonraker
/// - Otherwise → OctoPrint (requires `api_key`)
pub fn printer_from_url(url: &str, api_key: Option<&str>) -> Box<dyn PrinterApi> {
    if url.contains(":7125") || url.contains("/server") {
        Box::new(MoonrakerClient::new(url))
    } else {
        Box::new(OctoPrintClient::new(url, api_key.unwrap_or("")))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ---------------------------------------------------------

    fn connected_mock() -> MockPrinter {
        let mut p = MockPrinter::new("TestPrinter");
        p.connect().unwrap();
        p
    }

    // -- PrinterApi trait tests via MockPrinter --------------------------

    #[test]
    fn test_mock_name() {
        let p = MockPrinter::new("Prusa-MK4");
        assert_eq!(p.name(), "Prusa-MK4");
    }

    #[test]
    fn test_mock_connect_disconnect_lifecycle() {
        let mut p = MockPrinter::new("Lifecycle");
        assert_eq!(p.status().unwrap(), PrinterStatus::Disconnected);

        p.connect().unwrap();
        assert_eq!(p.status().unwrap(), PrinterStatus::Idle);

        p.force_disconnect();
        assert_eq!(p.status().unwrap(), PrinterStatus::Disconnected);
    }

    #[test]
    fn test_mock_submit_and_status() {
        let mut p = connected_mock();
        let job_id = p.submit_gcode("G28\nG1 X10 Y10\n").unwrap();
        assert!(job_id.starts_with("mock-job-"));

        match p.status().unwrap() {
            PrinterStatus::Printing { progress, job_id: id } => {
                assert_eq!(progress, 0.0);
                assert_eq!(id, job_id);
            }
            other => panic!("expected Printing, got {:?}", other),
        }
    }

    #[test]
    fn test_mock_cancel_job() {
        let mut p = connected_mock();
        let job_id = p.submit_gcode("G28\n").unwrap();
        p.cancel_job(&job_id).unwrap();

        // After cancellation the printer should be idle (no active jobs).
        assert_eq!(p.status().unwrap(), PrinterStatus::Idle);
        assert_eq!(p.completed_count(), 0); // cancelled, not completed
        assert!(p.active_jobs().is_empty());
    }

    #[test]
    fn test_mock_cancel_nonexistent_job() {
        let mut p = connected_mock();
        let result = p.cancel_job("no-such-job");
        assert!(result.is_err());
        match result.unwrap_err() {
            PrinterError::ApiError(msg) => assert!(msg.contains("no-such-job")),
            other => panic!("expected ApiError, got {:?}", other),
        }
    }

    #[test]
    fn test_mock_temperature_reading() {
        let mut p = connected_mock();
        p.set_nozzle_temperature(210.0, 210.0);
        p.set_bed_temperature(60.0, 60.0);

        let temps = p.get_temperatures().unwrap();
        assert_eq!(temps.nozzle_actual, 210.0);
        assert_eq!(temps.nozzle_target, 210.0);
        assert_eq!(temps.bed_actual, 60.0);
        assert_eq!(temps.bed_target, 60.0);
        assert!(temps.at_target(1.0));
    }

    #[test]
    fn test_temperature_at_target() {
        let temps = TemperatureReading {
            nozzle_actual: 208.0,
            nozzle_target: 210.0,
            bed_actual: 59.0,
            bed_target: 60.0,
        };
        assert!(!temps.at_target(1.0));
        assert!(temps.at_target(2.5));
    }

    #[test]
    fn test_mock_progress_advance() {
        let mut p = connected_mock();
        let job_id = p.submit_gcode("G28\nG1 Z5\n").unwrap();

        p.advance_progress(0.5);
        match p.status().unwrap() {
            PrinterStatus::Printing { progress, .. } => {
                assert!((progress - 0.5).abs() < f32::EPSILON);
            }
            other => panic!("expected Printing, got {:?}", other),
        }

        // Advance to completion.
        p.advance_progress(0.6); // clamped to 1.0
        assert_eq!(p.status().unwrap(), PrinterStatus::Idle);
        assert_eq!(p.completed_count(), 0); // auto-completed jobs are deactivated
        assert!(!p.all_jobs().iter().any(|j| j.job_id == job_id && j.active));
    }

    #[test]
    fn test_mock_error_injection() {
        let mut p = MockPrinter::new("ErrorTest");
        p.inject_error("thermal runaway");

        // Cannot connect while in error state.
        let result = p.connect();
        assert!(matches!(result, Err(PrinterError::ConnectionFailed(_))));

        // Clear error and connect.
        p.clear_error();
        p.connect().unwrap();

        // Inject error while connected — status reflects it.
        p.inject_error("nozzle jam");
        match p.status().unwrap() {
            PrinterStatus::Error(msg) => assert_eq!(msg, "nozzle jam"),
            other => panic!("expected Error, got {:?}", other),
        }

        // Cannot submit while in error state.
        let result = p.submit_gcode("G28\n");
        assert!(matches!(result, Err(PrinterError::ApiError(_))));
    }

    #[test]
    fn test_not_connected_errors() {
        let mut p = MockPrinter::new("Offline");
        assert!(matches!(
            p.submit_gcode("G28\n"),
            Err(PrinterError::NotConnected)
        ));
        assert!(matches!(
            p.cancel_job("x"),
            Err(PrinterError::NotConnected)
        ));
        assert!(matches!(
            p.get_temperatures(),
            Err(PrinterError::NotConnected)
        ));
    }

    #[test]
    fn test_empty_gcode_rejected() {
        let mut p = connected_mock();
        let result = p.submit_gcode("");
        assert!(matches!(result, Err(PrinterError::ApiError(_))));
    }

    #[test]
    fn test_multiple_jobs_queued() {
        let mut p = connected_mock();
        let id1 = p.submit_gcode("G28\n").unwrap();
        let id2 = p.submit_gcode("G1 X50\n").unwrap();
        assert_ne!(id1, id2);
        assert_eq!(p.active_jobs().len(), 2);
        assert_eq!(p.all_jobs().len(), 2);
    }

    // -- OctoPrint stub tests -------------------------------------------

    #[test]
    fn test_octoprint_stub_lifecycle() {
        let mut client = OctoPrintClient::new("http://localhost:5000", "ABCD1234");
        assert_eq!(client.name(), "OctoPrint");
        assert_eq!(client.base_url(), "http://localhost:5000");
        assert_eq!(client.api_key_redacted(), "ABCD...");

        client.connect().unwrap();
        assert_eq!(client.status().unwrap(), PrinterStatus::Idle);

        let temps = client.get_temperatures().unwrap();
        assert!(temps.nozzle_actual > 0.0);

        let job_id = client.submit_gcode("G28\n").unwrap();
        assert!(job_id.starts_with("octo-job-"));
        client.cancel_job(&job_id).unwrap();
    }

    // -- Moonraker stub tests -------------------------------------------

    #[test]
    fn test_moonraker_stub_lifecycle() {
        let mut client = MoonrakerClient::new("http://localhost:7125");
        assert_eq!(client.name(), "Moonraker");
        assert_eq!(client.base_url(), "http://localhost:7125");

        client.connect().unwrap();
        assert_eq!(client.status().unwrap(), PrinterStatus::Idle);

        let temps = client.get_temperatures().unwrap();
        assert!(temps.bed_actual > 0.0);

        let job_id = client.submit_gcode("G28\n").unwrap();
        assert!(job_id.starts_with("moon-job-"));
        client.cancel_job(&job_id).unwrap();
    }

    // -- printer_from_url ------------------------------------------------

    #[test]
    fn test_printer_from_url_moonraker() {
        let p = printer_from_url("http://192.168.1.60:7125", None);
        assert_eq!(p.name(), "Moonraker");
    }

    #[test]
    fn test_printer_from_url_octoprint() {
        let p = printer_from_url("http://192.168.1.50:5000", Some("key"));
        assert_eq!(p.name(), "OctoPrint");
    }

    // -- PrinterError display -------------------------------------------

    #[test]
    fn test_printer_error_display() {
        assert_eq!(
            PrinterError::ConnectionFailed("refused".into()).to_string(),
            "connection failed: refused"
        );
        assert_eq!(PrinterError::Timeout.to_string(), "printer timeout");
        assert_eq!(
            PrinterError::ApiError("bad request".into()).to_string(),
            "API error: bad request"
        );
        assert_eq!(
            PrinterError::NotConnected.to_string(),
            "printer not connected"
        );
    }
}

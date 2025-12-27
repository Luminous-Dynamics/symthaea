/*!
 * Trace format types and loading
 */

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use uuid::Uuid;

/// Complete execution trace for a Symthaea session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Trace format version
    pub version: String,

    /// Unique session identifier
    pub session_id: String,

    /// Session start time
    pub timestamp_start: String,

    /// Session end time (if session completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_end: Option<String>,

    /// Chronological list of events
    pub events: Vec<Event>,

    /// Session summary statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SessionSummary>,
}

impl Trace {
    /// Load trace from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .context("Failed to read trace file")?;

        serde_json::from_str(&content)
            .context("Failed to parse trace JSON")
    }

    /// Save trace to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)
            .context("Failed to write trace file")
    }

    /// Create a new empty trace
    pub fn new(session_id: String) -> Self {
        Self {
            version: "1.0".to_string(),
            session_id,
            timestamp_start: Utc::now().to_rfc3339(),
            timestamp_end: None,
            events: Vec::new(),
            summary: None,
        }
    }

    /// Add an event to the trace
    pub fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Finalize the trace (compute summary, set end time)
    pub fn finalize(&mut self) {
        self.timestamp_end = Some(Utc::now().to_rfc3339());
        self.summary = Some(self.compute_summary());
    }

    /// Compute summary statistics
    fn compute_summary(&self) -> SessionSummary {
        let mut router_distribution = HashMap::new();
        let mut phi_measurements = Vec::new();
        let mut ignition_count = 0;
        let mut errors = 0;
        let mut security_denials = 0;

        for event in &self.events {
            match &event.event_type {
                EventType::RouterSelection => {
                    if let Some(data) = &event.data {
                        if let Ok(router_data) = serde_json::from_value::<RouterSelectionData>(data.clone()) {
                            *router_distribution.entry(router_data.selected_router).or_insert(0) += 1;
                        }
                    }
                }
                EventType::WorkspaceIgnition => {
                    ignition_count += 1;
                    if let Some(data) = &event.data {
                        if let Ok(ws_data) = serde_json::from_value::<WorkspaceIgnitionData>(data.clone()) {
                            phi_measurements.push(ws_data.phi);
                        }
                    }
                }
                EventType::ErrorOccurred => errors += 1,
                EventType::SecurityCheck => {
                    if let Some(data) = &event.data {
                        if let Ok(sec_data) = serde_json::from_value::<SecurityCheckData>(data.clone()) {
                            if sec_data.decision == "denied" {
                                security_denials += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let average_phi = if !phi_measurements.is_empty() {
            phi_measurements.iter().sum::<f64>() / phi_measurements.len() as f64
        } else {
            0.0
        };

        let max_phi = phi_measurements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_phi = phi_measurements.iter().cloned().fold(f64::INFINITY, f64::min);

        let duration_ms = if let (Ok(start), Some(end_str)) = (
            DateTime::parse_from_rfc3339(&self.timestamp_start),
            &self.timestamp_end,
        ) {
            if let Ok(end) = DateTime::parse_from_rfc3339(end_str) {
                (end - start).num_milliseconds() as u64
            } else {
                0
            }
        } else {
            0
        };

        SessionSummary {
            total_events: self.events.len(),
            average_phi,
            max_phi: if max_phi.is_finite() { max_phi } else { 0.0 },
            min_phi: if min_phi.is_finite() { min_phi } else { 0.0 },
            router_distribution,
            ignition_count,
            duration_ms,
            errors,
            security_denials,
        }
    }
}

/// A single event in the trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event timestamp
    pub timestamp: String,

    /// Event type
    #[serde(rename = "type")]
    pub event_type: EventType,

    /// Type-specific event data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Event type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    RouterSelection,
    WorkspaceIgnition,
    PrimitiveActivation,
    PhiMeasurement,
    ResponseGenerated,
    ErrorOccurred,
    SecurityCheck,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RouterSelection => write!(f, "Router Selection"),
            Self::WorkspaceIgnition => write!(f, "Workspace Ignition"),
            Self::PrimitiveActivation => write!(f, "Primitive Activation"),
            Self::PhiMeasurement => write!(f, "Φ Measurement"),
            Self::ResponseGenerated => write!(f, "Response Generated"),
            Self::ErrorOccurred => write!(f, "Error Occurred"),
            Self::SecurityCheck => write!(f, "Security Check"),
        }
    }
}

/// Router selection event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterSelectionData {
    pub input: String,
    pub selected_router: String,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alternatives: Option<Vec<RouterAlternative>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandit_stats: Option<HashMap<String, BanditStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterAlternative {
    pub router: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditStats {
    pub count: u64,
    pub reward: f64,
}

/// Workspace ignition event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceIgnitionData {
    pub phi: f64,
    pub free_energy: f64,
    pub coalition_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_primitives: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub broadcast_payload_size: Option<usize>,
}

/// Response generation event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseData {
    pub content: String,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_verified: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_confirmation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
}

/// Security check event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCheckData {
    pub operation: String,
    pub decision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secrets_redacted: Option<usize>,
}

/// Session summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub total_events: usize,
    pub average_phi: f64,
    pub max_phi: f64,
    pub min_phi: f64,
    pub router_distribution: HashMap<String, usize>,
    pub ignition_count: usize,
    pub duration_ms: u64,
    pub errors: usize,
    pub security_denials: usize,
}

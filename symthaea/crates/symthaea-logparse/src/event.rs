//! Normalized log event. All sources (Evtx, syslog, SNMP traps, future) flatten
//! into this shape before encoding. Fields are deliberately conservative — the
//! encoder reads only what's here, not source-specific extensions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Severity {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Source {
    WindowsEvent,
    Syslog,
    Snmp,
    Other,
}

/// A normalized log event. Fields are ordered by encoder-importance, not
/// chronologically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    pub timestamp: DateTime<Utc>,
    pub source: Source,
    pub severity: Severity,
    /// Component/service name (e.g. "Service Control Manager", "sshd", "kernel").
    pub component: String,
    /// Provider/facility identifier (Evtx provider GUID, syslog facility).
    pub provider: String,
    /// Numeric event id (Evtx EventID, syslog msgid). 0 if unknown.
    pub event_id: u32,
    /// Human-readable message body. May be empty.
    pub message: String,
    /// Structured key-value pairs extracted from the event (Evtx EventData,
    /// syslog structured-data). BTreeMap so iteration is deterministic for
    /// reproducible encoding.
    pub fields: BTreeMap<String, String>,
    /// Host/machine that emitted the event. Used for per-host cluster
    /// attribution, not for encoding.
    pub host: Option<String>,
    /// Optional ground-truth label for purity evaluation on benchmark corpora.
    /// None in production ingestion.
    pub label: Option<String>,
}

impl LogEvent {
    /// Cheap fingerprint for deduplication before encoding. NOT a hash for
    /// integrity — the clustering pipeline discards duplicates by this key.
    pub fn dedupe_key(&self) -> String {
        format!(
            "{:?}|{}|{}|{}",
            self.source, self.provider, self.event_id, self.component
        )
    }
}

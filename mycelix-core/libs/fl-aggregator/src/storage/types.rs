//! Storage backend types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Backend type enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    /// Local file-based storage.
    LocalFile,
    /// PostgreSQL database.
    PostgreSQL,
    /// Holochain DHT.
    Holochain,
    /// Blockchain storage.
    Blockchain,
    /// IPFS distributed storage.
    Ipfs,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::LocalFile => write!(f, "localfile"),
            BackendType::PostgreSQL => write!(f, "postgresql"),
            BackendType::Holochain => write!(f, "holochain"),
            BackendType::Blockchain => write!(f, "blockchain"),
            BackendType::Ipfs => write!(f, "ipfs"),
        }
    }
}

/// Gradient record stored in backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientRecord {
    /// Unique identifier.
    pub id: String,
    /// Node that submitted the gradient.
    pub node_id: String,
    /// Training round number.
    pub round_num: u64,
    /// Gradient data (may be encrypted base64 or raw floats as JSON).
    pub gradient: GradientData,
    /// SHA-256 hash of gradient data.
    pub gradient_hash: String,
    /// PoGQ (Proof of Gradient Quality) score.
    pub pogq_score: Option<f32>,
    /// Whether ZKPoC verification passed.
    pub zkpoc_verified: bool,
    /// Whether validation passed.
    pub validation_passed: bool,
    /// Node's reputation score at submission time.
    pub reputation_score: f32,
    /// Submission timestamp (Unix epoch seconds).
    pub timestamp: f64,
    /// Whether gradient is encrypted.
    pub encrypted: bool,
    /// Backend-specific metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GradientRecord {
    /// Create a new gradient record.
    pub fn new(
        node_id: impl Into<String>,
        round_num: u64,
        gradient: Vec<f32>,
        gradient_hash: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            node_id: node_id.into(),
            round_num,
            gradient: GradientData::Raw(gradient),
            gradient_hash: gradient_hash.into(),
            pogq_score: None,
            zkpoc_verified: false,
            validation_passed: true,
            reputation_score: 0.5,
            timestamp: current_timestamp(),
            encrypted: false,
            metadata: HashMap::new(),
        }
    }

    /// Create with encrypted gradient.
    pub fn new_encrypted(
        node_id: impl Into<String>,
        round_num: u64,
        encrypted_data: impl Into<String>,
        gradient_hash: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            node_id: node_id.into(),
            round_num,
            gradient: GradientData::Encrypted(encrypted_data.into()),
            gradient_hash: gradient_hash.into(),
            pogq_score: None,
            zkpoc_verified: false,
            validation_passed: true,
            reputation_score: 0.5,
            timestamp: current_timestamp(),
            encrypted: true,
            metadata: HashMap::new(),
        }
    }

    /// Set PoGQ score.
    pub fn with_pogq(mut self, score: f32) -> Self {
        self.pogq_score = Some(score);
        self
    }

    /// Set validation status.
    pub fn with_validation(mut self, passed: bool) -> Self {
        self.validation_passed = passed;
        self
    }

    /// Set reputation score.
    pub fn with_reputation(mut self, score: f32) -> Self {
        self.reputation_score = score;
        self
    }

    /// Set custom ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set ZKPoC (Zero-Knowledge Proof of Contribution) verification status.
    ///
    /// This flag indicates whether the node's K-Vector trust metrics have been
    /// cryptographically verified using a STARK proof without revealing the
    /// actual values.
    pub fn with_zkpoc_verified(mut self, verified: bool) -> Self {
        self.zkpoc_verified = verified;
        self
    }
}

/// Gradient data representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GradientData {
    /// Raw float values.
    Raw(Vec<f32>),
    /// Encrypted base64 string.
    Encrypted(String),
}

impl GradientData {
    /// Check if data is encrypted.
    pub fn is_encrypted(&self) -> bool {
        matches!(self, GradientData::Encrypted(_))
    }

    /// Get raw values if not encrypted.
    pub fn as_raw(&self) -> Option<&Vec<f32>> {
        match self {
            GradientData::Raw(v) => Some(v),
            GradientData::Encrypted(_) => None,
        }
    }

    /// Get encrypted string if encrypted.
    pub fn as_encrypted(&self) -> Option<&str> {
        match self {
            GradientData::Raw(_) => None,
            GradientData::Encrypted(s) => Some(s),
        }
    }
}

/// Credit transaction record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditRecord {
    /// Transaction ID.
    pub transaction_id: String,
    /// Credit holder (node ID).
    pub holder: String,
    /// Credit amount.
    pub amount: u64,
    /// Reason for earning credit.
    pub earned_from: String,
    /// Transaction timestamp.
    pub timestamp: f64,
    /// Backend-specific metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CreditRecord {
    /// Create a new credit record.
    pub fn new(holder: impl Into<String>, amount: u64, earned_from: impl Into<String>) -> Self {
        Self {
            transaction_id: uuid_v4(),
            holder: holder.into(),
            amount,
            earned_from: earned_from.into(),
            timestamp: current_timestamp(),
            metadata: HashMap::new(),
        }
    }
}

/// Byzantine detection event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByzantineEvent {
    /// Event ID.
    pub event_id: String,
    /// Node that exhibited Byzantine behavior.
    pub node_id: String,
    /// Round when detected.
    pub round_num: u64,
    /// Detection method used.
    pub detection_method: String,
    /// Severity level.
    pub severity: ByzantineSeverity,
    /// Additional details.
    pub details: serde_json::Value,
    /// Detection timestamp.
    pub timestamp: f64,
}

impl ByzantineEvent {
    /// Create a new Byzantine event.
    pub fn new(
        node_id: impl Into<String>,
        round_num: u64,
        detection_method: impl Into<String>,
        severity: ByzantineSeverity,
    ) -> Self {
        Self {
            event_id: uuid_v4(),
            node_id: node_id.into(),
            round_num,
            detection_method: detection_method.into(),
            severity,
            details: serde_json::Value::Null,
            timestamp: current_timestamp(),
        }
    }

    /// Add details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = details;
        self
    }
}

/// Byzantine event severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ByzantineSeverity {
    /// Low severity (warning).
    Low,
    /// Medium severity.
    Medium,
    /// High severity.
    High,
    /// Critical (immediate action required).
    Critical,
}

impl std::fmt::Display for ByzantineSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ByzantineSeverity::Low => write!(f, "low"),
            ByzantineSeverity::Medium => write!(f, "medium"),
            ByzantineSeverity::High => write!(f, "high"),
            ByzantineSeverity::Critical => write!(f, "critical"),
        }
    }
}

/// Node reputation data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationData {
    /// Node ID.
    pub node_id: String,
    /// Reputation score (0.0 - 1.0).
    pub score: f32,
    /// Total gradients submitted.
    pub gradients_submitted: u64,
    /// Gradients accepted.
    pub gradients_accepted: u64,
    /// Byzantine events count.
    pub byzantine_events: u64,
    /// Total credits earned.
    pub total_credits: u64,
}

/// Backend health status.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Whether backend is healthy.
    pub healthy: bool,
    /// Response latency in milliseconds.
    pub latency_ms: Option<f64>,
    /// Whether storage is available.
    pub storage_available: bool,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            healthy: false,
            latency_ms: None,
            storage_available: false,
            metadata: HashMap::new(),
        }
    }
}

/// Backend statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendStats {
    /// Backend type.
    pub backend_type: BackendType,
    /// Total gradients stored.
    pub total_gradients: u64,
    /// Total credits issued.
    pub total_credits_issued: u64,
    /// Total Byzantine events.
    pub total_byzantine_events: u64,
    /// Storage size in bytes.
    pub storage_size_bytes: u64,
    /// Uptime in seconds.
    pub uptime_seconds: f64,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

// Helper functions

/// Generate UUID v4.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // Simple UUID-like string (not cryptographically secure, but unique enough)
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (timestamp >> 96) as u32,
        (timestamp >> 80) as u16,
        (timestamp >> 64) as u16 & 0x0FFF,
        ((timestamp >> 48) as u16 & 0x3FFF) | 0x8000,
        timestamp as u64 & 0xFFFFFFFFFFFF
    )
}

/// Get current Unix timestamp.
fn current_timestamp() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_record_creation() {
        let record = GradientRecord::new("node_1", 1, vec![1.0, 2.0, 3.0], "hash123");

        assert_eq!(record.node_id, "node_1");
        assert_eq!(record.round_num, 1);
        assert!(!record.encrypted);
        assert!(record.validation_passed);
    }

    #[test]
    fn test_gradient_record_encrypted() {
        let record = GradientRecord::new_encrypted("node_1", 1, "base64data", "hash123");

        assert!(record.encrypted);
        assert!(record.gradient.is_encrypted());
    }

    #[test]
    fn test_credit_record() {
        let record = CreditRecord::new("node_1", 100, "gradient_quality");

        assert_eq!(record.holder, "node_1");
        assert_eq!(record.amount, 100);
        assert_eq!(record.earned_from, "gradient_quality");
    }

    #[test]
    fn test_byzantine_event() {
        let event = ByzantineEvent::new("node_1", 5, "pogq", ByzantineSeverity::High)
            .with_details(serde_json::json!({"score": 0.1}));

        assert_eq!(event.node_id, "node_1");
        assert_eq!(event.round_num, 5);
        assert_eq!(event.severity, ByzantineSeverity::High);
    }

    #[test]
    fn test_uuid_uniqueness() {
        let id1 = uuid_v4();
        let id2 = uuid_v4();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::LocalFile.to_string(), "localfile");
        assert_eq!(BackendType::PostgreSQL.to_string(), "postgresql");
    }
}

//! Types for Holochain integration.

use serde::{Deserialize, Serialize};

/// Holochain entry hash (action hash).
pub type EntryHash = String;

/// Holochain agent public key.
pub type AgentPubKey = String;

/// Gradient record for storage in Holochain DHT.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientRecord {
    /// Node identifier.
    pub node_id: String,

    /// Training round number.
    pub round_num: u64,

    /// Gradient data (base64 encoded).
    pub gradient_data: String,

    /// Shape of gradient tensor.
    pub gradient_shape: Vec<usize>,

    /// Data type (e.g., "float32").
    pub gradient_dtype: String,

    /// Node's reputation score.
    pub reputation_score: f32,

    /// Whether gradient passed validation.
    pub validation_passed: bool,

    /// PoGQ (Proof of Gradient Quality) score.
    pub pogq_score: Option<f32>,

    /// Whether anomaly was detected.
    pub anomaly_detected: bool,

    /// Whether node is blacklisted.
    pub blacklisted: bool,
}

impl GradientRecord {
    /// Create a new gradient record from raw data.
    pub fn new(
        node_id: impl Into<String>,
        round_num: u64,
        gradient: &[f32],
        reputation_score: f32,
    ) -> Self {
        // Encode gradient as JSON then base64
        let gradient_json = serde_json::to_string(gradient).unwrap_or_default();
        let gradient_data = base64::engine::general_purpose::STANDARD.encode(gradient_json.as_bytes());

        Self {
            node_id: node_id.into(),
            round_num,
            gradient_data,
            gradient_shape: vec![gradient.len()],
            gradient_dtype: "float32".to_string(),
            reputation_score,
            validation_passed: true,
            pogq_score: None,
            anomaly_detected: false,
            blacklisted: false,
        }
    }

    /// Set PoGQ score.
    pub fn with_pogq(mut self, score: f32) -> Self {
        self.pogq_score = Some(score);
        self
    }

    /// Mark as anomaly.
    pub fn with_anomaly(mut self) -> Self {
        self.anomaly_detected = true;
        self.validation_passed = false;
        self
    }

    /// Decode gradient data back to float array.
    pub fn decode_gradient(&self) -> Result<Vec<f32>, HolochainError> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.gradient_data)
            .map_err(|e| HolochainError::Encoding(e.to_string()))?;

        let json_str = String::from_utf8(bytes)
            .map_err(|e| HolochainError::Encoding(e.to_string()))?;

        serde_json::from_str(&json_str)
            .map_err(|e| HolochainError::Encoding(e.to_string()))
    }
}

/// Credit record for the credit system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditRecord {
    /// Agent receiving credit.
    pub holder: AgentPubKey,

    /// Credit amount.
    pub amount: u64,

    /// Reason for earning credit.
    pub earned_from: EarnReason,

    /// Verifiers who approved this credit.
    pub verifiers: Vec<AgentPubKey>,

    /// Timestamp.
    pub timestamp: u64,
}

/// Reason for earning credit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum EarnReason {
    /// Credit for quality gradient submission.
    QualityGradient {
        pogq_score: f32,
        gradient_hash: String,
    },
    /// Credit for detecting Byzantine behavior.
    ByzantineDetection {
        caught_node_id: String,
        evidence_hash: String,
    },
    /// Credit for peer validation.
    PeerValidation {
        validated_node_id: String,
        gradient_hash: String,
    },
    /// Credit for network contribution.
    NetworkContribution {
        uptime_hours: u64,
    },
}

/// Byzantine event record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByzantineEvent {
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

    /// Timestamp.
    pub timestamp: u64,
}

/// Byzantine event severity.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ByzantineSeverity {
    /// Low severity (warning).
    Low,
    /// Medium severity.
    Medium,
    /// High severity (immediate action).
    High,
    /// Critical (blacklist node).
    Critical,
}

/// Signal from Holochain DHT.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "signal_type", content = "payload")]
pub enum HolochainSignal {
    /// New gradient submitted.
    GradientSubmitted {
        node_id: String,
        round_num: u64,
        entry_hash: EntryHash,
    },
    /// Reputation updated.
    ReputationUpdated {
        node_id: String,
        old_score: f32,
        new_score: f32,
        reason: String,
    },
    /// Round event.
    RoundEvent {
        round_num: u64,
        action: RoundAction,
        participants: Vec<String>,
    },
    /// Byzantine behavior detected.
    ByzantineDetected {
        node_ids: Vec<String>,
        detection_method: String,
        round_num: u64,
    },
    /// Ethereum payment request from Bridge zome.
    EthereumPaymentRequest {
        intent_id: String,
        model_id: String,
        round: u64,
        total_amount_wei: String,
        splits: Vec<PaymentSplitSignal>,
    },
    /// Ethereum anchor request from Bridge zome.
    EthereumAnchorRequest {
        intent_id: String,
        anchor_type: String,
        agent_id: String,
        score_bps: u64,
        round: u64,
        evidence_hash: Option<String>,
    },
}

/// Payment split for Ethereum distribution signal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaymentSplitSignal {
    /// Ethereum address of recipient.
    pub address: String,
    /// Basis points (10000 = 100%).
    pub basis_points: u64,
    /// Agent ID in Holochain.
    pub agent_id: String,
}

/// Round action type.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RoundAction {
    /// Round started.
    Start,
    /// Round ended.
    End,
    /// Aggregation triggered.
    Aggregate,
}

/// Holochain-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum HolochainError {
    /// Connection error.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Zome call error.
    #[error("Zome error ({zome}.{function}): {message}")]
    ZomeCall {
        zome: String,
        function: String,
        message: String,
    },

    /// Encoding/decoding error.
    #[error("Encoding error: {0}")]
    Encoding(String),

    /// Timeout error.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Not connected.
    #[error("Not connected to Holochain conductor")]
    NotConnected,

    /// Cell ID not found.
    #[error("Cell ID not configured")]
    NoCellId,

    /// WebSocket error.
    #[error("WebSocket error: {0}")]
    WebSocket(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for Holochain operations.
pub type HolochainResult<T> = Result<T, HolochainError>;

/// Response from Holochain admin interface.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdminResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub value: serde_json::Value,
}

/// Response from zome call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZomeResponse {
    pub data: Option<serde_json::Value>,
    pub error: Option<serde_json::Value>,
}

use base64::Engine;

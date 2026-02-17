//! Audit Logging System for Proof Operations
//!
//! Provides tamper-evident audit trails for all proof operations including:
//! - Proof generation events
//! - Verification events
//! - Access patterns
//! - Error conditions
//!
//! Features:
//! - Cryptographic chaining (hash chain)
//! - Structured logging with rich metadata
//! - Retention policies
//! - Export capabilities (JSON, CSV)
//! - Search and filter operations

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::proofs::{ProofType, SecurityLevel};

// ============================================================================
// Audit Event Types
// ============================================================================

/// Type of audit event
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Proof generation started
    GenerationStarted,
    /// Proof generation completed successfully
    GenerationCompleted,
    /// Proof generation failed
    GenerationFailed,
    /// Proof verification started
    VerificationStarted,
    /// Proof verification completed
    VerificationCompleted,
    /// Proof verification failed
    VerificationFailed,
    /// Proof accessed (read from cache/storage)
    ProofAccessed,
    /// Proof exported
    ProofExported,
    /// Proof revoked
    ProofRevoked,
    /// Configuration changed
    ConfigChanged,
    /// Rate limit hit
    RateLimitHit,
    /// Security event (suspicious activity)
    SecurityEvent,
    /// System event (startup, shutdown, etc.)
    SystemEvent,
}

impl AuditEventType {
    /// Get severity level (0=info, 1=warning, 2=error, 3=critical)
    pub fn severity(&self) -> u8 {
        match self {
            AuditEventType::GenerationStarted
            | AuditEventType::GenerationCompleted
            | AuditEventType::VerificationStarted
            | AuditEventType::VerificationCompleted
            | AuditEventType::ProofAccessed
            | AuditEventType::ProofExported
            | AuditEventType::ConfigChanged
            | AuditEventType::SystemEvent => 0, // Info
            AuditEventType::RateLimitHit | AuditEventType::ProofRevoked => 1, // Warning
            AuditEventType::GenerationFailed | AuditEventType::VerificationFailed => 2, // Error
            AuditEventType::SecurityEvent => 3, // Critical
        }
    }
}

/// Audit event entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: String,
    /// Event sequence number
    pub sequence: u64,
    /// Event type
    pub event_type: AuditEventType,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Associated proof type (if applicable)
    pub proof_type: Option<ProofType>,
    /// Security level used (if applicable)
    pub security_level: Option<SecurityLevel>,
    /// Request/operation ID
    pub request_id: Option<String>,
    /// Actor (user/service) that triggered the event
    pub actor: Option<String>,
    /// Client IP address (if applicable)
    pub client_ip: Option<String>,
    /// Duration in microseconds (for completed operations)
    pub duration_us: Option<u64>,
    /// Whether the operation was successful
    pub success: Option<bool>,
    /// Error message (if applicable)
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Hash of previous event (for chain integrity)
    pub prev_hash: [u8; 32],
    /// Hash of this event
    pub event_hash: [u8; 32],
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(event_type: AuditEventType, prev_hash: [u8; 32]) -> Self {
        let id = generate_event_id();
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut event = Self {
            id,
            sequence: 0,
            event_type,
            timestamp_ms,
            proof_type: None,
            security_level: None,
            request_id: None,
            actor: None,
            client_ip: None,
            duration_us: None,
            success: None,
            error_message: None,
            metadata: HashMap::new(),
            prev_hash,
            event_hash: [0u8; 32],
        };

        event.event_hash = event.compute_hash();
        event
    }

    /// Builder method: set proof type
    pub fn with_proof_type(mut self, proof_type: ProofType) -> Self {
        self.proof_type = Some(proof_type);
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set security level
    pub fn with_security_level(mut self, level: SecurityLevel) -> Self {
        self.security_level = Some(level);
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set request ID
    pub fn with_request_id(mut self, request_id: &str) -> Self {
        self.request_id = Some(request_id.to_string());
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set actor
    pub fn with_actor(mut self, actor: &str) -> Self {
        self.actor = Some(actor.to_string());
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set client IP
    pub fn with_client_ip(mut self, ip: &str) -> Self {
        self.client_ip = Some(ip.to_string());
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_us = Some(duration.as_micros() as u64);
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set success flag
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: set error message
    pub fn with_error(mut self, error: &str) -> Self {
        self.error_message = Some(error.to_string());
        self.success = Some(false);
        self.event_hash = self.compute_hash();
        self
    }

    /// Builder method: add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self.event_hash = self.compute_hash();
        self
    }

    /// Compute the hash of this event
    fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();

        hasher.update(self.id.as_bytes());
        hasher.update(&self.sequence.to_le_bytes());
        hasher.update(&(self.event_type as u32).to_le_bytes());
        hasher.update(&self.timestamp_ms.to_le_bytes());

        if let Some(pt) = &self.proof_type {
            hasher.update(&(*pt as u32).to_le_bytes());
        }

        if let Some(req_id) = &self.request_id {
            hasher.update(req_id.as_bytes());
        }

        if let Some(actor) = &self.actor {
            hasher.update(actor.as_bytes());
        }

        if let Some(err) = &self.error_message {
            hasher.update(err.as_bytes());
        }

        hasher.update(&self.prev_hash);

        *hasher.finalize().as_bytes()
    }

    /// Verify the integrity of this event
    pub fn verify_integrity(&self) -> bool {
        self.event_hash == self.compute_hash()
    }
}

/// Generate a unique event ID
fn generate_event_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);

    format!("audit-{}-{:04x}", timestamp, counter & 0xFFFF)
}

// ============================================================================
// Audit Log Configuration
// ============================================================================

/// Configuration for audit logging
#[derive(Clone, Debug)]
pub struct AuditConfig {
    /// Maximum number of events to retain in memory
    pub max_memory_events: usize,
    /// Retention period for events
    pub retention_period: Duration,
    /// Event types to include (empty = all)
    pub include_types: Vec<AuditEventType>,
    /// Event types to exclude
    pub exclude_types: Vec<AuditEventType>,
    /// Minimum severity level to log (0-3)
    pub min_severity: u8,
    /// Whether to log sensitive metadata
    pub log_sensitive: bool,
    /// Whether to verify chain on read
    pub verify_on_read: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            max_memory_events: 100_000,
            retention_period: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            include_types: Vec::new(),
            exclude_types: Vec::new(),
            min_severity: 0,
            log_sensitive: false,
            verify_on_read: true,
        }
    }
}

// ============================================================================
// Audit Log
// ============================================================================

/// Audit log manager
pub struct AuditLog {
    config: AuditConfig,
    events: Arc<RwLock<Vec<AuditEvent>>>,
    sequence: Arc<RwLock<u64>>,
    last_hash: Arc<RwLock<[u8; 32]>>,
    stats: Arc<RwLock<AuditStats>>,
}

/// Statistics for audit log
#[derive(Clone, Debug, Default)]
pub struct AuditStats {
    /// Total events logged
    pub total_events: u64,
    /// Events by type
    pub events_by_type: HashMap<AuditEventType, u64>,
    /// Events by severity
    pub events_by_severity: [u64; 4],
    /// Chain integrity verified
    pub chain_verified: bool,
    /// Last verification timestamp
    pub last_verification_ms: u64,
}

impl AuditLog {
    /// Create a new audit log
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            events: Arc::new(RwLock::new(Vec::new())),
            sequence: Arc::new(RwLock::new(0)),
            last_hash: Arc::new(RwLock::new([0u8; 32])),
            stats: Arc::new(RwLock::new(AuditStats::default())),
        }
    }

    /// Log an event
    pub async fn log(&self, mut event: AuditEvent) {
        // Check if event type should be logged
        if !self.should_log(&event) {
            return;
        }

        let mut sequence = self.sequence.write().await;
        let mut last_hash = self.last_hash.write().await;
        let mut events = self.events.write().await;
        let mut stats = self.stats.write().await;

        // Set sequence and previous hash
        event.sequence = *sequence;
        event.prev_hash = *last_hash;
        event.event_hash = event.compute_hash();

        // Update last hash and sequence
        *last_hash = event.event_hash;
        *sequence += 1;

        // Update statistics
        stats.total_events += 1;
        *stats
            .events_by_type
            .entry(event.event_type)
            .or_insert(0) += 1;
        let severity = event.event_type.severity() as usize;
        if severity < 4 {
            stats.events_by_severity[severity] += 1;
        }

        // Add event
        events.push(event);

        // Enforce memory limit
        let event_count = events.len();
        if event_count > self.config.max_memory_events {
            let drain_count = event_count - self.config.max_memory_events;
            events.drain(0..drain_count);
        }
    }

    /// Check if event should be logged
    fn should_log(&self, event: &AuditEvent) -> bool {
        // Check severity
        if event.event_type.severity() < self.config.min_severity {
            return false;
        }

        // Check exclude types
        if self.config.exclude_types.contains(&event.event_type) {
            return false;
        }

        // Check include types (if specified)
        if !self.config.include_types.is_empty()
            && !self.config.include_types.contains(&event.event_type)
        {
            return false;
        }

        true
    }

    /// Get all events
    pub async fn get_events(&self) -> Vec<AuditEvent> {
        self.events.read().await.clone()
    }

    /// Get events with filter
    pub async fn query(&self, filter: &AuditFilter) -> Vec<AuditEvent> {
        let events = self.events.read().await;

        events
            .iter()
            .filter(|e| filter.matches(e))
            .cloned()
            .collect()
    }

    /// Get statistics
    pub async fn stats(&self) -> AuditStats {
        self.stats.read().await.clone()
    }

    /// Verify chain integrity
    pub async fn verify_chain(&self) -> ChainVerificationResult {
        let events = self.events.read().await;

        if events.is_empty() {
            return ChainVerificationResult {
                valid: true,
                events_verified: 0,
                first_invalid_sequence: None,
                error_message: None,
            };
        }

        let mut expected_prev_hash = [0u8; 32];

        for (i, event) in events.iter().enumerate() {
            // Verify previous hash chain
            if event.prev_hash != expected_prev_hash {
                return ChainVerificationResult {
                    valid: false,
                    events_verified: i,
                    first_invalid_sequence: Some(event.sequence),
                    error_message: Some("Previous hash mismatch".to_string()),
                };
            }

            // Verify event hash
            if !event.verify_integrity() {
                return ChainVerificationResult {
                    valid: false,
                    events_verified: i,
                    first_invalid_sequence: Some(event.sequence),
                    error_message: Some("Event hash mismatch".to_string()),
                };
            }

            expected_prev_hash = event.event_hash;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.chain_verified = true;
        stats.last_verification_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        ChainVerificationResult {
            valid: true,
            events_verified: events.len(),
            first_invalid_sequence: None,
            error_message: None,
        }
    }

    /// Export events to JSON
    pub async fn export_json(&self, filter: Option<&AuditFilter>) -> String {
        let events = match filter {
            Some(f) => self.query(f).await,
            None => self.get_events().await,
        };

        serde_json::to_string_pretty(&events).unwrap_or_default()
    }

    /// Export events to CSV
    pub async fn export_csv(&self, filter: Option<&AuditFilter>) -> String {
        let events = match filter {
            Some(f) => self.query(f).await,
            None => self.get_events().await,
        };

        let mut csv = String::new();
        csv.push_str("id,sequence,event_type,timestamp_ms,proof_type,request_id,actor,success,duration_us,error_message\n");

        for event in events {
            csv.push_str(&format!(
                "{},{},{:?},{},{},{},{},{},{},{}\n",
                event.id,
                event.sequence,
                event.event_type,
                event.timestamp_ms,
                event
                    .proof_type
                    .map(|pt| format!("{:?}", pt))
                    .unwrap_or_default(),
                event.request_id.unwrap_or_default(),
                event.actor.unwrap_or_default(),
                event.success.map(|s| s.to_string()).unwrap_or_default(),
                event.duration_us.map(|d| d.to_string()).unwrap_or_default(),
                event.error_message.unwrap_or_default().replace(',', ";"),
            ));
        }

        csv
    }

    /// Clear old events based on retention policy
    pub async fn cleanup(&self) {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - self.config.retention_period.as_millis() as u64;

        let mut events = self.events.write().await;
        events.retain(|e| e.timestamp_ms >= cutoff);
    }
}

/// Chain verification result
#[derive(Clone, Debug)]
pub struct ChainVerificationResult {
    /// Whether the chain is valid
    pub valid: bool,
    /// Number of events verified
    pub events_verified: usize,
    /// First invalid sequence number (if any)
    pub first_invalid_sequence: Option<u64>,
    /// Error message (if any)
    pub error_message: Option<String>,
}

// ============================================================================
// Audit Filter
// ============================================================================

/// Filter for querying audit events
#[derive(Clone, Debug, Default)]
pub struct AuditFilter {
    /// Event types to include
    pub event_types: Option<Vec<AuditEventType>>,
    /// Proof types to include
    pub proof_types: Option<Vec<ProofType>>,
    /// Start timestamp (inclusive)
    pub start_time_ms: Option<u64>,
    /// End timestamp (exclusive)
    pub end_time_ms: Option<u64>,
    /// Actor filter
    pub actor: Option<String>,
    /// Request ID filter
    pub request_id: Option<String>,
    /// Success filter
    pub success: Option<bool>,
    /// Minimum severity
    pub min_severity: Option<u8>,
    /// Text search in metadata
    pub text_search: Option<String>,
}

impl AuditFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by event types
    pub fn event_types(mut self, types: Vec<AuditEventType>) -> Self {
        self.event_types = Some(types);
        self
    }

    /// Filter by proof types
    pub fn proof_types(mut self, types: Vec<ProofType>) -> Self {
        self.proof_types = Some(types);
        self
    }

    /// Filter by time range
    pub fn time_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.start_time_ms = Some(start_ms);
        self.end_time_ms = Some(end_ms);
        self
    }

    /// Filter by actor
    pub fn actor(mut self, actor: &str) -> Self {
        self.actor = Some(actor.to_string());
        self
    }

    /// Filter by request ID
    pub fn request_id(mut self, request_id: &str) -> Self {
        self.request_id = Some(request_id.to_string());
        self
    }

    /// Filter by success status
    pub fn success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }

    /// Filter by minimum severity
    pub fn min_severity(mut self, severity: u8) -> Self {
        self.min_severity = Some(severity);
        self
    }

    /// Check if event matches filter
    pub fn matches(&self, event: &AuditEvent) -> bool {
        // Check event type
        if let Some(types) = &self.event_types {
            if !types.contains(&event.event_type) {
                return false;
            }
        }

        // Check proof type
        if let Some(types) = &self.proof_types {
            match &event.proof_type {
                Some(pt) => {
                    if !types.contains(pt) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Check time range
        if let Some(start) = self.start_time_ms {
            if event.timestamp_ms < start {
                return false;
            }
        }
        if let Some(end) = self.end_time_ms {
            if event.timestamp_ms >= end {
                return false;
            }
        }

        // Check actor
        if let Some(actor) = &self.actor {
            match &event.actor {
                Some(e) => {
                    if e != actor {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Check request ID
        if let Some(request_id) = &self.request_id {
            match &event.request_id {
                Some(e) => {
                    if e != request_id {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Check success
        if let Some(success) = self.success {
            match event.success {
                Some(s) => {
                    if s != success {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Check severity
        if let Some(min_sev) = self.min_severity {
            if event.event_type.severity() < min_sev {
                return false;
            }
        }

        // Check text search
        if let Some(text) = &self.text_search {
            let text_lower = text.to_lowercase();
            let has_match = event.id.to_lowercase().contains(&text_lower)
                || event
                    .request_id
                    .as_ref()
                    .map(|r| r.to_lowercase().contains(&text_lower))
                    .unwrap_or(false)
                || event
                    .actor
                    .as_ref()
                    .map(|a| a.to_lowercase().contains(&text_lower))
                    .unwrap_or(false)
                || event
                    .error_message
                    .as_ref()
                    .map(|e| e.to_lowercase().contains(&text_lower))
                    .unwrap_or(false)
                || event
                    .metadata
                    .values()
                    .any(|v| v.to_lowercase().contains(&text_lower));

            if !has_match {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Audit Logger Trait
// ============================================================================

/// Trait for types that can log audit events
#[async_trait::async_trait]
pub trait AuditLogger: Send + Sync {
    /// Log a proof generation started event
    async fn log_generation_started(
        &self,
        proof_type: ProofType,
        request_id: &str,
        actor: Option<&str>,
    );

    /// Log a proof generation completed event
    async fn log_generation_completed(
        &self,
        proof_type: ProofType,
        request_id: &str,
        duration: Duration,
        proof_size: usize,
    );

    /// Log a proof generation failed event
    async fn log_generation_failed(
        &self,
        proof_type: ProofType,
        request_id: &str,
        error: &str,
    );

    /// Log a verification event
    async fn log_verification(
        &self,
        proof_type: ProofType,
        request_id: &str,
        valid: bool,
        duration: Duration,
    );
}

#[async_trait::async_trait]
impl AuditLogger for AuditLog {
    async fn log_generation_started(
        &self,
        proof_type: ProofType,
        request_id: &str,
        actor: Option<&str>,
    ) {
        let mut event = AuditEvent::new(AuditEventType::GenerationStarted, [0u8; 32])
            .with_proof_type(proof_type)
            .with_request_id(request_id);

        if let Some(a) = actor {
            event = event.with_actor(a);
        }

        self.log(event).await;
    }

    async fn log_generation_completed(
        &self,
        proof_type: ProofType,
        request_id: &str,
        duration: Duration,
        proof_size: usize,
    ) {
        let event = AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
            .with_proof_type(proof_type)
            .with_request_id(request_id)
            .with_duration(duration)
            .with_success(true)
            .with_metadata("proof_size", &proof_size.to_string());

        self.log(event).await;
    }

    async fn log_generation_failed(
        &self,
        proof_type: ProofType,
        request_id: &str,
        error: &str,
    ) {
        let event = AuditEvent::new(AuditEventType::GenerationFailed, [0u8; 32])
            .with_proof_type(proof_type)
            .with_request_id(request_id)
            .with_error(error);

        self.log(event).await;
    }

    async fn log_verification(
        &self,
        proof_type: ProofType,
        request_id: &str,
        valid: bool,
        duration: Duration,
    ) {
        let event = AuditEvent::new(
            if valid {
                AuditEventType::VerificationCompleted
            } else {
                AuditEventType::VerificationFailed
            },
            [0u8; 32],
        )
        .with_proof_type(proof_type)
        .with_request_id(request_id)
        .with_duration(duration)
        .with_success(valid);

        self.log(event).await;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_severity() {
        assert_eq!(AuditEventType::GenerationStarted.severity(), 0);
        assert_eq!(AuditEventType::RateLimitHit.severity(), 1);
        assert_eq!(AuditEventType::GenerationFailed.severity(), 2);
        assert_eq!(AuditEventType::SecurityEvent.severity(), 3);
    }

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(AuditEventType::GenerationStarted, [0u8; 32])
            .with_proof_type(ProofType::Range)
            .with_request_id("test-123")
            .with_actor("user@example.com");

        assert_eq!(event.event_type, AuditEventType::GenerationStarted);
        assert_eq!(event.proof_type, Some(ProofType::Range));
        assert_eq!(event.request_id, Some("test-123".to_string()));
        assert_eq!(event.actor, Some("user@example.com".to_string()));
    }

    #[test]
    fn test_event_integrity() {
        let event = AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
            .with_proof_type(ProofType::GradientIntegrity)
            .with_success(true);

        assert!(event.verify_integrity());
    }

    #[test]
    fn test_event_id_uniqueness() {
        let id1 = generate_event_id();
        let id2 = generate_event_id();
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_audit_log_basic() {
        let log = AuditLog::new(AuditConfig::default());

        let event = AuditEvent::new(AuditEventType::GenerationStarted, [0u8; 32])
            .with_proof_type(ProofType::Range);

        log.log(event).await;

        let events = log.get_events().await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 0);
    }

    #[tokio::test]
    async fn test_audit_chain_integrity() {
        let log = AuditLog::new(AuditConfig::default());

        // Log multiple events
        for i in 0..5 {
            let event = AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
                .with_request_id(&format!("req-{}", i));
            log.log(event).await;
        }

        // Verify chain
        let result = log.verify_chain().await;
        assert!(result.valid);
        assert_eq!(result.events_verified, 5);
    }

    #[tokio::test]
    async fn test_audit_filter_event_type() {
        let log = AuditLog::new(AuditConfig::default());

        log.log(
            AuditEvent::new(AuditEventType::GenerationStarted, [0u8; 32])
                .with_proof_type(ProofType::Range),
        )
        .await;
        log.log(
            AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
                .with_proof_type(ProofType::Range),
        )
        .await;
        log.log(
            AuditEvent::new(AuditEventType::VerificationCompleted, [0u8; 32])
                .with_proof_type(ProofType::Range),
        )
        .await;

        let filter = AuditFilter::new().event_types(vec![AuditEventType::GenerationCompleted]);
        let results = log.query(&filter).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event_type, AuditEventType::GenerationCompleted);
    }

    #[tokio::test]
    async fn test_audit_filter_success() {
        let log = AuditLog::new(AuditConfig::default());

        log.log(
            AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32]).with_success(true),
        )
        .await;
        log.log(
            AuditEvent::new(AuditEventType::GenerationFailed, [0u8; 32]).with_success(false),
        )
        .await;

        let filter = AuditFilter::new().success(false);
        let results = log.query(&filter).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].success, Some(false));
    }

    #[tokio::test]
    async fn test_audit_stats() {
        let log = AuditLog::new(AuditConfig::default());

        for _ in 0..3 {
            log.log(AuditEvent::new(
                AuditEventType::GenerationCompleted,
                [0u8; 32],
            ))
            .await;
        }
        for _ in 0..2 {
            log.log(AuditEvent::new(AuditEventType::GenerationFailed, [0u8; 32])).await;
        }

        let stats = log.stats().await;
        assert_eq!(stats.total_events, 5);
        assert_eq!(
            *stats
                .events_by_type
                .get(&AuditEventType::GenerationCompleted)
                .unwrap(),
            3
        );
        assert_eq!(
            *stats
                .events_by_type
                .get(&AuditEventType::GenerationFailed)
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn test_audit_export_json() {
        let log = AuditLog::new(AuditConfig::default());

        log.log(
            AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
                .with_proof_type(ProofType::Range)
                .with_request_id("test"),
        )
        .await;

        let json = log.export_json(None).await;
        assert!(json.contains("GenerationCompleted"));
        assert!(json.contains("test"));
    }

    #[tokio::test]
    async fn test_audit_export_csv() {
        let log = AuditLog::new(AuditConfig::default());

        log.log(
            AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
                .with_proof_type(ProofType::Range)
                .with_request_id("test-csv"),
        )
        .await;

        let csv = log.export_csv(None).await;
        assert!(csv.contains("test-csv"));
        assert!(csv.contains("GenerationCompleted"));
    }

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();
        assert_eq!(config.max_memory_events, 100_000);
        assert_eq!(config.min_severity, 0);
        assert!(config.verify_on_read);
    }

    #[tokio::test]
    async fn test_audit_logger_trait() {
        let log = AuditLog::new(AuditConfig::default());

        log.log_generation_started(ProofType::Range, "req-1", Some("alice"))
            .await;
        log.log_generation_completed(ProofType::Range, "req-1", Duration::from_millis(50), 15000)
            .await;

        let events = log.get_events().await;
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_filter_matches() {
        let event = AuditEvent::new(AuditEventType::GenerationCompleted, [0u8; 32])
            .with_proof_type(ProofType::Range)
            .with_request_id("req-123")
            .with_actor("alice")
            .with_success(true);

        // Should match
        assert!(AuditFilter::new().matches(&event));
        assert!(AuditFilter::new()
            .event_types(vec![AuditEventType::GenerationCompleted])
            .matches(&event));
        assert!(AuditFilter::new()
            .proof_types(vec![ProofType::Range])
            .matches(&event));
        assert!(AuditFilter::new().success(true).matches(&event));
        assert!(AuditFilter::new().actor("alice").matches(&event));

        // Should not match
        assert!(!AuditFilter::new()
            .event_types(vec![AuditEventType::GenerationFailed])
            .matches(&event));
        assert!(!AuditFilter::new().success(false).matches(&event));
        assert!(!AuditFilter::new().actor("bob").matches(&event));
    }
}

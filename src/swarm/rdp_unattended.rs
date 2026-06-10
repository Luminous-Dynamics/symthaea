// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unattended access daemon for Sovereign RDP.
//!
//! Accepts incoming RDP connections without user interaction on the remote end.
//! Identity-verified: only MFDI-approved identities can connect.
//! Time-bounded: sessions auto-expire.
//! Fully audit-logged.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Default maximum unattended session duration (2 hours).
const DEFAULT_MAX_DURATION_SECS: u64 = 7200;

/// Configuration for the unattended access daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnattendedConfig {
    /// MFDI identity DIDs allowed to connect without user approval.
    pub allowed_identities: Vec<String>,
    /// Maximum session duration before auto-disconnect.
    pub max_session_duration_secs: u64,
    /// Minimum MFDI assurance level required (0-4, maps to E0-E4).
    pub min_assurance_level: u8,
    /// Whether to require consciousness attestation.
    pub require_consciousness: bool,
    /// Minimum phi for session (if consciousness required).
    pub min_phi: f32,
}

impl Default for UnattendedConfig {
    fn default() -> Self {
        Self {
            allowed_identities: Vec::new(),
            max_session_duration_secs: DEFAULT_MAX_DURATION_SECS,
            min_assurance_level: 2, // E2 Verified minimum
            require_consciousness: true,
            min_phi: 0.1,
        }
    }
}

/// An active unattended session.
#[derive(Debug, Clone)]
pub struct UnattendedSession {
    /// RDP session ID.
    pub session_id: String,
    /// Connected agent's DID.
    pub agent_did: String,
    /// Session start time.
    pub started_at: Instant,
    /// Maximum duration.
    pub max_duration: Duration,
}

/// Audit log entry for unattended access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp (Unix seconds).
    pub timestamp: u64,
    /// Event type.
    pub event: AuditEvent,
    /// Agent DID (if applicable).
    pub agent_did: Option<String>,
    /// Session ID (if applicable).
    pub session_id: Option<String>,
}

/// Audit event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEvent {
    /// Connection attempt (may succeed or fail).
    ConnectionAttempt { allowed: bool, reason: String },
    /// Session started.
    SessionStarted,
    /// Session ended (reason: timeout, disconnect, kicked).
    SessionEnded { reason: String, duration_secs: u64 },
    /// Configuration changed.
    ConfigChanged { field: String },
}

/// Unattended access daemon.
pub struct UnattendedDaemon {
    config: UnattendedConfig,
    active_sessions: HashMap<String, UnattendedSession>,
    audit_log: Vec<AuditEntry>,
}

impl UnattendedDaemon {
    pub fn new(config: UnattendedConfig) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
            audit_log: Vec::new(),
        }
    }

    /// Check if an agent is allowed to connect.
    pub fn authorize(&mut self, agent_did: &str, assurance_level: u8, phi: f32) -> bool {
        // Check identity allowlist.
        let identity_ok = self
            .config
            .allowed_identities
            .contains(&agent_did.to_string());

        // Check assurance level.
        let assurance_ok = assurance_level >= self.config.min_assurance_level;

        // Check consciousness.
        let consciousness_ok = !self.config.require_consciousness || phi >= self.config.min_phi;

        let allowed = identity_ok && assurance_ok && consciousness_ok;

        let reason = if !identity_ok {
            "Identity not in allowlist".to_string()
        } else if !assurance_ok {
            format!(
                "Assurance level {} below minimum {}",
                assurance_level, self.config.min_assurance_level
            )
        } else if !consciousness_ok {
            format!("Phi {:.2} below minimum {:.2}", phi, self.config.min_phi)
        } else {
            "Authorized".to_string()
        };

        self.log(AuditEntry {
            timestamp: now_secs(),
            event: AuditEvent::ConnectionAttempt { allowed, reason },
            agent_did: Some(agent_did.to_string()),
            session_id: None,
        });

        allowed
    }

    /// Register an active session.
    pub fn start_session(&mut self, session_id: &str, agent_did: &str) {
        let session = UnattendedSession {
            session_id: session_id.to_string(),
            agent_did: agent_did.to_string(),
            started_at: Instant::now(),
            max_duration: Duration::from_secs(self.config.max_session_duration_secs),
        };

        self.active_sessions.insert(session_id.to_string(), session);

        self.log(AuditEntry {
            timestamp: now_secs(),
            event: AuditEvent::SessionStarted,
            agent_did: Some(agent_did.to_string()),
            session_id: Some(session_id.to_string()),
        });
    }

    /// End a session.
    pub fn end_session(&mut self, session_id: &str, reason: &str) {
        if let Some(session) = self.active_sessions.remove(session_id) {
            self.log(AuditEntry {
                timestamp: now_secs(),
                event: AuditEvent::SessionEnded {
                    reason: reason.to_string(),
                    duration_secs: session.started_at.elapsed().as_secs(),
                },
                agent_did: Some(session.agent_did),
                session_id: Some(session_id.to_string()),
            });
        }
    }

    /// Tick: check for expired sessions. Returns session IDs to disconnect.
    pub fn tick(&mut self) -> Vec<String> {
        let expired: Vec<String> = self
            .active_sessions
            .iter()
            .filter(|(_, s)| s.started_at.elapsed() > s.max_duration)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &expired {
            self.end_session(id, "session timeout");
        }

        expired
    }

    /// Add an identity to the allowlist.
    pub fn add_allowed_identity(&mut self, did: &str) {
        if !self.config.allowed_identities.contains(&did.to_string()) {
            self.config.allowed_identities.push(did.to_string());
            self.log(AuditEntry {
                timestamp: now_secs(),
                event: AuditEvent::ConfigChanged {
                    field: format!("allowed_identities += {}", did),
                },
                agent_did: None,
                session_id: None,
            });
        }
    }

    /// Remove an identity from the allowlist.
    pub fn remove_allowed_identity(&mut self, did: &str) {
        self.config.allowed_identities.retain(|d| d != did);
        self.log(AuditEntry {
            timestamp: now_secs(),
            event: AuditEvent::ConfigChanged {
                field: format!("allowed_identities -= {}", did),
            },
            agent_did: None,
            session_id: None,
        });
    }

    /// Number of active sessions.
    pub fn active_count(&self) -> usize {
        self.active_sessions.len()
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Serialize audit log as JSON.
    pub fn audit_log_json(&self) -> String {
        serde_json::to_string_pretty(&self.audit_log).unwrap_or_else(|_| "[]".to_string())
    }

    fn log(&mut self, entry: AuditEntry) {
        self.audit_log.push(entry);
        // Cap at 10,000 entries.
        if self.audit_log.len() > 10_000 {
            self.audit_log.drain(..1_000);
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> UnattendedConfig {
        UnattendedConfig {
            allowed_identities: vec!["did:mycelix:agent1".to_string()],
            max_session_duration_secs: 5, // Short for testing
            min_assurance_level: 2,
            require_consciousness: true,
            min_phi: 0.1,
        }
    }

    #[test]
    fn test_authorize_allowed() {
        let mut daemon = UnattendedDaemon::new(test_config());
        assert!(daemon.authorize("did:mycelix:agent1", 2, 0.5));
    }

    #[test]
    fn test_authorize_unknown_identity() {
        let mut daemon = UnattendedDaemon::new(test_config());
        assert!(!daemon.authorize("did:mycelix:stranger", 4, 0.9));
    }

    #[test]
    fn test_authorize_low_assurance() {
        let mut daemon = UnattendedDaemon::new(test_config());
        assert!(!daemon.authorize("did:mycelix:agent1", 1, 0.5)); // E1 < E2
    }

    #[test]
    fn test_authorize_low_phi() {
        let mut daemon = UnattendedDaemon::new(test_config());
        assert!(!daemon.authorize("did:mycelix:agent1", 2, 0.05)); // Below 0.1
    }

    #[test]
    fn test_session_lifecycle() {
        let mut daemon = UnattendedDaemon::new(test_config());
        daemon.start_session("rdp-1", "did:mycelix:agent1");
        assert_eq!(daemon.active_count(), 1);

        daemon.end_session("rdp-1", "user disconnect");
        assert_eq!(daemon.active_count(), 0);

        // Audit log should have: connection attempt, start, end = at least 2.
        assert!(daemon.audit_log().len() >= 2);
    }

    #[test]
    fn test_audit_log_entries() {
        let mut daemon = UnattendedDaemon::new(test_config());
        daemon.authorize("did:mycelix:agent1", 2, 0.5);

        let log = daemon.audit_log();
        assert_eq!(log.len(), 1);
        match &log[0].event {
            AuditEvent::ConnectionAttempt { allowed, .. } => assert!(allowed),
            _ => panic!("Expected ConnectionAttempt"),
        }
    }

    #[test]
    fn test_add_remove_identity() {
        let mut daemon = UnattendedDaemon::new(test_config());

        assert!(!daemon.authorize("did:mycelix:agent2", 3, 0.5));

        daemon.add_allowed_identity("did:mycelix:agent2");
        assert!(daemon.authorize("did:mycelix:agent2", 3, 0.5));

        daemon.remove_allowed_identity("did:mycelix:agent2");
        assert!(!daemon.authorize("did:mycelix:agent2", 3, 0.5));
    }
}

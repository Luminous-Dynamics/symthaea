// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sub-Passport -- Morally-constrained AI agent delegation.
//!
//! Allows a human identity holder to issue a constrained credential
//! to an AI agent (like Symthaea), specifying:
//!
//! - Maximum action type (Read, Write, CargoTest, etc.)
//! - Maximum moral severity for actions affecting others
//! - Trust tier ceiling (agent starts low, can earn up)
//! - Time-to-live (short-lived, must be renewed)
//!
//! ## Security Model
//!
//! A sub-passport is NOT an API key. It is a graduated credential:
//! - The agent starts at the delegator-specified tier (typically Participant)
//! - Actions are filtered through the moral algebra before execution
//! - The delegator's Ahimsa gate constraints are inherited
//! - Restorative justice applies: violations are tracked, trust can be earned back
//!
//! ## Differences from Web3 delegation
//!
//! | Property | API Key | Sub-Passport |
//! |----------|---------|--------------|
//! | Granularity | Binary (yes/no) | Graduated (tier + severity) |
//! | Moral constraints | None | Ahimsa gate inherited |
//! | Trust evolution | Static | Earnable via compliance |
//! | Accountability | Audit log | Restorative justice tracker |
//! | Renewal | Permanent until revoked | Short-lived, must be renewed |

use serde::{Deserialize, Serialize};

use crate::consciousness_profile::ConsciousnessTier;

/// Maximum moral severity for actions affecting others.
/// Actions above this threshold are blocked by the moral filter.
pub const DEFAULT_MAX_MORAL_SEVERITY: f32 = 0.3;

/// Default sub-passport TTL: 24 hours in microseconds.
pub const DEFAULT_TTL_US: u64 = 24 * 3600 * 1_000_000;

/// Action types that can be permitted via sub-passport.
/// Ordered by privilege level (lower = safer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DelegatedActionType {
    /// Read-only operations (file read, directory listing).
    Read = 0,
    /// Parse and analyze (no side effects).
    Analyze = 1,
    /// Build and test (cargo check, cargo test).
    Build = 2,
    /// Write files (code generation, edits).
    Write = 3,
    /// Version control (git commit).
    VersionControl = 4,
    /// Execute commands (shell execution).
    Execute = 5,
    /// Network operations (API calls, mesh communication).
    Network = 6,
    /// Governance participation (voting, proposals).
    Govern = 7,
}

/// A sub-passport issued to an AI agent by a human delegator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubPassport {
    /// DID of the AI agent receiving delegation.
    pub agent_did: String,
    /// DID of the human delegator.
    pub delegator_did: String,
    /// Maximum trust tier the agent can operate at.
    pub max_tier: ConsciousnessTier,
    /// Current effective tier (starts at max_tier, can be degraded by violations).
    pub effective_tier: ConsciousnessTier,
    /// Maximum action type permitted.
    pub max_action: DelegatedActionType,
    /// Maximum moral severity for actions affecting others.
    /// Actions above this are blocked by the moral filter.
    pub max_moral_severity: f32,
    /// Whether the Ahimsa (non-harm) gate is enforced.
    /// When true, ALL actions that could cause harm are blocked regardless of severity.
    pub ahimsa_enforced: bool,
    /// Issuance timestamp (microseconds).
    pub issued_at: u64,
    /// Expiry timestamp (microseconds).
    pub expires_at: u64,
    /// Whether the sub-passport has been revoked.
    pub revoked: bool,
    /// When the sub-passport was revoked (microseconds). None if not revoked.
    #[serde(default)]
    pub revoked_at: Option<u64>,
    /// Human-readable reason for revocation. None if not revoked.
    #[serde(default)]
    pub revoked_reason: Option<String>,
    /// Number of moral violations recorded.
    pub violation_count: u32,
    /// Number of corrective actions recorded.
    pub correction_count: u32,
    /// Human-readable purpose/scope description.
    pub purpose: String,
    /// Delegator signature binding this passport to the delegator's key.
    /// Empty if unsigned (local use only).
    #[serde(default)]
    pub delegator_signature: Vec<u8>,
}

impl SubPassport {
    /// Create a new sub-passport with default constraints.
    pub fn new(agent_did: String, delegator_did: String, purpose: String, now_us: u64) -> Self {
        Self {
            agent_did,
            delegator_did,
            max_tier: ConsciousnessTier::Participant,
            effective_tier: ConsciousnessTier::Participant,
            max_action: DelegatedActionType::Read,
            max_moral_severity: DEFAULT_MAX_MORAL_SEVERITY,
            ahimsa_enforced: true,
            issued_at: now_us,
            expires_at: now_us + DEFAULT_TTL_US,
            revoked: false,
            revoked_at: None,
            revoked_reason: None,
            violation_count: 0,
            correction_count: 0,
            purpose,
            delegator_signature: Vec::new(),
        }
    }

    /// Create with specific constraints.
    #[allow(clippy::too_many_arguments)]
    pub fn with_constraints(
        agent_did: String,
        delegator_did: String,
        purpose: String,
        max_tier: ConsciousnessTier,
        max_action: DelegatedActionType,
        max_moral_severity: f32,
        ttl_hours: u64,
        now_us: u64,
    ) -> Self {
        Self {
            agent_did,
            delegator_did,
            max_tier,
            effective_tier: max_tier,
            max_action,
            max_moral_severity: max_moral_severity.clamp(0.0, 1.0),
            ahimsa_enforced: true,
            issued_at: now_us,
            expires_at: now_us + ttl_hours * 3600 * 1_000_000,
            revoked: false,
            revoked_at: None,
            revoked_reason: None,
            violation_count: 0,
            correction_count: 0,
            purpose,
            delegator_signature: Vec::new(),
        }
    }

    /// Check if an action is permitted by this sub-passport.
    pub fn permits_action(&self, action: DelegatedActionType, now_us: u64) -> bool {
        if !self.is_valid(now_us) {
            return false;
        }
        action <= self.max_action
    }

    /// Check if a moral severity is within bounds.
    pub fn permits_severity(&self, severity: f32) -> bool {
        severity <= self.max_moral_severity
    }

    /// Whether the sub-passport is currently valid.
    pub fn is_valid(&self, now_us: u64) -> bool {
        !self.revoked && now_us < self.expires_at
    }

    /// Revoke the sub-passport with a reason and timestamp.
    pub fn revoke(&mut self, reason: impl Into<String>, now_us: u64) {
        self.revoked = true;
        self.revoked_at = Some(now_us);
        self.revoked_reason = Some(reason.into());
    }

    /// Record a moral violation. Degrades effective tier after threshold.
    pub fn record_violation(&mut self) {
        self.violation_count = self.violation_count.saturating_add(1);
        // Every 3 violations, degrade effective tier by 1
        if self.violation_count % 3 == 0 {
            self.effective_tier = self.effective_tier.degrade(1);
        }
    }

    /// Record a corrective action (restorative justice).
    pub fn record_correction(&mut self) {
        self.correction_count = self.correction_count.saturating_add(1);
        // Every 10 corrections can restore one tier level (up to max_tier)
        if self.correction_count % 10 == 0 {
            self.effective_tier = self.effective_tier.upgrade_capped(self.max_tier);
        }
    }

    /// Remaining time in hours.
    pub fn hours_remaining(&self, now_us: u64) -> f64 {
        if now_us >= self.expires_at {
            return 0.0;
        }
        (self.expires_at - now_us) as f64 / (3600.0 * 1_000_000.0)
    }

    /// Compliance ratio: corrections / (violations + corrections).
    pub fn compliance_ratio(&self) -> f64 {
        let total = self.violation_count + self.correction_count;
        if total == 0 {
            return 1.0; // Perfect compliance (no events)
        }
        self.correction_count as f64 / total as f64
    }

    /// Deterministic serialization of signable fields.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(self.agent_did.as_bytes());
        buf.extend_from_slice(self.delegator_did.as_bytes());
        buf.push(self.max_tier as u8);
        buf.push(self.max_action as u8);
        buf.extend_from_slice(&self.max_moral_severity.to_le_bytes());
        buf.extend_from_slice(&self.expires_at.to_le_bytes());
        buf
    }

    /// Sign with BLAKE3 keyed hash. Always available (no feature gate).
    pub fn sign_blake3(&mut self, key: &[u8; 32]) {
        let data = self.canonical_bytes();
        let mac = blake3::keyed_hash(key, &data);
        self.delegator_signature = mac.as_bytes().to_vec();
    }

    /// Verify BLAKE3 keyed hash signature.
    pub fn verify_blake3(&self, key: &[u8; 32]) -> bool {
        let data = self.canonical_bytes();
        let mac = blake3::keyed_hash(key, &data);
        self.delegator_signature == mac.as_bytes().as_slice()
    }

    /// Whether this passport has been signed.
    pub fn is_signed(&self) -> bool {
        !self.delegator_signature.is_empty()
    }

    /// Sign with Ed25519 (requires `identity` feature).
    #[cfg(feature = "identity")]
    pub fn sign_ed25519(&mut self, key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        let data = self.canonical_bytes();
        let sig = key.sign(&data);
        self.delegator_signature = sig.to_bytes().to_vec();
    }

    /// Verify Ed25519 signature (requires `identity` feature).
    #[cfg(feature = "identity")]
    pub fn verify_ed25519(&self, pubkey: &ed25519_dalek::VerifyingKey) -> bool {
        use ed25519_dalek::Verifier;
        if self.delegator_signature.len() != 64 {
            return false;
        }
        let sig_bytes: [u8; 64] = match self.delegator_signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
        let data = self.canonical_bytes();
        pubkey.verify(&data, &sig).is_ok()
    }

    /// Renew the sub-passport with a new TTL.
    /// Only possible if compliance ratio is above threshold.
    pub fn renew(&mut self, now_us: u64, ttl_hours: u64) -> bool {
        if self.compliance_ratio() < 0.5 {
            return false; // Too many violations
        }
        self.issued_at = now_us;
        self.expires_at = now_us + ttl_hours * 3600 * 1_000_000;
        self.revoked = false;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        1_000_000_000_000 // arbitrary timestamp
    }

    #[test]
    fn default_sub_passport() {
        let sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "code review".into(),
            now(),
        );

        assert_eq!(sp.max_tier, ConsciousnessTier::Participant);
        assert_eq!(sp.max_action, DelegatedActionType::Read);
        assert!(sp.ahimsa_enforced);
        assert!(sp.is_valid(now()));
        assert!(!sp.is_valid(now() + DEFAULT_TTL_US + 1));
    }

    #[test]
    fn action_permission() {
        let sp = SubPassport::with_constraints(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "development".into(),
            ConsciousnessTier::Citizen,
            DelegatedActionType::Build,
            0.3,
            24,
            now(),
        );

        assert!(sp.permits_action(DelegatedActionType::Read, now()));
        assert!(sp.permits_action(DelegatedActionType::Analyze, now()));
        assert!(sp.permits_action(DelegatedActionType::Build, now()));
        assert!(!sp.permits_action(DelegatedActionType::Write, now()));
        assert!(!sp.permits_action(DelegatedActionType::Execute, now()));
    }

    #[test]
    fn moral_severity_gating() {
        let sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "review".into(),
            now(),
        );

        assert!(sp.permits_severity(0.1)); // Low severity OK
        assert!(sp.permits_severity(0.3)); // At threshold OK
        assert!(!sp.permits_severity(0.5)); // Above threshold blocked
    }

    #[test]
    fn violation_degrades_tier() {
        let mut sp = SubPassport::with_constraints(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            ConsciousnessTier::Citizen,
            DelegatedActionType::Build,
            0.3,
            24,
            now(),
        );

        assert_eq!(sp.effective_tier, ConsciousnessTier::Citizen);

        // 3 violations -> degrade by 1
        sp.record_violation();
        sp.record_violation();
        sp.record_violation();
        assert_eq!(sp.effective_tier, ConsciousnessTier::Participant);

        // 3 more -> degrade again
        sp.record_violation();
        sp.record_violation();
        sp.record_violation();
        assert_eq!(sp.effective_tier, ConsciousnessTier::Observer);
    }

    #[test]
    fn correction_restores_tier() {
        let mut sp = SubPassport::with_constraints(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            ConsciousnessTier::Citizen,
            DelegatedActionType::Build,
            0.3,
            24,
            now(),
        );

        // Degrade to Participant
        for _ in 0..3 {
            sp.record_violation();
        }
        assert_eq!(sp.effective_tier, ConsciousnessTier::Participant);

        // 10 corrections -> restore one level
        for _ in 0..10 {
            sp.record_correction();
        }
        assert_eq!(sp.effective_tier, ConsciousnessTier::Citizen);
    }

    #[test]
    fn cannot_upgrade_past_max() {
        let mut sp = SubPassport::with_constraints(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            ConsciousnessTier::Participant,
            DelegatedActionType::Read,
            0.3,
            24,
            now(),
        );

        // 10 corrections
        for _ in 0..10 {
            sp.record_correction();
        }
        // Should stay at Participant (max_tier)
        assert_eq!(sp.effective_tier, ConsciousnessTier::Participant);
    }

    #[test]
    fn revocation() {
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );

        assert!(sp.is_valid(now()));
        sp.revoke("test revocation", now());
        assert!(!sp.is_valid(now()));
        assert_eq!(sp.revoked_at, Some(now()));
        assert_eq!(sp.revoked_reason.as_deref(), Some("test revocation"));
        assert!(!sp.permits_action(DelegatedActionType::Read, now()));
    }

    #[test]
    fn renewal_requires_compliance() {
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );

        // Good compliance -> can renew
        for _ in 0..5 {
            sp.record_correction();
        }
        assert!(sp.renew(now() + DEFAULT_TTL_US, 24));

        // Bad compliance -> cannot renew
        let mut bad_sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        for _ in 0..10 {
            bad_sp.record_violation();
        }
        assert!(!bad_sp.renew(now() + DEFAULT_TTL_US, 24));
    }

    #[test]
    fn compliance_ratio() {
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );

        assert_eq!(sp.compliance_ratio(), 1.0); // No events = perfect

        sp.record_violation();
        sp.record_correction();
        sp.record_correction();
        // 2 corrections / 3 total = 0.667
        assert!((sp.compliance_ratio() - 0.667).abs() < 0.01);
    }

    #[test]
    fn hours_remaining() {
        let sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );

        let remaining = sp.hours_remaining(now());
        assert!((remaining - 24.0).abs() < 0.01);

        let remaining_half = sp.hours_remaining(now() + 12 * 3600 * 1_000_000);
        assert!((remaining_half - 12.0).abs() < 0.01);
    }

    #[test]
    fn unsigned_passport_not_signed() {
        let sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        assert!(!sp.is_signed());
    }

    #[test]
    fn canonical_bytes_deterministic() {
        let sp1 = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        let sp2 = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        assert_eq!(sp1.canonical_bytes(), sp2.canonical_bytes());
    }

    #[test]
    fn blake3_sign_verify() {
        let key = [42u8; 32];
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        sp.sign_blake3(&key);
        assert!(sp.is_signed());
        assert!(sp.verify_blake3(&key));
    }

    #[test]
    fn blake3_tamper_detection() {
        let key = [42u8; 32];
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        sp.sign_blake3(&key);
        // Tamper with max_action
        sp.max_action = DelegatedActionType::Execute;
        assert!(!sp.verify_blake3(&key));
    }

    #[cfg(feature = "identity")]
    #[test]
    fn ed25519_sign_verify() {
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        sp.sign_ed25519(&signing_key);
        assert!(sp.verify_ed25519(&verifying_key));
    }

    #[cfg(feature = "identity")]
    #[test]
    fn ed25519_tamper_detection() {
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let mut sp = SubPassport::new(
            "did:mycelix:agent".into(),
            "did:mycelix:human".into(),
            "test".into(),
            now(),
        );
        sp.sign_ed25519(&signing_key);
        sp.max_moral_severity = 0.9;
        assert!(!sp.verify_ed25519(&verifying_key));
    }

    #[test]
    fn upgrade_capped_levels() {
        assert_eq!(
            ConsciousnessTier::Observer.upgrade_capped(ConsciousnessTier::Guardian),
            ConsciousnessTier::Participant
        );
        assert_eq!(
            ConsciousnessTier::Participant.upgrade_capped(ConsciousnessTier::Participant),
            ConsciousnessTier::Participant
        );
        assert_eq!(
            ConsciousnessTier::Citizen.upgrade_capped(ConsciousnessTier::Citizen),
            ConsciousnessTier::Citizen
        );
    }
}

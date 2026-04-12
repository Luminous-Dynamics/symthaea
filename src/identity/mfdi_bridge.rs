// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MFDI Bridge
//!
//! Bridge between Symthaea cognitive loop and the Multi-Factor Delegated Identity system.
//! Provides signature verification, output signing, and assurance level gating.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Assurance levels aligned with Epistemic Charter v2.0
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum AssuranceLevel {
    /// E0: Anonymous - Read-only access
    #[default]
    E0Anonymous = 0,
    /// E1: Basic - Single factor verified (post, message)
    E1Basic = 1,
    /// E2: Verified - 3+ factors, 2+ categories (FL participation, proposals)
    E2Verified = 2,
    /// E3: Highly Assured - 5+ factors, 3+ categories (governance voting)
    E3HighlyAssured = 3,
    /// E4: Constitutional - All critical factors (constitutional amendments)
    E4Constitutional = 4,
}

impl AssuranceLevel {
    /// Minimum score for each level
    pub fn min_score(&self) -> f32 {
        match self {
            AssuranceLevel::E0Anonymous => 0.0,
            AssuranceLevel::E1Basic => 0.3,
            AssuranceLevel::E2Verified => 0.5,
            AssuranceLevel::E3HighlyAssured => 0.7,
            AssuranceLevel::E4Constitutional => 0.9,
        }
    }

    /// From composite identity score
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 {
            AssuranceLevel::E4Constitutional
        } else if score >= 0.7 {
            AssuranceLevel::E3HighlyAssured
        } else if score >= 0.5 {
            AssuranceLevel::E2Verified
        } else if score >= 0.3 {
            AssuranceLevel::E1Basic
        } else {
            AssuranceLevel::E0Anonymous
        }
    }

    /// Check if level meets required minimum
    pub fn meets_requirement(&self, required: AssuranceLevel) -> bool {
        *self >= required
    }
}

/// Cognitive capabilities gated by assurance level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveCapability {
    /// Basic inference (read-only)
    BasicInference,
    /// Full cognitive cycle processing
    FullCycle,
    /// Memory write access
    MemoryWrite,
    /// Goal modification
    GoalModification,
    /// Federated learning participation
    FederatedLearning,
    /// Swarm coordination
    SwarmCoordination,
    /// Model weight updates
    WeightUpdate,
    /// Soul/identity modification
    SoulModification,
}

impl CognitiveCapability {
    /// Minimum assurance level required for this capability
    pub fn required_level(&self) -> AssuranceLevel {
        match self {
            CognitiveCapability::BasicInference => AssuranceLevel::E0Anonymous,
            CognitiveCapability::FullCycle => AssuranceLevel::E1Basic,
            CognitiveCapability::MemoryWrite => AssuranceLevel::E1Basic,
            CognitiveCapability::GoalModification => AssuranceLevel::E2Verified,
            CognitiveCapability::FederatedLearning => AssuranceLevel::E2Verified,
            CognitiveCapability::SwarmCoordination => AssuranceLevel::E2Verified,
            CognitiveCapability::WeightUpdate => AssuranceLevel::E3HighlyAssured,
            CognitiveCapability::SoulModification => AssuranceLevel::E4Constitutional,
        }
    }
}

/// MFDI Identity for an agent
#[derive(Debug, Clone)]
pub struct MfdiIdentity {
    /// Agent's unique identifier (DID format: did:mycelix:...)
    pub agent_id: String,
    /// Ed25519 public key for signature verification
    pub public_key: VerifyingKey,
    /// Current assurance level
    pub assurance_level: AssuranceLevel,
    /// Composite identity score (0.0 - 1.0)
    pub identity_score: f32,
    /// Reputation score from MATL (0.0 - 1.0)
    pub reputation: f32,
    /// Factor count contributing to identity
    pub factor_count: u8,
    /// Category diversity (number of unique factor categories)
    pub category_diversity: u8,
    /// When identity was last verified
    pub last_verified: SystemTime,
    /// Identity metadata (factor types, etc.)
    pub metadata: HashMap<String, String>,
}

impl MfdiIdentity {
    /// Create a new MFDI identity
    pub fn new(
        agent_id: impl Into<String>,
        public_key: VerifyingKey,
        identity_score: f32,
        reputation: f32,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            public_key,
            assurance_level: AssuranceLevel::from_score(identity_score),
            identity_score,
            reputation,
            factor_count: 1,
            category_diversity: 1,
            last_verified: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Check if identity can perform a capability
    pub fn can_perform(&self, capability: CognitiveCapability) -> bool {
        self.assurance_level
            .meets_requirement(capability.required_level())
    }

    /// Verify a signature from this identity
    pub fn verify_signature(&self, message: &[u8], signature: &Signature) -> bool {
        self.public_key.verify(message, signature).is_ok()
    }

    /// Check if identity is fresh (verified within duration)
    pub fn is_fresh(&self, max_age: Duration) -> bool {
        self.last_verified
            .elapsed()
            .map(|e| e < max_age)
            .unwrap_or(false)
    }

    /// Get DID from public key
    pub fn derive_did(public_key: &VerifyingKey) -> String {
        let hash = Sha256::digest(public_key.as_bytes());
        format!("did:mycelix:{}", hex::encode(&hash[..16]))
    }
}

/// Signed request from an agent
#[derive(Debug, Clone)]
pub struct SignedRequest {
    /// The request payload
    pub payload: String,
    /// Ed25519 signature of the payload
    pub signature: Signature,
    /// Agent's public key
    pub public_key: VerifyingKey,
    /// Request timestamp
    pub timestamp: SystemTime,
    /// Nonce for replay protection
    pub nonce: u64,
}

impl SignedRequest {
    /// Create a new signed request
    pub fn new(payload: impl Into<String>, signing_key: &SigningKey) -> Self {
        let payload = payload.into();
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Create signing message: payload + nonce
        let mut message = payload.as_bytes().to_vec();
        message.extend_from_slice(&nonce.to_le_bytes());

        let signature = signing_key.sign(&message);

        Self {
            payload,
            signature,
            public_key: signing_key.verifying_key(),
            timestamp: SystemTime::now(),
            nonce,
        }
    }

    /// Verify the request signature
    pub fn verify(&self) -> bool {
        let mut message = self.payload.as_bytes().to_vec();
        message.extend_from_slice(&self.nonce.to_le_bytes());
        self.public_key.verify(&message, &self.signature).is_ok()
    }

    /// Get agent DID from public key
    pub fn agent_did(&self) -> String {
        MfdiIdentity::derive_did(&self.public_key)
    }
}

/// Signed output from the cognitive loop
#[derive(Debug, Clone)]
pub struct SignedOutput {
    /// Output vector (compressed CfC state)
    pub output: Vec<f32>,
    /// Ed25519 signature of the output hash
    pub signature: Signature,
    /// Agent DID that produced this output
    pub agent_id: String,
    /// Timestamp of output generation
    pub timestamp: SystemTime,
    /// Cycle ID for tracking
    pub cycle_id: u64,
}

impl SignedOutput {
    /// Create a new signed output
    pub fn new(
        output: Vec<f32>,
        signing_key: &SigningKey,
        agent_id: impl Into<String>,
        cycle_id: u64,
    ) -> Self {
        // Hash the output for signing
        let output_hash = Self::hash_output(&output, cycle_id);
        let signature = signing_key.sign(&output_hash);

        Self {
            output,
            signature,
            agent_id: agent_id.into(),
            timestamp: SystemTime::now(),
            cycle_id,
        }
    }

    /// Hash output vector for signing
    fn hash_output(output: &[f32], cycle_id: u64) -> Vec<u8> {
        let mut hasher = Sha256::new();
        for val in output {
            hasher.update(val.to_le_bytes());
        }
        hasher.update(cycle_id.to_le_bytes());
        hasher.finalize().to_vec()
    }

    /// Verify the output signature
    pub fn verify(&self, public_key: &VerifyingKey) -> bool {
        let output_hash = Self::hash_output(&self.output, self.cycle_id);
        public_key.verify(&output_hash, &self.signature).is_ok()
    }
}

/// Configuration for the MFDI bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfdiBridgeConfig {
    /// Require signed requests
    pub require_signatures: bool,
    /// Sign outputs
    pub sign_outputs: bool,
    /// Maximum identity age before refresh required
    pub max_identity_age_secs: u64,
    /// Minimum assurance level for processing
    pub min_assurance_level: AssuranceLevel,
    /// Enable capability gating
    pub capability_gating: bool,
    /// Nonce window for replay protection (seconds)
    pub nonce_window_secs: u64,
}

impl Default for MfdiBridgeConfig {
    fn default() -> Self {
        Self {
            require_signatures: false, // Disabled by default for backward compatibility
            sign_outputs: false,
            max_identity_age_secs: 3600, // 1 hour
            min_assurance_level: AssuranceLevel::E0Anonymous,
            capability_gating: true,
            nonce_window_secs: 300, // 5 minutes
        }
    }
}

/// MFDI Bridge for cognitive loop integration
#[derive(Debug)]
pub struct MfdiBridge {
    /// Configuration
    config: MfdiBridgeConfig,
    /// Agent's signing key (if local agent)
    signing_key: Option<SigningKey>,
    /// Current identity (if authenticated)
    current_identity: Option<MfdiIdentity>,
    /// Seen nonces for replay protection
    seen_nonces: HashMap<u64, SystemTime>,
    /// Cycle counter for output signing
    cycle_counter: u64,
}

impl MfdiBridge {
    /// Create a new MFDI bridge
    pub fn new(config: MfdiBridgeConfig) -> Self {
        Self {
            config,
            signing_key: None,
            current_identity: None,
            seen_nonces: HashMap::new(),
            cycle_counter: 0,
        }
    }

    /// Create with a local signing key
    pub fn with_signing_key(config: MfdiBridgeConfig, signing_key: SigningKey) -> Self {
        let public_key = signing_key.verifying_key();
        let agent_id = MfdiIdentity::derive_did(&public_key);

        // Create default identity for local agent
        let identity = MfdiIdentity::new(agent_id, public_key, 0.5, 1.0);

        Self {
            config,
            signing_key: Some(signing_key),
            current_identity: Some(identity),
            seen_nonces: HashMap::new(),
            cycle_counter: 0,
        }
    }

    /// Generate a new signing key
    pub fn generate_key() -> SigningKey {
        use rand::RngCore;
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        SigningKey::from_bytes(&seed)
    }

    /// Get current identity
    pub fn identity(&self) -> Option<&MfdiIdentity> {
        self.current_identity.as_ref()
    }

    /// Get agent ID
    pub fn agent_id(&self) -> Option<&str> {
        self.current_identity.as_ref().map(|i| i.agent_id.as_str())
    }

    /// Get current assurance level
    pub fn assurance_level(&self) -> AssuranceLevel {
        self.current_identity
            .as_ref()
            .map(|i| i.assurance_level)
            .unwrap_or(AssuranceLevel::E0Anonymous)
    }

    /// Set identity from external verification
    pub fn set_identity(&mut self, identity: MfdiIdentity) {
        self.current_identity = Some(identity);
    }

    /// Update identity score and recalculate assurance level
    pub fn update_score(&mut self, new_score: f32, new_reputation: f32) {
        if let Some(ref mut identity) = self.current_identity {
            identity.identity_score = new_score;
            identity.reputation = new_reputation;
            identity.assurance_level = AssuranceLevel::from_score(new_score);
            identity.last_verified = SystemTime::now();
        }
    }

    /// Check if a capability is allowed
    pub fn check_capability(&self, capability: CognitiveCapability) -> Result<(), MfdiError> {
        if !self.config.capability_gating {
            return Ok(());
        }

        let level = self.assurance_level();
        if level.meets_requirement(capability.required_level()) {
            Ok(())
        } else {
            Err(MfdiError::InsufficientAssurance {
                required: capability.required_level(),
                actual: level,
                capability,
            })
        }
    }

    /// Verify a signed request
    pub fn verify_request(&mut self, request: &SignedRequest) -> Result<(), MfdiError> {
        // Check signature
        if !request.verify() {
            return Err(MfdiError::InvalidSignature);
        }

        // Check replay protection
        let now = SystemTime::now();
        let window = Duration::from_secs(self.config.nonce_window_secs);

        // Clean old nonces
        self.seen_nonces
            .retain(|_, t| t.elapsed().unwrap_or(window) < window);

        // Check if nonce was seen
        if self.seen_nonces.contains_key(&request.nonce) {
            return Err(MfdiError::ReplayDetected);
        }

        // Check timestamp freshness
        if let Ok(age) = request.timestamp.elapsed() {
            if age > window {
                return Err(MfdiError::RequestExpired);
            }
        }

        // Store nonce
        self.seen_nonces.insert(request.nonce, now);

        Ok(())
    }

    /// Sign an output
    pub fn sign_output(&mut self, output: &[f32]) -> Result<SignedOutput, MfdiError> {
        let signing_key = self.signing_key.as_ref().ok_or(MfdiError::NoSigningKey)?;
        let agent_id = self.agent_id().ok_or(MfdiError::NoIdentity)?.to_string();

        self.cycle_counter += 1;
        Ok(SignedOutput::new(
            output.to_vec(),
            signing_key,
            agent_id,
            self.cycle_counter,
        ))
    }

    /// Create a signed request
    pub fn sign_request(&self, payload: impl Into<String>) -> Result<SignedRequest, MfdiError> {
        let signing_key = self.signing_key.as_ref().ok_or(MfdiError::NoSigningKey)?;
        Ok(SignedRequest::new(payload, signing_key))
    }

    /// Process input with identity validation
    pub fn process_input(&mut self, input: &str) -> Result<ProcessedInput, MfdiError> {
        // Check minimum assurance for processing
        let level = self.assurance_level();
        if !level.meets_requirement(self.config.min_assurance_level) {
            return Err(MfdiError::InsufficientAssurance {
                required: self.config.min_assurance_level,
                actual: level,
                capability: CognitiveCapability::FullCycle,
            });
        }

        Ok(ProcessedInput {
            content: input.to_string(),
            agent_id: self.agent_id().map(String::from),
            assurance_level: level,
            reputation: self
                .current_identity
                .as_ref()
                .map(|i| i.reputation)
                .unwrap_or(0.0),
        })
    }

    /// Get summary for logging/debugging
    pub fn summary(&self) -> MfdiBridgeSummary {
        MfdiBridgeSummary {
            has_identity: self.current_identity.is_some(),
            agent_id: self.agent_id().map(String::from),
            assurance_level: self.assurance_level(),
            has_signing_key: self.signing_key.is_some(),
            require_signatures: self.config.require_signatures,
            sign_outputs: self.config.sign_outputs,
            cycle_count: self.cycle_counter,
        }
    }
}

/// Processed input with identity context
#[derive(Debug, Clone)]
pub struct ProcessedInput {
    /// Original content
    pub content: String,
    /// Agent ID if authenticated
    pub agent_id: Option<String>,
    /// Assurance level
    pub assurance_level: AssuranceLevel,
    /// Reputation score
    pub reputation: f32,
}

/// Summary of bridge state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfdiBridgeSummary {
    pub has_identity: bool,
    pub agent_id: Option<String>,
    pub assurance_level: AssuranceLevel,
    pub has_signing_key: bool,
    pub require_signatures: bool,
    pub sign_outputs: bool,
    pub cycle_count: u64,
}

/// MFDI errors
#[derive(Debug, Clone)]
pub enum MfdiError {
    /// Signature verification failed
    InvalidSignature,
    /// Replay attack detected
    ReplayDetected,
    /// Request too old
    RequestExpired,
    /// No signing key available
    NoSigningKey,
    /// No identity set
    NoIdentity,
    /// Assurance level too low
    InsufficientAssurance {
        required: AssuranceLevel,
        actual: AssuranceLevel,
        capability: CognitiveCapability,
    },
    /// Identity expired
    IdentityExpired,
}

impl std::fmt::Display for MfdiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MfdiError::InvalidSignature => write!(f, "Invalid signature"),
            MfdiError::ReplayDetected => write!(f, "Replay attack detected"),
            MfdiError::RequestExpired => write!(f, "Request expired"),
            MfdiError::NoSigningKey => write!(f, "No signing key available"),
            MfdiError::NoIdentity => write!(f, "No identity set"),
            MfdiError::InsufficientAssurance {
                required,
                actual,
                capability,
            } => {
                write!(
                    f,
                    "Insufficient assurance for {:?}: required {:?}, got {:?}",
                    capability, required, actual
                )
            }
            MfdiError::IdentityExpired => write!(f, "Identity verification expired"),
        }
    }
}

impl std::error::Error for MfdiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assurance_level_from_score() {
        assert_eq!(AssuranceLevel::from_score(0.0), AssuranceLevel::E0Anonymous);
        assert_eq!(
            AssuranceLevel::from_score(0.29),
            AssuranceLevel::E0Anonymous
        );
        assert_eq!(AssuranceLevel::from_score(0.3), AssuranceLevel::E1Basic);
        assert_eq!(AssuranceLevel::from_score(0.5), AssuranceLevel::E2Verified);
        assert_eq!(
            AssuranceLevel::from_score(0.7),
            AssuranceLevel::E3HighlyAssured
        );
        assert_eq!(
            AssuranceLevel::from_score(0.9),
            AssuranceLevel::E4Constitutional
        );
        assert_eq!(
            AssuranceLevel::from_score(1.0),
            AssuranceLevel::E4Constitutional
        );
    }

    #[test]
    fn test_assurance_level_ordering() {
        assert!(AssuranceLevel::E4Constitutional > AssuranceLevel::E3HighlyAssured);
        assert!(AssuranceLevel::E3HighlyAssured > AssuranceLevel::E2Verified);
        assert!(AssuranceLevel::E2Verified > AssuranceLevel::E1Basic);
        assert!(AssuranceLevel::E1Basic > AssuranceLevel::E0Anonymous);
    }

    #[test]
    fn test_capability_gating() {
        let identity = MfdiIdentity::new(
            "did:mycelix:test",
            MfdiBridge::generate_key().verifying_key(),
            0.5, // E2 level
            1.0,
        );

        // E2 can do basic and full cycle
        assert!(identity.can_perform(CognitiveCapability::BasicInference));
        assert!(identity.can_perform(CognitiveCapability::FullCycle));
        assert!(identity.can_perform(CognitiveCapability::FederatedLearning));

        // E2 cannot do weight updates (requires E3)
        assert!(!identity.can_perform(CognitiveCapability::WeightUpdate));

        // E2 cannot do soul modification (requires E4)
        assert!(!identity.can_perform(CognitiveCapability::SoulModification));
    }

    #[test]
    fn test_signed_request() {
        let signing_key = MfdiBridge::generate_key();
        let request = SignedRequest::new("test payload", &signing_key);

        assert!(request.verify());
        assert!(request.agent_did().starts_with("did:mycelix:"));
    }

    #[test]
    fn test_signed_output() {
        let signing_key = MfdiBridge::generate_key();
        let output = vec![0.1, 0.2, 0.3, 0.4];
        let signed = SignedOutput::new(output.clone(), &signing_key, "test-agent", 1);

        assert!(signed.verify(&signing_key.verifying_key()));
        assert_eq!(signed.output, output);
    }

    #[test]
    fn test_bridge_with_signing_key() {
        let signing_key = MfdiBridge::generate_key();
        let bridge = MfdiBridge::with_signing_key(MfdiBridgeConfig::default(), signing_key);

        assert!(bridge.identity().is_some());
        assert!(bridge.agent_id().is_some());
        assert!(bridge.agent_id().unwrap().starts_with("did:mycelix:"));
    }

    #[test]
    fn test_bridge_capability_check() {
        let signing_key = MfdiBridge::generate_key();
        let config = MfdiBridgeConfig {
            capability_gating: true,
            ..Default::default()
        };
        let bridge = MfdiBridge::with_signing_key(config, signing_key);

        // Default identity has score 0.5 (E2)
        assert!(bridge
            .check_capability(CognitiveCapability::FullCycle)
            .is_ok());
        assert!(bridge
            .check_capability(CognitiveCapability::FederatedLearning)
            .is_ok());
        assert!(bridge
            .check_capability(CognitiveCapability::WeightUpdate)
            .is_err());
    }

    #[test]
    fn test_bridge_sign_output() {
        let signing_key = MfdiBridge::generate_key();
        let mut bridge =
            MfdiBridge::with_signing_key(MfdiBridgeConfig::default(), signing_key.clone());

        let output = vec![1.0, 2.0, 3.0];
        let signed = bridge.sign_output(&output).unwrap();

        assert!(signed.verify(&signing_key.verifying_key()));
        assert_eq!(signed.cycle_id, 1);
    }

    #[test]
    fn test_replay_protection() {
        let signing_key = MfdiBridge::generate_key();
        let mut bridge =
            MfdiBridge::with_signing_key(MfdiBridgeConfig::default(), signing_key.clone());

        let request = SignedRequest::new("test", &signing_key);

        // First verification should succeed
        assert!(bridge.verify_request(&request).is_ok());

        // Same request again should fail (replay)
        assert!(matches!(
            bridge.verify_request(&request),
            Err(MfdiError::ReplayDetected)
        ));
    }

    #[test]
    fn test_process_input() {
        let signing_key = MfdiBridge::generate_key();
        let config = MfdiBridgeConfig {
            min_assurance_level: AssuranceLevel::E0Anonymous,
            ..Default::default()
        };
        let mut bridge = MfdiBridge::with_signing_key(config, signing_key);

        let result = bridge.process_input("test input").unwrap();
        assert_eq!(result.content, "test input");
        assert!(result.agent_id.is_some());
        assert_eq!(result.assurance_level, AssuranceLevel::E2Verified);
    }

    #[test]
    fn test_derive_did() {
        let key1 = MfdiBridge::generate_key();
        let key2 = MfdiBridge::generate_key();

        let did1 = MfdiIdentity::derive_did(&key1.verifying_key());
        let did2 = MfdiIdentity::derive_did(&key2.verifying_key());

        // DIDs should be deterministic from public key
        assert_eq!(did1, MfdiIdentity::derive_did(&key1.verifying_key()));
        // Different keys should have different DIDs
        assert_ne!(did1, did2);
        // DID format
        assert!(did1.starts_with("did:mycelix:"));
        assert_eq!(did1.len(), "did:mycelix:".len() + 32); // 16 bytes hex = 32 chars
    }
}

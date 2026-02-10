//! Byzantine Identity Bridge Integration for WebAuthn
//!
//! This module provides integration between the WebAuthn hardware key
//! authentication system and the Byzantine Identity Coordinator.
//!
//! # Overview
//!
//! Hardware key authentication provides an additional layer of security
//! for identity verification, particularly useful for:
//! - High-stakes governance operations
//! - Credential issuance and management
//! - Recovery operations
//! - Byzantine consensus participation
//!
//! # Architecture
//!
//! ```text
//! WebAuthn Authenticator
//!         │
//!         ▼
//! ┌─────────────────────┐
//! │  WebAuthnService    │ ─── Challenges, Verification
//! └─────────────────────┘
//!         │
//!         ▼
//! ┌─────────────────────┐
//! │  HardwareKeyBridge  │ ─── Credential Storage
//! └─────────────────────┘
//!         │
//!         ▼
//! ┌─────────────────────────────┐
//! │ ByzantineIdentityCoordinator│ ─── Trust Assessment
//! └─────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::identity::{HardwareKeyBridge, WebAuthnCredential};
//! use mycelix_sdk::bridge::ByzantineIdentityCoordinator;
//!
//! // Create the bridge
//! let mut bridge = HardwareKeyBridge::new(
//!     "mycelix.example.com".to_string(),
//!     "https://mycelix.example.com".to_string(),
//! );
//!
//! // Bind a hardware key to an agent
//! let agent_id = b"uhCAk..."; // Agent public key bytes
//! let credential = WebAuthnCredential::new(/* ... */);
//! bridge.bind_hardware_key(agent_id, credential)?;
//!
//! // Later, verify during high-stakes operation
//! let verified = bridge.verify_hardware_key(agent_id, &auth_response)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::webauthn::{
    AuthenticationChallenge, AuthenticationResponse, AuthenticationResult, RegistrationChallenge,
    RegistrationResponse, UserVerification, WebAuthnConfig, WebAuthnCredential, WebAuthnError,
    WebAuthnService,
};

// ============================================================================
// Error Types
// ============================================================================

/// Errors from hardware key bridge operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BridgeError {
    /// WebAuthn operation failed
    WebAuthn(WebAuthnError),

    /// Agent not found
    AgentNotFound(Vec<u8>),

    /// No hardware keys registered for agent
    NoHardwareKeys(Vec<u8>),

    /// Hardware key not found
    HardwareKeyNotFound {
        /// Agent identifier.
        agent_id: Vec<u8>,
        /// Credential identifier.
        credential_id: Vec<u8>,
    },

    /// Maximum keys per agent exceeded
    MaxKeysExceeded {
        /// Agent identifier.
        agent_id: Vec<u8>,
        /// Maximum allowed keys.
        max: usize,
    },

    /// Operation requires hardware key verification
    HardwareKeyRequired,

    /// Internal error
    InternalError(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WebAuthn(e) => write!(f, "WebAuthn error: {}", e),
            Self::AgentNotFound(id) => write!(f, "Agent not found: {:?}", id),
            Self::NoHardwareKeys(id) => {
                write!(f, "No hardware keys registered for agent: {:?}", id)
            }
            Self::HardwareKeyNotFound {
                agent_id,
                credential_id,
            } => {
                write!(
                    f,
                    "Hardware key {:?} not found for agent {:?}",
                    credential_id, agent_id
                )
            }
            Self::MaxKeysExceeded { agent_id, max } => {
                write!(f, "Maximum {} keys exceeded for agent {:?}", max, agent_id)
            }
            Self::HardwareKeyRequired => write!(f, "Operation requires hardware key verification"),
            Self::InternalError(s) => write!(f, "Internal error: {}", s),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<WebAuthnError> for BridgeError {
    fn from(e: WebAuthnError) -> Self {
        BridgeError::WebAuthn(e)
    }
}

// ============================================================================
// Bridge Types
// ============================================================================

/// Configuration for the hardware key bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareKeyBridgeConfig {
    /// WebAuthn configuration
    pub webauthn: WebAuthnConfig,

    /// Maximum hardware keys per agent
    pub max_keys_per_agent: usize,

    /// Require hardware key for high-stakes operations
    pub require_for_high_stakes: bool,

    /// Require user verification for hardware key auth
    pub require_user_verification: bool,

    /// Trust score boost for hardware key authentication
    pub hardware_key_trust_boost: f64,
}

impl Default for HardwareKeyBridgeConfig {
    fn default() -> Self {
        Self {
            webauthn: WebAuthnConfig::single_domain("mycelix.local", "Mycelix"),
            max_keys_per_agent: 5,
            require_for_high_stakes: true,
            require_user_verification: false,
            hardware_key_trust_boost: 0.1,
        }
    }
}

/// Record of a hardware key binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareKeyBinding {
    /// The WebAuthn credential
    pub credential: WebAuthnCredential,

    /// When the key was bound to the agent
    pub bound_at: u64,

    /// Last successful authentication with this key
    pub last_used: Option<u64>,

    /// Number of successful authentications
    pub use_count: u64,

    /// Whether this is the primary key for the agent
    pub is_primary: bool,

    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl HardwareKeyBinding {
    /// Create a new binding
    pub fn new(credential: WebAuthnCredential, is_primary: bool) -> Self {
        Self {
            credential,
            bound_at: current_timestamp(),
            last_used: None,
            use_count: 0,
            is_primary,
            metadata: HashMap::new(),
        }
    }

    /// Record a successful use
    pub fn record_use(&mut self, new_sign_count: u32) {
        self.last_used = Some(current_timestamp());
        self.use_count += 1;
        // Ignore error since we've already validated
        let _ = self.credential.update_sign_count(new_sign_count);
    }
}

/// Summary of hardware key authentication status for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareKeyStatus {
    /// Agent identifier
    pub agent_id: Vec<u8>,

    /// Number of registered keys
    pub key_count: usize,

    /// Whether agent has a primary key
    pub has_primary: bool,

    /// Last authentication timestamp
    pub last_authenticated: Option<u64>,

    /// Total authentications across all keys
    pub total_authentications: u64,
}

// ============================================================================
// Hardware Key Bridge
// ============================================================================

/// Bridge connecting hardware key authentication to identity system
///
/// This struct manages hardware key bindings for agents and provides
/// integration with the Byzantine Identity Coordinator.
pub struct HardwareKeyBridge {
    /// Configuration
    config: HardwareKeyBridgeConfig,

    /// WebAuthn service
    webauthn: WebAuthnService,

    /// Hardware key bindings by agent ID
    bindings: HashMap<Vec<u8>, Vec<HardwareKeyBinding>>,

    /// Pending registration challenges by agent ID
    pending_registrations: HashMap<Vec<u8>, RegistrationChallenge>,

    /// Pending authentication challenges by agent ID
    pending_authentications: HashMap<Vec<u8>, AuthenticationChallenge>,
}

impl HardwareKeyBridge {
    /// Create a new hardware key bridge
    pub fn new(rp_id: String, rp_origin: String) -> Self {
        let webauthn = WebAuthnService::new(rp_id.clone(), rp_origin.clone());

        Self {
            config: HardwareKeyBridgeConfig {
                webauthn: WebAuthnConfig::single_domain(&rp_id, &rp_id),
                ..Default::default()
            },
            webauthn,
            bindings: HashMap::new(),
            pending_registrations: HashMap::new(),
            pending_authentications: HashMap::new(),
        }
    }

    /// Create with full configuration
    pub fn with_config(config: HardwareKeyBridgeConfig) -> Self {
        let webauthn = WebAuthnService::with_config(config.webauthn.clone());

        Self {
            config,
            webauthn,
            bindings: HashMap::new(),
            pending_registrations: HashMap::new(),
            pending_authentications: HashMap::new(),
        }
    }

    // =========================================================================
    // Registration (Key Binding)
    // =========================================================================

    /// Start hardware key registration for an agent
    ///
    /// # Arguments
    /// * `agent_id` - Agent public key bytes
    /// * `display_name` - Human-readable name for the agent
    ///
    /// # Returns
    /// Registration challenge to send to the client
    pub fn start_registration(
        &mut self,
        agent_id: &[u8],
        display_name: &str,
    ) -> Result<RegistrationChallenge, BridgeError> {
        // Check max keys limit
        if let Some(existing) = self.bindings.get(agent_id) {
            if existing.len() >= self.config.max_keys_per_agent {
                return Err(BridgeError::MaxKeysExceeded {
                    agent_id: agent_id.to_vec(),
                    max: self.config.max_keys_per_agent,
                });
            }
        }

        // Get existing credential IDs to exclude
        let exclude_ids: Vec<Vec<u8>> = self
            .bindings
            .get(agent_id)
            .map(|bindings| {
                bindings
                    .iter()
                    .map(|b| b.credential.credential_id.clone())
                    .collect()
            })
            .unwrap_or_default();

        // Create challenge
        let challenge = self.webauthn.create_registration_challenge_with_options(
            agent_id,
            display_name,
            display_name,
            super::webauthn::AttestationConveyance::Direct,
            super::webauthn::AuthenticatorSelectionCriteria::default(),
            exclude_ids,
        )?;

        // Store pending registration
        self.pending_registrations
            .insert(agent_id.to_vec(), challenge.clone());

        Ok(challenge)
    }

    /// Complete hardware key registration
    ///
    /// # Arguments
    /// * `agent_id` - Agent public key bytes
    /// * `response` - Registration response from authenticator
    ///
    /// # Returns
    /// The bound credential
    pub fn complete_registration(
        &mut self,
        agent_id: &[u8],
        response: &RegistrationResponse,
    ) -> Result<WebAuthnCredential, BridgeError> {
        // Get pending challenge
        let challenge = self
            .pending_registrations
            .remove(agent_id)
            .ok_or_else(|| BridgeError::InternalError("No pending registration".to_string()))?;

        // Verify registration
        let credential = self.webauthn.verify_registration(&challenge, response)?;

        // Determine if this is the primary key
        let is_primary = self
            .bindings
            .get(agent_id)
            .map(|b| b.is_empty())
            .unwrap_or(true);

        // Create binding
        let binding = HardwareKeyBinding::new(credential.clone(), is_primary);

        // Store binding
        self.bindings
            .entry(agent_id.to_vec())
            .or_default()
            .push(binding);

        Ok(credential)
    }

    /// Bind an existing credential to an agent (for migration/import)
    pub fn bind_hardware_key(
        &mut self,
        agent_id: &[u8],
        credential: WebAuthnCredential,
    ) -> Result<(), BridgeError> {
        // Check max keys limit
        if let Some(existing) = self.bindings.get(agent_id) {
            if existing.len() >= self.config.max_keys_per_agent {
                return Err(BridgeError::MaxKeysExceeded {
                    agent_id: agent_id.to_vec(),
                    max: self.config.max_keys_per_agent,
                });
            }

            // Check for duplicate credential ID
            if existing
                .iter()
                .any(|b| b.credential.credential_id == credential.credential_id)
            {
                return Err(BridgeError::WebAuthn(
                    WebAuthnError::CredentialAlreadyRegistered,
                ));
            }
        }

        let is_primary = self
            .bindings
            .get(agent_id)
            .map(|b| b.is_empty())
            .unwrap_or(true);
        let binding = HardwareKeyBinding::new(credential, is_primary);

        self.bindings
            .entry(agent_id.to_vec())
            .or_default()
            .push(binding);

        Ok(())
    }

    // =========================================================================
    // Authentication (Key Verification)
    // =========================================================================

    /// Start hardware key authentication for an agent
    ///
    /// # Arguments
    /// * `agent_id` - Agent public key bytes
    ///
    /// # Returns
    /// Authentication challenge to send to the client
    pub fn start_authentication(
        &mut self,
        agent_id: &[u8],
    ) -> Result<AuthenticationChallenge, BridgeError> {
        // Get agent's credential IDs
        let credential_ids: Vec<Vec<u8>> = self
            .bindings
            .get(agent_id)
            .ok_or_else(|| BridgeError::NoHardwareKeys(agent_id.to_vec()))?
            .iter()
            .map(|b| b.credential.credential_id.clone())
            .collect();

        if credential_ids.is_empty() {
            return Err(BridgeError::NoHardwareKeys(agent_id.to_vec()));
        }

        // Create challenge
        let user_verification = if self.config.require_user_verification {
            UserVerification::Required
        } else {
            UserVerification::Preferred
        };

        let challenge = self
            .webauthn
            .create_authentication_challenge_with_options(&credential_ids, user_verification)?;

        // Store pending authentication
        self.pending_authentications
            .insert(agent_id.to_vec(), challenge.clone());

        Ok(challenge)
    }

    /// Verify hardware key authentication
    ///
    /// # Arguments
    /// * `agent_id` - Agent public key bytes
    /// * `response` - Authentication response from authenticator
    ///
    /// # Returns
    /// Authentication result with updated credential
    pub fn verify_hardware_key(
        &mut self,
        agent_id: &[u8],
        response: &AuthenticationResponse,
    ) -> Result<AuthenticationResult, BridgeError> {
        // Get pending challenge
        let challenge = self
            .pending_authentications
            .remove(agent_id)
            .ok_or_else(|| BridgeError::InternalError("No pending authentication".to_string()))?;

        // Find the credential
        let bindings = self
            .bindings
            .get_mut(agent_id)
            .ok_or_else(|| BridgeError::NoHardwareKeys(agent_id.to_vec()))?;

        let binding = bindings
            .iter_mut()
            .find(|b| b.credential.credential_id == response.credential_id)
            .ok_or_else(|| BridgeError::HardwareKeyNotFound {
                agent_id: agent_id.to_vec(),
                credential_id: response.credential_id.clone(),
            })?;

        // Verify authentication
        let result =
            self.webauthn
                .verify_authentication(&challenge, response, &binding.credential)?;

        // Update binding
        binding.record_use(result.new_sign_count);

        Ok(result)
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /// Get hardware key status for an agent
    pub fn get_status(&self, agent_id: &[u8]) -> Option<HardwareKeyStatus> {
        self.bindings.get(agent_id).map(|bindings| {
            let total_authentications: u64 = bindings.iter().map(|b| b.use_count).sum();
            let last_authenticated = bindings.iter().filter_map(|b| b.last_used).max();
            let has_primary = bindings.iter().any(|b| b.is_primary);

            HardwareKeyStatus {
                agent_id: agent_id.to_vec(),
                key_count: bindings.len(),
                has_primary,
                last_authenticated,
                total_authentications,
            }
        })
    }

    /// Get all credentials for an agent
    pub fn get_credentials(&self, agent_id: &[u8]) -> Vec<&WebAuthnCredential> {
        self.bindings
            .get(agent_id)
            .map(|bindings| bindings.iter().map(|b| &b.credential).collect())
            .unwrap_or_default()
    }

    /// Check if agent has any hardware keys
    pub fn has_hardware_keys(&self, agent_id: &[u8]) -> bool {
        self.bindings
            .get(agent_id)
            .map(|b| !b.is_empty())
            .unwrap_or(false)
    }

    /// Get the number of agents with hardware keys
    pub fn agent_count(&self) -> usize {
        self.bindings.len()
    }

    /// Get the total number of registered keys
    pub fn total_key_count(&self) -> usize {
        self.bindings.values().map(|b| b.len()).sum()
    }

    // =========================================================================
    // Management
    // =========================================================================

    /// Remove a hardware key binding
    pub fn unbind_hardware_key(
        &mut self,
        agent_id: &[u8],
        credential_id: &[u8],
    ) -> Result<WebAuthnCredential, BridgeError> {
        let bindings = self
            .bindings
            .get_mut(agent_id)
            .ok_or_else(|| BridgeError::NoHardwareKeys(agent_id.to_vec()))?;

        let idx = bindings
            .iter()
            .position(|b| b.credential.credential_id == credential_id)
            .ok_or_else(|| BridgeError::HardwareKeyNotFound {
                agent_id: agent_id.to_vec(),
                credential_id: credential_id.to_vec(),
            })?;

        let binding = bindings.remove(idx);

        // If we removed the primary, make the first remaining key primary
        if binding.is_primary && !bindings.is_empty() {
            bindings[0].is_primary = true;
        }

        // Remove agent entry if no keys left
        if bindings.is_empty() {
            self.bindings.remove(agent_id);
        }

        Ok(binding.credential)
    }

    /// Set a credential as the primary key for an agent
    pub fn set_primary_key(
        &mut self,
        agent_id: &[u8],
        credential_id: &[u8],
    ) -> Result<(), BridgeError> {
        let bindings = self
            .bindings
            .get_mut(agent_id)
            .ok_or_else(|| BridgeError::NoHardwareKeys(agent_id.to_vec()))?;

        let mut found = false;
        for binding in bindings.iter_mut() {
            if binding.credential.credential_id == credential_id {
                binding.is_primary = true;
                found = true;
            } else {
                binding.is_primary = false;
            }
        }

        if !found {
            return Err(BridgeError::HardwareKeyNotFound {
                agent_id: agent_id.to_vec(),
                credential_id: credential_id.to_vec(),
            });
        }

        Ok(())
    }

    /// Get the trust score boost for hardware key authentication
    pub fn trust_boost(&self) -> f64 {
        self.config.hardware_key_trust_boost
    }

    /// Check if hardware key is required for high-stakes operations
    pub fn requires_for_high_stakes(&self) -> bool {
        self.config.require_for_high_stakes
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::webauthn::AttestationFormat;

    fn create_test_bridge() -> HardwareKeyBridge {
        HardwareKeyBridge::new(
            "test.mycelix.local".to_string(),
            "https://test.mycelix.local".to_string(),
        )
    }

    fn create_test_credential(id: u8) -> WebAuthnCredential {
        WebAuthnCredential::new(
            vec![id, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9],
            AttestationFormat::None,
        )
    }

    #[test]
    fn test_bind_hardware_key() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();
        let cred = create_test_credential(1);

        assert!(bridge.bind_hardware_key(&agent_id, cred.clone()).is_ok());
        assert!(bridge.has_hardware_keys(&agent_id));
        assert_eq!(bridge.agent_count(), 1);
        assert_eq!(bridge.total_key_count(), 1);
    }

    #[test]
    fn test_multiple_keys() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        // Bind multiple keys
        for i in 0..3 {
            let cred = create_test_credential(i);
            assert!(bridge.bind_hardware_key(&agent_id, cred).is_ok());
        }

        assert_eq!(bridge.total_key_count(), 3);

        let status = bridge.get_status(&agent_id).unwrap();
        assert_eq!(status.key_count, 3);
        assert!(status.has_primary);
    }

    #[test]
    fn test_max_keys_limit() {
        let config = HardwareKeyBridgeConfig {
            max_keys_per_agent: 2,
            ..Default::default()
        };
        let mut bridge = HardwareKeyBridge::with_config(config);
        let agent_id = b"agent123".to_vec();

        // Bind up to limit
        bridge
            .bind_hardware_key(&agent_id, create_test_credential(1))
            .unwrap();
        bridge
            .bind_hardware_key(&agent_id, create_test_credential(2))
            .unwrap();

        // Should fail on third key
        let result = bridge.bind_hardware_key(&agent_id, create_test_credential(3));
        assert!(matches!(result, Err(BridgeError::MaxKeysExceeded { .. })));
    }

    #[test]
    fn test_unbind_hardware_key() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        bridge
            .bind_hardware_key(&agent_id, create_test_credential(1))
            .unwrap();
        bridge
            .bind_hardware_key(&agent_id, create_test_credential(2))
            .unwrap();

        // Remove first key
        let removed = bridge.unbind_hardware_key(&agent_id, &vec![1, 1, 2, 3]);
        assert!(removed.is_ok());
        assert_eq!(bridge.total_key_count(), 1);

        // Remove second key
        let removed = bridge.unbind_hardware_key(&agent_id, &vec![2, 1, 2, 3]);
        assert!(removed.is_ok());
        assert!(!bridge.has_hardware_keys(&agent_id));
    }

    #[test]
    fn test_set_primary_key() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        bridge
            .bind_hardware_key(&agent_id, create_test_credential(1))
            .unwrap();
        bridge
            .bind_hardware_key(&agent_id, create_test_credential(2))
            .unwrap();

        // Set second key as primary
        assert!(bridge.set_primary_key(&agent_id, &vec![2, 1, 2, 3]).is_ok());

        // Verify first key is no longer primary
        let bindings = bridge.bindings.get(&agent_id).unwrap();
        assert!(!bindings[0].is_primary);
        assert!(bindings[1].is_primary);
    }

    #[test]
    fn test_start_registration() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        let challenge = bridge.start_registration(&agent_id, "Test User");
        assert!(challenge.is_ok());

        let challenge = challenge.unwrap();
        assert_eq!(challenge.user_id, agent_id);
        assert!(!challenge.is_expired());
    }

    #[test]
    fn test_start_authentication_no_keys() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        let result = bridge.start_authentication(&agent_id);
        assert!(matches!(result, Err(BridgeError::NoHardwareKeys(_))));
    }

    #[test]
    fn test_start_authentication_with_keys() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        bridge
            .bind_hardware_key(&agent_id, create_test_credential(1))
            .unwrap();

        let challenge = bridge.start_authentication(&agent_id);
        assert!(challenge.is_ok());

        let challenge = challenge.unwrap();
        assert_eq!(challenge.allowed_credentials.len(), 1);
    }

    #[test]
    fn test_duplicate_credential_rejected() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();
        let cred = create_test_credential(1);

        assert!(bridge.bind_hardware_key(&agent_id, cred.clone()).is_ok());
        let result = bridge.bind_hardware_key(&agent_id, cred);
        assert!(matches!(
            result,
            Err(BridgeError::WebAuthn(
                WebAuthnError::CredentialAlreadyRegistered
            ))
        ));
    }

    #[test]
    fn test_hardware_key_status() {
        let mut bridge = create_test_bridge();
        let agent_id = b"agent123".to_vec();

        // No status before binding
        assert!(bridge.get_status(&agent_id).is_none());

        bridge
            .bind_hardware_key(&agent_id, create_test_credential(1))
            .unwrap();

        let status = bridge.get_status(&agent_id).unwrap();
        assert_eq!(status.key_count, 1);
        assert!(status.has_primary);
        assert_eq!(status.total_authentications, 0);
        assert!(status.last_authenticated.is_none());
    }

    #[test]
    fn test_bridge_error_display() {
        let err = BridgeError::NoHardwareKeys(vec![1, 2, 3]);
        assert!(err.to_string().contains("No hardware keys"));

        let err = BridgeError::MaxKeysExceeded {
            agent_id: vec![1],
            max: 5,
        };
        assert!(err.to_string().contains("Maximum 5 keys"));
    }

    #[test]
    fn test_config_defaults() {
        let config = HardwareKeyBridgeConfig::default();
        assert_eq!(config.max_keys_per_agent, 5);
        assert!(config.require_for_high_stakes);
        assert!(!config.require_user_verification);
        assert!(config.hardware_key_trust_boost > 0.0);
    }
}

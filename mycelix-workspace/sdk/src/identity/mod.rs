//! Mycelix Identity Module
//!
//! Provides identity-related functionality for the Mycelix ecosystem,
//! including WebAuthn/FIDO2 hardware key authentication integration.
//!
//! # Overview
//!
//! This module extends the identity capabilities beyond the core DID registry
//! (in zomes/identity) with SDK-level authentication and verification tools.
//!
//! # Components
//!
//! - **WebAuthn**: Hardware key authentication (YubiKey, platform authenticators)
//! - **Bridge Integration**: Connect WebAuthn to Byzantine Identity Coordinator
//!
//! # WebAuthn Example
//!
//! ```rust,ignore
//! use mycelix_sdk::identity::webauthn::{WebAuthnService, WebAuthnCredential};
//!
//! // Create service for your domain
//! let mut service = WebAuthnService::new(
//!     "mycelix.example.com".to_string(),
//!     "https://mycelix.example.com".to_string(),
//! );
//!
//! // Create registration challenge for user
//! let user_id = b"did:mycelix:agent123".to_vec();
//! let challenge = service.create_registration_challenge(&user_id, "alice").unwrap();
//!
//! // Challenge would be sent to client...
//! // Client returns RegistrationResponse...
//! // Verify and store credential...
//! ```
//!
//! # Bridge Integration Example
//!
//! ```rust,ignore
//! use mycelix_sdk::identity::{HardwareKeyBridge, WebAuthnCredential};
//!
//! // Create bridge for your domain
//! let mut bridge = HardwareKeyBridge::new(
//!     "mycelix.example.com".to_string(),
//!     "https://mycelix.example.com".to_string(),
//! );
//!
//! // Bind a hardware key to an agent
//! let agent_id = b"agent_pubkey_bytes";
//! let challenge = bridge.start_registration(agent_id, "Alice").unwrap();
//!
//! // After receiving response from authenticator...
//! // bridge.complete_registration(agent_id, &response)?;
//!
//! // Later, verify the key
//! // let auth_challenge = bridge.start_authentication(agent_id)?;
//! // let result = bridge.verify_hardware_key(agent_id, &auth_response)?;
//! ```

pub mod webauthn;
pub mod bridge_integration;

// Re-export WebAuthn types
pub use webauthn::{
    // Core types
    WebAuthnCredential,
    WebAuthnService,
    WebAuthnConfig,
    WebAuthnError,

    // Challenge types
    RegistrationChallenge,
    AuthenticationChallenge,
    AuthenticatorSelectionCriteria,

    // Response types
    RegistrationResponse,
    AuthenticationResponse,
    AuthenticationResult,

    // Enums
    AttestationFormat,
    AttestationConveyance,
    AuthenticatorTransport,
    AuthenticatorAttachment,
    UserVerification,
    ResidentKeyRequirement,

    // Utilities
    AllowedCredential,
    AuthenticatorFlags,
};

// Re-export bridge integration types
pub use bridge_integration::{
    HardwareKeyBridge,
    HardwareKeyBridgeConfig,
    HardwareKeyBinding,
    HardwareKeyStatus,
    BridgeError,
};

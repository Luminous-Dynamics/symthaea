// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federation Protocol Implementation
//!
//! Handles the actual network communication between instances

use super::*;
use async_trait::async_trait;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Federation protocol handler
pub struct FederationProtocol {
    config: FederationConfig,
    signing_key: SigningKey,
    http_client: Client,
    known_instances: Arc<RwLock<HashMap<String, FederatedInstance>>>,
    message_handlers: Arc<RwLock<Vec<Box<dyn MessageHandler>>>>,
}

#[async_trait]
pub trait MessageHandler: Send + Sync {
    async fn handle(&self, from: &str, message: &FederationMessage) -> Option<FederationMessage>;
    fn handles(&self, message_type: &str) -> bool;
}

impl FederationProtocol {
    pub fn new(config: FederationConfig, signing_key: SigningKey) -> Self {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent(format!("Mycelix-Mail-Federation/1.0 ({})", config.domain))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            signing_key,
            http_client,
            known_instances: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a message handler
    pub async fn register_handler(&self, handler: Box<dyn MessageHandler>) {
        self.message_handlers.write().await.push(handler);
    }

    /// Connect to a remote instance
    pub async fn connect(&self, domain: &str) -> Result<FederatedInstance, FederationError> {
        // Check if blocked
        if self.config.blocked_instances.contains(&domain.to_string()) {
            return Err(FederationError::Blocked(domain.to_string()));
        }

        // Check if allowed (if allowlist is set)
        if !self.config.allowed_instances.is_empty()
            && !self.config.allowed_instances.contains(&domain.to_string())
        {
            return Err(FederationError::NotAllowed(domain.to_string()));
        }

        // Discover instance metadata
        let instance = self.discover_instance(domain).await?;

        // Send hello message
        let hello = self.create_hello_message();
        let response = self.send_message(domain, &hello).await?;

        match response {
            FederationMessage::HelloAck(ack) => {
                if ack.accepted {
                    // Store instance
                    self.known_instances
                        .write()
                        .await
                        .insert(domain.to_string(), instance.clone());
                    Ok(instance)
                } else {
                    Err(FederationError::Rejected)
                }
            }
            FederationMessage::Error(e) => Err(FederationError::Remote(e.message)),
            _ => Err(FederationError::UnexpectedResponse),
        }
    }

    /// Discover instance metadata from well-known endpoint
    async fn discover_instance(&self, domain: &str) -> Result<FederatedInstance, FederationError> {
        let url = format!("https://{}/.well-known/mycelix-federation", domain);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| FederationError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(FederationError::DiscoveryFailed(domain.to_string()));
        }

        let metadata: InstanceMetadata = response
            .json()
            .await
            .map_err(|e| FederationError::InvalidResponse(e.to_string()))?;

        Ok(FederatedInstance {
            id: Uuid::new_v4(),
            domain: domain.to_string(),
            display_name: metadata.display_name,
            public_key: metadata.public_key,
            protocol_version: metadata.protocol_version,
            capabilities: metadata.capabilities,
            status: InstanceStatus::Active,
            last_seen: Utc::now(),
            created_at: Utc::now(),
        })
    }

    /// Create hello message for handshake
    fn create_hello_message(&self) -> FederationMessage {
        let payload = HelloPayload {
            instance_id: Uuid::new_v4(), // Should come from config
            domain: self.config.domain.clone(),
            public_key: hex::encode(self.signing_key.verifying_key().as_bytes()),
            protocol_version: "1.0".to_string(),
            capabilities: vec![
                FederationCapability::TrustSync,
                FederationCapability::IdentityVerification,
                FederationCapability::KeyDiscovery,
                FederationCapability::ReputationSharing,
            ],
            timestamp: Utc::now(),
            signature: String::new(), // Will be signed
        };

        FederationMessage::Hello(payload)
    }

    /// Send a message to a remote instance
    pub async fn send_message(
        &self,
        domain: &str,
        message: &FederationMessage,
    ) -> Result<FederationMessage, FederationError> {
        let url = format!("https://{}/api/federation/inbox", domain);

        // Sign the message
        let signed_message = self.sign_message(message)?;

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Federation-Instance", &self.config.domain)
            .json(&signed_message)
            .send()
            .await
            .map_err(|e| FederationError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(FederationError::HttpError(status.as_u16(), body));
        }

        let response_message: SignedMessage = response
            .json()
            .await
            .map_err(|e| FederationError::InvalidResponse(e.to_string()))?;

        // Verify signature
        self.verify_message(domain, &response_message)?;

        Ok(response_message.message)
    }

    /// Sign a federation message
    fn sign_message(&self, message: &FederationMessage) -> Result<SignedMessage, FederationError> {
        let payload = serde_json::to_vec(message)
            .map_err(|e| FederationError::SerializationError(e.to_string()))?;

        let signature = self.signing_key.sign(&payload);

        Ok(SignedMessage {
            message: message.clone(),
            signature: hex::encode(signature.to_bytes()),
            from_instance: self.config.domain.clone(),
            timestamp: Utc::now(),
        })
    }

    /// Verify a signed message from a remote instance
    fn verify_message(
        &self,
        from_domain: &str,
        signed: &SignedMessage,
    ) -> Result<(), FederationError> {
        // Look up instance public key
        let instances = futures::executor::block_on(self.known_instances.read());
        let instance = instances
            .get(from_domain)
            .ok_or_else(|| FederationError::UnknownInstance(from_domain.to_string()))?;

        // Decode public key
        let public_key_bytes = hex::decode(&instance.public_key)
            .map_err(|_| FederationError::InvalidSignature)?;

        let public_key = VerifyingKey::from_bytes(
            public_key_bytes
                .as_slice()
                .try_into()
                .map_err(|_| FederationError::InvalidSignature)?,
        )
        .map_err(|_| FederationError::InvalidSignature)?;

        // Decode signature
        let signature_bytes =
            hex::decode(&signed.signature).map_err(|_| FederationError::InvalidSignature)?;

        let signature = Signature::from_bytes(
            signature_bytes
                .as_slice()
                .try_into()
                .map_err(|_| FederationError::InvalidSignature)?,
        );

        // Verify
        let payload = serde_json::to_vec(&signed.message)
            .map_err(|e| FederationError::SerializationError(e.to_string()))?;

        public_key
            .verify(&payload, &signature)
            .map_err(|_| FederationError::InvalidSignature)?;

        Ok(())
    }

    /// Handle incoming federation message
    pub async fn handle_incoming(
        &self,
        from_domain: &str,
        signed: SignedMessage,
    ) -> Result<FederationMessage, FederationError> {
        // Verify signature (if instance is known)
        if self.known_instances.read().await.contains_key(from_domain) {
            self.verify_message(from_domain, &signed)?;
        }

        // Route to appropriate handler
        let handlers = self.message_handlers.read().await;

        for handler in handlers.iter() {
            if let Some(response) = handler.handle(from_domain, &signed.message).await {
                return Ok(response);
            }
        }

        // Default handlers
        match &signed.message {
            FederationMessage::Hello(hello) => {
                self.handle_hello(from_domain, hello).await
            }
            FederationMessage::TrustQuery(query) => {
                self.handle_trust_query(from_domain, query).await
            }
            _ => Err(FederationError::UnhandledMessage),
        }
    }

    async fn handle_hello(
        &self,
        from_domain: &str,
        hello: &HelloPayload,
    ) -> Result<FederationMessage, FederationError> {
        // Validate hello message
        if hello.protocol_version != "1.0" {
            return Ok(FederationMessage::Error(ErrorPayload {
                code: "UNSUPPORTED_VERSION".to_string(),
                message: format!("Protocol version {} not supported", hello.protocol_version),
                details: None,
            }));
        }

        // Check if we should accept
        let accepted = if self.config.blocked_instances.contains(&from_domain.to_string()) {
            false
        } else if !self.config.allowed_instances.is_empty() {
            self.config.allowed_instances.contains(&from_domain.to_string())
        } else {
            self.config.auto_accept
        };

        // Store instance if accepted
        if accepted {
            let instance = FederatedInstance {
                id: hello.instance_id,
                domain: hello.domain.clone(),
                display_name: hello.domain.clone(),
                public_key: hello.public_key.clone(),
                protocol_version: hello.protocol_version.clone(),
                capabilities: hello.capabilities.clone(),
                status: InstanceStatus::Active,
                last_seen: Utc::now(),
                created_at: Utc::now(),
            };

            self.known_instances
                .write()
                .await
                .insert(from_domain.to_string(), instance);
        }

        // Create response
        let ack = HelloAckPayload {
            instance_id: Uuid::new_v4(),
            accepted,
            capabilities: vec![
                FederationCapability::TrustSync,
                FederationCapability::IdentityVerification,
            ],
            challenge: if accepted {
                Some(hex::encode(rand::random::<[u8; 32]>()))
            } else {
                None
            },
            signature: String::new(),
        };

        Ok(FederationMessage::HelloAck(ack))
    }

    async fn handle_trust_query(
        &self,
        _from_domain: &str,
        query: &TrustQueryPayload,
    ) -> Result<FederationMessage, FederationError> {
        // This would query local trust database
        // For now, return empty response
        Ok(FederationMessage::TrustResponse(TrustResponsePayload {
            target_user: query.target_user.clone(),
            attestations: vec![],
            aggregate_score: None,
            path: vec![],
        }))
    }

    /// Query trust for a user from federated instances
    pub async fn query_federated_trust(
        &self,
        user_id: &str,
        context: Option<&str>,
    ) -> Vec<(String, TrustResponsePayload)> {
        let instances = self.known_instances.read().await;
        let mut results = Vec::new();

        for (domain, instance) in instances.iter() {
            if instance.capabilities.contains(&FederationCapability::TrustSync) {
                let query = FederationMessage::TrustQuery(TrustQueryPayload {
                    requester: format!("{}@{}", "local", self.config.domain),
                    target_user: user_id.to_string(),
                    context: context.map(String::from),
                    depth: self.config.max_trust_depth,
                });

                match self.send_message(domain, &query).await {
                    Ok(FederationMessage::TrustResponse(response)) => {
                        results.push((domain.clone(), response));
                    }
                    _ => continue,
                }
            }
        }

        results
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMessage {
    pub message: FederationMessage,
    pub signature: String,
    pub from_instance: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetadata {
    pub display_name: String,
    pub public_key: String,
    pub protocol_version: String,
    pub capabilities: Vec<FederationCapability>,
    pub admin_email: Option<String>,
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum FederationError {
    #[error("Instance {0} is blocked")]
    Blocked(String),

    #[error("Instance {0} is not in allowed list")]
    NotAllowed(String),

    #[error("Federation request rejected")]
    Rejected,

    #[error("Network error: {0}")]
    Network(String),

    #[error("Discovery failed for {0}")]
    DiscoveryFailed(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("HTTP error {0}: {1}")]
    HttpError(u16, String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Unknown instance: {0}")]
    UnknownInstance(String),

    #[error("Unexpected response type")]
    UnexpectedResponse,

    #[error("Message type not handled")]
    UnhandledMessage,

    #[error("Remote error: {0}")]
    Remote(String),
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Main identity client implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn, info};

use crate::error::{IdentityError, IdentityResult};
use crate::types::*;

/// Cache entry with expiration
struct CacheEntry<T> {
    value: T,
    expires_at: Instant,
}

impl<T: Clone> CacheEntry<T> {
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            expires_at: Instant::now() + ttl,
        }
    }

    fn is_valid(&self) -> bool {
        Instant::now() < self.expires_at
    }
}

/// Identity client for DID resolution, credential verification, and assurance levels
///
/// Designed for graceful degradation - if identity hApp is unavailable,
/// operations return warnings/defaults instead of hard failures.
pub struct IdentityClient {
    config: IdentityClientConfig,
    did_cache: Arc<RwLock<HashMap<String, CacheEntry<DidDocument>>>>,
    revocation_cache: Arc<RwLock<HashMap<String, CacheEntry<RevocationStatus>>>>,
    identity_happ_available: Arc<RwLock<bool>>,
    last_connection_check: Arc<RwLock<Instant>>,
}

impl IdentityClient {
    /// Create a new identity client
    pub fn new(config: IdentityClientConfig) -> Self {
        Self {
            config,
            did_cache: Arc::new(RwLock::new(HashMap::new())),
            revocation_cache: Arc::new(RwLock::new(HashMap::new())),
            identity_happ_available: Arc::new(RwLock::new(false)),
            last_connection_check: Arc::new(RwLock::new(Instant::now() - Duration::from_secs(60))),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(IdentityClientConfig::default())
    }

    /// Create with specified conductor URL
    pub fn with_conductor_url(conductor_url: impl Into<String>) -> Self {
        Self::new(IdentityClientConfig::with_conductor_url(conductor_url))
    }

    /// Get the conductor URL
    pub fn conductor_url(&self) -> &str {
        &self.config.conductor_url
    }

    // =========================================================================
    // Connection Management
    // =========================================================================

    /// Check if identity hApp is available
    ///
    /// This should be called with the actual Holochain client to verify connectivity.
    /// Returns false if the identity DNA is not installed.
    pub async fn check_connection(&self) -> bool {
        let mut last_check = self.last_connection_check.write().await;
        let check_interval = Duration::from_secs(30);

        if last_check.elapsed() < check_interval {
            return *self.identity_happ_available.read().await;
        }

        *last_check = Instant::now();

        // In actual implementation, this would check with Holochain
        // For now, assume available (will be updated by set_connection_status)
        *self.identity_happ_available.read().await
    }

    /// Set connection status (called by integration layer)
    pub async fn set_connection_status(&self, available: bool) {
        let mut status = self.identity_happ_available.write().await;
        *status = available;

        if available {
            info!("Identity hApp connection established");
        } else {
            warn!("Identity hApp not available");
        }
    }

    // =========================================================================
    // DID Resolution
    // =========================================================================

    /// Resolve a DID to its document
    pub async fn resolve_did(&self, did: &str) -> IdentityResult<DidResolutionResult> {
        let start = Instant::now();

        // Validate DID format
        if !did.starts_with("did:mycelix:") {
            return Ok(DidResolutionResult {
                success: false,
                error: Some("Invalid DID format. Must start with 'did:mycelix:'".to_string()),
                ..Default::default()
            });
        }

        // Check cache
        if self.config.enable_cache {
            let cache = self.did_cache.read().await;
            if let Some(entry) = cache.get(did) {
                if entry.is_valid() {
                    debug!("DID resolution cache hit: {}", did);
                    return Ok(DidResolutionResult {
                        success: true,
                        did_document: Some(entry.value.clone()),
                        cached: true,
                        metadata: Some(ResolutionMetadata {
                            resolve_time_ms: start.elapsed().as_millis() as u64,
                            source: ResolutionSource::Local,
                        }),
                        ..Default::default()
                    });
                }
            }
        }

        // Check if identity hApp is available
        let available = self.check_connection().await;

        if !available {
            match self.config.fallback_mode {
                FallbackMode::Error => {
                    return Err(IdentityError::IdentityHappUnavailable);
                }
                FallbackMode::Warn => {
                    warn!("Identity hApp not available for DID resolution: {}", did);
                }
                FallbackMode::Silent => {}
            }

            return Ok(DidResolutionResult {
                success: false,
                error: Some("Identity hApp not available - using degraded mode".to_string()),
                metadata: Some(ResolutionMetadata {
                    resolve_time_ms: start.elapsed().as_millis() as u64,
                    source: ResolutionSource::Fallback,
                }),
                ..Default::default()
            });
        }

        // In actual implementation, this would call the identity zome
        // For now, return a placeholder
        Ok(DidResolutionResult {
            success: false,
            error: Some("DID resolution requires Holochain integration".to_string()),
            metadata: Some(ResolutionMetadata {
                resolve_time_ms: start.elapsed().as_millis() as u64,
                source: ResolutionSource::Dht,
            }),
            ..Default::default()
        })
    }

    /// Resolve a DID from identity hApp (internal, requires actual zome call)
    ///
    /// This method should be called from the integration layer with the result
    /// from the actual zome call.
    pub async fn cache_did_document(&self, did: &str, doc: DidDocument) {
        if self.config.enable_cache {
            let mut cache = self.did_cache.write().await;
            let ttl = Duration::from_secs(self.config.cache_ttl_secs);
            cache.insert(did.to_string(), CacheEntry::new(doc, ttl));
            debug!("Cached DID document: {}", did);
        }
    }

    // =========================================================================
    // Credential Verification
    // =========================================================================

    /// Verify a credential
    pub async fn verify_credential(
        &self,
        credential: &VerifiableCredential,
    ) -> IdentityResult<CredentialVerificationResult> {
        let mut result = CredentialVerificationResult::default();

        // Check expiration
        if let Some(ref exp) = credential.expiration_date {
            // Simple date comparison (would use proper date parsing in production)
            let now = chrono_placeholder_now();
            result.checks.not_expired = exp > &now;
        } else {
            result.checks.not_expired = true;
        }

        // Check revocation
        if credential.credential_status.is_some() {
            let revocation = self.check_revocation_status(&credential.id).await?;
            result.checks.not_revoked = revocation.status == RevocationState::Active;
        } else {
            result.checks.not_revoked = true;
        }

        // Verify issuer
        let issuer_result = self.resolve_did(&credential.issuer).await?;
        result.checks.issuer_trusted = issuer_result.success;

        // Signature verification (simplified)
        result.checks.signature_valid = credential.proof.is_some() && issuer_result.success;

        // Schema validation
        result.checks.schema_valid = self.validate_credential_schema(credential);

        // Overall validity
        result.valid = result.checks.signature_valid
            && result.checks.not_expired
            && result.checks.not_revoked
            && result.checks.issuer_trusted
            && result.checks.schema_valid;

        if result.valid {
            result.assurance_level = Some(self.derive_assurance_level(credential));
        }

        Ok(result)
    }

    /// Check revocation status of a credential
    pub async fn check_revocation_status(
        &self,
        credential_id: &str,
    ) -> IdentityResult<RevocationStatus> {
        // Check cache
        if self.config.enable_cache {
            let cache = self.revocation_cache.read().await;
            if let Some(entry) = cache.get(credential_id) {
                if entry.is_valid() {
                    return Ok(entry.value.clone());
                }
            }
        }

        // In actual implementation, would call revocation zome
        // For now, return active (permissive default)
        let status = RevocationStatus {
            credential_id: credential_id.to_string(),
            status: RevocationState::Active,
            reason: None,
            checked_at: current_timestamp(),
        };

        // Cache the result
        if self.config.enable_cache {
            let mut cache = self.revocation_cache.write().await;
            let ttl = Duration::from_secs(self.config.cache_ttl_secs);
            cache.insert(credential_id.to_string(), CacheEntry::new(status.clone(), ttl));
        }

        Ok(status)
    }

    // =========================================================================
    // Assurance Level
    // =========================================================================

    /// Get the assurance level for a DID
    pub async fn get_assurance_level(&self, did: &str) -> IdentityResult<AssuranceLevel> {
        let available = self.check_connection().await;

        if !available {
            match self.config.fallback_mode {
                FallbackMode::Error => {
                    return Err(IdentityError::IdentityHappUnavailable);
                }
                _ => {
                    return Ok(AssuranceLevel::E0);
                }
            }
        }

        // Resolve DID
        let resolution = self.resolve_did(did).await?;
        if !resolution.success {
            return Ok(AssuranceLevel::E0);
        }

        // In actual implementation, would query credentials for this DID
        // For now, return E0
        Ok(AssuranceLevel::E0)
    }

    /// Check if a DID meets a minimum assurance level
    pub async fn meets_assurance_level(
        &self,
        did: &str,
        min_level: AssuranceLevel,
    ) -> IdentityResult<bool> {
        let current = self.get_assurance_level(did).await?;
        Ok(current >= min_level)
    }

    /// Verify identity for high-value transactions
    pub async fn verify_for_high_value_transaction(
        &self,
        did: &str,
        config: &HighValueTransactionConfig,
    ) -> IdentityResult<IdentityVerificationResponse> {
        let verification = self
            .verify_identity(IdentityVerificationRequest {
                did: did.to_string(),
                min_assurance_level: Some(config.required_assurance_level),
                required_credentials: config.required_credentials.clone(),
                source_happ: "marketplace".to_string(),
            })
            .await?;

        // Check if meets requirements
        if verification.assurance_level < config.required_assurance_level {
            return Err(IdentityError::InsufficientAssuranceLevel {
                required: config.required_assurance_level,
                actual: verification.assurance_level,
            });
        }

        Ok(verification)
    }

    // =========================================================================
    // Cross-hApp Verification
    // =========================================================================

    /// Verify identity for cross-hApp operations
    pub async fn verify_identity(
        &self,
        request: IdentityVerificationRequest,
    ) -> IdentityResult<IdentityVerificationResponse> {
        let now = current_timestamp();
        let available = self.check_connection().await;

        if !available {
            match self.config.fallback_mode {
                FallbackMode::Error => {
                    return Err(IdentityError::IdentityHappUnavailable);
                }
                _ => {
                    return Ok(IdentityVerificationResponse {
                        id: format!("verify_{}", now),
                        did: request.did,
                        is_valid: false,
                        is_deactivated: false,
                        assurance_level: AssuranceLevel::E0,
                        matl_score: 0.5,
                        credential_count: 0,
                        did_created: None,
                        verified_at: now,
                    });
                }
            }
        }

        // Resolve DID
        let resolution = self.resolve_did(&request.did).await?;
        let assurance = self.get_assurance_level(&request.did).await?;
        let reputation = self.get_cross_happ_reputation(&request.did).await.ok();

        Ok(IdentityVerificationResponse {
            id: format!("verify_{}", now),
            did: request.did,
            is_valid: resolution.success,
            is_deactivated: false,
            assurance_level: assurance,
            matl_score: reputation.map(|r| r.aggregate_score).unwrap_or(0.5),
            credential_count: 0,
            did_created: resolution.did_document.map(|d| d.created),
            verified_at: now,
        })
    }

    /// Get cross-hApp reputation for a DID
    pub async fn get_cross_happ_reputation(
        &self,
        did: &str,
    ) -> IdentityResult<CrossHappReputation> {
        let available = self.check_connection().await;

        if !available {
            match self.config.fallback_mode {
                FallbackMode::Error => {
                    return Err(IdentityError::IdentityHappUnavailable);
                }
                _ => {
                    return Ok(CrossHappReputation {
                        did: did.to_string(),
                        scores: vec![],
                        aggregate_score: 0.5,
                        last_updated: current_timestamp(),
                    });
                }
            }
        }

        // In actual implementation, would call identity_bridge zome
        Ok(CrossHappReputation {
            did: did.to_string(),
            scores: vec![],
            aggregate_score: 0.5,
            last_updated: current_timestamp(),
        })
    }

    // =========================================================================
    // Guardian Endorsement
    // =========================================================================

    /// Check guardian endorsements for a DID
    ///
    /// Returns the list of guardians and their endorsement status.
    /// Guardians are used for social recovery and identity verification.
    pub async fn check_guardian_endorsement(
        &self,
        did: &str,
    ) -> IdentityResult<Vec<Guardian>> {
        let available = self.check_connection().await;

        if !available {
            match self.config.fallback_mode {
                FallbackMode::Error => {
                    return Err(IdentityError::IdentityHappUnavailable);
                }
                FallbackMode::Warn => {
                    warn!("Identity hApp not available for guardian check: {}", did);
                }
                FallbackMode::Silent => {}
            }
            return Ok(vec![]);
        }

        // In actual implementation, would call guardian_registry zome
        // For now, return empty list
        debug!("Checking guardian endorsements for: {}", did);
        Ok(vec![])
    }

    /// Get full guardian endorsement result with statistics
    pub async fn get_guardian_endorsement_result(
        &self,
        did: &str,
    ) -> IdentityResult<GuardianEndorsementResult> {
        let guardians = self.check_guardian_endorsement(did).await?;

        let endorsement_count = guardians.iter().filter(|g| g.has_endorsed).count() as u32;
        let total_guardians = guardians.len() as u32;

        // Recovery typically requires 2/3 of guardians
        let meets_threshold = if total_guardians > 0 {
            endorsement_count * 3 >= total_guardians * 2
        } else {
            false
        };

        // Calculate aggregate trust from endorsed guardians
        let aggregate_trust = if endorsement_count > 0 {
            guardians
                .iter()
                .filter(|g| g.has_endorsed)
                .map(|g| g.trust_weight)
                .sum::<f64>() / endorsement_count as f64
        } else {
            0.0
        };

        Ok(GuardianEndorsementResult {
            did: did.to_string(),
            guardians,
            endorsement_count,
            total_guardians,
            meets_recovery_threshold: meets_threshold,
            aggregate_trust,
            checked_at: current_timestamp(),
        })
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Derive assurance level from a credential
    fn derive_assurance_level(&self, credential: &VerifiableCredential) -> AssuranceLevel {
        let types = &credential.type_;

        if types.iter().any(|t| t == "BiometricCredential" || t == "MultifactorCredential") {
            return AssuranceLevel::E4;
        }
        if types.iter().any(|t| t == "GovernmentIdCredential" || t == "KYCCredential") {
            return AssuranceLevel::E3;
        }
        if types.iter().any(|t| t == "PhoneCredential" || t == "RecoveryConfigured") {
            return AssuranceLevel::E2;
        }
        if types.iter().any(|t| t == "EmailCredential") {
            return AssuranceLevel::E1;
        }

        AssuranceLevel::E0
    }

    /// Validate credential schema (simplified)
    fn validate_credential_schema(&self, credential: &VerifiableCredential) -> bool {
        !credential.context.is_empty()
            && !credential.id.is_empty()
            && !credential.type_.is_empty()
            && !credential.issuer.is_empty()
            && !credential.issuance_date.is_empty()
    }

    /// Clear all caches
    pub async fn clear_cache(&self) {
        let mut did_cache = self.did_cache.write().await;
        let mut rev_cache = self.revocation_cache.write().await;
        did_cache.clear();
        rev_cache.clear();
        debug!("Identity client caches cleared");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let did_cache = self.did_cache.read().await;
        let rev_cache = self.revocation_cache.read().await;
        (did_cache.len(), rev_cache.len())
    }
}

// Placeholder for timestamp (would use actual time in production)
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// Placeholder for date formatting
fn chrono_placeholder_now() -> String {
    // ISO 8601 format placeholder
    "2026-01-08T00:00:00Z".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_did_format() {
        let client = IdentityClient::with_defaults();
        let result = client.resolve_did("invalid:did:format").await.unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_assurance_level_comparison() {
        assert!(AssuranceLevel::E4 > AssuranceLevel::E0);
        assert!(AssuranceLevel::E3 > AssuranceLevel::E2);
        assert_eq!(AssuranceLevel::E2.value(), 2);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let client = IdentityClient::with_defaults();
        let (did_count, rev_count) = client.cache_stats().await;
        assert_eq!(did_count, 0);
        assert_eq!(rev_count, 0);
    }
}

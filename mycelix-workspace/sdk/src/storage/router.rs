// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UESS Storage Router
//!
//! Routes data to appropriate storage backends based on E/N/M classification.
//! Implements INV-2 (monotonic classification transitions).

use super::types::{AccessControlMode, MutabilityMode, StorageBackend, StorageTier};
use crate::epistemic::{EmpiricalLevel, EpistemicClassification, MaterialityLevel, NormativeLevel};
use serde::{Deserialize, Serialize};

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// TTL for M0 (ephemeral) data in milliseconds
    pub m0_ttl_ms: u64,
    /// TTL for M1 (temporal) data in milliseconds
    pub m1_ttl_ms: u64,
    /// Default replication factor for DHT
    pub dht_default_replication: usize,
    /// Minimum replication for M3 (immutable)
    pub m3_min_replication: usize,
    /// Maximum replication for M3
    pub m3_max_replication: usize,
    /// Whether IPFS backend is enabled
    pub enable_ipfs: bool,
    /// Whether Filecoin backend is enabled
    pub enable_filecoin: bool,
    /// Network size for replication calculation
    pub network_size: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            m0_ttl_ms: 3600 * 1000,          // 1 hour
            m1_ttl_ms: 7 * 24 * 3600 * 1000, // 7 days
            dht_default_replication: 10,
            m3_min_replication: 5,
            m3_max_replication: 30,
            enable_ipfs: true,
            enable_filecoin: false,
            network_size: 100,
        }
    }
}

/// Storage router for E/N/M classification → tier mapping
#[derive(Debug, Clone)]
pub struct StorageRouter {
    config: RouterConfig,
}

impl StorageRouter {
    /// Create a new router with configuration
    pub fn new(config: RouterConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_router() -> Self {
        Self::new(RouterConfig::default())
    }

    /// Route classification to storage tier
    ///
    /// This is the core routing logic that maps E/N/M levels to storage behavior.
    pub fn route(&self, classification: &EpistemicClassification) -> StorageTier {
        // 1. Backend from Materiality
        let (backend, additional_backends) =
            self.backend_from_materiality(classification.materiality);

        // 2. Mutability from Empirical
        let mutability = self.mutability_from_empirical(classification.empirical);

        // 3. Access control from Normative
        let access_control = self.access_control_from_normative(classification.normative);

        // 4. Encryption required for N0/N1
        let encrypted = matches!(
            classification.normative,
            NormativeLevel::N0Personal | NormativeLevel::N1Communal
        );

        // 5. Content addressing for E3+
        let content_addressed = matches!(
            classification.empirical,
            EmpiricalLevel::E3Cryptographic | EmpiricalLevel::E4PublicRepro
        );

        // 6. TTL from Materiality
        let ttl_ms = self.ttl_from_materiality(classification.materiality);

        // 7. Replication factor
        let replication = self.replication_factor(classification);

        StorageTier {
            backend,
            additional_backends,
            replication,
            mutability,
            access_control,
            ttl_ms,
            encrypted,
            content_addressed,
        }
    }

    /// Determine backend from Materiality level
    fn backend_from_materiality(
        &self,
        materiality: MaterialityLevel,
    ) -> (StorageBackend, Vec<StorageBackend>) {
        match materiality {
            MaterialityLevel::M0Ephemeral => (StorageBackend::Memory, vec![]),
            MaterialityLevel::M1Temporal => (StorageBackend::Local, vec![]),
            MaterialityLevel::M2Persistent => (StorageBackend::DHT, vec![StorageBackend::Local]),
            MaterialityLevel::M3Foundational => {
                if self.config.enable_ipfs {
                    if self.config.enable_filecoin {
                        (
                            StorageBackend::IPFS,
                            vec![StorageBackend::Filecoin, StorageBackend::DHT],
                        )
                    } else {
                        (StorageBackend::IPFS, vec![StorageBackend::DHT])
                    }
                } else {
                    (StorageBackend::DHT, vec![StorageBackend::Local])
                }
            }
        }
    }

    /// Determine mutability mode from Empirical level
    fn mutability_from_empirical(&self, empirical: EmpiricalLevel) -> MutabilityMode {
        match empirical {
            EmpiricalLevel::E0Null | EmpiricalLevel::E1Testimonial => MutabilityMode::MutableCRDT,
            EmpiricalLevel::E2PrivateVerify => MutabilityMode::AppendOnly,
            EmpiricalLevel::E3Cryptographic | EmpiricalLevel::E4PublicRepro => {
                MutabilityMode::Immutable
            }
        }
    }

    /// Determine access control from Normative level
    fn access_control_from_normative(&self, normative: NormativeLevel) -> AccessControlMode {
        match normative {
            NormativeLevel::N0Personal => AccessControlMode::Owner,
            NormativeLevel::N1Communal => AccessControlMode::CapBAC,
            NormativeLevel::N2Network | NormativeLevel::N3Axiomatic => AccessControlMode::Public,
        }
    }

    /// Determine TTL from Materiality level
    fn ttl_from_materiality(&self, materiality: MaterialityLevel) -> Option<u64> {
        match materiality {
            MaterialityLevel::M0Ephemeral => Some(self.config.m0_ttl_ms),
            MaterialityLevel::M1Temporal => Some(self.config.m1_ttl_ms),
            MaterialityLevel::M2Persistent | MaterialityLevel::M3Foundational => None,
        }
    }

    /// Calculate replication factor
    fn replication_factor(&self, classification: &EpistemicClassification) -> usize {
        match classification.materiality {
            MaterialityLevel::M0Ephemeral => 0, // No replication for ephemeral
            MaterialityLevel::M1Temporal => 1,  // Local only
            MaterialityLevel::M2Persistent => {
                match classification.normative {
                    NormativeLevel::N0Personal => 3, // Moderate for personal
                    NormativeLevel::N1Communal => 5, // Higher for communal
                    _ => self.config.dht_default_replication,
                }
            }
            MaterialityLevel::M3Foundational => {
                // Logarithmic replication based on network size
                // Formula: ceil(log2(network_size)), bounded by [min, max]
                let log_replication = (self.config.network_size as f64).log2().ceil() as usize;
                log_replication.clamp(
                    self.config.m3_min_replication,
                    self.config.m3_max_replication,
                )
            }
        }
    }

    /// Validate a classification transition (INV-2: monotonic only)
    ///
    /// Returns Ok(()) if transition is valid, Err with reason if not.
    pub fn validate_transition(
        &self,
        from: &EpistemicClassification,
        to: &EpistemicClassification,
    ) -> Result<(), TransitionError> {
        // Empirical level must not decrease
        if (to.empirical as u8) < (from.empirical as u8) {
            return Err(TransitionError::EmpiricalDowngrade {
                from: from.empirical,
                to: to.empirical,
            });
        }

        // Normative level must not decrease
        if (to.normative as u8) < (from.normative as u8) {
            return Err(TransitionError::NormativeDowngrade {
                from: from.normative,
                to: to.normative,
            });
        }

        // Materiality level must not decrease
        if (to.materiality as u8) < (from.materiality as u8) {
            return Err(TransitionError::MaterialityDowngrade {
                from: from.materiality,
                to: to.materiality,
            });
        }

        Ok(())
    }

    /// Check if a transition requires migration (backend/encryption change)
    pub fn requires_migration(
        &self,
        from: &EpistemicClassification,
        to: &EpistemicClassification,
    ) -> bool {
        let from_tier = self.route(from);
        let to_tier = self.route(to);

        // Migration required if:
        // 1. Backend changes
        // 2. Encryption status changes
        // 3. Mutability mode changes
        // 4. Content addressing changes
        from_tier.backend != to_tier.backend
            || from_tier.encrypted != to_tier.encrypted
            || from_tier.mutability != to_tier.mutability
            || from_tier.content_addressed != to_tier.content_addressed
    }

    /// Get configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Update network size (for replication calculation)
    pub fn set_network_size(&mut self, size: usize) {
        self.config.network_size = size;
    }
}

/// Classification transition error
#[derive(Debug, Clone, thiserror::Error)]
pub enum TransitionError {
    /// Attempted to downgrade empirical level (violates monotonicity)
    #[error("Cannot downgrade Empirical level from {from:?} to {to:?}")]
    EmpiricalDowngrade {
        /// Original empirical level
        from: EmpiricalLevel,
        /// Attempted downgraded level
        to: EmpiricalLevel,
    },
    /// Attempted to downgrade normative level (violates monotonicity)
    #[error("Cannot downgrade Normative level from {from:?} to {to:?}")]
    NormativeDowngrade {
        /// Original normative level
        from: NormativeLevel,
        /// Attempted downgraded level
        to: NormativeLevel,
    },
    /// Attempted to downgrade materiality level (violates monotonicity)
    #[error("Cannot downgrade Materiality level from {from:?} to {to:?}")]
    MaterialityDowngrade {
        /// Original materiality level
        from: MaterialityLevel,
        /// Attempted downgraded level
        to: MaterialityLevel,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classification(
        e: EmpiricalLevel,
        n: NormativeLevel,
        m: MaterialityLevel,
    ) -> EpistemicClassification {
        EpistemicClassification {
            empirical: e,
            normative: n,
            materiality: m,
        }
    }

    #[test]
    fn test_backend_from_materiality() {
        let router = StorageRouter::default_router();

        // M0 → Memory
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.backend, StorageBackend::Memory);

        // M1 → Local
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        ));
        assert_eq!(tier.backend, StorageBackend::Local);

        // M2 → DHT
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M2Persistent,
        ));
        assert_eq!(tier.backend, StorageBackend::DHT);

        // M3 → IPFS (if enabled)
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M3Foundational,
        ));
        assert_eq!(tier.backend, StorageBackend::IPFS);
    }

    #[test]
    fn test_mutability_from_empirical() {
        let router = StorageRouter::default_router();

        // E0-E1 → MutableCRDT
        let tier = router.route(&classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.mutability, MutabilityMode::MutableCRDT);

        // E2 → AppendOnly
        let tier = router.route(&classification(
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.mutability, MutabilityMode::AppendOnly);

        // E3-E4 → Immutable
        let tier = router.route(&classification(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.mutability, MutabilityMode::Immutable);
    }

    #[test]
    fn test_access_control_from_normative() {
        let router = StorageRouter::default_router();

        // N0 → Owner
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.access_control, AccessControlMode::Owner);

        // N1 → CapBAC
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N1Communal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.access_control, AccessControlMode::CapBAC);

        // N2-N3 → Public
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N2Network,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.access_control, AccessControlMode::Public);
    }

    #[test]
    fn test_encryption_from_normative() {
        let router = StorageRouter::default_router();

        // N0 → encrypted
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert!(tier.encrypted);

        // N1 → encrypted
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N1Communal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert!(tier.encrypted);

        // N2+ → not encrypted
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N2Network,
            MaterialityLevel::M0Ephemeral,
        ));
        assert!(!tier.encrypted);
    }

    #[test]
    fn test_content_addressing_from_empirical() {
        let router = StorageRouter::default_router();

        // E0-E2 → not content addressed
        let tier = router.route(&classification(
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert!(!tier.content_addressed);

        // E3-E4 → content addressed
        let tier = router.route(&classification(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert!(tier.content_addressed);
    }

    #[test]
    fn test_ttl_from_materiality() {
        let router = StorageRouter::default_router();

        // M0 → has TTL
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert!(tier.ttl_ms.is_some());

        // M2-M3 → no TTL
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M2Persistent,
        ));
        assert!(tier.ttl_ms.is_none());
    }

    #[test]
    fn test_replication_factor() {
        let router = StorageRouter::default_router();

        // M0 → 0 (no replication)
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
        ));
        assert_eq!(tier.replication, 0);

        // M2/N0 → moderate (3)
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M2Persistent,
        ));
        assert_eq!(tier.replication, 3);

        // M2/N1 → higher (5)
        let tier = router.route(&classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
        ));
        assert_eq!(tier.replication, 5);
    }

    #[test]
    fn test_transition_validation() {
        let router = StorageRouter::default_router();

        // Valid upgrade
        let from = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        let to = classification(
            EmpiricalLevel::E2PrivateVerify,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
        );
        assert!(router.validate_transition(&from, &to).is_ok());

        // Invalid: E downgrade
        let to_bad_e = classification(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        assert!(router.validate_transition(&from, &to_bad_e).is_err());

        // Invalid: N downgrade
        let from_n1 = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N1Communal,
            MaterialityLevel::M1Temporal,
        );
        let to_n0 = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        assert!(router.validate_transition(&from_n1, &to_n0).is_err());

        // Invalid: M downgrade
        let from_m2 = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M2Persistent,
        );
        let to_m1 = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        assert!(router.validate_transition(&from_m2, &to_m1).is_err());
    }

    #[test]
    fn test_requires_migration() {
        let router = StorageRouter::default_router();

        // Same classification → no migration
        let c = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        assert!(!router.requires_migration(&c, &c));

        // M1 → M2: backend change → migration
        let from = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        let to = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M2Persistent,
        );
        assert!(router.requires_migration(&from, &to));

        // N0 → N2: encryption change → migration
        let from = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        let to = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N2Network,
            MaterialityLevel::M1Temporal,
        );
        assert!(router.requires_migration(&from, &to));

        // E1 → E3: mutability change → migration
        let from = classification(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        let to = classification(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N0Personal,
            MaterialityLevel::M1Temporal,
        );
        assert!(router.requires_migration(&from, &to));
    }
}

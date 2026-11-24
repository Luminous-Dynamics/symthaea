// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy — Sharing tier management and data filtering

/// Coarse user-configured sharing preference.
///
/// This is not proof that content is anonymous, scrubbed, consented, or safe
/// to disclose. A separate data-classification and privacy-validation step
/// must run before any customer data leaves its tenant boundary.
#[derive(Debug)]
pub struct PrivacyManager {
    sharing_tier: SharingTier,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SharingTier {
    /// No cross-tenant sharing. This is the default.
    LocalOnly,
    /// Sharing may be considered only after an independent anonymization and
    /// disclosure review; this label alone does not establish anonymity.
    Anonymized,
    /// Explicit full-sharing preference. Still subject to tenant consent,
    /// contract, data classification, and destination authorization.
    Full,
}

impl PrivacyManager {
    pub fn new(tier: SharingTier) -> Self {
        Self { sharing_tier: tier }
    }

    pub fn sharing_tier(&self) -> &SharingTier {
        &self.sharing_tier
    }

    pub fn set_sharing_tier(&mut self, tier: SharingTier) {
        self.sharing_tier = tier;
    }

    /// Check whether the configured preference permits considering sharing.
    ///
    /// A true result is not a privacy guarantee or an authorization token.
    pub fn can_share(&self) -> bool {
        !matches!(self.sharing_tier, SharingTier::LocalOnly)
    }

    /// Check whether the configured preference permits considering cognitive
    /// sharing. Callers must independently scrub, classify, and authorize the
    /// exact payload before transmission.
    pub fn can_share_cognitive(&self) -> bool {
        matches!(
            self.sharing_tier,
            SharingTier::Anonymized | SharingTier::Full
        )
    }
}

impl Default for PrivacyManager {
    fn default() -> Self {
        Self::new(SharingTier::LocalOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_only_blocks_sharing() {
        let pm = PrivacyManager::new(SharingTier::LocalOnly);
        assert!(!pm.can_share());
    }

    #[test]
    fn anonymized_allows_sharing() {
        let pm = PrivacyManager::new(SharingTier::Anonymized);
        assert!(pm.can_share());
    }

    #[test]
    fn full_allows_sharing() {
        let pm = PrivacyManager::new(SharingTier::Full);
        assert!(pm.can_share());
    }

    #[test]
    fn can_share_cognitive_works() {
        let local = PrivacyManager::new(SharingTier::LocalOnly);
        let anon = PrivacyManager::new(SharingTier::Anonymized);
        let full = PrivacyManager::new(SharingTier::Full);

        assert!(!local.can_share_cognitive());
        assert!(anon.can_share_cognitive());
        assert!(full.can_share_cognitive());
    }

    #[test]
    fn default_is_local_only() {
        let pm = PrivacyManager::default();
        assert_eq!(*pm.sharing_tier(), SharingTier::LocalOnly);
        assert!(!pm.can_share());
        assert!(!pm.can_share_cognitive());
    }

    #[test]
    fn set_sharing_tier_changes_tier() {
        let mut pm = PrivacyManager::new(SharingTier::LocalOnly);
        assert!(!pm.can_share());
        pm.set_sharing_tier(SharingTier::Full);
        assert!(pm.can_share());
    }
}

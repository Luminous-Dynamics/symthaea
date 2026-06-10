// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy — Sharing tier management and data filtering

/// Privacy manager for controlling what data can be shared.
#[derive(Debug)]
pub struct PrivacyManager {
    sharing_tier: SharingTier,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SharingTier {
    LocalOnly,
    Anonymized,
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

    /// Check if data can be shared at the current tier.
    pub fn can_share(&self) -> bool {
        !matches!(self.sharing_tier, SharingTier::LocalOnly)
    }

    /// Check if cognitive updates can be shared.
    pub fn can_share_cognitive(&self) -> bool {
        matches!(
            self.sharing_tier,
            SharingTier::Anonymized | SharingTier::Full
        )
    }
}

impl Default for PrivacyManager {
    fn default() -> Self {
        Self::new(SharingTier::Anonymized)
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
    fn default_is_anonymized() {
        let pm = PrivacyManager::default();
        assert_eq!(*pm.sharing_tier(), SharingTier::Anonymized);
    }

    #[test]
    fn set_sharing_tier_changes_tier() {
        let mut pm = PrivacyManager::new(SharingTier::LocalOnly);
        assert!(!pm.can_share());
        pm.set_sharing_tier(SharingTier::Full);
        assert!(pm.can_share());
    }
}

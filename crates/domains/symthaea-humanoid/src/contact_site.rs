// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed canonical site identities shared by morphology, dynamics, capability,
//! and skill qualification.
//!
//! The MuJoCo backend and procedural MJCF already use a stable eight-site
//! vocabulary. Keeping those identities as raw strings at policy boundaries
//! makes typo/drift failures unnecessarily late. This module centralizes the
//! vocabulary without deciding which sites or wrench axes any task must use;
//! those requirements remain explicit qualification policy.

use serde::{Deserialize, Serialize};

pub const HUMANOID_CONTACT_SITE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HumanoidContactSite {
    RightFoot,
    LeftFoot,
    RightHand,
    LeftHand,
    RightKnee,
    LeftKnee,
    RightForearm,
    LeftForearm,
}

impl HumanoidContactSite {
    pub const ALL: [Self; 8] = [
        Self::RightFoot,
        Self::LeftFoot,
        Self::RightHand,
        Self::LeftHand,
        Self::RightKnee,
        Self::LeftKnee,
        Self::RightForearm,
        Self::LeftForearm,
    ];

    pub const fn canonical_id(self) -> &'static str {
        match self {
            Self::RightFoot => "r_foot_site",
            Self::LeftFoot => "l_foot_site",
            Self::RightHand => "r_hand_site",
            Self::LeftHand => "l_hand_site",
            Self::RightKnee => "r_knee_site",
            Self::LeftKnee => "l_knee_site",
            Self::RightForearm => "r_forearm_site",
            Self::LeftForearm => "l_forearm_site",
        }
    }

    pub const fn side(self) -> HumanoidBodySide {
        match self {
            Self::RightFoot | Self::RightHand | Self::RightKnee | Self::RightForearm => {
                HumanoidBodySide::Right
            }
            Self::LeftFoot | Self::LeftHand | Self::LeftKnee | Self::LeftForearm => {
                HumanoidBodySide::Left
            }
        }
    }

    pub const fn region(self) -> HumanoidContactRegion {
        match self {
            Self::RightFoot | Self::LeftFoot => HumanoidContactRegion::Foot,
            Self::RightHand | Self::LeftHand => HumanoidContactRegion::Hand,
            Self::RightKnee | Self::LeftKnee => HumanoidContactRegion::Knee,
            Self::RightForearm | Self::LeftForearm => HumanoidContactRegion::Forearm,
        }
    }

    pub fn parse_canonical(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|site| site.canonical_id() == value)
    }
}

impl std::fmt::Display for HumanoidContactSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.canonical_id())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HumanoidBodySide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HumanoidContactRegion {
    Foot,
    Hand,
    Knee,
    Forearm,
}

/// Structural site inventory emitted by an embodiment/dynamics backend.
///
/// This is not contact-state evidence: a listed site need not currently be in
/// physical contact. It only proves that the backend exposes a Jacobian-bearing
/// site with the canonical identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidContactSiteInventory {
    pub schema_version: u32,
    pub backend_profile_id: String,
    pub sites: Vec<HumanoidContactSite>,
}

impl HumanoidContactSiteInventory {
    pub fn new(
        backend_profile_id: impl Into<String>,
        mut sites: Vec<HumanoidContactSite>,
    ) -> Self {
        sites.sort_unstable();
        sites.dedup();
        Self {
            schema_version: HUMANOID_CONTACT_SITE_SCHEMA_VERSION,
            backend_profile_id: backend_profile_id.into(),
            sites,
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_CONTACT_SITE_SCHEMA_VERSION
            && !self.backend_profile_id.trim().is_empty()
            && self.backend_profile_id == self.backend_profile_id.trim()
            && self.backend_profile_id.len() <= 256
            && !self.sites.is_empty()
            && self
                .sites
                .windows(2)
                .all(|pair| pair[0] < pair[1])
    }

    pub fn contains(&self, site: HumanoidContactSite) -> bool {
        self.sites.binary_search(&site).is_ok()
    }

    /// Build an inventory from runtime site ids. Any unknown identity fails
    /// closed rather than being silently ignored.
    pub fn from_runtime_ids(
        backend_profile_id: impl Into<String>,
        site_ids: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, HumanoidContactSiteInventoryError> {
        let mut sites = Vec::new();
        for raw in site_ids {
            let raw = raw.as_ref();
            let Some(site) = HumanoidContactSite::parse_canonical(raw) else {
                return Err(HumanoidContactSiteInventoryError::UnknownSite);
            };
            sites.push(site);
        }
        let inventory = Self::new(backend_profile_id, sites);
        inventory
            .validate()
            .then_some(inventory)
            .ok_or(HumanoidContactSiteInventoryError::InvalidInventory)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidContactSiteInventoryError {
    UnknownSite,
    InvalidInventory,
}

/// Convert a typed site to the existing dynamics/policy wire identity.
/// Keeping this conversion in one place avoids duplicated spelling constants.
pub const fn canonical_contact_site_id(site: HumanoidContactSite) -> &'static str {
    site.canonical_id()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_ids_round_trip() {
        for site in HumanoidContactSite::ALL {
            assert_eq!(HumanoidContactSite::parse_canonical(site.canonical_id()), Some(site));
        }
    }

    #[test]
    fn canonical_site_ids_are_unique() {
        for (index, site) in HumanoidContactSite::ALL.iter().enumerate() {
            assert!(!HumanoidContactSite::ALL[..index]
                .iter()
                .any(|previous| previous.canonical_id() == site.canonical_id()));
        }
    }

    #[test]
    fn inventory_sorts_and_deduplicates() {
        let inventory = HumanoidContactSiteInventory::new(
            "backend-v1",
            vec![
                HumanoidContactSite::LeftHand,
                HumanoidContactSite::RightFoot,
                HumanoidContactSite::LeftHand,
            ],
        );
        assert!(inventory.validate());
        assert_eq!(inventory.sites.len(), 2);
        assert!(inventory.contains(HumanoidContactSite::LeftHand));
        assert!(inventory.contains(HumanoidContactSite::RightFoot));
    }

    #[test]
    fn unknown_runtime_site_fails_closed() {
        assert_eq!(
            HumanoidContactSiteInventory::from_runtime_ids(
                "backend-v1",
                ["r_foot_site", "mystery_site"],
            ),
            Err(HumanoidContactSiteInventoryError::UnknownSite)
        );
    }

    #[test]
    fn sides_and_regions_are_structural_only() {
        assert_eq!(HumanoidContactSite::RightHand.side(), HumanoidBodySide::Right);
        assert_eq!(HumanoidContactSite::LeftFoot.side(), HumanoidBodySide::Left);
        assert_eq!(HumanoidContactSite::RightKnee.region(), HumanoidContactRegion::Knee);
        assert_eq!(
            HumanoidContactSite::LeftForearm.region(),
            HumanoidContactRegion::Forearm
        );
    }
}

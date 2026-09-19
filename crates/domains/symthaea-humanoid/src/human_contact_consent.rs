// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scoped human-contact consent semantics.
//!
//! This module is deliberately non-actuating. It represents what intentional
//! human contact has been explicitly scoped as permissible; it does not grant
//! motor authority, prove authentic consent, establish physical qualification,
//! or define contact-force limits.

use std::collections::BTreeSet;

/// Schema identity for the first scoped human-contact consent contract.
pub const HUMAN_CONTACT_CONSENT_SCOPE_SCHEMA_V1: &str =
    "symthaea.humanoid.human-contact-consent-scope.v1";

/// Coarse safety/authority-relevant contact class.
///
/// The enum is intentionally not a technique taxonomy. Increasing sensitivity
/// must be established by a fresh explicit scope rather than inferred from a
/// previous class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HumanContactClass {
    Social,
    Assistive,
    Therapeutic,
    AdultIntimate,
}

/// Coarse, non-graphic human body regions for consent scoping.
///
/// Unknown/unspecified regions are represented by absence from the permitted
/// set, not by an `Unknown` variant that might later be interpreted as allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HumanBodyRegion {
    Head,
    Face,
    Neck,
    Shoulder,
    Arm,
    Hand,
    UpperTorso,
    LowerTorso,
    Back,
    Hip,
    Leg,
    Foot,
    Intimate,
}

/// Stable semantic identity for a robot surface/site proposed for human contact.
///
/// Site eligibility is a separate qualification proposition. A site appearing
/// here means only that the participant's consent scope includes it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HumanContactRobotSite(String);

impl HumanContactRobotSite {
    pub fn new(value: impl Into<String>) -> Result<Self, HumanContactConsentError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.len() > 128 {
            return Err(HumanContactConsentError::InvalidRobotSite);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validation/narrowing failures for scoped consent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanContactConsentError {
    EmptyParticipant,
    EmptySession,
    InvalidEpoch,
    InvalidValidityWindow,
    EmptyHumanRegions,
    EmptyRobotSites,
    InvalidRobotSite,
    ParticipantMismatch,
    SessionMismatch,
    ContactClassMismatch,
    EpochMismatch,
    EmptyIntersection,
}

/// Explicitly scoped consent for intentional human contact.
///
/// This value is semantic evidence only. It is intentionally Clone-able for
/// planning/testing and must never be treated as a one-shot execution permit.
/// Later authority layers must bind an authenticated consent decision and a live
/// validation epoch before physical eligibility exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanContactConsentScopeV1 {
    participant_id: String,
    session_id: String,
    consent_epoch: u64,
    revocation_epoch: u64,
    contact_class: HumanContactClass,
    permitted_human_regions: BTreeSet<HumanBodyRegion>,
    permitted_robot_sites: BTreeSet<HumanContactRobotSite>,
    valid_from_ns: u64,
    valid_until_ns: u64,
}

impl HumanContactConsentScopeV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        participant_id: impl Into<String>,
        session_id: impl Into<String>,
        consent_epoch: u64,
        revocation_epoch: u64,
        contact_class: HumanContactClass,
        permitted_human_regions: impl IntoIterator<Item = HumanBodyRegion>,
        permitted_robot_sites: impl IntoIterator<Item = HumanContactRobotSite>,
        valid_from_ns: u64,
        valid_until_ns: u64,
    ) -> Result<Self, HumanContactConsentError> {
        let participant_id = participant_id.into().trim().to_owned();
        let session_id = session_id.into().trim().to_owned();
        if participant_id.is_empty() {
            return Err(HumanContactConsentError::EmptyParticipant);
        }
        if session_id.is_empty() {
            return Err(HumanContactConsentError::EmptySession);
        }
        if consent_epoch == 0 || revocation_epoch > consent_epoch {
            return Err(HumanContactConsentError::InvalidEpoch);
        }
        if valid_until_ns <= valid_from_ns {
            return Err(HumanContactConsentError::InvalidValidityWindow);
        }
        let permitted_human_regions = permitted_human_regions.into_iter().collect();
        if permitted_human_regions.is_empty() {
            return Err(HumanContactConsentError::EmptyHumanRegions);
        }
        let permitted_robot_sites = permitted_robot_sites.into_iter().collect();
        if permitted_robot_sites.is_empty() {
            return Err(HumanContactConsentError::EmptyRobotSites);
        }

        Ok(Self {
            participant_id,
            session_id,
            consent_epoch,
            revocation_epoch,
            contact_class,
            permitted_human_regions,
            permitted_robot_sites,
            valid_from_ns,
            valid_until_ns,
        })
    }

    pub fn participant_id(&self) -> &str {
        &self.participant_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn consent_epoch(&self) -> u64 {
        self.consent_epoch
    }

    pub const fn revocation_epoch(&self) -> u64 {
        self.revocation_epoch
    }

    pub const fn contact_class(&self) -> HumanContactClass {
        self.contact_class
    }

    pub fn permitted_human_regions(&self) -> &BTreeSet<HumanBodyRegion> {
        &self.permitted_human_regions
    }

    pub fn permitted_robot_sites(&self) -> &BTreeSet<HumanContactRobotSite> {
        &self.permitted_robot_sites
    }

    pub const fn valid_from_ns(&self) -> u64 {
        self.valid_from_ns
    }

    pub const fn valid_until_ns(&self) -> u64 {
        self.valid_until_ns
    }

    /// Whether this semantic scope is currently live with respect to its own
    /// time/epoch fields. This does not authenticate the scope or grant motion.
    pub fn is_live_at(&self, now_ns: u64) -> bool {
        self.revocation_epoch < self.consent_epoch
            && now_ns >= self.valid_from_ns
            && now_ns < self.valid_until_ns
    }

    /// Exact allowlist check; absence never implies permission.
    pub fn permits(
        &self,
        human_region: HumanBodyRegion,
        robot_site: &HumanContactRobotSite,
        contact_class: HumanContactClass,
        now_ns: u64,
    ) -> bool {
        self.is_live_at(now_ns)
            && contact_class == self.contact_class
            && self.permitted_human_regions.contains(&human_region)
            && self.permitted_robot_sites.contains(robot_site)
    }

    /// Narrow two scopes by intersection.
    ///
    /// This operation cannot widen participant/session/class/epoch/time/regions
    /// or robot sites. Class escalation is deliberately not represented as an
    /// ordering: different classes require fresh explicit authority.
    pub fn intersect(&self, other: &Self) -> Result<Self, HumanContactConsentError> {
        if self.participant_id != other.participant_id {
            return Err(HumanContactConsentError::ParticipantMismatch);
        }
        if self.session_id != other.session_id {
            return Err(HumanContactConsentError::SessionMismatch);
        }
        if self.contact_class != other.contact_class {
            return Err(HumanContactConsentError::ContactClassMismatch);
        }
        if self.consent_epoch != other.consent_epoch
            || self.revocation_epoch != other.revocation_epoch
        {
            return Err(HumanContactConsentError::EpochMismatch);
        }

        let permitted_human_regions = self
            .permitted_human_regions
            .intersection(&other.permitted_human_regions)
            .copied()
            .collect::<BTreeSet<_>>();
        let permitted_robot_sites = self
            .permitted_robot_sites
            .intersection(&other.permitted_robot_sites)
            .cloned()
            .collect::<BTreeSet<_>>();
        let valid_from_ns = self.valid_from_ns.max(other.valid_from_ns);
        let valid_until_ns = self.valid_until_ns.min(other.valid_until_ns);

        if permitted_human_regions.is_empty()
            || permitted_robot_sites.is_empty()
            || valid_until_ns <= valid_from_ns
        {
            return Err(HumanContactConsentError::EmptyIntersection);
        }

        Self::new(
            self.participant_id.clone(),
            self.session_id.clone(),
            self.consent_epoch,
            self.revocation_epoch,
            self.contact_class,
            permitted_human_regions,
            permitted_robot_sites,
            valid_from_ns,
            valid_until_ns,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(name: &str) -> HumanContactRobotSite {
        HumanContactRobotSite::new(name).unwrap()
    }

    fn scope(
        regions: impl IntoIterator<Item = HumanBodyRegion>,
        sites: impl IntoIterator<Item = HumanContactRobotSite>,
    ) -> HumanContactConsentScopeV1 {
        HumanContactConsentScopeV1::new(
            "participant-a",
            "session-a",
            2,
            1,
            HumanContactClass::Social,
            regions,
            sites,
            100,
            200,
        )
        .unwrap()
    }

    #[test]
    fn empty_participant_fails_closed() {
        let result = HumanContactConsentScopeV1::new(
            "",
            "session-a",
            1,
            0,
            HumanContactClass::Social,
            [HumanBodyRegion::Hand],
            [site("right_hand")],
            1,
            2,
        );
        assert_eq!(result, Err(HumanContactConsentError::EmptyParticipant));
    }

    #[test]
    fn expired_scope_does_not_permit_contact() {
        let s = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        assert!(!s.permits(
            HumanBodyRegion::Hand,
            &site("right_hand"),
            HumanContactClass::Social,
            200
        ));
    }

    #[test]
    fn revoked_epoch_does_not_permit_contact() {
        let s = HumanContactConsentScopeV1::new(
            "participant-a",
            "session-a",
            2,
            2,
            HumanContactClass::Social,
            [HumanBodyRegion::Hand],
            [site("right_hand")],
            100,
            200,
        )
        .unwrap();
        assert!(!s.is_live_at(150));
    }

    #[test]
    fn absent_region_or_site_never_implies_permission() {
        let s = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        assert!(!s.permits(
            HumanBodyRegion::Shoulder,
            &site("right_hand"),
            HumanContactClass::Social,
            150
        ));
        assert!(!s.permits(
            HumanBodyRegion::Hand,
            &site("left_hand"),
            HumanContactClass::Social,
            150
        ));
    }

    #[test]
    fn different_contact_class_cannot_be_inferred() {
        let s = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        assert!(!s.permits(
            HumanBodyRegion::Hand,
            &site("right_hand"),
            HumanContactClass::AdultIntimate,
            150
        ));
    }

    #[test]
    fn intersection_only_narrows_scope() {
        let broad = scope(
            [HumanBodyRegion::Hand, HumanBodyRegion::Shoulder],
            [site("right_hand"), site("left_hand")],
        );
        let narrow = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        let joined = broad.intersect(&narrow).unwrap();
        assert_eq!(joined.permitted_human_regions().len(), 1);
        assert_eq!(joined.permitted_robot_sites().len(), 1);
        assert!(joined.permitted_human_regions().contains(&HumanBodyRegion::Hand));
        assert!(joined.permitted_robot_sites().contains(&site("right_hand")));
    }

    #[test]
    fn disjoint_scope_has_no_authority() {
        let a = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        let b = scope([HumanBodyRegion::Shoulder], [site("left_hand")]);
        assert_eq!(
            a.intersect(&b),
            Err(HumanContactConsentError::EmptyIntersection)
        );
    }

    #[test]
    fn participant_substitution_fails_closed() {
        let a = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        let b = HumanContactConsentScopeV1::new(
            "participant-b",
            "session-a",
            2,
            1,
            HumanContactClass::Social,
            [HumanBodyRegion::Hand],
            [site("right_hand")],
            100,
            200,
        )
        .unwrap();
        assert_eq!(
            a.intersect(&b),
            Err(HumanContactConsentError::ParticipantMismatch)
        );
    }

    #[test]
    fn epoch_substitution_fails_closed() {
        let a = scope([HumanBodyRegion::Hand], [site("right_hand")]);
        let b = HumanContactConsentScopeV1::new(
            "participant-a",
            "session-a",
            3,
            1,
            HumanContactClass::Social,
            [HumanBodyRegion::Hand],
            [site("right_hand")],
            100,
            200,
        )
        .unwrap();
        assert_eq!(a.intersect(&b), Err(HumanContactConsentError::EpochMismatch));
    }
}

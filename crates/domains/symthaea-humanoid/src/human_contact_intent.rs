// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consent-bound human-contact intent semantics.
//!
//! This module is deliberately non-actuating. It binds one proposed intentional
//! human-contact target to the exact scoped consent currently admitted by the
//! human-contact authority session. The resulting value is not motor authority
//! and contains no joint command, controller gain, or production safety limit.

use crate::human_contact_consent::{
    HumanBodyRegion, HumanContactClass, HumanContactConsentScopeV1, HumanContactRobotSite,
};
use crate::human_contact_session::HumanContactAuthoritySessionV1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanContactIntentBindError {
    SessionNotActive,
    ExactScopeNotActive,
    ScopeDoesNotPermitContact,
    EmptySpatialGoalId,
    InvalidSpatialTarget,
}

/// One non-authoritative human-contact proposal bound to the exact live consent
/// scope that was admitted into the runtime lifecycle.
///
/// This type deliberately stores only semantic contact identity and spatial
/// intent. Physical force/pressure/temperature/velocity envelopes belong to
/// separately qualified safety policy and later whole-body lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsentBoundHumanContactIntentV1 {
    participant_id: String,
    session_id: String,
    consent_epoch: u64,
    contact_class: HumanContactClass,
    human_region: HumanBodyRegion,
    robot_site: HumanContactRobotSite,
    spatial_goal_id: String,
    target_root_m: [f64; 3],
}

impl ConsentBoundHumanContactIntentV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        session: &HumanContactAuthoritySessionV1,
        scope: &HumanContactConsentScopeV1,
        human_region: HumanBodyRegion,
        robot_site: &HumanContactRobotSite,
        spatial_goal_id: impl Into<String>,
        target_root_m: [f64; 3],
        now_ns: u64,
    ) -> Result<Self, HumanContactIntentBindError> {
        if !session.is_active() {
            return Err(HumanContactIntentBindError::SessionNotActive);
        }
        if !session.is_exact_scope_active(scope, now_ns) {
            return Err(HumanContactIntentBindError::ExactScopeNotActive);
        }
        if !scope.permits(human_region, robot_site, scope.contact_class(), now_ns) {
            return Err(HumanContactIntentBindError::ScopeDoesNotPermitContact);
        }
        let spatial_goal_id = spatial_goal_id.into().trim().to_owned();
        if spatial_goal_id.is_empty() || spatial_goal_id.len() > 256 {
            return Err(HumanContactIntentBindError::EmptySpatialGoalId);
        }
        if !target_root_m.iter().all(|value| value.is_finite()) {
            return Err(HumanContactIntentBindError::InvalidSpatialTarget);
        }

        Ok(Self {
            participant_id: scope.participant_id().to_owned(),
            session_id: scope.session_id().to_owned(),
            consent_epoch: scope.consent_epoch(),
            contact_class: scope.contact_class(),
            human_region,
            robot_site: robot_site.clone(),
            spatial_goal_id,
            target_root_m,
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

    pub const fn contact_class(&self) -> HumanContactClass {
        self.contact_class
    }

    pub const fn human_region(&self) -> HumanBodyRegion {
        self.human_region
    }

    pub fn robot_site(&self) -> &HumanContactRobotSite {
        &self.robot_site
    }

    pub fn spatial_goal_id(&self) -> &str {
        &self.spatial_goal_id
    }

    pub const fn target_root_m(&self) -> [f64; 3] {
        self.target_root_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::human_contact_session::HumanContactRevocationReason;

    fn site(name: &str) -> HumanContactRobotSite {
        HumanContactRobotSite::new(name).unwrap()
    }

    fn scope(
        epoch: u64,
        regions: impl IntoIterator<Item = HumanBodyRegion>,
        sites: impl IntoIterator<Item = HumanContactRobotSite>,
    ) -> HumanContactConsentScopeV1 {
        HumanContactConsentScopeV1::new(
            "participant-a",
            "session-a",
            epoch,
            epoch.saturating_sub(1),
            HumanContactClass::Social,
            regions,
            sites,
            100,
            300,
        )
        .unwrap()
    }

    #[test]
    fn exact_active_scope_can_bind_contact_intent() {
        let admitted = scope(2, [HumanBodyRegion::Hand], [site("right_hand")]);
        let mut session = HumanContactAuthoritySessionV1::new("participant-a", "session-a").unwrap();
        session.admit_scope(&admitted, 150).unwrap();

        let intent = ConsentBoundHumanContactIntentV1::bind(
            &session,
            &admitted,
            HumanBodyRegion::Hand,
            &site("right_hand"),
            "goal-1",
            [0.1, 0.2, 0.3],
            160,
        )
        .unwrap();

        assert_eq!(intent.consent_epoch(), 2);
        assert_eq!(intent.human_region(), HumanBodyRegion::Hand);
        assert_eq!(intent.robot_site(), &site("right_hand"));
    }

    #[test]
    fn same_epoch_broader_scope_substitution_fails_closed() {
        let admitted = scope(2, [HumanBodyRegion::Hand], [site("right_hand")]);
        let substituted = scope(
            2,
            [HumanBodyRegion::Hand, HumanBodyRegion::Shoulder],
            [site("right_hand"), site("left_hand")],
        );
        let mut session = HumanContactAuthoritySessionV1::new("participant-a", "session-a").unwrap();
        session.admit_scope(&admitted, 150).unwrap();

        assert_eq!(
            ConsentBoundHumanContactIntentV1::bind(
                &session,
                &substituted,
                HumanBodyRegion::Shoulder,
                &site("left_hand"),
                "goal-2",
                [0.0, 0.0, 0.0],
                160,
            ),
            Err(HumanContactIntentBindError::ExactScopeNotActive)
        );
    }

    #[test]
    fn region_or_site_outside_scope_fails_closed() {
        let admitted = scope(2, [HumanBodyRegion::Hand], [site("right_hand")]);
        let mut session = HumanContactAuthoritySessionV1::new("participant-a", "session-a").unwrap();
        session.admit_scope(&admitted, 150).unwrap();

        assert_eq!(
            ConsentBoundHumanContactIntentV1::bind(
                &session,
                &admitted,
                HumanBodyRegion::Shoulder,
                &site("right_hand"),
                "goal-3",
                [0.0, 0.0, 0.0],
                160,
            ),
            Err(HumanContactIntentBindError::ScopeDoesNotPermitContact)
        );
        assert_eq!(
            ConsentBoundHumanContactIntentV1::bind(
                &session,
                &admitted,
                HumanBodyRegion::Hand,
                &site("left_hand"),
                "goal-4",
                [0.0, 0.0, 0.0],
                160,
            ),
            Err(HumanContactIntentBindError::ScopeDoesNotPermitContact)
        );
    }

    #[test]
    fn revoked_session_cannot_bind_new_contact_intent() {
        let admitted = scope(2, [HumanBodyRegion::Hand], [site("right_hand")]);
        let mut session = HumanContactAuthoritySessionV1::new("participant-a", "session-a").unwrap();
        session.admit_scope(&admitted, 150).unwrap();
        session
            .revoke_active(HumanContactRevocationReason::ExplicitWithdrawal)
            .unwrap();

        assert_eq!(
            ConsentBoundHumanContactIntentV1::bind(
                &session,
                &admitted,
                HumanBodyRegion::Hand,
                &site("right_hand"),
                "goal-5",
                [0.0, 0.0, 0.0],
                160,
            ),
            Err(HumanContactIntentBindError::SessionNotActive)
        );
    }

    #[test]
    fn expired_scope_cannot_remain_exactly_active_for_binding() {
        let admitted = scope(2, [HumanBodyRegion::Hand], [site("right_hand")]);
        let mut session = HumanContactAuthoritySessionV1::new("participant-a", "session-a").unwrap();
        session.admit_scope(&admitted, 150).unwrap();

        assert_eq!(
            ConsentBoundHumanContactIntentV1::bind(
                &session,
                &admitted,
                HumanBodyRegion::Hand,
                &site("right_hand"),
                "goal-6",
                [0.0, 0.0, 0.0],
                300,
            ),
            Err(HumanContactIntentBindError::ExactScopeNotActive)
        );
    }

    #[test]
    fn malformed_spatial_target_is_rejected() {
        let admitted = scope(2, [HumanBodyRegion::Hand], [site("right_hand")]);
        let mut session = HumanContactAuthoritySessionV1::new("participant-a", "session-a").unwrap();
        session.admit_scope(&admitted, 150).unwrap();

        assert_eq!(
            ConsentBoundHumanContactIntentV1::bind(
                &session,
                &admitted,
                HumanBodyRegion::Hand,
                &site("right_hand"),
                "goal-7",
                [f64::NAN, 0.0, 0.0],
                160,
            ),
            Err(HumanContactIntentBindError::InvalidSpatialTarget)
        );
    }
}

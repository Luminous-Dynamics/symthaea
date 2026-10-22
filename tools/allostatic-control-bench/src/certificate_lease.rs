//! Assumption-bound, freshness-checked safety certificate leases.
//!
//! A certificate is tied to the exact state/model/limits/calibration/authority
//! revisions, monitor epoch, supporting-evidence digest, decision step and
//! certified horizon. Any mismatch invalidates the lease. This prevents a
//! previously valid action decision from being reused after its assumptions
//! change. It is a software lifecycle contract, not a proof that the underlying
//! model or evidence is physically correct.

use super::multistep_viability::{HorizonViableActionSet, MAX_CERTIFIED_HORIZON};
use super::Action;

const MAX_DIGEST: [u8; 32] = [0; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CertificateContext {
    step: u64,
    state_revision: u64,
    dynamics_revision: u64,
    limits_revision: u64,
    calibration_revision: u64,
    authority_revision: u64,
    monitor_epoch: u64,
    horizon: usize,
    evidence_digest: [u8; 32],
}

impl CertificateContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        step: u64,
        state_revision: u64,
        dynamics_revision: u64,
        limits_revision: u64,
        calibration_revision: u64,
        authority_revision: u64,
        monitor_epoch: u64,
        horizon: usize,
        evidence_digest: [u8; 32],
    ) -> Option<Self> {
        if horizon == 0 || horizon > MAX_CERTIFIED_HORIZON || evidence_digest == MAX_DIGEST {
            return None;
        }
        Some(Self {
            step,
            state_revision,
            dynamics_revision,
            limits_revision,
            calibration_revision,
            authority_revision,
            monitor_epoch,
            horizon,
            evidence_digest,
        })
    }

    pub const fn step(self) -> u64 {
        self.step
    }

    pub const fn state_revision(self) -> u64 {
        self.state_revision
    }

    pub const fn dynamics_revision(self) -> u64 {
        self.dynamics_revision
    }

    pub const fn limits_revision(self) -> u64 {
        self.limits_revision
    }

    pub const fn calibration_revision(self) -> u64 {
        self.calibration_revision
    }

    pub const fn authority_revision(self) -> u64 {
        self.authority_revision
    }

    pub const fn monitor_epoch(self) -> u64 {
        self.monitor_epoch
    }

    pub const fn horizon(self) -> usize {
        self.horizon
    }

    pub const fn evidence_digest(self) -> [u8; 32] {
        self.evidence_digest
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertificateRejection {
    InvalidContext,
    ClockRegressed,
    Expired,
    StateRevisionChanged,
    DynamicsRevisionChanged,
    LimitsRevisionChanged,
    CalibrationRevisionChanged,
    AuthorityRevisionChanged,
    MonitorEpochChanged,
    SupportingEvidenceChanged,
    HorizonInsufficient,
}

impl CertificateRejection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidContext => "invalid_context",
            Self::ClockRegressed => "clock_regressed",
            Self::Expired => "expired",
            Self::StateRevisionChanged => "state_revision_changed",
            Self::DynamicsRevisionChanged => "dynamics_revision_changed",
            Self::LimitsRevisionChanged => "limits_revision_changed",
            Self::CalibrationRevisionChanged => "calibration_revision_changed",
            Self::AuthorityRevisionChanged => "authority_revision_changed",
            Self::MonitorEpochChanged => "monitor_epoch_changed",
            Self::SupportingEvidenceChanged => "supporting_evidence_changed",
            Self::HorizonInsufficient => "horizon_insufficient",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertificateLease {
    action: Action,
    context: CertificateContext,
    expires_at_step: u64,
}

impl CertificateLease {
    /// Issue a lease only for an action present in the exact horizon-viable set
    /// produced for the same horizon. A lease cannot outlive its predicted
    /// horizon; a normal caller should use age zero and recertify each step.
    pub fn issue(
        action: Action,
        context: CertificateContext,
        certified_actions: HorizonViableActionSet,
        max_age_steps: u64,
    ) -> Option<Self> {
        if context.horizon != certified_actions.horizon()
            || !certified_actions.contains(action)
            || max_age_steps >= context.horizon as u64
        {
            return None;
        }
        let expires_at_step = context.step.checked_add(max_age_steps)?;
        Some(Self {
            action,
            context,
            expires_at_step,
        })
    }

    pub const fn action(self) -> Action {
        self.action
    }

    pub const fn issued_at_step(self) -> u64 {
        self.context.step
    }

    pub const fn expires_at_step(self) -> u64 {
        self.expires_at_step
    }

    pub const fn horizon(self) -> usize {
        self.context.horizon
    }

    pub const fn evidence_digest(self) -> [u8; 32] {
        self.context.evidence_digest
    }

    /// Validate the lease against a fresh context. Any revision, authority,
    /// monitor-epoch or evidence change invalidates the certificate. The caller
    /// must not downgrade a rejection to a warning or reuse the action anyway.
    pub fn validate(
        self,
        current: CertificateContext,
    ) -> Result<(), CertificateRejection> {
        if current.step < self.context.step {
            return Err(CertificateRejection::ClockRegressed);
        }
        if current.step > self.expires_at_step {
            return Err(CertificateRejection::Expired);
        }
        if current.state_revision != self.context.state_revision {
            return Err(CertificateRejection::StateRevisionChanged);
        }
        if current.dynamics_revision != self.context.dynamics_revision {
            return Err(CertificateRejection::DynamicsRevisionChanged);
        }
        if current.limits_revision != self.context.limits_revision {
            return Err(CertificateRejection::LimitsRevisionChanged);
        }
        if current.calibration_revision != self.context.calibration_revision {
            return Err(CertificateRejection::CalibrationRevisionChanged);
        }
        if current.authority_revision != self.context.authority_revision {
            return Err(CertificateRejection::AuthorityRevisionChanged);
        }
        if current.monitor_epoch != self.context.monitor_epoch {
            return Err(CertificateRejection::MonitorEpochChanged);
        }
        if current.evidence_digest != self.context.evidence_digest {
            return Err(CertificateRejection::SupportingEvidenceChanged);
        }
        if current.horizon > self.context.horizon {
            return Err(CertificateRejection::HorizonInsufficient);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multistep_viability::evaluate_horizon_action_set;
    use crate::robust_viability::{
        ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
    };

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("valid interval")
    }

    fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
        ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("valid effects")
    }

    fn context(
        step: u64,
        state_revision: u64,
        dynamics_revision: u64,
        limits_revision: u64,
        calibration_revision: u64,
        authority_revision: u64,
        monitor_epoch: u64,
        horizon: usize,
        digest: [u8; 32],
    ) -> CertificateContext {
        CertificateContext::new(
            step,
            state_revision,
            dynamics_revision,
            limits_revision,
            calibration_revision,
            authority_revision,
            monitor_epoch,
            horizon,
            digest,
        )
        .expect("valid certificate context")
    }

    fn certified_actions(horizon: usize) -> HorizonViableActionSet {
        let state = StateEnvelope::new(interval(0.50, 0.50), interval(0.50, 0.50))
            .expect("valid state");
        let dynamics = DynamicsBounds::new(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        );
        evaluate_horizon_action_set(
            state,
            dynamics,
            ViabilityLimits::new(0.15, 0.88).expect("valid limits"),
            true,
            horizon,
        )
        .expect("valid horizon evaluation")
    }

    fn valid_context() -> CertificateContext {
        context(100, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32])
    }

    fn lease() -> CertificateLease {
        CertificateLease::issue(
            Action::Maintain,
            valid_context(),
            certified_actions(3),
            1,
        )
        .expect("maintain is horizon-certified")
    }

    #[test]
    fn matching_context_accepts_certificate_within_lease_age() {
        let certificate = lease();
        let current = context(101, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32]);
        assert_eq!(certificate.validate(current), Ok(()));
        assert_eq!(certificate.expires_at_step(), 101);
    }

    #[test]
    fn state_dynamics_constraints_calibration_and_authority_changes_invalidate() {
        let certificate = lease();
        let base = valid_context();
        let variants = [
            (2, 11, 13, 17, 19, 23, CertificateRejection::StateRevisionChanged),
            (7, 12, 13, 17, 19, 23, CertificateRejection::DynamicsRevisionChanged),
            (7, 11, 14, 17, 19, 23, CertificateRejection::LimitsRevisionChanged),
            (7, 11, 13, 18, 19, 23, CertificateRejection::CalibrationRevisionChanged),
            (7, 11, 13, 17, 20, 23, CertificateRejection::AuthorityRevisionChanged),
        ];
        for (state_rev, dynamics_rev, limits_rev, calibration_rev, authority_rev,
            monitor_epoch, expected) in variants
        {
            let changed = context(
                base.step(),
                state_rev,
                dynamics_rev,
                limits_rev,
                calibration_rev,
                authority_rev,
                monitor_epoch,
                base.horizon(),
                base.evidence_digest(),
            );
            assert_eq!(certificate.validate(changed), Err(expected));
        }
    }

    #[test]
    fn monitor_epoch_and_supporting_evidence_changes_invalidate() {
        let certificate = lease();
        let base = valid_context();
        let changed_monitor = context(100, 7, 11, 13, 17, 19, 24, 3, [0x2a; 32]);
        let changed_evidence = context(100, 7, 11, 13, 17, 19, 23, 3, [0x2b; 32]);
        assert_eq!(
            certificate.validate(changed_monitor),
            Err(CertificateRejection::MonitorEpochChanged)
        );
        assert_eq!(
            certificate.validate(changed_evidence),
            Err(CertificateRejection::SupportingEvidenceChanged)
        );
        assert_eq!(base.monitor_epoch(), 23);
    }

    #[test]
    fn expired_or_rewound_clock_invalidates() {
        let certificate = lease();
        let base = valid_context();
        assert_eq!(
            certificate.validate(context(102, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32])),
            Err(CertificateRejection::Expired)
        );
        assert_eq!(
            certificate.validate(context(99, 7, 11, 13, 17, 19, 23, 3, [0x2a; 32])),
            Err(CertificateRejection::ClockRegressed)
        );
        assert_eq!(certificate.issued_at_step(), base.step());
    }

    #[test]
    fn a_request_for_a_longer_horizon_invalidates_a_shorter_certificate() {
        let certificate = lease();
        let current = context(100, 7, 11, 13, 17, 19, 23, 4, [0x2a; 32]);
        assert_eq!(
            certificate.validate(current),
            Err(CertificateRejection::HorizonInsufficient)
        );
    }

    #[test]
    fn issue_rejects_uncertified_actions_horizon_mismatch_and_excessive_age() {
        let actions = certified_actions(3);
        let current = valid_context();
        assert!(CertificateLease::issue(Action::Push, current, actions, 0).is_some());
        assert!(CertificateLease::issue(Action::Push, current, actions, 3).is_none());

        let shorter = context(100, 7, 11, 13, 17, 19, 23, 2, [0x2a; 32]);
        assert!(CertificateLease::issue(Action::Maintain, shorter, actions, 0).is_none());

        let boundary_state =
            StateEnvelope::new(interval(0.16, 0.16), interval(0.87, 0.87)).expect("valid state");
        let dead_end_dynamics = DynamicsBounds::new(
            effects((-0.006, -0.005), (0.01, 0.01)),
            effects((0.0, 0.0), (0.005, 0.005)),
            effects((-0.02, -0.02), (-0.30, -0.25)),
        );
        let boundary_actions = evaluate_horizon_action_set(
            boundary_state,
            dead_end_dynamics,
            ViabilityLimits::new(0.15, 0.88).expect("valid limits"),
            true,
            2,
        )
        .expect("valid horizon");
        assert!(!boundary_actions.push());
        assert!(CertificateLease::issue(
            Action::Push,
            shorter,
            boundary_actions,
            0,
        )
        .is_none());
    }

    #[test]
    fn invalid_digest_or_horizon_cannot_create_a_context() {
        assert!(CertificateContext::new(1, 1, 1, 1, 1, 1, 1, 3, [0; 32]).is_none());
        assert!(CertificateContext::new(
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            MAX_CERTIFIED_HORIZON + 1,
            [0x2a; 32],
        )
        .is_none());
    }
}

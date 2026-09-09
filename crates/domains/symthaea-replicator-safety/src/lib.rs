// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replicator Safety Kernel (RSK) constitutional authority core.
//!
//! This crate models authority around creation; it does not implement physical,
//! molecular, biological, or other replication mechanisms.
//!
//! Foundational invariant: **physical creation does not confer authority**.

#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubjectId([u8; 32]);
impl SubjectId {
    pub const fn new(bytes: [u8; 32]) -> Self { Self(bytes) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LineageId([u8; 32]);
impl LineageId {
    pub const fn new(bytes: [u8; 32]) -> Self { Self(bytes) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GrantId([u8; 32]);
impl GrantId {
    pub const fn new(bytes: [u8; 32]) -> Self { Self(bytes) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EvidenceDigest([u8; 32]);
impl EvidenceDigest {
    pub const fn new(bytes: [u8; 32]) -> Self { Self(bytes) }
}

/// Opaque caller-defined capability bits.
///
/// RSK assigns no physical meaning to these bits. It only enforces that a
/// request is a subset of both the ancestral/parent ceiling and the explicit
/// subject-bound grant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CapabilitySet(u64);
impl CapabilitySet {
    pub const NONE: Self = Self(0);
    pub const fn from_bits(bits: u64) -> Self { Self(bits) }
    pub const fn bits(self) -> u64 { self.0 }
    pub const fn is_empty(self) -> bool { self.0 == 0 }
    pub const fn intersect(self, other: Self) -> Self { Self(self.0 & other.0) }
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskClass { R0, R1, R2, R3, R4, R5 }
impl RiskClass {
    pub const fn requires_high_consequence_quorum(self) -> bool {
        matches!(self, Self::R4 | Self::R5)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrantIssuerClass {
    ExternalIndependent,
    Developer,
    Operator,
    ControlledSubject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicationGrant {
    pub grant_id: GrantId,
    pub subject: SubjectId,
    pub lineage: LineageId,
    pub issuer_class: GrantIssuerClass,
    pub allowed_capabilities: CapabilitySet,
    pub not_before_unix_secs: u64,
    pub expires_at_unix_secs: u64,
    pub generation: u64,
    pub max_direct_children: u64,
    pub max_total_descendants: u64,
    pub max_lineage_depth: u32,
    pub max_resource_units: u64,
    pub safety_case_digest: EvidenceDigest,
    pub containment_envelope_digest: EvidenceDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicationBudgetSnapshot {
    pub direct_children_consumed: u64,
    pub total_descendants_consumed: u64,
    pub current_lineage_depth: u32,
    pub resource_units_consumed: u64,
    pub requested_resource_units: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineageStatus {
    pub known: bool,
    pub acyclic: bool,
    pub parent_binding_valid: bool,
}
impl LineageStatus {
    pub const fn valid(self) -> bool {
        self.known && self.acyclic && self.parent_binding_valid
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonitoringStatus { pub healthy: bool, pub fresh: bool }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafetyCaseStatus {
    pub design_qualified: bool,
    pub policy_current: bool,
    pub evidence_current: bool,
    pub current_safety_case_digest: EvidenceDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainmentStatus {
    pub environment_matches: bool,
    pub current_envelope_digest: EvidenceDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuorumEvidence {
    pub independent_approvals: u16,
    pub required_independent_approvals: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicationAuthorityRequest {
    pub subject: SubjectId,
    pub lineage: LineageId,
    pub risk_class: RiskClass,
    pub requested_capabilities: CapabilitySet,
    pub parent_capability_ceiling: CapabilitySet,
    pub now_unix_secs: u64,
    pub expected_grant_generation: u64,
    pub quarantined: bool,
    pub revoked: bool,
    pub lineage_status: LineageStatus,
    pub monitoring: MonitoringStatus,
    pub safety_case: SafetyCaseStatus,
    pub containment: ContainmentStatus,
    pub budget: ReplicationBudgetSnapshot,
    pub quorum: QuorumEvidence,
}

/// Provenance-only record returned by physical creation.
///
/// There is deliberately no authority or grant field. A descendant must pass
/// a later, separate `evaluate_replication_authority` call with its own grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnprivilegedDescendant {
    pub subject: SubjectId,
    pub parent: SubjectId,
    pub lineage: LineageId,
}

pub const fn record_physical_creation(
    parent: SubjectId,
    child: SubjectId,
    lineage: LineageId,
) -> UnprivilegedDescendant {
    UnprivilegedDescendant { subject: child, parent, lineage }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthorizedReplication {
    pub subject: SubjectId,
    pub lineage: LineageId,
    pub grant_id: GrantId,
    pub grant_generation: u64,
    pub effective_capabilities: CapabilitySet,
    pub safety_case_digest: EvidenceDigest,
    pub containment_envelope_digest: EvidenceDigest,
    pub expires_at_unix_secs: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DenialReason {
    Quarantined,
    Revoked,
    MissingExplicitGrant,
    EmptyCapabilityRequest,
    RiskClassDoesNotPermitReplication,
    GrantSubjectMismatch,
    GrantLineageMismatch,
    ControlledSubjectIssuedGrant,
    GrantNotYetValid,
    GrantExpired,
    StaleGrantGeneration,
    DesignNotQualified,
    UnknownLineage,
    InvalidLineage,
    PolicyStale,
    SafetyEvidenceStale,
    SafetyCaseDigestMismatch,
    ContainmentDrift,
    ContainmentEnvelopeDigestMismatch,
    MonitorUnhealthy,
    MonitorStale,
    DirectChildBudgetExhausted,
    TotalDescendantBudgetExhausted,
    LineageDepthBudgetExhausted,
    ResourceBudgetExhausted,
    RequestedCapabilitiesExceedParent,
    RequestedCapabilitiesExceedGrant,
    InsufficientIndependentQuorum,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplicationDecision {
    Allow(AuthorizedReplication),
    Deny { reasons: Vec<DenialReason> },
}
impl ReplicationDecision {
    pub fn is_allowed(&self) -> bool { matches!(self, Self::Allow(_)) }
    pub fn denial_reasons(&self) -> &[DenialReason] {
        match self { Self::Allow(_) => &[], Self::Deny { reasons } => reasons }
    }
}

pub const MIN_HIGH_CONSEQUENCE_INDEPENDENT_QUORUM: u16 = 2;

/// Pure, deterministic, deny-by-default RSK decision.
///
/// This function evaluates an immutable snapshot. It does not mutate resource
/// or lineage ledgers. Monotonic consumption and atomic replay resistance are
/// intentionally left for the ledger tranche.
pub fn evaluate_replication_authority(
    request: &ReplicationAuthorityRequest,
    grant: Option<&ReplicationGrant>,
) -> ReplicationDecision {
    let mut reasons = Vec::new();

    // Negative containment is checked first and can never be canceled by a
    // later positive fact. Remaining checks are still collected for audit.
    if request.quarantined { reasons.push(DenialReason::Quarantined); }
    if request.revoked { reasons.push(DenialReason::Revoked); }
    if request.requested_capabilities.is_empty() {
        reasons.push(DenialReason::EmptyCapabilityRequest);
    }
    if request.risk_class == RiskClass::R0 {
        reasons.push(DenialReason::RiskClassDoesNotPermitReplication);
    }

    if !request.lineage_status.known {
        reasons.push(DenialReason::UnknownLineage);
    } else if !request.lineage_status.valid() {
        reasons.push(DenialReason::InvalidLineage);
    }
    if !request.safety_case.design_qualified {
        reasons.push(DenialReason::DesignNotQualified);
    }
    if !request.safety_case.policy_current {
        reasons.push(DenialReason::PolicyStale);
    }
    if !request.safety_case.evidence_current {
        reasons.push(DenialReason::SafetyEvidenceStale);
    }
    if !request.containment.environment_matches {
        reasons.push(DenialReason::ContainmentDrift);
    }
    if !request.monitoring.healthy { reasons.push(DenialReason::MonitorUnhealthy); }
    if !request.monitoring.fresh { reasons.push(DenialReason::MonitorStale); }

    if !request.requested_capabilities.is_subset_of(request.parent_capability_ceiling) {
        reasons.push(DenialReason::RequestedCapabilitiesExceedParent);
    }

    let required_quorum = if request.risk_class.requires_high_consequence_quorum() {
        request.quorum.required_independent_approvals
            .max(MIN_HIGH_CONSEQUENCE_INDEPENDENT_QUORUM)
    } else {
        request.quorum.required_independent_approvals
    };
    if request.quorum.independent_approvals < required_quorum {
        reasons.push(DenialReason::InsufficientIndependentQuorum);
    }

    let Some(grant) = grant else {
        reasons.push(DenialReason::MissingExplicitGrant);
        return ReplicationDecision::Deny { reasons };
    };

    if grant.subject != request.subject { reasons.push(DenialReason::GrantSubjectMismatch); }
    if grant.lineage != request.lineage { reasons.push(DenialReason::GrantLineageMismatch); }
    if grant.issuer_class == GrantIssuerClass::ControlledSubject {
        reasons.push(DenialReason::ControlledSubjectIssuedGrant);
    }
    if request.now_unix_secs < grant.not_before_unix_secs {
        reasons.push(DenialReason::GrantNotYetValid);
    }
    if request.now_unix_secs >= grant.expires_at_unix_secs {
        reasons.push(DenialReason::GrantExpired);
    }
    if grant.generation != request.expected_grant_generation {
        reasons.push(DenialReason::StaleGrantGeneration);
    }
    if grant.safety_case_digest != request.safety_case.current_safety_case_digest {
        reasons.push(DenialReason::SafetyCaseDigestMismatch);
    }
    if grant.containment_envelope_digest != request.containment.current_envelope_digest {
        reasons.push(DenialReason::ContainmentEnvelopeDigestMismatch);
    }
    if !request.requested_capabilities.is_subset_of(grant.allowed_capabilities) {
        reasons.push(DenialReason::RequestedCapabilitiesExceedGrant);
    }

    if request.budget.direct_children_consumed >= grant.max_direct_children {
        reasons.push(DenialReason::DirectChildBudgetExhausted);
    }
    if request.budget.total_descendants_consumed >= grant.max_total_descendants {
        reasons.push(DenialReason::TotalDescendantBudgetExhausted);
    }
    if request.budget.current_lineage_depth >= grant.max_lineage_depth {
        reasons.push(DenialReason::LineageDepthBudgetExhausted);
    }
    let resource_fits = request.budget.resource_units_consumed
        .checked_add(request.budget.requested_resource_units)
        .is_some_and(|total| total <= grant.max_resource_units);
    if !resource_fits { reasons.push(DenialReason::ResourceBudgetExhausted); }

    if !reasons.is_empty() { return ReplicationDecision::Deny { reasons }; }

    let effective_capabilities = request.requested_capabilities
        .intersect(request.parent_capability_ceiling)
        .intersect(grant.allowed_capabilities);

    ReplicationDecision::Allow(AuthorizedReplication {
        subject: request.subject,
        lineage: request.lineage,
        grant_id: grant.grant_id,
        grant_generation: grant.generation,
        effective_capabilities,
        safety_case_digest: grant.safety_case_digest,
        containment_envelope_digest: grant.containment_envelope_digest,
        expires_at_unix_secs: grant.expires_at_unix_secs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: CapabilitySet = CapabilitySet::from_bits(0b01);
    const B: CapabilitySet = CapabilitySet::from_bits(0b10);
    const AB: CapabilitySet = CapabilitySet::from_bits(0b11);

    fn sid(b: u8) -> SubjectId { SubjectId::new([b; 32]) }
    fn lid(b: u8) -> LineageId { LineageId::new([b; 32]) }
    fn dig(b: u8) -> EvidenceDigest { EvidenceDigest::new([b; 32]) }

    fn grant(subject: SubjectId, lineage: LineageId) -> ReplicationGrant {
        ReplicationGrant {
            grant_id: GrantId::new([9; 32]), subject, lineage,
            issuer_class: GrantIssuerClass::ExternalIndependent,
            allowed_capabilities: AB,
            not_before_unix_secs: 10, expires_at_unix_secs: 1_000, generation: 7,
            max_direct_children: 4, max_total_descendants: 16,
            max_lineage_depth: 4, max_resource_units: 100,
            safety_case_digest: dig(4), containment_envelope_digest: dig(5),
        }
    }

    fn request(subject: SubjectId, lineage: LineageId) -> ReplicationAuthorityRequest {
        ReplicationAuthorityRequest {
            subject, lineage, risk_class: RiskClass::R3,
            requested_capabilities: A, parent_capability_ceiling: AB,
            now_unix_secs: 100, expected_grant_generation: 7,
            quarantined: false, revoked: false,
            lineage_status: LineageStatus { known: true, acyclic: true, parent_binding_valid: true },
            monitoring: MonitoringStatus { healthy: true, fresh: true },
            safety_case: SafetyCaseStatus {
                design_qualified: true, policy_current: true, evidence_current: true,
                current_safety_case_digest: dig(4),
            },
            containment: ContainmentStatus {
                environment_matches: true, current_envelope_digest: dig(5),
            },
            budget: ReplicationBudgetSnapshot {
                direct_children_consumed: 0, total_descendants_consumed: 0,
                current_lineage_depth: 0, resource_units_consumed: 0,
                requested_resource_units: 1,
            },
            quorum: QuorumEvidence {
                independent_approvals: 1, required_independent_approvals: 1,
            },
        }
    }

    fn deny(decision: ReplicationDecision, reason: DenialReason) {
        assert!(!decision.is_allowed());
        assert!(decision.denial_reasons().contains(&reason), "{decision:?}");
    }

    #[test]
    fn green_request_is_allowed() {
        let s = sid(1); let l = lid(2); let g = grant(s, l); let r = request(s, l);
        let ReplicationDecision::Allow(a) = evaluate_replication_authority(&r, Some(&g))
            else { panic!("expected allow") };
        assert_eq!(a.subject, s);
        assert_eq!(a.effective_capabilities, A);
    }

    #[test]
    fn r0_never_authorizes_replication() {
        let s = sid(1); let l = lid(2); let g = grant(s, l); let mut r = request(s, l);
        r.risk_class = RiskClass::R0;
        deny(evaluate_replication_authority(&r, Some(&g)),
             DenialReason::RiskClassDoesNotPermitReplication);
    }

    #[test]
    fn physical_creation_is_unprivileged_and_parent_grant_does_not_transfer() {
        let p = sid(1); let c = sid(2); let l = lid(3);
        let child = record_physical_creation(p, c, l);
        assert_eq!(child.subject, c);
        let pg = grant(p, l);
        deny(evaluate_replication_authority(&request(c, l), Some(&pg)),
             DenialReason::GrantSubjectMismatch);
    }

    #[test]
    fn quarantine_and_revocation_dominate_green_authority() {
        let s = sid(1); let l = lid(2); let g = grant(s, l); let mut r = request(s, l);
        r.quarantined = true; r.revoked = true;
        let d = evaluate_replication_authority(&r, Some(&g));
        deny(d.clone(), DenialReason::Quarantined);
        deny(d, DenialReason::Revoked);
    }

    #[test]
    fn lineage_and_monitor_uncertainty_fail_closed() {
        let s = sid(1); let l = lid(2); let g = grant(s, l); let mut r = request(s, l);
        r.lineage_status.known = false; r.monitoring.healthy = false; r.monitoring.fresh = false;
        let d = evaluate_replication_authority(&r, Some(&g));
        for reason in [DenialReason::UnknownLineage, DenialReason::MonitorUnhealthy,
                       DenialReason::MonitorStale] {
            assert!(d.denial_reasons().contains(&reason), "{d:?}");
        }
    }

    #[test]
    fn evidence_and_containment_are_exactly_digest_bound() {
        let s = sid(1); let l = lid(2); let g = grant(s, l); let mut r = request(s, l);
        r.safety_case.current_safety_case_digest = dig(88);
        r.containment.current_envelope_digest = dig(89);
        let d = evaluate_replication_authority(&r, Some(&g));
        deny(d.clone(), DenialReason::SafetyCaseDigestMismatch);
        deny(d, DenialReason::ContainmentEnvelopeDigestMismatch);
    }

    #[test]
    fn all_budget_dimensions_fail_closed_and_overflow_is_denied() {
        let s = sid(1); let l = lid(2); let mut g = grant(s, l); let mut r = request(s, l);
        r.budget.direct_children_consumed = g.max_direct_children;
        r.budget.total_descendants_consumed = g.max_total_descendants;
        r.budget.current_lineage_depth = g.max_lineage_depth;
        r.budget.resource_units_consumed = g.max_resource_units;
        let d = evaluate_replication_authority(&r, Some(&g));
        for reason in [DenialReason::DirectChildBudgetExhausted,
                       DenialReason::TotalDescendantBudgetExhausted,
                       DenialReason::LineageDepthBudgetExhausted,
                       DenialReason::ResourceBudgetExhausted] {
            assert!(d.denial_reasons().contains(&reason), "{d:?}");
        }
        g.max_resource_units = u64::MAX;
        r.budget.direct_children_consumed = 0; r.budget.total_descendants_consumed = 0;
        r.budget.current_lineage_depth = 0; r.budget.resource_units_consumed = u64::MAX;
        deny(evaluate_replication_authority(&r, Some(&g)), DenialReason::ResourceBudgetExhausted);
    }

    #[test]
    fn capability_authority_is_intersection_bounded() {
        let s = sid(1); let l = lid(2); let mut g = grant(s, l); let mut r = request(s, l);
        r.requested_capabilities = AB; r.parent_capability_ceiling = A; g.allowed_capabilities = B;
        let d = evaluate_replication_authority(&r, Some(&g));
        deny(d.clone(), DenialReason::RequestedCapabilitiesExceedParent);
        deny(d, DenialReason::RequestedCapabilitiesExceedGrant);
    }

    #[test]
    fn controlled_subject_cannot_self_mint_external_authority() {
        let s = sid(1); let l = lid(2); let mut g = grant(s, l); g.issuer_class = GrantIssuerClass::ControlledSubject;
        deny(evaluate_replication_authority(&request(s, l), Some(&g)),
             DenialReason::ControlledSubjectIssuedGrant);
    }

    #[test]
    fn r4_r5_require_at_least_two_independent_approvals() {
        for class in [RiskClass::R4, RiskClass::R5] {
            let s = sid(1); let l = lid(2); let g = grant(s, l); let mut r = request(s, l);
            r.risk_class = class; r.quorum.independent_approvals = 1;
            deny(evaluate_replication_authority(&r, Some(&g)),
                 DenialReason::InsufficientIndependentQuorum);
            r.quorum.independent_approvals = 2;
            assert!(evaluate_replication_authority(&r, Some(&g)).is_allowed());
        }
    }

    #[test]
    fn stale_or_missing_grant_never_preserves_authority() {
        let s = sid(1); let l = lid(2); let mut g = grant(s, l); let r = request(s, l);
        g.generation = 6;
        deny(evaluate_replication_authority(&r, Some(&g)), DenialReason::StaleGrantGeneration);
        deny(evaluate_replication_authority(&r, None), DenialReason::MissingExplicitGrant);
    }

    #[test]
    fn time_bounds_are_half_open_and_fail_closed() {
        let s = sid(1); let l = lid(2); let g = grant(s, l);
        let mut early = request(s, l); early.now_unix_secs = 9;
        deny(evaluate_replication_authority(&early, Some(&g)), DenialReason::GrantNotYetValid);
        let mut expired = request(s, l); expired.now_unix_secs = 1_000;
        deny(evaluate_replication_authority(&expired, Some(&g)), DenialReason::GrantExpired);
    }

    #[test]
    fn every_single_hard_failure_breaks_an_all_green_request() {
        let s = sid(1); let l = lid(2); let g = grant(s, l);
        let mut cases = Vec::new();
        let mut r = request(s, l); r.quarantined = true; cases.push(r);
        let mut r = request(s, l); r.revoked = true; cases.push(r);
        let mut r = request(s, l); r.lineage_status.known = false; cases.push(r);
        let mut r = request(s, l); r.monitoring.healthy = false; cases.push(r);
        let mut r = request(s, l); r.safety_case.policy_current = false; cases.push(r);
        let mut r = request(s, l); r.safety_case.evidence_current = false; cases.push(r);
        let mut r = request(s, l); r.containment.environment_matches = false; cases.push(r);
        for failed in cases {
            assert!(!evaluate_replication_authority(&failed, Some(&g)).is_allowed());
        }
    }
}

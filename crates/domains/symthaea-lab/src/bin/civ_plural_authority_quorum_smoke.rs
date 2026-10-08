//! SYM-CIV-006: plural authority and separation-of-powers smoke.
//!
//! Dependency-free research control. Decision identity is represented as an
//! exact structured tuple in this fixture, not as a cryptographic digest.
//! A production implementation must canonicalize and cryptographically bind it.
//!
//! Core boundaries:
//! candidate != evaluator != authority
//! authority_1 != authority_2
//! quorum != legitimacy
//! PASS != authorization
//! current quorum != stale roster snapshot
//! suspension != authorization
//!
//! Claim ceiling: local authority/type-flow invariants only.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Digest(&'static str);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ConsequenceTier {
    Low,
    Significant,
    Critical,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AuthorityRole {
    SafetyGuardian,
    TechnicalAdjudicator,
    PublicSteward,
    AffectedPartyRepresentative,
    EmergencyGuardian,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AuthorityMember {
    id: Digest,
    role: AuthorityRole,
    authority_lineage: Digest,
    conflict: bool,
    active: bool,
    valid_from_epoch: u64,
    valid_until_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuthorityRoster {
    version: Digest,
    members: Vec<AuthorityMember>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GovernancePolicy {
    version: Digest,
    tier: ConsequenceTier,
    scope: Digest,
    min_members: usize,
    min_roles: usize,
    roster_version: Digest,
    valid_from_epoch: u64,
    valid_until_epoch: u64,
    max_evaluation_age: u64,
}

impl GovernancePolicy {
    fn for_tier(tier: ConsequenceTier, scope: Digest, roster_version: Digest) -> Self {
        let (min_members, min_roles) = match tier {
            ConsequenceTier::Low => (1, 1),
            ConsequenceTier::Significant => (2, 2),
            ConsequenceTier::Critical => (3, 3),
        };
        Self {
            version: Digest("governance-policy-v1"),
            tier,
            scope,
            min_members,
            min_roles,
            roster_version,
            valid_from_epoch: 90,
            valid_until_epoch: 200,
            max_evaluation_age: 30,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Verdict {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvaluationReceipt {
    id: Digest,
    candidate: Digest,
    evaluator: Digest,
    tier: ConsequenceTier,
    verdict: Verdict,
    issued_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuthorityDecisionIdentity {
    evaluation: Digest,
    candidate: Digest,
    tier: ConsequenceTier,
    scope: Digest,
    policy: GovernancePolicy,
    roster_version: Digest,
    approvers: Vec<AuthorityMember>,
    rationale: Digest,
    effective_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuthorityDecision {
    // Exact structured identity; not a content hash or signature.
    id: AuthorityDecisionIdentity,
    candidate: Digest,
    evaluation: Digest,
    tier: ConsequenceTier,
    scope: Digest,
    rationale: Digest,
    effective_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuorumFailure {
    TooFewMembers,
    DuplicateMember,
    TooFewDistinctRoles,
    SharedAuthorityLineage,
    ConflictOfInterest,
    CandidateIsApprover,
    EvaluatorIsApprover,
    ScopeNotPermitted,
    TierMismatch,
    EvaluationMismatch,
    NonPassingEvaluation,
    StaleRoster,
    MemberNotInCurrentRoster,
    MemberInactive,
    MemberOutsideValidity,
    PolicyNotCurrent,
    EvaluationTooOld,
    DecisionNotCurrent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EmergencySuspensionIdentity {
    deployment: Digest,
    authority: Digest,
    role: AuthorityRole,
    reason: Digest,
    policy_version: Digest,
    roster_version: Digest,
    effective_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EmergencySuspension {
    id: EmergencySuspensionIdentity,
    deployment: Digest,
    authority: Digest,
    reason: Digest,
    effective_epoch: u64,
}

fn validate_policy(
    policy: GovernancePolicy,
    roster: &AuthorityRoster,
    tier: ConsequenceTier,
    scope: Digest,
    now_epoch: u64,
) -> Result<(), QuorumFailure> {
    if policy.roster_version != roster.version {
        return Err(QuorumFailure::StaleRoster);
    }
    if policy.tier != tier {
        return Err(QuorumFailure::TierMismatch);
    }
    if policy.scope != scope {
        return Err(QuorumFailure::ScopeNotPermitted);
    }
    if now_epoch < policy.valid_from_epoch || now_epoch > policy.valid_until_epoch {
        return Err(QuorumFailure::PolicyNotCurrent);
    }
    Ok(())
}

fn approve(
    evaluation: EvaluationReceipt,
    tier: ConsequenceTier,
    scope: Digest,
    candidate: Digest,
    evaluator: Digest,
    approvers: &[AuthorityMember],
    roster: &AuthorityRoster,
    policy: GovernancePolicy,
    rationale: Digest,
    effective_epoch: u64,
) -> Result<AuthorityDecision, QuorumFailure> {
    if evaluation.verdict != Verdict::Pass {
        return Err(QuorumFailure::NonPassingEvaluation);
    }
    if evaluation.candidate != candidate || evaluation.evaluator != evaluator {
        return Err(QuorumFailure::EvaluationMismatch);
    }
    if evaluation.tier != tier || policy.tier != tier {
        return Err(QuorumFailure::TierMismatch);
    }
    if effective_epoch < evaluation.issued_epoch
        || effective_epoch - evaluation.issued_epoch > policy.max_evaluation_age
    {
        return Err(QuorumFailure::EvaluationTooOld);
    }
    validate_policy(policy, roster, tier, scope, effective_epoch)?;

    if approvers.len() < policy.min_members {
        return Err(QuorumFailure::TooFewMembers);
    }

    let mut ids = BTreeSet::new();
    let mut roles = BTreeSet::new();
    let mut authority_lineages = BTreeSet::new();
    let mut canonical_approvers = Vec::with_capacity(approvers.len());

    for approver in approvers {
        if approver.id == candidate {
            return Err(QuorumFailure::CandidateIsApprover);
        }
        if approver.id == evaluator {
            return Err(QuorumFailure::EvaluatorIsApprover);
        }
        if !ids.insert(approver.id) {
            return Err(QuorumFailure::DuplicateMember);
        }
        if approver.conflict {
            return Err(QuorumFailure::ConflictOfInterest);
        }
        if !approver.active {
            return Err(QuorumFailure::MemberInactive);
        }
        if effective_epoch < approver.valid_from_epoch
            || effective_epoch > approver.valid_until_epoch
        {
            return Err(QuorumFailure::MemberOutsideValidity);
        }

        let registered = roster
            .members
            .iter()
            .find(|member| member.id == approver.id)
            .ok_or(QuorumFailure::MemberNotInCurrentRoster)?;
        if registered != approver {
            return Err(QuorumFailure::StaleRoster);
        }

        roles.insert(approver.role);
        if !authority_lineages.insert(approver.authority_lineage) {
            return Err(QuorumFailure::SharedAuthorityLineage);
        }
        canonical_approvers.push(*approver);
    }

    if roles.len() < policy.min_roles {
        return Err(QuorumFailure::TooFewDistinctRoles);
    }

    canonical_approvers.sort_by_key(|member| member.id);

    let id = AuthorityDecisionIdentity {
        evaluation: evaluation.id,
        candidate,
        tier,
        scope,
        policy,
        roster_version: roster.version,
        approvers: canonical_approvers,
        rationale,
        effective_epoch,
    };

    Ok(AuthorityDecision {
        id,
        candidate,
        evaluation: evaluation.id,
        tier,
        scope,
        rationale,
        effective_epoch,
    })
}

fn validate_current_decision(
    decision: &AuthorityDecision,
    current_policy: GovernancePolicy,
    current_roster: &AuthorityRoster,
    requested_tier: ConsequenceTier,
    requested_scope: Digest,
    now_epoch: u64,
) -> Result<(), QuorumFailure> {
    validate_policy(
        current_policy,
        current_roster,
        requested_tier,
        requested_scope,
        now_epoch,
    )?;

    if decision.id.policy != current_policy
        || decision.id.roster_version != current_roster.version
        || decision.id.tier != requested_tier
        || decision.id.scope != requested_scope
    {
        return Err(QuorumFailure::DecisionNotCurrent);
    }

    for original_member in &decision.id.approvers {
        let current_member = current_roster
            .members
            .iter()
            .find(|member| member.id == original_member.id)
            .ok_or(QuorumFailure::MemberNotInCurrentRoster)?;

        if current_member != original_member {
            return Err(QuorumFailure::DecisionNotCurrent);
        }
        if !current_member.active {
            return Err(QuorumFailure::MemberInactive);
        }
        if current_member.conflict {
            return Err(QuorumFailure::ConflictOfInterest);
        }
        if now_epoch < current_member.valid_from_epoch
            || now_epoch > current_member.valid_until_epoch
        {
            return Err(QuorumFailure::MemberOutsideValidity);
        }
    }
    Ok(())
}

fn emergency_suspend(
    member: AuthorityMember,
    deployment: Digest,
    reason: Digest,
    policy: GovernancePolicy,
    roster: &AuthorityRoster,
    effective_epoch: u64,
) -> Result<EmergencySuspension, QuorumFailure> {
    if policy.roster_version != roster.version {
        return Err(QuorumFailure::StaleRoster);
    }
    if effective_epoch < policy.valid_from_epoch || effective_epoch > policy.valid_until_epoch {
        return Err(QuorumFailure::PolicyNotCurrent);
    }
    if member.role != AuthorityRole::EmergencyGuardian {
        return Err(QuorumFailure::MemberNotInCurrentRoster);
    }
    if member.conflict {
        return Err(QuorumFailure::ConflictOfInterest);
    }
    if !member.active {
        return Err(QuorumFailure::MemberInactive);
    }
    if effective_epoch < member.valid_from_epoch || effective_epoch > member.valid_until_epoch {
        return Err(QuorumFailure::MemberOutsideValidity);
    }
    let registered = roster
        .members
        .iter()
        .find(|current| current.id == member.id)
        .ok_or(QuorumFailure::MemberNotInCurrentRoster)?;
    if registered != &member {
        return Err(QuorumFailure::StaleRoster);
    }

    let id = EmergencySuspensionIdentity {
        deployment,
        authority: member.id,
        role: member.role,
        reason,
        policy_version: policy.version,
        roster_version: roster.version,
        effective_epoch,
    };

    Ok(EmergencySuspension {
        id,
        deployment,
        authority: member.id,
        reason,
        effective_epoch,
    })
}

fn main() {
    let candidate = Digest("candidate-v1");
    let evaluator = Digest("evaluator-v1");
    let roster_version = Digest("roster-v1");

    let evaluation = EvaluationReceipt {
        id: Digest("evaluation-v1"),
        candidate,
        evaluator,
        tier: ConsequenceTier::Critical,
        verdict: Verdict::Pass,
        issued_epoch: 95,
    };

    let safety = AuthorityMember {
        id: Digest("authority-safety-1"),
        role: AuthorityRole::SafetyGuardian,
        authority_lineage: Digest("lineage-safety"),
        conflict: false,
        active: true,
        valid_from_epoch: 1,
        valid_until_epoch: 500,
    };
    let technical = AuthorityMember {
        id: Digest("authority-technical-1"),
        role: AuthorityRole::TechnicalAdjudicator,
        authority_lineage: Digest("lineage-technical"),
        conflict: false,
        active: true,
        valid_from_epoch: 1,
        valid_until_epoch: 500,
    };
    let public = AuthorityMember {
        id: Digest("authority-public-1"),
        role: AuthorityRole::PublicSteward,
        authority_lineage: Digest("lineage-public"),
        conflict: false,
        active: true,
        valid_from_epoch: 1,
        valid_until_epoch: 500,
    };
    let affected = AuthorityMember {
        id: Digest("authority-affected-1"),
        role: AuthorityRole::AffectedPartyRepresentative,
        authority_lineage: Digest("lineage-affected"),
        conflict: false,
        active: true,
        valid_from_epoch: 1,
        valid_until_epoch: 500,
    };
    let emergency = AuthorityMember {
        id: Digest("emergency-guardian-1"),
        role: AuthorityRole::EmergencyGuardian,
        authority_lineage: Digest("lineage-emergency"),
        conflict: false,
        active: true,
        valid_from_epoch: 1,
        valid_until_epoch: 500,
    };

    let roster = AuthorityRoster {
        version: roster_version,
        members: vec![safety, technical, public, affected, emergency],
    };
    let policy = GovernancePolicy::for_tier(
        ConsequenceTier::Critical,
        Digest("critical-scope-v1"),
        roster.version,
    );

    let decision = approve(
        evaluation,
        ConsequenceTier::Critical,
        Digest("critical-scope-v1"),
        candidate,
        evaluator,
        &[safety, technical, public],
        &roster,
        policy,
        Digest("independent-critical-quorum"),
        100,
    )
    .expect("critical PASS and independent quorum should authorize");
    assert_eq!(decision.candidate, candidate);
    assert_eq!(decision.evaluation, evaluation.id);
    assert_eq!(decision.tier, ConsequenceTier::Critical);
    assert_eq!(decision.id.approvers.len(), 3);

    // Authorization fails closed unless evaluation is explicitly PASS.
    for verdict in [Verdict::Fail, Verdict::Indeterminate] {
        let nonpassing = EvaluationReceipt { verdict, ..evaluation };
        assert_eq!(
            approve(
                nonpassing,
                ConsequenceTier::Critical,
                Digest("critical-scope-v1"),
                candidate,
                evaluator,
                &[safety, technical, public],
                &roster,
                policy,
                Digest("nonpassing-must-not-authorize"),
                100,
            ),
            Err(QuorumFailure::NonPassingEvaluation)
        );
    }

    // Same signers repeated do not create a quorum.
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, safety, public],
            &roster,
            policy,
            Digest("duplicate-signer"),
            101,
        ),
        Err(QuorumFailure::DuplicateMember)
    );

    // Candidate and evaluator cannot manufacture authority.
    let candidate_member = AuthorityMember { id: candidate, ..public };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[candidate_member, technical, public],
            &roster,
            policy,
            Digest("candidate-self-authorized"),
            102,
        ),
        Err(QuorumFailure::CandidateIsApprover)
    );
    let evaluator_member = AuthorityMember { id: evaluator, ..technical };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, evaluator_member, public],
            &roster,
            policy,
            Digest("evaluator-self-authorized"),
            103,
        ),
        Err(QuorumFailure::EvaluatorIsApprover)
    );

    // Shared institutional lineage blocks a nominally plural quorum.
    let captured_technical = AuthorityMember {
        id: Digest("authority-captured-technical"),
        authority_lineage: Digest("lineage-safety"),
        ..technical
    };
    let captured_roster = AuthorityRoster {
        version: Digest("roster-captured-v1"),
        members: vec![safety, captured_technical, public, affected, emergency],
    };
    let captured_policy = GovernancePolicy {
        roster_version: captured_roster.version,
        ..policy
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, captured_technical, public],
            &captured_roster,
            captured_policy,
            Digest("shared-lineage"),
            104,
        ),
        Err(QuorumFailure::SharedAuthorityLineage)
    );

    // Conflict of interest is a hard quorum exclusion.
    let conflicted_technical = AuthorityMember {
        conflict: true,
        ..technical
    };
    let conflicted_roster = AuthorityRoster {
        version: Digest("roster-conflicted-v1"),
        members: vec![safety, conflicted_technical, public, affected, emergency],
    };
    let conflicted_policy = GovernancePolicy {
        roster_version: conflicted_roster.version,
        ..policy
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, conflicted_technical, public],
            &conflicted_roster,
            conflicted_policy,
            Digest("conflicted"),
            105,
        ),
        Err(QuorumFailure::ConflictOfInterest)
    );

    // Membership changes produce a different structured identity and force
    // current-state validation to reject the old decision.
    let changed_public = AuthorityMember {
        id: Digest("authority-public-2"),
        authority_lineage: Digest("lineage-public-2"),
        ..public
    };
    let changed_roster = AuthorityRoster {
        version: Digest("roster-v2"),
        members: vec![safety, technical, changed_public, affected, emergency],
    };
    let changed_policy = GovernancePolicy {
        version: Digest("governance-policy-v2"),
        roster_version: changed_roster.version,
        ..policy
    };
    assert_ne!(decision.id.roster_version, changed_roster.version);
    assert_eq!(
        validate_current_decision(
            &decision,
            changed_policy,
            &changed_roster,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            110,
        ),
        Err(QuorumFailure::DecisionNotCurrent)
    );

    // Decision identity includes the exact scope, policy, roster, members,
    // rationale and effective time; changing any one creates another identity.
    let expanded_policy = GovernancePolicy {
        version: Digest("governance-policy-expanded-v1"),
        scope: Digest("expanded-critical-scope"),
        ..policy
    };
    let expanded = approve(
        evaluation,
        ConsequenceTier::Critical,
        Digest("expanded-critical-scope"),
        candidate,
        evaluator,
        &[safety, technical, public],
        &roster,
        expanded_policy,
        Digest("expanded-scope"),
        120,
    )
    .expect("scope expansion requires a new matching policy and quorum");
    assert_ne!(decision.id, expanded.id);

    // A different eligible approver set for the same scope/time creates a
    // different identity even when the policy and candidate are unchanged.
    let alternate_quorum = approve(
        evaluation,
        ConsequenceTier::Critical,
        Digest("critical-scope-v1"),
        candidate,
        evaluator,
        &[safety, technical, affected],
        &roster,
        policy,
        Digest("independent-critical-quorum"),
        100,
    )
    .expect("alternate independent quorum should be representable");
    assert_ne!(decision.id, alternate_quorum.id);

    // Higher consequence cannot be retroactively inferred from a lower-tier
    // evaluation; the evaluator must explicitly evaluate at the requested tier.
    let low_evaluation = EvaluationReceipt {
        id: Digest("evaluation-low-v1"),
        tier: ConsequenceTier::Low,
        ..evaluation
    };
    assert_eq!(
        approve(
            low_evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, technical, public],
            &roster,
            policy,
            Digest("tier-upgrade"),
            106,
        ),
        Err(QuorumFailure::TierMismatch)
    );

    // Stale roster/policy and stale evaluation cannot be reused.
    let stale_roster = AuthorityRoster {
        version: Digest("roster-v2"),
        ..roster.clone()
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, technical, public],
            &stale_roster,
            policy,
            Digest("stale-roster"),
            107,
        ),
        Err(QuorumFailure::StaleRoster)
    );

    let stale_evaluation = EvaluationReceipt {
        issued_epoch: 60,
        ..evaluation
    };
    assert_eq!(
        approve(
            stale_evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, technical, public],
            &roster,
            policy,
            Digest("stale-evaluation"),
            100,
        ),
        Err(QuorumFailure::EvaluationTooOld)
    );

    // Emergency suspension has an event identity that binds reason, roster,
    // policy, authority, deployment, and effective time.
    let suspension = emergency_suspend(
        emergency,
        Digest("deployment-v1"),
        Digest("immediate-safety-concern"),
        policy,
        &roster,
        110,
    )
    .expect("registered emergency guardian may suspend");
    let second_suspension = emergency_suspend(
        emergency,
        Digest("deployment-v1"),
        Digest("second-safety-event"),
        policy,
        &roster,
        110,
    )
    .expect("distinct emergency reason is a distinct event");
    assert_ne!(suspension.id, second_suspension.id);

    assert_eq!(
        emergency_suspend(
            safety,
            Digest("deployment-v1"),
            Digest("fake-emergency"),
            policy,
            &roster,
            111,
        ),
        Err(QuorumFailure::MemberNotInCurrentRoster)
    );

    // Historical decisions retain their immutable roster/policy snapshot.
    assert_eq!(decision.id.roster_version, Digest("roster-v1"));
    assert_eq!(decision.id.policy.version, Digest("governance-policy-v1"));
    assert_eq!(decision.id.approvers, vec![public, safety, technical]);

    println!("SYM-CIV-006 PASS: plural-authority, policy-freshness and quorum controls hold.");
    println!("Claim ceiling: structured local identity/type-flow fixture only; not cryptographic proof or democratic legitimacy.");
}

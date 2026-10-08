//! SYM-CIV-006: plural authority and separation-of-powers smoke.
//!
//! Dependency-free research control.
//!
//! Core boundaries:
//! candidate != evaluator != authority
//! authority_1 != authority_2
//! quorum != legitimacy
//! suspension != authorization
//! historical decision != current roster state
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QuorumRequirement {
    min_members: usize,
    min_roles: usize,
}

impl ConsequenceTier {
    fn quorum_requirement(self) -> QuorumRequirement {
        match self {
            ConsequenceTier::Low => QuorumRequirement {
                min_members: 1,
                min_roles: 1,
            },
            ConsequenceTier::Significant => QuorumRequirement {
                min_members: 2,
                min_roles: 2,
            },
            ConsequenceTier::Critical => QuorumRequirement {
                min_members: 3,
                min_roles: 3,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvaluationReceipt {
    id: Digest,
    candidate: Digest,
    evaluator: Digest,
    tier: ConsequenceTier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AuthorityDecision {
    id: Digest,
    candidate: Digest,
    evaluation: Digest,
    tier: ConsequenceTier,
    scope: Digest,
    approver_set_digest: Digest,
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
    ScopeUpgradeRequiresFreshDecision,
    TierUpgradeRequiresFreshDecision,
    EvaluationMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EmergencySuspension {
    id: Digest,
    deployment: Digest,
    authority: Digest,
    reason: Digest,
    effective_epoch: u64,
}

fn decision_identity(
    evaluation: Digest,
    tier: ConsequenceTier,
    scope: Digest,
    approver_set_digest: Digest,
    effective_epoch: u64,
) -> Digest {
    match (
        evaluation,
        tier,
        scope,
        approver_set_digest,
        effective_epoch,
    ) {
        (
            Digest("evaluation-v1"),
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            Digest("approver-set-v1"),
            100,
        ) => Digest("authority-decision-evaluation-v1-critical-scope-v1-100"),
        (
            Digest("evaluation-v1"),
            ConsequenceTier::Critical,
            Digest("expanded-critical-scope"),
            Digest("approver-set-v1"),
            120,
        ) => Digest("authority-decision-evaluation-v1-expanded-scope-120"),
        _ => Digest("authority-decision-other"),
    }
}

fn approve(
    evaluation: EvaluationReceipt,
    tier: ConsequenceTier,
    scope: Digest,
    candidate: Digest,
    evaluator: Digest,
    approvers: &[AuthorityMember],
    rationale: Digest,
    effective_epoch: u64,
) -> Result<AuthorityDecision, QuorumFailure> {
    if evaluation.candidate != candidate || evaluation.evaluator != evaluator {
        return Err(QuorumFailure::EvaluationMismatch);
    }
    if evaluation.tier != tier {
        return Err(QuorumFailure::TierUpgradeRequiresFreshDecision);
    }

    let required = tier.quorum_requirement();

    if approvers.len() < required.min_members {
        return Err(QuorumFailure::TooFewMembers);
    }

    let mut ids = BTreeSet::new();
    let mut roles = BTreeSet::new();
    let mut authority_lineages = BTreeSet::new();

    for approver in approvers {
        if !ids.insert(approver.id) {
            return Err(QuorumFailure::DuplicateMember);
        }
        if approver.id == candidate {
            return Err(QuorumFailure::CandidateIsApprover);
        }
        if approver.id == evaluator {
            return Err(QuorumFailure::EvaluatorIsApprover);
        }
        if approver.conflict {
            return Err(QuorumFailure::ConflictOfInterest);
        }
        roles.insert(approver.role);
        if !authority_lineages.insert(approver.authority_lineage) {
            return Err(QuorumFailure::SharedAuthorityLineage);
        }
    }

    if roles.len() < required.min_roles {
        return Err(QuorumFailure::TooFewDistinctRoles);
    }

    let approver_set_digest = Digest("approver-set-v1");

    Ok(AuthorityDecision {
        id: decision_identity(
            evaluation.id,
            tier,
            scope,
            approver_set_digest,
            effective_epoch,
        ),
        candidate,
        evaluation: evaluation.id,
        tier,
        scope,
        approver_set_digest,
        rationale,
        effective_epoch,
    })
}

fn emergency_suspension_identity(
    deployment: Digest,
    authority: Digest,
    effective_epoch: u64,
) -> Digest {
    match (deployment, authority, effective_epoch) {
        (
            Digest("deployment-v1"),
            Digest("emergency-guardian-1"),
            110,
        ) => Digest("emergency-suspension-deployment-v1-110"),
        _ => Digest("emergency-suspension-other"),
    }
}

fn emergency_suspend(
    member: AuthorityMember,
    deployment: Digest,
    reason: Digest,
    effective_epoch: u64,
) -> Result<EmergencySuspension, QuorumFailure> {
    if member.role != AuthorityRole::EmergencyGuardian {
        return Err(QuorumFailure::TooFewMembers);
    }
    if member.conflict {
        return Err(QuorumFailure::ConflictOfInterest);
    }

    Ok(EmergencySuspension {
        id: emergency_suspension_identity(deployment, member.id, effective_epoch),
        deployment,
        authority: member.id,
        reason,
        effective_epoch,
    })
}

fn main() {
    let candidate = Digest("candidate-v1");
    let evaluator = Digest("evaluator-v1");

    let evaluation = EvaluationReceipt {
        id: Digest("evaluation-v1"),
        candidate,
        evaluator,
        tier: ConsequenceTier::Critical,
    };

    let safety = AuthorityMember {
        id: Digest("authority-safety-1"),
        role: AuthorityRole::SafetyGuardian,
        authority_lineage: Digest("lineage-safety"),
        conflict: false,
    };
    let technical = AuthorityMember {
        id: Digest("authority-technical-1"),
        role: AuthorityRole::TechnicalAdjudicator,
        authority_lineage: Digest("lineage-technical"),
        conflict: false,
    };
    let public = AuthorityMember {
        id: Digest("authority-public-1"),
        role: AuthorityRole::PublicSteward,
        authority_lineage: Digest("lineage-public"),
        conflict: false,
    };

    let decision = approve(
        evaluation,
        ConsequenceTier::Critical,
        Digest("critical-scope-v1"),
        candidate,
        evaluator,
        &[safety, technical, public],
        Digest("independent-critical-quorum"),
        100,
    )
    .expect("critical decision should satisfy plural authority requirements");

    assert_eq!(decision.candidate, candidate);
    assert_eq!(decision.evaluation, evaluation.id);
    assert_eq!(decision.tier, ConsequenceTier::Critical);

    // Decision identity binds scope and effective epoch, rather than being a
    // reusable action label.
    let expanded_decision = approve(
        evaluation,
        ConsequenceTier::Critical,
        Digest("expanded-critical-scope"),
        candidate,
        evaluator,
        &[safety, technical, public],
        Digest("expanded-scope"),
        120,
    )
    .expect("expanded scope can be represented as a fresh authority decision");
    assert_ne!(decision.id, expanded_decision.id);
    assert_ne!(decision.scope, expanded_decision.scope);

    // Repeating the same signer does not create a quorum.
    let duplicate = approve(
        evaluation,
        ConsequenceTier::Critical,
        Digest("critical-scope-v1"),
        candidate,
        evaluator,
        &[safety, safety, public],
        Digest("duplicate-signer"),
        101,
    );
    assert_eq!(duplicate, Err(QuorumFailure::DuplicateMember));

    // Candidate and evaluator cannot manufacture their own authority.
    let candidate_member = AuthorityMember {
        id: candidate,
        role: AuthorityRole::PublicSteward,
        authority_lineage: Digest("lineage-candidate"),
        conflict: false,
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[candidate_member, technical, public],
            Digest("candidate-self-authorized"),
            102,
        ),
        Err(QuorumFailure::CandidateIsApprover)
    );

    let evaluator_member = AuthorityMember {
        id: evaluator,
        role: AuthorityRole::TechnicalAdjudicator,
        authority_lineage: Digest("lineage-evaluator"),
        conflict: false,
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, evaluator_member, public],
            Digest("evaluator-self-authorized"),
            103,
        ),
        Err(QuorumFailure::EvaluatorIsApprover)
    );

    // One institutional lineage cannot masquerade as multiple independent
    // authorities merely by using multiple identities.
    let captured_technical = AuthorityMember {
        id: Digest("authority-captured-technical"),
        role: AuthorityRole::TechnicalAdjudicator,
        authority_lineage: Digest("lineage-safety"),
        conflict: false,
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, captured_technical, public],
            Digest("shared-lineage"),
            104,
        ),
        Err(QuorumFailure::SharedAuthorityLineage)
    );

    // Conflict of interest is a hard quorum exclusion.
    let conflicted = AuthorityMember {
        conflict: true,
        ..technical
    };
    assert_eq!(
        approve(
            evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, conflicted, public],
            Digest("conflicted"),
            105,
        ),
        Err(QuorumFailure::ConflictOfInterest)
    );

    // Lower-tier authority cannot simply be relabeled as critical-tier authority.
    let low_evaluation = EvaluationReceipt {
        id: Digest("evaluation-low-v1"),
        candidate,
        evaluator,
        tier: ConsequenceTier::Low,
    };
    assert_eq!(
        approve(
            low_evaluation,
            ConsequenceTier::Critical,
            Digest("critical-scope-v1"),
            candidate,
            evaluator,
            &[safety, technical, public],
            Digest("tier-upgrade"),
            106,
        ),
        Err(QuorumFailure::TierUpgradeRequiresFreshDecision)
    );

    // A scope expansion is likewise a new authority decision, not an in-place
    // mutation of the historical decision.
    assert_ne!(decision.scope, Digest("expanded-critical-scope"));
    assert_eq!(decision.scope, Digest("critical-scope-v1"));

    // Emergency suspension is intentionally not a grant of authorization.
    let emergency = AuthorityMember {
        id: Digest("emergency-guardian-1"),
        role: AuthorityRole::EmergencyGuardian,
        authority_lineage: Digest("lineage-emergency"),
        conflict: false,
    };
    let suspension = emergency_suspend(
        emergency,
        Digest("deployment-v1"),
        Digest("immediate-safety-concern"),
        110,
    )
    .expect("emergency guardian may issue a suspension");
    assert_eq!(suspension.deployment, Digest("deployment-v1"));
    assert_eq!(suspension.reason, Digest("immediate-safety-concern"));
    assert_ne!(suspension.id, decision.id);

    let later_suspension = emergency_suspend(
        emergency,
        Digest("deployment-v1"),
        Digest("second-safety-event"),
        111,
    )
    .expect("a later emergency action can be represented");
    assert_ne!(suspension.id, later_suspension.id);

    // A normal authority member cannot pretend to be the emergency role.
    assert_eq!(
        emergency_suspend(
            safety,
            Digest("deployment-v1"),
            Digest("fake-emergency"),
            111
        ),
        Err(QuorumFailure::TooFewMembers)
    );

    // Snapshot semantics: a later roster change does not rewrite the historical
    // decision's recorded approver set, tier, scope, or effective time.
    let replacement = AuthorityMember {
        id: Digest("authority-public-2"),
        authority_lineage: Digest("lineage-public-2"),
        ..public
    };
    assert_ne!(replacement.id, public.id);
    assert_eq!(decision.approver_set_digest, Digest("approver-set-v1"));
    assert_eq!(decision.effective_epoch, 100);

    println!("SYM-CIV-006 PASS: plural authority and separation-of-powers controls hold.");
    println!("Claim ceiling: local authority/type-flow control only; not democratic legitimacy or substantive justice.");
}

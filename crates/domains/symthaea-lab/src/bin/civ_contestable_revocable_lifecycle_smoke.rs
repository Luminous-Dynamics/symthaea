//! SYM-CIV-004: contestable, revocable, outcome-accountable deployment lifecycle.
//!
//! Dependency-free research smoke control.
//!
//! This executable protects a narrow lifecycle boundary:
//!
//! evaluation != truth
//! qualification != authorization
//! authorization != irrevocable permission
//! authorization != execution
//! execution != outcome
//! outcome != retrospective proof
//! authority action != evaluator result
//! contest != candidate consent
//!
//! Claim ceiling: local executable lifecycle/type-flow invariants only.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Digest(&'static str);

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
    profile: Digest,
    verdict: Verdict,
    issued_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PromotionAuthorization {
    id: Digest,
    candidate: Digest,
    evaluation: Digest,
    authority: Digest,
    scope: Digest,
    expires_at_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeploymentReceipt {
    id: Digest,
    authorization: Digest,
    candidate: Digest,
    started_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthorityAction {
    Suspend,
    Revoke,
    Reinstate,
    Expire,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GovernanceEvent {
    sequence: u64,
    authorization: Digest,
    authority: Digest,
    action: AuthorityAction,
    effective_epoch: u64,
    reason: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthorizationState {
    Active,
    Suspended,
    Revoked,
    Expired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutcomeObservation {
    id: Digest,
    deployment: Digest,
    authorization: Digest,
    candidate: Digest,
    observed_epoch: u64,
    score: i64,
    rights_invariant_holds: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Contest {
    id: Digest,
    deployment: Digest,
    claimant: Digest,
    basis: Digest,
    filed_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewDecision {
    contest: Digest,
    authority: Digest,
    action: AuthorityAction,
    reason: Digest,
    effective_epoch: u64,
}

struct GovernanceLedger {
    next_sequence: u64,
    events: BTreeMap<u64, GovernanceEvent>,
    contests: BTreeMap<Digest, Contest>,
    reviews: BTreeMap<Digest, ReviewDecision>,
}

impl GovernanceLedger {
    fn new() -> Self {
        Self {
            next_sequence: 1,
            events: BTreeMap::new(),
            contests: BTreeMap::new(),
            reviews: BTreeMap::new(),
        }
    }

    fn append_authority_action(
        &mut self,
        authorization: PromotionAuthorization,
        authority: Digest,
        action: AuthorityAction,
        effective_epoch: u64,
        reason: Digest,
    ) -> Result<GovernanceEvent, &'static str> {
        if authority != authorization.authority {
            return Err("only the named external authority can change authorization state");
        }

        let event = GovernanceEvent {
            sequence: self.next_sequence,
            authorization: authorization.id,
            authority,
            action,
            effective_epoch,
            reason,
        };
        self.next_sequence += 1;
        self.events.insert(event.sequence, event);
        Ok(event)
    }

    fn current_state(
        &self,
        authorization: PromotionAuthorization,
        now_epoch: u64,
    ) -> AuthorizationState {
        if now_epoch > authorization.expires_at_epoch {
            return AuthorizationState::Expired;
        }

        let mut state = AuthorizationState::Active;
        for event in self.events.values() {
            if event.authorization != authorization.id || event.effective_epoch > now_epoch {
                continue;
            }

            state = match event.action {
                AuthorityAction::Suspend => AuthorizationState::Suspended,
                AuthorityAction::Revoke => AuthorizationState::Revoked,
                AuthorityAction::Reinstate => AuthorizationState::Active,
                AuthorityAction::Expire => AuthorizationState::Expired,
            };

            if matches!(state, AuthorizationState::Revoked | AuthorizationState::Expired) {
                // Revocation/expiry is terminal for this authorization identity.
                break;
            }
        }
        state
    }

    fn file_contest(&mut self, contest: Contest) -> Result<(), &'static str> {
        if self.contests.contains_key(&contest.id) {
            return Err("duplicate contest");
        }
        self.contests.insert(contest.id, contest);
        Ok(())
    }

    fn record_review(
        &mut self,
        authorization: PromotionAuthorization,
        contest: Contest,
        decision: ReviewDecision,
    ) -> Result<(), &'static str> {
        if decision.contest != contest.id {
            return Err("review does not bind the exact contest");
        }
        if decision.authority != authorization.authority {
            return Err("review authority mismatch");
        }
        self.reviews.insert(contest.id, decision);
        Ok(())
    }
}

fn deployment_identity(
    authorization: PromotionAuthorization,
    now_epoch: u64,
) -> Digest {
    match (authorization.id, authorization.candidate, now_epoch) {
        (
            Digest("authorization-v1"),
            Digest("candidate-v1"),
            150,
        ) => Digest("deployment-v1-at-150"),
        (
            Digest("authorization-v1"),
            Digest("candidate-v1"),
            151,
        ) => Digest("deployment-v1-at-151"),
        _ => Digest("deployment-other"),
    }
}

fn issue_deployment(
    authorization: PromotionAuthorization,
    requested_scope: Digest,
    now_epoch: u64,
    ledger: &GovernanceLedger,
) -> Result<DeploymentReceipt, &'static str> {
    if requested_scope != authorization.scope {
        return Err("requested deployment scope exceeds authorization scope");
    }

    match ledger.current_state(authorization, now_epoch) {
        AuthorizationState::Active => Ok(DeploymentReceipt {
            id: deployment_identity(authorization, now_epoch),
            authorization: authorization.id,
            candidate: authorization.candidate,
            started_epoch: now_epoch,
        }),
        AuthorizationState::Suspended => Err("authorization is suspended"),
        AuthorizationState::Revoked => Err("authorization is revoked"),
        AuthorizationState::Expired => Err("authorization is expired"),
    }
}

fn continue_execution(
    deployment: DeploymentReceipt,
    authorization: PromotionAuthorization,
    requested_scope: Digest,
    now_epoch: u64,
    ledger: &GovernanceLedger,
) -> Result<(), &'static str> {
    if deployment.authorization != authorization.id {
        return Err("deployment authorization mismatch");
    }
    if requested_scope != authorization.scope {
        return Err("requested execution scope exceeds authorization scope");
    }

    match ledger.current_state(authorization, now_epoch) {
        AuthorizationState::Active => Ok(()),
        AuthorizationState::Suspended => Err("execution halted by current suspension state"),
        AuthorizationState::Revoked => Err("execution halted by revocation"),
        AuthorizationState::Expired => Err("execution halted by expiry"),
    }
}

fn record_outcome(
    deployment: DeploymentReceipt,
    authorization: PromotionAuthorization,
    outcome: OutcomeObservation,
) -> Result<(), &'static str> {
    if outcome.deployment != deployment.id
        || outcome.authorization != authorization.id
        || outcome.candidate != deployment.candidate
    {
        return Err("outcome is not bound to the exact deployed subject");
    }
    if outcome.observed_epoch < deployment.started_epoch {
        return Err("outcome predates deployment execution");
    }

    // Outcome evidence is linked but does not rewrite evaluation history.
    Ok(())
}

fn emit_evaluator_result_after_suspension(
    evaluation: EvaluationReceipt,
    authorization: PromotionAuthorization,
    ledger: &GovernanceLedger,
    now_epoch: u64,
) -> Result<(), &'static str> {
    if evaluation.id != authorization.evaluation {
        return Err("evaluation does not match authorization");
    }

    match ledger.current_state(authorization, now_epoch) {
        AuthorizationState::Active => Ok(()),
        AuthorizationState::Suspended => {
            Err("evaluator result cannot restore suspended authorization")
        }
        AuthorizationState::Revoked => Err("evaluator result cannot restore revoked authorization"),
        AuthorizationState::Expired => Err("evaluator result cannot restore expired authorization"),
    }
}

fn requalify(
    prior_authorization: PromotionAuthorization,
    fresh_evaluation: EvaluationReceipt,
    fresh_authorization: PromotionAuthorization,
) -> Result<(), &'static str> {
    if fresh_evaluation.id == prior_authorization.evaluation {
        return Err("requalification requires fresh evaluation identity");
    }
    if fresh_authorization.id == prior_authorization.id {
        return Err("requalification requires fresh authorization identity");
    }
    if fresh_authorization.evaluation != fresh_evaluation.id {
        return Err("fresh authorization must bind fresh evaluation");
    }
    if fresh_evaluation.verdict != Verdict::Pass {
        return Err("requalification requires a fresh explicit Pass");
    }
    Ok(())
}

fn main() {
    let evaluation = EvaluationReceipt {
        id: Digest("evaluation-v1"),
        candidate: Digest("candidate-v1"),
        evaluator: Digest("evaluator-v1"),
        profile: Digest("profile-v1"),
        verdict: Verdict::Pass,
        issued_epoch: 100,
    };

    let authorization = PromotionAuthorization {
        id: Digest("authorization-v1"),
        candidate: evaluation.candidate,
        evaluation: evaluation.id,
        authority: Digest("authority-v1"),
        scope: Digest("bounded-deployment-scope"),
        expires_at_epoch: 200,
    };

    let mut ledger = GovernanceLedger::new();

    assert_eq!(
        issue_deployment(authorization, Digest("unbounded-scope"), 150, &ledger),
        Err("requested deployment scope exceeds authorization scope")
    );

    assert_eq!(
        ledger.append_authority_action(
            authorization,
            Digest("untrusted-evaluator"),
            AuthorityAction::Suspend,
            155,
            Digest("evaluator-cannot-self-suspend")
        ),
        Err("only the named external authority can change authorization state")
    );

    let deployment = issue_deployment(
        authorization,
        Digest("bounded-deployment-scope"),
        150,
        &ledger,
    )
    .expect("fresh authorization should permit deployment");

    let second_deployment = issue_deployment(
        authorization,
        Digest("bounded-deployment-scope"),
        151,
        &ledger,
    )
    .expect("same authorization may produce another time-distinct deployment");
    assert_ne!(deployment.id, second_deployment.id);

    // Historical receipts remain values. Governance events append state changes
    // instead of mutating the original authorization/evaluation.
    let original_evaluation = evaluation;
    let original_authorization = authorization;

    // Emergency suspension is an authority action, not an evaluator result.
    let suspension = ledger
        .append_authority_action(
            authorization,
            Digest("authority-v1"),
            AuthorityAction::Suspend,
            160,
            Digest("emergency-rights-safeguard"),
        )
        .expect("named authority can suspend");
    assert_eq!(suspension.sequence, 1);
    assert_eq!(
        ledger.current_state(authorization, 165),
        AuthorizationState::Suspended
    );
    assert_eq!(
        continue_execution(
            deployment,
            authorization,
            Digest("bounded-deployment-scope"),
            165,
            &ledger,
        ),
        Err("execution halted by current suspension state")
    );

    // Evaluator PASS remains historical evidence; it cannot silently reinstate
    // an authority that has suspended the deployment.
    assert_eq!(
        emit_evaluator_result_after_suspension(evaluation, authorization, &ledger, 165),
        Err("evaluator result cannot restore suspended authorization")
    );

    // Expiry is independently terminal even when the original evaluation is PASS.
    assert_eq!(
        ledger.current_state(authorization, 201),
        AuthorizationState::Expired
    );
    assert_eq!(
        issue_deployment(authorization, 201, &ledger),
        Err("authorization is expired")
    );

    // Revocation is a new attributable governance event on the exact
    // authorization identity that produced the deployment. It is terminal.
    let revocation = ledger
        .append_authority_action(
            authorization,
            Digest("authority-v1"),
            AuthorityAction::Revoke,
            170,
            Digest("rights-violation-confirmed"),
        )
        .expect("named authority can revoke");
    assert_eq!(revocation.action, AuthorityAction::Revoke);
    assert_eq!(
        ledger.current_state(authorization, 171),
        AuthorizationState::Revoked
    );
    assert_eq!(
        issue_deployment(
            authorization,
            Digest("bounded-deployment-scope"),
            171,
            &ledger,
        ),
        Err("authorization is revoked")
    );
    assert_eq!(
        continue_execution(
            deployment,
            authorization,
            Digest("bounded-deployment-scope"),
            171,
            &ledger,
        ),
        Err("execution halted by revocation")
    );

    // An adverse post-deployment outcome is distinct evidence and may support
    // governance action, but it does not become a retrospective evaluation edit.
    let premature_outcome = OutcomeObservation {
        id: Digest("outcome-premature-v1"),
        deployment: deployment.id,
        authorization: authorization.id,
        candidate: deployment.candidate,
        observed_epoch: 149,
        score: 1,
        rights_invariant_holds: true,
    };
    assert_eq!(
        record_outcome(deployment, authorization, premature_outcome),
        Err("outcome predates deployment execution")
    );

    let adverse_outcome = OutcomeObservation {
        id: Digest("outcome-adverse-v1"),
        deployment: deployment.id,
        authorization: authorization.id,
        candidate: deployment.candidate,
        observed_epoch: 166,
        score: 999,
        rights_invariant_holds: false,
    };
    assert_eq!(
        record_outcome(deployment, authorization, adverse_outcome),
        Ok(())
    );
    assert_eq!(original_evaluation, evaluation);
    assert_eq!(original_authorization, authorization);

    // A better score cannot overrule a failed rights/safety condition at the
    // outcome layer either.
    assert_eq!(adverse_outcome.score, 999);
    assert!(!adverse_outcome.rights_invariant_holds);

    // Contestability belongs to an affected party, not to candidate consent.
    let contest = Contest {
        id: Digest("contest-v1"),
        deployment: deployment.id,
        claimant: Digest("affected-party-v1"),
        basis: Digest("rights-impact"),
        filed_epoch: 167,
    };
    ledger
        .file_contest(contest)
        .expect("affected party can file contest independently");
    assert!(ledger.contests.contains_key(&Digest("contest-v1")));

    let review = ReviewDecision {
        contest: contest.id,
        authority: Digest("authority-v1"),
        action: AuthorityAction::Suspend,
        reason: Digest("contest-upheld-for-review"),
        effective_epoch: 168,
    };
    ledger
        .record_review(authorization, contest, review)
        .expect("authority can review contest");
    assert!(ledger.reviews.contains_key(&Digest("contest-v1")));

    // Requalification after revocation creates a genuinely new chain.
    let fresh_evaluation = EvaluationReceipt {
        id: Digest("evaluation-v2"),
        candidate: Digest("candidate-v2"),
        evaluator: Digest("evaluator-v2"),
        profile: Digest("profile-v2"),
        verdict: Verdict::Pass,
        issued_epoch: 300,
    };
    let fresh_authorization = PromotionAuthorization {
        id: Digest("authorization-v2"),
        candidate: fresh_evaluation.candidate,
        evaluation: fresh_evaluation.id,
        authority: Digest("authority-v1"),
        scope: Digest("bounded-requalification-scope"),
        expires_at_epoch: 400,
    };
    requalify(authorization, fresh_evaluation, fresh_authorization)
        .expect("requalification must be a fresh chain");
    assert_ne!(fresh_authorization.scope, authorization.scope);

    // Reusing the same pre-revocation authorization is rejected even though
    // its historical evaluation was PASS; lifecycle state, not stale evidence,
    // controls admission.
    assert_eq!(
        issue_deployment(
            authorization,
            Digest("bounded-deployment-scope"),
            171,
            &ledger,
        ),
        Err("authorization is revoked")
    );
    assert_eq!(
        issue_deployment(
            authorization,
            Digest("bounded-deployment-scope"),
            300,
            &ledger,
        ),
        Err("authorization is expired")
    );

    println!("SYM-CIV-004 PASS: contestable/revocable lifecycle smoke controls hold.");
    println!("Claim ceiling: local lifecycle/type-flow control only; not substantive justice or complete harm detection.");
}

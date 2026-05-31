// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Access Control Tests
//!
//! Tests for party access control, role-based permissions, and authorization
//! in the mycelix-justice dispute resolution system.

#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod party_access_tests {
    use super::*;

    // ==========================================
    // Case Party Access Tests
    // ==========================================

    #[test]
    fn test_complainant_can_access_own_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let access = check_case_access("case-001", "did:mycelix:alice");
        assert!(
            access.can_view,
            "Complainant should be able to view own case"
        );
        assert!(
            access.can_submit_evidence,
            "Complainant should be able to submit evidence"
        );
        assert!(
            access.can_update,
            "Complainant should be able to update own case"
        );
    }

    #[test]
    fn test_respondent_can_access_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let access = check_case_access("case-001", "did:mycelix:bob");
        assert!(access.can_view, "Respondent should be able to view case");
        assert!(
            access.can_submit_evidence,
            "Respondent should be able to submit evidence"
        );
        assert!(
            !access.can_withdraw,
            "Respondent should not be able to withdraw case"
        );
    }

    #[test]
    fn test_third_party_cannot_access_private_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let access = check_case_access("case-001", "did:mycelix:charlie");
        assert!(
            !access.can_view,
            "Third party should not be able to view private case"
        );
        assert!(
            !access.can_submit_evidence,
            "Third party should not be able to submit evidence"
        );
    }

    #[test]
    fn test_witness_has_limited_access() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        add_party_to_case("case-001", "did:mycelix:witness1", PartyRole::Witness);

        let access = check_case_access("case-001", "did:mycelix:witness1");
        assert!(access.can_view, "Witness should be able to view case");
        assert!(
            access.can_submit_evidence,
            "Witness should be able to submit testimony"
        );
        assert!(
            !access.can_update,
            "Witness should not be able to update case"
        );
    }

    #[test]
    fn test_expert_has_specialized_access() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        add_party_to_case("case-001", "did:mycelix:expert1", PartyRole::Expert);

        let access = check_case_access("case-001", "did:mycelix:expert1");
        assert!(access.can_view, "Expert should be able to view case");
        assert!(
            access.can_submit_evidence,
            "Expert should be able to submit expert opinion"
        );
        assert!(
            !access.can_update,
            "Expert should not be able to update case"
        );
    }

    #[test]
    fn test_complainant_can_withdraw_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let access = check_case_access("case-001", "did:mycelix:alice");
        assert!(
            access.can_withdraw,
            "Complainant should be able to withdraw case"
        );
    }

    #[test]
    fn test_respondent_cannot_withdraw_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let access = check_case_access("case-001", "did:mycelix:bob");
        assert!(
            !access.can_withdraw,
            "Respondent should not be able to withdraw case"
        );
    }
}

#[cfg(test)]
mod arbitrator_access_tests {
    use super::*;

    // ==========================================
    // Arbitrator Access Tests
    // ==========================================

    #[test]
    fn test_assigned_arbitrator_can_access_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        assign_arbitrator("case-001", "did:mycelix:arb1");

        let access = check_case_access("case-001", "did:mycelix:arb1");
        assert!(
            access.can_view,
            "Assigned arbitrator should be able to view case"
        );
        assert!(
            access.can_view_all_evidence,
            "Arbitrator should be able to view all evidence"
        );
    }

    #[test]
    fn test_unassigned_arbitrator_cannot_access_case() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        // arb2 is not assigned to this case

        let access = check_case_access("case-001", "did:mycelix:arb2");
        assert!(
            !access.can_view,
            "Unassigned arbitrator should not be able to view case"
        );
    }

    #[test]
    fn test_arbitrator_can_render_decision() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        assign_arbitrator("case-001", "did:mycelix:arb1");

        let access = check_case_access("case-001", "did:mycelix:arb1");
        assert!(
            access.can_render_decision,
            "Arbitrator should be able to render decision"
        );
    }

    #[test]
    fn test_party_cannot_render_decision() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let alice_access = check_case_access("case-001", "did:mycelix:alice");
        let bob_access = check_case_access("case-001", "did:mycelix:bob");

        assert!(
            !alice_access.can_render_decision,
            "Complainant should not be able to render decision"
        );
        assert!(
            !bob_access.can_render_decision,
            "Respondent should not be able to render decision"
        );
    }

    #[test]
    fn test_recused_arbitrator_loses_access() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        assign_arbitrator("case-001", "did:mycelix:arb1");
        recuse_arbitrator("case-001", "did:mycelix:arb1");

        let access = check_case_access("case-001", "did:mycelix:arb1");
        assert!(
            !access.can_view,
            "Recused arbitrator should lose access to case"
        );
        assert!(
            !access.can_render_decision,
            "Recused arbitrator should not render decisions"
        );
    }

    #[test]
    fn test_arbitrator_panel_all_have_access() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        assign_arbitrator("case-001", "did:mycelix:arb1");
        assign_arbitrator("case-001", "did:mycelix:arb2");
        assign_arbitrator("case-001", "did:mycelix:arb3");

        for arb in &["did:mycelix:arb1", "did:mycelix:arb2", "did:mycelix:arb3"] {
            let access = check_case_access("case-001", arb);
            assert!(access.can_view, "All panel arbitrators should have access");
            assert!(
                access.can_render_decision,
                "All panel arbitrators should be able to vote"
            );
        }
    }
}

#[cfg(test)]
mod evidence_access_tests {
    use super::*;

    // ==========================================
    // Evidence Access Tests
    // ==========================================

    #[test]
    fn test_parties_can_view_public_evidence() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        create_evidence("evidence-001", "case-001", EvidenceVisibility::AllParties);

        let alice_access = check_evidence_access("evidence-001", "did:mycelix:alice");
        let bob_access = check_evidence_access("evidence-001", "did:mycelix:bob");

        assert!(
            alice_access.can_view,
            "Complainant should view public evidence"
        );
        assert!(
            bob_access.can_view,
            "Respondent should view public evidence"
        );
    }

    #[test]
    fn test_adjudicators_only_evidence() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        assign_arbitrator("case-001", "did:mycelix:arb1");
        create_evidence(
            "evidence-001",
            "case-001",
            EvidenceVisibility::AdjudicatorsOnly,
        );

        let arb_access = check_evidence_access("evidence-001", "did:mycelix:arb1");
        let alice_access = check_evidence_access("evidence-001", "did:mycelix:alice");
        let bob_access = check_evidence_access("evidence-001", "did:mycelix:bob");

        assert!(
            arb_access.can_view,
            "Arbitrator should view adjudicators-only evidence"
        );
        assert!(
            !alice_access.can_view,
            "Complainant should not view adjudicators-only evidence"
        );
        assert!(
            !bob_access.can_view,
            "Respondent should not view adjudicators-only evidence"
        );
    }

    #[test]
    fn test_restricted_evidence_access() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        create_evidence_with_restriction(
            "evidence-001",
            "case-001",
            vec!["did:mycelix:alice".to_string()], // Only Alice can view
        );

        let alice_access = check_evidence_access("evidence-001", "did:mycelix:alice");
        let bob_access = check_evidence_access("evidence-001", "did:mycelix:bob");

        assert!(
            alice_access.can_view,
            "Alice should view restricted evidence"
        );
        assert!(
            !bob_access.can_view,
            "Bob should not view restricted evidence"
        );
    }

    #[test]
    fn test_sealed_evidence_no_access() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        assign_arbitrator("case-001", "did:mycelix:arb1");
        create_evidence("evidence-001", "case-001", EvidenceVisibility::Sealed);

        let arb_access = check_evidence_access("evidence-001", "did:mycelix:arb1");
        let alice_access = check_evidence_access("evidence-001", "did:mycelix:alice");

        assert!(
            !arb_access.can_view,
            "Arbitrator should not view sealed evidence without order"
        );
        assert!(
            !alice_access.can_view,
            "Party should not view sealed evidence"
        );
    }

    #[test]
    fn test_evidence_submitter_can_always_view_own() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        create_evidence_by_submitter(
            "evidence-001",
            "case-001",
            "did:mycelix:alice",
            EvidenceVisibility::AdjudicatorsOnly,
        );

        // Even though it's adjudicators-only, submitter should see their own evidence
        let alice_access = check_evidence_access("evidence-001", "did:mycelix:alice");
        assert!(
            alice_access.can_view,
            "Submitter should always be able to view own evidence"
        );
    }
}

#[cfg(test)]
mod appeal_access_tests {
    use super::*;

    // ==========================================
    // Appeal Access Tests
    // ==========================================

    #[test]
    fn test_losing_party_can_file_appeal() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        render_decision("case-001", "decision-001", DecisionOutcome::ForComplainant);

        // Bob lost, so Bob should be able to appeal
        let bob_access = check_appeal_access("decision-001", "did:mycelix:bob");
        assert!(
            bob_access.can_file_appeal,
            "Losing party should be able to file appeal"
        );
    }

    #[test]
    fn test_winning_party_cannot_file_appeal() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        render_decision("case-001", "decision-001", DecisionOutcome::ForComplainant);

        // Alice won, so Alice should not be able to appeal
        let alice_access = check_appeal_access("decision-001", "did:mycelix:alice");
        assert!(
            !alice_access.can_file_appeal,
            "Winning party should not be able to file appeal"
        );
    }

    #[test]
    fn test_either_party_can_appeal_split_decision() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        render_decision("case-001", "decision-001", DecisionOutcome::SplitDecision);

        let alice_access = check_appeal_access("decision-001", "did:mycelix:alice");
        let bob_access = check_appeal_access("decision-001", "did:mycelix:bob");

        assert!(
            alice_access.can_file_appeal,
            "Either party can appeal split decision"
        );
        assert!(
            bob_access.can_file_appeal,
            "Either party can appeal split decision"
        );
    }

    #[test]
    fn test_third_party_cannot_file_appeal() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        render_decision("case-001", "decision-001", DecisionOutcome::ForComplainant);

        let charlie_access = check_appeal_access("decision-001", "did:mycelix:charlie");
        assert!(
            !charlie_access.can_file_appeal,
            "Third party should not be able to file appeal"
        );
    }

    #[test]
    fn test_cannot_appeal_after_finalization() {
        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);
        render_decision("case-001", "decision-001", DecisionOutcome::ForComplainant);
        finalize_decision("decision-001");

        let bob_access = check_appeal_access("decision-001", "did:mycelix:bob");
        assert!(
            !bob_access.can_file_appeal,
            "Cannot appeal finalized decision"
        );
    }
}

#[cfg(test)]
mod enforcement_access_tests {
    use super::*;

    // ==========================================
    // Enforcement Access Tests
    // ==========================================

    #[test]
    fn test_system_enforcer_can_execute() {
        create_enforcement("enforcement-001", "decision-001");
        set_enforcer("enforcement-001", "did:mycelix:system_enforcer");

        let access = check_enforcement_access("enforcement-001", "did:mycelix:system_enforcer");
        assert!(
            access.can_execute,
            "Assigned enforcer should be able to execute"
        );
        assert!(
            access.can_update_status,
            "Assigned enforcer should be able to update status"
        );
    }

    #[test]
    fn test_parties_cannot_execute_enforcement() {
        create_enforcement("enforcement-001", "decision-001");
        set_enforcement_case("enforcement-001", "case-001");

        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let alice_access = check_enforcement_access("enforcement-001", "did:mycelix:alice");
        let bob_access = check_enforcement_access("enforcement-001", "did:mycelix:bob");

        assert!(
            !alice_access.can_execute,
            "Party should not be able to execute enforcement"
        );
        assert!(
            !bob_access.can_execute,
            "Party should not be able to execute enforcement"
        );
    }

    #[test]
    fn test_parties_can_view_enforcement_status() {
        create_enforcement("enforcement-001", "decision-001");
        set_enforcement_case("enforcement-001", "case-001");

        let case = create_test_case("case-001", "did:mycelix:alice", "did:mycelix:bob");
        create_case(&case);

        let alice_access = check_enforcement_access("enforcement-001", "did:mycelix:alice");
        let bob_access = check_enforcement_access("enforcement-001", "did:mycelix:bob");

        assert!(
            alice_access.can_view,
            "Party should be able to view enforcement status"
        );
        assert!(
            bob_access.can_view,
            "Party should be able to view enforcement status"
        );
    }

    #[test]
    fn test_target_can_contest_enforcement() {
        create_enforcement("enforcement-001", "decision-001");
        set_enforcement_target("enforcement-001", "did:mycelix:bob");

        let bob_access = check_enforcement_access("enforcement-001", "did:mycelix:bob");
        assert!(
            bob_access.can_contest,
            "Enforcement target should be able to contest"
        );
    }

    #[test]
    fn test_non_target_cannot_contest_enforcement() {
        create_enforcement("enforcement-001", "decision-001");
        set_enforcement_target("enforcement-001", "did:mycelix:bob");

        let alice_access = check_enforcement_access("enforcement-001", "did:mycelix:alice");
        assert!(
            !alice_access.can_contest,
            "Non-target should not be able to contest enforcement"
        );
    }
}

#[cfg(test)]
mod restorative_circle_access_tests {
    use super::*;

    // ==========================================
    // Restorative Circle Access Tests
    // ==========================================

    #[test]
    fn test_facilitator_has_full_control() {
        create_circle("circle-001", "case-001", "did:mycelix:facilitator");

        let access = check_circle_access("circle-001", "did:mycelix:facilitator");
        assert!(access.can_view, "Facilitator should be able to view circle");
        assert!(
            access.can_record_session,
            "Facilitator should be able to record sessions"
        );
        assert!(
            access.can_add_agreement,
            "Facilitator should be able to add agreements"
        );
        assert!(
            access.can_complete,
            "Facilitator should be able to complete circle"
        );
    }

    #[test]
    fn test_participants_have_limited_access() {
        create_circle("circle-001", "case-001", "did:mycelix:facilitator");
        add_circle_participant("circle-001", "did:mycelix:alice", CircleRole::HarmReceiver);

        let access = check_circle_access("circle-001", "did:mycelix:alice");
        assert!(access.can_view, "Participant should be able to view circle");
        assert!(
            !access.can_record_session,
            "Participant should not record sessions"
        );
        assert!(
            !access.can_complete,
            "Participant should not complete circle"
        );
    }

    #[test]
    fn test_non_participant_cannot_access() {
        create_circle("circle-001", "case-001", "did:mycelix:facilitator");

        let access = check_circle_access("circle-001", "did:mycelix:charlie");
        assert!(!access.can_view, "Non-participant should not view circle");
    }

    #[test]
    fn test_consent_required_for_participation() {
        create_circle("circle-001", "case-001", "did:mycelix:facilitator");
        add_circle_participant("circle-001", "did:mycelix:alice", CircleRole::HarmReceiver);

        // Before consent
        let before_access = check_circle_participation("circle-001", "did:mycelix:alice");
        assert!(
            !before_access.can_fully_participate,
            "Should not fully participate before consent"
        );

        // After consent
        record_consent("circle-001", "did:mycelix:alice");
        let after_access = check_circle_participation("circle-001", "did:mycelix:alice");
        assert!(
            after_access.can_fully_participate,
            "Should fully participate after consent"
        );
    }
}

// ==========================================
// Test Helper Types and Functions
// ==========================================

#[derive(Clone, Debug)]
struct CaseAccess {
    can_view: bool,
    can_submit_evidence: bool,
    can_update: bool,
    can_withdraw: bool,
    can_view_all_evidence: bool,
    can_render_decision: bool,
}

#[derive(Clone, Debug)]
struct EvidenceAccess {
    can_view: bool,
}

#[derive(Clone, Debug)]
struct AppealAccess {
    can_file_appeal: bool,
}

#[derive(Clone, Debug)]
struct EnforcementAccess {
    can_view: bool,
    can_execute: bool,
    can_update_status: bool,
    can_contest: bool,
}

#[derive(Clone, Debug)]
struct CircleAccess {
    can_view: bool,
    can_record_session: bool,
    can_add_agreement: bool,
    can_complete: bool,
}

#[derive(Clone, Debug)]
struct CircleParticipationAccess {
    can_fully_participate: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum PartyRole {
    Complainant,
    Respondent,
    Witness,
    Expert,
}

#[derive(Clone, Debug, PartialEq)]
enum EvidenceVisibility {
    AllParties,
    AdjudicatorsOnly,
    Restricted(Vec<String>),
    Sealed,
}

#[derive(Clone, Debug, PartialEq)]
enum DecisionOutcome {
    ForComplainant,
    ForRespondent,
    SplitDecision,
}

#[derive(Clone, Debug, PartialEq)]
enum CircleRole {
    Facilitator,
    HarmReceiver,
    HarmDoer,
}

#[derive(Clone, Debug)]
struct TestCase {
    id: String,
    complainant: String,
    respondent: String,
    parties: Vec<(String, PartyRole)>,
    arbitrators: Vec<String>,
    recused_arbitrators: Vec<String>,
}

#[derive(Clone, Debug)]
struct TestEvidence {
    id: String,
    case_id: String,
    visibility: EvidenceVisibility,
    submitter: String,
}

#[derive(Clone, Debug)]
struct TestDecision {
    id: String,
    case_id: String,
    outcome: DecisionOutcome,
    finalized: bool,
}

#[derive(Clone, Debug)]
struct TestEnforcement {
    id: String,
    decision_id: String,
    case_id: Option<String>,
    enforcer: Option<String>,
    target: Option<String>,
}

#[derive(Clone, Debug)]
struct TestCircle {
    id: String,
    case_id: String,
    facilitator: String,
    participants: Vec<(String, CircleRole, bool)>, // (did, role, consented)
}

// In-memory stores
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CASES: Mutex<HashMap<String, TestCase>> = Mutex::new(HashMap::new());
    static ref EVIDENCE: Mutex<HashMap<String, TestEvidence>> = Mutex::new(HashMap::new());
    static ref DECISIONS: Mutex<HashMap<String, TestDecision>> = Mutex::new(HashMap::new());
    static ref ENFORCEMENTS: Mutex<HashMap<String, TestEnforcement>> = Mutex::new(HashMap::new());
    static ref CIRCLES: Mutex<HashMap<String, TestCircle>> = Mutex::new(HashMap::new());
}

fn create_test_case(id: &str, complainant: &str, respondent: &str) -> TestCase {
    TestCase {
        id: id.to_string(),
        complainant: complainant.to_string(),
        respondent: respondent.to_string(),
        parties: vec![],
        arbitrators: vec![],
        recused_arbitrators: vec![],
    }
}

fn create_case(case: &TestCase) {
    CASES.lock().unwrap().insert(case.id.clone(), case.clone());
}

fn add_party_to_case(case_id: &str, party_did: &str, role: PartyRole) {
    let mut store = CASES.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        case.parties.push((party_did.to_string(), role));
    }
}

fn assign_arbitrator(case_id: &str, arb_did: &str) {
    let mut store = CASES.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        case.arbitrators.push(arb_did.to_string());
    }
}

fn recuse_arbitrator(case_id: &str, arb_did: &str) {
    let mut store = CASES.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        case.arbitrators.retain(|a| a != arb_did);
        case.recused_arbitrators.push(arb_did.to_string());
    }
}

fn check_case_access(case_id: &str, accessor: &str) -> CaseAccess {
    let store = CASES.lock().unwrap();
    if let Some(case) = store.get(case_id) {
        let is_complainant = case.complainant == accessor;
        let is_respondent = case.respondent == accessor;
        let is_party = case.parties.iter().any(|(did, _)| did == accessor);
        let is_arbitrator = case.arbitrators.contains(&accessor.to_string());
        let is_recused = case.recused_arbitrators.contains(&accessor.to_string());

        CaseAccess {
            can_view: is_complainant || is_respondent || is_party || (is_arbitrator && !is_recused),
            can_submit_evidence: is_complainant || is_respondent || is_party,
            can_update: is_complainant,
            can_withdraw: is_complainant,
            can_view_all_evidence: is_arbitrator && !is_recused,
            can_render_decision: is_arbitrator && !is_recused,
        }
    } else {
        CaseAccess {
            can_view: false,
            can_submit_evidence: false,
            can_update: false,
            can_withdraw: false,
            can_view_all_evidence: false,
            can_render_decision: false,
        }
    }
}

fn create_evidence(id: &str, case_id: &str, visibility: EvidenceVisibility) {
    EVIDENCE.lock().unwrap().insert(
        id.to_string(),
        TestEvidence {
            id: id.to_string(),
            case_id: case_id.to_string(),
            visibility,
            submitter: "did:mycelix:unknown".to_string(),
        },
    );
}

fn create_evidence_with_restriction(id: &str, case_id: &str, allowed: Vec<String>) {
    EVIDENCE.lock().unwrap().insert(
        id.to_string(),
        TestEvidence {
            id: id.to_string(),
            case_id: case_id.to_string(),
            visibility: EvidenceVisibility::Restricted(allowed),
            submitter: "did:mycelix:unknown".to_string(),
        },
    );
}

fn create_evidence_by_submitter(
    id: &str,
    case_id: &str,
    submitter: &str,
    visibility: EvidenceVisibility,
) {
    EVIDENCE.lock().unwrap().insert(
        id.to_string(),
        TestEvidence {
            id: id.to_string(),
            case_id: case_id.to_string(),
            visibility,
            submitter: submitter.to_string(),
        },
    );
}

fn check_evidence_access(evidence_id: &str, accessor: &str) -> EvidenceAccess {
    let evidence_store = EVIDENCE.lock().unwrap();
    let case_store = CASES.lock().unwrap();

    if let Some(evidence) = evidence_store.get(evidence_id) {
        // Submitter can always view
        if evidence.submitter == accessor {
            return EvidenceAccess { can_view: true };
        }

        if let Some(case) = case_store.get(&evidence.case_id) {
            let is_arbitrator = case.arbitrators.contains(&accessor.to_string());

            match &evidence.visibility {
                EvidenceVisibility::AllParties => {
                    let is_party = case.complainant == accessor
                        || case.respondent == accessor
                        || case.parties.iter().any(|(did, _)| did == accessor)
                        || is_arbitrator;
                    EvidenceAccess { can_view: is_party }
                }
                EvidenceVisibility::AdjudicatorsOnly => EvidenceAccess {
                    can_view: is_arbitrator,
                },
                EvidenceVisibility::Restricted(allowed) => EvidenceAccess {
                    can_view: allowed.contains(&accessor.to_string()),
                },
                EvidenceVisibility::Sealed => EvidenceAccess { can_view: false },
            }
        } else {
            EvidenceAccess { can_view: false }
        }
    } else {
        EvidenceAccess { can_view: false }
    }
}

fn render_decision(case_id: &str, decision_id: &str, outcome: DecisionOutcome) {
    DECISIONS.lock().unwrap().insert(
        decision_id.to_string(),
        TestDecision {
            id: decision_id.to_string(),
            case_id: case_id.to_string(),
            outcome,
            finalized: false,
        },
    );
}

fn finalize_decision(decision_id: &str) {
    let mut store = DECISIONS.lock().unwrap();
    if let Some(decision) = store.get_mut(decision_id) {
        decision.finalized = true;
    }
}

fn check_appeal_access(decision_id: &str, accessor: &str) -> AppealAccess {
    let decision_store = DECISIONS.lock().unwrap();
    let case_store = CASES.lock().unwrap();

    if let Some(decision) = decision_store.get(decision_id) {
        if decision.finalized {
            return AppealAccess {
                can_file_appeal: false,
            };
        }

        if let Some(case) = case_store.get(&decision.case_id) {
            let is_complainant = case.complainant == accessor;
            let is_respondent = case.respondent == accessor;

            let can_appeal = match decision.outcome {
                DecisionOutcome::ForComplainant => is_respondent,
                DecisionOutcome::ForRespondent => is_complainant,
                DecisionOutcome::SplitDecision => is_complainant || is_respondent,
            };

            AppealAccess {
                can_file_appeal: can_appeal,
            }
        } else {
            AppealAccess {
                can_file_appeal: false,
            }
        }
    } else {
        AppealAccess {
            can_file_appeal: false,
        }
    }
}

fn create_enforcement(id: &str, decision_id: &str) {
    ENFORCEMENTS.lock().unwrap().insert(
        id.to_string(),
        TestEnforcement {
            id: id.to_string(),
            decision_id: decision_id.to_string(),
            case_id: None,
            enforcer: None,
            target: None,
        },
    );
}

fn set_enforcer(enforcement_id: &str, enforcer: &str) {
    let mut store = ENFORCEMENTS.lock().unwrap();
    if let Some(enforcement) = store.get_mut(enforcement_id) {
        enforcement.enforcer = Some(enforcer.to_string());
    }
}

fn set_enforcement_case(enforcement_id: &str, case_id: &str) {
    let mut store = ENFORCEMENTS.lock().unwrap();
    if let Some(enforcement) = store.get_mut(enforcement_id) {
        enforcement.case_id = Some(case_id.to_string());
    }
}

fn set_enforcement_target(enforcement_id: &str, target: &str) {
    let mut store = ENFORCEMENTS.lock().unwrap();
    if let Some(enforcement) = store.get_mut(enforcement_id) {
        enforcement.target = Some(target.to_string());
    }
}

fn check_enforcement_access(enforcement_id: &str, accessor: &str) -> EnforcementAccess {
    let enforcement_store = ENFORCEMENTS.lock().unwrap();
    let case_store = CASES.lock().unwrap();

    if let Some(enforcement) = enforcement_store.get(enforcement_id) {
        let is_enforcer = enforcement.enforcer.as_ref() == Some(&accessor.to_string());
        let is_target = enforcement.target.as_ref() == Some(&accessor.to_string());

        let is_party = if let Some(case_id) = &enforcement.case_id {
            if let Some(case) = case_store.get(case_id) {
                case.complainant == accessor || case.respondent == accessor
            } else {
                false
            }
        } else {
            false
        };

        EnforcementAccess {
            can_view: is_party || is_enforcer || is_target,
            can_execute: is_enforcer,
            can_update_status: is_enforcer,
            can_contest: is_target,
        }
    } else {
        EnforcementAccess {
            can_view: false,
            can_execute: false,
            can_update_status: false,
            can_contest: false,
        }
    }
}

fn create_circle(id: &str, case_id: &str, facilitator: &str) {
    CIRCLES.lock().unwrap().insert(
        id.to_string(),
        TestCircle {
            id: id.to_string(),
            case_id: case_id.to_string(),
            facilitator: facilitator.to_string(),
            participants: vec![],
        },
    );
}

fn add_circle_participant(circle_id: &str, did: &str, role: CircleRole) {
    let mut store = CIRCLES.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        circle.participants.push((did.to_string(), role, false));
    }
}

fn record_consent(circle_id: &str, did: &str) {
    let mut store = CIRCLES.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        for (p_did, _, consented) in &mut circle.participants {
            if p_did == did {
                *consented = true;
            }
        }
    }
}

fn check_circle_access(circle_id: &str, accessor: &str) -> CircleAccess {
    let store = CIRCLES.lock().unwrap();
    if let Some(circle) = store.get(circle_id) {
        let is_facilitator = circle.facilitator == accessor;
        let is_participant = circle
            .participants
            .iter()
            .any(|(did, _, _)| did == accessor);

        CircleAccess {
            can_view: is_facilitator || is_participant,
            can_record_session: is_facilitator,
            can_add_agreement: is_facilitator,
            can_complete: is_facilitator,
        }
    } else {
        CircleAccess {
            can_view: false,
            can_record_session: false,
            can_add_agreement: false,
            can_complete: false,
        }
    }
}

fn check_circle_participation(circle_id: &str, accessor: &str) -> CircleParticipationAccess {
    let store = CIRCLES.lock().unwrap();
    if let Some(circle) = store.get(circle_id) {
        if let Some((_, _, consented)) = circle
            .participants
            .iter()
            .find(|(did, _, _)| did == accessor)
        {
            CircleParticipationAccess {
                can_fully_participate: *consented,
            }
        } else {
            CircleParticipationAccess {
                can_fully_participate: false,
            }
        }
    } else {
        CircleParticipationAccess {
            can_fully_participate: false,
        }
    }
}

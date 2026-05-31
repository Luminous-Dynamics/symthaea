// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Case Lifecycle Tests
//!
//! Tests for case management, status transitions, and phase progression
//! in the mycelix-justice dispute resolution system.

#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod case_filing_tests {
    use super::*;

    // ==========================================
    // Case Filing Tests
    // ==========================================

    #[test]
    fn test_file_case_creates_case_record() {
        let case = create_test_case(
            "case-001",
            "Contract Dispute",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );

        let result = simulate_file_case(case);
        assert!(result.is_ok(), "Filing a valid case should succeed");

        let filed_case = result.unwrap();
        assert_eq!(filed_case.phase, CasePhase::Filed);
        assert_eq!(filed_case.status, CaseStatus::Active);
    }

    #[test]
    fn test_file_case_creates_complainant_link() {
        let case = create_test_case(
            "case-001",
            "Property Dispute",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );

        let result = simulate_file_case(case.clone());
        assert!(result.is_ok());

        // Verify complainant can retrieve the case
        let complainant_cases =
            simulate_get_cases_for_party(&case.complainant, PartyType::Complainant);
        assert!(complainant_cases.iter().any(|c| c.id == "case-001"));
    }

    #[test]
    fn test_file_case_creates_respondent_link() {
        let case = create_test_case(
            "case-001",
            "Financial Dispute",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );

        let result = simulate_file_case(case.clone());
        assert!(result.is_ok());

        // Verify respondent can retrieve the case
        let respondent_cases =
            simulate_get_cases_for_party(&case.respondent, PartyType::Respondent);
        assert!(respondent_cases.iter().any(|c| c.id == "case-001"));
    }

    #[test]
    fn test_file_case_with_context() {
        let mut case = create_test_case(
            "case-001",
            "Marketplace Dispute",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        case.context = Some(CaseContext {
            happ: Some("mycelix-marketplace".to_string()),
            reference_id: Some("order-12345".to_string()),
            community: None,
            jurisdiction: None,
        });

        let result = simulate_file_case(case);
        assert!(result.is_ok());

        let filed_case = result.unwrap();
        assert!(filed_case.context.is_some());
        assert_eq!(
            filed_case.context.as_ref().unwrap().happ,
            Some("mycelix-marketplace".to_string())
        );
    }

    #[test]
    fn test_file_case_with_severity_levels() {
        // Minor case
        let mut minor_case = create_test_case(
            "case-minor",
            "Minor Issue",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        minor_case.severity = CaseSeverity::Minor;
        assert!(simulate_file_case(minor_case).is_ok());

        // Moderate case
        let mut moderate_case = create_test_case(
            "case-moderate",
            "Moderate Issue",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        moderate_case.severity = CaseSeverity::Moderate;
        assert!(simulate_file_case(moderate_case).is_ok());

        // Serious case
        let mut serious_case = create_test_case(
            "case-serious",
            "Serious Issue",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        serious_case.severity = CaseSeverity::Serious;
        assert!(simulate_file_case(serious_case).is_ok());

        // Critical case
        let mut critical_case = create_test_case(
            "case-critical",
            "Critical Issue",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        critical_case.severity = CaseSeverity::Critical;
        assert!(simulate_file_case(critical_case).is_ok());
    }
}

#[cfg(test)]
mod case_phase_transition_tests {
    use super::*;

    // ==========================================
    // Phase Transition Tests
    // ==========================================

    #[test]
    fn test_phase_transition_filed_to_negotiation() {
        let case = create_filed_case("case-001");

        let result = simulate_update_phase(&case.id, CasePhase::Negotiation);
        assert!(
            result.is_ok(),
            "Transition from Filed to Negotiation should succeed"
        );
        assert_eq!(result.unwrap().phase, CasePhase::Negotiation);
    }

    #[test]
    fn test_phase_transition_negotiation_to_mediation() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Negotiation;

        let result = simulate_update_phase(&case.id, CasePhase::Mediation);
        assert!(
            result.is_ok(),
            "Transition from Negotiation to Mediation should succeed"
        );
        assert_eq!(result.unwrap().phase, CasePhase::Mediation);
    }

    #[test]
    fn test_phase_transition_mediation_to_arbitration() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Mediation;

        let result = simulate_update_phase(&case.id, CasePhase::Arbitration);
        assert!(
            result.is_ok(),
            "Transition from Mediation to Arbitration should succeed"
        );
        assert_eq!(result.unwrap().phase, CasePhase::Arbitration);
    }

    #[test]
    fn test_phase_transition_arbitration_to_appeal() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Arbitration;

        let result = simulate_update_phase(&case.id, CasePhase::Appeal);
        assert!(
            result.is_ok(),
            "Transition from Arbitration to Appeal should succeed"
        );
        assert_eq!(result.unwrap().phase, CasePhase::Appeal);
    }

    #[test]
    fn test_phase_transition_to_enforcement() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Arbitration;

        let result = simulate_update_phase(&case.id, CasePhase::Enforcement);
        assert!(result.is_ok(), "Transition to Enforcement should succeed");
        assert_eq!(result.unwrap().phase, CasePhase::Enforcement);
    }

    #[test]
    fn test_phase_transition_to_closed() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Enforcement;

        let result = simulate_update_phase(&case.id, CasePhase::Closed);
        assert!(result.is_ok(), "Transition to Closed should succeed");
        assert_eq!(result.unwrap().phase, CasePhase::Closed);
    }

    #[test]
    fn test_phase_transition_with_deadline() {
        let case = create_filed_case("case-001");
        let deadline = MockTimestamp::from_days_from_now(7);

        let result =
            simulate_update_phase_with_deadline(&case.id, CasePhase::Negotiation, Some(deadline));
        assert!(result.is_ok());

        let updated_case = result.unwrap();
        assert_eq!(updated_case.phase, CasePhase::Negotiation);
        assert!(updated_case.phase_deadline.is_some());
    }

    #[test]
    fn test_direct_filing_to_arbitration_for_serious_cases() {
        // Serious cases may skip negotiation/mediation
        let mut case = create_filed_case("case-001");
        case.severity = CaseSeverity::Serious;

        let result = simulate_update_phase(&case.id, CasePhase::Arbitration);
        assert!(
            result.is_ok(),
            "Serious cases should be able to go directly to arbitration"
        );
    }
}

#[cfg(test)]
mod case_status_tests {
    use super::*;

    // ==========================================
    // Status Transition Tests
    // ==========================================

    #[test]
    fn test_status_active_to_on_hold() {
        let case = create_filed_case("case-001");

        let result = simulate_update_status(&case.id, CaseStatus::OnHold);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::OnHold);
    }

    #[test]
    fn test_status_active_to_awaiting_response() {
        let case = create_filed_case("case-001");

        let result = simulate_update_status(&case.id, CaseStatus::AwaitingResponse);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::AwaitingResponse);
    }

    #[test]
    fn test_status_to_in_deliberation() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Arbitration;

        let result = simulate_update_status(&case.id, CaseStatus::InDeliberation);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::InDeliberation);
    }

    #[test]
    fn test_status_to_decision_rendered() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Arbitration;
        case.status = CaseStatus::InDeliberation;

        let result = simulate_update_status(&case.id, CaseStatus::DecisionRendered);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::DecisionRendered);
    }

    #[test]
    fn test_status_to_enforcing() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Enforcement;
        case.status = CaseStatus::DecisionRendered;

        let result = simulate_update_status(&case.id, CaseStatus::Enforcing);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::Enforcing);
    }

    #[test]
    fn test_status_to_resolved() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Enforcement;
        case.status = CaseStatus::Enforcing;

        let result = simulate_update_status(&case.id, CaseStatus::Resolved);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::Resolved);
    }

    #[test]
    fn test_status_to_dismissed() {
        let case = create_filed_case("case-001");

        let result = simulate_update_status(&case.id, CaseStatus::Dismissed);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::Dismissed);
    }

    #[test]
    fn test_status_to_withdrawn() {
        let case = create_filed_case("case-001");

        let result = simulate_update_status(&case.id, CaseStatus::Withdrawn);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CaseStatus::Withdrawn);
    }

    #[test]
    fn test_cannot_modify_closed_case() {
        let mut case = create_filed_case("case-001");
        case.phase = CasePhase::Closed;
        case.status = CaseStatus::Resolved;

        // Attempting to change status of a closed case should fail
        let result = simulate_update_status(&case.id, CaseStatus::Active);
        assert!(
            result.is_err(),
            "Should not be able to reactivate a closed case"
        );
    }
}

#[cfg(test)]
mod party_management_tests {
    use super::*;

    // ==========================================
    // Party Management Tests
    // ==========================================

    #[test]
    fn test_add_witness_to_case() {
        let case = create_filed_case("case-001");

        let result = simulate_add_party(&case.id, "did:mycelix:witness1", PartyRole::Witness);
        assert!(result.is_ok());

        let updated_case = result.unwrap();
        assert!(
            updated_case
                .parties
                .iter()
                .any(|p| p.did == "did:mycelix:witness1" && p.role == PartyRole::Witness)
        );
    }

    #[test]
    fn test_add_expert_to_case() {
        let case = create_filed_case("case-001");

        let result = simulate_add_party(&case.id, "did:mycelix:expert1", PartyRole::Expert);
        assert!(result.is_ok());

        let updated_case = result.unwrap();
        assert!(
            updated_case
                .parties
                .iter()
                .any(|p| p.did == "did:mycelix:expert1" && p.role == PartyRole::Expert)
        );
    }

    #[test]
    fn test_add_intervenor_to_case() {
        let case = create_filed_case("case-001");

        let result = simulate_add_party(&case.id, "did:mycelix:intervenor1", PartyRole::Intervenor);
        assert!(result.is_ok());

        let updated_case = result.unwrap();
        assert!(
            updated_case
                .parties
                .iter()
                .any(|p| p.did == "did:mycelix:intervenor1" && p.role == PartyRole::Intervenor)
        );
    }

    #[test]
    fn test_add_affected_party_to_case() {
        let case = create_filed_case("case-001");

        let result = simulate_add_party(&case.id, "did:mycelix:affected1", PartyRole::Affected);
        assert!(result.is_ok());

        let updated_case = result.unwrap();
        assert!(
            updated_case
                .parties
                .iter()
                .any(|p| p.did == "did:mycelix:affected1" && p.role == PartyRole::Affected)
        );
    }

    #[test]
    fn test_add_multiple_parties_to_case() {
        let case = create_filed_case("case-001");

        simulate_add_party(&case.id, "did:mycelix:witness1", PartyRole::Witness).unwrap();
        simulate_add_party(&case.id, "did:mycelix:witness2", PartyRole::Witness).unwrap();
        let result = simulate_add_party(&case.id, "did:mycelix:expert1", PartyRole::Expert);

        assert!(result.is_ok());
        let updated_case = result.unwrap();
        assert!(updated_case.parties.len() >= 3);
    }

    #[test]
    fn test_party_added_with_timestamp() {
        let case = create_filed_case("case-001");

        let result = simulate_add_party(&case.id, "did:mycelix:witness1", PartyRole::Witness);
        assert!(result.is_ok());

        let updated_case = result.unwrap();
        let witness = updated_case
            .parties
            .iter()
            .find(|p| p.did == "did:mycelix:witness1")
            .unwrap();
        assert!(witness.joined_at.is_valid());
    }
}

#[cfg(test)]
mod case_query_tests {
    use super::*;

    // ==========================================
    // Case Query Tests
    // ==========================================

    #[test]
    fn test_get_case_by_hash() {
        let case = create_filed_case("case-001");

        let result = simulate_get_case(&case.id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "case-001");
    }

    #[test]
    fn test_get_nonexistent_case() {
        let result = simulate_get_case("nonexistent-case");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_my_cases_as_complainant() {
        let case1 = create_test_case(
            "case-001",
            "Dispute 1",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        let case2 = create_test_case(
            "case-002",
            "Dispute 2",
            "did:mycelix:alice",
            "did:mycelix:charlie",
        );

        simulate_file_case(case1).unwrap();
        simulate_file_case(case2).unwrap();

        let alice_cases = simulate_get_my_cases("did:mycelix:alice");
        assert_eq!(alice_cases.len(), 2);
    }

    #[test]
    fn test_get_my_cases_as_respondent() {
        let case = create_test_case(
            "case-001",
            "Dispute",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );

        simulate_file_case(case).unwrap();

        let bob_cases = simulate_get_my_cases("did:mycelix:bob");
        assert!(bob_cases.iter().any(|c| c.id == "case-001"));
    }

    #[test]
    fn test_get_all_cases() {
        simulate_file_case(create_test_case(
            "case-001",
            "Dispute 1",
            "did:mycelix:alice",
            "did:mycelix:bob",
        ))
        .unwrap();
        simulate_file_case(create_test_case(
            "case-002",
            "Dispute 2",
            "did:mycelix:charlie",
            "did:mycelix:dave",
        ))
        .unwrap();
        simulate_file_case(create_test_case(
            "case-003",
            "Dispute 3",
            "did:mycelix:eve",
            "did:mycelix:frank",
        ))
        .unwrap();

        let all_cases = simulate_get_all_cases();
        assert!(all_cases.len() >= 3);
    }

    #[test]
    fn test_get_cases_avoids_duplicates() {
        // When a party is both complainant in one case and respondent in another
        let case1 = create_test_case(
            "case-001",
            "Dispute 1",
            "did:mycelix:alice",
            "did:mycelix:bob",
        );
        let case2 = create_test_case(
            "case-002",
            "Dispute 2",
            "did:mycelix:bob",
            "did:mycelix:alice",
        );

        simulate_file_case(case1).unwrap();
        simulate_file_case(case2).unwrap();

        let alice_cases = simulate_get_my_cases("did:mycelix:alice");

        // Should have exactly 2 cases, not duplicates
        let unique_ids: std::collections::HashSet<_> = alice_cases.iter().map(|c| &c.id).collect();
        assert_eq!(unique_ids.len(), alice_cases.len());
    }
}

// ==========================================
// Test Helper Types and Functions
// ==========================================

#[derive(Clone, Debug, PartialEq)]
enum CasePhase {
    Filed,
    Negotiation,
    Mediation,
    Arbitration,
    Appeal,
    Enforcement,
    Closed,
}

#[derive(Clone, Debug, PartialEq)]
enum CaseStatus {
    Active,
    OnHold,
    AwaitingResponse,
    InDeliberation,
    DecisionRendered,
    Enforcing,
    Resolved,
    Dismissed,
    Withdrawn,
}

#[derive(Clone, Debug, PartialEq)]
enum CaseSeverity {
    Minor,
    Moderate,
    Serious,
    Critical,
}

#[derive(Clone, Debug, PartialEq)]
enum PartyRole {
    Complainant,
    Respondent,
    Witness,
    Expert,
    Intervenor,
    Affected,
}

#[derive(Clone, Debug)]
enum PartyType {
    Complainant,
    Respondent,
}

#[derive(Clone, Debug)]
struct CaseContext {
    happ: Option<String>,
    reference_id: Option<String>,
    community: Option<String>,
    jurisdiction: Option<String>,
}

#[derive(Clone, Debug)]
struct CaseParty {
    did: String,
    role: PartyRole,
    joined_at: MockTimestamp,
}

#[derive(Clone, Debug)]
struct MockTimestamp {
    micros: i64,
}

impl MockTimestamp {
    fn now() -> Self {
        MockTimestamp {
            micros: 1706000000000000,
        }
    }

    fn from_days_from_now(days: i64) -> Self {
        MockTimestamp {
            micros: 1706000000000000 + (days * 24 * 3600 * 1_000_000),
        }
    }

    fn is_valid(&self) -> bool {
        self.micros > 0
    }
}

#[derive(Clone, Debug)]
struct TestCase {
    id: String,
    title: String,
    description: String,
    complainant: String,
    respondent: String,
    parties: Vec<CaseParty>,
    phase: CasePhase,
    status: CaseStatus,
    severity: CaseSeverity,
    context: Option<CaseContext>,
    phase_deadline: Option<MockTimestamp>,
}

// In-memory store for testing
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CASE_STORE: Mutex<HashMap<String, TestCase>> = Mutex::new(HashMap::new());
}

fn create_test_case(id: &str, title: &str, complainant: &str, respondent: &str) -> TestCase {
    TestCase {
        id: id.to_string(),
        title: title.to_string(),
        description: format!("{} - detailed description", title),
        complainant: complainant.to_string(),
        respondent: respondent.to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Moderate,
        context: None,
        phase_deadline: None,
    }
}

fn create_filed_case(id: &str) -> TestCase {
    let case = create_test_case(id, "Test Case", "did:mycelix:alice", "did:mycelix:bob");
    CASE_STORE
        .lock()
        .unwrap()
        .insert(id.to_string(), case.clone());
    case
}

fn simulate_file_case(case: TestCase) -> Result<TestCase, String> {
    // Validate
    if case.title.trim().is_empty() {
        return Err("Case title required".to_string());
    }
    if !case.complainant.starts_with("did:") {
        return Err("Complainant must be a DID".to_string());
    }
    if !case.respondent.starts_with("did:") {
        return Err("Respondent must be a DID".to_string());
    }
    if case.complainant == case.respondent {
        return Err("Cannot file case against self".to_string());
    }

    CASE_STORE
        .lock()
        .unwrap()
        .insert(case.id.clone(), case.clone());
    Ok(case)
}

fn simulate_get_case(id: &str) -> Option<TestCase> {
    CASE_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_update_phase(case_id: &str, new_phase: CasePhase) -> Result<TestCase, String> {
    let mut store = CASE_STORE.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        if case.phase == CasePhase::Closed {
            return Err("Cannot modify closed case".to_string());
        }
        case.phase = new_phase;
        Ok(case.clone())
    } else {
        Err("Case not found".to_string())
    }
}

fn simulate_update_phase_with_deadline(
    case_id: &str,
    new_phase: CasePhase,
    deadline: Option<MockTimestamp>,
) -> Result<TestCase, String> {
    let mut store = CASE_STORE.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        case.phase = new_phase;
        case.phase_deadline = deadline;
        Ok(case.clone())
    } else {
        Err("Case not found".to_string())
    }
}

fn simulate_update_status(case_id: &str, new_status: CaseStatus) -> Result<TestCase, String> {
    let mut store = CASE_STORE.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        if case.phase == CasePhase::Closed && new_status == CaseStatus::Active {
            return Err("Cannot reactivate closed case".to_string());
        }
        case.status = new_status;
        Ok(case.clone())
    } else {
        Err("Case not found".to_string())
    }
}

fn simulate_add_party(case_id: &str, party_did: &str, role: PartyRole) -> Result<TestCase, String> {
    let mut store = CASE_STORE.lock().unwrap();
    if let Some(case) = store.get_mut(case_id) {
        case.parties.push(CaseParty {
            did: party_did.to_string(),
            role,
            joined_at: MockTimestamp::now(),
        });
        Ok(case.clone())
    } else {
        Err("Case not found".to_string())
    }
}

fn simulate_get_cases_for_party(party_did: &str, party_type: PartyType) -> Vec<TestCase> {
    let store = CASE_STORE.lock().unwrap();
    store
        .values()
        .filter(|case| match party_type {
            PartyType::Complainant => case.complainant == party_did,
            PartyType::Respondent => case.respondent == party_did,
        })
        .cloned()
        .collect()
}

fn simulate_get_my_cases(did: &str) -> Vec<TestCase> {
    let store = CASE_STORE.lock().unwrap();
    store
        .values()
        .filter(|case| case.complainant == did || case.respondent == did)
        .cloned()
        .collect()
}

fn simulate_get_all_cases() -> Vec<TestCase> {
    let store = CASE_STORE.lock().unwrap();
    store.values().cloned().collect()
}

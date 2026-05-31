// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Restorative Justice Tests
//!
//! Tests for restorative justice circles, community healing processes,
//! and alternative dispute resolution in the mycelix-justice system.

#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod circle_creation_tests {
    use super::*;

    // ==========================================
    // Restorative Circle Creation Tests
    // ==========================================

    #[test]
    fn test_create_restorative_circle() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            vec![
                create_participant("did:mycelix:alice", CircleRole::HarmReceiver),
                create_participant("did:mycelix:bob", CircleRole::HarmDoer),
            ],
        );

        let result = simulate_create_circle(circle);
        assert!(result.is_ok(), "Creating valid circle should succeed");
    }

    #[test]
    fn test_circle_creates_case_link() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();

        let case_circle = simulate_get_case_circle("case-001");
        assert!(case_circle.is_some());
        assert_eq!(case_circle.unwrap().id, "circle-001");
    }

    #[test]
    fn test_circle_creates_facilitator_link() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:jane_facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();

        let facilitator_circles = simulate_get_facilitator_circles("did:mycelix:jane_facilitator");
        assert!(facilitator_circles.iter().any(|c| c.id == "circle-001"));
    }

    #[test]
    fn test_circle_initial_status() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );

        let result = simulate_create_circle(circle).unwrap();
        assert_eq!(result.status, CircleStatus::Forming);
    }

    #[test]
    fn test_circle_with_all_roles() {
        let participants = vec![
            create_participant("did:mycelix:alice", CircleRole::HarmReceiver),
            create_participant("did:mycelix:bob", CircleRole::HarmDoer),
            create_participant("did:mycelix:community1", CircleRole::CommunityMember),
            create_participant("did:mycelix:community2", CircleRole::CommunityMember),
            create_participant("did:mycelix:support_alice", CircleRole::SupportPerson),
            create_participant("did:mycelix:support_bob", CircleRole::SupportPerson),
            create_participant("did:mycelix:elder1", CircleRole::Elder),
        ];

        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            participants,
        );
        let result = simulate_create_circle(circle);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().participants.len(), 7);
    }

    #[test]
    fn test_circle_requires_participants() {
        let circle =
            create_test_circle("circle-001", "case-001", "did:mycelix:facilitator", vec![]);

        let result = simulate_create_circle(circle);
        assert!(
            result.is_err(),
            "Circle with no participants should be rejected"
        );
    }
}

#[cfg(test)]
mod consent_tests {
    use super::*;

    // ==========================================
    // Participant Consent Tests
    // ==========================================

    #[test]
    fn test_record_participant_consent() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();

        let result = simulate_record_consent("circle-001", "did:mycelix:alice");
        assert!(result.is_ok());

        let updated = result.unwrap();
        let alice = updated
            .participants
            .iter()
            .find(|p| p.did == "did:mycelix:alice")
            .unwrap();
        assert!(alice.consented);
    }

    #[test]
    fn test_all_consent_moves_to_active() {
        let participants = vec![
            create_participant("did:mycelix:alice", CircleRole::HarmReceiver),
            create_participant("did:mycelix:bob", CircleRole::HarmDoer),
        ];
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            participants,
        );
        simulate_create_circle(circle).unwrap();

        simulate_record_consent("circle-001", "did:mycelix:alice").unwrap();
        let result = simulate_record_consent("circle-001", "did:mycelix:bob");

        assert!(result.is_ok());
        let updated = result.unwrap();
        assert_eq!(updated.status, CircleStatus::Active);
    }

    #[test]
    fn test_partial_consent_stays_forming() {
        let participants = vec![
            create_participant("did:mycelix:alice", CircleRole::HarmReceiver),
            create_participant("did:mycelix:bob", CircleRole::HarmDoer),
            create_participant("did:mycelix:elder", CircleRole::Elder),
        ];
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            participants,
        );
        simulate_create_circle(circle).unwrap();

        simulate_record_consent("circle-001", "did:mycelix:alice").unwrap();
        let result = simulate_record_consent("circle-001", "did:mycelix:bob");

        assert!(result.is_ok());
        let updated = result.unwrap();
        assert_eq!(updated.status, CircleStatus::Forming); // Still waiting for elder's consent
    }

    #[test]
    fn test_consent_is_idempotent() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();

        simulate_record_consent("circle-001", "did:mycelix:alice").unwrap();
        let result = simulate_record_consent("circle-001", "did:mycelix:alice"); // Consent again

        assert!(result.is_ok()); // Should succeed without error
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;

    // ==========================================
    // Circle Session Tests
    // ==========================================

    #[test]
    fn test_record_session() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        let session = CircleSession {
            session_number: 1,
            summary: "Initial circle gathering. All parties shared their perspectives.".to_string(),
            next_steps: vec![
                "Review community impact".to_string(),
                "Prepare apology statement".to_string(),
            ],
        };

        let result = simulate_record_session(
            "circle-001",
            session,
            vec![
                "did:mycelix:alice".to_string(),
                "did:mycelix:bob".to_string(),
            ],
        );
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.sessions.len(), 1);
    }

    #[test]
    fn test_record_multiple_sessions() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        simulate_record_session(
            "circle-001",
            CircleSession {
                session_number: 1,
                summary: "Session 1".to_string(),
                next_steps: vec![],
            },
            vec!["did:mycelix:alice".to_string()],
        )
        .unwrap();

        simulate_record_session(
            "circle-001",
            CircleSession {
                session_number: 2,
                summary: "Session 2".to_string(),
                next_steps: vec![],
            },
            vec![
                "did:mycelix:alice".to_string(),
                "did:mycelix:bob".to_string(),
            ],
        )
        .unwrap();

        let updated = simulate_get_circle("circle-001").unwrap();
        assert_eq!(updated.sessions.len(), 2);
    }

    #[test]
    fn test_session_tracks_attendance() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        simulate_record_session(
            "circle-001",
            CircleSession {
                session_number: 1,
                summary: "First session".to_string(),
                next_steps: vec![],
            },
            vec!["did:mycelix:alice".to_string()],
        )
        .unwrap(); // Only Alice attended

        let updated = simulate_get_circle("circle-001").unwrap();
        let alice = updated
            .participants
            .iter()
            .find(|p| p.did == "did:mycelix:alice")
            .unwrap();
        let bob = updated
            .participants
            .iter()
            .find(|p| p.did == "did:mycelix:bob")
            .unwrap();

        assert!(alice.attended_sessions.contains(&1));
        assert!(!bob.attended_sessions.contains(&1));
    }

    #[test]
    fn test_session_in_forming_status_rejected() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        // Don't make active - still in Forming

        let session = CircleSession {
            session_number: 1,
            summary: "Attempted session".to_string(),
            next_steps: vec![],
        };

        let result =
            simulate_record_session("circle-001", session, vec!["did:mycelix:alice".to_string()]);
        assert!(
            result.is_err(),
            "Should not be able to record session while circle is forming"
        );
    }
}

#[cfg(test)]
mod agreement_tests {
    use super::*;

    // ==========================================
    // Circle Agreement Tests
    // ==========================================

    #[test]
    fn test_add_agreement() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        let result = simulate_add_agreement(
            "circle-001",
            "Bob will make a public apology to Alice within 7 days",
        );
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert!(
            updated
                .agreements
                .contains(&"Bob will make a public apology to Alice within 7 days".to_string())
        );
    }

    #[test]
    fn test_add_multiple_agreements() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        simulate_add_agreement("circle-001", "Agreement 1: Public apology").unwrap();
        simulate_add_agreement("circle-001", "Agreement 2: Restitution of 500 HBAR").unwrap();
        simulate_add_agreement("circle-001", "Agreement 3: Community service - 20 hours").unwrap();

        let updated = simulate_get_circle("circle-001").unwrap();
        assert_eq!(updated.agreements.len(), 3);
    }

    #[test]
    fn test_agreement_moves_to_agreement_reached() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        simulate_add_agreement("circle-001", "Parties have agreed to settlement terms").unwrap();

        // Manually update to AgreementReached status after agreements
        let result = simulate_update_circle_status("circle-001", CircleStatus::AgreementReached);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CircleStatus::AgreementReached);
    }
}

#[cfg(test)]
mod status_transition_tests {
    use super::*;

    // ==========================================
    // Circle Status Transition Tests
    // ==========================================

    #[test]
    fn test_status_forming_to_active() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();

        let result = simulate_update_circle_status("circle-001", CircleStatus::Active);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CircleStatus::Active);
    }

    #[test]
    fn test_status_active_to_agreement_reached() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        let result = simulate_update_circle_status("circle-001", CircleStatus::AgreementReached);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CircleStatus::AgreementReached);
    }

    #[test]
    fn test_status_agreement_reached_to_monitoring() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        simulate_update_circle_status("circle-001", CircleStatus::Active).unwrap();
        simulate_update_circle_status("circle-001", CircleStatus::AgreementReached).unwrap();

        let result = simulate_update_circle_status("circle-001", CircleStatus::Monitoring);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CircleStatus::Monitoring);
    }

    #[test]
    fn test_status_monitoring_to_completed() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        simulate_update_circle_status("circle-001", CircleStatus::Active).unwrap();
        simulate_update_circle_status("circle-001", CircleStatus::AgreementReached).unwrap();
        simulate_update_circle_status("circle-001", CircleStatus::Monitoring).unwrap();

        let result = simulate_update_circle_status("circle-001", CircleStatus::Completed);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CircleStatus::Completed);
    }

    #[test]
    fn test_status_to_discontinued() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        // Circle can be discontinued at any active stage
        let result = simulate_update_circle_status("circle-001", CircleStatus::Discontinued);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, CircleStatus::Discontinued);
    }

    #[test]
    fn test_get_circles_by_status() {
        simulate_create_circle(create_test_circle(
            "c-001",
            "case-001",
            "did:mycelix:f",
            create_default_participants(),
        ))
        .unwrap();
        simulate_create_circle(create_test_circle(
            "c-002",
            "case-002",
            "did:mycelix:f",
            create_default_participants(),
        ))
        .unwrap();
        simulate_create_circle(create_test_circle(
            "c-003",
            "case-003",
            "did:mycelix:f",
            create_default_participants(),
        ))
        .unwrap();

        make_circle_active("c-002");
        simulate_update_circle_status("c-003", CircleStatus::Completed).unwrap();

        let forming = simulate_get_circles_by_status(CircleStatus::Forming);
        let active = simulate_get_circles_by_status(CircleStatus::Active);
        let completed = simulate_get_circles_by_status(CircleStatus::Completed);

        assert!(forming.iter().any(|c| c.id == "c-001"));
        assert!(active.iter().any(|c| c.id == "c-002"));
        assert!(completed.iter().any(|c| c.id == "c-003"));
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    // ==========================================
    // Circle Completion Tests
    // ==========================================

    #[test]
    fn test_complete_circle() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");
        simulate_add_agreement("circle-001", "Initial agreement").unwrap();

        let result = simulate_complete_circle(
            "circle-001",
            vec!["Final agreement: All terms fulfilled".to_string()],
        );
        assert!(result.is_ok());

        let completed = result.unwrap();
        assert_eq!(completed.status, CircleStatus::Completed);
        assert!(
            completed
                .agreements
                .contains(&"Final agreement: All terms fulfilled".to_string())
        );
    }

    #[test]
    fn test_complete_circle_adds_final_agreements() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        simulate_add_agreement("circle-001", "Existing agreement").unwrap();

        let result = simulate_complete_circle(
            "circle-001",
            vec!["Final term 1".to_string(), "Final term 2".to_string()],
        );

        assert!(result.is_ok());
        let completed = result.unwrap();
        assert_eq!(completed.agreements.len(), 3);
    }

    #[test]
    fn test_complete_circle_deduplicates_agreements() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");

        simulate_add_agreement("circle-001", "Shared agreement").unwrap();

        let result = simulate_complete_circle(
            "circle-001",
            vec![
                "Shared agreement".to_string(), // Duplicate
                "New agreement".to_string(),
            ],
        );

        assert!(result.is_ok());
        let completed = result.unwrap();
        // Should have 2 agreements, not 3 (deduplicated)
        assert_eq!(completed.agreements.len(), 2);
    }

    #[test]
    fn test_completed_circle_cannot_add_sessions() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();
        make_circle_active("circle-001");
        simulate_complete_circle("circle-001", vec![]).unwrap();

        let session = CircleSession {
            session_number: 1,
            summary: "Late session".to_string(),
            next_steps: vec![],
        };

        let result =
            simulate_record_session("circle-001", session, vec!["did:mycelix:alice".to_string()]);
        assert!(
            result.is_err(),
            "Should not be able to add sessions to completed circle"
        );
    }
}

#[cfg(test)]
mod query_tests {
    use super::*;

    // ==========================================
    // Circle Query Tests
    // ==========================================

    #[test]
    fn test_get_circle_by_id() {
        let circle = create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:facilitator",
            create_default_participants(),
        );
        simulate_create_circle(circle).unwrap();

        let result = simulate_get_circle("circle-001");
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "circle-001");
    }

    #[test]
    fn test_get_nonexistent_circle() {
        let result = simulate_get_circle("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_facilitator_circles() {
        simulate_create_circle(create_test_circle(
            "c-001",
            "case-001",
            "did:mycelix:jane",
            create_default_participants(),
        ))
        .unwrap();
        simulate_create_circle(create_test_circle(
            "c-002",
            "case-002",
            "did:mycelix:jane",
            create_default_participants(),
        ))
        .unwrap();
        simulate_create_circle(create_test_circle(
            "c-003",
            "case-003",
            "did:mycelix:john",
            create_default_participants(),
        ))
        .unwrap();

        let jane_circles = simulate_get_facilitator_circles("did:mycelix:jane");
        let john_circles = simulate_get_facilitator_circles("did:mycelix:john");

        assert_eq!(jane_circles.len(), 2);
        assert_eq!(john_circles.len(), 1);
    }

    #[test]
    fn test_get_case_circle() {
        simulate_create_circle(create_test_circle(
            "circle-001",
            "case-001",
            "did:mycelix:f",
            create_default_participants(),
        ))
        .unwrap();

        let result = simulate_get_case_circle("case-001");
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "circle-001");
    }

    #[test]
    fn test_get_case_circle_nonexistent() {
        let result = simulate_get_case_circle("nonexistent-case");
        assert!(result.is_none());
    }
}

// ==========================================
// Test Helper Types and Functions
// ==========================================

#[derive(Clone, Debug, PartialEq)]
enum CircleStatus {
    Forming,
    Active,
    AgreementReached,
    Monitoring,
    Completed,
    Discontinued,
}

#[derive(Clone, Debug, PartialEq)]
enum CircleRole {
    Facilitator,
    HarmDoer,
    HarmReceiver,
    CommunityMember,
    SupportPerson,
    Elder,
}

#[derive(Clone, Debug)]
struct CircleParticipant {
    did: String,
    role: CircleRole,
    consented: bool,
    attended_sessions: Vec<u32>,
}

#[derive(Clone, Debug)]
struct CircleSession {
    session_number: u32,
    summary: String,
    next_steps: Vec<String>,
}

#[derive(Clone, Debug)]
struct TestRestorativeCircle {
    id: String,
    case_id: String,
    facilitator: String,
    participants: Vec<CircleParticipant>,
    status: CircleStatus,
    sessions: Vec<CircleSession>,
    agreements: Vec<String>,
}

// In-memory store
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CIRCLE_STORE: Mutex<HashMap<String, TestRestorativeCircle>> = Mutex::new(HashMap::new());
}

fn create_participant(did: &str, role: CircleRole) -> CircleParticipant {
    CircleParticipant {
        did: did.to_string(),
        role,
        consented: false,
        attended_sessions: vec![],
    }
}

fn create_default_participants() -> Vec<CircleParticipant> {
    vec![
        create_participant("did:mycelix:alice", CircleRole::HarmReceiver),
        create_participant("did:mycelix:bob", CircleRole::HarmDoer),
    ]
}

fn create_test_circle(
    id: &str,
    case_id: &str,
    facilitator: &str,
    participants: Vec<CircleParticipant>,
) -> TestRestorativeCircle {
    TestRestorativeCircle {
        id: id.to_string(),
        case_id: case_id.to_string(),
        facilitator: facilitator.to_string(),
        participants,
        status: CircleStatus::Forming,
        sessions: vec![],
        agreements: vec![],
    }
}

fn simulate_create_circle(circle: TestRestorativeCircle) -> Result<TestRestorativeCircle, String> {
    if !circle.facilitator.starts_with("did:") {
        return Err("Facilitator must be a DID".to_string());
    }
    if circle.participants.is_empty() {
        return Err("Restorative circle must have participants".to_string());
    }
    CIRCLE_STORE
        .lock()
        .unwrap()
        .insert(circle.id.clone(), circle.clone());
    Ok(circle)
}

fn simulate_get_circle(id: &str) -> Option<TestRestorativeCircle> {
    CIRCLE_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_get_case_circle(case_id: &str) -> Option<TestRestorativeCircle> {
    CIRCLE_STORE
        .lock()
        .unwrap()
        .values()
        .find(|c| c.case_id == case_id)
        .cloned()
}

fn simulate_get_facilitator_circles(facilitator_did: &str) -> Vec<TestRestorativeCircle> {
    CIRCLE_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|c| c.facilitator == facilitator_did)
        .cloned()
        .collect()
}

fn simulate_record_consent(
    circle_id: &str,
    participant_did: &str,
) -> Result<TestRestorativeCircle, String> {
    let mut store = CIRCLE_STORE.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        for p in &mut circle.participants {
            if p.did == participant_did {
                p.consented = true;
            }
        }
        // Check if all consented
        if circle.participants.iter().all(|p| p.consented) && circle.status == CircleStatus::Forming
        {
            circle.status = CircleStatus::Active;
        }
        Ok(circle.clone())
    } else {
        Err("Circle not found".to_string())
    }
}

fn simulate_record_session(
    circle_id: &str,
    session: CircleSession,
    attendees: Vec<String>,
) -> Result<TestRestorativeCircle, String> {
    let mut store = CIRCLE_STORE.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        if circle.status == CircleStatus::Forming {
            return Err("Cannot record session while circle is forming".to_string());
        }
        if circle.status == CircleStatus::Completed || circle.status == CircleStatus::Discontinued {
            return Err("Cannot record session on completed/discontinued circle".to_string());
        }

        let session_num = session.session_number;
        circle.sessions.push(session);

        // Update attendance
        for did in attendees {
            for p in &mut circle.participants {
                if p.did == did {
                    p.attended_sessions.push(session_num);
                }
            }
        }

        Ok(circle.clone())
    } else {
        Err("Circle not found".to_string())
    }
}

fn simulate_add_agreement(
    circle_id: &str,
    agreement: &str,
) -> Result<TestRestorativeCircle, String> {
    let mut store = CIRCLE_STORE.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        circle.agreements.push(agreement.to_string());
        Ok(circle.clone())
    } else {
        Err("Circle not found".to_string())
    }
}

fn simulate_update_circle_status(
    circle_id: &str,
    new_status: CircleStatus,
) -> Result<TestRestorativeCircle, String> {
    let mut store = CIRCLE_STORE.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        circle.status = new_status;
        Ok(circle.clone())
    } else {
        Err("Circle not found".to_string())
    }
}

fn simulate_complete_circle(
    circle_id: &str,
    final_agreements: Vec<String>,
) -> Result<TestRestorativeCircle, String> {
    let mut store = CIRCLE_STORE.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        circle.status = CircleStatus::Completed;
        for agreement in final_agreements {
            if !circle.agreements.contains(&agreement) {
                circle.agreements.push(agreement);
            }
        }
        Ok(circle.clone())
    } else {
        Err("Circle not found".to_string())
    }
}

fn simulate_get_circles_by_status(status: CircleStatus) -> Vec<TestRestorativeCircle> {
    CIRCLE_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|c| c.status == status)
        .cloned()
        .collect()
}

fn make_circle_active(circle_id: &str) {
    let mut store = CIRCLE_STORE.lock().unwrap();
    if let Some(circle) = store.get_mut(circle_id) {
        circle.status = CircleStatus::Active;
    }
}

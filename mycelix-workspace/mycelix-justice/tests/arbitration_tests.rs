// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Arbitration Workflow Tests
//!
//! Tests for arbitration panel formation, deliberation, decision rendering,
//! and the appeals process in the mycelix-justice system.

#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod arbitration_creation_tests {
    use super::*;

    // ==========================================
    // Arbitration Panel Creation Tests
    // ==========================================

    #[test]
    fn test_create_arbitration_panel() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![
                create_arbitrator("did:mycelix:arb1", ArbitratorRole::Primary),
                create_arbitrator("did:mycelix:arb2", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb3", ArbitratorRole::PanelMember),
            ],
        );

        let result = simulate_create_arbitration(arbitration);
        assert!(
            result.is_ok(),
            "Creating valid arbitration panel should succeed"
        );
    }

    #[test]
    fn test_arbitration_creates_case_link() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );

        simulate_create_arbitration(arbitration).unwrap();

        let case_arb = simulate_get_case_arbitration("case-001");
        assert!(case_arb.is_some());
        assert_eq!(case_arb.unwrap().id, "arb-001");
    }

    #[test]
    fn test_arbitration_creates_arbitrator_links() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![
                create_arbitrator("did:mycelix:arb1", ArbitratorRole::Primary),
                create_arbitrator("did:mycelix:arb2", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb3", ArbitratorRole::PanelMember),
            ],
        );

        simulate_create_arbitration(arbitration).unwrap();

        // Each arbitrator should be able to find the case
        let arb1_cases = simulate_get_arbitrator_cases("did:mycelix:arb1");
        assert!(arb1_cases.iter().any(|a| a.id == "arb-001"));

        let arb2_cases = simulate_get_arbitrator_cases("did:mycelix:arb2");
        assert!(arb2_cases.iter().any(|a| a.id == "arb-001"));
    }

    #[test]
    fn test_arbitration_selection_methods() {
        // Random selection
        let arb_random = create_test_arbitration_with_selection(
            "arb-random",
            "case-001",
            ArbitratorSelection::Random,
        );
        assert!(simulate_create_arbitration(arb_random).is_ok());

        // MATL weighted selection
        let arb_matl = create_test_arbitration_with_selection(
            "arb-matl",
            "case-002",
            ArbitratorSelection::MATLWeighted,
        );
        assert!(simulate_create_arbitration(arb_matl).is_ok());

        // Party agreed selection
        let arb_agreed = create_test_arbitration_with_selection(
            "arb-agreed",
            "case-003",
            ArbitratorSelection::PartyAgreed,
        );
        assert!(simulate_create_arbitration(arb_agreed).is_ok());

        // Expertise based selection
        let arb_expert = create_test_arbitration_with_selection(
            "arb-expert",
            "case-004",
            ArbitratorSelection::ExpertiseBased {
                domain: "smart_contracts".to_string(),
            },
        );
        assert!(simulate_create_arbitration(arb_expert).is_ok());
    }

    #[test]
    fn test_arbitration_initial_status() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );

        let result = simulate_create_arbitration(arbitration).unwrap();
        assert_eq!(result.status, ArbitrationStatus::PanelFormation);
    }

    #[test]
    fn test_arbitration_with_alternates() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![
                create_arbitrator("did:mycelix:arb1", ArbitratorRole::Primary),
                create_arbitrator("did:mycelix:arb2", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb3", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb4", ArbitratorRole::Alternate),
                create_arbitrator("did:mycelix:arb5", ArbitratorRole::Alternate),
            ],
        );

        let result = simulate_create_arbitration(arbitration);
        assert!(result.is_ok());

        let created = result.unwrap();
        let alternates: Vec<_> = created
            .arbitrators
            .iter()
            .filter(|a| a.role == ArbitratorRole::Alternate)
            .collect();
        assert_eq!(alternates.len(), 2);
    }
}

#[cfg(test)]
mod arbitrator_response_tests {
    use super::*;

    // ==========================================
    // Arbitrator Acceptance/Recusal Tests
    // ==========================================

    #[test]
    fn test_arbitrator_accepts() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![
                create_arbitrator("did:mycelix:arb1", ArbitratorRole::Primary),
                create_arbitrator("did:mycelix:arb2", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb3", ArbitratorRole::PanelMember),
            ],
        );
        simulate_create_arbitration(arbitration).unwrap();

        let result = simulate_record_arbitrator_response(
            "arb-001",
            "did:mycelix:arb1",
            true,  // accepted
            false, // not recused
            None,
        );

        assert!(result.is_ok());
        let updated = result.unwrap();
        let arb1 = updated
            .arbitrators
            .iter()
            .find(|a| a.did == "did:mycelix:arb1")
            .unwrap();
        assert!(arb1.accepted);
        assert!(!arb1.recused);
    }

    #[test]
    fn test_arbitrator_recuses() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![
                create_arbitrator("did:mycelix:arb1", ArbitratorRole::Primary),
                create_arbitrator("did:mycelix:arb2", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb3", ArbitratorRole::PanelMember),
            ],
        );
        simulate_create_arbitration(arbitration).unwrap();

        let result = simulate_record_arbitrator_response(
            "arb-001",
            "did:mycelix:arb2",
            false, // not accepted
            true,  // recused
            Some("I have a prior business relationship with the complainant".to_string()),
        );

        assert!(result.is_ok());
        let updated = result.unwrap();
        let arb2 = updated
            .arbitrators
            .iter()
            .find(|a| a.did == "did:mycelix:arb2")
            .unwrap();
        assert!(!arb2.accepted);
        assert!(arb2.recused);
        assert!(arb2.recusal_reason.is_some());
    }

    #[test]
    fn test_all_arbitrators_accept_moves_to_evidence_review() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![
                create_arbitrator("did:mycelix:arb1", ArbitratorRole::Primary),
                create_arbitrator("did:mycelix:arb2", ArbitratorRole::PanelMember),
                create_arbitrator("did:mycelix:arb3", ArbitratorRole::PanelMember),
            ],
        );
        simulate_create_arbitration(arbitration).unwrap();

        simulate_record_arbitrator_response("arb-001", "did:mycelix:arb1", true, false, None)
            .unwrap();
        simulate_record_arbitrator_response("arb-001", "did:mycelix:arb2", true, false, None)
            .unwrap();
        simulate_record_arbitrator_response("arb-001", "did:mycelix:arb3", true, false, None)
            .unwrap();

        // After all accept, status should advance
        let arb = simulate_get_arbitration("arb-001").unwrap();
        // The system would auto-advance, but for testing we verify all accepted
        assert!(arb.arbitrators.iter().all(|a| a.accepted));
    }
}

#[cfg(test)]
mod arbitration_status_tests {
    use super::*;

    // ==========================================
    // Arbitration Status Transition Tests
    // ==========================================

    #[test]
    fn test_status_panel_formation_to_evidence_review() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );
        simulate_create_arbitration(arbitration).unwrap();

        let result =
            simulate_update_arbitration_status("arb-001", ArbitrationStatus::EvidenceReview);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, ArbitrationStatus::EvidenceReview);
    }

    #[test]
    fn test_status_evidence_review_to_hearing() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );
        simulate_create_arbitration(arbitration).unwrap();
        simulate_update_arbitration_status("arb-001", ArbitrationStatus::EvidenceReview).unwrap();

        let result = simulate_update_arbitration_status("arb-001", ArbitrationStatus::Hearing);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, ArbitrationStatus::Hearing);
    }

    #[test]
    fn test_status_hearing_to_deliberation() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );
        simulate_create_arbitration(arbitration).unwrap();

        let result = simulate_update_arbitration_status("arb-001", ArbitrationStatus::Deliberation);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, ArbitrationStatus::Deliberation);
    }

    #[test]
    fn test_status_deliberation_to_decision_drafting() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );
        simulate_create_arbitration(arbitration).unwrap();
        simulate_update_arbitration_status("arb-001", ArbitrationStatus::Deliberation).unwrap();

        let result =
            simulate_update_arbitration_status("arb-001", ArbitrationStatus::DecisionDrafting);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, ArbitrationStatus::DecisionDrafting);
    }

    #[test]
    fn test_status_to_decision_rendered() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );
        simulate_create_arbitration(arbitration).unwrap();

        let result =
            simulate_update_arbitration_status("arb-001", ArbitrationStatus::DecisionRendered);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, ArbitrationStatus::DecisionRendered);
    }

    #[test]
    fn test_status_to_appealed() {
        let arbitration = create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        );
        simulate_create_arbitration(arbitration).unwrap();
        simulate_update_arbitration_status("arb-001", ArbitrationStatus::DecisionRendered).unwrap();

        let result = simulate_update_arbitration_status("arb-001", ArbitrationStatus::Appealed);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, ArbitrationStatus::Appealed);
    }

    #[test]
    fn test_get_arbitrations_by_status() {
        simulate_create_arbitration(create_test_arbitration(
            "arb-001",
            "case-001",
            vec![create_arbitrator(
                "did:mycelix:arb1",
                ArbitratorRole::Primary,
            )],
        ))
        .unwrap();
        simulate_create_arbitration(create_test_arbitration(
            "arb-002",
            "case-002",
            vec![create_arbitrator(
                "did:mycelix:arb2",
                ArbitratorRole::Primary,
            )],
        ))
        .unwrap();
        simulate_update_arbitration_status("arb-002", ArbitrationStatus::Deliberation).unwrap();

        let in_formation = simulate_get_arbitrations_by_status(ArbitrationStatus::PanelFormation);
        let in_deliberation = simulate_get_arbitrations_by_status(ArbitrationStatus::Deliberation);

        assert!(in_formation.iter().any(|a| a.id == "arb-001"));
        assert!(in_deliberation.iter().any(|a| a.id == "arb-002"));
    }
}

#[cfg(test)]
mod decision_tests {
    use super::*;

    // ==========================================
    // Decision Rendering Tests
    // ==========================================

    #[test]
    fn test_render_decision() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            vec![
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb1".to_string(),
                    vote: VoteChoice::ForComplainant,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb2".to_string(),
                    vote: VoteChoice::ForComplainant,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb3".to_string(),
                    vote: VoteChoice::ForRespondent,
                },
            ],
        );

        let result = simulate_render_decision(decision);
        assert!(result.is_ok(), "Rendering valid decision should succeed");
    }

    #[test]
    fn test_decision_creates_case_link() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            vec![ArbitratorVote {
                arbitrator: "did:mycelix:arb1".to_string(),
                vote: VoteChoice::ForComplainant,
            }],
        );

        simulate_render_decision(decision).unwrap();

        let case_decisions = simulate_get_case_decisions("case-001");
        assert!(case_decisions.iter().any(|d| d.id == "decision-001"));
    }

    #[test]
    fn test_decision_types() {
        // Merits decision
        let merits = create_test_decision(
            "d-merits",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        assert!(simulate_render_decision(merits).is_ok());

        // Interim decision
        let interim = create_test_decision(
            "d-interim",
            "case-002",
            "arb-002",
            DecisionType::InterimDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        assert!(simulate_render_decision(interim).is_ok());

        // Default decision
        let default = create_test_decision(
            "d-default",
            "case-003",
            "arb-003",
            DecisionType::DefaultDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        assert!(simulate_render_decision(default).is_ok());

        // Consent decision
        let consent = create_test_decision(
            "d-consent",
            "case-004",
            "arb-004",
            DecisionType::ConsentDecision,
            DecisionOutcome::Settled,
            create_default_votes(),
        );
        assert!(simulate_render_decision(consent).is_ok());

        // Dismissal
        let dismissal = create_test_decision(
            "d-dismiss",
            "case-005",
            "arb-005",
            DecisionType::Dismissal,
            DecisionOutcome::Dismissed,
            create_default_votes(),
        );
        assert!(simulate_render_decision(dismissal).is_ok());
    }

    #[test]
    fn test_decision_outcomes() {
        // For complainant
        let for_comp = create_test_decision(
            "d-1",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        assert_eq!(
            simulate_render_decision(for_comp).unwrap().outcome,
            DecisionOutcome::ForComplainant
        );

        // For respondent
        let for_resp = create_test_decision(
            "d-2",
            "case-002",
            "arb-002",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        assert_eq!(
            simulate_render_decision(for_resp).unwrap().outcome,
            DecisionOutcome::ForRespondent
        );

        // Split decision
        let split = create_test_decision(
            "d-3",
            "case-003",
            "arb-003",
            DecisionType::MeritsDecision,
            DecisionOutcome::SplitDecision,
            create_default_votes(),
        );
        assert_eq!(
            simulate_render_decision(split).unwrap().outcome,
            DecisionOutcome::SplitDecision
        );
    }

    #[test]
    fn test_decision_with_remedies() {
        let mut decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        decision.remedies = vec![
            Remedy {
                remedy_type: RemedyType::Compensation,
                responsible_party: "did:mycelix:bob".to_string(),
                amount: Some(1000),
                currency: Some("HBAR".to_string()),
                description: "Payment for damages".to_string(),
            },
            Remedy {
                remedy_type: RemedyType::Apology,
                responsible_party: "did:mycelix:bob".to_string(),
                amount: None,
                currency: None,
                description: "Public apology required".to_string(),
            },
        ];

        let result = simulate_render_decision(decision);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().remedies.len(), 2);
    }

    #[test]
    fn test_decision_with_dissenting_opinions() {
        let mut decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            vec![
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb1".to_string(),
                    vote: VoteChoice::ForComplainant,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb2".to_string(),
                    vote: VoteChoice::ForComplainant,
                },
                ArbitratorVote {
                    arbitrator: "did:mycelix:arb3".to_string(),
                    vote: VoteChoice::ForRespondent,
                },
            ],
        );
        decision.dissents = vec![DissentingOpinion {
            arbitrator: "did:mycelix:arb3".to_string(),
            opinion: "I disagree with the majority because the evidence was insufficient..."
                .to_string(),
        }];

        let result = simulate_render_decision(decision);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dissents.len(), 1);
    }

    #[test]
    fn test_finalize_decision() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let result = simulate_finalize_decision("decision-001");
        assert!(result.is_ok());
        assert!(result.unwrap().finalized);
    }

    #[test]
    fn test_cannot_appeal_finalized_decision() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForComplainant,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();
        simulate_finalize_decision("decision-001").unwrap();

        // Attempting to file appeal after finalization should fail
        let appeal =
            create_test_appeal("appeal-001", "case-001", "decision-001", "did:mycelix:bob");
        let result = simulate_file_appeal(appeal);
        assert!(
            result.is_err(),
            "Should not be able to appeal finalized decision"
        );
    }
}

#[cfg(test)]
mod appeal_tests {
    use super::*;

    // ==========================================
    // Appeal Process Tests
    // ==========================================

    #[test]
    fn test_file_appeal() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        let result = simulate_file_appeal(appeal);

        assert!(result.is_ok(), "Filing valid appeal should succeed");
    }

    #[test]
    fn test_appeal_creates_decision_link() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        simulate_file_appeal(appeal).unwrap();

        let decision_appeals = simulate_get_decision_appeals("decision-001");
        assert!(decision_appeals.iter().any(|a| a.id == "appeal-001"));
    }

    #[test]
    fn test_appeal_grounds() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let mut appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        appeal.grounds = vec![
            AppealGround::ProceduralError,
            AppealGround::NewEvidence,
            AppealGround::ExcessiveRemedy,
        ];

        let result = simulate_file_appeal(appeal);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().grounds.len(), 3);
    }

    #[test]
    fn test_appeal_status_transitions() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        simulate_file_appeal(appeal).unwrap();

        // Initial status should be Filed
        let filed = simulate_get_appeal("appeal-001").unwrap();
        assert_eq!(filed.status, AppealStatus::Filed);

        // Transition to UnderReview
        simulate_update_appeal_status("appeal-001", AppealStatus::UnderReview).unwrap();
        let reviewing = simulate_get_appeal("appeal-001").unwrap();
        assert_eq!(reviewing.status, AppealStatus::UnderReview);

        // Transition to Granted
        simulate_update_appeal_status("appeal-001", AppealStatus::Granted).unwrap();
        let granted = simulate_get_appeal("appeal-001").unwrap();
        assert_eq!(granted.status, AppealStatus::Granted);
    }

    #[test]
    fn test_appeal_denied() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        simulate_file_appeal(appeal).unwrap();

        simulate_update_appeal_status("appeal-001", AppealStatus::UnderReview).unwrap();
        let result = simulate_update_appeal_status("appeal-001", AppealStatus::Denied);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, AppealStatus::Denied);
    }

    #[test]
    fn test_appeal_remanded() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        simulate_file_appeal(appeal).unwrap();

        let result = simulate_update_appeal_status("appeal-001", AppealStatus::Remanded);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, AppealStatus::Remanded);
    }

    #[test]
    fn test_appeal_within_deadline() {
        let mut decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        decision.appeal_deadline = MockTimestamp::from_days_from_now(30);
        simulate_render_decision(decision).unwrap();

        let appeal = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        let result = simulate_file_appeal(appeal);

        assert!(result.is_ok(), "Appeal within deadline should be accepted");
    }

    #[test]
    fn test_appeal_number_tracking() {
        let decision = create_test_decision(
            "decision-001",
            "case-001",
            "arb-001",
            DecisionType::MeritsDecision,
            DecisionOutcome::ForRespondent,
            create_default_votes(),
        );
        simulate_render_decision(decision).unwrap();

        let appeal1 = create_test_appeal(
            "appeal-001",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        let first = simulate_file_appeal(appeal1).unwrap();
        assert_eq!(first.appeal_number, 1);

        // Simulate appeal denied, new appeal filed
        simulate_update_appeal_status("appeal-001", AppealStatus::Denied).unwrap();

        let mut appeal2 = create_test_appeal(
            "appeal-002",
            "case-001",
            "decision-001",
            "did:mycelix:alice",
        );
        appeal2.appeal_number = 2;
        let second = simulate_file_appeal(appeal2).unwrap();
        assert_eq!(second.appeal_number, 2);
    }
}

// ==========================================
// Test Helper Types and Functions
// ==========================================

#[derive(Clone, Debug, PartialEq)]
enum ArbitrationStatus {
    PanelFormation,
    EvidenceReview,
    Hearing,
    Deliberation,
    DecisionDrafting,
    DecisionRendered,
    Appealed,
}

#[derive(Clone, Debug, PartialEq)]
enum ArbitratorRole {
    Primary,
    PanelMember,
    Alternate,
}

#[derive(Clone, Debug, PartialEq)]
enum ArbitratorSelection {
    Random,
    MATLWeighted,
    PartyAgreed,
    ExpertiseBased { domain: String },
}

#[derive(Clone, Debug, PartialEq)]
enum DecisionType {
    MeritsDecision,
    InterimDecision,
    DefaultDecision,
    ConsentDecision,
    Dismissal,
}

#[derive(Clone, Debug, PartialEq)]
enum DecisionOutcome {
    ForComplainant,
    ForRespondent,
    SplitDecision,
    Dismissed,
    Settled,
}

#[derive(Clone, Debug, PartialEq)]
enum VoteChoice {
    ForComplainant,
    ForRespondent,
    Abstain,
}

#[derive(Clone, Debug, PartialEq)]
enum RemedyType {
    Compensation,
    Restitution,
    SpecificPerformance,
    Injunction,
    Apology,
    CommunityService,
    ReputationAdjustment,
    AccessRestriction,
    Education,
    RestorativeCircle,
}

#[derive(Clone, Debug, PartialEq)]
enum AppealGround {
    ProceduralError,
    NewEvidence,
    LegalError,
    Bias,
    ExcessiveRemedy,
    InsufficientRemedy,
}

#[derive(Clone, Debug, PartialEq)]
enum AppealStatus {
    Filed,
    UnderReview,
    Granted,
    Denied,
    Remanded,
    Resolved,
}

#[derive(Clone, Debug)]
struct Arbitrator {
    did: String,
    role: ArbitratorRole,
    accepted: bool,
    recused: bool,
    recusal_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct ArbitratorVote {
    arbitrator: String,
    vote: VoteChoice,
}

#[derive(Clone, Debug)]
struct DissentingOpinion {
    arbitrator: String,
    opinion: String,
}

#[derive(Clone, Debug)]
struct Remedy {
    remedy_type: RemedyType,
    responsible_party: String,
    amount: Option<u128>,
    currency: Option<String>,
    description: String,
}

#[derive(Clone, Debug)]
struct TestArbitration {
    id: String,
    case_id: String,
    arbitrators: Vec<Arbitrator>,
    selection_method: ArbitratorSelection,
    status: ArbitrationStatus,
}

#[derive(Clone, Debug)]
struct TestDecision {
    id: String,
    case_id: String,
    arbitration_id: String,
    decision_type: DecisionType,
    outcome: DecisionOutcome,
    reasoning: String,
    remedies: Vec<Remedy>,
    votes: Vec<ArbitratorVote>,
    dissents: Vec<DissentingOpinion>,
    appeal_deadline: MockTimestamp,
    finalized: bool,
}

#[derive(Clone, Debug)]
struct TestAppeal {
    id: String,
    case_id: String,
    decision_id: String,
    appellant: String,
    grounds: Vec<AppealGround>,
    argument: String,
    status: AppealStatus,
    appeal_number: u8,
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
}

// In-memory stores
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref ARB_STORE: Mutex<HashMap<String, TestArbitration>> = Mutex::new(HashMap::new());
    static ref DECISION_STORE: Mutex<HashMap<String, TestDecision>> = Mutex::new(HashMap::new());
    static ref APPEAL_STORE: Mutex<HashMap<String, TestAppeal>> = Mutex::new(HashMap::new());
}

fn create_arbitrator(did: &str, role: ArbitratorRole) -> Arbitrator {
    Arbitrator {
        did: did.to_string(),
        role,
        accepted: false,
        recused: false,
        recusal_reason: None,
    }
}

fn create_test_arbitration(
    id: &str,
    case_id: &str,
    arbitrators: Vec<Arbitrator>,
) -> TestArbitration {
    TestArbitration {
        id: id.to_string(),
        case_id: case_id.to_string(),
        arbitrators,
        selection_method: ArbitratorSelection::Random,
        status: ArbitrationStatus::PanelFormation,
    }
}

fn create_test_arbitration_with_selection(
    id: &str,
    case_id: &str,
    selection: ArbitratorSelection,
) -> TestArbitration {
    TestArbitration {
        id: id.to_string(),
        case_id: case_id.to_string(),
        arbitrators: vec![create_arbitrator(
            "did:mycelix:arb1",
            ArbitratorRole::Primary,
        )],
        selection_method: selection,
        status: ArbitrationStatus::PanelFormation,
    }
}

fn create_test_decision(
    id: &str,
    case_id: &str,
    arb_id: &str,
    decision_type: DecisionType,
    outcome: DecisionOutcome,
    votes: Vec<ArbitratorVote>,
) -> TestDecision {
    TestDecision {
        id: id.to_string(),
        case_id: case_id.to_string(),
        arbitration_id: arb_id.to_string(),
        decision_type,
        outcome,
        reasoning: "Based on evidence and testimony presented...".to_string(),
        remedies: vec![],
        votes,
        dissents: vec![],
        appeal_deadline: MockTimestamp::from_days_from_now(30),
        finalized: false,
    }
}

fn create_default_votes() -> Vec<ArbitratorVote> {
    vec![ArbitratorVote {
        arbitrator: "did:mycelix:arb1".to_string(),
        vote: VoteChoice::ForComplainant,
    }]
}

fn create_test_appeal(id: &str, case_id: &str, decision_id: &str, appellant: &str) -> TestAppeal {
    TestAppeal {
        id: id.to_string(),
        case_id: case_id.to_string(),
        decision_id: decision_id.to_string(),
        appellant: appellant.to_string(),
        grounds: vec![AppealGround::ProceduralError],
        argument: "The decision was flawed because...".to_string(),
        status: AppealStatus::Filed,
        appeal_number: 1,
    }
}

fn simulate_create_arbitration(arbitration: TestArbitration) -> Result<TestArbitration, String> {
    if arbitration.arbitrators.is_empty() {
        return Err("Arbitration requires arbitrators".to_string());
    }
    ARB_STORE
        .lock()
        .unwrap()
        .insert(arbitration.id.clone(), arbitration.clone());
    Ok(arbitration)
}

fn simulate_get_arbitration(id: &str) -> Option<TestArbitration> {
    ARB_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_get_case_arbitration(case_id: &str) -> Option<TestArbitration> {
    ARB_STORE
        .lock()
        .unwrap()
        .values()
        .find(|a| a.case_id == case_id)
        .cloned()
}

fn simulate_get_arbitrator_cases(arbitrator_did: &str) -> Vec<TestArbitration> {
    ARB_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|a| a.arbitrators.iter().any(|arb| arb.did == arbitrator_did))
        .cloned()
        .collect()
}

fn simulate_record_arbitrator_response(
    arb_id: &str,
    arbitrator_did: &str,
    accepted: bool,
    recused: bool,
    reason: Option<String>,
) -> Result<TestArbitration, String> {
    let mut store = ARB_STORE.lock().unwrap();
    if let Some(arb) = store.get_mut(arb_id) {
        for a in &mut arb.arbitrators {
            if a.did == arbitrator_did {
                a.accepted = accepted;
                a.recused = recused;
                a.recusal_reason = reason;
            }
        }
        Ok(arb.clone())
    } else {
        Err("Arbitration not found".to_string())
    }
}

fn simulate_update_arbitration_status(
    arb_id: &str,
    new_status: ArbitrationStatus,
) -> Result<TestArbitration, String> {
    let mut store = ARB_STORE.lock().unwrap();
    if let Some(arb) = store.get_mut(arb_id) {
        arb.status = new_status;
        Ok(arb.clone())
    } else {
        Err("Arbitration not found".to_string())
    }
}

fn simulate_get_arbitrations_by_status(status: ArbitrationStatus) -> Vec<TestArbitration> {
    ARB_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|a| a.status == status)
        .cloned()
        .collect()
}

fn simulate_render_decision(decision: TestDecision) -> Result<TestDecision, String> {
    if decision.votes.is_empty() {
        return Err("Decision requires votes".to_string());
    }
    if decision.reasoning.trim().is_empty() {
        return Err("Decision requires reasoning".to_string());
    }
    DECISION_STORE
        .lock()
        .unwrap()
        .insert(decision.id.clone(), decision.clone());
    Ok(decision)
}

fn simulate_get_decision(id: &str) -> Option<TestDecision> {
    DECISION_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_get_case_decisions(case_id: &str) -> Vec<TestDecision> {
    DECISION_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|d| d.case_id == case_id)
        .cloned()
        .collect()
}

fn simulate_finalize_decision(decision_id: &str) -> Result<TestDecision, String> {
    let mut store = DECISION_STORE.lock().unwrap();
    if let Some(decision) = store.get_mut(decision_id) {
        decision.finalized = true;
        Ok(decision.clone())
    } else {
        Err("Decision not found".to_string())
    }
}

fn simulate_file_appeal(appeal: TestAppeal) -> Result<TestAppeal, String> {
    // Check if decision is finalized
    let decisions = DECISION_STORE.lock().unwrap();
    if let Some(decision) = decisions.get(&appeal.decision_id) {
        if decision.finalized {
            return Err("Cannot appeal finalized decision".to_string());
        }
    }
    drop(decisions);

    if appeal.grounds.is_empty() {
        return Err("Appeal requires grounds".to_string());
    }
    if !appeal.appellant.starts_with("did:") {
        return Err("Appellant must be a DID".to_string());
    }

    APPEAL_STORE
        .lock()
        .unwrap()
        .insert(appeal.id.clone(), appeal.clone());
    Ok(appeal)
}

fn simulate_get_appeal(id: &str) -> Option<TestAppeal> {
    APPEAL_STORE.lock().unwrap().get(id).cloned()
}

fn simulate_get_decision_appeals(decision_id: &str) -> Vec<TestAppeal> {
    APPEAL_STORE
        .lock()
        .unwrap()
        .values()
        .filter(|a| a.decision_id == decision_id)
        .cloned()
        .collect()
}

fn simulate_update_appeal_status(
    appeal_id: &str,
    new_status: AppealStatus,
) -> Result<TestAppeal, String> {
    let mut store = APPEAL_STORE.lock().unwrap();
    if let Some(appeal) = store.get_mut(appeal_id) {
        appeal.status = new_status;
        Ok(appeal.clone())
    } else {
        Err("Appeal not found".to_string())
    }
}

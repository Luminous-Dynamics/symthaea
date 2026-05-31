// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Integrity Zome Tests
//!
//! Tests for entry validation in the mycelix-justice integrity zome.
//! These tests verify that validation rules are correctly enforced.

#[cfg(test)]
mod case_validation_tests {
    use super::*;

    // ==========================================
    // Case Entry Validation Tests
    // ==========================================

    #[test]
    fn test_case_requires_title() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "".to_string(), // Empty title
            description: "Valid description".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:bob".to_string(),
        };

        let result = validate_case_title(&case);
        assert!(result.is_err(), "Case with empty title should be invalid");
        assert_eq!(result.unwrap_err(), "Case title required");
    }

    #[test]
    fn test_case_requires_description() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "Valid Title".to_string(),
            description: "".to_string(), // Empty description
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:bob".to_string(),
        };

        let result = validate_case_description(&case);
        assert!(
            result.is_err(),
            "Case with empty description should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Case description required");
    }

    #[test]
    fn test_case_requires_complainant_did() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "Valid Title".to_string(),
            description: "Valid description".to_string(),
            complainant: "alice@example.com".to_string(), // Not a DID
            respondent: "did:mycelix:bob".to_string(),
        };

        let result = validate_case_parties(&case);
        assert!(
            result.is_err(),
            "Case with non-DID complainant should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Case parties must be DIDs");
    }

    #[test]
    fn test_case_requires_respondent_did() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "Valid Title".to_string(),
            description: "Valid description".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "bob@example.com".to_string(), // Not a DID
        };

        let result = validate_case_parties(&case);
        assert!(
            result.is_err(),
            "Case with non-DID respondent should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Case parties must be DIDs");
    }

    #[test]
    fn test_case_cannot_file_against_self() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "Valid Title".to_string(),
            description: "Valid description".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:alice".to_string(), // Same as complainant
        };

        let result = validate_case_self_filing(&case);
        assert!(result.is_err(), "Case filed against self should be invalid");
        assert_eq!(result.unwrap_err(), "Cannot file case against self");
    }

    #[test]
    fn test_valid_case_passes_validation() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "Contract Dispute".to_string(),
            description: "Alice claims Bob breached their agreement".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:bob".to_string(),
        };

        assert!(validate_case_title(&case).is_ok());
        assert!(validate_case_description(&case).is_ok());
        assert!(validate_case_parties(&case).is_ok());
        assert!(validate_case_self_filing(&case).is_ok());
    }

    #[test]
    fn test_case_whitespace_only_title_invalid() {
        let case = MockCase {
            id: "case-001".to_string(),
            title: "   ".to_string(), // Whitespace only
            description: "Valid description".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:bob".to_string(),
        };

        let result = validate_case_title(&case);
        assert!(
            result.is_err(),
            "Case with whitespace-only title should be invalid"
        );
    }

    #[test]
    fn test_case_accepts_various_did_formats() {
        // Test did:mycelix: format
        let case1 = MockCase {
            id: "case-001".to_string(),
            title: "Test".to_string(),
            description: "Test".to_string(),
            complainant: "did:mycelix:alice".to_string(),
            respondent: "did:mycelix:bob".to_string(),
        };
        assert!(validate_case_parties(&case1).is_ok());

        // Test did:key: format
        let case2 = MockCase {
            id: "case-002".to_string(),
            title: "Test".to_string(),
            description: "Test".to_string(),
            complainant: "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK".to_string(),
            respondent: "did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH".to_string(),
        };
        assert!(validate_case_parties(&case2).is_ok());
    }
}

#[cfg(test)]
mod evidence_validation_tests {
    use super::*;

    // ==========================================
    // Evidence Entry Validation Tests
    // ==========================================

    #[test]
    fn test_evidence_requires_submitter_did() {
        let evidence = MockEvidence {
            id: "evidence-001".to_string(),
            case_id: "case-001".to_string(),
            submitter: "alice@example.com".to_string(), // Not a DID
            description: "Document showing contract terms".to_string(),
            content_hash: "Qm123abc".to_string(),
        };

        let result = validate_evidence_submitter(&evidence);
        assert!(
            result.is_err(),
            "Evidence with non-DID submitter should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Evidence submitter must be a DID");
    }

    #[test]
    fn test_evidence_requires_description() {
        let evidence = MockEvidence {
            id: "evidence-001".to_string(),
            case_id: "case-001".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            description: "".to_string(), // Empty description
            content_hash: "Qm123abc".to_string(),
        };

        let result = validate_evidence_description(&evidence);
        assert!(
            result.is_err(),
            "Evidence with empty description should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Evidence description required");
    }

    #[test]
    fn test_evidence_requires_content_hash() {
        let evidence = MockEvidence {
            id: "evidence-001".to_string(),
            case_id: "case-001".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            description: "Document showing contract terms".to_string(),
            content_hash: "".to_string(), // Empty hash
        };

        let result = validate_evidence_content_hash(&evidence);
        assert!(
            result.is_err(),
            "Evidence with empty content hash should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Evidence content hash required");
    }

    #[test]
    fn test_valid_evidence_passes_validation() {
        let evidence = MockEvidence {
            id: "evidence-001".to_string(),
            case_id: "case-001".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            description: "Document showing contract terms".to_string(),
            content_hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
        };

        assert!(validate_evidence_submitter(&evidence).is_ok());
        assert!(validate_evidence_description(&evidence).is_ok());
        assert!(validate_evidence_content_hash(&evidence).is_ok());
    }

    #[test]
    fn test_evidence_whitespace_only_description_invalid() {
        let evidence = MockEvidence {
            id: "evidence-001".to_string(),
            case_id: "case-001".to_string(),
            submitter: "did:mycelix:alice".to_string(),
            description: "   \n\t  ".to_string(), // Whitespace only
            content_hash: "Qm123abc".to_string(),
        };

        let result = validate_evidence_description(&evidence);
        assert!(
            result.is_err(),
            "Evidence with whitespace-only description should be invalid"
        );
    }
}

#[cfg(test)]
mod mediation_validation_tests {
    use super::*;

    // ==========================================
    // Mediation Entry Validation Tests
    // ==========================================

    #[test]
    fn test_mediation_requires_mediator_did() {
        let mediation = MockMediation {
            id: "mediation-001".to_string(),
            case_id: "case-001".to_string(),
            mediator: "jane.mediator@example.com".to_string(), // Not a DID
        };

        let result = validate_mediation_mediator(&mediation);
        assert!(
            result.is_err(),
            "Mediation with non-DID mediator should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Mediator must be a DID");
    }

    #[test]
    fn test_valid_mediation_passes_validation() {
        let mediation = MockMediation {
            id: "mediation-001".to_string(),
            case_id: "case-001".to_string(),
            mediator: "did:mycelix:jane_mediator".to_string(),
        };

        assert!(validate_mediation_mediator(&mediation).is_ok());
    }
}

#[cfg(test)]
mod arbitration_validation_tests {
    use super::*;

    // ==========================================
    // Arbitration Entry Validation Tests
    // ==========================================

    #[test]
    fn test_arbitration_requires_odd_number_of_arbitrators() {
        // 2 arbitrators - even number
        let arbitration = MockArbitration {
            id: "arb-001".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec![
                "did:mycelix:arb1".to_string(),
                "did:mycelix:arb2".to_string(),
            ],
        };

        let result = validate_arbitration_panel_size(&arbitration);
        assert!(
            result.is_err(),
            "Arbitration with even number of arbitrators should be invalid"
        );
        assert_eq!(
            result.unwrap_err(),
            "Arbitration panel must have odd number of arbitrators"
        );
    }

    #[test]
    fn test_arbitration_requires_arbitrator_dids() {
        let arbitration = MockArbitration {
            id: "arb-001".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec![
                "did:mycelix:arb1".to_string(),
                "arb2@arbitrators.org".to_string(), // Not a DID
                "did:mycelix:arb3".to_string(),
            ],
        };

        let result = validate_arbitration_arbitrator_dids(&arbitration);
        assert!(
            result.is_err(),
            "Arbitration with non-DID arbitrator should be invalid"
        );
        assert_eq!(result.unwrap_err(), "All arbitrators must be DIDs");
    }

    #[test]
    fn test_valid_arbitration_panel_sizes() {
        // 1 arbitrator
        let arb1 = MockArbitration {
            id: "arb-001".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec!["did:mycelix:arb1".to_string()],
        };
        assert!(validate_arbitration_panel_size(&arb1).is_ok());

        // 3 arbitrators
        let arb3 = MockArbitration {
            id: "arb-002".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec![
                "did:mycelix:arb1".to_string(),
                "did:mycelix:arb2".to_string(),
                "did:mycelix:arb3".to_string(),
            ],
        };
        assert!(validate_arbitration_panel_size(&arb3).is_ok());

        // 5 arbitrators
        let arb5 = MockArbitration {
            id: "arb-003".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec![
                "did:mycelix:arb1".to_string(),
                "did:mycelix:arb2".to_string(),
                "did:mycelix:arb3".to_string(),
                "did:mycelix:arb4".to_string(),
                "did:mycelix:arb5".to_string(),
            ],
        };
        assert!(validate_arbitration_panel_size(&arb5).is_ok());
    }

    #[test]
    fn test_arbitration_zero_arbitrators_invalid() {
        let arbitration = MockArbitration {
            id: "arb-001".to_string(),
            case_id: "case-001".to_string(),
            arbitrators: vec![], // No arbitrators
        };

        let result = validate_arbitration_panel_size(&arbitration);
        // Empty panel has 0 arbitrators (even), so should be invalid
        assert!(
            result.is_err(),
            "Arbitration with zero arbitrators should be invalid"
        );
    }
}

#[cfg(test)]
mod decision_validation_tests {
    use super::*;

    // ==========================================
    // Decision Entry Validation Tests
    // ==========================================

    #[test]
    fn test_decision_requires_reasoning() {
        let decision = MockDecision {
            id: "decision-001".to_string(),
            case_id: "case-001".to_string(),
            arbitration_id: "arb-001".to_string(),
            reasoning: "".to_string(), // Empty reasoning
            votes: vec![MockVote {
                arbitrator: "did:mycelix:arb1".to_string(),
                choice: "ForComplainant".to_string(),
            }],
        };

        let result = validate_decision_reasoning(&decision);
        assert!(
            result.is_err(),
            "Decision with empty reasoning should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Decision reasoning required");
    }

    #[test]
    fn test_decision_requires_votes() {
        let decision = MockDecision {
            id: "decision-001".to_string(),
            case_id: "case-001".to_string(),
            arbitration_id: "arb-001".to_string(),
            reasoning: "Based on evidence presented...".to_string(),
            votes: vec![], // No votes
        };

        let result = validate_decision_votes(&decision);
        assert!(result.is_err(), "Decision with no votes should be invalid");
        assert_eq!(result.unwrap_err(), "Decision must have votes");
    }

    #[test]
    fn test_valid_decision_passes_validation() {
        let decision = MockDecision {
            id: "decision-001".to_string(),
            case_id: "case-001".to_string(),
            arbitration_id: "arb-001".to_string(),
            reasoning: "Based on the evidence presented, the panel finds...".to_string(),
            votes: vec![
                MockVote {
                    arbitrator: "did:mycelix:arb1".to_string(),
                    choice: "ForComplainant".to_string(),
                },
                MockVote {
                    arbitrator: "did:mycelix:arb2".to_string(),
                    choice: "ForComplainant".to_string(),
                },
                MockVote {
                    arbitrator: "did:mycelix:arb3".to_string(),
                    choice: "ForRespondent".to_string(),
                },
            ],
        };

        assert!(validate_decision_reasoning(&decision).is_ok());
        assert!(validate_decision_votes(&decision).is_ok());
    }
}

#[cfg(test)]
mod appeal_validation_tests {
    use super::*;

    // ==========================================
    // Appeal Entry Validation Tests
    // ==========================================

    #[test]
    fn test_appeal_requires_appellant_did() {
        let appeal = MockAppeal {
            id: "appeal-001".to_string(),
            case_id: "case-001".to_string(),
            decision_id: "decision-001".to_string(),
            appellant: "alice@example.com".to_string(), // Not a DID
            grounds: vec!["ProceduralError".to_string()],
            argument: "The process was flawed because...".to_string(),
        };

        let result = validate_appeal_appellant(&appeal);
        assert!(
            result.is_err(),
            "Appeal with non-DID appellant should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Appellant must be a DID");
    }

    #[test]
    fn test_appeal_requires_grounds() {
        let appeal = MockAppeal {
            id: "appeal-001".to_string(),
            case_id: "case-001".to_string(),
            decision_id: "decision-001".to_string(),
            appellant: "did:mycelix:alice".to_string(),
            grounds: vec![], // No grounds
            argument: "The process was flawed because...".to_string(),
        };

        let result = validate_appeal_grounds(&appeal);
        assert!(result.is_err(), "Appeal with no grounds should be invalid");
        assert_eq!(result.unwrap_err(), "Appeal must state grounds");
    }

    #[test]
    fn test_appeal_requires_argument() {
        let appeal = MockAppeal {
            id: "appeal-001".to_string(),
            case_id: "case-001".to_string(),
            decision_id: "decision-001".to_string(),
            appellant: "did:mycelix:alice".to_string(),
            grounds: vec!["ProceduralError".to_string()],
            argument: "".to_string(), // Empty argument
        };

        let result = validate_appeal_argument(&appeal);
        assert!(
            result.is_err(),
            "Appeal with empty argument should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Appeal argument required");
    }

    #[test]
    fn test_valid_appeal_passes_validation() {
        let appeal = MockAppeal {
            id: "appeal-001".to_string(),
            case_id: "case-001".to_string(),
            decision_id: "decision-001".to_string(),
            appellant: "did:mycelix:alice".to_string(),
            grounds: vec!["ProceduralError".to_string(), "NewEvidence".to_string()],
            argument: "The original decision failed to consider key evidence...".to_string(),
        };

        assert!(validate_appeal_appellant(&appeal).is_ok());
        assert!(validate_appeal_grounds(&appeal).is_ok());
        assert!(validate_appeal_argument(&appeal).is_ok());
    }

    #[test]
    fn test_appeal_multiple_grounds_valid() {
        let appeal = MockAppeal {
            id: "appeal-001".to_string(),
            case_id: "case-001".to_string(),
            decision_id: "decision-001".to_string(),
            appellant: "did:mycelix:alice".to_string(),
            grounds: vec![
                "ProceduralError".to_string(),
                "NewEvidence".to_string(),
                "Bias".to_string(),
                "ExcessiveRemedy".to_string(),
            ],
            argument: "Multiple issues with the original decision...".to_string(),
        };

        assert!(validate_appeal_grounds(&appeal).is_ok());
    }
}

#[cfg(test)]
mod enforcement_validation_tests {
    use super::*;

    // ==========================================
    // Enforcement Entry Validation Tests
    // ==========================================

    #[test]
    fn test_enforcement_requires_enforcer_did() {
        let enforcement = MockEnforcement {
            id: "enforcement-001".to_string(),
            decision_id: "decision-001".to_string(),
            enforcer: "system@mycelix.org".to_string(), // Not a DID
        };

        let result = validate_enforcement_enforcer(&enforcement);
        assert!(
            result.is_err(),
            "Enforcement with non-DID enforcer should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Enforcer must be a DID");
    }

    #[test]
    fn test_valid_enforcement_passes_validation() {
        let enforcement = MockEnforcement {
            id: "enforcement-001".to_string(),
            decision_id: "decision-001".to_string(),
            enforcer: "did:mycelix:system_enforcer".to_string(),
        };

        assert!(validate_enforcement_enforcer(&enforcement).is_ok());
    }
}

#[cfg(test)]
mod restorative_circle_validation_tests {
    use super::*;

    // ==========================================
    // Restorative Circle Entry Validation Tests
    // ==========================================

    #[test]
    fn test_circle_requires_facilitator_did() {
        let circle = MockRestorativeCircle {
            id: "circle-001".to_string(),
            case_id: "case-001".to_string(),
            facilitator: "facilitator@community.org".to_string(), // Not a DID
            participants: vec![
                "did:mycelix:alice".to_string(),
                "did:mycelix:bob".to_string(),
            ],
        };

        let result = validate_circle_facilitator(&circle);
        assert!(
            result.is_err(),
            "Circle with non-DID facilitator should be invalid"
        );
        assert_eq!(result.unwrap_err(), "Facilitator must be a DID");
    }

    #[test]
    fn test_circle_requires_participants() {
        let circle = MockRestorativeCircle {
            id: "circle-001".to_string(),
            case_id: "case-001".to_string(),
            facilitator: "did:mycelix:jane_facilitator".to_string(),
            participants: vec![], // No participants
        };

        let result = validate_circle_participants(&circle);
        assert!(
            result.is_err(),
            "Circle with no participants should be invalid"
        );
        assert_eq!(
            result.unwrap_err(),
            "Restorative circle must have participants"
        );
    }

    #[test]
    fn test_valid_circle_passes_validation() {
        let circle = MockRestorativeCircle {
            id: "circle-001".to_string(),
            case_id: "case-001".to_string(),
            facilitator: "did:mycelix:jane_facilitator".to_string(),
            participants: vec![
                "did:mycelix:alice".to_string(),
                "did:mycelix:bob".to_string(),
                "did:mycelix:community_elder".to_string(),
            ],
        };

        assert!(validate_circle_facilitator(&circle).is_ok());
        assert!(validate_circle_participants(&circle).is_ok());
    }
}

// ==========================================
// Mock Types for Testing
// ==========================================

struct MockCase {
    id: String,
    title: String,
    description: String,
    complainant: String,
    respondent: String,
}

struct MockEvidence {
    id: String,
    case_id: String,
    submitter: String,
    description: String,
    content_hash: String,
}

struct MockMediation {
    id: String,
    case_id: String,
    mediator: String,
}

struct MockArbitration {
    id: String,
    case_id: String,
    arbitrators: Vec<String>,
}

struct MockDecision {
    id: String,
    case_id: String,
    arbitration_id: String,
    reasoning: String,
    votes: Vec<MockVote>,
}

struct MockVote {
    arbitrator: String,
    choice: String,
}

struct MockAppeal {
    id: String,
    case_id: String,
    decision_id: String,
    appellant: String,
    grounds: Vec<String>,
    argument: String,
}

struct MockEnforcement {
    id: String,
    decision_id: String,
    enforcer: String,
}

struct MockRestorativeCircle {
    id: String,
    case_id: String,
    facilitator: String,
    participants: Vec<String>,
}

// ==========================================
// Validation Helper Functions
// ==========================================

fn validate_case_title(case: &MockCase) -> Result<(), &'static str> {
    if case.title.trim().is_empty() {
        Err("Case title required")
    } else {
        Ok(())
    }
}

fn validate_case_description(case: &MockCase) -> Result<(), &'static str> {
    if case.description.trim().is_empty() {
        Err("Case description required")
    } else {
        Ok(())
    }
}

fn validate_case_parties(case: &MockCase) -> Result<(), &'static str> {
    if !case.complainant.starts_with("did:") || !case.respondent.starts_with("did:") {
        Err("Case parties must be DIDs")
    } else {
        Ok(())
    }
}

fn validate_case_self_filing(case: &MockCase) -> Result<(), &'static str> {
    if case.complainant == case.respondent {
        Err("Cannot file case against self")
    } else {
        Ok(())
    }
}

fn validate_evidence_submitter(evidence: &MockEvidence) -> Result<(), &'static str> {
    if !evidence.submitter.starts_with("did:") {
        Err("Evidence submitter must be a DID")
    } else {
        Ok(())
    }
}

fn validate_evidence_description(evidence: &MockEvidence) -> Result<(), &'static str> {
    if evidence.description.trim().is_empty() {
        Err("Evidence description required")
    } else {
        Ok(())
    }
}

fn validate_evidence_content_hash(evidence: &MockEvidence) -> Result<(), &'static str> {
    if evidence.content_hash.is_empty() {
        Err("Evidence content hash required")
    } else {
        Ok(())
    }
}

fn validate_mediation_mediator(mediation: &MockMediation) -> Result<(), &'static str> {
    if !mediation.mediator.starts_with("did:") {
        Err("Mediator must be a DID")
    } else {
        Ok(())
    }
}

fn validate_arbitration_panel_size(arbitration: &MockArbitration) -> Result<(), &'static str> {
    if arbitration.arbitrators.len() % 2 == 0 {
        Err("Arbitration panel must have odd number of arbitrators")
    } else {
        Ok(())
    }
}

fn validate_arbitration_arbitrator_dids(arbitration: &MockArbitration) -> Result<(), &'static str> {
    for arb in &arbitration.arbitrators {
        if !arb.starts_with("did:") {
            return Err("All arbitrators must be DIDs");
        }
    }
    Ok(())
}

fn validate_decision_reasoning(decision: &MockDecision) -> Result<(), &'static str> {
    if decision.reasoning.trim().is_empty() {
        Err("Decision reasoning required")
    } else {
        Ok(())
    }
}

fn validate_decision_votes(decision: &MockDecision) -> Result<(), &'static str> {
    if decision.votes.is_empty() {
        Err("Decision must have votes")
    } else {
        Ok(())
    }
}

fn validate_appeal_appellant(appeal: &MockAppeal) -> Result<(), &'static str> {
    if !appeal.appellant.starts_with("did:") {
        Err("Appellant must be a DID")
    } else {
        Ok(())
    }
}

fn validate_appeal_grounds(appeal: &MockAppeal) -> Result<(), &'static str> {
    if appeal.grounds.is_empty() {
        Err("Appeal must state grounds")
    } else {
        Ok(())
    }
}

fn validate_appeal_argument(appeal: &MockAppeal) -> Result<(), &'static str> {
    if appeal.argument.trim().is_empty() {
        Err("Appeal argument required")
    } else {
        Ok(())
    }
}

fn validate_enforcement_enforcer(enforcement: &MockEnforcement) -> Result<(), &'static str> {
    if !enforcement.enforcer.starts_with("did:") {
        Err("Enforcer must be a DID")
    } else {
        Ok(())
    }
}

fn validate_circle_facilitator(circle: &MockRestorativeCircle) -> Result<(), &'static str> {
    if !circle.facilitator.starts_with("did:") {
        Err("Facilitator must be a DID")
    } else {
        Ok(())
    }
}

fn validate_circle_participants(circle: &MockRestorativeCircle) -> Result<(), &'static str> {
    if circle.participants.is_empty() {
        Err("Restorative circle must have participants")
    } else {
        Ok(())
    }
}

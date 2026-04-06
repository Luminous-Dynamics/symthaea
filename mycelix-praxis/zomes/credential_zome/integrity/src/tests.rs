// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit tests for Credential integrity zome validation logic
//!
//! Tests validate entry validation functions for:
//! - VerifiableCredential

use super::*;
use praxis_core::CourseId;

// =============================================================================
// Helper functions for creating test data
// =============================================================================

fn create_valid_credential() -> VerifiableCredential {
    VerifiableCredential {
        context: "https://www.w3.org/2018/credentials/v1".to_string(),
        credential_type: vec!["VerifiableCredential".to_string(), "EducationalCredential".to_string()],
        issuer: "did:example:issuer123".to_string(),
        issuance_date: "2024-01-15T12:00:00Z".to_string(),
        expiration_date: Some("2025-01-15T12:00:00Z".to_string()),
        subject_id: "did:example:learner456".to_string(),
        course_id: CourseId("course_rust_101".to_string()),
        model_id: "model_v1".to_string(),
        rubric_id: "rubric_r1".to_string(),
        score: Some(87.5),
        score_band: "Proficient".to_string(),
        subject_metadata: Some("{\"course_name\":\"Rust Basics\"}".to_string()),
        status_id: Some("https://example.com/status/1".to_string()),
        status_type: Some("StatusList2021Entry".to_string()),
        status_list_index: Some(42),
        status_purpose: Some("revocation".to_string()),
        proof_type: "Ed25519Signature2020".to_string(),
        proof_created: "2024-01-15T12:01:00Z".to_string(),
        verification_method: "did:example:issuer123#key-1".to_string(),
        proof_purpose: "assertionMethod".to_string(),
        proof_value: "abc123signature".to_string(),
        epistemic_empirical: Some(2),
        epistemic_normative: Some(1),
        epistemic_materiality: Some(1),
    }
}

// =============================================================================
// VerifiableCredential validation tests
// =============================================================================

#[cfg(test)]
mod credential_validation_tests {
    use super::*;

    #[test]
    fn test_valid_credential() {
        let cred = create_valid_credential();
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_missing_verifiable_credential_type() {
        let mut cred = create_valid_credential();
        cred.credential_type = vec!["EducationalCredential".to_string()];

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_issuer() {
        let mut cred = create_valid_credential();
        cred.issuer = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_issuance_date() {
        let mut cred = create_valid_credential();
        cred.issuance_date = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_subject_id() {
        let mut cred = create_valid_credential();
        cred.subject_id = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_score_band() {
        let mut cred = create_valid_credential();
        cred.score_band = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_score_negative() {
        let mut cred = create_valid_credential();
        cred.score = Some(-10.0);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_score_above_100() {
        let mut cred = create_valid_credential();
        cred.score = Some(110.0);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_score_at_zero() {
        let mut cred = create_valid_credential();
        cred.score = Some(0.0);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_score_at_100() {
        let mut cred = create_valid_credential();
        cred.score = Some(100.0);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_no_score() {
        let mut cred = create_valid_credential();
        cred.score = None;

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_empty_proof_type() {
        let mut cred = create_valid_credential();
        cred.proof_type = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_proof_created() {
        let mut cred = create_valid_credential();
        cred.proof_created = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_verification_method() {
        let mut cred = create_valid_credential();
        cred.verification_method = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_proof_purpose() {
        let mut cred = create_valid_credential();
        cred.proof_purpose = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_proof_value() {
        let mut cred = create_valid_credential();
        cred.proof_value = "".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_minimal_valid() {
        let cred = VerifiableCredential {
            context: "https://www.w3.org/2018/credentials/v1".to_string(),
            credential_type: vec!["VerifiableCredential".to_string()],
            issuer: "did:example:issuer".to_string(),
            issuance_date: "2024-01-15T12:00:00Z".to_string(),
            expiration_date: None,
            subject_id: "did:example:learner".to_string(),
            course_id: CourseId("course_001".to_string()),
            model_id: "model_v1".to_string(),
            rubric_id: "rubric_r1".to_string(),
            score: None,
            score_band: "Pass".to_string(),
            subject_metadata: None,
            status_id: None,
            status_type: None,
            status_list_index: None,
            status_purpose: None,
            proof_type: "Ed25519Signature2020".to_string(),
            proof_created: "2024-01-15T12:01:00Z".to_string(),
            verification_method: "did:example:issuer#key-1".to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "signature".to_string(),
            epistemic_empirical: None,
            epistemic_normative: None,
            epistemic_materiality: None,
        };

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_with_expiration() {
        let mut cred = create_valid_credential();
        cred.expiration_date = Some("2030-12-31T23:59:59Z".to_string());

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_without_expiration() {
        let mut cred = create_valid_credential();
        cred.expiration_date = None;

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_multiple_types() {
        let mut cred = create_valid_credential();
        cred.credential_type = vec![
            "VerifiableCredential".to_string(),
            "EducationalCredential".to_string(),
            "OpenBadgeCredential".to_string(),
        ];

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_with_metadata() {
        let mut cred = create_valid_credential();
        cred.subject_metadata = Some("{\"course\":\"Rust\",\"level\":\"Advanced\"}".to_string());

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_with_status_list() {
        let mut cred = create_valid_credential();
        cred.status_id = Some("https://example.com/status/revocation".to_string());
        cred.status_type = Some("StatusList2021Entry".to_string());
        cred.status_list_index = Some(1234);
        cred.status_purpose = Some("revocation".to_string());

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_different_proof_types() {
        let mut cred = create_valid_credential();
        cred.proof_type = "RsaSignature2018".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_different_proof_purposes() {
        let mut cred = create_valid_credential();
        cred.proof_purpose = "authentication".to_string();

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_long_proof_value() {
        let mut cred = create_valid_credential();
        cred.proof_value = "a".repeat(1000);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// Edge case tests
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_credential_score_very_small() {
        let mut cred = create_valid_credential();
        cred.score = Some(0.001);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_score_very_close_to_100() {
        let mut cred = create_valid_credential();
        cred.score = Some(99.999);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_very_long_issuer() {
        let mut cred = create_valid_credential();
        cred.issuer = format!("did:example:{}", "a".repeat(500));

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_very_long_subject_id() {
        let mut cred = create_valid_credential();
        cred.subject_id = format!("did:example:{}", "b".repeat(500));

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_many_types() {
        let mut cred = create_valid_credential();
        cred.credential_type = (0..20)
            .map(|i| format!("Type{}", i))
            .collect();
        cred.credential_type.insert(0, "VerifiableCredential".to_string());

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_large_status_index() {
        let mut cred = create_valid_credential();
        cred.status_list_index = Some(1_000_000);

        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// Assessment validation tests
// =============================================================================

#[cfg(test)]
mod assessment_validation_tests {
    use super::*;

    fn create_valid_assessment() -> Assessment {
        Assessment {
            node_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            assessment_type: AssessmentType::Quiz,
            title: "Rust Basics Quiz".to_string(),
            instructions: Some("Answer all questions".to_string()),
            max_score: 100,
            passing_score: 70,
            rubric: vec![RubricCriterion {
                name: "Understanding".to_string(),
                description: "Core concepts".to_string(),
                max_points: 50,
                levels: vec![
                    RubricLevel {
                        label: "Exceeds".to_string(),
                        points: 50,
                        description: "Demonstrates deep understanding".to_string(),
                    },
                    RubricLevel {
                        label: "Meets".to_string(),
                        points: 35,
                        description: "Meets expectations".to_string(),
                    },
                    RubricLevel {
                        label: "Beginning".to_string(),
                        points: 15,
                        description: "Developing understanding".to_string(),
                    },
                ],
            }],
            bloom_level: Some("Apply".to_string()),
            time_limit_minutes: Some(30),
            creator: AgentPubKey::from_raw_36(vec![0xdb; 36]),
            created_at: 1700000000,
        }
    }

    #[test]
    fn test_valid_assessment() {
        let a = create_valid_assessment();
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_assessment_empty_title() {
        let mut a = create_valid_assessment();
        a.title = "".to_string();
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_assessment_title_too_long() {
        let mut a = create_valid_assessment();
        a.title = "x".repeat(201);
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_assessment_title_at_max_length() {
        let mut a = create_valid_assessment();
        a.title = "x".repeat(200);
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_assessment_passing_exceeds_max() {
        let mut a = create_valid_assessment();
        a.passing_score = 101;
        a.max_score = 100;
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_assessment_passing_equals_max() {
        let mut a = create_valid_assessment();
        a.passing_score = 100;
        a.max_score = 100;
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_assessment_too_many_rubric_criteria() {
        let mut a = create_valid_assessment();
        a.rubric = (0..21)
            .map(|i| RubricCriterion {
                name: format!("Criterion {}", i),
                description: "desc".to_string(),
                max_points: 5,
                levels: vec![],
            })
            .collect();
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_assessment_rubric_name_too_long() {
        let mut a = create_valid_assessment();
        a.rubric[0].name = "x".repeat(101);
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_assessment_too_many_rubric_levels() {
        let mut a = create_valid_assessment();
        a.rubric[0].levels = (0..11)
            .map(|i| RubricLevel {
                label: format!("Level {}", i),
                points: i as u32,
                description: "desc".to_string(),
            })
            .collect();
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_assessment_no_rubric_is_valid() {
        let mut a = create_valid_assessment();
        a.rubric = vec![];
        let result = validate_assessment(&a);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_assessment_default_type() {
        assert_eq!(AssessmentType::default(), AssessmentType::Quiz);
    }
}

// =============================================================================
// StudentResult validation tests
// =============================================================================

#[cfg(test)]
mod student_result_validation_tests {
    use super::*;

    fn create_valid_student_result() -> StudentResult {
        StudentResult {
            assessment_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            student: AgentPubKey::from_raw_36(vec![0xdb; 36]),
            score: 85,
            rubric_scores: vec![40, 45],
            attempt_number: 1,
            reasoning_trace: None,
            consciousness_level_permille: Some(750),
            feedback: Some("Good work".to_string()),
            graded_by: Some(AgentPubKey::from_raw_36(vec![0xab; 36])),
            completed_at: 1700000000,
        }
    }

    #[test]
    fn test_valid_student_result() {
        let r = create_valid_student_result();
        let result = validate_student_result(&r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_student_result_zero_attempt() {
        let mut r = create_valid_student_result();
        r.attempt_number = 0;
        let result = validate_student_result(&r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_student_result_consciousness_over_1000() {
        let mut r = create_valid_student_result();
        r.consciousness_level_permille = Some(1001);
        let result = validate_student_result(&r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_student_result_consciousness_at_1000() {
        let mut r = create_valid_student_result();
        r.consciousness_level_permille = Some(1000);
        let result = validate_student_result(&r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_student_result_no_consciousness_level() {
        let mut r = create_valid_student_result();
        r.consciousness_level_permille = None;
        let result = validate_student_result(&r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// ReportCard validation tests
// =============================================================================

#[cfg(test)]
mod report_card_validation_tests {
    use super::*;

    fn create_valid_report_card() -> ReportCard {
        ReportCard {
            student: AgentPubKey::from_raw_36(vec![0xdb; 36]),
            classroom_hash: None,
            period: "2025-Spring".to_string(),
            subject_grades: vec![SubjectGrade {
                subject: "Mathematics".to_string(),
                grade_band: "A".to_string(),
                mastery_permille: 920,
                standards_met: 18,
                standards_total: 20,
            }],
            total_mastery_permille: 920,
            attendance_days: 170,
            absence_days: 10,
            teacher_narrative: Some("Excellent student".to_string()),
            generated_at: 1700000000,
        }
    }

    #[test]
    fn test_valid_report_card() {
        let rc = create_valid_report_card();
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_report_card_empty_period() {
        let mut rc = create_valid_report_card();
        rc.period = "".to_string();
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_report_card_mastery_over_1000() {
        let mut rc = create_valid_report_card();
        rc.total_mastery_permille = 1001;
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_report_card_subject_empty_name() {
        let mut rc = create_valid_report_card();
        rc.subject_grades[0].subject = "".to_string();
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_report_card_subject_mastery_over_1000() {
        let mut rc = create_valid_report_card();
        rc.subject_grades[0].mastery_permille = 1001;
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_report_card_standards_met_exceeds_total() {
        let mut rc = create_valid_report_card();
        rc.subject_grades[0].standards_met = 21;
        rc.subject_grades[0].standards_total = 20;
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_report_card_no_grades_is_valid() {
        let mut rc = create_valid_report_card();
        rc.subject_grades = vec![];
        let result = validate_report_card(&rc);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// Security validation tests (is_finite guards)
// =============================================================================

#[cfg(test)]
mod security_validation_tests {
    use super::*;

    /// Helper that creates a credential with all fields populated (including epistemic)
    fn test_credential() -> VerifiableCredential {
        let mut cred = create_valid_credential();
        cred.epistemic_empirical = Some(2);
        cred.epistemic_normative = Some(1);
        cred.epistemic_materiality = Some(1);
        cred
    }

    #[test]
    fn test_credential_valid() {
        let cred = test_credential();
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_credential_nan_score_rejected() {
        let mut cred = test_credential();
        cred.score = Some(f32::NAN);
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_inf_score_rejected() {
        let mut cred = test_credential();
        cred.score = Some(f32::INFINITY);
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_neg_inf_score_rejected() {
        let mut cred = test_credential();
        cred.score = Some(f32::NEG_INFINITY);
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_issuer_rejected() {
        let mut cred = test_credential();
        cred.issuer = "".to_string();
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_missing_type_rejected() {
        let mut cred = test_credential();
        cred.credential_type = vec!["EducationalCredential".to_string()];
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_credential_empty_type_vec_rejected() {
        let mut cred = test_credential();
        cred.credential_type = vec![];
        let result = validate_verifiable_credential(cred);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}

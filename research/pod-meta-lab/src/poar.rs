use std::collections::BTreeSet;

use crate::{
    apply_operation_receipt, choose_next, CandidateOperation, EvidenceVerdict, Obligation,
    ObligationState, OperationKind, OperationReceipt, OperationStatus, ProofVerdict,
    SchedulingClass, Selection, SelectionContext, StopReason, VerifierReceipt,
};

#[derive(Debug, Clone)]
pub struct ScheduleScenario {
    pub obligations: Vec<Obligation>,
    pub operations: Vec<CandidateOperation>,
    pub attempted_without_progress: BTreeSet<(String, String)>,
    pub remaining_work_units: u64,
    pub mandatory_reserve_units: u64,
    pub expected: Selection,
}

#[derive(Debug, Clone)]
pub struct ReceiptReplayScenario {
    pub obligations: Vec<Obligation>,
    pub original_receipt: OperationReceipt,
    pub revocation_receipt: OperationReceipt,
    pub obligation_id: String,
    pub expected_state: ObligationState,
    pub expected_state_identity: String,
}

#[derive(Debug, Clone)]
pub enum PoarScenario {
    Schedule(ScheduleScenario),
    ReceiptReplay(ReceiptReplayScenario),
}

#[derive(Debug, Clone)]
pub struct PoarFixture {
    pub id: &'static str,
    pub scenario: PoarScenario,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoarCaseResult {
    pub id: String,
    pub passed: bool,
    pub observation: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoarReport {
    pub case_results: Vec<PoarCaseResult>,
}

impl PoarReport {
    pub fn case_count(&self) -> usize {
        self.case_results.len()
    }

    pub fn passed_count(&self) -> usize {
        self.case_results.iter().filter(|case| case.passed).count()
    }

    pub fn failed_count(&self) -> usize {
        self.case_count() - self.passed_count()
    }
}

fn obligation(id: &str, state: ObligationState, priority: u8) -> Obligation {
    Obligation::new(id, state, priority).with_state_identity("fixture-state-v1")
}

fn operation(
    id: &str,
    kind: OperationKind,
    targets: &[&str],
    class: SchedulingClass,
    work_units: u64,
    progress_key: &str,
) -> CandidateOperation {
    CandidateOperation::new(
        id,
        kind,
        targets.iter().copied(),
        class,
        work_units,
        progress_key,
    )
}

fn schedule_case(
    id: &'static str,
    obligations: Vec<Obligation>,
    operations: Vec<CandidateOperation>,
    attempted_without_progress: BTreeSet<(String, String)>,
    remaining_work_units: u64,
    mandatory_reserve_units: u64,
    expected: Selection,
) -> PoarFixture {
    PoarFixture {
        id,
        scenario: PoarScenario::Schedule(ScheduleScenario {
            obligations,
            operations,
            attempted_without_progress,
            remaining_work_units,
            mandatory_reserve_units,
            expected,
        }),
    }
}

fn verifier_receipt(
    operation_id: &str,
    progress_key: &str,
    result_state_identity: &str,
    evidence_verdict: EvidenceVerdict,
) -> VerifierReceipt {
    VerifierReceipt {
        verifier_id: "independent-fixture-verifier".into(),
        operation_id: operation_id.into(),
        progress_key: progress_key.into(),
        obligation_ids: BTreeSet::from(["claim".to_string()]),
        proof_profile_id: "poar-profile-v1".into(),
        proof_identity: "synthetic-proof-reference".into(),
        result_state_identity: result_state_identity.into(),
        proof_verdict: ProofVerdict::Valid,
        evidence_verdict,
    }
}

fn receipt_replay_case() -> PoarFixture {
    let obligations = vec![
        Obligation::new(
            "claim",
            ObligationState::CandidateEvidenceAvailable,
            8,
        )
        .with_state_identity("evidence-v1"),
    ];
    let original_receipt = OperationReceipt {
        producer_id: "planner-v1".into(),
        operation_id: "verify-v1".into(),
        progress_key: "evidence-v1".into(),
        obligation_ids: BTreeSet::from(["claim".to_string()]),
        status: OperationStatus::Completed,
        verifier: Some(verifier_receipt(
            "verify-v1",
            "evidence-v1",
            "verified-v1",
            EvidenceVerdict::Admissible,
        )),
    };
    let revocation_receipt = OperationReceipt {
        producer_id: "ledger-v2".into(),
        operation_id: "refresh-v2".into(),
        progress_key: "verified-v1".into(),
        obligation_ids: BTreeSet::from(["claim".to_string()]),
        status: OperationStatus::Completed,
        verifier: Some(VerifierReceipt {
            verifier_id: "ledger-verifier-v2".into(),
            operation_id: "refresh-v2".into(),
            progress_key: "verified-v1".into(),
            obligation_ids: BTreeSet::from(["claim".to_string()]),
            proof_profile_id: "poar-profile-v1".into(),
            proof_identity: "revocation-proof-reference".into(),
            result_state_identity: "revoked-v2".into(),
            proof_verdict: ProofVerdict::Valid,
            evidence_verdict: EvidenceVerdict::StaleOrRetracted,
        }),
    };

    PoarFixture {
        id: "receipt-replay-after-revocation",
        scenario: PoarScenario::ReceiptReplay(ReceiptReplayScenario {
            obligations,
            original_receipt,
            revocation_receipt,
            obligation_id: "claim".into(),
            expected_state: ObligationState::StaleOrRetracted,
            expected_state_identity: "revoked-v2".into(),
        }),
    }
}

/// Return the public deterministic POAR-v1 conformance fixtures.
///
/// This is a scheduler/receipt contract smoke suite. It is not a model
/// leaderboard, a hidden-test evaluation, or an independent proof checker.
pub fn poar_v1_fixtures() -> Vec<PoarFixture> {
    vec![
        schedule_case(
            "single-missing-premise",
            vec![obligation("premise", ObligationState::Open, 7)],
            vec![
                operation("retrieve-premise", OperationKind::Retrieve, &["premise"], SchedulingClass::Competitive, 2, "q1"),
                operation("verify-missing", OperationKind::Verify, &["premise"], SchedulingClass::Competitive, 1, "q1"),
            ],
            BTreeSet::new(),
            10,
            2,
            Selection::Run {
                operation_id: "retrieve-premise".into(),
                kind: OperationKind::Retrieve,
                obligation_ids: vec!["premise".into()],
                progress_key: "q1".into(),
                work_units: 2,
            },
        ),
        schedule_case(
            "higher-consequence-open-obligation-selected",
            vec![
                obligation("decoy", ObligationState::Open, 2),
                obligation("decisive", ObligationState::Open, 9),
            ],
            vec![
                operation("retrieve-decoy", OperationKind::Retrieve, &["decoy"], SchedulingClass::Competitive, 1, "q2"),
                operation("retrieve-low-salience-decisive", OperationKind::Retrieve, &["decisive"], SchedulingClass::Competitive, 3, "q2"),
            ],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "retrieve-low-salience-decisive".into(),
                kind: OperationKind::Retrieve,
                obligation_ids: vec!["decisive".into()],
                progress_key: "q2".into(),
                work_units: 3,
            },
        ),
        schedule_case(
            "candidate-derivation-routes-to-verification",
            vec![obligation("derivation", ObligationState::CandidateEvidenceAvailable, 8)],
            vec![
                operation("verify-derivation", OperationKind::Verify, &["derivation"], SchedulingClass::Competitive, 1, "q3"),
                operation("repair-before-check", OperationKind::RepairDerivation, &["derivation"], SchedulingClass::Competitive, 1, "q3"),
            ],
            BTreeSet::new(),
            8,
            0,
            Selection::Run {
                operation_id: "verify-derivation".into(),
                kind: OperationKind::Verify,
                obligation_ids: vec!["derivation".into()],
                progress_key: "q3".into(),
                work_units: 1,
            },
        ),
        schedule_case(
            "overthinking-trap",
            vec![obligation("claim", ObligationState::SupportedUnderProfile, 8)],
            vec![operation("unneeded-retrieval", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 1, "q4")],
            BTreeSet::new(),
            10,
            0,
            Selection::Stop(StopReason::AllObligationsDischarged),
        ),
        schedule_case(
            "contested-obligation-routes-to-independent-evidence",
            vec![obligation("claim", ObligationState::Contested, 8)],
            vec![operation("seek-independent-evidence", OperationKind::SeekIndependentEvidence, &["claim"], SchedulingClass::Competitive, 3, "q5")],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "seek-independent-evidence".into(),
                kind: OperationKind::SeekIndependentEvidence,
                obligation_ids: vec!["claim".into()],
                progress_key: "q5".into(),
                work_units: 3,
            },
        ),
        schedule_case(
            "candidate-evidence-routes-to-alternative-comparison",
            vec![obligation("claim", ObligationState::CandidateEvidenceAvailable, 7)],
            vec![
                operation("verify-one-path", OperationKind::Verify, &["claim"], SchedulingClass::Competitive, 1, "q6"),
                operation("compare-alternatives", OperationKind::CompareAlternatives, &["claim"], SchedulingClass::Mandatory, 4, "q6"),
            ],
            BTreeSet::new(),
            10,
            2,
            Selection::Run {
                operation_id: "compare-alternatives".into(),
                kind: OperationKind::CompareAlternatives,
                obligation_ids: vec!["claim".into()],
                progress_key: "q6".into(),
                work_units: 4,
            },
        ),
        schedule_case(
            "stale-evidence-routes-to-refresh",
            vec![obligation("claim", ObligationState::StaleOrRetracted, 8)],
            vec![
                operation("refresh-revision", OperationKind::RefreshEvidence, &["claim"], SchedulingClass::Competitive, 2, "q7"),
                operation("repeat-retrieval", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 3, "q7"),
            ],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "refresh-revision".into(),
                kind: OperationKind::RefreshEvidence,
                obligation_ids: vec!["claim".into()],
                progress_key: "q7".into(),
                work_units: 2,
            },
        ),
        schedule_case(
            "retracted-target-selected-over-supported-target",
            vec![
                obligation("dependent", ObligationState::StaleOrRetracted, 9),
                obligation("unrelated", ObligationState::SupportedUnderProfile, 4),
            ],
            vec![
                operation("refresh-dependent", OperationKind::RefreshEvidence, &["dependent"], SchedulingClass::Mandatory, 2, "q8"),
                operation("verify-unrelated", OperationKind::Verify, &["unrelated"], SchedulingClass::Mandatory, 1, "q8"),
            ],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "refresh-dependent".into(),
                kind: OperationKind::RefreshEvidence,
                obligation_ids: vec!["dependent".into()],
                progress_key: "q8".into(),
                work_units: 2,
            },
        ),
        schedule_case(
            "scope-authority-mismatch",
            vec![obligation("claim", ObligationState::Unauthorized, 8)],
            vec![operation("request-clarification", OperationKind::RequestClarification, &["claim"], SchedulingClass::Mandatory, 1, "q9")],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "request-clarification".into(),
                kind: OperationKind::RequestClarification,
                obligation_ids: vec!["claim".into()],
                progress_key: "q9".into(),
                work_units: 1,
            },
        ),
        schedule_case(
            "one-unit-over-budget",
            vec![obligation("claim", ObligationState::Open, 8)],
            vec![operation("retrieve-exceeds-budget", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 7, "q10")],
            BTreeSet::new(),
            10,
            4,
            Selection::Stop(StopReason::BudgetExhausted),
        ),
        schedule_case(
            "contested-obligation-selects-independent-evidence",
            vec![obligation("claim", ObligationState::Contested, 8)],
            vec![operation("seek-independent-source", OperationKind::SeekIndependentEvidence, &["claim"], SchedulingClass::Mandatory, 2, "q11")],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "seek-independent-source".into(),
                kind: OperationKind::SeekIndependentEvidence,
                obligation_ids: vec!["claim".into()],
                progress_key: "q11".into(),
                work_units: 2,
            },
        ),
        schedule_case(
            "requires-external-observation",
            vec![obligation("physical-fact", ObligationState::UnresolvableWithoutExternalInput, 9)],
            Vec::new(),
            BTreeSet::new(),
            10,
            0,
            Selection::Stop(StopReason::NeedsExternalEvidence),
        ),
        schedule_case(
            "multi-obligation-targets-are-normalized",
            vec![
                obligation("p1", ObligationState::Open, 7),
                obligation("p2", ObligationState::Open, 7),
                obligation("p3", ObligationState::Open, 7),
            ],
            vec![operation("retrieve-chain-frontier", OperationKind::Retrieve, &["p3", "p1", "p2"], SchedulingClass::Mandatory, 5, "q13")],
            BTreeSet::new(),
            10,
            0,
            Selection::Run {
                operation_id: "retrieve-chain-frontier".into(),
                kind: OperationKind::Retrieve,
                obligation_ids: vec!["p1".into(), "p2".into(), "p3".into()],
                progress_key: "q13".into(),
                work_units: 5,
            },
        ),
        schedule_case(
            "null-benefit-cached-query",
            vec![obligation("cached", ObligationState::SupportedUnderProfile, 2)],
            vec![operation("repeat-cached-query", OperationKind::Retrieve, &["cached"], SchedulingClass::Opportunistic, 1, "q14")],
            BTreeSet::new(),
            10,
            0,
            Selection::Stop(StopReason::AllObligationsDischarged),
        ),
        receipt_replay_case(),
        schedule_case(
            "unchanged-state-no-progress-stop",
            vec![obligation("claim", ObligationState::Open, 5)],
            vec![operation("retrieve-again", OperationKind::Retrieve, &["claim"], SchedulingClass::Competitive, 1, "q16")],
            BTreeSet::from([("retrieve-again".into(), "q16".into())]),
            10,
            0,
            Selection::Stop(StopReason::NoSemanticProgress),
        ),
    ]
}

/// Execute the public POAR-v1 scheduler/receipt conformance smoke suite.
pub fn run_poar_v1() -> PoarReport {
    let mut case_results = Vec::new();

    for fixture in poar_v1_fixtures() {
        match fixture.scenario {
            PoarScenario::Schedule(scenario) => {
                let decision = choose_next(&SelectionContext {
                    obligations: &scenario.obligations,
                    operations: &scenario.operations,
                    attempted_without_progress: &scenario.attempted_without_progress,
                    remaining_work_units: scenario.remaining_work_units,
                    mandatory_reserve_units: scenario.mandatory_reserve_units,
                });
                let passed = decision.selection == scenario.expected;
                case_results.push(PoarCaseResult {
                    id: fixture.id.to_string(),
                    passed,
                    observation: format!("{:?}", decision.selection),
                });
            }
            PoarScenario::ReceiptReplay(scenario) => {
                let mut obligations = scenario.obligations;
                let initial = apply_operation_receipt(&mut obligations, &scenario.original_receipt);
                let revocation = apply_operation_receipt(&mut obligations, &scenario.revocation_receipt);
                let replay = apply_operation_receipt(&mut obligations, &scenario.original_receipt);
                let expected_state_matches = match obligations
                    .iter()
                    .find(|obligation| obligation.id == scenario.obligation_id)
                {
                    Some(obligation) => {
                        obligation.state == scenario.expected_state
                            && obligation.state_identity == scenario.expected_state_identity
                    }
                    None => false,
                };
                let passed = initial == Ok(1)
                    && revocation == Ok(1)
                    && replay == Err(crate::ReceiptError::ProgressBindingMismatch)
                    && expected_state_matches;
                case_results.push(PoarCaseResult {
                    id: fixture.id.to_string(),
                    passed,
                    observation: format!(
                        "initial={initial:?}; revocation={revocation:?}; replay={replay:?}; final_state_matches={expected_state_matches}"
                    ),
                });
            }
        }
    }

    PoarReport { case_results }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poar_v1_public_conformance_fixtures_match_expected_outcomes() {
        let report = run_poar_v1();
        assert_eq!(report.case_count(), 16);
        let fixture_ids = poar_v1_fixtures()
            .iter()
            .map(|fixture| fixture.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(fixture_ids.len(), report.case_count(), "fixture IDs must be unique");
        assert_eq!(
            report.passed_count(),
            16,
            "POAR fixture failures: {:#?}",
            report.case_results.iter().filter(|case| !case.passed).collect::<Vec<_>>()
        );
    }
}

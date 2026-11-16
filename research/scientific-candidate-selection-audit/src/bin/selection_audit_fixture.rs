use scientific_candidate_selection_audit::{
    audit_submission, run_selector, CandidateSelectionPolicy, CandidateState, DevelopmentEligibility,
    DevelopmentMetrics, EvaluatorAdmissibility, EvaluatorMetrics, EvaluatorOutcome,
    EvaluatorStatus, EvidenceStage, MetricMask, PermittedEvidence, SelectionAuditError,
    SelectionDecision, SelectionObservation, SelectorCandidate, SelectorPacket,
};

const POOL_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const EVIDENCE_SHA: &str =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

struct DevelopmentSelector;

impl CandidateSelectionPolicy for DevelopmentSelector {
    fn policy_id(&self) -> &str {
        "development-score-v0"
    }

    fn select(
        &self,
        packet: &SelectorPacket,
    ) -> Result<SelectionObservation, SelectionAuditError> {
        // Deliberately simple reference selector for the audit fixture.
        // It reads only development evidence and does not receive the
        // evaluator outcomes supplied later to audit_submission.
        let chosen = packet
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.state == CandidateState::Ready
                    && candidate.eligibility == DevelopmentEligibility::Eligible
            })
            .max_by(|left, right| {
                left.development
                    .predictive_score
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(
                        &right
                            .development
                            .predictive_score
                            .unwrap_or(f64::NEG_INFINITY),
                    )
                    .then_with(|| right.candidate_id.cmp(&left.candidate_id))
            });

        let decision = match chosen {
            Some(candidate) => SelectionDecision::Select {
                candidate_id: candidate.candidate_id.clone(),
            },
            None => SelectionDecision::Abstain {
                reason: "no development-eligible candidate".to_owned(),
            },
        };
        Ok(SelectionObservation {
            decision,
            consumed_evidence_ids: vec!["development-fold-01".to_owned()],
            all_in_cost_units: 3.0,
        })
    }
}

fn candidate(
    id: &str,
    model_sha: &str,
    development_score: f64,
) -> SelectorCandidate {
    SelectorCandidate {
        candidate_id: id.to_owned(),
        canonical_model_sha256: Some(model_sha.to_owned()),
        state: CandidateState::Ready,
        eligibility: DevelopmentEligibility::Eligible,
        development: DevelopmentMetrics {
            predictive_score: Some(development_score),
            stability_score: Some(0.9),
            complexity: Some(4.0),
        },
        evidence_ids: vec!["development-fold-01".to_owned()],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let packet = SelectorPacket {
        task_id: "scilaws-real-candidate-selection-fixture".to_owned(),
        candidate_pool_sha256: POOL_SHA.to_owned(),
        metric_mask: MetricMask::REAL_FIT_AND_VALIDITY,
        permitted_evidence: vec![PermittedEvidence {
            evidence_id: "development-fold-01".to_owned(),
            sha256: EVIDENCE_SHA.to_owned(),
            stage: EvidenceStage::Development,
        }],
        candidates: vec![
            candidate(
                "candidate-a",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                0.91,
            ),
            candidate(
                "candidate-b",
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                0.82,
            ),
            candidate(
                "candidate-c",
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                0.80,
            ),
            candidate(
                "candidate-d",
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                0.77,
            ),
        ],
    };

    let trace = run_selector(&DevelopmentSelector, &packet)?;
    let outcomes = vec![
        EvaluatorOutcome {
            candidate_id: "candidate-a".to_owned(),
            status: EvaluatorStatus::Scored,
            admissibility: EvaluatorAdmissibility::Qualified,
            metrics: EvaluatorMetrics {
                numeric_fit: Some(70.0),
                scientific_validity: Some(85.0),
                structure_recovery: None,
            },
        },
        EvaluatorOutcome {
            candidate_id: "candidate-b".to_owned(),
            status: EvaluatorStatus::Scored,
            admissibility: EvaluatorAdmissibility::Qualified,
            metrics: EvaluatorMetrics {
                numeric_fit: Some(75.0),
                scientific_validity: Some(90.0),
                structure_recovery: None,
            },
        },
        EvaluatorOutcome {
            candidate_id: "candidate-c".to_owned(),
            status: EvaluatorStatus::Scored,
            admissibility: EvaluatorAdmissibility::Qualified,
            metrics: EvaluatorMetrics {
                numeric_fit: Some(80.0),
                scientific_validity: Some(82.0),
                structure_recovery: None,
            },
        },
        EvaluatorOutcome {
            candidate_id: "candidate-d".to_owned(),
            status: EvaluatorStatus::TimedOut,
            admissibility: EvaluatorAdmissibility::NotAssessed,
            metrics: EvaluatorMetrics::default(),
        },
    ];

    // Evaluator labels enter only after the selector decision and evidence
    // trace have been frozen and validated.
    let report = audit_submission(&packet, &trace, &outcomes)?;
    println!("{}", report.to_json());
    Ok(())
}

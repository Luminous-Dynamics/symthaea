use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_domotic::home_intelligence::{
    validate_home_proposal, ApprovalRequirement, CandidatePlan, EpistemicStatus, EvidenceRef,
    HomeProposal, ProposalMode, ProposedEffect, RiskClass, UncertaintyLevel,
    HOME_PROPOSAL_SCHEMA_V1,
};

const CORPUS: &str = include_str!("fixtures/home_intelligence_corpus_v1.json");
const NOW_MS: u64 = 1_000;

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    corpus_id: String,
    schema_version: u16,
    reference_now_ms: u64,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
struct CorpusCase {
    id: String,
    scenario: String,
    class: String,
    expected: String,
    description: String,
}

fn evidence(status: EpistemicStatus, epoch: u64) -> EvidenceRef {
    EvidenceRef {
        evidence_id: "evidence-1".into(),
        source_ref: "ha-source-1".into(),
        entity_ref: "sensor.living_room_temperature".into(),
        observed_at_ms: 900,
        fresh_until_ms: 2_000,
        session_epoch: epoch,
        status,
        derived_from: Vec::new(),
    }
}

fn valid_proposal() -> HomeProposal {
    HomeProposal {
        schema_version: HOME_PROPOSAL_SCHEMA_V1,
        proposal_id: "proposal-1".into(),
        request_ref: "request-1".into(),
        generated_at_ms: 950,
        expires_at_ms: 1_500,
        mode: ProposalMode::Recommend,
        summary: "Consider turning on the living-room lamp.".into(),
        evidence: vec![evidence(EpistemicStatus::Observed, 7)],
        assumptions: vec!["The resident wants the room illuminated.".into()],
        unknowns: Vec::new(),
        uncertainty: UncertaintyLevel::Low,
        risk_class: RiskClass::Low,
        approval_requirement: ApprovalRequirement::ResidentConfirmation,
        candidates: vec![CandidatePlan {
            summary: "Turn on the named lamp.".into(),
            evidence_ids: vec!["evidence-1".into()],
            effects: vec![ProposedEffect::SetLightState {
                entity_ref: "light.living_room".into(),
                enabled: true,
            }],
            tradeoffs: vec!["Uses a small amount of electricity.".into()],
        }],
    }
}

fn current_epochs() -> BTreeMap<String, u64> {
    BTreeMap::from([("ha-source-1".to_string(), 7)])
}

fn derived_evidence(id: &str, parents: &[&str]) -> EvidenceRef {
    EvidenceRef {
        evidence_id: id.into(),
        source_ref: "ha-source-1".into(),
        entity_ref: "sensor.living_room_temperature".into(),
        observed_at_ms: 920,
        fresh_until_ms: 2_000,
        session_epoch: 7,
        status: EpistemicStatus::Derived,
        derived_from: parents.iter().map(|value| (*value).to_string()).collect(),
    }
}

fn build_scenario(scenario: &str) -> (HomeProposal, BTreeMap<String, u64>, u64) {
    let mut proposal = valid_proposal();
    let mut epochs = current_epochs();
    let now_ms = NOW_MS;

    match scenario {
        "valid-recommendation" => {}
        "stale-evidence-action" => proposal.evidence[0].fresh_until_ms = 999,
        "prior-session-action" => proposal.evidence[0].session_epoch = 6,
        "unknown-evidence-action" => proposal.evidence[0].status = EpistemicStatus::Unknown,
        "conflicting-evidence-action" => proposal.evidence[0].status = EpistemicStatus::Conflicting,
        "unknown-context-abstention" => {
            proposal.mode = ProposalMode::Abstain;
            proposal.candidates.clear();
            proposal.evidence[0].status = EpistemicStatus::Unknown;
            proposal.unknowns = vec!["The current sensor value is unknown.".into()];
        }
        "stale-context-abstention" => {
            proposal.mode = ProposalMode::Abstain;
            proposal.candidates.clear();
            proposal.evidence[0].fresh_until_ms = 999;
            proposal.unknowns = vec!["The last observation expired.".into()];
        }
        "prior-session-context-abstention" => {
            proposal.mode = ProposalMode::Abstain;
            proposal.candidates.clear();
            proposal.evidence[0].session_epoch = 6;
            proposal.unknowns = vec!["The provider reconnected; old data is not current.".into()];
        }
        "high-risk-candidate" => proposal.risk_class = RiskClass::High,
        "uncalibrated-candidate" => proposal.uncertainty = UncertaintyLevel::Uncalibrated,
        "missing-citation" => proposal.candidates[0].evidence_ids.clear(),
        "missing-source-evidence" => proposal.evidence.clear(),
        "duplicate-evidence-id" => proposal.evidence.push(proposal.evidence[0].clone()),
        "invalid-derived-parent" => {
            proposal.evidence.push(derived_evidence("derived-2", &["missing-parent"]));
        }
        "derived-parent-not-cited" => {
            proposal.evidence.push(derived_evidence("derived-2", &["evidence-1"]));
            proposal.candidates[0].evidence_ids = vec!["derived-2".into()];
        }
        "expired-proposal" => proposal.expires_at_ms = NOW_MS,
        "future-evidence" => proposal.evidence[0].observed_at_ms = NOW_MS + 1,
        "missing-current-epoch" => epochs.clear(),
        "invalid-mode-effects" => proposal.mode = ProposalMode::Clarify,
        "comfort-out-of-range" => {
            proposal.candidates[0].effects = vec![ProposedEffect::SuggestComfortTarget {
                entity_ref: "climate.living_room".into(),
                target_millicelsius: 31_001,
            }];
        }
        "oversized-summary" => proposal.summary = "x".repeat(600),
        "control-character-summary" => proposal.summary = "Bad\nsummary".into(),
        "missing-approval" => proposal.approval_requirement = ApprovalRequirement::None,
        "wrong-schema" => proposal.schema_version = HOME_PROPOSAL_SCHEMA_V1 + 1,
        "future-proposal" => proposal.generated_at_ms = NOW_MS + 1,
        "clarify-without-target" => {
            proposal.mode = ProposalMode::Clarify;
            proposal.candidates.clear();
            proposal.unknowns.clear();
        }
        "explain-unknown-context" => {
            proposal.mode = ProposalMode::Explain;
            proposal.candidates.clear();
            proposal.evidence[0].status = EpistemicStatus::Unknown;
            proposal.summary = "The current state cannot be confirmed.".into();
        }
        "duplicate-candidate-citation" => {
            proposal.candidates[0].evidence_ids =
                vec!["evidence-1".into(), "evidence-1".into()];
        }
        "invalid-derived-order" => {
            proposal.evidence.push(derived_evidence("derived-2", &["evidence-1"]));
            proposal.evidence.swap(0, 1);
        }
        "unavailable-context-explanation" => {
            proposal.mode = ProposalMode::Explain;
            proposal.candidates.clear();
            proposal.evidence[0].status = EpistemicStatus::Unavailable;
            proposal.summary = "The provider is unavailable.".into();
        }
        _ => panic!("unrecognized HOME-INTEL scenario: {scenario}"),
    }

    (proposal, epochs, now_ms)
}

#[test]
fn replay_frozen_home_intel_contract_corpus() {
    let manifest: CorpusManifest =
        serde_json::from_str(CORPUS).expect("corpus manifest must parse");

    assert_eq!(manifest.corpus_id, "HOME-INTEL-001B");
    assert_eq!(manifest.schema_version, 1, "unsupported corpus schema version");
    assert_eq!(manifest.reference_now_ms, NOW_MS);

    let mut case_ids = BTreeSet::new();
    let mut scenario_ids = BTreeSet::new();
    let mut positive = 0usize;
    let mut negative = 0usize;

    for case in &manifest.cases {
        assert!(!case.id.trim().is_empty(), "case ID must be nonempty");
        assert!(case_ids.insert(case.id.as_str()), "duplicate case ID: {}", case.id);
        assert!(
            scenario_ids.insert(case.scenario.as_str()),
            "duplicate scenario ID: {}",
            case.scenario
        );
        assert!(!case.description.trim().is_empty(), "case requires rationale: {}", case.id);
        match case.class.as_str() {
            "positive" => {
                positive += 1;
                assert_eq!(case.expected, "accept", "positive case has reject oracle: {}", case.id);
            }
            "negative" => {
                negative += 1;
                assert_ne!(case.expected, "accept", "negative control cannot expect acceptance: {}", case.id);
            }
            other => panic!("unknown corpus class {other} for {}", case.id),
        }

        let (proposal, epochs, now_ms) = build_scenario(&case.scenario);
        let result = validate_home_proposal(&proposal, now_ms, &epochs);
        let actual = match result {
            Ok(()) => "accept".to_string(),
            Err(error) => format!("{error:?}"),
        };

        assert_eq!(
            actual, case.expected,
            "scenario {} ({}) mismatch: {}",
            case.id, case.scenario, case.description
        );
    }

    assert_eq!(manifest.cases.len(), 30, "unexpected corpus case count");
    assert!(positive >= 4, "corpus needs multiple positive controls");
    assert!(negative >= 20, "corpus needs broad negative-control coverage");

    let digest = Sha256::digest(CORPUS.as_bytes());
    let digest_hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    eprintln!(
        "HOME_INTEL_CORPUS_REPLAY corpus={} schema={} cases={} positive={} negative={} sha256={}",
        manifest.corpus_id,
        manifest.schema_version,
        manifest.cases.len(),
        positive,
        negative,
        digest_hex
    );
}

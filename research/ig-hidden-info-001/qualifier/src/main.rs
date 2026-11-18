//! Structural qualifier for IG-HIDDEN-INFO-001 synthetic fixtures.
//!
//! This program checks event provenance and evidence availability. It does not
//! execute agents, infer evidence from prose, or judge logical entailment.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;

#[derive(Debug, Deserialize)]
struct FixtureSet {
    schema_version: u32,
    protocol_id: String,
    fixtures: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize)]
struct Scenario {
    id: String,
    evidence_catalog: Vec<EvidenceAtom>,
    required_decisive_atoms: Vec<String>,
    agents: Vec<Agent>,
    decision_agent_id: String,
    deliveries: Vec<Delivery>,
    final_record: Option<FinalRecord>,
    expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
struct EvidenceAtom {
    atom_id: String,
    decisive: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Agent {
    agent_id: String,
    initial_evidence_atoms: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Delivery {
    sequence: u64,
    from_agent_id: String,
    to_agent_id: String,
    evidence_atom_id: String,
    delivered: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct FinalRecord {
    status: String,
    cited_evidence_atoms: Vec<String>,
    answer_correct: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct Expected {
    disposition: String,
    decision_coverage_milli: u16,
    cited_decisive_milli: Option<u16>,
    provenance_violations: usize,
    duplicate_delivery_events: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Score {
    fixture_id: String,
    disposition: String,
    decision_coverage_milli: u16,
    cited_decisive_milli: Option<u16>,
    provenance_violations: usize,
    duplicate_delivery_events: usize,
    valid_delivery_events: usize,
    final_answer_correct: Option<bool>,
}

fn ratio_milli(numerator: usize, denominator: usize) -> u16 {
    if denominator == 0 {
        return 0;
    }
    ((numerator.saturating_mul(1_000) / denominator).min(1_000)) as u16
}

fn score_scenario(scenario: &Scenario) -> Score {
    let mut violations = 0usize;
    let mut duplicate_delivery_events = 0usize;
    let mut valid_delivery_events = 0usize;

    let mut catalog = HashMap::<String, bool>::new();
    for atom in &scenario.evidence_catalog {
        if catalog.insert(atom.atom_id.clone(), atom.decisive).is_some() {
            violations += 1;
        }
    }

    let required: HashSet<String> = scenario.required_decisive_atoms.iter().cloned().collect();
    if required.is_empty() || required.len() != scenario.required_decisive_atoms.len() {
        violations += 1;
    }
    for atom_id in &required {
        if catalog.get(atom_id) != Some(&true) {
            violations += 1;
        }
    }

    let mut known_agents = HashSet::<String>::new();
    let mut held = HashMap::<String, HashSet<String>>::new();
    for agent in &scenario.agents {
        if !known_agents.insert(agent.agent_id.clone()) {
            violations += 1;
        }
        let mut evidence = HashSet::new();
        for atom_id in &agent.initial_evidence_atoms {
            if !catalog.contains_key(atom_id) {
                violations += 1;
            }
            if !evidence.insert(atom_id.clone()) {
                violations += 1;
            }
        }
        held.insert(agent.agent_id.clone(), evidence);
    }

    if !known_agents.contains(&scenario.decision_agent_id) {
        violations += 1;
    }

    let mut previous_sequence: Option<u64> = None;
    for event in &scenario.deliveries {
        let sequence_valid = previous_sequence.is_none_or(|previous| event.sequence > previous);
        previous_sequence = Some(event.sequence);

        if !sequence_valid {
            violations += 1;
            continue;
        }

        let sender_exists = known_agents.contains(&event.from_agent_id);
        let recipient_exists = known_agents.contains(&event.to_agent_id);
        let atom_exists = catalog.contains_key(&event.evidence_atom_id);
        if !sender_exists || !recipient_exists || !atom_exists {
            violations += 1;
            continue;
        }
        if event.from_agent_id == event.to_agent_id {
            violations += 1;
            continue;
        }

        let sender_holds = held
            .get(&event.from_agent_id)
            .is_some_and(|evidence| evidence.contains(&event.evidence_atom_id));
        if !sender_holds {
            // A participant cannot create provenance by asserting that it sent
            // a fact it did not initially hold or previously receive.
            violations += 1;
            continue;
        }

        if !event.delivered {
            // Attempted but failed delivery is recorded, but changes no state.
            continue;
        }

        let Some(recipient_evidence) = held.get_mut(&event.to_agent_id) else {
            violations += 1;
            continue;
        };
        if recipient_evidence.insert(event.evidence_atom_id.clone()) {
            valid_delivery_events += 1;
        } else {
            // Duplicate delivery is measurable but never increases coverage.
            duplicate_delivery_events += 1;
        }
    }

    let decision_evidence = held
        .get(&scenario.decision_agent_id)
        .cloned()
        .unwrap_or_default();
    let decision_coverage = required
        .iter()
        .filter(|atom_id| decision_evidence.contains(*atom_id))
        .count();
    let decision_coverage_milli = ratio_milli(decision_coverage, required.len());

    let mut cited_decisive_milli = None;
    let mut final_answer_correct = None;
    if let Some(final_record) = &scenario.final_record {
        final_answer_correct = final_record.answer_correct;

        match final_record.status.as_str() {
            "complete" if final_record.answer_correct.is_none() => violations += 1,
            "abstain" | "indeterminate" if final_record.answer_correct.is_some() => {
                violations += 1
            }
            "complete" | "abstain" | "indeterminate" => {}
            _ => violations += 1,
        }

        let mut citations = HashSet::new();
        for atom_id in &final_record.cited_evidence_atoms {
            if !citations.insert(atom_id.clone()) {
                // Duplicate citations do not inflate the evidence-use metric.
                continue;
            }
            if !catalog.contains_key(atom_id) || !decision_evidence.contains(atom_id) {
                violations += 1;
            }
        }
        let cited_decisive = required
            .iter()
            .filter(|atom_id| citations.contains(*atom_id) && decision_evidence.contains(*atom_id))
            .count();
        cited_decisive_milli = Some(ratio_milli(cited_decisive, required.len()));
    }

    let disposition = if violations > 0 {
        "reject"
    } else if scenario.final_record.is_none() {
        "incomplete"
    } else {
        "qualified"
    };

    Score {
        fixture_id: scenario.id.clone(),
        disposition: disposition.to_string(),
        decision_coverage_milli,
        cited_decisive_milli,
        provenance_violations: violations,
        duplicate_delivery_events,
        valid_delivery_events,
        final_answer_correct,
    }
}

fn expected_matches(actual: &Score, expected: &Expected) -> bool {
    actual.disposition == expected.disposition
        && actual.decision_coverage_milli == expected.decision_coverage_milli
        && actual.cited_decisive_milli == expected.cited_decisive_milli
        && actual.provenance_violations == expected.provenance_violations
        && actual.duplicate_delivery_events == expected.duplicate_delivery_events
}

fn load_fixtures(path: &str) -> Result<FixtureSet, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read fixture file {path}: {error}"))?;
    serde_json::from_str(&raw).map_err(|error| format!("invalid fixture JSON: {error}"))
}

fn run(path: &str) -> Result<(), String> {
    let fixture_set = load_fixtures(path)?;
    if fixture_set.schema_version != 1 || fixture_set.protocol_id != "IG-HIDDEN-INFO-001" {
        return Err("unsupported fixture schema or protocol identity".to_string());
    }
    if fixture_set.fixtures.is_empty() {
        return Err("fixture set must not be empty".to_string());
    }

    let mut fixture_ids = HashSet::new();
    let mut qualified = 0usize;
    let mut incomplete = 0usize;
    let mut rejected = 0usize;
    let mut mismatches = 0usize;

    for scenario in &fixture_set.fixtures {
        if !fixture_ids.insert(scenario.id.clone()) {
            return Err(format!("duplicate fixture id: {}", scenario.id));
        }
        let score = score_scenario(scenario);
        match score.disposition.as_str() {
            "qualified" => qualified += 1,
            "incomplete" => incomplete += 1,
            _ => rejected += 1,
        }
        if !expected_matches(&score, &scenario.expected) {
            mismatches += 1;
            eprintln!(
                "EXPECTATION_MISMATCH {} expected={:?} actual={:?}",
                scenario.id, scenario.expected, score
            );
        }
        println!(
            "{}",
            serde_json::to_string(&score)
                .map_err(|error| format!("cannot serialize score: {error}"))?
        );
    }

    println!(
        "SUMMARY fixtures={} qualified={} incomplete={} rejected={} expectation_mismatches={}",
        fixture_set.fixtures.len(),
        qualified,
        incomplete,
        rejected,
        mismatches
    );
    if mismatches > 0 {
        return Err(format!("{mismatches} fixture expectation(s) did not match"));
    }
    Ok(())
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: ig-hidden-info-qualifier <HIDDEN_INFO_FIXTURES_V0_1.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one fixture path");
        std::process::exit(2);
    }
    if let Err(error) = run(&path) {
        eprintln!("qualification failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../../HIDDEN_INFO_FIXTURES_V0_1.json");

    fn fixture_set() -> FixtureSet {
        serde_json::from_str(FIXTURES).expect("checked-in fixtures must parse")
    }

    #[test]
    fn all_frozen_fixture_expectations_match() {
        let fixtures = fixture_set();
        assert_eq!(fixtures.schema_version, 1);
        assert_eq!(fixtures.protocol_id, "IG-HIDDEN-INFO-001");
        for scenario in &fixtures.fixtures {
            let actual = score_scenario(scenario);
            assert!(
                expected_matches(&actual, &scenario.expected),
                "fixture {} mismatch: actual={actual:?}, expected={:?}",
                scenario.id,
                scenario.expected
            );
        }
    }

    #[test]
    fn structural_qualification_does_not_mean_answer_correctness() {
        let fixtures = fixture_set();
        let no_communication = fixtures
            .fixtures
            .iter()
            .find(|fixture| fixture.id == "HIF-002-no-communication")
            .expect("no-communication fixture exists");
        let score = score_scenario(no_communication);
        assert_eq!(score.disposition, "qualified");
        assert_eq!(score.final_answer_correct, Some(false));
        assert_eq!(score.decision_coverage_milli, 333);
    }

    #[test]
    fn failed_delivery_does_not_transfer_evidence() {
        let fixtures = fixture_set();
        let scenario = fixtures
            .fixtures
            .iter()
            .find(|fixture| fixture.id == "HIF-005-failed-delivery-is-not-knowledge")
            .expect("failed-delivery fixture exists");
        let score = score_scenario(scenario);
        assert_eq!(score.decision_coverage_milli, 0);
        assert_eq!(score.valid_delivery_events, 0);
    }

    #[test]
    fn duplicate_delivery_does_not_inflate_coverage() {
        let fixtures = fixture_set();
        let scenario = fixtures
            .fixtures
            .iter()
            .find(|fixture| fixture.id == "HIF-004-duplicate-does-not-inflate")
            .expect("duplicate fixture exists");
        let score = score_scenario(scenario);
        assert_eq!(score.decision_coverage_milli, 1_000);
        assert_eq!(score.duplicate_delivery_events, 1);
        assert_eq!(score.valid_delivery_events, 1);
    }

    #[test]
    fn missing_post_measurement_is_incomplete_not_pass() {
        let fixtures = fixture_set();
        let scenario = fixtures
            .fixtures
            .iter()
            .find(|fixture| fixture.id == "HIF-009-missing-post-measurement")
            .expect("missing-measurement fixture exists");
        let score = score_scenario(scenario);
        assert_eq!(score.disposition, "incomplete");
        assert_eq!(score.cited_decisive_milli, None);
    }

    #[test]
    fn decision_agent_cannot_cite_undelivered_evidence() {
        let fixtures = fixture_set();
        let scenario = fixtures
            .fixtures
            .iter()
            .find(|fixture| fixture.id == "HIF-010-unsupported-citation")
            .expect("unsupported-citation fixture exists");
        let score = score_scenario(scenario);
        assert_eq!(score.disposition, "reject");
        assert_eq!(score.provenance_violations, 1);
        assert_eq!(score.decision_coverage_milli, 500);
    }

    #[test]
    fn a_correct_minority_fact_can_change_the_collective_answer() {
        let fixtures = fixture_set();
        let scenario = fixtures
            .fixtures
            .iter()
            .find(|fixture| fixture.id == "HIF-012-minority-held-correct-fact")
            .expect("minority fact fixture exists");
        let score = score_scenario(scenario);
        assert_eq!(score.disposition, "qualified");
        assert_eq!(score.final_answer_correct, Some(true));
        assert_eq!(score.decision_coverage_milli, 1_000);
    }
}

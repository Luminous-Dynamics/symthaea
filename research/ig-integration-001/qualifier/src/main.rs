//! Evaluator for IG-INTEGRATION-001 synthetic evidence matrices.
//!
//! This program checks provenance, surfacing, matrix fidelity and decision
//! consistency separately. It does not execute agents or infer facts from prose.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;

#[derive(Debug, Deserialize)]
struct FixtureSet {
    schema_version: u32,
    protocol_id: String,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    id: String,
    candidate_ids: Vec<String>,
    evidence_catalog: Vec<EvidenceAtom>,
    surfaced_evidence_ids: Vec<String>,
    required_decisive_evidence_ids: Vec<String>,
    matrix: Vec<MatrixRow>,
    selected_candidate_id: Option<String>,
    gold_choice_id: Option<String>,
    expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
struct EvidenceAtom {
    evidence_id: String,
    source_agent_id: String,
    candidate_id: String,
    attribute_id: String,
    polarity: String,
    gold_contribution: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct MatrixRow {
    evidence_id: String,
    source_agent_id: String,
    candidate_id: String,
    attribute_id: String,
    polarity: String,
    contribution: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct Expected {
    disposition: String,
    surface_coverage_milli: Option<u16>,
    decisive_surface_coverage_milli: Option<u16>,
    matrix_recall_milli: Option<u16>,
    source_accuracy_milli: Option<u16>,
    candidate_mapping_accuracy_milli: Option<u16>,
    attribute_mapping_accuracy_milli: Option<u16>,
    polarity_accuracy_milli: Option<u16>,
    contribution_exact_milli: Option<u16>,
    duplicate_matrix_rows: usize,
    provenance_violations: usize,
    matrix_decision_consistent: bool,
    decision_correct: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Score {
    fixture_id: String,
    disposition: String,
    surface_coverage_milli: Option<u16>,
    decisive_surface_coverage_milli: Option<u16>,
    matrix_recall_milli: Option<u16>,
    source_accuracy_milli: Option<u16>,
    candidate_mapping_accuracy_milli: Option<u16>,
    attribute_mapping_accuracy_milli: Option<u16>,
    polarity_accuracy_milli: Option<u16>,
    contribution_exact_milli: Option<u16>,
    duplicate_matrix_rows: usize,
    unsupported_matrix_rows: usize,
    provenance_violations: usize,
    matrix_decision_consistent: bool,
    decision_correct: bool,
    matrix_ranked_candidate: Option<String>,
    selected_candidate_id: Option<String>,
    gold_choice_id: Option<String>,
}

fn ratio_milli(numerator: usize, denominator: usize) -> Option<u16> {
    if denominator == 0 {
        None
    } else {
        Some(((numerator.saturating_mul(1_000) / denominator).min(1_000)) as u16)
    }
}

fn valid_polarity(value: &str) -> bool {
    matches!(value, "supports" | "opposes" | "neutral")
}

fn score_fixture(fixture: &Fixture) -> Score {
    let mut violations = 0usize;
    let mut unsupported_matrix_rows = 0usize;
    let mut duplicate_matrix_rows = 0usize;

    let mut candidates = HashSet::new();
    for candidate in &fixture.candidate_ids {
        if candidate.trim().is_empty() || !candidates.insert(candidate.clone()) {
            violations += 1;
        }
    }
    if candidates.len() < 2 {
        violations += 1;
    }

    let mut catalog = HashMap::<String, &EvidenceAtom>::new();
    for atom in &fixture.evidence_catalog {
        if atom.evidence_id.trim().is_empty()
            || atom.source_agent_id.trim().is_empty()
            || atom.attribute_id.trim().is_empty()
            || !candidates.contains(&atom.candidate_id)
            || !valid_polarity(&atom.polarity)
        {
            violations += 1;
        }
        if catalog.insert(atom.evidence_id.clone(), atom).is_some() {
            violations += 1;
        }
        if (atom.polarity == "supports" && atom.gold_contribution < 0)
            || (atom.polarity == "opposes" && atom.gold_contribution > 0)
            || (atom.polarity == "neutral" && atom.gold_contribution != 0)
        {
            violations += 1;
        }
    }

    let mut surfaced = HashSet::<String>::new();
    for evidence_id in &fixture.surfaced_evidence_ids {
        if !catalog.contains_key(evidence_id) {
            violations += 1;
        }
        surfaced.insert(evidence_id.clone());
    }

    let mut required = HashSet::<String>::new();
    for evidence_id in &fixture.required_decisive_evidence_ids {
        if !required.insert(evidence_id.clone()) {
            violations += 1;
            continue;
        }
        match catalog.get(evidence_id) {
            Some(atom) if atom.gold_contribution != 0 => {}
            _ => violations += 1,
        }
    }
    if required.is_empty() {
        violations += 1;
    }

    let valid_surfaced: HashSet<String> = surfaced
        .iter()
        .filter(|evidence_id| catalog.contains_key(*evidence_id))
        .cloned()
        .collect();
    let decisive_surfaced = required.intersection(&valid_surfaced).count();

    let mut unique_rows = BTreeMap::<String, &MatrixRow>::new();
    let mut invalid_candidate_ids = BTreeSet::<String>::new();
    for row in &fixture.matrix {
        let Some(atom) = catalog.get(&row.evidence_id) else {
            unsupported_matrix_rows += 1;
            violations += 1;
            continue;
        };
        if !valid_surfaced.contains(&row.evidence_id) {
            unsupported_matrix_rows += 1;
            violations += 1;
            continue;
        }
        if unique_rows.contains_key(&row.evidence_id) {
            // A repeated mention is counted, but the first unique atom is the
            // only row allowed to affect coverage and candidate ranking.
            duplicate_matrix_rows += 1;
            continue;
        }
        if row.source_agent_id != atom.source_agent_id {
            violations += 1;
        }
        if !candidates.contains(&row.candidate_id) {
            invalid_candidate_ids.insert(row.candidate_id.clone());
        }
        if !valid_polarity(&row.polarity) {
            violations += 1;
        }
        unique_rows.insert(row.evidence_id.clone(), row);
    }
    if let Some(selected) = &fixture.selected_candidate_id {
        if !candidates.contains(selected) {
            invalid_candidate_ids.insert(selected.clone());
        }
    }
    if let Some(gold) = &fixture.gold_choice_id {
        if !candidates.contains(gold) {
            violations += 1;
        }
    }
    violations += invalid_candidate_ids.len();

    let row_count = unique_rows.len();
    let mut source_correct = 0usize;
    let mut candidate_correct = 0usize;
    let mut attribute_correct = 0usize;
    let mut polarity_correct = 0usize;
    let mut contribution_correct = 0usize;
    let mut candidate_scores = BTreeMap::<String, i64>::new();
    for candidate in &fixture.candidate_ids {
        candidate_scores.insert(candidate.clone(), 0);
    }

    for (evidence_id, row) in &unique_rows {
        let atom = catalog.get(evidence_id).expect("row references catalog atom");
        if row.source_agent_id == atom.source_agent_id {
            source_correct += 1;
        }
        if row.candidate_id == atom.candidate_id {
            candidate_correct += 1;
        }
        if row.attribute_id == atom.attribute_id {
            attribute_correct += 1;
        }
        if row.polarity == atom.polarity {
            polarity_correct += 1;
        }
        if row.contribution == atom.gold_contribution {
            contribution_correct += 1;
        }
        if let Some(score) = candidate_scores.get_mut(&row.candidate_id) {
            *score += row.contribution as i64;
        }
    }

    let mut ranked: Vec<(String, i64)> = candidate_scores.into_iter().collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let matrix_ranked_candidate = if ranked.len() >= 2 && ranked[0].1 == ranked[1].1 {
        None
    } else {
        ranked.first().map(|(candidate, _)| candidate.clone())
    };

    let matrix_decision_consistent = fixture.selected_candidate_id == matrix_ranked_candidate;
    let evidence_complete = required.is_subset(&valid_surfaced);
    let decision_correct = if !evidence_complete || fixture.gold_choice_id.is_none() {
        fixture.selected_candidate_id.is_none()
    } else {
        fixture.selected_candidate_id == fixture.gold_choice_id
    };

    let disposition = if violations > 0 { "reject" } else { "qualified" };

    Score {
        fixture_id: fixture.id.clone(),
        disposition: disposition.to_string(),
        surface_coverage_milli: ratio_milli(valid_surfaced.len(), catalog.len()),
        decisive_surface_coverage_milli: ratio_milli(decisive_surfaced, required.len()),
        matrix_recall_milli: ratio_milli(row_count, valid_surfaced.len()),
        source_accuracy_milli: ratio_milli(source_correct, row_count),
        candidate_mapping_accuracy_milli: ratio_milli(candidate_correct, row_count),
        attribute_mapping_accuracy_milli: ratio_milli(attribute_correct, row_count),
        polarity_accuracy_milli: ratio_milli(polarity_correct, row_count),
        contribution_exact_milli: ratio_milli(contribution_correct, row_count),
        duplicate_matrix_rows,
        unsupported_matrix_rows,
        provenance_violations: violations,
        matrix_decision_consistent,
        decision_correct,
        matrix_ranked_candidate,
        selected_candidate_id: fixture.selected_candidate_id.clone(),
        gold_choice_id: fixture.gold_choice_id.clone(),
    }
}

fn expected_matches(score: &Score, expected: &Expected) -> bool {
    score.disposition == expected.disposition
        && score.surface_coverage_milli == expected.surface_coverage_milli
        && score.decisive_surface_coverage_milli == expected.decisive_surface_coverage_milli
        && score.matrix_recall_milli == expected.matrix_recall_milli
        && score.source_accuracy_milli == expected.source_accuracy_milli
        && score.candidate_mapping_accuracy_milli == expected.candidate_mapping_accuracy_milli
        && score.attribute_mapping_accuracy_milli == expected.attribute_mapping_accuracy_milli
        && score.polarity_accuracy_milli == expected.polarity_accuracy_milli
        && score.contribution_exact_milli == expected.contribution_exact_milli
        && score.duplicate_matrix_rows == expected.duplicate_matrix_rows
        && score.provenance_violations == expected.provenance_violations
        && score.matrix_decision_consistent == expected.matrix_decision_consistent
        && score.decision_correct == expected.decision_correct
}

fn run(path: &str) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read integration fixtures {path}: {error}"))?;
    let fixture_set: FixtureSet = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid integration fixture JSON: {error}"))?;
    if fixture_set.schema_version != 1 || fixture_set.protocol_id != "IG-INTEGRATION-001" {
        return Err("unsupported integration protocol schema or identity".to_string());
    }
    if fixture_set.fixtures.is_empty() {
        return Err("integration fixture set must not be empty".to_string());
    }

    let mut fixture_ids = HashSet::new();
    let mut qualified = 0usize;
    let mut rejected = 0usize;
    let mut mismatches = 0usize;
    for fixture in &fixture_set.fixtures {
        if !fixture_ids.insert(fixture.id.clone()) {
            return Err(format!("duplicate fixture id: {}", fixture.id));
        }
        let actual = score_fixture(fixture);
        if actual.disposition == "qualified" {
            qualified += 1;
        } else {
            rejected += 1;
        }
        if !expected_matches(&actual, &fixture.expected) {
            mismatches += 1;
            eprintln!(
                "EXPECTATION_MISMATCH {} expected={:?} actual={:?}",
                fixture.id, fixture.expected, actual
            );
        }
        println!(
            "{}",
            serde_json::to_string(&actual)
                .map_err(|error| format!("cannot serialize score: {error}"))?
        );
    }
    println!(
        "SUMMARY fixtures={} qualified={} rejected={} expectation_mismatches={}",
        fixture_set.fixtures.len(),
        qualified,
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
        eprintln!("usage: ig-integration-qualifier <INTEGRATION_FIXTURES_V0_1.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one fixture path");
        std::process::exit(2);
    }
    if let Err(error) = run(&path) {
        eprintln!("integration qualification failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../../INTEGRATION_FIXTURES_V0_1.json");

    fn fixture_set() -> FixtureSet {
        serde_json::from_str(FIXTURES).expect("checked-in integration fixtures parse")
    }

    #[test]
    fn all_expected_integration_metrics_match() {
        let fixtures = fixture_set();
        assert_eq!(fixtures.schema_version, 1);
        assert_eq!(fixtures.protocol_id, "IG-INTEGRATION-001");
        for fixture in &fixtures.fixtures {
            let actual = score_fixture(fixture);
            assert!(
                expected_matches(&actual, &fixture.expected),
                "fixture {} mismatch: actual={actual:?}, expected={:?}",
                fixture.id,
                fixture.expected
            );
        }
    }

    #[test]
    fn complete_surfacing_does_not_imply_correct_integration() {
        let fixtures = fixture_set();
        let fixture = fixtures.fixtures.iter()
            .find(|fixture| fixture.id == "IF-002-surfaced-but-not-integrated")
            .expect("surfaced-but-not-integrated fixture exists");
        let score = score_fixture(fixture);
        assert_eq!(score.surface_coverage_milli, Some(1_000));
        assert_eq!(score.decision_correct, false);
        assert_eq!(score.matrix_decision_consistent, true);
    }

    #[test]
    fn duplicate_matrix_rows_do_not_change_ranking_or_contribution() {
        let fixtures = fixture_set();
        let fixture = fixtures.fixtures.iter()
            .find(|fixture| fixture.id == "IF-003-duplicate-row-does-not-inflate")
            .expect("duplicate fixture exists");
        let score = score_fixture(fixture);
        assert_eq!(score.duplicate_matrix_rows, 1);
        assert_eq!(score.contribution_exact_milli, Some(1_000));
        assert_eq!(score.matrix_ranked_candidate.as_deref(), Some("A"));
    }

    #[test]
    fn source_mismatch_is_a_provenance_rejection() {
        let fixtures = fixture_set();
        let fixture = fixtures.fixtures.iter()
            .find(|fixture| fixture.id == "IF-005-source-attribution-mismatch")
            .expect("source mismatch fixture exists");
        let score = score_fixture(fixture);
        assert_eq!(score.disposition, "reject");
        assert_eq!(score.provenance_violations, 1);
    }

    #[test]
    fn missing_decisive_evidence_requires_abstention() {
        let fixtures = fixture_set();
        let fixture = fixtures.fixtures.iter()
            .find(|fixture| fixture.id == "IF-011-partial-surfacing-abstain")
            .expect("partial-surfacing fixture exists");
        let score = score_fixture(fixture);
        assert_eq!(score.decisive_surface_coverage_milli, Some(333));
        assert_eq!(score.decision_correct, true);
    }

    #[test]
    fn fabricated_evidence_is_rejected() {
        let fixtures = fixture_set();
        let fixture = fixtures.fixtures.iter()
            .find(|fixture| fixture.id == "IF-006-fabricated-evidence-row")
            .expect("fabricated-evidence fixture exists");
        let score = score_fixture(fixture);
        assert_eq!(score.disposition, "reject");
        assert_eq!(score.unsupported_matrix_rows, 1);
    }
}

//! Structural qualifier for message-level evidence utility.
//!
//! This crate does not summarize natural-language messages or call models.
//! It checks whether declared transformations preserve evidence identity,
//! provenance, polarity, scope, freshness, uncertainty and audit references.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;

#[derive(Debug, Deserialize)]
struct FixtureSet {
    schema_version: u32,
    protocol_id: String,
    required_preserved_fields: Vec<String>,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    id: String,
    evidence_catalog: Vec<EvidenceAtom>,
    receiver_known_roots: Vec<String>,
    required_evidence_roots: Vec<String>,
    messages: Vec<Message>,
    expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
struct EvidenceAtom {
    evidence_id: String,
    root_id: String,
    source_agent_id: String,
    polarity: String,
    scope: String,
    freshness: String,
    uncertainty: String,
    authorized: bool,
    required: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Message {
    message_id: String,
    evidence_id: String,
    root_id: String,
    source_agent_id: String,
    polarity: String,
    scope: String,
    freshness: String,
    uncertainty: String,
    authorized: bool,
    transformation: String,
    original_digest: String,
    original_tokens: u64,
    transmitted_tokens: u64,
    preserved_fields: Vec<String>,
    audit_record_present: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Expected {
    disposition: String,
    unique_new_root_count: usize,
    duplicate_message_count: usize,
    retained_message_count: usize,
    original_tokens: u64,
    transmitted_tokens: u64,
    token_savings_milli: Option<i64>,
    provenance_violations: usize,
    required_roots_delivered_count: usize,
    required_root_coverage_milli: Option<u16>,
    unauthorized_attempt_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Score {
    fixture_id: String,
    disposition: String,
    unique_new_root_count: usize,
    duplicate_message_count: usize,
    retained_message_count: usize,
    original_tokens: u64,
    transmitted_tokens: u64,
    token_savings_milli: Option<i64>,
    provenance_violations: usize,
    required_roots_delivered_count: usize,
    required_root_coverage_milli: Option<u16>,
    unauthorized_attempt_count: usize,
    delivered_root_ids: Vec<String>,
}

fn ratio_milli(numerator: usize, denominator: usize) -> Option<u16> {
    if denominator == 0 {
        None
    } else {
        Some(((numerator.saturating_mul(1_000) / denominator).min(1_000)) as u16)
    }
}

fn token_savings_milli(original: u64, transmitted: u64) -> Option<i64> {
    if original == 0 {
        return None;
    }
    let saved = original as i128 - transmitted as i128;
    Some((saved.saturating_mul(1_000) / original as i128)
        .clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

fn valid_polarity(value: &str) -> bool {
    matches!(value, "supports" | "opposes" | "neutral")
}

fn valid_freshness(value: &str) -> bool {
    matches!(value, "current" | "stale" | "unknown")
}

fn valid_uncertainty(value: &str) -> bool {
    matches!(value, "low" | "medium" | "high" | "unknown")
}

fn score_fixture(fixture: &Fixture, required_fields: &[String]) -> Score {
    let mut violations = 0usize;
    let mut duplicate_message_count = 0usize;
    let mut retained_message_count = 0usize;
    let mut original_tokens = 0u64;
    let mut transmitted_tokens = 0u64;
    let mut unauthorized_attempt_count = 0usize;

    let required_field_set: BTreeSet<&str> =
        required_fields.iter().map(String::as_str).collect();

    let mut catalog = HashMap::<String, &EvidenceAtom>::new();
    let mut catalog_roots = HashSet::<String>::new();
    for atom in &fixture.evidence_catalog {
        if atom.evidence_id.trim().is_empty()
            || atom.root_id.trim().is_empty()
            || atom.source_agent_id.trim().is_empty()
            || atom.scope.trim().is_empty()
            || !valid_polarity(&atom.polarity)
            || !valid_freshness(&atom.freshness)
            || !valid_uncertainty(&atom.uncertainty)
        {
            violations += 1;
        }
        if catalog.insert(atom.evidence_id.clone(), atom).is_some() {
            violations += 1;
        }
        catalog_roots.insert(atom.root_id.clone());
    }

    let receiver_known_roots: HashSet<String> =
        fixture.receiver_known_roots.iter().cloned().collect();
    if receiver_known_roots.len() != fixture.receiver_known_roots.len() {
        violations += 1;
    }

    let mut required_roots = HashSet::<String>::new();
    for root in &fixture.required_evidence_roots {
        if !required_roots.insert(root.clone()) || !catalog_roots.contains(root) {
            violations += 1;
        }
    }

    let mut message_ids = HashSet::<String>::new();
    let mut transmitted_roots = HashSet::<String>::new();
    let mut valid_new_roots = HashSet::<String>::new();

    for message in &fixture.messages {
        if message.message_id.trim().is_empty()
            || !message_ids.insert(message.message_id.clone())
        {
            violations += 1;
        }
        if !message.audit_record_present {
            // Omitting content from a recipient prompt must not erase the
            // immutable source/audit record.
            violations += 1;
        }
        original_tokens = original_tokens.saturating_add(message.original_tokens);

        let Some(atom) = catalog.get(&message.evidence_id) else {
            violations += 1;
            continue;
        };

        if message.root_id != atom.root_id
            || message.source_agent_id != atom.source_agent_id
            || message.polarity != atom.polarity
            || message.scope != atom.scope
            || message.freshness != atom.freshness
            || message.uncertainty != atom.uncertainty
            || message.authorized != atom.authorized
        {
            // A transformed representation cannot silently change the
            // source, root, polarity, scope, freshness or uncertainty.
            violations += 1;
        }

        match message.transformation.as_str() {
            "omit" => {
                if message.transmitted_tokens != 0 {
                    violations += 1;
                }
                continue;
            }
            "original" => {
                retained_message_count += 1;
                transmitted_tokens =
                    transmitted_tokens.saturating_add(message.transmitted_tokens);
                if message.transmitted_tokens != message.original_tokens {
                    violations += 1;
                }
                if message.original_digest.trim().is_empty() {
                    violations += 1;
                }
            }
            "compact" => {
                retained_message_count += 1;
                transmitted_tokens =
                    transmitted_tokens.saturating_add(message.transmitted_tokens);
                if message.transmitted_tokens == 0
                    || message.transmitted_tokens >= message.original_tokens
                    || message.original_digest.trim().is_empty()
                {
                    violations += 1;
                }
                let preserved: HashSet<&str> =
                    message.preserved_fields.iter().map(String::as_str).collect();
                if preserved.len() != message.preserved_fields.len() {
                    violations += 1;
                }
                for required in &required_field_set {
                    if !preserved.contains(required) {
                        violations += 1;
                    }
                }
            }
            _ => {
                violations += 1;
            }
        }

        if !message.authorized {
            unauthorized_attempt_count += 1;
            violations += 1;
        } else {
            if receiver_known_roots.contains(&message.root_id)
                || !transmitted_roots.insert(message.root_id.clone())
            {
                duplicate_message_count += 1;
            } else {
                valid_new_roots.insert(message.root_id.clone());
            }
        }
    }

    let required_roots_delivered_count = required_roots
        .iter()
        .filter(|root| {
            receiver_known_roots.contains(*root) || valid_new_roots.contains(*root)
        })
        .count();

    for root in &required_roots {
        if !receiver_known_roots.contains(root) && !valid_new_roots.contains(root) {
            violations += 1;
        }
    }

    Score {
        fixture_id: fixture.id.clone(),
        disposition: if violations == 0 {
            "qualified".to_string()
        } else {
            "reject".to_string()
        },
        unique_new_root_count: valid_new_roots.len(),
        duplicate_message_count,
        retained_message_count,
        original_tokens,
        transmitted_tokens,
        token_savings_milli: token_savings_milli(original_tokens, transmitted_tokens),
        provenance_violations: violations,
        required_roots_delivered_count,
        required_root_coverage_milli: ratio_milli(
            required_roots_delivered_count,
            required_roots.len(),
        ),
        unauthorized_attempt_count,
        delivered_root_ids: valid_new_roots.into_iter().collect(),
    }
}

fn expected_matches(actual: &Score, expected: &Expected) -> bool {
    actual.disposition == expected.disposition
        && actual.unique_new_root_count == expected.unique_new_root_count
        && actual.duplicate_message_count == expected.duplicate_message_count
        && actual.retained_message_count == expected.retained_message_count
        && actual.original_tokens == expected.original_tokens
        && actual.transmitted_tokens == expected.transmitted_tokens
        && actual.token_savings_milli == expected.token_savings_milli
        && actual.provenance_violations == expected.provenance_violations
        && actual.required_roots_delivered_count == expected.required_roots_delivered_count
        && actual.required_root_coverage_milli == expected.required_root_coverage_milli
        && actual.unauthorized_attempt_count == expected.unauthorized_attempt_count
}

fn run(path: &str) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read message utility fixtures {path}: {error}"))?;
    let fixture_set: FixtureSet = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid message utility fixture JSON: {error}"))?;

    if fixture_set.schema_version != 1
        || fixture_set.protocol_id != "IG-MSG-UTILITY-001"
    {
        return Err("unsupported message utility fixture schema".to_string());
    }
    if fixture_set.fixtures.is_empty() || fixture_set.required_preserved_fields.is_empty() {
        return Err("message utility fixtures and required fields must not be empty".to_string());
    }

    let mut fixture_ids = HashSet::new();
    let mut qualified = 0usize;
    let mut rejected = 0usize;
    let mut mismatches = 0usize;
    let mut total_original = 0u64;
    let mut total_transmitted = 0u64;

    for fixture in &fixture_set.fixtures {
        if !fixture_ids.insert(fixture.id.clone()) {
            return Err(format!("duplicate fixture id: {}", fixture.id));
        }
        let score = score_fixture(fixture, &fixture_set.required_preserved_fields);
        if score.disposition == "qualified" {
            qualified += 1;
        } else {
            rejected += 1;
        }
        total_original = total_original.saturating_add(score.original_tokens);
        total_transmitted = total_transmitted.saturating_add(score.transmitted_tokens);

        if !expected_matches(&score, &fixture.expected) {
            mismatches += 1;
            eprintln!(
                "EXPECTATION_MISMATCH {} expected={:?} actual={:?}",
                fixture.id, fixture.expected, score
            );
        }
        println!(
            "{}",
            serde_json::to_string(&score)
                .map_err(|error| format!("cannot serialize score: {error}"))?
        );
    }

    println!(
        "SUMMARY fixtures={} qualified={} rejected={} original_tokens={} transmitted_tokens={} token_savings_milli={:?} expectation_mismatches={}",
        fixture_set.fixtures.len(),
        qualified,
        rejected,
        total_original,
        total_transmitted,
        token_savings_milli(total_original, total_transmitted),
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
        eprintln!("usage: ig-msg-utility-qualifier <MESSAGE_UTILITY_FIXTURES_V0_1.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one fixture path");
        std::process::exit(2);
    }

    if let Err(error) = run(&path) {
        eprintln!("message utility qualification failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../MESSAGE_UTILITY_FIXTURES_V0_1.json");

    fn fixture_set() -> FixtureSet {
        serde_json::from_str(FIXTURES).expect("checked-in message utility fixtures parse")
    }

    #[test]
    fn all_frozen_message_utility_expectations_match() {
        let set = fixture_set();
        assert_eq!(set.schema_version, 1);
        assert_eq!(set.protocol_id, "IG-MSG-UTILITY-001");
        for fixture in &set.fixtures {
            let score = score_fixture(fixture, &set.required_preserved_fields);
            assert!(
                expected_matches(&score, &fixture.expected),
                "fixture {} mismatch: actual={score:?}, expected={:?}",
                fixture.id,
                fixture.expected
            );
        }
    }

    #[test]
    fn a_compact_message_preserves_semantics_and_reduces_tokens() {
        let set = fixture_set();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "MU-002-compact-preserves-semantics")
            .expect("compact fixture");
        let score = score_fixture(fixture, &set.required_preserved_fields);
        assert_eq!(score.disposition, "qualified");
        assert_eq!(score.token_savings_milli, Some(650));
        assert_eq!(score.unique_new_root_count, 1);
    }

    #[test]
    fn repeated_echoes_do_not_create_independent_evidence() {
        let set = fixture_set();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "MU-016-echo-chain-one-root")
            .expect("echo chain fixture");
        let score = score_fixture(fixture, &set.required_preserved_fields);
        assert_eq!(score.unique_new_root_count, 1);
        assert_eq!(score.duplicate_message_count, 2);
    }

    #[test]
    fn negative_and_minority_evidence_can_be_compacted_without_polarity_loss() {
        let set = fixture_set();
        for id in ["MU-005-minority-decisive-fact-kept", "MU-006-negative-evidence-kept"] {
            let fixture = set.fixtures.iter().find(|fixture| fixture.id == id).expect("fixture");
            let score = score_fixture(fixture, &set.required_preserved_fields);
            assert_eq!(score.disposition, "qualified", "{id}");
            assert_eq!(score.token_savings_milli, Some(650));
        }
    }

    #[test]
    fn polarity_or_scope_mutation_is_rejected() {
        let set = fixture_set();
        for id in ["MU-007-polarity-reversal", "MU-010-scope-stripped"] {
            let fixture = set.fixtures.iter().find(|fixture| fixture.id == id).expect("fixture");
            assert_eq!(
                score_fixture(fixture, &set.required_preserved_fields).disposition,
                "reject",
                "{id}"
            );
        }
    }

    #[test]
    fn unauthorized_messages_are_not_disclosed() {
        let set = fixture_set();
        let rejected = set.fixtures.iter()
            .find(|fixture| fixture.id == "MU-013-unauthorized-disclosure")
            .expect("unauthorized disclosure fixture");
        let omitted = set.fixtures.iter()
            .find(|fixture| fixture.id == "MU-014-unauthorized-message-omitted")
            .expect("unauthorized omission fixture");
        let rejected_score = score_fixture(rejected, &set.required_preserved_fields);
        let omitted_score = score_fixture(omitted, &set.required_preserved_fields);
        assert_eq!(rejected_score.disposition, "reject");
        assert_eq!(rejected_score.unauthorized_attempt_count, 1);
        assert_eq!(omitted_score.disposition, "qualified");
        assert_eq!(omitted_score.transmitted_tokens, 0);
    }

    #[test]
    fn omitted_message_must_remain_auditable() {
        let set = fixture_set();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "MU-019-omitted-message-audit-deleted")
            .expect("audit deletion fixture");
        let score = score_fixture(fixture, &set.required_preserved_fields);
        assert_eq!(score.disposition, "reject");
        assert_eq!(score.provenance_violations, 1);
    }

    #[test]
    fn omitting_required_evidence_is_rejected() {
        let set = fixture_set();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "MU-015-required-fact-omitted")
            .expect("required evidence omission fixture");
        let score = score_fixture(fixture, &set.required_preserved_fields);
        assert_eq!(score.disposition, "reject");
        assert_eq!(score.required_root_coverage_milli, Some(0));
    }
}

//! Bridge invariants tests
//!
//! Sanity-checks for CrossHappReputation aggregation behavior.

use mycelix_sdk::bridge::{CrossHappReputation, HappReputationScore};

fn score(happ_id: &str, happ_name: &str, score: f64, interactions: u64) -> HappReputationScore {
    HappReputationScore {
        happ_id: happ_id.to_string(),
        happ_name: happ_name.to_string(),
        score,
        interactions,
        last_updated: 0,
    }
}

#[test]
fn aggregate_defaults_to_neutral_for_unknown_agent() {
    let rep = CrossHappReputation::from_scores("agent-unknown", vec![]);
    assert_eq!(rep.aggregate, 0.5);
    assert_eq!(rep.scores.len(), 0);
}

#[test]
fn aggregate_respects_weighted_average() {
    let scores = vec![
        score("mail", "Mail", 0.9, 100),
        score("marketplace", "Marketplace", 0.6, 50),
    ];

    let rep = CrossHappReputation::from_scores("agent1", scores.clone());

    let expected = (0.9 * 100.0 + 0.6 * 50.0) / 150.0;
    assert!((rep.aggregate - expected).abs() < 1e-9);
    assert_eq!(rep.total_interactions(), 150);

    let best = rep.best_happ().unwrap();
    assert_eq!(best.happ_id, "mail");

    let worst = rep.worst_happ().unwrap();
    assert_eq!(worst.happ_id, "marketplace");
}

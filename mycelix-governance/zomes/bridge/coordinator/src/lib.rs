//! Governance Bridge Coordinator Zome
//!
//! Cross-hApp communication for proposal queries, voting status,
//! and execution requests across the Mycelix ecosystem.
//!
//! ## Consciousness Metrics Integration
//!
//! This module provides coordinator functions for Symthaea consciousness metrics:
//! - `record_consciousness_snapshot`: Store Φ measurements from Symthaea
//! - `verify_consciousness_gate`: Check Φ threshold for governance actions
//! - `assess_value_alignment`: Evaluate proposal alignment with Eight Harmonies
//! - `get_agent_consciousness_history`: Track consciousness over time
//!
//! Updated to use HDK 0.6 patterns

#![allow(clippy::unnecessary_sort_by)]

use hdk::prelude::*;
use governance_bridge_integrity::*;

mod query;
mod consciousness;
mod attestation;
mod consensus;
mod cross_cluster;
mod consciousness_config;
mod validation;

pub use query::*;
pub use consciousness::*;
pub use attestation::*;
pub use consensus::*;
pub use cross_cluster::*;
pub use consciousness_config::*;
pub use validation::*;

const GOVERNANCE_HAPP_ID: &str = "mycelix-governance";
const SYMTHAEA_SOURCE: &str = "symthaea";

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum BridgeSignal {
    ConsciousnessSnapshotRecorded {
        agent_did: String,
        consciousness_level: f64,
    },
    ConsciousnessGateVerified {
        agent_did: String,
        passed: bool,
        action_type: String,
    },
    ValueAlignmentAssessed {
        proposal_id: String,
        agent_did: String,
        recommendation: String,
    },
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get the latest ConsciousnessAttestation for an agent
fn get_latest_agent_attestation(
    agent_did: &str,
) -> ExternResult<Option<(f64, Option<ConsciousnessVectorEntry>, Record)>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToAttestations)?,
        GetStrategy::default(),
    )?;

    let mut latest: Option<(f64, Option<ConsciousnessVectorEntry>, Record, Timestamp)> = None;

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(attestation) = record
                .entry()
                .to_app_option::<ConsciousnessAttestation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                let ts = record.action().timestamp();
                match &latest {
                    None => latest = Some((attestation.consciousness_level, attestation.consciousness_vector, record, ts)),
                    Some((_, _, _, prev_ts)) if ts > *prev_ts => {
                        latest = Some((attestation.consciousness_level, attestation.consciousness_vector, record, ts));
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(latest.map(|(consciousness_level, cv, record, _)| (consciousness_level, cv, record)))
}

/// Get the latest consciousness snapshot for an agent
fn get_latest_agent_snapshot(
    agent_did: &str,
) -> ExternResult<Option<(Record, ConsciousnessSnapshot)>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToSnapshots)?,
        GetStrategy::default(),
    )?;

    // Get all snapshots and find the most recent
    let mut latest: Option<(Record, ConsciousnessSnapshot, Timestamp)> = None;

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(snapshot) = record
                .entry()
                .to_app_option::<ConsciousnessSnapshot>()
                .ok()
                .flatten()
            {
                let ts = snapshot.captured_at;
                match &latest {
                    Some((_, _, latest_ts)) if ts > *latest_ts => {
                        latest = Some((record, snapshot, ts));
                    }
                    None => {
                        latest = Some((record, snapshot, ts));
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(latest.map(|(r, s, _)| (r, s)))
}

/// Check if a keyword appears in a negated context.
/// Scans a window of 5 words before the keyword for negation markers.
fn is_negated(content_lower: &str, keyword: &str) -> bool {
    const NEGATION_MARKERS: &[&str] = &[
        "not", "no", "never", "without", "lack", "against", "anti",
        "eliminate", "destroy", "end", "remove", "ban", "block", "prevent",
        "oppose", "reject", "deny", "undermine", "erode",
    ];

    // Find all occurrences of the keyword
    let mut start = 0;
    while let Some(pos) = content_lower[start..].find(keyword) {
        let abs_pos = start + pos;
        // Extract up to 60 chars before the keyword match as context window
        let window_start = abs_pos.saturating_sub(60);
        let window = &content_lower[window_start..abs_pos];
        let words: Vec<&str> = window.split_whitespace().collect();
        // Check last 5 words for negation
        let check_words = if words.len() > 5 { &words[words.len() - 5..] } else { &words };
        for w in check_words {
            if NEGATION_MARKERS.iter().any(|neg| w.contains(neg)) {
                return true;
            }
        }
        start = abs_pos + keyword.len();
    }
    false
}

/// Calculate harmony scores for proposal content.
///
/// Uses keyword matching with negation-awareness: keywords preceded by
/// negation markers ("not", "destroy", "without", etc.) within a 5-word
/// window are counted as negative rather than positive signals.
/// Includes Sacred Stillness (8th Harmony).
fn calculate_harmony_scores(content: &str) -> Vec<HarmonyScore> {
    let content_lower = content.to_lowercase();

    // Eight Harmonies with keyword-based scoring
    let harmonies = [
        ("Resonant Coherence", vec!["integration", "wholeness", "coherent", "unified", "harmony"]),
        ("Pan-Sentient Flourishing", vec!["flourishing", "wellbeing", "care", "compassion", "help", "nurture"]),
        ("Integral Wisdom", vec!["wisdom", "truth", "knowledge", "understanding", "verify", "discern"]),
        ("Infinite Play", vec!["creative", "play", "possibility", "experiment", "novel", "explore"]),
        ("Universal Interconnectedness", vec!["connection", "network", "relationship", "web", "community"]),
        ("Sacred Reciprocity", vec!["reciprocity", "balance", "exchange", "mutual", "fair", "equitable"]),
        ("Evolutionary Progression", vec!["evolution", "growth", "progress", "development", "adapt"]),
        ("Sacred Stillness", vec!["stillness", "rest", "contemplation", "reflection", "pause", "silence"]),
    ];

    harmonies
        .iter()
        .map(|(name, keywords)| {
            let mut positive_count = 0u32;
            let mut negated_count = 0u32;

            for kw in keywords {
                if content_lower.contains(*kw) {
                    if is_negated(&content_lower, kw) {
                        negated_count += 1;
                    } else {
                        positive_count += 1;
                    }
                }
            }

            // Each positive keyword adds 0.2, each negated keyword subtracts 0.15
            let score = (positive_count as f64 * 0.2) - (negated_count as f64 * 0.15);

            // Global negative indicators (apply uniformly to all harmonies)
            let negative_keywords = ["harm", "damage", "destroy", "exclude", "discriminate"];
            let negative_count = negative_keywords
                .iter()
                .filter(|kw| {
                    // Only count if not itself negated (e.g., "prevent harm" shouldn't penalize)
                    content_lower.contains(**kw) && !is_negated(&content_lower, kw)
                })
                .count();

            let final_score = score - (negative_count as f64 * 0.3);

            HarmonyScore {
                harmony: name.to_string(),
                score: final_score.clamp(-1.0, 1.0),
            }
        })
        .collect()
}

/// Determine governance recommendation based on alignment and authenticity
fn determine_recommendation(
    alignment: f64,
    authenticity: f64,
    violations: &[String],
) -> GovernanceRecommendation {
    // Strong oppose if violations detected
    if !violations.is_empty() {
        return GovernanceRecommendation::StrongOppose;
    }

    // Cannot evaluate if authenticity is too low
    if authenticity < 0.2 {
        return GovernanceRecommendation::CannotEvaluate;
    }

    // Weighted score: 60% alignment, 40% authenticity
    let combined = alignment * 0.6 + authenticity * 0.4;

    match combined {
        c if c > 0.7 => GovernanceRecommendation::StrongSupport,
        c if c > 0.3 => GovernanceRecommendation::Support,
        c if c > -0.3 => GovernanceRecommendation::Neutral,
        c if c > -0.7 => GovernanceRecommendation::Oppose,
        _ => GovernanceRecommendation::StrongOppose,
    }
}

// =============================================================================
// ADDITIONAL HELPER FUNCTIONS
// =============================================================================

/// Get the consensus participant for an agent
fn get_agent_participant(agent_did: &str) -> ExternResult<Option<ConsensusParticipant>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToParticipant)?,
        GetStrategy::default(),
    )?;

    // Get the most recent participant record
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            return record
                .entry()
                .to_app_option::<ConsensusParticipant>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
        }
    }

    Ok(None)
}

/// Get the federated reputation for an agent
fn get_agent_federated_reputation(
    agent_did: &str,
) -> ExternResult<Option<(Record, FederatedReputation)>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToFederatedRep)?,
        GetStrategy::default(),
    )?;

    // Get the most recent federated reputation record
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            let fed_rep = record
                .entry()
                .to_app_option::<FederatedReputation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

            if let Some(fr) = fed_rep {
                return Ok(Some((record, fr)));
            }
        }
    }

    Ok(None)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- calculate_harmony_scores ---

    #[test]
    fn test_harmony_scores_empty_content() {
        let scores = calculate_harmony_scores("");
        assert_eq!(scores.len(), 8, "Should always return 8 harmony scores");
        for s in &scores {
            assert!((s.score - 0.0).abs() < 1e-10, "{} should be 0.0 for empty content", s.harmony);
        }
    }

    #[test]
    fn test_harmony_scores_flourishing_keywords() {
        let scores = calculate_harmony_scores("We care about flourishing and wellbeing of all beings");
        let flourishing = scores.iter().find(|s| s.harmony == "Pan-Sentient Flourishing").unwrap();
        assert!(flourishing.score > 0.0, "Should detect flourishing keywords");
        // "flourishing", "wellbeing", "care" = 3 keywords × 0.2 = 0.6
        assert!((flourishing.score - 0.6).abs() < 1e-10, "Expected 0.6, got {}", flourishing.score);
    }

    #[test]
    fn test_harmony_scores_max_capped_at_1() {
        let scores = calculate_harmony_scores(
            "integration wholeness coherent unified resonance harmony"
        );
        let coherence = scores.iter().find(|s| s.harmony == "Resonant Coherence").unwrap();
        // "integration", "wholeness", "coherent", "unified" = 4 keywords × 0.2 = 0.8
        assert!(coherence.score <= 1.0, "Score should be capped at 1.0");
    }

    #[test]
    fn test_harmony_scores_negative_indicators() {
        let scores = calculate_harmony_scores("We will destroy and harm and exclude others");
        // Every harmony gets negative score from "destroy", "harm", "exclude" = -0.9
        for s in &scores {
            assert!(s.score < 0.0, "{} should be negative with harmful content, got {}", s.harmony, s.score);
        }
    }

    #[test]
    fn test_harmony_scores_case_insensitive() {
        let scores = calculate_harmony_scores("WISDOM and TRUTH for all");
        let wisdom = scores.iter().find(|s| s.harmony == "Integral Wisdom").unwrap();
        assert!(wisdom.score > 0.0, "Should match uppercase keywords");
    }

    #[test]
    fn test_harmony_scores_mixed_positive_negative() {
        let scores = calculate_harmony_scores("We seek wisdom and truth but may harm in the process");
        let wisdom = scores.iter().find(|s| s.harmony == "Integral Wisdom").unwrap();
        // "wisdom" + "truth" = 2×0.2 = 0.4, minus "harm" = -0.3 → net 0.1
        assert!((wisdom.score - 0.1).abs() < 1e-10, "Expected 0.1, got {}", wisdom.score);
    }

    // --- is_negated (negation-awareness) ---

    #[test]
    fn test_negated_keyword_simple() {
        assert!(is_negated("we will not seek justice", "justice"));
        assert!(is_negated("never pursue fairness", "fairness"));
        assert!(is_negated("without any care for others", "care"));
    }

    #[test]
    fn test_non_negated_keyword() {
        assert!(!is_negated("we pursue justice for all", "justice"));
        assert!(!is_negated("fairness is our goal", "fairness"));
        assert!(!is_negated("care and compassion matter", "care"));
    }

    #[test]
    fn test_negation_window_boundary() {
        // Negation more than 5 words away should NOT trigger
        assert!(!is_negated(
            "not at this particular moment in time do we seek justice",
            "justice"
        ));
        // Negation within 5 words SHOULD trigger
        assert!(is_negated("we do not seek justice", "justice"));
    }

    #[test]
    fn test_negated_destroy_in_content() {
        // "destroy justice" — "destroy" is a negation marker
        assert!(is_negated("we will destroy justice", "justice"));
    }

    #[test]
    fn test_negated_keyword_not_found() {
        // Keyword not in content → false (no crash)
        assert!(!is_negated("hello world", "justice"));
    }

    #[test]
    fn test_prevent_harm_is_negated() {
        // "prevent" is a negation marker → "harm" is negated
        assert!(is_negated("we will prevent harm to all beings", "harm"));
    }

    // --- negation-aware harmony scoring ---

    #[test]
    fn test_harmony_negated_keyword_reduces_score() {
        // "not wisdom" should NOT give positive score to Integral Wisdom
        let scores = calculate_harmony_scores("We have not wisdom nor truth");
        let wisdom = scores.iter().find(|s| s.harmony == "Integral Wisdom").unwrap();
        // "wisdom" negated (-0.15) + "truth" negated (-0.15) = -0.3
        assert!(wisdom.score < 0.0, "Negated keywords should produce negative score, got {}", wisdom.score);
    }

    #[test]
    fn test_harmony_destroy_all_justice_scores_negative() {
        // The original bug: "destroy all justice" scored positive on Sacred Reciprocity
        // because "fair" wasn't matched but other keywords were.
        // With negation: "destroy" is a negation marker for any following keyword.
        let scores = calculate_harmony_scores("We must destroy all fairness and balance");
        let reciprocity = scores.iter().find(|s| s.harmony == "Sacred Reciprocity").unwrap();
        // "fairness" doesn't match "fair" (substring), but "balance" matches
        // "destroy" negates "balance" → -0.15
        // Plus "destroy" is a global negative keyword (not negated) → -0.3
        assert!(reciprocity.score < 0.0,
            "Destroying fairness should score negative, got {}", reciprocity.score);
    }

    #[test]
    fn test_harmony_prevent_harm_not_penalized() {
        // "prevent harm" — "harm" is a global negative keyword, but it's negated by "prevent"
        let scores = calculate_harmony_scores("Our goal is to prevent harm and prevent damage");
        // All global negatives are negated → should NOT penalize
        for s in &scores {
            assert!(s.score >= 0.0,
                "{} should not be penalized by negated harm/damage, got {}", s.harmony, s.score);
        }
    }

    #[test]
    fn test_harmony_sacred_stillness_detected() {
        let scores = calculate_harmony_scores("We need stillness and contemplation for reflection");
        let stillness = scores.iter().find(|s| s.harmony == "Sacred Stillness").unwrap();
        // "stillness" + "contemplation" + "reflection" = 3 × 0.2 = 0.6
        assert!((stillness.score - 0.6).abs() < 1e-10,
            "Expected 0.6 for Sacred Stillness, got {}", stillness.score);
    }

    // --- determine_recommendation ---

    #[test]
    fn test_recommendation_strong_support() {
        let rec = determine_recommendation(0.9, 0.8, &[]);
        // combined = 0.9×0.6 + 0.8×0.4 = 0.54 + 0.32 = 0.86 > 0.7
        assert_eq!(rec, GovernanceRecommendation::StrongSupport);
    }

    #[test]
    fn test_recommendation_support() {
        let rec = determine_recommendation(0.5, 0.5, &[]);
        // combined = 0.5×0.6 + 0.5×0.4 = 0.3 + 0.2 = 0.5 (> 0.3, <= 0.7)
        assert_eq!(rec, GovernanceRecommendation::Support);
    }

    #[test]
    fn test_recommendation_neutral() {
        let rec = determine_recommendation(0.0, 0.5, &[]);
        // combined = 0.0 + 0.2 = 0.2 (> -0.3, <= 0.3)
        assert_eq!(rec, GovernanceRecommendation::Neutral);
    }

    #[test]
    fn test_recommendation_oppose() {
        // -0.6×0.6 + 0.5×0.4 = -0.16 → Neutral (not strong enough)
        let mild = determine_recommendation(-0.6, 0.5, &[]);
        assert_eq!(mild, GovernanceRecommendation::Neutral);

        // -0.8×0.6 + 0.3×0.4 = -0.36 → Oppose (> -0.7, <= -0.3)
        let strong = determine_recommendation(-0.8, 0.3, &[]);
        assert_eq!(strong, GovernanceRecommendation::Oppose);
    }

    #[test]
    fn test_recommendation_strong_oppose_with_violations() {
        let violations = vec!["Charter violation".to_string()];
        let rec = determine_recommendation(0.9, 0.9, &violations);
        assert_eq!(rec, GovernanceRecommendation::StrongOppose,
            "Any violations should result in StrongOppose regardless of scores");
    }

    #[test]
    fn test_recommendation_cannot_evaluate_low_authenticity() {
        let rec = determine_recommendation(0.9, 0.1, &[]);
        assert_eq!(rec, GovernanceRecommendation::CannotEvaluate,
            "Authenticity < 0.2 should be CannotEvaluate");
    }

    #[test]
    fn test_recommendation_boundary_authenticity() {
        // Exactly at 0.2 boundary — should NOT be CannotEvaluate (< 0.2 triggers it)
        let rec = determine_recommendation(0.5, 0.2, &[]);
        assert_ne!(rec, GovernanceRecommendation::CannotEvaluate,
            "Authenticity exactly 0.2 should be evaluated");
    }

    // --- check_snapshot_input ---

    fn valid_snapshot_input() -> RecordSnapshotInput {
        RecordSnapshotInput {
            consciousness_level: 0.7,
            meta_awareness: 0.6,
            self_model_accuracy: 0.8,
            coherence: 0.9,
            affective_valence: 0.3,
            care_activation: 0.5,
            source: None,
        }
    }

    #[test]
    fn test_snapshot_valid_input() {
        assert!(check_snapshot_input(&valid_snapshot_input()).is_ok());
    }

    #[test]
    fn test_snapshot_consciousness_out_of_range() {
        let mut input = valid_snapshot_input();
        input.consciousness_level = 1.1;
        assert!(check_snapshot_input(&input).is_err());
        input.consciousness_level = -0.1;
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_meta_awareness_out_of_range() {
        let mut input = valid_snapshot_input();
        input.meta_awareness = 1.5;
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_self_model_accuracy_out_of_range() {
        let mut input = valid_snapshot_input();
        input.self_model_accuracy = -0.01;
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_coherence_out_of_range() {
        let mut input = valid_snapshot_input();
        input.coherence = 2.0;
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_valence_out_of_range() {
        let mut input = valid_snapshot_input();
        input.affective_valence = -1.1;
        assert!(check_snapshot_input(&input).is_err());
        input.affective_valence = 1.1;
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_valence_valid_negative() {
        let mut input = valid_snapshot_input();
        input.affective_valence = -1.0;
        assert!(check_snapshot_input(&input).is_ok());
        input.affective_valence = -0.5;
        assert!(check_snapshot_input(&input).is_ok());
    }

    #[test]
    fn test_snapshot_care_activation_out_of_range() {
        let mut input = valid_snapshot_input();
        input.care_activation = 1.01;
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_source_empty() {
        let mut input = valid_snapshot_input();
        input.source = Some("".into());
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_source_too_long() {
        let mut input = valid_snapshot_input();
        input.source = Some("x".repeat(257));
        assert!(check_snapshot_input(&input).is_err());
    }

    #[test]
    fn test_snapshot_source_valid() {
        let mut input = valid_snapshot_input();
        input.source = Some("symthaea".into());
        assert!(check_snapshot_input(&input).is_ok());
    }

    #[test]
    fn test_snapshot_boundary_values() {
        // All at exact boundaries should pass
        let input = RecordSnapshotInput {
            consciousness_level: 0.0,
            meta_awareness: 1.0,
            self_model_accuracy: 0.0,
            coherence: 1.0,
            affective_valence: -1.0,
            care_activation: 0.0,
            source: None,
        };
        assert!(check_snapshot_input(&input).is_ok());
    }

    // --- check_broadcast_event_input ---

    #[test]
    fn test_broadcast_valid_input() {
        let input = BroadcastGovernanceEventInput {
            event_type: GovernanceEventType::ProposalCreated,
            proposal_id: Some("prop-123".into()),
            subject: "New proposal".into(),
            payload: "{}".into(),
        };
        assert!(check_broadcast_event_input(&input).is_ok());
    }

    #[test]
    fn test_broadcast_empty_subject() {
        let input = BroadcastGovernanceEventInput {
            event_type: GovernanceEventType::VoteReceived,
            proposal_id: None,
            subject: "".into(),
            payload: "{}".into(),
        };
        assert!(check_broadcast_event_input(&input).is_err());
    }

    #[test]
    fn test_broadcast_subject_too_long() {
        let input = BroadcastGovernanceEventInput {
            event_type: GovernanceEventType::VoteReceived,
            proposal_id: None,
            subject: "x".repeat(257),
            payload: "{}".into(),
        };
        assert!(check_broadcast_event_input(&input).is_err());
    }

    #[test]
    fn test_broadcast_payload_too_long() {
        let input = BroadcastGovernanceEventInput {
            event_type: GovernanceEventType::ProposalExecuted,
            proposal_id: None,
            subject: "test".into(),
            payload: "x".repeat(4097),
        };
        assert!(check_broadcast_event_input(&input).is_err());
    }

    #[test]
    fn test_broadcast_empty_proposal_id() {
        let input = BroadcastGovernanceEventInput {
            event_type: GovernanceEventType::ProposalCreated,
            proposal_id: Some("".into()),
            subject: "test".into(),
            payload: "{}".into(),
        };
        assert!(check_broadcast_event_input(&input).is_err());
    }

    #[test]
    fn test_broadcast_none_proposal_id_ok() {
        let input = BroadcastGovernanceEventInput {
            event_type: GovernanceEventType::ConstitutionAmended,
            proposal_id: None,
            subject: "test".into(),
            payload: "{}".into(),
        };
        assert!(check_broadcast_event_input(&input).is_ok());
    }

    // --- check_execution_request_input ---

    #[test]
    fn test_execution_valid_input() {
        let input = RequestExecutionInput {
            proposal_id: "prop-1".into(),
            target_happ: "finance".into(),
            action: "transfer".into(),
            parameters: serde_json::json!({}),
        };
        assert!(check_execution_request_input(&input).is_ok());
    }

    #[test]
    fn test_execution_empty_proposal_id() {
        let input = RequestExecutionInput {
            proposal_id: "".into(),
            target_happ: "finance".into(),
            action: "transfer".into(),
            parameters: serde_json::json!({}),
        };
        assert!(check_execution_request_input(&input).is_err());
    }

    #[test]
    fn test_execution_empty_target_happ() {
        let input = RequestExecutionInput {
            proposal_id: "prop-1".into(),
            target_happ: "".into(),
            action: "transfer".into(),
            parameters: serde_json::json!({}),
        };
        assert!(check_execution_request_input(&input).is_err());
    }

    #[test]
    fn test_execution_empty_action() {
        let input = RequestExecutionInput {
            proposal_id: "prop-1".into(),
            target_happ: "finance".into(),
            action: "".into(),
            parameters: serde_json::json!({}),
        };
        assert!(check_execution_request_input(&input).is_err());
    }

    // --- check_alignment_input ---

    #[test]
    fn test_alignment_valid_input() {
        let input = AssessAlignmentInput {
            proposal_id: "prop-1".into(),
            proposal_content: "A flourishing community project".into(),
        };
        assert!(check_alignment_input(&input).is_ok());
    }

    #[test]
    fn test_alignment_empty_proposal_id() {
        let input = AssessAlignmentInput {
            proposal_id: "".into(),
            proposal_content: "content".into(),
        };
        assert!(check_alignment_input(&input).is_err());
    }

    #[test]
    fn test_alignment_empty_content() {
        let input = AssessAlignmentInput {
            proposal_id: "prop-1".into(),
            proposal_content: "".into(),
        };
        assert!(check_alignment_input(&input).is_err());
    }

    #[test]
    fn test_alignment_content_too_long() {
        let input = AssessAlignmentInput {
            proposal_id: "prop-1".into(),
            proposal_content: "x".repeat(4097),
        };
        assert!(check_alignment_input(&input).is_err());
    }

    // --- check_weighted_vote_input ---

    #[test]
    fn test_weighted_vote_valid_input() {
        let input = CastWeightedVoteInput {
            proposal_id: "prop-1".into(),
            proposal_type: ProposalType::Standard,
            round: 1,
            decision: VoteDecision::Approve,
            harmonic_alignment: Some(0.8),
            reason: Some("Good proposal".into()),
        };
        assert!(check_weighted_vote_input(&input).is_ok());
    }

    #[test]
    fn test_weighted_vote_empty_proposal_id() {
        let input = CastWeightedVoteInput {
            proposal_id: "".into(),
            proposal_type: ProposalType::Standard,
            round: 1,
            decision: VoteDecision::Approve,
            harmonic_alignment: None,
            reason: None,
        };
        assert!(check_weighted_vote_input(&input).is_err());
    }

    #[test]
    fn test_weighted_vote_reason_too_long() {
        let input = CastWeightedVoteInput {
            proposal_id: "prop-1".into(),
            proposal_type: ProposalType::Constitutional,
            round: 1,
            decision: VoteDecision::Reject,
            harmonic_alignment: None,
            reason: Some("x".repeat(4097)),
        };
        assert!(check_weighted_vote_input(&input).is_err());
    }

    #[test]
    fn test_weighted_vote_none_reason_ok() {
        let input = CastWeightedVoteInput {
            proposal_id: "prop-1".into(),
            proposal_type: ProposalType::Emergency,
            round: 1,
            decision: VoteDecision::Abstain,
            harmonic_alignment: None,
            reason: None,
        };
        assert!(check_weighted_vote_input(&input).is_ok());
    }

    // --- check_transfer_credits_input ---

    #[test]
    fn test_transfer_valid_input() {
        let input = TransferCreditsInput {
            from: "treasury-main".into(),
            to: "recipient-1".into(),
            amount: 100.0,
        };
        assert!(check_transfer_credits_input(&input).is_ok());
    }

    #[test]
    fn test_transfer_empty_from() {
        let input = TransferCreditsInput {
            from: "".into(),
            to: "recipient-1".into(),
            amount: 100.0,
        };
        assert!(check_transfer_credits_input(&input).is_err());
    }

    #[test]
    fn test_transfer_empty_to() {
        let input = TransferCreditsInput {
            from: "treasury".into(),
            to: "".into(),
            amount: 100.0,
        };
        assert!(check_transfer_credits_input(&input).is_err());
    }

    #[test]
    fn test_transfer_zero_amount() {
        let input = TransferCreditsInput {
            from: "a".into(),
            to: "b".into(),
            amount: 0.0,
        };
        assert!(check_transfer_credits_input(&input).is_err());
    }

    #[test]
    fn test_transfer_negative_amount() {
        let input = TransferCreditsInput {
            from: "a".into(),
            to: "b".into(),
            amount: -50.0,
        };
        assert!(check_transfer_credits_input(&input).is_err());
    }

    #[test]
    fn test_transfer_infinity() {
        let input = TransferCreditsInput {
            from: "a".into(),
            to: "b".into(),
            amount: f64::INFINITY,
        };
        assert!(check_transfer_credits_input(&input).is_err());
    }

    #[test]
    fn test_transfer_nan() {
        let input = TransferCreditsInput {
            from: "a".into(),
            to: "b".into(),
            amount: f64::NAN,
        };
        assert!(check_transfer_credits_input(&input).is_err());
    }

    // --- check_federated_reputation_input ---

    #[test]
    fn test_federated_rep_empty_input_valid() {
        let input = UpdateFederatedReputationInput::default();
        assert!(check_federated_reputation_input(&input).is_ok());
    }

    #[test]
    fn test_federated_rep_valid_partial() {
        let input = UpdateFederatedReputationInput {
            identity_verification: Some(0.9),
            credential_count: Some(5),
            voting_participation: Some(0.7),
            ..Default::default()
        };
        assert!(check_federated_reputation_input(&input).is_ok());
    }

    #[test]
    fn test_federated_rep_identity_verification_out_of_range() {
        let input = UpdateFederatedReputationInput {
            identity_verification: Some(1.5),
            ..Default::default()
        };
        assert!(check_federated_reputation_input(&input).is_err());
    }

    #[test]
    fn test_federated_rep_negative_score() {
        let input = UpdateFederatedReputationInput {
            pogq_score: Some(-0.1),
            ..Default::default()
        };
        assert!(check_federated_reputation_input(&input).is_err());
    }

    #[test]
    fn test_federated_rep_all_at_boundary() {
        let input = UpdateFederatedReputationInput {
            identity_verification: Some(0.0),
            credential_quality: Some(1.0),
            epistemic_contributions: Some(0.0),
            factcheck_accuracy: Some(1.0),
            stake_weight: Some(0.0),
            payment_reliability: Some(1.0),
            escrow_completion_rate: Some(0.5),
            pogq_score: Some(0.0),
            byzantine_clean_rate: Some(1.0),
            voting_participation: Some(0.0),
            proposal_success_rate: Some(1.0),
            consensus_alignment: Some(0.5),
            ..Default::default()
        };
        assert!(check_federated_reputation_input(&input).is_ok());
    }

    #[test]
    fn test_federated_rep_byzantine_rate_too_high() {
        let input = UpdateFederatedReputationInput {
            byzantine_clean_rate: Some(1.001),
            ..Default::default()
        };
        assert!(check_federated_reputation_input(&input).is_err());
    }

    // --- GovernanceActionType thresholds ---

    #[test]
    fn test_action_type_thresholds() {
        assert!((GovernanceActionType::Basic.consciousness_gate() - 0.2).abs() < f64::EPSILON);
        assert!((GovernanceActionType::ProposalSubmission.consciousness_gate() - 0.3).abs() < f64::EPSILON);
        assert!((GovernanceActionType::Voting.consciousness_gate() - 0.4).abs() < f64::EPSILON);
        assert!((GovernanceActionType::Constitutional.consciousness_gate() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_action_type_ordering() {
        // Thresholds must be monotonically increasing
        assert!(GovernanceActionType::Basic.consciousness_gate()
            < GovernanceActionType::ProposalSubmission.consciousness_gate());
        assert!(GovernanceActionType::ProposalSubmission.consciousness_gate()
            < GovernanceActionType::Voting.consciousness_gate());
        assert!(GovernanceActionType::Voting.consciousness_gate()
            < GovernanceActionType::Constitutional.consciousness_gate());
    }

    // --- ConsciousnessSnapshot quality_score ---

    #[test]
    fn test_quality_score_perfect() {
        let snap = ConsciousnessSnapshot {
            id: "s1".into(),
            agent_did: "did:test:1".into(),
            consciousness_level: 1.0,
            meta_awareness: 1.0,
            self_model_accuracy: 1.0,
            coherence: 1.0,
            affective_valence: 1.0,
            care_activation: 1.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".into(),
            consciousness_vector: None,
        };
        // 1.0*0.4 + 1.0*0.2 + 1.0*0.2 + 1.0*0.2 = 1.0
        assert!((snap.quality_score() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quality_score_zero() {
        let snap = ConsciousnessSnapshot {
            id: "s2".into(),
            agent_did: "did:test:2".into(),
            consciousness_level: 0.0,
            meta_awareness: 0.0,
            self_model_accuracy: 0.0,
            coherence: 0.0,
            affective_valence: 0.0,
            care_activation: 0.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".into(),
            consciousness_vector: None,
        };
        assert!((snap.quality_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_quality_score_consciousness_dominates() {
        // Consciousness has 0.4 weight, all others 0.2
        let snap = ConsciousnessSnapshot {
            id: "s3".into(),
            agent_did: "did:test:3".into(),
            consciousness_level: 1.0,
            meta_awareness: 0.0,
            self_model_accuracy: 0.0,
            coherence: 0.0,
            affective_valence: 0.0,
            care_activation: 0.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".into(),
            consciousness_vector: None,
        };
        assert!((snap.quality_score() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_quality_score_does_not_include_valence_or_care() {
        // affective_valence and care_activation are NOT in quality_score
        let base = ConsciousnessSnapshot {
            id: "s4".into(),
            agent_did: "did:test:4".into(),
            consciousness_level: 0.5,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: 0.0,
            care_activation: 0.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".into(),
            consciousness_vector: None,
        };
        let with_extras = ConsciousnessSnapshot {
            affective_valence: 1.0,
            care_activation: 1.0,
            consciousness_vector: None,
            ..base.clone()
        };
        assert!((base.quality_score() - with_extras.quality_score()).abs() < 1e-10,
            "Valence and care should not affect quality score");
    }

    #[test]
    fn test_snapshot_meets_threshold() {
        let snap = ConsciousnessSnapshot {
            id: "s5".into(),
            agent_did: "did:test:5".into(),
            consciousness_level: 0.5,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: 0.0,
            care_activation: 0.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".into(),
            consciousness_vector: None,
        };
        assert!(snap.meets_threshold(&GovernanceActionType::Basic));       // 0.5 >= 0.2
        assert!(snap.meets_threshold(&GovernanceActionType::ProposalSubmission)); // 0.5 >= 0.3
        assert!(snap.meets_threshold(&GovernanceActionType::Voting));      // 0.5 >= 0.4
        assert!(!snap.meets_threshold(&GovernanceActionType::Constitutional)); // 0.5 < 0.6
    }

    // --- HolisticVotingWeight ---

    #[test]
    fn test_holistic_weight_perfect_inputs() {
        let w = HolisticVotingWeight::calculate(1.0, 1.0, 1.0);
        // 1.0² × (0.7 + 0.3×1.0) × (1 + 0.2×1.0) = 1.0 × 1.0 × 1.2 = 1.2
        assert!((w.final_weight - 1.2).abs() < 1e-10);
        assert!(!w.was_capped);
    }

    #[test]
    fn test_holistic_weight_zero_consciousness() {
        let w = HolisticVotingWeight::calculate(1.0, 0.0, 0.0);
        // 1.0² × (0.7 + 0.0) × 1.0 = 0.7
        assert!((w.final_weight - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_zero_reputation() {
        let w = HolisticVotingWeight::calculate(0.0, 1.0, 1.0);
        // 0.0² × anything = 0.0
        assert!((w.final_weight - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_negative_alignment_no_penalty() {
        let w_neg = HolisticVotingWeight::calculate(0.8, 0.5, -1.0);
        let w_zero = HolisticVotingWeight::calculate(0.8, 0.5, 0.0);
        // Negative alignment is clamped to 0 in bonus, so both should be equal
        assert!((w_neg.final_weight - w_zero.final_weight).abs() < 1e-10,
            "Negative alignment should give same weight as zero alignment");
    }

    #[test]
    fn test_holistic_weight_capping() {
        // With extreme reputation > 1.0 (gets clamped to 1.0), so can't exceed cap
        // But let's verify the cap constant is 1.5
        assert!((HolisticVotingWeight::max_weight() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_calculate_base() {
        let w = HolisticVotingWeight::calculate_base(0.8, 0.6);
        let w_manual = HolisticVotingWeight::calculate(0.8, 0.6, 0.0);
        assert!((w.final_weight - w_manual.final_weight).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_consciousness_multiplier_range() {
        // Phi = 0 → multiplier = 0.7
        let w0 = HolisticVotingWeight::calculate(1.0, 0.0, 0.0);
        assert!((w0.consciousness_multiplier - 0.7).abs() < 1e-10);

        // Phi = 1 → multiplier = 1.0
        let w1 = HolisticVotingWeight::calculate(1.0, 1.0, 0.0);
        assert!((w1.consciousness_multiplier - 1.0).abs() < 1e-10);

        // Phi = 0.5 → multiplier = 0.85
        let w5 = HolisticVotingWeight::calculate(1.0, 0.5, 0.0);
        assert!((w5.consciousness_multiplier - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_harmonic_bonus_range() {
        // alignment = 0 → bonus = 1.0
        let w0 = HolisticVotingWeight::calculate(1.0, 0.5, 0.0);
        assert!((w0.harmonic_bonus - 1.0).abs() < 1e-10);

        // alignment = 1.0 → bonus = 1.2
        let w1 = HolisticVotingWeight::calculate(1.0, 0.5, 1.0);
        assert!((w1.harmonic_bonus - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_mid_reputation() {
        let w = HolisticVotingWeight::calculate(0.5, 0.5, 0.0);
        // 0.5² × (0.7 + 0.3×0.5) × 1.0 = 0.25 × 0.85 × 1.0 = 0.2125
        assert!((w.final_weight - 0.2125).abs() < 1e-10);
    }

    #[test]
    fn test_holistic_weight_breakdown_not_empty() {
        let w = HolisticVotingWeight::calculate(0.8, 0.6, 0.3);
        assert!(!w.calculation_breakdown.is_empty());
        // Breakdown should contain the constant values
        assert!(w.calculation_breakdown.contains("0.7"), "Should reference consciousness base");
        assert!(w.calculation_breakdown.contains("0.3"), "Should reference phi factor");
    }

    // --- AdaptiveThreshold ---

    #[test]
    fn test_adaptive_threshold_standard() {
        let t = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);
        assert!((t.base_threshold - 0.51).abs() < 1e-10);
        assert!((t.min_voter_consciousness - 0.2).abs() < 1e-10);
        assert_eq!(t.min_participation, 5);
        assert!((t.quorum - 0.20).abs() < 1e-10);
        assert_eq!(t.max_extension_secs, 86400);
    }

    #[test]
    fn test_adaptive_threshold_emergency() {
        let t = AdaptiveThreshold::for_proposal_type(&ProposalType::Emergency);
        assert!((t.base_threshold - 0.60).abs() < 1e-10);
        assert!((t.min_voter_consciousness - 0.3).abs() < 1e-10);
        assert_eq!(t.min_participation, 3);
        assert!((t.quorum - 0.10).abs() < 1e-10);
        assert_eq!(t.max_extension_secs, 3600);
    }

    #[test]
    fn test_adaptive_threshold_constitutional() {
        let t = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);
        assert!((t.base_threshold - 0.67).abs() < 1e-10);
        assert!((t.min_voter_consciousness - 0.5).abs() < 1e-10);
        assert_eq!(t.min_participation, 10);
        assert!((t.quorum - 0.40).abs() < 1e-10);
        assert_eq!(t.max_extension_secs, 604800);
    }

    #[test]
    fn test_adaptive_threshold_consciousness_ordering() {
        let standard = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);
        let emergency = AdaptiveThreshold::for_proposal_type(&ProposalType::Emergency);
        let constitutional = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);
        assert!(standard.min_voter_consciousness < emergency.min_voter_consciousness);
        assert!(emergency.min_voter_consciousness < constitutional.min_voter_consciousness);
    }

    #[test]
    fn test_adaptive_threshold_quorum_check() {
        let t = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);
        // 20% quorum, min 5 participants
        assert!(!t.quorum_met(4, 100), "4 voters < 5 min_participation");
        assert!(t.quorum_met(20, 100), "20/100 = 20% meets quorum");
        assert!(!t.quorum_met(19, 100), "19/100 = 19% below 20%");
        assert!(!t.quorum_met(5, 0), "0 eligible → never meets quorum");
    }

    #[test]
    fn test_adaptive_threshold_voter_consciousness_check() {
        let t = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional);
        assert!(!t.voter_meets_consciousness_requirement(0.49), "0.49 < 0.5");
        assert!(t.voter_meets_consciousness_requirement(0.50), "0.50 = 0.5");
        assert!(t.voter_meets_consciousness_requirement(0.80), "0.80 > 0.5");
    }

    #[test]
    fn test_adaptive_threshold_calculate_threshold() {
        let t = AdaptiveThreshold::for_proposal_type(&ProposalType::Standard);
        // base_threshold = 0.51, so for total_weight = 10.0: 5.1
        assert!((t.calculate_threshold(10.0) - 5.1).abs() < 1e-10);
    }

    // --- Threshold hierarchy documentation test ---

    #[test]
    fn test_threshold_hierarchy_voter_bar_below_proposer_bar() {
        // For Constitutional proposals:
        // - Proposer needs GovernanceActionType::Constitutional = 0.6
        // - Voter needs AdaptiveThreshold(Constitutional).min_voter_consciousness = 0.5
        // This is intentional: lower bar for voting than proposing
        let proposer_bar = GovernanceActionType::Constitutional.consciousness_gate();
        let voter_bar = AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional).min_voter_consciousness;
        assert!(voter_bar < proposer_bar,
            "Constitutional voter bar ({}) should be lower than proposer bar ({})",
            voter_bar, proposer_bar);
    }
}

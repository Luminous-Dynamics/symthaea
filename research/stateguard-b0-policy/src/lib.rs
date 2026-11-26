//! StateGuard B0: deterministic, offer-constrained research policy baseline.
//!
//! This policy consumes a public ResearchState projection and an upstream list
//! of currently permitted action offers. It only returns an offer ID; it never
//! executes tools, grants permission, changes evidence, adjudicates truth or
//! accesses the evaluator-only sequential oracle.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use typed_research_state_contract::validate_state;

pub const POLICY_ID: &str = "stateguard-b0";
pub const POLICY_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActionKind {
    RetrieveOrRecapture,
    VerifyArtifact,
    ClassifyClaimSource,
    SeekIndependentSource,
    RequestDiscriminatingExperiment,
    Replicate,
    RequestIndependentAdjudication,
    RefineScope,
    Escalate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecisionDisposition {
    Selected,
    Abstain,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionOffer {
    /// Stable ID supplied by the upstream action authority.
    pub offer_id: String,
    /// Type of action that the upstream authority has already permitted.
    pub kind: ActionKind,
    /// Claim target, if the action is claim-specific.
    pub target_claim_id: Option<String>,
    /// Evidence target, if the action is source-specific.
    pub target_evidence_id: Option<String>,
    /// Integer costs in every declared resource dimension.
    pub resource_cost: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyInput {
    /// Exact serialized public ResearchState bytes.
    pub research_state_json: String,
    /// Only actions already authorized by the AgentView/permission owner.
    pub permitted_action_offers: Vec<ActionOffer>,
    /// Remaining budget supplied by the same upstream authority.
    pub remaining_budget: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyDecision {
    pub policy_id: String,
    pub policy_version: String,
    pub disposition: DecisionDisposition,
    /// Present only for SELECTED. It must exactly match a current permitted offer.
    pub selected_offer_id: Option<String>,
    pub selected_action_kind: Option<ActionKind>,
    pub target_claim_id: Option<String>,
    pub target_evidence_id: Option<String>,
    pub reason_codes: Vec<String>,
    /// SHA-256 of the exact serialized candidate ResearchState bytes.
    pub research_state_sha256: String,
    /// Upstream evidence-prefix digest, copied unchanged.
    pub evidence_prefix_sha256: String,
    /// SHA-256 of the canonical serialized ordered action-offer list.
    pub action_offers_sha256: String,
    /// Copied from the selected offer; empty for ABSTAIN / STOP.
    pub resource_estimate: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
struct PolicyNeed {
    priority: u8,
    claim_id: Option<String>,
    evidence_id: Option<String>,
    action_kinds: Vec<ActionKind>,
    reason_codes: Vec<String>,
}

fn issue(issues: &mut Vec<PolicyIssue>, path: impl Into<String>, message: impl Into<String>) {
    issues.push(PolicyIssue {
        path: path.into(),
        message: message.into(),
    });
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn array<'a>(value: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    value.get(key).and_then(Value::as_array)
}

fn relation_has(relations: &[Value], relation: &str) -> bool {
    relations.iter().any(|item| get_str(item, "relation") == Some(relation))
}

fn first_relation_evidence(relations: &[Value], relation: &str) -> Option<String> {
    relations
        .iter()
        .filter(|item| get_str(item, "relation") == Some(relation))
        .filter_map(|item| get_str(item, "evidence_id").map(str::to_owned))
        .min()
}

fn evidence_for_claim<'a>(
    state: &'a Value,
    relations: &[Value],
) -> Vec<&'a Value> {
    let ids: HashSet<&str> = relations
        .iter()
        .filter_map(|item| get_str(item, "evidence_id"))
        .collect();
    array(state, "evidence")
        .into_iter()
        .flatten()
        .filter(|item| get_str(item, "evidence_id").is_some_and(|id| ids.contains(id)))
        .collect()
}

fn need_for_claim(
    state: &Value,
    claim: &Value,
    risk_class: &str,
) -> Option<PolicyNeed> {
    let claim_id = get_str(claim, "claim_id").map(str::to_owned);
    let usability = claim.get("decision_usability");
    let usability_status = usability.and_then(|value| get_str(value, "status"));
    let relations = array(claim, "evidence_relations").map(Vec::as_slice).unwrap_or(&[]);
    let evidence = evidence_for_claim(state, relations);

    // Repair broken artifact capture/attribution before reasoning over content.
    for item in &evidence {
        let retrieval = get_str(item, "retrieval_state").unwrap_or("UNKNOWN");
        let integrity = get_str(item, "integrity_status").unwrap_or("UNKNOWN");
        let correction = get_str(item, "correction_status").unwrap_or("UNKNOWN");
        if retrieval != "CAPTURED" || integrity != "CLEAN" || correction != "NONE" {
            let evidence_id = get_str(item, "evidence_id").map(str::to_owned);
            let source_changed = ["CORRECTED", "RETRACTED", "SUPERSEDED"].contains(&correction);
            return Some(PolicyNeed {
                priority: 0,
                claim_id,
                evidence_id,
                action_kinds: vec![
                    ActionKind::VerifyArtifact,
                    ActionKind::RetrieveOrRecapture,
                    ActionKind::Escalate,
                ],
                reason_codes: vec![if source_changed {
                    "SOURCE_CORRECTED_RETRACTED_OR_SUPERSEDED".to_owned()
                } else if correction == "UNKNOWN" {
                    "SOURCE_CORRECTION_STATUS_NOT_YET_VERIFIED".to_owned()
                } else if integrity == "UNKNOWN" || retrieval != "CAPTURED" {
                    "ARTIFACT_NOT_YET_CAPTURED_AND_INTEGRITY_VERIFIED".to_owned()
                } else {
                    "ARTIFACT_INTEGRITY_OR_ATTRIBUTION_GAP".to_owned()
                }],
            });
        }
    }

    // A stale or out-of-scope verdict must be refreshed or the question narrowed.
    if usability_status == Some("STALE") {
        return Some(PolicyNeed {
            priority: 1,
            claim_id,
            evidence_id: None,
            action_kinds: vec![
                ActionKind::RetrieveOrRecapture,
                ActionKind::RefineScope,
                ActionKind::Escalate,
            ],
            reason_codes: vec!["EVIDENCE_STALE_FOR_DECISION_REFERENCE_TIME".to_owned()],
        });
    }
    if usability_status == Some("OUT_OF_SCOPE") {
        return Some(PolicyNeed {
            priority: 1,
            claim_id,
            evidence_id: None,
            action_kinds: vec![ActionKind::RefineScope, ActionKind::Escalate],
            reason_codes: vec!["CLAIM_OR_EVIDENCE_OUT_OF_SCOPE".to_owned()],
        });
    }

    // Do not treat a page as claim support. Request classification of the exact pair.
    if relations.is_empty() || relation_has(relations, "UNCLASSIFIED") {
        let evidence_id = first_relation_evidence(relations, "UNCLASSIFIED")
            .or_else(|| {
                evidence.iter()
                    .filter_map(|item| get_str(item, "evidence_id").map(str::to_owned))
                    .min()
            });
        return Some(PolicyNeed {
            priority: 2,
            claim_id,
            evidence_id,
            action_kinds: vec![
                ActionKind::ClassifyClaimSource,
                ActionKind::SeekIndependentSource,
                ActionKind::Escalate,
            ],
            reason_codes: vec!["CLAIM_SOURCE_RELATION_UNCLASSIFIED".to_owned()],
        });
    }

    // Explicit conflict is a discriminating-question problem, not a majority vote.
    let has_support = relation_has(relations, "SUPPORTS");
    let has_refute = relation_has(relations, "REFUTES");
    if usability_status == Some("CONFLICTED") || (has_support && has_refute) {
        return Some(PolicyNeed {
            priority: 3,
            claim_id,
            evidence_id: None,
            action_kinds: vec![
                ActionKind::RequestDiscriminatingExperiment,
                ActionKind::SeekIndependentSource,
                ActionKind::Replicate,
                ActionKind::Escalate,
            ],
            reason_codes: vec![
                "MATERIAL_CLAIM_SOURCE_CONFLICT".to_owned(),
                "DO_NOT_MAJORITY_VOTE_CORRELATED_EVIDENCE".to_owned(),
            ],
        });
    }

    let adjudication = claim.get("truth_adjudication");
    let adjudication_status = adjudication.and_then(|value| get_str(value, "status"));
    let adjudication_result = adjudication.and_then(|value| get_str(value, "result"));
    if ["REFUTED", "MIXED", "UNDECIDABLE"].contains(&adjudication_result.unwrap_or("NOT_ADJUDICATED")) {
        return Some(PolicyNeed {
            priority: 4,
            claim_id,
            evidence_id: None,
            action_kinds: vec![
                ActionKind::RequestDiscriminatingExperiment,
                ActionKind::SeekIndependentSource,
                ActionKind::Replicate,
                ActionKind::Escalate,
            ],
            reason_codes: vec!["INDEPENDENT_ADJUDICATION_NOT_DECISIVELY_SUPPORTIVE".to_owned()],
        });
    }

    if ["HIGH", "CRITICAL"].contains(&risk_class)
        && (adjudication_status != Some("ADJUDICATED")
            || adjudication_result != Some("SUPPORTED"))
    {
        return Some(PolicyNeed {
            priority: 4,
            claim_id,
            evidence_id: None,
            action_kinds: vec![
                ActionKind::RequestIndependentAdjudication,
                ActionKind::Escalate,
            ],
            reason_codes: vec!["HIGH_RISK_REQUIRES_INDEPENDENT_SUPPORTED_ADJUDICATION".to_owned()],
        });
    }

    if !has_support || usability_status != Some("SUFFICIENT") {
        return Some(PolicyNeed {
            priority: 5,
            claim_id,
            evidence_id: None,
            action_kinds: vec![
                ActionKind::SeekIndependentSource,
                ActionKind::RequestDiscriminatingExperiment,
                ActionKind::RequestIndependentAdjudication,
                ActionKind::Escalate,
            ],
            reason_codes: vec!["DECISION_EVIDENCE_NOT_YET_SUFFICIENT".to_owned()],
        });
    }

    None
}

fn pareto_dominates(left: &BTreeMap<String, u64>, right: &BTreeMap<String, u64>) -> bool {
    let mut strictly_better = false;
    for (dimension, right_cost) in right {
        let Some(left_cost) = left.get(dimension) else {
            return false;
        };
        if left_cost > right_cost {
            return false;
        }
        if left_cost < right_cost {
            strictly_better = true;
        }
    }
    strictly_better
}

fn same_cost(left: &BTreeMap<String, u64>, right: &BTreeMap<String, u64>) -> bool {
    left == right
}

fn fits_budget(cost: &BTreeMap<String, u64>, budget: &BTreeMap<String, u64>) -> bool {
    if cost.len() != budget.len() || !cost.keys().eq(budget.keys()) {
        return false;
    }
    cost.iter()
        .all(|(dimension, amount)| budget.get(dimension).is_some_and(|remaining| amount <= remaining))
}

fn offer_matches_need(offer: &ActionOffer, need: &PolicyNeed) -> bool {
    let claim_matches = offer.target_claim_id.as_deref() == need.claim_id.as_deref()
        || (offer.kind == ActionKind::Escalate && offer.target_claim_id.is_none());
    let evidence_matches = need.evidence_id.is_none()
        || offer.target_evidence_id.as_deref() == need.evidence_id.as_deref()
        || offer.kind == ActionKind::Escalate;
    need.action_kinds.contains(&offer.kind) && claim_matches && evidence_matches
}

fn abstain(
    state_sha256: String,
    evidence_prefix_sha256: String,
    action_offers_sha256: String,
    reason: &str,
) -> PolicyDecision {
    PolicyDecision {
        policy_id: POLICY_ID.to_owned(),
        policy_version: POLICY_VERSION.to_owned(),
        disposition: DecisionDisposition::Abstain,
        selected_offer_id: None,
        selected_action_kind: None,
        target_claim_id: None,
        target_evidence_id: None,
        reason_codes: vec![reason.to_owned()],
        research_state_sha256: state_sha256,
        evidence_prefix_sha256,
        action_offers_sha256,
        resource_estimate: BTreeMap::new(),
    }
}

fn stop(
    state_sha256: String,
    evidence_prefix_sha256: String,
    action_offers_sha256: String,
) -> PolicyDecision {
    PolicyDecision {
        policy_id: POLICY_ID.to_owned(),
        policy_version: POLICY_VERSION.to_owned(),
        disposition: DecisionDisposition::Stop,
        selected_offer_id: None,
        selected_action_kind: None,
        target_claim_id: None,
        target_evidence_id: None,
        reason_codes: vec!["ALL_CLAIMS_SUFFICIENT_FOR_CURRENT_DECISION".to_owned()],
        research_state_sha256: state_sha256,
        evidence_prefix_sha256,
        action_offers_sha256,
        resource_estimate: BTreeMap::new(),
    }
}

fn select_offer(
    need: &PolicyNeed,
    offers: &[ActionOffer],
    budget: &BTreeMap<String, u64>,
) -> Result<Option<ActionOffer>, String> {
    for kind in &need.action_kinds {
        let mut feasible: Vec<ActionOffer> = offers
            .iter()
            .filter(|offer| offer.kind == *kind && offer_matches_need(offer, need))
            .filter(|offer| fits_budget(&offer.resource_cost, budget))
            .cloned()
            .collect();
        if feasible.is_empty() {
            continue;
        }
        feasible.sort_by(|left, right| left.offer_id.cmp(&right.offer_id));
        let nondominated: Vec<ActionOffer> = feasible
            .iter()
            .filter(|candidate| {
                !feasible.iter().any(|other| {
                    other.offer_id != candidate.offer_id
                        && pareto_dominates(&other.resource_cost, &candidate.resource_cost)
                })
            })
            .cloned()
            .collect();

        if nondominated.len() == 1 {
            return Ok(nondominated.into_iter().next());
        }
        if nondominated
            .iter()
            .all(|offer| same_cost(&offer.resource_cost, &nondominated[0].resource_cost))
        {
            return Ok(nondominated.into_iter().min_by(|a, b| a.offer_id.cmp(&b.offer_id)));
        }
        return Err("PARETO_INCOMPARABLE_PERMITTED_OFFERS".to_owned());
    }
    Ok(None)
}

/// Choose one action from the permitted offer list, abstain, or stop.
///
/// This is a deterministic baseline. It never executes the selected action and
/// never creates a new permission or tool offer.
pub fn decide(input: &PolicyInput) -> Result<PolicyDecision, Vec<PolicyIssue>> {
    decide_inner(input, false)
}

fn decide_inner(
    input: &PolicyInput,
    allow_synthetic_fixtures: bool,
) -> Result<PolicyDecision, Vec<PolicyIssue>> {
    let validation = validate_state(&input.research_state_json).map_err(|issues| {
        issues
            .into_iter()
            .map(|item| PolicyIssue {
                path: item.path,
                message: item.message,
            })
            .collect::<Vec<_>>()
    })?;
    let state: Value = serde_json::from_str(&input.research_state_json).map_err(|error| {
        vec![PolicyIssue {
            path: "$".to_owned(),
            message: format!("invalid ResearchState JSON: {error}"),
        }]
    })?;
    let state_origin = get_str(&state, "status").unwrap_or("UNKNOWN");
    if state_origin != "PRODUCTION_PUBLIC_STATE" && !allow_synthetic_fixtures {
        return Err(vec![PolicyIssue {
            path: "research_state_json.status".to_owned(),
            message: "synthetic ResearchState is not admissible at the production policy entrypoint"
                .to_owned(),
        }]);
    }
    let evidence_prefix_sha256 = get_str(
        state.get("evidence_prefix").unwrap_or(&Value::Null),
        "prefix_sha256",
    )
    .unwrap_or_default()
    .to_owned();
    let action_offer_bytes = serde_json::to_vec(&input.permitted_action_offers).map_err(|error| {
        vec![PolicyIssue {
            path: "permitted_action_offers".to_owned(),
            message: format!("cannot serialize action offers: {error}"),
        }]
    })?;
    let action_offers_sha256 = sha256_hex(&action_offer_bytes);

    let mut issues = Vec::new();
    for dimension in input.remaining_budget.keys() {
        if dimension.trim().is_empty() {
            issue(&mut issues, "remaining_budget", "resource dimension names must be non-empty");
        }
    }
    let mut offer_ids = HashSet::new();
    let known_claims: HashSet<&str> = array(&state, "claims")
        .into_iter()
        .flatten()
        .filter_map(|claim| get_str(claim, "claim_id"))
        .collect();
    let known_evidence: HashSet<&str> = array(&state, "evidence")
        .into_iter()
        .flatten()
        .filter_map(|evidence| get_str(evidence, "evidence_id"))
        .collect();

    for (index, offer) in input.permitted_action_offers.iter().enumerate() {
        if offer.offer_id.trim().is_empty() {
            issue(&mut issues, format!("permitted_action_offers[{index}].offer_id"), "offer ID must be non-empty");
        }
        if !offer_ids.insert(offer.offer_id.clone()) {
            issue(&mut issues, format!("permitted_action_offers[{index}].offer_id"), "duplicate permitted offer ID");
        }
        if let Some(claim_id) = offer.target_claim_id.as_deref() {
            if !known_claims.contains(claim_id) {
                issue(&mut issues, format!("permitted_action_offers[{index}].target_claim_id"), "offer references unknown claim");
            }
        }
        if let Some(evidence_id) = offer.target_evidence_id.as_deref() {
            if !known_evidence.contains(evidence_id) {
                issue(&mut issues, format!("permitted_action_offers[{index}].target_evidence_id"), "offer references unknown evidence");
            }
        }
        if !offer.resource_cost.keys().eq(input.remaining_budget.keys()) {
            issue(
                &mut issues,
                format!("permitted_action_offers[{index}].resource_cost"),
                "cost vector must use exactly the same named dimensions as the remaining budget",
            );
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let decision = state.get("decision_context").unwrap_or(&Value::Null);
    let risk_class = get_str(decision, "risk_class").unwrap_or("UNKNOWN");
    let mut needs = Vec::new();
    for claim in array(&state, "claims").into_iter().flatten() {
        if let Some(need) = need_for_claim(&state, claim, risk_class) {
            needs.push(need);
        }
    }
    needs.sort_by(|left, right| {
        left.priority
            .cmp(&right.priority)
            .then_with(|| left.claim_id.cmp(&right.claim_id))
    });

    if needs.is_empty() {
        return Ok(stop(
            validation.state_sha256,
            evidence_prefix_sha256,
            action_offers_sha256,
        ));
    }

    let need = &needs[0];
    match select_offer(need, &input.permitted_action_offers, &input.remaining_budget) {
        Ok(Some(offer)) => Ok(PolicyDecision {
            policy_id: POLICY_ID.to_owned(),
            policy_version: POLICY_VERSION.to_owned(),
            disposition: DecisionDisposition::Selected,
            selected_offer_id: Some(offer.offer_id),
            selected_action_kind: Some(offer.kind),
            target_claim_id: need.claim_id.clone(),
            target_evidence_id: need.evidence_id.clone(),
            reason_codes: need.reason_codes.clone(),
            research_state_sha256: validation.state_sha256,
            evidence_prefix_sha256,
            action_offers_sha256,
            resource_estimate: offer.resource_cost,
        }),
        Ok(None) => Ok(abstain(
            validation.state_sha256,
            evidence_prefix_sha256,
            action_offers_sha256,
            "NO_FEASIBLE_PERMITTED_ACTION",
        )),
        Err(reason) => Ok(abstain(
            validation.state_sha256,
            evidence_prefix_sha256,
            action_offers_sha256,
            &reason,
        )),
    }
}

pub fn decide_json(input_json: &str) -> Result<String, Vec<PolicyIssue>> {
    let input: PolicyInput = serde_json::from_str(input_json).map_err(|error| {
        vec![PolicyIssue {
            path: "$".to_owned(),
            message: format!("invalid policy input: {error}"),
        }]
    })?;
    let decision = decide(&input)?;
    serde_json::to_string_pretty(&decision).map_err(|error| {
        vec![PolicyIssue {
            path: "decision".to_owned(),
            message: format!("cannot serialize policy decision: {error}"),
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state(index: usize) -> String {
        let pack: Value = serde_json::from_str(include_str!(
            "../typed-research-state/fixtures/research-state-examples-v0.1.json"
        ))
        .unwrap();
        pack["examples"][index]["state"].to_string()
    }

    fn artifact_verified_state(index: usize) -> String {
        let mut state: Value = serde_json::from_str(&sample_state(index)).unwrap();
        for item in state["evidence"].as_array_mut().unwrap() {
            if item["integrity_status"] == "UNKNOWN" {
                item["integrity_status"] = Value::String("CLEAN".to_owned());
            }
            if item["correction_status"] == "UNKNOWN" {
                item["correction_status"] = Value::String("NONE".to_owned());
            }
        }
        let evidence = state["evidence"].as_array().unwrap().clone();
        let episode = get_str(&state, "research_episode_id").unwrap().to_owned();
        let sequence = state["evidence_prefix"]["sequence_number"].as_u64().unwrap();
        let prefix = typed_research_state_contract::evidence_prefix_sha256(
            &episode,
            sequence,
            &evidence,
        );
        state["evidence_prefix"]["prefix_sha256"] = Value::String(prefix.clone());
        for claim in state["claims"].as_array_mut().unwrap() {
            for relation in claim["evidence_relations"].as_array_mut().unwrap() {
                if relation["classifier_provenance"].is_object() {
                    relation["classifier_provenance"]["evidence_prefix_sha256"] =
                        Value::String(prefix.clone());
                }
            }
        }
        state.to_string()
    }

    fn offer(
        offer_id: &str,
        kind: ActionKind,
        claim: Option<&str>,
        evidence: Option<&str>,
        cost: &[(&str, u64)],
    ) -> ActionOffer {
        ActionOffer {
            offer_id: offer_id.to_owned(),
            kind,
            target_claim_id: claim.map(str::to_owned),
            target_evidence_id: evidence.map(str::to_owned),
            resource_cost: cost.iter().map(|(key, value)| (key.to_string(), *value)).collect(),
        }
    }

    fn input(state: String, offers: Vec<ActionOffer>) -> PolicyInput {
        PolicyInput {
            research_state_json: state,
            permitted_action_offers: offers,
            remaining_budget: [
                ("query_units".to_owned(), 10),
                ("experiment_units".to_owned(), 10),
                ("review_minutes".to_owned(), 10),
            ]
            .into_iter()
            .collect(),
        }
    }


    #[test]
    fn production_entrypoint_rejects_synthetic_conformance_fixture() {
        let input: PolicyInput = serde_json::from_str(include_str!(
            "../fixtures/policy-input-synthetic-example-v0.1.json"
        ))
        .unwrap();
        let errors = decide(&input).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.path == "research_state_json.status"
                && error.message.contains("synthetic ResearchState")
        }));
    }

    #[test]
    fn production_entrypoint_accepts_explicit_production_state_and_selects_permitted_offer() {
        let mut input: PolicyInput = serde_json::from_str(include_str!(
            "../fixtures/policy-input-synthetic-example-v0.1.json"
        ))
        .unwrap();
        let mut state: Value = serde_json::from_str(&input.research_state_json).unwrap();
        state["status"] = Value::String("PRODUCTION_PUBLIC_STATE".to_owned());
        input.research_state_json = state.to_string();

        let decision = decide(&input).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Selected);
        assert_eq!(
            decision.selected_offer_id.as_deref(),
            Some("classify-claim-a-ev-a1")
        );
        assert_eq!(decision.selected_action_kind, Some(ActionKind::ClassifyClaimSource));
    }

    #[test]
    fn internal_synthetic_fixture_path_preserves_known_answer_policy_replay() {
        let input: PolicyInput = serde_json::from_str(include_str!(
            "../fixtures/policy-input-synthetic-example-v0.1.json"
        ))
        .unwrap();
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Selected);
        assert_eq!(
            decision.selected_offer_id.as_deref(),
            Some("classify-claim-a-ev-a1")
        );
    }

    #[test]
    fn unclassified_retrieval_selects_only_an_offered_classifier_action() {
        let input = input(
            artifact_verified_state(0),
            vec![offer(
                "classify-claim-a",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Selected);
        assert_eq!(decision.selected_offer_id.as_deref(), Some("classify-claim-a"));
        assert_eq!(decision.selected_action_kind, Some(ActionKind::ClassifyClaimSource));
    }

    #[test]
    fn unknown_resource_dimensions_fail_closed() {
        let mut input = input(sample_state(0), vec![]);
        input.remaining_budget.insert(String::new(), 1);
        assert!(decide_inner(&input, true).is_err());
    }

    #[test]
    fn unknown_integrity_requires_verification_before_classification() {
        let input = input(
            sample_state(0),
            vec![
                offer(
                    "verify-artifact",
                    ActionKind::VerifyArtifact,
                    Some("claim-a"),
                    Some("ev-a1"),
                    &[("query_units", 0), ("experiment_units", 0), ("review_minutes", 1)],
                ),
                offer(
                    "classify-claim-a",
                    ActionKind::ClassifyClaimSource,
                    Some("claim-a"),
                    Some("ev-a1"),
                    &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)],
                ),
            ],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("verify-artifact"));
        assert_eq!(
            decision.reason_codes,
            vec!["ARTIFACT_NOT_YET_CAPTURED_AND_INTEGRITY_VERIFIED"]
        );
    }

    #[test]
    fn unknown_correction_status_must_be_resolved_before_claim_use() {
        let mut state: Value = serde_json::from_str(&artifact_verified_state(0)).unwrap();
        state["evidence"][0]["correction_status"] = Value::String("UNKNOWN".to_owned());
        let input = input(
            state.to_string(),
            vec![offer(
                "check-source-corrections",
                ActionKind::VerifyArtifact,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 0), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("check-source-corrections"));
        assert_eq!(
            decision.reason_codes,
            vec!["SOURCE_CORRECTION_STATUS_NOT_YET_VERIFIED"]
        );
    }

    #[test]
    fn fetched_but_not_captured_source_requires_artifact_verification() {
        let mut state: Value = serde_json::from_str(&sample_state(1)).unwrap();
        state["evidence"][0]["retrieval_state"] = Value::String("FETCHED".to_owned());
        let evidence = state["evidence"].as_array().unwrap().clone();
        let episode = get_str(&state, "research_episode_id").unwrap().to_owned();
        let sequence = state["evidence_prefix"]["sequence_number"].as_u64().unwrap();
        let prefix = typed_research_state_contract::evidence_prefix_sha256(
            &episode,
            sequence,
            &evidence,
        );
        state["evidence_prefix"]["prefix_sha256"] = Value::String(prefix.clone());
        for relation in state["claims"][0]["evidence_relations"].as_array_mut().unwrap() {
            if relation["classifier_provenance"].is_object() {
                relation["classifier_provenance"]["evidence_prefix_sha256"] =
                    Value::String(prefix.clone());
            }
        }
        let input = input(
            state.to_string(),
            vec![offer(
                "verify-before-use",
                ActionKind::VerifyArtifact,
                Some("claim-b"),
                Some("ev-b1"),
                &[("query_units", 0), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("verify-before-use"));
    }

    #[test]
    fn deterministic_replay_returns_identical_decision() {
        let input = input(
            artifact_verified_state(0),
            vec![offer(
                "classify-claim-a",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let first = decide_inner(&input, true).unwrap();
        let second = decide_inner(&input, true).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn resource_dimension_mismatch_in_offer_is_rejected() {
        let input = input(
            artifact_verified_state(0),
            vec![offer(
                "classify-missing-review-dimension",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 1), ("experiment_units", 0)],
            )],
        );
        assert!(decide_inner(&input, true).is_err());
    }

    #[test]
    fn action_offer_change_changes_offer_digest_not_state_digest() {
        let state = artifact_verified_state(0);
        let first_input = input(
            state.clone(),
            vec![offer(
                "classify-a",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let second_input = input(
            state,
            vec![offer(
                "classify-b",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let first = decide(&first_input).unwrap();
        let second = decide(&second_input).unwrap();
        assert_eq!(first.research_state_sha256, second.research_state_sha256);
        assert_ne!(first.action_offers_sha256, second.action_offers_sha256);
    }

    #[test]
    fn no_permitted_action_causes_abstention() {
        let decision = decide(&input(sample_state(0), vec![])).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Abstain);
        assert_eq!(decision.reason_codes, vec!["NO_FEASIBLE_PERMITTED_ACTION"]);
    }

    #[test]
    fn retracted_source_is_refreshed_before_using_its_old_support_label() {
        let mut state: Value = serde_json::from_str(&sample_state(1)).unwrap();
        state["evidence"][0]["correction_status"] = Value::String("RETRACTED".to_owned());
        let input = input(
            state.to_string(),
            vec![offer(
                "refresh-retracted",
                ActionKind::RetrieveOrRecapture,
                Some("claim-b"),
                Some("ev-b1"),
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 0)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("refresh-retracted"));
        assert_eq!(
            decision.reason_codes,
            vec!["SOURCE_CORRECTED_RETRACTED_OR_SUPERSEDED"]
        );
    }

    #[test]
    fn refuted_adjudication_requests_discriminating_work_not_repeat_adjudication() {
        let mut state: Value = serde_json::from_str(&sample_state(1)).unwrap();
        state["decision_context"]["risk_class"] = Value::String("HIGH".to_owned());
        state["claims"][0]["truth_adjudication"] = serde_json::json!({
            "status": "ADJUDICATED",
            "result": "REFUTED",
            "authority_id": "independent-reviewer",
            "evidence_reference": "urn:adjudication:refuted",
            "adjudicated_at": "2026-11-25T10:10:00Z"
        });
        let input = input(
            state.to_string(),
            vec![offer(
                "discriminate",
                ActionKind::RequestDiscriminatingExperiment,
                Some("claim-b"),
                None,
                &[("query_units", 0), ("experiment_units", 2), ("review_minutes", 0)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_action_kind, Some(ActionKind::RequestDiscriminatingExperiment));
    }

    #[test]
    fn global_escalation_offer_is_usable_as_a_fail_safe_fallback() {
        let input = input(
            sample_state(0),
            vec![offer(
                "escalate-episode",
                ActionKind::Escalate,
                None,
                None,
                &[("query_units", 0), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("escalate-episode"));
    }

    #[test]
    fn conflict_prefers_a_permitted_discriminating_experiment() {
        let input = input(
            sample_state(2),
            vec![offer(
                "probe-conflict",
                ActionKind::RequestDiscriminatingExperiment,
                Some("claim-c"),
                None,
                &[("query_units", 0), ("experiment_units", 2), ("review_minutes", 0)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("probe-conflict"));
        assert_eq!(decision.selected_action_kind, Some(ActionKind::RequestDiscriminatingExperiment));
    }

    #[test]
    fn stale_historical_claim_selects_retrieval_not_stop() {
        let input = input(
            sample_state(3),
            vec![offer(
                "refresh-policy",
                ActionKind::RetrieveOrRecapture,
                Some("claim-d"),
                None,
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 0)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Selected);
        assert_eq!(decision.selected_offer_id.as_deref(), Some("refresh-policy"));
    }

    #[test]
    fn high_risk_supported_but_unadjudicated_claim_requests_adjudication() {
        let mut state: Value = serde_json::from_str(&sample_state(1)).unwrap();
        state["decision_context"]["risk_class"] = Value::String("HIGH".to_owned());
        let input = input(
            state.to_string(),
            vec![offer(
                "independent-adjudication",
                ActionKind::RequestIndependentAdjudication,
                Some("claim-b"),
                None,
                &[("query_units", 0), ("experiment_units", 0), ("review_minutes", 2)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_action_kind, Some(ActionKind::RequestIndependentAdjudication));
    }

    #[test]
    fn action_not_in_the_offer_list_cannot_be_invented() {
        let input = input(artifact_verified_state(0), vec![offer(
            "only-search",
            ActionKind::SeekIndependentSource,
            Some("claim-a"),
            None,
            &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 0)],
        )]);
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Selected);
        assert_eq!(decision.selected_offer_id.as_deref(), Some("only-search"));
        assert_ne!(decision.selected_action_kind, Some(ActionKind::ClassifyClaimSource));
    }

    #[test]
    fn insufficient_component_budget_causes_abstention() {
        let mut input = input(
            artifact_verified_state(0),
            vec![offer(
                "classify-too-expensive",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 11), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        input.remaining_budget.insert("query_units".to_owned(), 10);
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Abstain);
    }

    #[test]
    fn pareto_incomparable_same_intent_offers_abstain() {
        let input = input(
            artifact_verified_state(0),
            vec![
                offer("offer-a", ActionKind::ClassifyClaimSource, Some("claim-a"), Some("ev-a1"),
                    &[("query_units", 2), ("experiment_units", 0), ("review_minutes", 1)]),
                offer("offer-b", ActionKind::ClassifyClaimSource, Some("claim-a"), Some("ev-a1"),
                    &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 2)]),
            ],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Abstain);
        assert_eq!(decision.reason_codes, vec!["PARETO_INCOMPARABLE_PERMITTED_OFFERS"]);
    }

    #[test]
    fn dominated_same_intent_offer_is_not_selected() {
        let input = input(
            artifact_verified_state(0),
            vec![
                offer("offer-a", ActionKind::ClassifyClaimSource, Some("claim-a"), Some("ev-a1"),
                    &[("query_units", 2), ("experiment_units", 0), ("review_minutes", 1)]),
                offer("offer-b", ActionKind::ClassifyClaimSource, Some("claim-a"), Some("ev-a1"),
                    &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)]),
            ],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.selected_offer_id.as_deref(), Some("offer-b"));
    }

    #[test]
    fn selected_decision_binds_state_prefix_and_offer_hashes() {
        let input = input(
            artifact_verified_state(0),
            vec![offer(
                "classify",
                ActionKind::ClassifyClaimSource,
                Some("claim-a"),
                Some("ev-a1"),
                &[("query_units", 1), ("experiment_units", 0), ("review_minutes", 1)],
            )],
        );
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.research_state_sha256.len(), 64);
        assert_eq!(decision.evidence_prefix_sha256.len(), 64);
        assert_eq!(decision.action_offers_sha256.len(), 64);
    }

    #[test]
    fn all_sufficient_claims_stop_without_running_extra_actions() {
        let mut state: Value = serde_json::from_str(&sample_state(1)).unwrap();
        state["claims"][0]["decision_usability"]["status"] = Value::String("SUFFICIENT".to_owned());
        state["claims"][0]["truth_adjudication"] = serde_json::json!({
            "status": "ADJUDICATED",
            "result": "SUPPORTED",
            "authority_id": "independent-review",
            "evidence_reference": "urn:adjudication:case",
            "adjudicated_at": "2026-11-25T10:09:00Z"
        });
        // Moderate risk: a supported claim can be decision-sufficient under
        // the declared policy; the candidate still does not create the label.
        let input = input(state.to_string(), vec![]);
        let decision = decide_inner(&input, true).unwrap();
        assert_eq!(decision.disposition, DecisionDisposition::Stop);
    }
}

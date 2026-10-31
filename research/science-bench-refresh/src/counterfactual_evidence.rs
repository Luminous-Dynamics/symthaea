//! Pre-registered counterfactual evidence-intervention plan contract.
//!
//! This validates a plan for paired evidence interventions. It does not run an
//! agent, establish that a runner applied the interventions, authenticate the
//! plan's pre-run timestamp, or decide whether a claim follows from a source.
//! The plan commitment must be anchored independently before the first trial.

use crate::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const EVIDENCE_INTERVENTION_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_counterfactual_evidence_plan.v1";
pub const EVIDENCE_INTERVENTION_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_counterfactual_evidence_report.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.counterfactual_evidence.v1\0";
const MAX_PAIRS: usize = 50_000;
const MIN_CONDITIONS: usize = 2;
const CONDITIONS: &[&str] = &[
    "natural_retrieval",
    "network_isolated",
    "gold_evidence_injected",
    "gold_evidence_withheld",
    "decoy_noise",
    "source_contradiction",
];

/// The precommit input excludes only `plan_commitment_sha256`. Preserve this
/// field order: it is the language-neutral typed-commitment serialization order.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceInterventionPlanFieldsV1 {
    pub schema: String,
    pub generation_id: String,
    pub plan_id: String,
    pub policy_revision_sha256: String,
    pub configuration_sha256: String,
    pub prompt_sha256: String,
    pub tool_manifest_sha256: String,
    pub budget_sha256: String,
    pub analysis_plan_sha256: String,
    /// Evaluator implementation must be frozen before the first trial.
    pub evaluator_revision_sha256: String,
    /// Outcome labels and decision thresholds must be frozen before the first trial.
    pub decision_rules_sha256: String,
    pub randomization_commitment_sha256: String,
    pub required_conditions: Vec<String>,
    pub pairs: Vec<EvidenceInterventionPairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceInterventionPlanV1 {
    pub schema: String,
    pub generation_id: String,
    pub plan_id: String,
    pub policy_revision_sha256: String,
    pub configuration_sha256: String,
    pub prompt_sha256: String,
    pub tool_manifest_sha256: String,
    pub budget_sha256: String,
    pub analysis_plan_sha256: String,
    pub evaluator_revision_sha256: String,
    pub decision_rules_sha256: String,
    pub randomization_commitment_sha256: String,
    pub required_conditions: Vec<String>,
    pub pairs: Vec<EvidenceInterventionPairV1>,
    pub plan_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceInterventionPairV1 {
    /// Opaque, stable identity of one task/world and replicate.
    pub pair_id: String,
    /// Evaluator-side task identity digest. Never include target answer bytes.
    pub task_ref_sha256: String,
    pub task_family_id: String,
    pub trials: Vec<EvidenceInterventionTrialV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceInterventionTrialV1 {
    pub trial_id: String,
    pub condition: String,
    /// Digest of the exact policy-visible input artifact for this condition.
    pub policy_input_sha256: String,
    /// Digest of the evaluator/runner evidence manifest for this condition.
    pub evidence_manifest_sha256: String,
    /// Precommitted order within the pair; must be a permutation of 0..N.
    pub randomization_order: u32,
    /// Opaque commitment to this trial's seed, not the raw seed.
    pub seed_commitment_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceInterventionPlanReportV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_expected: String,
    pub plan_commitment_recomputed: String,
    pub plan_commitment_matches: bool,
    pub structural_valid: bool,
    pub valid: bool,
    pub generation_id: String,
    pub plan_id: String,
    pub pair_count: u64,
    pub planned_trial_count: u64,
    pub conditions_per_pair: u64,
    pub issues: Vec<String>,
    /// A pre-run commitment must be anchored outside this repository/runner.
    pub external_anchor_required: bool,
    /// Structural validity cannot prove the runner executed the intervention.
    pub runner_execution_proven: bool,
    pub admission: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && !value.bytes().all(|byte| byte == b'0')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_domain_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len =
        u64::try_from(domain.len()).map_err(|_| "hash domain exceeds u64 length".to_owned())?;
    let payload_len =
        u64::try_from(payload.len()).map_err(|_| "hash payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update(domain_len.to_be_bytes());
    hasher.update(domain);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn fields_from_plan(plan: &EvidenceInterventionPlanV1) -> EvidenceInterventionPlanFieldsV1 {
    EvidenceInterventionPlanFieldsV1 {
        schema: plan.schema.clone(),
        generation_id: plan.generation_id.clone(),
        plan_id: plan.plan_id.clone(),
        policy_revision_sha256: plan.policy_revision_sha256.clone(),
        configuration_sha256: plan.configuration_sha256.clone(),
        prompt_sha256: plan.prompt_sha256.clone(),
        tool_manifest_sha256: plan.tool_manifest_sha256.clone(),
        budget_sha256: plan.budget_sha256.clone(),
        analysis_plan_sha256: plan.analysis_plan_sha256.clone(),
        evaluator_revision_sha256: plan.evaluator_revision_sha256.clone(),
        decision_rules_sha256: plan.decision_rules_sha256.clone(),
        randomization_commitment_sha256: plan.randomization_commitment_sha256.clone(),
        required_conditions: plan.required_conditions.clone(),
        pairs: plan.pairs.clone(),
    }
}

fn validate_fields(fields: &EvidenceInterventionPlanFieldsV1) -> Vec<String> {
    let mut issues = Vec::new();
    if fields.schema != EVIDENCE_INTERVENTION_PLAN_SCHEMA {
        issues.push(format!(
            "schema must be {EVIDENCE_INTERVENTION_PLAN_SCHEMA}"
        ));
    }
    for (label, value, max_len) in [
        ("generation_id", fields.generation_id.as_str(), 128usize),
        ("plan_id", fields.plan_id.as_str(), 128usize),
    ] {
        if value.trim().is_empty() || value.len() > max_len {
            issues.push(format!(
                "{label} must be 1..={max_len} non-whitespace characters"
            ));
        }
    }

    for (label, value) in [
        (
            "policy_revision_sha256",
            fields.policy_revision_sha256.as_str(),
        ),
        ("configuration_sha256", fields.configuration_sha256.as_str()),
        ("prompt_sha256", fields.prompt_sha256.as_str()),
        ("tool_manifest_sha256", fields.tool_manifest_sha256.as_str()),
        ("budget_sha256", fields.budget_sha256.as_str()),
        ("analysis_plan_sha256", fields.analysis_plan_sha256.as_str()),
        (
            "evaluator_revision_sha256",
            fields.evaluator_revision_sha256.as_str(),
        ),
        (
            "decision_rules_sha256",
            fields.decision_rules_sha256.as_str(),
        ),
        (
            "randomization_commitment_sha256",
            fields.randomization_commitment_sha256.as_str(),
        ),
    ] {
        if !valid_digest(value) {
            issues.push(format!(
                "{label} must be 64 lowercase hexadecimal characters"
            ));
        }
    }

    if fields.required_conditions.len() < MIN_CONDITIONS
        || fields.required_conditions.len() > CONDITIONS.len()
    {
        issues.push(format!(
            "required_conditions must contain {MIN_CONDITIONS}..={} conditions",
            CONDITIONS.len()
        ));
    }
    let mut seen_conditions = BTreeSet::new();
    let mut previous_rank = None;
    for (index, condition) in fields.required_conditions.iter().enumerate() {
        let Some(rank) = CONDITIONS
            .iter()
            .position(|candidate| *candidate == condition)
        else {
            issues.push(format!(
                "required_conditions[{index}] is unsupported: {condition}"
            ));
            continue;
        };
        if !seen_conditions.insert(condition.as_str()) {
            issues.push(format!("duplicate required condition: {condition}"));
        }
        if previous_rank.is_some_and(|previous| previous >= rank) {
            issues.push(
                "required_conditions must use the canonical order declared by the schema"
                    .to_owned(),
            );
        }
        previous_rank = Some(rank);
    }

    if fields.pairs.is_empty() || fields.pairs.len() > MAX_PAIRS {
        issues.push(format!(
            "pairs must contain 1..={MAX_PAIRS} paired task records"
        ));
    }
    let required: BTreeSet<&str> = fields
        .required_conditions
        .iter()
        .map(String::as_str)
        .collect();
    let mut pair_ids = BTreeSet::new();
    let mut trial_ids = BTreeSet::new();
    for (pair_index, pair) in fields.pairs.iter().enumerate() {
        let pair_prefix = format!("pairs[{pair_index}]");
        if pair.pair_id.trim().is_empty() || pair.pair_id.len() > 128 {
            issues.push(format!("{pair_prefix}.pair_id must be 1..=128 characters"));
        } else if !pair_ids.insert(pair.pair_id.as_str()) {
            issues.push(format!("duplicate pair_id: {}", pair.pair_id));
        }
        if !valid_digest(&pair.task_ref_sha256) {
            issues.push(format!(
                "{pair_prefix}.task_ref_sha256 must be a lowercase SHA-256 digest"
            ));
        }
        if pair.task_family_id.trim().is_empty() || pair.task_family_id.len() > 128 {
            issues.push(format!(
                "{pair_prefix}.task_family_id must be 1..=128 characters"
            ));
        }
        if pair.trials.len() != fields.required_conditions.len() {
            issues.push(format!(
                "{pair_prefix} must have exactly {} trials; observed {}",
                fields.required_conditions.len(),
                pair.trials.len()
            ));
        }

        let mut conditions = BTreeSet::new();
        let mut evidence_manifest_digests = BTreeSet::new();
        let mut orders = BTreeSet::new();
        for (trial_index, trial) in pair.trials.iter().enumerate() {
            let trial_prefix = format!("{pair_prefix}.trials[{trial_index}]");
            if trial.trial_id.trim().is_empty() || trial.trial_id.len() > 160 {
                issues.push(format!(
                    "{trial_prefix}.trial_id must be 1..=160 characters"
                ));
            } else if !trial_ids.insert(trial.trial_id.as_str()) {
                issues.push(format!("duplicate trial_id: {}", trial.trial_id));
            }
            if !required.contains(trial.condition.as_str()) {
                issues.push(format!(
                    "{trial_prefix}.condition is not declared in required_conditions: {}",
                    trial.condition
                ));
            }
            if !conditions.insert(trial.condition.as_str()) {
                issues.push(format!(
                    "{pair_prefix} contains duplicate condition {}",
                    trial.condition
                ));
            }
            if !valid_digest(&trial.policy_input_sha256) {
                issues.push(format!(
                    "{trial_prefix}.policy_input_sha256 must be a lowercase SHA-256 digest"
                ));
            }
            if !valid_digest(&trial.evidence_manifest_sha256) {
                issues.push(format!(
                    "{trial_prefix}.evidence_manifest_sha256 must be a nonzero lowercase SHA-256 digest"
                ));
            } else if !evidence_manifest_digests.insert(trial.evidence_manifest_sha256.as_str()) {
                issues.push(format!(
                    "{pair_prefix} reuses evidence_manifest_sha256 across conditions"
                ));
            }
            if !valid_digest(&trial.seed_commitment_sha256) {
                issues.push(format!(
                    "{trial_prefix}.seed_commitment_sha256 must be a lowercase SHA-256 digest"
                ));
            }
            if trial.randomization_order as usize >= fields.required_conditions.len() {
                issues.push(format!(
                    "{trial_prefix}.randomization_order is outside the pair's condition range"
                ));
            }
            if !orders.insert(trial.randomization_order) {
                issues.push(format!(
                    "{pair_prefix} contains duplicate randomization_order {}",
                    trial.randomization_order
                ));
            }
        }
        if conditions != required {
            issues.push(format!(
                "{pair_prefix} does not contain each required condition exactly once"
            ));
        }
        if orders.len() == fields.required_conditions.len()
            && (0..fields.required_conditions.len()).any(|order| !orders.contains(&(order as u32)))
        {
            issues.push(format!(
                "{pair_prefix}.randomization_order must be a complete permutation starting at zero"
            ));
        }
    }
    issues
}

/// Compute the typed, domain-separated commitment for a validated fields file.
/// This does not anchor the commitment or prove it predates execution.
pub fn compute_evidence_intervention_plan_commitment(bytes: &[u8]) -> Result<String, String> {
    let fields: EvidenceInterventionPlanFieldsV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("plan fields do not match strict v1 shape: {error}"))?;
    let issues = validate_fields(&fields);
    if !issues.is_empty() {
        return Err(format!("plan fields invalid: {}", issues.join("; ")));
    }
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize plan fields: {error}"))?;
    digest_domain_framed(b"plan.v1", &payload)
}

/// Verify strict shape, paired-condition completeness and commitment consistency.
/// A valid report is a qualified *plan contract* only; execution remains external.
pub fn verify_evidence_intervention_plan(
    bytes: &[u8],
) -> Result<EvidenceInterventionPlanReportV1, String> {
    let plan: EvidenceInterventionPlanV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("plan does not match strict v1 shape: {error}"))?;
    let fields = fields_from_plan(&plan);
    let mut issues = validate_fields(&fields);
    if !valid_digest(&plan.plan_commitment_sha256) {
        issues.push("plan_commitment_sha256 must be a lowercase SHA-256 digest".to_owned());
    }
    let recomputed = match serde_json::to_vec(&fields) {
        Ok(payload) => match digest_domain_framed(b"plan.v1", &payload) {
            Ok(value) => value,
            Err(error) => {
                issues.push(error);
                String::new()
            }
        },
        Err(error) => {
            issues.push(format!("cannot serialize plan fields: {error}"));
            String::new()
        }
    };
    let commitment_matches =
        valid_digest(&plan.plan_commitment_sha256) && plan.plan_commitment_sha256 == recomputed;
    if !commitment_matches {
        issues.push("plan commitment does not match the typed plan fields".to_owned());
    }
    let structural_valid = validate_fields(&fields).is_empty();
    let valid = structural_valid && commitment_matches && issues.is_empty();
    let pair_count = u64::try_from(plan.pairs.len()).unwrap_or(u64::MAX);
    let conditions_per_pair = u64::try_from(plan.required_conditions.len()).unwrap_or(u64::MAX);
    let planned_trial_count = pair_count.saturating_mul(conditions_per_pair);
    Ok(EvidenceInterventionPlanReportV1 {
        schema: EVIDENCE_INTERVENTION_REPORT_SCHEMA.to_owned(),
        plan_sha256: sha256_hex(bytes),
        plan_commitment_expected: plan.plan_commitment_sha256,
        plan_commitment_recomputed: recomputed,
        plan_commitment_matches: commitment_matches,
        structural_valid,
        valid,
        generation_id: plan.generation_id,
        plan_id: plan.plan_id,
        pair_count,
        planned_trial_count,
        conditions_per_pair,
        issues,
        external_anchor_required: true,
        runner_execution_proven: false,
        admission: if valid {
            "valid_precommit_contract_external_anchor_and_runner_evidence_required".to_owned()
        } else {
            "blocked_invalid_counterfactual_intervention_plan".to_owned()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fields() -> EvidenceInterventionPlanFieldsV1 {
        let required_conditions = CONDITIONS
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>();
        let trials = CONDITIONS
            .iter()
            .enumerate()
            .map(|(index, condition)| EvidenceInterventionTrialV1 {
                trial_id: format!("trial-{index}"),
                condition: (*condition).to_owned(),
                policy_input_sha256: digest(char::from(b'a' + index as u8)),
                evidence_manifest_sha256: digest(char::from(b'1' + index as u8)),
                randomization_order: index as u32,
                seed_commitment_sha256: digest(['7', '8', '9', 'a', 'b', 'c'][index]),
            })
            .collect();
        EvidenceInterventionPlanFieldsV1 {
            schema: EVIDENCE_INTERVENTION_PLAN_SCHEMA.to_owned(),
            generation_id: "generation-1".to_owned(),
            plan_id: "plan-1".to_owned(),
            policy_revision_sha256: digest('1'),
            configuration_sha256: digest('2'),
            prompt_sha256: digest('3'),
            tool_manifest_sha256: digest('4'),
            budget_sha256: digest('5'),
            analysis_plan_sha256: digest('6'),
            evaluator_revision_sha256: digest('7'),
            decision_rules_sha256: digest('9'),
            randomization_commitment_sha256: digest('8'),
            required_conditions,
            pairs: vec![EvidenceInterventionPairV1 {
                pair_id: "pair-1".to_owned(),
                task_ref_sha256: digest('b'),
                task_family_id: "causal-science".to_owned(),
                trials,
            }],
        }
    }

    #[test]
    fn accepts_complete_intervention_pairs() {
        assert!(validate_fields(&fields()).is_empty());
    }

    #[test]
    fn rejects_missing_condition_even_when_other_fields_are_valid() {
        let mut fields = fields();
        fields.pairs[0].trials.pop();
        assert!(
            validate_fields(&fields)
                .iter()
                .any(|issue| issue.contains("exactly") || issue.contains("each required condition"))
        );
    }

    #[test]
    fn rejects_duplicate_trial_id_and_order() {
        let mut fields = fields();
        let first_trial_id = fields.pairs[0].trials[0].trial_id.clone();
        let first_order = fields.pairs[0].trials[0].randomization_order;
        fields.pairs[0].trials[1].trial_id = first_trial_id;
        fields.pairs[0].trials[1].randomization_order = first_order;
        let issues = validate_fields(&fields);
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("duplicate trial_id"))
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("duplicate randomization_order"))
        );
    }

    #[test]
    fn rejects_noncanonical_condition_order_and_invalid_digest() {
        let mut fields = fields();
        fields.required_conditions.swap(0, 1);
        fields.pairs[0].trials[0].policy_input_sha256 = "not-a-digest".to_owned();
        let first_evidence_manifest = fields.pairs[0].trials[0].evidence_manifest_sha256.clone();
        fields.pairs[0].trials[1].evidence_manifest_sha256 = first_evidence_manifest;
        let issues = validate_fields(&fields);
        assert!(issues.iter().any(|issue| issue.contains("canonical order")));
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("policy_input_sha256"))
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("reuses evidence_manifest_sha256"))
        );
    }

    #[test]
    fn rejects_all_zero_digest_placeholders() {
        assert!(!valid_digest(&digest('0')));
    }
}

//! Gold-label-side oracle bounds for selective workspace revalidation.
//!
//! The D6 scorer must validate the scenario before this module is called.
//! This module then uses evaluator-side affected-obligation labels to compute
//! the minimum set of active obligations that must receive fresh evidence at
//! each post-baseline epoch: newly introduced obligations plus obligations
//! explicitly marked affected by that epoch. It measures both over- and
//! under-revalidation. The gold labels are never intended for candidate use.
//!
//! This is a count-based lower bound (one unit per obligation), not a claim
//! that real checks have equal execution cost or that the gold impact labels
//! are complete/authentic. It does not execute checks or inspect artifacts.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct EpochRevalidationOracle {
    pub epoch: u64,
    pub active_obligation_count: usize,
    pub gold_required_revalidation_obligations: Vec<String>,
    pub agent_freshly_revalidated_obligations: Vec<String>,
    pub agent_fresh_pass_obligations: Vec<String>,
    pub unnecessary_revalidation_obligations: Vec<String>,
    pub missing_required_revalidation_obligations: Vec<String>,
    pub required_without_fresh_pass_obligations: Vec<String>,
    pub oracle_minimum_revalidation_units: usize,
    pub agent_fresh_revalidation_units: usize,
    pub full_rerun_reference_units: usize,
    pub over_revalidation_units: usize,
    pub under_revalidation_units: usize,
    pub selection_set_distance_units: usize,
    pub oracle_lower_bound_savings_fraction: Option<f64>,
    pub agent_observed_refresh_fraction: Option<f64>,
    pub revalidation_precision: Option<f64>,
    pub revalidation_recall: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArmRevalidationOracle {
    pub arm_id: String,
    pub condition: String,
    pub epochs: Vec<EpochRevalidationOracle>,
    pub total_oracle_minimum_revalidation_units: usize,
    pub total_agent_fresh_revalidation_units: usize,
    pub total_full_rerun_reference_units: usize,
    pub total_over_revalidation_units: usize,
    pub total_under_revalidation_units: usize,
    pub total_selection_set_distance_units: usize,
    pub oracle_lower_bound_savings_fraction: Option<f64>,
    pub agent_observed_refresh_fraction: Option<f64>,
    pub aggregate_revalidation_precision: Option<f64>,
    pub aggregate_revalidation_recall: Option<f64>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SelectiveRevalidationOracleReport {
    pub scenario_id: String,
    pub arm_reports: Vec<ArmRevalidationOracle>,
    pub claim_ceiling: String,
}

fn get_str<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a str, String> {
    value.get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn get_u64(value: &Value, key: &str, label: &str) -> Result<u64, String> {
    value.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} must be an unsigned integer"))
}

fn parse_id_set(value: &Value, key: &str, label: &str) -> Result<BTreeSet<String>, String> {
    let values = value.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} must be an array"))?;
    let mut set = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = item.as_str()
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| format!("{label}[{index}] must be a non-empty string"))?;
        if !set.insert(id.to_owned()) {
            return Err(format!("{label} contains duplicate obligation {id:?}"));
        }
    }
    Ok(set)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn score_arm(
    arm: &Value,
    known_obligations: &BTreeSet<String>,
) -> Result<ArmRevalidationOracle, String> {
    let arm_id = get_str(arm, "arm_id", "arm.arm_id")?.to_owned();
    let condition = get_str(arm, "condition", "arm.condition")?.to_owned();
    let epochs = arm.get("epochs")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("arm {arm_id}: epochs must be an array"))?;
    let checkpoints = arm.get("checkpoints")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("arm {arm_id}: checkpoints must be an array"))?;
    if epochs.len() < 2 || epochs.len() != checkpoints.len() {
        return Err(format!(
            "arm {arm_id}: at least two epochs and an equal number of checkpoints are required"
        ));
    }

    let mut previous_active = BTreeSet::new();
    let mut epoch_reports = Vec::new();
    let mut total_oracle = 0usize;
    let mut total_refresh = 0usize;
    let mut total_full_rerun = 0usize;
    let mut total_over = 0usize;
    let mut total_under = 0usize;
    let mut total_distance = 0usize;

    for (index, epoch) in epochs.iter().enumerate() {
        let epoch_number = get_u64(epoch, "epoch", "epoch.epoch")?;
        let active = parse_id_set(epoch, "active_obligation_ids", "epoch.active_obligation_ids")?;
        let affected = parse_id_set(epoch, "affected_obligation_ids", "epoch.affected_obligation_ids")?;
        for id in active.iter().chain(affected.iter()) {
            if !known_obligations.contains(id) {
                return Err(format!(
                    "arm {arm_id} epoch {epoch_number}: unknown obligation {id:?}"
                ));
            }
        }

        if index == 0 {
            if active.is_empty() {
                return Err(format!("arm {arm_id}: initial epoch must have active obligations"));
            }
            previous_active = active;
            continue;
        }

        let checkpoint = checkpoints.iter()
            .find(|item| item.get("epoch").and_then(Value::as_u64) == Some(epoch_number))
            .ok_or_else(|| format!("arm {arm_id}: checkpoint missing for epoch {epoch_number}"))?;

        let newly_introduced: BTreeSet<_> = active.difference(&previous_active).cloned().collect();
        let gold_required: BTreeSet<_> = affected.union(&newly_introduced)
            .filter(|id| active.contains(*id))
            .cloned()
            .collect();

        let receipts = checkpoint.get("receipts")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("arm {arm_id} checkpoint {epoch_number}: receipts must be an array"))?;
        let mut current_receipts: BTreeMap<String, String> = BTreeMap::new();
        for (receipt_index, receipt) in receipts.iter().enumerate() {
            let obligation_id = get_str(
                receipt,
                "obligation_id",
                &format!("checkpoint[{epoch_number}].receipts[{receipt_index}].obligation_id"),
            )?.to_owned();
            if !known_obligations.contains(&obligation_id) {
                return Err(format!(
                    "arm {arm_id} checkpoint {epoch_number}: unknown receipt obligation {obligation_id:?}"
                ));
            }
            let source_epoch = get_u64(receipt, "source_epoch", "receipt.source_epoch")?;
            if source_epoch > epoch_number {
                return Err(format!(
                    "arm {arm_id} checkpoint {epoch_number}: receipt for {obligation_id:?} cites future source epoch {source_epoch}"
                ));
            }
            let outcome = get_str(receipt, "outcome", "receipt.outcome")?.to_owned();
            if !["pass", "fail", "unknown"].contains(&outcome.as_str()) {
                return Err(format!(
                    "arm {arm_id} checkpoint {epoch_number}: unsupported outcome {outcome:?}"
                ));
            }
            if current_receipts.insert(obligation_id.clone(), format!("{source_epoch}:{outcome}")).is_some() {
                return Err(format!(
                    "arm {arm_id} checkpoint {epoch_number}: duplicate current receipt for {obligation_id:?}"
                ));
            }
        }

        let mut refreshed = BTreeSet::new();
        let mut fresh_pass = BTreeSet::new();
        for (id, source_and_outcome) in &current_receipts {
            if !active.contains(id) {
                continue;
            }
            let (source, outcome) = source_and_outcome.split_once(':')
                .ok_or_else(|| "internal receipt encoding failure".to_owned())?;
            if source.parse::<u64>().ok() == Some(epoch_number) {
                refreshed.insert(id.clone());
                if outcome == "pass" {
                    fresh_pass.insert(id.clone());
                }
            }
        }

        let unnecessary: BTreeSet<_> = refreshed.difference(&gold_required).cloned().collect();
        let missing: BTreeSet<_> = gold_required.difference(&refreshed).cloned().collect();
        let required_without_pass: BTreeSet<_> = gold_required.difference(&fresh_pass).cloned().collect();
        let oracle_units = gold_required.len();
        let refresh_units = refreshed.len();
        let full_rerun_units = active.len();
        let over_units = unnecessary.len();
        let under_units = missing.len();
        let set_distance = over_units + under_units;

        epoch_reports.push(EpochRevalidationOracle {
            epoch: epoch_number,
            active_obligation_count: active.len(),
            gold_required_revalidation_obligations: gold_required.iter().cloned().collect(),
            agent_freshly_revalidated_obligations: refreshed.iter().cloned().collect(),
            agent_fresh_pass_obligations: fresh_pass.iter().cloned().collect(),
            unnecessary_revalidation_obligations: unnecessary.iter().cloned().collect(),
            missing_required_revalidation_obligations: missing.iter().cloned().collect(),
            required_without_fresh_pass_obligations: required_without_pass.iter().cloned().collect(),
            oracle_minimum_revalidation_units: oracle_units,
            agent_fresh_revalidation_units: refresh_units,
            full_rerun_reference_units: full_rerun_units,
            over_revalidation_units: over_units,
            under_revalidation_units: under_units,
            selection_set_distance_units: set_distance,
            oracle_lower_bound_savings_fraction: (full_rerun_units > 0)
                .then_some(1.0 - oracle_units as f64 / full_rerun_units as f64),
            agent_observed_refresh_fraction: ratio(refresh_units, full_rerun_units),
            revalidation_precision: ratio(
                refreshed.intersection(&gold_required).count(),
                refresh_units,
            ),
            revalidation_recall: ratio(
                refreshed.intersection(&gold_required).count(),
                oracle_units,
            ),
        });

        total_oracle += oracle_units;
        total_refresh += refresh_units;
        total_full_rerun += full_rerun_units;
        total_over += over_units;
        total_under += under_units;
        total_distance += set_distance;
        previous_active = active;
    }

    Ok(ArmRevalidationOracle {
        arm_id,
        condition,
        epochs: epoch_reports,
        total_oracle_minimum_revalidation_units: total_oracle,
        total_agent_fresh_revalidation_units: total_refresh,
        total_full_rerun_reference_units: total_full_rerun,
        total_over_revalidation_units: total_over,
        total_under_revalidation_units: total_under,
        total_selection_set_distance_units: total_distance,
        oracle_lower_bound_savings_fraction: (total_full_rerun > 0)
            .then_some(1.0 - total_oracle as f64 / total_full_rerun as f64),
        agent_observed_refresh_fraction: ratio(total_refresh, total_full_rerun),
        aggregate_revalidation_precision: ratio(
            total_refresh.saturating_sub(total_over),
            total_refresh,
        ),
        aggregate_revalidation_recall: ratio(
            total_refresh.saturating_sub(total_over),
            total_oracle,
        ),
        claim_ceiling: "Count-based lower bound from supplied evaluator-side impact labels. Assumes one equal-cost revalidation unit per active obligation; does not authenticate gold labels, execute checks, measure heterogeneous real check costs, or prove safe reuse.".to_owned(),
    })
}

/// Validate the source trace using the D6 scorer, then compute gold-label-side
/// selective-revalidation bounds. This function never writes into the trace.
pub fn score_selective_revalidation_oracle_json(
    input: &str,
) -> Result<SelectiveRevalidationOracleReport, String> {
    crate::workspace_shift::evaluate_shift_scenario_json(input)
        .map_err(|error| format!("D6 scenario validation failed: {error}"))?;

    let scenario: Value = serde_json::from_str(input)
        .map_err(|error| format!("scenario JSON parse failed: {error}"))?;
    let scenario_id = get_str(&scenario, "scenario_id", "scenario_id")?.to_owned();
    let obligations = scenario.get("obligations")
        .and_then(Value::as_array)
        .ok_or_else(|| "obligations must be an array".to_owned())?;
    let mut known_obligations = BTreeSet::new();
    for (index, obligation) in obligations.iter().enumerate() {
        let id = get_str(obligation, "obligation_id", &format!("obligations[{index}].obligation_id"))?;
        if !known_obligations.insert(id.to_owned()) {
            return Err(format!("duplicate obligation definition {id:?}"));
        }
    }

    let arms = scenario.get("arms")
        .and_then(Value::as_array)
        .ok_or_else(|| "arms must be an array".to_owned())?;
    if arms.len() != 2 {
        return Err("exactly two matched arms are required".to_owned());
    }

    let mut arm_reports = Vec::new();
    for arm in arms {
        arm_reports.push(score_arm(arm, &known_obligations)?);
    }
    arm_reports.sort_by(|a, b| a.condition.cmp(&b.condition));

    Ok(SelectiveRevalidationOracleReport {
        scenario_id,
        arm_reports,
        claim_ceiling: "Deterministic lower-bound analysis of a supplied, D6-validated trace using evaluator-side gold impact labels. Not a candidate-facing signal, execution oracle, signature, causal proof, or empirical agent result.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str =
        include_str!("../../../docs/development-map/fixtures/egddm-shiftbench-v0.1.json");

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn refresh_receipt_digest(
        scenario: &mut Value,
        arm_index: usize,
        checkpoint_index: usize,
        receipt_index: usize,
    ) {
        use sha2::{Digest, Sha256};

        #[derive(Serialize)]
        struct ReceiptMaterial {
            schema_version: String,
            scenario_id: String,
            repository_full_name: String,
            arm_id: String,
            receipt_id: String,
            obligation_id: String,
            check_id: String,
            outcome: String,
            source_epoch: u64,
            observed_at: String,
            available_at: String,
            revision: String,
            tree_oid: String,
            configuration_world_digest: String,
        }

        let source_epoch = scenario["arms"][arm_index]["checkpoints"][checkpoint_index]["receipts"][receipt_index]["source_epoch"]
            .as_u64()
            .unwrap();
        let source = scenario["arms"][arm_index]["epochs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|epoch| epoch["epoch"].as_u64() == Some(source_epoch))
            .unwrap();
        let revision = source["snapshot"]["revision"].clone();
        let tree_oid = source["snapshot"]["tree_oid"].clone();
        let configuration_world_digest =
            source["snapshot"]["configuration_world_digest"].clone();

        {
            let receipt = &mut scenario["arms"][arm_index]["checkpoints"][checkpoint_index]["receipts"][receipt_index];
            receipt["revision"] = revision;
            receipt["tree_oid"] = tree_oid;
            receipt["configuration_world_digest"] = configuration_world_digest;
            if source_epoch == 0 {
                receipt["observed_at"] = json!("2026-11-15T07:01:00Z");
                receipt["available_at"] = json!("2026-11-15T07:02:00Z");
            } else {
                receipt["observed_at"] = json!("2026-11-15T07:12:00Z");
                receipt["available_at"] = json!("2026-11-15T07:13:00Z");
            }
        }

        let arm_id = scenario["arms"][arm_index]["arm_id"].as_str().unwrap().to_owned();
        let receipt = &scenario["arms"][arm_index]["checkpoints"][checkpoint_index]["receipts"][receipt_index];
        let material = ReceiptMaterial {
            schema_version: scenario["schema_version"].as_str().unwrap().to_owned(),
            scenario_id: scenario["scenario_id"].as_str().unwrap().to_owned(),
            repository_full_name: scenario["repository_full_name"].as_str().unwrap().to_owned(),
            arm_id,
            receipt_id: receipt["receipt_id"].as_str().unwrap().to_owned(),
            obligation_id: receipt["obligation_id"].as_str().unwrap().to_owned(),
            check_id: receipt["check_id"].as_str().unwrap().to_owned(),
            outcome: receipt["outcome"].as_str().unwrap().to_owned(),
            source_epoch: receipt["source_epoch"].as_u64().unwrap(),
            observed_at: receipt["observed_at"].as_str().unwrap().to_owned(),
            available_at: receipt["available_at"].as_str().unwrap().to_owned(),
            revision: receipt["revision"].as_str().unwrap().to_owned(),
            tree_oid: receipt["tree_oid"].as_str().unwrap().to_owned(),
            configuration_world_digest: receipt["configuration_world_digest"].as_str().unwrap().to_owned(),
        };
        let bytes = serde_json::to_vec(&material).unwrap();
        let digest = Sha256::digest(bytes);
        let encoded = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        scenario["arms"][arm_index]["checkpoints"][checkpoint_index]["receipts"][receipt_index]["receipt_digest"] =
            json!(encoded);
    }

    #[test]
    fn digest_helper_reproduces_committed_d6_receipt_digest() {
        let mut scenario = fixture_value();
        let original = scenario["arms"][1]["checkpoints"][1]["receipts"][0]["receipt_digest"]
            .as_str()
            .unwrap()
            .to_owned();
        refresh_receipt_digest(&mut scenario, 1, 1, 0);
        assert_eq!(
            scenario["arms"][1]["checkpoints"][1]["receipts"][0]["receipt_digest"],
            original
        );
    }

    #[test]
    fn computes_oracle_minimum_for_shifted_synthetic_trace() {
        let report = score_selective_revalidation_oracle_json(FIXTURE).unwrap();
        let shifted = report.arm_reports.iter()
            .find(|arm| arm.condition == "shifted")
            .unwrap();
        assert_eq!(shifted.total_oracle_minimum_revalidation_units, 2);
        assert_eq!(shifted.total_full_rerun_reference_units, 3);
        assert_eq!(shifted.total_agent_fresh_revalidation_units, 2);
        assert_eq!(shifted.total_over_revalidation_units, 0);
        assert_eq!(shifted.total_under_revalidation_units, 0);
        assert_eq!(shifted.total_selection_set_distance_units, 0);
        assert_eq!(shifted.epochs[0].gold_required_revalidation_obligations, vec![
            "api-contract".to_owned(), "audit-requirement".to_owned()
        ]);
        assert_eq!(shifted.epochs[0].revalidation_precision, Some(1.0));
        assert_eq!(shifted.epochs[0].revalidation_recall, Some(1.0));
        assert!((shifted.oracle_lower_bound_savings_fraction.unwrap() - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn matches_committed_known_answer_report() {
        const EXPECTED: &str = include_str!("../../../docs/development-map/fixtures/egddm-selective-revalidation-oracle-v0.1.expected.json");
        let actual = score_selective_revalidation_oracle_json(FIXTURE).unwrap();
        let actual_value = serde_json::to_value(actual).unwrap();
        let expected_value: Value = serde_json::from_str(EXPECTED).unwrap();
        assert_eq!(actual_value, expected_value);
    }

    #[test]
    fn clean_arm_needs_no_post_baseline_revalidation() {
        let report = score_selective_revalidation_oracle_json(FIXTURE).unwrap();
        let clean = report.arm_reports.iter()
            .find(|arm| arm.condition == "clean")
            .unwrap();
        assert_eq!(clean.total_oracle_minimum_revalidation_units, 0);
        assert_eq!(clean.total_agent_fresh_revalidation_units, 0);
        assert_eq!(clean.total_full_rerun_reference_units, 2);
    }

    #[test]
    fn detects_unnecessary_refresh_of_unaffected_obligation() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["checkpoints"][1]["receipts"][1]["receipt_id"] = json!("shifted-epoch1-build-overcheck");
        scenario["arms"][1]["checkpoints"][1]["receipts"][1]["source_epoch"] = json!(1);
        refresh_receipt_digest(&mut scenario, 1, 1, 1);
        let report = score_selective_revalidation_oracle_json(&scenario.to_string()).unwrap();
        let shifted = report.arm_reports.iter().find(|arm| arm.condition == "shifted").unwrap();
        assert_eq!(shifted.total_over_revalidation_units, 1);
        assert_eq!(shifted.total_under_revalidation_units, 0);
        assert!(shifted.aggregate_revalidation_precision.unwrap() < 1.0);
    }

    #[test]
    fn detects_missing_required_revalidation() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["checkpoints"][1]["receipts"][0]["source_epoch"] = json!(0);
        refresh_receipt_digest(&mut scenario, 1, 1, 0);
        let report = score_selective_revalidation_oracle_json(&scenario.to_string()).unwrap();
        let shifted = report.arm_reports.iter().find(|arm| arm.condition == "shifted").unwrap();
        assert_eq!(shifted.total_under_revalidation_units, 1);
        assert_eq!(shifted.total_selection_set_distance_units, 1);
        assert!(shifted.aggregate_revalidation_recall.unwrap() < 1.0);
    }

    #[test]
    fn distinguishes_fresh_failure_from_missing_revalidation() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["checkpoints"][1]["receipts"][0]["outcome"] = json!("fail");
        refresh_receipt_digest(&mut scenario, 1, 1, 0);
        let report = score_selective_revalidation_oracle_json(&scenario.to_string()).unwrap();
        let shifted = report.arm_reports.iter().find(|arm| arm.condition == "shifted").unwrap();
        assert_eq!(shifted.total_under_revalidation_units, 0);
        assert_eq!(shifted.epochs[0].required_without_fresh_pass_obligations, vec!["api-contract".to_owned()]);
    }

    #[test]
    fn rejects_unknown_gold_affected_obligation() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["epochs"][1]["affected_obligation_ids"] = json!(["not-declared"]);
        assert!(score_selective_revalidation_oracle_json(&scenario.to_string()).is_err());
    }

    #[test]
    fn rejects_unknown_receipt_obligation() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["checkpoints"][1]["receipts"][0]["obligation_id"] = json!("not-declared");
        assert!(score_selective_revalidation_oracle_json(&scenario.to_string()).is_err());
    }

    #[test]
    fn rejects_future_source_epoch() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["checkpoints"][1]["receipts"][0]["source_epoch"] = json!(99);
        assert!(score_selective_revalidation_oracle_json(&scenario.to_string()).is_err());
    }

    #[test]
    fn rejects_duplicate_active_obligation() {
        let mut scenario = fixture_value();
        scenario["arms"][1]["epochs"][1]["active_obligation_ids"] = json!([
            "api-contract", "build-repro", "audit-requirement", "audit-requirement"
        ]);
        assert!(score_selective_revalidation_oracle_json(&scenario.to_string()).is_err());
    }

    #[test]
    fn reports_claim_ceiling_that_keeps_gold_labels_outside_candidate_scope() {
        let report = score_selective_revalidation_oracle_json(FIXTURE).unwrap();
        assert!(report.claim_ceiling.contains("evaluator-side gold impact labels"));
        assert!(report.claim_ceiling.contains("Not a candidate-facing signal"));
    }

    #[test]
    fn rejects_malformed_scenario_before_oracle_scoring() {
        let mut scenario = fixture_value();
        scenario["scenario_id"] = json!("");
        assert!(score_selective_revalidation_oracle_json(&scenario.to_string()).is_err());
    }
}

use crate::{is_full_git_sha, sha256, verify_output_directory, Manifest, Oracle, TruthLabel};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const PREDICTIONS_VERSION: &str = "cyber-hunt-predictions-v1";
const EVALUATION_VERSION: &str = "cyber-hunt-evaluation-v1";
const EXPECTED_VERIFIER_DISPOSITION: &str = "STRUCTURAL_PASS_NOT_CAPABILITY_PASS";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Suspect {
    record_id: String,
    evidence_refs: Vec<String>,
    rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PredictionSet {
    schema_version: String,
    run_id: String,
    source_commit: String,
    solver_manifest_sha256: String,
    event_store_sha256: String,
    suspects: Vec<Suspect>,
    abstained_record_ids: Vec<String>,
    non_claims: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PhaseMetrics {
    phase: String,
    ground_truth_malicious: usize,
    detected_malicious: usize,
    missed_malicious: usize,
    recall_basis_points: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CampaignMetrics {
    campaign_id: String,
    ground_truth_malicious: usize,
    detected_malicious: usize,
    missed_malicious: usize,
    recall_basis_points: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EvaluationReceipt {
    schema_version: String,
    disposition: String,
    run_id: String,
    source_commit: String,
    source_commit_pinned: bool,
    dataset_manifest_sha256: String,
    solver_manifest_sha256: String,
    event_store_sha256: String,
    prediction_set_sha256: String,
    total_events: usize,
    ground_truth_malicious: usize,
    ground_truth_benign: usize,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    true_negatives: usize,
    precision_basis_points: Option<u64>,
    recall_basis_points: Option<u64>,
    false_positives_per_10000_benign_milli: Option<u64>,
    abstained_events: usize,
    abstained_malicious_events: usize,
    abstained_benign_events: usize,
    benign_change_decoy_false_positives: usize,
    phase_metrics: Vec<PhaseMetrics>,
    campaign_metrics: Vec<CampaignMetrics>,
    non_claims: Vec<String>,
}

#[derive(Clone, Debug)]
struct EvaluationConfig {
    dataset_root: PathBuf,
    predictions_file: PathBuf,
}

fn basis_points(numerator: usize, denominator: usize) -> Option<u64> {
    if denominator == 0 {
        return None;
    }
    let n = numerator as u128;
    let d = denominator as u128;
    Some(((n * 10_000 + d / 2) / d) as u64)
}

fn false_positives_per_10000_milli(false_positives: usize, benign_events: usize) -> Option<u64> {
    if benign_events == 0 {
        return None;
    }
    let n = false_positives as u128 * 10_000_000u128;
    let d = benign_events as u128;
    Some(((n + d / 2) / d) as u64)
}

fn parse_jsonl_ids(bytes: &[u8], label: &str) -> Result<BTreeSet<String>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    if !text.ends_with('\n') {
        return Err(format!("{label} JSONL must end with LF"));
    }
    let mut ids = BTreeSet::new();
    for (index, line) in text.split_terminator('\n').enumerate() {
        if line.trim().is_empty() {
            return Err(format!("{label} has a blank record at line {}", index + 1));
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("{label} has invalid JSON at line {}: {error}", index + 1))?;
        let id = value.get("record_id").and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{label} record {} has no string record_id", index + 1))?;
        if !ids.insert(id.to_string()) {
            return Err(format!("{label} contains duplicate record ID {id}"));
        }
    }
    if ids.is_empty() {
        return Err(format!("{label} is empty"));
    }
    Ok(ids)
}

fn evaluate_prediction_bytes(
    dataset_root: &Path,
    prediction_bytes: &[u8],
) -> Result<EvaluationReceipt, String> {
    let structural = verify_output_directory(dataset_root)?;
    if structural.disposition != EXPECTED_VERIFIER_DISPOSITION {
        return Err(format!("dataset did not qualify for structural evaluation: {}", structural.disposition));
    }

    let manifest_bytes = fs::read(dataset_root.join("manifest.json"))
        .map_err(|error| format!("cannot read operator manifest: {error}"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("cannot parse operator manifest: {error}"))?;
    let solver_manifest_bytes = fs::read(dataset_root.join("solver/manifest.json"))
        .map_err(|error| format!("cannot read solver manifest: {error}"))?;
    let event_bytes = fs::read(dataset_root.join("solver/events.jsonl"))
        .map_err(|error| format!("cannot read solver event stream: {error}"))?;
    let oracle_bytes = fs::read(dataset_root.join("evaluator/oracle.json"))
        .map_err(|error| format!("cannot read evaluator oracle: {error}"))?;

    let prediction_set: PredictionSet = serde_json::from_slice(prediction_bytes)
        .map_err(|error| format!("prediction set is invalid JSON/schema shape: {error}"))?;
    if prediction_set.schema_version != PREDICTIONS_VERSION
        || prediction_set.run_id.trim().len() < 3
        || prediction_set.run_id.len() > 128
        || prediction_set.source_commit != manifest.source_commit
        || prediction_set.solver_manifest_sha256 != sha256(&solver_manifest_bytes)
        || prediction_set.event_store_sha256 != sha256(&event_bytes)
        || prediction_set.non_claims.is_empty()
        || prediction_set.non_claims.iter().any(|claim| claim.trim().is_empty())
    {
        return Err("prediction schema/version, source, artifact pin, or non-claim is invalid".to_string());
    }

    let solver_ids = parse_jsonl_ids(&event_bytes, "solver event stream")?;
    let oracle: Oracle = serde_json::from_slice(&oracle_bytes)
        .map_err(|error| format!("cannot parse evaluator oracle: {error}"))?;
    if oracle.schema_version != "cyber-hunt-oracle-v1"
        || oracle.generator_version != "cyber-hunt-fixture-v1"
        || oracle.seed != manifest.seed
        || oracle.record_count != manifest.actual_record_count
        || oracle.labels.len() != manifest.actual_record_count
    {
        return Err("oracle version/seed/count does not agree with the structural manifest".to_string());
    }

    let mut labels = BTreeMap::<String, TruthLabel>::new();
    for label in &oracle.labels {
        if !matches!(label.security_label.as_str(), "benign" | "malicious") {
            return Err(format!("oracle record {} has unknown security label", label.record_id));
        }
        if labels.insert(label.record_id.clone(), label.clone()).is_some() {
            return Err(format!("oracle contains duplicate record ID {}", label.record_id));
        }
    }
    if solver_ids != labels.keys().cloned().collect::<BTreeSet<_>>() {
        return Err("solver event IDs and oracle label IDs do not match exactly".to_string());
    }

    let mut suspects = BTreeMap::<String, Suspect>::new();
    for suspect in &prediction_set.suspects {
        if !solver_ids.contains(&suspect.record_id) {
            return Err(format!("prediction references unknown record ID {}", suspect.record_id));
        }
        if suspect.rationale.trim().len() < 8 || suspect.evidence_refs.is_empty() {
            return Err(format!("prediction for {} lacks evidence or rationale", suspect.record_id));
        }
        if suspects.insert(suspect.record_id.clone(), suspect.clone()).is_some() {
            return Err(format!("prediction set repeats suspected record ID {}", suspect.record_id));
        }
        let mut seen_refs = BTreeSet::new();
        for reference in &suspect.evidence_refs {
            if !solver_ids.contains(reference) {
                return Err(format!("prediction for {} references unknown evidence ID {reference}", suspect.record_id));
            }
            if !seen_refs.insert(reference.clone()) {
                return Err(format!("prediction for {} repeats evidence ID {reference}", suspect.record_id));
            }
        }
    }

    let mut abstained = BTreeSet::new();
    for record_id in &prediction_set.abstained_record_ids {
        if !solver_ids.contains(record_id) {
            return Err(format!("abstention references unknown record ID {record_id}"));
        }
        if suspects.contains_key(record_id) {
            return Err(format!("record {record_id} is both suspected and abstained"));
        }
        if !abstained.insert(record_id.clone()) {
            return Err(format!("prediction set repeats abstained record ID {record_id}"));
        }
    }

    let ground_truth_malicious = labels.values().filter(|label| label.security_label == "malicious").count();
    let ground_truth_benign = labels.values().filter(|label| label.security_label == "benign").count();
    let true_positives = suspects.keys()
        .filter(|id| labels.get(*id).is_some_and(|label| label.security_label == "malicious"))
        .count();
    let false_positives = suspects.len() - true_positives;
    let false_negatives = ground_truth_malicious - true_positives;
    let true_negatives = ground_truth_benign - false_positives;

    let abstained_malicious_events = abstained.iter()
        .filter(|id| labels.get(*id).is_some_and(|label| label.security_label == "malicious"))
        .count();
    let abstained_benign_events = abstained.len() - abstained_malicious_events;

    let benign_decoy_ids: BTreeSet<String> = oracle.campaigns.iter()
        .filter(|campaign| campaign.disposition == "benign_decoy")
        .flat_map(|campaign| campaign.event_record_ids.iter().cloned())
        .collect();
    let benign_change_decoy_false_positives = suspects.keys()
        .filter(|id| benign_decoy_ids.contains(*id))
        .count();

    let mut phases = BTreeMap::<String, (usize, usize)>::new();
    for label in labels.values().filter(|label| label.security_label == "malicious") {
        if let Some(phase) = label.phase.as_ref() {
            let entry = phases.entry(phase.clone()).or_insert((0, 0));
            entry.0 += 1;
            if suspects.contains_key(&label.record_id) {
                entry.1 += 1;
            }
        }
    }
    let phase_metrics = phases.into_iter().map(|(phase, (ground_truth, detected))| PhaseMetrics {
        phase,
        ground_truth_malicious: ground_truth,
        detected_malicious: detected,
        missed_malicious: ground_truth - detected,
        recall_basis_points: basis_points(detected, ground_truth),
    }).collect();

    let campaign_metrics = oracle.campaigns.iter()
        .filter(|campaign| campaign.disposition == "malicious")
        .map(|campaign| {
            let ground_truth = campaign.event_record_ids.len();
            let detected = campaign.event_record_ids.iter()
                .filter(|id| suspects.contains_key(*id))
                .count();
            CampaignMetrics {
                campaign_id: campaign.campaign_id.clone(),
                ground_truth_malicious: ground_truth,
                detected_malicious: detected,
                missed_malicious: ground_truth - detected,
                recall_basis_points: basis_points(detected, ground_truth),
            }
        })
        .collect();

    Ok(EvaluationReceipt {
        schema_version: EVALUATION_VERSION.to_string(),
        disposition: "SCORED_UNQUALIFIED_SYNTHETIC_ONLY".to_string(),
        run_id: prediction_set.run_id,
        source_commit: manifest.source_commit.clone(),
        source_commit_pinned: is_full_git_sha(&manifest.source_commit),
        dataset_manifest_sha256: sha256(&manifest_bytes),
        solver_manifest_sha256: sha256(&solver_manifest_bytes),
        event_store_sha256: sha256(&event_bytes),
        prediction_set_sha256: sha256(prediction_bytes),
        total_events: labels.len(),
        ground_truth_malicious,
        ground_truth_benign,
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
        precision_basis_points: basis_points(true_positives, suspects.len()),
        recall_basis_points: basis_points(true_positives, ground_truth_malicious),
        false_positives_per_10000_benign_milli: false_positives_per_10000_milli(false_positives, ground_truth_benign),
        abstained_events: abstained.len(),
        abstained_malicious_events,
        abstained_benign_events,
        benign_change_decoy_false_positives,
        phase_metrics,
        campaign_metrics,
        non_claims: vec![
            "This evaluator scores only the named synthetic corpus, prediction set, and source version.".to_string(),
            "Unflagged records, including abstained records, are not counted as alerts for event-level precision/recall.".to_string(),
            "Exact synthetic labels do not establish real incident truth, production detector effectiveness, analyst skill, or real-world security.".to_string(),
            "A score or benchmark pass grants no remediation or execution authority.".to_string(),
            "The operator/evaluator process must be isolated from the solver process and its query filesystem mount.".to_string(),
        ],
    })
}

fn run_cli(args: &[String]) -> Result<(), String> {
    let mut dataset_root = None;
    let mut predictions_file = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--dataset-root" => {
                dataset_root = Some(PathBuf::from(iter.next().ok_or_else(|| "--dataset-root requires a path".to_string())?));
            }
            "--predictions" => {
                predictions_file = Some(PathBuf::from(iter.next().ok_or_else(|| "--predictions requires a file".to_string())?));
            }
            other => return Err(format!("unknown evaluation argument {other}")),
        }
    }
    let dataset_root = dataset_root.ok_or_else(|| "evaluate mode requires --dataset-root".to_string())?;
    let predictions_file = predictions_file.ok_or_else(|| "evaluate mode requires --predictions".to_string())?;
    let prediction_bytes = fs::read(&predictions_file)
        .map_err(|error| format!("cannot read prediction set {}: {error}", predictions_file.display()))?;
    if prediction_bytes.len() > 16 * 1024 * 1024 {
        return Err("prediction set exceeds 16 MiB".to_string());
    }
    let receipt = evaluate_prediction_bytes(&dataset_root, &prediction_bytes)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("cannot serialize evaluation receipt: {error}"))?
    );
    Ok(())
}

pub fn run_cli_or_exit(args: &[String]) {
    if let Err(error) = run_cli(args) {
        eprintln!("cyber-hunt-fixture evaluate: FAIL: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{write_dataset, Config};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root() -> (PathBuf, Manifest, Oracle) {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)
            .expect("clock should be after UNIX epoch").as_nanos();
        let root = env::temp_dir().join(format!("cyber-hunt-evaluator-test-{}-{nonce}", process::id()));
        let config = Config {
            records: 1_000,
            seed: 42,
            out_dir: root.clone(),
            source_commit: "test-source-commit".to_string(),
        };
        let _receipt = write_dataset(&config).expect("synthetic dataset should be generated and verified");
        let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let oracle: Oracle = serde_json::from_slice(&fs::read(root.join("evaluator/oracle.json")).unwrap()).unwrap();
        (root, manifest, oracle)
    }

    fn prediction_set(root: &Path, manifest: &Manifest, oracle: &Oracle, only_malicious: bool) -> PredictionSet {
        let solver_manifest_bytes = fs::read(root.join("solver/manifest.json")).unwrap();
        let event_bytes = fs::read(root.join("solver/events.jsonl")).unwrap();
        let suspects = oracle.labels.iter()
            .filter(|label| !only_malicious || label.security_label == "malicious")
            .map(|label| Suspect {
                record_id: label.record_id.clone(),
                evidence_refs: vec![label.record_id.clone()],
                rationale: "Synthetic evaluator test assertion tied to this exact event record.".to_string(),
            })
            .collect();
        PredictionSet {
            schema_version: PREDICTIONS_VERSION.to_string(),
            run_id: "evaluator-test-run-001".to_string(),
            source_commit: manifest.source_commit.clone(),
            solver_manifest_sha256: sha256(&solver_manifest_bytes),
            event_store_sha256: sha256(&event_bytes),
            suspects,
            abstained_record_ids: Vec::new(),
            non_claims: vec!["Test fixture only.".to_string()],
        }
    }

    #[test]
    fn prediction_and_evaluation_receipts_match_public_schema_fields() {
        let (root, manifest, oracle) = fixture_root();
        let predictions = prediction_set(&root, &manifest, &oracle, true);
        let prediction_value = serde_json::to_value(&predictions).expect("prediction set should serialize");
        let prediction_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/predictions-v1.schema.json"))
                .expect("prediction schema should be valid JSON");
        let prediction_fields: BTreeSet<String> =
            prediction_value.as_object().unwrap().keys().cloned().collect();
        let prediction_required: BTreeSet<String> = prediction_schema["required"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_string()).collect();
        assert_eq!(prediction_fields, prediction_required);

        let receipt = evaluate_prediction_bytes(&root, &serde_json::to_vec(&predictions).unwrap())
            .expect("prediction set should score");
        let receipt_value = serde_json::to_value(&receipt).expect("receipt should serialize");
        let receipt_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/evaluation-receipt-v1.schema.json"))
                .expect("evaluation receipt schema should be valid JSON");
        let receipt_fields: BTreeSet<String> =
            receipt_value.as_object().unwrap().keys().cloned().collect();
        let receipt_required: BTreeSet<String> = receipt_schema["required"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_string()).collect();
        assert_eq!(receipt_fields, receipt_required);
        fs::remove_dir_all(root).expect("temporary dataset should be removed");
    }

    #[test]
    fn perfect_synthetic_predictions_score_separately_from_qualification() {
        let (root, manifest, oracle) = fixture_root();
        let predictions = prediction_set(&root, &manifest, &oracle, true);
        let bytes = serde_json::to_vec(&predictions).unwrap();
        let receipt = evaluate_prediction_bytes(&root, &bytes).expect("predictions should score");
        assert_eq!(receipt.disposition, "SCORED_UNQUALIFIED_SYNTHETIC_ONLY");
        assert_eq!(receipt.precision_basis_points, Some(10_000));
        assert_eq!(receipt.recall_basis_points, Some(10_000));
        assert_eq!(receipt.false_positives, 0);
        assert_eq!(receipt.false_negatives, 0);
        assert_eq!(receipt.campaign_metrics.len(), 2);
        assert!(!receipt.source_commit_pinned);
        fs::remove_dir_all(root).expect("temporary dataset should be removed");
    }

    #[test]
    fn empty_alert_set_reports_zero_recall_and_undefined_precision() {
        let (root, manifest, oracle) = fixture_root();
        let mut predictions = prediction_set(&root, &manifest, &oracle, true);
        predictions.suspects.clear();
        let bytes = serde_json::to_vec(&predictions).unwrap();
        let receipt = evaluate_prediction_bytes(&root, &bytes).expect("empty alerts should score");
        assert_eq!(receipt.precision_basis_points, None);
        assert_eq!(receipt.recall_basis_points, Some(0));
        assert_eq!(receipt.false_positives, 0);
        assert_eq!(receipt.false_negatives, receipt.ground_truth_malicious);
        fs::remove_dir_all(root).expect("temporary dataset should be removed");
    }

    #[test]
    fn evaluator_rejects_unknown_ids_duplicate_predictions_and_hash_mismatch() {
        let (root, manifest, oracle) = fixture_root();
        let mut predictions = prediction_set(&root, &manifest, &oracle, true);
        predictions.suspects[0].record_id = "evt-99999999".to_string();
        let bytes = serde_json::to_vec(&predictions).unwrap();
        assert!(evaluate_prediction_bytes(&root, &bytes).unwrap_err().contains("unknown record ID"));

        let mut predictions = prediction_set(&root, &manifest, &oracle, true);
        predictions.suspects.push(predictions.suspects[0].clone());
        let bytes = serde_json::to_vec(&predictions).unwrap();
        assert!(evaluate_prediction_bytes(&root, &bytes).unwrap_err().contains("repeats suspected"));

        let mut predictions = prediction_set(&root, &manifest, &oracle, true);
        predictions.event_store_sha256 = "0".repeat(64);
        let bytes = serde_json::to_vec(&predictions).unwrap();
        assert!(evaluate_prediction_bytes(&root, &bytes).unwrap_err().contains("artifact pin"));

        fs::remove_dir_all(root).expect("temporary dataset should be removed");
    }

    #[test]
    fn evaluator_rejects_overlapping_abstention_and_missing_evidence() {
        let (root, manifest, oracle) = fixture_root();
        let mut predictions = prediction_set(&root, &manifest, &oracle, true);
        predictions.abstained_record_ids.push(predictions.suspects[0].record_id.clone());
        let bytes = serde_json::to_vec(&predictions).unwrap();
        assert!(evaluate_prediction_bytes(&root, &bytes).unwrap_err().contains("both suspected and abstained"));

        let mut predictions = prediction_set(&root, &manifest, &oracle, true);
        predictions.suspects[0].evidence_refs = vec!["evt-99999999".to_string()];
        let bytes = serde_json::to_vec(&predictions).unwrap();
        assert!(evaluate_prediction_bytes(&root, &bytes).unwrap_err().contains("unknown evidence ID"));
        fs::remove_dir_all(root).expect("temporary dataset should be removed");
    }
}

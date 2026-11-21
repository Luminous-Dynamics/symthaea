use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const CONTRACT: &str = "SC-PRECURSOR-GENOME-GRAPH-SCORE-0.1";
const CLAIM_CEILING: &str = "Synthetic cold-start graph scoring contract only. No real model performance, synthesis recommendation, phase yield, or material discovery is established.";

#[derive(Clone, Debug)]
struct Record {
    id: String,
    sample_id: String,
    precursor_a: String,
    precursor_b: String,
    category: String,
    included: bool,
    label: Option<u8>,
}
#[derive(Clone, Debug)]
struct Graph {
    nodes: Vec<String>,
    records: Vec<Record>,
    included_count: usize,
    positive_count: usize,
    negative_count: usize,
}
#[derive(Clone, Copy, Debug)]
struct Prediction {
    probability: f64,
    label: u8,
}
#[derive(Clone, Copy, Debug)]
struct Metrics {
    brier: f64,
    log_loss: f64,
    roc_auc: f64,
    average_precision: f64,
    balanced_accuracy: f64,
    precision: f64,
    recall: f64,
    specificity: f64,
    mcc: Option<f64>,
}
#[derive(Clone, Debug)]
struct FoldReceipt {
    json: Value,
    metrics: Metrics,
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {key}"))
}
fn required_number(value: &Value, key: &str) -> Result<f64, String> {
    let number = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !number.is_finite() {
        return Err(format!("{key} must be finite"));
    }
    Ok(number)
}
fn strings(value: &Value, key: &str) -> Result<Vec<String>, String> {
    let values = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))?;
    let mut output = Vec::with_capacity(values.len());
    let mut seen = HashSet::new();
    for item in values {
        let text = item.as_str().filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{key} contains a non-string or empty value"))?;
        if !seen.insert(text.to_owned()) {
            return Err(format!("{key} contains duplicate value {text}"));
        }
        output.push(text.to_owned());
    }
    Ok(output)
}
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn sha256_ids(ids: &[String]) -> String {
    let mut sorted = ids.to_vec();
    sorted.sort();
    sha256_hex(sorted.join("\n").as_bytes())
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn category_label(category: &str) -> Option<u8> {
    match category {
        "completely_reacted" | "partially_reacted" => Some(1),
        "transformed" | "unreacted" => Some(0),
        _ => None,
    }
}
fn parse_graph(cohort: &Value) -> Result<Graph, String> {
    let nodes = strings(cohort, "nodes")?;
    if nodes.len() < 3 || nodes.len() > 46 {
        return Err(format!("node count must be in [3,46], found {}", nodes.len()));
    }
    let node_set = nodes.iter().cloned().collect::<BTreeSet<_>>();
    let rows = cohort.get("records").and_then(Value::as_array)
        .ok_or("cohort records must be an array")?;
    let expected_edges = nodes.len() * (nodes.len() - 1) / 2;
    if rows.len() != expected_edges {
        return Err(format!(
            "complete undirected graph requires {expected_edges} records for {} nodes, found {}",
            nodes.len(), rows.len()
        ));
    }
    let mut records = Vec::with_capacity(rows.len());
    let mut record_ids = HashSet::new();
    let mut sample_ids = HashSet::new();
    let mut edge_set = HashSet::new();
    let mut included_count = 0;
    let mut positive_count = 0;
    let mut negative_count = 0;
    for row in rows {
        let id = required_text(row, "record_id")?.to_owned();
        let sample_id = required_text(row, "sample_id")?.to_owned();
        let precursor_a = required_text(row, "precursor_a")?.to_owned();
        let precursor_b = required_text(row, "precursor_b")?.to_owned();
        let category = required_text(row, "category")?.to_owned();
        let state = required_text(row, "cohort_state")?;
        if !record_ids.insert(id.clone()) {
            return Err(format!("duplicate record_id {id}"));
        }
        if !sample_ids.insert(sample_id.clone()) {
            return Err(format!("duplicate sample_id {sample_id}"));
        }
        if precursor_a >= precursor_b {
            return Err(format!(
                "record {id} precursor pair must be canonical with precursor_a < precursor_b"
            ));
        }
        if !node_set.contains(&precursor_a) || !node_set.contains(&precursor_b) {
            return Err(format!("record {id} references a precursor outside the frozen node set"));
        }
        if !edge_set.insert((precursor_a.clone(), precursor_b.clone())) {
            return Err(format!("duplicate unordered precursor edge {precursor_a} / {precursor_b}"));
        }
        let label = match row.get("label") {
            Some(value) if value.is_null() => None,
            Some(value) => {
                let number = value.as_u64().ok_or_else(|| format!("record {id} label must be 0, 1, or null"))?;
                if number > 1 {
                    return Err(format!("record {id} label must be 0 or 1"));
                }
                Some(number as u8)
            }
            None => return Err(format!("record {id} is missing label")),
        };
        let included = match state {
            "included" => {
                let expected = category_label(&category)
                    .ok_or_else(|| format!("record {id} has unsupported included category {category}"))?;
                if label != Some(expected) {
                    return Err(format!("record {id} category {category} conflicts with label"));
                }
                included_count += 1;
                if expected == 1 { positive_count += 1; } else { negative_count += 1; }
                true
            }
            "physical_failure_excluded" | "poor_refinement_excluded" => {
                if label.is_some() {
                    return Err(format!("excluded record {id} must have a null label"));
                }
                required_text(row, "exclusion_reason")?;
                // Preserve the upstream category as descriptive source data,
                // but never map an excluded specimen into the scored endpoint.
                false
            }
            _ => return Err(format!("record {id} has unsupported cohort_state {state}")),
        };
        records.push(Record {
            id, sample_id, precursor_a, precursor_b, category, included, label
        });
    }
    if edge_set.len() != expected_edges {
        return Err(format!("graph has {} unique edges; expected {expected_edges}", edge_set.len()));
    }
    for (index, left) in nodes.iter().enumerate() {
        for right in nodes.iter().skip(index + 1) {
            if !edge_set.contains(&(left.clone(), right.clone())) {
                return Err(format!("complete graph is missing edge {left} / {right}"));
            }
        }
    }
    if included_count == 0 || positive_count == 0 || negative_count == 0 {
        return Err("included cohort must contain both positive and negative labels".to_owned());
    }
    Ok(Graph {
        nodes: node_set.into_iter().collect(),
        records,
        included_count,
        positive_count,
        negative_count,
    })
}
fn fold_id(kind: &str, heldout: &[String]) -> String {
    if kind == "lopo" {
        format!("lopo:{}", heldout[0])
    } else {
        format!("double:{}|{}", heldout[0], heldout[1])
    }
}
fn expected_training_nodes(graph: &Graph, heldout: &BTreeSet<String>) -> Vec<String> {
    graph.nodes.iter().filter(|node| !heldout.contains(*node)).cloned().collect()
}
fn expected_training_ids(graph: &Graph, heldout: &BTreeSet<String>) -> Vec<String> {
    let mut ids = graph.records.iter()
        .filter(|record| record.included
            && !heldout.contains(&record.precursor_a)
            && !heldout.contains(&record.precursor_b))
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}
fn expected_test_records<'a>(graph: &'a Graph, heldout: &BTreeSet<String>) -> Vec<&'a Record> {
    graph.records.iter()
        .filter(|record| record.included
            && (heldout.contains(&record.precursor_a) || heldout.contains(&record.precursor_b)))
        .collect()
}
fn metric_values(predictions: &[Prediction], threshold: f64) -> Result<Metrics, String> {
    let positive = predictions.iter().filter(|row| row.label == 1).count();
    let negative = predictions.len() - positive;
    if positive == 0 || negative == 0 {
        return Err("each scoring fold must contain both positive and negative test records".to_owned());
    }
    let mut brier = 0.0;
    let mut log_loss = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut tn = 0.0;
    let mut fn_count = 0.0;
    let epsilon = 1.0e-15;
    for row in predictions {
        let p = row.probability.clamp(epsilon, 1.0 - epsilon);
        let y = f64::from(row.label);
        brier += (row.probability - y).powi(2);
        log_loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        match (row.label == 1, row.probability >= threshold) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (false, false) => tn += 1.0,
            (true, false) => fn_count += 1.0,
        }
    }
    let count = predictions.len() as f64;
    brier /= count;
    log_loss /= count;
    let recall = tp / (tp + fn_count);
    let specificity = tn / (tn + fp);
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let balanced_accuracy = (recall + specificity) / 2.0;
    let denominator = ((tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count)).sqrt();
    let mcc = if denominator > 0.0 {
        Some((tp * tn - fp * fn_count) / denominator)
    } else {
        None
    };
    let positives = predictions.iter().filter(|row| row.label == 1)
        .map(|row| row.probability).collect::<Vec<_>>();
    let negatives = predictions.iter().filter(|row| row.label == 0)
        .map(|row| row.probability).collect::<Vec<_>>();
    let mut auc = 0.0;
    for p in &positives {
        for n in &negatives {
            if p > n { auc += 1.0; }
            else if p == n { auc += 0.5; }
        }
    }
    auc /= (positive * negative) as f64;
    let mut ranked = predictions.to_vec();
    ranked.sort_by(|left, right| right.probability.total_cmp(&left.probability));
    let mut average_precision = 0.0;
    let mut cumulative_positive = 0usize;
    let mut cumulative_total = 0usize;
    let mut index = 0usize;
    while index < ranked.len() {
        let score = ranked[index].probability;
        let mut group_positive = 0usize;
        let mut group_total = 0usize;
        while index < ranked.len() && ranked[index].probability == score {
            group_total += 1;
            if ranked[index].label == 1 { group_positive += 1; }
            index += 1;
        }
        cumulative_positive += group_positive;
        cumulative_total += group_total;
        if group_positive > 0 {
            let precision_at_group = cumulative_positive as f64 / cumulative_total as f64;
            average_precision += (group_positive as f64 / positive as f64) * precision_at_group;
        }
    }
    Ok(Metrics {
        brier, log_loss, roc_auc: auc, average_precision,
        balanced_accuracy, precision, recall, specificity, mcc
    })
}
fn metrics_json(metrics: Metrics) -> Value {
    json!({
        "brier_score": metrics.brier,
        "log_loss": metrics.log_loss,
        "roc_auc": metrics.roc_auc,
        "average_precision": metrics.average_precision,
        "balanced_accuracy": metrics.balanced_accuracy,
        "precision_at_fixed_threshold": metrics.precision,
        "recall_at_fixed_threshold": metrics.recall,
        "specificity_at_fixed_threshold": metrics.specificity,
        "matthews_correlation_coefficient": metrics.mcc
    })
}
fn evaluate_fold(
    graph: &Graph,
    fold: &Value,
    kind: &str,
    expected_heldout: &[String],
    threshold: f64,
) -> Result<FoldReceipt, String> {
    let id = required_text(fold, "fold_id")?;
    let expected_id = fold_id(kind, expected_heldout);
    if id != expected_id {
        return Err(format!("expected fold_id {expected_id}, got {id}"));
    }
    let heldout = strings(fold, "heldout_precursors")?;
    if heldout != expected_heldout {
        return Err(format!("fold {id} heldout_precursors do not match the frozen split"));
    }
    let heldout_set = heldout.iter().cloned().collect::<BTreeSet<_>>();
    let training_nodes = strings(fold, "training_precursors")?;
    let expected_nodes = expected_training_nodes(graph, &heldout_set);
    if training_nodes != expected_nodes {
        return Err(format!("fold {id} has train/test precursor leakage or an incomplete training node set"));
    }
    let training_ids = expected_training_ids(graph, &heldout_set);
    let expected_training_count = training_ids.len();
    let declared_count = fold.get("training_record_count").and_then(Value::as_u64)
        .ok_or_else(|| format!("fold {id} training_record_count must be an integer"))? as usize;
    if declared_count != expected_training_count {
        return Err(format!(
            "fold {id} training_record_count is {declared_count}; expected {expected_training_count}"
        ));
    }
    let test_records = expected_test_records(graph, &heldout_set);
    let rows = fold.get("predictions").and_then(Value::as_array)
        .ok_or_else(|| format!("fold {id} predictions must be an array"))?;
    let expected_test_ids = test_records.iter().map(|record| record.id.clone()).collect::<HashSet<_>>();
    if rows.len() != expected_test_ids.len() {
        return Err(format!(
            "fold {id} prediction count is {}; expected {} eligible held-out edges",
            rows.len(), expected_test_ids.len()
        ));
    }
    let labels = test_records.iter().map(|record| (record.id.clone(), record.label.unwrap())).collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut scored = Vec::with_capacity(rows.len());
    for row in rows {
        let record_id = required_text(row, "record_id")?.to_owned();
        if !expected_test_ids.contains(&record_id) {
            return Err(format!("fold {id} prediction {record_id} is not an eligible held-out edge"));
        }
        if !seen.insert(record_id.clone()) {
            return Err(format!("fold {id} has duplicate prediction for {record_id}"));
        }
        let probability = required_number(row, "probability")?;
        if !(0.0..=1.0).contains(&probability) {
            return Err(format!("fold {id} prediction {record_id} probability must be in [0,1]"));
        }
        scored.push(Prediction {
            probability,
            label: *labels.get(&record_id).expect("expected test record label"),
        });
    }
    if seen != expected_test_ids {
        return Err(format!("fold {id} does not predict every eligible held-out edge exactly once"));
    }
    let metrics = metric_values(&scored, threshold)?;
    Ok(FoldReceipt {
        json: json!({
            "fold_id": id,
            "heldout_precursors": heldout,
            "training_precursors": training_nodes,
            "training_record_count": expected_training_count,
            "training_record_ids_sha256": sha256_ids(&training_ids),
            "test_record_count": test_records.len(),
            "test_positive_count": scored.iter().filter(|row| row.label == 1).count(),
            "test_negative_count": scored.iter().filter(|row| row.label == 0).count(),
            "test_record_ids_sha256": sha256_ids(&test_records.iter().map(|record| record.id.clone()).collect::<Vec<_>>()),
            "metrics": metrics_json(metrics)
        }),
        metrics,
    })
}
fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let collected = values.collect::<Vec<_>>();
    if collected.is_empty() { None } else {
        Some(collected.iter().sum::<f64>() / collected.len() as f64)
    }
}
fn summarize(folds: &[FoldReceipt], kind: &str) -> Value {
    let metrics = folds.iter().map(|fold| fold.metrics).collect::<Vec<_>>();
    json!({
        "split": kind,
        "fold_count": folds.len(),
        "mean_test_records_per_fold": mean(folds.iter().map(|fold| fold.json["test_record_count"].as_u64().unwrap() as f64)),
        "mean_fold_brier_score": mean(metrics.iter().map(|m| m.brier)),
        "worst_fold_brier_score": metrics.iter().map(|m| m.brier).fold(0.0, f64::max),
        "mean_fold_log_loss": mean(metrics.iter().map(|m| m.log_loss)),
        "worst_fold_log_loss": metrics.iter().map(|m| m.log_loss).fold(0.0, f64::max),
        "mean_fold_roc_auc": mean(metrics.iter().map(|m| m.roc_auc)),
        "mean_fold_average_precision": mean(metrics.iter().map(|m| m.average_precision)),
        "mean_fold_balanced_accuracy": mean(metrics.iter().map(|m| m.balanced_accuracy)),
        "mean_fold_mcc": mean(metrics.iter().filter_map(|m| m.mcc)),
        "mcc_defined_fold_count": metrics.iter().filter(|m| m.mcc.is_some()).count(),
        "aggregation": "unweighted mean across folds; folds overlap and are not independent samples"
    })
}
fn evaluate_model(graph: &Graph, model: &Value) -> Result<Value, String> {
    let model_id = required_text(model, "model_id")?.to_owned();
    let artifact = required_text(model, "model_artifact_sha256")?;
    if !valid_digest(artifact) {
        return Err(format!("model {model_id} has malformed model_artifact_sha256"));
    }
    let threshold = required_number(model, "decision_threshold")?;
    if !(0.0..1.0).contains(&threshold) {
        return Err(format!("model {model_id} decision_threshold must be in [0,1)"));
    }
    let lopo_rows = model.get("lopo_folds").and_then(Value::as_array)
        .ok_or_else(|| format!("model {model_id} lopo_folds must be an array"))?;
    if lopo_rows.len() != graph.nodes.len() {
        return Err(format!("model {model_id} requires exactly {} LOPO folds", graph.nodes.len()));
    }
    let mut lopo_by_id = HashMap::new();
    for row in lopo_rows {
        let id = required_text(row, "fold_id")?.to_owned();
        if lopo_by_id.insert(id.clone(), row).is_some() {
            return Err(format!("model {model_id} has duplicate LOPO fold {id}"));
        }
    }
    let mut lopo_folds = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let id = fold_id("lopo", std::slice::from_ref(node));
        let row = lopo_by_id.get(&id).ok_or_else(|| format!("model {model_id} is missing LOPO fold {id}"))?;
        lopo_folds.push(evaluate_fold(graph, row, "lopo", std::slice::from_ref(node), threshold)?);
    }
    let expected_double_count = graph.nodes.len() * (graph.nodes.len() - 1) / 2;
    let double_rows = model.get("double_node_folds").and_then(Value::as_array)
        .ok_or_else(|| format!("model {model_id} double_node_folds must be an array"))?;
    if double_rows.len() != expected_double_count {
        return Err(format!("model {model_id} requires exactly {expected_double_count} double-node folds"));
    }
    let mut double_by_id = HashMap::new();
    for row in double_rows {
        let id = required_text(row, "fold_id")?.to_owned();
        if double_by_id.insert(id.clone(), row).is_some() {
            return Err(format!("model {model_id} has duplicate double-node fold {id}"));
        }
    }
    let mut double_folds = Vec::with_capacity(expected_double_count);
    for i in 0..graph.nodes.len() {
        for j in i + 1..graph.nodes.len() {
            let heldout = vec![graph.nodes[i].clone(), graph.nodes[j].clone()];
            let id = fold_id("double", &heldout);
            let row = double_by_id.get(&id).ok_or_else(|| format!("model {model_id} is missing double-node fold {id}"))?;
            double_folds.push(evaluate_fold(graph, row, "double_node", &heldout, threshold)?);
        }
    }
    Ok(json!({
        "model_id": model_id,
        "model_artifact_sha256": artifact,
        "decision_threshold": threshold,
        "lopo": summarize(&lopo_folds, "leave_one_precursor_out"),
        "double_node_holdout": summarize(&double_folds, "two_precursor_cold_start"),
        "lopo_folds": lopo_folds.iter().map(|fold| fold.json.clone()).collect::<Vec<_>>(),
        "double_node_folds": double_folds.iter().map(|fold| fold.json.clone()).collect::<Vec<_>>()
    }))
}
fn valid_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn graph_from_audit_receipt(receipt: &Value) -> Result<Graph, String> {
    if required_text(receipt, "status")? != "precursor_genome_graph_audit_pass"
        || required_text(receipt, "contract")? != "SC-PRECURSOR-GENOME-GRAPH-AUDIT-0.1"
    {
        return Err("input is not a passing Precursor Genome graph audit receipt".to_owned());
    }
    let source = receipt.get("source").ok_or("graph receipt source is missing")?;
    if required_text(source, "repository")? != "lauren-walters/precursor-genome" {
        return Err("graph receipt is not from the pinned Precursor Genome repository".to_owned());
    }
    let ledger_commit = required_text(source, "commit")?;
    if ledger_commit != "78cb026aa6c9b53aa0ec26ea5611d523bc3c320a" {
        return Err(format!("unexpected Precursor Genome source commit {ledger_commit}"));
    }
    if !valid_sha1(required_text(source, "ledger_git_blob_sha1")?) {
        return Err("graph receipt ledger_git_blob_sha1 is malformed".to_owned());
    }
    if !valid_digest(required_text(source, "ledger_sha256")?) {
        return Err("graph receipt ledger_sha256 is malformed".to_owned());
    }
    let pair_graph = receipt.get("pair_graph").ok_or("graph receipt pair_graph is missing")?;
    if pair_graph.get("complete_undirected_graph").and_then(Value::as_bool) != Some(true) {
        return Err("source graph receipt does not establish a complete undirected graph".to_owned());
    }
    let precursor_rows = pair_graph.get("per_precursor").and_then(Value::as_array)
        .ok_or("graph receipt pair_graph.per_precursor must be an array")?;
    let mut nodes = Vec::with_capacity(precursor_rows.len());
    for row in precursor_rows {
        nodes.push(required_text(row, "precursor_formula")?.to_owned());
    }
    nodes.sort();
    let expected_edges = nodes.len() * (nodes.len() - 1) / 2;
    if pair_graph.get("precursor_count").and_then(Value::as_u64) != Some(nodes.len() as u64)
        || pair_graph.get("unique_unordered_pair_count").and_then(Value::as_u64) != Some(expected_edges as u64)
        || pair_graph.get("expected_degree").and_then(Value::as_u64) != Some((nodes.len() - 1) as u64)
    {
        return Err("graph receipt node, edge or degree counts do not describe a complete graph".to_owned());
    }
    let source_rows = receipt.get("sample_records").and_then(Value::as_array)
        .ok_or("graph receipt sample_records must be an array")?;
    let mut records = Vec::with_capacity(source_rows.len());
    for row in source_rows {
        let sample_id = required_text(row, "sample_id")?.to_owned();
        let pair = row.get("precursor_pair_canonical").and_then(Value::as_array)
            .ok_or_else(|| format!("source sample {sample_id} lacks canonical precursor pair"))?;
        if pair.len() != 2 {
            return Err(format!("source sample {sample_id} precursor pair must have exactly two values"));
        }
        let precursor_a = pair[0].as_str().ok_or("canonical precursor A must be string")?.to_owned();
        let precursor_b = pair[1].as_str().ok_or("canonical precursor B must be string")?.to_owned();
        let category = row.get("raw_reaction_category").and_then(Value::as_str)
            .unwrap_or("unclassified").to_owned();
        let state = required_text(row, "cohort_status")?;
        let (cohort_state, label, exclusion_reason) = match state {
            "analyzable" => {
                let endpoint = row.get("cross_precursor_product_endpoint").and_then(Value::as_bool)
                    .ok_or_else(|| format!("analyzable sample {sample_id} lacks binary cross-precursor endpoint"))?;
                ("included", json!(u8::from(endpoint)), None)
            }
            "physical_failure_no_powder" => ("physical_failure_excluded", Value::Null, Some(state)),
            "poor_xrd_refinement_quality_3" => ("poor_refinement_excluded", Value::Null, Some(state)),
            other => return Err(format!("source sample {sample_id} has unsupported cohort status {other}")),
        };
        let mut item = json!({
            "record_id": sample_id,
            "sample_id": sample_id,
            "precursor_a": precursor_a,
            "precursor_b": precursor_b,
            "category": category,
            "cohort_state": cohort_state,
            "label": label
        });
        if let Some(reason) = exclusion_reason {
            item["exclusion_reason"] = json!(reason);
        }
        records.push(item);
    }
    let cohort_value = json!({
        "nodes": nodes,
        "records": records,
        "source_snapshot_sha256": required_text(source, "ledger_sha256")?,
        "graph_receipt_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    });
    let graph = parse_graph(&cohort_value)?;
    let summary = receipt.get("cohort").ok_or("graph receipt cohort summary is missing")?;
    for (field, observed) in [
        ("source_sample_count", graph.records.len()),
        ("analyzable_count", graph.included_count),
        ("cross_precursor_product_positive_count", graph.positive_count),
        ("cross_precursor_product_negative_count", graph.negative_count),
    ] {
        if summary.get(field).and_then(Value::as_u64) != Some(observed as u64) {
            return Err(format!("graph receipt cohort.{field} conflicts with source sample records"));
        }
    }
    Ok(graph)
}
fn training_prevalence_fold(graph: &Graph, kind: &str, heldout: &[String]) -> Result<Value, String> {
    let heldout_set = heldout.iter().cloned().collect::<BTreeSet<_>>();
    let training_ids = expected_training_ids(graph, &heldout_set);
    if training_ids.is_empty() {
        return Err(format!("fold {} has no eligible training records", fold_id(kind, heldout)));
    }
    let training_id_set = training_ids.iter().cloned().collect::<HashSet<_>>();
    let positives = graph.records.iter()
        .filter(|record| training_id_set.contains(&record.id) && record.label == Some(1))
        .count();
    let probability = positives as f64 / training_ids.len() as f64;
    let training_nodes = expected_training_nodes(graph, &heldout_set);
    let predictions = expected_test_records(graph, &heldout_set).iter()
        .map(|record| json!({"record_id": record.id, "probability": probability}))
        .collect::<Vec<_>>();
    Ok(json!({
        "fold_id": fold_id(kind, heldout),
        "heldout_precursors": heldout,
        "training_precursors": training_nodes,
        "training_record_count": training_ids.len(),
        "predictions": predictions
    }))
}
fn training_prevalence_model(graph: &Graph, source_identity: &str) -> Result<Value, String> {
    let mut lopo = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        lopo.push(training_prevalence_fold(graph, "lopo", std::slice::from_ref(node))?);
    }
    let mut double_node = Vec::with_capacity(graph.nodes.len() * (graph.nodes.len() - 1) / 2);
    for i in 0..graph.nodes.len() {
        for j in i + 1..graph.nodes.len() {
            let heldout = vec![graph.nodes[i].clone(), graph.nodes[j].clone()];
            double_node.push(training_prevalence_fold(graph, "double", &heldout)?);
        }
    }
    let artifact_identity = format!("training-prevalence-baseline-v0.1:{source_identity}");
    Ok(json!({
        "model_id": "fold_local_training_prevalence_baseline",
        "model_artifact_sha256": sha256_hex(artifact_identity.as_bytes()),
        "decision_threshold": 0.5,
        "lopo_folds": lopo,
        "double_node_folds": double_node
    }))
}
fn audit_graph_receipt(receipt_bytes: &[u8]) -> Result<Value, String> {
    let receipt: Value = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("graph receipt JSON parse failed: {error}"))?;
    let graph = graph_from_audit_receipt(&receipt)?;
    let receipt_sha256 = sha256_hex(receipt_bytes);
    let source = receipt.get("source").ok_or("graph receipt source is missing")?;
    let model = training_prevalence_model(&graph, &receipt_sha256)?;
    let evaluation = evaluate_model(&graph, &model)?;
    let prevalence = graph.positive_count as f64 / graph.included_count as f64;
    let global_baseline_brier = prevalence * (1.0 - prevalence);
    let global_baseline_log_loss = -prevalence * prevalence.ln()
        - (1.0 - prevalence) * (1.0 - prevalence).ln();
    Ok(json!({
        "status": "precursor_genome_training_prevalence_baseline_pass",
        "contract": CONTRACT,
        "benchmark_id": "precursor_genome_k46_fold_local_training_prevalence_baseline_001",
        "source": {
            "repository": required_text(source, "repository")?,
            "commit": required_text(source, "commit")?,
            "ledger_git_blob_sha1": required_text(source, "ledger_git_blob_sha1")?,
            "ledger_sha256": required_text(source, "ledger_sha256")?,
            "license": required_text(source, "license")?
        },
        "graph_receipt_sha256": receipt_sha256,
        "graph": {
            "precursor_count": graph.nodes.len(),
            "complete_undirected_edge_count": graph.records.len(),
            "included_edge_count": graph.included_count,
            "excluded_edge_count": graph.records.len() - graph.included_count,
            "positive_count": graph.positive_count,
            "negative_count": graph.negative_count,
            "positive_prevalence": prevalence
        },
        "global_constant_prevalence_reference": {
            "brier_score": global_baseline_brier,
            "log_loss": global_baseline_log_loss,
            "roc_auc": 0.5,
            "average_precision": prevalence,
            "note": "Uses the full evaluation cohort prevalence and is a descriptive no-skill reference, not a fold-valid prediction."
        },
        "fold_local_training_prevalence_baseline": evaluation,
        "claim_ceiling": "A deterministic, fold-local prevalence baseline replayed on the source-pinned Precursor Genome graph. This is not a learned predictor, does not establish model superiority, and makes no new synthesis or material claim."
    }))
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!("graph score schema validation failed: {error} at {}", error.instance_path()));
    }
    if required_text(&fixture, "contract")? != CONTRACT {
        return Err("unsupported precursor graph score contract".to_owned());
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("v0.1 graph score auditor accepts synthetic fixtures only".to_owned());
    }
    let cohort = fixture.get("cohort").ok_or("fixture cohort is missing")?;
    let graph = parse_graph(cohort)?;
    let models = fixture.get("models").and_then(Value::as_array).ok_or("models must be an array")?;
    if models.is_empty() {
        return Err("at least one model is required".to_owned());
    }
    let mut model_ids = HashSet::new();
    let mut model_receipts = Vec::with_capacity(models.len());
    for model in models {
        let id = required_text(model, "model_id")?.to_owned();
        if !model_ids.insert(id.clone()) {
            return Err(format!("duplicate model_id {id}"));
        }
        model_receipts.push(evaluate_model(&graph, model)?);
    }
    let prevalence = graph.positive_count as f64 / graph.included_count as f64;
    let no_skill_brier = prevalence * (1.0 - prevalence);
    let no_skill_log_loss = -prevalence * prevalence.ln() - (1.0 - prevalence) * (1.0 - prevalence).ln();
    Ok(json!({
        "status": "synthetic_precursor_graph_score_pass",
        "contract": CONTRACT,
        "benchmark_id": required_text(&fixture, "benchmark_id")?,
        "source_snapshot_sha256": required_text(cohort, "source_snapshot_sha256")?,
        "graph_receipt_sha256": required_text(cohort, "graph_receipt_sha256")?,
        "schema_sha256": sha256_hex(schema_bytes),
        "fixture_sha256": sha256_hex(fixture_bytes),
        "graph": {
            "precursor_count": graph.nodes.len(),
            "complete_undirected_edge_count": graph.records.len(),
            "included_edge_count": graph.included_count,
            "excluded_edge_count": graph.records.len() - graph.included_count,
            "positive_count": graph.positive_count,
            "negative_count": graph.negative_count,
            "positive_prevalence": prevalence
        },
        "no_skill_baseline": {
            "brier_score": no_skill_brier,
            "log_loss": no_skill_log_loss,
            "roc_auc": 0.5,
            "average_precision": prevalence,
            "description": "Constant probability equal to the evaluated cohort prevalence; ranking is random."
        },
        "models": model_receipts,
        "claim_ceiling": CLAIM_CEILING
    }))
}
fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]");
    eprintln!("   or: {program} --graph-receipt <source-audit-receipt.json> [--out <receipt.json>]");
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut graph_receipt_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--graph-receipt" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--graph-receipt" => graph_receipt_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let receipt = match graph_receipt_path {
        Some(path) => {
            if schema_path.is_some() || fixture_path.is_some() {
                eprintln!("--graph-receipt cannot be combined with --schema or --fixture");
                usage(&args[0]);
                process::exit(2);
            }
            audit_graph_receipt(&read(&path))
        }
        None => match (schema_path, fixture_path) {
            (Some(schema), Some(fixture)) => audit(&read(&schema), &read(&fixture)),
            _ => {
                usage(&args[0]);
                process::exit(2);
            }
        },
    }.unwrap_or_else(|error| {
        eprintln!("SC-PRECURSOR-GRAPH-SCORE-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/precursor_genome_graph_score_v0_1.schema.json")).unwrap(),
            fs::read(root.join("tools/superconductivity-discovery-harness/fixtures/precursor_genome_graph_score_smoke_v0_1.json")).unwrap(),
        )
    }
    fn mutate_fixture(change: impl FnOnce(&mut Value)) -> Result<Value, String> {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        change(&mut value);
        audit(&schema, &serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn smoke_fixture_scores_every_lopo_and_double_node_fold() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        assert_eq!(receipt["graph"]["precursor_count"], 4);
        assert_eq!(receipt["graph"]["complete_undirected_edge_count"], 6);
        assert_eq!(receipt["models"][0]["lopo"]["fold_count"], 4);
        assert_eq!(receipt["models"][0]["double_node_holdout"]["fold_count"], 6);
        assert_eq!(receipt["models"][0]["lopo"]["mean_fold_roc_auc"], 1.0);
        assert_eq!(receipt["models"][0]["double_node_holdout"]["mean_fold_average_precision"], 1.0);
    }

    #[test]
    fn heldout_precursor_cannot_appear_in_training_nodes() {
        let error = mutate_fixture(|value| {
            value["models"][0]["lopo_folds"][0]["training_precursors"].as_array_mut().unwrap().push(json!("A"));
        }).unwrap_err();
        assert!(error.contains("train/test precursor leakage"));
    }

    #[test]
    fn missing_heldout_prediction_is_rejected() {
        let error = mutate_fixture(|value| {
            value["models"][0]["lopo_folds"][0]["predictions"].as_array_mut().unwrap().pop();
        }).unwrap_err();
        assert!(error.contains("prediction count"));
    }

    #[test]
    fn prediction_for_an_unheld_edge_is_rejected() {
        let error = mutate_fixture(|value| {
            let mut row = value["models"][0]["lopo_folds"][0]["predictions"][0].clone();
            row["record_id"] = json!("edge_cd");
            value["models"][0]["lopo_folds"][0]["predictions"].as_array_mut().unwrap().push(row);
        }).unwrap_err();
        assert!(error.contains("prediction count") || error.contains("not an eligible held-out edge"));
    }

    #[test]
    fn duplicate_folds_are_rejected() {
        let error = mutate_fixture(|value| {
            let duplicate = value["models"][0]["lopo_folds"][0].clone();
            value["models"][0]["lopo_folds"].as_array_mut().unwrap().push(duplicate);
        }).unwrap_err();
        assert!(error.contains("requires exactly 4 LOPO folds") || error.contains("duplicate LOPO fold"));
    }

    #[test]
    fn incomplete_or_duplicate_graph_edges_are_rejected() {
        let error = mutate_fixture(|value| {
            value["cohort"]["records"].as_array_mut().unwrap().pop();
        }).unwrap_err();
        assert!(error.contains("complete undirected graph requires"));
    }

    #[test]
    fn invalid_probability_is_rejected() {
        let error = mutate_fixture(|value| {
            value["models"][0]["lopo_folds"][0]["predictions"][0]["probability"] = json!(1.5);
        }).unwrap_err();
        assert!(error.contains("probability must be in [0,1]"));
    }

    #[test]
    fn mismatched_training_record_count_is_rejected() {
        let error = mutate_fixture(|value| {
            value["models"][0]["lopo_folds"][0]["training_record_count"] = json!(0);
        }).unwrap_err();
        assert!(error.contains("training_record_count"));
    }

    fn synthetic_graph_receipt() -> Value {
        let (_, fixture) = inputs();
        let value: Value = serde_json::from_slice(&fixture).unwrap();
        let cohort = &value["cohort"];
        let nodes = cohort["nodes"].as_array().unwrap().clone();
        let records = cohort["records"].as_array().unwrap();
        let sample_records = records.iter().map(|row| json!({
            "sample_id": row["sample_id"],
            "precursor_pair_canonical": [row["precursor_a"], row["precursor_b"]],
            "cohort_status": "analyzable",
            "raw_reaction_category": row["category"],
            "cross_precursor_product_endpoint": row["label"] == 1
        })).collect::<Vec<_>>();
        json!({
            "status": "precursor_genome_graph_audit_pass",
            "contract": "SC-PRECURSOR-GENOME-GRAPH-AUDIT-0.1",
            "source": {
                "repository": "lauren-walters/precursor-genome",
                "commit": "78cb026aa6c9b53aa0ec26ea5611d523bc3c320a",
                "ledger_git_blob_sha1": "a".repeat(40),
                "ledger_sha256": "b".repeat(64),
                "license": "CC-BY-4.0"
            },
            "cohort": {
                "source_sample_count": records.len(),
                "analyzable_count": records.len(),
                "cross_precursor_product_positive_count": 3,
                "cross_precursor_product_negative_count": 3
            },
            "pair_graph": {
                "complete_undirected_graph": true,
                "precursor_count": nodes.len(),
                "unique_unordered_pair_count": records.len(),
                "expected_degree": nodes.len() - 1,
                "per_precursor": nodes.iter().map(|node| json!({
                    "precursor_formula": node
                })).collect::<Vec<_>>()
            },
            "sample_records": sample_records
        })
    }

    #[test]
    fn real_receipt_mode_parses_pinned_graph_and_scores_fold_local_prevalence() {
        let graph_receipt = synthetic_graph_receipt();
        let bytes = serde_json::to_vec(&graph_receipt).unwrap();
        let receipt = audit_graph_receipt(&bytes).unwrap();
        assert_eq!(receipt["graph"]["precursor_count"], 4);
        assert_eq!(receipt["graph"]["complete_undirected_edge_count"], 6);
        assert_eq!(receipt["fold_local_training_prevalence_baseline"]["lopo"]["fold_count"], 4);
        assert_eq!(receipt["fold_local_training_prevalence_baseline"]["double_node_holdout"]["fold_count"], 6);
        assert_eq!(receipt["fold_local_training_prevalence_baseline"]["lopo"]["mean_fold_roc_auc"], 0.5);
    }

    #[test]
    fn real_receipt_mode_rejects_wrong_pinned_source_revision() {
        let mut receipt = synthetic_graph_receipt();
        receipt["source"]["commit"] = json!("0000000000000000000000000000000000000000");
        let error = audit_graph_receipt(&serde_json::to_vec(&receipt).unwrap()).unwrap_err();
        assert!(error.contains("unexpected Precursor Genome source commit"));
    }

    #[test]
    fn fold_local_prevalence_is_computed_from_training_edges_only() {
        let (schema, fixture) = inputs();
        let data: Value = serde_json::from_slice(&fixture).unwrap();
        let graph = parse_graph(&data["cohort"]).unwrap();
        let model = training_prevalence_model(&graph, "synthetic-source").unwrap();
        let fold = model["lopo_folds"].as_array().unwrap().iter()
            .find(|fold| fold["fold_id"] == "lopo:A").unwrap();
        assert_eq!(fold["training_record_count"], 3);
        for prediction in fold["predictions"].as_array().unwrap() {
            assert_eq!(prediction["probability"], 1.0 / 3.0);
        }
        let _ = schema;
    }

}

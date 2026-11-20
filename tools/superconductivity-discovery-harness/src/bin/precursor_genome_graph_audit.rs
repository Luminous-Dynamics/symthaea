use jsonschema::validator_for;
use serde_json::{json, Value};
use sha1::{Digest, Sha1};
use sha2::Sha256;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const SOURCE_COMMIT: &str = "78cb026aa6c9b53aa0ec26ea5611d523bc3c320a";
const SOURCE_BLOB_SHA1: &str = "a8ef03cee29670b22ed04b9cebb1686fe26e1acd";
const CONTRACT: &str = "SC-PRECURSOR-GENOME-GRAPH-0.1";

#[derive(Clone, Debug, Default)]
struct NodeCounts {
    raw_pairs: u64,
    analyzable_pairs: u64,
    positive_pairs: u64,
    negative_pairs: u64,
    transformed_pairs: u64,
    unreacted_pairs: u64,
    completely_reacted_pairs: u64,
    partially_reacted_pairs: u64,
}
#[derive(Clone, Debug)]
struct ParsedSample {
    sample_id: String,
    pair: (String, String),
    category: Option<String>,
    quality_score: Option<u64>,
    has_physical_failure: bool,
    dwell_temperature_c: Option<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CohortStatus {
    PhysicalFailure,
    PoorRefinement,
    Analyzable,
    UnknownOutcome,
}
impl CohortStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::PhysicalFailure => "physical_failure_no_powder",
            Self::PoorRefinement => "poor_xrd_refinement_quality_3",
            Self::Analyzable => "analyzable",
            Self::UnknownOutcome => "unknown_or_unclassified_outcome",
        }
    }
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|x| !x.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn optional_number(value: &Value, key: &str) -> Result<Option<f64>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(item) => {
            let number = item.as_f64().ok_or_else(|| format!("{key} must be numeric or null"))?;
            if !number.is_finite() {
                return Err(format!("{key} must be finite"));
            }
            Ok(Some(number))
        }
    }
}
fn git_blob_sha1(bytes: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn canonical_pair(first: &str, second: &str) -> Result<(String, String), String> {
    if first.is_empty() || second.is_empty() {
        return Err("precursor formula must not be empty".to_owned());
    }
    if first == second {
        return Err(format!("self-pair detected for precursor {first}"));
    }
    if first < second {
        Ok((first.to_owned(), second.to_owned()))
    } else {
        Ok((second.to_owned(), first.to_owned()))
    }
}
fn category_label(category: &str) -> Option<bool> {
    match category {
        "completely_reacted" | "partially_reacted" => Some(true),
        "transformed" | "unreacted" => Some(false),
        _ => None,
    }
}
fn cohort_status(has_physical_failure: bool, quality_score: Option<u64>, category: Option<&str>) -> CohortStatus {
    // Physical handling failure takes precedence; a failed specimen cannot
    // become an XRD-quality exclusion or a non-reactive label.
    if has_physical_failure {
        return CohortStatus::PhysicalFailure;
    }
    if quality_score == Some(3) {
        return CohortStatus::PoorRefinement;
    }
    if category.and_then(category_label).is_some() {
        CohortStatus::Analyzable
    } else {
        CohortStatus::UnknownOutcome
    }
}
fn quality_score(sample: &Value) -> Option<u64> {
    let active_scan_index = sample.get("active_scan_index")?.as_u64()? as usize;
    let scans = sample.get("characterization")?.get("xrd")?.get("scans")?.as_array()?;
    let scan = scans.get(active_scan_index)?;
    let active_case_index = scan.get("active_case_index")?.as_u64()? as usize;
    let refinements = scan.get("refinement_cases")?.as_array()?;
    let refinement = refinements.get(active_case_index)?;
    refinement.get("verification")?.get("human_quality_score")?.as_u64()
}
fn parse_sample(value: &Value) -> Result<ParsedSample, String> {
    let sample_id = text_field(value, "sample_id")?.to_owned();
    let precursors = value.get("precursors").and_then(Value::as_array)
        .ok_or_else(|| format!("sample {sample_id} has no precursor array"))?;
    if precursors.len() != 2 {
        return Err(format!("sample {sample_id} must have exactly two precursors"));
    }
    let formula = |precursor: &Value| -> Result<String, String> {
        let cleaned = precursor.get("formula_clean").and_then(Value::as_str)
            .filter(|x| !x.trim().is_empty());
        let fallback = precursor.get("formula").and_then(Value::as_str)
            .filter(|x| !x.trim().is_empty());
        cleaned.or(fallback).map(str::to_owned)
            .ok_or_else(|| format!("sample {sample_id} precursor missing formula and formula_clean"))
    };
    let first = formula(&precursors[0])?;
    let second = formula(&precursors[1])?;
    let pair = canonical_pair(&first, &second)?;
    let category = value.get("outcome").and_then(|x| x.get("reaction_category"))
        .and_then(Value::as_str).map(str::to_owned);
    let has_physical_failure = value.get("physical_failure")
        .is_some_and(|x| !x.is_null());
    let quality_score = quality_score(value);
    if quality_score.is_some_and(|x| !(1..=3).contains(&x)) {
        return Err(format!("sample {sample_id} has an out-of-contract XRD quality score"));
    }
    let dwell_temperature_c = value.get("synthesis")
        .map(|synthesis| optional_number(synthesis, "heating_temperature"))
        .transpose()?.flatten();
    Ok(ParsedSample {
        sample_id,
        pair,
        category,
        quality_score,
        has_physical_failure,
        dwell_temperature_c,
    })
}
fn validate_complete_graph(
    nodes: &BTreeSet<String>,
    observed_edges: &HashSet<(String, String)>,
    expected_degree: usize,
) -> Result<BTreeMap<String, u64>, String> {
    if nodes.len() < 2 {
        return Err("pair graph must contain at least two precursor nodes".to_owned());
    }
    let mut degree: BTreeMap<String, u64> = nodes.iter()
        .map(|node| (node.clone(), 0)).collect();
    for (left, right) in observed_edges {
        if left >= right {
            return Err(format!("edge is not in canonical order: {left} / {right}"));
        }
        if !nodes.contains(left) || !nodes.contains(right) {
            return Err(format!("edge contains unknown precursor: {left} / {right}"));
        }
        *degree.get_mut(left).expect("node existence checked") += 1;
        *degree.get_mut(right).expect("node existence checked") += 1;
    }
    let expected_edges = nodes.len() * (nodes.len() - 1) / 2;
    if observed_edges.len() != expected_edges {
        return Err(format!(
            "complete graph requires {expected_edges} edges for {} precursors, found {}",
            nodes.len(), observed_edges.len()
        ));
    }
    for (index, left) in nodes.iter().enumerate() {
        for right in nodes.iter().skip(index + 1) {
            if !observed_edges.contains(&(left.clone(), right.clone())) {
                return Err(format!("missing precursor pair edge: {left} / {right}"));
            }
        }
    }
    if let Some((node, count)) = degree.iter().find(|(_, count)| **count != expected_degree as u64) {
        return Err(format!(
            "precursor {node} degree is {count}; expected {expected_degree}"
        ));
    }
    Ok(degree)
}
fn expected_u64(manifest: &Value, field: &str) -> Result<u64, String> {
    manifest.get("expected").and_then(|x| x.get(field)).and_then(Value::as_u64)
        .ok_or_else(|| format!("manifest expected.{field} must be an unsigned integer"))
}
fn expected_category_counts(manifest: &Value, group: &str) -> Result<BTreeMap<String, u64>, String> {
    let object = manifest.get("expected").and_then(|x| x.get(group))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("manifest expected.{group} must be an object"))?;
    object.iter().map(|(key, value)| {
        let count = value.as_u64().ok_or_else(|| {
            format!("manifest expected.{group}.{key} must be an unsigned integer")
        })?;
        Ok((key.clone(), count))
    }).collect()
}
fn count_map_json(map: &BTreeMap<String, u64>) -> Value {
    let mut output = serde_json::Map::new();
    for (key, value) in map {
        output.insert(key.clone(), json!(value));
    }
    Value::Object(output)
}
fn audit(schema_bytes: &[u8], manifest_bytes: &[u8], ledger_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|e| format!("manifest schema JSON parse failed: {e}"))?;
    let manifest: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|e| format!("manifest JSON parse failed: {e}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("manifest schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|e| format!("manifest schema compilation failed: {e}"))?;
    if let Some(error) = validator.iter_errors(&manifest).next() {
        return Err(format!(
            "precursor graph manifest validation failed: {error} at {}",
            error.instance_path()
        ));
    }
    if text_field(&manifest, "contract")? != CONTRACT {
        return Err("unsupported precursor graph benchmark contract".to_owned());
    }
    let source = manifest.get("source").ok_or("manifest source missing")?;
    let source_commit = text_field(source, "commit")?;
    let expected_blob = text_field(source, "ledger_git_blob_sha1")?;
    if source_commit != SOURCE_COMMIT || expected_blob != SOURCE_BLOB_SHA1 {
        return Err("manifest source commit/blob differs from the code-pinned release".to_owned());
    }
    let actual_blob = git_blob_sha1(ledger_bytes);
    if actual_blob != expected_blob {
        return Err(format!(
            "source ledger Git blob mismatch: expected {expected_blob}, received {actual_blob}"
        ));
    }
    let ledger: Value = serde_json::from_slice(ledger_bytes)
        .map_err(|e| format!("source ledger JSON parse failed: {e}"))?;
    let rows = ledger.get("samples").and_then(Value::as_array)
        .ok_or("source ledger.samples must be an array")?;
    let sample_count = expected_u64(&manifest, "sample_count")?;
    if rows.len() as u64 != sample_count {
        return Err(format!(
            "source sample count mismatch: expected {sample_count}, found {}",
            rows.len()
        ));
    }

    let mut parsed_samples = Vec::with_capacity(rows.len());
    let mut sample_ids = HashSet::new();
    let mut nodes = BTreeSet::new();
    let mut edges = HashSet::new();
    for row in rows {
        let sample = parse_sample(row)?;
        if !sample_ids.insert(sample.sample_id.clone()) {
            return Err(format!("duplicate sample_id {}", sample.sample_id));
        }
        nodes.insert(sample.pair.0.clone());
        nodes.insert(sample.pair.1.clone());
        if !edges.insert(sample.pair.clone()) {
            return Err(format!(
                "duplicate unordered precursor pair {} / {}",
                sample.pair.0, sample.pair.1
            ));
        }
        parsed_samples.push(sample);
    }
    let precursor_count = expected_u64(&manifest, "precursor_count")? as usize;
    let edge_count = expected_u64(&manifest, "edge_count")?;
    let expected_degree = expected_u64(&manifest, "degree_per_precursor")? as usize;
    if nodes.len() != precursor_count {
        return Err(format!(
            "precursor count mismatch: expected {precursor_count}, found {}",
            nodes.len()
        ));
    }
    if edges.len() as u64 != edge_count {
        return Err(format!("edge count mismatch: expected {edge_count}, found {}", edges.len()));
    }
    let raw_degrees = validate_complete_graph(&nodes, &edges, expected_degree)?;

    let mut raw_category_counts = BTreeMap::<String, u64>::new();
    let mut analyzable_category_counts = BTreeMap::<String, u64>::new();
    let mut exclusion_counts = BTreeMap::<String, u64>::new();
    let mut quality_counts = BTreeMap::<String, u64>::new();
    let mut dwell_counts = BTreeMap::<String, u64>::new();
    let mut node_counts: BTreeMap<String, NodeCounts> = nodes.iter()
        .map(|node| (node.clone(), NodeCounts::default())).collect();
    let mut analyzable_count = 0_u64;
    let mut positive_count = 0_u64;
    let mut negative_count = 0_u64;
    let mut dwell_count = 0_u64;
    let mut dwell_missing_count = 0_u64;
    let mut physical_failure_exclusions = 0_u64;
    let mut poor_refinement_exclusions = 0_u64;

    for sample in &parsed_samples {
        let category_key = sample.category.as_deref().unwrap_or("unclassified").to_owned();
        *raw_category_counts.entry(category_key).or_default() += 1;
        if let Some(score) = sample.quality_score {
            *quality_counts.entry(score.to_string()).or_default() += 1;
        } else {
            *quality_counts.entry("missing".to_owned()).or_default() += 1;
        }
        let status = cohort_status(
            sample.has_physical_failure,
            sample.quality_score,
            sample.category.as_deref(),
        );
        *exclusion_counts.entry(status.as_str().to_owned()).or_default() += 1;

        match status {
            CohortStatus::PhysicalFailure => {
                physical_failure_exclusions += 1;
                continue;
            }
            CohortStatus::PoorRefinement => {
                poor_refinement_exclusions += 1;
                continue;
            }
            CohortStatus::UnknownOutcome => {
                return Err(format!(
                    "sample {} has no supported outcome and no declared exclusion reason",
                    sample.sample_id
                ));
            }
            CohortStatus::Analyzable => {}
        }
        let category = sample.category.as_deref().expect("analyzable sample has a category");
        let positive = category_label(category).expect("analyzable category has a label");
        *analyzable_category_counts.entry(category.to_owned()).or_default() += 1;
        analyzable_count += 1;
        if positive {
            positive_count += 1;
        } else {
            negative_count += 1;
        }
        for node in [&sample.pair.0, &sample.pair.1] {
            let counts = node_counts.get_mut(node).expect("node collected from this sample");
            counts.analyzable_pairs += 1;
            if positive {
                counts.positive_pairs += 1;
            } else {
                counts.negative_pairs += 1;
            }
            match category {
                "transformed" => counts.transformed_pairs += 1,
                "unreacted" => counts.unreacted_pairs += 1,
                "completely_reacted" => counts.completely_reacted_pairs += 1,
                "partially_reacted" => counts.partially_reacted_pairs += 1,
                _ => unreachable!("category_label admits only known categories"),
            }
        }
        if let Some(dwell) = sample.dwell_temperature_c {
            dwell_count += 1;
            let key = format!("{dwell:.0}");
            *dwell_counts.entry(key).or_default() += 1;
        } else {
            dwell_missing_count += 1;
        }
    }
    for (node, degree) in &raw_degrees {
        let counts = node_counts.get_mut(node).expect("same node set");
        counts.raw_pairs = *degree;
    }

    let expected_physical = expected_u64(&manifest, "physical_failure_exclusions")?;
    let expected_poor = expected_u64(&manifest, "poor_refinement_exclusions")?;
    let expected_analyzable = expected_u64(&manifest, "analyzable_count")?;
    let expected_positive = expected_u64(&manifest, "positive_count")?;
    let expected_negative = expected_u64(&manifest, "negative_count")?;
    if physical_failure_exclusions != expected_physical {
        return Err(format!(
            "physical-failure exclusion count mismatch: expected {expected_physical}, found {physical_failure_exclusions}"
        ));
    }
    if poor_refinement_exclusions != expected_poor {
        return Err(format!(
            "poor-refinement exclusion count mismatch: expected {expected_poor}, found {poor_refinement_exclusions}"
        ));
    }
    if analyzable_count != expected_analyzable {
        return Err(format!(
            "analyzable cohort mismatch: expected {expected_analyzable}, found {analyzable_count}"
        ));
    }
    if positive_count != expected_positive || negative_count != expected_negative {
        return Err(format!(
            "label count mismatch: expected positive/negative {expected_positive}/{expected_negative}, found {positive_count}/{negative_count}"
        ));
    }
    let expected_raw = expected_category_counts(&manifest, "raw_category_counts")?;
    let expected_analyzable_categories = expected_category_counts(&manifest, "analyzable_category_counts")?;
    if raw_category_counts != expected_raw {
        return Err(format!(
            "raw outcome category counts differ from pinned cohort: expected {:?}, received {:?}",
            expected_raw, raw_category_counts
        ));
    }
    if analyzable_category_counts != expected_analyzable_categories {
        return Err(format!(
            "analyzable outcome category counts differ from paper cohort: expected {:?}, received {:?}",
            expected_analyzable_categories, analyzable_category_counts
        ));
    }
    if analyzable_count + physical_failure_exclusions + poor_refinement_exclusions != sample_count {
        return Err("cohort accounting does not close to the complete source ledger".to_owned());
    }

    // Export a compact, sample-indexed dataset view so downstream benchmarks
    // can preserve row provenance rather than attempting to reconstruct labels
    // from aggregate category counts. Raw XRD files remain external.
    let sample_records: Vec<Value> = rows.iter().zip(parsed_samples.iter()).map(|(row, sample)| {
        let precursors = row.get("precursors").and_then(Value::as_array);
        let precursor_records: Vec<Value> = precursors.into_iter().flatten().map(|precursor| {
            json!({
                "formula_clean": precursor.get("formula_clean").and_then(Value::as_str),
                "formula": precursor.get("formula").and_then(Value::as_str),
                "cas_number": precursor.get("cas_number").and_then(Value::as_str),
                "catalog_melting_point_c": precursor.get("melting_point_c").and_then(Value::as_f64),
                "catalog_purity_fraction": precursor.get("sa_purity").and_then(Value::as_f64)
            })
        }).collect();

        let scan_index = row.get("active_scan_index").and_then(Value::as_u64).map(|x| x as usize);
        let scans = row.get("characterization").and_then(|x| x.get("xrd"))
            .and_then(|x| x.get("scans")).and_then(Value::as_array);
        let active_scan = scan_index.and_then(|index| scans.and_then(|items| items.get(index)));
        let active_case_index = active_scan.and_then(|scan| scan.get("active_case_index"))
            .and_then(Value::as_u64).map(|x| x as usize);
        let active_refinement = active_case_index.and_then(|index| {
            active_scan.and_then(|scan| scan.get("refinement_cases"))
                .and_then(Value::as_array).and_then(|cases| cases.get(index))
        });
        let verification = active_refinement.and_then(|case| case.get("verification"));
        let synthesis = row.get("synthesis");
        let profile = synthesis.and_then(|s| s.get("temperature_profile"));
        let profile_times = profile.and_then(|p| p.get("time_minutes")).and_then(Value::as_array);
        let profile_temps = profile.and_then(|p| p.get("temperature_celsius")).and_then(Value::as_array);
        let finite_temps: Vec<f64> = profile_temps.into_iter().flatten()
            .filter_map(Value::as_f64).filter(|x| x.is_finite()).collect();
        let profile_min = finite_temps.iter().copied().reduce(f64::min);
        let profile_max = finite_temps.iter().copied().reduce(f64::max);
        let category = sample.category.as_deref();
        let status = cohort_status(
            sample.has_physical_failure,
            sample.quality_score,
            category,
        );
        let endpoint = if status == CohortStatus::Analyzable {
            category.and_then(category_label)
        } else {
            None
        };
        let phases = row.get("outcome").and_then(|o| o.get("phases"))
            .cloned().unwrap_or_else(|| json!([]));
        json!({
            "sample_id": sample.sample_id,
            "precursor_order_source": precursor_records,
            "precursor_pair_canonical": [sample.pair.0, sample.pair.1],
            "raw_reaction_category": category,
            "cross_precursor_product_endpoint": endpoint,
            "cohort_status": status.as_str(),
            "physical_failure_observation": row.get("physical_failure")
                .and_then(|failure| failure.get("observation")).and_then(Value::as_str),
            "active_xrd": {
                "active_scan_index": scan_index,
                "scan_status": active_scan.and_then(|scan| scan.get("status")).and_then(Value::as_str),
                "prep_is_good": active_scan.and_then(|scan| scan.get("prep_verification"))
                    .and_then(|v| v.get("is_prep_good")).and_then(Value::as_bool),
                "active_refinement_case_index": active_case_index,
                "rwp": active_refinement.and_then(|case| case.get("rwp")).and_then(Value::as_f64),
                "is_accepted": verification.and_then(|v| v.get("is_accepted")).and_then(Value::as_bool),
                "human_quality_score": sample.quality_score,
                "phase_weights": active_refinement.and_then(|case| case.get("phase_weights"))
                    .cloned().unwrap_or_else(|| json!({}))
            },
            "reported_outcome_phases": phases,
            "synthesis": {
                "recorded_heating_temperature_c": sample.dwell_temperature_c,
                "recorded_heating_time": synthesis.and_then(|s| s.get("heating_time")).and_then(Value::as_f64),
                "furnace_name": synthesis.and_then(|s| s.get("furnace_name")).and_then(Value::as_str),
                "mass_recovered_mg": synthesis.and_then(|s| s.get("mass_recovered_mg")).and_then(Value::as_f64),
                "met_target_mass": synthesis.and_then(|s| s.get("met_target_mass")).and_then(Value::as_bool),
                "thermal_profile": {
                    "time_point_count": profile_times.map(Vec::len).unwrap_or(0),
                    "temperature_point_count": profile_temps.map(Vec::len).unwrap_or(0),
                    "minimum_recorded_temperature_c": profile_min,
                    "maximum_recorded_temperature_c": profile_max
                }
            },
            "target_compound": row.get("target_compound").and_then(Value::as_str),
            "target_stoichiometry": row.get("target_stoichiometry").and_then(Value::as_str),
            "reaction_energy_eV_per_atom": row.get("reaction_energy_ev_per_atom").and_then(Value::as_f64)
        })
    }).collect();

    let per_precursor: Vec<Value> = node_counts.iter().map(|(formula, counts)| {
        json!({
            "precursor_formula": formula,
            "raw_pair_degree": counts.raw_pairs,
            "analyzable_pair_degree": counts.analyzable_pairs,
            "positive_pairs": counts.positive_pairs,
            "negative_pairs": counts.negative_pairs,
            "cross_precursor_product_fraction": if counts.analyzable_pairs > 0 {
                Some(counts.positive_pairs as f64 / counts.analyzable_pairs as f64)
            } else { None },
            "category_counts": {
                "transformed": counts.transformed_pairs,
                "unreacted": counts.unreacted_pairs,
                "completely_reacted": counts.completely_reacted_pairs,
                "partially_reacted": counts.partially_reacted_pairs
            }
        })
    }).collect();

    let precursor_names: Vec<String> = nodes.iter().cloned().collect();
    let node_count = precursor_names.len();
    let lopo_train_edges = (node_count - 1) * (node_count - 2) / 2;
    let lopo_test_edges = node_count - 1;
    let double_node_train_edges = (node_count - 2) * (node_count - 3) / 2;
    let double_node_test_edges = 2 * (node_count - 2) + 1;
    let leave_one_precursor_out_folds: Vec<Value> = precursor_names.iter().map(|held| {
        json!({
            "fold_id": format!("leave_one_precursor_out__{held}"),
            "held_precursor": held,
            "training_node_count": node_count - 1,
            "training_edge_count": lopo_train_edges,
            "test_edge_count": lopo_test_edges,
            "test_edge_predicate": "canonical_pair contains held_precursor"
        })
    }).collect();
    let mut double_node_holdout_folds = Vec::with_capacity(node_count * (node_count - 1) / 2);
    for (index, left) in precursor_names.iter().enumerate() {
        for right in precursor_names.iter().skip(index + 1) {
            double_node_holdout_folds.push(json!({
                "fold_id": format!("double_node_holdout__{left}__{right}"),
                "held_precursors": [left, right],
                "training_node_count": node_count - 2,
                "training_edge_count": double_node_train_edges,
                "test_edge_count": double_node_test_edges,
                "test_edge_predicate": "canonical_pair contains either held_precursor"
            }));
        }
    }

    Ok(json!({
        "status": "precursor_genome_graph_audit_pass",
        "contract": "SC-PRECURSOR-GENOME-GRAPH-AUDIT-0.1",
        "benchmark_id": text_field(&manifest, "benchmark_id")?,
        "source": {
            "repository": text_field(source, "repository")?,
            "commit": source_commit,
            "ledger_path": text_field(source, "ledger_path")?,
            "license": text_field(source, "license")?,
            "attribution": text_field(source, "attribution")?,
            "ledger_bytes": ledger_bytes.len(),
            "ledger_git_blob_sha1": actual_blob,
            "ledger_sha256": sha256_hex(ledger_bytes)
        },
        "pair_graph": {
            "precursor_count": nodes.len(),
            "unique_unordered_pair_count": edges.len(),
            "complete_undirected_graph": true,
            "expected_degree": expected_degree,
            "per_precursor": per_precursor
        },
        "sample_records": sample_records,
        "validation_protocols": {
            "random_pair_split": {
                "status": "diagnostic_only",
                "reason": "Every precursor appears in 45 source edges; random edge splits leak precursor identities into train and test."
            },
            "leave_one_precursor_out": {
                "fold_count": leave_one_precursor_out_folds.len(),
                "training_edge_count_per_fold": lopo_train_edges,
                "test_edge_count_per_fold": lopo_test_edges,
                "rule": "Remove all source edges incident to one precursor from training; test only on those incident edges.",
                "folds": leave_one_precursor_out_folds
            },
            "double_node_holdout": {
                "fold_count": double_node_holdout_folds.len(),
                "training_edge_count_per_fold": double_node_train_edges,
                "test_edge_count_per_fold": double_node_test_edges,
                "rule": "Remove every edge incident to either of two held precursors from training; test on the union of both held-node neighborhoods.",
                "folds": double_node_holdout_folds
            },
            "chemistry_family_holdout": {
                "status": "not_materialized",
                "reason": "A precursor family assignment must be declared independently of outcomes and frozen before evaluation; no family labels are inferred from this ledger."
            }
        },
        "cohort": {
            "source_sample_count": sample_count,
            "physical_failure_exclusions": physical_failure_exclusions,
            "poor_refinement_quality_score": expected_u64(&manifest, "poor_refinement_quality_score")?,
            "poor_refinement_exclusions": poor_refinement_exclusions,
            "analyzable_count": analyzable_count,
            "raw_category_counts": count_map_json(&raw_category_counts),
            "analyzable_category_counts": count_map_json(&analyzable_category_counts),
            "cross_precursor_product_positive_count": positive_count,
            "cross_precursor_product_negative_count": negative_count,
            "positive_fraction": positive_count as f64 / analyzable_count as f64,
            "negative_fraction": negative_count as f64 / analyzable_count as f64,
            "exclusion_reason_counts": count_map_json(&exclusion_counts)
        },
        "quality_and_thermal_diagnostics": {
            "active_refinement_quality_score_counts": count_map_json(&quality_counts),
            "recorded_dwell_temperature_c_count": dwell_count,
            "recorded_dwell_temperature_c_missing": dwell_missing_count,
            "recorded_dwell_temperature_c_histogram": count_map_json(&dwell_counts),
            "thermal_feature_policy": "Dwell setpoints are copied from synthesis.heating_temperature only; no missing temperature or reference-transition proxy is inferred."
        },
        "label_semantics": {
            "positive": "completely_reacted or partially_reacted: at least one phase incorporates ions from both precursors",
            "negative": "unreacted or transformed: no cross-precursor product; transformed may still contain precursor decomposition, oxidation or reduction products",
            "excluded": "physical_failure_no_powder or poor_xrd_refinement_quality_3; neither is a negative example"
        },
        "claim_ceiling": text_field(&manifest, "claim_ceiling")?
    }))
}
fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --manifest <manifest.json> --manifest-schema <schema.json> --ledger <ledger.json> [--out <receipt.json>]"
    );
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut manifest_path = None;
    let mut schema_path = None;
    let mut ledger_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest" | "--manifest-schema" | "--ledger" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--manifest" => manifest_path = Some(path),
                    "--manifest-schema" => schema_path = Some(path),
                    "--ledger" => ledger_path = Some(path),
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
    let (Some(manifest_path), Some(schema_path), Some(ledger_path)) =
        (manifest_path, schema_path, ledger_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = audit(&read(&schema_path), &read(&manifest_path), &read(&ledger_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-PRECURSOR-GENOME-GRAPH-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("JSON receipt serialization cannot fail");
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

    #[test]
    fn git_blob_sha1_matches_the_empty_git_blob() {
        assert_eq!(
            git_blob_sha1(b""),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        );
    }
    #[test]
    fn unordered_pair_canonicalization_is_symmetric() {
        assert_eq!(canonical_pair("A", "B").unwrap(), canonical_pair("B", "A").unwrap());
        assert_eq!(canonical_pair("A", "B").unwrap(), ("A".to_owned(), "B".to_owned()));
    }
    #[test]
    fn self_pair_is_rejected() {
        assert!(canonical_pair("A", "A").is_err());
    }
    #[test]
    fn complete_three_node_graph_has_degree_two() {
        let nodes = ["A".to_owned(), "B".to_owned(), "C".to_owned()]
            .into_iter().collect::<BTreeSet<_>>();
        let edges = [
            ("A".to_owned(), "B".to_owned()),
            ("A".to_owned(), "C".to_owned()),
            ("B".to_owned(), "C".to_owned()),
        ].into_iter().collect::<HashSet<_>>();
        let degrees = validate_complete_graph(&nodes, &edges, 2).unwrap();
        assert_eq!(degrees.get("A"), Some(&2));
        assert_eq!(degrees.get("B"), Some(&2));
        assert_eq!(degrees.get("C"), Some(&2));
    }
    #[test]
    fn incomplete_graph_is_rejected() {
        let nodes = ["A".to_owned(), "B".to_owned(), "C".to_owned()]
            .into_iter().collect::<BTreeSet<_>>();
        let edges = [
            ("A".to_owned(), "B".to_owned()),
            ("A".to_owned(), "C".to_owned()),
        ].into_iter().collect::<HashSet<_>>();
        assert!(validate_complete_graph(&nodes, &edges, 2).unwrap_err().contains("requires 3 edges"));
    }
    #[test]
    fn product_endpoint_preserves_transformed_as_negative_subtype() {
        assert_eq!(category_label("completely_reacted"), Some(true));
        assert_eq!(category_label("partially_reacted"), Some(true));
        assert_eq!(category_label("unreacted"), Some(false));
        assert_eq!(category_label("transformed"), Some(false));
        assert_eq!(category_label("unclassified"), None);
    }
    #[test]
    fn cold_start_split_sizes_match_complete_pair_graph_combinatorics() {
        let nodes = 46_usize;
        let lopo_train = (nodes - 1) * (nodes - 2) / 2;
        let lopo_test = nodes - 1;
        let double_train = (nodes - 2) * (nodes - 3) / 2;
        let double_test = 2 * (nodes - 2) + 1;
        assert_eq!((lopo_train, lopo_test), (990, 45));
        assert_eq!((double_train, double_test), (946, 89));
        assert_eq!(nodes * (nodes - 1) / 2, 1035);
    }

    #[test]
    fn physical_failure_and_poor_refinement_are_exclusions_not_negatives() {
        assert_eq!(
            cohort_status(false, Some(2), Some("transformed")),
            CohortStatus::Analyzable
        );
        assert_eq!(
            cohort_status(true, Some(1), Some("transformed")),
            CohortStatus::PhysicalFailure
        );
        assert_eq!(
            cohort_status(false, Some(3), Some("unreacted")),
            CohortStatus::PoorRefinement
        );
        assert_eq!(
            cohort_status(false, None, None),
            CohortStatus::UnknownOutcome
        );
    }
}

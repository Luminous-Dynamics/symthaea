//! Fail-closed ConnectomeBench2 V2 sidecar audit.
//!
//! Consumes JSONL exported from the metadata-only Parquet sidecars. This audit
//! checks the pinned dataset's declared identity, labels, routes, masks, source
//! operation grouping, and spatial split assignment before model training.
//! Root/neuron overlap across spatial splits is reported as a warning, not
//! automatically labelled leakage: biological neurons may span multiple cubes.

use clap::Parser;
use md5::Md5;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const DATASET_ID: &str = "jeffbbrown2/ConnectomeBench2";
const DATASET_REVISION: &str = "v2";
const CUBE_EDGE_NM: f64 = 50_000.0;
const KNOWN_SAMPLE_TYPES: &[&str] = &[
    "merge_edit",
    "split_edit",
    "adjacent_control",
    "junction_control",
    "synapse_control",
];
const KNOWN_SLOTS: &[&str] = &[
    "geometry",
    "geometry_single",
    "em_xy",
    "em_xz",
    "em_yz",
    "em_best",
];

#[derive(Debug, Parser)]
#[command(
    name = "connectome-dataset-audit",
    version,
    about = "Audit ConnectomeBench2 V2 metadata identity, labels, routes, lineage, and split isolation"
)]
struct Args {
    /// JSONL exported from metadata/{train,val,test}.parquet.
    #[arg(long)]
    input: PathBuf,

    /// Exact 40-character Hugging Face commit SHA resolved from revision=v2.
    /// This is operator-supplied provenance; the offline auditor does not fetch the Hub ref.
    #[arg(long)]
    dataset_revision_commit_sha: String,

    /// Spatial split cube edge in nanometres. ConnectomeBench2 V2 uses 50 µm cubes.
    #[arg(long, default_value_t = CUBE_EDGE_NM)]
    cube_edge_nm: f64,

    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Optional content-addressed sample-ID cohort manifest. Emitted only when the audit status is PASS.
    #[arg(long)]
    cohort_manifest_output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Row {
    combined_sample_hash: String,
    source_archive_sample_hash: String,
    source_archive: String,
    sample_type: String,
    same_neuron: bool,
    has_single_mask: bool,
    has_dual_mask: bool,
    task_routing: Vec<String>,
    false_split_correction_label: bool,
    false_merge_identification_label: bool,
    split: String,
    species: String,
    has_em: bool,
    present_slots: Vec<String>,
    metadata: String,
}

#[derive(Debug, Clone, Serialize)]
struct Finding {
    code: String,
    sample_id: Option<String>,
    detail: String,
}

impl Finding {
    fn row(code: &str, row: &Row, detail: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            sample_id: Some(row.combined_sample_hash.clone()),
            detail: detail.into(),
        }
    }

    fn global(code: &str, detail: impl Into<String>) -> Self {
        Self { code: code.to_string(), sample_id: None, detail: detail.into() }
    }
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    input_sha256: String,
    dataset_id: &'static str,
    dataset_revision: &'static str,
    dataset_revision_commit_sha: String,
    pinned_upstream_commit: &'static str,
    cube_edge_nm: f64,
    row_count: usize,
    split_counts: BTreeMap<String, u64>,
    species_counts: BTreeMap<String, u64>,
    sample_type_counts: BTreeMap<String, u64>,
    route_counts: BTreeMap<String, u64>,
    /// Metadata-sidecar declared modality coverage by deterministic population stratum.
    modality_coverage: BTreeMap<String, ModalityCoverage>,
    operation_lineage_group_count: usize,
    spatial_cube_group_count: usize,
    root_ids_shared_across_splits: usize,
    rows_without_spatial_coordinate: u64,
    status: &'static str,
    errors: Vec<Finding>,
    warnings: Vec<Finding>,
    scope_note: &'static str,
}

#[derive(Debug, Default, Serialize)]
struct ModalityCoverage {
    total_rows: u64,
    geometry_rows: u64,
    em_any_rows: u64,
    em_missing_rows: u64,
    /// Rows with at least one non-label-conditioned EM view (not necessarily all views).
    em_usable_rows: u64,
    em_xy_rows: u64,
    em_xz_rows: u64,
    em_yz_rows: u64,
    em_full_triplet_rows: u64,
    em_best_present_rows: u64,
    em_best_only_rows: u64,
    geometry_without_usable_em_rows: u64,
    fused_geometry_em_rows: u64,
    em_any_fraction: Option<f64>,
    em_usable_fraction: Option<f64>,
    em_full_triplet_fraction: Option<f64>,
    fused_fraction: Option<f64>,
}

impl ModalityCoverage {
    fn observe(&mut self, row: &Row, slots: &BTreeSet<String>) {
        self.total_rows += 1;
        let has_geometry = row.has_single_mask || row.has_dual_mask;
        let has_xy = slots.contains("em_xy");
        let has_xz = slots.contains("em_xz");
        let has_yz = slots.contains("em_yz");
        let has_usable_em = has_xy || has_xz || has_yz;
        let has_full_triplet = has_xy && has_xz && has_yz;
        let has_em_best = slots.contains("em_best");
        if has_geometry {
            self.geometry_rows += 1;
        }
        if row.has_em {
            self.em_any_rows += 1;
        } else {
            self.em_missing_rows += 1;
        }
        if has_usable_em {
            self.em_usable_rows += 1;
        }
        if has_xy {
            self.em_xy_rows += 1;
        }
        if has_xz {
            self.em_xz_rows += 1;
        }
        if has_yz {
            self.em_yz_rows += 1;
        }
        if has_full_triplet {
            self.em_full_triplet_rows += 1;
        }
        if has_em_best {
            self.em_best_present_rows += 1;
            if !has_usable_em {
                self.em_best_only_rows += 1;
            }
        }
        if has_geometry && !has_usable_em {
            self.geometry_without_usable_em_rows += 1;
        }
        if has_geometry && has_usable_em {
            self.fused_geometry_em_rows += 1;
        }
    }

    fn finalize(&mut self) {
        let total = self.total_rows as f64;
        self.em_any_fraction = (self.total_rows > 0)
            .then_some(self.em_any_rows as f64 / total);
        self.em_usable_fraction = (self.total_rows > 0)
            .then_some(self.em_usable_rows as f64 / total);
        self.em_full_triplet_fraction = (self.total_rows > 0)
            .then_some(self.em_full_triplet_rows as f64 / total);
        self.fused_fraction = (self.total_rows > 0)
            .then_some(self.fused_geometry_em_rows as f64 / total);
    }
}

fn modality_coverage_keys(row: &Row, split: &str) -> Vec<String> {
    let mut keys = vec![
        "overall".to_string(),
        format!("sample_type|{}", row.sample_type),
        format!("species|{}", row.species),
        format!("split|{split}"),
        format!("source_archive|{}", row.source_archive),
        format!("sample_type_species_split|{}|{}|{split}", row.sample_type, row.species),
    ];
    let routes = row.task_routing.iter().cloned().collect::<BTreeSet<_>>();
    for route in routes {
        keys.push(format!("task_route|{route}"));
        keys.push(format!("task_route_species_split|{route}|{}|{split}", row.species));
    }
    keys.sort();
    keys.dedup();
    keys
}

#[derive(Debug, Serialize)]
struct CohortManifest {
    schema_version: u32,
    dataset_id: &'static str,
    dataset_revision: &'static str,
    dataset_revision_commit_sha: String,
    pinned_upstream_commit: &'static str,
    source_sidecar_jsonl_sha256: String,
    eligibility_policy_version: &'static str,
    cohort_set_sha256: String,
    cohorts: BTreeMap<String, CohortMembership>,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct CohortMembership {
    membership_status: &'static str,
    eligibility_rule: String,
    sample_count: usize,
    sample_ids_sha256: String,
    sample_ids: Vec<String>,
}

fn hash_cohort_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn cohort_membership_sha256(name: &str, sample_ids: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"connectomebench2-v2-cohort-members-v1");
    hash_cohort_field(&mut hasher, name);
    hasher.update((sample_ids.len() as u64).to_be_bytes());
    for sample_id in sample_ids {
        hash_cohort_field(&mut hasher, sample_id);
    }
    hex::encode(hasher.finalize())
}

fn cohort_set_sha256(
    dataset_revision_commit_sha: &str,
    source_sidecar_jsonl_sha256: &str,
    cohorts: &BTreeMap<String, CohortMembership>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"connectomebench2-v2-cohort-set-v1");
    hash_cohort_field(&mut hasher, dataset_revision_commit_sha);
    hash_cohort_field(&mut hasher, source_sidecar_jsonl_sha256);
    hash_cohort_field(&mut hasher, "connectomebench2-v2-modality-cohorts-v1");
    hasher.update((cohorts.len() as u64).to_be_bytes());
    for (name, cohort) in cohorts {
        hash_cohort_field(&mut hasher, name);
        hash_cohort_field(&mut hasher, &cohort.eligibility_rule);
        hasher.update((cohort.sample_count as u64).to_be_bytes());
        hash_cohort_field(&mut hasher, &cohort.sample_ids_sha256);
    }
    hex::encode(hasher.finalize())
}

fn add_cohort_member(
    cohorts: &mut BTreeMap<String, (String, BTreeSet<String>)>,
    name: &str,
    eligibility_rule: &str,
    sample_id: &str,
) {
    cohorts
        .entry(name.to_string())
        .or_insert_with(|| (eligibility_rule.to_string(), BTreeSet::new()))
        .1
        .insert(sample_id.to_string());
}

fn build_cohort_manifest(
    rows: &[ParsedRow],
    dataset_revision_commit_sha: &str,
    source_sidecar_jsonl_sha256: &str,
) -> CohortManifest {
    let mut members = BTreeMap::<String, (String, BTreeSet<String>)>::new();
    let rules = [
        ("geometry_all", "has_single_mask OR has_dual_mask"),
        ("em_legal_any_view", "present_slots contains at least one of em_xy/em_xz/em_yz; excludes em_best"),
        ("em_legal_full_triplet", "present_slots contains em_xy AND em_xz AND em_yz; excludes em_best"),
        ("fused_geometry_legal_em", "geometry is present AND at least one of em_xy/em_xz/em_yz is present"),
        ("paired_geometry_only", "exact same sample IDs as fused_geometry_legal_em; score geometry-only on this cohort"),
        ("paired_em_legal_views", "exact same sample IDs as fused_geometry_legal_em; score legal EM views on this cohort"),
        ("geometry_without_legal_em", "geometry is present AND none of em_xy/em_xz/em_yz is present"),
        ("em_best_only_excluded", "em_best is present AND none of em_xy/em_xz/em_yz is present"),
    ];
    for (name, rule) in rules {
        members.insert(name.to_string(), (rule.to_string(), BTreeSet::new()));
    }

    for parsed in rows {
        let row = &parsed.row;
        let slots = row.present_slots.iter().cloned().collect::<BTreeSet<_>>();
        let has_geometry = row.has_single_mask || row.has_dual_mask;
        let has_xy = slots.contains("em_xy");
        let has_xz = slots.contains("em_xz");
        let has_yz = slots.contains("em_yz");
        let has_legal_em = has_xy || has_xz || has_yz;
        let has_em_best = slots.contains("em_best");
        let sample_id = row.combined_sample_hash.as_str();

        if has_geometry {
            add_cohort_member(&mut members, "geometry_all", rules[0].1, sample_id);
        }
        if has_legal_em {
            add_cohort_member(&mut members, "em_legal_any_view", rules[1].1, sample_id);
        }
        if has_xy && has_xz && has_yz {
            add_cohort_member(&mut members, "em_legal_full_triplet", rules[2].1, sample_id);
        }
        if has_geometry && has_legal_em {
            add_cohort_member(&mut members, "fused_geometry_legal_em", rules[3].1, sample_id);
            add_cohort_member(&mut members, "paired_geometry_only", rules[4].1, sample_id);
            add_cohort_member(&mut members, "paired_em_legal_views", rules[5].1, sample_id);
        }
        if has_geometry && !has_legal_em {
            add_cohort_member(&mut members, "geometry_without_legal_em", rules[6].1, sample_id);
        }
        if has_em_best && !has_legal_em {
            add_cohort_member(&mut members, "em_best_only_excluded", rules[7].1, sample_id);
        }
    }

    let cohorts = members
        .into_iter()
        .map(|(name, (eligibility_rule, ids))| {
            let sample_ids = ids.into_iter().collect::<Vec<_>>();
            let sample_count = sample_ids.len();
            let sample_ids_sha256 = cohort_membership_sha256(&name, &sample_ids);
            (
                name,
                CohortMembership {
                    membership_status: if sample_count == 0 { "NOT_ESTIMABLE" } else { "ELIGIBLE" },
                    eligibility_rule,
                    sample_count,
                    sample_ids_sha256,
                    sample_ids,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let cohort_set_digest = cohort_set_sha256(
        dataset_revision_commit_sha,
        source_sidecar_jsonl_sha256,
        &cohorts,
    );

    CohortManifest {
        schema_version: 1,
        dataset_id: DATASET_ID,
        dataset_revision: DATASET_REVISION,
        dataset_revision_commit_sha: dataset_revision_commit_sha.to_string(),
        pinned_upstream_commit: "5679b39aa2641d8d0afeed88b441e696f87a66df",
        source_sidecar_jsonl_sha256: source_sidecar_jsonl_sha256.to_string(),
        eligibility_policy_version: "connectomebench2-v2-modality-cohorts-v1",
        cohort_set_sha256: cohort_set_digest,
        cohorts,
        scope_note: "Membership is derived from metadata-sidecar declared slots and masks only. It does not verify Parquet payload bytes, image readability, biological correctness, or population inclusion probabilities. paired_geometry_only and paired_em_legal_views contain exactly the same sample IDs. EM-best-only rows are diagnostic exclusions, not usable EM examples.",
    }
}

#[derive(Debug, Clone)]
struct ParsedRow {
    row: Row,
    metadata: Value,
    normalized_split: String,
    bank: String,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn md5_sample_hash(source_archive: &str, source_hash: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(format!("{source_archive}|{source_hash}").as_bytes());
    hex::encode(hasher.finalize())
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn normalize_split(split: &str) -> Option<&'static str> {
    match split {
        "train" => Some("train"),
        "val" | "validation" => Some("validation"),
        "test" => Some("test"),
        _ => None,
    }
}

fn archive_bank(source_archive: &str) -> Option<&'static str> {
    if source_archive.starts_with("edits_and_adj_controls_") {
        Some("operation_bank")
    } else if source_archive.starts_with("junction_controls_") {
        Some("operation_bank_junction_controls")
    } else if source_archive.starts_with("synapse_controls_") {
        Some("operation_bank_synapse_controls")
    } else {
        None
    }
}

fn scalar_key(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed == "None" {
                return None;
            }
            if let Ok(integer) = trimmed.parse::<i128>() {
                return Some(integer.to_string());
            }
            if let Ok(integer) = trimmed.parse::<u128>() {
                return Some(integer.to_string());
            }
            Some(trimmed.to_string())
        }
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                return Some(integer.to_string());
            }
            if let Some(integer) = value.as_u64() {
                return Some(integer.to_string());
            }
            Some(value.to_string())
        }
        _ => None,
    }
}

fn metadata_string_array(metadata: &Value, field: &str) -> Result<Vec<String>, String> {
    let value = metadata.get(field).ok_or_else(|| format!("metadata.{field} is missing"))?;
    let array = value.as_array().ok_or_else(|| format!("metadata.{field} must be an array"))?;
    array.iter().map(|item| {
        item.as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("metadata.{field} must contain only strings"))
    }).collect()
}

fn derive_sample_type(source_archive: &str, strategy: Option<&str>) -> Option<&'static str> {
    if source_archive.contains("edits_and_adj") {
        match strategy {
            None | Some("merge_correction") => Some("merge_edit"),
            Some("inversion") | Some("split_correction") => Some("split_edit"),
            Some(value) if value.to_lowercase().contains("adj") => Some("adjacent_control"),
            _ => None,
        }
    } else if source_archive.contains("junction") {
        Some("junction_control")
    } else if source_archive.contains("synapse") {
        Some("synapse_control")
    } else {
        None
    }
}

fn expected_same_neuron(sample_type: &str) -> Option<bool> {
    match sample_type {
        "merge_edit" | "junction_control" => Some(true),
        "split_edit" | "adjacent_control" | "synapse_control" => Some(false),
        _ => None,
    }
}

fn expected_routes(sample_type: &str, has_single: bool, has_dual: bool) -> BTreeSet<String> {
    let mut routes = BTreeSet::new();
    if matches!(sample_type, "merge_edit" | "synapse_control" | "adjacent_control") && has_dual {
        routes.insert("false_split_correction".to_string());
    }
    if matches!(sample_type, "split_edit" | "junction_control") && has_single {
        routes.insert("false_merge_identification".to_string());
    }
    if sample_type == "split_edit" && has_single {
        routes.insert("split_mask_generation".to_string());
    }
    routes
}

fn lineage_key(row: &Row, metadata: &Value, bank: &str) -> Option<String> {
    let key = match bank {
        "operation_bank" => scalar_key(metadata.get("operation_id"))?,
        "operation_bank_junction_controls" => scalar_key(metadata.get("source_operation_id"))?,
        "operation_bank_synapse_controls" => {
            let pre = scalar_key(metadata.get("pre_pt_root_id"))?;
            let post = scalar_key(metadata.get("post_pt_root_id"))?;
            let center = metadata.get("synapse_ctr_pt_nm")?.as_array()?;
            if center.len() != 3 {
                return None;
            }
            let mut coords = Vec::with_capacity(3);
            for value in center {
                let value = value.as_f64()?;
                if !value.is_finite() {
                    return None;
                }
                // Python float equality treats -0.0 and 0.0 as the same tuple element.
                coords.push(if value == 0.0 { "0".to_string() } else { value.to_string() });
            }
            format!("{pre}|{post}|{}", coords.join(","))
        }
        _ => return None,
    };
    Some(format!("{}|{}|{}", row.species, bank, key))
}

fn coordinate(metadata: &Value, bank: &str) -> Option<[f64; 3]> {
    let field = if bank == "operation_bank_synapse_controls" {
        "synapse_ctr_pt_nm"
    } else {
        "interface_point_nm"
    };
    let values = metadata.get(field)?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    let mut out = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        out[index] = value.as_f64()?;
        if !out[index].is_finite() {
            return None;
        }
    }
    Some(out)
}

fn cube_key(row: &Row, point: [f64; 3], edge_nm: f64) -> Option<String> {
    let mut indices = [0_i64; 3];
    for (index, value) in point.iter().enumerate() {
        let scaled = (*value / edge_nm).floor();
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return None;
        }
        indices[index] = scaled as i64;
    }
    Some(format!(
        "{}|{},{},{}",
        row.species, indices[0], indices[1], indices[2]
    ))
}

fn root_ids(metadata: &Value) -> Vec<String> {
    let mut roots = Vec::new();
    for field in ["before_root_ids", "after_root_ids"] {
        if let Some(values) = metadata.get(field).and_then(Value::as_array) {
            for value in values {
                if let Some(root) = scalar_key(Some(value)) {
                    roots.push(root);
                }
            }
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

fn parse_rows(bytes: &[u8]) -> Result<Vec<ParsedRow>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("input is not UTF-8: {error}"))?;
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Row>(line).map_err(|error| {
            format!("line {} is invalid or contains an unknown/missing field: {error}", line_index + 1)
        })?;
        let metadata: Value = serde_json::from_str(&row.metadata).map_err(|error| {
            format!("line {} metadata is not valid JSON: {error}", line_index + 1)
        })?;
        if !metadata.is_object() {
            return Err(format!("line {} metadata must be a JSON object", line_index + 1));
        }
        let normalized_split = normalize_split(&row.split)
            .ok_or_else(|| format!("line {} has unsupported split {:?}", line_index + 1, row.split))?
            .to_string();
        let bank = archive_bank(&row.source_archive)
            .ok_or_else(|| format!("line {} has unsupported source_archive {:?}", line_index + 1, row.source_archive))?
            .to_string();
        rows.push(ParsedRow { row, metadata, normalized_split, bank });
    }
    if rows.is_empty() {
        return Err("input contains no metadata rows".to_string());
    }
    Ok(rows)
}

fn audit_rows(rows: &[ParsedRow], input_hash: String, cube_edge_nm: f64, dataset_revision_commit_sha: String) -> Report {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut split_counts = BTreeMap::<String, u64>::new();
    let mut species_counts = BTreeMap::<String, u64>::new();
    let mut sample_type_counts = BTreeMap::<String, u64>::new();
    let mut route_counts = BTreeMap::<String, u64>::new();
    let mut modality_coverage = BTreeMap::<String, ModalityCoverage>::new();
    let mut seen_samples = BTreeSet::<String>::new();
    let mut lineage_splits = BTreeMap::<String, BTreeSet<String>>::new();
    let mut cube_splits = BTreeMap::<String, BTreeSet<String>>::new();
    let mut root_splits = BTreeMap::<String, BTreeSet<String>>::new();
    let mut rows_without_spatial_coordinate = 0_u64;

    for parsed in rows {
        let row = &parsed.row;
        let metadata = &parsed.metadata;
        let split = &parsed.normalized_split;
        let bank = &parsed.bank;

        *split_counts.entry(split.clone()).or_default() += 1;
        *species_counts.entry(row.species.clone()).or_default() += 1;
        *sample_type_counts.entry(row.sample_type.clone()).or_default() += 1;
        for route in &row.task_routing {
            *route_counts.entry(route.clone()).or_default() += 1;
        }

        if !seen_samples.insert(row.combined_sample_hash.clone()) {
            errors.push(Finding::row("duplicate_combined_sample_hash", row, "combined_sample_hash is not globally unique"));
        }
        if !is_lower_hex(&row.combined_sample_hash, 32) {
            errors.push(Finding::row("combined_hash_format", row, "combined_sample_hash must be 32 lowercase hexadecimal characters"));
        }
        if !is_lower_hex(&row.source_archive_sample_hash, 12) {
            errors.push(Finding::row("source_hash_format", row, "source_archive_sample_hash must be 12 lowercase hexadecimal characters"));
        } else {
            let expected = md5_sample_hash(&row.source_archive, &row.source_archive_sample_hash);
            if row.combined_sample_hash != expected {
                errors.push(Finding::row(
                    "combined_hash_mismatch",
                    row,
                    format!("combined_sample_hash does not equal MD5(source_archive|source_archive_sample_hash); expected {expected}"),
                ));
            }
        }

        if !KNOWN_SAMPLE_TYPES.contains(&row.sample_type.as_str()) {
            errors.push(Finding::row("unknown_sample_type", row, format!("unsupported sample_type {:?}", row.sample_type)));
        }

        if !row.source_archive.ends_with(&format!("_{}", row.species)) {
            errors.push(Finding::row(
                "archive_species_mismatch",
                row,
                format!("source_archive {:?} does not end with species {:?}", row.source_archive, row.species),
            ));
        }

        let expected_same = expected_same_neuron(&row.sample_type);
        if expected_same != Some(row.same_neuron) {
            errors.push(Finding::row("same_neuron_mismatch", row, "same_neuron disagrees with the dataset sample_type mapping"));
        }
        if row.false_split_correction_label != row.same_neuron {
            errors.push(Finding::row("false_split_label_mismatch", row, "false_split_correction_label must equal same_neuron"));
        }
        if row.false_merge_identification_label == row.same_neuron {
            errors.push(Finding::row("false_merge_label_mismatch", row, "false_merge_identification_label must equal NOT same_neuron"));
        }

        let image_types = match metadata_string_array(metadata, "image_types") {
            Ok(values) => values,
            Err(detail) => {
                errors.push(Finding::row("metadata_image_types_invalid", row, detail));
                Vec::new()
            }
        };
        let image_type_set: BTreeSet<String> = image_types.iter().cloned().collect();
        if image_type_set.len() != image_types.len() {
            errors.push(Finding::row("duplicate_image_type", row, "metadata.image_types contains duplicate entries"));
        }
        let derived_single = image_type_set.contains("geometry_single");
        let derived_dual = image_type_set.contains("geometry");
        if row.has_single_mask != derived_single {
            errors.push(Finding::row("single_mask_flag_mismatch", row, "has_single_mask must reflect geometry_single in metadata.image_types"));
        }
        if row.has_dual_mask != derived_dual {
            errors.push(Finding::row("dual_mask_flag_mismatch", row, "has_dual_mask must reflect geometry in metadata.image_types"));
        }
        for value in &image_types {
            if !matches!(value.as_str(), "geometry" | "geometry_single") {
                errors.push(Finding::row("unknown_image_type", row, format!("unsupported metadata.image_types entry {value:?}")));
            }
        }

        let expected_route_set = expected_routes(&row.sample_type, row.has_single_mask, row.has_dual_mask);
        let actual_route_set: BTreeSet<String> = row.task_routing.iter().cloned().collect();
        if actual_route_set.len() != row.task_routing.len() {
            errors.push(Finding::row("duplicate_task_route", row, "task_routing contains duplicate entries"));
        }
        if actual_route_set != expected_route_set {
            errors.push(Finding::row(
                "task_routing_mismatch",
                row,
                format!("expected routes {:?}, got {:?}", expected_route_set, actual_route_set),
            ));
        }

        let present_slot_set: BTreeSet<String> = row.present_slots.iter().cloned().collect();
        for key in modality_coverage_keys(row, split) {
            modality_coverage.entry(key).or_default().observe(row, &present_slot_set);
        }
        if present_slot_set.len() != row.present_slots.len() {
            errors.push(Finding::row("duplicate_present_slot", row, "present_slots contains duplicate entries"));
        }
        for slot in &present_slot_set {
            if !KNOWN_SLOTS.contains(&slot.as_str()) {
                errors.push(Finding::row("unknown_present_slot", row, format!("unsupported present_slots entry {slot:?}")));
            }
        }
        if row.has_single_mask != present_slot_set.contains("geometry_single") {
            errors.push(Finding::row("present_single_mask_mismatch", row, "present_slots and has_single_mask disagree"));
        }
        if row.has_dual_mask != present_slot_set.contains("geometry") {
            errors.push(Finding::row("present_dual_mask_mismatch", row, "present_slots and has_dual_mask disagree"));
        }
        let present_em = present_slot_set.iter().any(|slot| slot.starts_with("em_"));
        if row.has_em != present_em {
            errors.push(Finding::row("has_em_mismatch", row, "has_em must equal presence of at least one em_* slot"));
        }

        let strategy = metadata.get("strategy").and_then(Value::as_str);
        let derived_type = derive_sample_type(&row.source_archive, strategy);
        if derived_type != Some(row.sample_type.as_str()) {
            errors.push(Finding::row(
                "sample_type_derivation_mismatch",
                row,
                format!("source_archive/metadata.strategy imply {:?}, not {:?}", derived_type, row.sample_type),
            ));
        }

        match lineage_key(row, metadata, bank) {
            Some(key) => {
                lineage_splits.entry(key).or_default().insert(split.clone());
            }
            None => errors.push(Finding::row(
                "lineage_key_missing",
                row,
                format!("cannot reconstruct the upstream split-group key for bank {bank}"),
            )),
        }

        let coordinate_field = if bank == "operation_bank_synapse_controls" {
            "synapse_ctr_pt_nm"
        } else {
            "interface_point_nm"
        };
        match coordinate(metadata, bank) {
            Some(point) => match cube_key(row, point, cube_edge_nm) {
                Some(key) => {
                    cube_splits.entry(key).or_default().insert(split.clone());
                }
                None => errors.push(Finding::row("spatial_cube_invalid", row, "spatial coordinate cannot be mapped to an integer cube")),
            },
            None if metadata.get(coordinate_field).is_some() => errors.push(Finding::row(
                "spatial_coordinate_invalid",
                row,
                format!("metadata.{coordinate_field} is present but is not a finite three-number coordinate"),
            )),
            None => rows_without_spatial_coordinate += 1,
        }

        for root in root_ids(metadata) {
            root_splits
                .entry(format!("{}|{}", row.species, root))
                .or_default()
                .insert(split.clone());
        }
    }

    for (key, splits) in &lineage_splits {
        if splits.len() > 1 {
            errors.push(Finding::global(
                "operation_lineage_cross_split",
                format!("upstream split-group key {key:?} appears in partitions {splits:?}"),
            ));
        }
    }
    for (key, splits) in &cube_splits {
        if splits.len() > 1 {
            errors.push(Finding::global(
                "spatial_cube_cross_split",
                format!("50 µm cube {key:?} appears in partitions {splits:?}"),
            ));
        }
    }

    let shared_root_entries: Vec<_> = root_splits
        .iter()
        .filter(|(_, splits)| splits.len() > 1)
        .collect();
    if !shared_root_entries.is_empty() {
        warnings.push(Finding::global(
            "neuron_identity_cross_split",
            format!(
                "{} species-scoped root identities appear in multiple spatial partitions; this is a dependence/leakage risk to quantify, not proof of leakage because neurons can span cubes",
                shared_root_entries.len()
            ),
        ));
    }
    if rows_without_spatial_coordinate > 0 {
        warnings.push(Finding::global(
            "spatial_coordinate_missing",
            format!("{rows_without_spatial_coordinate} rows lack the coordinate needed to independently audit cube assignment"),
        ));
    }

    for coverage in modality_coverage.values_mut() {
        coverage.finalize();
    }

    let operation_lineage_group_count = lineage_splits.len();
    let spatial_cube_group_count = cube_splits.len();
    let root_ids_shared_across_splits = shared_root_entries.len();
    errors.sort_by(|a, b| a.code.cmp(&b.code).then_with(|| a.sample_id.cmp(&b.sample_id)).then_with(|| a.detail.cmp(&b.detail)));
    warnings.sort_by(|a, b| a.code.cmp(&b.code).then_with(|| a.sample_id.cmp(&b.sample_id)).then_with(|| a.detail.cmp(&b.detail)));

    Report {
        schema_version: 1,
        tool: "connectome-dataset-audit",
        input_sha256: input_hash,
        dataset_id: DATASET_ID,
        dataset_revision: DATASET_REVISION,
        dataset_revision_commit_sha,
        pinned_upstream_commit: "5679b39aa2641d8d0afeed88b441e696f87a66df",
        cube_edge_nm,
        row_count: rows.len(),
        split_counts,
        species_counts,
        sample_type_counts,
        route_counts,
        modality_coverage,
        operation_lineage_group_count,
        spatial_cube_group_count,
        root_ids_shared_across_splits,
        rows_without_spatial_coordinate,
        status: if errors.is_empty() { "PASS" } else { "FAIL" },
        errors,
        warnings,
        scope_note: "PASS validates row-level dataset identity, declared label/route/mask consistency, upstream operation-group isolation, and available 50 um spatial-cube isolation. modality_coverage is computed from sidecar-declared present_slots/has_em only; it does not verify Parquet payload bytes, prove ground-truth correctness, biological independence, or model performance. em_usable_rows counts em_xy/em_xz/em_yz only and excludes label-conditioned em_best; this is metadata eligibility, not proof of readable payload bytes. Cross-split root identities are warnings because neurons may span spatial cubes.",
    }
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    if !is_lower_hex(&args.dataset_revision_commit_sha, 40) {
        return Err("--dataset-revision-commit-sha must be a lowercase 40-character Git SHA resolved from the V2 revision".to_string());
    }
    if !args.cube_edge_nm.is_finite() || args.cube_edge_nm <= 0.0 {
        return Err("--cube-edge-nm must be finite and positive".to_string());
    }
    let bytes = fs::read(&args.input)
        .map_err(|error| format!("failed to read {}: {error}", args.input.display()))?;
    let input_hash = sha256(&bytes);
    let dataset_revision_commit_sha = args.dataset_revision_commit_sha.clone();
    let rows = parse_rows(&bytes)?;
    let report = audit_rows(
        &rows,
        input_hash.clone(),
        args.cube_edge_nm,
        dataset_revision_commit_sha.clone(),
    );
    let status = report.status;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))? + "\n";
    if let Some(path) = args.output {
        fs::write(&path, json.as_bytes())
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    } else {
        print!("{json}");
    }

    if let Some(path) = args.cohort_manifest_output {
        if status == "PASS" {
            let manifest = build_cohort_manifest(
                &rows,
                &dataset_revision_commit_sha,
                &input_hash,
            );
            let rendered = serde_json::to_string_pretty(&manifest)
                .map_err(|error| format!("failed to serialize cohort manifest: {error}"))?
                + "\n";
            fs::write(&path, rendered.as_bytes())
                .map_err(|error| format!("failed to write cohort manifest {}: {error}", path.display()))?;
        } else {
            eprintln!("cohort manifest withheld: metadata audit status is {status}, not PASS");
        }
    }
    Ok(if status == "PASS" { 0 } else { 1 })
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(code) => ExitCode::from(code as u8),
        Err(error) => {
            eprintln!("connectome-dataset-audit: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(
        sample_hash: &str,
        archive: &str,
        sample_type: &str,
        split: &str,
        operation: u64,
        same_neuron: bool,
        single: bool,
        dual: bool,
        coordinate: [f64; 3],
    ) -> ParsedRow {
        let mut metadata = json!({
            "image_types": [],
            "interface_point_nm": coordinate,
            "before_root_ids": [format!("before-{operation}")],
            "after_root_ids": [format!("after-{operation}")]
        });
        let image_types = match (single, dual) {
            (true, true) => vec!["geometry".to_string(), "geometry_single".to_string()],
            (true, false) => vec!["geometry_single".to_string()],
            (false, true) => vec!["geometry".to_string()],
            (false, false) => vec![],
        };
        metadata["image_types"] = json!(image_types);
        match sample_type {
            "merge_edit" => metadata["strategy"] = json!("merge_correction"),
            "split_edit" => metadata["strategy"] = json!("split_correction"),
            "adjacent_control" => metadata["strategy"] = json!("adjacent_control"),
            "junction_control" => metadata["source_operation_id"] = json!(operation),
            "synapse_control" => {
                metadata["pre_pt_root_id"] = json!(1000 + operation);
                metadata["post_pt_root_id"] = json!(2000 + operation);
                metadata["synapse_ctr_pt_nm"] = json!(coordinate);
            }
            _ => {}
        }
        if archive.starts_with("edits_and_adj_controls_") {
            metadata["operation_id"] = json!(operation);
        }
        let mut routes = expected_routes(sample_type, single, dual);
        let present_slots: Vec<String> = image_types_from_metadata(&metadata);
        if sample_type == "synapse_control" {
            routes.insert("false_split_correction".to_string());
        }
        let source_archive_sample_hash = format!("{sample_hash:0>12}");
        let sample_id = md5_sample_hash(archive, &source_archive_sample_hash);
        let row = Row {
            combined_sample_hash: sample_id,
            source_archive_sample_hash,
            source_archive: archive.to_string(),
            sample_type: sample_type.to_string(),
            same_neuron,
            has_single_mask: single,
            has_dual_mask: dual,
            task_routing: routes.into_iter().collect(),
            false_split_correction_label: same_neuron,
            false_merge_identification_label: !same_neuron,
            split: split.to_string(),
            species: "mouse".to_string(),
            has_em: false,
            present_slots,
            metadata: metadata.to_string(),
        };
        let normalized_split = normalize_split(split).unwrap().to_string();
        let bank = archive_bank(archive).unwrap().to_string();
        ParsedRow { row, metadata, normalized_split, bank }
    }

    fn image_types_from_metadata(metadata: &Value) -> Vec<String> {
        metadata
            .get("image_types")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect()
    }

    fn valid_dataset() -> Vec<ParsedRow> {
        vec![
            row("1", "edits_and_adj_controls_mouse", "merge_edit", "train", 1, true, true, true, [10.0, 10.0, 10.0]),
            row("2", "edits_and_adj_controls_mouse", "split_edit", "validation", 2, false, true, true, [60_010.0, 10.0, 10.0]),
            row("3", "edits_and_adj_controls_mouse", "adjacent_control", "test", 3, false, false, true, [120_010.0, 10.0, 10.0]),
            row("4", "junction_controls_mouse", "junction_control", "train", 4, true, true, false, [180_010.0, 10.0, 10.0]),
            row("5", "synapse_controls_mouse", "synapse_control", "validation", 5, false, false, true, [240_010.0, 10.0, 10.0]),
        ]
    }

    fn report(rows: &[ParsedRow]) -> Report {
        audit_rows(rows, "0".repeat(64), CUBE_EDGE_NM, "1".repeat(40))
    }

    #[test]
    fn accepts_all_five_sample_types_with_upstream_semantics() {
        let report = report(&valid_dataset());
        assert_eq!(report.status, "PASS", "{:?}", report.errors);
        assert_eq!(report.row_count, 5);
        assert_eq!(report.sample_type_counts.len(), 5);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn recomputes_combined_sample_hash_from_archive_and_source_hash() {
        let rows = valid_dataset();
        assert_eq!(
            md5_sample_hash(&rows[0].row.source_archive, &rows[0].row.source_archive_sample_hash),
            rows[0].row.combined_sample_hash
        );
    }

    #[test]
    fn rejects_combined_sample_hash_mismatch() {
        let mut rows = valid_dataset();
        rows[0].row.combined_sample_hash = "0".repeat(32);
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "combined_hash_mismatch"));
    }

    #[test]
    fn rejects_duplicate_combined_sample_hashes() {
        let mut rows = valid_dataset();
        rows[1].row.combined_sample_hash = rows[0].row.combined_sample_hash.clone();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "duplicate_combined_sample_hash"));
    }

    #[test]
    fn rejects_operation_group_crossing_partitions() {
        let mut rows = valid_dataset();
        let mut second = rows[0].clone();
        second.row.source_archive_sample_hash = "abcdef000001".to_string();
        second.row.combined_sample_hash = md5_sample_hash(&second.row.source_archive, &second.row.source_archive_sample_hash);
        second.metadata["interface_point_nm"] = json!([310_010.0, 10.0, 10.0]);
        second.row.metadata = second.metadata.to_string();
        second.normalized_split = "test".to_string();
        rows.push(second);
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "operation_lineage_cross_split"));
    }

    #[test]
    fn rejects_spatial_cube_assigned_to_multiple_partitions() {
        let mut rows = valid_dataset();
        let mut second = rows[0].clone();
        second.row.source_archive_sample_hash = "abcdef000002".to_string();
        second.row.combined_sample_hash = md5_sample_hash(&second.row.source_archive, &second.row.source_archive_sample_hash);
        second.metadata["operation_id"] = json!(99);
        second.metadata["interface_point_nm"] = json!([20.0, 20.0, 20.0]);
        second.row.metadata = second.metadata.to_string();
        second.normalized_split = "test".to_string();
        rows.push(second);
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "spatial_cube_cross_split"));
    }

    #[test]
    fn reports_neuron_identity_overlap_as_warning_not_automatic_failure() {
        let mut rows = valid_dataset();
        rows[1].metadata["before_root_ids"] = json!(["before-1"]);
        rows[1].row.metadata = rows[1].metadata.to_string();
        let report = report(&rows);
        assert_eq!(report.status, "PASS");
        assert!(report.warnings.iter().any(|finding| finding.code == "neuron_identity_cross_split"));
        assert_eq!(report.root_ids_shared_across_splits, 1);
    }

    #[test]
    fn rejects_sample_type_that_disagrees_with_archive_strategy() {
        let mut rows = valid_dataset();
        rows[0].row.sample_type = "split_edit".to_string();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "sample_type_derivation_mismatch"));
    }

    #[test]
    fn rejects_incorrect_label_mapping() {
        let mut rows = valid_dataset();
        rows[0].row.false_merge_identification_label = true;
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "false_merge_label_mismatch"));
    }

    #[test]
    fn rejects_task_route_drift() {
        let mut rows = valid_dataset();
        rows[1].row.task_routing = vec!["false_split_correction".to_string()];
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "task_routing_mismatch"));
    }

    #[test]
    fn rejects_mask_flag_that_disagrees_with_metadata() {
        let mut rows = valid_dataset();
        rows[0].row.has_single_mask = false;
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "single_mask_flag_mismatch"));
    }

    #[test]
    fn rejects_em_flag_that_disagrees_with_present_slots() {
        let mut rows = valid_dataset();
        rows[0].row.has_em = true;
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "has_em_mismatch"));
    }

    #[test]
    fn reports_stratified_em_eligibility_and_excludes_em_best_from_usable_cohort() {
        let mut rows = valid_dataset();
        rows[0].row.present_slots.push("em_xy".to_string());
        rows[0].row.has_em = true;
        rows[1].row.present_slots.push("em_best".to_string());
        rows[1].row.has_em = true;
        rows[4].row.present_slots.push("em_xz".to_string());
        rows[4].row.has_em = true;

        let report = report(&rows);
        assert_eq!(report.status, "PASS", "{:?}", report.errors);
        let overall = report.modality_coverage.get("overall").unwrap();
        assert_eq!(overall.total_rows, 5);
        assert_eq!(overall.geometry_rows, 5);
        assert_eq!(overall.em_any_rows, 3);
        assert_eq!(overall.em_missing_rows, 2);
        assert_eq!(overall.em_usable_rows, 2);
        assert_eq!(overall.em_xy_rows, 1);
        assert_eq!(overall.em_xz_rows, 1);
        assert_eq!(overall.em_yz_rows, 0);
        assert_eq!(overall.em_full_triplet_rows, 0);
        assert_eq!(overall.em_best_present_rows, 1);
        assert_eq!(overall.em_best_only_rows, 1);
        assert_eq!(overall.fused_geometry_em_rows, 2);
        assert!((overall.em_any_fraction.unwrap() - 0.6).abs() < 1.0e-12);
        assert!((overall.em_usable_fraction.unwrap() - 0.4).abs() < 1.0e-12);
        assert!((overall.em_full_triplet_fraction.unwrap() - 0.0).abs() < 1.0e-12);

        assert!(report.modality_coverage.contains_key("task_route|false_split_correction"));
        assert!(report.modality_coverage.contains_key("sample_type|merge_edit"));
        assert!(report.modality_coverage.contains_key("species|mouse"));
        assert!(report.modality_coverage.contains_key("split|train"));
        assert!(report.modality_coverage.contains_key("source_archive|edits_and_adj_controls_mouse"));
        assert!(report.modality_coverage.contains_key("sample_type_species_split|merge_edit|mouse|train"));
        assert!(report.modality_coverage.contains_key("task_route_species_split|false_split_correction|mouse|train"));
    }

    #[test]
    fn requires_upstream_lineage_key_for_each_archive_bank() {
        let mut rows = valid_dataset();
        rows[3].metadata.as_object_mut().unwrap().remove("source_operation_id");
        rows[3].row.metadata = rows[3].metadata.to_string();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "lineage_key_missing"));
    }

    #[test]
    fn synapse_lineage_key_includes_pair_and_synapse_coordinate() {
        let mut rows = valid_dataset();
        let mut sibling = rows[4].clone();
        sibling.row.source_archive_sample_hash = "abcdef000003".to_string();
        sibling.row.combined_sample_hash = md5_sample_hash(&sibling.row.source_archive, &sibling.row.source_archive_sample_hash);
        sibling.normalized_split = "test".to_string();
        let report = report(&[rows.remove(4), sibling]);
        assert!(report.errors.iter().any(|finding| finding.code == "operation_lineage_cross_split"));
    }

    #[test]
    fn validation_alias_is_normalized_for_reporting() {
        let mut rows = valid_dataset();
        rows[1].row.split = "val".to_string();
        rows[1].normalized_split = "validation".to_string();
        let report = report(&rows);
        assert_eq!(report.split_counts.get("validation"), Some(&2));
    }

    #[test]
    fn missing_spatial_coordinates_are_reported_without_fabricating_a_cube() {
        let mut rows = valid_dataset();
        rows[0].metadata.as_object_mut().unwrap().remove("interface_point_nm");
        rows[0].row.metadata = rows[0].metadata.to_string();
        let report = report(&rows);
        assert_eq!(report.status, "PASS");
        assert_eq!(report.rows_without_spatial_coordinate, 1);
        assert!(report.warnings.iter().any(|finding| finding.code == "spatial_coordinate_missing"));
    }

    #[test]
    fn parser_rejects_unknown_sidecar_fields() {
        let row = valid_dataset().remove(0).row;
        let mut value = serde_json::to_value(row).unwrap();
        value["unexpected_future_field"] = json!(true);
        let line = value.to_string();
        assert!(parse_rows(line.as_bytes()).unwrap_err().contains("unknown"));
    }

    #[test]
    fn parser_rejects_malformed_nested_metadata() {
        let mut row = valid_dataset().remove(0).row;
        row.metadata = "{".to_string();
        let line = serde_json::to_string(&row).unwrap();
        assert!(parse_rows(line.as_bytes()).unwrap_err().contains("metadata is not valid JSON"));
    }

    #[test]
    fn parser_rejects_empty_input() {
        assert!(parse_rows(b"\n \n").unwrap_err().contains("no metadata rows"));
    }

    #[test]
    fn rejects_invalid_source_hash_format() {
        let mut rows = valid_dataset();
        rows[0].row.source_archive_sample_hash = "not-a-hash".to_string();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "source_hash_format"));
    }

    #[test]
    fn audit_output_is_deterministic_for_frozen_input() {
        let rows = valid_dataset();
        let first = serde_json::to_string(&report(&rows)).unwrap();
        let second = serde_json::to_string(&report(&rows)).unwrap();
        assert_eq!(first, second);
    }
    #[test]
    fn checked_in_metadata_lineage_fixture_passes_the_audit() {
        let rows = parse_rows(include_str!("../../../../docs/neuroscience/examples/CONNECTOMEBENCH2_V2_METADATA_LINEAGE.example.jsonl").as_bytes())
            .expect("synthetic JSONL fixture parses");
        let report = audit_rows(&rows, "0".repeat(64), CUBE_EDGE_NM, "1".repeat(40));
        assert_eq!(report.status, "PASS", "{:?}", report.errors);
        assert_eq!(report.row_count, 5);
        assert_eq!(report.sample_type_counts.len(), 5);
    }

    #[test]
    fn rejects_empty_strategy_for_edit_archive() {
        let mut rows = valid_dataset();
        rows[0].metadata["strategy"] = json!("");
        rows[0].row.metadata = rows[0].metadata.to_string();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "sample_type_derivation_mismatch"));
    }

    #[test]
    fn report_carries_the_operator_supplied_immutable_dataset_revision() {
        let rows = valid_dataset();
        let revision = "a".repeat(40);
        let report = audit_rows(&rows, "0".repeat(64), CUBE_EDGE_NM, revision.clone());
        assert_eq!(report.dataset_revision, "v2");
        assert_eq!(report.dataset_revision_commit_sha, revision);
        assert_eq!(report.pinned_upstream_commit, "5679b39aa2641d8d0afeed88b441e696f87a66df");
    }

    #[test]
    fn rejects_spatial_cube_crossing_partitions_across_different_sample_banks() {
        let mut rows = valid_dataset();
        rows[4].metadata["synapse_ctr_pt_nm"] = json!([20.0, 20.0, 20.0]);
        rows[4].row.metadata = rows[4].metadata.to_string();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "spatial_cube_cross_split"));
    }

    #[test]
    fn canonicalizes_numeric_strings_like_upstream_python_int_split_keys() {
        let mut rows = valid_dataset();
        rows[0].metadata["operation_id"] = json!("0001");
        rows[1].metadata["operation_id"] = json!(1);
        let first = lineage_key(&rows[0].row, &rows[0].metadata, &rows[0].bank).unwrap();
        let second = lineage_key(&rows[1].row, &rows[1].metadata, &rows[1].bank).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_present_but_malformed_spatial_coordinates() {
        let mut rows = valid_dataset();
        rows[0].metadata["interface_point_nm"] = json!("not-a-coordinate");
        rows[0].row.metadata = rows[0].metadata.to_string();
        let report = report(&rows);
        assert!(report.errors.iter().any(|finding| finding.code == "spatial_coordinate_invalid"));
    }

}

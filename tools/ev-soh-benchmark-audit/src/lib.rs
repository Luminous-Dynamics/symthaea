use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA_ID: &str = "https://luminousdynamics.io/schemas/transport/ev-soh-benchmark-manifest/0.1";
const CORPUS_VERSION: &str = "ev-soh-benchmark-manifest-corpus/0.1";
const PROTOCOL_VERSION: &str = "ev-soh-benchmark-manifest/0.1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    manifest: Value,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    protocol_version: String,
    benchmark_id: String,
    evidence_class: String,
    dataset: Dataset,
    target: Target,
    evaluation: Evaluation,
    splits: Splits,
    preprocessing: Vec<Preprocessing>,
    target_adaptation: TargetAdaptation,
    metrics: Vec<MetricResult>,
    uncertainty: Uncertainty,
    execution: Execution,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    source_id: String,
    version: String,
    source_ref: String,
    source_digest: String,
    license_ref: String,
    retrieved_at: String,
    selected_subset_digest: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    metric: String,
    definition_digest: String,
    numerator: String,
    denominator: String,
    reference_protocol_digest: String,
    reference_condition_digest: String,
    unit: String,
    label_uncertainty: LabelUncertainty,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelUncertainty {
    method: String,
    standard_uncertainty: f64,
    unit: String,
    evidence_digest: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evaluation {
    mode: String,
    group_level: String,
    generalization_claim: String,
    split_manifest_digest: String,
    split_frozen_at_ms: u64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Splits {
    train: Split,
    validation: Split,
    test: Split,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Split {
    split_digest: String,
    window_start_ms: u64,
    window_end_ms: u64,
    entities: Vec<Entity>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entity {
    sample_digest: String,
    group_identity_digest: String,
    cell_digest: Option<String>,
    module_digest: Option<String>,
    pack_digest: Option<String>,
    vehicle_digest: Option<String>,
    lab_digest: Option<String>,
    domain_digest: Option<String>,
    observation_start_ms: u64,
    observation_end_ms: u64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preprocessing {
    transform_id: String,
    method: String,
    fit_scope: String,
    uses_validation_or_test_labels: bool,
    transform_digest: String,
    predeclared_at_ms: Option<u64>,
    predeclaration_digest: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetAdaptation {
    mode: String,
    unlabeled_target_budget: usize,
    labeled_target_budget: usize,
    protocol_digest: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetricResult {
    name: String,
    formula: String,
    unit: String,
    aggregation: String,
    value: f64,
    sample_count: usize,
    result_digest: String,
    strata: Vec<StratumResult>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StratumResult {
    identity_digest: String,
    value: f64,
    sample_count: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Uncertainty {
    status: String,
    method: Option<String>,
    nominal_levels: Vec<f64>,
    calibration_partition: String,
    domain_stratified: bool,
    calibration_source_digest: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    source_commit: String,
    dependency_lock_digest: String,
    toolchain: String,
    toolchain_digest: String,
    hardware: String,
    random_seeds: Vec<u64>,
    wall_time_ms: u64,
    peak_memory_bytes: u64,
    energy_wh: Option<f64>,
    environment_digest: String,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    benchmark_id: Option<String>,
    claim_ceiling: String,
    train_entities: usize,
    validation_entities: usize,
    test_entities: usize,
}
#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    cases: Vec<CaseAudit>,
}
#[derive(Debug, Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}
fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}
fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value = serde_json::from_slice(bytes).map_err(|e| format!("schema JSON: {e}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some("https://json-schema.org/draft/2020-12/schema") {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match SOH manifest v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options().should_validate_formats(true).build(&schema)
        .map_err(|e| format!("schema compilation: {e}"))
}
fn expected_claim(mode: &str) -> Option<&'static str> {
    match mode {
        "same_cell_temporal" => Some("future_within_seen_cell"),
        "unseen_cell" => Some("unseen_cell_transfer"),
        "unseen_module" => Some("unseen_module_transfer"),
        "unseen_pack" => Some("unseen_pack_transfer"),
        "unseen_vehicle" => Some("unseen_vehicle_transfer"),
        "unseen_lab" => Some("unseen_lab_transfer"),
        "unseen_domain" => Some("unseen_domain_transfer"),
        _ => None,
    }
}
fn expected_group(mode: &str) -> Option<&'static str> {
    match mode {
        "same_cell_temporal" | "unseen_cell" => Some("cell"),
        "unseen_module" => Some("module"),
        "unseen_pack" => Some("pack"),
        "unseen_vehicle" => Some("vehicle"),
        "unseen_lab" => Some("lab"),
        "unseen_domain" => Some("domain"),
        _ => None,
    }
}
fn entity_group<'a>(entity: &'a Entity, level: &str) -> Option<&'a str> {
    match level {
        "cell" => entity.cell_digest.as_deref(),
        "module" => entity.module_digest.as_deref(),
        "pack" => entity.pack_digest.as_deref(),
        "vehicle" => entity.vehicle_digest.as_deref(),
        "lab" => entity.lab_digest.as_deref(),
        "domain" => entity.domain_digest.as_deref(),
        _ => None,
    }
}
fn validate_split(name: &str, split: &Split, errors: &mut Vec<String>) {
    if !is_digest(&split.split_digest) || split.entities.is_empty() {
        errors.push(format!("{name} split needs a digest and at least one entity"));
    }
    if split.window_start_ms > split.window_end_ms {
        errors.push(format!("{name} split has an inverted time window"));
    }
    let mut samples = BTreeSet::new();
    for entity in &split.entities {
        if !is_digest(&entity.sample_digest) || !is_digest(&entity.group_identity_digest)
            || !samples.insert(entity.sample_digest.clone()) {
            errors.push(format!("{name} split has invalid or duplicate sample/group identity"));
        }
        if entity.observation_start_ms > entity.observation_end_ms
            || entity.observation_start_ms < split.window_start_ms
            || entity.observation_end_ms > split.window_end_ms {
            errors.push(format!("{name} entity falls outside its declared observation window"));
        }
        for item in [&entity.cell_digest,&entity.module_digest,&entity.pack_digest,&entity.vehicle_digest,&entity.lab_digest,&entity.domain_digest] {
            if item.as_ref().is_some_and(|digest| !is_digest(digest)) {
                errors.push(format!("{name} entity contains a malformed hierarchy digest"));
            }
        }
    }
}
fn collect_groups(split: &Split, level: &str) -> Result<BTreeSet<String>, Vec<String>> {
    let mut groups = BTreeSet::new();
    let mut errors = Vec::new();
    for entity in &split.entities {
        match entity_group(entity, level) {
            Some(digest) if is_digest(digest) => {
                if entity.group_identity_digest != digest {
                    errors.push(format!("{level} group identity does not match selected hierarchy digest"));
                }
                groups.insert(digest.to_owned());
            }
            _ => errors.push(format!("entity lacks a valid {level} identity")),
        }
    }
    if errors.is_empty() { Ok(groups) } else { Err(errors) }
}
fn validate_manifest(m: &Manifest) -> Vec<String> {
    let mut errors = Vec::new();
    if m.protocol_version != PROTOCOL_VERSION || !nonempty(&m.benchmark_id) {
        errors.push("protocol version or benchmark identity is invalid".to_owned());
    }
    if !matches!(m.evidence_class.as_str(),"physical_observation"|"derived_from_physical"|"simulation"|"synthetic_fixture"|"vendor_claim") {
        errors.push("unknown benchmark evidence class".to_owned());
    }
    let d=&m.dataset;
    if !nonempty(&d.source_id)||!nonempty(&d.version)||!nonempty(&d.source_ref)||!nonempty(&d.license_ref)
        ||d.retrieved_at.len()<10||!is_digest(&d.source_digest)||!is_digest(&d.selected_subset_digest) {
        errors.push("dataset provenance, license or selected-subset identity is incomplete".to_owned());
    }
    let t=&m.target;
    if !nonempty(&t.numerator)||!nonempty(&t.denominator)||!is_digest(&t.definition_digest)
        ||!is_digest(&t.reference_protocol_digest)||!is_digest(&t.reference_condition_digest)
        ||!nonempty(&t.label_uncertainty.method)||!t.label_uncertainty.standard_uncertainty.is_finite()
        ||t.label_uncertainty.standard_uncertainty<0.0||!nonempty(&t.label_uncertainty.unit)
        ||!is_digest(&t.label_uncertainty.evidence_digest) {
        errors.push("operational target or label-uncertainty definition is incomplete".to_owned());
    }
    let e=&m.evaluation;
    if !is_digest(&e.split_manifest_digest)
        || e.split_frozen_at_ms == 0
        || expected_claim(&e.mode) != Some(e.generalization_claim.as_str())
        || expected_group(&e.mode) != Some(e.group_level.as_str())
    {
        errors.push("evaluation mode, physical grouping level, claim or split digest is inconsistent".to_owned());
    }
    validate_split("train",&m.splits.train,&mut errors);
    validate_split("validation",&m.splits.validation,&mut errors);
    validate_split("test",&m.splits.test,&mut errors);

    let mut samples=BTreeSet::new();
    for (name,split) in [("train",&m.splits.train),("validation",&m.splits.validation),("test",&m.splits.test)] {
        for entity in &split.entities {
            if !samples.insert(entity.sample_digest.clone()) {
                errors.push(format!("sample identity overlaps partitions (detected in {name})"));
            }
        }
    }
    let train=collect_groups(&m.splits.train,&e.group_level);
    let validation=collect_groups(&m.splits.validation,&e.group_level);
    let test=collect_groups(&m.splits.test,&e.group_level);
    match (train,validation,test) {
        (Ok(tr),Ok(va),Ok(te)) if e.mode=="same_cell_temporal" => {
            if !te.is_subset(&tr)||!va.is_subset(&tr) {
                errors.push("same-cell temporal evaluation requires validation/test cells to occur in training".to_owned());
            }
            if m.splits.train.window_end_ms>=m.splits.validation.window_start_ms
                ||m.splits.validation.window_end_ms>=m.splits.test.window_start_ms {
                errors.push("same-cell temporal windows must be strictly chronological and non-overlapping".to_owned());
            }
        }
        (Ok(tr),Ok(va),Ok(te)) => {
            let overlap=tr.intersection(&va).count()+tr.intersection(&te).count()+va.intersection(&te).count();
            if overlap>0 {
                errors.push(format!("{}-level physical identities overlap across partitions: {overlap}",e.group_level));
            }
        }
        (tr,va,te) => {
            for result in [tr,va,te] {
                if let Err(mut local)=result { errors.append(&mut local); }
            }
        }
    }

    let mut transform_ids=BTreeSet::new();
    for x in &m.preprocessing {
        if !nonempty(&x.transform_id)||!transform_ids.insert(x.transform_id.clone())
            ||!nonempty(&x.method)||!is_digest(&x.transform_digest) {
            errors.push("preprocessing transform identity or digest is invalid".to_owned());
        }
        if x.uses_validation_or_test_labels {
            errors.push(format!("preprocessing transform {} uses validation/test labels",x.transform_id));
        }
        match x.fit_scope.as_str() {
            "none" | "train_only" => {
                if x.predeclared_at_ms.is_some() || x.predeclaration_digest.is_some() {
                    errors.push(format!(
                        "transform {} unexpectedly carries a fixed-predeclaration receipt",
                        x.transform_id
                    ));
                }
            }
            "fixed_predeclared" => {
                if x.predeclared_at_ms.is_none_or(|at| at >= m.evaluation.split_frozen_at_ms)
                    || x.predeclaration_digest.as_deref().is_none_or(|digest| !is_digest(digest))
                {
                    errors.push(format!(
                        "fixed transform {} was not evidenced before the split freeze",
                        x.transform_id
                    ));
                }
            }
            "unlabeled_target" => {
                if m.target_adaptation.mode != "unlabeled_target"
                    || m.target_adaptation.unlabeled_target_budget == 0
                    || m.target_adaptation.labeled_target_budget != 0
                    || x.predeclared_at_ms.is_some()
                    || x.predeclaration_digest.is_some()
                {
                    errors.push(format!(
                        "transform {} uses target adaptation without an explicit label-free budget",
                        x.transform_id
                    ));
                }
            }
            _ => errors.push(format!("transform {} has a prohibited fit scope", x.transform_id)),
        }
    }
    let a=&m.target_adaptation;
    if !is_digest(&a.protocol_digest)
        || a.labeled_target_budget != 0
        || (a.mode == "none" && a.unlabeled_target_budget != 0)
        || (a.mode == "unlabeled_target" && a.unlabeled_target_budget == 0)
        || !matches!(a.mode.as_str(), "none" | "unlabeled_target")
    {
        errors.push("target adaptation must be explicitly label-free and budgeted".to_owned());
    }

    let mut metric_keys = BTreeSet::new();
    let mut seen_metrics = BTreeMap::<String, BTreeSet<String>>::new();
    let mut required_coverage_levels = BTreeSet::new();
    let mut test_group_cache = BTreeMap::<String, BTreeSet<String>>::new();
    for metric in &m.metrics {
        if !matches!(
            metric.name.as_str(),
            "mae" | "rmse" | "r2" | "interval_coverage_90" | "interval_coverage_95"
                | "mean_interval_score" | "p95_absolute_error"
        ) || !metric_keys.insert((metric.name.clone(), metric.aggregation.clone()))
            || !nonempty(&metric.formula)
            || !nonempty(&metric.unit)
            || !metric.value.is_finite()
            || metric.sample_count == 0
            || !is_digest(&metric.result_digest)
        {
            errors.push("metric definition, value or result evidence is invalid".to_owned());
        }
        let required_group = match metric.aggregation.as_str() {
            "macro_cell" | "p95_cell" => Some("cell"),
            "macro_pack" => Some("pack"),
            "macro_vehicle" => Some("vehicle"),
            "macro_domain" | "worst_domain" => Some("domain"),
            "micro_sample" => None,
            _ => {
                errors.push(format!(
                    "metric {} has an unknown aggregation rule",
                    metric.name
                ));
                None
            }
        };
        if let Some(group) = required_group {
            let expected = test_group_cache.entry(group.to_owned()).or_insert_with(|| {
                m.splits
                    .test
                    .entities
                    .iter()
                    .filter_map(|entity| entity_group(entity, group))
                    .map(str::to_owned)
                    .collect()
            });
            if expected.is_empty() {
                errors.push(format!("test split has no {group} identities for metric {}", metric.name));
            }
            let mut strata = BTreeSet::new();
            let mut count_sum = 0usize;
            let mut values = Vec::new();
            for row in &metric.strata {
                if !is_digest(&row.identity_digest)
                    || !strata.insert(row.identity_digest.clone())
                    || !row.value.is_finite()
                    || row.sample_count == 0
                {
                    errors.push(format!("metric {} has invalid or duplicate strata", metric.name));
                }
                if !expected.contains(&row.identity_digest) {
                    errors.push(format!(
                        "metric {} includes a stratum identity absent from the test split",
                        metric.name
                    ));
                }
                count_sum = count_sum.saturating_add(row.sample_count);
                values.push(row.value);
            }
            if &strata != expected {
                errors.push(format!(
                    "metric {} strata do not exactly cover test {} identities",
                    metric.name, group
                ));
            }
            if count_sum != metric.sample_count {
                errors.push(format!(
                    "metric {} stratum sample counts do not sum to its reported sample count",
                    metric.name
                ));
            }
            if matches!(
                metric.aggregation.as_str(),
                "macro_cell" | "macro_pack" | "macro_vehicle" | "macro_domain"
            ) {
                let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
                if !mean.is_finite() || (metric.value - mean).abs() > 1e-9 {
                    errors.push(format!(
                        "metric {} macro value does not equal the mean of its test strata",
                        metric.name
                    ));
                }
            }
            if metric.aggregation == "p95_cell" {
                let mut sorted = values.clone();
                sorted.sort_by(f64::total_cmp);
                let rank = ((0.95 * sorted.len() as f64).ceil() as usize)
                    .max(1)
                    .min(sorted.len());
                if sorted.is_empty() || (metric.value - sorted[rank - 1]).abs() > 1e-9 {
                    errors.push(format!(
                        "metric {} p95 value does not equal the nearest-rank 95th percentile",
                        metric.name
                    ));
                }
            }
            if metric.aggregation == "worst_domain" {
                if group != "domain" || values.len() < 2 {
                    errors.push(format!(
                        "worst-domain metric {} needs at least two test-domain strata",
                        metric.name
                    ));
                }
                let worst = if matches!(
                    metric.name.as_str(),
                    "r2" | "interval_coverage_90" | "interval_coverage_95"
                ) {
                    values.iter().copied().fold(f64::INFINITY, f64::min)
                } else {
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                };
                if !worst.is_finite() || (metric.value - worst).abs() > 1e-9 {
                    errors.push(format!(
                        "metric {} worst-domain value does not match its domain strata",
                        metric.name
                    ));
                }
                if !m.uncertainty.domain_stratified {
                    errors.push(format!(
                        "worst-domain metric {} requires domain-stratified uncertainty reporting",
                        metric.name
                    ));
                }
            }
        } else if !metric.strata.is_empty() {
            errors.push(format!(
                "micro-sample metric {} must not declare physical-group strata",
                metric.name
            ));
        }
        if metric.name == "interval_coverage_90" {
            required_coverage_levels.insert(0.90f64.to_bits());
        }
        if metric.name == "interval_coverage_95" {
            required_coverage_levels.insert(0.95f64.to_bits());
        }
        seen_metrics
            .entry(metric.name.clone())
            .or_default()
            .insert(metric.aggregation.clone());
    }
    if metric_keys.is_empty() {
        errors.push("benchmark metric set must not be empty".to_owned());
    }
    let u=&m.uncertainty;
    match u.status.as_str() {
        "reported" => {
            if u.method.as_deref().is_none_or(|s| !nonempty(s))
                || u.nominal_levels.is_empty()
                || !matches!(u.calibration_partition.as_str(), "validation_only" | "external_calibration")
            {
                errors.push("reported uncertainty requires method, nominal levels and held-out calibration provenance".to_owned());
            }
            if u.calibration_partition == "external_calibration"
                && u.calibration_source_digest.as_deref().is_none_or(|digest| !is_digest(digest))
            {
                errors.push("external uncertainty calibration requires a source digest".to_owned());
            }
            if u.calibration_partition == "validation_only" && u.calibration_source_digest.is_some() {
                errors.push("validation-only calibration must not claim an external calibration source".to_owned());
            }
            for level in &u.nominal_levels {
                let metric_name = if (*level - 0.90).abs() < 1e-9 {
                    "interval_coverage_90"
                } else if (*level - 0.95).abs() < 1e-9 {
                    "interval_coverage_95"
                } else {
                    ""
                };
                if metric_name.is_empty()
                    || !seen_metrics.get(metric_name).is_some_and(|aggregations| {
                        aggregations.contains("worst_domain")
                    })
                {
                    errors.push(format!(
                        "reported uncertainty level {level} lacks a matching worst-domain coverage metric"
                    ));
                }
            }
            if u.nominal_levels.iter().any(|x|!x.is_finite()||*x<=0.0||*x>=1.0) {
                errors.push("uncertainty nominal levels are invalid".to_owned());
            }
        }
        "not_reported" => {
            if u.method.is_some()
                || !u.nominal_levels.is_empty()
                || u.calibration_partition != "none"
                || u.domain_stratified
                || u.calibration_source_digest.is_some()
            {
                errors.push("not_reported uncertainty must not claim calibration evidence".to_owned());
            }
        }
        "not_applicable" => {
            if u.method.is_some()
                || !u.nominal_levels.is_empty()
                || u.calibration_partition != "none"
                || u.domain_stratified
                || u.calibration_source_digest.is_some()
            {
                errors.push(
                    "not_applicable uncertainty must not claim method or calibration evidence"
                        .to_owned(),
                );
            }
        }
        _ => errors.push("unknown uncertainty status".to_owned()),
    }
    if !matches!(u.calibration_partition.as_str(),"validation_only"|"train_only"|"external_calibration"|"none") {
        errors.push("unknown uncertainty calibration partition".to_owned());
    }
    let x=&m.execution;
    let mut seeds=x.random_seeds.clone();seeds.sort_unstable();
    if !is_commit(&x.source_commit)||!is_digest(&x.dependency_lock_digest)||!nonempty(&x.toolchain)
        ||!is_digest(&x.toolchain_digest)||!nonempty(&x.hardware)||seeds.is_empty()
        ||seeds.windows(2).any(|pair|pair[0]==pair[1])
        ||x.energy_wh.is_some_and(|v|!v.is_finite()||v<0.0)||!is_digest(&x.environment_digest) {
        errors.push("execution/reproducibility manifest is incomplete or invalid".to_owned());
    }
    errors
}

fn canonicalize(value:&mut Value) {
    let Some(cases)=value.get_mut("cases").and_then(Value::as_array_mut) else{return};
    cases.sort_by(|a,b|a.get("case_id").and_then(Value::as_str).cmp(&b.get("case_id").and_then(Value::as_str)));
    for case in cases {
        if let Some(manifest)=case.get_mut("manifest") {
            for partition in ["train","validation","test"] {
                if let Some(rows)=manifest.get_mut("splits").and_then(|s|s.get_mut(partition))
                    .and_then(|s|s.get_mut("entities")).and_then(Value::as_array_mut) {
                    rows.sort_by(|a,b|a.get("sample_digest").and_then(Value::as_str).cmp(&b.get("sample_digest").and_then(Value::as_str)));
                }
            }
            if let Some(rows)=manifest.get_mut("preprocessing").and_then(Value::as_array_mut) {
                rows.sort_by(|a,b|a.get("transform_id").and_then(Value::as_str).cmp(&b.get("transform_id").and_then(Value::as_str)));
            }
            if let Some(seeds)=manifest.get_mut("execution")
                .and_then(|execution| execution.get_mut("random_seeds"))
                .and_then(Value::as_array_mut) {
                seeds.sort_by(|a,b| a.as_u64().cmp(&b.as_u64()));
            }
            if let Some(rows)=manifest.get_mut("metrics").and_then(Value::as_array_mut) {
                rows.sort_by(|a,b|a.get("name").and_then(Value::as_str).cmp(&b.get("name").and_then(Value::as_str)));
                for metric in rows {
                    if let Some(strata)=metric.get_mut("strata").and_then(Value::as_array_mut) {
                        strata.sort_by(|a,b|a.get("identity_digest").and_then(Value::as_str).cmp(&b.get("identity_digest").and_then(Value::as_str)));
                    }
                }
            }
        }
    }
}

pub fn audit_corpus(schema_bytes:&[u8],corpus_bytes:&[u8])->Result<CorpusReport,String> {
    let validator=schema_validator(schema_bytes)?;
    let mut value:Value=serde_json::from_slice(corpus_bytes).map_err(|e|format!("corpus JSON: {e}"))?;
    canonicalize(&mut value);
    let canonical=serde_json::to_vec(&value).map_err(|e|format!("canonical corpus: {e}"))?;
    let corpus:Corpus=serde_json::from_slice(&canonical).map_err(|e|format!("typed corpus: {e}"))?;
    if corpus.corpus_version!=CORPUS_VERSION||corpus.schema_id!=SCHEMA_ID {
        return Err("corpus version or schema identity mismatch".to_owned());
    }
    let mut seen=BTreeSet::new();
    let mut cases=Vec::new();
    let mut accepted_count=0usize;
    let mut rejected_count=0usize;
    for case in corpus.cases {
        if !nonempty(&case.case_id)||!seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}",case.case_id));
        }
        if !matches!(case.expectation.as_str(),"accept"|"reject") {
            return Err(format!("case {} has invalid expectation",case.case_id));
        }
        let schema_errors:Vec<String>=validator.iter_errors(&case.manifest).map(|e|e.to_string()).collect();
        let result=if schema_errors.is_empty() {
            match serde_json::from_value::<Manifest>(case.manifest) {
                Ok(manifest)=> {
                    let errors=validate_manifest(&manifest);
                    if errors.is_empty(){Ok(manifest)}else{Err(errors)}
                }
                Err(e)=>Err(vec![format!("typed manifest parse: {e}")])
            }
        } else {Err(schema_errors)};
        let accepted=result.is_ok();
        if accepted!=(case.expectation=="accept") {
            let detail=match &result {Ok(_)=>"unexpected acceptance".to_owned(),Err(e)=>e.join("; ")};
            return Err(format!("case {} expectation mismatch: {}",case.case_id,detail));
        }
        let (manifest,errors)=match result {
            Ok(manifest)=>{accepted_count+=1;(Some(manifest),Vec::new())}
            Err(errors)=>{rejected_count+=1;(None,errors)}
        };
        let claim_ceiling=match manifest.as_ref().map(|m|m.evidence_class.as_str()) {
            Some("physical_observation"|"derived_from_physical")=>"producer_declared_physical_protocol_not_independently_attested",
            Some("simulation")=>"simulation_only",
            Some("synthetic_fixture")=>"synthetic_software_validation_only",
            Some("vendor_claim")=>"vendor_claim_only",
            _=>"rejected_manifest"
        }.to_owned();
        cases.push(CaseAudit{
            case_id:case.case_id,
            expectation:case.expectation,
            accepted,
            errors,
            benchmark_id:manifest.as_ref().map(|m|m.benchmark_id.clone()),
            claim_ceiling,
            train_entities:manifest.as_ref().map_or(0,|m|m.splits.train.entities.len()),
            validation_entities:manifest.as_ref().map_or(0,|m|m.splits.validation.entities.len()),
            test_entities:manifest.as_ref().map_or(0,|m|m.splits.test.entities.len())
        });
    }
    let payload=ReportPayload{
        corpus_version:CORPUS_VERSION.to_owned(),
        schema_id:SCHEMA_ID.to_owned(),
        schema_sha256:sha256(schema_bytes),
        canonical_corpus_sha256:sha256(&canonical),
        case_count:cases.len(),
        accepted_count,
        rejected_count,
        cases
    };
    let bytes=serde_json::to_vec(&payload).map_err(|e|format!("report serialization: {e}"))?;
    Ok(CorpusReport{payload,report_sha256:sha256(&bytes)})
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    const SCHEMA:&[u8]=include_bytes!("../../../schemas/transport/ev_soh_benchmark_manifest_v0_1.schema.json");
    const CORPUS:&[u8]=include_bytes!("../../../fixtures/ev_battery/soh_benchmark_manifest_scenarios_v0_1.json");
    fn report()->CorpusReport{audit_corpus(SCHEMA,CORPUS).expect("frozen SOH benchmark manifest corpus")}

    #[test]
    fn frozen_corpus_accepts_three_and_rejects_three_controls(){
        let report=report();assert_eq!(report.payload.case_count,6);assert_eq!(report.payload.accepted_count,3);assert_eq!(report.payload.rejected_count,3);
    }
    #[test]
    fn metric_strata_are_bound_to_test_physical_groups_and_aggregate_values() {
        let r = report();
        let clean = r.payload.cases.iter()
            .find(|c| c.case_id == "clean-unseen-cell")
            .expect("clean benchmark");
        assert!(clean.accepted);
        for case_id in [
            "metric-stratum-identity-outside-test-rejected",
            "metric-stratum-count-mismatch-rejected",
            "metric-aggregate-value-mismatch-rejected",
        ] {
            let case = r.payload.cases.iter().find(|c| c.case_id == case_id).unwrap();
            assert!(!case.accepted);
        }
    }

    #[test]
    fn fixed_transforms_require_predeclaration_before_split_freeze() {
        let r = report();
        assert!(r.payload.cases.iter()
            .find(|c| c.case_id == "fixed-predeclared-transform-before-split")
            .unwrap().accepted);
        assert!(!r.payload.cases.iter()
            .find(|c| c.case_id == "fixed-predeclared-transform-after-split-rejected")
            .unwrap().accepted);
    }

    #[test]
    fn reported_uncertainty_requires_heldout_coverage_by_domain() {
        let r = report();
        let case = r.payload.cases.iter()
            .find(|c| c.case_id == "reported-uncertainty-without-coverage-metrics-rejected")
            .unwrap();
        assert!(!case.accepted);
        let train_calibration = r.payload.cases.iter()
            .find(|c| c.case_id == "train-only-uncertainty-calibration-rejected")
            .unwrap();
        assert!(!train_calibration.accepted);
    }

    #[test]
    fn unseen_cell_task_rejects_shared_physical_cell_identity(){
        let r=report();let c=r.payload.cases.iter().find(|c|c.case_id=="shared-cell-leak-in-unseen-cell-task").unwrap();assert!(!c.accepted);
    }
    #[test]
    fn same_cell_temporal_task_is_not_mislabeled_unseen_cell_transfer(){
        let r=report();let c=r.payload.cases.iter().find(|c|c.case_id=="same-cell-temporal-forward").unwrap();assert!(c.accepted);
    }
    #[test]
    fn validation_or_test_label_preprocessing_is_rejected(){
        let r=report();let c=r.payload.cases.iter().find(|c|c.case_id=="test-label-preprocessing-leak").unwrap();assert!(!c.accepted);
    }
    #[test]
    fn explicit_unlabeled_target_adaptation_has_zero_target_label_budget(){
        let r=report();let c=r.payload.cases.iter().find(|c|c.case_id=="unlabeled-target-adaptation-explicit").unwrap();assert!(c.accepted);
    }
    #[test]
    fn target_adaptation_requires_nonzero_unlabeled_budget_and_zero_labels() {
        let r = report();
        let c = r.payload.cases.iter()
            .find(|c| c.case_id == "zero-budget-target-adaptation-rejected")
            .expect("zero budget adaptation control");
        assert!(!c.accepted);
    }

    #[test]
    fn metric_aggregation_must_match_declared_physical_group() {
        let r = report();
        let c = r.payload.cases.iter()
            .find(|c| c.case_id == "aggregation-must-match-physical-group-rejected")
            .expect("aggregation mismatch control");
        assert!(!c.accepted);
    }

    #[test]
    fn duplicate_sample_digest_across_partitions_is_rejected(){
        let r=report();let c=r.payload.cases.iter().find(|c|c.case_id=="duplicate-sample-cross-split").unwrap();assert!(!c.accepted);
    }
    #[test]
    fn canonicalization_is_invariant_to_entity_and_metric_order(){
        let original=report();let mut value:Value=serde_json::from_slice(CORPUS).unwrap();value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap(){
            for partition in ["train","validation","test"]{case["manifest"]["splits"][partition]["entities"].as_array_mut().unwrap().reverse();}
            case["manifest"]["metrics"].as_array_mut().unwrap().reverse();
            case["manifest"]["execution"]["random_seeds"].as_array_mut().unwrap().reverse();
            for metric in case["manifest"]["metrics"].as_array_mut().unwrap(){metric["strata"].as_array_mut().unwrap().reverse();}
        }
        let reversed=serde_json::to_vec(&value).unwrap();let replay=audit_corpus(SCHEMA,&reversed).unwrap();assert_eq!(original.report_sha256,replay.report_sha256);
    }
}

use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-pack-capability-metrics/0.2";
const CORPUS_VERSION: &str = "ev-pack-capability-metrics-corpus/0.2";
const PROFILE_VERSION: &str = "ev-pack-capability-metrics/0.2";

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
    profile: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    profile_version: String,
    profile_id: String,
    asset: Asset,
    population: Population,
    topology: Topology,
    reference_conditions: ReferenceConditions,
    provenance: Provenance,
    units: Vec<Unit>,
    soc_snapshot: Option<SocSnapshot>,
    lifetime_evidence: Option<LifetimeEvidence>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    vehicle_ref: String,
    vehicle_configuration_digest: String,
    pack_ref: String,
    pack_configuration_digest: String,
    chemistry: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Population {
    comparison_resolution: String,
    declared_unit_count: usize,
    observed_unit_count: usize,
    inventory_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Topology {
    status: String,
    architecture: String,
    topology_ref: String,
    topology_digest: String,
    qualification: Option<TopologyQualification>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TopologyQualification {
    qualification_ref: String,
    authority_ref: String,
    evidence_digest: String,
    evidence_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceConditions {
    capacity: CapacityCondition,
    resistance: ResistanceCondition,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityCondition {
    condition_ref: String,
    temperature_c: f64,
    maximum_measurement_skew_ms: u64,
    c_rate: f64,
    lower_cutoff_voltage_v: f64,
    upper_cutoff_voltage_v: f64,
    protocol_ref: String,
    protocol_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResistanceCondition {
    condition_ref: String,
    temperature_c: f64,
    soc_fraction: f64,
    pulse_current_a: f64,
    pulse_duration_ms: u64,
    protocol_ref: String,
    protocol_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    source_batch_digest: String,
    analysis_method_ref: String,
    analysis_method_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Unit {
    unit_ref: String,
    nominal_capacity_ah: f64,
    capacity: CapacityMeasurement,
    resistance: ResistanceMeasurement,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityMeasurement {
    value_ah: f64,
    uncertainty_ah: f64,
    measurement_time_ms: u64,
    evidence_class: String,
    method: String,
    evidence_digest: String,
    condition_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResistanceMeasurement {
    value_mohm: f64,
    uncertainty_mohm: f64,
    evidence_class: String,
    method: String,
    evidence_digest: String,
    condition_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SocSnapshot {
    event_type: String,
    snapshot_ref: String,
    pack_configuration_digest: String,
    condition_ref: String,
    maximum_sample_skew_ms: u64,
    maximum_soh_measurement_age_ms: u64,
    snapshot_digest: String,
    observations: Vec<SocObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SocObservation {
    unit_ref: String,
    soc_fraction: f64,
    sample_time_ms: u64,
    evidence_digest: String,
    evidence_class: String,
    soh_measurement_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LifetimeEvidence {
    eol_rule_ref: String,
    eol_threshold_soh_fraction: f64,
    evidence_digest: String,
    pack_configuration_digest: String,
    eol_rule_digest: String,
    service_history_digest: String,
    efc_method_ref: String,
    efc_method_digest: String,
    units: Vec<LifetimeUnit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LifetimeUnit {
    unit_ref: String,
    status: String,
    efc_to_eol: Option<f64>,
    eol_event_ref: Option<String>,
    evidence_digest: Option<String>,
    evidence_class: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricEvidence {
    pub role: String,
    pub digest: String,
    pub evidence_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricResult {
    pub status: String,
    pub value: Option<f64>,
    pub unit: String,
    pub population_scope: String,
    pub evidence: Vec<MetricEvidence>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct InconsistencyMetrics {
    pub sigma_soh: MetricResult,
    pub beta_soh: MetricResult,
    pub beta_life: MetricResult,
    pub beta_soc: MetricResult,
    pub beta_power: MetricResult,
    pub beta_energy: MetricResult,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DistributionSummary {
    pub mean: f64,
    pub median_nearest_rank: f64,
    pub minimum: f64,
    pub p05_nearest_rank: f64,
    pub p95_nearest_rank: f64,
    pub population_standard_deviation: f64,
    pub coefficient_of_variation: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfileSummary {
    pub profile_id: String,
    pub vehicle_ref: String,
    pub vehicle_configuration_digest: String,
    pub pack_ref: String,
    pub pack_configuration_digest: String,
    pub chemistry: String,
    pub comparison_resolution: String,
    pub topology_architecture: String,
    pub declared_unit_count: usize,
    pub observed_unit_count: usize,
    pub coverage_fraction: f64,
    pub population_scope: String,
    pub soh_fraction: DistributionSummary,
    pub resistance_mohm: DistributionSummary,
    pub inconsistency_metrics: InconsistencyMetrics,
    pub uncertainty_status: String,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    summary: Option<ProfileSummary>,
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
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn schema_validator(schema_bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(schema_bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema ID does not match pack-metrics v0.2".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_profile(profile: &Profile) -> Vec<String> {
    let mut errors = Vec::new();
    if profile.profile_version != PROFILE_VERSION || !nonempty(&profile.profile_id) {
        errors.push("profile version or identity is invalid".to_owned());
    }
    if !nonempty(&profile.asset.vehicle_ref)
        || !nonempty(&profile.asset.pack_ref)
        || !nonempty(&profile.asset.chemistry)
        || !is_digest(&profile.asset.vehicle_configuration_digest)
        || !is_digest(&profile.asset.pack_configuration_digest)
        || !is_digest(&profile.population.inventory_digest)
        || !is_digest(&profile.provenance.source_batch_digest)
        || !is_digest(&profile.provenance.analysis_method_digest)
        || !nonempty(&profile.provenance.analysis_method_ref)
    {
        errors.push("asset, inventory or provenance identity is invalid".to_owned());
    }
    if !matches!(
        profile.population.comparison_resolution.as_str(),
        "cell" | "parallel_group" | "module"
    ) {
        errors.push("unsupported comparison resolution".to_owned());
    }
    if profile.population.declared_unit_count < 2
        || profile.population.observed_unit_count < 2
        || profile.population.observed_unit_count > profile.population.declared_unit_count
        || profile.population.observed_unit_count != profile.units.len()
    {
        errors.push("declared/observed counts do not match the unit inventory".to_owned());
    }
    match profile.topology.status.as_str() {
        "qualified" => {
            if profile.topology.architecture == "unknown" {
                errors.push("qualified topology cannot have unknown architecture".to_owned());
            }
        }
        "unknown" => {
            if profile.topology.architecture != "unknown" {
                errors.push("unknown topology must use unknown architecture".to_owned());
            }
        }
        _ => errors.push("unknown topology status".to_owned()),
    }
    if !nonempty(&profile.topology.topology_ref) || !is_digest(&profile.topology.topology_digest) {
        errors.push("topology reference or digest is invalid".to_owned());
    }
    match (
        profile.topology.status.as_str(),
        &profile.topology.qualification,
    ) {
        ("qualified", Some(qualification)) => {
            if !nonempty(&qualification.qualification_ref)
                || !nonempty(&qualification.authority_ref)
                || !is_digest(&qualification.evidence_digest)
                || !matches!(
                    qualification.evidence_class.as_str(),
                    "reference_test_measurement"
                        | "derived_estimate"
                        | "model_prediction"
                        | "simulation_output"
                        | "synthetic_fixture"
                        | "vendor_claim"
                )
            {
                errors.push("topology qualification evidence is invalid".to_owned());
            }
        }
        ("qualified", None) => {
            errors.push(
                "qualified topology requires authority-backed qualification evidence".to_owned(),
            );
        }
        ("unknown", None) => {}
        ("unknown", Some(_)) => {
            errors.push("unknown topology cannot carry a qualification record".to_owned());
        }
        _ => errors.push("topology status is invalid".to_owned()),
    }

    let capacity = &profile.reference_conditions.capacity;
    if !nonempty(&capacity.condition_ref)
        || !nonempty(&capacity.protocol_ref)
        || !is_digest(&capacity.protocol_digest)
        || !capacity.temperature_c.is_finite()
        || capacity.maximum_measurement_skew_ms == 0
        || !capacity.c_rate.is_finite()
        || capacity.c_rate <= 0.0
        || !capacity.lower_cutoff_voltage_v.is_finite()
        || !capacity.upper_cutoff_voltage_v.is_finite()
        || capacity.lower_cutoff_voltage_v <= 0.0
        || capacity.upper_cutoff_voltage_v <= capacity.lower_cutoff_voltage_v
    {
        errors.push("capacity reference condition is invalid".to_owned());
    }
    let capacity_times: Vec<u64> = profile.units.iter()
        .map(|unit| unit.capacity.measurement_time_ms)
        .collect();
    if let (Some(minimum), Some(maximum)) =
        (capacity_times.iter().min(), capacity_times.iter().max())
    {
        if maximum - minimum > capacity.maximum_measurement_skew_ms {
            errors.push(
                "per-cell SOH capacity measurements exceed the declared common-time window"
                    .to_owned(),
            );
        }
    }
    let resistance = &profile.reference_conditions.resistance;
    if !nonempty(&resistance.condition_ref)
        || !nonempty(&resistance.protocol_ref)
        || !is_digest(&resistance.protocol_digest)
        || !resistance.temperature_c.is_finite()
        || !resistance.soc_fraction.is_finite()
        || !(0.0..=1.0).contains(&resistance.soc_fraction)
        || !resistance.pulse_current_a.is_finite()
        || resistance.pulse_current_a <= 0.0
        || resistance.pulse_duration_ms == 0
    {
        errors.push("resistance reference condition is invalid".to_owned());
    }

    let mut unit_ids = BTreeSet::new();
    for unit in &profile.units {
        if !nonempty(&unit.unit_ref) || !unit_ids.insert(unit.unit_ref.clone()) {
            errors.push(format!(
                "unit identity is empty or duplicated: {}",
                unit.unit_ref
            ));
        }
        if !unit.nominal_capacity_ah.is_finite() || unit.nominal_capacity_ah <= 0.0 {
            errors.push(format!(
                "unit {} has invalid nominal capacity",
                unit.unit_ref
            ));
        }
        if !unit.capacity.value_ah.is_finite()
            || unit.capacity.value_ah <= 0.0
            || !unit.capacity.uncertainty_ah.is_finite()
            || unit.capacity.uncertainty_ah < 0.0
            || !unit.resistance.value_mohm.is_finite()
            || unit.resistance.value_mohm <= 0.0
            || !unit.resistance.uncertainty_mohm.is_finite()
            || unit.resistance.uncertainty_mohm < 0.0
        {
            errors.push(format!(
                "unit {} has invalid capacity or resistance",
                unit.unit_ref
            ));
        }
        if unit.capacity.condition_ref != capacity.condition_ref
            || unit.resistance.condition_ref != resistance.condition_ref
        {
            errors.push(format!(
                "unit {} has mismatched reference conditions",
                unit.unit_ref
            ));
        }
        if !is_digest(&unit.capacity.evidence_digest)
            || !is_digest(&unit.resistance.evidence_digest)
            || !nonempty(&unit.capacity.method)
            || !nonempty(&unit.resistance.method)
        {
            errors.push(format!(
                "unit {} has invalid measurement provenance",
                unit.unit_ref
            ));
        }
        for class in [
            unit.capacity.evidence_class.as_str(),
            unit.resistance.evidence_class.as_str(),
        ] {
            if !matches!(
                class,
                "reference_test_measurement"
                    | "derived_estimate"
                    | "model_prediction"
                    | "simulation_output"
                    | "synthetic_fixture"
                    | "vendor_claim"
            ) {
                errors.push(format!(
                    "unit {} has an unknown evidence class",
                    unit.unit_ref
                ));
            }
        }
    }

    if let Some(snapshot) = &profile.soc_snapshot {
        if snapshot.event_type != "end_of_charge"
            || !nonempty(&snapshot.snapshot_ref)
            || !is_digest(&snapshot.snapshot_digest)
            || snapshot.pack_configuration_digest != profile.asset.pack_configuration_digest
            || !nonempty(&snapshot.condition_ref)
            || snapshot.maximum_sample_skew_ms == 0
        {
            errors.push(
                "SOC snapshot identity, configuration or event context is invalid".to_owned(),
            );
        }
        let mut observed = BTreeSet::new();
        let mut times = Vec::new();
        if snapshot.maximum_soh_measurement_age_ms == 0 {
            errors.push("maximum SOH measurement age must be positive".to_owned());
        }
        for row in &snapshot.observations {
            if !unit_ids.contains(&row.unit_ref) || !observed.insert(row.unit_ref.clone()) {
                errors.push(format!(
                    "SOC snapshot has unknown or duplicate unit: {}",
                    row.unit_ref
                ));
            }
            if !row.soc_fraction.is_finite()
                || !(0.0..=1.0).contains(&row.soc_fraction)
                || !is_digest(&row.evidence_digest)
                || !is_digest(&row.soh_measurement_digest)
                || !matches!(
                    row.evidence_class.as_str(),
                    "reference_test_measurement"
                        | "derived_estimate"
                        | "model_prediction"
                        | "simulation_output"
                        | "synthetic_fixture"
                        | "vendor_claim"
                )
            {
                errors.push(format!("SOC observation for {} is invalid", row.unit_ref));
            }
            let unit = profile.units.iter().find(|unit| unit.unit_ref == row.unit_ref);
            if let Some(unit) = unit {
                if unit.capacity.evidence_digest != row.soh_measurement_digest {
                    errors.push(format!(
                        "SOC observation for {} references a different SOH measurement",
                        row.unit_ref
                    ));
                }
                if row.sample_time_ms.abs_diff(unit.capacity.measurement_time_ms)
                    > snapshot.maximum_soh_measurement_age_ms
                {
                    errors.push(format!(
                        "SOC/SOH temporal alignment for {} exceeds the declared age window",
                        row.unit_ref
                    ));
                }
            }
            times.push(row.sample_time_ms);
        }
        if let (Some(minimum), Some(maximum)) = (times.iter().min(), times.iter().max()) {
            if maximum - minimum > snapshot.maximum_sample_skew_ms {
                errors
                    .push("SOC observations exceed the declared synchronization window".to_owned());
            }
        }
    }

    if let Some(lifetime) = &profile.lifetime_evidence {
        if !nonempty(&lifetime.eol_rule_ref)
            || lifetime.pack_configuration_digest != profile.asset.pack_configuration_digest
            || !is_digest(&lifetime.eol_rule_digest)
            || !is_digest(&lifetime.service_history_digest)
            || !nonempty(&lifetime.efc_method_ref)
            || !is_digest(&lifetime.efc_method_digest)
            || !lifetime.eol_threshold_soh_fraction.is_finite()
            || !(0.0..=1.0).contains(&lifetime.eol_threshold_soh_fraction)
            || lifetime.eol_threshold_soh_fraction == 0.0
            || !is_digest(&lifetime.evidence_digest)
        {
            errors.push("lifetime evidence rule, threshold or digest is invalid".to_owned());
        }
        let mut lifetime_ids = BTreeSet::new();
        for row in &lifetime.units {
            if !unit_ids.contains(&row.unit_ref) || !lifetime_ids.insert(row.unit_ref.clone()) {
                errors.push(format!(
                    "lifetime evidence has unknown or duplicate unit: {}",
                    row.unit_ref
                ));
            }
            if !matches!(
                row.evidence_class.as_str(),
                "reference_test_measurement"
                    | "derived_estimate"
                    | "model_prediction"
                    | "simulation_output"
                    | "synthetic_fixture"
                    | "vendor_claim"
            ) {
                errors.push(format!(
                    "lifetime evidence for {} has unknown evidence class",
                    row.unit_ref
                ));
            }
            match row.status.as_str() {
                "observed_eol" => {
                    if row
                        .efc_to_eol
                        .is_none_or(|value| !value.is_finite() || value <= 0.0)
                        || row
                            .eol_event_ref
                            .as_deref()
                            .is_none_or(|value| !nonempty(value))
                        || row
                            .evidence_digest
                            .as_deref()
                            .is_none_or(|value| !is_digest(value))
                    {
                        errors.push(format!(
                            "observed EOL evidence for {} is incomplete",
                            row.unit_ref
                        ));
                    }
                }
                "right_censored" | "unknown" => {
                    if row.efc_to_eol.is_some() || row.eol_event_ref.is_some() {
                        errors.push(format!(
                            "censored/unknown EOL row {} contains a fabricated EOL result",
                            row.unit_ref
                        ));
                    }
                    if row
                        .evidence_digest
                        .as_deref()
                        .is_none_or(|value| !is_digest(value))
                    {
                        errors.push(format!(
                            "censored/unknown EOL row {} lacks evidence digest",
                            row.unit_ref
                        ));
                    }
                }
                _ => errors.push(format!(
                    "lifetime evidence for {} has unknown status",
                    row.unit_ref
                )),
            }
        }
    }
    errors
}

fn nearest_rank(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.max(1).min(sorted.len()) - 1]
}

fn distribution(values: &[f64]) -> DistributionSummary {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let standard_deviation = variance.sqrt();
    DistributionSummary {
        mean,
        median_nearest_rank: nearest_rank(values, 0.50),
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
        p05_nearest_rank: nearest_rank(values, 0.05),
        p95_nearest_rank: nearest_rank(values, 0.95),
        population_standard_deviation: standard_deviation,
        coefficient_of_variation: standard_deviation / mean,
    }
}

fn scope(profile: &Profile) -> String {
    if profile.population.observed_unit_count == profile.population.declared_unit_count {
        "complete_declared_population".to_owned()
    } else {
        "observed_subset_only".to_owned()
    }
}

fn not_computable(unit: &str, scope: &str, reason: &str) -> MetricResult {
    MetricResult {
        status: "NOT_COMPUTABLE".to_owned(),
        value: None,
        unit: unit.to_owned(),
        population_scope: scope.to_owned(),
        evidence: Vec::new(),
        reason: Some(reason.to_owned()),
    }
}

fn computed(
    value: f64,
    unit: &str,
    scope: &str,
    mut evidence: Vec<MetricEvidence>,
) -> MetricResult {
    evidence.sort_by(|left, right| {
        (&left.role, &left.digest, &left.evidence_class).cmp(&(
            &right.role,
            &right.digest,
            &right.evidence_class,
        ))
    });
    evidence.dedup_by(|left, right| {
        left.role == right.role
            && left.digest == right.digest
            && left.evidence_class == right.evidence_class
    });
    MetricResult {
        status: "COMPUTED".to_owned(),
        value: Some(value),
        unit: unit.to_owned(),
        population_scope: scope.to_owned(),
        evidence,
        reason: None,
    }
}

fn metric_evidence(role: String, digest: String, evidence_class: Option<String>) -> MetricEvidence {
    MetricEvidence {
        role,
        digest,
        evidence_class,
    }
}

fn beta_soh_gate(profile: &Profile) -> Result<(), &'static str> {
    series_cell_gate(profile)?;
    let minimum = profile.units.iter()
        .map(|unit| unit.nominal_capacity_ah)
        .fold(f64::INFINITY, f64::min);
    let maximum = profile.units.iter()
        .map(|unit| unit.nominal_capacity_ah)
        .fold(f64::NEG_INFINITY, f64::max);
    if !minimum.is_finite() || maximum - minimum > maximum * 1e-9 {
        return Err("HOMOGENEOUS_NOMINAL_CELL_CAPACITY_REQUIRED");
    }
    Ok(())
}

fn series_cell_gate(profile: &Profile) -> Result<(), &'static str> {
    if profile.population.comparison_resolution != "cell" {
        return Err("CELL_RESOLUTION_REQUIRED");
    }
    if profile.population.observed_unit_count != profile.population.declared_unit_count {
        return Err("COMPLETE_DECLARED_CELL_POPULATION_REQUIRED");
    }
    if profile.topology.status != "qualified"
        || profile.topology.architecture != "series_single_parallel"
    {
        return Err("QUALIFIED_SERIES_SINGLE_PARALLEL_TOPOLOGY_REQUIRED");
    }
    Ok(())
}

fn calculate_metrics(
    profile: &Profile,
    soh: &[f64],
    soh_distribution: &DistributionSummary,
) -> InconsistencyMetrics {
    let population_scope = scope(profile);
    let soh_evidence: Vec<MetricEvidence> = profile
        .units
        .iter()
        .map(|unit| {
            metric_evidence(
                format!("capacity_soh:{}", unit.unit_ref),
                unit.capacity.evidence_digest.clone(),
                Some(unit.capacity.evidence_class.clone()),
            )
        })
        .chain(std::iter::once(metric_evidence(
            "source_batch".to_owned(),
            profile.provenance.source_batch_digest.clone(),
            None,
        )))
        .chain(std::iter::once(metric_evidence(
            "unit_inventory_and_nominal_capacity".to_owned(),
            profile.population.inventory_digest.clone(),
            None,
        )))
        .collect();

    let topology_evidence = || {
        let mut evidence = vec![metric_evidence(
            "topology_descriptor".to_owned(),
            profile.topology.topology_digest.clone(),
            None,
        )];
        if let Some(qualification) = &profile.topology.qualification {
            evidence.push(metric_evidence(
                "topology_qualification".to_owned(),
                qualification.evidence_digest.clone(),
                Some(qualification.evidence_class.clone()),
            ));
        }
        evidence
    };

    let sigma_soh = computed(
        soh_distribution.population_standard_deviation,
        "SOH_fraction",
        &population_scope,
        soh_evidence.clone(),
    );

    let beta_soh = match beta_soh_gate(profile) {
        Ok(()) => {
            let mean = soh.iter().sum::<f64>() / soh.len() as f64;
            let minimum = soh.iter().copied().fold(f64::INFINITY, f64::min);
            computed(
                minimum / mean,
                "ratio",
                &population_scope,
                soh_evidence
                    .iter()
                    .cloned()
                    .chain(topology_evidence())
                    .collect(),
            )
        }
        Err(reason) => not_computable("ratio", &population_scope, reason),
    };

    let beta_life = match series_cell_gate(profile) {
        Err(reason) => not_computable("ratio", &population_scope, reason),
        Ok(()) => match &profile.lifetime_evidence {
            None => not_computable("ratio", &population_scope, "EOL_EFC_EVIDENCE_ABSENT"),
            Some(lifetime) if lifetime.units.len() != profile.population.declared_unit_count => {
                not_computable(
                    "ratio",
                    &population_scope,
                    "COMPLETE_EOL_EFC_RECORDS_REQUIRED",
                )
            }
            Some(lifetime)
                if lifetime
                    .units
                    .iter()
                    .any(|row| row.status != "observed_eol") =>
            {
                not_computable(
                    "ratio",
                    &population_scope,
                    "RIGHT_CENSORED_OR_UNKNOWN_EOL_UNIT",
                )
            }
            Some(lifetime) => {
                let efcs: Vec<f64> = lifetime
                    .units
                    .iter()
                    .filter_map(|row| row.efc_to_eol)
                    .collect();
                if efcs.len() != profile.population.declared_unit_count
                    || efcs.iter().any(|value| !value.is_finite() || *value <= 0.0)
                    || efcs.iter().sum::<f64>() <= 0.0
                {
                    not_computable(
                        "ratio",
                        &population_scope,
                        "VALID_EFC_TO_EOL_REQUIRED_FOR_EVERY_CELL",
                    )
                } else {
                    let minimum = efcs.iter().copied().fold(f64::INFINITY, f64::min);
                    let value = efcs.len() as f64 * minimum / efcs.iter().sum::<f64>();
                    let lifetime_evidence: Vec<MetricEvidence> = lifetime
                        .units
                        .iter()
                        .map(|row| {
                            metric_evidence(
                                format!("efc_to_eol:{}", row.unit_ref),
                                row.evidence_digest.clone().unwrap_or_default(),
                                Some(row.evidence_class.clone()),
                            )
                        })
                        .chain(std::iter::once(metric_evidence(
                            "eol_evidence_batch".to_owned(),
                            lifetime.evidence_digest.clone(),
                            None,
                        )))
                        .chain(std::iter::once(metric_evidence(
                            "eol_rule".to_owned(),
                            lifetime.eol_rule_digest.clone(),
                            None,
                        )))
                        .chain(std::iter::once(metric_evidence(
                            "service_history".to_owned(),
                            lifetime.service_history_digest.clone(),
                            None,
                        )))
                        .chain(std::iter::once(metric_evidence(
                            "efc_method".to_owned(),
                            lifetime.efc_method_digest.clone(),
                            None,
                        )))
                        .chain(topology_evidence())
                        .collect();
                    computed(value, "ratio", &population_scope, lifetime_evidence)
                }
            }
        },
    };

    let beta_soc = match series_cell_gate(profile) {
        Err(reason) => not_computable("ratio", &population_scope, reason),
        Ok(()) => match &profile.soc_snapshot {
            None => not_computable(
                "ratio",
                &population_scope,
                "SYNCHRONIZED_END_OF_CHARGE_SOC_SNAPSHOT_ABSENT",
            ),
            Some(snapshot)
                if snapshot.observations.len() != profile.population.declared_unit_count =>
            {
                not_computable(
                    "ratio",
                    &population_scope,
                    "FULL_CELL_SOC_SNAPSHOT_REQUIRED",
                )
            }
            Some(snapshot) => {
                let soh_by_unit: BTreeMap<&str, f64> = profile
                    .units
                    .iter()
                    .zip(soh.iter())
                    .map(|(unit, value)| (unit.unit_ref.as_str(), *value))
                    .collect();
                let soc_max = snapshot
                    .observations
                    .iter()
                    .map(|row| row.soc_fraction)
                    .fold(f64::NEG_INFINITY, f64::max);
                let total_soh: f64 = soh.iter().sum();
                let weighted_gap: f64 = snapshot
                    .observations
                    .iter()
                    .map(|row| {
                        let unit_soh = soh_by_unit
                            .get(row.unit_ref.as_str())
                            .copied()
                            .unwrap_or(0.0);
                        (soc_max - row.soc_fraction) * unit_soh
                    })
                    .sum();
                if total_soh <= 0.0 || !total_soh.is_finite() {
                    not_computable(
                        "ratio",
                        &population_scope,
                        "POSITIVE_COMPARABLE_SOH_WEIGHT_REQUIRED",
                    )
                } else {
                    let soc_evidence: Vec<MetricEvidence> = snapshot
                        .observations
                        .iter()
                        .map(|row| {
                            metric_evidence(
                                format!("end_of_charge_soc:{}", row.unit_ref),
                                row.evidence_digest.clone(),
                                Some(row.evidence_class.clone()),
                            )
                        })
                        .chain(std::iter::once(metric_evidence(
                            "soc_snapshot".to_owned(),
                            snapshot.snapshot_digest.clone(),
                            None,
                        )))
                        .chain(snapshot.observations.iter().map(|row| {
                            let unit = profile.units.iter()
                                .find(|unit| unit.unit_ref == row.unit_ref)
                                .expect("validated full cell population has matching SOH unit");
                            metric_evidence(
                                format!("soc_soh_binding:{}", row.unit_ref),
                                row.soh_measurement_digest.clone(),
                                Some(unit.capacity.evidence_class.clone()),
                            )
                        }))
                        .chain(soh_evidence.iter().cloned())
                        .chain(topology_evidence())
                        .collect();
                    computed(
                        1.0 - weighted_gap / total_soh,
                        "ratio",
                        &population_scope,
                        soc_evidence,
                    )
                }
            }
        },
    };

    let beta_power = not_computable(
        "ratio",
        &population_scope,
        "QUALIFIED_CELL_AND_PACK_POWER_MODEL_REQUIRED",
    );
    let beta_energy = not_computable(
        "ratio",
        &population_scope,
        "LIFETIME_WEIGHTED_SOC_HISTORY_AND_QUALIFIED_BETA_LIFE_REQUIRED",
    );

    InconsistencyMetrics {
        sigma_soh,
        beta_soh,
        beta_life,
        beta_soc,
        beta_power,
        beta_energy,
    }
}

fn summarize(profile: &Profile) -> ProfileSummary {
    let soh: Vec<f64> = profile
        .units
        .iter()
        .map(|unit| unit.capacity.value_ah / unit.nominal_capacity_ah)
        .collect();
    let resistance: Vec<f64> = profile
        .units
        .iter()
        .map(|unit| unit.resistance.value_mohm)
        .collect();
    let soh_summary = distribution(&soh);
    let resistance_summary = distribution(&resistance);
    let metrics = calculate_metrics(profile, &soh, &soh_summary);
    ProfileSummary {
        profile_id: profile.profile_id.clone(),
        vehicle_ref: profile.asset.vehicle_ref.clone(),
        vehicle_configuration_digest: profile.asset.vehicle_configuration_digest.clone(),
        pack_ref: profile.asset.pack_ref.clone(),
        pack_configuration_digest: profile.asset.pack_configuration_digest.clone(),
        chemistry: profile.asset.chemistry.clone(),
        comparison_resolution: profile.population.comparison_resolution.clone(),
        topology_architecture: profile.topology.architecture.clone(),
        declared_unit_count: profile.population.declared_unit_count,
        observed_unit_count: profile.population.observed_unit_count,
        coverage_fraction: profile.population.observed_unit_count as f64
            / profile.population.declared_unit_count as f64,
        population_scope: scope(profile),
        soh_fraction: soh_summary,
        resistance_mohm: resistance_summary,
        inconsistency_metrics: metrics,
        uncertainty_status: "per_unit_uncertainty_preserved_not_propagated".to_owned(),
    }
}

fn canonicalize_corpus(value: &mut Value) {
    if let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) {
        cases.sort_by(|left, right| {
            left.get("case_id")
                .and_then(Value::as_str)
                .cmp(&right.get("case_id").and_then(Value::as_str))
        });
        for case in cases {
            if let Some(profile) = case.get_mut("profile") {
                if let Some(units) = profile.get_mut("units").and_then(Value::as_array_mut) {
                    units.sort_by(|left, right| {
                        left.get("unit_ref")
                            .and_then(Value::as_str)
                            .cmp(&right.get("unit_ref").and_then(Value::as_str))
                    });
                }
                if let Some(observations) = profile
                    .get_mut("soc_snapshot")
                    .and_then(|value| value.get_mut("observations"))
                    .and_then(Value::as_array_mut)
                {
                    observations.sort_by(|left, right| {
                        left.get("unit_ref")
                            .and_then(Value::as_str)
                            .cmp(&right.get("unit_ref").and_then(Value::as_str))
                    });
                }
                if let Some(units) = profile
                    .get_mut("lifetime_evidence")
                    .and_then(|value| value.get_mut("units"))
                    .and_then(Value::as_array_mut)
                {
                    units.sort_by(|left, right| {
                        left.get("unit_ref")
                            .and_then(Value::as_str)
                            .cmp(&right.get("unit_ref").and_then(Value::as_str))
                    });
                }
            }
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus_value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize_corpus(&mut corpus_value);
    let canonical_corpus = serde_json::to_vec(&corpus_value)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let corpus: Corpus = serde_json::from_slice(&canonical_corpus)
        .map_err(|error| format!("typed corpus: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity does not match v0.2".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;

    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!(
                "case identity is empty or duplicated: {}",
                case.case_id
            ));
        }
        if !matches!(case.expectation.as_str(), "accept" | "reject") {
            return Err(format!("case {} has unknown expectation", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.profile)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Profile>(case.profile) {
                Ok(profile) => {
                    let errors = validate_profile(&profile);
                    if errors.is_empty() {
                        Ok(summarize(&profile))
                    } else {
                        Err(errors)
                    }
                }
                Err(error) => Err(vec![format!("typed profile parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };
        let accepted = outcome.is_ok();
        if accepted != (case.expectation == "accept") {
            let detail = match &outcome {
                Ok(_) => "unexpected acceptance".to_owned(),
                Err(errors) => errors.join("; "),
            };
            return Err(format!(
                "case {} expected {} but accepted={}: {}",
                case.case_id, case.expectation, accepted, detail
            ));
        }
        let (summary, errors) = match outcome {
            Ok(summary) => {
                accepted_count += 1;
                (Some(summary), Vec::new())
            }
            Err(errors) => {
                rejected_count += 1;
                (None, errors)
            }
        };
        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            summary,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical_corpus),
        case_count: cases.len(),
        accepted_count,
        rejected_count,
        cases,
    };
    let report_bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&report_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_pack_capability_metrics_v0_2.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/pack_capability_metrics_v0_2.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen v0.2 corpus")
    }

    #[test]
    fn frozen_corpus_accepts_six_cases_and_rejects_two_controls() {
        let report = report();
        assert_eq!(report.payload.case_count, 12);
        assert_eq!(report.payload.accepted_count, 7);
        assert_eq!(report.payload.rejected_count, 5);
    }

    #[test]
    fn all_six_metrics_have_explicit_status() {
        let report = report();
        let complete = report
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "complete-qualified-series-pack")
            .and_then(|case| case.summary.as_ref())
            .expect("complete pack summary");
        let metrics = &complete.inconsistency_metrics;
        assert_eq!(metrics.sigma_soh.status, "COMPUTED");
        assert_eq!(metrics.beta_soh.status, "COMPUTED");
        assert_eq!(metrics.beta_life.status, "COMPUTED");
        assert_eq!(metrics.beta_soc.status, "COMPUTED");
        assert_eq!(metrics.beta_power.status, "NOT_COMPUTABLE");
        assert_eq!(metrics.beta_energy.status, "NOT_COMPUTABLE");
        assert!((metrics.beta_soh.value.unwrap() - (0.85 / 0.90)).abs() < 1e-12);
        assert!((metrics.beta_life.value.unwrap() - (3.0 * 800.0 / 2700.0)).abs() < 1e-12);
    }

    #[test]
    fn computed_metrics_preserve_digest_to_evidence_class_bindings() {
        let report = report();
        let complete = report
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "complete-qualified-series-pack")
            .and_then(|case| case.summary.as_ref())
            .expect("complete pack summary");
        let metrics = &complete.inconsistency_metrics;
        assert!(metrics.sigma_soh.evidence.iter().any(|item| {
            item.role == "capacity_soh:cell-001"
                && item.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
        assert!(
            metrics
                .sigma_soh
                .evidence
                .iter()
                .any(|item| { item.role == "unit_inventory_and_nominal_capacity" })
        );
        assert!(metrics.beta_soh.evidence.iter().any(|item| {
            item.role == "topology_qualification"
                && item.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
        assert!(metrics.beta_life.evidence.iter().any(|item| {
            item.role == "efc_to_eol:cell-001"
                && item.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
        assert!(metrics.beta_soc.evidence.iter().any(|item| {
            item.role == "end_of_charge_soc:cell-001"
                && item.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
    }

    #[test]
    fn soc_metric_binds_exact_soh_digest_for_each_cell() {
        let audited = report();
        let complete = audited.payload.cases.iter()
            .find(|case| case.case_id == "complete-qualified-series-pack")
            .and_then(|case| case.summary.as_ref())
            .expect("complete pack summary");
        assert!(complete.inconsistency_metrics.beta_soc.evidence.iter().any(|item| {
            item.role == "soc_soh_binding:cell-001"
                && item.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
    }

    #[test]
    fn temporal_and_pack_identity_negative_controls_are_rejected() {
        let audited = report();
        for case_id in [
            "soc-soh-reference-digest-mismatch",
            "stale-soh-reference-rejected",
            "lifetime-pack-configuration-drift-rejected",
        ] {
            let case = audited.payload.cases.iter()
                .find(|case| case.case_id == case_id)
                .expect("negative control case");
            assert!(!case.accepted);
            assert!(case.plan.is_none());
        }
    }

    #[test]
    fn qualified_topology_without_authority_evidence_is_rejected() {
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus");
        value["cases"][0]["profile"]["topology"]
            .as_object_mut()
            .expect("topology")
            .remove("qualification");
        let bytes = serde_json::to_vec(&value).expect("mutated corpus");
        assert!(audit_corpus(SCHEMA, &bytes).is_err());
    }

    #[test]
    fn heterogeneous_nominal_capacity_keeps_beta_soh_not_computable() {
        let audited = report();
        let case = audited.payload.cases.iter()
            .find(|case| case.case_id == "heterogeneous-nominal-capacity-no-beta-soh")
            .and_then(|case| case.summary.as_ref())
            .expect("heterogeneous nominal capacity summary");
        assert_eq!(
            case.inconsistency_metrics.beta_soh.status,
            "NOT_COMPUTABLE"
        );
        assert_eq!(
            case.inconsistency_metrics.beta_soh.reason.as_deref(),
            Some("HOMOGENEOUS_NOMINAL_CELL_CAPACITY_REQUIRED")
        );
        assert_eq!(case.inconsistency_metrics.beta_life.status, "COMPUTED");
        assert_eq!(case.inconsistency_metrics.beta_soc.status, "COMPUTED");
    }

    #[test]
    fn censored_eol_never_gets_dropped_from_beta_life() {
        let audited = report();
        let row = audited
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "censored-eol-explicit")
            .unwrap()
            .summary
            .as_ref()
            .unwrap();
        assert_eq!(row.inconsistency_metrics.beta_life.status, "NOT_COMPUTABLE");
        assert_eq!(
            row.inconsistency_metrics.beta_life.reason.as_deref(),
            Some("RIGHT_CENSORED_OR_UNKNOWN_EOL_UNIT")
        );
    }

    #[test]
    fn partial_coverage_computes_sigma_only_for_observed_subset() {
        let audited = report();
        let row = audited
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "partial-population-explicit")
            .unwrap()
            .summary
            .as_ref()
            .unwrap();
        assert_eq!(row.inconsistency_metrics.sigma_soh.status, "COMPUTED");
        assert_eq!(
            row.inconsistency_metrics.sigma_soh.population_scope,
            "observed_subset_only"
        );
        assert_eq!(row.inconsistency_metrics.beta_soh.status, "NOT_COMPUTABLE");
        assert_eq!(row.inconsistency_metrics.beta_soc.status, "NOT_COMPUTABLE");
    }

    #[test]
    fn module_resolution_never_uses_cell_specific_beta_metrics() {
        let audited = report();
        let row = audited
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "module-resolution-explicit")
            .unwrap()
            .summary
            .as_ref()
            .unwrap();
        assert_eq!(row.inconsistency_metrics.sigma_soh.status, "COMPUTED");
        assert_eq!(row.inconsistency_metrics.beta_soh.status, "NOT_COMPUTABLE");
        assert_eq!(row.inconsistency_metrics.beta_life.status, "NOT_COMPUTABLE");
        assert_eq!(row.inconsistency_metrics.beta_soc.status, "NOT_COMPUTABLE");
    }

    #[test]
    fn corpus_and_unit_order_permutations_have_identical_digest() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["profile"]["units"].as_array_mut().unwrap().reverse();
            if case["profile"]["soc_snapshot"].is_object() {
                case["profile"]["soc_snapshot"]["observations"]
                    .as_array_mut()
                    .unwrap()
                    .reverse();
            }
            if case["profile"]["lifetime_evidence"].is_object() {
                case["profile"]["lifetime_evidence"]["units"]
                    .as_array_mut()
                    .unwrap()
                    .reverse();
            }
        }
        let reversed = serde_json::to_vec(&value).expect("reversed corpus");
        let replay = audit_corpus(SCHEMA, &reversed).expect("permuted corpus");
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256,
            "canonical input identity must ignore corpus and unit order"
        );
        assert_eq!(
            serde_json::to_value(&original.payload).unwrap(),
            serde_json::to_value(&replay.payload).unwrap(),
            "canonicalized report payload must ignore evidence row order"
        );
        assert_eq!(original.report_sha256, replay.report_sha256);
    }
}

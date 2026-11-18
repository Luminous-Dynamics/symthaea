// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission-profile to textile-exposure admission for Atelier Lightshell.
//!
//! This adapter preserves robot-derived exposure distributions and sequence
//! semantics for downstream textile test planning. It does not acquire robot
//! telemetry, infer service life, accelerate failure, execute tests, or
//! authorize robot operation.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MissionExposureManifest {
    pub schema_version: String,
    pub profile_id: String,
    pub state: String,
    pub platform: PlatformBinding,
    pub operating_domain: OperatingDomainBinding,
    pub zones: Vec<ZoneBinding>,
    pub source_runs: Vec<SourceRun>,
    pub observations: Vec<ExposureObservation>,
    pub test_blocks: Vec<TestBlock>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformBinding {
    pub platform_id: String,
    pub article_revision: Option<String>,
    pub geometry_digest: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatingDomainBinding {
    pub odd_profile_ref: Option<String>,
    pub approval_ref: Option<String>,
    pub mission_mix_ref: Option<String>,
    pub service_horizon_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneBinding {
    pub zone_id: String,
    pub function_refs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRun {
    pub source_run_id: String,
    pub source_kind: String,
    pub article_revision: Option<String>,
    pub run_ref: Option<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub clock_sync_ref: Option<String>,
    pub raw_data_digest: Option<String>,
    pub instrument_refs: Vec<String>,
    pub calibration_refs: Vec<String>,
    pub data_quality_state: String,
    pub missing_data_fraction: Option<f64>,
    pub saturation_observed: bool,
    pub gap_disposition_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureObservation {
    pub observation_id: String,
    pub zone_id: String,
    pub source_run_ref: Option<String>,
    pub exposure_kind: String,
    pub evidence_origin: String,
    pub metric_id: String,
    pub unit: String,
    pub sample_count: u64,
    pub minimum: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub maximum: Option<f64>,
    pub frequency_bins: Vec<FrequencyBin>,
    pub method_ref: Option<String>,
    pub uncertainty_model_ref: Option<String>,
    pub derivation_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencyBin {
    pub lower_hz: f64,
    pub upper_hz: f64,
    pub level: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestBlock {
    pub block_id: String,
    pub sequence_index: u32,
    pub sequence_group: Option<String>,
    pub zone_id: String,
    pub exposure_kind: String,
    pub observation_refs: Vec<String>,
    pub method_selection_ref: Option<String>,
    pub protocol_ref: Option<String>,
    pub block_intent: String,
    pub acceleration_mode: String,
    pub acceleration_model_ref: Option<String>,
    pub correlation_evidence_ref: Option<String>,
    pub lifetime_equivalence_claim: bool,
    pub state: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub missing_data_never_becomes_zero: bool,
    pub simulated_or_estimated_exposure_is_non_qualifying: bool,
    pub frequency_distribution_preserved: bool,
    pub unvalidated_acceleration_blocks_lifetime_claims: bool,
    pub coupon_plan_is_not_robot_lifetime_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub zone_count: usize,
    pub source_run_count: usize,
    pub observation_count: usize,
    pub test_block_count: usize,
    pub measured_observation_count: usize,
    pub manifest_profiled: bool,
    pub manifest_preregistered: bool,
    pub lifetime_equivalence_claim_count: usize,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn can_preregister(&self) -> bool {
        self.is_valid() && self.manifest_preregistered
    }

    /// True only when at least one lifetime-equivalence claim passed structural
    /// evidence admission. This is not an engineering validation of the model.
    pub fn has_admitted_lifetime_equivalence_claim(&self) -> bool {
        self.is_valid() && self.lifetime_equivalence_claim_count > 0
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: MissionExposureManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl MissionExposureManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut zones: HashMap<&str, &ZoneBinding> = HashMap::new();
        let mut source_runs: HashMap<&str, &SourceRun> = HashMap::new();
        let mut observations: HashMap<&str, &ExposureObservation> = HashMap::new();
        let mut block_ids = HashSet::new();
        let mut sequence_indices = HashSet::new();
        let mut measured_observation_count = 0usize;
        let mut lifetime_equivalence_claim_count = 0usize;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell mission-exposure schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.profile_id) {
            push(&mut findings, "missing_profile_id", "profile_id",
                "Mission exposure profile identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "inputs_pending" | "profiled" | "preregistered") {
            push(&mut findings, "invalid_manifest_state", "state",
                "Manifest state must be proposed, inputs_pending, profiled or preregistered.");
        }
        if is_blank(&self.platform.platform_id) {
            push(&mut findings, "missing_platform_identity", "platform.platform_id",
                "Platform identity must be non-empty.");
        }
        if self.platform.geometry_digest.as_deref().is_some_and(|v| !is_sha256_digest(v)) {
            push(&mut findings, "invalid_geometry_digest", "platform.geometry_digest",
                "Geometry digest must use sha256: followed by 64 hexadecimal characters.");
        }
        if self.zones.is_empty() {
            push(&mut findings, "no_zones", "zones",
                "At least one garment zone must be declared.");
        }

        for (index, zone) in self.zones.iter().enumerate() {
            let path = format!("zones[{index}]");
            if is_blank(&zone.zone_id) {
                push(&mut findings, "blank_zone_id", format!("{path}.zone_id"),
                    "Zone identity must be non-empty.");
            }
            if zones.insert(zone.zone_id.as_str(), zone).is_some() {
                push(&mut findings, "duplicate_zone_id", format!("{path}.zone_id"),
                    "Zone identities must be unique.");
            }
            if zone.function_refs.is_empty() || zone.function_refs.iter().any(|v| is_blank(v)) {
                push(&mut findings, "zone_without_function_refs", format!("{path}.function_refs"),
                    "Each zone must bind at least one non-empty functional requirement reference.");
            }
        }

        for (index, run) in self.source_runs.iter().enumerate() {
            let path = format!("source_runs[{index}]");
            if is_blank(&run.source_run_id) {
                push(&mut findings, "blank_source_run_id", format!("{path}.source_run_id"),
                    "Source run identity must be non-empty.");
            }
            if source_runs.insert(run.source_run_id.as_str(), run).is_some() {
                push(&mut findings, "duplicate_source_run_id", format!("{path}.source_run_id"),
                    "Source run IDs must be unique.");
            }
            if !matches!(run.source_kind.as_str(), "jointlab_run" | "field_mission_run" | "other_calibrated_run" | "synthetic_fixture") {
                push(&mut findings, "unknown_source_kind", format!("{path}.source_kind"),
                    "Unknown mission exposure source kind.");
            }
            if !matches!(run.data_quality_state.as_str(), "synthetic_fixture" | "pending" | "reviewed_usable" | "reviewed_with_bounds" | "rejected") {
                push(&mut findings, "invalid_source_quality_state", format!("{path}.data_quality_state"),
                    "Unknown source-data quality state.");
            }
            if run.missing_data_fraction.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
                push(&mut findings, "invalid_missing_data_fraction", format!("{path}.missing_data_fraction"),
                    "Missing-data fraction must be within 0..=1.");
            }
            if run.raw_data_digest.as_deref().is_some_and(|v| !is_sha256_digest(v)) {
                push(&mut findings, "invalid_source_raw_digest", format!("{path}.raw_data_digest"),
                    "Source raw-data digest must use sha256: followed by 64 hexadecimal characters.");
            }
            if run.data_quality_state == "reviewed_usable" || run.data_quality_state == "reviewed_with_bounds" {
                for (field, value) in [
                    ("article_revision", run.article_revision.as_deref()),
                    ("run_ref", run.run_ref.as_deref()),
                    ("start_time", run.start_time.as_deref()),
                    ("end_time", run.end_time.as_deref()),
                    ("clock_sync_ref", run.clock_sync_ref.as_deref()),
                    ("raw_data_digest", run.raw_data_digest.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "reviewed_source_missing_provenance", format!("{path}.{field}"),
                            "A reviewed physical source requires article, time, clock and raw-data identity.");
                    }
                }
                if run.missing_data_fraction.is_none() {
                    push(&mut findings, "reviewed_source_missing_data_status", format!("{path}.missing_data_fraction"),
                        "A reviewed source must explicitly report its missing-data fraction, including zero.");
                }
                if run.instrument_refs.is_empty() || run.instrument_refs.iter().any(|v| is_blank(v)) {
                    push(&mut findings, "reviewed_source_missing_instruments", format!("{path}.instrument_refs"),
                        "A reviewed source requires non-empty instrument identities.");
                }
                if run.calibration_refs.is_empty() || run.calibration_refs.iter().any(|v| is_blank(v)) {
                    push(&mut findings, "reviewed_source_missing_calibration", format!("{path}.calibration_refs"),
                        "A reviewed source requires non-empty calibration references.");
                }
                if run.saturation_observed {
                    push(&mut findings, "reviewed_source_saturated", format!("{path}.saturation_observed"),
                        "A saturated source cannot support a profiled exposure without a separately validated recovery/bounding method.");
                }
                if run.missing_data_fraction.unwrap_or(0.0) > 0.0 {
                    if run.data_quality_state != "reviewed_with_bounds" || is_blank_option(&run.gap_disposition_ref) {
                        push(&mut findings, "missing_data_without_bounded_disposition", format!("{path}.gap_disposition_ref"),
                            "Missing data must remain explicit and have a reviewed bounded-data disposition; it is never converted to zero.");
                    }
                }
                if run.article_revision.as_deref() != self.platform.article_revision.as_deref() {
                    push(&mut findings, "source_article_revision_mismatch", format!("{path}.article_revision"),
                        "Source run article revision must match the mission-profile platform revision.");
                }
            }
        }

        for (index, observation) in self.observations.iter().enumerate() {
            let path = format!("observations[{index}]");
            if is_blank(&observation.observation_id) {
                push(&mut findings, "blank_observation_id", format!("{path}.observation_id"),
                    "Exposure observation identity must be non-empty.");
            }
            if observations.insert(observation.observation_id.as_str(), observation).is_some() {
                push(&mut findings, "duplicate_observation_id", format!("{path}.observation_id"),
                    "Exposure observation IDs must be unique.");
            }
            if !zones.contains_key(observation.zone_id.as_str()) {
                push(&mut findings, "observation_unknown_zone", format!("{path}.zone_id"),
                    "Exposure observation references an unknown garment zone.");
            }
            if !is_known_exposure_kind(&observation.exposure_kind) {
                push(&mut findings, "unknown_exposure_kind", format!("{path}.exposure_kind"),
                    "Unknown textile exposure kind.");
            }
            if !origin_is_known(&observation.evidence_origin) {
                push(&mut findings, "unknown_evidence_origin", format!("{path}.evidence_origin"),
                    "Unknown exposure evidence-origin classification.");
            }
            if !metric_matches_exposure(&observation.exposure_kind, &observation.metric_id, &observation.unit) {
                push(&mut findings, "exposure_metric_unit_mismatch", format!("{path}.metric_id/unit"),
                    "Metric and unit do not match the declared exposure kind.");
            }
            if observation.sample_count == 0 {
                push(&mut findings, "exposure_missing_sample_count", format!("{path}.sample_count"),
                    "Exposure distribution requires a nonzero sample count.");
            }
            for (field, value) in [
                ("minimum", observation.minimum),
                ("p50", observation.p50),
                ("p95", observation.p95),
                ("p99", observation.p99),
                ("maximum", observation.maximum),
            ] {
                if value.is_some_and(|v| !v.is_finite()) {
                    push(&mut findings, "nonfinite_exposure_statistic", format!("{path}.{field}"),
                        "Exposure statistics must be finite.");
                }
            }
            if observation.minimum.is_some_and(|v| observation.maximum.is_some_and(|m| v > m)) {
                push(&mut findings, "exposure_min_max_reversed", path.clone(),
                    "Exposure minimum must not exceed maximum.");
            }
            for (lower_name, lower, upper_name, upper) in [
                ("minimum", observation.minimum, "p50", observation.p50),
                ("p50", observation.p50, "p95", observation.p95),
                ("p95", observation.p95, "p99", observation.p99),
                ("p99", observation.p99, "maximum", observation.maximum),
            ] {
                if lower.is_some_and(|v| upper.is_some_and(|u| v > u)) {
                    push(&mut findings, "exposure_quantiles_not_monotone", format!("{path}.{lower_name}/{upper_name}"),
                        "Available exposure quantiles must be monotonically nondecreasing.");
                }
            }
            if observation.p95.is_none() && observation.p99.is_none() {
                push(&mut findings, "exposure_missing_tail_statistic", format!("{path}.p95/p99"),
                    "A mission exposure must preserve at least a p95 or p99 tail statistic.");
            }
            if observation.evidence_origin == "measured" || observation.evidence_origin == "calibrated_derived" {
                measured_observation_count += 1;
            }
            if observation.evidence_origin == "calibrated_derived" && is_blank_option(&observation.derivation_ref) {
                push(&mut findings, "derived_exposure_missing_derivation", format!("{path}.derivation_ref"),
                    "A calibrated-derived exposure requires its transformation/derivation reference.");
            }
            if let Some(source_ref) = observation.source_run_ref.as_deref() {
                if !source_runs.contains_key(source_ref) {
                    push(&mut findings, "observation_unknown_source_run", format!("{path}.source_run_ref"),
                        "Exposure references a source run absent from this manifest.");
                }
            } else if observation.evidence_origin == "measured" || observation.evidence_origin == "calibrated_derived" {
                push(&mut findings, "physical_exposure_missing_source_run", format!("{path}.source_run_ref"),
                    "Measured or calibrated-derived exposures must bind an exact source run.");
            }
            if self.state == "profiled" || self.state == "preregistered" {
                if observation.minimum.is_none() || observation.maximum.is_none() {
                    push(&mut findings, "profiled_exposure_missing_range", format!("{path}.minimum/maximum"),
                        "Profiled exposures must preserve both minimum and maximum, in addition to their tail statistics.");
                }
                if !matches!(observation.evidence_origin.as_str(), "measured" | "calibrated_derived") {
                    push(&mut findings, "nonphysical_exposure_in_profiled_manifest", format!("{path}.evidence_origin"),
                        "Only measured or calibrated-derived channels can support a profiled mission exposure.");
                }
                if is_blank_option(&observation.method_ref) || is_blank_option(&observation.uncertainty_model_ref) {
                    push(&mut findings, "profiled_exposure_missing_method_uncertainty", format!("{path}.method_ref/uncertainty_model_ref"),
                        "Profiled exposure requires method and uncertainty-model references.");
                }
            }
            if observation.exposure_kind == "vibration" {
                validate_frequency_bins(observation, &path, &mut findings);
            } else if !observation.frequency_bins.is_empty() {
                push(&mut findings, "unexpected_frequency_bins", format!("{path}.frequency_bins"),
                    "Frequency bins are only valid for vibration exposure records.");
            }
        }

        for (index, block) in self.test_blocks.iter().enumerate() {
            let path = format!("test_blocks[{index}]");
            if is_blank(&block.block_id) {
                push(&mut findings, "blank_block_id", format!("{path}.block_id"),
                    "Test block identity must be non-empty.");
            }
            if !block_ids.insert(block.block_id.as_str()) {
                push(&mut findings, "duplicate_block_id", format!("{path}.block_id"),
                    "Test block IDs must be unique.");
            }
            if !sequence_indices.insert(block.sequence_index) {
                push(&mut findings, "duplicate_sequence_index", format!("{path}.sequence_index"),
                    "Test block sequence indices must be unique.");
            }
            if !zones.contains_key(block.zone_id.as_str()) {
                push(&mut findings, "block_unknown_zone", format!("{path}.zone_id"),
                    "Test block references an unknown garment zone.");
            }
            if !is_known_exposure_kind(&block.exposure_kind) {
                push(&mut findings, "unknown_block_exposure_kind", format!("{path}.exposure_kind"),
                    "Unknown test-block exposure kind.");
            }
            if !matches!(block.block_intent.as_str(), "mission_replay" | "tail_challenge" | "screening" | "sequence_interaction") {
                push(&mut findings, "invalid_block_intent", format!("{path}.block_intent"),
                    "Unknown test-block intent.");
            }
            if !matches!(block.acceleration_mode.as_str(), "none" | "unvalidated_hypothesis" | "validated") {
                push(&mut findings, "invalid_acceleration_mode", format!("{path}.acceleration_mode"),
                    "Acceleration mode must be none, unvalidated_hypothesis or validated.");
            }
            if block.lifetime_equivalence_claim {
                lifetime_equivalence_claim_count += 1;
                if block.acceleration_mode != "validated"
                    || is_blank_option(&block.acceleration_model_ref)
                    || is_blank_option(&block.correlation_evidence_ref)
                {
                    push(&mut findings, "unvalidated_lifetime_equivalence", format!("{path}.lifetime_equivalence_claim"),
                        "Lifetime equivalence requires a validated acceleration model and representative correlation evidence.");
                }
            }
            if block.acceleration_mode == "validated"
                && (is_blank_option(&block.acceleration_model_ref)
                    || is_blank_option(&block.correlation_evidence_ref))
            {
                push(&mut findings, "validated_acceleration_missing_evidence", format!("{path}.acceleration_model_ref/correlation_evidence_ref"),
                    "A validated acceleration profile requires both a model reference and representative correlation evidence.");
            }
            if block.acceleration_mode == "unvalidated_hypothesis" && block.lifetime_equivalence_claim {
                push(&mut findings, "hypothesis_promoted_to_lifetime_claim", path.clone(),
                    "An unvalidated acceleration hypothesis cannot support a lifetime-equivalence claim.");
            }
            if block.state != "proposed" && block.state != "preregistered" {
                push(&mut findings, "invalid_block_state", format!("{path}.state"),
                    "Test block state must be proposed or preregistered.");
            }
            if block.observation_refs.is_empty() {
                push(&mut findings, "block_without_observations", format!("{path}.observation_refs"),
                    "A test block must cite at least one exposure observation.");
            }
            for observation_ref in &block.observation_refs {
                match observations.get(observation_ref.as_str()) {
                    Some(observation) => {
                        if observation.zone_id != block.zone_id || observation.exposure_kind != block.exposure_kind {
                            push(&mut findings, "block_observation_scope_mismatch", format!("{path}.observation_refs"),
                                "Block zone and exposure kind must match each referenced observation.");
                        }
                    }
                    None => push(&mut findings, "block_unknown_observation", format!("{path}.observation_refs"),
                        "Test block references an exposure observation absent from this manifest."),
                }
            }
            if block.state == "preregistered" {
                for (field, value) in [
                    ("method_selection_ref", block.method_selection_ref.as_deref()),
                    ("protocol_ref", block.protocol_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "preregistered_block_missing_method", format!("{path}.{field}"),
                            "A preregistered block must bind an admitted test-method selection and protocol.");
                    }
                }
            }
            if block.block_intent == "sequence_interaction" && is_blank_option(&block.sequence_group) {
                push(&mut findings, "sequence_interaction_missing_group", format!("{path}.sequence_group"),
                    "A sequence-interaction block must bind an explicit interaction group.");
            }
        }

        let mut sequence_groups: HashMap<&str, HashSet<&str>> = HashMap::new();
        for block in &self.test_blocks {
            if block.block_intent == "sequence_interaction" {
                if let Some(group) = block.sequence_group.as_deref() {
                    sequence_groups.entry(group).or_default().insert(block.exposure_kind.as_str());
                }
            }
        }
        for (group, exposure_kinds) in sequence_groups {
            if exposure_kinds.len() < 2 {
                push(&mut findings, "sequence_interaction_without_distinct_exposures", "test_blocks.sequence_group",
                    format!("Sequence-interaction group {group} must include at least two distinct exposure kinds."));
            }
        }

        if self.state == "profiled" || self.state == "preregistered" {
            for (field, value) in [
                ("platform.article_revision", self.platform.article_revision.as_deref()),
                ("platform.geometry_digest", self.platform.geometry_digest.as_deref()),
                ("operating_domain.odd_profile_ref", self.operating_domain.odd_profile_ref.as_deref()),
                ("operating_domain.approval_ref", self.operating_domain.approval_ref.as_deref()),
                ("operating_domain.mission_mix_ref", self.operating_domain.mission_mix_ref.as_deref()),
                ("operating_domain.service_horizon_ref", self.operating_domain.service_horizon_ref.as_deref()),
            ] {
                if value.is_none_or(is_blank) {
                    push(&mut findings, "profiled_manifest_missing_identity", field,
                        "A profiled mission exposure requires exact article, geometry, approved ODD, mission mix and service-horizon references.");
                }
            }
            if self.source_runs.is_empty() || self.observations.is_empty() {
                push(&mut findings, "profiled_manifest_without_physical_observations", "source_runs/observations",
                    "A profiled mission exposure requires physical source runs and their exposure observations.");
            }
            if self.source_runs.iter().any(|run| !matches!(run.data_quality_state.as_str(), "reviewed_usable" | "reviewed_with_bounds")) {
                push(&mut findings, "profiled_manifest_has_unreviewed_source", "source_runs",
                    "A profiled mission exposure cannot depend on pending, synthetic, rejected or unreviewed source runs.");
            }
            if self.test_blocks.is_empty() {
                push(&mut findings, "profiled_manifest_without_test_blocks", "test_blocks",
                    "A profiled mission exposure requires a downstream zone-specific test plan.");
            }
        }
        if self.state == "preregistered" && self.test_blocks.iter().any(|block| block.state != "preregistered") {
            push(&mut findings, "preregistered_manifest_has_unfrozen_blocks", "test_blocks",
                "A preregistered manifest cannot contain proposed test blocks.");
        }

        if !self.evidence_policy.missing_data_never_becomes_zero
            || !self.evidence_policy.simulated_or_estimated_exposure_is_non_qualifying
            || !self.evidence_policy.frequency_distribution_preserved
            || !self.evidence_policy.unvalidated_acceleration_blocks_lifetime_claims
            || !self.evidence_policy.coupon_plan_is_not_robot_lifetime_qualification
        {
            push(&mut findings, "unsafe_mission_profile_policy", "evidence_policy",
                "Mission exposure unknowns, frequency tails, acceleration and coupon/lifetime boundaries must remain explicit.");
        }

        ValidationReport {
            findings,
            zone_count: self.zones.len(),
            source_run_count: self.source_runs.len(),
            observation_count: self.observations.len(),
            test_block_count: self.test_blocks.len(),
            measured_observation_count,
            manifest_profiled: self.state == "profiled",
            manifest_preregistered: self.state == "preregistered",
            lifetime_equivalence_claim_count,
        }
    }
}

fn validate_frequency_bins(
    observation: &ExposureObservation,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    if observation.frequency_bins.len() < 3 {
        push(findings, "vibration_spectrum_missing_bins", format!("{path}.frequency_bins"),
            "Vibration exposure must preserve at least three frequency bins; a scalar RMS value is not a frequency profile.");
    }
    let mut previous_upper: Option<f64> = None;
    for (index, bin) in observation.frequency_bins.iter().enumerate() {
        if !bin.lower_hz.is_finite() || !bin.upper_hz.is_finite() || !bin.level.is_finite()
            || bin.lower_hz < 0.0 || bin.upper_hz <= bin.lower_hz || bin.level < 0.0
        {
            push(findings, "invalid_frequency_bin", format!("{path}.frequency_bins[{index}]"),
                "Frequency bins require finite ordered nonnegative bounds and nonnegative level.");
        }
        if previous_upper.is_some_and(|upper| bin.lower_hz < upper) {
            push(findings, "overlapping_frequency_bins", format!("{path}.frequency_bins[{index}]"),
                "Frequency bins must be ordered and non-overlapping.");
        }
        previous_upper = Some(bin.upper_hz);
    }
}

fn metric_matches_exposure(kind: &str, metric: &str, unit: &str) -> bool {
    let expected = match kind {
        "joint_flex" => match metric {
            "joint_rom_deg" => "deg",
            "flex_cycles" => "cycles",
            _ => return false,
        },
        "vibration" => match metric {
            "vibration_acceleration_m_s2" => "m/s2",
            _ => return false,
        },
        "thermal" => match metric {
            "temperature_c" => "degC",
            _ => return false,
        },
        "humidity" => match metric {
            "relative_humidity_pct" => "%",
            _ => return false,
        },
        "abrasion_contact" => match metric {
            "abrasion_cycles" => "cycles",
            "contact_events" => "count",
            _ => return false,
        },
        "cleaning" => match metric {
            "cleaning_cycles" => "cycles",
            "cleaning_agent_concentration_pct" => "%",
            _ => return false,
        },
        "repair_service" => match metric {
            "service_intervention_count" => "count",
            "repair_duration_min" => "min",
            _ => return false,
        },
        _ => return false,
    };
    expected == unit
}

fn is_known_exposure_kind(value: &str) -> bool {
    matches!(value,
        "joint_flex" | "vibration" | "thermal" | "humidity" |
        "abrasion_contact" | "cleaning" | "repair_service")
}

fn origin_is_known(value: &str) -> bool {
    matches!(value, "measured" | "calibrated_derived" | "estimated" | "simulation" | "synthetic")
}

fn push(findings: &mut Vec<Finding>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn base_manifest() -> Value {
        json!({
            "schema_version":"0.1.0",
            "profile_id":"LIGHTSHELL-MISSION-SYNTHETIC-001",
            "state":"proposed",
            "platform":{"platform_id":"SYNTHETIC-ROBOT-001","article_revision":null,"geometry_digest":null},
            "operating_domain":{"odd_profile_ref":null,"approval_ref":null,"mission_mix_ref":null,"service_horizon_ref":null},
            "zones":[{"zone_id":"elbow_gaiter","function_refs":["REQ-ARTICULATION-01","REQ-FLEX-FATIGUE-01"]}],
            "source_runs":[{
                "source_run_id":"SYNTHETIC-SOURCE-001",
                "source_kind":"synthetic_fixture",
                "article_revision":null,
                "run_ref":null,
                "start_time":null,
                "end_time":null,
                "clock_sync_ref":null,
                "raw_data_digest":null,
                "instrument_refs":[],
                "calibration_refs":[],
                "data_quality_state":"synthetic_fixture",
                "missing_data_fraction":null,
                "saturation_observed":false,
                "gap_disposition_ref":null
            }],
            "observations":[{
                "observation_id":"OBS-JOINT-FLEX-001",
                "zone_id":"elbow_gaiter",
                "source_run_ref":"SYNTHETIC-SOURCE-001",
                "exposure_kind":"joint_flex",
                "evidence_origin":"synthetic",
                "metric_id":"joint_rom_deg",
                "unit":"deg",
                "sample_count":100,
                "minimum":0.0,
                "p50":45.0,
                "p95":80.0,
                "p99":88.0,
                "maximum":90.0,
                "frequency_bins":[],
                "method_ref":null,
                "uncertainty_model_ref":null,
                "derivation_ref":null
            }],
            "test_blocks":[{
                "block_id":"BLOCK-FLEX-001",
                "sequence_index":0,
                "sequence_group":null,
                "zone_id":"elbow_gaiter",
                "exposure_kind":"joint_flex",
                "observation_refs":["OBS-JOINT-FLEX-001"],
                "method_selection_ref":null,
                "protocol_ref":null,
                "block_intent":"screening",
                "acceleration_mode":"none",
                "acceleration_model_ref":null,
                "correlation_evidence_ref":null,
                "lifetime_equivalence_claim":false,
                "state":"proposed"
            }],
            "evidence_policy":{
                "missing_data_never_becomes_zero":true,
                "simulated_or_estimated_exposure_is_non_qualifying":true,
                "frequency_distribution_preserved":true,
                "unvalidated_acceleration_blocks_lifetime_claims":true,
                "coupon_plan_is_not_robot_lifetime_qualification":true
            }
        })
    }

    fn make_reviewed_source(manifest: &mut Value) {
        manifest["platform"]["article_revision"] = json!("ROBOT-REV-A");
        manifest["platform"]["geometry_digest"] = json!(format!("sha256:{}", "a".repeat(64)));
        manifest["operating_domain"] = json!({
            "odd_profile_ref":"ODD-REV-A",
            "approval_ref":"ODD-APPROVAL-01",
            "mission_mix_ref":"MISSION-MIX-01",
            "service_horizon_ref":"SERVICE-HORIZON-01"
        });
        let run = &mut manifest["source_runs"][0];
        run["source_kind"] = json!("jointlab_run");
        run["article_revision"] = json!("ROBOT-REV-A");
        run["run_ref"] = json!("JOINTLAB-RUN-001");
        run["start_time"] = json!("2026-11-18T08:00:00+02:00");
        run["end_time"] = json!("2026-11-18T08:30:00+02:00");
        run["clock_sync_ref"] = json!("CLOCK-SYNC-01");
        run["raw_data_digest"] = json!(format!("sha256:{}", "b".repeat(64)));
        run["instrument_refs"] = json!(["ENCODER-01"]);
        run["calibration_refs"] = json!(["CAL-ENCODER-01"]);
        run["data_quality_state"] = json!("reviewed_usable");
        run["missing_data_fraction"] = json!(0.0);
        let observation = &mut manifest["observations"][0];
        observation["evidence_origin"] = json!("measured");
        observation["method_ref"] = json!("JOINT-ROM-METHOD-01");
        observation["uncertainty_model_ref"] = json!("JOINT-UQ-01");
        let block = &mut manifest["test_blocks"][0];
        block["method_selection_ref"] = json!("METHOD-SELECTION-12759-01");
        block["protocol_ref"] = json!("FLEX-PROTOCOL-REV-A");
    }

    #[test]
    fn proposed_synthetic_mission_fixture_is_admitted_but_not_profiled() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.zone_count, 1);
        assert_eq!(report.observation_count, 1);
        assert_eq!(report.measured_observation_count, 0);
        assert!(!report.can_preregister());
    }

    #[test]
    fn profiled_manifest_requires_exact_platform_odd_and_physical_source() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "profiled_manifest_missing_identity"));
        assert!(report.findings.iter().any(|f| f.code == "profiled_manifest_without_physical_observations"));
    }

    #[test]
    fn measured_profile_with_source_provenance_is_admitted() {
        let mut manifest = base_manifest();
        make_reviewed_source(&mut manifest);
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.measured_observation_count, 1);
        assert!(report.manifest_profiled);
    }

    #[test]
    fn source_article_revision_must_match_platform_revision() {
        let mut manifest = base_manifest();
        make_reviewed_source(&mut manifest);
        manifest["source_runs"][0]["article_revision"] = json!("ROBOT-REV-B");
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "source_article_revision_mismatch"));
    }

    #[test]
    fn missing_data_is_not_silently_converted_to_zero() {
        let mut manifest = base_manifest();
        make_reviewed_source(&mut manifest);
        manifest["source_runs"][0]["missing_data_fraction"] = json!(0.15);
        manifest["source_runs"][0]["data_quality_state"] = json!("reviewed_usable");
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "missing_data_without_bounded_disposition"));
    }

    #[test]
    fn bounded_missing_data_requires_an_explicit_disposition() {
        let mut manifest = base_manifest();
        make_reviewed_source(&mut manifest);
        manifest["source_runs"][0]["missing_data_fraction"] = json!(0.15);
        manifest["source_runs"][0]["data_quality_state"] = json!("reviewed_with_bounds");
        manifest["source_runs"][0]["gap_disposition_ref"] = json!("GAP-BOUNDING-01");
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn saturated_physical_source_is_rejected() {
        let mut manifest = base_manifest();
        make_reviewed_source(&mut manifest);
        manifest["source_runs"][0]["saturation_observed"] = json!(true);
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "reviewed_source_saturated"));
    }

    #[test]
    fn vibration_requires_frequency_bins_not_only_scalar_rms() {
        let mut manifest = base_manifest();
        manifest["observations"][0]["exposure_kind"] = json!("vibration");
        manifest["observations"][0]["metric_id"] = json!("vibration_acceleration_m_s2");
        manifest["observations"][0]["unit"] = json!("m/s2");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "vibration_spectrum_missing_bins"));
    }

    #[test]
    fn overlapping_frequency_bins_are_rejected() {
        let mut manifest = base_manifest();
        manifest["observations"][0]["exposure_kind"] = json!("vibration");
        manifest["observations"][0]["metric_id"] = json!("vibration_acceleration_m_s2");
        manifest["observations"][0]["unit"] = json!("m/s2");
        manifest["observations"][0]["frequency_bins"] = json!([
            {"lower_hz":10.0,"upper_hz":40.0,"level":0.3},
            {"lower_hz":30.0,"upper_hz":70.0,"level":0.5},
            {"lower_hz":70.0,"upper_hz":100.0,"level":0.2}
        ]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "overlapping_frequency_bins"));
    }

    #[test]
    fn nonmonotone_exposure_quantiles_are_rejected() {
        let mut manifest = base_manifest();
        manifest["observations"][0]["p95"] = json!(70.0);
        manifest["observations"][0]["p99"] = json!(60.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "exposure_quantiles_not_monotone"));
    }

    #[test]
    fn unvalidated_acceleration_cannot_support_lifetime_equivalence() {
        let mut manifest = base_manifest();
        manifest["test_blocks"][0]["acceleration_mode"] = json!("unvalidated_hypothesis");
        manifest["test_blocks"][0]["lifetime_equivalence_claim"] = json!(true);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unvalidated_lifetime_equivalence"));
    }

    #[test]
    fn sequence_interaction_requires_distinct_exposure_kinds() {
        let mut manifest = base_manifest();
        manifest["test_blocks"][0]["block_intent"] = json!("sequence_interaction");
        manifest["test_blocks"][0]["sequence_group"] = json!("WASH-THEN-FLEX");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "sequence_interaction_without_distinct_exposures"));
    }

    #[test]
    fn validated_acceleration_requires_model_and_correlation_evidence() {
        let mut manifest = base_manifest();
        manifest["test_blocks"][0]["acceleration_mode"] = json!("validated");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "validated_acceleration_missing_evidence"));
    }

    #[test]
    fn sequence_interaction_requires_named_sequence_group() {
        let mut manifest = base_manifest();
        manifest["test_blocks"][0]["block_intent"] = json!("sequence_interaction");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "sequence_interaction_missing_group"));
    }

    #[test]
    fn simulated_observations_cannot_be_promoted_into_profiled_mission() {
        let mut manifest = base_manifest();
        make_reviewed_source(&mut manifest);
        manifest["observations"][0]["evidence_origin"] = json!("simulation");
        manifest["state"] = json!("profiled");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "nonphysical_exposure_in_profiled_manifest"));
    }

    #[test]
    fn duplicate_sequence_indices_are_rejected() {
        let mut manifest = base_manifest();
        let duplicate = manifest["test_blocks"][0].clone();
        manifest["test_blocks"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_sequence_index"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["invented_lifetime_hours"] = json!(100000);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}

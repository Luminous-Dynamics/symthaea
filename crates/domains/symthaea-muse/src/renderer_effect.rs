// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual qualification of renderer controls against independently
//! observed output. An accepted API request is never treated as proof of effect.

use crate::evidence_digest::{canonical_json_sha256, decode_hex_32};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RENDERER_EFFECT_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererControlDimensionV1 {
    TempoBpm,
    ChromaDistributionDistance,
    NoteDensity,
    Brightness,
    Dynamics,
    OnsetRate,
    Register,
    StereoWidth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererObservationSourceV1 {
    /// Independently captured rendered waveform. Required for acoustic claims.
    RenderedAudioLoopback,
    /// Independently tapped MIDI output. Does not establish audible onset or timbre.
    IndependentMidiTap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererControlConditionKindV1 {
    Baseline,
    Sham,
    Active,
    Reversal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererRequestStatusV1 {
    NotSent,
    Rejected,
    Accepted,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererObservationClosureStatusV1 {
    StructurallyComplete,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererExpectedDirectionV1 {
    Increase,
    Decrease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererEffectOutcomeV1 {
    /// Active, sham, reversal and collateral tests meet the preregistered rules.
    StructurallyQualified,
    /// Complete observations exist, but one or more causal-effect rules failed.
    NotQualified,
    /// At least one required observation window is incomplete or unknown.
    Unobserved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererObservationProvenanceV1 {
    pub source: RendererObservationSourceV1,
    pub closure_status: RendererObservationClosureStatusV1,
    pub capture_id: String,
    pub capture_payload_sha256: String,
    pub closure_receipt_sha256: String,
    pub event_manifest_sha256: String,
    pub event_manifest_capture_payload_sha256: String,
    pub capture_observer_identity_sha256: String,
    pub capture_observer_build_sha256: String,
    pub capture_observer_configuration_sha256: String,
    pub capture_observer_receipt_sha256: String,
    pub event_observer_identity_sha256: String,
    pub event_observer_build_sha256: String,
    pub event_observer_configuration_sha256: String,
    pub event_observer_qualification_receipt_sha256: String,
    pub event_observer_receipt_sha256: String,
    pub producer_identity_sha256: String,
    pub clock_domain_id: String,
    pub clock_alignment_receipt_sha256: String,
    pub captured_duration_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererCollateralMetricV1 {
    pub metric_id: String,
    pub value: f64,
    pub uncertainty_abs: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererControlConditionV1 {
    pub kind: RendererControlConditionKindV1,
    /// Randomized position within this trial; all four positions must be unique.
    pub randomized_order_slot: u8,
    pub request_status: RendererRequestStatusV1,
    pub requested_value: Option<f64>,
    /// Adapter-reported effective setting, not an observation of the sound.
    pub effective_value: Option<f64>,
    pub renderer_identity_sha256: String,
    pub adapter_identity_sha256: String,
    pub non_target_configuration_sha256: String,
    pub capability_manifest_sha256: String,
    /// Measured from the declared independent observation source.
    pub observed_metric_value: f64,
    pub observed_metric_uncertainty_abs: f64,
    pub observed_metric_unit: String,
    pub observation: RendererObservationProvenanceV1,
    pub collateral_metrics: Vec<RendererCollateralMetricV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererControlEffectTrialV1 {
    pub schema_version: u32,
    pub trial_id: String,
    pub control_id: String,
    pub dimension: RendererControlDimensionV1,
    pub expected_direction: RendererExpectedDirectionV1,
    pub metric_unit: String,
    pub observation_source: RendererObservationSourceV1,
    pub required_observation_duration_ns: u64,
    pub baseline_setting_value: f64,
    pub target_setting_value: f64,
    pub setting_tolerance_abs: f64,
    pub minimum_effect_abs: f64,
    pub maximum_sham_effect_abs: f64,
    pub maximum_reversal_residual_abs: f64,
    pub maximum_collateral_drift_abs: f64,
    pub renderer_identity_sha256: String,
    pub adapter_identity_sha256: String,
    pub non_target_configuration_sha256: String,
    pub capability_manifest_sha256: String,
    /// Commits to the randomized assignment; authenticity is a separate layer.
    pub randomization_receipt_sha256: String,
    pub conditions: Vec<RendererControlConditionV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RendererControlEffectAssessmentV1 {
    pub schema_version: u32,
    pub trial_id: String,
    pub trial_sha256: String,
    pub outcome: RendererEffectOutcomeV1,
    pub measured_active_effect: Option<f64>,
    pub conservative_active_effect_abs_lower_bound: Option<f64>,
    pub conservative_sham_effect_abs_upper_bound: Option<f64>,
    pub conservative_reversal_residual_abs_upper_bound: Option<f64>,
    pub maximum_collateral_drift_abs_upper_bound: Option<f64>,
    pub findings: Vec<String>,
    /// This is a structural assessment of linked observations, not signer or
    /// detector authentication and not a human-perception result.
    pub claim_ceiling: String,
}

fn require_text(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

fn require_sha256(value: &str, field: &str) -> Result<(), String> {
    if decode_hex_32(value).is_none() || value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(format!(
            "{field} must be a lowercase 64-character SHA-256 digest"
        ));
    }
    Ok(())
}

fn require_finite(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{field} must be finite"));
    }
    Ok(())
}

fn require_finite_non_negative(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{field} must be finite and non-negative"));
    }
    Ok(())
}

fn close_enough(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

fn condition_by_kind(
    conditions: &[RendererControlConditionV1],
) -> BTreeMap<RendererControlConditionKindV1, &RendererControlConditionV1> {
    conditions
        .iter()
        .map(|condition| (condition.kind, condition))
        .collect()
}

fn validate_trial_shape(trial: &RendererControlEffectTrialV1) -> Result<(), String> {
    if trial.schema_version != RENDERER_EFFECT_SCHEMA_VERSION_V1 {
        return Err(format!(
            "unsupported renderer effect schema version {}",
            trial.schema_version
        ));
    }
    require_text(&trial.trial_id, "trial_id")?;
    require_text(&trial.control_id, "control_id")?;
    require_text(&trial.metric_unit, "metric_unit")?;
    if trial.required_observation_duration_ns == 0 {
        return Err("required observation duration must be non-zero".to_owned());
    }
    require_finite(trial.baseline_setting_value, "baseline setting")?;
    require_finite(trial.target_setting_value, "target setting")?;
    for (value, name) in [
        (trial.setting_tolerance_abs, "setting tolerance"),
        (trial.minimum_effect_abs, "minimum effect"),
        (trial.maximum_sham_effect_abs, "maximum sham effect"),
        (
            trial.maximum_reversal_residual_abs,
            "maximum reversal residual",
        ),
        (
            trial.maximum_collateral_drift_abs,
            "maximum collateral drift",
        ),
    ] {
        require_finite_non_negative(value, name)?;
    }
    if close_enough(
        trial.baseline_setting_value,
        trial.target_setting_value,
        trial.setting_tolerance_abs,
    ) {
        return Err(
            "active target setting must differ from baseline beyond setting tolerance".to_owned(),
        );
    }
    for (digest, name) in [
        (&trial.renderer_identity_sha256, "renderer identity digest"),
        (&trial.adapter_identity_sha256, "adapter identity digest"),
        (
            &trial.non_target_configuration_sha256,
            "non-target configuration digest",
        ),
        (
            &trial.capability_manifest_sha256,
            "capability manifest digest",
        ),
        (
            &trial.randomization_receipt_sha256,
            "randomization receipt digest",
        ),
    ] {
        require_sha256(digest, name)?;
    }
    if matches!(
        trial.dimension,
        RendererControlDimensionV1::Brightness
            | RendererControlDimensionV1::Dynamics
            | RendererControlDimensionV1::StereoWidth
    ) && trial.observation_source != RendererObservationSourceV1::RenderedAudioLoopback
    {
        return Err(
            "acoustic control dimensions require rendered-audio loopback observation".to_owned(),
        );
    }
    if trial.conditions.len() != 4 {
        return Err(
            "trial must contain exactly baseline, sham, active and reversal conditions".to_owned(),
        );
    }

    let mut kinds = BTreeSet::new();
    let mut slots = BTreeSet::new();
    let mut capture_ids = BTreeSet::new();
    for condition in &trial.conditions {
        if !kinds.insert(condition.kind) {
            return Err(format!("duplicate condition kind: {:?}", condition.kind));
        }
        if condition.randomized_order_slot > 3 || !slots.insert(condition.randomized_order_slot) {
            return Err("randomized order slots must be a permutation of 0, 1, 2, 3".to_owned());
        }
        for (digest, name) in [
            (
                &condition.renderer_identity_sha256,
                "condition renderer identity",
            ),
            (
                &condition.adapter_identity_sha256,
                "condition adapter identity",
            ),
            (
                &condition.non_target_configuration_sha256,
                "condition non-target configuration",
            ),
            (
                &condition.capability_manifest_sha256,
                "condition capability manifest",
            ),
        ] {
            require_sha256(digest, name)?;
        }
        if condition.renderer_identity_sha256 != trial.renderer_identity_sha256
            || condition.adapter_identity_sha256 != trial.adapter_identity_sha256
            || condition.non_target_configuration_sha256 != trial.non_target_configuration_sha256
            || condition.capability_manifest_sha256 != trial.capability_manifest_sha256
        {
            return Err(format!(
                "condition {:?} does not use the trial's pinned renderer, adapter, configuration and capability manifest",
                condition.kind
            ));
        }
        require_finite(condition.observed_metric_value, "observed metric value")?;
        if let Some(value) = condition.requested_value {
            require_finite(value, "requested control value")?;
        }
        if let Some(value) = condition.effective_value {
            require_finite(value, "effective control value")?;
        }
        require_finite_non_negative(
            condition.observed_metric_uncertainty_abs,
            "observed metric uncertainty",
        )?;
        require_text(&condition.observed_metric_unit, "observed metric unit")?;
        if condition.observed_metric_unit != trial.metric_unit {
            return Err(format!(
                "condition {:?} metric unit differs from trial",
                condition.kind
            ));
        }
        require_text(&condition.observation.capture_id, "capture_id")?;
        if !capture_ids.insert(condition.observation.capture_id.as_str()) {
            return Err("each randomized condition must use a distinct capture ID".to_owned());
        }
        require_text(&condition.observation.clock_domain_id, "clock_domain_id")?;
        for (digest, name) in [
            (
                &condition.observation.capture_payload_sha256,
                "capture payload digest",
            ),
            (
                &condition.observation.closure_receipt_sha256,
                "closure receipt digest",
            ),
            (
                &condition.observation.event_manifest_sha256,
                "event manifest digest",
            ),
            (
                &condition.observation.event_manifest_capture_payload_sha256,
                "event manifest capture payload digest",
            ),
            (
                &condition.observation.capture_observer_identity_sha256,
                "capture observer identity digest",
            ),
            (
                &condition.observation.capture_observer_build_sha256,
                "capture observer build digest",
            ),
            (
                &condition.observation.capture_observer_configuration_sha256,
                "capture observer configuration digest",
            ),
            (
                &condition.observation.capture_observer_receipt_sha256,
                "capture observer receipt digest",
            ),
            (
                &condition.observation.event_observer_identity_sha256,
                "event observer identity digest",
            ),
            (
                &condition.observation.event_observer_build_sha256,
                "event observer build digest",
            ),
            (
                &condition.observation.event_observer_configuration_sha256,
                "event observer configuration digest",
            ),
            (
                &condition
                    .observation
                    .event_observer_qualification_receipt_sha256,
                "event observer qualification receipt digest",
            ),
            (
                &condition.observation.event_observer_receipt_sha256,
                "event observer receipt digest",
            ),
            (
                &condition.observation.producer_identity_sha256,
                "producer identity digest",
            ),
            (
                &condition.observation.clock_alignment_receipt_sha256,
                "clock alignment receipt digest",
            ),
        ] {
            require_sha256(digest, name)?;
        }
        if condition.observation.event_manifest_capture_payload_sha256
            != condition.observation.capture_payload_sha256
        {
            return Err(format!(
                "condition {:?} event manifest is not bound to the captured audio payload",
                condition.kind
            ));
        }
        if condition.observation.capture_observer_identity_sha256
            == condition.observation.producer_identity_sha256
            || condition.observation.event_observer_identity_sha256
                == condition.observation.producer_identity_sha256
        {
            return Err(format!(
                "condition {:?} uses the producer as its independent capture or event observer",
                condition.kind
            ));
        }
        if condition.observation.source != trial.observation_source {
            return Err(format!(
                "condition {:?} uses a different observation source from the trial",
                condition.kind
            ));
        }
        if condition.observation.captured_duration_ns == 0 {
            return Err(format!(
                "condition {:?} capture duration must be non-zero",
                condition.kind
            ));
        }
        for collateral in &condition.collateral_metrics {
            require_text(&collateral.metric_id, "collateral metric_id")?;
            require_text(&collateral.unit, "collateral metric unit")?;
            if !collateral.value.is_finite() {
                return Err(format!(
                    "collateral metric {} must be finite",
                    collateral.metric_id
                ));
            }
            require_finite_non_negative(
                collateral.uncertainty_abs,
                "collateral metric uncertainty",
            )?;
        }
        let collateral_ids: BTreeSet<_> = condition
            .collateral_metrics
            .iter()
            .map(|metric| metric.metric_id.as_str())
            .collect();
        if collateral_ids.len() != condition.collateral_metrics.len() {
            return Err(format!(
                "condition {:?} has duplicate collateral metric IDs",
                condition.kind
            ));
        }
    }
    if kinds.len() != 4 || slots.len() != 4 {
        return Err(
            "trial must include all four unique conditions and randomized order slots".to_owned(),
        );
    }
    let slot_for = |kind| {
        trial
            .conditions
            .iter()
            .find(|condition| condition.kind == kind)
            .map(|condition| condition.randomized_order_slot)
            .expect("the four unique condition kinds were checked above")
    };
    let baseline_slot = slot_for(RendererControlConditionKindV1::Baseline);
    let sham_slot = slot_for(RendererControlConditionKindV1::Sham);
    let active_slot = slot_for(RendererControlConditionKindV1::Active);
    let reversal_slot = slot_for(RendererControlConditionKindV1::Reversal);
    if baseline_slot >= active_slot || sham_slot >= active_slot || active_slot >= reversal_slot {
        return Err(
            "randomized order must place baseline and sham before active, and reversal after active"
                .to_owned(),
        );
    }
    let baseline_condition = trial
        .conditions
        .iter()
        .find(|condition| condition.kind == RendererControlConditionKindV1::Baseline)
        .expect("four unique condition kinds include baseline");
    let baseline_collateral_units: BTreeMap<_, _> = baseline_condition
        .collateral_metrics
        .iter()
        .map(|metric| (metric.metric_id.as_str(), metric.unit.as_str()))
        .collect();
    for condition in &trial.conditions {
        let condition_collateral_units: BTreeMap<_, _> = condition
            .collateral_metrics
            .iter()
            .map(|metric| (metric.metric_id.as_str(), metric.unit.as_str()))
            .collect();
        if baseline_collateral_units != condition_collateral_units {
            return Err(format!(
                "condition {:?} collateral metric set or units differ from baseline",
                condition.kind
            ));
        }
    }
    let mut ordered_conditions: Vec<_> = trial.conditions.iter().collect();
    ordered_conditions.sort_by_key(|condition| condition.randomized_order_slot);
    let randomized_assignment: Vec<_> = ordered_conditions
        .iter()
        .map(|condition| condition.kind)
        .collect();
    let expected_randomization_receipt =
        canonical_json_sha256(&randomized_assignment).map_err(|error| error.to_string())?;
    if expected_randomization_receipt != trial.randomization_receipt_sha256 {
        return Err(
            "randomization receipt digest does not match the recorded condition order".to_owned(),
        );
    }
    Ok(())
}

fn validate_condition_request_contract(
    trial: &RendererControlEffectTrialV1,
    conditions: &BTreeMap<RendererControlConditionKindV1, &RendererControlConditionV1>,
) -> Vec<String> {
    let mut findings = Vec::new();
    let expected = [
        (
            RendererControlConditionKindV1::Baseline,
            RendererRequestStatusV1::NotSent,
            None,
            trial.baseline_setting_value,
        ),
        (
            RendererControlConditionKindV1::Sham,
            RendererRequestStatusV1::Accepted,
            Some(trial.baseline_setting_value),
            trial.baseline_setting_value,
        ),
        (
            RendererControlConditionKindV1::Active,
            RendererRequestStatusV1::Accepted,
            Some(trial.target_setting_value),
            trial.target_setting_value,
        ),
        (
            RendererControlConditionKindV1::Reversal,
            RendererRequestStatusV1::Accepted,
            Some(trial.baseline_setting_value),
            trial.baseline_setting_value,
        ),
    ];
    for (kind, request_status, requested_value, effective_value) in expected {
        let condition = conditions[&kind];
        if condition.request_status != request_status {
            findings.push(format!(
                "{kind:?} request status is {:?}, expected {:?}",
                condition.request_status, request_status
            ));
        }
        match (condition.requested_value, requested_value) {
            (None, None) => {}
            (Some(actual), Some(expected))
                if close_enough(actual, expected, trial.setting_tolerance_abs) => {}
            _ => findings.push(format!(
                "{kind:?} requested setting does not match the preregistered condition"
            )),
        }
        match condition.effective_value {
            Some(actual) if close_enough(actual, effective_value, trial.setting_tolerance_abs) => {}
            _ => findings.push(format!(
                "{kind:?} effective setting does not match the preregistered setting"
            )),
        }
    }
    findings
}

fn validate_observation_completeness(
    trial: &RendererControlEffectTrialV1,
    conditions: &BTreeMap<RendererControlConditionKindV1, &RendererControlConditionV1>,
) -> Vec<String> {
    let mut findings = Vec::new();
    let mut capture_observer_identities = BTreeSet::new();
    let mut capture_observer_builds = BTreeSet::new();
    let mut capture_observer_configurations = BTreeSet::new();
    let mut event_observer_identities = BTreeSet::new();
    let mut event_observer_builds = BTreeSet::new();
    let mut event_observer_configurations = BTreeSet::new();
    let mut event_observer_qualifications = BTreeSet::new();
    let mut producer_identities = BTreeSet::new();
    let mut clock_domains = BTreeSet::new();
    for (kind, condition) in conditions {
        if condition.observation.closure_status
            != RendererObservationClosureStatusV1::StructurallyComplete
        {
            findings.push(format!(
                "{kind:?} observation window is not structurally complete"
            ));
        }
        if condition.observation.captured_duration_ns < trial.required_observation_duration_ns {
            findings.push(format!(
                "{kind:?} observation is shorter than the required target window"
            ));
        }
        capture_observer_identities.insert(
            condition
                .observation
                .capture_observer_identity_sha256
                .as_str(),
        );
        capture_observer_builds
            .insert(condition.observation.capture_observer_build_sha256.as_str());
        capture_observer_configurations.insert(
            condition
                .observation
                .capture_observer_configuration_sha256
                .as_str(),
        );
        event_observer_identities.insert(
            condition
                .observation
                .event_observer_identity_sha256
                .as_str(),
        );
        event_observer_builds.insert(condition.observation.event_observer_build_sha256.as_str());
        event_observer_configurations.insert(
            condition
                .observation
                .event_observer_configuration_sha256
                .as_str(),
        );
        event_observer_qualifications.insert(
            condition
                .observation
                .event_observer_qualification_receipt_sha256
                .as_str(),
        );
        producer_identities.insert(condition.observation.producer_identity_sha256.as_str());
        clock_domains.insert(condition.observation.clock_domain_id.as_str());
    }
    if capture_observer_identities.len() != 1 {
        findings.push("capture observer identity changes between randomized conditions".to_owned());
    }
    if capture_observer_builds.len() != 1 {
        findings.push("capture observer build changes between randomized conditions".to_owned());
    }
    if capture_observer_configurations.len() != 1 {
        findings.push(
            "capture observer configuration changes between randomized conditions".to_owned(),
        );
    }
    if event_observer_identities.len() != 1 {
        findings.push("event observer identity changes between randomized conditions".to_owned());
    }
    if event_observer_builds.len() != 1 {
        findings.push("event observer build changes between randomized conditions".to_owned());
    }
    if event_observer_configurations.len() != 1 {
        findings
            .push("event observer configuration changes between randomized conditions".to_owned());
    }
    if event_observer_qualifications.len() != 1 {
        findings
            .push("event observer qualification changes between randomized conditions".to_owned());
    }
    if producer_identities.len() != 1 {
        findings.push("producer identity changes between randomized conditions".to_owned());
    }
    if clock_domains.len() != 1 {
        findings.push("observation clock domain changes between randomized conditions".to_owned());
    }
    findings
}

fn collateral_drift_upper_bound(
    baseline: &RendererControlConditionV1,
    active: &RendererControlConditionV1,
) -> Result<f64, String> {
    let baseline_map: BTreeMap<_, _> = baseline
        .collateral_metrics
        .iter()
        .map(|metric| (metric.metric_id.as_str(), metric))
        .collect();
    let active_map: BTreeMap<_, _> = active
        .collateral_metrics
        .iter()
        .map(|metric| (metric.metric_id.as_str(), metric))
        .collect();
    if baseline_map.keys().collect::<Vec<_>>() != active_map.keys().collect::<Vec<_>>() {
        return Err("baseline and active collateral metric sets differ".to_owned());
    }
    let mut maximum: f64 = 0.0;
    for (metric_id, baseline_metric) in &baseline_map {
        let active_metric = active_map
            .get(*metric_id)
            .copied()
            .ok_or_else(|| format!("active collateral metric {metric_id} is missing"))?;
        if baseline_metric.unit != active_metric.unit {
            return Err(format!(
                "collateral metric {metric_id} changes unit between conditions"
            ));
        }
        let upper = (active_metric.value - baseline_metric.value).abs()
            + active_metric.uncertainty_abs
            + baseline_metric.uncertainty_abs;
        maximum = maximum.max(upper);
    }
    Ok(maximum)
}

/// Assess a renderer control against independent output observations.
///
/// A structurally qualified result means only that the supplied complete-window
/// receipts, request/effective settings, observed target metric, sham,
/// reversal and collateral-drift tests satisfy this trial's declared limits.
/// It does not authenticate those receipts or establish human-perceived value.
pub fn assess_renderer_control_effect_v1(
    trial: &RendererControlEffectTrialV1,
) -> Result<RendererControlEffectAssessmentV1, String> {
    validate_trial_shape(trial)?;
    let mut normalized_trial = trial.clone();
    normalized_trial
        .conditions
        .sort_by_key(|condition| condition.kind);
    let trial_sha256 =
        canonical_json_sha256(&normalized_trial).map_err(|error| error.to_string())?;
    let conditions = condition_by_kind(&trial.conditions);
    let mut findings = validate_condition_request_contract(trial, &conditions);
    let observation_findings = validate_observation_completeness(trial, &conditions);
    if !observation_findings.is_empty() {
        findings.extend(observation_findings);
        return Ok(RendererControlEffectAssessmentV1 {
            schema_version: RENDERER_EFFECT_SCHEMA_VERSION_V1,
            trial_id: trial.trial_id.clone(),
            trial_sha256,
            outcome: RendererEffectOutcomeV1::Unobserved,
            measured_active_effect: None,
            conservative_active_effect_abs_lower_bound: None,
            conservative_sham_effect_abs_upper_bound: None,
            conservative_reversal_residual_abs_upper_bound: None,
            maximum_collateral_drift_abs_upper_bound: None,
            findings,
            claim_ceiling: "structural capture/manifest eligibility only; no signer, detector, acoustic truth or human-preference authentication".to_owned(),
        });
    }

    let baseline = conditions[&RendererControlConditionKindV1::Baseline];
    let sham = conditions[&RendererControlConditionKindV1::Sham];
    let active = conditions[&RendererControlConditionKindV1::Active];
    let reversal = conditions[&RendererControlConditionKindV1::Reversal];

    let active_effect = active.observed_metric_value - baseline.observed_metric_value;
    let active_effect_abs_lower = active_effect.abs()
        - active.observed_metric_uncertainty_abs
        - baseline.observed_metric_uncertainty_abs;
    let sham_effect_abs_upper = (sham.observed_metric_value - baseline.observed_metric_value).abs()
        + sham.observed_metric_uncertainty_abs
        + baseline.observed_metric_uncertainty_abs;
    let reversal_residual_abs_upper =
        (reversal.observed_metric_value - baseline.observed_metric_value).abs()
            + reversal.observed_metric_uncertainty_abs
            + baseline.observed_metric_uncertainty_abs;
    let collateral_upper = collateral_drift_upper_bound(baseline, active)?;
    for (value, field) in [
        (active_effect, "active effect"),
        (active_effect_abs_lower, "active effect lower bound"),
        (sham_effect_abs_upper, "sham effect upper bound"),
        (reversal_residual_abs_upper, "reversal residual upper bound"),
        (collateral_upper, "collateral drift upper bound"),
    ] {
        if !value.is_finite() {
            return Err(format!("{field} overflowed finite floating-point range"));
        }
    }

    let correct_direction = match trial.expected_direction {
        RendererExpectedDirectionV1::Increase => active_effect > 0.0,
        RendererExpectedDirectionV1::Decrease => active_effect < 0.0,
    };
    if !correct_direction {
        findings.push(
            "active intervention did not change the observed metric in the expected direction"
                .to_owned(),
        );
    }
    if active_effect_abs_lower < trial.minimum_effect_abs {
        findings.push(
            "conservative active effect lower bound is below the preregistered minimum".to_owned(),
        );
    }
    if sham_effect_abs_upper > trial.maximum_sham_effect_abs {
        findings.push(
            "sham condition changed the observed metric beyond its maximum allowance".to_owned(),
        );
    }
    if reversal_residual_abs_upper > trial.maximum_reversal_residual_abs {
        findings.push(
            "reversal condition did not return the observed metric close enough to baseline"
                .to_owned(),
        );
    }
    if collateral_upper > trial.maximum_collateral_drift_abs {
        findings.push(
            "active intervention caused collateral drift beyond its declared budget".to_owned(),
        );
    }

    let outcome = if findings.is_empty() {
        RendererEffectOutcomeV1::StructurallyQualified
    } else {
        RendererEffectOutcomeV1::NotQualified
    };
    Ok(RendererControlEffectAssessmentV1 {
        schema_version: RENDERER_EFFECT_SCHEMA_VERSION_V1,
        trial_id: trial.trial_id.clone(),
        trial_sha256,
        outcome,
        measured_active_effect: Some(active_effect),
        conservative_active_effect_abs_lower_bound: Some(active_effect_abs_lower),
        conservative_sham_effect_abs_upper_bound: Some(sham_effect_abs_upper),
        conservative_reversal_residual_abs_upper_bound: Some(reversal_residual_abs_upper),
        maximum_collateral_drift_abs_upper_bound: Some(collateral_upper),
        findings,
        claim_ceiling: "structural capture/manifest eligibility and observed-metric counterfactual checks only; no signer, detector, acoustic truth or human-preference authentication".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RENDERER: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const ADAPTER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONFIG: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CAPABILITY: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const OBSERVER: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const PRODUCER: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const CLOCK: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const RECEIPT: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const MANIFEST: &str = "3333333333333333333333333333333333333333333333333333333333333333";

    fn observation(capture_id: &str) -> RendererObservationProvenanceV1 {
        RendererObservationProvenanceV1 {
            source: RendererObservationSourceV1::RenderedAudioLoopback,
            closure_status: RendererObservationClosureStatusV1::StructurallyComplete,
            capture_id: capture_id.to_owned(),
            capture_payload_sha256: RECEIPT.to_owned(),
            closure_receipt_sha256: RECEIPT.to_owned(),
            event_manifest_sha256: MANIFEST.to_owned(),
            event_manifest_capture_payload_sha256: RECEIPT.to_owned(),
            capture_observer_identity_sha256:
                "8888888888888888888888888888888888888888888888888888888888888888".to_owned(),
            capture_observer_build_sha256:
                "9999999999999999999999999999999999999999999999999999999999999999".to_owned(),
            capture_observer_configuration_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            capture_observer_receipt_sha256:
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            event_observer_identity_sha256: OBSERVER.to_owned(),
            event_observer_build_sha256:
                "4444444444444444444444444444444444444444444444444444444444444444".to_owned(),
            event_observer_configuration_sha256:
                "5555555555555555555555555555555555555555555555555555555555555555".to_owned(),
            event_observer_qualification_receipt_sha256:
                "6666666666666666666666666666666666666666666666666666666666666666".to_owned(),
            event_observer_receipt_sha256:
                "7777777777777777777777777777777777777777777777777777777777777777".to_owned(),
            producer_identity_sha256: PRODUCER.to_owned(),
            clock_domain_id: "audio-loopback-clock-v1".to_owned(),
            clock_alignment_receipt_sha256: CLOCK.to_owned(),
            captured_duration_ns: 4_000_000_000,
        }
    }

    fn condition(
        kind: RendererControlConditionKindV1,
        order: u8,
        metric: f64,
        uncertainty: f64,
        request_status: RendererRequestStatusV1,
        requested: Option<f64>,
        effective: f64,
    ) -> RendererControlConditionV1 {
        RendererControlConditionV1 {
            kind,
            randomized_order_slot: order,
            request_status,
            requested_value: requested,
            effective_value: Some(effective),
            renderer_identity_sha256: RENDERER.to_owned(),
            adapter_identity_sha256: ADAPTER.to_owned(),
            non_target_configuration_sha256: CONFIG.to_owned(),
            capability_manifest_sha256: CAPABILITY.to_owned(),
            observed_metric_value: metric,
            observed_metric_uncertainty_abs: uncertainty,
            observed_metric_unit: "bpm".to_owned(),
            observation: observation(&format!("{kind:?}").to_lowercase()),
            collateral_metrics: vec![
                RendererCollateralMetricV1 {
                    metric_id: "pitch_class_entropy".to_owned(),
                    value: 0.6,
                    uncertainty_abs: 0.01,
                    unit: "bits".to_owned(),
                },
                RendererCollateralMetricV1 {
                    metric_id: "stereo_width".to_owned(),
                    value: 0.4,
                    uncertainty_abs: 0.01,
                    unit: "ratio".to_owned(),
                },
            ],
        }
    }

    fn valid_trial() -> RendererControlEffectTrialV1 {
        RendererControlEffectTrialV1 {
            schema_version: RENDERER_EFFECT_SCHEMA_VERSION_V1,
            trial_id: "tempo-control-trial-001".to_owned(),
            control_id: "tempo_bpm".to_owned(),
            dimension: RendererControlDimensionV1::TempoBpm,
            expected_direction: RendererExpectedDirectionV1::Increase,
            metric_unit: "bpm".to_owned(),
            observation_source: RendererObservationSourceV1::RenderedAudioLoopback,
            required_observation_duration_ns: 3_000_000_000,
            baseline_setting_value: 100.0,
            target_setting_value: 120.0,
            setting_tolerance_abs: 0.01,
            minimum_effect_abs: 10.0,
            maximum_sham_effect_abs: 2.0,
            maximum_reversal_residual_abs: 3.0,
            maximum_collateral_drift_abs: 0.05,
            renderer_identity_sha256: RENDERER.to_owned(),
            adapter_identity_sha256: ADAPTER.to_owned(),
            non_target_configuration_sha256: CONFIG.to_owned(),
            capability_manifest_sha256: CAPABILITY.to_owned(),
            randomization_receipt_sha256: canonical_json_sha256(&vec![
                RendererControlConditionKindV1::Sham,
                RendererControlConditionKindV1::Baseline,
                RendererControlConditionKindV1::Active,
                RendererControlConditionKindV1::Reversal,
            ])
            .unwrap(),
            conditions: vec![
                condition(
                    RendererControlConditionKindV1::Baseline,
                    1,
                    100.0,
                    0.2,
                    RendererRequestStatusV1::NotSent,
                    None,
                    100.0,
                ),
                condition(
                    RendererControlConditionKindV1::Sham,
                    0,
                    100.5,
                    0.2,
                    RendererRequestStatusV1::Accepted,
                    Some(100.0),
                    100.0,
                ),
                condition(
                    RendererControlConditionKindV1::Active,
                    2,
                    119.0,
                    0.2,
                    RendererRequestStatusV1::Accepted,
                    Some(120.0),
                    120.0,
                ),
                condition(
                    RendererControlConditionKindV1::Reversal,
                    3,
                    101.0,
                    0.2,
                    RendererRequestStatusV1::Accepted,
                    Some(100.0),
                    100.0,
                ),
            ],
        }
    }

    #[test]
    fn accepted_control_without_observed_effect_does_not_qualify() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .observed_metric_value = 100.5;
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::NotQualified);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.contains("minimum"))
        );
    }

    #[test]
    fn active_effect_with_null_sham_reversal_and_collateral_drift_qualifies() {
        let result = assess_renderer_control_effect_v1(&valid_trial()).unwrap();
        assert_eq!(
            result.outcome,
            RendererEffectOutcomeV1::StructurallyQualified
        );
        assert!(result.findings.is_empty());
        assert!((result.measured_active_effect.unwrap() - 19.0).abs() < 1e-9);
        assert_eq!(result.trial_sha256.len(), 64);
    }

    #[test]
    fn sham_effect_is_not_mistaken_for_active_control_effect() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Sham)
            .unwrap()
            .observed_metric_value = 115.0;
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::NotQualified);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.contains("sham"))
        );
    }

    #[test]
    fn sticky_control_fails_the_reversal_test() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Reversal)
            .unwrap()
            .observed_metric_value = 118.0;
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::NotQualified);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.contains("reversal"))
        );
    }

    #[test]
    fn collateral_musical_change_can_fail_a_successful_target_control() {
        let mut trial = valid_trial();
        let active = trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap();
        active.collateral_metrics[0].value = 0.9;
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::NotQualified);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.contains("collateral"))
        );
    }

    #[test]
    fn incomplete_audio_capture_is_unobserved_not_a_failed_control() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .observation
            .closure_status = RendererObservationClosureStatusV1::Incomplete;
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::Unobserved);
        assert!(result.measured_active_effect.is_none());
    }

    #[test]
    fn accepted_request_with_wrong_effective_value_fails() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .effective_value = Some(100.0);
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::NotQualified);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.contains("effective setting"))
        );
    }

    #[test]
    fn changed_renderer_identity_between_conditions_is_rejected() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .renderer_identity_sha256 = ADAPTER.to_owned();
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("renderer, adapter, configuration"));
    }

    #[test]
    fn producer_cannot_act_as_independent_observer() {
        let mut trial = valid_trial();
        for condition in &mut trial.conditions {
            condition.observation.event_observer_identity_sha256 = PRODUCER.to_owned();
        }
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("producer as its independent capture or event observer"));
    }

    #[test]
    fn randomized_order_slots_must_be_unique() {
        let mut trial = valid_trial();
        trial.conditions[0].randomized_order_slot = trial.conditions[1].randomized_order_slot;
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("permutation"));
    }

    #[test]
    fn condition_order_independent_of_array_order() {
        let mut trial = valid_trial();
        let expected_digest = assess_renderer_control_effect_v1(&trial)
            .unwrap()
            .trial_sha256;
        trial.conditions.reverse();
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(
            result.outcome,
            RendererEffectOutcomeV1::StructurallyQualified
        );
        assert_eq!(result.trial_sha256, expected_digest);
    }

    #[test]
    fn incomplete_capture_is_not_the_same_as_observed_silence() {
        let mut trial = valid_trial();
        trial.conditions[2].observation.captured_duration_ns = 1;
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::Unobserved);
    }

    #[test]
    fn reversal_must_follow_the_active_intervention() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .randomized_order_slot = 3;
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Reversal)
            .unwrap()
            .randomized_order_slot = 2;
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("reversal after active"));
    }

    #[test]
    fn negative_thresholds_are_rejected() {
        let mut trial = valid_trial();
        trial.maximum_sham_effect_abs = -1.0;
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("maximum sham effect must be finite and non-negative"));
    }

    #[test]
    fn non_finite_effective_setting_is_rejected() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .effective_value = Some(f64::INFINITY);
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("effective control value must be finite"));
    }

    #[test]
    fn randomization_receipt_must_bind_the_recorded_condition_order() {
        let mut trial = valid_trial();
        trial.randomization_receipt_sha256 = RECEIPT.to_owned();
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("randomization receipt digest"));
    }

    #[test]
    fn observer_build_or_configuration_change_makes_trial_unobserved() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .observation
            .event_observer_build_sha256 =
            "8888888888888888888888888888888888888888888888888888888888888888".to_owned();
        let result = assess_renderer_control_effect_v1(&trial).unwrap();
        assert_eq!(result.outcome, RendererEffectOutcomeV1::Unobserved);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.contains("event observer build"))
        );
    }

    #[test]
    fn event_manifest_must_reference_the_exact_captured_payload() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Active)
            .unwrap()
            .observation
            .event_manifest_capture_payload_sha256 =
            "8888888888888888888888888888888888888888888888888888888888888888".to_owned();
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("not bound to the captured audio payload"));
    }

    #[test]
    fn all_arms_must_report_the_same_collateral_metric_inventory() {
        let mut trial = valid_trial();
        trial
            .conditions
            .iter_mut()
            .find(|condition| condition.kind == RendererControlConditionKindV1::Sham)
            .unwrap()
            .collateral_metrics
            .pop();
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("collateral metric set or units"));
    }

    #[test]
    fn acoustic_dimensions_cannot_be_qualified_from_midi_tap_alone() {
        let mut trial = valid_trial();
        trial.dimension = RendererControlDimensionV1::Brightness;
        trial.metric_unit = "brightness_0_1".to_owned();
        for condition in &mut trial.conditions {
            condition.observed_metric_unit = "brightness_0_1".to_owned();
            condition.observation.source = RendererObservationSourceV1::IndependentMidiTap;
        }
        trial.observation_source = RendererObservationSourceV1::IndependentMidiTap;
        let error = assess_renderer_control_effect_v1(&trial).unwrap_err();
        assert!(error.contains("acoustic control dimensions require rendered-audio loopback"));
    }
}

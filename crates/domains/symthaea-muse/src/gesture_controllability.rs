// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired, renderer-neutral counterfactual benchmark for instrument gestures.
//!
//! A successful audio render does not prove that a requested gesture was
//! consumed. This module verifies paired baseline/intervention measurements
//! under one score, note-identity map, source manifest, renderer build and
//! feature-extractor build. It evaluates declared effects and side-effect
//! budgets; it does not independently qualify a capability or replace the
//! signed exact-cell verifier.

use crate::performance_gesture::{
    is_sha256_hex, scalar_domain_is_valid, value_matches_parameter, BeatPosition, GestureParameter,
    GestureTarget, GestureUnit, GestureValue, PerformanceGestureEvent, ScoreNoteIdentityMapV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

/// Wire schema for paired gesture-controllability receipts.
pub const GESTURE_CONTROLLABILITY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureUnit {
    Decibels,
    Hertz,
    Seconds,
    Milliseconds,
    Cents,
    Normalized,
    Ratio,
    Samples,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedDirection {
    Increase,
    Decrease,
    Either,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureMeasurement {
    pub target: GestureTarget,
    pub instrument_id: String,
    pub source_model_id: Option<String>,
    pub feature_id: String,
    pub value: f64,
    pub unit: FeatureUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedFeatureEffect {
    pub target: GestureTarget,
    pub feature_id: String,
    pub unit: FeatureUnit,
    pub minimum_abs_delta: f64,
    pub direction: ExpectedDirection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SideEffectBudget {
    pub target: GestureTarget,
    pub feature_id: String,
    pub unit: FeatureUnit,
    pub maximum_abs_delta: f64,
}

/// One intervention compared with its matched baseline. Shared identity fields
/// guarantee that the declared comparison uses one score, one instrument/model,
/// one asset manifest, one renderer build and one feature-extractor build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GestureCounterfactualReceiptV1 {
    pub schema_version: u16,
    pub score_sha256: String,
    pub note_identity_map_sha256: String,
    /// Digest of the preregistered, versioned intervention hypothesis and
    /// side-effect budget profile. Must resolve to a frozen protocol record.
    pub protocol_sha256: String,
    pub gesture_event_sha256: String,
    pub source_manifest_sha256: String,
    pub renderer_id: String,
    pub renderer_build_sha256: String,
    pub feature_extractor_id: String,
    pub feature_extractor_build_sha256: String,
    /// Digest of feature extraction configuration: windows, resampling,
    /// normalization and per-feature parameters.
    pub feature_extractor_config_sha256: String,
    pub event_id: String,
    pub instrument_id: String,
    pub source_model_id: String,
    pub target: GestureTarget,
    pub at: BeatPosition,
    pub parameter: GestureParameter,
    pub baseline_value: GestureValue,
    pub intervention_value: GestureValue,
    pub baseline_output_sha256: String,
    pub intervention_output_sha256: String,
    /// Must be sorted by feature_id, unique, finite and unit-stable.
    pub baseline_features: Vec<FeatureMeasurement>,
    /// Same feature IDs, order and units as baseline_features.
    pub intervention_features: Vec<FeatureMeasurement>,
    /// Must be sorted by feature_id and unique.
    pub expected_effects: Vec<ExpectedFeatureEffect>,
    /// Must be sorted by feature_id and unique; disjoint from expected_effects.
    pub side_effect_budgets: Vec<SideEffectBudget>,
    pub receipt_sha256: String,
}

impl GestureCounterfactualReceiptV1 {
    /// Recompute the receipt digest after constructing or editing a fixture.
    /// This is content integrity, not a signature or qualification attestation.
    pub fn seal(&mut self) {
        self.receipt_sha256 = self.compute_digest();
    }

    /// Check all shared conditions and prove that expected effects occurred
    /// while declared side effects remained within their budgets.
    pub fn verify(&self) -> Result<(), Vec<String>> {
        let mut issues = Vec::new();
        if self.schema_version != GESTURE_CONTROLLABILITY_SCHEMA_VERSION {
            issues.push(format!(
                "unsupported gesture controllability schema version: {}",
                self.schema_version
            ));
        }

        for (name, digest) in [
            ("score_sha256", self.score_sha256.as_str()),
            ("note_identity_map_sha256", self.note_identity_map_sha256.as_str()),
            ("protocol_sha256", self.protocol_sha256.as_str()),
            ("gesture_event_sha256", self.gesture_event_sha256.as_str()),
            ("source_manifest_sha256", self.source_manifest_sha256.as_str()),
            ("renderer_build_sha256", self.renderer_build_sha256.as_str()),
            (
                "feature_extractor_build_sha256",
                self.feature_extractor_build_sha256.as_str(),
            ),
            ("baseline_output_sha256", self.baseline_output_sha256.as_str()),
            (
                "intervention_output_sha256",
                self.intervention_output_sha256.as_str(),
            ),
        ] {
            if !is_sha256_hex(digest) {
                issues.push(format!("{name} is not a lowercase SHA-256 digest"));
            }
        }

        for (name, value) in [
            ("renderer_id", self.renderer_id.as_str()),
            ("feature_extractor_id", self.feature_extractor_id.as_str()),
            ("event_id", self.event_id.as_str()),
            ("instrument_id", self.instrument_id.as_str()),
            ("source_model_id", self.source_model_id.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(format!("{name} is empty"));
            }
        }

        match &self.target {
            GestureTarget::Part { part_id } if *part_id == u16::MAX => {
                issues.push("target part identity is unassigned".to_string());
            }
            GestureTarget::Note { part_id, note_id } => {
                if *part_id == u16::MAX || note_id.trim().is_empty() {
                    issues.push("target note identity is invalid".to_string());
                }
            }
            GestureTarget::Part { .. } => {}
        }

        if !self.at.is_canonical() {
            issues.push("gesture beat position is not canonical".to_string());
        }
        if !value_matches_parameter(self.parameter, &self.baseline_value)
            || !value_matches_parameter(self.parameter, &self.intervention_value)
        {
            issues.push("gesture values do not match the parameter's type/unit".to_string());
        } else {
            for value in [&self.baseline_value, &self.intervention_value] {
                if let GestureValue::Scalar { amount, unit } = value {
                    if !amount.is_finite()
                        || !scalar_domain_is_valid(self.parameter, *amount, *unit)
                    {
                        issues.push("gesture value is non-finite or outside its declared domain".to_string());
                        break;
                    }
                }
                if let GestureValue::Choice { value } = value {
                    if value.trim().is_empty() {
                        issues.push("gesture categorical value is empty".to_string());
                    }
                }
            }
        }
        if self.baseline_value == self.intervention_value {
            issues.push("baseline and intervention gesture values are identical".to_string());
        }

        validate_measurements("baseline_features", &self.baseline_features, &mut issues);
        validate_measurements(
            "intervention_features",
            &self.intervention_features,
            &mut issues,
        );
        if self.baseline_features.len() != self.intervention_features.len() {
            issues.push("baseline/intervention feature sets have different lengths".to_string());
        } else {
            for (baseline, intervention) in self
                .baseline_features
                .iter()
                .zip(self.intervention_features.iter())
            {
                if baseline.feature_id != intervention.feature_id
                    || baseline.target != intervention.target
                    || baseline.instrument_id != intervention.instrument_id
                    || baseline.source_model_id != intervention.source_model_id
                    || baseline.unit != intervention.unit
                {
                    issues.push("baseline/intervention feature identities or units differ".to_string());
                    break;
                }
            }
        }

        validate_sorted_unique_effects(&self.expected_effects, &mut issues);
        validate_sorted_unique_budgets(&self.side_effect_budgets, &mut issues);
        if self.expected_effects.is_empty() {
            issues.push("at least one expected feature effect is required".to_string());
        }
        if self.side_effect_budgets.iter().any(|budget| {
            self.expected_effects.iter().any(|effect| {
                effect.feature_id == budget.feature_id && effect.target == budget.target
            })
        }) {
            issues.push("the same target feature cannot be both an expected effect and a side-effect budget".to_string());
        }

        for effect in &self.expected_effects {
            let (Some(baseline), Some(intervention)) = (
                find_feature(&self.baseline_features, &effect.target, &effect.feature_id),
                find_feature(&self.intervention_features, &effect.target, &effect.feature_id),
            ) else {
                issues.push(format!(
                    "expected feature {} is missing from a paired measurement",
                    effect.feature_id
                ));
                continue;
            };
            if baseline.unit != effect.unit || intervention.unit != effect.unit {
                issues.push(format!("expected feature {} has a unit mismatch", effect.feature_id));
                continue;
            }
            if baseline.target != effect.target || intervention.target != effect.target {
                issues.push(format!("expected feature {} does not belong to its declared target", effect.feature_id));
                continue;
            }
            if effect.target != self.target {
                issues.push(format!("expected feature {} is not measured on the gesture target", effect.feature_id));
                continue;
            }
            let delta = intervention.value - baseline.value;
            if !delta.is_finite() {
                issues.push(format!("expected feature {} delta is non-finite", effect.feature_id));
                continue;
            }
            let magnitude = delta.abs();
            let direction_matches = match effect.direction {
                ExpectedDirection::Increase => delta > 0.0,
                ExpectedDirection::Decrease => delta < 0.0,
                ExpectedDirection::Either => magnitude > 0.0,
            };
            if !direction_matches || magnitude < effect.minimum_abs_delta {
                issues.push(format!(
                    "expected effect {} not observed: delta={delta}, minimum_abs_delta={}",
                    effect.feature_id, effect.minimum_abs_delta
                ));
            }
        }

        for budget in &self.side_effect_budgets {
            let (Some(baseline), Some(intervention)) = (
                find_feature(&self.baseline_features, &budget.target, &budget.feature_id),
                find_feature(&self.intervention_features, &budget.target, &budget.feature_id),
            ) else {
                issues.push(format!(
                    "side-effect feature {} is missing from a paired measurement",
                    budget.feature_id
                ));
                continue;
            };
            if baseline.unit != budget.unit || intervention.unit != budget.unit {
                issues.push(format!("side-effect feature {} has a unit mismatch", budget.feature_id));
                continue;
            }
            if baseline.target != budget.target || intervention.target != budget.target {
                issues.push(format!("side-effect feature {} does not belong to its declared target", budget.feature_id));
                continue;
            }
            let signed_delta = intervention.value - baseline.value;
            if !signed_delta.is_finite() {
                issues.push(format!("side-effect feature {} delta is non-finite", budget.feature_id));
                continue;
            }
            let delta = signed_delta.abs();
            if delta > budget.maximum_abs_delta {
                issues.push(format!(
                    "side-effect budget exceeded for {}: delta={delta}, maximum={}",
                    budget.feature_id, budget.maximum_abs_delta
                ));
            }
        }

        if self.baseline_output_sha256 == self.intervention_output_sha256
            && self.baseline_features != self.intervention_features
        {
            issues.push(
                "identical output audio hashes cannot have different measurements from the same extractor".to_string(),
            );
        }

        if !is_sha256_hex(&self.receipt_sha256)
            || self.receipt_sha256 != self.compute_digest()
        {
            issues.push("gesture counterfactual receipt digest is invalid or stale".to_string());
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Verify the intervention against the exact producer-assigned gesture
    /// event. The baseline must equal the event's original value; only the
    /// intervention value may differ.
    pub fn verify_against_event(
        &self,
        event: &PerformanceGestureEvent,
    ) -> Result<(), Vec<String>> {
        let mut issues = self.verify().err().unwrap_or_default();
        let event_digest = event.canonical_sha256();
        match event_digest {
            Ok(digest) if digest == self.gesture_event_sha256 => {}
            _ => issues.push("gesture event digest does not match the receipt".to_string()),
        }
        if event.event_id != self.event_id
            || event.instrument_id != self.instrument_id
            || event.source_model_id.as_deref() != Some(self.source_model_id.as_str())
            || event.target != self.target
            || event.at != self.at
            || event.parameter != self.parameter
            || event.value != self.baseline_value
        {
            issues.push("receipt identity or baseline value does not match the source gesture event".to_string());
        }
        if event.value == self.intervention_value {
            issues.push("intervention must change the source gesture event value".to_string());
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }

    /// Combined gate for the exact score, note identity map and source gesture.
    pub fn verify_against_event_and_identity_map(
        &self,
        expected_score_sha256: &str,
        expected_note_count: usize,
        identity_map: &ScoreNoteIdentityMapV1,
        event: &PerformanceGestureEvent,
    ) -> Result<(), Vec<String>> {
        let mut issues = self.verify_against_identity_map(
            expected_score_sha256,
            expected_note_count,
            identity_map,
        ).err().unwrap_or_default();
        if let Err(event_issues) = self.verify_against_event(event) {
            issues.extend(event_issues);
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }

    /// Bind the receipt to the exact producer-assigned identity map. A valid
    /// counterfactual receipt is still not a capability certificate.
    pub fn verify_against_identity_map(
        &self,
        expected_score_sha256: &str,
        expected_note_count: usize,
        identity_map: &ScoreNoteIdentityMapV1,
    ) -> Result<(), Vec<String>> {
        let mut issues = self.verify().err().unwrap_or_default();
        if let Err(map_issues) =
            identity_map.validate_for_score(expected_score_sha256, expected_note_count)
        {
            issues.extend(map_issues);
        }
        if self.score_sha256 != expected_score_sha256
            || identity_map.score_sha256 != expected_score_sha256
        {
            issues.push("counterfactual and identity map do not bind to the requested score".to_string());
        }
        if self.note_identity_map_sha256 != identity_map.identity_map_sha256 {
            issues.push("counterfactual uses a different note identity map".to_string());
        }

        for measurement in self.baseline_features.iter().chain(self.intervention_features.iter()) {
            match &measurement.target {
                GestureTarget::Note { part_id, note_id } => match identity_map.note(*part_id, note_id) {
                    None => issues.push(format!(
                        "feature {} references an unknown part/note identity",
                        measurement.feature_id
                    )),
                    Some(note) => {
                        if note.instrument_id != measurement.instrument_id
                            || note.source_model_id != measurement.source_model_id
                        {
                            issues.push(format!(
                                "feature {} instrument/model does not match its score note",
                                measurement.feature_id
                            ));
                        }
                    }
                },
                GestureTarget::Part { part_id } => {
                    let part_notes: Vec<_> = identity_map.notes.iter()
                        .filter(|note| note.part_id == *part_id).collect();
                    let homogeneous = !part_notes.is_empty() && part_notes.iter().all(|note| {
                        note.instrument_id == measurement.instrument_id
                            && note.source_model_id == measurement.source_model_id
                    });
                    if !homogeneous {
                        issues.push(format!(
                            "feature {} part target is absent or has heterogeneous instrument/model identity",
                            measurement.feature_id
                        ));
                    }
                }
            }
        }

        match &self.target {
            GestureTarget::Note { part_id, note_id } => match identity_map.note(*part_id, note_id) {
                None => issues.push("counterfactual references an unknown part/note identity".to_string()),
                Some(note) => {
                    if note.instrument_id != self.instrument_id
                        || note.source_model_id.as_deref() != Some(self.source_model_id.as_str())
                    {
                        issues.push("counterfactual instrument/model does not match target note".to_string());
                    }
                    if self.at.cmp_exact(note.onset) == Ordering::Less {
                        issues.push("counterfactual occurs before target note onset".to_string());
                    }
                    if note.onset.checked_add(note.duration).is_none_or(|end| {
                        self.at.cmp_exact(end) == Ordering::Greater
                    }) {
                        issues.push("counterfactual occurs after target note end".to_string());
                    }
                }
            },
            GestureTarget::Part { part_id } => {
                let part_notes: Vec<_> = identity_map
                    .notes
                    .iter()
                    .filter(|note| note.part_id == *part_id)
                    .collect();
                if part_notes.is_empty()
                    || !part_notes.iter().all(|note| {
                        note.instrument_id == self.instrument_id
                            && note.source_model_id.as_deref() == Some(self.source_model_id.as_str())
                    })
                {
                    issues.push("counterfactual part has no homogeneous matching instrument/model assignment".to_string());
                } else {
                    let first_onset = part_notes.iter()
                        .map(|note| note.onset)
                        .min_by(|left, right| left.cmp_exact(*right));
                    let latest_end = part_notes.iter()
                        .filter_map(|note| note.onset.checked_add(note.duration))
                        .max_by(|left, right| left.cmp_exact(*right));
                    match (first_onset, latest_end) {
                        (Some(first), Some(last))
                            if self.at.cmp_exact(first) != Ordering::Less
                                && self.at.cmp_exact(last) != Ordering::Greater => {}
                        _ => issues.push(
                            "counterfactual part-scoped gesture lies outside the part score-time envelope or has unrepresentable timing".to_string()
                        ),
                    }
                }
            }
        }

        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }

    fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:muse:gesture-counterfactual-receipt:v1\n");
        hasher.update(self.schema_version.to_be_bytes());
        hash_string(&mut hasher, &self.score_sha256);
        hash_string(&mut hasher, &self.note_identity_map_sha256);
        hash_string(&mut hasher, &self.protocol_sha256);
        hash_string(&mut hasher, &self.gesture_event_sha256);
        hash_string(&mut hasher, &self.source_manifest_sha256);
        hash_string(&mut hasher, &self.renderer_id);
        hash_string(&mut hasher, &self.renderer_build_sha256);
        hash_string(&mut hasher, &self.feature_extractor_id);
        hash_string(&mut hasher, &self.feature_extractor_build_sha256);
        hash_string(&mut hasher, &self.feature_extractor_config_sha256);
        hash_string(&mut hasher, &self.event_id);
        hash_string(&mut hasher, &self.instrument_id);
        hash_string(&mut hasher, &self.source_model_id);
        hash_target(&mut hasher, &self.target);
        hash_position(&mut hasher, self.at);
        hasher.update([parameter_code(self.parameter)]);
        hash_value(&mut hasher, &self.baseline_value);
        hash_value(&mut hasher, &self.intervention_value);
        hash_string(&mut hasher, &self.baseline_output_sha256);
        hash_string(&mut hasher, &self.intervention_output_sha256);
        hash_measurements(&mut hasher, &self.baseline_features);
        hash_measurements(&mut hasher, &self.intervention_features);
        hasher.update((self.expected_effects.len() as u64).to_be_bytes());
        for effect in &self.expected_effects {
            hash_target(&mut hasher, &effect.target);
            hash_string(&mut hasher, &effect.feature_id);
            hasher.update([feature_unit_code(effect.unit)]);
            hasher.update(effect.minimum_abs_delta.to_bits().to_be_bytes());
            hasher.update([direction_code(effect.direction)]);
        }
        hasher.update((self.side_effect_budgets.len() as u64).to_be_bytes());
        for budget in &self.side_effect_budgets {
            hash_target(&mut hasher, &budget.target);
            hash_string(&mut hasher, &budget.feature_id);
            hasher.update([feature_unit_code(budget.unit)]);
            hasher.update(budget.maximum_abs_delta.to_bits().to_be_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

fn validate_measurements(
    name: &str,
    measurements: &[FeatureMeasurement],
    issues: &mut Vec<String>,
) {
    let mut previous: Option<(&GestureTarget, &str)> = None;
    for measurement in measurements {
        if measurement.feature_id.trim().is_empty()
            || measurement.instrument_id.trim().is_empty()
            || measurement.source_model_id.as_ref().is_some_and(|model| model.trim().is_empty())
            || !measurement.value.is_finite()
        {
            issues.push(format!("{name} contains an empty identity/feature ID or non-finite value"));
        }
        match &measurement.target {
            GestureTarget::Part { part_id } if *part_id == u16::MAX => {
                issues.push(format!("{name} contains an unassigned part target"));
            }
            GestureTarget::Note { part_id, note_id }
                if *part_id == u16::MAX || note_id.trim().is_empty() =>
            {
                issues.push(format!("{name} contains an invalid note target"));
            }
            _ => {}
        }
        if previous.is_some_and(|(prior_target, prior_id)| {
            compare_feature_key(prior_target, prior_id, &measurement.target, &measurement.feature_id)
                != Ordering::Less
        }) {
            issues.push(format!("{name} must be sorted by unique (target, feature_id)"));
        }
        previous = Some((&measurement.target, measurement.feature_id.as_str()));
    }
}

fn validate_sorted_unique_effects(
    effects: &[ExpectedFeatureEffect],
    issues: &mut Vec<String>,
) {
    let mut previous: Option<(&GestureTarget, &str)> = None;
    for effect in effects {
        if effect.feature_id.trim().is_empty()
            || !effect.minimum_abs_delta.is_finite()
            || effect.minimum_abs_delta <= 0.0
        {
            issues.push("expected effect has an empty ID or invalid minimum delta".to_string());
        }
        if previous.is_some_and(|(prior_target, prior_id)| {
            compare_feature_key(prior_target, prior_id, &effect.target, &effect.feature_id)
                != Ordering::Less
        }) {
            issues.push("expected effects must be sorted by unique (target, feature_id)".to_string());
        }
        previous = Some((&effect.target, effect.feature_id.as_str()));
    }
}

fn validate_sorted_unique_budgets(
    budgets: &[SideEffectBudget],
    issues: &mut Vec<String>,
) {
    let mut previous: Option<(&GestureTarget, &str)> = None;
    for budget in budgets {
        if budget.feature_id.trim().is_empty()
            || !budget.maximum_abs_delta.is_finite()
            || budget.maximum_abs_delta < 0.0
        {
            issues.push("side-effect budget has an empty ID or invalid maximum delta".to_string());
        }
        if previous.is_some_and(|(prior_target, prior_id)| {
            compare_feature_key(prior_target, prior_id, &budget.target, &budget.feature_id)
                != Ordering::Less
        }) {
            issues.push("side-effect budgets must be sorted by unique (target, feature_id)".to_string());
        }
        previous = Some((&budget.target, budget.feature_id.as_str()));
    }
}

fn compare_target(left: &GestureTarget, right: &GestureTarget) -> Ordering {
    match (left, right) {
        (GestureTarget::Part { part_id: left }, GestureTarget::Part { part_id: right }) => {
            left.cmp(right)
        }
        (GestureTarget::Part { .. }, GestureTarget::Note { .. }) => Ordering::Less,
        (GestureTarget::Note { .. }, GestureTarget::Part { .. }) => Ordering::Greater,
        (
            GestureTarget::Note { part_id: left_part, note_id: left_note },
            GestureTarget::Note { part_id: right_part, note_id: right_note },
        ) => left_part.cmp(right_part).then_with(|| left_note.cmp(right_note)),
    }
}

fn compare_feature_key(
    left_target: &GestureTarget,
    left_id: &str,
    right_target: &GestureTarget,
    right_id: &str,
) -> Ordering {
    compare_target(left_target, right_target).then_with(|| left_id.cmp(right_id))
}

fn find_feature<'a>(
    measurements: &'a [FeatureMeasurement],
    target: &GestureTarget,
    feature_id: &str,
) -> Option<&'a FeatureMeasurement> {
    measurements.iter().find(|measurement| {
        &measurement.target == target && measurement.feature_id == feature_id
    })
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hash_target(hasher: &mut Sha256, target: &GestureTarget) {
    match target {
        GestureTarget::Part { part_id } => {
            hasher.update([0u8]);
            hasher.update(part_id.to_be_bytes());
        }
        GestureTarget::Note { part_id, note_id } => {
            hasher.update([1u8]);
            hasher.update(part_id.to_be_bytes());
            hash_string(hasher, note_id);
        }
    }
}

fn hash_position(hasher: &mut Sha256, at: BeatPosition) {
    hasher.update(at.numerator().to_be_bytes());
    hasher.update(at.denominator().to_be_bytes());
}

fn hash_value(hasher: &mut Sha256, value: &GestureValue) {
    match value {
        GestureValue::Scalar { amount, unit } => {
            hasher.update([0u8]);
            hasher.update(amount.to_bits().to_be_bytes());
            hasher.update([gesture_unit_code(*unit)]);
        }
        GestureValue::Boolean { value } => {
            hasher.update([1u8, u8::from(*value)]);
        }
        GestureValue::Choice { value } => {
            hasher.update([2u8]);
            hash_string(hasher, value);
        }
    }
}

fn hash_measurements(hasher: &mut Sha256, measurements: &[FeatureMeasurement]) {
    hasher.update((measurements.len() as u64).to_be_bytes());
    for measurement in measurements {
        hash_target(hasher, &measurement.target);
        hash_string(hasher, &measurement.instrument_id);
        match measurement.source_model_id.as_deref() {
            Some(model) => { hasher.update([1u8]); hash_string(hasher, model); }
            None => hasher.update([0u8]),
        }
        hash_string(hasher, &measurement.feature_id);
        hasher.update(measurement.value.to_bits().to_be_bytes());
        hasher.update([feature_unit_code(measurement.unit)]);
    }
}

// Hash codes are explicit and append-only. Never derive these from Rust enum
// declaration order: inserting/reordering a variant must not reinterpret old
// version-1 receipts.
fn parameter_code(parameter: GestureParameter) -> u8 {
    match parameter {
        GestureParameter::BowSpeed => 0,
        GestureParameter::BowForce => 1,
        GestureParameter::BowContactPoint => 2,
        GestureParameter::BreathPressure => 3,
        GestureParameter::TonguingMode => 4,
        GestureParameter::PluckForce => 5,
        GestureParameter::PluckPosition => 6,
        GestureParameter::DampingAmount => 7,
        GestureParameter::MalletProfile => 8,
        GestureParameter::StrikePosition => 9,
        GestureParameter::DamperPedal => 10,
        GestureParameter::VibraphoneMotorEnabled => 11,
        GestureParameter::TromboneSlidePosition => 12,
        GestureParameter::BellowsPressure => 13,
        GestureParameter::BellowsDirection => 14,
        GestureParameter::RegisterStop => 15,
        GestureParameter::VowelIdentity => 16,
        GestureParameter::FormantShift => 17,
        GestureParameter::OscillatorFrequency => 18,
        GestureParameter::FilterCutoff => 19,
        GestureParameter::EnvelopeLevel => 20,
        GestureParameter::PitchBend => 21,
        GestureParameter::VibratoRate => 22,
        GestureParameter::VibratoDepth => 23,
        GestureParameter::PianoDamperPedalDepth => 24,
        GestureParameter::PianoSostenutoPedal => 25,
        GestureParameter::PianoUnaCordaPedal => 26,
        GestureParameter::PianoStringResonance => 27,
        GestureParameter::PianoHammerHardness => 28,
    }
}

fn gesture_unit_code(unit: GestureUnit) -> u8 {
    match unit {
        GestureUnit::MetersPerSecond => 0,
        GestureUnit::Newtons => 1,
        GestureUnit::Pascals => 2,
        GestureUnit::Normalized => 3,
        GestureUnit::Millimeters => 4,
        GestureUnit::Hertz => 5,
        GestureUnit::Cents => 6,
    }
}

fn feature_unit_code(unit: FeatureUnit) -> u8 {
    match unit {
        FeatureUnit::Decibels => 0,
        FeatureUnit::Hertz => 1,
        FeatureUnit::Seconds => 2,
        FeatureUnit::Milliseconds => 3,
        FeatureUnit::Cents => 4,
        FeatureUnit::Normalized => 5,
        FeatureUnit::Ratio => 6,
        FeatureUnit::Samples => 7,
    }
}

fn direction_code(direction: ExpectedDirection) -> u8 {
    match direction {
        ExpectedDirection::Increase => 0,
        ExpectedDirection::Decrease => 1,
        ExpectedDirection::Either => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_gesture::{ScoreNoteIdentityV1, ScoreNoteIdentityMapV1};

    fn identity_map() -> ScoreNoteIdentityMapV1 {
        ScoreNoteIdentityMapV1::new(
            "a".repeat(64),
            2,
            vec![
                ScoreNoteIdentityV1 {
                    note_index: 0,
                    note_id: "vln-note-1".to_string(),
                    part_id: 4,
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    onset: BeatPosition::new(0, 1).unwrap(),
                    duration: BeatPosition::new(2, 1).unwrap(),
                },
                ScoreNoteIdentityV1 {
                    note_index: 1,
                    note_id: "vln-note-2".to_string(),
                    part_id: 4,
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    onset: BeatPosition::new(2, 1).unwrap(),
                    duration: BeatPosition::new(2, 1).unwrap(),
                },
            ],
        )
    }

    fn piano_identity_map() -> ScoreNoteIdentityMapV1 {
        ScoreNoteIdentityMapV1::new(
            "a".repeat(64),
            2,
            vec![
                ScoreNoteIdentityV1 {
                    note_index: 0,
                    note_id: "pno-note-1".to_string(),
                    part_id: 8,
                    instrument_id: "piano".to_string(),
                    source_model_id: Some("piano.model-a".to_string()),
                    onset: BeatPosition::new(0, 1).unwrap(),
                    duration: BeatPosition::new(2, 1).unwrap(),
                },
                ScoreNoteIdentityV1 {
                    note_index: 1,
                    note_id: "pno-note-2".to_string(),
                    part_id: 8,
                    instrument_id: "piano".to_string(),
                    source_model_id: Some("piano.model-a".to_string()),
                    onset: BeatPosition::new(2, 1).unwrap(),
                    duration: BeatPosition::new(2, 1).unwrap(),
                },
            ],
        )
    }

    fn piano_pedal_event() -> PerformanceGestureEvent {
        PerformanceGestureEvent {
            event_id: "piano-damper-depth-1".to_string(),
            instrument_id: "piano".to_string(),
            source_model_id: Some("piano.model-a".to_string()),
            target: GestureTarget::Part { part_id: 8 },
            at: BeatPosition::new(1, 1).unwrap(),
            parameter: GestureParameter::PianoDamperPedalDepth,
            value: GestureValue::Scalar {
                amount: 0.2,
                unit: GestureUnit::Normalized,
            },
            interpolation: crate::performance_gesture::GestureInterpolation::Linear,
        }
    }

    fn gesture_event() -> PerformanceGestureEvent {
        PerformanceGestureEvent {
            event_id: "bow-force-event-1".to_string(),
            instrument_id: "violin".to_string(),
            source_model_id: Some("violin.model-a".to_string()),
            target: GestureTarget::Note {
                part_id: 4,
                note_id: "vln-note-1".to_string(),
            },
            at: BeatPosition::new(1, 1).unwrap(),
            parameter: GestureParameter::BowForce,
            value: GestureValue::Scalar {
                amount: 1.0,
                unit: GestureUnit::Newtons,
            },
            interpolation: crate::performance_gesture::GestureInterpolation::Linear,
        }
    }

    fn receipt() -> GestureCounterfactualReceiptV1 {
        let mut value = GestureCounterfactualReceiptV1 {
            schema_version: GESTURE_CONTROLLABILITY_SCHEMA_VERSION,
            score_sha256: "a".repeat(64),
            note_identity_map_sha256: identity_map().identity_map_sha256,
            protocol_sha256: "2".repeat(64),
            gesture_event_sha256: gesture_event().canonical_sha256().unwrap(),
            source_manifest_sha256: "b".repeat(64),
            renderer_id: "muse.procedural.violin".to_string(),
            renderer_build_sha256: "c".repeat(64),
            feature_extractor_id: "muse.audio.features.v1".to_string(),
            feature_extractor_build_sha256: "d".repeat(64),
            feature_extractor_config_sha256: "3".repeat(64),
            event_id: "bow-force-event-1".to_string(),
            instrument_id: "violin".to_string(),
            source_model_id: "violin.model-a".to_string(),
            target: GestureTarget::Note {
                part_id: 4,
                note_id: "vln-note-1".to_string(),
            },
            at: BeatPosition::new(1, 1).unwrap(),
            parameter: GestureParameter::BowForce,
            baseline_value: GestureValue::Scalar {
                amount: 1.0,
                unit: GestureUnit::Newtons,
            },
            intervention_value: GestureValue::Scalar {
                amount: 2.0,
                unit: GestureUnit::Newtons,
            },
            baseline_output_sha256: "e".repeat(64),
            intervention_output_sha256: "f".repeat(64),
            baseline_features: vec![
                FeatureMeasurement {
                    target: GestureTarget::Note {
                    part_id: 4,
                    note_id: "vln-note-1".to_string(),
                },
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    feature_id: "onset_latency_ms".to_string(),
                    value: 18.0,
                    unit: FeatureUnit::Milliseconds,
                },
                FeatureMeasurement {
                    target: GestureTarget::Note {
                    part_id: 4,
                    note_id: "vln-note-1".to_string(),
                },
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    feature_id: "spectral_centroid_hz".to_string(),
                    value: 1800.0,
                    unit: FeatureUnit::Hertz,
                },
            ],
            intervention_features: vec![
                FeatureMeasurement {
                    target: GestureTarget::Note {
                        part_id: 4,
                        note_id: "vln-note-1".to_string(),
                    },
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    feature_id: "onset_latency_ms".to_string(),
                    value: 19.0,
                    unit: FeatureUnit::Milliseconds,
                },
                FeatureMeasurement {
                    target: GestureTarget::Note {
                        part_id: 4,
                        note_id: "vln-note-1".to_string(),
                    },
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    feature_id: "spectral_centroid_hz".to_string(),
                    value: 2050.0,
                    unit: FeatureUnit::Hertz,
                },
            ],
            expected_effects: vec![ExpectedFeatureEffect {
                target: GestureTarget::Note {
                    part_id: 4,
                    note_id: "vln-note-1".to_string(),
                },
                feature_id: "spectral_centroid_hz".to_string(),
                unit: FeatureUnit::Hertz,
                minimum_abs_delta: 100.0,
                direction: ExpectedDirection::Increase,
            }],
            side_effect_budgets: vec![SideEffectBudget {
                target: GestureTarget::Note {
                    part_id: 4,
                    note_id: "vln-note-1".to_string(),
                },
                feature_id: "onset_latency_ms".to_string(),
                unit: FeatureUnit::Milliseconds,
                maximum_abs_delta: 3.0,
            }],
            receipt_sha256: String::new(),
        };
        value.seal();
        value
    }

    #[test]
    fn passes_when_expected_feature_changes_and_side_effects_stay_bounded() {
        let value = receipt();
        assert!(value.verify().is_ok());
        assert!(value
            .verify_against_event_and_identity_map(
                &"a".repeat(64), 2, &identity_map(), &gesture_event()
            )
            .is_ok());
    }

    #[test]
    fn piano_half_pedal_counterfactual_binds_part_scope_and_model() {
        let identity_map = piano_identity_map();
        let source_event = piano_pedal_event();
        let mut value = receipt();
        value.note_identity_map_sha256 = identity_map.identity_map_sha256.clone();
        value.protocol_sha256 = "4".repeat(64);
        value.gesture_event_sha256 = source_event.canonical_sha256().unwrap();
        value.source_manifest_sha256 = "5".repeat(64);
        value.renderer_id = "muse.physical-piano-model".to_string();
        value.renderer_build_sha256 = "6".repeat(64);
        value.feature_extractor_config_sha256 = "7".repeat(64);
        value.event_id = source_event.event_id.clone();
        value.instrument_id = "piano".to_string();
        value.source_model_id = "piano.model-a".to_string();
        value.target = GestureTarget::Part { part_id: 8 };
        value.parameter = GestureParameter::PianoDamperPedalDepth;
        value.baseline_value = GestureValue::Scalar {
            amount: 0.2,
            unit: GestureUnit::Normalized,
        };
        value.intervention_value = GestureValue::Scalar {
            amount: 0.8,
            unit: GestureUnit::Normalized,
        };
        value.baseline_features = vec![
            FeatureMeasurement {
                target: GestureTarget::Part { part_id: 8 },
                instrument_id: "piano".to_string(),
                source_model_id: Some("piano.model-a".to_string()),
                feature_id: "pedal_decay_ms".to_string(),
                value: 800.0,
                unit: FeatureUnit::Milliseconds,
            },
            FeatureMeasurement {
                target: GestureTarget::Part { part_id: 8 },
                instrument_id: "piano".to_string(),
                source_model_id: Some("piano.model-a".to_string()),
                feature_id: "note_attack_ms".to_string(),
                value: 18.0,
                unit: FeatureUnit::Milliseconds,
            },
        ];
        value.intervention_features = vec![
            FeatureMeasurement {
                target: GestureTarget::Part { part_id: 8 },
                instrument_id: "piano".to_string(),
                source_model_id: Some("piano.model-a".to_string()),
                feature_id: "pedal_decay_ms".to_string(),
                value: 1200.0,
                unit: FeatureUnit::Milliseconds,
            },
            FeatureMeasurement {
                target: GestureTarget::Part { part_id: 8 },
                instrument_id: "piano".to_string(),
                source_model_id: Some("piano.model-a".to_string()),
                feature_id: "note_attack_ms".to_string(),
                value: 19.0,
                unit: FeatureUnit::Milliseconds,
            },
        ];
        value.expected_effects = vec![ExpectedFeatureEffect {
            target: GestureTarget::Part { part_id: 8 },
            feature_id: "pedal_decay_ms".to_string(),
            unit: FeatureUnit::Milliseconds,
            minimum_abs_delta: 200.0,
            direction: ExpectedDirection::Increase,
        }];
        value.side_effect_budgets = vec![SideEffectBudget {
            target: GestureTarget::Part { part_id: 8 },
            feature_id: "note_attack_ms".to_string(),
            unit: FeatureUnit::Milliseconds,
            maximum_abs_delta: 2.0,
        }];
        value.seal();

        assert!(value.verify().is_ok());
        assert!(value.verify_against_event_and_identity_map(
            &"a".repeat(64),
            2,
            &identity_map,
            &source_event,
        ).is_ok());

        value.intervention_value = GestureValue::Scalar {
            amount: 0.8,
            unit: GestureUnit::Hertz,
        };
        value.seal();
        assert!(value.verify().is_err());
    }

    #[test]
    fn rejects_receipt_with_different_source_gesture_event_or_baseline_value() {
        let value = receipt();
        let mut mismatched = gesture_event();
        mismatched.value = GestureValue::Scalar {
            amount: 1.5,
            unit: GestureUnit::Newtons,
        };
        assert!(value.verify_against_event(&mismatched).unwrap_err()
            .iter().any(|issue| issue.contains("gesture event digest")));

        let mut wrong_parameter = gesture_event();
        wrong_parameter.parameter = GestureParameter::PluckForce;
        assert!(value.verify_against_event(&wrong_parameter).is_err());
    }

    #[test]
    fn rejects_a_renderer_that_ignores_the_requested_gesture() {
        let mut value = receipt();
        value.intervention_features[1].value = 1820.0;
        value.seal();
        assert!(value
            .verify()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("expected effect spectral_centroid_hz not observed")));
    }

    #[test]
    fn rejects_wrong_direction_even_when_magnitude_is_large() {
        let mut value = receipt();
        value.expected_effects[0].direction = ExpectedDirection::Decrease;
        value.seal();
        assert!(value
            .verify()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("expected effect spectral_centroid_hz not observed")));
    }

    #[test]
    fn rejects_side_effects_outside_the_declared_budget() {
        let mut value = receipt();
        value.intervention_features[0].value = 40.0;
        value.seal();
        assert!(value
            .verify()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("side-effect budget exceeded")));
    }

    #[test]
    fn detects_cross_note_leakage_with_a_note_scoped_side_effect_budget() {
        let mut value = receipt();
        let other_target = GestureTarget::Note {
            part_id: 4,
            note_id: "vln-note-2".to_string(),
        };
        value.baseline_features.push(FeatureMeasurement {
            target: other_target.clone(),
            instrument_id: "violin".to_string(),
            source_model_id: Some("violin.model-a".to_string()),
            feature_id: "spectral_centroid_hz".to_string(),
            value: 0.3,
            unit: FeatureUnit::Normalized,
        });
        value.intervention_features.push(FeatureMeasurement {
            target: other_target.clone(),
            instrument_id: "violin".to_string(),
            source_model_id: Some("violin.model-a".to_string()),
            feature_id: "spectral_centroid_hz".to_string(),
            value: 0.3,
            unit: FeatureUnit::Normalized,
        });
        value.baseline_features.sort_by(|a, b| {
            compare_feature_key(&a.target, &a.feature_id, &b.target, &b.feature_id)
        });
        value.intervention_features.sort_by(|a, b| {
            compare_feature_key(&a.target, &a.feature_id, &b.target, &b.feature_id)
        });
        value.side_effect_budgets.push(SideEffectBudget {
            target: other_target.clone(),
            feature_id: "spectral_centroid_hz".to_string(),
            unit: FeatureUnit::Normalized,
            maximum_abs_delta: 0.0,
        });
        value.side_effect_budgets.sort_by(|a, b| {
            compare_feature_key(&a.target, &a.feature_id, &b.target, &b.feature_id)
        });
        value.seal();
        assert!(value.verify().is_ok());
        assert!(value
            .verify_against_identity_map(&"a".repeat(64), 2, &identity_map())
            .is_ok());

        value.intervention_features.iter_mut()
            .find(|feature| feature.target == other_target
                && feature.feature_id == "spectral_centroid_hz")
            .unwrap().value = 0.31;
        value.seal();
        assert!(value.verify().unwrap_err().iter()
            .any(|issue| issue.contains("side-effect budget exceeded for spectral_centroid_hz")));
    }

    #[test]
    fn rejects_identical_audio_with_inconsistent_feature_measurements() {
        let mut value = receipt();
        value.intervention_output_sha256 = value.baseline_output_sha256.clone();
        value.seal();
        assert!(value
            .verify()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("identical output audio hashes")));
    }

    #[test]
    fn rejects_stale_or_tampered_counterfactual_receipts() {
        let mut value = receipt();
        value.intervention_output_sha256 = "1".repeat(64);
        assert!(value
            .verify()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("receipt digest is invalid or stale")));
    }

    #[test]
    fn rejects_counterfactuals_for_the_wrong_note_or_instrument() {
        let mut value = receipt();
        value.target = GestureTarget::Note {
            part_id: 99,
            note_id: "vln-note-1".to_string(),
        };
        value.seal();
        assert!(value
            .verify_against_identity_map(&"a".repeat(64), 2, &identity_map())
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("unknown part/note")));

        let mut wrong_instrument = receipt();
        wrong_instrument.instrument_id = "flute".to_string();
        wrong_instrument.seal();
        assert!(wrong_instrument
            .verify_against_identity_map(&"a".repeat(64), 2, &identity_map())
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("instrument/model")));
    }

    #[test]
    fn rejects_missing_or_non_finite_feature_measurements() {
        let mut missing = receipt();
        missing.intervention_features.pop();
        missing.seal();
        assert!(missing.verify().is_err());

        let mut non_finite = receipt();
        non_finite.intervention_features[1].value = f64::NAN;
        non_finite.seal();
        assert!(non_finite
            .verify()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("non-finite")));
    }
}

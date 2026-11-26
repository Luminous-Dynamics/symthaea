// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Information-set guard for observable semiconductor process-regime routing.
//!
//! This gate distinguishes deployable observable-context routing from
//! retrospective target-conditioned oracle stratification. It verifies
//! feature source, schema, availability, clock and model-training declarations
//! before a route can be marked deployable. It does not train or authenticate
//! the external router model.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const REGIME_ROUTING_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.regime-routing-input.v1";
pub const REGIME_ROUTER_TRAINING_MANIFEST_SCHEMA_V1: &str =
    "luminous.semi-bench.regime-router-training-manifest.v1";
pub const REGIME_ROUTING_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.regime-routing-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeRoutingPurposeV1 {
    ObservableContext,
    RetrospectiveOracle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeFeatureSourceKindV1 {
    ObservableProcessContext,
    SensorObservation,
    TargetDerived,
    PostDecision,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeRoutingStateV1 {
    Eligible,
    Unknown,
    Rejected,
    RetrospectiveOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeRoutingReasonV1 {
    NoFeaturesProvided,
    DuplicateFeatureId,
    FeatureNotInTrainingAllowlist,
    FeatureForbiddenByTrainingManifest,
    TargetFeatureNotForbiddenInTrainingManifest,
    TrainingManifestContainsTargetFeature,
    TrainingCutoffAfterDecision,
    TrainingManifestDigestMismatch,
    RouterArtifactMismatch,
    TargetMeasurandMismatch,
    FeatureSchemaMismatch,
    FeatureSchemaDigestMalformed,
    FeatureValueNonFinite,
    FeatureUnitMissing,
    FeatureSourceUnknown,
    FeatureTargetDerived,
    FeaturePostDecision,
    FeatureEventAfterDecision,
    FeatureNotAvailableAtDecision,
    FeatureClockDomainMismatch,
    FeatureClockMappingMismatch,
    FeatureClockMappingMissing,
    FeatureSourceDigestMissing,
    FeatureTransformDigestMissing,
    FeatureContextIdentityMissing,
    RouterClockDomainMismatch,
    RouterClockMappingMismatch,
    RouteIdentityMissing,
    RouteScoreInvalid,
    RetrospectiveTargetFeatureUsed,
    RetrospectivePostDecisionFeatureUsed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegimeFeatureObservationV1 {
    pub feature_id: String,
    pub value: f64,
    pub unit: String,
    pub source_kind: RegimeFeatureSourceKindV1,
    pub source_artifact_sha256: Option<String>,
    pub transform_sha256: Option<String>,
    pub event_time_unix_seconds: i64,
    pub available_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub clock_mapping_sha256: String,
    pub feature_schema_sha256: String,
    pub subject_context_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegimeRouterTrainingManifestV1 {
    pub schema: String,
    pub router_artifact_sha256: String,
    pub source_dataset_manifest_sha256: String,
    pub feature_schema_sha256: String,
    pub target_measurand_id: String,
    pub allowed_feature_ids: Vec<String>,
    pub forbidden_target_feature_ids: Vec<String>,
    pub fit_cutoff_unix_seconds: i64,
    pub clock_domain_id: String,
    pub clock_mapping_sha256: String,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegimeRoutingInputV1 {
    pub schema: String,
    pub decision_id: String,
    pub purpose: RegimeRoutingPurposeV1,
    pub decision_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub clock_mapping_sha256: String,
    pub feature_schema_sha256: String,
    pub target_measurand_id: String,
    pub route_id: String,
    pub route_score: f64,
    pub router_artifact_sha256: String,
    pub training_manifest: RegimeRouterTrainingManifestV1,
    pub consumed_features: Vec<RegimeFeatureObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegimeRoutingReceiptV1 {
    pub schema: String,
    pub decision_id: String,
    pub purpose: RegimeRoutingPurposeV1,
    pub state: RegimeRoutingStateV1,
    pub route_id: String,
    pub route_score: f64,
    pub decision_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub clock_mapping_sha256: String,
    pub router_artifact_sha256: String,
    pub training_manifest_sha256: String,
    pub input_sha256: String,
    pub consumed_feature_set_sha256: String,
    pub reasons: Vec<RegimeRoutingReasonV1>,
    pub deployable_router_eligible: bool,
    pub retrospective_oracle_only: bool,
    pub router_issuer_authenticity_proven: bool,
    pub training_manifest_issuer_authenticity_proven: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct RegimeRoutingGuardErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), RegimeRoutingGuardErrorV1> {
    if value.trim().is_empty() {
        return Err(RegimeRoutingGuardErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, RegimeRoutingGuardErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        RegimeRoutingGuardErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn canonical_manifest(
    manifest: &RegimeRouterTrainingManifestV1,
) -> RegimeRouterTrainingManifestV1 {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    canonical.allowed_feature_ids.sort();
    canonical.forbidden_target_feature_ids.sort();
    canonical
}

fn canonical_features(features: &[RegimeFeatureObservationV1]) -> Vec<RegimeFeatureObservationV1> {
    let mut canonical = features.to_vec();
    canonical.sort_by(|left, right| left.feature_id.cmp(&right.feature_id));
    canonical
}

fn canonical_input(input: &RegimeRoutingInputV1) -> RegimeRoutingInputV1 {
    let mut canonical = input.clone();
    canonical.training_manifest = canonical_manifest(&canonical.training_manifest);
    canonical.consumed_features = canonical_features(&canonical.consumed_features);
    canonical
}

fn training_manifest_digest(
    manifest: &RegimeRouterTrainingManifestV1,
) -> Result<String, RegimeRoutingGuardErrorV1> {
    json_sha256(&canonical_manifest(manifest))
}

fn feature_set_digest(
    features: &[RegimeFeatureObservationV1],
) -> Result<String, RegimeRoutingGuardErrorV1> {
    json_sha256(&canonical_features(features))
}

fn receipt_digest(
    receipt: &RegimeRoutingReceiptV1,
) -> Result<String, RegimeRoutingGuardErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

/// Seals a router training manifest. A digest establishes content identity,
/// not proof that the declared training process was faithfully executed.
pub fn seal_regime_router_training_manifest_v1(
    manifest: &RegimeRouterTrainingManifestV1,
) -> Result<String, RegimeRoutingGuardErrorV1> {
    if manifest.schema != REGIME_ROUTER_TRAINING_MANIFEST_SCHEMA_V1 {
        return Err(RegimeRoutingGuardErrorV1(
            "router training manifest schema is unsupported".into(),
        ));
    }
    Ok(training_manifest_digest(manifest)?)
}

/// Replays the feature-information-set checks and seals a route receipt.
///
/// ObservableContext decisions can be ELIGIBLE only when every consumed
/// feature is observable, in the declared schema/clock, available by the
/// decision frontier, and not target-derived. RetrospectiveOracle decisions
/// are always marked RETROSPECTIVE_ONLY and never deployable.
pub fn evaluate_regime_routing_v1(
    input: &RegimeRoutingInputV1,
) -> Result<RegimeRoutingReceiptV1, RegimeRoutingGuardErrorV1> {
    if input.schema != REGIME_ROUTING_INPUT_SCHEMA_V1 {
        return Err(RegimeRoutingGuardErrorV1(
            "regime routing input schema is unsupported".into(),
        ));
    }
    require_text(&input.decision_id, "input.decision_id")?;
    require_text(&input.clock_domain_id, "input.clock_domain_id")?;
    require_text(&input.target_measurand_id, "input.target_measurand_id")?;
    require_text(&input.route_id, "input.route_id")?;
    require_text(
        &input.training_manifest.clock_domain_id,
        "training_manifest.clock_domain_id",
    )?;

    if input.decision_at_unix_seconds < 0 {
        return Err(RegimeRoutingGuardErrorV1(
            "decision frontier must be a non-negative Unix timestamp".into(),
        ));
    }
    if !valid_sha256(&input.clock_mapping_sha256)
        || !valid_sha256(&input.feature_schema_sha256)
        || !valid_sha256(&input.router_artifact_sha256)
    {
        return Err(RegimeRoutingGuardErrorV1(
            "routing input contains a malformed clock, feature-schema or router digest".into(),
        ));
    }
    if !input.route_score.is_finite() || !(0.0..=1.0).contains(&input.route_score) {
        return Err(RegimeRoutingGuardErrorV1(
            "route score must be finite and lie in [0, 1]".into(),
        ));
    }

    let manifest = &input.training_manifest;
    if manifest.schema != REGIME_ROUTER_TRAINING_MANIFEST_SCHEMA_V1 {
        return Err(RegimeRoutingGuardErrorV1(
            "router training manifest schema is unsupported".into(),
        ));
    }
    let mut reasons = Vec::new();
    let mut rejected = false;
    let mut unknown = false;

    if !valid_sha256(&manifest.router_artifact_sha256)
        || !valid_sha256(&manifest.source_dataset_manifest_sha256)
        || !valid_sha256(&manifest.feature_schema_sha256)
        || !valid_sha256(&manifest.clock_mapping_sha256)
        || !valid_sha256(&manifest.manifest_sha256)
        || training_manifest_digest(manifest)? != manifest.manifest_sha256
    {
        reasons.push(RegimeRoutingReasonV1::TrainingManifestDigestMismatch);
        rejected = true;
    }
    if manifest.router_artifact_sha256 != input.router_artifact_sha256 {
        reasons.push(RegimeRoutingReasonV1::RouterArtifactMismatch);
        rejected = true;
    }
    if manifest.target_measurand_id != input.target_measurand_id {
        reasons.push(RegimeRoutingReasonV1::TargetMeasurandMismatch);
        rejected = true;
    }
    if manifest.feature_schema_sha256 != input.feature_schema_sha256 {
        reasons.push(RegimeRoutingReasonV1::FeatureSchemaMismatch);
        rejected = true;
    }
    if manifest.clock_domain_id != input.clock_domain_id {
        reasons.push(RegimeRoutingReasonV1::RouterClockDomainMismatch);
        unknown = true;
    }
    if manifest.clock_mapping_sha256 != input.clock_mapping_sha256 {
        reasons.push(RegimeRoutingReasonV1::RouterClockMappingMismatch);
        unknown = true;
    }
    if manifest.fit_cutoff_unix_seconds < 0 {
        reasons.push(RegimeRoutingReasonV1::TrainingCutoffAfterDecision);
        rejected = true;
    } else if manifest.fit_cutoff_unix_seconds > input.decision_at_unix_seconds {
        reasons.push(RegimeRoutingReasonV1::TrainingCutoffAfterDecision);
        rejected = true;
    }

    let allowed: BTreeSet<&str> = manifest
        .allowed_feature_ids
        .iter()
        .map(String::as_str)
        .collect();
    let forbidden: BTreeSet<&str> = manifest
        .forbidden_target_feature_ids
        .iter()
        .map(String::as_str)
        .collect();

    if allowed.len() != manifest.allowed_feature_ids.len()
        || forbidden.len() != manifest.forbidden_target_feature_ids.len()
        || manifest.allowed_feature_ids.iter().any(|id| id.trim().is_empty())
        || manifest.forbidden_target_feature_ids.iter().any(|id| id.trim().is_empty())
    {
        reasons.push(RegimeRoutingReasonV1::DuplicateFeatureId);
        rejected = true;
    }
    if !forbidden.contains(input.target_measurand_id.as_str()) {
        reasons.push(RegimeRoutingReasonV1::TargetFeatureNotForbiddenInTrainingManifest);
        rejected = true;
    }
    if let Some(target_feature) = allowed.intersection(&forbidden).next() {
        let _ = target_feature;
        reasons.push(RegimeRoutingReasonV1::TrainingManifestContainsTargetFeature);
        rejected = true;
    }

    if input.consumed_features.is_empty() {
        reasons.push(RegimeRoutingReasonV1::NoFeaturesProvided);
        unknown = true;
    }

    let mut seen = BTreeSet::new();
    for feature in &input.consumed_features {
        require_text(&feature.feature_id, "feature.feature_id")?;
        if !seen.insert(feature.feature_id.as_str()) {
            reasons.push(RegimeRoutingReasonV1::DuplicateFeatureId);
            rejected = true;
            continue;
        }
        let retrospective_target_feature = input.purpose == RegimeRoutingPurposeV1::RetrospectiveOracle
            && feature.source_kind == RegimeFeatureSourceKindV1::TargetDerived;
        if !allowed.contains(feature.feature_id.as_str()) && !retrospective_target_feature {
            reasons.push(RegimeRoutingReasonV1::FeatureNotInTrainingAllowlist);
            rejected = true;
        }
        if forbidden.contains(feature.feature_id.as_str()) && !retrospective_target_feature {
            reasons.push(RegimeRoutingReasonV1::FeatureForbiddenByTrainingManifest);
            rejected = true;
        }
        if feature.feature_id == input.target_measurand_id {
            if input.purpose == RegimeRoutingPurposeV1::ObservableContext {
                reasons.push(RegimeRoutingReasonV1::FeatureTargetDerived);
                rejected = true;
            } else {
                reasons.push(RegimeRoutingReasonV1::RetrospectiveTargetFeatureUsed);
            }
        }

        if !feature.value.is_finite() {
            reasons.push(RegimeRoutingReasonV1::FeatureValueNonFinite);
            rejected = true;
        }
        if feature.unit.trim().is_empty() {
            reasons.push(RegimeRoutingReasonV1::FeatureUnitMissing);
            unknown = true;
        }
        if feature.feature_schema_sha256 != input.feature_schema_sha256
            || !valid_sha256(&feature.feature_schema_sha256)
        {
            reasons.push(RegimeRoutingReasonV1::FeatureSchemaMismatch);
            rejected = true;
        }
        if feature.clock_domain_id != input.clock_domain_id {
            reasons.push(RegimeRoutingReasonV1::FeatureClockDomainMismatch);
            unknown = true;
        }
        if !valid_sha256(&feature.clock_mapping_sha256)
            || feature.clock_mapping_sha256 != input.clock_mapping_sha256
        {
            reasons.push(RegimeRoutingReasonV1::FeatureClockMappingMismatch);
            unknown = true;
        }
        match feature.source_kind {
            RegimeFeatureSourceKindV1::ObservableProcessContext
            | RegimeFeatureSourceKindV1::SensorObservation => {}
            RegimeFeatureSourceKindV1::TargetDerived => {
                if input.purpose == RegimeRoutingPurposeV1::ObservableContext {
                    reasons.push(RegimeRoutingReasonV1::FeatureTargetDerived);
                    rejected = true;
                } else {
                    reasons.push(RegimeRoutingReasonV1::RetrospectiveTargetFeatureUsed);
                }
            }
            RegimeFeatureSourceKindV1::PostDecision => {
                if input.purpose == RegimeRoutingPurposeV1::ObservableContext {
                    reasons.push(RegimeRoutingReasonV1::FeaturePostDecision);
                    rejected = true;
                } else {
                    reasons.push(RegimeRoutingReasonV1::RetrospectivePostDecisionFeatureUsed);
                }
            }
            RegimeFeatureSourceKindV1::Unknown => {
                reasons.push(RegimeRoutingReasonV1::FeatureSourceUnknown);
                unknown = true;
            }
        }

        if feature.event_time_unix_seconds < 0
            || feature.available_at_unix_seconds < feature.event_time_unix_seconds
        {
            reasons.push(RegimeRoutingReasonV1::FeatureSourceUnknown);
            unknown = true;
        } else if feature.event_time_unix_seconds > input.decision_at_unix_seconds {
            if input.purpose == RegimeRoutingPurposeV1::ObservableContext {
                reasons.push(RegimeRoutingReasonV1::FeatureEventAfterDecision);
                rejected = true;
            } else {
                reasons.push(RegimeRoutingReasonV1::RetrospectivePostDecisionFeatureUsed);
            }
        }
        if feature.available_at_unix_seconds > input.decision_at_unix_seconds {
            if input.purpose == RegimeRoutingPurposeV1::ObservableContext {
                reasons.push(RegimeRoutingReasonV1::FeatureNotAvailableAtDecision);
                rejected = true;
            } else {
                reasons.push(RegimeRoutingReasonV1::RetrospectivePostDecisionFeatureUsed);
            }
        }
        if feature.source_artifact_sha256.as_deref().is_none_or(|sha| !valid_sha256(sha)) {
            reasons.push(RegimeRoutingReasonV1::FeatureSourceDigestMissing);
            unknown = true;
        }
        if feature.transform_sha256.as_deref().is_none_or(|sha| !valid_sha256(sha)) {
            reasons.push(RegimeRoutingReasonV1::FeatureTransformDigestMissing);
            unknown = true;
        }
        if feature.subject_context_id.trim().is_empty() {
            reasons.push(RegimeRoutingReasonV1::FeatureContextIdentityMissing);
            unknown = true;
        }
    }

    reasons.sort();
    reasons.dedup();
    let state = if input.purpose == RegimeRoutingPurposeV1::RetrospectiveOracle
        && !rejected
        && !unknown
    {
        RegimeRoutingStateV1::RetrospectiveOnly
    } else if rejected {
        RegimeRoutingStateV1::Rejected
    } else if unknown {
        RegimeRoutingStateV1::Unknown
    } else if input.purpose == RegimeRoutingPurposeV1::ObservableContext {
        RegimeRoutingStateV1::Eligible
    } else {
        RegimeRoutingStateV1::RetrospectiveOnly
    };

    let canonical = canonical_input(input);
    let input_sha256 = json_sha256(&canonical)?;
    let mut receipt = RegimeRoutingReceiptV1 {
        schema: REGIME_ROUTING_RECEIPT_SCHEMA_V1.into(),
        decision_id: input.decision_id.clone(),
        purpose: input.purpose,
        state,
        route_id: input.route_id.clone(),
        route_score: input.route_score,
        decision_at_unix_seconds: input.decision_at_unix_seconds,
        clock_domain_id: input.clock_domain_id.clone(),
        clock_mapping_sha256: input.clock_mapping_sha256.clone(),
        router_artifact_sha256: input.router_artifact_sha256.clone(),
        training_manifest_sha256: manifest.manifest_sha256.clone(),
        input_sha256,
        consumed_feature_set_sha256: feature_set_digest(&input.consumed_features)?,
        reasons,
        deployable_router_eligible: state == RegimeRoutingStateV1::Eligible,
        retrospective_oracle_only: state == RegimeRoutingStateV1::RetrospectiveOnly,
        router_issuer_authenticity_proven: false,
        training_manifest_issuer_authenticity_proven: false,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    Ok(receipt)
}

/// Verifies the exact route receipt by re-evaluating the original feature set.
pub fn verify_regime_routing_receipt_v1(
    input: &RegimeRoutingInputV1,
    supplied: &RegimeRoutingReceiptV1,
) -> Result<bool, RegimeRoutingGuardErrorV1> {
    if !valid_sha256(&supplied.receipt_sha256)
        || receipt_digest(supplied)? != supplied.receipt_sha256
    {
        return Ok(false);
    }
    let expected = evaluate_regime_routing_v1(input)?;
    Ok(expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn manifest() -> RegimeRouterTrainingManifestV1 {
        let mut value = RegimeRouterTrainingManifestV1 {
            schema: REGIME_ROUTER_TRAINING_MANIFEST_SCHEMA_V1.into(),
            router_artifact_sha256: digest('a'),
            source_dataset_manifest_sha256: digest('b'),
            feature_schema_sha256: digest('c'),
            target_measurand_id: "avg_removal_rate".into(),
            allowed_feature_ids: vec!["stage".into(), "chamber".into(), "pressure_mean".into()],
            forbidden_target_feature_ids: vec!["avg_removal_rate".into(), "avg_removal_rate_bin".into()],
            fit_cutoff_unix_seconds: 90,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('d'),
            manifest_sha256: String::new(),
        };
        value.manifest_sha256 = seal_regime_router_training_manifest_v1(&value).unwrap();
        value
    }

    fn feature(
        id: &str,
        source_kind: RegimeFeatureSourceKindV1,
        event_time: i64,
        available_at: i64,
    ) -> RegimeFeatureObservationV1 {
        RegimeFeatureObservationV1 {
            feature_id: id.into(),
            value: 1.25,
            unit: "declared_unit".into(),
            source_kind,
            source_artifact_sha256: Some(digest('e')),
            transform_sha256: Some(digest('f')),
            event_time_unix_seconds: event_time,
            available_at_unix_seconds: available_at,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('d'),
            feature_schema_sha256: digest('c'),
            subject_context_id: "wafer-001/tool-02/chamber-01".into(),
        }
    }

    fn input(
        purpose: RegimeRoutingPurposeV1,
        consumed_features: Vec<RegimeFeatureObservationV1>,
    ) -> RegimeRoutingInputV1 {
        RegimeRoutingInputV1 {
            schema: REGIME_ROUTING_INPUT_SCHEMA_V1.into(),
            decision_id: "route-decision-001".into(),
            purpose,
            decision_at_unix_seconds: 100,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('d'),
            feature_schema_sha256: digest('c'),
            target_measurand_id: "avg_removal_rate".into(),
            route_id: "mrr-low-mode".into(),
            route_score: 0.90,
            router_artifact_sha256: digest('a'),
            training_manifest: manifest(),
            consumed_features,
        }
    }

    #[test]
    fn observable_stage_chamber_and_process_features_are_eligible() {
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![
                feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90),
                feature("chamber", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90),
                feature("pressure_mean", RegimeFeatureSourceKindV1::SensorObservation, 85, 95),
            ],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Eligible);
        assert!(receipt.deployable_router_eligible);
        assert!(!receipt.retrospective_oracle_only);
        assert!(!receipt.router_issuer_authenticity_proven);
        assert!(verify_regime_routing_receipt_v1(&input, &receipt).unwrap());
    }

    #[test]
    fn direct_target_feature_is_rejected_for_observable_routing() {
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("avg_removal_rate", RegimeFeatureSourceKindV1::TargetDerived, 80, 90)],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(!receipt.deployable_router_eligible);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureTargetDerived));
    }

    #[test]
    fn transformed_or_thresholded_target_feature_is_rejected_by_lineage() {
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature(
                "avg_removal_rate_bin",
                RegimeFeatureSourceKindV1::ObservableProcessContext,
                80,
                90,
            )],
        );
        // The training manifest marks the target-derived bin as forbidden even
        // though its superficial source label says observable context.
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(
            receipt
                .reasons
                .contains(&RegimeRoutingReasonV1::FeatureForbiddenByTrainingManifest)
        );
    }

    #[test]
    fn feature_event_after_decision_is_rejected() {
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 101, 101)],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureEventAfterDecision));
    }

    #[test]
    fn feature_available_after_decision_is_rejected_even_when_event_is_earlier() {
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 101)],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureNotAvailableAtDecision));
    }

    #[test]
    fn clock_domain_mismatch_returns_unknown() {
        let mut feature = feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90);
        feature.clock_domain_id = "other-clock".into();
        let receipt = evaluate_regime_routing_v1(&input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature],
        ))
        .unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Unknown);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureClockDomainMismatch));
    }

    #[test]
    fn clock_mapping_mismatch_returns_unknown() {
        let mut feature = feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90);
        feature.clock_mapping_sha256 = digest('8');
        let receipt = evaluate_regime_routing_v1(&input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature],
        ))
        .unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Unknown);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureClockMappingMismatch));
    }

    #[test]
    fn missing_source_transform_digest_or_context_returns_unknown() {
        let mut observation = feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90);
        observation.source_artifact_sha256 = None;
        observation.transform_sha256 = None;
        observation.subject_context_id.clear();
        let receipt = evaluate_regime_routing_v1(&input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![observation],
        ))
        .unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Unknown);
    }

    #[test]
    fn target_conditioned_two_mode_oracle_is_retrospective_only() {
        let input = input(
            RegimeRoutingPurposeV1::RetrospectiveOracle,
            vec![
                feature("avg_removal_rate", RegimeFeatureSourceKindV1::TargetDerived, 101, 102),
            ],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::RetrospectiveOnly);
        assert!(receipt.retrospective_oracle_only);
        assert!(!receipt.deployable_router_eligible);
        assert!(
            receipt
                .reasons
                .contains(&RegimeRoutingReasonV1::RetrospectiveTargetFeatureUsed)
        );
    }

    #[test]
    fn retrospective_target_value_is_stratification_only_not_a_model_input() {
        let input = input(
            RegimeRoutingPurposeV1::RetrospectiveOracle,
            vec![
                feature(
                    "avg_removal_rate_bin",
                    RegimeFeatureSourceKindV1::TargetDerived,
                    101,
                    102,
                ),
                feature(
                    "stage",
                    RegimeFeatureSourceKindV1::ObservableProcessContext,
                    80,
                    90,
                ),
            ],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::RetrospectiveOnly);
        assert!(receipt.retrospective_oracle_only);
        assert!(!receipt.deployable_router_eligible);
        assert!(!receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureNotInTrainingAllowlist));
        assert!(!receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureForbiddenByTrainingManifest));
    }

    #[test]
    fn target_plus_stage_three_mode_oracle_is_retrospective_only() {
        let input = input(
            RegimeRoutingPurposeV1::RetrospectiveOracle,
            vec![
                feature("avg_removal_rate", RegimeFeatureSourceKindV1::TargetDerived, 101, 102),
                feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90),
            ],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::RetrospectiveOnly);
        assert!(!receipt.deployable_router_eligible);
    }

    #[test]
    fn unknown_feature_source_returns_unknown() {
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::Unknown, 80, 90)],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Unknown);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureSourceUnknown));
    }

    #[test]
    fn training_manifest_cannot_allow_the_target_as_a_feature() {
        let mut input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90)],
        );
        input.training_manifest.allowed_feature_ids.push("avg_removal_rate".into());
        input.training_manifest.manifest_sha256 =
            seal_regime_router_training_manifest_v1(&input.training_manifest).unwrap();
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::TrainingManifestContainsTargetFeature));
    }

    #[test]
    fn future_router_fit_cutoff_is_rejected() {
        let mut input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90)],
        );
        input.training_manifest.fit_cutoff_unix_seconds = 101;
        input.training_manifest.manifest_sha256 =
            seal_regime_router_training_manifest_v1(&input.training_manifest).unwrap();
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::TrainingCutoffAfterDecision));
    }

    #[test]
    fn duplicate_feature_ids_are_rejected() {
        let feature = feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90);
        let input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature.clone(), feature],
        );
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::DuplicateFeatureId));
    }

    #[test]
    fn feature_order_does_not_change_receipt_digest() {
        let left = feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90);
        let right = feature("chamber", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90);
        let first = input(RegimeRoutingPurposeV1::ObservableContext, vec![left.clone(), right.clone()]);
        let second = input(RegimeRoutingPurposeV1::ObservableContext, vec![right, left]);
        let a = evaluate_regime_routing_v1(&first).unwrap();
        let b = evaluate_regime_routing_v1(&second).unwrap();
        assert_eq!(a.receipt_sha256, b.receipt_sha256);
    }

    #[test]
    fn changed_post_decision_feature_changes_historical_receipt_and_is_rejected() {
        let first = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90)],
        );
        let a = evaluate_regime_routing_v1(&first).unwrap();
        let mut changed = first.clone();
        changed.consumed_features.push(feature(
            "pressure_mean",
            RegimeFeatureSourceKindV1::PostDecision,
            101,
            102,
        ));
        let b = evaluate_regime_routing_v1(&changed).unwrap();
        assert_ne!(a.receipt_sha256, b.receipt_sha256);
        assert_eq!(b.state, RegimeRoutingStateV1::Rejected);
    }

    #[test]
    fn input_and_training_manifest_digest_tampering_is_rejected() {
        let mut input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90)],
        );
        input.training_manifest.router_artifact_sha256 = digest('9');
        let receipt = evaluate_regime_routing_v1(&input).unwrap();
        assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
        assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::TrainingManifestDigestMismatch)
            || receipt.reasons.contains(&RegimeRoutingReasonV1::RouterArtifactMismatch));
    }

    #[test]
    fn invalid_route_score_is_rejected_before_receipt() {
        let mut input = input(
            RegimeRoutingPurposeV1::ObservableContext,
            vec![feature("stage", RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90)],
        );
        input.route_score = 1.5;
        assert!(evaluate_regime_routing_v1(&input).is_err());
    }
}

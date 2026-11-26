use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    REGIME_ROUTER_TRAINING_MANIFEST_SCHEMA_V1, REGIME_ROUTING_INPUT_SCHEMA_V1,
    RegimeFeatureObservationV1, RegimeFeatureSourceKindV1, RegimeRouterTrainingManifestV1,
    RegimeRoutingInputV1, RegimeRoutingPurposeV1, RegimeRoutingReasonV1,
    RegimeRoutingStateV1, evaluate_regime_routing_v1, seal_regime_router_training_manifest_v1,
    verify_regime_routing_receipt_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_digest<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn manifest() -> RegimeRouterTrainingManifestV1 {
    let mut value = RegimeRouterTrainingManifestV1 {
        schema: REGIME_ROUTER_TRAINING_MANIFEST_SCHEMA_V1.into(),
        router_artifact_sha256: digest('a'),
        source_dataset_manifest_sha256: digest('b'),
        feature_schema_sha256: digest('c'),
        target_measurand_id: "avg_removal_rate".into(),
        allowed_feature_ids: vec!["stage".into(), "chamber".into(), "pressure_mean".into()],
        forbidden_target_feature_ids: vec![
            "avg_removal_rate".into(),
            "avg_removal_rate_bin".into(),
            "mrr_threshold_class".into(),
        ],
        fit_cutoff_unix_seconds: 90,
        clock_domain_id: "fab-clock".into(),
        clock_mapping_sha256: digest('d'),
        manifest_sha256: String::new(),
    };
    value.manifest_sha256 = independent_manifest_digest(&value);
    assert_eq!(
        value.manifest_sha256,
        seal_regime_router_training_manifest_v1(&value).unwrap()
    );
    value
}

fn independent_manifest_digest(
    manifest: &RegimeRouterTrainingManifestV1,
) -> String {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    canonical.allowed_feature_ids.sort();
    canonical.forbidden_target_feature_ids.sort();
    json_digest(&canonical)
}

fn feature(
    id: &str,
    value: f64,
    source_kind: RegimeFeatureSourceKindV1,
    event_time: i64,
    available_at: i64,
) -> RegimeFeatureObservationV1 {
    RegimeFeatureObservationV1 {
        feature_id: id.into(),
        value,
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
    features: Vec<RegimeFeatureObservationV1>,
) -> RegimeRoutingInputV1 {
    RegimeRoutingInputV1 {
        schema: REGIME_ROUTING_INPUT_SCHEMA_V1.into(),
        decision_id: "cmp-routing-known-answer.v1".into(),
        purpose,
        decision_at_unix_seconds: 100,
        clock_domain_id: "fab-clock".into(),
        clock_mapping_sha256: digest('d'),
        feature_schema_sha256: digest('c'),
        target_measurand_id: "avg_removal_rate".into(),
        route_id: "low-removal-context-expert".into(),
        route_score: 0.9,
        router_artifact_sha256: digest('a'),
        training_manifest: manifest(),
        consumed_features: features,
    }
}

#[test]
fn independent_observable_context_decision_is_eligible_and_replayable() {
    let input = input(
        RegimeRoutingPurposeV1::ObservableContext,
        vec![
            feature("stage", 1.0, RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90),
            feature("chamber", 2.0, RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90),
            feature("pressure_mean", 40.0, RegimeFeatureSourceKindV1::SensorObservation, 85, 95),
        ],
    );
    let receipt = evaluate_regime_routing_v1(&input).unwrap();
    assert_eq!(receipt.state, RegimeRoutingStateV1::Eligible);
    assert!(receipt.deployable_router_eligible);
    assert!(verify_regime_routing_receipt_v1(&input, &receipt).unwrap());

    let mut tampered = receipt;
    tampered.route_id = "different-expert".into();
    assert!(!verify_regime_routing_receipt_v1(&input, &tampered).unwrap());
}

#[test]
fn target_conditioned_mrr_mode_is_retrospective_not_deployable() {
    let input = input(
        RegimeRoutingPurposeV1::RetrospectiveOracle,
        vec![
            feature("avg_removal_rate_bin", 0.0, RegimeFeatureSourceKindV1::TargetDerived, 101, 102),
            feature("stage", 1.0, RegimeFeatureSourceKindV1::ObservableProcessContext, 80, 90),
        ],
    );
    let receipt = evaluate_regime_routing_v1(&input).unwrap();
    assert_eq!(receipt.state, RegimeRoutingStateV1::RetrospectiveOnly);
    assert!(receipt.retrospective_oracle_only);
    assert!(!receipt.deployable_router_eligible);
    assert!(receipt.reasons.contains(&RegimeRoutingReasonV1::RetrospectiveTargetFeatureUsed));
}

#[test]
fn target_conditioned_feature_cannot_enter_deployable_router_even_when_manifest_is_resealed() {
    let input = input(
        RegimeRoutingPurposeV1::ObservableContext,
        vec![
            feature("avg_removal_rate_bin", 1.0, RegimeFeatureSourceKindV1::TargetDerived, 80, 90),
        ],
    );
    let receipt = evaluate_regime_routing_v1(&input).unwrap();
    assert_eq!(receipt.state, RegimeRoutingStateV1::Rejected);
    assert!(!receipt.deployable_router_eligible);
    assert!(
        receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureForbiddenByTrainingManifest)
            || receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureNotInTrainingAllowlist)
            || receipt.reasons.contains(&RegimeRoutingReasonV1::FeatureTargetDerived)
    );
}

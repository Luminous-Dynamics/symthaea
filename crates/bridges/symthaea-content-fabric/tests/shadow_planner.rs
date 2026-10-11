use std::collections::BTreeSet;

use symthaea_content_fabric::{
    plan_hdc_shadow_v1, recommend_json_v1, recompute_profile_id_v1, recompute_request_id_v1,
    validate_request_v1, ActionRefV1, AgentRefV1, ContentDigestV1, DigestAlgorithmV1,
    ExternalAcceptedFailureDomainV1, ExternalFailureDomainRequirementV1,
    ExternalPlannerCandidateV1, ExternalPlannerPreferencesV1, ExternalPlannerProfileV1,
    ExternalPlannerRecommendationV1, ExternalPlannerRequestIdV1, ExternalPlannerRequestV1,
    FailureDomainKindV1, ObjectIdV1, PlacementProposalIdV1, PlacementTargetV1,
    PlannerInputIdV1, PlannerProfileIdV1, ProtocolErrorV1, SoftEvidenceStateV1,
    StorageIntentIdV1, EXTERNAL_PLANNER_PROTOCOL_V1,
};
use symthaea_core::hdc::unified_hv::set_cognitive_stride;

fn action(byte: u8) -> ActionRefV1 {
    ActionRefV1(vec![byte; 39])
}

fn agent(byte: u8) -> AgentRefV1 {
    AgentRefV1(vec![byte; 39])
}

fn candidate(
    rank: u64,
    action_byte: u8,
    site: &str,
    cost_penalty_ppm: u32,
) -> ExternalPlannerCandidateV1 {
    ExternalPlannerCandidateV1 {
        availability_action: action(action_byte),
        advertisement_action: action(action_byte.saturating_add(20)),
        provider: agent(action_byte.saturating_add(40)),
        baseline_rank: rank,
        baseline_weighted_penalty: u64::from(cost_penalty_ppm) * 100,
        cost_penalty_ppm,
        latency_penalty_ppm: 0,
        energy_penalty_ppm: 0,
        locality_penalty_ppm: 0,
        evidence_state: SoftEvidenceStateV1::Fresh,
        missing_weighted_metrics: Vec::new(),
        accepted_failure_domains: vec![ExternalAcceptedFailureDomainV1 {
            kind: FailureDomainKindV1::Site,
            value: site.to_string(),
        }],
    }
}

fn fixture() -> ExternalPlannerRequestV1 {
    let candidates = vec![
        candidate(0, 1, "site-a", 0),
        candidate(1, 2, "site-a", 100_000),
        candidate(2, 3, "site-b", 900_000),
    ];
    let mut request = ExternalPlannerRequestV1 {
        schema_version: EXTERNAL_PLANNER_PROTOCOL_V1,
        id: ExternalPlannerRequestIdV1([0; 32]),
        planner_input_id: PlannerInputIdV1([7; 32]),
        storage_intent_id: StorageIntentIdV1([8; 32]),
        target: PlacementTargetV1 {
            object_id: ObjectIdV1([9; 32]),
            digest: ContentDigestV1 {
                algorithm: DigestAlgorithmV1::Blake3_256,
                bytes: [10; 32],
            },
            size_bytes: 4096,
            client_side_encrypted: false,
        },
        evaluated_at_unix_ms: 1_800_000_000_000,
        profile: ExternalPlannerProfileV1 {
            id: PlannerProfileIdV1([0; 32]),
            cost_ceiling_microunits_per_gib: 10_000,
            latency_ceiling_ms: 1_000,
            energy_ceiling_millijoules_per_gib: 100_000,
            locality_ceiling_km: 20_000,
        },
        preferences: ExternalPlannerPreferencesV1 {
            target_latency_ms: Some(100),
            cost_weight: 100,
            latency_weight: 0,
            energy_weight: 0,
            locality_weight: 0,
        },
        minimum_replicas: 2,
        failure_domain_requirements: vec![ExternalFailureDomainRequirementV1 {
            kind: FailureDomainKindV1::Site,
            minimum_distinct: 2,
        }],
        baseline_proposal_id: PlacementProposalIdV1([12; 32]),
        baseline_selected_availability_actions: vec![action(1), action(3)],
        candidates,
    };
    request.profile.id = recompute_profile_id_v1(&request.profile);
    request.id = recompute_request_id_v1(&request);
    request
}

#[test]
fn golden_request_vector_matches_mycelix_cf06c() {
    let json = include_str!("fixtures/external_planner_request_v1.json");
    let request: ExternalPlannerRequestV1 = serde_json::from_str(json).expect("decode golden request");
    let expected_profile = PlannerProfileIdV1([
        0xcb, 0x4f, 0x6c, 0x8e, 0xba, 0x88, 0xb5, 0x50, 0xd9, 0x5f, 0x6f, 0x7f, 0xdd,
        0x78, 0x66, 0x2a, 0x0c, 0xa1, 0xf1, 0x76, 0xea, 0x17, 0xfa, 0xd3, 0x62, 0x72,
        0xe0, 0x32, 0x0b, 0x7c, 0x7a, 0x86,
    ]);
    let expected_request = ExternalPlannerRequestIdV1([
        0xd3, 0xf8, 0xe7, 0xf6, 0x4b, 0x1e, 0xae, 0xe2, 0x63, 0xbd, 0x7c, 0x1f, 0x05,
        0xd9, 0x90, 0xb2, 0xeb, 0x43, 0xe8, 0x84, 0x73, 0x73, 0x31, 0x39, 0xac, 0x6a,
        0x2a, 0xc9, 0x72, 0xba, 0xf4, 0x54,
    ]);
    assert_eq!(request.profile.id, expected_profile);
    assert_eq!(recompute_profile_id_v1(&request.profile), expected_profile);
    assert_eq!(request.id, expected_request);
    assert_eq!(recompute_request_id_v1(&request), expected_request);
    validate_request_v1(&request).expect("golden request is structurally valid");
}

#[test]
fn nested_profile_commitment_survives_outer_request_rehash() {
    let mut request = fixture();
    request.profile.cost_ceiling_microunits_per_gib += 1;
    request.id = recompute_request_id_v1(&request);
    assert_eq!(
        validate_request_v1(&request),
        Err(ProtocolErrorV1::ProfileIdMismatch)
    );
}

#[test]
fn request_commitment_detects_mutation() {
    let mut request = fixture();
    validate_request_v1(&request).expect("fixture is valid");
    request.candidates[0].cost_penalty_ppm = 1;
    assert_eq!(
        validate_request_v1(&request),
        Err(ProtocolErrorV1::RequestIdMismatch)
    );
}

#[test]
fn hdc_shadow_preserves_required_site_diversity() {
    let request = fixture();
    let plan = plan_hdc_shadow_v1(&request).expect("plan succeeds");
    assert_eq!(plan.recommendation.ranking[0], action(1));
    assert_eq!(plan.recommendation.ranking[1], action(2));
    let selected = plan
        .recommendation
        .selected_availability_actions
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(selected.len(), 2);
    assert!(selected.contains(&action(1)));
    assert!(selected.contains(&action(3)));
}

#[test]
fn zero_preferences_preserve_baseline_exactly() {
    let mut request = fixture();
    request.preferences.cost_weight = 0;
    request.id = recompute_request_id_v1(&request);
    let plan = plan_hdc_shadow_v1(&request).expect("plan succeeds");
    assert_eq!(
        plan.recommendation.ranking,
        request
            .candidates
            .iter()
            .map(|candidate| candidate.availability_action.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        plan.recommendation.selected_availability_actions,
        request.baseline_selected_availability_actions
    );
    assert!(!plan.trace.ranking_changed);
    assert!(!plan.trace.selection_changed);
}

#[test]
fn output_is_independent_of_global_cognitive_stride() {
    let request = fixture();
    set_cognitive_stride(1);
    let full = plan_hdc_shadow_v1(&request).expect("full-stride plan");
    set_cognitive_stride(31);
    let throttled = plan_hdc_shadow_v1(&request).expect("throttled plan");
    set_cognitive_stride(4);
    assert_eq!(full, throttled);
}

#[test]
fn repeated_planning_is_deterministic() {
    let request = fixture();
    let first = plan_hdc_shadow_v1(&request).expect("first plan");
    let second = plan_hdc_shadow_v1(&request).expect("second plan");
    assert_eq!(first, second);
}

#[test]
fn recommendation_is_complete_and_json_compatible() {
    let request = fixture();
    let request_json = serde_json::to_string(&request).expect("serialize request");
    let recommendation_json = recommend_json_v1(&request_json).expect("recommendation JSON");
    let recommendation: ExternalPlannerRecommendationV1 =
        serde_json::from_str(&recommendation_json).expect("decode recommendation");
    assert_eq!(recommendation.request_id, request.id);
    assert_eq!(recommendation.planner_input_id, request.planner_input_id);
    assert_eq!(recommendation.ranking.len(), request.candidates.len());
    let ranked = recommendation.ranking.into_iter().collect::<BTreeSet<_>>();
    let expected = request
        .candidates
        .into_iter()
        .map(|candidate| candidate.availability_action)
        .collect::<BTreeSet<_>>();
    assert_eq!(ranked, expected);
}

#[test]
fn malformed_opaque_reference_fails_closed() {
    let mut request = fixture();
    request.candidates[0].availability_action = ActionRefV1(vec![1; 38]);
    request.id = recompute_request_id_v1(&request);
    assert_eq!(
        validate_request_v1(&request),
        Err(ProtocolErrorV1::MalformedOpaqueReference)
    );
}

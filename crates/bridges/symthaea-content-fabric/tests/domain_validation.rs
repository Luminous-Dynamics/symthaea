use symthaea_content_fabric::{
    recompute_request_id_v1, validate_request_v1, ExternalAcceptedFailureDomainV1,
    ExternalFailureDomainRequirementV1, ExternalPlannerRequestV1, FailureDomainKindV1,
    ProtocolErrorV1,
};

#[test]
fn non_canonical_failure_domain_value_fails_closed() {
    let json = include_str!("fixtures/external_planner_request_v1.json");
    let mut request: ExternalPlannerRequestV1 =
        serde_json::from_str(json).expect("decode golden request");
    request.failure_domain_requirements = vec![ExternalFailureDomainRequirementV1 {
        kind: FailureDomainKindV1::Site,
        minimum_distinct: 1,
    }];
    request.candidates[0].accepted_failure_domains = vec![ExternalAcceptedFailureDomainV1 {
        kind: FailureDomainKindV1::Site,
        value: "SITE-A".to_string(),
    }];
    request.id = recompute_request_id_v1(&request);

    assert_eq!(
        validate_request_v1(&request),
        Err(ProtocolErrorV1::InvalidCandidateFailureDomains)
    );
}

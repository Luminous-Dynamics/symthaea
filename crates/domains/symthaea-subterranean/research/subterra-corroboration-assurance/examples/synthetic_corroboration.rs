use subterra_corroboration_assurance::{
    CandidateCorroborationPortfolio, CommonCauseDisposition, CorroborationPolicy,
    DependencyIdentity, ExpectedCorroborationContext, EvidenceSource, FaultSemanticsDisposition,
    InventoryDisposition, PortfolioAssuranceAssessment, SourceDependencyClosure,
    VerifierDependencyClosure, VerifierSeparationDisposition, admit_corroboration_portfolio,
};
use subterra_inspection_evidence_gate::{
    CandidateInspectionOption, EvidenceManifest, EvidencePurpose, ExpectedInspectionContext,
    ReceiptVerificationDisposition, RightsDisposition, SpatialScopeDisposition, admit_candidate,
};
use subterra_inspection_campaign::{InspectionOption, ResourceVector};

fn dependency(dimension: &str, identity: &str) -> DependencyIdentity {
    DependencyIdentity { dimension: dimension.into(), identity: identity.into() }
}

fn main() {
    let now = 1_800_000_000_000_u64;
    let expected_upstream = ExpectedInspectionContext {
        decision_id: "decision-7".into(),
        option_id: "inspect-7".into(),
        asset_id: "asset-7".into(),
        asset_revision: "rev-4".into(),
        coordinate_frame_digest: "sha256:frame".into(),
        region_scope_digest: "sha256:region".into(),
        sensor_model_digest: "sha256:sensor".into(),
        likelihood_model_digest: "sha256:likelihood".into(),
        loss_table_digest: "sha256:loss".into(),
        evsi_assessment_digest: "sha256:evsi".into(),
        source_dataset_digest: "sha256:dataset-a".into(),
        deployment_bundle_id: "corridor-1".into(),
        deployment_bundle_digest: "sha256:bundle".into(),
        resource_model_digest: "sha256:resource-model".into(),
        resource_assessment_digest: "sha256:resource".into(),
        risk_model_digest: "sha256:risk-model".into(),
        risk_assessment_digest: "sha256:risk".into(),
        purpose: EvidencePurpose::AssetConditionInspection,
        evaluation_time_unix_ms: now,
        maximum_observation_age_ms: 3_600_000,
        allowed_future_clock_skew_ms: 5_000,
    };
    let upstream = CandidateInspectionOption {
        option: InspectionOption {
            option_id: "inspect-7".into(),
            decision_id: "decision-7".into(),
            deployment_bundle_id: "corridor-1".into(),
            gross_evsi_micro_units: 100,
            unknown_state_debt_reduction_units: 1,
            variable_resources: ResourceVector {
                energy_millijoules: 5,
                duration_milliseconds: 10,
                data_bytes: 20,
                exposure_milliunits: 1,
            },
            mission_interruption_upper_ppm: 1,
            robot_loss_upper_ppm: 1,
            external_gate_passed: true,
        },
        manifest: EvidenceManifest {
            candidate_id: "candidate-7".into(),
            decision_id: "decision-7".into(),
            option_id: "inspect-7".into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            sensor_model_digest: "sha256:sensor".into(),
            likelihood_model_digest: "sha256:likelihood".into(),
            loss_table_digest: "sha256:loss".into(),
            evsi_assessment_digest: "sha256:evsi".into(),
            source_dataset_digest: "sha256:dataset-a".into(),
            deployment_bundle_id: "corridor-1".into(),
            deployment_bundle_digest: "sha256:bundle".into(),
            resource_model_digest: "sha256:resource-model".into(),
            resource_assessment_digest: "sha256:resource".into(),
            risk_model_digest: "sha256:risk-model".into(),
            risk_assessment_digest: "sha256:risk".into(),
            assessed_resources: ResourceVector {
                energy_millijoules: 5,
                duration_milliseconds: 10,
                data_bytes: 20,
                exposure_milliunits: 1,
            },
            assessed_mission_interruption_upper_ppm: 1,
            assessed_robot_loss_upper_ppm: 1,
            resource_assessment_valid_until_unix_ms: now + 10_000,
            risk_assessment_valid_until_unix_ms: now + 10_000,
            observation_receipt_digest: "sha256:obs-a".into(),
            calibration_receipt_digest: "sha256:cal-a".into(),
            rights_receipt_digest: "sha256:rights".into(),
            verifier_receipt_digest: "sha256:upstream-verifier".into(),
            observation_captured_at_unix_ms: now - 1_000,
            evidence_valid_until_unix_ms: now + 10_000,
            calibration_valid_from_unix_ms: now - 10_000,
            calibration_valid_until_unix_ms: now + 10_000,
            rights_valid_until_unix_ms: now + 10_000,
            rights_purpose: EvidencePurpose::AssetConditionInspection,
            rights_disposition: RightsDisposition::AuthorizedForDeclaredPurpose,
            receipt_verification: ReceiptVerificationDisposition::VerifiedByExternalService,
            spatial_scope: SpatialScopeDisposition::CompleteWithinDeclaredScope,
        },
    };
    let anchor = admit_candidate(&upstream, &expected_upstream)
        .admitted_candidate.expect("synthetic upstream option admitted");

    let sources = vec![
        EvidenceSource {
            source_id: "robot-a-thermal".into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            source_dataset_digest: "sha256:dataset-a".into(),
            observation_receipt_digest: "sha256:obs-a".into(),
            calibration_receipt_digest: "sha256:cal-a".into(),
            captured_at_unix_ms: now - 1_000,
            valid_until_unix_ms: now + 10_000,
        },
        EvidenceSource {
            source_id: "robot-b-lidar".into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            source_dataset_digest: "sha256:dataset-b".into(),
            observation_receipt_digest: "sha256:obs-b".into(),
            calibration_receipt_digest: "sha256:cal-b".into(),
            captured_at_unix_ms: now - 2_000,
            valid_until_unix_ms: now + 10_000,
        },
    ];
    let assessment = PortfolioAssuranceAssessment {
        claim_id: "claim-7".into(),
        asset_id: "asset-7".into(),
        asset_revision: "rev-4".into(),
        coordinate_frame_digest: "sha256:frame".into(),
        region_scope_digest: "sha256:region".into(),
        evidence_set_digest: "sha256:evidence-set".into(),
        dependency_graph_digest: "sha256:graph".into(),
        fault_domain_profile_digest: "sha256:profile".into(),
        assessment_digest: "sha256:assessment".into(),
        verifier_receipt_digest: "sha256:portfolio-verifier".into(),
        evaluated_at_unix_ms: now - 100,
        valid_until_unix_ms: now + 10_000,
        inventory: InventoryDisposition::CompleteWithinDeclaredScope,
        common_cause: CommonCauseDisposition::NoSharedClaimCriticalCauseWithinDeclaredScope,
        verifier_separation: VerifierSeparationDisposition::IndependentWithinDeclaredScope,
        fault_semantics: FaultSemanticsDisposition::FullyRepresentedByDeclaredModel,
        sources: vec![
            SourceDependencyClosure {
                source_id: "robot-a-thermal".into(),
                inventory: InventoryDisposition::CompleteWithinDeclaredScope,
                claim_critical_dependencies: vec![
                    dependency("robot", "robot-a"),
                    dependency("power", "power-a"),
                    dependency("clock", "clock-a"),
                    dependency("network", "network-a"),
                ],
            },
            SourceDependencyClosure {
                source_id: "robot-b-lidar".into(),
                inventory: InventoryDisposition::CompleteWithinDeclaredScope,
                claim_critical_dependencies: vec![
                    dependency("robot", "robot-b"),
                    dependency("power", "power-b"),
                    dependency("clock", "clock-b"),
                    dependency("network", "network-b"),
                ],
            },
        ],
        verifier: VerifierDependencyClosure {
            verifier_id: "verifier-7".into(),
            inventory: InventoryDisposition::CompleteWithinDeclaredScope,
            claim_critical_dependencies: vec![
                dependency("host", "verifier-host"),
                dependency("clock", "verifier-clock"),
                dependency("ground-truth", "sealed-independent-truth"),
            ],
        },
    };
    let expected = ExpectedCorroborationContext {
        claim_id: "claim-7".into(),
        option_id: "inspect-7".into(),
        asset_id: "asset-7".into(),
        asset_revision: "rev-4".into(),
        coordinate_frame_digest: "sha256:frame".into(),
        region_scope_digest: "sha256:region".into(),
        evidence_set_digest: "sha256:evidence-set".into(),
        dependency_graph_digest: "sha256:graph".into(),
        fault_domain_profile_digest: "sha256:profile".into(),
        assessment_digest: "sha256:assessment".into(),
        verifier_receipt_digest: "sha256:portfolio-verifier".into(),
    };
    let report = admit_corroboration_portfolio(
        &CandidateCorroborationPortfolio { admitted_anchor: anchor, sources, assessment },
        &expected,
        CorroborationPolicy {
            minimum_distinct_sources: 2,
            evaluation_time_unix_ms: now,
            maximum_observation_age_ms: 3_600_000,
            allowed_future_clock_skew_ms: 5_000,
            maximum_assessment_age_ms: 3_600_000,
        },
    );
    println!("option={}", report.admitted.as_ref().map_or("rejected", |p| p.option_id()));
    println!("admitted={}", report.accepted());
    println!("rejections={:?}", report.rejection_reasons);
    println!("independence_claim=within declared complete dependency/fault model only");
    println!("synthetic_assessments=true; Mycelix signatures were not authenticated");
}

use subterra_inspection_evidence_gate::{
    CandidateInspectionOption, EvidencePurpose, ExpectedInspectionContext,
    EvidenceManifest, ReceiptVerificationDisposition, RightsDisposition,
    SpatialScopeDisposition, admit_candidate,
};
use subterra_inspection_campaign::{InspectionOption, ResourceVector};

fn main() {
    let now = 1_800_000_000_000_u64;
    let expected = ExpectedInspectionContext {
        decision_id: "support-08".into(),
        option_id: "support-08-independent-followup".into(),
        asset_id: "asset-support-08".into(),
        asset_revision: "rev-17".into(),
        coordinate_frame_digest: "sha256:frame".into(),
        region_scope_digest: "sha256:region".into(),
        sensor_model_digest: "sha256:sensor".into(),
        likelihood_model_digest: "sha256:likelihood".into(),
        loss_table_digest: "sha256:loss".into(),
        evsi_assessment_digest: "sha256:evsi".into(),
        source_dataset_digest: "sha256:synthetic-source".into(),
        purpose: EvidencePurpose::AssetConditionInspection,
        evaluation_time_unix_ms: now,
        maximum_observation_age_ms: 3_600_000,
        allowed_future_clock_skew_ms: 5_000,
    };
    let candidate = CandidateInspectionOption {
        option: InspectionOption {
            option_id: "support-08-independent-followup".into(),
            decision_id: "support-08".into(),
            deployment_bundle_id: "corridor-west".into(),
            gross_evsi_micro_units: 250_000,
            unknown_state_debt_reduction_units: 20,
            variable_resources: ResourceVector {
                energy_millijoules: 5_000,
                duration_milliseconds: 20_000,
                data_bytes: 1_024,
                exposure_milliunits: 2,
            },
            mission_interruption_upper_ppm: 100,
            robot_loss_upper_ppm: 10,
            external_gate_passed: true,
        },
        manifest: EvidenceManifest {
            candidate_id: "candidate-synthetic-001".into(),
            decision_id: "support-08".into(),
            asset_id: "asset-support-08".into(),
            asset_revision: "rev-17".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            sensor_model_digest: "sha256:sensor".into(),
            likelihood_model_digest: "sha256:likelihood".into(),
            loss_table_digest: "sha256:loss".into(),
            evsi_assessment_digest: "sha256:evsi".into(),
            source_dataset_digest: "sha256:synthetic-source".into(),
            observation_receipt_digest: "sha256:synthetic-observation".into(),
            calibration_receipt_digest: "sha256:synthetic-calibration".into(),
            rights_receipt_digest: "sha256:synthetic-rights".into(),
            verifier_receipt_digest: "sha256:synthetic-verifier".into(),
            observation_captured_at_unix_ms: now - 60_000,
            evidence_valid_until_unix_ms: now + 600_000,
            calibration_valid_until_unix_ms: now + 600_000,
            rights_purpose: EvidencePurpose::AssetConditionInspection,
            rights_disposition: RightsDisposition::AuthorizedForDeclaredPurpose,
            receipt_verification: ReceiptVerificationDisposition::VerifiedByExternalService,
            spatial_scope: SpatialScopeDisposition::CompleteWithinDeclaredScope,
        },
    };

    let report = admit_candidate(&candidate, &expected);
    println!("candidate={}", report.candidate_id);
    println!("admitted={}", report.admitted());
    println!("rejections={:?}", report.rejection_reasons);
    println!(
        "synthetic_receipts_only=true; external authentication and legal rights review were not performed"
    );
}

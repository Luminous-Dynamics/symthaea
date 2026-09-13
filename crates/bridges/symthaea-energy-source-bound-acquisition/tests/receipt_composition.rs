use symthaea_energy_material_campaign::{
    AcquisitionDeclaration, CampaignAdmissionReceipt,
    CAPABILITY_CLASSIFICATION as COMPATIBILITY_CAPABILITY_CLASSIFICATION,
};
use symthaea_energy_material_screening::EvidenceDimension;
use symthaea_energy_native_campaign_admission::{
    AdmittedEnvelopeBinding, NativeCampaignAdmissionReceipt,
    CAPABILITY_CLASSIFICATION as NATIVE_CAPABILITY_CLASSIFICATION,
};
use symthaea_energy_source_bound_acquisition::{
    AcquisitionSourceIdentity, SourceBoundAcquisitionDeclaration,
    SourceBoundAdmissionError, SourceBoundCompatibilityAdmissionReceipt,
    SourceBoundNativeAdmissionReceipt, CAPABILITY_CLASSIFICATION,
};

const CAMPAIGN_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const DOSSIER_SHA: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const CANDIDATE_SHA: &str =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const POLICY_SHA: &str =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const QUERY_SHA: &str =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const ARTIFACT_SHA: &str =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
const RECEIPT_SHA: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";

fn source_bound_declaration() -> SourceBoundAcquisitionDeclaration {
    SourceBoundAcquisitionDeclaration {
        dimension: EvidenceDimension::FunctionalPerformance,
        source: AcquisitionSourceIdentity {
            source_name: "fixture-source".into(),
            source_version: "2026.09".into(),
            source_uri: "https://example.invalid/fixture-source".into(),
        },
        acquisition_query_sha256: QUERY_SHA.into(),
        acquired_artifact_sha256: ARTIFACT_SHA.into(),
        source_receipt_sha256: RECEIPT_SHA.into(),
        reviewer: "fixture-reviewer".into(),
        note: "fixture review note".into(),
    }
}

fn v0_declaration(note: &str) -> AcquisitionDeclaration {
    AcquisitionDeclaration {
        dimension: EvidenceDimension::FunctionalPerformance,
        acquisition_query_sha256: QUERY_SHA.into(),
        acquired_artifact_sha256: ARTIFACT_SHA.into(),
        source_receipt_sha256: RECEIPT_SHA.into(),
        reviewer: "fixture-reviewer".into(),
        note: note.into(),
    }
}

fn compatibility_inner(note: &str) -> CampaignAdmissionReceipt {
    CampaignAdmissionReceipt {
        schema: "symthaea.energy-material.campaign-admission.v0".into(),
        capability_classification: COMPATIBILITY_CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: CAMPAIGN_SHA.into(),
        candidate_bound_dossier_sha256: DOSSIER_SHA.into(),
        candidate_sha256: CANDIDATE_SHA.into(),
        screening_policy_sha256: POLICY_SHA.into(),
        acquisition_declarations: vec![v0_declaration(note)],
        admitted_dimensions: EvidenceDimension::ALL.to_vec(),
    }
}

fn compatibility_outer(note: &str) -> SourceBoundCompatibilityAdmissionReceipt {
    let inner = compatibility_inner(note);
    SourceBoundCompatibilityAdmissionReceipt {
        schema: "symthaea.energy-material.source-bound-compatibility-admission.v1".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: CAMPAIGN_SHA.into(),
        compatibility_admission_sha256: inner.sha256().unwrap(),
        compatibility_admission: inner,
        acquisition_declarations: vec![source_bound_declaration()],
    }
}

fn native_inner(note: &str) -> NativeCampaignAdmissionReceipt {
    NativeCampaignAdmissionReceipt {
        schema: "symthaea.energy-material.native-campaign-admission.v0".into(),
        capability_classification: NATIVE_CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: CAMPAIGN_SHA.into(),
        native_dossier_sha256: DOSSIER_SHA.into(),
        candidate_sha256: CANDIDATE_SHA.into(),
        screening_policy_sha256: POLICY_SHA.into(),
        admitted_envelopes: EvidenceDimension::ALL
            .into_iter()
            .enumerate()
            .map(|(index, dimension)| AdmittedEnvelopeBinding {
                dimension,
                envelope_sha256: format!("{:064x}", index + 2),
            })
            .collect(),
        acquisition_declarations: vec![v0_declaration(note)],
        outcome_disclosure: "fixture outcome disclosure".into(),
    }
}

fn native_outer(note: &str) -> SourceBoundNativeAdmissionReceipt {
    let inner = native_inner(note);
    SourceBoundNativeAdmissionReceipt {
        schema: "symthaea.energy-material.source-bound-native-admission.v1".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: CAMPAIGN_SHA.into(),
        native_admission_sha256: inner.sha256().unwrap(),
        native_admission: inner,
        acquisition_declarations: vec![source_bound_declaration()],
    }
}

#[test]
fn clean_synthetic_receipt_composition_is_structurally_valid() {
    compatibility_outer("fixture review note").validate().unwrap();
    native_outer("fixture review note").validate().unwrap();
}

#[test]
fn compatibility_rejects_cross_wired_inner_declaration_before_digest_trust() {
    let receipt = compatibility_outer("different inner note");
    assert!(matches!(
        receipt.validate(),
        Err(SourceBoundAdmissionError::InnerDeclarationMismatch)
    ));
}

#[test]
fn native_rejects_cross_wired_inner_declaration_before_digest_trust() {
    let receipt = native_outer("different inner note");
    assert!(matches!(
        receipt.validate(),
        Err(SourceBoundAdmissionError::InnerDeclarationMismatch)
    ));
}

#[test]
fn compatibility_rejects_incomplete_inner_dimension_coverage() {
    let mut receipt = compatibility_outer("fixture review note");
    receipt.compatibility_admission.admitted_dimensions.pop();
    receipt.compatibility_admission_sha256 = receipt.compatibility_admission.sha256().unwrap();

    assert!(matches!(
        receipt.validate(),
        Err(SourceBoundAdmissionError::InvalidReceipt(_))
    ));
}

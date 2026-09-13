use std::collections::BTreeMap;

use symthaea_discovery::{
    Candidate, CandidateId, CandidateOrigin, EvidenceKind, EvidenceRef, FidelityLevel,
    ModelProvenance, ObjectiveDirection, Prediction, UncertaintyEstimate,
};
use symthaea_energy_evidence_envelope::wrap_evidence_payload_json;
use symthaea_energy_material_campaign::{
    admit_campaign_result, freeze_campaign_manifest, AcquisitionDeclaration,
    CampaignAdmissionReceipt, CampaignError, EvidenceLanePlan, SourceCommitment,
    Tier1CampaignManifest,
};
use symthaea_energy_material_candidate_version::{
    anchor_candidate, bind_dossier_to_candidate_version, ReceiptVersionAttestation,
};
use symthaea_energy_material_dossier::{
    assemble_dossier, EvidenceContribution, IdentityAssertion, IdentityAssertionBasis,
};
use symthaea_energy_material_screening::{
    EnergyMaterialScreeningPolicy, EvidenceDimension, MetricContract,
};
use symthaea_energy_native_campaign_admission::{
    admit_native_campaign_result, NativeAdmissionError, NativeCampaignAdmissionReceipt,
};
use symthaea_energy_native_dossier::assemble_native_envelope_dossier;

const INTERNAL_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const QUERY_SHA: &str =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const ARTIFACT_SHA: &str =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const WRONG_SHA: &str =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mutation {
    Clean,
    WrongQuery,
    MissingArtifactProvenance,
    WrongReceipt,
    UnexpectedDeclaration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    Setup,
    AcquisitionDeclarationBinding,
    AcquiredArtifactProvenance,
    UnexpectedDeclaration,
    OtherAdmission,
}

impl Mutation {
    fn expected_failure(self) -> Option<FailureClass> {
        match self {
            Self::Clean => None,
            // The compatibility API intentionally combines query and receipt
            // mismatch into AcquisitionDeclarationMismatch. The native API is
            // more specific, but both belong to one semantic binding class.
            Self::WrongQuery | Self::WrongReceipt => {
                Some(FailureClass::AcquisitionDeclarationBinding)
            }
            Self::MissingArtifactProvenance => Some(FailureClass::AcquiredArtifactProvenance),
            Self::UnexpectedDeclaration => Some(FailureClass::UnexpectedDeclaration),
        }
    }
}

fn classify_compatibility(error: &CampaignError) -> FailureClass {
    match error {
        CampaignError::AcquisitionDeclarationMismatch(_) => {
            FailureClass::AcquisitionDeclarationBinding
        }
        CampaignError::AcquiredArtifactNotInProvenance(_) => {
            FailureClass::AcquiredArtifactProvenance
        }
        CampaignError::UnexpectedAcquisitionDeclaration => FailureClass::UnexpectedDeclaration,
        _ => FailureClass::OtherAdmission,
    }
}

fn classify_native(error: &NativeAdmissionError) -> FailureClass {
    match error {
        NativeAdmissionError::AcquisitionQueryMismatch(_)
        | NativeAdmissionError::AcquisitionReceiptMismatch(_) => {
            FailureClass::AcquisitionDeclarationBinding
        }
        NativeAdmissionError::AcquiredArtifactNotInProvenance(_) => {
            FailureClass::AcquiredArtifactProvenance
        }
        NativeAdmissionError::UnexpectedAcquisitionDeclaration => {
            FailureClass::UnexpectedDeclaration
        }
        _ => FailureClass::OtherAdmission,
    }
}

fn code(dimension: EvidenceDimension) -> u8 {
    match dimension {
        EvidenceDimension::FunctionalPerformance => 0,
        EvidenceDimension::ThermodynamicStability => 1,
        EvidenceDimension::CriticalMaterialBurden => 2,
        EvidenceDimension::SupplyResilience => 3,
        EvidenceDimension::HumanEnvironmentalHazard => 4,
        EvidenceDimension::Circularity => 5,
        EvidenceDimension::Manufacturability => 6,
    }
}

fn candidate() -> Candidate {
    Candidate {
        id: CandidateId::new("prospective-parity-candidate").unwrap(),
        kind: "energy_material".into(),
        specification: BTreeMap::from([("formula".into(), "LiFePO4".into())]),
        origin: CandidateOrigin::UserProposed,
    }
}

fn policy() -> EnergyMaterialScreeningPolicy {
    EnergyMaterialScreeningPolicy {
        policy_id: "prospective-admission-parity-v0".into(),
        contracts: EvidenceDimension::ALL
            .into_iter()
            .map(|dimension| MetricContract {
                dimension,
                metric: format!("metric-{}", code(dimension)),
                unit: "score".into(),
                direction: ObjectiveDirection::Minimize,
                minimum_fidelity: FidelityLevel::Surrogate,
                accepted_evidence_kinds: vec![EvidenceKind::Dataset],
            })
            .collect(),
        constraints: vec![],
    }
}

fn manifest() -> Tier1CampaignManifest {
    let lanes = EvidenceDimension::ALL
        .into_iter()
        .map(|dimension| EvidenceLanePlan {
            dimension,
            adapter_name: format!("adapter-{}", code(dimension)),
            adapter_version: "v0".into(),
            expected_model_name: format!("model-{}", code(dimension)),
            expected_model_version: Some("v0".into()),
            method_parameters: BTreeMap::new(),
            source_commitment: if dimension == EvidenceDimension::FunctionalPerformance {
                SourceCommitment::ProspectiveAcquisition {
                    source_name: "fixture-source".into(),
                    source_version: "2026-09".into(),
                    source_uri: "https://example.invalid/prospective".into(),
                    acquisition_query_sha256: QUERY_SHA.into(),
                }
            } else {
                SourceCommitment::InternalLineage {
                    sha256: INTERNAL_SHA.into(),
                }
            },
            required_evidence_kinds: vec![EvidenceKind::Dataset],
        })
        .collect();

    freeze_campaign_manifest(
        "prospective-admission-parity-v0",
        anchor_candidate(candidate()).unwrap(),
        policy(),
        lanes,
        None,
        vec![],
    )
    .unwrap()
}

fn prediction(dimension: EvidenceDimension, mutation: Mutation) -> Prediction {
    let prospective = dimension == EvidenceDimension::FunctionalPerformance;
    let digest = if prospective {
        if matches!(mutation, Mutation::MissingArtifactProvenance) {
            WRONG_SHA
        } else {
            ARTIFACT_SHA
        }
    } else {
        INTERNAL_SHA
    };

    Prediction {
        metric: format!("metric-{}", code(dimension)),
        value: 1.0 + f64::from(code(dimension)),
        unit: "score".into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0).unwrap(),
        fidelity: FidelityLevel::Surrogate,
        model: ModelProvenance {
            name: format!("model-{}", code(dimension)),
            version: Some("v0".into()),
            implementation_digest: None,
            input_digest: Some(format!("sha256:{digest}")),
            output_digest: None,
        },
        assumptions: vec![],
        evidence: vec![EvidenceRef {
            id: format!("evidence-{}", code(dimension)),
            kind: EvidenceKind::Dataset,
            uri: None,
            digest: Some(format!("sha256:{digest}")),
            note: None,
        }],
    }
}

fn compatibility_receipt_sha(dimension: EvidenceDimension) -> String {
    let nibble = char::from(b'1' + code(dimension));
    std::iter::repeat(nibble).take(64).collect()
}

fn declaration(source_receipt_sha256: String, mutation: Mutation) -> AcquisitionDeclaration {
    AcquisitionDeclaration {
        dimension: EvidenceDimension::FunctionalPerformance,
        acquisition_query_sha256: if matches!(mutation, Mutation::WrongQuery) {
            WRONG_SHA.into()
        } else {
            QUERY_SHA.into()
        },
        acquired_artifact_sha256: ARTIFACT_SHA.into(),
        source_receipt_sha256: if matches!(mutation, Mutation::WrongReceipt) {
            WRONG_SHA.into()
        } else {
            source_receipt_sha256
        },
        reviewer: "prospective-parity-fixture".into(),
        note: "Fixture declaration for compatibility/native acquisition parity.".into(),
    }
}

fn unexpected_declaration() -> AcquisitionDeclaration {
    AcquisitionDeclaration {
        dimension: EvidenceDimension::ThermodynamicStability,
        acquisition_query_sha256: QUERY_SHA.into(),
        acquired_artifact_sha256: ARTIFACT_SHA.into(),
        source_receipt_sha256: WRONG_SHA.into(),
        reviewer: "prospective-parity-fixture".into(),
        note: "This declaration is intentionally unexpected for a non-prospective lane.".into(),
    }
}

fn run_compatibility(
    mutation: Mutation,
) -> Result<CampaignAdmissionReceipt, FailureClass> {
    let manifest = manifest();
    let candidate = candidate();
    let candidate_sha = manifest.candidate_anchor.candidate_sha256.clone();
    let mut assertions = Vec::new();
    let mut contributions = Vec::new();
    let mut attestations = Vec::new();

    for dimension in EvidenceDimension::ALL {
        let receipt_sha = compatibility_receipt_sha(dimension);
        let assertion_id = format!("compat-identity-{}", code(dimension));
        assertions.push(IdentityAssertion {
            assertion_id: assertion_id.clone(),
            candidate_id: candidate.id.clone(),
            namespace: "symthaea".into(),
            subject_id: candidate.id.0.clone(),
            basis: IdentityAssertionBasis::InternalCandidate,
            source_receipt_sha256: receipt_sha.clone(),
        });
        contributions.push(EvidenceContribution {
            dimension,
            candidate_id: candidate.id.clone(),
            identity_assertion_id: assertion_id,
            source_receipt_sha256: receipt_sha.clone(),
            prediction: prediction(dimension, mutation),
        });
        attestations.push(ReceiptVersionAttestation {
            attestation_id: format!("compat-attestation-{}", code(dimension)),
            dimension,
            source_receipt_sha256: receipt_sha,
            candidate_sha256: candidate_sha.clone(),
            reviewer: "prospective-parity-fixture".into(),
            review_note: "Fixture binds compatibility evidence to the candidate version.".into(),
        });
    }

    let dossier = assemble_dossier(
        candidate.id.clone(),
        &manifest.screening_policy,
        assertions,
        contributions,
    )
    .map_err(|_| FailureClass::Setup)?;
    let bound = bind_dossier_to_candidate_version(candidate, dossier, attestations)
        .map_err(|_| FailureClass::Setup)?;

    let mut declarations = vec![declaration(
        compatibility_receipt_sha(EvidenceDimension::FunctionalPerformance),
        mutation,
    )];
    if matches!(mutation, Mutation::UnexpectedDeclaration) {
        declarations.push(unexpected_declaration());
    }

    admit_campaign_result(&manifest, &bound, declarations)
        .map_err(|error| classify_compatibility(&error))
}

fn run_native(
    mutation: Mutation,
) -> Result<NativeCampaignAdmissionReceipt, FailureClass> {
    let manifest = manifest();
    let candidate_id = manifest.candidate_anchor.candidate.id.clone();
    let mut assertions = Vec::new();
    let mut envelopes = Vec::new();
    let mut functional_receipt = None;

    for dimension in EvidenceDimension::ALL {
        let envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension, mutation),
            "prospective-admission-parity-fixture-v0",
            format!("{{\"dimension\":{}}}", code(dimension)),
        )
        .map_err(|_| FailureClass::Setup)?;
        let receipt_sha = envelope.sha256().map_err(|_| FailureClass::Setup)?;
        if dimension == EvidenceDimension::FunctionalPerformance {
            functional_receipt = Some(receipt_sha.clone());
        }
        assertions.push(IdentityAssertion {
            assertion_id: format!("native-identity-{}", code(dimension)),
            candidate_id: candidate_id.clone(),
            namespace: "symthaea".into(),
            subject_id: candidate_id.0.clone(),
            basis: IdentityAssertionBasis::InternalCandidate,
            source_receipt_sha256: receipt_sha,
        });
        envelopes.push(envelope);
    }

    let dossier = assemble_native_envelope_dossier(&manifest, assertions, envelopes)
        .map_err(|_| FailureClass::Setup)?;
    let mut declarations = vec![declaration(
        functional_receipt.expect("functional-performance envelope is mandatory"),
        mutation,
    )];
    if matches!(mutation, Mutation::UnexpectedDeclaration) {
        declarations.push(unexpected_declaration());
    }

    admit_native_campaign_result(&manifest, &dossier, declarations)
        .map_err(|error| classify_native(&error))
}

#[test]
fn clean_prospective_acquisition_is_accepted_by_both_paths() {
    let compatibility = run_compatibility(Mutation::Clean).unwrap();
    let native = run_native(Mutation::Clean).unwrap();
    assert_eq!(
        compatibility.campaign_manifest_sha256,
        native.campaign_manifest_sha256
    );
    assert_eq!(compatibility.candidate_sha256, native.candidate_sha256);
    assert_eq!(
        compatibility.screening_policy_sha256,
        native.screening_policy_sha256
    );
}

#[test]
fn prospective_acquisition_failures_have_semantic_class_parity() {
    for mutation in [
        Mutation::WrongQuery,
        Mutation::MissingArtifactProvenance,
        Mutation::WrongReceipt,
        Mutation::UnexpectedDeclaration,
    ] {
        let expected = mutation.expected_failure().unwrap();
        let compatibility = run_compatibility(mutation);
        let native = run_native(mutation);
        assert_eq!(
            compatibility,
            Err(expected),
            "compatibility prospective path failed in the wrong semantic class for {mutation:?}"
        );
        assert_eq!(
            native,
            Err(expected),
            "native prospective path failed in the wrong semantic class for {mutation:?}"
        );
    }
}

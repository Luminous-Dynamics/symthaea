use std::collections::BTreeMap;

use symthaea_discovery::{
    Candidate, CandidateId, CandidateOrigin, EvidenceKind, EvidenceRef, FidelityLevel,
    ModelProvenance, ObjectiveDirection, Prediction, UncertaintyEstimate,
};
use symthaea_energy_evidence_envelope::wrap_evidence_payload_json;
use symthaea_energy_material_campaign::{
    freeze_campaign_manifest, EvidenceLanePlan, SourceCommitment, Tier1CampaignManifest,
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
use symthaea_energy_native_dossier::assemble_native_envelope_dossier;
use symthaea_energy_source_bound_acquisition::{
    admit_source_bound_compatibility, admit_source_bound_native, AcquisitionSourceIdentity,
    SourceBoundAcquisitionDeclaration, SourceBoundAdmissionError,
};

const INTERNAL_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const QUERY_SHA: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const ARTIFACT_SHA: &str =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mutation {
    Clean,
    WrongName,
    WrongVersion,
    WrongUri,
    WrongQuery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    Setup,
    SourceIdentity,
    Query,
    Other,
}

fn classify(error: &SourceBoundAdmissionError) -> FailureClass {
    match error {
        SourceBoundAdmissionError::AcquisitionSourceIdentityMismatch(_) => {
            FailureClass::SourceIdentity
        }
        SourceBoundAdmissionError::AcquisitionQueryMismatch(_) => FailureClass::Query,
        _ => FailureClass::Other,
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

fn source_name(dimension: EvidenceDimension) -> String {
    format!("source-{}", code(dimension))
}

fn source_version(dimension: EvidenceDimension) -> String {
    format!("2026.{}", code(dimension) + 1)
}

fn source_uri(dimension: EvidenceDimension) -> String {
    format!("https://example.invalid/source/{}", code(dimension))
}

fn candidate() -> Candidate {
    Candidate {
        id: CandidateId::new("source-bound-parity-candidate").unwrap(),
        kind: "energy_material".into(),
        specification: BTreeMap::from([("formula".into(), "LiFePO4".into())]),
        origin: CandidateOrigin::UserProposed,
    }
}

fn policy() -> EnergyMaterialScreeningPolicy {
    EnergyMaterialScreeningPolicy {
        policy_id: "source-bound-acquisition-v1-parity".into(),
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

fn manifest(prospective_dimension: EvidenceDimension) -> Tier1CampaignManifest {
    let lanes = EvidenceDimension::ALL
        .into_iter()
        .map(|dimension| EvidenceLanePlan {
            dimension,
            adapter_name: format!("adapter-{}", code(dimension)),
            adapter_version: "v1".into(),
            expected_model_name: format!("model-{}", code(dimension)),
            expected_model_version: Some("v1".into()),
            method_parameters: BTreeMap::new(),
            source_commitment: if dimension == prospective_dimension {
                SourceCommitment::ProspectiveAcquisition {
                    source_name: source_name(dimension),
                    source_version: source_version(dimension),
                    source_uri: source_uri(dimension),
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
        "source-bound-acquisition-v1-parity",
        anchor_candidate(candidate()).unwrap(),
        policy(),
        lanes,
        None,
        vec![],
    )
    .unwrap()
}

fn prediction(
    dimension: EvidenceDimension,
    prospective_dimension: EvidenceDimension,
) -> Prediction {
    let provenance_sha = if dimension == prospective_dimension {
        ARTIFACT_SHA
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
            version: Some("v1".into()),
            implementation_digest: None,
            input_digest: Some(format!("sha256:{provenance_sha}")),
            output_digest: None,
        },
        assumptions: vec![],
        evidence: vec![EvidenceRef {
            id: format!("evidence-{}", code(dimension)),
            kind: EvidenceKind::Dataset,
            uri: None,
            digest: Some(format!("sha256:{provenance_sha}")),
            note: None,
        }],
    }
}

fn compatibility_receipt_sha(dimension: EvidenceDimension) -> String {
    let nibble = char::from(b'1' + code(dimension));
    std::iter::repeat_n(nibble, 64).collect()
}

fn declaration(
    dimension: EvidenceDimension,
    source_receipt_sha256: String,
    mutation: Mutation,
) -> SourceBoundAcquisitionDeclaration {
    SourceBoundAcquisitionDeclaration {
        dimension,
        source: AcquisitionSourceIdentity {
            source_name: if matches!(mutation, Mutation::WrongName) {
                "wrong-source".into()
            } else {
                source_name(dimension)
            },
            source_version: if matches!(mutation, Mutation::WrongVersion) {
                "wrong-version".into()
            } else {
                source_version(dimension)
            },
            source_uri: if matches!(mutation, Mutation::WrongUri) {
                "https://example.invalid/wrong".into()
            } else {
                source_uri(dimension)
            },
        },
        acquisition_query_sha256: if matches!(mutation, Mutation::WrongQuery) {
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into()
        } else {
            QUERY_SHA.into()
        },
        acquired_artifact_sha256: ARTIFACT_SHA.into(),
        source_receipt_sha256,
        reviewer: "source-bound-parity-fixture".into(),
        note: "Exact source identity fixture; reviewer identity itself is not authenticated here."
            .into(),
    }
}

fn compatibility_inputs(
    prospective_dimension: EvidenceDimension,
) -> Result<(
    Tier1CampaignManifest,
    symthaea_energy_material_candidate_version::CandidateVersionBoundDossier,
), FailureClass> {
    let manifest = manifest(prospective_dimension);
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
            prediction: prediction(dimension, prospective_dimension),
        });
        attestations.push(ReceiptVersionAttestation {
            attestation_id: format!("attestation-{}", code(dimension)),
            dimension,
            source_receipt_sha256: receipt_sha,
            candidate_sha256: candidate_sha.clone(),
            reviewer: "source-bound-parity-fixture".into(),
            review_note: "Fixture candidate-version attestation.".into(),
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
    Ok((manifest, bound))
}

fn native_inputs(
    prospective_dimension: EvidenceDimension,
) -> Result<(
    Tier1CampaignManifest,
    symthaea_energy_native_dossier::NativeEnvelopeDossier,
    String,
), FailureClass> {
    let manifest = manifest(prospective_dimension);
    let candidate_id = manifest.candidate_anchor.candidate.id.clone();
    let mut assertions = Vec::new();
    let mut envelopes = Vec::new();
    let mut prospective_receipt = None;

    for dimension in EvidenceDimension::ALL {
        let envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension, prospective_dimension),
            "source-bound-acquisition-v1-parity-fixture",
            format!("{{\"dimension\":{}}}", code(dimension)),
        )
        .map_err(|_| FailureClass::Setup)?;
        let receipt_sha = envelope.sha256().map_err(|_| FailureClass::Setup)?;
        if dimension == prospective_dimension {
            prospective_receipt = Some(receipt_sha.clone());
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
    Ok((
        manifest,
        dossier,
        prospective_receipt.expect("prospective envelope must exist"),
    ))
}

fn run_compatibility(
    prospective_dimension: EvidenceDimension,
    mutation: Mutation,
) -> Result<String, FailureClass> {
    let (manifest, dossier) = compatibility_inputs(prospective_dimension)?;
    let result = admit_source_bound_compatibility(
        &manifest,
        &dossier,
        vec![declaration(
            prospective_dimension,
            compatibility_receipt_sha(prospective_dimension),
            mutation,
        )],
    )
    .map_err(|error| classify(&error))?;
    result
        .validate_with_inputs(&manifest, &dossier)
        .map_err(|_| FailureClass::Other)?;
    result.sha256().map_err(|_| FailureClass::Other)
}

fn run_native(
    prospective_dimension: EvidenceDimension,
    mutation: Mutation,
) -> Result<String, FailureClass> {
    let (manifest, dossier, receipt_sha) = native_inputs(prospective_dimension)?;
    let result = admit_source_bound_native(
        &manifest,
        &dossier,
        vec![declaration(prospective_dimension, receipt_sha, mutation)],
    )
    .map_err(|error| classify(&error))?;
    result
        .validate_with_inputs(&manifest, &dossier)
        .map_err(|_| FailureClass::Other)?;
    result.sha256().map_err(|_| FailureClass::Other)
}

#[test]
fn every_dimension_enforces_exact_source_identity_on_both_paths() {
    for prospective_dimension in EvidenceDimension::ALL {
        assert!(run_compatibility(prospective_dimension, Mutation::Clean).is_ok());
        assert!(run_native(prospective_dimension, Mutation::Clean).is_ok());

        for mutation in [
            Mutation::WrongName,
            Mutation::WrongVersion,
            Mutation::WrongUri,
        ] {
            assert_eq!(
                run_compatibility(prospective_dimension, mutation),
                Err(FailureClass::SourceIdentity),
                "compatibility source identity mismatch escaped or was misclassified for {prospective_dimension:?} / {mutation:?}"
            );
            assert_eq!(
                run_native(prospective_dimension, mutation),
                Err(FailureClass::SourceIdentity),
                "native source identity mismatch escaped or was misclassified for {prospective_dimension:?} / {mutation:?}"
            );
        }
        assert_eq!(
            run_compatibility(prospective_dimension, Mutation::WrongQuery),
            Err(FailureClass::Query)
        );
        assert_eq!(
            run_native(prospective_dimension, Mutation::WrongQuery),
            Err(FailureClass::Query)
        );
    }
}

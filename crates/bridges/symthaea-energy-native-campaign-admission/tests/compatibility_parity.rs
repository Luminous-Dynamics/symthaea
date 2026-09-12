use std::collections::BTreeMap;

use symthaea_discovery::{
    Candidate, CandidateId, CandidateOrigin, EvidenceKind, EvidenceRef, FidelityLevel,
    ModelProvenance, ObjectiveDirection, Prediction, UncertaintyEstimate,
};
use symthaea_energy_evidence_envelope::wrap_evidence_payload_json;
use symthaea_energy_material_campaign::{
    admit_campaign_result, freeze_campaign_manifest, CampaignAdmissionReceipt, EvidenceLanePlan,
    SourceCommitment, Tier1CampaignManifest,
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
    admit_native_campaign_result, NativeCampaignAdmissionReceipt,
};
use symthaea_energy_native_dossier::assemble_native_envelope_dossier;

const LINEAGE_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OTHER_SHA: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[derive(Debug, Clone, Copy)]
enum Mutation {
    Clean,
    WrongModel,
    MissingLineage,
    MissingRequiredEvidence,
    LowFidelity,
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
        id: CandidateId::new("parity-candidate").unwrap(),
        kind: "energy_material".into(),
        specification: BTreeMap::from([
            ("formula".into(), "LiFePO4".into()),
            ("phase".into(), "olivine".into()),
        ]),
        origin: CandidateOrigin::UserProposed,
    }
}

fn policy() -> EnergyMaterialScreeningPolicy {
    EnergyMaterialScreeningPolicy {
        policy_id: "campaign-admission-parity-v0".into(),
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
            method_parameters: BTreeMap::from([("mode".into(), "parity-fixture".into())]),
            source_commitment: SourceCommitment::InternalLineage {
                sha256: LINEAGE_SHA.into(),
            },
            required_evidence_kinds: vec![EvidenceKind::Dataset],
        })
        .collect();

    freeze_campaign_manifest(
        "campaign-admission-parity-v0",
        anchor_candidate(candidate()).unwrap(),
        policy(),
        lanes,
        None,
        vec!["Compatibility/native admission parity fixture.".into()],
    )
    .unwrap()
}

fn prediction(dimension: EvidenceDimension, mutation: Mutation) -> Prediction {
    let targeted = dimension == EvidenceDimension::FunctionalPerformance;
    let wrong_model = targeted && matches!(mutation, Mutation::WrongModel);
    let missing_lineage = targeted && matches!(mutation, Mutation::MissingLineage);
    let missing_required = targeted && matches!(mutation, Mutation::MissingRequiredEvidence);
    let low_fidelity = targeted && matches!(mutation, Mutation::LowFidelity);

    Prediction {
        metric: format!("metric-{}", code(dimension)),
        value: 1.0 + f64::from(code(dimension)),
        unit: "score".into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0).unwrap(),
        fidelity: if low_fidelity {
            FidelityLevel::Heuristic
        } else {
            FidelityLevel::Surrogate
        },
        model: ModelProvenance {
            name: if wrong_model {
                "wrong-model".into()
            } else {
                format!("model-{}", code(dimension))
            },
            version: Some("v0".into()),
            implementation_digest: None,
            input_digest: (!missing_lineage).then(|| format!("sha256:{LINEAGE_SHA}")),
            output_digest: None,
        },
        assumptions: vec![],
        evidence: vec![EvidenceRef {
            id: format!("fixture-evidence-{}", code(dimension)),
            kind: if missing_required {
                EvidenceKind::Literature
            } else {
                EvidenceKind::Dataset
            },
            uri: None,
            digest: Some(format!(
                "sha256:{}",
                if missing_lineage { OTHER_SHA } else { LINEAGE_SHA }
            )),
            note: None,
        }],
    }
}

fn compatibility_receipt_sha(dimension: EvidenceDimension) -> String {
    let nibble = char::from(b'1' + code(dimension));
    std::iter::repeat(nibble).take(64).collect()
}

fn run_compatibility(mutation: Mutation) -> Result<CampaignAdmissionReceipt, String> {
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
            reviewer: "parity-fixture".into(),
            review_note: "Fixture binds the compatibility receipt to the exact candidate version."
                .into(),
        });
    }

    let dossier = assemble_dossier(
        candidate.id.clone(),
        &manifest.screening_policy,
        assertions,
        contributions,
    )
    .map_err(|error| format!("assemble compatibility dossier: {error:?}"))?;
    let bound = bind_dossier_to_candidate_version(candidate, dossier, attestations)
        .map_err(|error| format!("bind compatibility dossier: {error:?}"))?;
    admit_campaign_result(&manifest, &bound, vec![])
        .map_err(|error| format!("compatibility admission: {error:?}"))
}

fn run_native(mutation: Mutation) -> Result<NativeCampaignAdmissionReceipt, String> {
    let manifest = manifest();
    let candidate_id = manifest.candidate_anchor.candidate.id.clone();

    let mut assertions = Vec::new();
    let mut envelopes = Vec::new();

    for dimension in EvidenceDimension::ALL {
        let envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension, mutation),
            "campaign-admission-parity-fixture-v0",
            format!("{{\"dimension\":{}}}", code(dimension)),
        )
        .map_err(|error| format!("wrap native envelope: {error:?}"))?;
        let receipt_sha = envelope
            .sha256()
            .map_err(|error| format!("hash native envelope: {error:?}"))?;
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
        .map_err(|error| format!("assemble native dossier: {error:?}"))?;
    admit_native_campaign_result(&manifest, &dossier, vec![])
        .map_err(|error| format!("native admission: {error:?}"))
}

#[test]
fn clean_campaign_is_accepted_by_both_admission_paths() {
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
fn compatibility_and_native_paths_reject_the_same_frozen_plan_violations() {
    for mutation in [
        Mutation::WrongModel,
        Mutation::MissingLineage,
        Mutation::MissingRequiredEvidence,
        Mutation::LowFidelity,
    ] {
        let compatibility = run_compatibility(mutation);
        let native = run_native(mutation);
        assert_eq!(
            compatibility.is_ok(),
            native.is_ok(),
            "admission-path parity diverged for {mutation:?}: compatibility={compatibility:?}, native={native:?}"
        );
        assert!(
            compatibility.is_err(),
            "corrupted campaign unexpectedly passed compatibility admission for {mutation:?}"
        );
    }
}

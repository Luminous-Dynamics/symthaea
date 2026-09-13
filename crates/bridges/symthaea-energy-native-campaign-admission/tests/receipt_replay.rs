use std::collections::BTreeMap;

use symthaea_discovery::{
    Candidate, CandidateId, CandidateOrigin, EvidenceKind, EvidenceRef, FidelityLevel,
    ModelProvenance, ObjectiveDirection, Prediction, UncertaintyEstimate,
};
use symthaea_energy_evidence_envelope::wrap_evidence_payload_json;
use symthaea_energy_material_campaign::{
    freeze_campaign_manifest, EvidenceLanePlan, SourceCommitment, Tier1CampaignManifest,
};
use symthaea_energy_material_candidate_version::anchor_candidate;
use symthaea_energy_material_dossier::{IdentityAssertion, IdentityAssertionBasis};
use symthaea_energy_material_screening::{
    EnergyMaterialScreeningPolicy, EvidenceDimension, MetricContract,
};
use symthaea_energy_native_campaign_admission::{
    admit_native_campaign_result, NativeAdmissionError, NativeCampaignAdmissionReceipt,
};
use symthaea_energy_native_dossier::{
    assemble_native_envelope_dossier, NativeEnvelopeDossier,
};

const LINEAGE_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

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

fn manifest() -> Tier1CampaignManifest {
    let candidate = Candidate {
        id: CandidateId::new("receipt-replay-candidate").unwrap(),
        kind: "energy_material".into(),
        specification: BTreeMap::from([
            ("formula".into(), "LiFePO4".into()),
            ("phase".into(), "olivine".into()),
        ]),
        origin: CandidateOrigin::UserProposed,
    };
    let policy = EnergyMaterialScreeningPolicy {
        policy_id: "receipt-replay-policy-v0".into(),
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
    };
    let lanes = EvidenceDimension::ALL
        .into_iter()
        .map(|dimension| EvidenceLanePlan {
            dimension,
            adapter_name: format!("adapter-{}", code(dimension)),
            adapter_version: "v0".into(),
            expected_model_name: format!("model-{}", code(dimension)),
            expected_model_version: Some("v0".into()),
            method_parameters: BTreeMap::new(),
            source_commitment: SourceCommitment::InternalLineage {
                sha256: LINEAGE_SHA.into(),
            },
            required_evidence_kinds: vec![EvidenceKind::Dataset],
        })
        .collect();

    freeze_campaign_manifest(
        "receipt-replay-campaign-v0",
        anchor_candidate(candidate).unwrap(),
        policy,
        lanes,
        None,
        vec![],
    )
    .unwrap()
}

fn prediction(dimension: EvidenceDimension) -> Prediction {
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
            input_digest: Some(format!("sha256:{LINEAGE_SHA}")),
            output_digest: None,
        },
        assumptions: vec![],
        evidence: vec![EvidenceRef {
            id: format!("receipt-replay-evidence-{}", code(dimension)),
            kind: EvidenceKind::Dataset,
            uri: None,
            digest: Some(format!("sha256:{LINEAGE_SHA}")),
            note: None,
        }],
    }
}

fn clean_inputs() -> (
    Tier1CampaignManifest,
    NativeEnvelopeDossier,
    NativeCampaignAdmissionReceipt,
) {
    let manifest = manifest();
    let candidate_id = manifest.candidate_anchor.candidate.id.clone();
    let mut assertions = Vec::new();
    let mut envelopes = Vec::new();

    for dimension in EvidenceDimension::ALL {
        let envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension),
            "receipt-replay-fixture-v0",
            format!("{{\"dimension\":{}}}", code(dimension)),
        )
        .unwrap();
        let envelope_sha = envelope.sha256().unwrap();
        assertions.push(IdentityAssertion {
            assertion_id: format!("receipt-replay-identity-{}", code(dimension)),
            candidate_id: candidate_id.clone(),
            namespace: "symthaea".into(),
            subject_id: candidate_id.0.clone(),
            basis: IdentityAssertionBasis::InternalCandidate,
            source_receipt_sha256: envelope_sha,
        });
        envelopes.push(envelope);
    }

    let dossier = assemble_native_envelope_dossier(&manifest, assertions, envelopes).unwrap();
    let receipt = admit_native_campaign_result(&manifest, &dossier, vec![]).unwrap();
    (manifest, dossier, receipt)
}

#[test]
fn clean_receipt_replays_deterministically() {
    let (manifest, dossier, receipt) = clean_inputs();
    receipt.validate_with_inputs(&manifest, &dossier).unwrap();
    let recomputed = admit_native_campaign_result(&manifest, &dossier, vec![]).unwrap();
    assert_eq!(receipt, recomputed);
    assert_eq!(receipt.sha256().unwrap(), recomputed.sha256().unwrap());
}

#[test]
fn swapped_dimension_envelope_bindings_pass_structure_but_fail_replay() {
    let (manifest, dossier, receipt) = clean_inputs();
    let mut tampered = receipt.clone();
    let first = tampered.admitted_envelopes[0].envelope_sha256.clone();
    let second = tampered.admitted_envelopes[1].envelope_sha256.clone();
    tampered.admitted_envelopes[0].envelope_sha256 = second;
    tampered.admitted_envelopes[1].envelope_sha256 = first;

    // The set of seven digests is still valid and dimensions remain canonical;
    // only the semantic dimension->receipt mapping was corrupted.
    tampered.validate().unwrap();
    assert!(matches!(
        tampered.validate_with_inputs(&manifest, &dossier),
        Err(NativeAdmissionError::ReceiptReplayMismatch)
    ));
}

#[test]
fn altered_nonempty_disclosure_passes_structure_but_fails_replay() {
    let (manifest, dossier, receipt) = clean_inputs();
    let mut tampered = receipt.clone();
    tampered.outcome_disclosure =
        "Structurally non-empty but not the disclosure emitted by admission.".into();

    tampered.validate().unwrap();
    assert!(matches!(
        tampered.validate_with_inputs(&manifest, &dossier),
        Err(NativeAdmissionError::ReceiptReplayMismatch)
    ));
}

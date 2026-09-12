// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native Tier-1 material dossier assembly from campaign-bound evidence envelopes.
//!
//! Generic dossier contributions are derived from validated native envelopes
//! rather than separately supplied by the caller. External identity assertions
//! remain mandatory through `symthaea-energy-material-dossier` because native
//! candidate/campaign lineage does not prove that an external polymorph,
//! substance, or process record actually refers to the intended candidate.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_energy_evidence_envelope::EnergyEvidenceEnvelope;
use symthaea_energy_material_campaign::Tier1CampaignManifest;
use symthaea_energy_material_dossier::{
    assemble_dossier, EnergyMaterialDossier, EvidenceContribution, IdentityAssertion,
};
use symthaea_energy_material_screening::EvidenceDimension;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "NATIVE ENERGY-MATERIAL DOSSIER ASSEMBLY ONLY -- envelope-derived contributions preserve provenance but do not prove external identity mappings, scientific validity, certification, synthesis success, or deployment authority.";
const NATIVE_DOSSIER_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.native-dossier.v0\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeEnvelopeDossier {
    pub schema: String,
    pub capability_classification: String,
    pub campaign_manifest_sha256: String,
    pub candidate_sha256: String,
    pub dossier_sha256: String,
    pub dossier: EnergyMaterialDossier,
    pub envelopes: Vec<EnergyEvidenceEnvelope>,
}

impl NativeEnvelopeDossier {
    pub fn validate_with_manifest(
        &self,
        manifest: &Tier1CampaignManifest,
    ) -> Result<(), NativeDossierError> {
        if self.schema != "symthaea.energy-material.native-dossier.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(NativeDossierError::InvalidNativeDossier(
                "native-dossier schema/capability classification was altered".into(),
            ));
        }
        manifest.validate()?;
        let expected_manifest_sha = manifest.sha256()?;
        validate_sha256(&self.campaign_manifest_sha256, "campaign-manifest SHA-256")?;
        if self.campaign_manifest_sha256 != expected_manifest_sha {
            return Err(NativeDossierError::CampaignManifestMismatch);
        }
        validate_sha256(&self.candidate_sha256, "candidate SHA-256")?;
        if self.candidate_sha256 != manifest.candidate_anchor.candidate_sha256 {
            return Err(NativeDossierError::CandidateVersionMismatch);
        }

        self.dossier.validate_integrity()?;
        validate_sha256(&self.dossier_sha256, "dossier SHA-256")?;
        let expected_dossier_sha = self.dossier.sha256()?;
        if self.dossier_sha256 != expected_dossier_sha {
            return Err(NativeDossierError::DossierDigestMismatch {
                expected: expected_dossier_sha,
                actual: self.dossier_sha256.clone(),
            });
        }
        if self.dossier.candidate_id != manifest.candidate_anchor.candidate.id
            || self.dossier.policy_sha256 != manifest.screening_policy_sha256
        {
            return Err(NativeDossierError::DossierCampaignMismatch);
        }
        if self.envelopes.len() != self.dossier.contributions.len() {
            return Err(NativeDossierError::EnvelopeContributionCountMismatch {
                envelopes: self.envelopes.len(),
                contributions: self.dossier.contributions.len(),
            });
        }

        let mut envelope_map = BTreeMap::new();
        for envelope in &self.envelopes {
            envelope.validate_with_manifest(manifest)?;
            let code = dimension_code(envelope.binding.dimension);
            if envelope_map.insert(code, envelope).is_some() {
                return Err(NativeDossierError::DuplicateEnvelopeDimension(
                    envelope.binding.dimension,
                ));
            }
        }

        let mut seen_contributions = BTreeSet::new();
        for contribution in &self.dossier.contributions {
            let code = dimension_code(contribution.dimension);
            if !seen_contributions.insert(code) {
                return Err(NativeDossierError::DuplicateContributionDimension(
                    contribution.dimension,
                ));
            }
            let envelope = envelope_map
                .get(&code)
                .copied()
                .ok_or(NativeDossierError::MissingEnvelope(contribution.dimension))?;
            if contribution.candidate_id != envelope.binding.candidate_id {
                return Err(NativeDossierError::CandidateIdMismatch(
                    contribution.dimension,
                ));
            }
            let envelope_sha = envelope.sha256()?;
            if contribution.source_receipt_sha256 != envelope_sha {
                return Err(NativeDossierError::SourceReceiptMismatch(
                    contribution.dimension,
                ));
            }
            if contribution.prediction != envelope.prediction {
                return Err(NativeDossierError::PredictionMismatch(
                    contribution.dimension,
                ));
            }
        }
        Ok(())
    }

    pub fn sha256(&self, manifest: &Tier1CampaignManifest) -> Result<String, NativeDossierError> {
        self.validate_with_manifest(manifest)?;
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(NATIVE_DOSSIER_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn assemble_native_envelope_dossier(
    manifest: &Tier1CampaignManifest,
    identity_assertions: Vec<IdentityAssertion>,
    mut envelopes: Vec<EnergyEvidenceEnvelope>,
) -> Result<NativeEnvelopeDossier, NativeDossierError> {
    manifest.validate()?;
    let mut seen = BTreeSet::new();
    for envelope in &envelopes {
        envelope.validate_with_manifest(manifest)?;
        let code = dimension_code(envelope.binding.dimension);
        if !seen.insert(code) {
            return Err(NativeDossierError::DuplicateEnvelopeDimension(
                envelope.binding.dimension,
            ));
        }
    }
    envelopes.sort_by_key(|envelope| dimension_code(envelope.binding.dimension));

    let mut contributions = Vec::with_capacity(envelopes.len());
    for envelope in &envelopes {
        contributions.push(EvidenceContribution {
            dimension: envelope.binding.dimension,
            candidate_id: envelope.binding.candidate_id.clone(),
            identity_assertion_id: identity_assertion_for_dimension(
                envelope.binding.dimension,
                &identity_assertions,
                &envelope.sha256()?,
            )?,
            source_receipt_sha256: envelope.sha256()?,
            prediction: envelope.prediction.clone(),
        });
    }

    let dossier = assemble_dossier(
        manifest.candidate_anchor.candidate.id.clone(),
        &manifest.screening_policy,
        identity_assertions,
        contributions,
    )?;
    let dossier_sha256 = dossier.sha256()?;
    let native = NativeEnvelopeDossier {
        schema: "symthaea.energy-material.native-dossier.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: manifest.sha256()?,
        candidate_sha256: manifest.candidate_anchor.candidate_sha256.clone(),
        dossier_sha256,
        dossier,
        envelopes,
    };
    native.validate_with_manifest(manifest)?;
    Ok(native)
}

fn identity_assertion_for_dimension(
    dimension: EvidenceDimension,
    assertions: &[IdentityAssertion],
    source_receipt_sha256: &str,
) -> Result<String, NativeDossierError> {
    let matches: Vec<&IdentityAssertion> = assertions
        .iter()
        .filter(|assertion| assertion.source_receipt_sha256 == source_receipt_sha256)
        .collect();
    match matches.as_slice() {
        [assertion] => Ok(assertion.assertion_id.clone()),
        [] => Err(NativeDossierError::MissingIdentityAssertion(dimension)),
        _ => Err(NativeDossierError::AmbiguousIdentityAssertion(dimension)),
    }
}

#[derive(Debug, Error)]
pub enum NativeDossierError {
    #[error("invalid native dossier: {0}")]
    InvalidNativeDossier(String),
    #[error("campaign-manifest digest differs from supplied manifest")]
    CampaignManifestMismatch,
    #[error("candidate-version digest differs from supplied manifest")]
    CandidateVersionMismatch,
    #[error("native dossier's generic dossier does not belong to supplied campaign")]
    DossierCampaignMismatch,
    #[error("dossier SHA mismatch: expected {expected}, got {actual}")]
    DossierDigestMismatch { expected: String, actual: String },
    #[error("envelope/contribution count mismatch: {envelopes} envelopes, {contributions} contributions")]
    EnvelopeContributionCountMismatch {
        envelopes: usize,
        contributions: usize,
    },
    #[error("duplicate native envelope for dimension {0:?}")]
    DuplicateEnvelopeDimension(EvidenceDimension),
    #[error("duplicate generic contribution for dimension {0:?}")]
    DuplicateContributionDimension(EvidenceDimension),
    #[error("missing native envelope for contribution dimension {0:?}")]
    MissingEnvelope(EvidenceDimension),
    #[error("native envelope candidate id differs from generic contribution for {0:?}")]
    CandidateIdMismatch(EvidenceDimension),
    #[error("native envelope digest differs from generic contribution source receipt for {0:?}")]
    SourceReceiptMismatch(EvidenceDimension),
    #[error("native envelope prediction differs from generic dossier prediction for {0:?}")]
    PredictionMismatch(EvidenceDimension),
    #[error("no identity assertion binds native envelope for dimension {0:?}")]
    MissingIdentityAssertion(EvidenceDimension),
    #[error("multiple identity assertions bind the same native envelope for dimension {0:?}")]
    AmbiguousIdentityAssertion(EvidenceDimension),
    #[error(transparent)]
    Envelope(#[from] symthaea_energy_evidence_envelope::EnvelopeError),
    #[error(transparent)]
    Campaign(#[from] symthaea_energy_material_campaign::CampaignError),
    #[error(transparent)]
    Dossier(#[from] symthaea_energy_material_dossier::DossierError),
    #[error("native-dossier JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_sha256(value: &str, name: &str) -> Result<(), NativeDossierError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NativeDossierError::InvalidNativeDossier(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn dimension_code(dimension: EvidenceDimension) -> u8 {
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

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_discovery::{
        Candidate, CandidateId, CandidateOrigin, EvidenceKind, EvidenceRef, FidelityLevel,
        ModelProvenance, ObjectiveDirection, Prediction, UncertaintyEstimate,
    };
    use symthaea_energy_evidence_envelope::wrap_evidence_payload_json;
    use symthaea_energy_material_campaign::{
        freeze_campaign_manifest, EvidenceLanePlan, SourceCommitment,
    };
    use symthaea_energy_material_candidate_version::anchor_candidate;
    use symthaea_energy_material_dossier::{IdentityAssertion, IdentityAssertionBasis};
    use symthaea_energy_material_screening::{
        EnergyMaterialScreeningPolicy, EvidenceCompleteness, MetricContract,
    };

    fn manifest() -> Tier1CampaignManifest {
        let candidate = Candidate {
            id: CandidateId::new("candidate-a").unwrap(),
            kind: "energy_material".into(),
            specification: BTreeMap::from([("formula".into(), "LiFePO4".into())]),
            origin: CandidateOrigin::UserProposed,
        };
        let policy = EnergyMaterialScreeningPolicy {
            policy_id: "native-dossier-test".into(),
            contracts: EvidenceDimension::ALL
                .into_iter()
                .map(|dimension| MetricContract {
                    dimension,
                    metric: format!("metric-{dimension:?}"),
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
                adapter_name: format!("adapter-{dimension:?}"),
                adapter_version: "v0".into(),
                expected_model_name: format!("model-{dimension:?}"),
                expected_model_version: Some("v0".into()),
                method_parameters: BTreeMap::new(),
                source_commitment: SourceCommitment::InternalLineage {
                    sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                },
                required_evidence_kinds: vec![EvidenceKind::Dataset],
            })
            .collect();
        freeze_campaign_manifest(
            "native-dossier-campaign",
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
            metric: format!("metric-{dimension:?}"),
            value: 1.0,
            unit: "score".into(),
            uncertainty: UncertaintyEstimate::new(1.0, 0.0).unwrap(),
            fidelity: FidelityLevel::Surrogate,
            model: ModelProvenance {
                name: format!("model-{dimension:?}"),
                version: Some("v0".into()),
                implementation_digest: None,
                input_digest: Some(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                ),
                output_digest: None,
            },
            assumptions: vec![],
            evidence: vec![EvidenceRef {
                id: "fixture".into(),
                kind: EvidenceKind::Dataset,
                uri: None,
                digest: Some(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                ),
                note: None,
            }],
        }
    }

    fn envelope(
        manifest: &Tier1CampaignManifest,
        dimension: EvidenceDimension,
    ) -> EnergyEvidenceEnvelope {
        wrap_evidence_payload_json(
            manifest,
            dimension,
            prediction(dimension),
            "fixture-receipt-v0",
            format!("{{\"dimension\":\"{dimension:?}\"}}"),
        )
        .unwrap()
    }

    fn assertion(
        manifest: &Tier1CampaignManifest,
        envelope: &EnergyEvidenceEnvelope,
        id: &str,
    ) -> IdentityAssertion {
        IdentityAssertion {
            assertion_id: id.into(),
            candidate_id: manifest.candidate_anchor.candidate.id.clone(),
            namespace: "symthaea".into(),
            subject_id: manifest.candidate_anchor.candidate.id.0.clone(),
            basis: IdentityAssertionBasis::InternalCandidate,
            source_receipt_sha256: envelope.sha256().unwrap(),
        }
    }

    #[test]
    fn partial_native_dossier_stays_partial() {
        let manifest = manifest();
        let env = envelope(&manifest, EvidenceDimension::FunctionalPerformance);
        let native = assemble_native_envelope_dossier(
            &manifest,
            vec![assertion(&manifest, &env, "functional")],
            vec![env],
        )
        .unwrap();
        assert_eq!(
            native.dossier.screening_assessment.completeness,
            EvidenceCompleteness::Incomplete
        );
        native.validate_with_manifest(&manifest).unwrap();
    }

    #[test]
    fn generic_prediction_cannot_diverge_from_envelope() {
        let manifest = manifest();
        let env = envelope(&manifest, EvidenceDimension::FunctionalPerformance);
        let mut native = assemble_native_envelope_dossier(
            &manifest,
            vec![assertion(&manifest, &env, "functional")],
            vec![env],
        )
        .unwrap();
        native.dossier.contributions[0].prediction.value = 99.0;
        assert!(matches!(
            native.validate_with_manifest(&manifest),
            Err(NativeDossierError::PredictionMismatch(_))
        ));
    }

    #[test]
    fn identity_assertion_must_bind_exact_envelope_digest() {
        let manifest = manifest();
        let env = envelope(&manifest, EvidenceDimension::FunctionalPerformance);
        let mut wrong = assertion(&manifest, &env, "functional");
        wrong.source_receipt_sha256 =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into();
        assert!(matches!(
            assemble_native_envelope_dossier(&manifest, vec![wrong], vec![env]),
            Err(NativeDossierError::MissingIdentityAssertion(_))
        ));
    }

    #[test]
    fn envelope_input_order_does_not_change_native_dossier_identity() {
        let manifest = manifest();
        let first_env = envelope(&manifest, EvidenceDimension::FunctionalPerformance);
        let second_env = envelope(&manifest, EvidenceDimension::ThermodynamicStability);
        let assertions = vec![
            assertion(&manifest, &first_env, "functional"),
            assertion(&manifest, &second_env, "stability"),
        ];
        let first = assemble_native_envelope_dossier(
            &manifest,
            assertions.clone(),
            vec![first_env.clone(), second_env.clone()],
        )
        .unwrap();
        let second = assemble_native_envelope_dossier(
            &manifest,
            assertions,
            vec![second_env, first_env],
        )
        .unwrap();
        assert_eq!(first.sha256(&manifest).unwrap(), second.sha256(&manifest).unwrap());
    }
}

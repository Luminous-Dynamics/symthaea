// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::canonical::{
    AcceptanceRecordDigestV1, AnalysisConfigurationDigestV1, AnalysisRequirementRevisionIdV1,
    AnalysisTrustErrorV1, CurrentnessAssertionIdV2, CurrentnessAttestationDigestV1,
    ModelRevisionDigestV1, ObligationRevisionIdV1, SubjectRevisionIdV1, SubjectStateDigestV1,
    TwinRevisionIdV1, TwinSchemaDigestV1, TwinStateDigestV1, ValidityDimensionDigestV1,
    ValidityDomainRevisionIdV1, CURRENTNESS_DOMAIN_V2, OBLIGATION_DOMAIN_V1,
    REQUIREMENT_DOMAIN_V1, SUBJECT_DOMAIN_V1, TWIN_DOMAIN_V1, VALIDITY_DOMAIN_V1,
    canonical_text, domain_hash,
};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_formal_safety::{EvidenceKind, ProofObligation};

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedAnalysisRequirementV1 {
    revision_id: AnalysisRequirementRevisionIdV1,
    max_bending_stress_pa: f64,
    acceptance_record_digest: AcceptanceRecordDigestV1,
}

impl AcceptedAnalysisRequirementV1 {
    pub fn civil_service_stress_250_mpa(
        acceptance_record_digest: AcceptanceRecordDigestV1,
    ) -> Self {
        let structural_invariants = canonical_invariants(["stress <= 250 MPa".to_string()])
            .expect("static canary invariant must be canonical");
        let preimage = json!({
            "acceptance_record_digest": acceptance_record_digest.as_str(),
            "criticality": "Blocking",
            "domain": "Civil",
            "expected_evidence_kind": EvidenceKind::Analysis.canonical_name(),
            "logical_requirement_id": "REQ-STRESS",
            "schema": "symthaea.etk-accepted-requirement.v1",
            "statement": "stress remains below allowable",
            "structural_invariants": structural_invariants,
        });
        Self {
            revision_id: AnalysisRequirementRevisionIdV1::from_digest(domain_hash(
                REQUIREMENT_DOMAIN_V1,
                &preimage,
            )),
            max_bending_stress_pa: 250.0e6,
            acceptance_record_digest,
        }
    }

    pub fn revision_id(&self) -> &AnalysisRequirementRevisionIdV1 {
        &self.revision_id
    }

    pub fn max_bending_stress_pa(&self) -> f64 {
        self.max_bending_stress_pa
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "acceptance_record_digest": self.acceptance_record_digest.as_str(),
            "authority": "accepted-analysis-requirement-only",
            "criticality": "Blocking",
            "domain": "Civil",
            "expected_evidence_kind": EvidenceKind::Analysis.canonical_name(),
            "logical_requirement_id": "REQ-STRESS",
            "max_bending_stress_pa": self.max_bending_stress_pa,
            "requirement_revision_id": self.revision_id.as_str(),
            "statement": "stress remains below allowable",
            "structural_invariants": ["stress <= 250 MPa"],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectRevisionV1 {
    revision_id: SubjectRevisionIdV1,
    namespace: String,
    subject_key: String,
    state_digest: SubjectStateDigestV1,
}

impl SubjectRevisionV1 {
    pub fn new(
        namespace: impl Into<String>,
        subject_key: impl Into<String>,
        state_digest: SubjectStateDigestV1,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        let namespace = canonical_text(namespace.into(), "subject namespace")?;
        let subject_key = canonical_text(subject_key.into(), "subject key")?;
        let preimage = json!({
            "namespace": namespace,
            "schema": "symthaea.etk-engineering-subject.v1",
            "state_digest": state_digest.as_str(),
            "subject_key": subject_key,
        });
        Ok(Self {
            revision_id: SubjectRevisionIdV1::from_digest(domain_hash(SUBJECT_DOMAIN_V1, &preimage)),
            namespace,
            subject_key,
            state_digest,
        })
    }

    pub fn revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.revision_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "authority": "subject-binding-only",
            "namespace": self.namespace.as_str(),
            "state_digest": self.state_digest.as_str(),
            "subject_key": self.subject_key.as_str(),
            "subject_revision_id": self.revision_id.as_str(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwinKindV1 {
    Design,
    AsBuilt,
    Operational,
}

impl TwinKindV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Design => "Design",
            Self::AsBuilt => "AsBuilt",
            Self::Operational => "Operational",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwinRevisionV1 {
    revision_id: TwinRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    kind: TwinKindV1,
    state_digest: TwinStateDigestV1,
    schema_digest: TwinSchemaDigestV1,
    parent_revision_id: Option<TwinRevisionIdV1>,
}

impl TwinRevisionV1 {
    pub fn new(
        subject: &SubjectRevisionV1,
        kind: TwinKindV1,
        state_digest: TwinStateDigestV1,
        schema_digest: TwinSchemaDigestV1,
        parent: Option<&TwinRevisionV1>,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if let Some(parent) = parent
            && (parent.subject_revision_id != *subject.revision_id() || parent.kind != kind)
        {
            return Err(AnalysisTrustErrorV1::TwinLineageMismatch);
        }
        let parent_revision_id = parent.map(|value| value.revision_id.clone());
        let preimage = json!({
            "kind": kind.as_str(),
            "parent_revision_id": parent_revision_id.as_ref().map(TwinRevisionIdV1::as_str),
            "schema": "symthaea.etk-twin-revision.v1",
            "schema_digest": schema_digest.as_str(),
            "state_digest": state_digest.as_str(),
            "subject_revision_id": subject.revision_id().as_str(),
        });
        Ok(Self {
            revision_id: TwinRevisionIdV1::from_digest(domain_hash(TWIN_DOMAIN_V1, &preimage)),
            subject_revision_id: subject.revision_id().clone(),
            kind,
            state_digest,
            schema_digest,
            parent_revision_id,
        })
    }

    pub fn revision_id(&self) -> &TwinRevisionIdV1 {
        &self.revision_id
    }

    pub fn subject_revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.subject_revision_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "authority": "twin-binding-only",
            "kind": self.kind.as_str(),
            "parent_revision_id": self.parent_revision_id.as_ref().map(TwinRevisionIdV1::as_str),
            "schema_digest": self.schema_digest.as_str(),
            "state_digest": self.state_digest.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.revision_id.as_str(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityDomainRevisionV1 {
    revision_id: ValidityDomainRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    model_revision_digest: ModelRevisionDigestV1,
    analysis_configuration_digest: AnalysisConfigurationDigestV1,
    dimensions: BTreeMap<String, ValidityDimensionDigestV1>,
}

impl ValidityDomainRevisionV1 {
    pub fn new(
        subject: &SubjectRevisionV1,
        twin: &TwinRevisionV1,
        model_revision_digest: ModelRevisionDigestV1,
        analysis_configuration_digest: AnalysisConfigurationDigestV1,
        dimensions: impl IntoIterator<Item = (String, ValidityDimensionDigestV1)>,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if twin.subject_revision_id() != subject.revision_id() {
            return Err(AnalysisTrustErrorV1::TwinSubjectMismatch);
        }
        let mut normalized = BTreeMap::new();
        for (name, digest) in dimensions {
            let name = canonical_text(name, "validity dimension")?;
            if normalized.insert(name.clone(), digest).is_some() {
                return Err(AnalysisTrustErrorV1::DuplicateValidityDimension(name));
            }
        }
        let mut dimension_object = Map::new();
        for (name, digest) in &normalized {
            dimension_object.insert(name.clone(), Value::String(digest.as_str().to_string()));
        }
        let preimage = json!({
            "dimensions": Value::Object(dimension_object),
            "model_revision_digest": model_revision_digest.as_str(),
            "schema": "symthaea.etk-validity-domain.v1",
            "solver_configuration_digest": analysis_configuration_digest.as_str(),
            "subject_revision_id": subject.revision_id().as_str(),
            "twin_revision_id": twin.revision_id().as_str(),
        });
        Ok(Self {
            revision_id: ValidityDomainRevisionIdV1::from_digest(domain_hash(
                VALIDITY_DOMAIN_V1,
                &preimage,
            )),
            subject_revision_id: subject.revision_id().clone(),
            twin_revision_id: twin.revision_id().clone(),
            model_revision_digest,
            analysis_configuration_digest,
            dimensions: normalized,
        })
    }

    pub fn revision_id(&self) -> &ValidityDomainRevisionIdV1 {
        &self.revision_id
    }

    pub fn subject_revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.subject_revision_id
    }

    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }

    pub fn audit_record_v1(&self) -> Value {
        let dimensions = self
            .dimensions
            .iter()
            .map(|(name, digest)| (name.clone(), digest.as_str().to_string()))
            .collect::<BTreeMap<_, _>>();
        json!({
            "analysis_configuration_digest": self.analysis_configuration_digest.as_str(),
            "authority": "validity-binding-only",
            "dimensions": dimensions,
            "model_revision_digest": self.model_revision_digest.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.revision_id.as_str(),
        })
    }
}

/// Bounded present-applicability assertion.
///
/// The attestation remains an unauthenticated content premise here, but the
/// semantic identity cannot represent an unbounded validity interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentnessAssertionV2 {
    assertion_id: CurrentnessAssertionIdV2,
    twin_revision_id: TwinRevisionIdV1,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    attestation_digest: CurrentnessAttestationDigestV1,
    observed_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl CurrentnessAssertionV2 {
    pub fn new(
        twin: &TwinRevisionV1,
        validity_domain: &ValidityDomainRevisionV1,
        attestation_digest: CurrentnessAttestationDigestV1,
        observed_at_unix_ms: u64,
        valid_until_unix_ms: u64,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if validity_domain.twin_revision_id() != twin.revision_id() {
            return Err(AnalysisTrustErrorV1::ValidityContextMismatch);
        }
        if valid_until_unix_ms <= observed_at_unix_ms {
            return Err(AnalysisTrustErrorV1::InvalidCurrentnessWindow);
        }
        let preimage = json!({
            "attestation_digest": attestation_digest.as_str(),
            "observed_at_unix_ms": observed_at_unix_ms,
            "schema": "symthaea.etk-currentness-assertion.v2",
            "twin_revision_id": twin.revision_id().as_str(),
            "valid_until_unix_ms": valid_until_unix_ms,
            "validity_domain_revision_id": validity_domain.revision_id().as_str(),
        });
        Ok(Self {
            assertion_id: CurrentnessAssertionIdV2::from_digest(domain_hash(
                CURRENTNESS_DOMAIN_V2,
                &preimage,
            )),
            twin_revision_id: twin.revision_id().clone(),
            validity_domain_revision_id: validity_domain.revision_id().clone(),
            attestation_digest,
            observed_at_unix_ms,
            valid_until_unix_ms,
        })
    }

    pub fn assertion_id(&self) -> &CurrentnessAssertionIdV2 {
        &self.assertion_id
    }

    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }

    pub fn validity_domain_revision_id(&self) -> &ValidityDomainRevisionIdV1 {
        &self.validity_domain_revision_id
    }

    pub fn observed_at_unix_ms(&self) -> u64 {
        self.observed_at_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "attestation_digest": self.attestation_digest.as_str(),
            "authority": "bounded-currentness-binding-only",
            "currentness_assertion_id": self.assertion_id.as_str(),
            "observed_at_unix_ms": self.observed_at_unix_ms,
            "schema": "symthaea.etk-currentness-assertion.v2",
            "twin_revision_id": self.twin_revision_id.as_str(),
            "valid_until_unix_ms": self.valid_until_unix_ms,
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
        })
    }
}

pub fn analytical_obligation_revision_v1(
    obligation: &ProofObligation,
) -> Result<ObligationRevisionIdV1, AnalysisTrustErrorV1> {
    if obligation.expected_evidence != EvidenceKind::Analysis {
        return Err(AnalysisTrustErrorV1::NotAnalysisObligation);
    }
    let preimage = json!({
        "claim": obligation.claim.as_str(),
        "expected_evidence_kind": EvidenceKind::Analysis.canonical_name(),
        "obligation_id": obligation.id.to_string(),
        "schema": "symthaea.etk-proof-obligation-snapshot.v1",
    });
    Ok(ObligationRevisionIdV1::from_digest(domain_hash(
        OBLIGATION_DOMAIN_V1,
        &preimage,
    )))
}

fn canonical_invariants(
    invariants: impl IntoIterator<Item = String>,
) -> Result<Vec<String>, AnalysisTrustErrorV1> {
    let mut seen = BTreeSet::new();
    let mut values = Vec::new();
    for invariant in invariants {
        let invariant = canonical_text(invariant, "structural invariant")?;
        if !seen.insert(invariant.clone()) {
            return Err(AnalysisTrustErrorV1::InvalidText("duplicate structural invariant"));
        }
        values.push(invariant);
    }
    values.sort();
    Ok(values)
}

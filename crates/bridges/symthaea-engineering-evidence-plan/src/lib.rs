// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed semantic bindings for the Engineering Trust Kernel.
//!
//! This crate freezes the ETK-3B theorem:
//!
//! ```text
//! matching labels != matching engineering semantics
//! ```
//!
//! An engineering evidence plan binds exact, independently content-addressed
//! revisions of the requirement, subject, twin, simulation request, evidence
//! policy, validity domain, proof obligation, currentness assertion, and rendered
//! solver input.  It does not admit evidence or issue a discharge receipt.
//!
//! A syntactically valid SHA-256 identifier is only an identity.  It does not by
//! itself authenticate the producer, prove currentness, prove physical truth, or
//! grant any engineering authority.

#![deny(unsafe_code)]

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_engineering_trust::proof_obligation_snapshot_id_v1;
use symthaea_formal_safety::{EvidenceKind, ProofObligation};
use symthaea_sim_bridge::{
    EngineeringDomain, SimulationRequest, SolverKind, UncertaintyEstimate,
};
use thiserror::Error;

const REQUIREMENT_DOMAIN_V1: &[u8] = b"symthaea.etk-accepted-requirement.v1\0";
const SUBJECT_DOMAIN_V1: &[u8] = b"symthaea.etk-engineering-subject.v1\0";
const TWIN_DOMAIN_V1: &[u8] = b"symthaea.etk-twin-revision.v1\0";
const REQUEST_DOMAIN_V1: &[u8] = b"symthaea.etk-simulation-request.v1\0";
const POLICY_DOMAIN_V1: &[u8] = b"symthaea.etk-simulation-evidence-policy.v1\0";
const VALIDITY_DOMAIN_V1: &[u8] = b"symthaea.etk-validity-domain.v1\0";
const CURRENTNESS_DOMAIN_V1: &[u8] = b"symthaea.etk-currentness-assertion.v1\0";
const EVIDENCE_PLAN_DOMAIN_V1: &[u8] = b"symthaea.etk-simulation-evidence-plan.v1\0";

/// Errors at the semantic-identity / evidence-plan boundary.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EvidencePlanErrorV1 {
    #[error("{0} cannot be empty or have leading/trailing whitespace")]
    InvalidText(&'static str),
    #[error("invalid SHA-256 identity: {0}")]
    InvalidDigest(String),
    #[error("{0} must be finite")]
    NonFinite(&'static str),
    #[error("{0} must be within [0, 1]")]
    UnitInterval(&'static str),
    #[error("duplicate structural invariant: {0}")]
    DuplicateInvariant(String),
    #[error("duplicate simulation parameter name: {0}")]
    DuplicateParameter(String),
    #[error("duplicate requested metric: {0}")]
    DuplicateMetric(String),
    #[error("duplicate exact warning allowance: {0}")]
    DuplicateWarning(String),
    #[error("duplicate validity-domain dimension: {0}")]
    DuplicateValidityDimension(String),
    #[error("invalid simulation request: {0}")]
    InvalidSimulationRequest(String),
    #[error("twin revision does not belong to the supplied subject revision")]
    TwinSubjectMismatch,
    #[error("twin parent belongs to a different subject or twin kind")]
    TwinLineageMismatch,
    #[error("validity domain does not bind the supplied subject/twin revisions")]
    ValidityContextMismatch,
    #[error("currentness assertion does not bind the supplied twin/validity revisions")]
    CurrentnessContextMismatch,
    #[error("proof obligation does not expect Simulation evidence")]
    NotSimulationObligation,
    #[error("accepted requirement does not expect Simulation evidence")]
    RequirementNotSimulation,
    #[error("accepted requirement and simulation request use different engineering domains")]
    RequirementDomainMismatch,
    #[error("required policy metric is not requested by the bound simulation request")]
    RequiredMetricNotRequested,
}

/// Strict textual SHA-256 identity (`sha256:` + 64 lowercase hexadecimal digits).
///
/// Parsing proves only canonical syntax.  The caller still owns the theorem that
/// the digest was computed over the intended artifact with the intended domain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256DigestV1(String);

impl Sha256DigestV1 {
    pub fn parse(value: impl Into<String>) -> Result<Self, EvidencePlanErrorV1> {
        let value = value.into();
        let Some(hex_part) = value.strip_prefix("sha256:") else {
            return Err(EvidencePlanErrorV1::InvalidDigest(value));
        };
        if hex_part.len() != 64
            || !hex_part
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(EvidencePlanErrorV1::InvalidDigest(value));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256DigestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

macro_rules! semantic_id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            fn from_digest(digest: Sha256DigestV1) -> Self {
                Self(digest)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

semantic_id_type!(AcceptedRequirementRevisionIdV1);
semantic_id_type!(SubjectRevisionIdV1);
semantic_id_type!(TwinRevisionIdV1);
semantic_id_type!(SimulationRequestRevisionIdV1);
semantic_id_type!(EvidencePolicyRevisionIdV1);
semantic_id_type!(ValidityDomainRevisionIdV1);
semantic_id_type!(CurrentnessAssertionIdV1);
semantic_id_type!(EvidencePlanIdV1);
semantic_id_type!(ObligationRevisionIdV1);

impl ObligationRevisionIdV1 {
    /// Wrap the ETK-2B content-addressed obligation theorem in a non-interchangeable type.
    pub fn for_obligation(obligation: &ProofObligation) -> Result<Self, EvidencePlanErrorV1> {
        let digest = Sha256DigestV1::parse(proof_obligation_snapshot_id_v1(obligation))?;
        Ok(Self::from_digest(digest))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequirementCriticalityV1 {
    Low,
    Medium,
    High,
    Blocking,
}

impl RequirementCriticalityV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Blocking => "Blocking",
        }
    }
}

/// Exact accepted requirement semantics.
///
/// `acceptance_record_digest` commits to the acceptance record but does not
/// authenticate who accepted it; signer/organizational authority remains a
/// separate ETK boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedRequirementRevisionV1 {
    revision_id: AcceptedRequirementRevisionIdV1,
    logical_requirement_id: String,
    domain: EngineeringDomain,
    statement: String,
    criticality: RequirementCriticalityV1,
    expected_evidence_kind: String,
    structural_invariants: Vec<String>,
    acceptance_record_digest: Sha256DigestV1,
}

impl AcceptedRequirementRevisionV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        logical_requirement_id: impl Into<String>,
        domain: EngineeringDomain,
        statement: impl Into<String>,
        criticality: RequirementCriticalityV1,
        expected_evidence: EvidenceKind,
        structural_invariants: impl IntoIterator<Item = impl Into<String>>,
        acceptance_record_digest: Sha256DigestV1,
    ) -> Result<Self, EvidencePlanErrorV1> {
        let logical_requirement_id = canonical_text(
            logical_requirement_id.into(),
            "logical requirement id",
        )?;
        let statement = canonical_text(statement.into(), "requirement statement")?;
        let mut invariants = Vec::new();
        let mut seen = BTreeSet::new();
        for invariant in structural_invariants {
            let invariant = canonical_text(invariant.into(), "structural invariant")?;
            if !seen.insert(invariant.clone()) {
                return Err(EvidencePlanErrorV1::DuplicateInvariant(invariant));
            }
            invariants.push(invariant);
        }
        invariants.sort();
        let expected_evidence_kind = evidence_kind_name(&expected_evidence).to_string();
        let preimage = json!({
            "acceptance_record_digest": acceptance_record_digest.as_str(),
            "criticality": criticality.as_str(),
            "domain": engineering_domain_name(domain),
            "expected_evidence_kind": expected_evidence_kind,
            "logical_requirement_id": logical_requirement_id,
            "schema": "symthaea.etk-accepted-requirement.v1",
            "statement": statement,
            "structural_invariants": invariants,
        });
        let revision_id = AcceptedRequirementRevisionIdV1::from_digest(domain_hash(
            REQUIREMENT_DOMAIN_V1,
            &preimage,
        ));
        Ok(Self {
            revision_id,
            logical_requirement_id,
            domain,
            statement,
            criticality,
            expected_evidence_kind,
            structural_invariants: invariants,
            acceptance_record_digest,
        })
    }

    pub fn revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.revision_id
    }

    pub fn logical_requirement_id(&self) -> &str {
        &self.logical_requirement_id
    }

    pub fn domain(&self) -> EngineeringDomain {
        self.domain
    }

    pub fn statement(&self) -> &str {
        &self.statement
    }

    pub fn criticality(&self) -> RequirementCriticalityV1 {
        self.criticality
    }

    pub fn expected_evidence_kind(&self) -> &str {
        &self.expected_evidence_kind
    }

    pub fn structural_invariants(&self) -> &[String] {
        &self.structural_invariants
    }

    pub fn acceptance_record_digest(&self) -> &Sha256DigestV1 {
        &self.acceptance_record_digest
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "acceptance_record_digest": self.acceptance_record_digest.as_str(),
            "criticality": self.criticality.as_str(),
            "domain": engineering_domain_name(self.domain),
            "expected_evidence_kind": self.expected_evidence_kind,
            "logical_requirement_id": self.logical_requirement_id,
            "revision_id": self.revision_id.as_str(),
            "statement": self.statement,
            "structural_invariants": self.structural_invariants,
        })
    }
}

/// Content-addressed subject state.  The external state digest is syntax-bound,
/// not authenticated by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectRevisionV1 {
    revision_id: SubjectRevisionIdV1,
    namespace: String,
    subject_key: String,
    state_digest: Sha256DigestV1,
}

impl SubjectRevisionV1 {
    pub fn new(
        namespace: impl Into<String>,
        subject_key: impl Into<String>,
        state_digest: Sha256DigestV1,
    ) -> Result<Self, EvidencePlanErrorV1> {
        let namespace = canonical_text(namespace.into(), "subject namespace")?;
        let subject_key = canonical_text(subject_key.into(), "subject key")?;
        let preimage = json!({
            "namespace": namespace,
            "schema": "symthaea.etk-engineering-subject.v1",
            "state_digest": state_digest.as_str(),
            "subject_key": subject_key,
        });
        let revision_id = SubjectRevisionIdV1::from_digest(domain_hash(SUBJECT_DOMAIN_V1, &preimage));
        Ok(Self {
            revision_id,
            namespace,
            subject_key,
            state_digest,
        })
    }

    pub fn revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.revision_id
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn subject_key(&self) -> &str {
        &self.subject_key
    }

    pub fn state_digest(&self) -> &Sha256DigestV1 {
        &self.state_digest
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

/// One exact design/as-built/operational twin revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwinRevisionV1 {
    revision_id: TwinRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    kind: TwinKindV1,
    state_digest: Sha256DigestV1,
    schema_digest: Sha256DigestV1,
    parent_revision_id: Option<TwinRevisionIdV1>,
}

impl TwinRevisionV1 {
    pub fn new(
        subject: &SubjectRevisionV1,
        kind: TwinKindV1,
        state_digest: Sha256DigestV1,
        schema_digest: Sha256DigestV1,
        parent: Option<&TwinRevisionV1>,
    ) -> Result<Self, EvidencePlanErrorV1> {
        if let Some(parent) = parent
            && (parent.subject_revision_id != *subject.revision_id() || parent.kind != kind)
        {
            return Err(EvidencePlanErrorV1::TwinLineageMismatch);
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
        let revision_id = TwinRevisionIdV1::from_digest(domain_hash(TWIN_DOMAIN_V1, &preimage));
        Ok(Self {
            revision_id,
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

    pub fn kind(&self) -> TwinKindV1 {
        self.kind
    }

    pub fn state_digest(&self) -> &Sha256DigestV1 {
        &self.state_digest
    }

    pub fn schema_digest(&self) -> &Sha256DigestV1 {
        &self.schema_digest
    }

    pub fn parent_revision_id(&self) -> Option<&TwinRevisionIdV1> {
        self.parent_revision_id.as_ref()
    }
}

/// Content-addressed normalized `SimulationRequest`.
///
/// Parameter order and requested-metric order are canonicalized because they do
/// not change the intended request semantics. Duplicate names are rejected rather
/// than silently collapsed. Parameter provenance text is committed but not
/// authenticated; provenance authentication remains a later ETK theorem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationRequestRevisionV1 {
    revision_id: SimulationRequestRevisionIdV1,
    logical_request_id: String,
    domain: EngineeringDomain,
    solver: SolverKind,
    requested_metrics: Vec<String>,
}

impl SimulationRequestRevisionV1 {
    pub fn from_request(request: &SimulationRequest) -> Result<Self, EvidencePlanErrorV1> {
        request
            .validate()
            .map_err(|error| EvidencePlanErrorV1::InvalidSimulationRequest(error.to_string()))?;
        canonical_text(request.id.clone(), "simulation request id")?;
        canonical_text(request.objective.clone(), "simulation objective")?;

        let mut parameters = request.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|left, right| left.name.cmp(&right.name));
        let mut seen_parameters = BTreeSet::new();
        let mut parameter_values = Vec::with_capacity(parameters.len());
        for parameter in parameters {
            canonical_text(parameter.name.clone(), "simulation parameter name")?;
            canonical_text(parameter.unit.clone(), "simulation parameter unit")?;
            canonical_text(parameter.provenance.clone(), "simulation parameter provenance")?;
            if !seen_parameters.insert(parameter.name.clone()) {
                return Err(EvidencePlanErrorV1::DuplicateParameter(parameter.name.clone()));
            }
            if !parameter.value.is_finite() {
                return Err(EvidencePlanErrorV1::NonFinite("simulation parameter value"));
            }
            parameter_values.push(json!({
                "name": parameter.name,
                "provenance": parameter.provenance,
                "uncertainty": parameter.uncertainty.map(uncertainty_value).unwrap_or(Value::Null),
                "unit": parameter.unit,
                "value": parameter.value,
            }));
        }

        let mut requested_metrics = request.requested_metrics.clone();
        for metric in &requested_metrics {
            canonical_text(metric.clone(), "requested metric")?;
        }
        requested_metrics.sort();
        for pair in requested_metrics.windows(2) {
            if pair[0] == pair[1] {
                return Err(EvidencePlanErrorV1::DuplicateMetric(pair[0].clone()));
            }
        }

        let preimage = json!({
            "domain": engineering_domain_name(request.domain),
            "logical_request_id": request.id,
            "objective": request.objective,
            "parameters": parameter_values,
            "requested_metrics": requested_metrics,
            "schema": "symthaea.etk-simulation-request.v1",
            "solver": solver_kind_name(request.solver),
        });
        let revision_id = SimulationRequestRevisionIdV1::from_digest(domain_hash(
            REQUEST_DOMAIN_V1,
            &preimage,
        ));
        Ok(Self {
            revision_id,
            logical_request_id: request.id.clone(),
            domain: request.domain,
            solver: request.solver,
            requested_metrics,
        })
    }

    pub fn revision_id(&self) -> &SimulationRequestRevisionIdV1 {
        &self.revision_id
    }

    pub fn logical_request_id(&self) -> &str {
        &self.logical_request_id
    }

    pub fn domain(&self) -> EngineeringDomain {
        self.domain
    }

    pub fn solver(&self) -> SolverKind {
        self.solver
    }

    pub fn requested_metrics(&self) -> &[String] {
        &self.requested_metrics
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricOperatorV1 {
    Lt,
    Le,
    Gt,
    Ge,
}

impl MetricOperatorV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }
}

/// Exact scalar acceptance predicate for the evidence policy.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricPredicateV1 {
    name: String,
    unit: String,
    operator: MetricOperatorV1,
    threshold: f64,
    max_epistemic: f64,
    max_aleatoric: f64,
}

impl MetricPredicateV1 {
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        operator: MetricOperatorV1,
        threshold: f64,
        max_epistemic: f64,
        max_aleatoric: f64,
    ) -> Result<Self, EvidencePlanErrorV1> {
        let name = canonical_text(name.into(), "required metric name")?;
        let unit = canonical_text(unit.into(), "required metric unit")?;
        if !threshold.is_finite() {
            return Err(EvidencePlanErrorV1::NonFinite("required metric threshold"));
        }
        check_unit_interval(max_epistemic, "maximum epistemic uncertainty")?;
        check_unit_interval(max_aleatoric, "maximum aleatoric uncertainty")?;
        Ok(Self {
            name,
            unit,
            operator,
            threshold,
            max_epistemic,
            max_aleatoric,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }

    pub fn operator(&self) -> MetricOperatorV1 {
        self.operator
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn max_epistemic(&self) -> f64 {
        self.max_epistemic
    }

    pub fn max_aleatoric(&self) -> f64 {
        self.max_aleatoric
    }

    fn as_value(&self) -> Value {
        json!({
            "max_aleatoric": self.max_aleatoric,
            "max_epistemic": self.max_epistemic,
            "name": self.name,
            "operator": self.operator.as_str(),
            "threshold": self.threshold,
            "unit": self.unit,
        })
    }
}

/// How free-form solver warnings affect *automatic* evidence admission.
///
/// `AllowExact` is a transitional compatibility mode for today's unstructured
/// `Vec<String>` warning channel. Structured warning codes should replace it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningPolicyV1 {
    DenyAny,
    ReviewRequired,
    AllowExact(Vec<String>),
}

/// Content-addressed evidence-policy semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationEvidencePolicyRevisionV1 {
    revision_id: EvidencePolicyRevisionIdV1,
    policy_label: String,
    required_metric: MetricPredicateV1,
    warning_policy: WarningPolicyV1,
}

impl SimulationEvidencePolicyRevisionV1 {
    pub fn new(
        policy_label: impl Into<String>,
        required_metric: MetricPredicateV1,
        warning_policy: WarningPolicyV1,
    ) -> Result<Self, EvidencePlanErrorV1> {
        let policy_label = canonical_text(policy_label.into(), "evidence policy label")?;
        let warning_policy = normalize_warning_policy(warning_policy)?;
        let preimage = json!({
            "execution_mode": "external_solver",
            "policy_label": policy_label,
            "required_metric": required_metric.as_value(),
            "schema": "symthaea.etk-simulation-evidence-policy.v1",
            "warning_policy": warning_policy_value(&warning_policy),
        });
        let revision_id = EvidencePolicyRevisionIdV1::from_digest(domain_hash(
            POLICY_DOMAIN_V1,
            &preimage,
        ));
        Ok(Self {
            revision_id,
            policy_label,
            required_metric,
            warning_policy,
        })
    }

    pub fn revision_id(&self) -> &EvidencePolicyRevisionIdV1 {
        &self.revision_id
    }

    pub fn policy_label(&self) -> &str {
        &self.policy_label
    }

    pub fn required_metric(&self) -> &MetricPredicateV1 {
        &self.required_metric
    }

    pub fn warning_policy(&self) -> &WarningPolicyV1 {
        &self.warning_policy
    }
}

/// Exact applicability domain for one evidence plan.
///
/// Core model and solver-configuration revisions are mandatory; additional
/// domain-specific dimensions are an ordered map of semantic labels to exact
/// SHA-256 identities.  The labels are committed but are not a substitute for a
/// future shared ontology of validity dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityDomainRevisionV1 {
    revision_id: ValidityDomainRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    model_revision_digest: Sha256DigestV1,
    solver_configuration_digest: Sha256DigestV1,
    dimensions: BTreeMap<String, Sha256DigestV1>,
}

impl ValidityDomainRevisionV1 {
    pub fn new(
        subject: &SubjectRevisionV1,
        twin: &TwinRevisionV1,
        model_revision_digest: Sha256DigestV1,
        solver_configuration_digest: Sha256DigestV1,
        dimensions: impl IntoIterator<Item = (String, Sha256DigestV1)>,
    ) -> Result<Self, EvidencePlanErrorV1> {
        if twin.subject_revision_id() != subject.revision_id() {
            return Err(EvidencePlanErrorV1::TwinSubjectMismatch);
        }
        let mut normalized = BTreeMap::new();
        for (name, digest) in dimensions {
            let name = canonical_text(name, "validity dimension")?;
            if normalized.insert(name.clone(), digest).is_some() {
                return Err(EvidencePlanErrorV1::DuplicateValidityDimension(name));
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
            "solver_configuration_digest": solver_configuration_digest.as_str(),
            "subject_revision_id": subject.revision_id().as_str(),
            "twin_revision_id": twin.revision_id().as_str(),
        });
        let revision_id = ValidityDomainRevisionIdV1::from_digest(domain_hash(
            VALIDITY_DOMAIN_V1,
            &preimage,
        ));
        Ok(Self {
            revision_id,
            subject_revision_id: subject.revision_id().clone(),
            twin_revision_id: twin.revision_id().clone(),
            model_revision_digest,
            solver_configuration_digest,
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

    pub fn model_revision_digest(&self) -> &Sha256DigestV1 {
        &self.model_revision_digest
    }

    pub fn solver_configuration_digest(&self) -> &Sha256DigestV1 {
        &self.solver_configuration_digest
    }

    pub fn dimensions(&self) -> &BTreeMap<String, Sha256DigestV1> {
        &self.dimensions
    }
}

/// Identity of a currentness assertion, not proof that the attestation is trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentnessAssertionV1 {
    assertion_id: CurrentnessAssertionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    attestation_digest: Sha256DigestV1,
    observed_at_unix_ms: u64,
}

impl CurrentnessAssertionV1 {
    pub fn new(
        twin: &TwinRevisionV1,
        validity_domain: &ValidityDomainRevisionV1,
        attestation_digest: Sha256DigestV1,
        observed_at_unix_ms: u64,
    ) -> Result<Self, EvidencePlanErrorV1> {
        if validity_domain.twin_revision_id() != twin.revision_id() {
            return Err(EvidencePlanErrorV1::ValidityContextMismatch);
        }
        let preimage = json!({
            "attestation_digest": attestation_digest.as_str(),
            "observed_at_unix_ms": observed_at_unix_ms,
            "schema": "symthaea.etk-currentness-assertion.v1",
            "twin_revision_id": twin.revision_id().as_str(),
            "validity_domain_revision_id": validity_domain.revision_id().as_str(),
        });
        let assertion_id = CurrentnessAssertionIdV1::from_digest(domain_hash(
            CURRENTNESS_DOMAIN_V1,
            &preimage,
        ));
        Ok(Self {
            assertion_id,
            twin_revision_id: twin.revision_id().clone(),
            validity_domain_revision_id: validity_domain.revision_id().clone(),
            attestation_digest,
            observed_at_unix_ms,
        })
    }

    pub fn assertion_id(&self) -> &CurrentnessAssertionIdV1 {
        &self.assertion_id
    }

    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }

    pub fn validity_domain_revision_id(&self) -> &ValidityDomainRevisionIdV1 {
        &self.validity_domain_revision_id
    }

    pub fn attestation_digest(&self) -> &Sha256DigestV1 {
        &self.attestation_digest
    }

    pub fn observed_at_unix_ms(&self) -> u64 {
        self.observed_at_unix_ms
    }
}

/// Exact semantic binding that says which computation *may later be considered*
/// as evidence for which obligation.  It grants no admission or discharge authority.
#[must_use = "an evidence plan is a binding plan, not admitted evidence or a discharge receipt"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationEvidencePlanV1 {
    plan_id: EvidencePlanIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    request_id: String,
    request_revision_id: SimulationRequestRevisionIdV1,
    evidence_policy_revision_id: EvidencePolicyRevisionIdV1,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    currentness_assertion_id: CurrentnessAssertionIdV1,
    expected_rendered_input_digest: Sha256DigestV1,
}

impl SimulationEvidencePlanV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &SubjectRevisionV1,
        twin: &TwinRevisionV1,
        requirement: &AcceptedRequirementRevisionV1,
        obligation: &ProofObligation,
        request: &SimulationRequestRevisionV1,
        evidence_policy: &SimulationEvidencePolicyRevisionV1,
        validity_domain: &ValidityDomainRevisionV1,
        currentness: &CurrentnessAssertionV1,
        expected_rendered_input_digest: Sha256DigestV1,
    ) -> Result<Self, EvidencePlanErrorV1> {
        if evidence_kind_name(&obligation.expected_evidence) != "Simulation" {
            return Err(EvidencePlanErrorV1::NotSimulationObligation);
        }
        if requirement.expected_evidence_kind() != "Simulation" {
            return Err(EvidencePlanErrorV1::RequirementNotSimulation);
        }
        if requirement.domain() != request.domain() {
            return Err(EvidencePlanErrorV1::RequirementDomainMismatch);
        }
        if twin.subject_revision_id() != subject.revision_id()
            || validity_domain.subject_revision_id() != subject.revision_id()
            || validity_domain.twin_revision_id() != twin.revision_id()
        {
            return Err(EvidencePlanErrorV1::ValidityContextMismatch);
        }
        if currentness.twin_revision_id() != twin.revision_id()
            || currentness.validity_domain_revision_id() != validity_domain.revision_id()
        {
            return Err(EvidencePlanErrorV1::CurrentnessContextMismatch);
        }
        if !request
            .requested_metrics()
            .iter()
            .any(|metric| metric == evidence_policy.required_metric().name())
        {
            return Err(EvidencePlanErrorV1::RequiredMetricNotRequested);
        }

        let obligation_revision_id = ObligationRevisionIdV1::for_obligation(obligation)?;
        let obligation_id = obligation.id.to_string();
        let preimage = json!({
            "currentness_assertion_id": currentness.assertion_id().as_str(),
            "evidence_policy_revision_id": evidence_policy.revision_id().as_str(),
            "expected_rendered_input_digest": expected_rendered_input_digest.as_str(),
            "obligation_id": obligation_id,
            "obligation_revision_id": obligation_revision_id.as_str(),
            "request_id": request.logical_request_id(),
            "request_revision_id": request.revision_id().as_str(),
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": "symthaea.etk-simulation-evidence-plan.v1",
            "subject_revision_id": subject.revision_id().as_str(),
            "twin_revision_id": twin.revision_id().as_str(),
            "validity_domain_revision_id": validity_domain.revision_id().as_str(),
        });
        let plan_id = EvidencePlanIdV1::from_digest(domain_hash(EVIDENCE_PLAN_DOMAIN_V1, &preimage));
        Ok(Self {
            plan_id,
            subject_revision_id: subject.revision_id().clone(),
            twin_revision_id: twin.revision_id().clone(),
            requirement_revision_id: requirement.revision_id().clone(),
            obligation_id,
            obligation_revision_id,
            request_id: request.logical_request_id().to_string(),
            request_revision_id: request.revision_id().clone(),
            evidence_policy_revision_id: evidence_policy.revision_id().clone(),
            validity_domain_revision_id: validity_domain.revision_id().clone(),
            currentness_assertion_id: currentness.assertion_id().clone(),
            expected_rendered_input_digest,
        })
    }

    pub fn plan_id(&self) -> &EvidencePlanIdV1 {
        &self.plan_id
    }

    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }

    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }

    pub fn request_revision_id(&self) -> &SimulationRequestRevisionIdV1 {
        &self.request_revision_id
    }

    pub fn evidence_policy_revision_id(&self) -> &EvidencePolicyRevisionIdV1 {
        &self.evidence_policy_revision_id
    }

    pub fn validity_domain_revision_id(&self) -> &ValidityDomainRevisionIdV1 {
        &self.validity_domain_revision_id
    }

    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }

    pub fn currentness_assertion_id(&self) -> &CurrentnessAssertionIdV1 {
        &self.currentness_assertion_id
    }

    pub fn expected_rendered_input_digest(&self) -> &Sha256DigestV1 {
        &self.expected_rendered_input_digest
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "authority": "binding-plan-only",
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "evidence_policy_revision_id": self.evidence_policy_revision_id.as_str(),
            "expected_rendered_input_digest": self.expected_rendered_input_digest.as_str(),
            "obligation_id": self.obligation_id,
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "plan_id": self.plan_id.as_str(),
            "request_id": self.request_id,
            "request_revision_id": self.request_revision_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
        })
    }
}

fn canonical_text(value: String, field: &'static str) -> Result<String, EvidencePlanErrorV1> {
    if value.is_empty() || value.trim() != value {
        Err(EvidencePlanErrorV1::InvalidText(field))
    } else {
        Ok(value)
    }
}

fn check_unit_interval(value: f64, field: &'static str) -> Result<(), EvidencePlanErrorV1> {
    if !value.is_finite() {
        return Err(EvidencePlanErrorV1::NonFinite(field));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(EvidencePlanErrorV1::UnitInterval(field));
    }
    Ok(())
}

fn normalize_warning_policy(
    policy: WarningPolicyV1,
) -> Result<WarningPolicyV1, EvidencePlanErrorV1> {
    match policy {
        WarningPolicyV1::AllowExact(mut warnings) => {
            for warning in &warnings {
                canonical_text(warning.clone(), "exact warning allowance")?;
            }
            warnings.sort();
            for pair in warnings.windows(2) {
                if pair[0] == pair[1] {
                    return Err(EvidencePlanErrorV1::DuplicateWarning(pair[0].clone()));
                }
            }
            Ok(WarningPolicyV1::AllowExact(warnings))
        }
        other => Ok(other),
    }
}

fn warning_policy_value(policy: &WarningPolicyV1) -> Value {
    match policy {
        WarningPolicyV1::DenyAny => json!({"mode": "deny_any"}),
        WarningPolicyV1::ReviewRequired => json!({"mode": "review_required"}),
        WarningPolicyV1::AllowExact(warnings) => json!({
            "allowed_exact_messages": warnings,
            "mode": "allow_exact",
        }),
    }
}

fn uncertainty_value(uncertainty: UncertaintyEstimate) -> Value {
    let interval = uncertainty.interval.map_or(Value::Null, |interval| {
        json!({"lower": interval.lower, "upper": interval.upper})
    });
    json!({
        "aleatoric": uncertainty.aleatoric,
        "epistemic": uncertainty.epistemic,
        "interval": interval,
    })
}

fn engineering_domain_name(domain: EngineeringDomain) -> &'static str {
    match domain {
        EngineeringDomain::Civil => "Civil",
        EngineeringDomain::Mechanical => "Mechanical",
        EngineeringDomain::Electrical => "Electrical",
        EngineeringDomain::Aerospace => "Aerospace",
        EngineeringDomain::ChemicalProcess => "ChemicalProcess",
        EngineeringDomain::Robotics => "Robotics",
        EngineeringDomain::Nuclear => "Nuclear",
        EngineeringDomain::Materials => "Materials",
        EngineeringDomain::Environmental => "Environmental",
        EngineeringDomain::Systems => "Systems",
    }
}

fn solver_kind_name(solver: SolverKind) -> &'static str {
    match solver {
        SolverKind::FiniteElement => "FiniteElement",
        SolverKind::ComputationalFluidDynamics => "ComputationalFluidDynamics",
        SolverKind::MultibodyDynamics => "MultibodyDynamics",
        SolverKind::Circuit => "Circuit",
        SolverKind::Process => "Process",
        SolverKind::CadGeometry => "CadGeometry",
        SolverKind::MultiPhysics => "MultiPhysics",
        SolverKind::Custom => "Custom",
    }
}

fn evidence_kind_name(kind: &EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::FormalProof => "FormalProof",
        EvidenceKind::Simulation => "Simulation",
        EvidenceKind::Test => "Test",
        EvidenceKind::Telemetry => "Telemetry",
        EvidenceKind::Standard => "Standard",
    }
}

fn domain_hash(domain: &[u8], value: &Value) -> Sha256DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(canonical_json(value).as_bytes());
    Sha256DigestV1(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value)
            .expect("serializing an in-memory JSON string cannot fail"),
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let encoded_key = serde_json::to_string(key)
                        .expect("serializing an in-memory JSON key cannot fail");
                    format!("{encoded_key}:{}", canonical_json(&map[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{Interval, ModelParameter};

    fn digest(ch: char) -> Sha256DigestV1 {
        Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
    }

    fn requirement(statement: &str, invariants: &[&str]) -> AcceptedRequirementRevisionV1 {
        AcceptedRequirementRevisionV1::new(
            "REQ-STRESS",
            EngineeringDomain::Civil,
            statement,
            RequirementCriticalityV1::Blocking,
            EvidenceKind::Simulation,
            invariants.iter().copied(),
            digest('a'),
        )
        .unwrap()
    }

    fn request() -> SimulationRequestRevisionV1 {
        let mut raw = SimulationRequest::new(
            "sim-static-G17-LC9",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "check bracket service stress",
        )
        .with_parameter("load_n", 10_000.0, "N", "load-case:LC9")
        .with_parameter("thickness_mm", 8.0, "mm", "design:G17");
        raw.requested_metrics = vec!["max_stress_mpa".into(), "max_displacement_mm".into()];
        SimulationRequestRevisionV1::from_request(&raw).unwrap()
    }

    fn policy() -> SimulationEvidencePolicyRevisionV1 {
        SimulationEvidencePolicyRevisionV1::new(
            "service-stress-policy",
            MetricPredicateV1::new(
                "max_stress_mpa",
                "MPa",
                MetricOperatorV1::Le,
                250.0,
                0.2,
                0.1,
            )
            .unwrap(),
            WarningPolicyV1::DenyAny,
        )
        .unwrap()
    }

    struct Fixture {
        subject: SubjectRevisionV1,
        twin: TwinRevisionV1,
        validity: ValidityDomainRevisionV1,
        currentness: CurrentnessAssertionV1,
        requirement: AcceptedRequirementRevisionV1,
        obligation: ProofObligation,
        request: SimulationRequestRevisionV1,
        policy: SimulationEvidencePolicyRevisionV1,
    }

    fn fixture() -> Fixture {
        let subject = SubjectRevisionV1::new("design", "bracket-alpha", digest('b')).unwrap();
        let twin = TwinRevisionV1::new(
            &subject,
            TwinKindV1::Design,
            digest('c'),
            digest('d'),
            None,
        )
        .unwrap();
        let validity = ValidityDomainRevisionV1::new(
            &subject,
            &twin,
            digest('e'),
            digest('f'),
            vec![
                ("load_case".into(), digest('1')),
                ("material_state".into(), digest('2')),
                ("boundary_conditions".into(), digest('3')),
            ],
        )
        .unwrap();
        let currentness =
            CurrentnessAssertionV1::new(&twin, &validity, digest('4'), 1_789_123_456_000)
                .unwrap();
        Fixture {
            subject,
            twin,
            validity,
            currentness,
            requirement: requirement("stress remains below allowable", &["stress <= 250 MPa"]),
            obligation: ProofObligation::new(
                "stress remains below allowable under service load",
                EvidenceKind::Simulation,
            ),
            request: request(),
            policy: policy(),
        }
    }

    #[test]
    fn sha256_syntax_is_strict_and_canonical() {
        assert!(Sha256DigestV1::parse(format!("sha256:{}", "a".repeat(64))).is_ok());
        assert!(Sha256DigestV1::parse(format!("sha256:{}", "A".repeat(64))).is_err());
        assert!(Sha256DigestV1::parse("sha256:abc").is_err());
        assert!(Sha256DigestV1::parse(format!("blake3:{}", "a".repeat(64))).is_err());
    }

    #[test]
    fn requirement_invariant_order_is_canonical_but_semantic_change_is_not() {
        let a = requirement("stress remains below allowable", &["b", "a"]);
        let b = requirement("stress remains below allowable", &["a", "b"]);
        assert_eq!(a.revision_id(), b.revision_id());

        let changed = requirement("stress remains below revised allowable", &["a", "b"]);
        assert_ne!(a.revision_id(), changed.revision_id());
    }

    #[test]
    fn duplicate_requirement_invariant_is_denied() {
        let result = AcceptedRequirementRevisionV1::new(
            "REQ",
            EngineeringDomain::Civil,
            "statement",
            RequirementCriticalityV1::Blocking,
            EvidenceKind::Simulation,
            ["same", "same"],
            digest('a'),
        );
        assert_eq!(
            result.unwrap_err(),
            EvidencePlanErrorV1::DuplicateInvariant("same".into())
        );
    }

    #[test]
    fn request_parameter_and_metric_order_are_semantically_canonical() {
        let mut a = SimulationRequest::new(
            "r",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "objective",
        );
        a.parameters = vec![
            ModelParameter {
                name: "z".into(),
                value: 2.0,
                unit: "m".into(),
                provenance: "p2".into(),
                uncertainty: None,
            },
            ModelParameter {
                name: "a".into(),
                value: 1.0,
                unit: "m".into(),
                provenance: "p1".into(),
                uncertainty: Some(UncertaintyEstimate {
                    epistemic: 0.1,
                    aleatoric: 0.05,
                    interval: Some(Interval {
                        lower: 0.9,
                        upper: 1.1,
                    }),
                }),
            },
        ];
        a.requested_metrics = vec!["z_metric".into(), "a_metric".into()];
        let mut b = a.clone();
        b.parameters.reverse();
        b.requested_metrics.reverse();
        assert_eq!(
            SimulationRequestRevisionV1::from_request(&a)
                .unwrap()
                .revision_id(),
            SimulationRequestRevisionV1::from_request(&b)
                .unwrap()
                .revision_id()
        );
    }

    #[test]
    fn duplicate_parameter_names_are_denied() {
        let mut raw = SimulationRequest::new(
            "r",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "objective",
        );
        raw.parameters = vec![
            ModelParameter {
                name: "x".into(),
                value: 1.0,
                unit: "m".into(),
                provenance: "p1".into(),
                uncertainty: None,
            },
            ModelParameter {
                name: "x".into(),
                value: 2.0,
                unit: "m".into(),
                provenance: "p2".into(),
                uncertainty: None,
            },
        ];
        assert_eq!(
            SimulationRequestRevisionV1::from_request(&raw).unwrap_err(),
            EvidencePlanErrorV1::DuplicateParameter("x".into())
        );
    }

    #[test]
    fn validity_dimension_order_is_canonical() {
        let subject = SubjectRevisionV1::new("design", "x", digest('a')).unwrap();
        let twin =
            TwinRevisionV1::new(&subject, TwinKindV1::Design, digest('b'), digest('c'), None)
                .unwrap();
        let a = ValidityDomainRevisionV1::new(
            &subject,
            &twin,
            digest('d'),
            digest('e'),
            vec![("z".into(), digest('1')), ("a".into(), digest('2'))],
        )
        .unwrap();
        let b = ValidityDomainRevisionV1::new(
            &subject,
            &twin,
            digest('d'),
            digest('e'),
            vec![("a".into(), digest('2')), ("z".into(), digest('1'))],
        )
        .unwrap();
        assert_eq!(a.revision_id(), b.revision_id());
    }

    #[test]
    fn currentness_refresh_changes_identity_without_changing_twin() {
        let f = fixture();
        let refreshed =
            CurrentnessAssertionV1::new(&f.twin, &f.validity, digest('5'), 1_789_123_457_000)
                .unwrap();
        assert_eq!(refreshed.twin_revision_id(), f.twin.revision_id());
        assert_ne!(refreshed.assertion_id(), f.currentness.assertion_id());
    }

    #[test]
    fn evidence_plan_binds_every_semantic_revision_but_grants_no_authority() {
        let f = fixture();
        let plan = SimulationEvidencePlanV1::new(
            &f.subject,
            &f.twin,
            &f.requirement,
            &f.obligation,
            &f.request,
            &f.policy,
            &f.validity,
            &f.currentness,
            digest('6'),
        )
        .unwrap();
        let audit = plan.audit_record_v1();
        assert_eq!(audit["authority"], "binding-plan-only");
        assert_eq!(
            audit["requirement_revision_id"],
            f.requirement.revision_id().as_str()
        );
        assert_eq!(
            audit["obligation_revision_id"],
            ObligationRevisionIdV1::for_obligation(&f.obligation)
                .unwrap()
                .as_str()
        );
    }

    #[test]
    fn changing_requirement_or_currentness_changes_plan_identity() {
        let f = fixture();
        let baseline = SimulationEvidencePlanV1::new(
            &f.subject,
            &f.twin,
            &f.requirement,
            &f.obligation,
            &f.request,
            &f.policy,
            &f.validity,
            &f.currentness,
            digest('6'),
        )
        .unwrap();

        let changed_requirement =
            requirement("stress remains below revised allowable", &["stress <= 250 MPa"]);
        let changed_requirement_plan = SimulationEvidencePlanV1::new(
            &f.subject,
            &f.twin,
            &changed_requirement,
            &f.obligation,
            &f.request,
            &f.policy,
            &f.validity,
            &f.currentness,
            digest('6'),
        )
        .unwrap();
        assert_ne!(baseline.plan_id(), changed_requirement_plan.plan_id());

        let refreshed =
            CurrentnessAssertionV1::new(&f.twin, &f.validity, digest('5'), 1_789_123_457_000)
                .unwrap();
        let refreshed_plan = SimulationEvidencePlanV1::new(
            &f.subject,
            &f.twin,
            &f.requirement,
            &f.obligation,
            &f.request,
            &f.policy,
            &f.validity,
            &refreshed,
            digest('6'),
        )
        .unwrap();
        assert_ne!(baseline.plan_id(), refreshed_plan.plan_id());
    }

    #[test]
    fn mismatched_requirement_domain_is_denied() {
        let f = fixture();
        let electrical_requirement = AcceptedRequirementRevisionV1::new(
            "REQ-E",
            EngineeringDomain::Electrical,
            "voltage remains in band",
            RequirementCriticalityV1::Blocking,
            EvidenceKind::Simulation,
            std::iter::empty::<&str>(),
            digest('a'),
        )
        .unwrap();
        let result = SimulationEvidencePlanV1::new(
            &f.subject,
            &f.twin,
            &electrical_requirement,
            &f.obligation,
            &f.request,
            &f.policy,
            &f.validity,
            &f.currentness,
            digest('6'),
        );
        assert_eq!(
            result.unwrap_err(),
            EvidencePlanErrorV1::RequirementDomainMismatch
        );
    }

    #[test]
    fn policy_metric_must_be_explicitly_requested() {
        let f = fixture();
        let policy = SimulationEvidencePolicyRevisionV1::new(
            "different-metric",
            MetricPredicateV1::new(
                "fatigue_life_cycles",
                "cycles",
                MetricOperatorV1::Ge,
                1_000_000.0,
                0.2,
                0.1,
            )
            .unwrap(),
            WarningPolicyV1::DenyAny,
        )
        .unwrap();
        let result = SimulationEvidencePlanV1::new(
            &f.subject,
            &f.twin,
            &f.requirement,
            &f.obligation,
            &f.request,
            &policy,
            &f.validity,
            &f.currentness,
            digest('6'),
        );
        assert_eq!(
            result.unwrap_err(),
            EvidencePlanErrorV1::RequiredMetricNotRequested
        );
    }
}

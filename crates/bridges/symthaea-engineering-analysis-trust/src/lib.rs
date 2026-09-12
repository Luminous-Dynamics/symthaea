// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed Engineering Trust Kernel boundary for native analytical evidence.
//!
//! ```text
//! native calculation
//! != admitted analytical evidence
//! != historical discharge receipt
//! != current analytical discharge fact
//! != requirement satisfaction
//! != qualified design / certification / manufacturing / deployment / actuation
//! ```
//!
//! This crate intentionally does not reuse the external-solver `Simulation`
//! admission path. It accepts only proof obligations whose evidence class is
//! `EvidenceKind::Analysis`.

#![deny(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use symthaea_formal_safety::{EvidenceKind, ProofObligation};
use thiserror::Error;

const REQUIREMENT_DOMAIN_V1: &[u8] = b"symthaea.etk-accepted-requirement.v1\0";
const OBLIGATION_DOMAIN_V1: &[u8] = b"symthaea.etk-proof-obligation-snapshot.v1\0";
const METHOD_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-method.v1\0";
const INPUT_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-input.v1\0";
const POLICY_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-policy.v1\0";
const PLAN_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-plan.v1\0";
const ADMITTED_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-admitted-evidence.v1\0";
const RECEIPT_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-discharge-receipt.v1\0";
const FACT_DOMAIN_V1: &[u8] = b"symthaea.etk-current-native-analytical-discharge-fact.v1\0";
const EQUATION_RELATIVE_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AnalysisTrustErrorV1 {
    #[error("{0} cannot be empty or have leading/trailing whitespace")]
    InvalidText(&'static str),
    #[error("{0} is not a canonical SHA-256 identity")]
    InvalidDigest(&'static str),
    #[error("{0} must be finite")]
    NonFinite(&'static str),
    #[error("{0} must be greater than zero")]
    NonPositive(&'static str),
    #[error("{0} must be within [0, 1]")]
    UnitInterval(&'static str),
    #[error("proof obligation does not expect Analysis evidence")]
    NotAnalysisObligation,
    #[error("analytical input was created for a different method revision")]
    MethodInputMismatch,
    #[error("analytical result does not target the exact bound method/input/policy")]
    PlanBindingMismatch,
    #[error("reported {0} is inconsistent with the bound analytical input")]
    AnalyticalEquationMismatch(&'static str),
    #[error("reported factor of safety is inconsistent with yield strength / bending stress")]
    InconsistentFactorOfSafety,
    #[error("model-relative-error bound exceeds the admission policy")]
    ModelErrorBudgetExceeded,
    #[error("conservative analytical acceptance predicate failed")]
    AcceptancePredicateFailed,
    #[error("admitted evidence targets a different analytical plan")]
    AdmittedPlanMismatch,
    #[error("discharge receipt targets a historical analytical plan")]
    HistoricalPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256DigestV1(String);

impl Sha256DigestV1 {
    pub fn parse(value: impl Into<String>) -> Result<Self, AnalysisTrustErrorV1> {
        let value = value.into();
        let Some(hex_part) = value.strip_prefix("sha256:") else {
            return Err(AnalysisTrustErrorV1::InvalidDigest("SHA-256 digest"));
        };
        if hex_part.len() != 64
            || !hex_part
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AnalysisTrustErrorV1::InvalidDigest("SHA-256 digest"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256DigestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

macro_rules! premise_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            pub fn parse(value: impl Into<String>) -> Result<Self, AnalysisTrustErrorV1> {
                Sha256DigestV1::parse(value).map(Self)
            }

            pub fn from_digest(digest: Sha256DigestV1) -> Self {
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

macro_rules! authority_id {
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

premise_id!(SubjectRevisionIdV1);
premise_id!(TwinRevisionIdV1);
premise_id!(ValidityDomainRevisionIdV1);
premise_id!(CurrentnessAssertionIdV1);
premise_id!(ModelQualificationRecordDigestV1);
premise_id!(ExecutionArtifactDigestV1);

authority_id!(AnalysisRequirementRevisionIdV1);
authority_id!(ObligationRevisionIdV1);
authority_id!(AnalyticalMethodRevisionIdV1);
authority_id!(AnalyticalInputRevisionIdV1);
authority_id!(AnalyticalPolicyRevisionIdV1);
authority_id!(AnalyticalPlanIdV1);
authority_id!(AdmittedAnalyticalEvidenceIdV1);
authority_id!(NativeAnalyticalDischargeReceiptIdV1);
authority_id!(CurrentNativeAnalyticalDischargeFactIdV1);

/// Cross-language identity encoding for one finite IEEE-754 binary64 value.
/// Signed zero is intentionally normalized.
pub fn canonical_binary64_v1(value: f64) -> Result<String, AnalysisTrustErrorV1> {
    if !value.is_finite() {
        return Err(AnalysisTrustErrorV1::NonFinite("binary64"));
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    Ok(format!("f64:{:016x}", normalized.to_bits()))
}

/// Exact accepted Analysis requirement semantics for the initial Civil canary.
/// The acceptance-record digest is content-addressed but not authenticated here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedAnalysisRequirementV1 {
    revision_id: AnalysisRequirementRevisionIdV1,
    logical_requirement_id: String,
    statement: String,
    structural_invariants: Vec<String>,
    acceptance_record_digest: Sha256DigestV1,
}

impl AcceptedAnalysisRequirementV1 {
    pub fn civil_blocking(
        logical_requirement_id: impl Into<String>,
        statement: impl Into<String>,
        structural_invariants: impl IntoIterator<Item = impl Into<String>>,
        acceptance_record_digest: Sha256DigestV1,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        let logical_requirement_id =
            canonical_text(logical_requirement_id.into(), "logical requirement id")?;
        let statement = canonical_text(statement.into(), "requirement statement")?;
        let mut seen = BTreeSet::new();
        let mut invariants = Vec::new();
        for invariant in structural_invariants {
            let invariant = canonical_text(invariant.into(), "structural invariant")?;
            if !seen.insert(invariant.clone()) {
                return Err(AnalysisTrustErrorV1::InvalidText(
                    "duplicate structural invariant",
                ));
            }
            invariants.push(invariant);
        }
        invariants.sort();

        let preimage = json!({
            "acceptance_record_digest": acceptance_record_digest.as_str(),
            "criticality": "Blocking",
            "domain": "Civil",
            "expected_evidence_kind": "Analysis",
            "logical_requirement_id": logical_requirement_id.as_str(),
            "schema": "symthaea.etk-accepted-requirement.v1",
            "statement": statement.as_str(),
            "structural_invariants": invariants,
        });
        Ok(Self {
            revision_id: AnalysisRequirementRevisionIdV1::from_digest(domain_hash(
                REQUIREMENT_DOMAIN_V1,
                &preimage,
            )),
            logical_requirement_id,
            statement,
            structural_invariants: invariants,
            acceptance_record_digest,
        })
    }

    pub fn revision_id(&self) -> &AnalysisRequirementRevisionIdV1 {
        &self.revision_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "acceptance_record_digest": self.acceptance_record_digest.as_str(),
            "authority": "accepted-analysis-requirement-only",
            "criticality": "Blocking",
            "domain": "Civil",
            "expected_evidence_kind": "Analysis",
            "logical_requirement_id": self.logical_requirement_id.as_str(),
            "requirement_revision_id": self.revision_id.as_str(),
            "statement": self.statement.as_str(),
            "structural_invariants": self.structural_invariants,
        })
    }
}

/// Content-addressed Analysis proof-obligation semantics. Lifecycle state and
/// evidence refs are intentionally excluded from the semantic snapshot.
pub fn analytical_obligation_revision_v1(
    obligation: &ProofObligation,
) -> Result<ObligationRevisionIdV1, AnalysisTrustErrorV1> {
    if obligation.expected_evidence != EvidenceKind::Analysis {
        return Err(AnalysisTrustErrorV1::NotAnalysisObligation);
    }
    let preimage = json!({
        "claim": obligation.claim.as_str(),
        "expected_evidence_kind": "Analysis",
        "obligation_id": obligation.id.to_string(),
        "schema": "symthaea.etk-proof-obligation-snapshot.v1",
    });
    Ok(ObligationRevisionIdV1::from_digest(domain_hash(
        OBLIGATION_DOMAIN_V1,
        &preimage,
    )))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalMethodV1 {
    revision_id: AnalyticalMethodRevisionIdV1,
}

impl AnalyticalMethodV1 {
    /// Initial ETK-3C canary: Euler-Bernoulli rectangular-beam analysis.
    pub fn euler_bernoulli_beam(
        implementation_artifact_digest: Sha256DigestV1,
        algorithm_revision_digest: Sha256DigestV1,
    ) -> Self {
        let preimage = json!({
            "algorithm_revision_digest": algorithm_revision_digest.as_str(),
            "assumptions": [
                "euler_bernoulli_kinematics",
                "linear_elastic_material",
                "prismatic_beam",
                "single_span",
                "small_deflection",
                "statically_determinate",
            ],
            "implementation_artifact_digest": implementation_artifact_digest.as_str(),
            "method_key": "symthaea-structural/euler-bernoulli-beam",
            "outputs": [
                {"name": "factor_of_safety", "unit": "1"},
                {"name": "max_bending_stress", "unit": "Pa"},
                {"name": "max_deflection", "unit": "m"},
                {"name": "max_moment", "unit": "N*m"},
            ],
            "schema": "symthaea.etk-native-analytical-method.v1",
            "supported_load_cases": [
                "cantilever_end_point",
                "cantilever_udl",
                "simply_supported_center_point",
                "simply_supported_udl",
            ],
            "unit_system": "SI",
        });
        Self {
            revision_id: AnalyticalMethodRevisionIdV1::from_digest(domain_hash(
                METHOD_DOMAIN_V1,
                &preimage,
            )),
        }
    }

    pub fn revision_id(&self) -> &AnalyticalMethodRevisionIdV1 {
        &self.revision_id
    }
}

/// Exact canary input: rectangular cantilever with an end point load.
#[derive(Debug, Clone, PartialEq)]
pub struct RectangularCantileverInputV1 {
    revision_id: AnalyticalInputRevisionIdV1,
    method_revision_id: AnalyticalMethodRevisionIdV1,
    length_m: f64,
    width_m: f64,
    height_m: f64,
    youngs_modulus_pa: f64,
    yield_strength_pa: f64,
    load_n: f64,
}

impl RectangularCantileverInputV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        method: &AnalyticalMethodV1,
        length_m: f64,
        width_m: f64,
        height_m: f64,
        youngs_modulus_pa: f64,
        yield_strength_pa: f64,
        load_n: f64,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        for (field, value) in [
            ("beam length", length_m),
            ("section width", width_m),
            ("section height", height_m),
            ("Young's modulus", youngs_modulus_pa),
            ("yield strength", yield_strength_pa),
            ("point load", load_n),
        ] {
            if !value.is_finite() {
                return Err(AnalysisTrustErrorV1::NonFinite(field));
            }
            if value <= 0.0 {
                return Err(AnalysisTrustErrorV1::NonPositive(field));
            }
        }
        let preimage = json!({
            "beam": {
                "length_m": canonical_binary64_v1(length_m)?,
                "material": {
                    "youngs_modulus_pa": canonical_binary64_v1(youngs_modulus_pa)?,
                    "yield_strength_pa": canonical_binary64_v1(yield_strength_pa)?,
                },
                "section": {
                    "height_m": canonical_binary64_v1(height_m)?,
                    "kind": "rectangular",
                    "width_m": canonical_binary64_v1(width_m)?,
                },
            },
            "load": {
                "kind": "cantilever_end_point",
                "unit": "N",
                "value": canonical_binary64_v1(load_n)?,
            },
            "method_revision_id": method.revision_id().as_str(),
            "schema": "symthaea.etk-native-analytical-input.v1",
        });
        Ok(Self {
            revision_id: AnalyticalInputRevisionIdV1::from_digest(domain_hash(
                INPUT_DOMAIN_V1,
                &preimage,
            )),
            method_revision_id: method.revision_id().clone(),
            length_m,
            width_m,
            height_m,
            youngs_modulus_pa,
            yield_strength_pa,
            load_n,
        })
    }

    pub fn revision_id(&self) -> &AnalyticalInputRevisionIdV1 {
        &self.revision_id
    }

    pub fn method_revision_id(&self) -> &AnalyticalMethodRevisionIdV1 {
        &self.method_revision_id
    }

    pub fn yield_strength_pa(&self) -> f64 {
        self.yield_strength_pa
    }

    fn expected_max_moment_nm(&self) -> f64 {
        self.load_n * self.length_m
    }

    fn expected_max_bending_stress_pa(&self) -> f64 {
        let section_modulus = self.width_m * self.height_m.powi(2) / 6.0;
        self.expected_max_moment_nm() / section_modulus
    }

    fn expected_max_deflection_m(&self) -> f64 {
        let moment_of_inertia = self.width_m * self.height_m.powi(3) / 12.0;
        self.load_n * self.length_m.powi(3)
            / (3.0 * self.youngs_modulus_pa * moment_of_inertia)
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "authority": "analytical-input-only",
            "height_m": self.height_m,
            "input_revision_id": self.revision_id.as_str(),
            "length_m": self.length_m,
            "load_n": self.load_n,
            "method_revision_id": self.method_revision_id.as_str(),
            "width_m": self.width_m,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "yield_strength_pa": self.yield_strength_pa,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyticalAcceptancePolicyV1 {
    revision_id: AnalyticalPolicyRevisionIdV1,
    threshold: f64,
    max_model_relative_error_bound: f64,
    model_qualification_record_digest: ModelQualificationRecordDigestV1,
}

impl AnalyticalAcceptancePolicyV1 {
    pub fn factor_of_safety_ge(
        threshold: f64,
        max_model_relative_error_bound: f64,
        model_qualification_record_digest: ModelQualificationRecordDigestV1,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if !threshold.is_finite() {
            return Err(AnalysisTrustErrorV1::NonFinite("factor-of-safety threshold"));
        }
        if !max_model_relative_error_bound.is_finite()
            || !(0.0..=1.0).contains(&max_model_relative_error_bound)
        {
            return Err(AnalysisTrustErrorV1::UnitInterval(
                "maximum model-relative-error bound",
            ));
        }
        let preimage = json!({
            "max_model_relative_error_bound": canonical_binary64_v1(max_model_relative_error_bound)?,
            "metric": "factor_of_safety",
            "model_qualification_record_digest": model_qualification_record_digest.as_str(),
            "operator": ">=",
            "schema": "symthaea.etk-native-analytical-policy.v1",
            "threshold": canonical_binary64_v1(threshold)?,
        });
        Ok(Self {
            revision_id: AnalyticalPolicyRevisionIdV1::from_digest(domain_hash(
                POLICY_DOMAIN_V1,
                &preimage,
            )),
            threshold,
            max_model_relative_error_bound,
            model_qualification_record_digest,
        })
    }

    pub fn revision_id(&self) -> &AnalyticalPolicyRevisionIdV1 {
        &self.revision_id
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn max_model_relative_error_bound(&self) -> f64 {
        self.max_model_relative_error_bound
    }

    pub fn model_qualification_record_digest(&self) -> &ModelQualificationRecordDigestV1 {
        &self.model_qualification_record_digest
    }
}

#[must_use = "an analytical plan is binding structure, not admitted evidence"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAnalyticalPlanV1 {
    plan_id: AnalyticalPlanIdV1,
    requirement_revision_id: AnalysisRequirementRevisionIdV1,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    currentness_assertion_id: CurrentnessAssertionIdV1,
    method_revision_id: AnalyticalMethodRevisionIdV1,
    input_revision_id: AnalyticalInputRevisionIdV1,
    policy_revision_id: AnalyticalPolicyRevisionIdV1,
}

impl NativeAnalyticalPlanV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        requirement: &AcceptedAnalysisRequirementV1,
        subject_revision_id: SubjectRevisionIdV1,
        twin_revision_id: TwinRevisionIdV1,
        validity_domain_revision_id: ValidityDomainRevisionIdV1,
        currentness_assertion_id: CurrentnessAssertionIdV1,
        obligation: &ProofObligation,
        method: &AnalyticalMethodV1,
        input: &RectangularCantileverInputV1,
        policy: &AnalyticalAcceptancePolicyV1,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if input.method_revision_id() != method.revision_id() {
            return Err(AnalysisTrustErrorV1::MethodInputMismatch);
        }
        let obligation_revision_id = analytical_obligation_revision_v1(obligation)?;
        let obligation_id = obligation.id.to_string();
        let preimage = json!({
            "acceptance_policy_revision_id": policy.revision_id().as_str(),
            "currentness_assertion_id": currentness_assertion_id.as_str(),
            "input_revision_id": input.revision_id().as_str(),
            "method_revision_id": method.revision_id().as_str(),
            "obligation_id": obligation_id.as_str(),
            "obligation_revision_id": obligation_revision_id.as_str(),
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": "symthaea.etk-native-analytical-plan.v1",
            "subject_revision_id": subject_revision_id.as_str(),
            "twin_revision_id": twin_revision_id.as_str(),
            "validity_domain_revision_id": validity_domain_revision_id.as_str(),
        });
        Ok(Self {
            plan_id: AnalyticalPlanIdV1::from_digest(domain_hash(PLAN_DOMAIN_V1, &preimage)),
            requirement_revision_id: requirement.revision_id().clone(),
            obligation_id,
            obligation_revision_id,
            subject_revision_id,
            twin_revision_id,
            validity_domain_revision_id,
            currentness_assertion_id,
            method_revision_id: method.revision_id().clone(),
            input_revision_id: input.revision_id().clone(),
            policy_revision_id: policy.revision_id().clone(),
        })
    }

    pub fn plan_id(&self) -> &AnalyticalPlanIdV1 {
        &self.plan_id
    }

    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "acceptance_policy_revision_id": self.policy_revision_id.as_str(),
            "analytical_plan_id": self.plan_id.as_str(),
            "authority": "binding-plan-only",
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "input_revision_id": self.input_revision_id.as_str(),
            "method_revision_id": self.method_revision_id.as_str(),
            "obligation_id": self.obligation_id.as_str(),
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
        })
    }
}

/// Data-only candidate result. Construction binds it to one exact method/input
/// pair; it still has no authority until admission independently rechecks the
/// plan, equations, model-error budget, and acceptance predicate.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeAnalyticalResultV1 {
    method_revision_id: AnalyticalMethodRevisionIdV1,
    input_revision_id: AnalyticalInputRevisionIdV1,
    execution_artifact_digest: ExecutionArtifactDigestV1,
    factor_of_safety: f64,
    max_bending_stress_pa: f64,
    max_deflection_m: f64,
    max_moment_nm: f64,
    model_relative_error_bound: f64,
}

impl NativeAnalyticalResultV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn for_input(
        method: &AnalyticalMethodV1,
        input: &RectangularCantileverInputV1,
        execution_artifact_digest: ExecutionArtifactDigestV1,
        factor_of_safety: f64,
        max_bending_stress_pa: f64,
        max_deflection_m: f64,
        max_moment_nm: f64,
        model_relative_error_bound: f64,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if input.method_revision_id() != method.revision_id() {
            return Err(AnalysisTrustErrorV1::MethodInputMismatch);
        }
        for (field, value) in [
            ("factor of safety", factor_of_safety),
            ("maximum bending stress", max_bending_stress_pa),
            ("maximum deflection", max_deflection_m),
            ("maximum moment", max_moment_nm),
            ("model-relative-error bound", model_relative_error_bound),
        ] {
            if !value.is_finite() {
                return Err(AnalysisTrustErrorV1::NonFinite(field));
            }
        }
        if max_bending_stress_pa <= 0.0 {
            return Err(AnalysisTrustErrorV1::NonPositive("maximum bending stress"));
        }
        if factor_of_safety <= 0.0 {
            return Err(AnalysisTrustErrorV1::NonPositive("factor of safety"));
        }
        if !(0.0..=1.0).contains(&model_relative_error_bound) {
            return Err(AnalysisTrustErrorV1::UnitInterval(
                "model-relative-error bound",
            ));
        }
        Ok(Self {
            method_revision_id: method.revision_id().clone(),
            input_revision_id: input.revision_id().clone(),
            execution_artifact_digest,
            factor_of_safety,
            max_bending_stress_pa,
            max_deflection_m,
            max_moment_nm,
            model_relative_error_bound,
        })
    }
}

#[must_use = "admitted analytical evidence is not a discharge receipt"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedAnalyticalEvidenceV1 {
    admitted_evidence_id: AdmittedAnalyticalEvidenceIdV1,
    analytical_plan_id: AnalyticalPlanIdV1,
}

impl AdmittedAnalyticalEvidenceV1 {
    pub fn admitted_evidence_id(&self) -> &AdmittedAnalyticalEvidenceIdV1 {
        &self.admitted_evidence_id
    }

    pub fn analytical_plan_id(&self) -> &AnalyticalPlanIdV1 {
        &self.analytical_plan_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "admitted_analytical_evidence_id": self.admitted_evidence_id.as_str(),
            "analytical_plan_id": self.analytical_plan_id.as_str(),
            "authority": "admitted-analytical-evidence-only",
        })
    }
}

pub fn admit_native_analytical_evidence_v1(
    plan: &NativeAnalyticalPlanV1,
    method: &AnalyticalMethodV1,
    input: &RectangularCantileverInputV1,
    policy: &AnalyticalAcceptancePolicyV1,
    result: &NativeAnalyticalResultV1,
) -> Result<AdmittedAnalyticalEvidenceV1, AnalysisTrustErrorV1> {
    if method.revision_id() != &plan.method_revision_id
        || input.revision_id() != &plan.input_revision_id
        || policy.revision_id() != &plan.policy_revision_id
        || result.method_revision_id != plan.method_revision_id
        || result.input_revision_id != plan.input_revision_id
    {
        return Err(AnalysisTrustErrorV1::PlanBindingMismatch);
    }

    let expected_moment = input.expected_max_moment_nm();
    if !relative_close(
        result.max_moment_nm,
        expected_moment,
        EQUATION_RELATIVE_TOLERANCE,
    ) {
        return Err(AnalysisTrustErrorV1::AnalyticalEquationMismatch(
            "maximum moment",
        ));
    }
    let expected_stress = input.expected_max_bending_stress_pa();
    if !relative_close(
        result.max_bending_stress_pa,
        expected_stress,
        EQUATION_RELATIVE_TOLERANCE,
    ) {
        return Err(AnalysisTrustErrorV1::AnalyticalEquationMismatch(
            "maximum bending stress",
        ));
    }
    let expected_deflection = input.expected_max_deflection_m();
    if !relative_close(
        result.max_deflection_m,
        expected_deflection,
        EQUATION_RELATIVE_TOLERANCE,
    ) {
        return Err(AnalysisTrustErrorV1::AnalyticalEquationMismatch(
            "maximum deflection",
        ));
    }

    let derived_fos = input.yield_strength_pa() / result.max_bending_stress_pa;
    if !relative_close(
        result.factor_of_safety,
        derived_fos,
        EQUATION_RELATIVE_TOLERANCE,
    ) {
        return Err(AnalysisTrustErrorV1::InconsistentFactorOfSafety);
    }
    if result.model_relative_error_bound > policy.max_model_relative_error_bound() {
        return Err(AnalysisTrustErrorV1::ModelErrorBudgetExceeded);
    }
    let conservative_factor_of_safety =
        result.factor_of_safety / (1.0 + result.model_relative_error_bound);
    if conservative_factor_of_safety < policy.threshold() {
        return Err(AnalysisTrustErrorV1::AcceptancePredicateFailed);
    }

    let normalized_result = json!({
        "execution_artifact_digest": result.execution_artifact_digest.as_str(),
        "factor_of_safety": canonical_binary64_v1(result.factor_of_safety)?,
        "max_bending_stress_pa": canonical_binary64_v1(result.max_bending_stress_pa)?,
        "max_deflection_m": canonical_binary64_v1(result.max_deflection_m)?,
        "max_moment_nm": canonical_binary64_v1(result.max_moment_nm)?,
        "model_relative_error_bound": canonical_binary64_v1(result.model_relative_error_bound)?,
    });
    let preimage = json!({
        "analytical_plan_id": plan.plan_id.as_str(),
        "conservative_factor_of_safety": canonical_binary64_v1(conservative_factor_of_safety)?,
        "normalized_result": normalized_result,
        "schema": "symthaea.etk-native-analytical-admitted-evidence.v1",
    });
    Ok(AdmittedAnalyticalEvidenceV1 {
        admitted_evidence_id: AdmittedAnalyticalEvidenceIdV1::from_digest(domain_hash(
            ADMITTED_DOMAIN_V1,
            &preimage,
        )),
        analytical_plan_id: plan.plan_id.clone(),
    })
}

#[must_use = "a historical analytical receipt is not present-tense discharge"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAnalyticalDischargeReceiptV1 {
    receipt_id: NativeAnalyticalDischargeReceiptIdV1,
    analytical_plan_id: AnalyticalPlanIdV1,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
}

impl NativeAnalyticalDischargeReceiptV1 {
    pub fn receipt_id(&self) -> &NativeAnalyticalDischargeReceiptIdV1 {
        &self.receipt_id
    }

    pub fn analytical_plan_id(&self) -> &AnalyticalPlanIdV1 {
        &self.analytical_plan_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "analytical_plan_id": self.analytical_plan_id.as_str(),
            "authority": "historical-analytical-discharge-only",
            "obligation_id": self.obligation_id.as_str(),
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "receipt_id": self.receipt_id.as_str(),
        })
    }
}

pub fn issue_native_analytical_discharge_receipt_v1(
    plan: &NativeAnalyticalPlanV1,
    admitted: &AdmittedAnalyticalEvidenceV1,
) -> Result<NativeAnalyticalDischargeReceiptV1, AnalysisTrustErrorV1> {
    if admitted.analytical_plan_id() != plan.plan_id() {
        return Err(AnalysisTrustErrorV1::AdmittedPlanMismatch);
    }
    let preimage = json!({
        "admitted_analytical_evidence_id": admitted.admitted_evidence_id().as_str(),
        "analytical_plan_id": plan.plan_id.as_str(),
        "obligation_id": plan.obligation_id.as_str(),
        "obligation_revision_id": plan.obligation_revision_id.as_str(),
        "schema": "symthaea.etk-native-analytical-discharge-receipt.v1",
    });
    Ok(NativeAnalyticalDischargeReceiptV1 {
        receipt_id: NativeAnalyticalDischargeReceiptIdV1::from_digest(domain_hash(
            RECEIPT_DOMAIN_V1,
            &preimage,
        )),
        analytical_plan_id: plan.plan_id.clone(),
        obligation_id: plan.obligation_id.clone(),
        obligation_revision_id: plan.obligation_revision_id.clone(),
    })
}

#[must_use = "a current analytical fact is not complete requirement satisfaction"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentNativeAnalyticalDischargeFactV1 {
    fact_id: CurrentNativeAnalyticalDischargeFactIdV1,
    analytical_plan_id: AnalyticalPlanIdV1,
    witness_receipt_id: NativeAnalyticalDischargeReceiptIdV1,
}

impl CurrentNativeAnalyticalDischargeFactV1 {
    pub fn fact_id(&self) -> &CurrentNativeAnalyticalDischargeFactIdV1 {
        &self.fact_id
    }

    pub fn analytical_plan_id(&self) -> &AnalyticalPlanIdV1 {
        &self.analytical_plan_id
    }

    pub fn witness_receipt_id(&self) -> &NativeAnalyticalDischargeReceiptIdV1 {
        &self.witness_receipt_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "analytical_plan_id": self.analytical_plan_id.as_str(),
            "authority": "current-analytical-discharge-only",
            "current_analytical_discharge_fact_id": self.fact_id.as_str(),
            "witness_receipt_id": self.witness_receipt_id.as_str(),
        })
    }
}

pub fn derive_current_native_analytical_discharge_fact_v1(
    current_plan: &NativeAnalyticalPlanV1,
    receipt: &NativeAnalyticalDischargeReceiptV1,
) -> Result<CurrentNativeAnalyticalDischargeFactV1, AnalysisTrustErrorV1> {
    if receipt.analytical_plan_id != current_plan.plan_id
        || receipt.obligation_id != current_plan.obligation_id
        || receipt.obligation_revision_id != current_plan.obligation_revision_id
    {
        return Err(AnalysisTrustErrorV1::HistoricalPlan);
    }
    let preimage = json!({
        "acceptance_policy_revision_id": current_plan.policy_revision_id.as_str(),
        "analytical_plan_id": current_plan.plan_id.as_str(),
        "currentness_assertion_id": current_plan.currentness_assertion_id.as_str(),
        "input_revision_id": current_plan.input_revision_id.as_str(),
        "method_revision_id": current_plan.method_revision_id.as_str(),
        "native_analytical_discharge_receipt_id": receipt.receipt_id.as_str(),
        "obligation_id": current_plan.obligation_id.as_str(),
        "obligation_revision_id": current_plan.obligation_revision_id.as_str(),
        "requirement_revision_id": current_plan.requirement_revision_id.as_str(),
        "schema": "symthaea.etk-current-native-analytical-discharge-fact.v1",
        "subject_revision_id": current_plan.subject_revision_id.as_str(),
        "twin_revision_id": current_plan.twin_revision_id.as_str(),
        "validity_domain_revision_id": current_plan.validity_domain_revision_id.as_str(),
    });
    Ok(CurrentNativeAnalyticalDischargeFactV1 {
        fact_id: CurrentNativeAnalyticalDischargeFactIdV1::from_digest(domain_hash(
            FACT_DOMAIN_V1,
            &preimage,
        )),
        analytical_plan_id: current_plan.plan_id.clone(),
        witness_receipt_id: receipt.receipt_id.clone(),
    })
}

fn relative_close(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() / expected.abs().max(1.0) <= tolerance
}

fn canonical_text(value: String, field: &'static str) -> Result<String, AnalysisTrustErrorV1> {
    if value.is_empty() || value.trim() != value {
        Err(AnalysisTrustErrorV1::InvalidText(field))
    } else {
        Ok(value)
    }
}

fn domain_hash(domain: &[u8], value: &Value) -> Sha256DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(canonical_json(value).as_bytes());
    Sha256DigestV1::parse(format!("sha256:{}", hex::encode(hasher.finalize())))
        .expect("SHA-256 output is canonical lowercase hex")
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => {
            serde_json::to_string(value).expect("serializing an in-memory JSON string cannot fail")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let encoded = serde_json::to_string(key)
                        .expect("serializing an in-memory JSON key cannot fail");
                    format!("{encoded}:{}", canonical_json(&map[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

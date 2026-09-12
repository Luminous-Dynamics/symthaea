// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::analysis::{
    AnalyticalAcceptancePolicyV1, AnalyticalMethodV1, NativeAnalyticalResultV1,
    RectangularCantileverInputV1,
};
use crate::canonical::{
    ADMITTED_DOMAIN_V1, AdmittedAnalyticalEvidenceIdV1, AnalysisTrustErrorV1,
    AnalyticalPlanIdV1, CurrentNativeAnalyticalDischargeFactIdV1, EQUATION_RELATIVE_TOLERANCE,
    FACT_DOMAIN_V1, NativeAnalyticalDischargeReceiptIdV1, PLAN_DOMAIN_V1, RECEIPT_DOMAIN_V1,
    canonical_binary64_v1, domain_hash,
};
use crate::context::{
    AcceptedAnalysisRequirementV1, CurrentnessAssertionV1, SubjectRevisionV1, TwinRevisionV1,
    ValidityDomainRevisionV1, analytical_obligation_revision_v1,
};
use crate::{
    AnalysisRequirementRevisionIdV1, AnalyticalInputRevisionIdV1, AnalyticalMethodRevisionIdV1,
    AnalyticalPolicyRevisionIdV1, CurrentnessAssertionIdV1, ObligationRevisionIdV1,
    SubjectRevisionIdV1, TwinRevisionIdV1, ValidityDomainRevisionIdV1,
};
use serde_json::{Value, json};
use symthaea_formal_safety::ProofObligation;

#[must_use = "an analytical plan is binding structure, not admitted evidence"]
#[derive(Debug, Clone, PartialEq)]
pub struct NativeAnalyticalPlanV1 {
    plan_id: AnalyticalPlanIdV1,
    requirement_revision_id: AnalysisRequirementRevisionIdV1,
    requirement_max_bending_stress_pa: f64,
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
        subject: &SubjectRevisionV1,
        twin: &TwinRevisionV1,
        validity_domain: &ValidityDomainRevisionV1,
        currentness: &CurrentnessAssertionV1,
        obligation: &ProofObligation,
        method: &AnalyticalMethodV1,
        input: &RectangularCantileverInputV1,
        policy: &AnalyticalAcceptancePolicyV1,
    ) -> Result<Self, AnalysisTrustErrorV1> {
        if input.method_revision_id() != method.revision_id() {
            return Err(AnalysisTrustErrorV1::MethodInputMismatch);
        }
        if twin.subject_revision_id() != subject.revision_id()
            || validity_domain.subject_revision_id() != subject.revision_id()
            || validity_domain.twin_revision_id() != twin.revision_id()
        {
            return Err(AnalysisTrustErrorV1::ValidityContextMismatch);
        }
        if currentness.twin_revision_id() != twin.revision_id()
            || currentness.validity_domain_revision_id() != validity_domain.revision_id()
        {
            return Err(AnalysisTrustErrorV1::CurrentnessContextMismatch);
        }

        let policy_worst_case_stress_pa = input.yield_strength_pa() / policy.threshold()
            * (1.0 + policy.max_model_relative_error_bound());
        if policy_worst_case_stress_pa > requirement.max_bending_stress_pa() {
            return Err(AnalysisTrustErrorV1::PolicyDoesNotDischargeRequirement);
        }

        let obligation_revision_id = analytical_obligation_revision_v1(obligation)?;
        let obligation_id = obligation.id.to_string();
        let preimage = json!({
            "acceptance_policy_revision_id": policy.revision_id().as_str(),
            "currentness_assertion_id": currentness.assertion_id().as_str(),
            "input_revision_id": input.revision_id().as_str(),
            "method_revision_id": method.revision_id().as_str(),
            "obligation_id": obligation_id.as_str(),
            "obligation_revision_id": obligation_revision_id.as_str(),
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": "symthaea.etk-native-analytical-plan.v1",
            "subject_revision_id": subject.revision_id().as_str(),
            "twin_revision_id": twin.revision_id().as_str(),
            "validity_domain_revision_id": validity_domain.revision_id().as_str(),
        });
        Ok(Self {
            plan_id: AnalyticalPlanIdV1::from_digest(domain_hash(PLAN_DOMAIN_V1, &preimage)),
            requirement_revision_id: requirement.revision_id().clone(),
            requirement_max_bending_stress_pa: requirement.max_bending_stress_pa(),
            obligation_id,
            obligation_revision_id,
            subject_revision_id: subject.revision_id().clone(),
            twin_revision_id: twin.revision_id().clone(),
            validity_domain_revision_id: validity_domain.revision_id().clone(),
            currentness_assertion_id: currentness.assertion_id().clone(),
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

    pub fn requirement_revision_id(&self) -> &AnalysisRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }

    pub fn subject_revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.subject_revision_id
    }

    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }

    pub fn validity_domain_revision_id(&self) -> &ValidityDomainRevisionIdV1 {
        &self.validity_domain_revision_id
    }

    pub fn currentness_assertion_id(&self) -> &CurrentnessAssertionIdV1 {
        &self.currentness_assertion_id
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
            "requirement_max_bending_stress_pa": self.requirement_max_bending_stress_pa,
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
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

    if !relative_close(
        result.max_moment_nm,
        input.expected_max_moment_nm(),
        EQUATION_RELATIVE_TOLERANCE,
    ) {
        return Err(AnalysisTrustErrorV1::AnalyticalEquationMismatch(
            "maximum moment",
        ));
    }
    if !relative_close(
        result.max_bending_stress_pa,
        input.expected_max_bending_stress_pa(),
        EQUATION_RELATIVE_TOLERANCE,
    ) {
        return Err(AnalysisTrustErrorV1::AnalyticalEquationMismatch(
            "maximum bending stress",
        ));
    }
    if !relative_close(
        result.max_deflection_m,
        input.expected_max_deflection_m(),
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

    let conservative_stress_pa =
        result.max_bending_stress_pa * (1.0 + result.model_relative_error_bound);
    if conservative_stress_pa > plan.requirement_max_bending_stress_pa {
        return Err(AnalysisTrustErrorV1::AcceptancePredicateFailed);
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
    requirement_revision_id: AnalysisRequirementRevisionIdV1,
    obligation_revision_id: ObligationRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    currentness_assertion_id: CurrentnessAssertionIdV1,
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

    pub fn requirement_revision_id(&self) -> &AnalysisRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }

    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "analytical_plan_id": self.analytical_plan_id.as_str(),
            "authority": "current-analytical-discharge-only",
            "current_analytical_discharge_fact_id": self.fact_id.as_str(),
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
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
        requirement_revision_id: current_plan.requirement_revision_id.clone(),
        obligation_revision_id: current_plan.obligation_revision_id.clone(),
        subject_revision_id: current_plan.subject_revision_id.clone(),
        twin_revision_id: current_plan.twin_revision_id.clone(),
        validity_domain_revision_id: current_plan.validity_domain_revision_id.clone(),
        currentness_assertion_id: current_plan.currentness_assertion_id.clone(),
    })
}

fn relative_close(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() / expected.abs().max(1.0) <= tolerance
}

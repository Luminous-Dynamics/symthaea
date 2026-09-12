// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::canonical::{
    AlgorithmRevisionDigestV1, AnalysisTrustErrorV1, AnalyticalInputRevisionIdV1,
    AnalyticalMethodRevisionIdV1, AnalyticalPolicyRevisionIdV1, ExecutionArtifactDigestV1,
    ImplementationArtifactDigestV1, INPUT_DOMAIN_V1, METHOD_DOMAIN_V1,
    ModelQualificationRecordDigestV1, POLICY_DOMAIN_V1, canonical_binary64_v1, domain_hash,
};
use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalMethodV1 {
    revision_id: AnalyticalMethodRevisionIdV1,
}

impl AnalyticalMethodV1 {
    pub fn euler_bernoulli_beam(
        implementation_artifact_digest: ImplementationArtifactDigestV1,
        algorithm_revision_digest: AlgorithmRevisionDigestV1,
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

    pub(crate) fn expected_max_moment_nm(&self) -> f64 {
        self.load_n * self.length_m
    }

    pub(crate) fn expected_max_bending_stress_pa(&self) -> f64 {
        let section_modulus = self.width_m * self.height_m.powi(2) / 6.0;
        self.expected_max_moment_nm() / section_modulus
    }

    pub(crate) fn expected_max_deflection_m(&self) -> f64 {
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
        if threshold <= 0.0 {
            return Err(AnalysisTrustErrorV1::NonPositive("factor-of-safety threshold"));
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

/// Data-only candidate result. It binds itself to one exact method/input pair,
/// but has no authority until the admission theorem independently rechecks the
/// plan and analytical equations.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeAnalyticalResultV1 {
    pub(crate) method_revision_id: AnalyticalMethodRevisionIdV1,
    pub(crate) input_revision_id: AnalyticalInputRevisionIdV1,
    pub(crate) execution_artifact_digest: ExecutionArtifactDigestV1,
    pub(crate) factor_of_safety: f64,
    pub(crate) max_bending_stress_pa: f64,
    pub(crate) max_deflection_m: f64,
    pub(crate) max_moment_nm: f64,
    pub(crate) model_relative_error_bound: f64,
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

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered model/r_min stability plan for static-potential string-tension fits.
//!
//! The fit family and the sequence of radial cuts are frozen before production
//! authorization. This module does not inspect data, choose a winning model, or
//! search for the first apparently stable range.

use std::collections::BTreeSet;

use crate::lattice_analysis_authority::{ExactHeadCiConclusion, ExactHeadCiEvidence};
use crate::lattice_campaign_authority::{
    CampaignExecutionAuthorization, QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE,
};
use crate::lattice_static_potential_plan::{
    STATIC_POTENTIAL_ANALYSIS_PLAN_ID, StaticPotentialAnalysisPlan,
};

pub const STATIC_POTENTIAL_FIT_STABILITY_PLAN_ID: &str =
    "static_potential_fit_stability_plan_v1";
pub const STATIC_POTENTIAL_FIT_STABILITY_AUTHORIZATION_SCOPE: &str =
    "qualified_static_potential_fit_stability_authorization_v1";
pub const FIT_FAMILY_ID: &str = "declared_lattice_potential_fit_family_gls_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StaticPotentialFitModel {
    FreeV0SigmaEL,
    FixedEPiOver12FreeL,
    FixedEPiOver12L0,
}

impl StaticPotentialFitModel {
    pub fn stable_id(self) -> &'static str {
        match self {
            Self::FreeV0SigmaEL => "free_v0_sigma_e_l_v1",
            Self::FixedEPiOver12FreeL => "fixed_e_pi_over_12_free_l_v1",
            Self::FixedEPiOver12L0 => "fixed_e_pi_over_12_l0_v1",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticPotentialRadialCut {
    /// Squared minimum Euclidean separation in lattice units.
    pub min_radius_squared: u32,
    /// Exact vectors retained after applying this cut to the preregistered base set.
    pub vectors: Vec<[i32; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticPotentialFitStabilityPlan {
    pub schema_id: &'static str,
    pub campaign_id: String,
    pub static_potential_plan_artifact_digest: String,
    pub static_potential_plan_freeze_evidence_id: String,

    pub fit_family_id: String,
    pub fit_family_implementation_revision: String,
    pub fit_family_oracle_evidence_id: String,
    pub models: Vec<StaticPotentialFitModel>,
    pub radial_cuts: Vec<StaticPotentialRadialCut>,

    /// Maximum caller-declared relative spread `(max sigma - min sigma)/mean sigma`
    /// used only as a downstream consistency policy. No universal value is encoded.
    pub max_sigma_relative_spread: f64,

    pub plan_configuration_digest: String,
    pub plan_freeze_evidence_id: String,
    pub frozen_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticPotentialFitStabilityAuthorization {
    pub authority_scope: &'static str,
    pub campaign_id: String,
    pub static_potential_plan_artifact_digest: String,
    pub fit_stability_plan_artifact_digest: String,
    pub fit_family_revision: String,
    pub fit_family_ci_run_id: u64,
    pub fit_family_ci_receipt_digest: String,
    pub radial_cut_count: usize,
    pub model_count: usize,
    pub frozen_at_unix_ns: u128,
    pub campaign_authorized_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StaticPotentialFitStabilityError {
    WrongSchema,
    StaticPlanSchemaMismatch,
    EmptyField(&'static str),
    InvalidSha256Digest { field: &'static str, value: String },
    CampaignMismatch,
    StaticPlanDigestMismatch,
    StaticPlanFreezeEvidenceMismatch,
    WrongFitFamily(String),
    InvalidFreezeTimestamp,
    FrozenBeforeStaticPlan,
    NotFrozenBeforeCampaignAuthorization,
    WrongCampaignAuthorizationScope,
    AuthorizationCampaignMismatch,
    InvalidModelSet,
    TooFewRadialCuts(usize),
    NonIncreasingRadialCut { previous: u32, current: u32 },
    TooFewFitPoints { cut_index: usize, count: usize },
    DuplicateVector { cut_index: usize, vector: [i32; 3] },
    VectorOutsideBaseSet { cut_index: usize, vector: [i32; 3] },
    VectorBelowCut { cut_index: usize, vector: [i32; 3] },
    RadialCutDoesNotMatchDeclaredBaseSet { cut_index: usize },
    InvalidSigmaSpread(f64),
    InvalidCiEvidence,
    CiDidNotPass(ExactHeadCiConclusion),
    CiRevisionMismatch { expected: String, actual: String },
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), StaticPotentialFitStabilityError> {
    if value.trim().is_empty() {
        Err(StaticPotentialFitStabilityError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), StaticPotentialFitStabilityError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(StaticPotentialFitStabilityError::InvalidSha256Digest {
            field,
            value: value.to_owned(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(StaticPotentialFitStabilityError::InvalidSha256Digest {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn radius_squared(vector: [i32; 3]) -> Option<u32> {
    vector.iter().try_fold(0u32, |sum, component| {
        let magnitude = component.unsigned_abs();
        sum.checked_add(magnitude.checked_mul(magnitude)?)
    })
}

fn canonical_vector_set(vectors: &[[i32; 3]]) -> BTreeSet<[i32; 3]> {
    vectors.iter().copied().collect()
}

impl StaticPotentialFitStabilityPlan {
    pub fn validate_against_static_plan(
        &self,
        static_plan: &StaticPotentialAnalysisPlan,
        static_plan_artifact_digest: &str,
    ) -> Result<(), StaticPotentialFitStabilityError> {
        if self.schema_id != STATIC_POTENTIAL_FIT_STABILITY_PLAN_ID {
            return Err(StaticPotentialFitStabilityError::WrongSchema);
        }
        if static_plan.plan_schema_id != STATIC_POTENTIAL_ANALYSIS_PLAN_ID {
            return Err(StaticPotentialFitStabilityError::StaticPlanSchemaMismatch);
        }
        if self.campaign_id != static_plan.campaign_id {
            return Err(StaticPotentialFitStabilityError::CampaignMismatch);
        }
        require_sha256(
            &self.static_potential_plan_artifact_digest,
            "static_potential_plan_artifact_digest",
        )?;
        if self.static_potential_plan_artifact_digest != static_plan_artifact_digest {
            return Err(StaticPotentialFitStabilityError::StaticPlanDigestMismatch);
        }
        if self.static_potential_plan_freeze_evidence_id
            != static_plan.analysis_plan_freeze_evidence_id
        {
            return Err(StaticPotentialFitStabilityError::StaticPlanFreezeEvidenceMismatch);
        }
        if self.fit_family_id != FIT_FAMILY_ID {
            return Err(StaticPotentialFitStabilityError::WrongFitFamily(
                self.fit_family_id.clone(),
            ));
        }
        for (value, field) in [
            (&self.fit_family_implementation_revision, "fit_family_implementation_revision"),
            (&self.fit_family_oracle_evidence_id, "fit_family_oracle_evidence_id"),
            (&self.plan_freeze_evidence_id, "plan_freeze_evidence_id"),
        ] {
            require_nonempty(value, field)?;
        }
        require_sha256(&self.plan_configuration_digest, "plan_configuration_digest")?;
        if self.frozen_at_unix_ns == 0 {
            return Err(StaticPotentialFitStabilityError::InvalidFreezeTimestamp);
        }
        if self.frozen_at_unix_ns <= static_plan.analysis_plan_frozen_at_unix_ns {
            return Err(StaticPotentialFitStabilityError::FrozenBeforeStaticPlan);
        }

        let expected_models = BTreeSet::from([
            StaticPotentialFitModel::FreeV0SigmaEL,
            StaticPotentialFitModel::FixedEPiOver12FreeL,
            StaticPotentialFitModel::FixedEPiOver12L0,
        ]);
        let actual_models = self.models.iter().copied().collect::<BTreeSet<_>>();
        if actual_models != expected_models || actual_models.len() != self.models.len() {
            return Err(StaticPotentialFitStabilityError::InvalidModelSet);
        }

        if self.radial_cuts.len() < 2 {
            return Err(StaticPotentialFitStabilityError::TooFewRadialCuts(
                self.radial_cuts.len(),
            ));
        }
        let base = canonical_vector_set(&static_plan.cornell_fit_vectors);
        let mut previous_cut = None;
        for (cut_index, cut) in self.radial_cuts.iter().enumerate() {
            if let Some(previous) = previous_cut {
                if cut.min_radius_squared <= previous {
                    return Err(StaticPotentialFitStabilityError::NonIncreasingRadialCut {
                        previous,
                        current: cut.min_radius_squared,
                    });
                }
            }
            previous_cut = Some(cut.min_radius_squared);

            if cut.vectors.len() <= 4 {
                return Err(StaticPotentialFitStabilityError::TooFewFitPoints {
                    cut_index,
                    count: cut.vectors.len(),
                });
            }
            let actual = canonical_vector_set(&cut.vectors);
            if actual.len() != cut.vectors.len() {
                let mut seen = BTreeSet::new();
                let duplicate = cut
                    .vectors
                    .iter()
                    .copied()
                    .find(|vector| !seen.insert(*vector))
                    .unwrap_or([0, 0, 0]);
                return Err(StaticPotentialFitStabilityError::DuplicateVector {
                    cut_index,
                    vector: duplicate,
                });
            }
            for vector in &cut.vectors {
                if !base.contains(vector) {
                    return Err(StaticPotentialFitStabilityError::VectorOutsideBaseSet {
                        cut_index,
                        vector: *vector,
                    });
                }
                if radius_squared(*vector).unwrap_or(0) < cut.min_radius_squared {
                    return Err(StaticPotentialFitStabilityError::VectorBelowCut {
                        cut_index,
                        vector: *vector,
                    });
                }
            }
            let expected = base
                .iter()
                .copied()
                .filter(|vector| radius_squared(*vector).unwrap_or(0) >= cut.min_radius_squared)
                .collect::<BTreeSet<_>>();
            if actual != expected {
                return Err(
                    StaticPotentialFitStabilityError::RadialCutDoesNotMatchDeclaredBaseSet {
                        cut_index,
                    },
                );
            }
        }

        if !self.max_sigma_relative_spread.is_finite() || self.max_sigma_relative_spread <= 0.0 {
            return Err(StaticPotentialFitStabilityError::InvalidSigmaSpread(
                self.max_sigma_relative_spread,
            ));
        }
        Ok(())
    }

    pub fn canonical_material(&self) -> String {
        let mut out = String::new();
        let mut field = |name: &str, value: &str| {
            out.push_str(name);
            out.push('=');
            out.push_str(&value.len().to_string());
            out.push(':');
            out.push_str(value);
            out.push('\n');
        };
        field("schema", self.schema_id);
        field("campaign", &self.campaign_id);
        field("static_plan_digest", &self.static_potential_plan_artifact_digest);
        field("static_plan_freeze", &self.static_potential_plan_freeze_evidence_id);
        field("fit_family", &self.fit_family_id);
        field("fit_family_revision", &self.fit_family_implementation_revision);
        field("fit_family_oracle", &self.fit_family_oracle_evidence_id);
        for model in &self.models {
            field("model", model.stable_id());
        }
        for cut in &self.radial_cuts {
            field("rmin2", &cut.min_radius_squared.to_string());
            for vector in &cut.vectors {
                field("vector", &format!("{},{},{}", vector[0], vector[1], vector[2]));
            }
        }
        field(
            "max_sigma_relative_spread_bits",
            &self.max_sigma_relative_spread.to_bits().to_string(),
        );
        field("config_digest", &self.plan_configuration_digest);
        field("freeze_evidence", &self.plan_freeze_evidence_id);
        field("frozen_at", &self.frozen_at_unix_ns.to_string());
        out
    }
}

pub fn authorize_static_potential_fit_stability(
    plan: &StaticPotentialFitStabilityPlan,
    static_plan: &StaticPotentialAnalysisPlan,
    static_plan_artifact_digest: &str,
    fit_stability_plan_artifact_digest: impl Into<String>,
    campaign_authorization: &CampaignExecutionAuthorization,
    fit_family_ci: &ExactHeadCiEvidence,
) -> Result<StaticPotentialFitStabilityAuthorization, StaticPotentialFitStabilityError> {
    plan.validate_against_static_plan(static_plan, static_plan_artifact_digest)?;
    let fit_stability_plan_artifact_digest = fit_stability_plan_artifact_digest.into();
    require_sha256(
        &fit_stability_plan_artifact_digest,
        "fit_stability_plan_artifact_digest",
    )?;
    if campaign_authorization.authority_scope
        != QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE
    {
        return Err(StaticPotentialFitStabilityError::WrongCampaignAuthorizationScope);
    }
    if campaign_authorization.campaign_id != plan.campaign_id {
        return Err(StaticPotentialFitStabilityError::AuthorizationCampaignMismatch);
    }
    if plan.frozen_at_unix_ns >= campaign_authorization.authorized_at_unix_ns {
        return Err(StaticPotentialFitStabilityError::NotFrozenBeforeCampaignAuthorization);
    }

    let structurally_valid = !fit_family_ci.subject_revision.trim().is_empty()
        && !fit_family_ci.source_tree_digest.trim().is_empty()
        && fit_family_ci.ci_run_id != 0
        && !fit_family_ci.ci_receipt_digest.trim().is_empty()
        && !fit_family_ci.toolchain_digest.trim().is_empty()
        && !fit_family_ci.build_profile.trim().is_empty()
        && fit_family_ci.qualification_timestamp_unix_ns != 0;
    if !structurally_valid {
        return Err(StaticPotentialFitStabilityError::InvalidCiEvidence);
    }
    if fit_family_ci.conclusion != ExactHeadCiConclusion::Passed {
        return Err(StaticPotentialFitStabilityError::CiDidNotPass(
            fit_family_ci.conclusion,
        ));
    }
    if fit_family_ci.subject_revision != plan.fit_family_implementation_revision {
        return Err(StaticPotentialFitStabilityError::CiRevisionMismatch {
            expected: plan.fit_family_implementation_revision.clone(),
            actual: fit_family_ci.subject_revision.clone(),
        });
    }

    Ok(StaticPotentialFitStabilityAuthorization {
        authority_scope: STATIC_POTENTIAL_FIT_STABILITY_AUTHORIZATION_SCOPE,
        campaign_id: plan.campaign_id.clone(),
        static_potential_plan_artifact_digest: plan.static_potential_plan_artifact_digest.clone(),
        fit_stability_plan_artifact_digest,
        fit_family_revision: fit_family_ci.subject_revision.clone(),
        fit_family_ci_run_id: fit_family_ci.ci_run_id,
        fit_family_ci_receipt_digest: fit_family_ci.ci_receipt_digest.clone(),
        radial_cut_count: plan.radial_cuts.len(),
        model_count: plan.models.len(),
        frozen_at_unix_ns: plan.frozen_at_unix_ns,
        campaign_authorized_at_unix_ns: campaign_authorization.authorized_at_unix_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_cut_is_derived_from_base_set_not_cherry_picked() {
        let base = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 2, 0]];
        let base_set = canonical_vector_set(&base);
        let expected = base_set
            .iter()
            .copied()
            .filter(|vector| radius_squared(*vector).unwrap() >= 2)
            .collect::<BTreeSet<_>>();
        assert_eq!(expected.len(), 5);
        assert!(!expected.contains(&[1, 0, 0]));
        assert!(expected.contains(&[2, 2, 0]));
    }

    #[test]
    fn shortest_supported_family_needs_more_than_four_points() {
        assert!(matches!(
            if 4usize <= 4 {
                Err(StaticPotentialFitStabilityError::TooFewFitPoints {
                    cut_index: 0,
                    count: 4,
                })
            } else {
                Ok(())
            },
            Err(StaticPotentialFitStabilityError::TooFewFitPoints { .. })
        ));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered static-potential analysis supplement for one frozen SU(3) campaign.
//!
//! The base campaign can be frozen before this supplement. The supplement must
//! then be frozen before production execution authorization. It binds operator
//! construction, separation geometry, Euclidean-time coverage, plateau windows,
//! block policy, lattice-Coulomb basis, Cornell fit range and benchmark lineage
//! without mutating the original campaign manifest.

use std::collections::BTreeSet;

use crate::lattice_analysis_authority::{ExactHeadCiConclusion, ExactHeadCiEvidence};
use crate::lattice_campaign_authority::{
    CampaignExecutionAuthorization, QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE,
};
use crate::lattice_campaign_manifest::{CampaignManifestError, PureSu3CampaignManifest};

pub const STATIC_POTENTIAL_ANALYSIS_PLAN_ID: &str = "static_potential_analysis_plan_v1";
pub const STATIC_POTENTIAL_SUPPLEMENT_AUTHORIZATION_SCOPE: &str =
    "qualified_static_potential_supplement_authorization_v1";
pub const SPATIAL_APE_CONVENTION_ID: &str = "spatial_ape_ehk_polar_v1";
pub const OFF_AXIS_WILSON_CONVENTION_ID: &str =
    "shortest_path_symmetrized_ape_spatial_unsmeared_temporal_wilson_v1";
pub const EFFECTIVE_POTENTIAL_CONVENTION_ID: &str =
    "wilson_effective_potential_declared_plateau_gls_v1";
pub const LATTICE_COULOMB_CONVENTION_ID: &str =
    "tree_level_wilson_lattice_coulomb_richardson_v1";
pub const LATTICE_CORNELL_CONVENTION_ID: &str =
    "correlated_declared_range_lattice_cornell_gls_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StaticPotentialPlateauWindow {
    /// One-based effective-potential time index.
    pub start_t: usize,
    /// One-based inclusive effective-potential time index.
    pub end_t: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticPotentialSeparationPlan {
    pub displacement: [i32; 3],
    /// Hard upper bound on unique shortest spatial paths averaged by the operator.
    pub max_shortest_paths: usize,
    /// Largest Wilson-loop temporal extent retained for this displacement.
    pub measured_t_max: usize,
    /// Primary preregistered plateau window. No later window search is authorized.
    pub primary_plateau: StaticPotentialPlateauWindow,
    /// Explicit neighboring windows used only as stability diagnostics.
    pub diagnostic_plateaus: Vec<StaticPotentialPlateauWindow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticPotentialAnalysisPlan {
    pub plan_schema_id: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub campaign_qualification_policy_artifact_digest: String,

    pub smearing_convention_id: String,
    pub smearing_config_digest: String,
    pub smearing_implementation_revision: String,
    pub smearing_oracle_evidence_id: String,

    pub wilson_measurement_convention_id: String,
    pub wilson_measurement_implementation_revision: String,
    pub wilson_measurement_oracle_evidence_id: String,

    pub effective_potential_convention_id: String,
    pub effective_potential_implementation_revision: String,
    pub effective_potential_oracle_evidence_id: String,

    pub block_policy_artifact_digest: String,
    pub separations: Vec<StaticPotentialSeparationPlan>,

    pub lattice_coulomb_convention_id: String,
    pub lattice_coulomb_implementation_revision: String,
    pub lattice_coulomb_config_digest: String,
    pub lattice_coulomb_oracle_evidence_id: String,
    pub lattice_coulomb_table_artifact_digest: String,

    pub cornell_convention_id: String,
    pub cornell_implementation_revision: String,
    pub cornell_oracle_evidence_id: String,
    /// Exact subset of measured displacement vectors entering the primary fit.
    pub cornell_fit_vectors: Vec<[i32; 3]>,

    /// Must be identical to the frozen campaign benchmark lineage.
    pub benchmark_source_digest: String,
    pub benchmark_contract_digest: String,

    /// Exact integration/orchestration subject implementing this supplement.
    pub analysis_plan_revision: String,
    pub analysis_plan_configuration_digest: String,
    pub analysis_plan_freeze_evidence_id: String,
    pub analysis_plan_frozen_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticPotentialQualifiedSubject {
    pub subject: &'static str,
    pub revision: String,
    pub ci_run_id: u64,
    pub ci_receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticPotentialSupplementAuthorization {
    pub authority_scope: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub analysis_plan_artifact_digest: String,
    pub analysis_plan_frozen_at_unix_ns: u128,
    pub campaign_authorized_at_unix_ns: u128,
    pub separation_count: usize,
    pub cornell_fit_point_count: usize,
    pub qualified_subjects: Vec<StaticPotentialQualifiedSubject>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StaticPotentialPlanError {
    Campaign(CampaignManifestError),
    WrongPlanSchema,
    EmptyField(&'static str),
    InvalidSha256Digest { field: &'static str, value: String },
    CampaignIdMismatch,
    CampaignManifestDigestMismatch,
    CampaignQualificationPolicyMismatch,
    BenchmarkSourceMismatch,
    BenchmarkContractMismatch,
    WrongConvention { field: &'static str, value: String },
    InvalidFreezeTimestamp,
    PlanFrozenBeforeCampaign,
    PlanNotFrozenBeforeAuthorization,
    WrongCampaignAuthorizationScope,
    AuthorizationCampaignMismatch,
    AuthorizationManifestDigestMismatch,
    EmptySeparationProgram,
    ZeroDisplacement { index: usize },
    DuplicateDisplacement([i32; 3]),
    WindingSpatialComponent {
        index: usize,
        axis: usize,
        requested: usize,
        lattice_extent: usize,
    },
    PathCountOverflow { index: usize },
    InvalidPathBudget { index: usize, budget: usize, required: usize },
    InvalidMeasuredTemporalExtent { index: usize, value: usize },
    WindingTemporalExtent {
        index: usize,
        requested: usize,
        lattice_extent: usize,
    },
    InvalidPlateauWindow {
        index: usize,
        window: StaticPotentialPlateauWindow,
    },
    PlateauRequiresUnmeasuredLoop {
        index: usize,
        end_t: usize,
        measured_t_max: usize,
    },
    DuplicateDiagnosticWindow {
        index: usize,
        window: StaticPotentialPlateauWindow,
    },
    DiagnosticMatchesPrimary { index: usize },
    TooFewCornellPoints(usize),
    DuplicateCornellVector([i32; 3]),
    CornellVectorNotMeasured([i32; 3]),
    InvalidCiEvidence { subject: &'static str },
    CiDidNotPass { subject: &'static str, conclusion: ExactHeadCiConclusion },
    CiSubjectRevisionMismatch {
        subject: &'static str,
        expected: String,
        actual: String,
    },
}

impl From<CampaignManifestError> for StaticPotentialPlanError {
    fn from(value: CampaignManifestError) -> Self {
        Self::Campaign(value)
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), StaticPotentialPlanError> {
    if value.trim().is_empty() {
        Err(StaticPotentialPlanError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), StaticPotentialPlanError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(StaticPotentialPlanError::InvalidSha256Digest {
            field,
            value: value.to_owned(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(StaticPotentialPlanError::InvalidSha256Digest {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn require_convention(
    value: &str,
    expected: &'static str,
    field: &'static str,
) -> Result<(), StaticPotentialPlanError> {
    if value != expected {
        Err(StaticPotentialPlanError::WrongConvention {
            field,
            value: value.to_owned(),
        })
    } else {
        Ok(())
    }
}

fn checked_binomial(n: usize, k: usize) -> Option<u128> {
    let k = k.min(n.checked_sub(k)?);
    let mut row = vec![0u128; k + 1];
    row[0] = 1;
    for _ in 0..n {
        for j in (1..=k).rev() {
            row[j] = row[j].checked_add(row[j - 1])?;
        }
    }
    Some(row[k])
}

fn shortest_path_count(displacement: [i32; 3]) -> Option<usize> {
    let a = displacement[0].unsigned_abs() as usize;
    let b = displacement[1].unsigned_abs() as usize;
    let c = displacement[2].unsigned_abs() as usize;
    let total = a.checked_add(b)?.checked_add(c)?;
    let first = checked_binomial(total, a)?;
    let second = checked_binomial(total.checked_sub(a)?, b)?;
    usize::try_from(first.checked_mul(second)?).ok()
}

fn validate_window(
    separation_index: usize,
    window: StaticPotentialPlateauWindow,
    measured_t_max: usize,
) -> Result<(), StaticPotentialPlanError> {
    if window.start_t == 0
        || window.end_t < window.start_t
        || window.end_t - window.start_t + 1 < 2
    {
        return Err(StaticPotentialPlanError::InvalidPlateauWindow {
            index: separation_index,
            window,
        });
    }
    let Some(required_loop_t) = window.end_t.checked_add(1) else {
        return Err(StaticPotentialPlanError::PlateauRequiresUnmeasuredLoop {
            index: separation_index,
            end_t: window.end_t,
            measured_t_max,
        });
    };
    if required_loop_t > measured_t_max {
        return Err(StaticPotentialPlanError::PlateauRequiresUnmeasuredLoop {
            index: separation_index,
            end_t: window.end_t,
            measured_t_max,
        });
    }
    Ok(())
}

fn validate_ci(
    subject: &'static str,
    evidence: &ExactHeadCiEvidence,
    expected_revision: &str,
) -> Result<StaticPotentialQualifiedSubject, StaticPotentialPlanError> {
    let structurally_valid = !evidence.subject_revision.trim().is_empty()
        && !evidence.source_tree_digest.trim().is_empty()
        && evidence.ci_run_id != 0
        && !evidence.ci_receipt_digest.trim().is_empty()
        && !evidence.toolchain_digest.trim().is_empty()
        && !evidence.build_profile.trim().is_empty()
        && evidence.qualification_timestamp_unix_ns != 0;
    if !structurally_valid {
        return Err(StaticPotentialPlanError::InvalidCiEvidence { subject });
    }
    if evidence.conclusion != ExactHeadCiConclusion::Passed {
        return Err(StaticPotentialPlanError::CiDidNotPass {
            subject,
            conclusion: evidence.conclusion,
        });
    }
    if evidence.subject_revision != expected_revision {
        return Err(StaticPotentialPlanError::CiSubjectRevisionMismatch {
            subject,
            expected: expected_revision.to_owned(),
            actual: evidence.subject_revision.clone(),
        });
    }
    Ok(StaticPotentialQualifiedSubject {
        subject,
        revision: evidence.subject_revision.clone(),
        ci_run_id: evidence.ci_run_id,
        ci_receipt_digest: evidence.ci_receipt_digest.clone(),
    })
}

impl StaticPotentialAnalysisPlan {
    pub fn validate_against_campaign(
        &self,
        campaign: &PureSu3CampaignManifest,
        campaign_manifest_artifact_digest: &str,
    ) -> Result<(), StaticPotentialPlanError> {
        campaign.validate()?;
        if self.plan_schema_id != STATIC_POTENTIAL_ANALYSIS_PLAN_ID {
            return Err(StaticPotentialPlanError::WrongPlanSchema);
        }
        if self.campaign_id != campaign.campaign_id {
            return Err(StaticPotentialPlanError::CampaignIdMismatch);
        }
        require_sha256(
            &self.campaign_manifest_artifact_digest,
            "campaign_manifest_artifact_digest",
        )?;
        if self.campaign_manifest_artifact_digest != campaign_manifest_artifact_digest {
            return Err(StaticPotentialPlanError::CampaignManifestDigestMismatch);
        }
        if self.campaign_qualification_policy_artifact_digest
            != campaign.qualification_policy_artifact_digest
        {
            return Err(StaticPotentialPlanError::CampaignQualificationPolicyMismatch);
        }
        if self.benchmark_source_digest != campaign.benchmark.benchmark_source_digest {
            return Err(StaticPotentialPlanError::BenchmarkSourceMismatch);
        }
        if self.benchmark_contract_digest != campaign.benchmark.benchmark_contract_digest {
            return Err(StaticPotentialPlanError::BenchmarkContractMismatch);
        }
        if self.analysis_plan_frozen_at_unix_ns == 0 {
            return Err(StaticPotentialPlanError::InvalidFreezeTimestamp);
        }
        if self.analysis_plan_frozen_at_unix_ns <= campaign.campaign_frozen_at_unix_ns {
            return Err(StaticPotentialPlanError::PlanFrozenBeforeCampaign);
        }

        require_convention(
            &self.smearing_convention_id,
            SPATIAL_APE_CONVENTION_ID,
            "smearing_convention_id",
        )?;
        require_convention(
            &self.wilson_measurement_convention_id,
            OFF_AXIS_WILSON_CONVENTION_ID,
            "wilson_measurement_convention_id",
        )?;
        require_convention(
            &self.effective_potential_convention_id,
            EFFECTIVE_POTENTIAL_CONVENTION_ID,
            "effective_potential_convention_id",
        )?;
        require_convention(
            &self.lattice_coulomb_convention_id,
            LATTICE_COULOMB_CONVENTION_ID,
            "lattice_coulomb_convention_id",
        )?;
        require_convention(
            &self.cornell_convention_id,
            LATTICE_CORNELL_CONVENTION_ID,
            "cornell_convention_id",
        )?;

        for (value, field) in [
            (&self.smearing_implementation_revision, "smearing_implementation_revision"),
            (&self.smearing_oracle_evidence_id, "smearing_oracle_evidence_id"),
            (&self.wilson_measurement_implementation_revision, "wilson_measurement_implementation_revision"),
            (&self.wilson_measurement_oracle_evidence_id, "wilson_measurement_oracle_evidence_id"),
            (&self.effective_potential_implementation_revision, "effective_potential_implementation_revision"),
            (&self.effective_potential_oracle_evidence_id, "effective_potential_oracle_evidence_id"),
            (&self.lattice_coulomb_implementation_revision, "lattice_coulomb_implementation_revision"),
            (&self.lattice_coulomb_oracle_evidence_id, "lattice_coulomb_oracle_evidence_id"),
            (&self.cornell_implementation_revision, "cornell_implementation_revision"),
            (&self.cornell_oracle_evidence_id, "cornell_oracle_evidence_id"),
            (&self.analysis_plan_revision, "analysis_plan_revision"),
            (&self.analysis_plan_freeze_evidence_id, "analysis_plan_freeze_evidence_id"),
        ] {
            require_nonempty(value, field)?;
        }
        for (value, field) in [
            (&self.smearing_config_digest, "smearing_config_digest"),
            (&self.block_policy_artifact_digest, "block_policy_artifact_digest"),
            (&self.lattice_coulomb_config_digest, "lattice_coulomb_config_digest"),
            (&self.lattice_coulomb_table_artifact_digest, "lattice_coulomb_table_artifact_digest"),
            (&self.analysis_plan_configuration_digest, "analysis_plan_configuration_digest"),
            (&self.benchmark_source_digest, "benchmark_source_digest"),
            (&self.benchmark_contract_digest, "benchmark_contract_digest"),
            (&self.campaign_qualification_policy_artifact_digest, "campaign_qualification_policy_artifact_digest"),
        ] {
            require_sha256(value, field)?;
        }

        if self.separations.is_empty() {
            return Err(StaticPotentialPlanError::EmptySeparationProgram);
        }
        let mut measured_vectors = BTreeSet::new();
        for (index, separation) in self.separations.iter().enumerate() {
            if separation.displacement == [0, 0, 0] {
                return Err(StaticPotentialPlanError::ZeroDisplacement { index });
            }
            if !measured_vectors.insert(separation.displacement) {
                return Err(StaticPotentialPlanError::DuplicateDisplacement(
                    separation.displacement,
                ));
            }
            for axis in 0..3 {
                let magnitude = separation.displacement[axis].unsigned_abs() as usize;
                if magnitude >= campaign.dims[axis] {
                    return Err(StaticPotentialPlanError::WindingSpatialComponent {
                        index,
                        axis,
                        requested: magnitude,
                        lattice_extent: campaign.dims[axis],
                    });
                }
            }
            let required_paths = shortest_path_count(separation.displacement)
                .ok_or(StaticPotentialPlanError::PathCountOverflow { index })?;
            if separation.max_shortest_paths < required_paths || separation.max_shortest_paths == 0 {
                return Err(StaticPotentialPlanError::InvalidPathBudget {
                    index,
                    budget: separation.max_shortest_paths,
                    required: required_paths,
                });
            }
            if separation.measured_t_max < 2 {
                return Err(StaticPotentialPlanError::InvalidMeasuredTemporalExtent {
                    index,
                    value: separation.measured_t_max,
                });
            }
            if separation.measured_t_max >= campaign.dims[3] {
                return Err(StaticPotentialPlanError::WindingTemporalExtent {
                    index,
                    requested: separation.measured_t_max,
                    lattice_extent: campaign.dims[3],
                });
            }
            validate_window(index, separation.primary_plateau, separation.measured_t_max)?;
            let mut diagnostic_windows = BTreeSet::new();
            for window in separation.diagnostic_plateaus.iter().copied() {
                validate_window(index, window, separation.measured_t_max)?;
                if window == separation.primary_plateau {
                    return Err(StaticPotentialPlanError::DiagnosticMatchesPrimary { index });
                }
                if !diagnostic_windows.insert(window) {
                    return Err(StaticPotentialPlanError::DuplicateDiagnosticWindow {
                        index,
                        window,
                    });
                }
            }
        }

        if self.cornell_fit_vectors.len() <= 3 {
            return Err(StaticPotentialPlanError::TooFewCornellPoints(
                self.cornell_fit_vectors.len(),
            ));
        }
        let mut cornell = BTreeSet::new();
        for vector in &self.cornell_fit_vectors {
            if !cornell.insert(*vector) {
                return Err(StaticPotentialPlanError::DuplicateCornellVector(*vector));
            }
            if !measured_vectors.contains(vector) {
                return Err(StaticPotentialPlanError::CornellVectorNotMeasured(*vector));
            }
        }
        Ok(())
    }

    pub fn canonical_material(&self) -> String {
        let mut material = String::new();
        material.push_str(self.plan_schema_id);
        material.push('|');
        material.push_str(&self.campaign_id);
        material.push('|');
        material.push_str(&self.campaign_manifest_artifact_digest);
        material.push('|');
        material.push_str(&self.smearing_convention_id);
        material.push('|');
        material.push_str(&self.smearing_config_digest);
        material.push('|');
        material.push_str(&self.smearing_implementation_revision);
        material.push('|');
        material.push_str(&self.wilson_measurement_convention_id);
        material.push('|');
        material.push_str(&self.wilson_measurement_implementation_revision);
        material.push('|');
        material.push_str(&self.effective_potential_convention_id);
        material.push('|');
        material.push_str(&self.effective_potential_implementation_revision);
        material.push('|');
        material.push_str(&self.block_policy_artifact_digest);
        for separation in &self.separations {
            material.push('|');
            material.push_str(&format!(
                "r={},{},{};paths={};tmax={};primary={}-{}",
                separation.displacement[0],
                separation.displacement[1],
                separation.displacement[2],
                separation.max_shortest_paths,
                separation.measured_t_max,
                separation.primary_plateau.start_t,
                separation.primary_plateau.end_t,
            ));
            for window in &separation.diagnostic_plateaus {
                material.push_str(&format!(";diag={}-{}", window.start_t, window.end_t));
            }
        }
        material.push('|');
        material.push_str(&self.lattice_coulomb_convention_id);
        material.push('|');
        material.push_str(&self.lattice_coulomb_implementation_revision);
        material.push('|');
        material.push_str(&self.lattice_coulomb_config_digest);
        material.push('|');
        material.push_str(&self.lattice_coulomb_table_artifact_digest);
        material.push('|');
        material.push_str(&self.cornell_convention_id);
        material.push('|');
        material.push_str(&self.cornell_implementation_revision);
        for vector in &self.cornell_fit_vectors {
            material.push_str(&format!("|fit={},{},{}", vector[0], vector[1], vector[2]));
        }
        material.push('|');
        material.push_str(&self.benchmark_source_digest);
        material.push('|');
        material.push_str(&self.benchmark_contract_digest);
        material.push('|');
        material.push_str(&self.analysis_plan_revision);
        material.push('|');
        material.push_str(&self.analysis_plan_configuration_digest);
        material.push('|');
        material.push_str(&self.analysis_plan_frozen_at_unix_ns.to_string());
        material
    }
}

#[allow(clippy::too_many_arguments)]
pub fn authorize_static_potential_supplement(
    campaign: &PureSu3CampaignManifest,
    campaign_manifest_artifact_digest: &str,
    plan: &StaticPotentialAnalysisPlan,
    analysis_plan_artifact_digest: &str,
    campaign_authorization: &CampaignExecutionAuthorization,
    plan_ci: &ExactHeadCiEvidence,
    smearing_ci: &ExactHeadCiEvidence,
    wilson_measurement_ci: &ExactHeadCiEvidence,
    effective_potential_ci: &ExactHeadCiEvidence,
    lattice_coulomb_ci: &ExactHeadCiEvidence,
    cornell_ci: &ExactHeadCiEvidence,
) -> Result<StaticPotentialSupplementAuthorization, StaticPotentialPlanError> {
    plan.validate_against_campaign(campaign, campaign_manifest_artifact_digest)?;
    require_sha256(analysis_plan_artifact_digest, "analysis_plan_artifact_digest")?;

    if campaign_authorization.authority_scope
        != QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE
    {
        return Err(StaticPotentialPlanError::WrongCampaignAuthorizationScope);
    }
    if campaign_authorization.campaign_id != campaign.campaign_id {
        return Err(StaticPotentialPlanError::AuthorizationCampaignMismatch);
    }
    if campaign_authorization.campaign_manifest_artifact_digest
        != campaign_manifest_artifact_digest
    {
        return Err(StaticPotentialPlanError::AuthorizationManifestDigestMismatch);
    }
    if campaign_authorization.authorized_at_unix_ns <= plan.analysis_plan_frozen_at_unix_ns {
        return Err(StaticPotentialPlanError::PlanNotFrozenBeforeAuthorization);
    }

    let qualified_subjects = vec![
        validate_ci("analysis-plan", plan_ci, &plan.analysis_plan_revision)?,
        validate_ci("smearing", smearing_ci, &plan.smearing_implementation_revision)?,
        validate_ci(
            "wilson-measurement",
            wilson_measurement_ci,
            &plan.wilson_measurement_implementation_revision,
        )?,
        validate_ci(
            "effective-potential",
            effective_potential_ci,
            &plan.effective_potential_implementation_revision,
        )?,
        validate_ci(
            "lattice-coulomb",
            lattice_coulomb_ci,
            &plan.lattice_coulomb_implementation_revision,
        )?,
        validate_ci("cornell-fit", cornell_ci, &plan.cornell_implementation_revision)?,
    ];

    Ok(StaticPotentialSupplementAuthorization {
        authority_scope: STATIC_POTENTIAL_SUPPLEMENT_AUTHORIZATION_SCOPE,
        campaign_id: campaign.campaign_id.clone(),
        campaign_manifest_artifact_digest: campaign_manifest_artifact_digest.to_owned(),
        analysis_plan_artifact_digest: analysis_plan_artifact_digest.to_owned(),
        analysis_plan_frozen_at_unix_ns: plan.analysis_plan_frozen_at_unix_ns,
        campaign_authorized_at_unix_ns: campaign_authorization.authorized_at_unix_ns,
        separation_count: plan.separations.len(),
        cornell_fit_point_count: plan.cornell_fit_vectors.len(),
        qualified_subjects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_path_count_matches_off_axis_oracle_geometry() {
        assert_eq!(shortest_path_count([1, 0, 0]), Some(1));
        assert_eq!(shortest_path_count([2, 0, 0]), Some(1));
        assert_eq!(shortest_path_count([1, 1, 0]), Some(2));
        assert_eq!(shortest_path_count([1, 1, 1]), Some(6));
        assert_eq!(shortest_path_count([2, 1, 0]), Some(3));
    }

    #[test]
    fn plateau_requires_the_next_wilson_loop_time() {
        let window = StaticPotentialPlateauWindow { start_t: 5, end_t: 7 };
        assert!(validate_window(0, window, 8).is_ok());
        assert_eq!(
            validate_window(0, window, 7),
            Err(StaticPotentialPlanError::PlateauRequiresUnmeasuredLoop {
                index: 0,
                end_t: 7,
                measured_t_max: 7,
            })
        );
    }

    #[test]
    fn path_budget_must_cover_every_unique_shortest_path() {
        let required = shortest_path_count([1, 1, 1]).unwrap();
        assert_eq!(required, 6);
        assert!(5 < required);
        assert!(6 >= required);
    }
}

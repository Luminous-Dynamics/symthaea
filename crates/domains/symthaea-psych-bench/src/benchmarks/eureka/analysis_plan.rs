// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered EUREKA-002C analysis plan.
//!
//! This module freezes comparator selection, metric strata, practical-effect
//! gates, replication minima, uncertainty settings, exclusions, and non-claims
//! before FullSymthaea held-out outcomes are inspected.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2099>

use super::baselines::ShortcutBaselineKind;

pub(super) const ANALYSIS_PLAN_REVISION: &str = "EUREKA.002C.ANALYSIS_PLAN.v1";
pub(super) const CONSEQUENCE_SCORER_REVISION: &str = "EUREKA.CONSEQUENCE_SCORER.v1";
pub(super) const TRANSITION_IDENTITY_REVISION: &str = "eureka.transition.v1";

/// Basis points avoid ambiguous floating-point serialization in the evidence
/// identity. 10_000 basis points == 1.0.
pub(super) const BASIS_POINTS_ONE: i32 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MetricStratum {
    ChangeBearing,
    TrueNoChange,
    Coverage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MetricRole {
    Primary,
    Secondary,
    DescriptiveOnly,
    Coverage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MetricId {
    ChangedFieldF1,
    ChangedValueAccuracy,
    FalseChangeRate,
    MissedChangeRate,
    UnchangedStatePreservation,
    FullStateAccuracy,
    ValidCoverage,
    AbstentionRate,
    OutOfDomainRate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct MetricSpec {
    pub id: MetricId,
    pub stratum: MetricStratum,
    pub role: MetricRole,
    /// If false, undefined observations must remain undefined and cannot be
    /// imputed as zero or one for aggregation.
    pub defined_on_every_valid_trial: bool,
}

pub(super) const METRIC_SPECS: [MetricSpec; 9] = [
    MetricSpec {
        id: MetricId::ChangedFieldF1,
        stratum: MetricStratum::ChangeBearing,
        role: MetricRole::Primary,
        defined_on_every_valid_trial: false,
    },
    MetricSpec {
        id: MetricId::ChangedValueAccuracy,
        stratum: MetricStratum::ChangeBearing,
        role: MetricRole::Primary,
        defined_on_every_valid_trial: false,
    },
    MetricSpec {
        id: MetricId::FalseChangeRate,
        stratum: MetricStratum::ChangeBearing,
        role: MetricRole::Secondary,
        defined_on_every_valid_trial: false,
    },
    MetricSpec {
        id: MetricId::MissedChangeRate,
        stratum: MetricStratum::ChangeBearing,
        role: MetricRole::Secondary,
        defined_on_every_valid_trial: false,
    },
    MetricSpec {
        id: MetricId::UnchangedStatePreservation,
        stratum: MetricStratum::TrueNoChange,
        role: MetricRole::Primary,
        defined_on_every_valid_trial: false,
    },
    MetricSpec {
        id: MetricId::FullStateAccuracy,
        stratum: MetricStratum::ChangeBearing,
        role: MetricRole::DescriptiveOnly,
        defined_on_every_valid_trial: true,
    },
    MetricSpec {
        id: MetricId::ValidCoverage,
        stratum: MetricStratum::Coverage,
        role: MetricRole::Coverage,
        defined_on_every_valid_trial: true,
    },
    MetricSpec {
        id: MetricId::AbstentionRate,
        stratum: MetricStratum::Coverage,
        role: MetricRole::Coverage,
        defined_on_every_valid_trial: true,
    },
    MetricSpec {
        id: MetricId::OutOfDomainRate,
        stratum: MetricStratum::Coverage,
        role: MetricRole::Coverage,
        defined_on_every_valid_trial: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CampaignRowDisposition {
    ValidScored,
    ValidAbstained,
    ValidOutOfDomain,
    InvalidLeakage,
    InvalidLineageMismatch,
    ScorerFailure,
    TargetExecutionFailure,
    InfrastructureIndeterminate,
}

pub(super) const REQUIRED_ROW_DISPOSITIONS: [CampaignRowDisposition; 8] = [
    CampaignRowDisposition::ValidScored,
    CampaignRowDisposition::ValidAbstained,
    CampaignRowDisposition::ValidOutOfDomain,
    CampaignRowDisposition::InvalidLeakage,
    CampaignRowDisposition::InvalidLineageMismatch,
    CampaignRowDisposition::ScorerFailure,
    CampaignRowDisposition::TargetExecutionFailure,
    CampaignRowDisposition::InfrastructureIndeterminate,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ComparatorSelectionRule {
    /// Select on a calibration-only selection corpus that is transition-ID
    /// disjoint from the baseline fit corpus. Rank by changed-field F1,
    /// require minimum coverage, then tie-break by stable baseline ID.
    CalibrationChangedFieldF1ThenStableId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum IntervalMethod {
    PairedBootstrapFixedSeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AggregationRequirement {
    PerTransition,
    PerSeed,
    PerWorldFamily,
    Overall,
    ExternalReplicationSeparate,
}

pub(super) const AGGREGATION_REQUIREMENTS: [AggregationRequirement; 5] = [
    AggregationRequirement::PerTransition,
    AggregationRequirement::PerSeed,
    AggregationRequirement::PerWorldFamily,
    AggregationRequirement::Overall,
    AggregationRequirement::ExternalReplicationSeparate,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AllowedExclusionReason {
    EvaluatorInvariantViolation,
    MalformedWorldSchema,
    DuplicateCanonicalTransition,
    InfrastructureFailureBeforeTargetExecution,
    ProvenLeakage,
}

pub(super) const ALLOWED_EXCLUSIONS: [AllowedExclusionReason; 5] = [
    AllowedExclusionReason::EvaluatorInvariantViolation,
    AllowedExclusionReason::MalformedWorldSchema,
    AllowedExclusionReason::DuplicateCanonicalTransition,
    AllowedExclusionReason::InfrastructureFailureBeforeTargetExecution,
    AllowedExclusionReason::ProvenLeakage,
];

pub(super) const ELIGIBLE_SHORTCUT_COMPARATORS: [ShortcutBaselineKind; 4] = [
    ShortcutBaselineKind::ActionMarginalDelta,
    ShortcutBaselineKind::ExactLookup,
    ShortcutBaselineKind::NearestTransition,
    ShortcutBaselineKind::SimpleMarkov,
];

pub(super) const EUREKA_002_NON_CLAIMS: [&str; 8] = [
    "causal understanding",
    "counterfactual understanding",
    "concept formation",
    "consciousness",
    "sentience",
    "general intelligence",
    "physical-world truth",
    "action authority",
];

/// Frozen EUREKA-002C V1 promotion/analysis contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Eureka002AnalysisPlanV1 {
    pub revision: &'static str,
    pub claim_id: &'static str,
    pub scorer_revision: &'static str,
    pub transition_identity_revision: &'static str,
    pub comparator_selection_rule: ComparatorSelectionRule,
    pub comparator_selection_must_be_disjoint_from_fit: bool,
    pub eligible_comparators: &'static [ShortcutBaselineKind],
    pub copy_current_state_is_mandatory_construct_control: bool,
    pub comparator_min_coverage_bps: u16,
    pub candidate_min_coverage_bps: u16,
    pub changed_field_f1_margin_bps: i16,
    pub changed_value_accuracy_margin_bps: i16,
    /// Candidate may regress by at most this many basis points on true-no-change
    /// preservation relative to the frozen primary comparator.
    pub no_change_preservation_max_regression_bps: u16,
    pub minimum_world_families: u8,
    pub held_out_seeds_per_family_min: u16,
    pub external_replication_seeds_per_family_min: u16,
    pub interval_method: IntervalMethod,
    pub confidence_level_bps: u16,
    pub bootstrap_resamples: u32,
    pub bootstrap_seed: u64,
    pub metric_specs: &'static [MetricSpec],
    pub required_row_dispositions: &'static [CampaignRowDisposition],
    pub aggregation_requirements: &'static [AggregationRequirement],
    pub allowed_exclusions: &'static [AllowedExclusionReason],
    pub zero_allowed_leakage_rows: bool,
    pub zero_allowed_lineage_invalid_rows_for_support: bool,
    pub external_replication_cannot_be_pooled_into_held_out_for_promotion: bool,
    pub explicit_non_claims: &'static [&'static str],
}

pub(super) const EUREKA_002_ANALYSIS_PLAN_V1: Eureka002AnalysisPlanV1 =
    Eureka002AnalysisPlanV1 {
        revision: ANALYSIS_PLAN_REVISION,
        claim_id: "EUREKA.PREDICTIVE_MODELING.v1",
        scorer_revision: CONSEQUENCE_SCORER_REVISION,
        transition_identity_revision: TRANSITION_IDENTITY_REVISION,
        comparator_selection_rule:
            ComparatorSelectionRule::CalibrationChangedFieldF1ThenStableId,
        comparator_selection_must_be_disjoint_from_fit: true,
        eligible_comparators: &ELIGIBLE_SHORTCUT_COMPARATORS,
        copy_current_state_is_mandatory_construct_control: true,
        comparator_min_coverage_bps: 9_000,
        candidate_min_coverage_bps: 9_000,
        changed_field_f1_margin_bps: 500,
        changed_value_accuracy_margin_bps: 500,
        no_change_preservation_max_regression_bps: 200,
        minimum_world_families: 2,
        held_out_seeds_per_family_min: 64,
        external_replication_seeds_per_family_min: 32,
        interval_method: IntervalMethod::PairedBootstrapFixedSeed,
        confidence_level_bps: 9_500,
        bootstrap_resamples: 10_000,
        bootstrap_seed: 0x4555_5245_4b41_3032,
        metric_specs: &METRIC_SPECS,
        required_row_dispositions: &REQUIRED_ROW_DISPOSITIONS,
        aggregation_requirements: &AGGREGATION_REQUIREMENTS,
        allowed_exclusions: &ALLOWED_EXCLUSIONS,
        zero_allowed_leakage_rows: true,
        zero_allowed_lineage_invalid_rows_for_support: true,
        external_replication_cannot_be_pooled_into_held_out_for_promotion: true,
        explicit_non_claims: &EUREKA_002_NON_CLAIMS,
    };

impl Eureka002AnalysisPlanV1 {
    /// Deterministic replay/evidence identity. This is not cryptographic
    /// authentication and is not a trusted external timestamp.
    pub(super) fn replay_digest(&self) -> u64 {
        let mut bytes = Vec::new();
        encode_str(&mut bytes, self.revision);
        encode_str(&mut bytes, self.claim_id);
        encode_str(&mut bytes, self.scorer_revision);
        encode_str(&mut bytes, self.transition_identity_revision);
        bytes.push(selection_rule_tag(self.comparator_selection_rule));
        bytes.push(u8::from(self.comparator_selection_must_be_disjoint_from_fit));
        bytes.extend_from_slice(&(self.eligible_comparators.len() as u64).to_le_bytes());
        for comparator in self.eligible_comparators {
            encode_str(&mut bytes, comparator.stable_id());
        }
        bytes.push(u8::from(self.copy_current_state_is_mandatory_construct_control));
        bytes.extend_from_slice(&self.comparator_min_coverage_bps.to_le_bytes());
        bytes.extend_from_slice(&self.candidate_min_coverage_bps.to_le_bytes());
        bytes.extend_from_slice(&self.changed_field_f1_margin_bps.to_le_bytes());
        bytes.extend_from_slice(&self.changed_value_accuracy_margin_bps.to_le_bytes());
        bytes.extend_from_slice(&self.no_change_preservation_max_regression_bps.to_le_bytes());
        bytes.push(self.minimum_world_families);
        bytes.extend_from_slice(&self.held_out_seeds_per_family_min.to_le_bytes());
        bytes.extend_from_slice(&self.external_replication_seeds_per_family_min.to_le_bytes());
        bytes.push(interval_method_tag(self.interval_method));
        bytes.extend_from_slice(&self.confidence_level_bps.to_le_bytes());
        bytes.extend_from_slice(&self.bootstrap_resamples.to_le_bytes());
        bytes.extend_from_slice(&self.bootstrap_seed.to_le_bytes());

        bytes.extend_from_slice(&(self.metric_specs.len() as u64).to_le_bytes());
        for metric in self.metric_specs {
            bytes.push(metric_id_tag(metric.id));
            bytes.push(metric_stratum_tag(metric.stratum));
            bytes.push(metric_role_tag(metric.role));
            bytes.push(u8::from(metric.defined_on_every_valid_trial));
        }

        bytes.extend_from_slice(&(self.required_row_dispositions.len() as u64).to_le_bytes());
        for disposition in self.required_row_dispositions {
            bytes.push(row_disposition_tag(*disposition));
        }

        bytes.extend_from_slice(&(self.aggregation_requirements.len() as u64).to_le_bytes());
        for requirement in self.aggregation_requirements {
            bytes.push(aggregation_tag(*requirement));
        }

        bytes.extend_from_slice(&(self.allowed_exclusions.len() as u64).to_le_bytes());
        for exclusion in self.allowed_exclusions {
            bytes.push(exclusion_tag(*exclusion));
        }

        bytes.push(u8::from(self.zero_allowed_leakage_rows));
        bytes.push(u8::from(self.zero_allowed_lineage_invalid_rows_for_support));
        bytes.push(u8::from(
            self.external_replication_cannot_be_pooled_into_held_out_for_promotion,
        ));
        bytes.extend_from_slice(&(self.explicit_non_claims.len() as u64).to_le_bytes());
        for non_claim in self.explicit_non_claims {
            encode_str(&mut bytes, non_claim);
        }
        fnv1a64(&bytes)
    }
}

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

const fn selection_rule_tag(value: ComparatorSelectionRule) -> u8 {
    match value {
        ComparatorSelectionRule::CalibrationChangedFieldF1ThenStableId => 1,
    }
}

const fn interval_method_tag(value: IntervalMethod) -> u8 {
    match value {
        IntervalMethod::PairedBootstrapFixedSeed => 1,
    }
}

const fn metric_stratum_tag(value: MetricStratum) -> u8 {
    match value {
        MetricStratum::ChangeBearing => 1,
        MetricStratum::TrueNoChange => 2,
        MetricStratum::Coverage => 3,
    }
}

const fn metric_role_tag(value: MetricRole) -> u8 {
    match value {
        MetricRole::Primary => 1,
        MetricRole::Secondary => 2,
        MetricRole::DescriptiveOnly => 3,
        MetricRole::Coverage => 4,
    }
}

const fn metric_id_tag(value: MetricId) -> u8 {
    match value {
        MetricId::ChangedFieldF1 => 1,
        MetricId::ChangedValueAccuracy => 2,
        MetricId::FalseChangeRate => 3,
        MetricId::MissedChangeRate => 4,
        MetricId::UnchangedStatePreservation => 5,
        MetricId::FullStateAccuracy => 6,
        MetricId::ValidCoverage => 7,
        MetricId::AbstentionRate => 8,
        MetricId::OutOfDomainRate => 9,
    }
}

const fn row_disposition_tag(value: CampaignRowDisposition) -> u8 {
    match value {
        CampaignRowDisposition::ValidScored => 1,
        CampaignRowDisposition::ValidAbstained => 2,
        CampaignRowDisposition::ValidOutOfDomain => 3,
        CampaignRowDisposition::InvalidLeakage => 4,
        CampaignRowDisposition::InvalidLineageMismatch => 5,
        CampaignRowDisposition::ScorerFailure => 6,
        CampaignRowDisposition::TargetExecutionFailure => 7,
        CampaignRowDisposition::InfrastructureIndeterminate => 8,
    }
}

const fn aggregation_tag(value: AggregationRequirement) -> u8 {
    match value {
        AggregationRequirement::PerTransition => 1,
        AggregationRequirement::PerSeed => 2,
        AggregationRequirement::PerWorldFamily => 3,
        AggregationRequirement::Overall => 4,
        AggregationRequirement::ExternalReplicationSeparate => 5,
    }
}

const fn exclusion_tag(value: AllowedExclusionReason) -> u8 {
    match value {
        AllowedExclusionReason::EvaluatorInvariantViolation => 1,
        AllowedExclusionReason::MalformedWorldSchema => 2,
        AllowedExclusionReason::DuplicateCanonicalTransition => 3,
        AllowedExclusionReason::InfrastructureFailureBeforeTargetExecution => 4,
        AllowedExclusionReason::ProvenLeakage => 5,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn v1_plan_freezes_expected_preregistered_effect_gates() {
        let plan = EUREKA_002_ANALYSIS_PLAN_V1;
        assert_eq!(plan.comparator_min_coverage_bps, 9_000);
        assert_eq!(plan.candidate_min_coverage_bps, 9_000);
        assert_eq!(plan.changed_field_f1_margin_bps, 500);
        assert_eq!(plan.changed_value_accuracy_margin_bps, 500);
        assert_eq!(plan.no_change_preservation_max_regression_bps, 200);
        assert_eq!(plan.minimum_world_families, 2);
        assert_eq!(plan.held_out_seeds_per_family_min, 64);
        assert_eq!(plan.external_replication_seeds_per_family_min, 32);
        assert_eq!(plan.confidence_level_bps, 9_500);
        assert_eq!(plan.bootstrap_resamples, 10_000);
    }

    #[test]
    fn changed_metrics_remain_separate_from_true_no_change_metrics() {
        let f1 = METRIC_SPECS
            .iter()
            .find(|spec| spec.id == MetricId::ChangedFieldF1)
            .unwrap();
        let no_change = METRIC_SPECS
            .iter()
            .find(|spec| spec.id == MetricId::UnchangedStatePreservation)
            .unwrap();
        assert_eq!(f1.stratum, MetricStratum::ChangeBearing);
        assert_eq!(no_change.stratum, MetricStratum::TrueNoChange);
        assert!(!f1.defined_on_every_valid_trial);
        assert!(!no_change.defined_on_every_valid_trial);
    }

    #[test]
    fn full_state_accuracy_is_never_a_primary_metric() {
        let full = METRIC_SPECS
            .iter()
            .find(|spec| spec.id == MetricId::FullStateAccuracy)
            .unwrap();
        assert_eq!(full.role, MetricRole::DescriptiveOnly);
    }

    #[test]
    fn comparator_ids_are_unique_and_stably_ordered() {
        let plan = EUREKA_002_ANALYSIS_PLAN_V1;
        let ids: Vec<&str> = plan
            .eligible_comparators
            .iter()
            .map(|kind| kind.stable_id())
            .collect();
        let unique: HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len());
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn every_required_row_disposition_is_unique() {
        let unique: HashSet<CampaignRowDisposition> = REQUIRED_ROW_DISPOSITIONS
            .iter()
            .copied()
            .collect();
        assert_eq!(unique.len(), REQUIRED_ROW_DISPOSITIONS.len());
    }

    #[test]
    fn plan_requires_disjoint_comparator_selection_and_fit_corpora() {
        let plan = EUREKA_002_ANALYSIS_PLAN_V1;
        assert!(plan.comparator_selection_must_be_disjoint_from_fit);
        assert!(plan.copy_current_state_is_mandatory_construct_control);
        assert!(plan.zero_allowed_leakage_rows);
        assert!(plan.zero_allowed_lineage_invalid_rows_for_support);
        assert!(plan.external_replication_cannot_be_pooled_into_held_out_for_promotion);
    }

    #[test]
    fn replay_identity_is_deterministic_and_sensitive_to_primary_margin() {
        let plan = EUREKA_002_ANALYSIS_PLAN_V1;
        let a = plan.replay_digest();
        let b = plan.replay_digest();
        assert_eq!(a, b);
        assert_ne!(a, 0);

        let changed = Eureka002AnalysisPlanV1 {
            changed_field_f1_margin_bps: plan.changed_field_f1_margin_bps + 1,
            ..plan
        };
        assert_ne!(a, changed.replay_digest());
    }

    #[test]
    fn nonclaims_explicitly_block_causal_and_consciousness_promotion() {
        let plan = EUREKA_002_ANALYSIS_PLAN_V1;
        assert!(plan.explicit_non_claims.contains(&"causal understanding"));
        assert!(plan.explicit_non_claims.contains(&"consciousness"));
        assert!(plan.explicit_non_claims.contains(&"action authority"));
    }

    #[test]
    fn thresholds_are_valid_basis_point_values() {
        let plan = EUREKA_002_ANALYSIS_PLAN_V1;
        assert!(i32::from(plan.comparator_min_coverage_bps) <= BASIS_POINTS_ONE);
        assert!(i32::from(plan.candidate_min_coverage_bps) <= BASIS_POINTS_ONE);
        assert!(i32::from(plan.confidence_level_bps) < BASIS_POINTS_ONE);
        assert!(plan.changed_field_f1_margin_bps > 0);
        assert!(plan.changed_value_accuracy_margin_bps > 0);
    }
}

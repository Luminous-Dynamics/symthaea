// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim-boundary audits for SYM-ARCH-002B1 baseline contrasts.
//!
//! Sharing an RLS algorithm is not enough to call two conditions readout-matched.
//! The effective feature dimension determines the number of trainable readout
//! weights and the O(d^2) inverse-covariance state. This module makes that
//! distinction executable so a capacity-changing reference contrast cannot be
//! surfaced as clean representation-level evidence.

use crate::experiment_baselines::{
    MatchedBaselineFamilySpec, SimpleBaselineKind, SimpleBaselineSpec,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadoutShape {
    pub input_dimension: usize,
    pub state_dimension: usize,
    pub trainable_parameters: usize,
    pub covariance_elements: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContrastClaimCeiling {
    /// Schema, RLS protocol, effective readout shape, and paired random-seed index
    /// are matched. Differences may be attributed at the representation level,
    /// but not automatically to equal resource efficiency or a specific algebraic
    /// mechanism inside either encoder.
    RepresentationLevel,
    /// Useful as a benchmark/reference comparison, but capacity or protocol differs
    /// enough that a representation-only interpretation is not admissible.
    ReferenceOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineContrastAudit {
    pub left_kind: SimpleBaselineKind,
    pub right_kind: SimpleBaselineKind,
    pub left_spec_digest: String,
    pub right_spec_digest: String,
    pub same_feature_schema: bool,
    pub same_rls_contract: bool,
    pub same_effective_feature_dimension: bool,
    pub same_readout_shape: bool,
    /// Applies only when both encoders use representation randomness. One-hot has
    /// no representation seed and therefore does not fail this field by design.
    pub paired_random_seed_index: bool,
    pub left_readout_shape: ReadoutShape,
    pub right_readout_shape: ReadoutShape,
    pub replay_examples_left: usize,
    pub replay_examples_right: usize,
    pub temporal_state_bytes_left: usize,
    pub temporal_state_bytes_right: usize,
    pub claim_ceiling: ContrastClaimCeiling,
    pub qualifiers: Vec<String>,
}

fn effective_feature_dimension(spec: &SimpleBaselineSpec) -> Result<usize, String> {
    spec.validate()?;
    match spec.kind {
        SimpleBaselineKind::OneHotRls => spec.feature_schema.one_hot_dimension(),
        SimpleBaselineKind::FixedRandomTanhRls | SimpleBaselineKind::VanillaHdcRls => {
            Ok(spec.encoded_dimension)
        }
    }
}

fn readout_shape(spec: &SimpleBaselineSpec) -> Result<ReadoutShape, String> {
    let input_dimension = effective_feature_dimension(spec)?;
    let state_dimension = input_dimension
        .checked_add(usize::from(spec.rls.include_bias))
        .ok_or_else(|| "baseline readout state dimension overflow".to_string())?;
    let covariance_elements = state_dimension
        .checked_mul(state_dimension)
        .ok_or_else(|| "baseline readout covariance shape overflow".to_string())?;
    Ok(ReadoutShape {
        input_dimension,
        state_dimension,
        trainable_parameters: state_dimension,
        covariance_elements,
    })
}

fn uses_representation_randomness(kind: SimpleBaselineKind) -> bool {
    matches!(
        kind,
        SimpleBaselineKind::FixedRandomTanhRls | SimpleBaselineKind::VanillaHdcRls
    )
}

/// Audit whether a pair of simple baselines supports a representation-level
/// contrast under the frozen B1 contract.
///
/// The audit intentionally does not require equal fixed-encoder storage: encoder
/// resource cost is an outcome of the representation choice and must be reported
/// separately. It does require equal learner-visible schema, RLS settings, and
/// effective readout shape so predictive-capacity differences are not silently
/// attributed to the encoder.
pub fn audit_baseline_contrast(
    left: &SimpleBaselineSpec,
    right: &SimpleBaselineSpec,
) -> Result<BaselineContrastAudit, String> {
    left.validate()?;
    right.validate()?;
    if left.kind == right.kind {
        return Err("baseline contrast requires two different baseline kinds".into());
    }

    let left_readout_shape = readout_shape(left)?;
    let right_readout_shape = readout_shape(right)?;
    let same_feature_schema = left.feature_schema == right.feature_schema;
    let same_rls_contract = left.rls == right.rls;
    let same_effective_feature_dimension =
        left_readout_shape.input_dimension == right_readout_shape.input_dimension;
    let same_readout_shape = left_readout_shape == right_readout_shape;
    let paired_random_seed_index = if uses_representation_randomness(left.kind)
        && uses_representation_randomness(right.kind)
    {
        left.representation_seed == right.representation_seed
    } else {
        true
    };

    let mut qualifiers = Vec::new();
    if !same_feature_schema {
        qualifiers.push("learner-visible categorical schemas differ".into());
    }
    if !same_rls_contract {
        qualifiers.push("RLS ridge/forgetting/bias contract differs".into());
    }
    if !same_effective_feature_dimension {
        qualifiers.push(format!(
            "effective feature dimensions differ: left={} right={}",
            left_readout_shape.input_dimension, right_readout_shape.input_dimension
        ));
    }
    if !same_readout_shape {
        qualifiers.push(format!(
            "RLS trainable/covariance shape differs: left_state={} left_cov={} right_state={} right_cov={}",
            left_readout_shape.state_dimension,
            left_readout_shape.covariance_elements,
            right_readout_shape.state_dimension,
            right_readout_shape.covariance_elements
        ));
    }
    if !paired_random_seed_index {
        qualifiers.push("randomized encoders use different representation-seed indices".into());
    }

    let clean_representation_contrast = same_feature_schema
        && same_rls_contract
        && same_effective_feature_dimension
        && same_readout_shape
        && paired_random_seed_index;

    Ok(BaselineContrastAudit {
        left_kind: left.kind,
        right_kind: right.kind,
        left_spec_digest: left.digest()?,
        right_spec_digest: right.digest()?,
        same_feature_schema,
        same_rls_contract,
        same_effective_feature_dimension,
        same_readout_shape,
        paired_random_seed_index,
        left_readout_shape,
        right_readout_shape,
        // B1 simple baselines are replay-free and stateless in time by contract.
        replay_examples_left: 0,
        replay_examples_right: 0,
        temporal_state_bytes_left: 0,
        temporal_state_bytes_right: 0,
        claim_ceiling: if clean_representation_contrast {
            ContrastClaimCeiling::RepresentationLevel
        } else {
            ContrastClaimCeiling::ReferenceOnly
        },
        qualifiers,
    })
}

/// Audit all three pairwise contrasts emitted by one matched B1 family.
pub fn audit_matched_family(
    family: &MatchedBaselineFamilySpec,
) -> Result<Vec<BaselineContrastAudit>, String> {
    let specs = family.specs()?;
    debug_assert_eq!(specs.len(), 3);
    Ok(vec![
        audit_baseline_contrast(&specs[0], &specs[1])?,
        audit_baseline_contrast(&specs[0], &specs[2])?,
        audit_baseline_contrast(&specs[1], &specs[2])?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment_baselines::{
        CategoricalFeatureSchema, CategoricalFeatureSpec, RlsConfig,
        SIMPLE_BASELINE_SCHEMA_V1,
    };

    fn schema() -> CategoricalFeatureSchema {
        CategoricalFeatureSchema {
            features: vec![
                CategoricalFeatureSpec {
                    name: "a".into(),
                    values: vec![0, 1],
                },
                CategoricalFeatureSpec {
                    name: "b".into(),
                    values: vec![0, 1],
                },
            ],
        }
    }

    fn family(encoded_dimension: usize) -> MatchedBaselineFamilySpec {
        MatchedBaselineFamilySpec {
            feature_schema: schema(),
            encoded_dimension,
            representation_seed: 17,
            rls: RlsConfig {
                ridge: 1.0,
                forgetting_factor: 1.0,
                include_bias: true,
            },
        }
    }

    #[test]
    fn random_hdc_is_representation_level_when_readout_shape_is_matched() {
        let audits = audit_matched_family(&family(64)).unwrap();
        let random_hdc = audits
            .iter()
            .find(|audit| {
                audit.left_kind == SimpleBaselineKind::FixedRandomTanhRls
                    && audit.right_kind == SimpleBaselineKind::VanillaHdcRls
            })
            .unwrap();
        assert_eq!(
            random_hdc.claim_ceiling,
            ContrastClaimCeiling::RepresentationLevel
        );
        assert!(random_hdc.same_readout_shape);
        assert!(random_hdc.paired_random_seed_index);
        assert!(random_hdc.qualifiers.is_empty());
    }

    #[test]
    fn one_hot_random_is_reference_only_when_capacity_changes() {
        // The categorical source has four one-hot coordinates while the random
        // representation has 64, so the RLS trainable/covariance shape differs.
        let audits = audit_matched_family(&family(64)).unwrap();
        let one_hot_random = audits
            .iter()
            .find(|audit| {
                audit.left_kind == SimpleBaselineKind::OneHotRls
                    && audit.right_kind == SimpleBaselineKind::FixedRandomTanhRls
            })
            .unwrap();
        assert_eq!(
            one_hot_random.claim_ceiling,
            ContrastClaimCeiling::ReferenceOnly
        );
        assert!(!one_hot_random.same_effective_feature_dimension);
        assert!(!one_hot_random.same_readout_shape);
        assert!(!one_hot_random.qualifiers.is_empty());
    }

    #[test]
    fn one_hot_random_can_be_clean_when_effective_dimensions_match() {
        // one-hot dimension is exactly four here.
        let audits = audit_matched_family(&family(4)).unwrap();
        let one_hot_random = audits
            .iter()
            .find(|audit| {
                audit.left_kind == SimpleBaselineKind::OneHotRls
                    && audit.right_kind == SimpleBaselineKind::FixedRandomTanhRls
            })
            .unwrap();
        assert_eq!(
            one_hot_random.claim_ceiling,
            ContrastClaimCeiling::RepresentationLevel
        );
        assert!(one_hot_random.same_readout_shape);
    }

    #[test]
    fn randomized_representation_contrast_requires_paired_seed_index() {
        let specs = family(32).specs().unwrap();
        let random = specs
            .iter()
            .find(|spec| spec.kind == SimpleBaselineKind::FixedRandomTanhRls)
            .unwrap();
        let hdc = specs
            .iter()
            .find(|spec| spec.kind == SimpleBaselineKind::VanillaHdcRls)
            .unwrap();
        let mut different_seed = hdc.clone();
        different_seed.representation_seed += 1;
        let audit = audit_baseline_contrast(random, &different_seed).unwrap();
        assert_eq!(audit.claim_ceiling, ContrastClaimCeiling::ReferenceOnly);
        assert!(!audit.paired_random_seed_index);
    }

    #[test]
    fn rls_protocol_mismatch_downgrades_the_claim_ceiling() {
        let specs = family(32).specs().unwrap();
        let random = specs
            .iter()
            .find(|spec| spec.kind == SimpleBaselineKind::FixedRandomTanhRls)
            .unwrap();
        let hdc = specs
            .iter()
            .find(|spec| spec.kind == SimpleBaselineKind::VanillaHdcRls)
            .unwrap();
        let mut altered = hdc.clone();
        altered.rls.ridge = 2.0;
        let audit = audit_baseline_contrast(random, &altered).unwrap();
        assert_eq!(audit.claim_ceiling, ContrastClaimCeiling::ReferenceOnly);
        assert!(!audit.same_rls_contract);
    }

    #[test]
    fn same_kind_is_not_a_scientific_contrast() {
        let spec = SimpleBaselineSpec {
            schema: SIMPLE_BASELINE_SCHEMA_V1.into(),
            kind: SimpleBaselineKind::FixedRandomTanhRls,
            feature_schema: schema(),
            encoded_dimension: 16,
            representation_seed: 5,
            rls: RlsConfig::default(),
        };
        assert!(audit_baseline_contrast(&spec, &spec).is_err());
    }
}

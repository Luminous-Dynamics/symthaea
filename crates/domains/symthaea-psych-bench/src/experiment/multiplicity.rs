// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multiplicity-safe analysis-plan primitives for SYM-ARCH-002A7.
//!
//! This module does not manufacture p-values or effect estimates. It freezes the
//! claim-bearing hypothesis family and applies standard family-wise corrections to
//! valid preregistered inferential outputs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MULTIPLICITY_PLAN_SCHEMA_V1: &str = "symthaea.multiplicity-plan/v1";
const MULTIPLICITY_PLAN_HASH_DOMAIN: &[u8] = b"symthaea.multiplicity-plan.hash/v1";

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisTail {
    Greater,
    Less,
    TwoSided,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisRole {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisSpec {
    pub hypothesis_id: String,
    pub tail: HypothesisTail,
    pub role: HypothesisRole,
}

impl HypothesisSpec {
    fn validate(&self) -> Result<(), String> {
        if self.hypothesis_id.trim().is_empty() || self.hypothesis_id.trim() != self.hypothesis_id {
            return Err("hypothesis ids must be non-empty and already normalized".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiplicityPlan {
    pub schema: String,
    pub family_id: String,
    /// Family-wise Type-I error rate.
    pub family_alpha: f64,
    pub hypotheses: Vec<HypothesisSpec>,
}

impl MultiplicityPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MULTIPLICITY_PLAN_SCHEMA_V1 {
            return Err(format!("unsupported multiplicity-plan schema: {}", self.schema));
        }
        if self.family_id.trim().is_empty() || self.family_id.trim() != self.family_id {
            return Err("family_id must be non-empty and already normalized".into());
        }
        if !self.family_alpha.is_finite()
            || self.family_alpha <= 0.0
            || self.family_alpha >= 1.0
        {
            return Err("family_alpha must be finite in (0,1)".into());
        }
        if self.hypotheses.is_empty() {
            return Err("multiplicity family must contain at least one hypothesis".into());
        }
        let mut ids = BTreeSet::new();
        for hypothesis in &self.hypotheses {
            hypothesis.validate()?;
            if !ids.insert(hypothesis.hypothesis_id.as_str()) {
                return Err("duplicate hypothesis id in multiplicity family".into());
            }
        }
        Ok(())
    }

    /// Canonical digest independent of the input ordering of hypothesis specs.
    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let mut hypotheses = self.hypotheses.clone();
        hypotheses.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));
        canonical_hash(
            MULTIPLICITY_PLAN_HASH_DOMAIN,
            &(
                self.schema.as_str(),
                self.family_id.as_str(),
                self.family_alpha,
                hypotheses,
            ),
        )
    }

    pub fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }

    /// Per-comparison alpha for Bonferroni simultaneous confidence intervals.
    pub fn bonferroni_per_comparison_alpha(&self) -> Result<f64, String> {
        self.validate()?;
        Ok(self.family_alpha / self.hypothesis_count() as f64)
    }

    pub fn bonferroni_confidence_level(&self) -> Result<f64, String> {
        Ok(1.0 - self.bonferroni_per_comparison_alpha()?)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawHypothesisPValue {
    pub hypothesis_id: String,
    /// Raw p-value produced by the separately frozen test and tail.
    pub raw_p: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HolmAdjustedHypothesis {
    pub hypothesis_id: String,
    pub tail: HypothesisTail,
    pub role: HypothesisRole,
    pub raw_p: f64,
    pub holm_adjusted_p: f64,
    pub reject_at_family_alpha: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HolmFamilyResult {
    pub family_id: String,
    pub plan_digest: String,
    pub family_alpha: f64,
    pub hypotheses: Vec<HolmAdjustedHypothesis>,
}

/// Apply Holm's step-down family-wise correction.
///
/// Raw p-values must correspond exactly to the hypothesis ids/tails frozen in the
/// plan. This function cannot verify how a p-value was generated; that test remains
/// part of the preregistered analysis contract.
pub fn apply_holm(
    plan: &MultiplicityPlan,
    raw: &[RawHypothesisPValue],
) -> Result<HolmFamilyResult, String> {
    plan.validate()?;
    if raw.len() != plan.hypothesis_count() {
        return Err("raw p-value count must match the frozen hypothesis family".into());
    }

    let mut raw_ids = BTreeSet::new();
    for value in raw {
        if value.hypothesis_id.trim().is_empty() || value.hypothesis_id.trim() != value.hypothesis_id {
            return Err("raw p-value hypothesis ids must be non-empty and normalized".into());
        }
        if !value.raw_p.is_finite() || !(0.0..=1.0).contains(&value.raw_p) {
            return Err("raw p-values must be finite in [0,1]".into());
        }
        if !raw_ids.insert(value.hypothesis_id.as_str()) {
            return Err("duplicate hypothesis id in raw p-values".into());
        }
    }

    let planned_ids: BTreeSet<&str> = plan
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.hypothesis_id.as_str())
        .collect();
    if raw_ids != planned_ids {
        return Err("raw p-value hypothesis ids do not match frozen family".into());
    }

    let mut work: Vec<(HypothesisSpec, f64)> = plan
        .hypotheses
        .iter()
        .cloned()
        .map(|spec| {
            let raw_p = raw
                .iter()
                .find(|value| value.hypothesis_id == spec.hypothesis_id)
                .expect("id-set equality established")
                .raw_p;
            (spec, raw_p)
        })
        .collect();

    // Deterministic tie-break by frozen hypothesis id.
    work.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.hypothesis_id.cmp(&right.0.hypothesis_id))
    });

    let count = work.len();
    let mut running_adjusted = 0.0f64;
    let mut adjusted = Vec::with_capacity(count);
    for (rank, (spec, raw_p)) in work.into_iter().enumerate() {
        let step_adjusted = ((count - rank) as f64 * raw_p).min(1.0);
        running_adjusted = running_adjusted.max(step_adjusted);
        adjusted.push(HolmAdjustedHypothesis {
            hypothesis_id: spec.hypothesis_id,
            tail: spec.tail,
            role: spec.role,
            raw_p,
            holm_adjusted_p: running_adjusted,
            reject_at_family_alpha: running_adjusted <= plan.family_alpha,
        });
    }

    // Output ordering is canonical by id, not by observed significance rank.
    adjusted.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));
    Ok(HolmFamilyResult {
        family_id: plan.family_id.clone(),
        plan_digest: plan.digest()?,
        family_alpha: plan.family_alpha,
        hypotheses: adjusted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(order: &[&str]) -> MultiplicityPlan {
        MultiplicityPlan {
            schema: MULTIPLICITY_PLAN_SCHEMA_V1.into(),
            family_id: "sym-arch-002-primary-family".into(),
            family_alpha: 0.05,
            hypotheses: order
                .iter()
                .enumerate()
                .map(|(index, id)| HypothesisSpec {
                    hypothesis_id: (*id).into(),
                    tail: if index % 2 == 0 {
                        HypothesisTail::Greater
                    } else {
                        HypothesisTail::TwoSided
                    },
                    role: HypothesisRole::Primary,
                })
                .collect(),
        }
    }

    #[test]
    fn holm_matches_known_step_down_example() {
        let plan = MultiplicityPlan {
            schema: MULTIPLICITY_PLAN_SCHEMA_V1.into(),
            family_id: "family".into(),
            family_alpha: 0.05,
            hypotheses: vec![
                HypothesisSpec {
                    hypothesis_id: "h1".into(),
                    tail: HypothesisTail::Greater,
                    role: HypothesisRole::Primary,
                },
                HypothesisSpec {
                    hypothesis_id: "h2".into(),
                    tail: HypothesisTail::Greater,
                    role: HypothesisRole::Primary,
                },
                HypothesisSpec {
                    hypothesis_id: "h3".into(),
                    tail: HypothesisTail::Greater,
                    role: HypothesisRole::Primary,
                },
            ],
        };
        let result = apply_holm(
            &plan,
            &[
                RawHypothesisPValue {
                    hypothesis_id: "h1".into(),
                    raw_p: 0.01,
                },
                RawHypothesisPValue {
                    hypothesis_id: "h2".into(),
                    raw_p: 0.04,
                },
                RawHypothesisPValue {
                    hypothesis_id: "h3".into(),
                    raw_p: 0.03,
                },
            ],
        )
        .unwrap();
        let h1 = result
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.hypothesis_id == "h1")
            .unwrap();
        let h2 = result
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.hypothesis_id == "h2")
            .unwrap();
        let h3 = result
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.hypothesis_id == "h3")
            .unwrap();
        assert!((h1.holm_adjusted_p - 0.03).abs() < 1e-12);
        assert!((h2.holm_adjusted_p - 0.06).abs() < 1e-12);
        assert!((h3.holm_adjusted_p - 0.06).abs() < 1e-12);
        assert!(h1.reject_at_family_alpha);
        assert!(!h2.reject_at_family_alpha);
        assert!(!h3.reject_at_family_alpha);
    }

    #[test]
    fn plan_digest_is_order_independent_when_specs_are_identical() {
        let first = MultiplicityPlan {
            schema: MULTIPLICITY_PLAN_SCHEMA_V1.into(),
            family_id: "family".into(),
            family_alpha: 0.05,
            hypotheses: vec![
                HypothesisSpec {
                    hypothesis_id: "a".into(),
                    tail: HypothesisTail::Greater,
                    role: HypothesisRole::Primary,
                },
                HypothesisSpec {
                    hypothesis_id: "b".into(),
                    tail: HypothesisTail::TwoSided,
                    role: HypothesisRole::Secondary,
                },
            ],
        };
        let second = MultiplicityPlan {
            hypotheses: first.hypotheses.iter().cloned().rev().collect(),
            ..first.clone()
        };
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn plan_digest_changes_when_tail_changes() {
        let first = plan(&["a", "b"]);
        let mut second = first.clone();
        second.hypotheses[0].tail = HypothesisTail::Less;
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn bonferroni_interval_alpha_controls_family_rate() {
        let plan = MultiplicityPlan {
            schema: MULTIPLICITY_PLAN_SCHEMA_V1.into(),
            family_id: "family".into(),
            family_alpha: 0.05,
            hypotheses: (0..4)
                .map(|index| HypothesisSpec {
                    hypothesis_id: format!("h{index}"),
                    tail: HypothesisTail::TwoSided,
                    role: HypothesisRole::Secondary,
                })
                .collect(),
        };
        assert!((plan.bonferroni_per_comparison_alpha().unwrap() - 0.0125).abs() < 1e-12);
        assert!((plan.bonferroni_confidence_level().unwrap() - 0.9875).abs() < 1e-12);
    }

    #[test]
    fn raw_family_must_match_frozen_ids_exactly() {
        let plan = plan(&["a", "b"]);
        let missing = [RawHypothesisPValue {
            hypothesis_id: "a".into(),
            raw_p: 0.01,
        }];
        assert!(apply_holm(&plan, &missing).is_err());

        let wrong = [
            RawHypothesisPValue {
                hypothesis_id: "a".into(),
                raw_p: 0.01,
            },
            RawHypothesisPValue {
                hypothesis_id: "c".into(),
                raw_p: 0.02,
            },
        ];
        assert!(apply_holm(&plan, &wrong).is_err());
    }

    #[test]
    fn non_finite_or_out_of_range_p_values_fail_closed() {
        let plan = plan(&["a"]);
        for raw_p in [f64::NAN, -0.01, 1.01] {
            let raw = [RawHypothesisPValue {
                hypothesis_id: "a".into(),
                raw_p,
            }];
            assert!(apply_holm(&plan, &raw).is_err());
        }
    }

    #[test]
    fn duplicate_hypothesis_ids_fail_closed() {
        let mut invalid = plan(&["a", "b"]);
        invalid.hypotheses[1].hypothesis_id = "a".into();
        assert!(invalid.validate().is_err());
    }
}

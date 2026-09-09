// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit primary estimand for Genesis mutation-rescue counterfactuals.
//!
//! The causal replay session freezes the world, stochastic inputs, intervention, shock protocol,
//! and evaluation window. This module closes the remaining analysis-choice gap by binding that
//! session to one versioned primary contrast. V1 uses normalized population-deficit area because it
//! preserves the existing evolvability metric's interpretation: lower is better. The causal effect
//! is therefore `revert - sham`; a positive value means removing the candidate mutation worsened
//! resilience under the matched experimental world.
//!
//! This module still does not claim causality from one effect value. The sham branch must first
//! reproduce the observed natural endpoint exactly, and replication/inference remain later layers.

use serde::{Deserialize, Serialize};

use crate::{
    CausalRescueReplaySessionErrorV1, CausalRescueReplaySessionV1,
    ValidatedCausalRescueReplaySessionV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRescuePrimaryMetricV1 {
    NormalizedPopulationDeficitArea,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRescueContrastV1 {
    /// Primary effect = `revert_metric - sham_metric`.
    RevertMinusSham,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRescueShamFidelityV1 {
    /// The sham branch must reproduce the observed natural execution capsule exactly at the
    /// protocol endpoint before the intervention contrast can be interpreted.
    ExactNaturalExecutionCapsule,
}

/// Versioned analysis choice for one causal replay session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalRescueEstimandV1 {
    primary_metric: CausalRescuePrimaryMetricV1,
    contrast: CausalRescueContrastV1,
    sham_fidelity: CausalRescueShamFidelityV1,
}

impl CausalRescueEstimandV1 {
    pub const fn canonical() -> Self {
        Self {
            primary_metric: CausalRescuePrimaryMetricV1::NormalizedPopulationDeficitArea,
            contrast: CausalRescueContrastV1::RevertMinusSham,
            sham_fidelity: CausalRescueShamFidelityV1::ExactNaturalExecutionCapsule,
        }
    }

    pub fn primary_metric(self) -> CausalRescuePrimaryMetricV1 {
        self.primary_metric
    }

    pub fn contrast(self) -> CausalRescueContrastV1 {
        self.contrast
    }

    pub fn sham_fidelity(self) -> CausalRescueShamFidelityV1 {
        self.sham_fidelity
    }

    pub fn compute_effect(
        self,
        sham_normalized_deficit_area: f64,
        revert_normalized_deficit_area: f64,
    ) -> Result<CausalRescueEffectV1, CausalRescueEstimandErrorV1> {
        validate_metric("sham", sham_normalized_deficit_area)?;
        validate_metric("revert", revert_normalized_deficit_area)?;
        let effect = revert_normalized_deficit_area - sham_normalized_deficit_area;
        if !effect.is_finite() {
            return Err(CausalRescueEstimandErrorV1::NonFiniteEffect {
                sham_bits: sham_normalized_deficit_area.to_bits(),
                revert_bits: revert_normalized_deficit_area.to_bits(),
            });
        }
        Ok(CausalRescueEffectV1 {
            sham_metric_bits: sham_normalized_deficit_area.to_bits(),
            revert_metric_bits: revert_normalized_deficit_area.to_bits(),
            effect_bits: effect.to_bits(),
        })
    }
}

impl Default for CausalRescueEstimandV1 {
    fn default() -> Self {
        Self::canonical()
    }
}

/// Persisted experiment-level binding of one replay session to its primary estimand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRescueExperimentSpecV1 {
    session: CausalRescueReplaySessionV1,
    estimand: CausalRescueEstimandV1,
}

impl CausalRescueExperimentSpecV1 {
    pub fn new(session: CausalRescueReplaySessionV1) -> Self {
        Self {
            session,
            estimand: CausalRescueEstimandV1::canonical(),
        }
    }

    pub fn validate(
        self,
    ) -> Result<ValidatedCausalRescueExperimentSpecV1, CausalRescueExperimentSpecErrorV1> {
        if self.estimand != CausalRescueEstimandV1::canonical() {
            return Err(CausalRescueExperimentSpecErrorV1::NonCanonicalEstimand);
        }
        let session = self
            .session
            .validate()
            .map_err(CausalRescueExperimentSpecErrorV1::Session)?;
        Ok(ValidatedCausalRescueExperimentSpecV1 {
            session,
            estimand: self.estimand,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ValidatedCausalRescueExperimentSpecV1 {
    session: ValidatedCausalRescueReplaySessionV1,
    estimand: CausalRescueEstimandV1,
}

impl ValidatedCausalRescueExperimentSpecV1 {
    pub fn session(&self) -> &ValidatedCausalRescueReplaySessionV1 {
        &self.session
    }

    pub fn estimand(&self) -> CausalRescueEstimandV1 {
        self.estimand
    }
}

#[derive(Debug)]
pub enum CausalRescueExperimentSpecErrorV1 {
    NonCanonicalEstimand,
    Session(CausalRescueReplaySessionErrorV1),
}

/// Exact persisted primary-effect value. This is an effect estimate, not a causal conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalRescueEffectV1 {
    sham_metric_bits: u64,
    revert_metric_bits: u64,
    effect_bits: u64,
}

impl CausalRescueEffectV1 {
    pub fn sham_metric(self) -> f64 {
        f64::from_bits(self.sham_metric_bits)
    }

    pub fn revert_metric(self) -> f64 {
        f64::from_bits(self.revert_metric_bits)
    }

    /// `revert - sham`; positive means the reversion worsened normalized population deficit area.
    pub fn effect(self) -> f64 {
        f64::from_bits(self.effect_bits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalRescueEstimandErrorV1 {
    NonFiniteMetric {
        role: &'static str,
        bits: u64,
    },
    NegativeMetric {
        role: &'static str,
        bits: u64,
    },
    NonFiniteEffect {
        sham_bits: u64,
        revert_bits: u64,
    },
}

fn validate_metric(role: &'static str, value: f64) -> Result<(), CausalRescueEstimandErrorV1> {
    if !value.is_finite() {
        return Err(CausalRescueEstimandErrorV1::NonFiniteMetric {
            role,
            bits: value.to_bits(),
        });
    }
    if value < 0.0 {
        return Err(CausalRescueEstimandErrorV1::NegativeMetric {
            role,
            bits: value.to_bits(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_estimand_names_metric_contrast_and_sham_fidelity_explicitly() {
        let estimand = CausalRescueEstimandV1::canonical();
        assert_eq!(
            estimand.primary_metric(),
            CausalRescuePrimaryMetricV1::NormalizedPopulationDeficitArea
        );
        assert_eq!(estimand.contrast(), CausalRescueContrastV1::RevertMinusSham);
        assert_eq!(
            estimand.sham_fidelity(),
            CausalRescueShamFidelityV1::ExactNaturalExecutionCapsule
        );
    }

    #[test]
    fn positive_effect_means_reversion_worsened_resilience() {
        let effect = CausalRescueEstimandV1::canonical()
            .compute_effect(1.25, 2.0)
            .expect("finite nonnegative metrics");
        assert_eq!(effect.sham_metric().to_bits(), 1.25f64.to_bits());
        assert_eq!(effect.revert_metric().to_bits(), 2.0f64.to_bits());
        assert_eq!(effect.effect().to_bits(), 0.75f64.to_bits());
    }

    #[test]
    fn effect_preserves_signed_zero_inputs_but_rejects_invalid_metric_domain() {
        let effect = CausalRescueEstimandV1::canonical()
            .compute_effect(-0.0, 0.0)
            .expect("signed zero is a valid nonnegative metric");
        assert_eq!(effect.sham_metric().to_bits(), (-0.0f64).to_bits());
        assert_eq!(effect.revert_metric().to_bits(), 0.0f64.to_bits());

        assert!(matches!(
            CausalRescueEstimandV1::canonical().compute_effect(-0.01, 1.0),
            Err(CausalRescueEstimandErrorV1::NegativeMetric { role: "sham", .. })
        ));
        assert!(matches!(
            CausalRescueEstimandV1::canonical().compute_effect(f64::NAN, 1.0),
            Err(CausalRescueEstimandErrorV1::NonFiniteMetric { role: "sham", .. })
        ));
    }

    #[test]
    fn estimand_json_round_trip_is_exact() {
        let estimand = CausalRescueEstimandV1::canonical();
        let json = serde_json::to_string(&estimand).expect("serialize estimand");
        let restored: CausalRescueEstimandV1 = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, estimand);
    }
}

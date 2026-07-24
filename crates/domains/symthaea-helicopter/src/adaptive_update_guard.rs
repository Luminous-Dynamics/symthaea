// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Guardrails for adaptive-controller parameter updates.
//!
//! Flight-time learning is locked by default. Ground-training updates are
//! bounded by parameter limits, absolute and relative step size, gradient norm,
//! loss improvement, lineage, and evidence identifiers. Shadow evaluation may
//! assess a proposal but cannot authorize it for active control.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveUpdateMode {
    GroundTraining,
    ShadowEvaluation,
    FlightLocked,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveParameterBound {
    pub parameter_id: String,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveUpdateGuardConfig {
    pub schema_version: String,
    pub guard_id: String,
    pub maximum_absolute_delta: f64,
    pub maximum_relative_delta: f64,
    pub maximum_gradient_norm: f64,
    pub minimum_loss_improvement: f64,
    pub require_dataset_digest: bool,
    pub require_evidence_id: bool,
    pub parameter_bounds: Vec<AdaptiveParameterBound>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveUpdateProposal {
    pub update_id: String,
    pub parameter_id: String,
    pub current_value: f64,
    pub proposed_value: f64,
    pub gradient_norm: f64,
    pub validation_loss_before: f64,
    pub validation_loss_after: f64,
    pub parent_checkpoint_digest: String,
    pub dataset_digest: Option<String>,
    pub evidence_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveUpdateDisposition {
    Accepted,
    ShadowOnly,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdaptiveUpdateRejection {
    FlightUpdatesLocked,
    UnknownParameter,
    ProposedValueOutOfBounds { minimum: f64, maximum: f64 },
    AbsoluteDeltaExceeded { observed: f64, maximum: f64 },
    RelativeDeltaExceeded { observed: f64, maximum: f64 },
    GradientNormExceeded { observed: f64, maximum: f64 },
    InsufficientLossImprovement { observed: f64, minimum: f64 },
    MissingParentCheckpoint,
    MissingDatasetDigest,
    MissingEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveUpdateDecision {
    pub schema_version: String,
    pub guard_id: String,
    pub update_id: String,
    pub mode: AdaptiveUpdateMode,
    pub disposition: AdaptiveUpdateDisposition,
    pub absolute_delta: f64,
    pub relative_delta: f64,
    pub loss_improvement: f64,
    pub rollback_token: Option<String>,
    pub rejections: Vec<AdaptiveUpdateRejection>,
}

impl AdaptiveUpdateDecision {
    pub fn canonical_json(&self) -> Result<Vec<u8>, AdaptiveUpdateGuardError> {
        let mut canonical = self.clone();
        canonical.rejections.sort_by_key(rejection_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| AdaptiveUpdateGuardError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, AdaptiveUpdateGuardError> {
        fnv1a64(&self.canonical_json()?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveUpdateGuardError {
    InvalidConfiguration,
    DuplicateParameterBound(String),
    InvalidProposal,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct AdaptiveUpdateGuard {
    config: AdaptiveUpdateGuardConfig,
    bounds: BTreeMap<String, AdaptiveParameterBound>,
    accepted_updates: BTreeSet<String>,
}

impl AdaptiveUpdateGuard {
    pub fn new(config: AdaptiveUpdateGuardConfig) -> Result<Self, AdaptiveUpdateGuardError> {
        if config.schema_version.trim().is_empty()
            || config.guard_id.trim().is_empty()
            || !config.maximum_absolute_delta.is_finite()
            || config.maximum_absolute_delta < 0.0
            || !config.maximum_relative_delta.is_finite()
            || config.maximum_relative_delta < 0.0
            || !config.maximum_gradient_norm.is_finite()
            || config.maximum_gradient_norm < 0.0
            || !config.minimum_loss_improvement.is_finite()
            || config.parameter_bounds.is_empty()
        {
            return Err(AdaptiveUpdateGuardError::InvalidConfiguration);
        }
        let mut bounds = BTreeMap::new();
        for bound in &config.parameter_bounds {
            if bound.parameter_id.trim().is_empty()
                || !bound.minimum.is_finite()
                || !bound.maximum.is_finite()
                || bound.minimum > bound.maximum
            {
                return Err(AdaptiveUpdateGuardError::InvalidConfiguration);
            }
            if bounds
                .insert(bound.parameter_id.clone(), bound.clone())
                .is_some()
            {
                return Err(AdaptiveUpdateGuardError::DuplicateParameterBound(
                    bound.parameter_id.clone(),
                ));
            }
        }
        Ok(Self {
            config,
            bounds,
            accepted_updates: BTreeSet::new(),
        })
    }

    pub fn evaluate(
        &mut self,
        mode: AdaptiveUpdateMode,
        proposal: &AdaptiveUpdateProposal,
    ) -> Result<AdaptiveUpdateDecision, AdaptiveUpdateGuardError> {
        validate_proposal(proposal)?;
        let absolute_delta = (proposal.proposed_value - proposal.current_value).abs();
        let denominator = proposal.current_value.abs().max(1e-9);
        let relative_delta = absolute_delta / denominator;
        let loss_improvement = proposal.validation_loss_before - proposal.validation_loss_after;
        let mut rejections = Vec::new();

        if mode == AdaptiveUpdateMode::FlightLocked {
            rejections.push(AdaptiveUpdateRejection::FlightUpdatesLocked);
        }
        let Some(bound) = self.bounds.get(&proposal.parameter_id) else {
            rejections.push(AdaptiveUpdateRejection::UnknownParameter);
            return Ok(self.decision(
                mode,
                proposal,
                absolute_delta,
                relative_delta,
                loss_improvement,
                rejections,
            )?);
        };
        if proposal.proposed_value < bound.minimum || proposal.proposed_value > bound.maximum {
            rejections.push(AdaptiveUpdateRejection::ProposedValueOutOfBounds {
                minimum: bound.minimum,
                maximum: bound.maximum,
            });
        }
        if absolute_delta > self.config.maximum_absolute_delta {
            rejections.push(AdaptiveUpdateRejection::AbsoluteDeltaExceeded {
                observed: absolute_delta,
                maximum: self.config.maximum_absolute_delta,
            });
        }
        if relative_delta > self.config.maximum_relative_delta {
            rejections.push(AdaptiveUpdateRejection::RelativeDeltaExceeded {
                observed: relative_delta,
                maximum: self.config.maximum_relative_delta,
            });
        }
        if proposal.gradient_norm > self.config.maximum_gradient_norm {
            rejections.push(AdaptiveUpdateRejection::GradientNormExceeded {
                observed: proposal.gradient_norm,
                maximum: self.config.maximum_gradient_norm,
            });
        }
        if loss_improvement < self.config.minimum_loss_improvement {
            rejections.push(AdaptiveUpdateRejection::InsufficientLossImprovement {
                observed: loss_improvement,
                minimum: self.config.minimum_loss_improvement,
            });
        }
        if proposal.parent_checkpoint_digest.trim().is_empty() {
            rejections.push(AdaptiveUpdateRejection::MissingParentCheckpoint);
        }
        if self.config.require_dataset_digest
            && proposal
                .dataset_digest
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            rejections.push(AdaptiveUpdateRejection::MissingDatasetDigest);
        }
        if self.config.require_evidence_id
            && proposal
                .evidence_id
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            rejections.push(AdaptiveUpdateRejection::MissingEvidence);
        }

        let decision = self.decision(
            mode,
            proposal,
            absolute_delta,
            relative_delta,
            loss_improvement,
            rejections,
        )?;
        if decision.disposition == AdaptiveUpdateDisposition::Accepted {
            self.accepted_updates.insert(proposal.update_id.clone());
        }
        Ok(decision)
    }

    pub fn was_accepted(&self, update_id: &str) -> bool {
        self.accepted_updates.contains(update_id)
    }

    fn decision(
        &self,
        mode: AdaptiveUpdateMode,
        proposal: &AdaptiveUpdateProposal,
        absolute_delta: f64,
        relative_delta: f64,
        loss_improvement: f64,
        mut rejections: Vec<AdaptiveUpdateRejection>,
    ) -> Result<AdaptiveUpdateDecision, AdaptiveUpdateGuardError> {
        rejections.sort_by_key(rejection_sort_key);
        let disposition = if !rejections.is_empty() {
            AdaptiveUpdateDisposition::Rejected
        } else if mode == AdaptiveUpdateMode::ShadowEvaluation {
            AdaptiveUpdateDisposition::ShadowOnly
        } else {
            AdaptiveUpdateDisposition::Accepted
        };
        let rollback_token = if disposition == AdaptiveUpdateDisposition::Accepted {
            Some(fnv1a64(&serde_json::to_vec(proposal).map_err(|_| {
                AdaptiveUpdateGuardError::SerializationFailed
            })?)?)
        } else {
            None
        };
        Ok(AdaptiveUpdateDecision {
            schema_version: self.config.schema_version.clone(),
            guard_id: self.config.guard_id.clone(),
            update_id: proposal.update_id.clone(),
            mode,
            disposition,
            absolute_delta,
            relative_delta,
            loss_improvement,
            rollback_token,
            rejections,
        })
    }
}

fn validate_proposal(proposal: &AdaptiveUpdateProposal) -> Result<(), AdaptiveUpdateGuardError> {
    let values = [
        proposal.current_value,
        proposal.proposed_value,
        proposal.gradient_norm,
        proposal.validation_loss_before,
        proposal.validation_loss_after,
    ];
    if proposal.update_id.trim().is_empty()
        || proposal.parameter_id.trim().is_empty()
        || values.iter().any(|value| !value.is_finite())
        || proposal.gradient_norm < 0.0
        || proposal.validation_loss_before < 0.0
        || proposal.validation_loss_after < 0.0
    {
        return Err(AdaptiveUpdateGuardError::InvalidProposal);
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> Result<String, AdaptiveUpdateGuardError> {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn rejection_sort_key(rejection: &AdaptiveUpdateRejection) -> String {
    format!("{rejection:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> AdaptiveUpdateGuard {
        AdaptiveUpdateGuard::new(AdaptiveUpdateGuardConfig {
            schema_version: "symthaea.helicopter.adaptive-update.v1".into(),
            guard_id: "guard-a".into(),
            maximum_absolute_delta: 0.2,
            maximum_relative_delta: 0.5,
            maximum_gradient_norm: 2.0,
            minimum_loss_improvement: 0.01,
            require_dataset_digest: true,
            require_evidence_id: true,
            parameter_bounds: vec![AdaptiveParameterBound {
                parameter_id: "gain.roll".into(),
                minimum: 0.0,
                maximum: 2.0,
            }],
        })
        .unwrap()
    }

    fn proposal() -> AdaptiveUpdateProposal {
        AdaptiveUpdateProposal {
            update_id: "update-1".into(),
            parameter_id: "gain.roll".into(),
            current_value: 1.0,
            proposed_value: 1.1,
            gradient_norm: 0.5,
            validation_loss_before: 0.5,
            validation_loss_after: 0.4,
            parent_checkpoint_digest: "sha256:parent".into(),
            dataset_digest: Some("sha256:data".into()),
            evidence_id: Some("evidence:validation".into()),
        }
    }

    #[test]
    fn ground_update_can_be_accepted() {
        let mut guard = guard();
        let decision = guard
            .evaluate(AdaptiveUpdateMode::GroundTraining, &proposal())
            .unwrap();
        assert_eq!(decision.disposition, AdaptiveUpdateDisposition::Accepted);
        assert!(decision.rollback_token.is_some());
        assert!(guard.was_accepted("update-1"));
    }

    #[test]
    fn shadow_update_is_never_committed() {
        let mut guard = guard();
        let decision = guard
            .evaluate(AdaptiveUpdateMode::ShadowEvaluation, &proposal())
            .unwrap();
        assert_eq!(decision.disposition, AdaptiveUpdateDisposition::ShadowOnly);
        assert!(!guard.was_accepted("update-1"));
    }

    #[test]
    fn flight_update_is_rejected() {
        let mut guard = guard();
        let decision = guard
            .evaluate(AdaptiveUpdateMode::FlightLocked, &proposal())
            .unwrap();
        assert_eq!(decision.disposition, AdaptiveUpdateDisposition::Rejected);
        assert!(
            decision
                .rejections
                .contains(&AdaptiveUpdateRejection::FlightUpdatesLocked)
        );
    }

    #[test]
    fn excessive_step_is_rejected() {
        let mut guard = guard();
        let mut update = proposal();
        update.proposed_value = 1.8;
        let decision = guard
            .evaluate(AdaptiveUpdateMode::GroundTraining, &update)
            .unwrap();
        assert_eq!(decision.disposition, AdaptiveUpdateDisposition::Rejected);
        assert!(decision.rejections.iter().any(|reason| matches!(
            reason,
            AdaptiveUpdateRejection::AbsoluteDeltaExceeded { .. }
        )));
    }

    #[test]
    fn missing_lineage_is_rejected() {
        let mut guard = guard();
        let mut update = proposal();
        update.parent_checkpoint_digest.clear();
        let decision = guard
            .evaluate(AdaptiveUpdateMode::GroundTraining, &update)
            .unwrap();
        assert!(
            decision
                .rejections
                .contains(&AdaptiveUpdateRejection::MissingParentCheckpoint)
        );
    }
}

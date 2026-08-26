// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Preregistered decision semantics for chemosensation experiments.
//!
//! This module deliberately separates three outcomes:
//!
//! - [`ExperimentDecision::Confirmed`]: every preregistered confirmation gate passes;
//! - [`ExperimentDecision::NotConfirmed`]: at least one metric crosses a separately
//!   preregistered practical-failure boundary;
//! - [`ExperimentDecision::Inconclusive`]: neither confirmation nor practical failure
//!   has been established.
//!
//! Merely missing a confirmation threshold is therefore **not** a negative result.
//! The distinction is important for small pilots, noisy sensors, and underpowered
//! physical experiments.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Evidence source admitted by a particular frozen decision protocol.
///
/// These variants are intentionally categorical rather than ordinal. A protocol
/// that requires held-out physical observations cannot be satisfied by a simulator
/// merely because the simulator is otherwise high quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChemicalEvidenceLevel {
    /// Deterministic software fixture or simulator output.
    SimulatedFixture,
    /// Replay of a previously recorded sensor trace.
    RecordedReplay,
    /// New physical sensor measurements collected on a bench/dev setup.
    BenchPhysicalObservation,
    /// New physical measurements from a frozen held-out evaluation set/session.
    HeldOutPhysicalObservation,
}

/// Dataset/session role for one evaluation receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaluationPartition {
    Calibration,
    Development,
    Holdout,
}

/// Direction in which a metric is considered better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateDirection {
    /// Larger values are better (for example classification accuracy).
    AtLeast,
    /// Smaller values are better (for example carryover or error).
    AtMost,
}

/// Outcome of one preregistered metric gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateOutcome {
    ConfirmationPass,
    PracticalFailure,
    Indeterminate,
}

/// Aggregate decision for one protocol evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperimentDecision {
    Confirmed,
    NotConfirmed,
    Inconclusive,
}

/// One metric gate frozen before outcome-bearing evaluation begins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricGate {
    pub id: String,
    pub direction: GateDirection,
    pub confirmation_threshold: f64,
    /// Boundary beyond which there is positive evidence of a practically
    /// important failure. `None` means this metric alone cannot produce a
    /// `NotConfirmed` decision.
    pub practical_failure_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecisionProtocolError {
    BlankProtocolId,
    BlankVersion,
    EmptyGates,
    BlankGateId,
    DuplicateGate(String),
    NonFiniteConfirmationThreshold(String),
    NonFinitePracticalFailureThreshold(String),
    InvalidThresholdOrdering(String),
}

impl MetricGate {
    pub fn new(
        id: impl Into<String>,
        direction: GateDirection,
        confirmation_threshold: f64,
        practical_failure_threshold: Option<f64>,
    ) -> Result<Self, DecisionProtocolError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(DecisionProtocolError::BlankGateId);
        }
        if !confirmation_threshold.is_finite() {
            return Err(DecisionProtocolError::NonFiniteConfirmationThreshold(id));
        }
        if let Some(failure) = practical_failure_threshold {
            if !failure.is_finite() {
                return Err(DecisionProtocolError::NonFinitePracticalFailureThreshold(id));
            }
            let ordered = match direction {
                GateDirection::AtLeast => failure < confirmation_threshold,
                GateDirection::AtMost => failure > confirmation_threshold,
            };
            if !ordered {
                return Err(DecisionProtocolError::InvalidThresholdOrdering(id));
            }
        }

        Ok(Self {
            id,
            direction,
            confirmation_threshold,
            practical_failure_threshold,
        })
    }

    fn assess(&self, value: f64) -> GateOutcome {
        let passes = match self.direction {
            GateDirection::AtLeast => value >= self.confirmation_threshold,
            GateDirection::AtMost => value <= self.confirmation_threshold,
        };
        if passes {
            return GateOutcome::ConfirmationPass;
        }

        let practical_failure = self
            .practical_failure_threshold
            .is_some_and(|failure| match self.direction {
                GateDirection::AtLeast => value <= failure,
                GateDirection::AtMost => value >= failure,
            });

        if practical_failure {
            GateOutcome::PracticalFailure
        } else {
            GateOutcome::Indeterminate
        }
    }
}

/// Frozen decision contract for one experiment family/version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalDecisionProtocol {
    pub protocol_id: String,
    pub version: String,
    pub required_evidence: ChemicalEvidenceLevel,
    pub required_partition: EvaluationPartition,
    pub gates: Vec<MetricGate>,
}

impl ChemicalDecisionProtocol {
    pub fn new(
        protocol_id: impl Into<String>,
        version: impl Into<String>,
        required_evidence: ChemicalEvidenceLevel,
        required_partition: EvaluationPartition,
        gates: Vec<MetricGate>,
    ) -> Result<Self, DecisionProtocolError> {
        let protocol_id = protocol_id.into();
        if protocol_id.trim().is_empty() {
            return Err(DecisionProtocolError::BlankProtocolId);
        }
        let version = version.into();
        if version.trim().is_empty() {
            return Err(DecisionProtocolError::BlankVersion);
        }
        if gates.is_empty() {
            return Err(DecisionProtocolError::EmptyGates);
        }

        let mut seen = BTreeSet::new();
        for gate in &gates {
            if !seen.insert(gate.id.clone()) {
                return Err(DecisionProtocolError::DuplicateGate(gate.id.clone()));
            }
        }

        Ok(Self {
            protocol_id,
            version,
            required_evidence,
            required_partition,
            gates,
        })
    }

    /// Evaluate an outcome-bearing run against the frozen decision contract.
    ///
    /// Evidence source and partition must match the preregistration exactly.
    /// This prevents development/simulator results from silently satisfying a
    /// held-out physical claim.
    pub fn evaluate(
        &self,
        evidence: ChemicalEvidenceLevel,
        partition: EvaluationPartition,
        metrics: &[MetricObservation],
    ) -> Result<ChemicalDecisionReceipt, DecisionError> {
        if evidence != self.required_evidence {
            return Err(DecisionError::EvidenceMismatch {
                expected: self.required_evidence,
                actual: evidence,
            });
        }
        if partition != self.required_partition {
            return Err(DecisionError::PartitionMismatch {
                expected: self.required_partition,
                actual: partition,
            });
        }

        let gate_ids: BTreeSet<&str> = self.gates.iter().map(|gate| gate.id.as_str()).collect();
        let mut observed = BTreeMap::new();
        for metric in metrics {
            if metric.id.trim().is_empty() {
                return Err(DecisionError::BlankMetricId);
            }
            if !metric.value.is_finite() {
                return Err(DecisionError::NonFiniteMetric(metric.id.clone()));
            }
            if !gate_ids.contains(metric.id.as_str()) {
                return Err(DecisionError::UnexpectedMetric(metric.id.clone()));
            }
            if observed.insert(metric.id.as_str(), metric.value).is_some() {
                return Err(DecisionError::DuplicateMetric(metric.id.clone()));
            }
        }

        let mut results = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let value = observed
                .get(gate.id.as_str())
                .copied()
                .ok_or_else(|| DecisionError::MissingMetric(gate.id.clone()))?;
            results.push(MetricGateResult {
                id: gate.id.clone(),
                value,
                outcome: gate.assess(value),
            });
        }

        let decision = if results
            .iter()
            .any(|result| result.outcome == GateOutcome::PracticalFailure)
        {
            ExperimentDecision::NotConfirmed
        } else if results
            .iter()
            .all(|result| result.outcome == GateOutcome::ConfirmationPass)
        {
            ExperimentDecision::Confirmed
        } else {
            ExperimentDecision::Inconclusive
        };

        Ok(ChemicalDecisionReceipt {
            protocol_id: self.protocol_id.clone(),
            version: self.version.clone(),
            evidence,
            partition,
            metrics: results,
            decision,
        })
    }
}

/// Observed scalar supplied to a frozen metric gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricObservation {
    pub id: String,
    pub value: f64,
}

impl MetricObservation {
    pub fn new(id: impl Into<String>, value: f64) -> Self {
        Self {
            id: id.into(),
            value,
        }
    }
}

/// Evaluated form of one metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricGateResult {
    pub id: String,
    pub value: f64,
    pub outcome: GateOutcome,
}

/// Machine-readable decision receipt for one admissible evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalDecisionReceipt {
    pub protocol_id: String,
    pub version: String,
    pub evidence: ChemicalEvidenceLevel,
    pub partition: EvaluationPartition,
    pub metrics: Vec<MetricGateResult>,
    pub decision: ExperimentDecision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecisionError {
    EvidenceMismatch {
        expected: ChemicalEvidenceLevel,
        actual: ChemicalEvidenceLevel,
    },
    PartitionMismatch {
        expected: EvaluationPartition,
        actual: EvaluationPartition,
    },
    BlankMetricId,
    NonFiniteMetric(String),
    DuplicateMetric(String),
    UnexpectedMetric(String),
    MissingMetric(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol() -> ChemicalDecisionProtocol {
        ChemicalDecisionProtocol::new(
            "od001-v1",
            "1.0.0",
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
            EvaluationPartition::Holdout,
            vec![
                MetricGate::new("identity_accuracy", GateDirection::AtLeast, 0.85, Some(0.60))
                    .unwrap(),
                MetricGate::new("concentration_leakage", GateDirection::AtMost, 0.15, Some(0.35))
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    fn evaluate(values: [f64; 2]) -> Result<ChemicalDecisionReceipt, DecisionError> {
        protocol().evaluate(
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
            EvaluationPartition::Holdout,
            &[
                MetricObservation::new("identity_accuracy", values[0]),
                MetricObservation::new("concentration_leakage", values[1]),
            ],
        )
    }

    #[test]
    fn all_confirmation_gates_must_pass() {
        let receipt = evaluate([0.90, 0.10]).unwrap();
        assert_eq!(receipt.decision, ExperimentDecision::Confirmed);
        assert!(receipt
            .metrics
            .iter()
            .all(|metric| metric.outcome == GateOutcome::ConfirmationPass));
    }

    #[test]
    fn missing_confirmation_is_inconclusive_not_negative() {
        let receipt = evaluate([0.75, 0.20]).unwrap();
        assert_eq!(receipt.decision, ExperimentDecision::Inconclusive);
        assert!(receipt
            .metrics
            .iter()
            .any(|metric| metric.outcome == GateOutcome::Indeterminate));
    }

    #[test]
    fn practical_failure_is_a_positive_negative_result() {
        let receipt = evaluate([0.55, 0.10]).unwrap();
        assert_eq!(receipt.decision, ExperimentDecision::NotConfirmed);
        assert!(receipt
            .metrics
            .iter()
            .any(|metric| metric.outcome == GateOutcome::PracticalFailure));
    }

    #[test]
    fn development_data_cannot_satisfy_holdout_protocol() {
        let error = protocol()
            .evaluate(
                ChemicalEvidenceLevel::HeldOutPhysicalObservation,
                EvaluationPartition::Development,
                &[],
            )
            .unwrap_err();
        assert!(matches!(error, DecisionError::PartitionMismatch { .. }));
    }

    #[test]
    fn simulator_cannot_satisfy_physical_protocol() {
        let error = protocol()
            .evaluate(
                ChemicalEvidenceLevel::SimulatedFixture,
                EvaluationPartition::Holdout,
                &[],
            )
            .unwrap_err();
        assert!(matches!(error, DecisionError::EvidenceMismatch { .. }));
    }

    #[test]
    fn invalid_threshold_ordering_is_rejected() {
        assert!(matches!(
            MetricGate::new("bad", GateDirection::AtLeast, 0.8, Some(0.9)),
            Err(DecisionProtocolError::InvalidThresholdOrdering(id)) if id == "bad"
        ));
        assert!(matches!(
            MetricGate::new("bad", GateDirection::AtMost, 0.2, Some(0.1)),
            Err(DecisionProtocolError::InvalidThresholdOrdering(id)) if id == "bad"
        ));
    }

    #[test]
    fn non_finite_metric_is_never_decision_evidence() {
        let error = protocol()
            .evaluate(
                ChemicalEvidenceLevel::HeldOutPhysicalObservation,
                EvaluationPartition::Holdout,
                &[
                    MetricObservation::new("identity_accuracy", f64::NAN),
                    MetricObservation::new("concentration_leakage", 0.1),
                ],
            )
            .unwrap_err();
        assert!(matches!(error, DecisionError::NonFiniteMetric(id) if id == "identity_accuracy"));
    }
}

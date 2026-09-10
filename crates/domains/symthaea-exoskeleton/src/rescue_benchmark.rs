// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-environment rescue benchmark for SX-015.
//!
//! The benchmark compares mobility strategies under the same declared case.
//! It does not encode one universally best rescue method and does not convert
//! simulation evidence into qualification evidence.

use serde::{Deserialize, Serialize};

use crate::rescue_propulsion::RescueEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RescueEnvironmentKind {
    OrbitalEva,
    LunarSurface,
    MarsSurface,
    SmallBodyMicrogravity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RescueStrategy {
    ExoskeletonMobility,
    TetherOrWinch,
    ColdGasSelfRescue,
    SurfaceRover,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueBenchmarkEnvironment {
    pub kind: RescueEnvironmentKind,
    pub gravity_m_s2: f64,
    pub reliable_ground_contact: bool,
    pub safe_haven_distance_m: f64,
}

impl RescueBenchmarkEnvironment {
    pub fn is_valid(&self) -> bool {
        self.gravity_m_s2.is_finite()
            && self.gravity_m_s2 >= 0.0
            && self.safe_haven_distance_m.is_finite()
            && self.safe_haven_distance_m >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueMeasurement {
    pub protocol_id: String,
    pub environment: RescueBenchmarkEnvironment,
    pub strategy: RescueStrategy,
    pub completion_time_s: f64,
    pub electrical_energy_wh: f64,
    pub propellant_kg: f64,
    pub human_work_kj: f64,
    pub peak_risk_index: f64,
    pub completed: bool,
    pub evidence: RescueEvidenceLevel,
}

impl RescueMeasurement {
    pub fn is_valid(&self) -> bool {
        !self.protocol_id.trim().is_empty()
            && self.environment.is_valid()
            && self.completion_time_s.is_finite()
            && self.completion_time_s >= 0.0
            && self.electrical_energy_wh.is_finite()
            && self.electrical_energy_wh >= 0.0
            && self.propellant_kg.is_finite()
            && self.propellant_kg >= 0.0
            && self.human_work_kj.is_finite()
            && self.human_work_kj >= 0.0
            && self.peak_risk_index.is_finite()
            && (0.0..=1.0).contains(&self.peak_risk_index)
    }

    /// Coarse applicability guard used before comparing measured outcomes.
    /// Surface cold-gas remains a contingency option, not a routine transport
    /// recommendation. This function does not prohibit testing it.
    pub fn nominal_role(&self) -> RescueRole {
        match (self.environment.kind, self.strategy) {
            (RescueEnvironmentKind::OrbitalEva, RescueStrategy::ColdGasSelfRescue) => {
                RescueRole::PrimarySelfRescue
            }
            (RescueEnvironmentKind::SmallBodyMicrogravity, RescueStrategy::ColdGasSelfRescue) => {
                RescueRole::BoundedTranslation
            }
            (RescueEnvironmentKind::LunarSurface, RescueStrategy::ColdGasSelfRescue)
            | (RescueEnvironmentKind::MarsSurface, RescueStrategy::ColdGasSelfRescue) => {
                RescueRole::ContingencyImpulseOnly
            }
            (_, RescueStrategy::TetherOrWinch) => RescueRole::LocalPhysicalRescue,
            (_, RescueStrategy::SurfaceRover) => RescueRole::SurfaceTransport,
            (_, RescueStrategy::ExoskeletonMobility) => RescueRole::HumanMobilityAssist,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueRole {
    HumanMobilityAssist,
    LocalPhysicalRescue,
    PrimarySelfRescue,
    BoundedTranslation,
    ContingencyImpulseOnly,
    SurfaceTransport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueComparisonOutcome {
    CandidatePreferred,
    BaselinePreferred,
    Tradeoff,
    Incomparable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueComparison {
    pub outcome: RescueComparisonOutcome,
    pub candidate_strategy: RescueStrategy,
    pub baseline_strategy: RescueStrategy,
    pub notes: Vec<String>,
}

/// Compare two measurements only when they are from the same declared
/// protocol/environment and meet the requested evidence floor.
///
/// A candidate is preferred only when it completes the rescue, has no worse
/// peak risk, and is no worse on every tracked resource/time metric with at
/// least one strict improvement. Otherwise the result remains an explicit
/// tradeoff rather than hiding weights in a scalar score.
pub fn compare_rescue_measurements(
    candidate: &RescueMeasurement,
    baseline: &RescueMeasurement,
    minimum_evidence: RescueEvidenceLevel,
) -> RescueComparison {
    let incomparable = || RescueComparison {
        outcome: RescueComparisonOutcome::Incomparable,
        candidate_strategy: candidate.strategy,
        baseline_strategy: baseline.strategy,
        notes: vec!["measurements are not directly comparable".to_string()],
    };

    if !candidate.is_valid()
        || !baseline.is_valid()
        || candidate.protocol_id != baseline.protocol_id
        || candidate.environment.kind != baseline.environment.kind
        || (candidate.environment.gravity_m_s2 - baseline.environment.gravity_m_s2).abs() > 1e-9
        || candidate.environment.reliable_ground_contact
            != baseline.environment.reliable_ground_contact
        || candidate.evidence < minimum_evidence
        || baseline.evidence < minimum_evidence
    {
        return incomparable();
    }

    if candidate.completed && !baseline.completed {
        return RescueComparison {
            outcome: RescueComparisonOutcome::CandidatePreferred,
            candidate_strategy: candidate.strategy,
            baseline_strategy: baseline.strategy,
            notes: vec!["candidate completed while baseline did not".to_string()],
        };
    }
    if !candidate.completed && baseline.completed {
        return RescueComparison {
            outcome: RescueComparisonOutcome::BaselinePreferred,
            candidate_strategy: candidate.strategy,
            baseline_strategy: baseline.strategy,
            notes: vec!["baseline completed while candidate did not".to_string()],
        };
    }
    if !candidate.completed && !baseline.completed {
        return RescueComparison {
            outcome: RescueComparisonOutcome::Tradeoff,
            candidate_strategy: candidate.strategy,
            baseline_strategy: baseline.strategy,
            notes: vec!["neither strategy completed the rescue".to_string()],
        };
    }

    let candidate_no_worse = candidate.completion_time_s <= baseline.completion_time_s
        && candidate.electrical_energy_wh <= baseline.electrical_energy_wh
        && candidate.propellant_kg <= baseline.propellant_kg
        && candidate.human_work_kj <= baseline.human_work_kj
        && candidate.peak_risk_index <= baseline.peak_risk_index;
    let candidate_strictly_better = candidate.completion_time_s < baseline.completion_time_s
        || candidate.electrical_energy_wh < baseline.electrical_energy_wh
        || candidate.propellant_kg < baseline.propellant_kg
        || candidate.human_work_kj < baseline.human_work_kj
        || candidate.peak_risk_index < baseline.peak_risk_index;

    let baseline_no_worse = baseline.completion_time_s <= candidate.completion_time_s
        && baseline.electrical_energy_wh <= candidate.electrical_energy_wh
        && baseline.propellant_kg <= candidate.propellant_kg
        && baseline.human_work_kj <= candidate.human_work_kj
        && baseline.peak_risk_index <= candidate.peak_risk_index;
    let baseline_strictly_better = baseline.completion_time_s < candidate.completion_time_s
        || baseline.electrical_energy_wh < candidate.electrical_energy_wh
        || baseline.propellant_kg < candidate.propellant_kg
        || baseline.human_work_kj < candidate.human_work_kj
        || baseline.peak_risk_index < candidate.peak_risk_index;

    if candidate_no_worse && candidate_strictly_better {
        RescueComparison {
            outcome: RescueComparisonOutcome::CandidatePreferred,
            candidate_strategy: candidate.strategy,
            baseline_strategy: baseline.strategy,
            notes: vec!["candidate Pareto-dominates baseline for this case".to_string()],
        }
    } else if baseline_no_worse && baseline_strictly_better {
        RescueComparison {
            outcome: RescueComparisonOutcome::BaselinePreferred,
            candidate_strategy: candidate.strategy,
            baseline_strategy: baseline.strategy,
            notes: vec!["baseline Pareto-dominates candidate for this case".to_string()],
        }
    } else {
        RescueComparison {
            outcome: RescueComparisonOutcome::Tradeoff,
            candidate_strategy: candidate.strategy,
            baseline_strategy: baseline.strategy,
            notes: vec!["no strategy dominates across all tracked dimensions".to_string()],
        }
    }
}

pub fn reference_environments() -> [RescueBenchmarkEnvironment; 4] {
    [
        RescueBenchmarkEnvironment {
            kind: RescueEnvironmentKind::OrbitalEva,
            gravity_m_s2: 0.0,
            reliable_ground_contact: false,
            safe_haven_distance_m: 20.0,
        },
        RescueBenchmarkEnvironment {
            kind: RescueEnvironmentKind::LunarSurface,
            gravity_m_s2: 1.62,
            reliable_ground_contact: true,
            safe_haven_distance_m: 500.0,
        },
        RescueBenchmarkEnvironment {
            kind: RescueEnvironmentKind::MarsSurface,
            gravity_m_s2: 3.71,
            reliable_ground_contact: true,
            safe_haven_distance_m: 500.0,
        },
        RescueBenchmarkEnvironment {
            kind: RescueEnvironmentKind::SmallBodyMicrogravity,
            gravity_m_s2: 0.001,
            reliable_ground_contact: false,
            safe_haven_distance_m: 50.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(
        environment: RescueBenchmarkEnvironment,
        strategy: RescueStrategy,
    ) -> RescueMeasurement {
        RescueMeasurement {
            protocol_id: "sx015-case-a".into(),
            environment,
            strategy,
            completion_time_s: 100.0,
            electrical_energy_wh: 20.0,
            propellant_kg: 0.1,
            human_work_kj: 10.0,
            peak_risk_index: 0.2,
            completed: true,
            evidence: RescueEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn environment_roles_do_not_treat_surface_thrust_as_routine_flight() {
        let env = reference_environments()[1];
        let m = measurement(env, RescueStrategy::ColdGasSelfRescue);
        assert_eq!(m.nominal_role(), RescueRole::ContingencyImpulseOnly);
    }

    #[test]
    fn orbital_cold_gas_is_a_self_rescue_role() {
        let env = reference_environments()[0];
        let m = measurement(env, RescueStrategy::ColdGasSelfRescue);
        assert_eq!(m.nominal_role(), RescueRole::PrimarySelfRescue);
    }

    #[test]
    fn protocol_mismatch_is_incomparable() {
        let env = reference_environments()[0];
        let a = measurement(env, RescueStrategy::ColdGasSelfRescue);
        let mut b = measurement(env, RescueStrategy::TetherOrWinch);
        b.protocol_id = "different".into();
        let result = compare_rescue_measurements(&a, &b, RescueEvidenceLevel::Simulation);
        assert_eq!(result.outcome, RescueComparisonOutcome::Incomparable);
    }

    #[test]
    fn qualification_claim_cannot_use_simulation_measurements() {
        let env = reference_environments()[0];
        let a = measurement(env, RescueStrategy::ColdGasSelfRescue);
        let b = measurement(env, RescueStrategy::TetherOrWinch);
        let result = compare_rescue_measurements(&a, &b, RescueEvidenceLevel::Qualification);
        assert_eq!(result.outcome, RescueComparisonOutcome::Incomparable);
    }

    #[test]
    fn pareto_dominance_is_explicit_without_hidden_weights() {
        let env = reference_environments()[0];
        let mut candidate = measurement(env, RescueStrategy::ColdGasSelfRescue);
        let baseline = measurement(env, RescueStrategy::TetherOrWinch);
        candidate.completion_time_s = 80.0;
        let result = compare_rescue_measurements(
            &candidate,
            &baseline,
            RescueEvidenceLevel::Simulation,
        );
        assert_eq!(result.outcome, RescueComparisonOutcome::CandidatePreferred);
    }
}

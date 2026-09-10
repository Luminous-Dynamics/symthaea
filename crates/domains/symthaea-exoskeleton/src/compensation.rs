// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-compensation contract for human/exoskeleton co-embodiment.
//!
//! The exoskeleton may compensate for an externally established reduction in
//! natural capability, but this layer never diagnoses the cause and never rewrites
//! the underlying condition as healed. It reports both supplied compensation and
//! any remaining task deficit.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityDemand {
    /// Estimated unaided capability available from the wearer, [0,1].
    pub natural_capacity: f64,
    /// Capability fraction required by the current task, [0,1].
    pub task_demand: f64,
    /// Confidence in the capacity estimate, [0,1].
    pub observation_confidence: f64,
}

impl CapabilityDemand {
    pub fn is_valid(&self) -> bool {
        self.natural_capacity.is_finite()
            && (0.0..=1.0).contains(&self.natural_capacity)
            && self.task_demand.is_finite()
            && (0.0..=1.0).contains(&self.task_demand)
            && self.observation_confidence.is_finite()
            && (0.0..=1.0).contains(&self.observation_confidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompensationPolicy {
    /// Maximum capability fraction the frame may add for this control context.
    pub max_assistance: f64,
    /// Below this confidence the controller may not provide predictive compensation.
    pub min_confidence: f64,
    /// Conservative assistance ceiling used when confidence is below the threshold.
    pub low_confidence_assistance: f64,
}

impl Default for CompensationPolicy {
    fn default() -> Self {
        Self {
            max_assistance: 0.60,
            min_confidence: 0.70,
            low_confidence_assistance: 0.15,
        }
    }
}

impl CompensationPolicy {
    pub fn is_valid(&self) -> bool {
        self.max_assistance.is_finite()
            && (0.0..=1.0).contains(&self.max_assistance)
            && self.min_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_confidence)
            && self.low_confidence_assistance.is_finite()
            && (0.0..=self.max_assistance).contains(&self.low_confidence_assistance)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompensationPlan {
    /// Task deficit before powered assistance.
    pub unaided_deficit: f64,
    /// Capability fraction the frame is permitted to attempt to provide.
    pub requested_assistance: f64,
    /// Capability still missing even if the requested assistance is delivered.
    pub unresolved_deficit: f64,
    /// The unchanged underlying unaided capability estimate.
    pub natural_capacity: f64,
    /// Confidence carried through from the observation used to form the plan.
    pub observation_confidence: f64,
}

/// Derive a bounded assistance request without changing the underlying capability.
pub fn plan_compensation(
    demand: CapabilityDemand,
    policy: CompensationPolicy,
) -> CompensationPlan {
    assert!(demand.is_valid(), "CapabilityDemand must be bounded and finite");
    assert!(policy.is_valid(), "CompensationPolicy must be bounded and finite");

    let deficit = (demand.task_demand - demand.natural_capacity).max(0.0);
    let assistance_ceiling = if demand.observation_confidence >= policy.min_confidence {
        policy.max_assistance
    } else {
        policy.low_confidence_assistance
    };
    let requested_assistance = deficit.min(assistance_ceiling);
    let unresolved_deficit = (deficit - requested_assistance).max(0.0);

    CompensationPlan {
        unaided_deficit: deficit,
        requested_assistance,
        unresolved_deficit,
        natural_capacity: demand.natural_capacity,
        observation_confidence: demand.observation_confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_deficit_requires_no_assistance() {
        let plan = plan_compensation(
            CapabilityDemand {
                natural_capacity: 0.8,
                task_demand: 0.6,
                observation_confidence: 1.0,
            },
            CompensationPolicy::default(),
        );
        assert_eq!(plan.unaided_deficit, 0.0);
        assert_eq!(plan.requested_assistance, 0.0);
        assert_eq!(plan.unresolved_deficit, 0.0);
    }

    #[test]
    fn compensation_does_not_rewrite_natural_capacity() {
        let demand = CapabilityDemand {
            natural_capacity: 0.25,
            task_demand: 0.75,
            observation_confidence: 0.95,
        };
        let plan = plan_compensation(demand, CompensationPolicy::default());
        assert_eq!(plan.natural_capacity, demand.natural_capacity);
        assert_eq!(plan.unaided_deficit, 0.5);
        assert_eq!(plan.requested_assistance, 0.5);
    }

    #[test]
    fn insufficient_frame_capacity_leaves_explicit_deficit() {
        let plan = plan_compensation(
            CapabilityDemand {
                natural_capacity: 0.1,
                task_demand: 1.0,
                observation_confidence: 1.0,
            },
            CompensationPolicy {
                max_assistance: 0.4,
                ..Default::default()
            },
        );
        assert!((plan.unresolved_deficit - 0.5).abs() < 1e-12);
    }

    #[test]
    fn low_confidence_limits_assistance() {
        let policy = CompensationPolicy::default();
        let plan = plan_compensation(
            CapabilityDemand {
                natural_capacity: 0.1,
                task_demand: 0.9,
                observation_confidence: 0.2,
            },
            policy,
        );
        assert_eq!(plan.requested_assistance, policy.low_confidence_assistance);
        assert!(plan.unresolved_deficit > 0.0);
    }

    #[test]
    fn deterministic_inputs_produce_identical_plan() {
        let demand = CapabilityDemand {
            natural_capacity: 0.42,
            task_demand: 0.91,
            observation_confidence: 0.88,
        };
        assert_eq!(
            plan_compensation(demand, CompensationPolicy::default()),
            plan_compensation(demand, CompensationPolicy::default())
        );
    }
}

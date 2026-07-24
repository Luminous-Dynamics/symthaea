// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit resource budgets for simultaneously active operational objectives.
//!
//! The budget is an admission contract, not an optimizer. Protected objectives
//! must remain fundable before discretionary work may consume remaining
//! battery, thermal, time, or recovery capacity.

use serde::{Deserialize, Serialize};

pub const OBJECTIVE_BUDGET_SCHEMA_VERSION: u16 = 1;
pub const NUM_CONFLICT_OBJECTIVES: usize = 8;
pub const MAX_OBJECTIVE_DEMANDS: usize = NUM_CONFLICT_OBJECTIVES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConflictObjective {
    PhysicalSafety,
    ReturnReserve,
    EnvironmentalContainment,
    AssetIntegrity,
    Restoration,
    PeerAssistance,
    Communications,
    MissionWork,
}

impl ConflictObjective {
    pub const ALL: [Self; NUM_CONFLICT_OBJECTIVES] = [
        Self::PhysicalSafety,
        Self::ReturnReserve,
        Self::EnvironmentalContainment,
        Self::AssetIntegrity,
        Self::Restoration,
        Self::PeerAssistance,
        Self::Communications,
        Self::MissionWork,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::PhysicalSafety => 0,
            Self::ReturnReserve => 1,
            Self::EnvironmentalContainment => 2,
            Self::AssetIntegrity => 3,
            Self::Restoration => 4,
            Self::PeerAssistance => 5,
            Self::Communications => 6,
            Self::MissionWork => 7,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::PhysicalSafety => "physical_safety",
            Self::ReturnReserve => "return_reserve",
            Self::EnvironmentalContainment => "environmental_containment",
            Self::AssetIntegrity => "asset_integrity",
            Self::Restoration => "restoration",
            Self::PeerAssistance => "peer_assistance",
            Self::Communications => "communications",
            Self::MissionWork => "mission_work",
        }
    }

    pub const fn class(self) -> ObjectiveClass {
        match self {
            Self::PhysicalSafety | Self::ReturnReserve | Self::EnvironmentalContainment => {
                ObjectiveClass::Protected
            }
            Self::AssetIntegrity | Self::Restoration | Self::PeerAssistance => {
                ObjectiveClass::Obligatory
            }
            Self::Communications | Self::MissionWork => ObjectiveClass::Discretionary,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObjectiveClass {
    Discretionary,
    Obligatory,
    Protected,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceVector {
    pub battery: f32,
    pub thermal: f32,
    pub time: f32,
    pub recovery: f32,
}

impl ResourceVector {
    pub const fn zero() -> Self {
        Self {
            battery: 0.0,
            thermal: 0.0,
            time: 0.0,
            recovery: 0.0,
        }
    }

    pub const fn unit() -> Self {
        Self {
            battery: 1.0,
            thermal: 1.0,
            time: 1.0,
            recovery: 1.0,
        }
    }

    pub fn validate(self) -> bool {
        [self.battery, self.thermal, self.time, self.recovery]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }

    pub fn fits_within(self, capacity: Self) -> bool {
        self.battery <= capacity.battery + f32::EPSILON
            && self.thermal <= capacity.thermal + f32::EPSILON
            && self.time <= capacity.time + f32::EPSILON
            && self.recovery <= capacity.recovery + f32::EPSILON
    }

    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            battery: (self.battery + other.battery).min(1.0),
            thermal: (self.thermal + other.thermal).min(1.0),
            time: (self.time + other.time).min(1.0),
            recovery: (self.recovery + other.recovery).min(1.0),
        }
    }

    pub fn headroom_after(self, demand: Self) -> Self {
        Self {
            battery: (self.battery - demand.battery).max(0.0),
            thermal: (self.thermal - demand.thermal).max(0.0),
            time: (self.time - demand.time).max(0.0),
            recovery: (self.recovery - demand.recovery).max(0.0),
        }
    }

    pub fn maximum_fraction_of(self, capacity: Self) -> f32 {
        let fraction = |demand: f32, available: f32| {
            if demand <= f32::EPSILON {
                0.0
            } else if available <= f32::EPSILON {
                1.0
            } else {
                (demand / available).clamp(0.0, 1.0)
            }
        };
        fraction(self.battery, capacity.battery)
            .max(fraction(self.thermal, capacity.thermal))
            .max(fraction(self.time, capacity.time))
            .max(fraction(self.recovery, capacity.recovery))
    }
}

impl Default for ResourceVector {
    fn default() -> Self {
        Self::zero()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveDemand {
    pub objective: ConflictObjective,
    pub active: bool,
    pub urgency: f32,
    pub demand: ResourceVector,
    pub deadline_step: Option<u64>,
    pub stakeholder: Option<u64>,
}

impl ObjectiveDemand {
    pub fn validate(self, current_step: u64) -> bool {
        self.urgency.is_finite()
            && (0.0..=1.0).contains(&self.urgency)
            && self.demand.validate()
            && self.deadline_step.is_none_or(|deadline| deadline >= current_step)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveBudget {
    pub schema_version: u16,
    pub capacity: ResourceVector,
    pub protected_reserve: ResourceVector,
    pub demands: Vec<ObjectiveDemand>,
}

impl ObjectiveBudget {
    pub fn new(capacity: ResourceVector, protected_reserve: ResourceVector) -> Self {
        Self {
            schema_version: OBJECTIVE_BUDGET_SCHEMA_VERSION,
            capacity,
            protected_reserve,
            demands: Vec::new(),
        }
    }

    pub fn validate(&self, current_step: u64) -> bool {
        self.schema_version == OBJECTIVE_BUDGET_SCHEMA_VERSION
            && self.capacity.validate()
            && self.protected_reserve.validate()
            && self.protected_reserve.fits_within(self.capacity)
            && self.demands.len() <= MAX_OBJECTIVE_DEMANDS
            && self
                .demands
                .iter()
                .copied()
                .all(|demand| demand.validate(current_step))
            && ConflictObjective::ALL.into_iter().all(|objective| {
                self.demands
                    .iter()
                    .filter(|demand| demand.objective == objective)
                    .count()
                    <= 1
            })
    }

    pub fn push(&mut self, demand: ObjectiveDemand) -> bool {
        if self.demands.len() >= MAX_OBJECTIVE_DEMANDS
            || self
                .demands
                .iter()
                .any(|existing| existing.objective == demand.objective)
        {
            return false;
        }
        self.demands.push(demand);
        true
    }

    pub fn active(&self) -> impl Iterator<Item = ObjectiveDemand> + '_ {
        self.demands.iter().copied().filter(|demand| demand.active)
    }

    pub fn usable_capacity(&self) -> ResourceVector {
        self.capacity.headroom_after(self.protected_reserve)
    }
}

impl Default for ObjectiveBudget {
    fn default() -> Self {
        Self::new(ResourceVector::unit(), ResourceVector::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_objectives_are_rejected() {
        let mut budget = ObjectiveBudget::default();
        let demand = ObjectiveDemand {
            objective: ConflictObjective::MissionWork,
            active: true,
            urgency: 0.5,
            demand: ResourceVector::zero(),
            deadline_step: None,
            stakeholder: None,
        };
        assert!(budget.push(demand));
        assert!(!budget.push(demand));
    }

    #[test]
    fn protected_reserve_is_removed_from_discretionary_capacity() {
        let budget = ObjectiveBudget::new(
            ResourceVector::unit(),
            ResourceVector {
                battery: 0.3,
                thermal: 0.2,
                time: 0.1,
                recovery: 0.4,
            },
        );
        assert_eq!(
            budget.usable_capacity(),
            ResourceVector {
                battery: 0.7,
                thermal: 0.8,
                time: 0.9,
                recovery: 0.6,
            }
        );
    }
}

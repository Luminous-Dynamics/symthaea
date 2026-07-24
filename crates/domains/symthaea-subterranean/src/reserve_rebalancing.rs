// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hysteretic protected-reserve rebalancing.

use crate::objective_budget::ResourceVector;
use serde::{Deserialize, Serialize};

pub const RESERVE_REBALANCING_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReserveRebalancingPolicy {
    pub maximum_step: f32,
    pub low_margin_threshold: f32,
    pub recovery_pressure_threshold: f32,
}

impl Default for ReserveRebalancingPolicy {
    fn default() -> Self {
        Self {
            maximum_step: 0.03,
            low_margin_threshold: 0.2,
            recovery_pressure_threshold: 0.65,
        }
    }
}

impl ReserveRebalancingPolicy {
    pub fn validate(self) -> bool {
        self.maximum_step.is_finite()
            && (0.0..=0.1).contains(&self.maximum_step)
            && self.low_margin_threshold.is_finite()
            && (0.0..=1.0).contains(&self.low_margin_threshold)
            && self.recovery_pressure_threshold.is_finite()
            && (0.0..=1.0).contains(&self.recovery_pressure_threshold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReserveRebalancer {
    schema_version: u16,
    policy: ReserveRebalancingPolicy,
    current: ResourceVector,
}

impl Default for ReserveRebalancer {
    fn default() -> Self {
        Self {
            schema_version: RESERVE_REBALANCING_SCHEMA_VERSION,
            policy: ReserveRebalancingPolicy::default(),
            current: ResourceVector {
                battery: 0.15,
                thermal: 0.10,
                time: 0.08,
                recovery: 0.15,
            },
        }
    }
}

impl ReserveRebalancer {
    pub fn validate(&self) -> bool {
        self.schema_version == RESERVE_REBALANCING_SCHEMA_VERSION
            && self.policy.validate()
            && self.current.validate()
    }

    pub fn rebalance(
        &mut self,
        capacity: ResourceVector,
        return_margin: f32,
        recovery_pressure: f32,
    ) -> ResourceVector {
        if !capacity.validate() || !return_margin.is_finite() || !recovery_pressure.is_finite() {
            self.current = capacity;
            return self.current;
        }
        let mut target = self.current;
        if return_margin < self.policy.low_margin_threshold {
            target.battery = (target.battery + self.policy.maximum_step).min(capacity.battery);
            target.time = (target.time + self.policy.maximum_step).min(capacity.time);
        } else {
            target.battery = (target.battery - self.policy.maximum_step * 0.25).max(0.12);
            target.time = (target.time - self.policy.maximum_step * 0.25).max(0.06);
        }
        if recovery_pressure > self.policy.recovery_pressure_threshold {
            target.recovery = (target.recovery + self.policy.maximum_step).min(capacity.recovery);
            target.thermal = (target.thermal + self.policy.maximum_step * 0.5).min(capacity.thermal);
        } else {
            target.recovery = (target.recovery - self.policy.maximum_step * 0.25).max(0.10);
            target.thermal = (target.thermal - self.policy.maximum_step * 0.25).max(0.06);
        }
        self.current = target;
        self.current
    }

    pub const fn current(&self) -> ResourceVector {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_return_margin_increases_protected_battery_reserve_gradually() {
        let mut rebalancer = ReserveRebalancer::default();
        let before = rebalancer.current();
        let after = rebalancer.rebalance(ResourceVector::unit(), 0.05, 0.0);
        assert!(after.battery > before.battery);
        assert!(after.battery - before.battery <= 0.031);
    }
}

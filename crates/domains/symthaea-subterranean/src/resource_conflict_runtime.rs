// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime derivation of objective demands from already-authoritative state.
//!
//! This module does not reinterpret hazards or approvals. It converts their
//! existing decisions into explicit normalized demands so conflicts become
//! measurable and replayable.

use crate::objective_budget::{
    ConflictObjective, ObjectiveBudget, ObjectiveDemand, ResourceVector,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceConflictRuntimeInputs {
    pub battery_ratio: f64,
    pub thermal_margin: f64,
    pub return_battery_margin: f64,
    pub return_feasible: bool,
    pub hazard_active: bool,
    pub hazard_severity: f32,
    pub environmental_restriction: bool,
    pub restoration_due: bool,
    pub maintenance_due: bool,
    pub mission_abort_required: bool,
    pub peer_assistance_requested: bool,
    pub distressed_peer: Option<u64>,
    pub communications_service_requested: bool,
    pub mission_work_requested: bool,
    pub recovery_capacity: f32,
}

impl ResourceConflictRuntimeInputs {
    pub fn validate(self) -> bool {
        self.battery_ratio.is_finite()
            && (0.0..=1.0).contains(&self.battery_ratio)
            && self.thermal_margin.is_finite()
            && (0.0..=1.0).contains(&self.thermal_margin)
            && self.return_battery_margin.is_finite()
            && (-1.0..=1.0).contains(&self.return_battery_margin)
            && self.hazard_severity.is_finite()
            && (0.0..=1.0).contains(&self.hazard_severity)
            && self.recovery_capacity.is_finite()
            && (0.0..=1.0).contains(&self.recovery_capacity)
    }

    pub fn derive(self, current_step: u64) -> ObjectiveBudget {
        if !self.validate() {
            let mut invalid = ObjectiveBudget::new(ResourceVector::zero(), ResourceVector::zero());
            let _ = invalid.push(ObjectiveDemand {
                objective: ConflictObjective::PhysicalSafety,
                active: true,
                urgency: 1.0,
                demand: ResourceVector::unit(),
                deadline_step: Some(current_step),
                stakeholder: None,
            });
            return invalid;
        }

        let capacity = ResourceVector {
            battery: self.battery_ratio as f32,
            thermal: self.thermal_margin as f32,
            time: 1.0,
            recovery: self.recovery_capacity,
        };
        let protected_reserve = ResourceVector {
            battery: (0.12 + (1.0 - self.return_battery_margin.max(0.0) as f32) * 0.18)
                .clamp(0.12, 0.30)
                .min(capacity.battery),
            thermal: 0.10f32.min(capacity.thermal),
            time: 0.08,
            recovery: 0.15f32.min(capacity.recovery),
        };
        let mut budget = ObjectiveBudget::new(capacity, protected_reserve);

        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::PhysicalSafety,
            active: self.hazard_active,
            urgency: self.hazard_severity,
            demand: ResourceVector {
                battery: 0.05,
                thermal: 0.05,
                time: 0.05,
                recovery: (0.15 + self.hazard_severity * 0.35).clamp(0.0, 0.5),
            },
            deadline_step: Some(current_step),
            stakeholder: None,
        });
        let return_urgency = if !self.return_feasible {
            1.0
        } else {
            (0.35 - self.return_battery_margin as f32).clamp(0.0, 1.0)
        };
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::ReturnReserve,
            active: !self.return_feasible || self.return_battery_margin < 0.20,
            urgency: return_urgency,
            demand: ResourceVector {
                battery: (0.12 + return_urgency * 0.28).clamp(0.0, 0.5),
                thermal: 0.08,
                time: 0.30,
                recovery: 0.05,
            },
            deadline_step: Some(current_step.saturating_add(1)),
            stakeholder: None,
        });
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::EnvironmentalContainment,
            active: self.environmental_restriction,
            urgency: if self.environmental_restriction { 0.9 } else { 0.0 },
            demand: ResourceVector {
                battery: 0.12,
                thermal: 0.05,
                time: 0.20,
                recovery: 0.30,
            },
            deadline_step: Some(current_step.saturating_add(1)),
            stakeholder: None,
        });
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::AssetIntegrity,
            active: self.maintenance_due || self.mission_abort_required,
            urgency: if self.mission_abort_required { 1.0 } else { 0.65 },
            demand: ResourceVector {
                battery: 0.10,
                thermal: 0.10,
                time: 0.25,
                recovery: 0.10,
            },
            deadline_step: Some(current_step.saturating_add(200)),
            stakeholder: None,
        });
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::Restoration,
            active: self.restoration_due,
            urgency: if self.restoration_due { 0.65 } else { 0.0 },
            demand: ResourceVector {
                battery: 0.22,
                thermal: 0.12,
                time: 0.30,
                recovery: 0.25,
            },
            deadline_step: Some(current_step.saturating_add(500)),
            stakeholder: Some(0x4553_5401),
        });
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::PeerAssistance,
            active: self.peer_assistance_requested,
            urgency: if self.peer_assistance_requested { 0.70 } else { 0.0 },
            demand: ResourceVector {
                battery: 0.24,
                thermal: 0.12,
                time: 0.35,
                recovery: 0.15,
            },
            deadline_step: Some(current_step.saturating_add(300)),
            stakeholder: self.distressed_peer,
        });
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::Communications,
            active: self.communications_service_requested,
            urgency: if self.communications_service_requested { 0.55 } else { 0.0 },
            demand: ResourceVector {
                battery: 0.10,
                thermal: 0.04,
                time: 0.12,
                recovery: 0.12,
            },
            deadline_step: Some(current_step.saturating_add(400)),
            stakeholder: Some(0x434f_4d01),
        });
        let _ = budget.push(ObjectiveDemand {
            objective: ConflictObjective::MissionWork,
            active: self.mission_work_requested,
            urgency: if self.mission_work_requested { 0.45 } else { 0.0 },
            demand: ResourceVector {
                battery: 0.35,
                thermal: 0.35,
                time: 0.40,
                recovery: 0.10,
            },
            deadline_step: None,
            stakeholder: Some(0x4d49_5301),
        });
        budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_runtime_inputs_fail_closed_as_physical_safety() {
        let budget = ResourceConflictRuntimeInputs {
            battery_ratio: f64::NAN,
            thermal_margin: 1.0,
            return_battery_margin: 1.0,
            return_feasible: true,
            hazard_active: false,
            hazard_severity: 0.0,
            environmental_restriction: false,
            restoration_due: false,
            maintenance_due: false,
            mission_abort_required: false,
            peer_assistance_requested: false,
            distressed_peer: None,
            communications_service_requested: false,
            mission_work_requested: true,
            recovery_capacity: 1.0,
        }
        .derive(10);
        assert!(budget
            .active()
            .any(|demand| demand.objective == ConflictObjective::PhysicalSafety));
    }

    #[test]
    fn low_return_margin_creates_protected_return_demand() {
        let budget = ResourceConflictRuntimeInputs {
            battery_ratio: 0.5,
            thermal_margin: 0.8,
            return_battery_margin: 0.1,
            return_feasible: true,
            hazard_active: false,
            hazard_severity: 0.0,
            environmental_restriction: false,
            restoration_due: false,
            maintenance_due: false,
            mission_abort_required: false,
            peer_assistance_requested: false,
            distressed_peer: None,
            communications_service_requested: false,
            mission_work_requested: true,
            recovery_capacity: 1.0,
        }
        .derive(10);
        assert!(budget
            .active()
            .any(|demand| demand.objective == ConflictObjective::ReturnReserve));
    }
}

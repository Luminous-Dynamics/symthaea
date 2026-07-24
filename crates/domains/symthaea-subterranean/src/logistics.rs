// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission logistics admission and bounded consumable accounting.
//!
//! A route being geometrically reachable does not make a work order feasible.
//! This module refuses work that cannot fund travel, execution, contingency,
//! and a protected return reserve with the resources currently onboard.

use crate::simulator::RecoveryResources;
use crate::tunnel_graph::TunnelRoute;
use crate::types::SubterraneanState;
use crate::work_orders::{WorkOrder, WorkResourceEstimate};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogisticsPolicy {
    pub protected_battery_reserve: f64,
    pub contingency_fraction: f64,
    pub maximum_sample_fill: f64,
    pub maximum_spoil_fill: f64,
}

impl LogisticsPolicy {
    pub fn validate(self) -> Result<Self, LogisticsError> {
        let valid = [
            self.protected_battery_reserve,
            self.contingency_fraction,
            self.maximum_sample_fill,
            self.maximum_spoil_fill,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value));
        if !valid {
            return Err(LogisticsError::InvalidPolicy);
        }
        Ok(self)
    }
}

impl Default for LogisticsPolicy {
    fn default() -> Self {
        Self {
            protected_battery_reserve: 0.12,
            contingency_fraction: 0.2,
            maximum_sample_fill: 0.9,
            maximum_spoil_fill: 0.85,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogisticsLedger {
    pub sample_fill: f64,
    pub spoil_fill: f64,
    pub coolant_health: f64,
    pub completed_work_orders: u64,
    pub refused_work_orders: u64,
}

impl LogisticsLedger {
    pub const fn new() -> Self {
        Self {
            sample_fill: 0.0,
            spoil_fill: 0.0,
            coolant_health: 1.0,
            completed_work_orders: 0,
            refused_work_orders: 0,
        }
    }

    pub fn validate(self) -> bool {
        [self.sample_fill, self.spoil_fill, self.coolant_health]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }

    pub fn sanitize(&mut self) {
        self.sample_fill = finite_ratio(self.sample_fill, 1.0);
        self.spoil_fill = finite_ratio(self.spoil_fill, 1.0);
        self.coolant_health = finite_ratio(self.coolant_health, 0.0);
    }

    pub fn apply_completion(&mut self, estimate: WorkResourceEstimate) {
        self.sample_fill = (self.sample_fill + estimate.sample_capacity).clamp(0.0, 1.0);
        self.spoil_fill = (self.spoil_fill + estimate.spoil_capacity).clamp(0.0, 1.0);
        self.coolant_health =
            (self.coolant_health - estimate.battery_fraction * 0.02).clamp(0.0, 1.0);
        self.completed_work_orders = self.completed_work_orders.saturating_add(1);
    }

    pub fn record_refusal(&mut self) {
        self.refused_work_orders = self.refused_work_orders.saturating_add(1);
    }

    pub fn unload_at_surface(&mut self) {
        self.sample_fill = 0.0;
        self.spoil_fill = 0.0;
    }
}

impl Default for LogisticsLedger {
    fn default() -> Self {
        Self::new()
    }
}

fn finite_ratio(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionResourceEnvelope {
    pub battery_available: f64,
    pub battery_required: f64,
    pub battery_after_return: f64,
    pub sealant_available: f64,
    pub relay_units_available: u8,
    pub roof_support_units_available: u8,
    pub sample_fill_after: f64,
    pub spoil_fill_after: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionRefusal {
    InvalidEstimate,
    NoOutboundRoute,
    NoReturnRoute,
    BatteryReserve,
    Sealant,
    Relay,
    RoofSupport,
    SampleCapacity,
    SpoilCapacity,
    CoolantUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkAdmission {
    pub admitted: bool,
    pub refusal: Option<AdmissionRefusal>,
    pub envelope: MissionResourceEnvelope,
}

impl WorkAdmission {
    fn refused(reason: AdmissionRefusal, envelope: MissionResourceEnvelope) -> Self {
        Self {
            admitted: false,
            refusal: Some(reason),
            envelope,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogisticsError {
    InvalidPolicy,
}

#[derive(Debug, Clone, Copy)]
pub struct LogisticsPlanner {
    policy: LogisticsPolicy,
}

impl LogisticsPlanner {
    pub fn new(policy: LogisticsPolicy) -> Result<Self, LogisticsError> {
        Ok(Self {
            policy: policy.validate()?,
        })
    }

    pub fn policy(self) -> LogisticsPolicy {
        self.policy
    }

    pub fn assess(
        self,
        order: &WorkOrder,
        outbound: Option<&TunnelRoute>,
        return_route: Option<&TunnelRoute>,
        state: &SubterraneanState,
        recovery: RecoveryResources,
        ledger: LogisticsLedger,
    ) -> WorkAdmission {
        let zero = MissionResourceEnvelope {
            battery_available: state.battery_ratio(),
            battery_required: f64::INFINITY,
            battery_after_return: f64::NEG_INFINITY,
            sealant_available: recovery.sealant_ratio,
            relay_units_available: recovery.relay_units,
            roof_support_units_available: recovery.roof_support_units,
            sample_fill_after: ledger.sample_fill,
            spoil_fill_after: ledger.spoil_fill,
        };
        if !order.resources.is_valid() {
            return WorkAdmission::refused(AdmissionRefusal::InvalidEstimate, zero);
        }
        let Some(outbound) = outbound.filter(|route| route.feasible()) else {
            return WorkAdmission::refused(AdmissionRefusal::NoOutboundRoute, zero);
        };
        let Some(return_route) = return_route.filter(|route| route.feasible()) else {
            return WorkAdmission::refused(AdmissionRefusal::NoReturnRoute, zero);
        };

        let travel_energy = outbound.estimated_energy + return_route.estimated_energy;
        let nominal_required = travel_energy + order.resources.battery_fraction;
        let battery_required = nominal_required * (1.0 + self.policy.contingency_fraction)
            + self.policy.protected_battery_reserve;
        let envelope = MissionResourceEnvelope {
            battery_available: state.battery_ratio(),
            battery_required,
            battery_after_return: state.battery_ratio() - battery_required,
            sealant_available: recovery.sealant_ratio,
            relay_units_available: recovery.relay_units,
            roof_support_units_available: recovery.roof_support_units,
            sample_fill_after: ledger.sample_fill + order.resources.sample_capacity,
            spoil_fill_after: ledger.spoil_fill + order.resources.spoil_capacity,
        };

        let refusal = if ledger.coolant_health <= 0.05 {
            Some(AdmissionRefusal::CoolantUnavailable)
        } else if state.battery_ratio() < battery_required {
            Some(AdmissionRefusal::BatteryReserve)
        } else if recovery.sealant_ratio < order.resources.sealant_fraction {
            Some(AdmissionRefusal::Sealant)
        } else if recovery.relay_units < order.resources.relay_units {
            Some(AdmissionRefusal::Relay)
        } else if recovery.roof_support_units < order.resources.roof_support_units {
            Some(AdmissionRefusal::RoofSupport)
        } else if envelope.sample_fill_after > self.policy.maximum_sample_fill {
            Some(AdmissionRefusal::SampleCapacity)
        } else if envelope.spoil_fill_after > self.policy.maximum_spoil_fill {
            Some(AdmissionRefusal::SpoilCapacity)
        } else {
            None
        };
        match refusal {
            Some(reason) => WorkAdmission::refused(reason, envelope),
            None => WorkAdmission {
                admitted: true,
                refusal: None,
                envelope,
            },
        }
    }
}

impl Default for LogisticsPlanner {
    fn default() -> Self {
        Self {
            policy: LogisticsPolicy::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel_graph::TunnelNodeId;
    use crate::work_orders::{WorkKind, WorkOrderId, WorkPriority, WorkStatus};

    fn order(energy: f64) -> WorkOrder {
        WorkOrder {
            id: WorkOrderId(1),
            kind: WorkKind::Bore,
            target: TunnelNodeId(2),
            priority: WorkPriority::Routine,
            prerequisites: [None; 4],
            estimated_steps: 10,
            deadline_step: None,
            resources: WorkResourceEstimate {
                battery_fraction: energy,
                sealant_fraction: 0.0,
                relay_units: 0,
                roof_support_units: 0,
                sample_capacity: 0.0,
                spoil_capacity: 0.1,
            },
            status: WorkStatus::Pending,
            completed_steps: 0,
        }
    }

    fn route(energy: f64) -> TunnelRoute {
        TunnelRoute {
            nodes: vec![TunnelNodeId(0), TunnelNodeId(2)],
            distance_m: 10.0,
            estimated_energy: energy,
            maximum_risk: 0.1,
            minimum_confidence: 0.9,
            total_cost: 10.0,
        }
    }

    #[test]
    fn work_is_refused_when_return_reserve_cannot_be_preserved() {
        let mut state = SubterraneanState::home();
        state.channels[crate::types::BATTERY_RATIO] = 0.3;
        let admission = LogisticsPlanner::default().assess(
            &order(0.1),
            Some(&route(0.05)),
            Some(&route(0.05)),
            &state,
            RecoveryResources::full(),
            LogisticsLedger::new(),
        );
        assert!(!admission.admitted);
        assert_eq!(admission.refusal, Some(AdmissionRefusal::BatteryReserve));
    }

    #[test]
    fn work_is_refused_before_payload_overflow() {
        let mut ledger = LogisticsLedger::new();
        ledger.sample_fill = 0.85;
        let mut sample_order = order(0.01);
        sample_order.resources.sample_capacity = 0.1;
        let admission = LogisticsPlanner::default().assess(
            &sample_order,
            Some(&route(0.01)),
            Some(&route(0.01)),
            &SubterraneanState::home(),
            RecoveryResources::full(),
            ledger,
        );
        assert_eq!(admission.refusal, Some(AdmissionRefusal::SampleCapacity));
    }

    #[test]
    fn admitted_work_reports_complete_resource_envelope() {
        let admission = LogisticsPlanner::default().assess(
            &order(0.03),
            Some(&route(0.02)),
            Some(&route(0.02)),
            &SubterraneanState::home(),
            RecoveryResources::full(),
            LogisticsLedger::new(),
        );
        assert!(admission.admitted);
        assert!(admission.envelope.battery_after_return > 0.6);
    }
}

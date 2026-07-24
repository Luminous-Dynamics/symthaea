// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent component health, command derating, and finite maintenance.
//!
//! Long missions cannot reset mechanical truth at each work order. Health is
//! accumulated from actual post-arbitration commands and measured plant load.
//! Degraded components restrict only the actuators they can no longer support;
//! critical mobility or cooling failures are surfaced to the mission executive.

use crate::types::{
    SLIP_RATIO, SubterraneanActuator, SubterraneanCommand, SubterraneanState, TOOL_WEAR,
    VIBRATION_LEVEL,
};
use serde::{Deserialize, Serialize};

pub const NUM_COMPONENTS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentKind {
    Cutter,
    Auger,
    LeftTrack,
    RightTrack,
    ThermalPump,
    DewateringPump,
    PressureSeal,
    Communications,
}

impl ComponentKind {
    pub const ALL: [Self; NUM_COMPONENTS] = [
        Self::Cutter,
        Self::Auger,
        Self::LeftTrack,
        Self::RightTrack,
        Self::ThermalPump,
        Self::DewateringPump,
        Self::PressureSeal,
        Self::Communications,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Cutter => 0,
            Self::Auger => 1,
            Self::LeftTrack => 2,
            Self::RightTrack => 3,
            Self::ThermalPump => 4,
            Self::DewateringPump => 5,
            Self::PressureSeal => 6,
            Self::Communications => 7,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cutter => "cutter",
            Self::Auger => "auger",
            Self::LeftTrack => "left_track",
            Self::RightTrack => "right_track",
            Self::ThermalPump => "thermal_pump",
            Self::DewateringPump => "dewatering_pump",
            Self::PressureSeal => "pressure_seal",
            Self::Communications => "communications",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceResources {
    pub spare_parts_ratio: f64,
    pub lubricant_ratio: f64,
}

impl MaintenanceResources {
    pub const fn full() -> Self {
        Self {
            spare_parts_ratio: 1.0,
            lubricant_ratio: 1.0,
        }
    }

    fn sanitize(&mut self) {
        self.spare_parts_ratio = finite_ratio(self.spare_parts_ratio);
        self.lubricant_ratio = finite_ratio(self.lubricant_ratio);
    }
}

fn finite_ratio(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceAssessment {
    pub minimum_health: f64,
    pub critical_component: Option<ComponentKind>,
    pub maintenance_due: bool,
    pub mission_abort_required: bool,
    pub mobility_available: bool,
    pub cooling_available: bool,
}

impl MaintenanceAssessment {
    pub const fn nominal() -> Self {
        Self {
            minimum_health: 1.0,
            critical_component: None,
            maintenance_due: false,
            mission_abort_required: false,
            mobility_available: true,
            cooling_available: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceError {
    InvalidAmount,
    InsufficientSpares,
    InsufficientLubricant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceMonitor {
    health: [f64; NUM_COMPONENTS],
    accumulated_runtime_s: [f64; NUM_COMPONENTS],
    resources: MaintenanceResources,
}

impl MaintenanceMonitor {
    pub fn new() -> Self {
        Self {
            health: [1.0; NUM_COMPONENTS],
            accumulated_runtime_s: [0.0; NUM_COMPONENTS],
            resources: MaintenanceResources::full(),
        }
    }

    pub fn validate(&self) -> bool {
        self.health
            .iter()
            .chain(self.accumulated_runtime_s.iter())
            .all(|value| value.is_finite() && *value >= 0.0)
            && self.health.iter().all(|value| *value <= 1.0)
            && self.resources.spare_parts_ratio.is_finite()
            && (0.0..=1.0).contains(&self.resources.spare_parts_ratio)
            && self.resources.lubricant_ratio.is_finite()
            && (0.0..=1.0).contains(&self.resources.lubricant_ratio)
    }

    pub fn health(&self, component: ComponentKind) -> f64 {
        self.health[component.index()]
    }

    pub fn resources(&self) -> MaintenanceResources {
        self.resources
    }

    pub fn set_health_for_test(&mut self, component: ComponentKind, health: f64) {
        self.health[component.index()] = finite_ratio(health);
    }

    fn wear(&mut self, component: ComponentKind, rate_per_second: f64, dt: f64) {
        let index = component.index();
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        self.health[index] = (self.health[index] - rate_per_second.max(0.0) * dt).clamp(0.0, 1.0);
        self.accumulated_runtime_s[index] += dt;
    }

    pub fn observe(&mut self, command: &SubterraneanCommand, state: &SubterraneanState, dt: f64) {
        let vibration = state.channels[VIBRATION_LEVEL].clamp(0.0, 1.0);
        let slip = state.channels[SLIP_RATIO].clamp(0.0, 1.0);
        let tool_wear = state.channels[TOOL_WEAR].clamp(0.0, 1.0);
        self.wear(
            ComponentKind::Cutter,
            command.cutter_head().abs() as f64
                * (0.00002 + vibration * 0.00008 + tool_wear * 0.00004),
            dt,
        );
        self.wear(
            ComponentKind::Auger,
            command.auger_feed().abs() as f64 * (0.000015 + state.spoil_buffer_fill() * 0.00007),
            dt,
        );
        self.wear(
            ComponentKind::LeftTrack,
            command.left_track().abs() as f64 * (0.00001 + slip * 0.00008),
            dt,
        );
        self.wear(
            ComponentKind::RightTrack,
            command.right_track().abs() as f64 * (0.00001 + slip * 0.00008),
            dt,
        );
        self.wear(
            ComponentKind::ThermalPump,
            command.thermal_pump().max(0.0) as f64
                * (0.00001 + (state.cutter_temp_c() / 180.0) * 0.00005),
            dt,
        );
        self.wear(
            ComponentKind::DewateringPump,
            command.recovery.dewatering_pump as f64 * (0.00002 + state.slurry_load() * 0.00008),
            dt,
        );
        self.wear(
            ComponentKind::PressureSeal,
            state.water_ingress_ratio() * 0.00003 + (1.0 - state.seal_integrity()) * 0.00002,
            dt,
        );
        self.wear(
            ComponentKind::Communications,
            (1.0 - state.relay_link_quality()) * 0.000005,
            dt,
        );
        self.resources.sanitize();
    }

    pub fn assessment(&self) -> MaintenanceAssessment {
        let mut minimum_health = 1.0f64;
        let mut critical_component = None;
        for component in ComponentKind::ALL {
            let health = self.health(component);
            if health < minimum_health {
                minimum_health = health;
                critical_component = Some(component);
            }
        }
        let left = self.health(ComponentKind::LeftTrack);
        let right = self.health(ComponentKind::RightTrack);
        let mobility_available = left > 0.12 && right > 0.12;
        let cooling_available = self.health(ComponentKind::ThermalPump) > 0.12;
        let pressure_available = self.health(ComponentKind::PressureSeal) > 0.1;
        MaintenanceAssessment {
            minimum_health,
            critical_component,
            maintenance_due: minimum_health < 0.55,
            mission_abort_required: minimum_health < 0.22
                || !mobility_available
                || !cooling_available
                || !pressure_available,
            mobility_available,
            cooling_available,
        }
    }

    fn derating(self_health: f64) -> f32 {
        if self_health <= 0.12 {
            0.0
        } else if self_health >= 0.7 {
            1.0
        } else {
            ((self_health - 0.12) / 0.58).clamp(0.0, 1.0) as f32
        }
    }

    pub fn derate_command(&self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        let mappings = [
            (ComponentKind::Cutter, SubterraneanActuator::CutterHead),
            (ComponentKind::Auger, SubterraneanActuator::AugerFeed),
            (ComponentKind::LeftTrack, SubterraneanActuator::LeftTrack),
            (ComponentKind::RightTrack, SubterraneanActuator::RightTrack),
            (
                ComponentKind::ThermalPump,
                SubterraneanActuator::ThermalPump,
            ),
        ];
        for (component, actuator) in mappings {
            let value = command.get(actuator) * Self::derating(self.health(component));
            command.set(actuator, value);
        }
        command.recovery.dewatering_pump *=
            Self::derating(self.health(ComponentKind::DewateringPump));
        if self.health(ComponentKind::PressureSeal) <= 0.12 {
            command.recovery.sealant_injector = 0.0;
        }
        if self.health(ComponentKind::Communications) <= 0.12 {
            command.recovery.relay_deployer = 0.0;
        }
        command.sanitize();
        command
    }

    pub fn service(
        &mut self,
        component: ComponentKind,
        amount: f64,
    ) -> Result<f64, MaintenanceError> {
        if !amount.is_finite() || amount <= 0.0 || amount > 1.0 {
            return Err(MaintenanceError::InvalidAmount);
        }
        let spare_cost = amount * 0.6;
        let lubricant_cost = amount * 0.25;
        if self.resources.spare_parts_ratio < spare_cost {
            return Err(MaintenanceError::InsufficientSpares);
        }
        if self.resources.lubricant_ratio < lubricant_cost {
            return Err(MaintenanceError::InsufficientLubricant);
        }
        self.resources.spare_parts_ratio -= spare_cost;
        self.resources.lubricant_ratio -= lubricant_cost;
        let index = component.index();
        self.health[index] = (self.health[index] + amount).clamp(0.0, 1.0);
        Ok(self.health[index])
    }

    pub fn reset_runtime(&mut self) {
        self.accumulated_runtime_s = [0.0; NUM_COMPONENTS];
    }
}

impl Default for MaintenanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damage_persists_across_observations() {
        let mut monitor = MaintenanceMonitor::new();
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        let mut state = SubterraneanState::home();
        state.channels[VIBRATION_LEVEL] = 1.0;
        for _ in 0..1000 {
            monitor.observe(&command, &state, 1.0);
        }
        assert!(monitor.health(ComponentKind::Cutter) < 0.9);
    }

    #[test]
    fn failed_track_cannot_receive_motion_authority() {
        let mut monitor = MaintenanceMonitor::new();
        monitor.set_health_for_test(ComponentKind::LeftTrack, 0.05);
        let mut command = SubterraneanCommand::zero();
        command.set_left_track(1.0);
        command.set_right_track(1.0);
        let command = monitor.derate_command(command);
        assert_eq!(command.left_track(), 0.0);
        assert!(command.right_track() > 0.9);
    }

    #[test]
    fn service_consumes_finite_resources() {
        let mut monitor = MaintenanceMonitor::new();
        monitor.set_health_for_test(ComponentKind::Cutter, 0.2);
        let restored = monitor
            .service(ComponentKind::Cutter, 0.5)
            .expect("service");
        assert_eq!(restored, 0.7);
        assert!(monitor.resources().spare_parts_ratio < 1.0);
    }
}

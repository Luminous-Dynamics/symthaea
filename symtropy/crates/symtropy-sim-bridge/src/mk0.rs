// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

/// Named stations in the room-scale Mk0 bootstrapper loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mk0Station {
    SeedNode,
    Helios,
    Detritivore,
    Fabricator,
    Manipulator,
    Vector,
}

impl Mk0Station {
    pub fn as_str(self) -> &'static str {
        match self {
            Mk0Station::SeedNode => "seed_node",
            Mk0Station::Helios => "helios",
            Mk0Station::Detritivore => "detritivore",
            Mk0Station::Fabricator => "fabricator",
            Mk0Station::Manipulator => "manipulator",
            Mk0Station::Vector => "vector",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mk0BenchmarkSubassembly {
    RoverBracketSet,
    SensorMast,
    BatteryTray,
    ManipulatorFixture,
    CableGuide,
}

impl Mk0BenchmarkSubassembly {
    pub fn as_str(self) -> &'static str {
        match self {
            Mk0BenchmarkSubassembly::RoverBracketSet => "rover_bracket_set",
            Mk0BenchmarkSubassembly::SensorMast => "sensor_mast",
            Mk0BenchmarkSubassembly::BatteryTray => "battery_tray",
            Mk0BenchmarkSubassembly::ManipulatorFixture => "manipulator_fixture",
            Mk0BenchmarkSubassembly::CableGuide => "cable_guide",
        }
    }

    pub fn default_material_grams(self) -> u32 {
        match self {
            Mk0BenchmarkSubassembly::RoverBracketSet => 180,
            Mk0BenchmarkSubassembly::SensorMast => 140,
            Mk0BenchmarkSubassembly::BatteryTray => 220,
            Mk0BenchmarkSubassembly::ManipulatorFixture => 160,
            Mk0BenchmarkSubassembly::CableGuide => 60,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mk0ScenarioPreset {
    SingleRoomLinear,
    PowerConstrainedRoom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0SubassemblySpec {
    pub id: Mk0BenchmarkSubassembly,
    pub display_name: String,
    pub required_material_grams: u32,
    pub printer_station: Mk0Station,
    pub assembler_station: Mk0Station,
}

impl Mk0SubassemblySpec {
    pub fn from_benchmark(id: Mk0BenchmarkSubassembly) -> Self {
        Self {
            display_name: id.as_str().to_string(),
            required_material_grams: id.default_material_grams(),
            printer_station: Mk0Station::Fabricator,
            assembler_station: Mk0Station::Manipulator,
            id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0LayoutNode {
    pub station: Mk0Station,
    pub zone: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0LayoutEdge {
    pub from: Mk0Station,
    pub to: Mk0Station,
    pub route_label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0RoomLayout {
    pub name: String,
    pub nodes: Vec<Mk0LayoutNode>,
    pub edges: Vec<Mk0LayoutEdge>,
}

impl Mk0RoomLayout {
    pub fn preset(preset: Mk0ScenarioPreset) -> Self {
        match preset {
            Mk0ScenarioPreset::SingleRoomLinear => Self {
                name: "single_room_linear".to_string(),
                nodes: vec![
                    Mk0LayoutNode {
                        station: Mk0Station::SeedNode,
                        zone: "control_rack".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Helios,
                        zone: "power_wall".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Detritivore,
                        zone: "recycling_bay".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Fabricator,
                        zone: "print_farm".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Manipulator,
                        zone: "assembly_bench".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Vector,
                        zone: "logistics_lane".to_string(),
                    },
                ],
                edges: vec![Mk0LayoutEdge {
                    from: Mk0Station::Fabricator,
                    to: Mk0Station::Manipulator,
                    route_label: "printer_to_bench".to_string(),
                }],
            },
            Mk0ScenarioPreset::PowerConstrainedRoom => Self {
                name: "power_constrained_room".to_string(),
                nodes: vec![
                    Mk0LayoutNode {
                        station: Mk0Station::SeedNode,
                        zone: "control_rack".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Helios,
                        zone: "battery_corner".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Detritivore,
                        zone: "recycling_bay".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Fabricator,
                        zone: "print_farm".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Manipulator,
                        zone: "assembly_bench".to_string(),
                    },
                    Mk0LayoutNode {
                        station: Mk0Station::Vector,
                        zone: "tight_lane".to_string(),
                    },
                ],
                edges: vec![Mk0LayoutEdge {
                    from: Mk0Station::Fabricator,
                    to: Mk0Station::Manipulator,
                    route_label: "constrained_printer_to_bench".to_string(),
                }],
            },
        }
    }

    pub fn has_route(&self, from: Mk0Station, to: Mk0Station) -> bool {
        self.edges
            .iter()
            .any(|edge| edge.from == from && edge.to == to)
    }

    pub fn route_label(&self, from: Mk0Station, to: Mk0Station) -> Option<&str> {
        self.edges
            .iter()
            .find(|edge| edge.from == from && edge.to == to)
            .map(|edge| edge.route_label.as_str())
    }
}

/// Minimal Mycelix-shaped record classes emitted by the Mk0 loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mk0RecordKind {
    WorkOrder,
    MaterialBatch,
    PrintClaim,
    DeliveryTask,
    HandoffReceipt,
    EnergyWindow,
    MaintenanceFlag,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0WorkOrder {
    pub id: String,
    pub subassembly: String,
    pub requested_by: Mk0Station,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0MaterialBatch {
    pub id: String,
    pub source: String,
    pub available_grams: u32,
    pub recycled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0PrintClaim {
    pub id: String,
    pub work_order_id: String,
    pub recipe_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0DeliveryTask {
    pub id: String,
    pub work_order_id: String,
    pub from: Mk0Station,
    pub to: Mk0Station,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0HandoffReceipt {
    pub id: String,
    pub work_order_id: String,
    pub accepted_by: Mk0Station,
    pub completed_subassembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0EnergyWindow {
    pub id: String,
    pub fabrication_allowed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mk0MaintenanceFlag {
    pub id: String,
    pub station: Mk0Station,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "payload_type", rename_all = "snake_case")]
pub enum Mk0RecordPayload {
    WorkOrder(Mk0WorkOrder),
    MaterialBatch(Mk0MaterialBatch),
    PrintClaim(Mk0PrintClaim),
    DeliveryTask(Mk0DeliveryTask),
    HandoffReceipt(Mk0HandoffReceipt),
    EnergyWindow(Mk0EnergyWindow),
    MaintenanceFlag(Mk0MaintenanceFlag),
}

/// One event in the bootstrapper loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mk0ScenarioEvent {
    pub tick: u32,
    pub station: Mk0Station,
    pub record_kind: Mk0RecordKind,
    pub record_id: String,
    pub detail: String,
    pub payload: Mk0RecordPayload,
}

/// Deterministic config for a single room-scale bootstrapper run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mk0ScenarioConfig {
    pub preset: Mk0ScenarioPreset,
    pub seed: u64,
    pub work_order_id: String,
    pub subassembly: Mk0SubassemblySpec,
    pub room_layout: Mk0RoomLayout,
    pub initial_energy_reserve: f32,
    pub initial_feedstock_grams: u32,
    pub allow_recycled_feedstock: bool,
}

impl Default for Mk0ScenarioConfig {
    fn default() -> Self {
        Self::for_preset(
            Mk0ScenarioPreset::SingleRoomLinear,
            Mk0BenchmarkSubassembly::RoverBracketSet,
        )
    }
}

impl Mk0ScenarioConfig {
    pub fn for_preset(preset: Mk0ScenarioPreset, subassembly: Mk0BenchmarkSubassembly) -> Self {
        Self {
            preset,
            seed: 42,
            work_order_id: format!("wo-mk0-{}-001", subassembly.as_str()),
            subassembly: Mk0SubassemblySpec::from_benchmark(subassembly),
            room_layout: Mk0RoomLayout::preset(preset),
            initial_energy_reserve: match preset {
                Mk0ScenarioPreset::SingleRoomLinear => 0.82,
                Mk0ScenarioPreset::PowerConstrainedRoom => 0.32,
            },
            initial_feedstock_grams: 750,
            allow_recycled_feedstock: true,
        }
    }
}

/// First-pass report for the Mk0 work-order -> handoff loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mk0ScenarioReport {
    pub config: Mk0ScenarioConfig,
    pub success: bool,
    pub completed_work_orders: u32,
    pub handoff_receipts: u32,
    pub recycled_feedstock_grams: u32,
    pub energy_window_open: bool,
    pub events: Vec<Mk0ScenarioEvent>,
}

/// Deterministic stub for the first Mk0 bootstrapper round-trip.
pub fn run_mk0_bootstrapper_scenario(config: Mk0ScenarioConfig) -> Mk0ScenarioReport {
    let mut events = Vec::new();
    let energy_window_open = config.initial_energy_reserve >= 0.25;
    let recycled_feedstock_grams = if config.allow_recycled_feedstock {
        config.initial_feedstock_grams.min(
            config
                .subassembly
                .required_material_grams
                .saturating_add(70),
        )
    } else {
        0
    };
    let has_delivery_route = config.room_layout.has_route(
        config.subassembly.printer_station,
        config.subassembly.assembler_station,
    );
    let route_label = config
        .room_layout
        .route_label(
            config.subassembly.printer_station,
            config.subassembly.assembler_station,
        )
        .unwrap_or("missing_route")
        .to_string();

    let work_order = Mk0WorkOrder {
        id: config.work_order_id.clone(),
        subassembly: config.subassembly.display_name.clone(),
        requested_by: Mk0Station::SeedNode,
    };
    events.push(Mk0ScenarioEvent {
        tick: 0,
        station: Mk0Station::SeedNode,
        record_kind: Mk0RecordKind::WorkOrder,
        record_id: work_order.id.clone(),
        detail: format!(
            "posted work order for benchmark subassembly '{}'",
            work_order.subassembly
        ),
        payload: Mk0RecordPayload::WorkOrder(work_order),
    });

    if recycled_feedstock_grams == 0 {
        let maintenance = Mk0MaintenanceFlag {
            id: format!("maint-{}", config.work_order_id),
            station: Mk0Station::Detritivore,
            reason: "no recycled feedstock batch available".to_string(),
        };
        events.push(Mk0ScenarioEvent {
            tick: 1,
            station: Mk0Station::Detritivore,
            record_kind: Mk0RecordKind::MaintenanceFlag,
            record_id: maintenance.id.clone(),
            detail: maintenance.reason.clone(),
            payload: Mk0RecordPayload::MaintenanceFlag(maintenance),
        });
    } else {
        let batch = Mk0MaterialBatch {
            id: format!("batch-{}", config.work_order_id),
            source: "failed_prints_and_local_plastic".to_string(),
            available_grams: recycled_feedstock_grams,
            recycled: true,
        };
        events.push(Mk0ScenarioEvent {
            tick: 1,
            station: Mk0Station::Detritivore,
            record_kind: Mk0RecordKind::MaterialBatch,
            record_id: batch.id.clone(),
            detail: format!(
                "published recycled filament batch with {} g available",
                batch.available_grams
            ),
            payload: Mk0RecordPayload::MaterialBatch(batch),
        });
    }

    if energy_window_open {
        let print_claim = Mk0PrintClaim {
            id: format!("print-{}", config.work_order_id),
            work_order_id: config.work_order_id.clone(),
            recipe_version: "v1".to_string(),
        };
        events.push(Mk0ScenarioEvent {
            tick: 2,
            station: config.subassembly.printer_station,
            record_kind: Mk0RecordKind::PrintClaim,
            record_id: print_claim.id.clone(),
            detail: format!(
                "claimed print job with recipe {}",
                print_claim.recipe_version
            ),
            payload: Mk0RecordPayload::PrintClaim(print_claim),
        });

        if has_delivery_route {
            let delivery = Mk0DeliveryTask {
                id: format!("delivery-{}", config.work_order_id),
                work_order_id: config.work_order_id.clone(),
                from: config.subassembly.printer_station,
                to: config.subassembly.assembler_station,
            };
            events.push(Mk0ScenarioEvent {
                tick: 3,
                station: Mk0Station::Vector,
                record_kind: Mk0RecordKind::DeliveryTask,
                record_id: delivery.id.clone(),
                detail: format!("assigned {} handoff", route_label),
                payload: Mk0RecordPayload::DeliveryTask(delivery),
            });

            let handoff = Mk0HandoffReceipt {
                id: format!("handoff-{}", config.work_order_id),
                work_order_id: config.work_order_id.clone(),
                accepted_by: config.subassembly.assembler_station,
                completed_subassembly: config.subassembly.display_name.clone(),
            };
            events.push(Mk0ScenarioEvent {
                tick: 4,
                station: config.subassembly.assembler_station,
                record_kind: Mk0RecordKind::HandoffReceipt,
                record_id: handoff.id.clone(),
                detail: "assembled and transferred benchmark subassembly".to_string(),
                payload: Mk0RecordPayload::HandoffReceipt(handoff),
            });
        } else {
            let maintenance = Mk0MaintenanceFlag {
                id: format!("maint-route-{}", config.work_order_id),
                station: Mk0Station::Vector,
                reason: format!(
                    "no route from {} to {} in room layout '{}'",
                    config.subassembly.printer_station.as_str(),
                    config.subassembly.assembler_station.as_str(),
                    config.room_layout.name
                ),
            };
            events.push(Mk0ScenarioEvent {
                tick: 3,
                station: Mk0Station::Vector,
                record_kind: Mk0RecordKind::MaintenanceFlag,
                record_id: maintenance.id.clone(),
                detail: maintenance.reason.clone(),
                payload: Mk0RecordPayload::MaintenanceFlag(maintenance),
            });
        }
    } else {
        let maintenance = Mk0MaintenanceFlag {
            id: format!("maint-{}", config.work_order_id),
            station: Mk0Station::Helios,
            reason: "energy reserve below fabrication threshold".to_string(),
        };
        events.push(Mk0ScenarioEvent {
            tick: 2,
            station: Mk0Station::Helios,
            record_kind: Mk0RecordKind::MaintenanceFlag,
            record_id: maintenance.id.clone(),
            detail: maintenance.reason.clone(),
            payload: Mk0RecordPayload::MaintenanceFlag(maintenance),
        });
    }

    let energy_window = Mk0EnergyWindow {
        id: format!("energy-{}", config.work_order_id),
        fabrication_allowed: energy_window_open,
        reason: if energy_window_open {
            "one additional cycle allowed".to_string()
        } else {
            "noncritical fabrication denied".to_string()
        },
    };
    events.push(Mk0ScenarioEvent {
        tick: 5,
        station: Mk0Station::Helios,
        record_kind: Mk0RecordKind::EnergyWindow,
        record_id: energy_window.id.clone(),
        detail: format!("published energy window: {}", energy_window.reason),
        payload: Mk0RecordPayload::EnergyWindow(energy_window),
    });

    let handoff_receipts = events
        .iter()
        .filter(|event| event.record_kind == Mk0RecordKind::HandoffReceipt)
        .count() as u32;

    Mk0ScenarioReport {
        config,
        success: energy_window_open
            && recycled_feedstock_grams >= config.subassembly.required_material_grams
            && has_delivery_route
            && handoff_receipts > 0,
        completed_work_orders: handoff_receipts.min(1),
        handoff_receipts,
        recycled_feedstock_grams,
        energy_window_open,
        events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mk0_bootstrapper_scenario_emits_expected_mycelix_records() {
        let report = run_mk0_bootstrapper_scenario(Mk0ScenarioConfig::default());
        assert!(report.success);
        assert!(report.energy_window_open);
        assert_eq!(report.completed_work_orders, 1);
        assert_eq!(report.handoff_receipts, 1);

        let record_kinds: Vec<Mk0RecordKind> = report
            .events
            .iter()
            .map(|event| event.record_kind)
            .collect();
        assert_eq!(
            record_kinds,
            vec![
                Mk0RecordKind::WorkOrder,
                Mk0RecordKind::MaterialBatch,
                Mk0RecordKind::PrintClaim,
                Mk0RecordKind::DeliveryTask,
                Mk0RecordKind::HandoffReceipt,
                Mk0RecordKind::EnergyWindow,
            ]
        );
    }

    #[test]
    fn mk0_bootstrapper_scenario_blocks_when_energy_window_is_closed() {
        let report = run_mk0_bootstrapper_scenario(Mk0ScenarioConfig {
            initial_energy_reserve: 0.1,
            ..Mk0ScenarioConfig::default()
        });
        assert!(!report.success);
        assert!(!report.energy_window_open);
        assert_eq!(report.completed_work_orders, 0);
        assert_eq!(report.handoff_receipts, 0);
        assert!(report
            .events
            .iter()
            .any(|event| event.record_kind == Mk0RecordKind::MaintenanceFlag));
    }

    #[test]
    fn mk0_bootstrapper_scenario_blocks_when_room_layout_has_no_delivery_route() {
        let mut config = Mk0ScenarioConfig::default();
        config.room_layout.edges.clear();
        let report = run_mk0_bootstrapper_scenario(config);
        assert!(!report.success);
        assert_eq!(report.handoff_receipts, 0);
        assert!(report.events.iter().any(|event| {
            event.record_kind == Mk0RecordKind::MaintenanceFlag && event.detail.contains("no route")
        }));
    }

    #[test]
    fn mk0_bootstrapper_payloads_are_serializable() {
        let report = run_mk0_bootstrapper_scenario(Mk0ScenarioConfig::default());
        let json = serde_json::to_string(&report).expect("mk0 report should serialize");
        assert!(json.contains("work_order"));
        assert!(json.contains("handoff_receipt"));
        assert!(json.contains("energy_window"));
    }

    #[test]
    fn mk0_room_layout_preset_exposes_printer_to_bench_route() {
        let layout = Mk0RoomLayout::preset(Mk0ScenarioPreset::SingleRoomLinear);
        assert!(layout.has_route(Mk0Station::Fabricator, Mk0Station::Manipulator));
        assert_eq!(
            layout.route_label(Mk0Station::Fabricator, Mk0Station::Manipulator),
            Some("printer_to_bench")
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manufacturing Common Types
//!
//! Shared types and validation for the mycelix-manufacturing cluster.
//! All zomes in the cluster depend on this crate for canonical type definitions.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Work Orders
// ============================================================================

/// A manufacturing work order that tracks production of a specific product.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkOrder {
    pub product_id: String,
    pub quantity: u64,
    pub due_date: Timestamp,
    pub status: WorkOrderStatus,
    pub priority: WorkOrderPriority,
    pub notes: Option<String>,
    pub bom_hash: Option<String>,
    pub routing_hash: Option<String>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

/// Status lifecycle: Draft -> Released -> InProgress -> Completed | OnHold | Cancelled
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum WorkOrderStatus {
    Draft,
    Released,
    InProgress,
    Completed,
    OnHold,
    Cancelled,
    Closed,
}

impl WorkOrderStatus {
    /// Returns the set of valid next states from this status.
    pub fn valid_transitions(&self) -> Vec<WorkOrderStatus> {
        match self {
            Self::Draft => vec![Self::Released, Self::Cancelled],
            Self::Released => vec![Self::InProgress, Self::OnHold, Self::Cancelled],
            Self::InProgress => vec![Self::Completed, Self::OnHold, Self::Cancelled],
            Self::OnHold => vec![Self::Released, Self::InProgress, Self::Cancelled],
            Self::Completed => vec![Self::Closed],
            Self::Cancelled => vec![],
            Self::Closed => vec![],
        }
    }

    /// Check whether transitioning to `target` is valid from the current status.
    pub fn can_transition_to(&self, target: &WorkOrderStatus) -> bool {
        self.valid_transitions().contains(target)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum WorkOrderPriority {
    Low,
    Normal,
    High,
    Urgent,
}

// ============================================================================
// Bill of Materials (BOM)
// ============================================================================

/// A bill of materials for a product design.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BillOfMaterials {
    pub design_id: String,
    pub revision: String,
    pub items: Vec<BomItem>,
    pub created_at: Timestamp,
}

/// A single line item in a BOM.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BomItem {
    pub part_id: String,
    pub quantity_per: u64,
    pub unit: String,
    /// If set, this item is a sub-assembly referencing another BOM.
    pub sub_assembly_bom_hash: Option<String>,
    pub notes: Option<String>,
}

// ============================================================================
// Operations & Routing
// ============================================================================

/// A single manufacturing operation (e.g., "Mill profile", "Deburr").
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Operation {
    pub name: String,
    pub description: String,
    pub machine_type: String,
    pub setup_time_min: u32,
    pub cycle_time_min: u32,
    pub tooling: Option<String>,
}

/// An ordered sequence of operations that defines how a product is manufactured.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoutingSequence {
    pub design_id: String,
    pub revision: String,
    pub steps: Vec<RoutingStep>,
    pub created_at: Timestamp,
}

/// A single step in a routing sequence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoutingStep {
    pub sequence: u32,
    pub operation: Operation,
}

// ============================================================================
// Machines
// ============================================================================

/// A registered manufacturing machine or workstation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Machine {
    pub name: String,
    pub machine_type: MachineType,
    pub capabilities: Vec<String>,
    pub location: String,
    pub max_throughput_per_hour: u32,
    pub status: MachineStatus,
    pub current_work_order: Option<String>,
    pub registered_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum MachineType {
    /// Fused Deposition Modeling (3D printing)
    FDM,
    /// Stereolithography (resin 3D printing)
    SLA,
    /// Selective Laser Sintering (powder 3D printing)
    SLS,
    /// 3-axis CNC milling
    CNC3Axis,
    /// 5-axis CNC milling
    CNC5Axis,
    /// Laser cutting / engraving
    LaserCutter,
    /// Manual or CNC lathe
    Lathe,
    /// Manual or automated assembly station
    Assembly,
    /// Custom / other machine type
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum MachineStatus {
    Available,
    Running,
    Maintenance,
    Offline,
}

impl MachineStatus {
    pub fn can_transition_to(&self, target: &MachineStatus) -> bool {
        match self {
            Self::Available => matches!(target, Self::Running | Self::Maintenance | Self::Offline),
            Self::Running => matches!(target, Self::Available | Self::Maintenance | Self::Offline),
            Self::Maintenance => matches!(target, Self::Available | Self::Offline),
            Self::Offline => matches!(target, Self::Available | Self::Maintenance),
        }
    }
}

// ============================================================================
// MRP (Material Requirements Planning)
// ============================================================================

/// Result of an MRP planning run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MrpResult {
    pub planned_orders: Vec<PlannedOrder>,
    pub scheduled_operations: Vec<ScheduledOperation>,
    pub capacity_warnings: Vec<CapacityWarning>,
    pub material_shortages: Vec<MaterialShortage>,
    pub feasible: bool,
}

/// A planned procurement or production order generated by MRP.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlannedOrder {
    pub part_id: String,
    pub quantity_needed: u64,
    pub quantity_available: u64,
    pub quantity_to_order: u64,
    pub due_date: Timestamp,
}

/// A scheduled operation assigned to a machine with time windows.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScheduledOperation {
    pub work_order_id: String,
    pub operation_name: String,
    pub machine_id: String,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub setup_minutes: u32,
    pub run_minutes: u32,
}

/// A warning that machine capacity may be exceeded.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CapacityWarning {
    pub machine_id: String,
    pub machine_name: String,
    pub utilization_pct: f64,
    pub period: String,
    pub message: String,
}

/// A material shortage detected during MRP planning.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MaterialShortage {
    pub part_id: String,
    pub quantity_needed: u64,
    pub quantity_available: u64,
    pub short_quantity: u64,
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate a work order has required fields and sane values.
pub fn validate_work_order(wo: &WorkOrder) -> Result<(), String> {
    if wo.product_id.is_empty() {
        return Err("product_id is required".to_string());
    }
    if wo.product_id.len() > 200 {
        return Err("product_id must be <= 200 characters".to_string());
    }
    if wo.quantity == 0 {
        return Err("quantity must be > 0".to_string());
    }
    if wo.quantity > 1_000_000 {
        return Err("quantity must be <= 1,000,000".to_string());
    }
    Ok(())
}

/// Validate a BOM has at least one item and valid fields.
pub fn validate_bom(bom: &BillOfMaterials) -> Result<(), String> {
    if bom.design_id.is_empty() {
        return Err("design_id is required".to_string());
    }
    if bom.revision.is_empty() {
        return Err("revision is required".to_string());
    }
    if bom.items.is_empty() {
        return Err("BOM must have at least one item".to_string());
    }
    for (i, item) in bom.items.iter().enumerate() {
        if item.part_id.is_empty() && item.sub_assembly_bom_hash.is_none() {
            return Err(format!("item {i}: part_id or sub_assembly_bom_hash required"));
        }
        if item.quantity_per == 0 {
            return Err(format!("item {i}: quantity_per must be > 0"));
        }
    }
    Ok(())
}

/// Validate a machine has required fields.
pub fn validate_machine(machine: &Machine) -> Result<(), String> {
    if machine.name.is_empty() {
        return Err("machine name is required".to_string());
    }
    if machine.name.len() > 200 {
        return Err("machine name must be <= 200 characters".to_string());
    }
    if machine.location.is_empty() {
        return Err("machine location is required".to_string());
    }
    if machine.max_throughput_per_hour == 0 {
        return Err("max_throughput_per_hour must be > 0".to_string());
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_timestamp() -> Timestamp {
        Timestamp::from_micros(1_700_000_000_000_000)
    }

    fn make_work_order(product_id: &str, qty: u64) -> WorkOrder {
        WorkOrder {
            product_id: product_id.to_string(),
            quantity: qty,
            due_date: test_timestamp(),
            status: WorkOrderStatus::Draft,
            priority: WorkOrderPriority::Normal,
            notes: None,
            bom_hash: None,
            routing_hash: None,
            created_at: test_timestamp(),
            updated_at: test_timestamp(),
        }
    }

    #[test]
    fn test_work_order_status_transitions() {
        let draft = WorkOrderStatus::Draft;
        assert!(draft.can_transition_to(&WorkOrderStatus::Released));
        assert!(draft.can_transition_to(&WorkOrderStatus::Cancelled));
        assert!(!draft.can_transition_to(&WorkOrderStatus::Completed));
        assert!(!draft.can_transition_to(&WorkOrderStatus::InProgress));
    }

    #[test]
    fn test_status_completed_only_to_closed() {
        let completed = WorkOrderStatus::Completed;
        assert!(completed.can_transition_to(&WorkOrderStatus::Closed));
        assert!(!completed.can_transition_to(&WorkOrderStatus::Draft));
        assert!(!completed.can_transition_to(&WorkOrderStatus::Released));
    }

    #[test]
    fn test_terminal_states_no_transitions() {
        assert!(WorkOrderStatus::Cancelled.valid_transitions().is_empty());
        assert!(WorkOrderStatus::Closed.valid_transitions().is_empty());
    }

    #[test]
    fn test_on_hold_transitions() {
        let hold = WorkOrderStatus::OnHold;
        assert!(hold.can_transition_to(&WorkOrderStatus::Released));
        assert!(hold.can_transition_to(&WorkOrderStatus::InProgress));
        assert!(hold.can_transition_to(&WorkOrderStatus::Cancelled));
        assert!(!hold.can_transition_to(&WorkOrderStatus::Completed));
    }

    #[test]
    fn test_validate_work_order_ok() {
        let wo = make_work_order("WIDGET-001", 100);
        assert!(validate_work_order(&wo).is_ok());
    }

    #[test]
    fn test_validate_work_order_empty_product() {
        let wo = make_work_order("", 100);
        assert_eq!(validate_work_order(&wo).unwrap_err(), "product_id is required");
    }

    #[test]
    fn test_validate_work_order_zero_qty() {
        let wo = make_work_order("W", 0);
        assert_eq!(validate_work_order(&wo).unwrap_err(), "quantity must be > 0");
    }

    #[test]
    fn test_validate_bom_ok() {
        let bom = BillOfMaterials {
            design_id: "D-001".to_string(),
            revision: "A".to_string(),
            items: vec![BomItem {
                part_id: "BOLT-M6".to_string(),
                quantity_per: 4,
                unit: "each".to_string(),
                sub_assembly_bom_hash: None,
                notes: None,
            }],
            created_at: test_timestamp(),
        };
        assert!(validate_bom(&bom).is_ok());
    }

    #[test]
    fn test_validate_bom_empty_items() {
        let bom = BillOfMaterials {
            design_id: "D-001".to_string(),
            revision: "A".to_string(),
            items: vec![],
            created_at: test_timestamp(),
        };
        assert!(validate_bom(&bom).unwrap_err().contains("at least one item"));
    }

    #[test]
    fn test_validate_machine_ok() {
        let m = Machine {
            name: "CNC Mill #1".to_string(),
            machine_type: MachineType::CNC3Axis,
            capabilities: vec!["milling".to_string()],
            location: "Bay 1".to_string(),
            max_throughput_per_hour: 10,
            status: MachineStatus::Available,
            current_work_order: None,
            registered_at: test_timestamp(),
        };
        assert!(validate_machine(&m).is_ok());
    }

    #[test]
    fn test_validate_machine_empty_name() {
        let m = Machine {
            name: String::new(),
            machine_type: MachineType::Lathe,
            capabilities: vec![],
            location: "Bay 1".to_string(),
            max_throughput_per_hour: 5,
            status: MachineStatus::Available,
            current_work_order: None,
            registered_at: test_timestamp(),
        };
        assert!(validate_machine(&m).unwrap_err().contains("name is required"));
    }

    #[test]
    fn test_machine_status_transitions() {
        let avail = MachineStatus::Available;
        assert!(avail.can_transition_to(&MachineStatus::Running));
        assert!(avail.can_transition_to(&MachineStatus::Maintenance));
        assert!(!avail.can_transition_to(&MachineStatus::Available));
    }

    #[test]
    fn test_serde_roundtrip_work_order() {
        let wo = make_work_order("TEST-001", 50);
        let json = serde_json::to_string(&wo).unwrap();
        let back: WorkOrder = serde_json::from_str(&json).unwrap();
        assert_eq!(back.product_id, "TEST-001");
        assert_eq!(back.quantity, 50);
    }

    #[test]
    fn test_serde_roundtrip_machine_type() {
        let custom = MachineType::Custom("Waterjet".to_string());
        let json = serde_json::to_string(&custom).unwrap();
        let back: MachineType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, MachineType::Custom("Waterjet".to_string()));
    }

    #[test]
    fn test_mrp_result_construction() {
        let result = MrpResult {
            planned_orders: vec![],
            scheduled_operations: vec![],
            capacity_warnings: vec![],
            material_shortages: vec![MaterialShortage {
                part_id: "BOLT-M6".to_string(),
                quantity_needed: 100,
                quantity_available: 20,
                short_quantity: 80,
            }],
            feasible: false,
        };
        assert!(!result.feasible);
        assert_eq!(result.material_shortages[0].short_quantity, 80);
    }
}

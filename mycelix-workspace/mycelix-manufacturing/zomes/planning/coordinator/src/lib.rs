// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Planning Coordinator Zome
//!
//! Material Requirements Planning (MRP) engine. Queries work orders, BOMs,
//! machine capacity, and optionally supplychain inventory via cross-cluster
//! `CallTargetCell::Local` calls within the manufacturing DNA, and
//! `CallTargetCell::OtherRole` for cross-cluster inventory lookups.

use hdk::prelude::*;
use planning_integrity::*;
use manufacturing_common::{MaterialShortage, MrpResult, PlannedOrder};
use std::collections::HashMap;

/// Minimal projection of WorkOrderEntry for BOM linkage.
/// Only the fields we need, avoiding a dependency on workorders_integrity
/// (which would cause duplicate HDK symbol conflicts).
/// Uses `#[serde(default)]` on all fields so extra fields in WorkOrderEntry
/// are silently ignored during deserialization.
#[derive(Serialize, Deserialize, SerializedBytes, Debug)]
struct WorkOrderProjection {
    #[serde(default)]
    quantity: u64,
    #[serde(default)]
    bom_hash: Option<ActionHash>,
}

// ============================================================================
// Input / Output types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RunMrpInput {
    pub work_order_hashes: Vec<ActionHash>,
    pub horizon_days: Option<u32>,
    pub check_external_inventory: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MrpOutput {
    pub mrp_run_hash: ActionHash,
    pub result: MrpResult,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InventoryQueryInput {
    pub sku: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InventoryLevel {
    pub sku: String,
    pub quantity: u64,
    pub source: String,
}

// ============================================================================
// Extern functions
// ============================================================================

/// Run an MRP planning cycle for the given work orders.
///
/// Steps:
/// 1. Fetch each work order via `CallTargetCell::Local` to the workorders zome
/// 2. For each work order, fetch the linked BOM via `CallTargetCell::Local` to bom zome
/// 3. Explode BOMs to get material requirements
/// 4. Optionally query supplychain inventory via `CallTargetCell::OtherRole`
/// 5. Compute shortages, planned orders, and machine schedule
/// 6. Persist the MRP run result
#[hdk_extern]
pub fn run_mrp(input: RunMrpInput) -> ExternResult<MrpOutput> {
    if input.work_order_hashes.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Must provide at least one work order hash".to_string()
        )));
    }

    let horizon = input.horizon_days.unwrap_or(30);
    if horizon == 0 || horizon > 365 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "horizon_days must be 1-365".to_string()
        )));
    }

    // --- Step 1: Fetch work orders via local zome call ---
    let mut material_needs: HashMap<String, u64> = HashMap::new();

    for wo_hash in &input.work_order_hashes {
        // Call workorders coordinator locally to get work order
        let response = call(
            CallTargetCell::Local,
            ZomeName::from("workorders"),
            FunctionName::from("get_work_order"),
            None,
            ExternIO::encode(wo_hash.clone())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        )?;
        let wo_record = match response {
            ZomeCallResponse::Ok(data) => {
                let record: Option<Record> = data.decode()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
                record.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                    "Work order not found: {:?}", wo_hash
                ))))?
            }
            other => return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to fetch work order: {:?}", other
            )))),
        };

        // --- Step 2-3: Fetch and explode BOM ---
        // Use a minimal projection struct to decode just the fields we need,
        // avoiding a dependency on workorders_integrity (HDK symbol conflicts).
        let wo_proj: Option<WorkOrderProjection> = wo_record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

        if let Some(proj) = wo_proj {
            if let Some(bom_hash) = proj.bom_hash {
                let quantity = proj.quantity.max(1);

                // Call BOM coordinator to explode the BOM.
                let explode_input = ExternIO::encode(serde_json::json!({
                    "bom_hash": bom_hash,
                    "quantity": quantity,
                    "flatten": true
                }))
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

                let bom_response = call(
                    CallTargetCell::Local,
                    ZomeName::from("bom"),
                    FunctionName::from("explode_bom"),
                    None,
                    explode_input,
                )?;

                if let ZomeCallResponse::Ok(data) = bom_response {
                    // Decode as a generic JSON value to extract explosion lines.
                    if let Ok(output) = data.decode::<serde_json::Value>() {
                        if let Some(lines) = output.get("lines").and_then(|l| l.as_array()) {
                            for line in lines {
                                if let (Some(part_id), Some(total_qty)) = (
                                    line.get("part_id").and_then(|p| p.as_str()),
                                    line.get("total_quantity").and_then(|q| q.as_u64()),
                                ) {
                                    *material_needs.entry(part_id.to_string()).or_insert(0) += total_qty;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let material_needs: Vec<(String, u64)> = material_needs.into_iter().collect();

    // --- Step 4: Optionally query external inventory ---
    let mut external_inventory: Vec<InventoryLevel> = Vec::new();
    if input.check_external_inventory {
        // Query supplychain cluster via OtherRole bridge
        // This call resolves if the unified hApp has a supplychain role installed
        for (sku, _qty) in &material_needs {
            let query = InventoryQueryInput {
                sku: sku.clone(),
            };
            let payload = ExternIO::encode(query)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
            match call(
                CallTargetCell::OtherRole("supplychain".into()),
                ZomeName::from("inventory"),
                FunctionName::from("get_stock_level_by_sku"),
                None,
                payload,
            ) {
                Ok(ZomeCallResponse::Ok(data)) => {
                    if let Ok(Some(level)) = data.decode::<Option<InventoryLevel>>() {
                        external_inventory.push(level);
                    }
                }
                _ => {
                    // Cross-cluster call failed -- supplychain may not be installed.
                    // Continue with local-only planning.
                }
            }
        }
    }

    // --- Step 5: Compute shortages and plan ---
    let mut shortages = Vec::new();
    let mut planned_orders = Vec::new();
    let now = sys_time()?;

    for (sku, qty_needed) in &material_needs {
        let available = external_inventory
            .iter()
            .find(|l| l.sku == *sku)
            .map(|l| l.quantity)
            .unwrap_or(0);

        if available < *qty_needed {
            let short = qty_needed - available;
            shortages.push(MaterialShortage {
                part_id: sku.clone(),
                quantity_needed: *qty_needed,
                quantity_available: available,
                short_quantity: short,
            });
            planned_orders.push(PlannedOrder {
                part_id: sku.clone(),
                quantity_needed: *qty_needed,
                quantity_available: available,
                quantity_to_order: short,
                due_date: now,
            });
        }
    }

    let feasible = shortages.is_empty();

    let result = MrpResult {
        planned_orders,
        scheduled_operations: Vec::new(), // TODO: machine scheduling
        capacity_warnings: Vec::new(),
        material_shortages: shortages,
        feasible,
    };

    // --- Step 6: Persist the MRP run ---
    let run_entry = MrpRunEntry {
        work_order_hashes: input.work_order_hashes.clone(),
        horizon_days: horizon,
        run_at: now,
        feasible,
    };
    let run_hash = create_entry(EntryTypes::MrpRun(run_entry))?;

    // Link from "all_mrp_runs" anchor
    let all_path = Path::from("all_mrp_runs").typed(LinkTypes::AllMrpRuns)?;
    all_path.ensure()?;
    create_link(
        all_path.path_entry_hash()?,
        run_hash.clone(),
        LinkTypes::AllMrpRuns,
        (),
    )?;

    // Link each work order to this MRP run
    for wo_hash in &input.work_order_hashes {
        create_link(
            wo_hash.clone(),
            run_hash.clone(),
            LinkTypes::WorkOrderToMrpRuns,
            (),
        )?;
    }

    // Persist shortage entries
    for shortage in &result.material_shortages {
        let entry = MaterialShortageEntry {
            mrp_run_hash: run_hash.clone(),
            part_id: shortage.part_id.clone(),
            quantity_needed: shortage.quantity_needed,
            quantity_available: shortage.quantity_available,
            short_quantity: shortage.short_quantity,
        };
        let sh = create_entry(EntryTypes::MaterialShortage(entry))?;
        create_link(
            run_hash.clone(),
            sh,
            LinkTypes::MrpRunToShortages,
            (),
        )?;
    }

    // Persist planned order entries
    for po in &result.planned_orders {
        let entry = PlannedOrderEntry {
            mrp_run_hash: run_hash.clone(),
            part_id: po.part_id.clone(),
            quantity_needed: po.quantity_needed,
            quantity_available: po.quantity_available,
            quantity_to_order: po.quantity_to_order,
            due_date: po.due_date,
        };
        let ph = create_entry(EntryTypes::PlannedOrder(entry))?;
        create_link(
            run_hash.clone(),
            ph,
            LinkTypes::MrpRunToPlannedOrders,
            (),
        )?;
    }

    Ok(MrpOutput {
        mrp_run_hash: run_hash,
        result,
    })
}

/// Get an MRP run by its action hash.
#[hdk_extern]
pub fn get_mrp_run(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// List all MRP runs.
#[hdk_extern]
pub fn list_mrp_runs(_: ()) -> ExternResult<Vec<Link>> {
    let all_path = Path::from("all_mrp_runs").typed(LinkTypes::AllMrpRuns)?;
    get_links(
        GetLinksInputBuilder::try_new(all_path.path_entry_hash()?, LinkTypes::AllMrpRuns)?.build(),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_mrp_input_serde() {
        let input = RunMrpInput {
            work_order_hashes: vec![ActionHash::from_raw_36(vec![0u8; 36])],
            horizon_days: Some(30),
            check_external_inventory: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RunMrpInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.work_order_hashes.len(), 1);
        assert_eq!(back.horizon_days, Some(30));
        assert!(back.check_external_inventory);
    }

    #[test]
    fn test_inventory_level_serde() {
        let level = InventoryLevel {
            sku: "BOLT-M6".to_string(),
            quantity: 500,
            source: "supplychain".to_string(),
        };
        let json = serde_json::to_string(&level).unwrap();
        let back: InventoryLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sku, "BOLT-M6");
        assert_eq!(back.quantity, 500);
    }

    #[test]
    fn test_mrp_result_serde_roundtrip() {
        let result = MrpResult {
            planned_orders: vec![PlannedOrder {
                part_id: "P1".to_string(),
                quantity_needed: 100,
                quantity_available: 20,
                quantity_to_order: 80,
                due_date: Timestamp::from_micros(0),
            }],
            scheduled_operations: vec![],
            capacity_warnings: vec![],
            material_shortages: vec![MaterialShortage {
                part_id: "P1".to_string(),
                quantity_needed: 100,
                quantity_available: 20,
                short_quantity: 80,
            }],
            feasible: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: MrpResult = serde_json::from_str(&json).unwrap();
        assert!(!back.feasible);
        assert_eq!(back.material_shortages.len(), 1);
        assert_eq!(back.planned_orders[0].quantity_to_order, 80);
    }

    #[test]
    fn test_material_needs_accumulation() {
        // Simulate the HashMap merge logic used in run_mrp
        let mut material_needs: HashMap<String, u64> = HashMap::new();

        // First BOM explosion: BOLT-M6 x 10, NUT-M6 x 20
        *material_needs.entry("BOLT-M6".to_string()).or_insert(0) += 10;
        *material_needs.entry("NUT-M6".to_string()).or_insert(0) += 20;

        // Second BOM explosion: BOLT-M6 x 5, WASHER-M6 x 30
        *material_needs.entry("BOLT-M6".to_string()).or_insert(0) += 5;
        *material_needs.entry("WASHER-M6".to_string()).or_insert(0) += 30;

        assert_eq!(*material_needs.get("BOLT-M6").unwrap(), 15);
        assert_eq!(*material_needs.get("NUT-M6").unwrap(), 20);
        assert_eq!(*material_needs.get("WASHER-M6").unwrap(), 30);
        assert_eq!(material_needs.len(), 3);

        // Convert to Vec (as done in run_mrp)
        let needs_vec: Vec<(String, u64)> = material_needs.into_iter().collect();
        assert_eq!(needs_vec.len(), 3);
        let total: u64 = needs_vec.iter().map(|(_, q)| q).sum();
        assert_eq!(total, 65);
    }

    #[test]
    fn test_material_shortage_math() {
        let needed = 100u64;
        let available = 20u64;
        let short = needed.saturating_sub(available);
        assert_eq!(short, 80);
    }

    #[test]
    fn test_mrp_output_serde() {
        let output = MrpOutput {
            mrp_run_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            result: MrpResult {
                planned_orders: vec![],
                scheduled_operations: vec![],
                capacity_warnings: vec![],
                material_shortages: vec![],
                feasible: true,
            },
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: MrpOutput = serde_json::from_str(&json).unwrap();
        assert!(back.result.feasible);
    }
}

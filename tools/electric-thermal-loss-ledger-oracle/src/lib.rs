// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent electrical-load and thermal-loss allocation reference ledgers.
//!
//! The electrical ledger sums unique, evidence-bound meter boundaries.
//! The thermal ledger allocates declared component losses to explicit thermal
//! nodes and preserves any unallocated remainder. The two ledgers remain
//! separate so a conversion loss is not added twice to electrical demand and
//! auxiliary/TMS power is not automatically assigned as battery heat.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const MODEL_VERSION: &str = "electric-thermal-loss-ledger-oracle-v0.1.0";

/// Electrical load categories at declared non-overlapping meter boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ElectricalLoadKind {
    PropulsionDcInput,
    ThermalManagement,
    OtherAuxiliary,
    UpstreamDistributionLoss,
    BatteryTerminalTotal,
}

/// One power measurement at a unique electrical boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectricalLoad {
    pub load_id: String,
    pub meter_boundary_id: String,
    pub kind: ElectricalLoadKind,
    pub power_w: f64,
    pub evidence_manifest_digest: String,
}

/// One per-load energy result, preserving source identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectricalLoadResult {
    pub load_id: String,
    pub meter_boundary_id: String,
    pub kind: ElectricalLoadKind,
    pub power_w: f64,
    pub energy_j: f64,
    pub evidence_manifest_digest: String,
}

/// Total battery-terminal demand and deterministic category subtotals.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectricalLoadReport {
    pub model_version: &'static str,
    pub duration_s: f64,
    pub propulsion_dc_input_power_w: f64,
    pub thermal_management_power_w: f64,
    pub other_auxiliary_power_w: f64,
    pub upstream_distribution_loss_power_w: f64,
    pub battery_terminal_power_w: f64,
    pub propulsion_dc_input_energy_j: f64,
    pub thermal_management_energy_j: f64,
    pub other_auxiliary_energy_j: f64,
    pub upstream_distribution_loss_energy_j: f64,
    pub battery_terminal_energy_j: f64,
    pub loads: Vec<ElectricalLoadResult>,
}

/// Physical loss categories that may have explicit thermal allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HeatLossKind {
    Transmission,
    Motor,
    Inverter,
    BatteryOhmic,
    DcDistribution,
    ThermalManagementInternal,
    Other,
}

/// Fraction of one declared component loss allocated to a thermal node.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalNodeAllocation {
    pub thermal_node_id: String,
    pub fraction: f64,
}

/// A declared component loss and its explicit source-to-node allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatLossSource {
    pub source_id: String,
    pub component_id: String,
    pub kind: HeatLossKind,
    pub loss_power_w: f64,
    pub evidence_manifest_digest: String,
    pub allocations: Vec<ThermalNodeAllocation>,
}

/// One source's allocation result. Unallocated loss remains visible.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatLossSourceResult {
    pub source_id: String,
    pub component_id: String,
    pub kind: HeatLossKind,
    pub loss_power_w: f64,
    pub allocated_heat_power_w: f64,
    pub unallocated_loss_power_w: f64,
    pub allocation_residual_w: f64,
    pub evidence_manifest_digest: String,
    pub allocations: Vec<ThermalNodeAllocation>,
}

/// Aggregated generation at one declared thermal node.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalNodeHeat {
    pub thermal_node_id: String,
    pub generated_heat_power_w: f64,
}

/// Complete heat-allocation report, not a thermal network balance.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatAllocationReport {
    pub model_version: &'static str,
    pub total_declared_loss_power_w: f64,
    pub total_allocated_heat_power_w: f64,
    pub total_unallocated_loss_power_w: f64,
    pub total_allocation_residual_w: f64,
    pub sources: Vec<HeatLossSourceResult>,
    pub thermal_nodes: Vec<ThermalNodeHeat>,
}

/// Explicit errors for invalid identities, accounting boundaries or values.
#[derive(Debug, Clone, PartialEq)]
pub enum LedgerError {
    InvalidInput(&'static str),
    DuplicateIdentity(&'static str),
    OverlappingElectricalBoundary,
    InvalidAllocation(&'static str),
    NumericalOverflow,
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LedgerError {}

fn add_finite(total: &mut f64, value: f64) -> Result<(), LedgerError> {
    let next = *total + value;
    if !next.is_finite() {
        return Err(LedgerError::NumericalOverflow);
    }
    *total = next;
    Ok(())
}

/// Sum non-overlapping electrical load readings into battery-terminal demand.
///
/// A BatteryTerminalTotal load must be the only supplied load. Otherwise, the
/// caller supplies branch loads and each meter boundary may appear exactly once.
/// Branch power/energy subtotals remain separate by category.
pub fn evaluate_electrical_loads(
    loads: &[ElectricalLoad],
    duration_s: f64,
) -> Result<ElectricalLoadReport, LedgerError> {
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(LedgerError::InvalidInput(
            "duration must be finite and positive",
        ));
    }
    if loads.is_empty() {
        return Err(LedgerError::InvalidInput(
            "at least one electrical load must be supplied",
        ));
    }

    let mut load_ids = BTreeSet::new();
    let mut boundary_ids = BTreeSet::new();
    let mut total = 0.0;
    let mut propulsion = 0.0;
    let mut tms = 0.0;
    let mut auxiliary = 0.0;
    let mut distribution = 0.0;
    let mut battery_total = None;

    let mut ordered_loads = loads.to_vec();
    ordered_loads.sort_by(|a, b| a.load_id.cmp(&b.load_id));

    for load in &ordered_loads {
        if load.load_id.trim().is_empty()
            || load.meter_boundary_id.trim().is_empty()
            || load.evidence_manifest_digest.trim().is_empty()
        {
            return Err(LedgerError::InvalidInput(
                "load, meter boundary and evidence identities must not be empty",
            ));
        }
        if !load.power_w.is_finite() || load.power_w < 0.0 {
            return Err(LedgerError::InvalidInput(
                "electrical load power must be finite and non-negative",
            ));
        }
        if !load_ids.insert(load.load_id.as_str()) {
            return Err(LedgerError::DuplicateIdentity("electrical load ID"));
        }
        if !boundary_ids.insert(load.meter_boundary_id.as_str()) {
            return Err(LedgerError::DuplicateIdentity("meter boundary ID"));
        }
        if load.kind == ElectricalLoadKind::BatteryTerminalTotal {
            if battery_total.replace(load.power_w).is_some() {
                return Err(LedgerError::DuplicateIdentity(
                    "battery-terminal total meter",
                ));
            }
        } else {
            add_finite(&mut total, load.power_w)?;
            match load.kind {
                ElectricalLoadKind::PropulsionDcInput => {
                    add_finite(&mut propulsion, load.power_w)?;
                }
                ElectricalLoadKind::ThermalManagement => {
                    add_finite(&mut tms, load.power_w)?;
                }
                ElectricalLoadKind::OtherAuxiliary => {
                    add_finite(&mut auxiliary, load.power_w)?;
                }
                ElectricalLoadKind::UpstreamDistributionLoss => {
                    add_finite(&mut distribution, load.power_w)?;
                }
                ElectricalLoadKind::BatteryTerminalTotal => unreachable!(),
            }
        }
    }

    let battery_terminal_power_w = if let Some(power) = battery_total {
        if ordered_loads.len() != 1 {
            return Err(LedgerError::OverlappingElectricalBoundary);
        }
        power
    } else {
        total
    };

    let to_energy = |power: f64| {
        let energy = power * duration_s;
        if energy.is_finite() {
            Ok(energy)
        } else {
            Err(LedgerError::NumericalOverflow)
        }
    };

    let load_results = ordered_loads
        .into_iter()
        .map(|load| {
            Ok(ElectricalLoadResult {
                load_id: load.load_id,
                meter_boundary_id: load.meter_boundary_id,
                kind: load.kind,
                power_w: load.power_w,
                energy_j: to_energy(load.power_w)?,
                evidence_manifest_digest: load.evidence_manifest_digest,
            })
        })
        .collect::<Result<Vec<_>, LedgerError>>()?;

    let (
        propulsion_power,
        tms_power,
        auxiliary_power,
        distribution_power,
    ) = if battery_total.is_some() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (propulsion, tms, auxiliary, distribution)
    };

    Ok(ElectricalLoadReport {
        model_version: MODEL_VERSION,
        duration_s,
        propulsion_dc_input_power_w: propulsion_power,
        thermal_management_power_w: tms_power,
        other_auxiliary_power_w: auxiliary_power,
        upstream_distribution_loss_power_w: distribution_power,
        battery_terminal_power_w,
        propulsion_dc_input_energy_j: to_energy(propulsion_power)?,
        thermal_management_energy_j: to_energy(tms_power)?,
        other_auxiliary_energy_j: to_energy(auxiliary_power)?,
        upstream_distribution_loss_energy_j: to_energy(distribution_power)?,
        battery_terminal_energy_j: to_energy(battery_terminal_power_w)?,
        loads: load_results,
    })
}

/// Allocate explicitly declared loss power to thermal nodes without changing
/// the electrical load ledger. Unallocated loss remains a first-class result.
pub fn allocate_loss_heat(
    sources: &[HeatLossSource],
) -> Result<HeatAllocationReport, LedgerError> {
    if sources.is_empty() {
        return Err(LedgerError::InvalidInput(
            "at least one heat-loss source must be supplied",
        ));
    }

    let mut source_ids = BTreeSet::new();
    let mut node_totals = BTreeMap::<String, f64>::new();
    let mut total_loss = 0.0;
    let mut total_allocated = 0.0;
    let mut total_unallocated = 0.0;
    let mut total_residual = 0.0;

    let mut ordered_sources = sources.to_vec();
    ordered_sources.sort_by(|a, b| a.source_id.cmp(&b.source_id));

    let mut source_results = Vec::with_capacity(ordered_sources.len());
    for source in ordered_sources {
        if source.source_id.trim().is_empty()
            || source.component_id.trim().is_empty()
            || source.evidence_manifest_digest.trim().is_empty()
        {
            return Err(LedgerError::InvalidInput(
                "source, component and evidence identities must not be empty",
            ));
        }
        if !source_ids.insert(source.source_id.as_str()) {
            return Err(LedgerError::DuplicateIdentity("thermal loss source ID"));
        }
        if !source.loss_power_w.is_finite() || source.loss_power_w < 0.0 {
            return Err(LedgerError::InvalidInput(
                "loss source power must be finite and non-negative",
            ));
        }

        let mut seen_nodes = BTreeSet::new();
        let mut ordered_allocations = source.allocations.clone();
        ordered_allocations.sort_by(|a, b| a.thermal_node_id.cmp(&b.thermal_node_id));
        let mut fraction_total = 0.0;
        for allocation in &ordered_allocations {
            if allocation.thermal_node_id.trim().is_empty() {
                return Err(LedgerError::InvalidAllocation(
                    "thermal node ID must not be empty",
                ));
            }
            if !seen_nodes.insert(allocation.thermal_node_id.as_str()) {
                return Err(LedgerError::DuplicateIdentity(
                    "thermal node allocation within one source",
                ));
            }
            if !allocation.fraction.is_finite()
                || !(0.0..=1.0).contains(&allocation.fraction)
            {
                return Err(LedgerError::InvalidAllocation(
                    "allocation fraction must be finite and in [0, 1]",
                ));
            }
            add_finite(&mut fraction_total, allocation.fraction)?;
        }
        if fraction_total > 1.0 {
            return Err(LedgerError::InvalidAllocation(
                "allocation fractions for one source must sum to at most 1",
            ));
        }

        let mut allocated_power = 0.0;
        for allocation in &ordered_allocations {
            let node_power = source.loss_power_w * allocation.fraction;
            if !node_power.is_finite() {
                return Err(LedgerError::NumericalOverflow);
            }
            add_finite(&mut allocated_power, node_power)?;
            let node_total = node_totals
                .entry(allocation.thermal_node_id.clone())
                .or_insert(0.0);
            add_finite(node_total, node_power)?;
        }

        let unallocated_power = source.loss_power_w * (1.0 - fraction_total);
        if !unallocated_power.is_finite() {
            return Err(LedgerError::NumericalOverflow);
        }
        let reconstructed = allocated_power + unallocated_power;
        if !reconstructed.is_finite() {
            return Err(LedgerError::NumericalOverflow);
        }
        let residual = source.loss_power_w - reconstructed;

        add_finite(&mut total_loss, source.loss_power_w)?;
        add_finite(&mut total_allocated, allocated_power)?;
        add_finite(&mut total_unallocated, unallocated_power)?;
        add_finite(&mut total_residual, residual)?;

        source_results.push(HeatLossSourceResult {
            source_id: source.source_id,
            component_id: source.component_id,
            kind: source.kind,
            loss_power_w: source.loss_power_w,
            allocated_heat_power_w: allocated_power,
            unallocated_loss_power_w: unallocated_power,
            allocation_residual_w: residual,
            evidence_manifest_digest: source.evidence_manifest_digest,
            allocations: ordered_allocations,
        });
    }

    let thermal_nodes = node_totals
        .into_iter()
        .map(|(thermal_node_id, generated_heat_power_w)| ThermalNodeHeat {
            thermal_node_id,
            generated_heat_power_w,
        })
        .collect();

    Ok(HeatAllocationReport {
        model_version: MODEL_VERSION,
        total_declared_loss_power_w: total_loss,
        total_allocated_heat_power_w: total_allocated,
        total_unallocated_loss_power_w: total_unallocated,
        total_allocation_residual_w: total_residual,
        sources: source_results,
        thermal_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn electrical_load(
        load_id: &str,
        boundary: &str,
        kind: ElectricalLoadKind,
        power_w: f64,
    ) -> ElectricalLoad {
        ElectricalLoad {
            load_id: load_id.to_owned(),
            meter_boundary_id: boundary.to_owned(),
            kind,
            power_w,
            evidence_manifest_digest: format!("evidence-{load_id}"),
        }
    }

    fn source(
        source_id: &str,
        component_id: &str,
        kind: HeatLossKind,
        loss_power_w: f64,
        allocations: Vec<ThermalNodeAllocation>,
    ) -> HeatLossSource {
        HeatLossSource {
            source_id: source_id.to_owned(),
            component_id: component_id.to_owned(),
            kind,
            loss_power_w,
            evidence_manifest_digest: format!("evidence-{source_id}"),
            allocations,
        }
    }

    #[test]
    fn electrical_loads_sum_to_battery_terminal_power_and_energy() {
        let loads = vec![
            electrical_load(
                "propulsion",
                "inverter-dc-terminal",
                ElectricalLoadKind::PropulsionDcInput,
                11_000.0,
            ),
            electrical_load(
                "tms",
                "tms-terminal",
                ElectricalLoadKind::ThermalManagement,
                800.0,
            ),
            electrical_load(
                "avionics",
                "avionics-terminal",
                ElectricalLoadKind::OtherAuxiliary,
                200.0,
            ),
            electrical_load(
                "distribution",
                "upstream-bus-loss",
                ElectricalLoadKind::UpstreamDistributionLoss,
                100.0,
            ),
        ];
        let report = evaluate_electrical_loads(&loads, 12.0).unwrap();
        assert!((report.battery_terminal_power_w - 12_100.0).abs() < 1.0e-12);
        assert!((report.battery_terminal_energy_j - 145_200.0).abs() < 1.0e-9);
        assert_eq!(report.thermal_management_power_w, 800.0);
        assert_eq!(report.propulsion_dc_input_power_w, 11_000.0);
        assert_eq!(report.loads.len(), 4);
    }

    #[test]
    fn thermal_management_power_is_not_automatically_battery_heat() {
        let loads = vec![electrical_load(
            "tms",
            "tms-terminal",
            ElectricalLoadKind::ThermalManagement,
            800.0,
        )];
        let electrical = evaluate_electrical_loads(&loads, 10.0).unwrap();
        assert_eq!(electrical.thermal_management_power_w, 800.0);

        let heat = allocate_loss_heat(&[source(
            "motor-loss",
            "motor-1",
            HeatLossKind::Motor,
            600.0,
            vec![],
        )])
        .unwrap();
        assert_eq!(heat.total_declared_loss_power_w, 600.0);
        assert_eq!(heat.total_allocated_heat_power_w, 0.0);
        assert_eq!(heat.total_unallocated_loss_power_w, 600.0);
    }

    #[test]
    fn battery_terminal_total_cannot_be_double_counted_with_branch_loads() {
        let loads = vec![
            electrical_load(
                "battery-total",
                "battery-terminal",
                ElectricalLoadKind::BatteryTerminalTotal,
                12_000.0,
            ),
            electrical_load(
                "tms",
                "tms-terminal",
                ElectricalLoadKind::ThermalManagement,
                500.0,
            ),
        ];
        assert_eq!(
            evaluate_electrical_loads(&loads, 1.0),
            Err(LedgerError::OverlappingElectricalBoundary)
        );
    }

    #[test]
    fn one_battery_terminal_total_is_accepted_as_the_exclusive_boundary() {
        let loads = vec![electrical_load(
            "battery-total",
            "battery-terminal",
            ElectricalLoadKind::BatteryTerminalTotal,
            12_000.0,
        )];
        let report = evaluate_electrical_loads(&loads, 2.0).unwrap();
        assert_eq!(report.battery_terminal_power_w, 12_000.0);
        assert_eq!(report.battery_terminal_energy_j, 24_000.0);
        assert_eq!(report.propulsion_dc_input_power_w, 0.0);
        assert_eq!(report.thermal_management_power_w, 0.0);
    }

    #[test]
    fn duplicate_load_and_meter_boundary_identities_are_rejected() {
        let duplicated_load = vec![
            electrical_load("same", "meter-a", ElectricalLoadKind::OtherAuxiliary, 1.0),
            electrical_load("same", "meter-b", ElectricalLoadKind::OtherAuxiliary, 2.0),
        ];
        assert_eq!(
            evaluate_electrical_loads(&duplicated_load, 1.0),
            Err(LedgerError::DuplicateIdentity("electrical load ID"))
        );

        let duplicated_boundary = vec![
            electrical_load("load-a", "same-meter", ElectricalLoadKind::OtherAuxiliary, 1.0),
            electrical_load("load-b", "same-meter", ElectricalLoadKind::OtherAuxiliary, 2.0),
        ];
        assert_eq!(
            evaluate_electrical_loads(&duplicated_boundary, 1.0),
            Err(LedgerError::DuplicateIdentity("meter boundary ID"))
        );
    }

    #[test]
    fn empty_identity_non_finite_power_and_invalid_duration_fail_closed() {
        let mut load = electrical_load("load", "meter", ElectricalLoadKind::OtherAuxiliary, 1.0);
        load.evidence_manifest_digest.clear();
        assert!(matches!(
            evaluate_electrical_loads(&[load], 1.0),
            Err(LedgerError::InvalidInput(_))
        ));

        let load = electrical_load("load", "meter", ElectricalLoadKind::OtherAuxiliary, f64::NAN);
        assert!(matches!(
            evaluate_electrical_loads(&[load], 1.0),
            Err(LedgerError::InvalidInput(_))
        ));

        let load = electrical_load("load", "meter", ElectricalLoadKind::OtherAuxiliary, 1.0);
        assert!(matches!(
            evaluate_electrical_loads(&[load], 0.0),
            Err(LedgerError::InvalidInput(_))
        ));
    }

    #[test]
    fn heat_loss_fully_allocated_to_one_node_closes() {
        let report = allocate_loss_heat(&[source(
            "motor-loss",
            "motor-1",
            HeatLossKind::Motor,
            600.0,
            vec![ThermalNodeAllocation {
                thermal_node_id: "motor-stator".to_owned(),
                fraction: 1.0,
            }],
        )])
        .unwrap();
        assert_eq!(report.total_declared_loss_power_w, 600.0);
        assert_eq!(report.total_allocated_heat_power_w, 600.0);
        assert_eq!(report.total_unallocated_loss_power_w, 0.0);
        assert_eq!(report.thermal_nodes[0].generated_heat_power_w, 600.0);
        assert_eq!(report.sources[0].allocation_residual_w, 0.0);
    }

    #[test]
    fn heat_loss_split_between_nodes_closes_and_preserves_lineage() {
        let report = allocate_loss_heat(&[source(
            "gearbox-loss",
            "gearbox-1",
            HeatLossKind::Transmission,
            1_000.0,
            vec![
                ThermalNodeAllocation {
                    thermal_node_id: "oil".to_owned(),
                    fraction: 0.6,
                },
                ThermalNodeAllocation {
                    thermal_node_id: "housing".to_owned(),
                    fraction: 0.4,
                },
            ],
        )])
        .unwrap();
        assert_eq!(report.total_allocated_heat_power_w, 1_000.0);
        assert_eq!(report.total_unallocated_loss_power_w, 0.0);
        assert_eq!(report.thermal_nodes.len(), 2);
        assert_eq!(report.sources[0].source_id, "gearbox-loss");
    }

    #[test]
    fn partial_and_zero_allocations_keep_unallocated_loss_visible() {
        let partial = allocate_loss_heat(&[source(
            "inverter-loss",
            "inverter-1",
            HeatLossKind::Inverter,
            500.0,
            vec![ThermalNodeAllocation {
                thermal_node_id: "power-module".to_owned(),
                fraction: 0.7,
            }],
        )])
        .unwrap();
        assert!((partial.total_allocated_heat_power_w - 350.0).abs() < 1.0e-12);
        assert!((partial.total_unallocated_loss_power_w - 150.0).abs() < 1.0e-12);

        let zero = allocate_loss_heat(&[source(
            "unknown-loss",
            "component-unknown",
            HeatLossKind::Other,
            200.0,
            vec![],
        )])
        .unwrap();
        assert_eq!(zero.total_allocated_heat_power_w, 0.0);
        assert_eq!(zero.total_unallocated_loss_power_w, 200.0);
    }

    #[test]
    fn duplicate_sources_nodes_and_invalid_fractions_fail_closed() {
        let duplicate_sources = vec![
            source("same", "motor-1", HeatLossKind::Motor, 100.0, vec![]),
            source("same", "motor-2", HeatLossKind::Motor, 100.0, vec![]),
        ];
        assert_eq!(
            allocate_loss_heat(&duplicate_sources),
            Err(LedgerError::DuplicateIdentity("thermal loss source ID"))
        );

        let duplicate_nodes = vec![source(
            "source",
            "motor-1",
            HeatLossKind::Motor,
            100.0,
            vec![
                ThermalNodeAllocation {
                    thermal_node_id: "stator".to_owned(),
                    fraction: 0.5,
                },
                ThermalNodeAllocation {
                    thermal_node_id: "stator".to_owned(),
                    fraction: 0.5,
                },
            ],
        )];
        assert_eq!(
            allocate_loss_heat(&duplicate_nodes),
            Err(LedgerError::DuplicateIdentity(
                "thermal node allocation within one source"
            ))
        );

        for fractions in [vec![1.1], vec![0.7, 0.4]] {
            let allocations = fractions
                .iter()
                .enumerate()
                .map(|(index, fraction)| ThermalNodeAllocation {
                    thermal_node_id: format!("node-{index}"),
                    fraction: *fraction,
                })
                .collect();
            assert!(matches!(
                allocate_loss_heat(&[source(
                    "source",
                    "motor-1",
                    HeatLossKind::Motor,
                    100.0,
                    allocations,
                )]),
                Err(LedgerError::InvalidAllocation(_))
            ));
        }
    }

    #[test]
    fn multiple_sources_aggregate_deterministically_without_losing_lineage() {
        let sources = vec![
            source(
                "loss-b",
                "inverter-1",
                HeatLossKind::Inverter,
                200.0,
                vec![ThermalNodeAllocation {
                    thermal_node_id: "shared-coolant".to_owned(),
                    fraction: 0.5,
                }],
            ),
            source(
                "loss-a",
                "motor-1",
                HeatLossKind::Motor,
                400.0,
                vec![ThermalNodeAllocation {
                    thermal_node_id: "shared-coolant".to_owned(),
                    fraction: 0.5,
                }],
            ),
        ];
        let first = allocate_loss_heat(&sources).unwrap();
        let second = allocate_loss_heat(&sources).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sources[0].source_id, "loss-a");
        assert_eq!(first.thermal_nodes[0].thermal_node_id, "shared-coolant");
        assert_eq!(first.thermal_nodes[0].generated_heat_power_w, 300.0);
    }

    #[test]
    fn overflow_and_empty_inputs_fail_closed() {
        let load = electrical_load(
            "huge",
            "huge-meter",
            ElectricalLoadKind::OtherAuxiliary,
            f64::MAX,
        );
        assert_eq!(
            evaluate_electrical_loads(&[load], 2.0),
            Err(LedgerError::NumericalOverflow)
        );

        let source_value = source(
            "huge-loss",
            "component",
            HeatLossKind::Other,
            f64::MAX,
            vec![ThermalNodeAllocation {
                thermal_node_id: "node".to_owned(),
                fraction: 1.0,
            }],
        );
        assert_eq!(
            allocate_loss_heat(&[source_value]),
            Err(LedgerError::NumericalOverflow)
        );
        assert!(matches!(
            evaluate_electrical_loads(&[], 1.0),
            Err(LedgerError::InvalidInput(_))
        ));
        assert!(matches!(
            allocate_loss_heat(&[]),
            Err(LedgerError::InvalidInput(_))
        ));
    }
}

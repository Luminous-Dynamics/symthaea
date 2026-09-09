// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scale-safe resource accounting across hierarchical infrastructure boundaries.
//!
//! Internal transfers disappear when viewed from their enclosing boundary, while
//! their physical transfer losses remain explicit. The same realized transfer may
//! therefore be a boundary export at one scale and an internal flow at a larger
//! scale without changing the underlying evidence.
//!
//! Node-local `modeled_losses` must contain local losses only. Loss between nodes
//! is represented exactly once by `RealizedTransfer::sent - delivered`.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_resource_hierarchy::ResourceHierarchy;
use symthaea_resource_model::{ResourceAmount, ResourceBalance, ResourceKey};
use thiserror::Error;

const CANCELLATION_EPSILON: f64 = 1e-9;

/// Resource balance attributed to one hierarchy node for an accounting window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeBalanceRecord {
    pub node_id: String,
    pub balance: ResourceBalance,
}

/// Realized inter-node transfer inside an accounting window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealizedTransfer {
    pub id: String,
    pub from_node: String,
    pub to_node: String,
    pub sent: ResourceAmount,
    pub delivered: ResourceAmount,
    pub occurred_at: DateTime<Utc>,
}

impl RealizedTransfer {
    pub fn validate(&self) -> Result<(), AccountingError> {
        validate_amount(self.sent)?;
        validate_amount(self.delivered)?;
        if self.from_node == self.to_node {
            return Err(AccountingError::SelfTransfer(self.id.clone()));
        }
        if self.sent.key != self.delivered.key {
            return Err(AccountingError::TransferDimensionMismatch(self.id.clone()));
        }
        if self.delivered.value > self.sent.value + CANCELLATION_EPSILON {
            return Err(AccountingError::DeliveredExceedsSent {
                id: self.id.clone(),
                sent: self.sent.value,
                delivered: self.delivered.value,
            });
        }
        Ok(())
    }

    pub fn loss(&self) -> f64 {
        (self.sent.value - self.delivered.value).max(0.0)
    }
}

/// Evidence for one closed accounting interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountingWindow {
    pub id: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub balances: Vec<NodeBalanceRecord>,
    pub transfers: Vec<RealizedTransfer>,
}

impl AccountingWindow {
    pub fn validate(&self, hierarchy: &ResourceHierarchy) -> Result<(), AccountingError> {
        if self.end <= self.start {
            return Err(AccountingError::InvalidWindow);
        }

        for record in &self.balances {
            if hierarchy.node(&record.node_id).is_none() {
                return Err(AccountingError::UnknownNode(record.node_id.clone()));
            }
            record
                .balance
                .validate()
                .map_err(|error| AccountingError::InvalidBalance {
                    node_id: record.node_id.clone(),
                    reason: error.to_string(),
                })?;
        }

        let mut transfer_ids = BTreeSet::new();
        for transfer in &self.transfers {
            transfer.validate()?;
            if !transfer_ids.insert(transfer.id.clone()) {
                return Err(AccountingError::DuplicateTransfer(transfer.id.clone()));
            }
            if hierarchy.node(&transfer.from_node).is_none() {
                return Err(AccountingError::UnknownNode(transfer.from_node.clone()));
            }
            if hierarchy.node(&transfer.to_node).is_none() {
                return Err(AccountingError::UnknownNode(transfer.to_node.clone()));
            }
            if transfer.occurred_at < self.start || transfer.occurred_at >= self.end {
                return Err(AccountingError::TransferOutsideWindow(transfer.id.clone()));
            }
        }
        Ok(())
    }

    /// Aggregate all node balances inside `node_id`'s subtree to that boundary.
    pub fn aggregate(
        &self,
        hierarchy: &ResourceHierarchy,
        node_id: &str,
    ) -> Result<ScaleAccountingReport, AccountingError> {
        self.validate(hierarchy)?;
        if hierarchy.node(node_id).is_none() {
            return Err(AccountingError::UnknownNode(node_id.to_owned()));
        }

        let mut subtree = BTreeSet::new();
        collect_subtree(hierarchy, node_id, &mut subtree)?;

        let mut totals: BTreeMap<ResourceKey, ResourceBalance> = BTreeMap::new();
        for record in self
            .balances
            .iter()
            .filter(|record| subtree.contains(&record.node_id))
        {
            let entry = totals.entry(record.balance.key).or_insert(ResourceBalance {
                key: record.balance.key,
                produced: 0.0,
                consumed: 0.0,
                imported: 0.0,
                exported: 0.0,
                storage_delta: 0.0,
                modeled_losses: 0.0,
            });
            entry.produced += record.balance.produced;
            entry.consumed += record.balance.consumed;
            entry.imported += record.balance.imported;
            entry.exported += record.balance.exported;
            entry.storage_delta += record.balance.storage_delta;
            entry.modeled_losses += record.balance.modeled_losses;
        }

        let mut internal_transfer_ids = Vec::new();
        let mut boundary_import_transfer_ids = Vec::new();
        let mut boundary_export_transfer_ids = Vec::new();

        for transfer in &self.transfers {
            let from_inside = subtree.contains(&transfer.from_node);
            let to_inside = subtree.contains(&transfer.to_node);
            match (from_inside, to_inside) {
                (true, true) => {
                    let entry = totals.entry(transfer.sent.key).or_insert(ResourceBalance {
                        key: transfer.sent.key,
                        produced: 0.0,
                        consumed: 0.0,
                        imported: 0.0,
                        exported: 0.0,
                        storage_delta: 0.0,
                        modeled_losses: 0.0,
                    });
                    cancel_internal_transfer(entry, transfer)?;
                    internal_transfer_ids.push(transfer.id.clone());
                }
                (false, true) => boundary_import_transfer_ids.push(transfer.id.clone()),
                (true, false) => boundary_export_transfer_ids.push(transfer.id.clone()),
                (false, false) => {}
            }
        }

        let mut balances: Vec<ResourceBalance> = totals.into_values().collect();
        for balance in &mut balances {
            normalize_near_zero(&mut balance.imported);
            normalize_near_zero(&mut balance.exported);
            balance
                .validate()
                .map_err(|error| AccountingError::InvalidAggregate(error.to_string()))?;
        }

        Ok(ScaleAccountingReport {
            window_id: self.id.clone(),
            node_id: node_id.to_owned(),
            subtree_nodes: subtree.into_iter().collect(),
            balances,
            internal_transfer_ids,
            boundary_import_transfer_ids,
            boundary_export_transfer_ids,
        })
    }
}

fn validate_amount(amount: ResourceAmount) -> Result<(), AccountingError> {
    if !amount.key.unit.supports(amount.key.kind) {
        return Err(AccountingError::InvalidAmountDimension);
    }
    if !amount.value.is_finite() || amount.value < 0.0 {
        return Err(AccountingError::InvalidAmountValue(amount.value));
    }
    Ok(())
}

fn collect_subtree(
    hierarchy: &ResourceHierarchy,
    node_id: &str,
    result: &mut BTreeSet<String>,
) -> Result<(), AccountingError> {
    if !result.insert(node_id.to_owned()) {
        return Err(AccountingError::ContainmentCycle(node_id.to_owned()));
    }
    let children: Vec<String> = hierarchy.children(node_id).map(str::to_owned).collect();
    for child in children {
        collect_subtree(hierarchy, &child, result)?;
    }
    Ok(())
}

fn cancel_internal_transfer(
    balance: &mut ResourceBalance,
    transfer: &RealizedTransfer,
) -> Result<(), AccountingError> {
    if balance.exported + CANCELLATION_EPSILON < transfer.sent.value {
        return Err(AccountingError::TransferExceedsRecordedExport {
            id: transfer.id.clone(),
            recorded: balance.exported,
            sent: transfer.sent.value,
        });
    }
    if balance.imported + CANCELLATION_EPSILON < transfer.delivered.value {
        return Err(AccountingError::TransferExceedsRecordedImport {
            id: transfer.id.clone(),
            recorded: balance.imported,
            delivered: transfer.delivered.value,
        });
    }
    balance.exported -= transfer.sent.value;
    balance.imported -= transfer.delivered.value;
    balance.modeled_losses += transfer.loss();
    Ok(())
}

fn normalize_near_zero(value: &mut f64) {
    if value.abs() <= CANCELLATION_EPSILON {
        *value = 0.0;
    }
}

/// Resource balance seen from one hierarchy boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleAccountingReport {
    pub window_id: String,
    pub node_id: String,
    pub subtree_nodes: Vec<String>,
    pub balances: Vec<ResourceBalance>,
    pub internal_transfer_ids: Vec<String>,
    pub boundary_import_transfer_ids: Vec<String>,
    pub boundary_export_transfer_ids: Vec<String>,
}

impl ScaleAccountingReport {
    pub fn balance(&self, key: ResourceKey) -> Option<&ResourceBalance> {
        self.balances.iter().find(|balance| balance.key == key)
    }

    pub fn residuals(&self) -> Result<Vec<(ResourceKey, f64)>, AccountingError> {
        self.balances
            .iter()
            .map(|balance| {
                balance
                    .residual()
                    .map(|residual| (balance.key, residual))
                    .map_err(|error| AccountingError::InvalidAggregate(error.to_string()))
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AccountingError {
    #[error("accounting window end must be after start")]
    InvalidWindow,
    #[error("unknown hierarchy node {0}")]
    UnknownNode(String),
    #[error("invalid balance for node {node_id}: {reason}")]
    InvalidBalance { node_id: String, reason: String },
    #[error("invalid aggregate balance: {0}")]
    InvalidAggregate(String),
    #[error("duplicate realized transfer {0}")]
    DuplicateTransfer(String),
    #[error("self-transfer is not allowed: {0}")]
    SelfTransfer(String),
    #[error("transfer {0} changes resource dimension")]
    TransferDimensionMismatch(String),
    #[error("transfer {id} delivered {delivered} > sent {sent}")]
    DeliveredExceedsSent {
        id: String,
        sent: f64,
        delivered: f64,
    },
    #[error("transfer {0} falls outside the accounting window")]
    TransferOutsideWindow(String),
    #[error("invalid resource amount dimension")]
    InvalidAmountDimension,
    #[error("invalid resource amount value {0}")]
    InvalidAmountValue(f64),
    #[error("containment cycle encountered at {0}")]
    ContainmentCycle(String),
    #[error("transfer {id} sent {sent} exceeds recorded aggregate export {recorded}")]
    TransferExceedsRecordedExport {
        id: String,
        recorded: f64,
        sent: f64,
    },
    #[error("transfer {id} delivered {delivered} exceeds recorded aggregate import {recorded}")]
    TransferExceedsRecordedImport {
        id: String,
        recorded: f64,
        delivered: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::{ResourceEnvelope, ResourceKind, ResourceUnit};

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn electricity(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Joule, value).unwrap()
    }

    fn key() -> ResourceKey {
        electricity(0.0).key
    }

    fn node(id: &str, scale: NodeScale) -> ResourceNode {
        ResourceNode::new(id, id, scale, ResourceEnvelope::default())
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(node("region", NodeScale::Region))
            .unwrap();
        hierarchy
            .insert_child("region", node("site-a", NodeScale::Site))
            .unwrap();
        hierarchy
            .insert_child("region", node("site-b", NodeScale::Site))
            .unwrap();
        hierarchy
            .insert_child("site-a", node("rack-a", NodeScale::Rack))
            .unwrap();
        hierarchy
            .insert_child("site-b", node("rack-b", NodeScale::Rack))
            .unwrap();
        hierarchy
    }

    fn balance(
        node_id: &str,
        produced: f64,
        consumed: f64,
        imported: f64,
        exported: f64,
        storage_delta: f64,
    ) -> NodeBalanceRecord {
        NodeBalanceRecord {
            node_id: node_id.into(),
            balance: ResourceBalance {
                key: key(),
                produced,
                consumed,
                imported,
                exported,
                storage_delta,
                modeled_losses: 0.0,
            },
        }
    }

    fn transfer(id: &str, from: &str, to: &str, sent: f64, delivered: f64) -> RealizedTransfer {
        RealizedTransfer {
            id: id.into(),
            from_node: from.into(),
            to_node: to.into(),
            sent: electricity(sent),
            delivered: electricity(delivered),
            occurred_at: t0() + Duration::minutes(10),
        }
    }

    #[test]
    fn internal_transfer_disappears_but_loss_remains() {
        let hierarchy = hierarchy();
        let window = AccountingWindow {
            id: "w1".into(),
            start: t0(),
            end: t0() + Duration::hours(1),
            balances: vec![
                balance("rack-a", 100.0, 40.0, 0.0, 60.0, 0.0),
                balance("rack-b", 0.0, 50.0, 57.0, 0.0, 7.0),
            ],
            transfers: vec![transfer("x", "rack-a", "rack-b", 60.0, 57.0)],
        };

        let report = window.aggregate(&hierarchy, "region").unwrap();
        let balance = report.balance(key()).unwrap();
        assert_eq!(balance.imported, 0.0);
        assert_eq!(balance.exported, 0.0);
        assert_eq!(balance.modeled_losses, 3.0);
        assert_eq!(balance.residual().unwrap(), 0.0);
        assert_eq!(report.internal_transfer_ids, vec!["x"]);
    }

    #[test]
    fn same_transfer_is_boundary_flow_at_smaller_scales() {
        let hierarchy = hierarchy();
        let window = AccountingWindow {
            id: "w1".into(),
            start: t0(),
            end: t0() + Duration::hours(1),
            balances: vec![
                balance("rack-a", 100.0, 40.0, 0.0, 60.0, 0.0),
                balance("rack-b", 0.0, 50.0, 57.0, 0.0, 7.0),
            ],
            transfers: vec![transfer("x", "rack-a", "rack-b", 60.0, 57.0)],
        };

        let site_a = window.aggregate(&hierarchy, "site-a").unwrap();
        assert_eq!(site_a.balance(key()).unwrap().exported, 60.0);
        assert_eq!(site_a.boundary_export_transfer_ids, vec!["x"]);

        let site_b = window.aggregate(&hierarchy, "site-b").unwrap();
        assert_eq!(site_b.balance(key()).unwrap().imported, 57.0);
        assert_eq!(site_b.boundary_import_transfer_ids, vec!["x"]);

        let region = window.aggregate(&hierarchy, "region").unwrap();
        assert_eq!(region.balance(key()).unwrap().exported, 0.0);
        assert_eq!(region.balance(key()).unwrap().imported, 0.0);
    }

    #[test]
    fn external_transfer_remains_visible_at_target_boundary() {
        let mut hierarchy = hierarchy();
        hierarchy
            .insert_child("region", node("grid", NodeScale::Site))
            .unwrap();
        let window = AccountingWindow {
            id: "w1".into(),
            start: t0(),
            end: t0() + Duration::hours(1),
            balances: vec![balance("rack-a", 0.0, 20.0, 20.0, 0.0, 0.0)],
            transfers: vec![transfer("grid-in", "grid", "rack-a", 21.0, 20.0)],
        };

        let report = window.aggregate(&hierarchy, "site-a").unwrap();
        assert_eq!(report.balance(key()).unwrap().imported, 20.0);
        assert_eq!(report.boundary_import_transfer_ids, vec!["grid-in"]);
    }

    #[test]
    fn transfer_outside_window_is_rejected() {
        let hierarchy = hierarchy();
        let mut late = transfer("late", "rack-a", "rack-b", 1.0, 1.0);
        late.occurred_at = t0() + Duration::hours(2);
        let window = AccountingWindow {
            id: "w1".into(),
            start: t0(),
            end: t0() + Duration::hours(1),
            balances: vec![],
            transfers: vec![late],
        };
        assert!(matches!(
            window.validate(&hierarchy),
            Err(AccountingError::TransferOutsideWindow(id)) if id == "late"
        ));
    }

    #[test]
    fn missing_recorded_boundary_flow_blocks_internal_cancellation() {
        let hierarchy = hierarchy();
        let window = AccountingWindow {
            id: "w1".into(),
            start: t0(),
            end: t0() + Duration::hours(1),
            balances: vec![
                balance("rack-a", 60.0, 60.0, 0.0, 0.0, 0.0),
                balance("rack-b", 0.0, 50.0, 57.0, 0.0, 7.0),
            ],
            transfers: vec![transfer("x", "rack-a", "rack-b", 60.0, 57.0)],
        };
        assert!(matches!(
            window.aggregate(&hierarchy, "region"),
            Err(AccountingError::TransferExceedsRecordedExport { .. })
        ));
    }

    #[test]
    fn report_surfaces_nonzero_measurement_residual_instead_of_hiding_it() {
        let hierarchy = hierarchy();
        let window = AccountingWindow {
            id: "w1".into(),
            start: t0(),
            end: t0() + Duration::hours(1),
            balances: vec![balance("rack-a", 10.0, 9.0, 0.0, 0.0, 0.0)],
            transfers: vec![],
        };
        let report = window.aggregate(&hierarchy, "site-a").unwrap();
        assert_eq!(report.residuals().unwrap(), vec![(key(), 1.0)]);
    }
}

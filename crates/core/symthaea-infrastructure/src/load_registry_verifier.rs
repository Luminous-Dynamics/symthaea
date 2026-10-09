// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent verification of per-load service ledgers.
//!
//! This checker deliberately does not call the registry allocator, the
//! registry convenience verifier, or the ledger's own validity method. It
//! independently recomputes dispatch ordering, per-load energy allocation,
//! and summary totals from the exact registry and demand inputs.
//!
//! It checks consistency with the supplied registry; it does not authenticate
//! the registry, its provenance labels, the energy measurement, or plant state.

use std::collections::BTreeMap;

use crate::load_registry::{LoadClass, LoadRegistry, LoadServiceLedger};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadLedgerVerificationError {
    RegistryVersionMismatch,
    RegistryDigestMismatch,
    InvalidTimeBasis,
    InvalidAvailableEnergy,
    RegistryRecordCountMismatch,
    InvalidLoadId,
    DuplicateLoadId(String),
    UnknownLoadId(String),
    MetadataMismatch(String),
    InvalidDemand(String),
    DemandExceedsRating(String),
    InvalidEnergyValue(String),
    DemandEnergyMismatch(String),
    AllocationMismatch(String),
    SummaryMismatch,
    EnergyBudgetMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedLoadServiceLedger {
    pub registry_version: String,
    pub registry_digest: String,
    pub record_count: usize,
    pub recomputed_demand_kwh: f64,
    pub recomputed_served_kwh: f64,
    pub recomputed_intentional_shed_kwh: f64,
    pub recomputed_unserved_kwh: f64,
    pub recomputed_unused_energy_kwh: f64,
    pub step_duration_hours: f64,
}

impl VerifiedLoadServiceLedger {
    pub fn residual_kwh(&self) -> f64 {
        self.recomputed_demand_kwh
            - self.recomputed_served_kwh
            - self.recomputed_intentional_shed_kwh
            - self.recomputed_unserved_kwh
    }
}

/// Verify one ledger against an exact, caller-selected registry.
///
/// The checker performs a separate dispatch calculation. It checks identity,
/// declared class, provenance labels, rated power, shedding/restoration
/// priorities, demand-to-energy conversion, allocation order, each record's
/// deficit classification, and aggregate energy totals. A successful result
/// establishes consistency among supplied values only, not factual truth.
pub fn verify_load_service_ledger(
    registry: &LoadRegistry,
    ledger: &LoadServiceLedger,
) -> Result<VerifiedLoadServiceLedger, LoadLedgerVerificationError> {
    if ledger.registry_version != registry.version() {
        return Err(LoadLedgerVerificationError::RegistryVersionMismatch);
    }
    if ledger.registry_digest != registry.registry_digest() {
        return Err(LoadLedgerVerificationError::RegistryDigestMismatch);
    }
    if !ledger.step_duration_hours.is_finite() || ledger.step_duration_hours <= 0.0 {
        return Err(LoadLedgerVerificationError::InvalidTimeBasis);
    }
    if !ledger.available_energy_kwh.is_finite() || ledger.available_energy_kwh < 0.0 {
        return Err(LoadLedgerVerificationError::InvalidAvailableEnergy);
    }

    let entries: BTreeMap<&str, _> = registry
        .entries()
        .iter()
        .map(|entry| (entry.load_id.as_str(), entry))
        .collect();
    if ledger.records.len() != entries.len() {
        return Err(LoadLedgerVerificationError::RegistryRecordCountMismatch);
    }

    let mut records: BTreeMap<&str, _> = BTreeMap::new();
    for record in &ledger.records {
        if !well_formed_label(&record.load_id) {
            return Err(LoadLedgerVerificationError::InvalidLoadId);
        }
        if records.insert(record.load_id.as_str(), record).is_some() {
            return Err(LoadLedgerVerificationError::DuplicateLoadId(
                record.load_id.clone(),
            ));
        }
        let Some(entry) = entries.get(record.load_id.as_str()) else {
            return Err(LoadLedgerVerificationError::UnknownLoadId(
                record.load_id.clone(),
            ));
        };
        if record.class != entry.class
            || record.provenance != entry.provenance
            || record.rated_power_kw != entry.rated_power_kw
            || record.shed_priority != entry.shed_priority
            || record.restore_priority != entry.restore_priority
        {
            return Err(LoadLedgerVerificationError::MetadataMismatch(
                record.load_id.clone(),
            ));
        }

        if !record.requested_power_kw.is_finite() || record.requested_power_kw < 0.0 {
            return Err(LoadLedgerVerificationError::InvalidDemand(
                record.load_id.clone(),
            ));
        }
        if record.requested_power_kw > entry.rated_power_kw {
            return Err(LoadLedgerVerificationError::DemandExceedsRating(
                record.load_id.clone(),
            ));
        }
        let values = [
            record.requested_kwh,
            record.served_kwh,
            record.intentional_shed_kwh,
            record.unserved_kwh,
        ];
        if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(LoadLedgerVerificationError::InvalidEnergyValue(
                record.load_id.clone(),
            ));
        }

        let independently_computed_demand_kwh =
            record.requested_power_kw * ledger.step_duration_hours;
        if !close(independently_computed_demand_kwh, record.requested_kwh) {
            return Err(LoadLedgerVerificationError::DemandEnergyMismatch(
                record.load_id.clone(),
            ));
        }
        if !close(
            record.requested_kwh,
            record.served_kwh + record.intentional_shed_kwh + record.unserved_kwh,
        ) {
            return Err(LoadLedgerVerificationError::EnergyBudgetMismatch);
        }

        if is_critical(entry.class) {
            if !close(record.intentional_shed_kwh, 0.0) {
                return Err(LoadLedgerVerificationError::AllocationMismatch(
                    record.load_id.clone(),
                ));
            }
        } else if !close(record.unserved_kwh, 0.0) {
            return Err(LoadLedgerVerificationError::AllocationMismatch(
                record.load_id.clone(),
            ));
        }
    }

    for entry in registry.entries() {
        if !records.contains_key(entry.load_id.as_str()) {
            return Err(LoadLedgerVerificationError::UnknownLoadId(
                entry.load_id.clone(),
            ));
        }
    }

    // Independent canonical ordering: class tier, descending retention
    // priority, and ID. This intentionally does not call allocator helpers.
    let mut canonical: Vec<_> = registry.entries().iter().collect();
    canonical.sort_by(|a, b| {
        independent_class_tier(a.class)
            .cmp(&independent_class_tier(b.class))
            .then_with(|| b.shed_priority.cmp(&a.shed_priority))
            .then_with(|| a.load_id.cmp(&b.load_id))
    });

    let mut remaining_supply_kwh = ledger.available_energy_kwh;
    for entry in canonical {
        let Some(record) = records.get(entry.load_id.as_str()) else {
            return Err(LoadLedgerVerificationError::UnknownLoadId(
                entry.load_id.clone(),
            ));
        };
        let expected_demand = record.requested_power_kw * ledger.step_duration_hours;
        let expected_served = expected_demand.min(remaining_supply_kwh).max(0.0);
        remaining_supply_kwh = (remaining_supply_kwh - expected_served).max(0.0);
        let expected_deficit = (expected_demand - expected_served).max(0.0);
        let expected_shed = if is_critical(entry.class) {
            0.0
        } else {
            expected_deficit
        };
        let expected_unserved = if is_critical(entry.class) {
            expected_deficit
        } else {
            0.0
        };

        if !close(record.requested_kwh, expected_demand)
            || !close(record.served_kwh, expected_served)
            || !close(record.intentional_shed_kwh, expected_shed)
            || !close(record.unserved_kwh, expected_unserved)
        {
            return Err(LoadLedgerVerificationError::AllocationMismatch(
                entry.load_id.clone(),
            ));
        }
    }

    let recomputed_demand = ledger
        .records
        .iter()
        .map(|record| record.requested_power_kw * ledger.step_duration_hours)
        .sum::<f64>();
    let recomputed_served = ledger.records.iter().map(|record| record.served_kwh).sum::<f64>();
    let recomputed_shed = ledger
        .records
        .iter()
        .map(|record| record.intentional_shed_kwh)
        .sum::<f64>();
    let recomputed_unserved = ledger.records.iter().map(|record| record.unserved_kwh).sum::<f64>();
    let recomputed_unused = (ledger.available_energy_kwh - recomputed_served).max(0.0);

    let summaries = [
        ledger.total_demand_kwh,
        ledger.total_served_kwh,
        ledger.intentional_shed_kwh,
        ledger.total_unserved_kwh,
        ledger.unused_energy_kwh,
    ];
    if summaries
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !close(ledger.total_demand_kwh, recomputed_demand)
        || !close(ledger.total_served_kwh, recomputed_served)
        || !close(ledger.intentional_shed_kwh, recomputed_shed)
        || !close(ledger.total_unserved_kwh, recomputed_unserved)
    {
        return Err(LoadLedgerVerificationError::SummaryMismatch);
    }
    if !close(ledger.unused_energy_kwh, recomputed_unused)
        || !close(recomputed_served + recomputed_unused, ledger.available_energy_kwh)
        || recomputed_served > ledger.available_energy_kwh + tolerance(ledger.available_energy_kwh)
        || !close(
            recomputed_demand,
            recomputed_served + recomputed_shed + recomputed_unserved,
        )
    {
        return Err(LoadLedgerVerificationError::EnergyBudgetMismatch);
    }

    Ok(VerifiedLoadServiceLedger {
        registry_version: registry.version().to_string(),
        registry_digest: registry.registry_digest().to_string(),
        record_count: ledger.records.len(),
        recomputed_demand_kwh: recomputed_demand,
        recomputed_served_kwh: recomputed_served,
        recomputed_intentional_shed_kwh: recomputed_shed,
        recomputed_unserved_kwh: recomputed_unserved,
        recomputed_unused_energy_kwh: recomputed_unused,
        step_duration_hours: ledger.step_duration_hours,
    })
}

fn well_formed_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn is_critical(class: LoadClass) -> bool {
    matches!(class, LoadClass::Critical)
}

fn independent_class_tier(class: LoadClass) -> u8 {
    match class {
        LoadClass::Critical => 10,
        LoadClass::Deferrable => 20,
        LoadClass::Auxiliary => 30,
    }
}

fn tolerance(value: f64) -> f64 {
    1e-9 * value.abs().max(1.0)
}

fn close(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= tolerance(left.abs().max(right.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_registry::{ClassificationProvenance, LoadDemand, LoadRegistryEntry};

    fn provenance() -> ClassificationProvenance {
        ClassificationProvenance::SyntheticScenario {
            scenario_id: "independent-verifier-fixture-v1".to_string(),
        }
    }

    fn registry() -> LoadRegistry {
        LoadRegistry::new(
            "independent-verifier-registry-v1",
            vec![
                LoadRegistryEntry {
                    load_id: "critical-water".into(),
                    class: LoadClass::Critical,
                    rated_power_kw: 2.0,
                    shed_priority: 10,
                    restore_priority: 2,
                    provenance: provenance(),
                },
                LoadRegistryEntry {
                    load_id: "deferrable-comms".into(),
                    class: LoadClass::Deferrable,
                    rated_power_kw: 1.0,
                    shed_priority: 10,
                    restore_priority: 1,
                    provenance: provenance(),
                },
                LoadRegistryEntry {
                    load_id: "aux-lighting".into(),
                    class: LoadClass::Auxiliary,
                    rated_power_kw: 2.0,
                    shed_priority: 0,
                    restore_priority: 3,
                    provenance: provenance(),
                },
            ],
        )
        .unwrap()
    }

    fn demands() -> Vec<LoadDemand> {
        vec![
            LoadDemand {
                load_id: "critical-water".into(),
                requested_power_kw: 2.0,
            },
            LoadDemand {
                load_id: "deferrable-comms".into(),
                requested_power_kw: 1.0,
            },
            LoadDemand {
                load_id: "aux-lighting".into(),
                requested_power_kw: 2.0,
            },
        ]
    }

    #[test]
    fn accepts_canonical_ledger_and_reports_independently_recomputed_totals() {
        let registry = registry();
        let ledger = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        let verified = verify_load_service_ledger(&registry, &ledger).unwrap();

        assert_eq!(verified.registry_version, "independent-verifier-registry-v1");
        assert_eq!(verified.registry_digest, registry.registry_digest());
        assert_eq!(verified.record_count, 3);
        assert_eq!(verified.recomputed_demand_kwh, 2.5);
        assert_eq!(verified.recomputed_served_kwh, 1.25);
        assert_eq!(verified.recomputed_intentional_shed_kwh, 1.25);
        assert_eq!(verified.recomputed_unserved_kwh, 0.0);
        assert!(verified.residual_kwh().abs() < 1e-12);
    }

    #[test]
    fn rejects_ledger_with_mismatched_registry_digest() {
        let registry = registry();
        let mut ledger = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        ledger.registry_digest = "0".repeat(64);

        assert_eq!(
            verify_load_service_ledger(&registry, &ledger).unwrap_err(),
            LoadLedgerVerificationError::RegistryDigestMismatch
        );
    }

    #[test]
    fn rejects_forged_aggregate_even_when_record_values_are_unchanged() {
        let registry = registry();
        let mut ledger = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        ledger.total_served_kwh += 0.25;

        assert_eq!(
            verify_load_service_ledger(&registry, &ledger).unwrap_err(),
            LoadLedgerVerificationError::SummaryMismatch
        );
    }

    #[test]
    fn rejects_priority_violation_even_when_energy_totals_still_balance() {
        let registry = registry();
        let mut ledger = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        let critical = ledger
            .records
            .iter_mut()
            .find(|record| record.load_id == "critical-water")
            .unwrap();
        // Move 0.25 kWh away from critical load and to a lower-tier load;
        // preserve served sum and per-record demand balance.
        critical.served_kwh -= 0.25;
        critical.unserved_kwh += 0.25;
        let auxiliary = ledger
            .records
            .iter_mut()
            .find(|record| record.load_id == "aux-lighting")
            .unwrap();
        auxiliary.served_kwh += 0.25;
        auxiliary.intentional_shed_kwh -= 0.25;
        ledger.total_unserved_kwh += 0.25;
        ledger.total_intentional_shed_kwh -= 0.25;

        assert_eq!(
            verify_load_service_ledger(&registry, &ledger).unwrap_err(),
            LoadLedgerVerificationError::AllocationMismatch("critical-water".into())
        );
    }

    #[test]
    fn rejects_classification_provenance_priority_and_rating_tampering() {
        let registry = registry();

        let mut class_changed = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        class_changed
            .records
            .iter_mut()
            .find(|record| record.load_id == "deferrable-comms")
            .unwrap()
            .class = LoadClass::Critical;
        assert_eq!(
            verify_load_service_ledger(&registry, &class_changed).unwrap_err(),
            LoadLedgerVerificationError::MetadataMismatch("deferrable-comms".into())
        );

        let mut provenance_changed = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        provenance_changed
            .records
            .iter_mut()
            .find(|record| record.load_id == "deferrable-comms")
            .unwrap()
            .provenance = ClassificationProvenance::ReviewedRecord {
                record_id: "review-99".into(),
                record_revision: "r2".into(),
                reviewer_id: "reviewer-a".into(),
            };
        assert_eq!(
            verify_load_service_ledger(&registry, &provenance_changed).unwrap_err(),
            LoadLedgerVerificationError::MetadataMismatch("deferrable-comms".into())
        );

        let mut priority_changed = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        priority_changed
            .records
            .iter_mut()
            .find(|record| record.load_id == "deferrable-comms")
            .unwrap()
            .shed_priority += 1;
        assert_eq!(
            verify_load_service_ledger(&registry, &priority_changed).unwrap_err(),
            LoadLedgerVerificationError::MetadataMismatch("deferrable-comms".into())
        );

        let mut rating_changed = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        rating_changed
            .records
            .iter_mut()
            .find(|record| record.load_id == "deferrable-comms")
            .unwrap()
            .rated_power_kw += 0.5;
        assert_eq!(
            verify_load_service_ledger(&registry, &rating_changed).unwrap_err(),
            LoadLedgerVerificationError::MetadataMismatch("deferrable-comms".into())
        );
    }

    #[test]
    fn rejects_duplicate_missing_unknown_and_unregistered_records() {
        let registry = registry();
        let original = registry.allocate(&demands(), 0.5, 1.25).unwrap();

        let mut duplicate = original.clone();
        duplicate.records[0].load_id = duplicate.records[1].load_id.clone();
        assert!(matches!(
            verify_load_service_ledger(&registry, &duplicate),
            Err(LoadLedgerVerificationError::DuplicateLoadId(_))
        ));

        let mut missing = original.clone();
        missing.records.pop();
        assert_eq!(
            verify_load_service_ledger(&registry, &missing).unwrap_err(),
            LoadLedgerVerificationError::RegistryRecordCountMismatch
        );

        let mut unknown = original;
        unknown.records[0].load_id = "outside-registry".into();
        assert!(matches!(
            verify_load_service_ledger(&registry, &unknown),
            Err(LoadLedgerVerificationError::UnknownLoadId(_))
        ));
    }

    #[test]
    fn rejects_invalid_time_and_supply_basis() {
        let registry = registry();
        let mut invalid = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        invalid.step_duration_hours = 0.0;
        assert_eq!(
            verify_load_service_ledger(&registry, &invalid).unwrap_err(),
            LoadLedgerVerificationError::InvalidTimeBasis
        );

        let mut invalid = registry.allocate(&demands(), 0.5, 1.25).unwrap();
        invalid.available_energy_kwh = f64::NAN;
        assert_eq!(
            verify_load_service_ledger(&registry, &invalid).unwrap_err(),
            LoadLedgerVerificationError::InvalidAvailableEnergy
        );
    }
}

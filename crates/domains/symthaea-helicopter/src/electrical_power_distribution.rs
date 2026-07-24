// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic electrical-power distribution and load-shedding evidence.
//!
//! This is a bounded power-allocation model, not a circuit simulator. It makes
//! source capacity, bus faults, load priority, brownout, and essential-load
//! continuity explicit so degraded operation cannot assume unlimited power.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ElectricalBus {
    EssentialA,
    EssentialB,
    Mission,
    Payload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ElectricalLoadPriority {
    FlightCritical,
    Essential,
    Mission,
    Payload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalSourceState {
    pub source_id: String,
    pub failure_domain: String,
    pub available: bool,
    pub maximum_power_w: f64,
    pub voltage_v: f64,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalLoadDemand {
    pub load_id: String,
    pub priority: ElectricalLoadPriority,
    pub demanded_power_w: f64,
    pub minimum_voltage_v: f64,
    pub permitted_buses: Vec<ElectricalBus>,
    pub required_for_safe_flight: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalDistributionInput {
    pub timestamp_ms: u64,
    pub sources: Vec<ElectricalSourceState>,
    pub loads: Vec<ElectricalLoadDemand>,
    pub bus_available: BTreeMap<ElectricalBus, bool>,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElectricalLoadDisposition {
    Powered,
    Shed,
    Brownout,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalLoadResult {
    pub load_id: String,
    pub disposition: ElectricalLoadDisposition,
    pub allocated_power_w: f64,
    pub assigned_bus: Option<ElectricalBus>,
    pub source_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElectricalDistributionStatus {
    Nominal,
    Degraded,
    Emergency,
    Blackout,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElectricalDistributionIssue {
    DuplicateSource(String),
    DuplicateLoad(String),
    InvalidSource(String),
    InvalidLoad(String),
    MissingEvidence(String),
    MissingBusState(ElectricalBus),
    NoIndependentPowerDomains,
    LoadShed(String),
    LoadBrownout(String),
    SafeFlightLoadUnavailable(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalDistributionReport {
    pub timestamp_ms: u64,
    pub status: ElectricalDistributionStatus,
    pub available_power_w: f64,
    pub allocated_power_w: f64,
    pub remaining_power_w: f64,
    pub independent_power_domains: usize,
    pub loads: Vec<ElectricalLoadResult>,
    pub issues: Vec<ElectricalDistributionIssue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectricalDistributionPolicy {
    pub nominal_voltage_v: f64,
    pub brownout_voltage_v: f64,
    pub minimum_independent_power_domains: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElectricalDistributionError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct ElectricalPowerDistributor {
    policy: ElectricalDistributionPolicy,
}

impl ElectricalPowerDistributor {
    pub fn new(policy: ElectricalDistributionPolicy) -> Result<Self, ElectricalDistributionError> {
        if !policy.nominal_voltage_v.is_finite()
            || !policy.brownout_voltage_v.is_finite()
            || policy.nominal_voltage_v <= 0.0
            || policy.brownout_voltage_v <= 0.0
            || policy.brownout_voltage_v >= policy.nominal_voltage_v
            || policy.minimum_independent_power_domains == 0
        {
            return Err(ElectricalDistributionError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn distribute(&self, input: &ElectricalDistributionInput) -> ElectricalDistributionReport {
        let mut issues = Vec::new();
        let mut source_ids = BTreeSet::new();
        let mut load_ids = BTreeSet::new();
        let mut domains = BTreeSet::new();
        let mut usable_sources = Vec::new();

        if input.evidence_ids.is_empty() || input.evidence_ids.iter().any(|id| id.trim().is_empty())
        {
            issues.push(ElectricalDistributionIssue::MissingEvidence(
                "distribution-input".into(),
            ));
        }
        for bus in [
            ElectricalBus::EssentialA,
            ElectricalBus::EssentialB,
            ElectricalBus::Mission,
            ElectricalBus::Payload,
        ] {
            if !input.bus_available.contains_key(&bus) {
                issues.push(ElectricalDistributionIssue::MissingBusState(bus));
            }
        }
        for source in &input.sources {
            if source.source_id.trim().is_empty() || !source_ids.insert(source.source_id.clone()) {
                issues.push(ElectricalDistributionIssue::DuplicateSource(
                    source.source_id.clone(),
                ));
            }
            if source.failure_domain.trim().is_empty()
                || !source.maximum_power_w.is_finite()
                || source.maximum_power_w < 0.0
                || !source.voltage_v.is_finite()
                || source.voltage_v < 0.0
            {
                issues.push(ElectricalDistributionIssue::InvalidSource(
                    source.source_id.clone(),
                ));
                continue;
            }
            if source.evidence_ids.is_empty()
                || source.evidence_ids.iter().any(|id| id.trim().is_empty())
            {
                issues.push(ElectricalDistributionIssue::MissingEvidence(
                    source.source_id.clone(),
                ));
            }
            if source.available && source.maximum_power_w > 0.0 {
                domains.insert(source.failure_domain.clone());
                usable_sources.push(source);
            }
        }
        if domains.len() < self.policy.minimum_independent_power_domains {
            issues.push(ElectricalDistributionIssue::NoIndependentPowerDomains);
        }

        let available_power_w: f64 = usable_sources
            .iter()
            .map(|source| source.maximum_power_w)
            .sum();
        let minimum_available_voltage = usable_sources
            .iter()
            .map(|source| source.voltage_v)
            .reduce(f64::min)
            .unwrap_or(0.0);
        let available_source_ids: Vec<_> = usable_sources
            .iter()
            .map(|source| source.source_id.clone())
            .collect();
        let mut remaining_power = available_power_w;
        let mut sorted_loads: Vec<_> = input.loads.iter().collect();
        sorted_loads.sort_by(|left, right| {
            left.priority
                .cmp(&right.priority)
                .then_with(|| left.load_id.cmp(&right.load_id))
        });
        let mut results = Vec::new();

        for load in sorted_loads {
            if load.load_id.trim().is_empty() || !load_ids.insert(load.load_id.clone()) {
                issues.push(ElectricalDistributionIssue::DuplicateLoad(
                    load.load_id.clone(),
                ));
            }
            let valid = load.demanded_power_w.is_finite()
                && load.demanded_power_w >= 0.0
                && load.minimum_voltage_v.is_finite()
                && load.minimum_voltage_v > 0.0
                && !load.permitted_buses.is_empty();
            if !valid {
                issues.push(ElectricalDistributionIssue::InvalidLoad(
                    load.load_id.clone(),
                ));
                results.push(ElectricalLoadResult {
                    load_id: load.load_id.clone(),
                    disposition: ElectricalLoadDisposition::Unavailable,
                    allocated_power_w: 0.0,
                    assigned_bus: None,
                    source_ids: Vec::new(),
                });
                continue;
            }
            let assigned_bus = load
                .permitted_buses
                .iter()
                .copied()
                .find(|bus| input.bus_available.get(bus).copied().unwrap_or(false));
            let (disposition, allocated_power_w) =
                if assigned_bus.is_none() || usable_sources.is_empty() {
                    (ElectricalLoadDisposition::Unavailable, 0.0)
                } else if minimum_available_voltage < self.policy.brownout_voltage_v
                    || minimum_available_voltage < load.minimum_voltage_v
                {
                    let allocated = remaining_power.min(load.demanded_power_w);
                    remaining_power -= allocated;
                    (ElectricalLoadDisposition::Brownout, allocated)
                } else if remaining_power + f64::EPSILON >= load.demanded_power_w {
                    remaining_power -= load.demanded_power_w;
                    (ElectricalLoadDisposition::Powered, load.demanded_power_w)
                } else {
                    (ElectricalLoadDisposition::Shed, 0.0)
                };
            match disposition {
                ElectricalLoadDisposition::Shed => {
                    issues.push(ElectricalDistributionIssue::LoadShed(load.load_id.clone()));
                }
                ElectricalLoadDisposition::Brownout => {
                    issues.push(ElectricalDistributionIssue::LoadBrownout(
                        load.load_id.clone(),
                    ));
                }
                ElectricalLoadDisposition::Unavailable => {}
                ElectricalLoadDisposition::Powered => {}
            }
            if load.required_for_safe_flight && disposition != ElectricalLoadDisposition::Powered {
                issues.push(ElectricalDistributionIssue::SafeFlightLoadUnavailable(
                    load.load_id.clone(),
                ));
            }
            results.push(ElectricalLoadResult {
                load_id: load.load_id.clone(),
                disposition,
                allocated_power_w,
                assigned_bus,
                source_ids: if disposition == ElectricalLoadDisposition::Unavailable {
                    Vec::new()
                } else {
                    available_source_ids.clone()
                },
            });
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                ElectricalDistributionIssue::DuplicateSource(_)
                    | ElectricalDistributionIssue::DuplicateLoad(_)
                    | ElectricalDistributionIssue::InvalidSource(_)
                    | ElectricalDistributionIssue::InvalidLoad(_)
                    | ElectricalDistributionIssue::MissingEvidence(_)
                    | ElectricalDistributionIssue::MissingBusState(_)
            )
        });
        let safe_flight_lost = issues.iter().any(|issue| {
            matches!(
                issue,
                ElectricalDistributionIssue::SafeFlightLoadUnavailable(_)
            )
        });
        let any_powered = results
            .iter()
            .any(|result| result.disposition == ElectricalLoadDisposition::Powered);
        let degraded = issues.iter().any(|issue| {
            matches!(
                issue,
                ElectricalDistributionIssue::NoIndependentPowerDomains
                    | ElectricalDistributionIssue::LoadShed(_)
                    | ElectricalDistributionIssue::LoadBrownout(_)
            )
        });
        let status = if incomplete {
            ElectricalDistributionStatus::Incomplete
        } else if available_power_w <= f64::EPSILON || !any_powered {
            ElectricalDistributionStatus::Blackout
        } else if safe_flight_lost {
            ElectricalDistributionStatus::Emergency
        } else if degraded {
            ElectricalDistributionStatus::Degraded
        } else {
            ElectricalDistributionStatus::Nominal
        };

        ElectricalDistributionReport {
            timestamp_ms: input.timestamp_ms,
            status,
            available_power_w,
            allocated_power_w: available_power_w - remaining_power,
            remaining_power_w: remaining_power,
            independent_power_domains: domains.len(),
            loads: results,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distributor() -> ElectricalPowerDistributor {
        ElectricalPowerDistributor::new(ElectricalDistributionPolicy {
            nominal_voltage_v: 28.0,
            brownout_voltage_v: 22.0,
            minimum_independent_power_domains: 2,
        })
        .unwrap()
    }

    fn source(id: &str, domain: &str, power: f64) -> ElectricalSourceState {
        ElectricalSourceState {
            source_id: id.into(),
            failure_domain: domain.into(),
            available: true,
            maximum_power_w: power,
            voltage_v: 28.0,
            evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    fn load(
        id: &str,
        priority: ElectricalLoadPriority,
        power: f64,
        critical: bool,
    ) -> ElectricalLoadDemand {
        ElectricalLoadDemand {
            load_id: id.into(),
            priority,
            demanded_power_w: power,
            minimum_voltage_v: 20.0,
            permitted_buses: vec![ElectricalBus::EssentialA, ElectricalBus::EssentialB],
            required_for_safe_flight: critical,
        }
    }

    fn input() -> ElectricalDistributionInput {
        ElectricalDistributionInput {
            timestamp_ms: 1_000,
            sources: vec![
                source("gen", "engine", 800.0),
                source("battery", "battery", 400.0),
            ],
            loads: vec![
                load(
                    "flight-computer",
                    ElectricalLoadPriority::FlightCritical,
                    300.0,
                    true,
                ),
                load("payload", ElectricalLoadPriority::Payload, 500.0, false),
            ],
            bus_available: BTreeMap::from([
                (ElectricalBus::EssentialA, true),
                (ElectricalBus::EssentialB, true),
                (ElectricalBus::Mission, true),
                (ElectricalBus::Payload, true),
            ]),
            evidence_ids: vec!["distribution-evidence".into()],
        }
    }

    #[test]
    fn critical_loads_are_allocated_before_payload() {
        let mut input = input();
        input.sources[0].maximum_power_w = 200.0;
        input.sources[1].maximum_power_w = 200.0;
        let report = distributor().distribute(&input);
        assert_eq!(report.status, ElectricalDistributionStatus::Degraded);
        assert_eq!(report.loads[0].load_id, "flight-computer");
        assert_eq!(
            report.loads[0].disposition,
            ElectricalLoadDisposition::Powered
        );
        assert_eq!(report.loads[1].disposition, ElectricalLoadDisposition::Shed);
    }

    #[test]
    fn loss_of_critical_load_is_emergency() {
        let mut input = input();
        input.sources.clear();
        let report = distributor().distribute(&input);
        assert_eq!(report.status, ElectricalDistributionStatus::Blackout);
        assert!(
            report.issues.iter().any(|issue| matches!(issue,
            ElectricalDistributionIssue::SafeFlightLoadUnavailable(id) if id == "flight-computer"))
        );
    }

    #[test]
    fn single_domain_is_degraded_even_with_capacity() {
        let mut input = input();
        input.sources[1].failure_domain = "engine".into();
        let report = distributor().distribute(&input);
        assert_eq!(report.status, ElectricalDistributionStatus::Degraded);
    }

    #[test]
    fn missing_bus_state_is_incomplete() {
        let mut input = input();
        input.bus_available.remove(&ElectricalBus::Payload);
        let report = distributor().distribute(&input);
        assert_eq!(report.status, ElectricalDistributionStatus::Incomplete);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Time-scoped resource allocation over potential topology and explicit capacity.
//!
//! This crate is intentionally a planner-side conservation layer:
//!
//! `ResourceTopology -> CapacitySchedule -> PlannedAllocation -> AdmittedAllocation`
//!
//! It does not choose workloads, optimize objectives, create Mycelix leases, or
//! execute effects. Admission means only that one proposed transfer fits the exact
//! topology and temporal capacity schedule presented to this allocation book.
//!
//! Two capacity theorems are implemented without conflating them:
//!
//! - `Concurrent`: exact half-open interval sweeping enforces peak simultaneous
//!   utilization. Disjoint reservations are not falsely summed together.
//! - `WindowBudget`: every allocation bound to the same capacity window consumes
//!   from one cumulative budget even when allocations are temporally disjoint.
//!
//! Source ports and topology links are charged against sent quantity. Destination
//! ports are charged against delivered quantity after the link's explicit loss.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_resource_capacity::{
    CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
};
use symthaea_resource_model::ResourceAmount;
use symthaea_resource_topology::{ResourceLink, ResourceTopology};
use thiserror::Error;

const ALLOCATION_EPSILON: f64 = 1e-9;

/// One proposed transfer along an exact potential topology link.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedAllocation {
    pub id: String,
    pub link_id: String,
    pub valid_from: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
    /// Sent quantity or rate in the topology link's exact resource dimension.
    pub sent: ResourceAmount,
}

impl PlannedAllocation {
    pub fn validate_structure(&self) -> Result<(), AllocationError> {
        if self.id.trim().is_empty() {
            return Err(AllocationError::EmptyAllocationId);
        }
        if self.link_id.trim().is_empty() {
            return Err(AllocationError::EmptyLinkId);
        }
        if self.valid_until <= self.valid_from {
            return Err(AllocationError::InvalidValidityWindow);
        }
        if !self.sent.value.is_finite() || self.sent.value <= 0.0 {
            return Err(AllocationError::InvalidAmount(self.sent.value));
        }
        Ok(())
    }

    pub fn delivered(&self, topology: &ResourceTopology) -> Result<ResourceAmount, AllocationError> {
        let link = topology
            .link(&self.link_id)
            .ok_or_else(|| AllocationError::UnknownLink(self.link_id.clone()))?;
        delivered_amount(self.sent, link)
    }
}

/// A plan that has passed topology and temporal-capacity conservation checks.
///
/// The exact capacity windows used during validation are retained by value so a
/// later reviewer can see which temporal interpretation supported the plan. This
/// type is intentionally not deserializable and is not execution authority.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedAllocation {
    allocation: PlannedAllocation,
    source_capacity: CapacityWindow,
    link_capacity: CapacityWindow,
    destination_capacity: CapacityWindow,
}

impl AdmittedAllocation {
    pub fn allocation(&self) -> &PlannedAllocation {
        &self.allocation
    }

    pub fn source_capacity(&self) -> &CapacityWindow {
        &self.source_capacity
    }

    pub fn link_capacity(&self) -> &CapacityWindow {
        &self.link_capacity
    }

    pub fn destination_capacity(&self) -> &CapacityWindow {
        &self.destination_capacity
    }

    fn binding_for(&self, subject: &CapacitySubject) -> Option<&CapacityWindow> {
        [
            &self.source_capacity,
            &self.link_capacity,
            &self.destination_capacity,
        ]
        .into_iter()
        .find(|window| &window.subject == subject)
    }
}

/// Mutable planning book of allocations already admitted under one explicit model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AllocationBook {
    allocations: BTreeMap<String, AdmittedAllocation>,
}

impl AllocationBook {
    pub fn admit(
        &mut self,
        topology: &ResourceTopology,
        capacities: &CapacitySchedule,
        allocation: PlannedAllocation,
    ) -> Result<(), AllocationError> {
        allocation.validate_structure()?;
        if self.allocations.contains_key(&allocation.id) {
            return Err(AllocationError::DuplicateAllocation(allocation.id));
        }

        let link = topology
            .link(&allocation.link_id)
            .ok_or_else(|| AllocationError::UnknownLink(allocation.link_id.clone()))?;
        validate_against_link(&allocation, link)?;
        let delivered = delivered_amount(allocation.sent, link)?;

        let source_subject = CapacitySubject::Port {
            node_id: link.from_node.clone(),
            port_id: link.from_port.clone(),
        };
        let link_subject = CapacitySubject::Link {
            link_id: link.id.clone(),
        };
        let destination_subject = CapacitySubject::Port {
            node_id: link.to_node.clone(),
            port_id: link.to_port.clone(),
        };

        let source_capacity = covering_capacity(capacities, &source_subject, &allocation)?;
        let link_capacity = covering_capacity(capacities, &link_subject, &allocation)?;
        let destination_capacity =
            covering_capacity(capacities, &destination_subject, &allocation)?;

        for window in [&source_capacity, &link_capacity, &destination_capacity] {
            window.validate_against(topology).map_err(|error| {
                AllocationError::InvalidCapacityWindow {
                    window_id: window.id.clone(),
                    reason: error.to_string(),
                }
            })?;
        }

        self.check_subject_capacity(
            topology,
            &source_subject,
            &source_capacity,
            &allocation,
            allocation.sent.value,
        )?;
        self.check_subject_capacity(
            topology,
            &link_subject,
            &link_capacity,
            &allocation,
            allocation.sent.value,
        )?;
        self.check_subject_capacity(
            topology,
            &destination_subject,
            &destination_capacity,
            &allocation,
            delivered.value,
        )?;

        let admitted = AdmittedAllocation {
            allocation,
            source_capacity,
            link_capacity,
            destination_capacity,
        };
        self.allocations
            .insert(admitted.allocation.id.clone(), admitted);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&AdmittedAllocation> {
        self.allocations.get(id)
    }

    pub fn allocations(&self) -> impl Iterator<Item = &AdmittedAllocation> {
        self.allocations.values()
    }

    pub fn len(&self) -> usize {
        self.allocations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    fn check_subject_capacity(
        &self,
        topology: &ResourceTopology,
        subject: &CapacitySubject,
        window: &CapacityWindow,
        candidate: &PlannedAllocation,
        candidate_value: f64,
    ) -> Result<(), AllocationError> {
        if candidate.sent.key != window.capacity.key {
            return Err(AllocationError::CapacityResourceMismatch {
                window_id: window.id.clone(),
            });
        }

        match window.semantics {
            CapacitySemantics::Concurrent => {
                let mut reservations = Vec::new();
                for admitted in self.allocations.values() {
                    let Some(binding) = admitted.binding_for(subject) else {
                        continue;
                    };
                    if binding.id != window.id {
                        continue;
                    }
                    if let Some(value) = contribution(
                        topology,
                        admitted.allocation(),
                        subject,
                    )? {
                        reservations.push(Reservation {
                            start: admitted.allocation.valid_from,
                            end: admitted.allocation.valid_until,
                            value,
                        });
                    }
                }
                reservations.push(Reservation {
                    start: candidate.valid_from,
                    end: candidate.valid_until,
                    value: candidate_value,
                });
                let peak = peak_concurrent_utilization(&reservations);
                if peak > window.capacity.value + ALLOCATION_EPSILON {
                    return Err(AllocationError::ConcurrentCapacityExceeded {
                        subject: subject.clone(),
                        window_id: window.id.clone(),
                        capacity: window.capacity.value,
                        attempted_peak: peak,
                    });
                }
            }
            CapacitySemantics::WindowBudget => {
                let mut total = candidate_value;
                for admitted in self.allocations.values() {
                    let Some(binding) = admitted.binding_for(subject) else {
                        continue;
                    };
                    if binding.id != window.id {
                        continue;
                    }
                    if let Some(value) = contribution(
                        topology,
                        admitted.allocation(),
                        subject,
                    )? {
                        total += value;
                    }
                }
                if total > window.capacity.value + ALLOCATION_EPSILON {
                    return Err(AllocationError::WindowBudgetExceeded {
                        subject: subject.clone(),
                        window_id: window.id.clone(),
                        capacity: window.capacity.value,
                        attempted_total: total,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Reservation {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    value: f64,
}

fn peak_concurrent_utilization(reservations: &[Reservation]) -> f64 {
    let mut points = Vec::with_capacity(reservations.len() * 2);
    for reservation in reservations {
        points.push(reservation.start);
        points.push(reservation.end);
    }
    points.sort();
    points.dedup();

    let mut peak: f64 = 0.0;
    for point in points.into_iter().take_while(|point| {
        reservations.iter().any(|reservation| *point < reservation.end)
    }) {
        let active: f64 = reservations
            .iter()
            .filter(|reservation| reservation.start <= point && point < reservation.end)
            .map(|reservation| reservation.value)
            .sum();
        peak = peak.max(active);
    }
    peak
}

fn covering_capacity(
    capacities: &CapacitySchedule,
    subject: &CapacitySubject,
    allocation: &PlannedAllocation,
) -> Result<CapacityWindow, AllocationError> {
    capacities
        .covering_window(subject, allocation.valid_from, allocation.valid_until)
        .cloned()
        .ok_or_else(|| AllocationError::MissingCapacityWindow {
            subject: subject.clone(),
            allocation_id: allocation.id.clone(),
        })
}

fn validate_against_link(
    allocation: &PlannedAllocation,
    link: &ResourceLink,
) -> Result<(), AllocationError> {
    if allocation.sent.key != link.capacity.key {
        return Err(AllocationError::ResourceKeyMismatch {
            allocation_id: allocation.id.clone(),
            link_id: link.id.clone(),
        });
    }
    if allocation.sent.value > link.capacity.value + ALLOCATION_EPSILON {
        return Err(AllocationError::AllocationExceedsLinkLimit {
            allocation_id: allocation.id.clone(),
            link_id: link.id.clone(),
            requested: allocation.sent.value,
            link_limit: link.capacity.value,
        });
    }
    Ok(())
}

fn delivered_amount(
    sent: ResourceAmount,
    link: &ResourceLink,
) -> Result<ResourceAmount, AllocationError> {
    if !link.loss_fraction.is_finite() || !(0.0..=1.0).contains(&link.loss_fraction) {
        return Err(AllocationError::InvalidLinkLoss {
            link_id: link.id.clone(),
            loss_fraction: link.loss_fraction,
        });
    }
    sent.scaled(1.0 - link.loss_fraction)
        .map_err(|error| AllocationError::InvalidResourceAmount(error.to_string()))
}

fn contribution(
    topology: &ResourceTopology,
    allocation: &PlannedAllocation,
    subject: &CapacitySubject,
) -> Result<Option<f64>, AllocationError> {
    let link = topology
        .link(&allocation.link_id)
        .ok_or_else(|| AllocationError::UnknownLink(allocation.link_id.clone()))?;
    match subject {
        CapacitySubject::Link { link_id } => {
            if &allocation.link_id == link_id {
                Ok(Some(allocation.sent.value))
            } else {
                Ok(None)
            }
        }
        CapacitySubject::Port { node_id, port_id } => {
            if &link.from_node == node_id && &link.from_port == port_id {
                Ok(Some(allocation.sent.value))
            } else if &link.to_node == node_id && &link.to_port == port_id {
                Ok(Some(delivered_amount(allocation.sent, link)?.value))
            } else {
                Ok(None)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AllocationError {
    #[error("allocation id must not be empty")]
    EmptyAllocationId,
    #[error("allocation link id must not be empty")]
    EmptyLinkId,
    #[error("allocation window end must be after start")]
    InvalidValidityWindow,
    #[error("allocation amount must be finite and positive, got {0}")]
    InvalidAmount(f64),
    #[error("duplicate allocation {0}")]
    DuplicateAllocation(String),
    #[error("unknown topology link {0}")]
    UnknownLink(String),
    #[error("allocation {allocation_id} resource dimension does not match link {link_id}")]
    ResourceKeyMismatch {
        allocation_id: String,
        link_id: String,
    },
    #[error(
        "allocation {allocation_id} requests {requested}, exceeding static link {link_id} limit {link_limit}"
    )]
    AllocationExceedsLinkLimit {
        allocation_id: String,
        link_id: String,
        requested: f64,
        link_limit: f64,
    },
    #[error("link {link_id} has invalid loss fraction {loss_fraction}")]
    InvalidLinkLoss {
        link_id: String,
        loss_fraction: f64,
    },
    #[error("invalid resource amount: {0}")]
    InvalidResourceAmount(String),
    #[error("allocation {allocation_id} has no capacity window for {subject:?}")]
    MissingCapacityWindow {
        subject: CapacitySubject,
        allocation_id: String,
    },
    #[error("capacity window {window_id} is invalid against current topology: {reason}")]
    InvalidCapacityWindow { window_id: String, reason: String },
    #[error("capacity window {window_id} resource dimension does not match allocation")]
    CapacityResourceMismatch { window_id: String },
    #[error(
        "concurrent utilization {attempted_peak} exceeds capacity {capacity} for {subject:?} in window {window_id}"
    )]
    ConcurrentCapacityExceeded {
        subject: CapacitySubject,
        window_id: String,
        capacity: f64,
        attempted_peak: f64,
    },
    #[error(
        "cumulative utilization {attempted_total} exceeds budget {capacity} for {subject:?} in window {window_id}"
    )]
    WindowBudgetExceeded {
        subject: CapacitySubject,
        window_id: String,
        capacity: f64,
        attempted_total: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_capacity::CapacitySemantics;
    use symthaea_resource_model::{
        PortDirection, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_topology::ResourceLink;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn compute(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Compute, ResourceUnit::CpuSecond, value).unwrap()
    }

    fn port(id: &str, direction: PortDirection, capacity: ResourceAmount) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            direction,
            capacity,
        }
    }

    fn simple_topology(amount: fn(f64) -> ResourceAmount, loss_fraction: f64) -> ResourceTopology {
        let mut topology = ResourceTopology::default();
        topology
            .add_node("source", [port("out", PortDirection::Output, amount(100.0))])
            .unwrap();
        topology
            .add_node("sink", [port("in", PortDirection::Input, amount(100.0))])
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: amount(100.0),
                loss_fraction,
            })
            .unwrap();
        topology
    }

    fn subject_source() -> CapacitySubject {
        CapacitySubject::Port {
            node_id: "source".into(),
            port_id: "out".into(),
        }
    }

    fn subject_link() -> CapacitySubject {
        CapacitySubject::Link {
            link_id: "line".into(),
        }
    }

    fn subject_sink() -> CapacitySubject {
        CapacitySubject::Port {
            node_id: "sink".into(),
            port_id: "in".into(),
        }
    }

    fn capacity_window(
        id: &str,
        subject: CapacitySubject,
        capacity: ResourceAmount,
        semantics: CapacitySemantics,
    ) -> CapacityWindow {
        CapacityWindow {
            id: id.into(),
            subject,
            valid_from: t0(),
            valid_until: t0() + Duration::minutes(60),
            capacity,
            semantics,
        }
    }

    fn schedule(
        topology: &ResourceTopology,
        amount: fn(f64) -> ResourceAmount,
        semantics: CapacitySemantics,
        source_limit: f64,
        link_limit: f64,
        sink_limit: f64,
    ) -> CapacitySchedule {
        let mut schedule = CapacitySchedule::default();
        for window in [
            capacity_window("source-cap", subject_source(), amount(source_limit), semantics),
            capacity_window("link-cap", subject_link(), amount(link_limit), semantics),
            capacity_window("sink-cap", subject_sink(), amount(sink_limit), semantics),
        ] {
            schedule.add_window(topology, window).unwrap();
        }
        schedule
    }

    fn allocation(
        id: &str,
        start_minutes: i64,
        end_minutes: i64,
        amount: ResourceAmount,
    ) -> PlannedAllocation {
        PlannedAllocation {
            id: id.into(),
            link_id: "line".into(),
            valid_from: t0() + Duration::minutes(start_minutes),
            valid_until: t0() + Duration::minutes(end_minutes),
            sent: amount,
        }
    }

    #[test]
    fn concurrent_disjoint_allocations_can_each_use_full_capacity() {
        let topology = simple_topology(power, 0.0);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 30, power(100.0)))
            .unwrap();
        book.admit(&topology, &capacities, allocation("b", 30, 60, power(100.0)))
            .unwrap();
        assert_eq!(book.len(), 2);
    }

    #[test]
    fn concurrent_overlap_is_rejected() {
        let topology = simple_topology(power, 0.0);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 40, power(60.0)))
            .unwrap();
        let result = book.admit(
            &topology,
            &capacities,
            allocation("b", 20, 60, power(50.0)),
        );
        assert!(matches!(
            result,
            Err(AllocationError::ConcurrentCapacityExceeded { attempted_peak, .. })
                if (attempted_peak - 110.0).abs() < 1e-9
        ));
    }

    #[test]
    fn concurrent_sweep_does_not_sum_disjoint_existing_reservations() {
        let topology = simple_topology(power, 0.0);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 30, power(40.0)))
            .unwrap();
        book.admit(&topology, &capacities, allocation("b", 30, 60, power(40.0)))
            .unwrap();
        book.admit(&topology, &capacities, allocation("spanning", 0, 60, power(60.0)))
            .unwrap();
        assert_eq!(book.len(), 3);
    }

    #[test]
    fn window_budget_sums_disjoint_allocations() {
        let topology = simple_topology(compute, 0.0);
        let capacities = schedule(
            &topology,
            compute,
            CapacitySemantics::WindowBudget,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 30, compute(60.0)))
            .unwrap();
        let result = book.admit(
            &topology,
            &capacities,
            allocation("b", 30, 60, compute(50.0)),
        );
        assert!(matches!(
            result,
            Err(AllocationError::WindowBudgetExceeded { attempted_total, .. })
                if (attempted_total - 110.0).abs() < 1e-9
        ));
    }

    #[test]
    fn destination_capacity_is_charged_after_transfer_loss() {
        let topology = simple_topology(power, 0.2);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            80.0,
        );
        let mut book = AllocationBook::default();
        book.admit(
            &topology,
            &capacities,
            allocation("lossy", 0, 60, power(100.0)),
        )
        .unwrap();
        assert_eq!(
            book.get("lossy")
                .unwrap()
                .allocation()
                .delivered(&topology)
                .unwrap()
                .value,
            80.0
        );
    }

    #[test]
    fn all_three_capacity_boundaries_are_required() {
        let topology = simple_topology(power, 0.0);
        let mut capacities = CapacitySchedule::default();
        capacities
            .add_window(
                &topology,
                capacity_window(
                    "source-cap",
                    subject_source(),
                    power(100.0),
                    CapacitySemantics::Concurrent,
                ),
            )
            .unwrap();
        capacities
            .add_window(
                &topology,
                capacity_window(
                    "sink-cap",
                    subject_sink(),
                    power(100.0),
                    CapacitySemantics::Concurrent,
                ),
            )
            .unwrap();

        let mut book = AllocationBook::default();
        assert!(matches!(
            book.admit(&topology, &capacities, allocation("missing", 0, 60, power(10.0))),
            Err(AllocationError::MissingCapacityWindow {
                subject: CapacitySubject::Link { link_id },
                ..
            }) if link_id == "line"
        ));
    }

    #[test]
    fn admitted_plan_retains_exact_capacity_bindings() {
        let topology = simple_topology(power, 0.0);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 60, power(10.0)))
            .unwrap();
        let admitted = book.get("a").unwrap();
        assert_eq!(admitted.source_capacity().id, "source-cap");
        assert_eq!(admitted.link_capacity().id, "link-cap");
        assert_eq!(admitted.destination_capacity().id, "sink-cap");
    }

    #[test]
    fn duplicate_allocation_identity_is_rejected() {
        let topology = simple_topology(power, 0.0);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("same", 0, 20, power(10.0)))
            .unwrap();
        assert!(matches!(
            book.admit(&topology, &capacities, allocation("same", 20, 40, power(10.0))),
            Err(AllocationError::DuplicateAllocation(id)) if id == "same"
        ));
    }

    #[test]
    fn allocation_cannot_change_resource_dimension() {
        let topology = simple_topology(power, 0.0);
        let capacities = schedule(
            &topology,
            power,
            CapacitySemantics::Concurrent,
            100.0,
            100.0,
            100.0,
        );
        let mut book = AllocationBook::default();
        assert!(matches!(
            book.admit(&topology, &capacities, allocation("wrong", 0, 20, compute(10.0))),
            Err(AllocationError::ResourceKeyMismatch { .. })
        ));
    }

    #[test]
    fn bidirectional_port_shares_concurrent_capacity_across_directions() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node("bus", [port("io", PortDirection::Bidirectional, power(100.0))])
            .unwrap();
        topology
            .add_node("source", [port("out", PortDirection::Output, power(100.0))])
            .unwrap();
        topology
            .add_node("sink", [port("in", PortDirection::Input, power(100.0))])
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "outbound".into(),
                from_node: "bus".into(),
                from_port: "io".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: power(100.0),
                loss_fraction: 0.0,
            })
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "inbound".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "bus".into(),
                to_port: "io".into(),
                capacity: power(100.0),
                loss_fraction: 0.0,
            })
            .unwrap();

        let mut capacities = CapacitySchedule::default();
        let subjects = [
            ("bus-cap", CapacitySubject::Port { node_id: "bus".into(), port_id: "io".into() }),
            ("source-cap", CapacitySubject::Port { node_id: "source".into(), port_id: "out".into() }),
            ("sink-cap", CapacitySubject::Port { node_id: "sink".into(), port_id: "in".into() }),
            ("outbound-cap", CapacitySubject::Link { link_id: "outbound".into() }),
            ("inbound-cap", CapacitySubject::Link { link_id: "inbound".into() }),
        ];
        for (id, subject) in subjects {
            capacities
                .add_window(
                    &topology,
                    capacity_window(id, subject, power(100.0), CapacitySemantics::Concurrent),
                )
                .unwrap();
        }

        let mut book = AllocationBook::default();
        let mut outbound = allocation("out", 0, 60, power(70.0));
        outbound.link_id = "outbound".into();
        book.admit(&topology, &capacities, outbound).unwrap();
        let mut inbound = allocation("in", 0, 60, power(40.0));
        inbound.link_id = "inbound".into();
        assert!(matches!(
            book.admit(&topology, &capacities, inbound),
            Err(AllocationError::ConcurrentCapacityExceeded {
                subject: CapacitySubject::Port { node_id, port_id },
                attempted_peak,
                ..
            }) if node_id == "bus" && port_id == "io" && (attempted_peak - 110.0).abs() < 1e-9
        ));
    }
}

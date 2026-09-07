// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit temporal meaning for resource capacity.
//!
//! Resource units alone do not determine how a scheduler may conserve capacity.
//! `Watt` and `bit/s` are naturally rate-like, while `CPU-second`, `Joule`,
//! `Byte`, `m³`, and `kg` may represent work, inventory, storage, or a bounded
//! service budget depending on deployment context. Inferring scheduling semantics
//! from the unit would therefore create false physical claims.
//!
//! This crate makes the distinction explicit and caller-declared:
//!
//! - `Concurrent` means the declared amount is available at every instant in the
//!   window; later allocations must sum simultaneous reservations.
//! - `WindowBudget` means the amount is a total budget across the complete
//!   window; later allocations must sum all consumption assigned to that window,
//!   regardless of whether those allocations overlap in time.
//!
//! Capacity windows are descriptive constraints only. They do not reserve
//! resources, choose workloads, mint Mycelix leases, or create effect authority.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_resource_model::ResourceAmount;
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

const CAPACITY_EPSILON: f64 = 1e-9;

/// The physical boundary whose capacity is being qualified for a time window.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CapacitySubject {
    Port { node_id: String, port_id: String },
    Link { link_id: String },
}

impl CapacitySubject {
    fn validate(&self) -> Result<(), CapacityError> {
        match self {
            Self::Port { node_id, port_id } => {
                if node_id.trim().is_empty() {
                    return Err(CapacityError::EmptyNodeId);
                }
                if port_id.trim().is_empty() {
                    return Err(CapacityError::EmptyPortId);
                }
            }
            Self::Link { link_id } => {
                if link_id.trim().is_empty() {
                    return Err(CapacityError::EmptyLinkId);
                }
            }
        }
        Ok(())
    }
}

/// How a later allocation layer must conserve the declared capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitySemantics {
    /// Capacity is simultaneously available throughout the window.
    ///
    /// Future allocation rule: at every instant, the sum of active reservations
    /// must remain within the declared amount.
    Concurrent,
    /// Capacity is a total quantity available across the whole window.
    ///
    /// Future allocation rule: all consumption assigned to this budget window is
    /// summed, even when individual allocations do not overlap.
    WindowBudget,
}

/// One explicit interpretation of a topology boundary's capacity over time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityWindow {
    pub id: String,
    pub subject: CapacitySubject,
    pub valid_from: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
    pub capacity: ResourceAmount,
    pub semantics: CapacitySemantics,
}

impl CapacityWindow {
    pub fn validate_structure(&self) -> Result<(), CapacityError> {
        if self.id.trim().is_empty() {
            return Err(CapacityError::EmptyWindowId);
        }
        self.subject.validate()?;
        if self.valid_until <= self.valid_from {
            return Err(CapacityError::InvalidValidityWindow);
        }
        if !self.capacity.value.is_finite() || self.capacity.value <= 0.0 {
            return Err(CapacityError::InvalidCapacity(self.capacity.value));
        }
        Ok(())
    }

    /// Validate this temporal capacity against the static physical topology.
    ///
    /// This proves only that the declared time-window capacity does not exceed the
    /// topology boundary's static typed limit. It does not prove that telemetry is
    /// current, that the capacity is available, or that any reservation is valid.
    pub fn validate_against(&self, topology: &ResourceTopology) -> Result<(), CapacityError> {
        self.validate_structure()?;
        let static_limit = static_limit(topology, &self.subject)?;
        if self.capacity.key != static_limit.key {
            return Err(CapacityError::ResourceKeyMismatch);
        }
        if self.capacity.value > static_limit.value + CAPACITY_EPSILON {
            return Err(CapacityError::ExceedsStaticLimit {
                declared: self.capacity.value,
                static_limit: static_limit.value,
            });
        }
        Ok(())
    }

    /// Whether this half-open capacity window fully contains `[start, end)`.
    pub fn contains(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> bool {
        start < end && self.valid_from <= start && end <= self.valid_until
    }

    /// Half-open overlap test.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.valid_from < other.valid_until && other.valid_from < self.valid_until
    }
}

/// Non-ambiguous temporal capacity declarations over one topology.
///
/// v1 deliberately rejects overlapping windows for the same physical subject.
/// This avoids hidden precedence, last-writer-wins, or implicit `min()` semantics.
/// A deployment that needs derating should split/restate the affected intervals
/// explicitly before presenting the schedule to a planner.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CapacitySchedule {
    windows: BTreeMap<String, CapacityWindow>,
}

impl CapacitySchedule {
    pub fn add_window(
        &mut self,
        topology: &ResourceTopology,
        window: CapacityWindow,
    ) -> Result<(), CapacityError> {
        window.validate_against(topology)?;
        if self.windows.contains_key(&window.id) {
            return Err(CapacityError::DuplicateWindow(window.id));
        }

        for existing in self.windows.values() {
            if existing.subject == window.subject && existing.overlaps(&window) {
                return Err(CapacityError::OverlappingSubjectWindows {
                    existing_id: existing.id.clone(),
                    candidate_id: window.id,
                });
            }
        }

        self.windows.insert(window.id.clone(), window);
        Ok(())
    }

    pub fn window(&self, id: &str) -> Option<&CapacityWindow> {
        self.windows.get(id)
    }

    pub fn windows(&self) -> impl Iterator<Item = &CapacityWindow> {
        self.windows.values()
    }

    pub fn windows_for<'a>(
        &'a self,
        subject: &'a CapacitySubject,
    ) -> impl Iterator<Item = &'a CapacityWindow> + 'a {
        self.windows
            .values()
            .filter(move |window| &window.subject == subject)
    }

    /// Find the unique capacity window that fully covers a requested interval.
    ///
    /// Overlap rejection in `add_window` means there can be at most one match.
    pub fn covering_window(
        &self,
        subject: &CapacitySubject,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Option<&CapacityWindow> {
        self.windows_for(subject)
            .find(|window| window.contains(start, end))
    }
}

fn static_limit(
    topology: &ResourceTopology,
    subject: &CapacitySubject,
) -> Result<ResourceAmount, CapacityError> {
    match subject {
        CapacitySubject::Port { node_id, port_id } => topology
            .port(node_id, port_id)
            .map(|port| port.capacity)
            .ok_or_else(|| CapacityError::UnknownPort {
                node_id: node_id.clone(),
                port_id: port_id.clone(),
            }),
        CapacitySubject::Link { link_id } => topology
            .link(link_id)
            .map(|link| link.capacity)
            .ok_or_else(|| CapacityError::UnknownLink(link_id.clone())),
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CapacityError {
    #[error("capacity window id must not be empty")]
    EmptyWindowId,
    #[error("capacity subject node id must not be empty")]
    EmptyNodeId,
    #[error("capacity subject port id must not be empty")]
    EmptyPortId,
    #[error("capacity subject link id must not be empty")]
    EmptyLinkId,
    #[error("capacity window end must be after start")]
    InvalidValidityWindow,
    #[error("capacity must be finite and positive, got {0}")]
    InvalidCapacity(f64),
    #[error("unknown topology port {node_id}/{port_id}")]
    UnknownPort { node_id: String, port_id: String },
    #[error("unknown topology link {0}")]
    UnknownLink(String),
    #[error("capacity resource dimension does not match the topology subject")]
    ResourceKeyMismatch,
    #[error("declared capacity {declared} exceeds static topology limit {static_limit}")]
    ExceedsStaticLimit {
        declared: f64,
        static_limit: f64,
    },
    #[error("duplicate capacity window {0}")]
    DuplicateWindow(String),
    #[error(
        "capacity windows {existing_id} and {candidate_id} overlap for the same physical subject"
    )]
    OverlappingSubjectWindows {
        existing_id: String,
        candidate_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_model::{
        PortDirection, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_topology::ResourceLink;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn electricity(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn compute(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Compute, ResourceUnit::CpuSecond, value).unwrap()
    }

    fn topology() -> ResourceTopology {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [
                    ResourcePort {
                        id: "power".into(),
                        direction: PortDirection::Output,
                        capacity: electricity(100.0),
                    },
                    ResourcePort {
                        id: "compute".into(),
                        direction: PortDirection::Output,
                        capacity: compute(3600.0),
                    },
                ],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [ResourcePort {
                    id: "power".into(),
                    direction: PortDirection::Input,
                    capacity: electricity(100.0),
                }],
            )
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "power".into(),
                to_node: "sink".into(),
                to_port: "power".into(),
                capacity: electricity(80.0),
                loss_fraction: 0.05,
            })
            .unwrap();
        topology
    }

    fn port_subject(port: &str) -> CapacitySubject {
        CapacitySubject::Port {
            node_id: "source".into(),
            port_id: port.into(),
        }
    }

    fn window(
        id: &str,
        subject: CapacitySubject,
        start_minutes: i64,
        end_minutes: i64,
        capacity: ResourceAmount,
        semantics: CapacitySemantics,
    ) -> CapacityWindow {
        CapacityWindow {
            id: id.into(),
            subject,
            valid_from: t0() + Duration::minutes(start_minutes),
            valid_until: t0() + Duration::minutes(end_minutes),
            capacity,
            semantics,
        }
    }

    #[test]
    fn temporal_semantics_are_explicit_not_inferred_from_unit() {
        let topology = topology();
        let mut schedule = CapacitySchedule::default();
        schedule
            .add_window(
                &topology,
                window(
                    "compute-budget-a",
                    port_subject("compute"),
                    0,
                    60,
                    compute(3600.0),
                    CapacitySemantics::WindowBudget,
                ),
            )
            .unwrap();
        schedule
            .add_window(
                &topology,
                window(
                    "compute-concurrent-b",
                    port_subject("compute"),
                    60,
                    120,
                    compute(1800.0),
                    CapacitySemantics::Concurrent,
                ),
            )
            .unwrap();

        assert_eq!(
            schedule.window("compute-budget-a").unwrap().semantics,
            CapacitySemantics::WindowBudget
        );
        assert_eq!(
            schedule.window("compute-concurrent-b").unwrap().semantics,
            CapacitySemantics::Concurrent
        );
    }

    #[test]
    fn capacity_cannot_exceed_static_port_limit() {
        let topology = topology();
        let too_large = window(
            "too-large",
            port_subject("power"),
            0,
            60,
            electricity(101.0),
            CapacitySemantics::Concurrent,
        );
        assert!(matches!(
            too_large.validate_against(&topology),
            Err(CapacityError::ExceedsStaticLimit { .. })
        ));
    }

    #[test]
    fn link_capacity_window_is_bound_to_exact_link_limit() {
        let topology = topology();
        let subject = CapacitySubject::Link {
            link_id: "line".into(),
        };
        let valid = window(
            "line-window",
            subject.clone(),
            0,
            60,
            electricity(80.0),
            CapacitySemantics::Concurrent,
        );
        valid.validate_against(&topology).unwrap();

        let too_large = window(
            "line-too-large",
            subject,
            60,
            120,
            electricity(81.0),
            CapacitySemantics::Concurrent,
        );
        assert!(matches!(
            too_large.validate_against(&topology),
            Err(CapacityError::ExceedsStaticLimit { .. })
        ));
    }

    #[test]
    fn overlapping_windows_for_same_subject_fail_closed() {
        let topology = topology();
        let mut schedule = CapacitySchedule::default();
        schedule
            .add_window(
                &topology,
                window(
                    "first",
                    port_subject("power"),
                    0,
                    60,
                    electricity(100.0),
                    CapacitySemantics::Concurrent,
                ),
            )
            .unwrap();
        let result = schedule.add_window(
            &topology,
            window(
                "overlap",
                port_subject("power"),
                30,
                90,
                electricity(90.0),
                CapacitySemantics::Concurrent,
            ),
        );
        assert!(matches!(
            result,
            Err(CapacityError::OverlappingSubjectWindows { existing_id, candidate_id })
                if existing_id == "first" && candidate_id == "overlap"
        ));
    }

    #[test]
    fn adjacent_half_open_windows_are_allowed() {
        let topology = topology();
        let mut schedule = CapacitySchedule::default();
        for (id, start, end) in [("a", 0, 60), ("b", 60, 120)] {
            schedule
                .add_window(
                    &topology,
                    window(
                        id,
                        port_subject("power"),
                        start,
                        end,
                        electricity(100.0),
                        CapacitySemantics::Concurrent,
                    ),
                )
                .unwrap();
        }
        assert_eq!(schedule.windows_for(&port_subject("power")).count(), 2);
    }

    #[test]
    fn different_subjects_may_have_overlapping_windows() {
        let topology = topology();
        let mut schedule = CapacitySchedule::default();
        schedule
            .add_window(
                &topology,
                window(
                    "power",
                    port_subject("power"),
                    0,
                    60,
                    electricity(100.0),
                    CapacitySemantics::Concurrent,
                ),
            )
            .unwrap();
        schedule
            .add_window(
                &topology,
                window(
                    "compute",
                    port_subject("compute"),
                    0,
                    60,
                    compute(3600.0),
                    CapacitySemantics::WindowBudget,
                ),
            )
            .unwrap();
        assert_eq!(schedule.windows().count(), 2);
    }

    #[test]
    fn covering_window_requires_full_interval_containment() {
        let topology = topology();
        let subject = port_subject("power");
        let mut schedule = CapacitySchedule::default();
        schedule
            .add_window(
                &topology,
                window(
                    "capacity",
                    subject.clone(),
                    0,
                    60,
                    electricity(100.0),
                    CapacitySemantics::Concurrent,
                ),
            )
            .unwrap();
        assert!(
            schedule
                .covering_window(
                    &subject,
                    t0() + Duration::minutes(10),
                    t0() + Duration::minutes(50),
                )
                .is_some()
        );
        assert!(
            schedule
                .covering_window(
                    &subject,
                    t0() + Duration::minutes(50),
                    t0() + Duration::minutes(70),
                )
                .is_none()
        );
    }

    #[test]
    fn resource_dimension_must_match_static_subject() {
        let topology = topology();
        let wrong = window(
            "wrong-key",
            port_subject("power"),
            0,
            60,
            compute(10.0),
            CapacitySemantics::WindowBudget,
        );
        assert!(matches!(
            wrong.validate_against(&topology),
            Err(CapacityError::ResourceKeyMismatch)
        ));
    }

    #[test]
    fn invalid_time_window_is_rejected() {
        let topology = topology();
        let invalid = window(
            "invalid",
            port_subject("power"),
            60,
            60,
            electricity(100.0),
            CapacitySemantics::Concurrent,
        );
        assert!(matches!(
            invalid.validate_against(&topology),
            Err(CapacityError::InvalidValidityWindow)
        ));
    }
}

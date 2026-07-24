// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded mission work orders and deterministic scheduling.
//!
//! Long-duration autonomy needs a queue of explicit, inspectable commitments.
//! A work order is not allowed to become motor authority by itself: this
//! scheduler only selects eligible work. Physical safety, logistics admission,
//! and the command arbiter remain independent gates.

use crate::mission::SubterraneanMissionIntent;
use crate::tunnel_graph::TunnelNodeId;
use serde::{Deserialize, Serialize};

pub const MAX_WORK_ORDERS: usize = 64;
pub const MAX_WORK_PREREQUISITES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkOrderId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkKind {
    Survey,
    Bore,
    StabilizeRoof,
    Dewater,
    DeployRelay,
    ExtractSample,
    ClearSpoil,
    ReturnToBase,
    Maintenance,
}

impl WorkKind {
    pub const fn mission_intent(self) -> SubterraneanMissionIntent {
        match self {
            Self::Survey => SubterraneanMissionIntent::ProbeAhead,
            Self::Bore | Self::ExtractSample => SubterraneanMissionIntent::FollowVein,
            Self::DeployRelay => SubterraneanMissionIntent::MaintainRelay,
            Self::ReturnToBase => SubterraneanMissionIntent::ReturnHome,
            Self::StabilizeRoof | Self::Dewater | Self::ClearSpoil | Self::Maintenance => {
                SubterraneanMissionIntent::HoldPosition
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkResourceEstimate {
    /// Estimated battery fraction consumed by the work itself, excluding the
    /// route to and from the target.
    pub battery_fraction: f64,
    pub sealant_fraction: f64,
    pub relay_units: u8,
    pub roof_support_units: u8,
    pub sample_capacity: f64,
    pub spoil_capacity: f64,
}

impl WorkResourceEstimate {
    pub const fn zero() -> Self {
        Self {
            battery_fraction: 0.0,
            sealant_fraction: 0.0,
            relay_units: 0,
            roof_support_units: 0,
            sample_capacity: 0.0,
            spoil_capacity: 0.0,
        }
    }

    pub fn is_valid(self) -> bool {
        [
            self.battery_fraction,
            self.sealant_fraction,
            self.sample_capacity,
            self.spoil_capacity,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkPriority {
    Routine,
    Important,
    Urgent,
    Emergency,
}

impl WorkPriority {
    const fn rank(self) -> u8 {
        match self {
            Self::Routine => 0,
            Self::Important => 1,
            Self::Urgent => 2,
            Self::Emergency => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkStatus {
    Pending,
    Active,
    Suspended,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkOrder {
    pub id: WorkOrderId,
    pub kind: WorkKind,
    pub target: TunnelNodeId,
    pub priority: WorkPriority,
    pub prerequisites: [Option<WorkOrderId>; MAX_WORK_PREREQUISITES],
    pub estimated_steps: u64,
    pub deadline_step: Option<u64>,
    pub resources: WorkResourceEstimate,
    pub status: WorkStatus,
    pub completed_steps: u64,
}

impl WorkOrder {
    pub fn validate(&self) -> Result<(), WorkOrderError> {
        if self.id.0 == 0 || self.estimated_steps == 0 || !self.resources.is_valid() {
            return Err(WorkOrderError::InvalidOrder);
        }
        if self
            .prerequisites
            .iter()
            .flatten()
            .any(|prerequisite| *prerequisite == self.id)
        {
            return Err(WorkOrderError::SelfDependency);
        }
        Ok(())
    }

    pub fn progress_ratio(&self) -> f64 {
        self.completed_steps as f64 / self.estimated_steps.max(1) as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkOrderError {
    InvalidOrder,
    DuplicateId,
    MissingPrerequisite,
    SelfDependency,
    Capacity,
    NotFound,
    InvalidTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkPreemptionReason {
    PhysicalHazard,
    ReturnReserve,
    ResourceLimit,
    Maintenance,
    TeamRightOfWay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerSnapshot {
    pub queued: usize,
    pub completed: usize,
    pub failed: usize,
    pub active: Option<WorkOrderId>,
    pub last_preemption: Option<WorkPreemptionReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkScheduler {
    orders: Vec<WorkOrder>,
    active: Option<WorkOrderId>,
    #[serde(skip)]
    last_preemption: Option<WorkPreemptionReason>,
}

impl WorkScheduler {
    pub fn new() -> Self {
        Self {
            orders: Vec::with_capacity(MAX_WORK_ORDERS),
            active: None,
            last_preemption: None,
        }
    }

    pub fn validate(&self) -> Result<(), WorkOrderError> {
        if self.orders.len() > MAX_WORK_ORDERS {
            return Err(WorkOrderError::Capacity);
        }
        for (index, order) in self.orders.iter().enumerate() {
            order.validate()?;
            if self.orders[..index]
                .iter()
                .any(|candidate| candidate.id == order.id)
            {
                return Err(WorkOrderError::DuplicateId);
            }
            for prerequisite in order.prerequisites.iter().flatten() {
                if self.order(*prerequisite).is_none() {
                    return Err(WorkOrderError::MissingPrerequisite);
                }
            }
        }
        if let Some(active) = self.active {
            if !self
                .order(active)
                .is_some_and(|order| order.status == WorkStatus::Active)
            {
                return Err(WorkOrderError::InvalidTransition);
            }
        } else if self
            .orders
            .iter()
            .any(|order| order.status == WorkStatus::Active)
        {
            return Err(WorkOrderError::InvalidTransition);
        }
        Ok(())
    }

    pub fn orders(&self) -> &[WorkOrder] {
        &self.orders
    }

    pub fn order(&self, id: WorkOrderId) -> Option<&WorkOrder> {
        self.orders.iter().find(|order| order.id == id)
    }

    fn order_mut(&mut self, id: WorkOrderId) -> Option<&mut WorkOrder> {
        self.orders.iter_mut().find(|order| order.id == id)
    }

    pub fn submit(&mut self, mut order: WorkOrder) -> Result<(), WorkOrderError> {
        order.validate()?;
        if self.order(order.id).is_some() {
            return Err(WorkOrderError::DuplicateId);
        }
        if self.orders.len() >= MAX_WORK_ORDERS {
            return Err(WorkOrderError::Capacity);
        }
        for prerequisite in order.prerequisites.iter().flatten() {
            if self.order(*prerequisite).is_none() {
                return Err(WorkOrderError::MissingPrerequisite);
            }
        }
        order.status = WorkStatus::Pending;
        order.completed_steps = 0;
        self.orders.push(order);
        self.orders.sort_by_key(|candidate| candidate.id.0);
        Ok(())
    }

    fn prerequisites_complete(&self, order: &WorkOrder) -> bool {
        order.prerequisites.iter().flatten().all(|prerequisite| {
            self.order(*prerequisite)
                .is_some_and(|candidate| candidate.status == WorkStatus::Complete)
        })
    }

    fn ready_index(&self, current_step: u64) -> Option<usize> {
        let mut selected = None;
        for (index, order) in self.orders.iter().enumerate() {
            if !matches!(order.status, WorkStatus::Pending | WorkStatus::Suspended)
                || !self.prerequisites_complete(order)
            {
                continue;
            }
            let overdue = order
                .deadline_step
                .is_some_and(|deadline| current_step >= deadline);
            let key = (overdue, order.priority.rank(), u64::MAX - order.id.0);
            if selected.is_none_or(|(_, selected_key)| key > selected_key) {
                selected = Some((index, key));
            }
        }
        selected.map(|(index, _)| index)
    }

    pub fn select_next(&mut self, current_step: u64) -> Option<WorkOrderId> {
        if self.active.is_some() {
            return self.active;
        }
        let index = self.ready_index(current_step)?;
        let id = self.orders[index].id;
        self.orders[index].status = WorkStatus::Active;
        self.active = Some(id);
        Some(id)
    }

    pub fn active_order(&self) -> Option<&WorkOrder> {
        self.active.and_then(|id| self.order(id))
    }

    pub fn advance_active(&mut self, steps: u64) -> Result<WorkStatus, WorkOrderError> {
        let id = self.active.ok_or(WorkOrderError::InvalidTransition)?;
        let status = {
            let order = self.order_mut(id).ok_or(WorkOrderError::NotFound)?;
            if order.status != WorkStatus::Active {
                return Err(WorkOrderError::InvalidTransition);
            }
            order.completed_steps = order.completed_steps.saturating_add(steps);
            if order.completed_steps >= order.estimated_steps {
                order.completed_steps = order.estimated_steps;
                order.status = WorkStatus::Complete;
            }
            order.status
        };
        if status == WorkStatus::Complete {
            self.active = None;
        }
        Ok(status)
    }

    pub fn preempt(&mut self, reason: WorkPreemptionReason) -> Result<(), WorkOrderError> {
        let id = self.active.ok_or(WorkOrderError::InvalidTransition)?;
        let order = self.order_mut(id).ok_or(WorkOrderError::NotFound)?;
        if order.status != WorkStatus::Active {
            return Err(WorkOrderError::InvalidTransition);
        }
        order.status = WorkStatus::Suspended;
        self.active = None;
        self.last_preemption = Some(reason);
        Ok(())
    }

    pub fn fail_active(&mut self) -> Result<(), WorkOrderError> {
        let id = self.active.ok_or(WorkOrderError::InvalidTransition)?;
        let order = self.order_mut(id).ok_or(WorkOrderError::NotFound)?;
        order.status = WorkStatus::Failed;
        self.active = None;
        Ok(())
    }

    pub fn cancel(&mut self, id: WorkOrderId) -> Result<(), WorkOrderError> {
        let order = self.order_mut(id).ok_or(WorkOrderError::NotFound)?;
        if matches!(order.status, WorkStatus::Complete | WorkStatus::Failed) {
            return Err(WorkOrderError::InvalidTransition);
        }
        order.status = WorkStatus::Cancelled;
        if self.active == Some(id) {
            self.active = None;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            queued: self
                .orders
                .iter()
                .filter(|order| matches!(order.status, WorkStatus::Pending | WorkStatus::Suspended))
                .count(),
            completed: self
                .orders
                .iter()
                .filter(|order| order.status == WorkStatus::Complete)
                .count(),
            failed: self
                .orders
                .iter()
                .filter(|order| order.status == WorkStatus::Failed)
                .count(),
            active: self.active,
            last_preemption: self.last_preemption,
        }
    }

    pub fn reset_runtime(&mut self) {
        self.active = None;
        self.last_preemption = None;
        for order in &mut self.orders {
            if matches!(order.status, WorkStatus::Active | WorkStatus::Suspended) {
                order.status = WorkStatus::Pending;
            }
        }
    }
}

impl Default for WorkScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(id: u64, priority: WorkPriority, prerequisite: Option<WorkOrderId>) -> WorkOrder {
        WorkOrder {
            id: WorkOrderId(id),
            kind: WorkKind::Survey,
            target: TunnelNodeId(id as u32),
            priority,
            prerequisites: [prerequisite, None, None, None],
            estimated_steps: 10,
            deadline_step: None,
            resources: WorkResourceEstimate::zero(),
            status: WorkStatus::Pending,
            completed_steps: 0,
        }
    }

    #[test]
    fn prerequisites_block_dependent_work_until_complete() {
        let mut scheduler = WorkScheduler::new();
        scheduler
            .submit(order(1, WorkPriority::Routine, None))
            .expect("first");
        scheduler
            .submit(order(2, WorkPriority::Emergency, Some(WorkOrderId(1))))
            .expect("dependent");
        assert_eq!(scheduler.select_next(0), Some(WorkOrderId(1)));
        scheduler.advance_active(10).expect("complete");
        assert_eq!(scheduler.select_next(10), Some(WorkOrderId(2)));
    }

    #[test]
    fn emergency_priority_preempts_queue_order_not_active_safety() {
        let mut scheduler = WorkScheduler::new();
        scheduler
            .submit(order(1, WorkPriority::Routine, None))
            .expect("routine");
        scheduler
            .submit(order(2, WorkPriority::Emergency, None))
            .expect("emergency");
        assert_eq!(scheduler.select_next(0), Some(WorkOrderId(2)));
        scheduler
            .preempt(WorkPreemptionReason::PhysicalHazard)
            .expect("preempt");
        assert!(scheduler.active_order().is_none());
        assert_eq!(
            scheduler.snapshot().last_preemption,
            Some(WorkPreemptionReason::PhysicalHazard)
        );
    }

    #[test]
    fn equal_priority_selection_is_stable_by_lowest_id() {
        let mut scheduler = WorkScheduler::new();
        scheduler
            .submit(order(9, WorkPriority::Important, None))
            .expect("order");
        scheduler
            .submit(order(3, WorkPriority::Important, None))
            .expect("order");
        assert_eq!(scheduler.select_next(0), Some(WorkOrderId(3)));
    }
}

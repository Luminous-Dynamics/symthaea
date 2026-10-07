//! Vulkan synchronization lowering plan for the semantic execution schedule.
//!
//! This module contains no Vulkan handles and performs no device calls. It
//! converts semantic dependencies into a deterministic logical-queue plan.
//! Each logical queue owns its own timeline. Cross-queue edges become waits on
//! the producer queue timeline; same-queue edges rely on submission order.

use std::collections::{BTreeMap, HashMap, HashSet};

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ExecutionSchedule, ScheduleError, MAX_GRAPH_NODES};

pub const VULKAN_SYNC_PLAN_VERSION: u16 = 1;
pub const MAX_VULKAN_LOGICAL_QUEUES: u16 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VulkanQueueId(u16);

impl VulkanQueueId {
    pub fn new(value: u16) -> Result<Self, VulkanSyncError> {
        if value >= MAX_VULKAN_LOGICAL_QUEUES {
            return Err(VulkanSyncError::QueueLimitExceeded(value));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 { self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanQueueAssignment {
    pub node_id: u32,
    pub queue: VulkanQueueId,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VulkanTimelineWait {
    pub producer_node: u32,
    pub producer_queue: VulkanQueueId,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanTimelineSignal {
    pub queue: VulkanQueueId,
    pub value: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanSubmission {
    pub node_id: u32,
    pub ordinal: u32,
    pub queue: VulkanQueueId,
    pub waits: Vec<VulkanTimelineWait>,
    pub signal: VulkanTimelineSignal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanSyncPlan {
    pub version: u16,
    pub schedule_digest: String,
    pub queue_count: u16,
    pub assignments: Vec<VulkanQueueAssignment>,
    pub submissions: Vec<VulkanSubmission>,
}

impl VulkanSyncPlan {
    /// Lower a semantic schedule into logical Vulkan queues and per-queue timelines.
    ///
    /// Queue ids are logical lanes, not Vulkan queue-family indices or handles.
    /// The actual device mapping remains a separate capability decision.
    pub fn from_schedule(
        schedule: &ExecutionSchedule,
        assignments: &[VulkanQueueAssignment],
    ) -> Result<Self, VulkanSyncError> {
        let schedule_digest = schedule.digest_hex().map_err(VulkanSyncError::Schedule)?;
        if assignments.len() != schedule.nodes.len() {
            return Err(VulkanSyncError::AssignmentCountMismatch {
                expected: schedule.nodes.len(),
                actual: assignments.len(),
            });
        }

        let mut assignment_by_node = BTreeMap::new();
        for assignment in assignments {
            if assignment.node_id == u32::MAX {
                return Err(VulkanSyncError::InvalidNodeId);
            }
            if assignment_by_node.insert(assignment.node_id, assignment.queue).is_some() {
                return Err(VulkanSyncError::DuplicateAssignment(assignment.node_id));
            }
            VulkanQueueId::new(assignment.queue.get())?;
        }

        for node in &schedule.nodes {
            if !assignment_by_node.contains_key(&node.id) {
                return Err(VulkanSyncError::MissingAssignment(node.id));
            }
        }

        let queue_count = assignment_by_node.values().map(|queue| queue.get()).max().map_or(0, |max| max + 1);
        let mut assignments_canonical = assignment_by_node
            .iter()
            .map(|(&node_id, &queue)| VulkanQueueAssignment { node_id, queue })
            .collect::<Vec<_>>();
        assignments_canonical.sort_by_key(|assignment| assignment.node_id);

        let ordinal_by_node = schedule.nodes.iter().map(|node| (node.id, node.ordinal)).collect::<HashMap<_, _>>();
        let mut signal_by_node = BTreeMap::new();
        let mut next_value_by_queue = BTreeMap::<VulkanQueueId, u64>::new();
        for node in &schedule.nodes {
            let queue = assignments_canonical
                .iter()
                .find(|assignment| assignment.node_id == node.id)
                .map(|assignment| assignment.queue)
                .ok_or(VulkanSyncError::MissingAssignment(node.id))?;
            let next = next_value_by_queue.entry(queue).or_insert(0);
            *next = next.checked_add(1).ok_or(VulkanSyncError::TimelineValueOverflow)?;
            signal_by_node.insert(node.id, (queue, *next));
        }

        let mut waits_by_node = BTreeMap::<u32, Vec<VulkanTimelineWait>>::new();
        for dependency in &schedule.dependencies {
            let (producer_queue, value) = *signal_by_node
                .get(&dependency.from)
                .ok_or(VulkanSyncError::MissingSignal(dependency.from))?;
            let consumer_queue = assignments_canonical
                .iter()
                .find(|assignment| assignment.node_id == dependency.to)
                .map(|assignment| assignment.queue)
                .ok_or(VulkanSyncError::MissingAssignment(dependency.to))?;
            if producer_queue == consumer_queue {
                continue;
            }
            waits_by_node.entry(dependency.to).or_default().push(VulkanTimelineWait {
                producer_node: dependency.from,
                producer_queue,
                value,
            });
        }

        let mut submissions = Vec::with_capacity(schedule.nodes.len());
        for node in &schedule.nodes {
            let (queue, value) = signal_by_node[&node.id];
            let mut waits = waits_by_node.remove(&node.id).unwrap_or_default();
            waits.sort();
            waits.dedup();
            submissions.push(VulkanSubmission {
                node_id: node.id,
                ordinal: ordinal_by_node[&node.id],
                queue,
                waits,
                signal: VulkanTimelineSignal { queue, value },
            });
        }

        if !waits_by_node.is_empty() {
            return Err(VulkanSyncError::UnexpectedWaitTarget);
        }

        let plan = Self {
            version: VULKAN_SYNC_PLAN_VERSION,
            schedule_digest,
            queue_count,
            assignments: assignments_canonical,
            submissions,
        };
        plan.validate_internal()?;
        Ok(plan)
    }

    pub fn queue_submission_order(&self, queue: VulkanQueueId) -> Result<Vec<u32>, VulkanSyncError> {
        self.validate_internal()?;
        let mut order = self.submissions.iter().filter(|submission| submission.queue == queue).collect::<Vec<_>>();
        order.sort_by_key(|submission| submission.ordinal);
        Ok(order.into_iter().map(|submission| submission.node_id).collect())
    }

    pub fn digest(&self) -> Result<[u8; 32], VulkanSyncError> {
        self.validate_internal()?;
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-gpu-fabric.vulkan-sync-plan.v1\0");
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&(self.schedule_digest.len() as u32).to_le_bytes());
        hasher.update(self.schedule_digest.as_bytes());
        hasher.update(&self.queue_count.to_le_bytes());
        hasher.update(&(self.assignments.len() as u32).to_le_bytes());
        for assignment in &self.assignments {
            hasher.update(&assignment.node_id.to_le_bytes());
            hasher.update(&assignment.queue.get().to_le_bytes());
        }
        hasher.update(&(self.submissions.len() as u32).to_le_bytes());
        for submission in &self.submissions {
            hasher.update(&submission.node_id.to_le_bytes());
            hasher.update(&submission.ordinal.to_le_bytes());
            hasher.update(&submission.queue.get().to_le_bytes());
            hasher.update(&(submission.waits.len() as u32).to_le_bytes());
            for wait in &submission.waits {
                hasher.update(&wait.producer_node.to_le_bytes());
                hasher.update(&wait.producer_queue.get().to_le_bytes());
                hasher.update(&wait.value.to_le_bytes());
            }
            hasher.update(&submission.signal.queue.get().to_le_bytes());
            hasher.update(&submission.signal.value.to_le_bytes());
        }
        Ok(*hasher.finalize().as_bytes())
    }

    pub fn digest_hex(&self) -> Result<String, VulkanSyncError> {
        Ok(self.digest()?.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    fn validate_internal(&self) -> Result<(), VulkanSyncError> {
        if self.version != VULKAN_SYNC_PLAN_VERSION {
            return Err(VulkanSyncError::UnsupportedVersion(self.version));
        }
        if self.submissions.len() > MAX_GRAPH_NODES {
            return Err(VulkanSyncError::SubmissionLimitExceeded(self.submissions.len()));
        }
        if self.queue_count > MAX_VULKAN_LOGICAL_QUEUES {
            return Err(VulkanSyncError::QueueLimitExceeded(self.queue_count));
        }
        if self.assignments.len() != self.submissions.len() {
            return Err(VulkanSyncError::InternalCardinalityMismatch);
        }

        let mut seen_nodes = HashSet::with_capacity(self.submissions.len());
        let assignment_map = self.assignments.iter().map(|item| (item.node_id, item.queue)).collect::<BTreeMap<_, _>>();
        if assignment_map.len() != self.assignments.len() {
            return Err(VulkanSyncError::DuplicateAssignment(u32::MAX));
        }

        let mut last_ordinal = None;
        let mut last_signal_by_queue = BTreeMap::<VulkanQueueId, u64>::new();
        let mut signal_index = BTreeMap::<(VulkanQueueId, u64), u32>::new();
        for submission in &self.submissions {
            if !seen_nodes.insert(submission.node_id) {
                return Err(VulkanSyncError::DuplicateSubmission(submission.node_id));
            }
            if assignment_map.get(&submission.node_id).copied() != Some(submission.queue) {
                return Err(VulkanSyncError::AssignmentMismatch(submission.node_id));
            }
            if let Some(previous) = last_ordinal {
                if submission.ordinal <= previous {
                    return Err(VulkanSyncError::SubmissionOrderMismatch);
                }
            }
            last_ordinal = Some(submission.ordinal);
            if submission.signal.queue != submission.queue || submission.signal.value == 0 {
                return Err(VulkanSyncError::InvalidSignal(submission.node_id));
            }
            if let Some(previous) = last_signal_by_queue.insert(submission.queue, submission.signal.value) {
                if submission.signal.value <= previous {
                    return Err(VulkanSyncError::NonMonotonicSignal(submission.queue));
                }
            }
            if signal_index.insert((submission.queue, submission.signal.value), submission.node_id).is_some() {
                return Err(VulkanSyncError::DuplicateSignal);
            }
            let mut waits = HashSet::with_capacity(submission.waits.len());
            for wait in &submission.waits {
                if wait.producer_queue == submission.queue {
                    return Err(VulkanSyncError::SameQueueWait(submission.node_id));
                }
                if wait.value == 0 {
                    return Err(VulkanSyncError::InvalidWait(submission.node_id));
                }
                if !waits.insert(wait) {
                    return Err(VulkanSyncError::DuplicateWait(submission.node_id));
                }
                if signal_index.get(&(wait.producer_queue, wait.value)) != Some(&wait.producer_node) {
                    return Err(VulkanSyncError::MissingSignal(wait.producer_node));
                }
            }
        }

        if seen_nodes.len() != self.assignments.len() {
            return Err(VulkanSyncError::InternalCardinalityMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VulkanSyncError {
    #[error("invalid execution schedule: {0}")]
    Schedule(ScheduleError),
    #[error("assignment count mismatch: expected {expected}, got {actual}")]
    AssignmentCountMismatch { expected: usize, actual: usize },
    #[error("missing queue assignment for node {0}")]
    MissingAssignment(u32),
    #[error("duplicate queue assignment for node {0}")]
    DuplicateAssignment(u32),
    #[error("queue id {0} exceeds the Vulkan logical-queue bound")]
    QueueLimitExceeded(u16),
    #[error("invalid node id")]
    InvalidNodeId,
    #[error("timeline value overflow")]
    TimelineValueOverflow,
    #[error("missing producer signal for node {0}")]
    MissingSignal(u32),
    #[error("wait table contains an unexpected target")]
    UnexpectedWaitTarget,
    #[error("unsupported Vulkan synchronization-plan version {0}")]
    UnsupportedVersion(u16),
    #[error("submission count {0} exceeds the graph bound")]
    SubmissionLimitExceeded(usize),
    #[error("submission/assignment cardinality mismatch")]
    InternalCardinalityMismatch,
    #[error("duplicate submission for node {0}")]
    DuplicateSubmission(u32),
    #[error("submission queue assignment mismatch for node {0}")]
    AssignmentMismatch(u32),
    #[error("submission order is not strictly increasing by semantic ordinal")]
    SubmissionOrderMismatch,
    #[error("invalid timeline signal for node {0}")]
    InvalidSignal(u32),
    #[error("timeline signal is not strictly increasing on queue {0:?}")]
    NonMonotonicSignal(VulkanQueueId),
    #[error("duplicate timeline signal")]
    DuplicateSignal,
    #[error("same-queue semaphore wait was emitted for node {0}")]
    SameQueueWait(u32),
    #[error("invalid timeline wait for node {0}")]
    InvalidWait(u32),
    #[error("duplicate timeline wait for node {0}")]
    DuplicateWait(u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccessKind, DependencyEdge, ExecutionGraph, ExecutionNode, GpuOperation, ResourceId, ResourceUse};

    fn resource() -> ResourceId { ResourceId::new("hv").unwrap() }

    fn node(id: u32, r: &ResourceId, access: AccessKind) -> ExecutionNode {
        ExecutionNode::new(id, GpuOperation::HdcBindXor { dimensions: 8 }, vec![ResourceUse::new(r.clone(), access)])
    }

    fn graph_chain() -> ExecutionGraph {
        let r = resource();
        ExecutionGraph::new(
            vec![node(1, &r, AccessKind::Write), node(2, &r, AccessKind::ReadWrite), node(3, &r, AccessKind::Read)],
            vec![
                DependencyEdge::new(1, 2, r.clone(), crate::DependencyKind::ReadAfterWrite),
                DependencyEdge::new(2, 3, r, crate::DependencyKind::ReadAfterWrite),
            ],
        ).unwrap()
    }

    #[test]
    fn single_queue_chain_has_ordered_signals_and_no_waits() {
        let schedule = ExecutionSchedule::from_graph(&graph_chain()).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let plan = VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q0 },
            VulkanQueueAssignment { node_id: 3, queue: q0 },
        ]).unwrap();
        assert_eq!(plan.queue_submission_order(q0).unwrap(), vec![1, 2, 3]);
        assert_eq!(plan.submissions.iter().map(|submission| submission.signal.value).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(plan.submissions.iter().all(|submission| submission.waits.is_empty()));
    }

    #[test]
    fn cross_queue_chain_uses_local_timeline_values() {
        let schedule = ExecutionSchedule::from_graph(&graph_chain()).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let q1 = VulkanQueueId::new(1).unwrap();
        let plan = VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q1 },
            VulkanQueueAssignment { node_id: 3, queue: q0 },
        ]).unwrap();
        assert!(plan.submissions[0].waits.is_empty());
        assert_eq!(plan.submissions[1].waits, vec![VulkanTimelineWait { producer_node: 1, producer_queue: q0, value: 1 }]);
        assert_eq!(plan.submissions[1].signal.value, 1);
        assert_eq!(plan.submissions[2].waits, vec![VulkanTimelineWait { producer_node: 2, producer_queue: q1, value: 1 }]);
        assert_eq!(plan.submissions[2].signal.value, 2);
    }

    #[test]
    fn assignment_insertion_order_does_not_change_digest() {
        let schedule = ExecutionSchedule::from_graph(&graph_chain()).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let q1 = VulkanQueueId::new(1).unwrap();
        let a = VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q1 },
            VulkanQueueAssignment { node_id: 3, queue: q0 },
        ]).unwrap();
        let b = VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 3, queue: q0 },
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q1 },
        ]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn parallel_queues_use_independent_local_timeline_values() {
        let r = resource();
        let graph = ExecutionGraph::new(
            vec![
                node(1, &r, AccessKind::Write),
                node(2, &r, AccessKind::Read),
                node(3, &r, AccessKind::Read),
            ],
            vec![
                DependencyEdge::new(1, 2, r.clone(), crate::DependencyKind::ReadAfterWrite),
                DependencyEdge::new(1, 3, r, crate::DependencyKind::ReadAfterWrite),
            ],
        ).unwrap();
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let q1 = VulkanQueueId::new(1).unwrap();
        let q2 = VulkanQueueId::new(2).unwrap();
        let plan = VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q1 },
            VulkanQueueAssignment { node_id: 3, queue: q2 },
        ]).unwrap();

        assert_eq!(plan.submissions[0].signal.value, 1);
        assert_eq!(plan.submissions[1].waits[0].value, 1);
        assert_eq!(plan.submissions[1].signal.value, 1);
        assert_eq!(plan.submissions[2].waits[0].value, 1);
        assert_eq!(plan.submissions[2].signal.value, 1);
    }

    #[test]
    fn incomplete_queue_assignment_is_rejected() {
        let schedule = ExecutionSchedule::from_graph(&graph_chain()).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let error = VulkanSyncPlan::from_schedule(
            &schedule,
            &[
                VulkanQueueAssignment { node_id: 1, queue: q0 },
                VulkanQueueAssignment { node_id: 2, queue: q0 },
            ],
        ).unwrap_err();
        assert!(matches!(error, VulkanSyncError::AssignmentCountMismatch { .. }));
    }

    #[test]
    fn non_monotonic_signal_is_rejected() {
        let schedule = ExecutionSchedule::from_graph(&graph_chain()).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let mut plan = VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q0 },
            VulkanQueueAssignment { node_id: 3, queue: q0 },
        ]).unwrap();
        plan.submissions[1].signal.value = 1;
        assert!(matches!(plan.digest(), Err(VulkanSyncError::NonMonotonicSignal(_))));
    }
}
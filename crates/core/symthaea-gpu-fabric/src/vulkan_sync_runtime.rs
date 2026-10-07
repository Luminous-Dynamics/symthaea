//! Real Vulkan synchronization qualification runtime.
//!
//! This module deliberately does not execute a Symthaea kernel. It binds the
//! backend-neutral `VulkanSyncPlan` to actual Vulkan 1.3 timeline semaphores and
//! `vkQueueSubmit2`, then observes completion from the host.
//!
//! The runtime uses one physical compute queue for all logical lanes. This is
//! intentionally conservative: it proves semaphore submission and completion
//! semantics without claiming that separate logical lanes are separate hardware
//! queues.

use std::ffi::CString;

use ash::{vk, Device, Entry, Instance};
use thiserror::Error;

use crate::{VulkanQueueId, VulkanSyncPlan};

const VULKAN_SYNC_API_VERSION: u32 = vk::API_VERSION_1_3;

#[derive(Debug, Error)]
pub enum VulkanSyncRuntimeError {
    #[error("Vulkan loader unavailable: {0}")]
    Loader(String),
    #[error("Vulkan call failed: {0:?}")]
    Vk(vk::Result),
    #[error("no Vulkan 1.3 compute device with synchronization2 and timeline semaphores")]
    NoQualifiedDevice,
    #[error("logical queue id {0} exceeds the runtime bound")]
    QueueLimitExceeded(u16),
    #[error("invalid synchronization plan: {0}")]
    Plan(crate::VulkanSyncError),
    #[error("missing logical queue {0}")]
    MissingQueue(VulkanQueueId),
    #[error("timeline value {value} for queue {queue:?} was not reached; observed {observed}")]
    CompletionNotReached { queue: VulkanQueueId, value: u64, observed: u64 },
    #[error("runtime produced an unexpected submission count")]
    SubmissionCountMismatch,
}

/// Concrete Vulkan runtime for synchronization-only qualification.
///
/// Every logical queue in the lowering plan is mapped onto this single actual
/// Vulkan queue. Cross-logical-queue waits still become real timeline waits.
pub struct VulkanSyncRuntime {
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    semaphores: Vec<(VulkanQueueId, vk::Semaphore)>,
}

impl VulkanSyncRuntime {
    pub fn new() -> Result<Self, VulkanSyncRuntimeError> {
        let entry = unsafe { Entry::load() }
            .map_err(|error| VulkanSyncRuntimeError::Loader(error.to_string()))?;

        let app_name = CString::new("symthaea-gpu-fabric-sync")
            .expect("static Vulkan application name has no NUL bytes");
        let engine_name = CString::new("Symthaea")
            .expect("static Vulkan engine name has no NUL bytes");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(VULKAN_SYNC_API_VERSION);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe {
            entry.create_instance(&instance_info, None)
                .map_err(VulkanSyncRuntimeError::Vk)?
        };

        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(VulkanSyncRuntimeError::Vk)?;

        let mut selected = None;
        for physical in devices {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            if properties.api_version < VULKAN_SYNC_API_VERSION {
                continue;
            }

            let mut timeline_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
            let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default();
            features2 = features2.push_next(&mut timeline_features).push_next(&mut sync2_features);
            unsafe { instance.get_physical_device_features2(physical, &mut features2) };
            if timeline_features.timeline_semaphore == 0 || sync2_features.synchronization2 == 0 {
                continue;
            }

            let queue_family = unsafe {
                instance.get_physical_device_queue_family_properties(physical)
            }
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32);

            if let Some(queue_family_index) = queue_family {
                selected = Some((physical, queue_family_index));
                break;
            }
        }

        let (physical, queue_family_index) = match selected {
            Some(value) => value,
            None => {
                unsafe { instance.destroy_instance(None) };
                return Err(VulkanSyncRuntimeError::NoQualifiedDevice);
            }
        };

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let mut timeline_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
            .timeline_semaphore(true);
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .push_next(&mut timeline_features)
            .push_next(&mut sync2_features);

        let device = unsafe {
            instance.create_device(physical, &device_info, None)
                .map_err(|error| {
                    instance.destroy_instance(None);
                    VulkanSyncRuntimeError::Vk(error)
                })?
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Self { instance, device, queue, semaphores: Vec::new() })
    }

    pub fn execute(&mut self, plan: &VulkanSyncPlan) -> Result<Vec<u64>, VulkanSyncRuntimeError> {
        plan.digest_hex().map_err(VulkanSyncRuntimeError::Plan)?;
        if plan.submissions.len() != plan.assignments.len() {
            return Err(VulkanSyncRuntimeError::SubmissionCountMismatch);
        }
        self.ensure_semaphores(plan.queue_count)?;

        for submission in &plan.submissions {
            let signal_semaphore = self
                .semaphore_for(submission.signal.queue)
                .ok_or(VulkanSyncRuntimeError::MissingQueue(submission.signal.queue))?;

            let mut wait_infos = Vec::with_capacity(submission.waits.len());
            for wait in &submission.waits {
                let semaphore = self
                    .semaphore_for(wait.producer_queue)
                    .ok_or(VulkanSyncRuntimeError::MissingQueue(wait.producer_queue))?;
                wait_infos.push(
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(semaphore)
                        .value(wait.value)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .device_index(0),
                );
            }

            let signal = vk::SemaphoreSubmitInfo::default()
                .semaphore(signal_semaphore)
                .value(submission.signal.value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .device_index(0);
            let submit = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(std::slice::from_ref(&signal));

            unsafe {
                self.device
                    .queue_submit2(self.queue, std::slice::from_ref(&submit), vk::Fence::null())
                    .map_err(VulkanSyncRuntimeError::Vk)?;
            }
        }

        let mut final_values = vec![0_u64; self.semaphores.len()];
        for (queue, _) in &self.semaphores {
            let final_value = plan
                .submissions
                .iter()
                .filter(|submission| submission.signal.queue == *queue)
                .map(|submission| submission.signal.value)
                .max()
                .unwrap_or(0);
            final_values[usize::from(queue.get())] = final_value;
        }

        let mut final_semaphores = Vec::new();
        let mut final_wait_values = Vec::new();
        for (index, (_, semaphore)) in self.semaphores.iter().enumerate() {
            if final_values[index] != 0 {
                final_semaphores.push(*semaphore);
                final_wait_values.push(final_values[index]);
            }
        }

        if !final_semaphores.is_empty() {
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&final_semaphores)
                .values(&final_wait_values);
            unsafe {
                self.device
                    .wait_semaphores(&wait_info, u64::MAX)
                    .map_err(VulkanSyncRuntimeError::Vk)?;
            }
        }

        for (index, (queue, semaphore)) in self.semaphores.iter().enumerate() {
            let expected = final_values.get(index).copied().unwrap_or(0);
            if expected == 0 {
                continue;
            }
            let observed = unsafe {
                self.device
                    .get_semaphore_counter_value(*semaphore)
                    .map_err(VulkanSyncRuntimeError::Vk)?
            };
            if observed < expected {
                return Err(VulkanSyncRuntimeError::CompletionNotReached {
                    queue: *queue,
                    value: expected,
                    observed,
                });
            }
        }

        Ok(final_values)
    }

    fn ensure_semaphores(&mut self, queue_count: u16) -> Result<(), VulkanSyncRuntimeError> {
        while self.semaphores.len() < usize::from(queue_count) {
            let queue = VulkanQueueId::new(self.semaphores.len() as u16)
                .map_err(|error| VulkanSyncRuntimeError::QueueLimitExceeded(error.get()))?;
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            let semaphore = unsafe {
                self.device
                    .create_semaphore(&create_info, None)
                    .map_err(VulkanSyncRuntimeError::Vk)?
            };
            self.semaphores.push((queue, semaphore));
        }
        Ok(())
    }

    fn semaphore_for(&self, queue: VulkanQueueId) -> Option<vk::Semaphore> {
        self.semaphores
            .iter()
            .find(|(candidate, _)| *candidate == queue)
            .map(|(_, semaphore)| *semaphore)
    }
}

impl Drop for VulkanSyncRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for (_, semaphore) in &self.semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccessKind, DependencyEdge, ExecutionGraph, ExecutionNode, GpuOperation,
        ResourceId, ResourceUse, VulkanQueueAssignment, VulkanSyncPlan,
    };

    fn test_plan() -> VulkanSyncPlan {
        let r = ResourceId::new("hv").unwrap();
        let graph = ExecutionGraph::new(
            vec![
                ExecutionNode::new(1, GpuOperation::HdcBindXor { dimensions: 8 }, vec![ResourceUse::new(r.clone(), AccessKind::Write)]),
                ExecutionNode::new(2, GpuOperation::HdcBindXor { dimensions: 8 }, vec![ResourceUse::new(r.clone(), AccessKind::Read)]),
            ],
            vec![DependencyEdge::new(1, 2, r, crate::DependencyKind::ReadAfterWrite)],
        ).unwrap();
        let schedule = crate::ExecutionSchedule::from_graph(&graph).unwrap();
        let q0 = VulkanQueueId::new(0).unwrap();
        let q1 = VulkanQueueId::new(1).unwrap();
        VulkanSyncPlan::from_schedule(&schedule, &[
            VulkanQueueAssignment { node_id: 1, queue: q0 },
            VulkanQueueAssignment { node_id: 2, queue: q1 },
        ]).unwrap()
    }

    #[test]
    #[ignore = "requires a Vulkan 1.3 qualification runner"]
    fn real_vulkan_timeline_submission_reaches_completion() {
        let mut runtime = VulkanSyncRuntime::new().expect("qualified Vulkan 1.3 device");
        let plan = test_plan();
        let final_values = runtime.execute(&plan).expect("timeline submissions complete");
        assert_eq!(final_values.len(), 2);
        assert_eq!(final_values[0], 1);
        assert_eq!(final_values[1], 1);
    }
}
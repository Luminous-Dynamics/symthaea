use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::ptr;

use ash::{vk, Device, Entry, Instance};
use blake3::Hasher;
use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AccessKind, BinaryHypervector, DependencyKind, ExecutionGraph, ExecutionNode,
    ExecutionSchedule, GpuOperation, ResourceId, VulkanBarrierRequirement, VulkanSyncPlan,
};

const MAX_WORKLOAD_NODES: usize = 64;
const WORKGROUP_SIZE: u32 = 64;
const VULKAN_API_VERSION: u32 = vk::API_VERSION_1_3;
const VULKAN_TIMELINE_TIMEOUT_NS: u64 = 5_000_000_000;
const RECEIPT_VERSION: u16 = 3;

const WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> lhs: array<u32>;
@group(0) @binding(1)
var<storage, read> rhs: array<u32>;
@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output)) {
        output[id.x] = lhs[id.x] ^ rhs[id.x];
    }
}
"#;

#[derive(Debug, Error)]
pub enum VulkanBarrierError {
    #[error("Vulkan loader unavailable: {0}")]
    Loader(String),
    #[error("Vulkan call failed: {0:?}")]
    Vk(vk::Result),
    #[error("no Vulkan 1.3 compute device with synchronization2")]
    NoQualifiedDevice,
    #[error("workload has {0} nodes; bound is {MAX_WORKLOAD_NODES}")]
    WorkloadNodeLimit(usize),
    #[error("multiple logical queues are not supported by the single-queue workload runtime")]
    MultipleLogicalQueues,
    #[error("invalid semantic schedule: {0}")]
    Schedule(crate::ScheduleError),
    #[error("invalid execution graph: {0}")]
    Graph(crate::GraphError),
    #[error("non-canonical Vulkan synchronization plan: {0}")]
    SyncPlan(crate::VulkanSyncError),
    #[error("resource {0} has no initial binding")]
    MissingResource(ResourceId),
    #[error("initial resource binding {0} is not referenced by the execution graph")]
    UnexpectedResource(ResourceId),
    #[error("resource {resource} has dimensions {actual}; expected {expected}")]
    ResourceDimensions { resource: ResourceId, actual: u32, expected: u32 },
    #[error("node {0} has unsupported HDC-XOR resource shape")]
    UnsupportedNodeShape(u32),
    #[error("resource {0} is too large for this Vulkan device")]
    ResourceTooLarge(ResourceId),
    #[error("resource {0} exceeds the compute dispatch limit")]
    DispatchTooLarge(ResourceId),
    #[error("no host-visible memory type")]
    NoHostVisibleMemory,
    #[error("allocation size overflow")]
    AllocationOverflow,
    #[error("CPU oracle mismatch for resource {0}")]
    OracleMismatch(ResourceId),
    #[error("timeline semaphore creation failed: {0:?}")]
    TimelineSemaphoreCreate(vk::Result),
    #[error("timeline queue submission failed: {0:?}")]
    TimelineSubmit(vk::Result),
    #[error("timeline semaphore wait failed: {0:?}")]
    TimelineWait(vk::Result),
    #[error("timeline semaphore counter query failed: {0:?}")]
    TimelineCounter(vk::Result),
    #[error("timeline completion did not reach {expected}; observed {observed}")]
    TimelineCompletionNotReached { expected: u64, observed: u64 },
    #[error("barrier receipt verification failed: {0}")]
    Receipt(VulkanBarrierReceiptError),
}

#[derive(Debug, Error)]
pub enum VulkanBarrierReceiptError {
    #[error("unsupported receipt version {0}")]
    Version(u16),
    #[error("graph digest mismatch")]
    GraphDigest,
    #[error("schedule digest mismatch")]
    ScheduleDigest,
    #[error("sync plan digest mismatch")]
    SyncPlanDigest,
    #[error("barrier digest mismatch")]
    BarrierDigest,
    #[error("node count mismatch")]
    NodeCount,
    #[error("barrier count mismatch")]
    BarrierCount,
    #[error("resource count mismatch")]
    ResourceCount,
    #[error("resource digest mismatch for {0}")]
    ResourceDigest(ResourceId),
    #[error("resource storage size mismatch for {0}")]
    ResourceStorageSize(ResourceId),
    #[error("missing concrete storage size for barrier resource {0}")]
    MissingResourceStorageSize(ResourceId),
    #[error("receipt Vulkan API version does not match the qualified runtime")]
    ApiVersion,
    #[error("receipt physical-device API version is below the qualified Vulkan API version")]
    PhysicalDeviceApiVersion,
    #[error("receipt physical-device API version does not match the execution runtime")]
    PhysicalDeviceApiVersionBinding,
    #[error("receipt queue family does not match the execution runtime")]
    QueueFamilyBinding,
    #[error("receipt expected timeline value does not match the synchronization plan")]
    TimelineExpected,
    #[error("receipt uses an unsupported multi-queue synchronization plan")]
    MultipleLogicalQueues,
    #[error("receipt completion lowering digest mismatch")]
    CompletionLoweringDigest,
    #[error("receipt observed timeline value {observed} does not equal expected {expected}")]
    TimelineCompletion { expected: u64, observed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanBarrierExecutionReceipt {
    pub version: u16,
    pub graph_digest: String,
    pub schedule_digest: String,
    pub sync_plan_digest: String,
    pub barrier_digest: String,
    pub barrier_lowering_digest: String,
    pub completion_lowering_digest: String,
    pub node_count: u32,
    pub barrier_count: u32,
    pub resource_digests: BTreeMap<ResourceId, String>,
    pub resource_storage_sizes: BTreeMap<ResourceId, u64>,
    pub completion_expected: u64,
    pub completion_observed: u64,
    pub vulkan_api_version: u32,
    pub physical_device_api_version: u32,
    pub queue_family_index: u32,
}

impl VulkanBarrierExecutionReceipt {
    pub fn verify_against(
        &self,
        graph: &ExecutionGraph,
        schedule: &ExecutionSchedule,
        plan: &VulkanSyncPlan,
        final_resources: &BTreeMap<ResourceId, BinaryHypervector>,
    ) -> Result<(), VulkanBarrierReceiptError> {
        if self.version != RECEIPT_VERSION { return Err(VulkanBarrierReceiptError::Version(self.version)); }
        if self.graph_digest != graph.digest_hex().map_err(|_| VulkanBarrierReceiptError::GraphDigest)? {
            return Err(VulkanBarrierReceiptError::GraphDigest);
        }
        if self.schedule_digest != schedule.digest_hex().map_err(|_| VulkanBarrierReceiptError::ScheduleDigest)? {
            return Err(VulkanBarrierReceiptError::ScheduleDigest);
        }
        if self.sync_plan_digest != plan.digest_hex().map_err(|_| VulkanBarrierReceiptError::SyncPlanDigest)? {
            return Err(VulkanBarrierReceiptError::SyncPlanDigest);
        }
        if self.barrier_digest != barrier_digest(plan) { return Err(VulkanBarrierReceiptError::BarrierDigest); }
        let expected_storage_sizes = final_resources
            .iter()
            .map(|(resource, value)| {
                (
                    resource.clone(),
                    rounded_storage_bytes(value.as_bytes().len() as u64),
                )
            })
            .collect::<BTreeMap<_, _>>();
        if self.resource_storage_sizes != expected_storage_sizes {
            let resource = self
                .resource_storage_sizes
                .keys()
                .chain(expected_storage_sizes.keys())
                .find(|resource| {
                    self.resource_storage_sizes.get(*resource)
                        != expected_storage_sizes.get(*resource)
                })
                .cloned();
            return match resource {
                Some(resource) => Err(VulkanBarrierReceiptError::ResourceStorageSize(resource)),
                None => Err(VulkanBarrierReceiptError::ResourceCount),
            };
        }
        if self.barrier_lowering_digest
            != barrier_lowering_digest(plan, &expected_storage_sizes)
                .map_err(|resource| VulkanBarrierReceiptError::MissingResourceStorageSize(resource))?
        {
            return Err(VulkanBarrierReceiptError::BarrierDigest);
        }
        if self.node_count != schedule.nodes.len() as u32 { return Err(VulkanBarrierReceiptError::NodeCount); }
        let count = plan.submissions.iter().map(|s| s.barriers.len() as u32).sum::<u32>();
        if self.barrier_count != count { return Err(VulkanBarrierReceiptError::BarrierCount); }
        if self.resource_digests.len() != final_resources.len() {
            return Err(VulkanBarrierReceiptError::ResourceCount);
        }
        for (resource, digest) in &self.resource_digests {
            let actual = final_resources.get(resource).map(resource_digest)
                .ok_or_else(|| VulkanBarrierReceiptError::ResourceDigest(resource.clone()))?;
            if &actual != digest { return Err(VulkanBarrierReceiptError::ResourceDigest(resource.clone())); }
        }
        if self.vulkan_api_version != VULKAN_API_VERSION {
            return Err(VulkanBarrierReceiptError::ApiVersion);
        }
        if self.physical_device_api_version < VULKAN_API_VERSION {
            return Err(VulkanBarrierReceiptError::PhysicalDeviceApiVersion);
        }
        if plan.queue_count > 1 || plan.assignments.iter().any(|assignment| assignment.queue.get() != 0) {
            return Err(VulkanBarrierReceiptError::MultipleLogicalQueues);
        }
        let expected_completion = expected_final_timeline_value(plan);
        if self.completion_expected != expected_completion {
            return Err(VulkanBarrierReceiptError::TimelineExpected);
        }
        if self.completion_lowering_digest
            != completion_lowering_digest(plan, expected_completion, self.queue_family_index)
        {
            return Err(VulkanBarrierReceiptError::CompletionLoweringDigest);
        }
        if self.completion_observed != self.completion_expected {
            return Err(VulkanBarrierReceiptError::TimelineCompletion {
                expected: self.completion_expected,
                observed: self.completion_observed,
            });
        }
        Ok(())
    }

    pub fn verify_runtime_binding(
        &self,
        physical_device_api_version: u32,
        queue_family_index: u32,
    ) -> Result<(), VulkanBarrierReceiptError> {
        if self.physical_device_api_version != physical_device_api_version {
            return Err(VulkanBarrierReceiptError::PhysicalDeviceApiVersionBinding);
        }
        if self.queue_family_index != queue_family_index {
            return Err(VulkanBarrierReceiptError::QueueFamilyBinding);
        }
        Ok(())
    }
}

pub struct VulkanBarrierWorkloadRuntime {
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_storage_buffer_range: u64,
    max_compute_workgroup_count_x: u32,
    physical_device_api_version: u32,
    queue_family_index: u32,
}

impl VulkanBarrierWorkloadRuntime {
    pub fn new() -> Result<Self, VulkanBarrierError> {
        let entry = unsafe { Entry::load() }.map_err(|e| VulkanBarrierError::Loader(e.to_string()))?;
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(VulkanBarrierError::Vk)?.unwrap_or(vk::API_VERSION_1_0);
        if loader_version < VULKAN_API_VERSION { return Err(VulkanBarrierError::NoQualifiedDevice); }

        let app = CString::new("symthaea-gpu-fabric-barrier").unwrap();
        let engine = CString::new("Symthaea").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app).application_version(1)
            .engine_name(&engine).engine_version(1)
            .api_version(VULKAN_API_VERSION);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None).map_err(VulkanBarrierError::Vk)? };

        let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(error) => {
                unsafe { instance.destroy_instance(None); }
                return Err(VulkanBarrierError::Vk(error));
            }
        };

        let mut selected = None;
        for physical in physical_devices {
            let props = unsafe { instance.get_physical_device_properties(physical) };
            if props.api_version < VULKAN_API_VERSION { continue; }
            let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
            let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut timeline)
                .push_next(&mut sync2);
            unsafe { instance.get_physical_device_features2(physical, &mut features2); }
            if timeline.timeline_semaphore == 0 || sync2.synchronization2 == 0 { continue; }
            let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
                .iter().enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32);
            if let Some(family) = family { selected = Some((physical, family)); break; }
        }

        let (physical, family) = match selected {
            Some(value) => value,
            None => {
                unsafe { instance.destroy_instance(None); }
                return Err(VulkanBarrierError::NoQualifiedDevice);
            }
        };
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities);
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .push_next(&mut timeline)
            .push_next(&mut sync2);
        let device = unsafe {
            instance.create_device(physical, &device_info, None).map_err(|e| {
                instance.destroy_instance(None);
                VulkanBarrierError::Vk(e)
            })?
        };
        let queue = unsafe { device.get_device_queue(family, 0) };

        let spirv = compile_spirv().map_err(|error| {
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            error
        })?;
        let shader = match create_shader_module(&device, &spirv) {
            Ok(shader) => shader,
            Err(error) => {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(error);
            }
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_layout = match unsafe {
            device.create_descriptor_set_layout(&layout_info, None)
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(shader, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanBarrierError::Vk(error));
            }
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_layout));
        let pipeline_layout = match unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                    device.destroy_shader_module(shader, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanBarrierError::Vk(error));
            }
        };
        let entry_point = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(&entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
        let pipeline = unsafe {
            match device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            ) {
                Ok(mut pipelines) => match pipelines.pop() {
                    Some(pipeline) => pipeline,
                    None => {
                        device.destroy_pipeline_layout(pipeline_layout, None);
                        device.destroy_descriptor_set_layout(descriptor_layout, None);
                        device.destroy_shader_module(shader, None);
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                        return Err(VulkanBarrierError::AllocationOverflow);
                    }
                },
                Err((pipelines, error)) => {
                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                    device.destroy_shader_module(shader, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                    return Err(VulkanBarrierError::Vk(error));
                }
            }
        };
        let command_pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        let command_pool = match unsafe { device.create_command_pool(&command_pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                    device.destroy_shader_module(shader, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanBarrierError::Vk(error));
            }
        };
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count((MAX_WORKLOAD_NODES * 3) as u32);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(MAX_WORKLOAD_NODES as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                    device.destroy_shader_module(shader, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanBarrierError::Vk(error));
            }
        };

        Ok(Self {
            instance, device, queue, command_pool, descriptor_layout, descriptor_pool,
            pipeline_layout, pipeline, shader, memory_properties,
            max_storage_buffer_range: u64::from(props.limits.max_storage_buffer_range),
            max_compute_workgroup_count_x: props.limits.max_compute_work_group_count[0],
            physical_device_api_version: props.api_version,
            queue_family_index: family,
        })
    }

    pub fn execute_verified(
        &self,
        graph: &ExecutionGraph,
        schedule: &ExecutionSchedule,
        plan: &VulkanSyncPlan,
        initial: &BTreeMap<ResourceId, BinaryHypervector>,
    ) -> Result<(BTreeMap<ResourceId, BinaryHypervector>, VulkanBarrierExecutionReceipt), VulkanBarrierError> {
        schedule.verify_against(graph).map_err(VulkanBarrierError::Schedule)?;
        plan.verify_against_schedule(schedule).map_err(VulkanBarrierError::SyncPlan)?;
        if schedule.nodes.len() > MAX_WORKLOAD_NODES { return Err(VulkanBarrierError::WorkloadNodeLimit(schedule.nodes.len())); }
        if plan.queue_count > 1 || plan.assignments.iter().any(|a| a.queue.get() != 0) { return Err(VulkanBarrierError::MultipleLogicalQueues); }

        let _resources = validate_initial_resources(graph, initial)?;
        let expected = simulate(graph, schedule, initial)?;
        let mut buffers = BTreeMap::new();
        for (resource, value) in initial {
            let physical = rounded_storage_bytes(value.as_bytes().len() as u64);
            if physical > self.max_storage_buffer_range { return Err(VulkanBarrierError::ResourceTooLarge(resource.clone())); }
            buffers.insert(resource.clone(), WorkloadBuffer::new(&self.device, &self.memory_properties, physical)?);
        }
        for (resource, value) in initial { buffers[resource].write(&self.device, value.as_bytes())?; }

        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe {
            self.device
                .allocate_command_buffers(&command_info)
                .map_err(VulkanBarrierError::Vk)?
                .into_iter()
                .next()
                .ok_or(VulkanBarrierError::AllocationOverflow)?
        };
        let command_guard = CommandBufferGuard::new(
            self.device.clone(),
            self.command_pool,
            command,
        );

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_guard.command(), &begin)
                .map_err(VulkanBarrierError::Vk)?;
        }

        let mut set_guard =
            DescriptorSetGuard::new(self.device.clone(), self.descriptor_pool);
        for scheduled in &schedule.nodes {
            let node = graph
                .nodes
                .iter()
                .find(|n| n.id == scheduled.id)
                .ok_or(VulkanBarrierError::UnsupportedNodeShape(scheduled.id))?;
            let (reads, writes) = canonical_workload_resources(node);
            if reads.len() != 2 || writes.len() != 1 || node.resources.len() != 3 {
                return Err(VulkanBarrierError::UnsupportedNodeShape(node.id));
            }

            let submission = plan
                .submissions
                .iter()
                .find(|s| s.node_id == node.id)
                .ok_or(VulkanBarrierError::UnsupportedNodeShape(node.id))?;

            record_barriers(
                &self.device,
                command_guard.command(),
                &submission.barriers,
                &buffers,
            )?;

            let range = buffers[&writes[0].resource].storage_size;
            let set = allocate_set(
                &self.device,
                self.descriptor_pool,
                self.descriptor_layout,
                [
                    &buffers[&reads[0].resource],
                    &buffers[&reads[1].resource],
                    &buffers[&writes[0].resource],
                ],
                range,
            )?;
            set_guard.push(set);

            let groups = ((range / 4) as u32)
                .saturating_add(WORKGROUP_SIZE - 1)
                / WORKGROUP_SIZE;
            if groups.max(1) > self.max_compute_workgroup_count_x {
                return Err(VulkanBarrierError::DispatchTooLarge(
                    writes[0].resource.clone(),
                ));
            }
            unsafe {
                self.device.cmd_bind_pipeline(
                    command_guard.command(),
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command_guard.command(),
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
                self.device
                    .cmd_dispatch(command_guard.command(), groups.max(1), 1, 1);
            }
        }

        unsafe {
            self.device
                .end_command_buffer(command_guard.command())
                .map_err(VulkanBarrierError::Vk)?;
        }
        let completion_expected = expected_final_timeline_value(plan);
        if completion_expected == 0 && !schedule.nodes.is_empty() {
            return Err(VulkanBarrierError::TimelineCompletionNotReached {
                expected: 1,
                observed: 0,
            });
        }

        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let semaphore = unsafe {
            self.device
                .create_semaphore(&semaphore_info, None)
                .map_err(VulkanBarrierError::TimelineSemaphoreCreate)?
        };
        let mut semaphore_guard = TimelineSemaphoreGuard::new(self.device.clone(), semaphore);

        let command_buffer_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(command_guard.command())
            .device_mask(1);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .value(completion_expected)
            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .device_index(0);
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&command_buffer_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        unsafe {
            self.device
                .queue_submit2(self.queue, std::slice::from_ref(&submit), vk::Fence::null())
                .map_err(VulkanBarrierError::TimelineSubmit)?;
        }
        semaphore_guard.mark_submitted();

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&semaphore))
            .values(std::slice::from_ref(&completion_expected));
        unsafe {
            self.device
                .wait_semaphores(&wait_info, VULKAN_TIMELINE_TIMEOUT_NS)
                .map_err(VulkanBarrierError::TimelineWait)?;
        }
        semaphore_guard.mark_completed();

        let completion_observed = unsafe {
            self.device
                .get_semaphore_counter_value(semaphore)
                .map_err(VulkanBarrierError::TimelineCounter)?
        };
        if completion_observed != completion_expected {
            return Err(VulkanBarrierError::TimelineCompletionNotReached {
                expected: completion_expected,
                observed: completion_observed,
            });
        }

        let mut observed = BTreeMap::new();
        for (resource, value) in initial {
            let bytes = buffers[resource].read(&self.device, value.as_bytes().len())?;
            observed.insert(resource.clone(), BinaryHypervector::from_bytes(value.dimensions, bytes)
                .map_err(|_| VulkanBarrierError::OracleMismatch(resource.clone()))?);
        }
        for (resource, expected_value) in &expected {
            if observed.get(resource) != Some(expected_value) { return Err(VulkanBarrierError::OracleMismatch(resource.clone())); }
        }
        let mut digests = BTreeMap::new();
        let mut storage_sizes = BTreeMap::new();
        for (resource, value) in &observed {
            digests.insert(resource.clone(), resource_digest(value));
            storage_sizes.insert(resource.clone(), buffers[resource].storage_size);
        }
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().map_err(VulkanBarrierError::Graph)?,
            schedule_digest: schedule.digest_hex().map_err(VulkanBarrierError::Schedule)?,
            sync_plan_digest: plan.digest_hex().map_err(|e| VulkanBarrierError::SyncPlan(e))?,
            barrier_digest: barrier_digest(plan),
            barrier_lowering_digest: barrier_lowering_digest(plan, &storage_sizes)
                .map_err(|resource| {
                    VulkanBarrierError::Receipt(
                        VulkanBarrierReceiptError::MissingResourceStorageSize(resource),
                    )
                })?,
            completion_lowering_digest: completion_lowering_digest(
                plan,
                completion_expected,
                self.queue_family_index,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: plan.submissions.iter().map(|s| s.barriers.len() as u32).sum(),
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected,
            completion_observed,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: self.physical_device_api_version,
            queue_family_index: self.queue_family_index,
        };
        receipt.verify_against(graph, schedule, plan, &observed).map_err(VulkanBarrierError::Receipt)?;
        receipt
            .verify_runtime_binding(self.physical_device_api_version, self.queue_family_index)
            .map_err(VulkanBarrierError::Receipt)?;
        Ok((observed, receipt))
    }
}

fn validate_initial_resources(
    graph: &ExecutionGraph,
    initial: &BTreeMap<ResourceId, BinaryHypervector>,
) -> Result<BTreeSet<ResourceId>, VulkanBarrierError> {
    let resources = graph
        .nodes
        .iter()
        .flat_map(|node| node.resources.iter().map(|use_| use_.resource.clone()))
        .collect::<BTreeSet<_>>();

    for resource in &resources {
        if !initial.contains_key(resource) {
            return Err(VulkanBarrierError::MissingResource(resource.clone()));
        }
    }
    if let Some(extra) = initial.keys().find(|resource| !resources.contains(*resource)) {
        return Err(VulkanBarrierError::UnexpectedResource(extra.clone()));
    }

    Ok(resources)
}

fn canonical_workload_resources(
    node: &ExecutionNode,
) -> (Vec<&crate::ResourceUse>, Vec<&crate::ResourceUse>) {
    let mut reads = node
        .resources
        .iter()
        .filter(|use_| use_.access == AccessKind::Read)
        .collect::<Vec<_>>();
    let mut writes = node
        .resources
        .iter()
        .filter(|use_| use_.access == AccessKind::Write)
        .collect::<Vec<_>>();

    reads.sort_by(|left, right| left.resource.cmp(&right.resource));
    writes.sort_by(|left, right| left.resource.cmp(&right.resource));

    (reads, writes)
}

fn simulate(
    graph: &ExecutionGraph,
    schedule: &ExecutionSchedule,
    initial: &BTreeMap<ResourceId, BinaryHypervector>,
) -> Result<BTreeMap<ResourceId, BinaryHypervector>, VulkanBarrierError> {
    let mut state = initial.clone();
    for scheduled in &schedule.nodes {
        let node = graph.nodes.iter().find(|n| n.id == scheduled.id).ok_or(VulkanBarrierError::UnsupportedNodeShape(scheduled.id))?;
        let (reads, writes) = canonical_workload_resources(node);
        if reads.len() != 2 || writes.len() != 1 {
            return Err(VulkanBarrierError::UnsupportedNodeShape(node.id));
        }
        let dimensions = match node.operation { GpuOperation::HdcBindXor { dimensions } => dimensions };
        let lhs = state.get(&reads[0].resource)
            .ok_or_else(|| VulkanBarrierError::MissingResource(reads[0].resource.clone()))?;
        let rhs = state.get(&reads[1].resource)
            .ok_or_else(|| VulkanBarrierError::MissingResource(reads[1].resource.clone()))?;
        let output = state.get(&writes[0].resource)
            .ok_or_else(|| VulkanBarrierError::MissingResource(writes[0].resource.clone()))?;
        if lhs.dimensions != dimensions || rhs.dimensions != dimensions || output.dimensions != dimensions {
            return Err(VulkanBarrierError::ResourceDimensions {
                resource: writes[0].resource.clone(),
                actual: output.dimensions,
                expected: dimensions,
            });
        }
        let bytes = lhs
            .as_bytes()
            .iter()
            .zip(rhs.as_bytes())
            .map(|(a, b)| a ^ b)
            .collect::<Vec<_>>();
        state.insert(writes[0].resource.clone(), BinaryHypervector::from_bytes(dimensions, bytes).map_err(|_| VulkanBarrierError::OracleMismatch(writes[0].resource.clone()))?);
    }
    Ok(state)
}

fn record_barriers(
    device: &Device,
    command: vk::CommandBuffer,
    requirements: &[VulkanBarrierRequirement],
    buffers: &BTreeMap<ResourceId, WorkloadBuffer>,
) -> Result<(), VulkanBarrierError> {
    if requirements.is_empty() { return Ok(()); }
    let mut buffer_barriers = Vec::new();
    let mut execution_barriers = Vec::new();
    for req in requirements {
        if req.requires_memory_dependency() {
            let buffer = buffers.get(&req.resource).ok_or(VulkanBarrierError::MissingResource(req.resource.clone()))?;
            let (src_access, dst_access) = barrier_access_masks(req.kind);
            buffer_barriers.push(vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buffer.buffer)
                .offset(0).size(buffer.storage_size));
        } else {
            execution_barriers.push(vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::empty()));
        }
    }
    let dependency = vk::DependencyInfo::default().memory_barriers(&execution_barriers).buffer_memory_barriers(&buffer_barriers);
    unsafe { device.cmd_pipeline_barrier2(command, &dependency); }
    Ok(())
}

struct CommandBufferGuard {
    device: Device,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
}

impl CommandBufferGuard {
    fn new(device: Device, pool: vk::CommandPool, command: vk::CommandBuffer) -> Self {
        Self { device, pool, command }
    }

    fn command(&self) -> vk::CommandBuffer {
        self.command
    }
}

impl Drop for CommandBufferGuard {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(self.pool, std::slice::from_ref(&self.command));
        }
    }
}

struct DescriptorSetGuard {
    device: Device,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl DescriptorSetGuard {
    fn new(device: Device, pool: vk::DescriptorPool) -> Self {
        Self { device, pool, sets: Vec::new() }
    }

    fn push(&mut self, set: vk::DescriptorSet) {
        self.sets.push(set);
    }
}

impl Drop for DescriptorSetGuard {
    fn drop(&mut self) {
        if self.sets.is_empty() {
            return;
        }
        unsafe {
            let _ = self.device.free_descriptor_sets(self.pool, &self.sets);
        }
    }
}

struct WorkloadBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocation_size: vk::DeviceSize,
    storage_size: vk::DeviceSize,
    coherent: bool,
}

impl WorkloadBuffer {
    fn new(device: &Device, props: &vk::PhysicalDeviceMemoryProperties, size: u64) -> Result<Self, VulkanBarrierError> {
        let info = vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None).map_err(VulkanBarrierError::Vk)? };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut selected = None;
        for coherent in [true, false] {
            for i in 0..props.memory_type_count as usize {
                let flags = props.memory_types[i].property_flags;
                if (req.memory_type_bits & (1_u32 << i)) != 0 && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) && (!coherent || flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)) {
                    selected = Some((i as u32, flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)));
                    break;
                }
            }
            if selected.is_some() { break; }
        }
        let (index, coherent) = match selected {
            Some(value) => value,
            None => {
                unsafe { device.destroy_buffer(buffer, None); }
                return Err(VulkanBarrierError::NoHostVisibleMemory);
            }
        };
        let alloc = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index);
        let memory = match unsafe { device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(error) => { unsafe { device.destroy_buffer(buffer, None); } return Err(VulkanBarrierError::Vk(error)); }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe { device.free_memory(memory, None); device.destroy_buffer(buffer, None); }
            return Err(VulkanBarrierError::Vk(error));
        }
        Ok(Self { device: device.clone(), buffer, memory, allocation_size: req.size, storage_size: size, coherent })
    }

    fn write(&self, device: &Device, bytes: &[u8]) -> Result<(), VulkanBarrierError> {
        if bytes.len() as u64 > self.allocation_size { return Err(VulkanBarrierError::AllocationOverflow); }
        let mapped = unsafe { device.map_memory(self.memory, 0, self.allocation_size, vk::MemoryMapFlags::empty()).map_err(VulkanBarrierError::Vk)? };
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            if bytes.len() < self.allocation_size as usize { ptr::write_bytes(mapped.cast::<u8>().add(bytes.len()), 0, self.allocation_size as usize - bytes.len()); }
            if !self.coherent {
                let range = vk::MappedMemoryRange::default().memory(self.memory).offset(0).size(vk::WHOLE_SIZE);
                if let Err(error) = device.flush_mapped_memory_ranges(std::slice::from_ref(&range)) {
                    device.unmap_memory(self.memory);
                    return Err(VulkanBarrierError::Vk(error));
                }
            }
            device.unmap_memory(self.memory);
        }
        Ok(())
    }

    fn read(&self, device: &Device, len: usize) -> Result<Vec<u8>, VulkanBarrierError> {
        if len as u64 > self.allocation_size { return Err(VulkanBarrierError::AllocationOverflow); }
        let mapped = unsafe { device.map_memory(self.memory, 0, self.allocation_size, vk::MemoryMapFlags::empty()).map_err(VulkanBarrierError::Vk)? };
        if !self.coherent {
            let range = vk::MappedMemoryRange::default().memory(self.memory).offset(0).size(vk::WHOLE_SIZE);
            if let Err(error) = unsafe { device.invalidate_mapped_memory_ranges(std::slice::from_ref(&range)) } {
                unsafe { device.unmap_memory(self.memory); }
                return Err(VulkanBarrierError::Vk(error));
            }
        }
        let mut bytes = vec![0_u8; len];
        unsafe { ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), len); device.unmap_memory(self.memory); }
        Ok(bytes)
    }
}

impl Drop for WorkloadBuffer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_buffer(self.buffer, None); self.device.free_memory(self.memory, None); }
    }
}

fn allocate_set(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    buffers: [&WorkloadBuffer; 3],
    range: u64,
) -> Result<vk::DescriptorSet, VulkanBarrierError> {
    let info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&layout));
    let set = unsafe {
        match device
            .allocate_descriptor_sets(&info)
            .map_err(VulkanBarrierError::Vk)?
            .into_iter()
            .next()
        {
            Some(set) => set,
            None => return Err(VulkanBarrierError::AllocationOverflow),
        }
    };
    let infos = [
        vk::DescriptorBufferInfo::default().buffer(buffers[0].buffer).offset(0).range(range),
        vk::DescriptorBufferInfo::default().buffer(buffers[1].buffer).offset(0).range(range),
        vk::DescriptorBufferInfo::default().buffer(buffers[2].buffer).offset(0).range(range),
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[0])),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[1])),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[2])),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]); }
    Ok(set)
}

fn compile_spirv() -> Result<Vec<u32>, VulkanBarrierError> {
    let module = wgsl::parse_str(WGSL).map_err(|e| VulkanBarrierError::Loader(e.to_string()))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).map_err(|e| VulkanBarrierError::Loader(e.to_string()))?;
    let options = spv::Options::default();
    let pipeline = spv::PipelineOptions { entry_point: "main".into(), shader_stage: naga::ShaderStage::Compute };
    spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|e| VulkanBarrierError::Loader(e.to_string()))
}

fn create_shader_module(device: &Device, spirv: &[u32]) -> Result<vk::ShaderModule, VulkanBarrierError> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    unsafe { device.create_shader_module(&info, None).map_err(VulkanBarrierError::Vk) }
}

fn rounded_storage_bytes(bytes: u64) -> u64 { (bytes.saturating_add(3) / 4) * 4 }

fn resource_digest(value: &BinaryHypervector) -> String {
    let mut h = Hasher::new();
    h.update(b"symthaea.gpu-fabric.vulkan-barrier-resource.v1\0");
    h.update(&value.dimensions.to_le_bytes());
    h.update(value.as_bytes());
    h.finalize().to_hex().to_string()
}

fn barrier_access_masks(kind: DependencyKind) -> (vk::AccessFlags2, vk::AccessFlags2) {
    match kind {
        DependencyKind::ReadAfterWrite => (
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        ),
        DependencyKind::WriteAfterRead => (
            vk::AccessFlags2::empty(),
            vk::AccessFlags2::empty(),
        ),
        DependencyKind::WriteAfterWrite => (
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        ),
    }
}

fn expected_final_timeline_value(plan: &VulkanSyncPlan) -> u64 {
    plan.submissions
        .iter()
        .map(|submission| submission.signal.value)
        .max()
        .unwrap_or(0)
}

fn completion_lowering_digest(
    plan: &VulkanSyncPlan,
    completion_expected: u64,
    queue_family_index: u32,
) -> String {
    let mut h = Hasher::new();
    h.update(b"symthaea.gpu-fabric.vulkan-completion-lowering.v2\0");
    h.update(b"semaphore-type:timeline\0");
    h.update(b"initial-value:0\0");
    h.update(b"recording-policy:single-primary-command-buffer\0");
    h.update(b"submission-policy:single-vkQueueSubmit2-batch\0");
    h.update(b"signal-policy:single-final-signal\0");
    h.update(b"completion-policy:max-plan-signal-value\0");
    h.update(b"submit-api:vkQueueSubmit2\0");
    h.update(b"signal-api:VkSemaphoreSubmitInfo\0");
    h.update(&vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw().to_le_bytes());
    h.update(b"wait-api:vkWaitSemaphores\0");
    h.update(b"counter-api:vkGetSemaphoreCounterValue\0");
    h.update(&VULKAN_TIMELINE_TIMEOUT_NS.to_le_bytes());
    h.update(&queue_family_index.to_le_bytes());
    h.update(&0_u32.to_le_bytes()); // queue index within selected family
    h.update(&0_u32.to_le_bytes()); // semaphore device index
    h.update(&1_u32.to_le_bytes()); // command-buffer device mask
    h.update(&completion_expected.to_le_bytes());
    h.update(&(plan.submissions.len() as u32).to_le_bytes());
    h.update(&1_u32.to_le_bytes()); // command-buffer count
    h.update(&1_u32.to_le_bytes()); // queue-submit batch count
    h.update(&1_u32.to_le_bytes()); // final signal count
    h.finalize().to_hex().to_string()
}

fn barrier_lowering_digest(
    plan: &VulkanSyncPlan,
    resource_storage_sizes: &BTreeMap<ResourceId, u64>,
) -> Result<String, ResourceId> {
    let mut h = Hasher::new();
    h.update(b"symthaea.gpu-fabric.vulkan-barrier-lowering.v2\0");
    h.update(b"src-stage:compute-shader\0");
    h.update(b"dst-stage:compute-shader\0");
    h.update(b"range-policy:rounded-storage-bytes\0");
    h.update(b"queue-family:ignored\0");
    h.update(b"descriptor-policy:reads-sorted-by-resource-id\0");
    h.update(b"descriptor-policy:single-write-slot\0");
    h.update(b"offset-policy:zero\0");

    for kind in [
        DependencyKind::ReadAfterWrite,
        DependencyKind::WriteAfterRead,
        DependencyKind::WriteAfterWrite,
    ] {
        h.update(&[match kind {
            DependencyKind::ReadAfterWrite => 1,
            DependencyKind::WriteAfterRead => 2,
            DependencyKind::WriteAfterWrite => 3,
        }]);
        let (src_access, dst_access) = barrier_access_masks(kind);
        h.update(&src_access.as_raw().to_le_bytes());
        h.update(&dst_access.as_raw().to_le_bytes());
    }

    for submission in &plan.submissions {
        for barrier in &submission.barriers {
            h.update(&barrier.from.to_le_bytes());
            h.update(&barrier.to.to_le_bytes());
            h.update(&(barrier.resource.as_str().len() as u32).to_le_bytes());
            h.update(barrier.resource.as_str().as_bytes());
            h.update(&[match barrier.kind {
                DependencyKind::ReadAfterWrite => 1,
                DependencyKind::WriteAfterRead => 2,
                DependencyKind::WriteAfterWrite => 3,
            }]);
            let size = resource_storage_sizes
                .get(&barrier.resource)
                .copied()
                .ok_or_else(|| barrier.resource.clone())?;
            h.update(&size.to_le_bytes());
        }
    }
    Ok(h.finalize().to_hex().to_string())
}

fn barrier_digest(plan: &VulkanSyncPlan) -> String {
    let mut h = Hasher::new();
    h.update(b"symthaea.gpu-fabric.vulkan-barriers.v1\0");
    for submission in &plan.submissions {
        for barrier in &submission.barriers {
            h.update(&barrier.from.to_le_bytes());
            h.update(&barrier.to.to_le_bytes());
            h.update(&(barrier.resource.as_str().len() as u32).to_le_bytes());
            h.update(barrier.resource.as_str().as_bytes());
            h.update(&[match barrier.kind {
                DependencyKind::ReadAfterWrite => 1,
                DependencyKind::WriteAfterRead => 2,
                DependencyKind::WriteAfterWrite => 3,
            }]);
        }
    }
    h.finalize().to_hex().to_string()
}

struct TimelineSemaphoreGuard {
    device: Device,
    semaphore: vk::Semaphore,
    submitted: bool,
    completed: bool,
}

impl TimelineSemaphoreGuard {
    fn new(device: Device, semaphore: vk::Semaphore) -> Self {
        Self { device, semaphore, submitted: false, completed: false }
    }

    fn mark_submitted(&mut self) {
        self.submitted = true;
    }

    fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for TimelineSemaphoreGuard {
    fn drop(&mut self) {
        unsafe {
            if self.submitted && !self.completed {
                let _ = self.device.device_wait_idle();
            }
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

impl Drop for VulkanBarrierWorkloadRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_shader_module(self.shader, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (
        ExecutionGraph,
        ExecutionSchedule,
        VulkanSyncPlan,
        BTreeMap<ResourceId, BinaryHypervector>,
    ) {
        let lhs = ResourceId::new("lhs").unwrap();
        let rhs = ResourceId::new("rhs").unwrap();
        let mid = ResourceId::new("mid").unwrap();
        let out = ResourceId::new("out").unwrap();

        let n1 = ExecutionNode::new(
            1,
            GpuOperation::HdcBindXor { dimensions: 32 },
            vec![
                crate::ResourceUse::new(lhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(rhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(mid.clone(), AccessKind::Write),
            ],
        );
        let n2 = ExecutionNode::new(
            2,
            GpuOperation::HdcBindXor { dimensions: 32 },
            vec![
                crate::ResourceUse::new(mid.clone(), AccessKind::Read),
                crate::ResourceUse::new(rhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(out.clone(), AccessKind::Write),
            ],
        );

        let graph = ExecutionGraph::new(
            vec![n1, n2],
            vec![crate::DependencyEdge::new(
                1,
                2,
                mid.clone(),
                DependencyKind::ReadAfterWrite,
            )],
        )
        .unwrap();
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        let queue = crate::VulkanQueueId::new(0).unwrap();
        let plan = VulkanSyncPlan::from_schedule(
            &schedule,
            &[
                crate::VulkanQueueAssignment { node_id: 1, queue },
                crate::VulkanQueueAssignment { node_id: 2, queue },
            ],
        )
        .unwrap();

        let mut initial = BTreeMap::new();
        initial.insert(
            lhs,
            BinaryHypervector::from_bytes(32, vec![0x0f, 0xf0, 0xaa, 0x55]).unwrap(),
        );
        initial.insert(
            rhs,
            BinaryHypervector::from_bytes(32, vec![0x33, 0xcc, 0x55, 0xaa]).unwrap(),
        );
        initial.insert(mid, BinaryHypervector::zeros(32));
        initial.insert(out, BinaryHypervector::zeros(32));
        (graph, schedule, plan, initial)
    }

    fn hazard_fixture() -> (
        ExecutionGraph,
        ExecutionSchedule,
        VulkanSyncPlan,
        BTreeMap<ResourceId, BinaryHypervector>,
    ) {
        let lhs = ResourceId::new("lhs").unwrap();
        let rhs = ResourceId::new("rhs").unwrap();
        let mid = ResourceId::new("mid").unwrap();

        let n1 = ExecutionNode::new(
            1,
            GpuOperation::HdcBindXor { dimensions: 32 },
            vec![
                crate::ResourceUse::new(lhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(rhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(mid.clone(), AccessKind::Write),
            ],
        );
        let n2 = ExecutionNode::new(
            2,
            GpuOperation::HdcBindXor { dimensions: 32 },
            vec![
                crate::ResourceUse::new(mid.clone(), AccessKind::Read),
                crate::ResourceUse::new(lhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(rhs.clone(), AccessKind::Write),
            ],
        );
        let n3 = ExecutionNode::new(
            3,
            GpuOperation::HdcBindXor { dimensions: 32 },
            vec![
                crate::ResourceUse::new(lhs.clone(), AccessKind::Read),
                crate::ResourceUse::new(mid.clone(), AccessKind::Read),
                crate::ResourceUse::new(rhs.clone(), AccessKind::Write),
            ],
        );

        let graph = ExecutionGraph::new(
            vec![n1, n2, n3],
            vec![
                crate::DependencyEdge::new(
                    1,
                    2,
                    mid.clone(),
                    DependencyKind::ReadAfterWrite,
                ),
                crate::DependencyEdge::new(
                    1,
                    2,
                    rhs.clone(),
                    DependencyKind::WriteAfterRead,
                ),
                crate::DependencyEdge::new(
                    2,
                    3,
                    rhs.clone(),
                    DependencyKind::WriteAfterWrite,
                ),
            ],
        )
        .unwrap();
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        let queue = crate::VulkanQueueId::new(0).unwrap();
        let plan = VulkanSyncPlan::from_schedule(
            &schedule,
            &[
                crate::VulkanQueueAssignment { node_id: 1, queue },
                crate::VulkanQueueAssignment { node_id: 2, queue },
                crate::VulkanQueueAssignment { node_id: 3, queue },
            ],
        )
        .unwrap();

        let mut initial = BTreeMap::new();
        initial.insert(
            lhs,
            BinaryHypervector::from_bytes(32, vec![0x0f, 0xf0, 0xaa, 0x55]).unwrap(),
        );
        initial.insert(
            rhs,
            BinaryHypervector::from_bytes(32, vec![0x33, 0xcc, 0x55, 0xaa]).unwrap(),
        );
        initial.insert(mid, BinaryHypervector::zeros(32));
        (graph, schedule, plan, initial)
    }

    #[test]
    fn barrier_access_policy_matches_vulkan_hazards() {
        assert_eq!(
            barrier_access_masks(DependencyKind::ReadAfterWrite),
            (
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        );
        assert_eq!(
            barrier_access_masks(DependencyKind::WriteAfterWrite),
            (
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        );
        assert_eq!(
            barrier_access_masks(DependencyKind::WriteAfterRead),
            (
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::empty(),
            )
        );
    }

    #[test]
    fn barrier_lowering_digest_is_distinct_from_semantic_barrier_digest() {
        let (_, _, plan, final_state) = fixture();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| {
                (
                    resource.clone(),
                    rounded_storage_bytes(value.as_bytes().len() as u64),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_ne!(
            barrier_digest(&plan),
            barrier_lowering_digest(&plan, &storage_sizes).unwrap()
        );
        assert!(!barrier_lowering_digest(&plan, &storage_sizes).unwrap().is_empty());
    }

    #[test]
    fn barrier_lowering_digest_binds_concrete_resource_ranges() {
        let (_, _, plan, final_state) = fixture();
        let mut storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let baseline = barrier_lowering_digest(&plan, &storage_sizes).unwrap();

        let mid = ResourceId::new("mid").unwrap();
        storage_sizes.insert(mid.clone(), storage_sizes[&mid] + 4);
        assert_ne!(
            baseline,
            barrier_lowering_digest(&plan, &storage_sizes).unwrap()
        );
    }

    #[test]
    fn barrier_lowering_digest_rejects_missing_resource_size() {
        let (_, _, plan, _) = fixture();
        let empty = BTreeMap::new();
        assert_eq!(
            barrier_lowering_digest(&plan, &empty).unwrap_err(),
            ResourceId::new("mid").unwrap()
        );
    }

    #[test]
    fn workload_binding_is_canonical_by_resource_id() {
        let (mut graph, schedule, plan, initial) = fixture();
        let canonical_graph = graph.clone();
        for node in &mut graph.nodes {
            node.resources.reverse();
        }

        assert_eq!(graph.digest_hex().unwrap(), canonical_graph.digest_hex().unwrap());

        for node in &graph.nodes {
            let (reads, writes) = canonical_workload_resources(node);
            assert_eq!(reads.len(), 2);
            assert_eq!(writes.len(), 1);
            assert!(reads[0].resource <= reads[1].resource);
        }

        let reordered_final = simulate(&graph, &schedule, &initial).unwrap();
        let canonical_final = simulate(&canonical_graph, &schedule, &initial).unwrap();
        assert_eq!(reordered_final, canonical_final);
        assert_eq!(plan.queue_count, 1);
    }

    #[test]
    fn workload_plan_exercises_raw_war_and_waw() {
        let (graph, schedule, plan, initial) = hazard_fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        assert_eq!(
            plan.submissions[1].barriers,
            vec![
                VulkanBarrierRequirement {
                    from: 1,
                    to: 2,
                    resource: ResourceId::new("mid").unwrap(),
                    kind: DependencyKind::ReadAfterWrite,
                },
                VulkanBarrierRequirement {
                    from: 1,
                    to: 2,
                    resource: ResourceId::new("rhs").unwrap(),
                    kind: DependencyKind::WriteAfterRead,
                },
            ]
        );
        assert_eq!(
            plan.submissions[2].barriers,
            vec![
                VulkanBarrierRequirement {
                    from: 2,
                    to: 3,
                    resource: ResourceId::new("rhs").unwrap(),
                    kind: DependencyKind::WriteAfterWrite,
                },
            ]
        );
        assert!(plan.submissions[1].barriers[0].requires_memory_dependency());
        assert!(!plan.submissions[1].barriers[1].requires_memory_dependency());
        assert!(plan.submissions[2].barriers[0].requires_memory_dependency());
        assert_eq!(
            final_state[&ResourceId::new("rhs").unwrap()].as_bytes(),
            &[0x33, 0xcc, 0x55, 0xaa]
        );
    }

    #[test]
    fn barrier_digest_is_stable_and_bound_to_plan() {
        let (_, _, plan, _) = fixture();
        assert_eq!(barrier_digest(&plan), barrier_digest(&plan));
        assert_eq!(plan.submissions[1].barriers.len(), 1);
        assert!(plan.submissions[1].barriers[0].requires_memory_dependency());
    }

    #[test]
    fn cpu_oracle_propagates_hdc_resource_chain() {
        let (graph, schedule, _, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        assert_eq!(
            final_state[&ResourceId::new("mid").unwrap()].as_bytes(),
            &[0x3c, 0x3c, 0xff, 0xff]
        );
        assert_eq!(
            final_state[&ResourceId::new("out").unwrap()].as_bytes(),
            &[0x0f, 0xf0, 0xaa, 0x55]
        );
    }

    #[test]
    fn extra_resource_bindings_are_rejected() {
        let (graph, _, _, mut initial) = fixture();
        initial.insert(
            ResourceId::new("unused").unwrap(),
            BinaryHypervector::zeros(32),
        );
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        let queue = crate::VulkanQueueId::new(0).unwrap();
        let plan = VulkanSyncPlan::from_schedule(
            &schedule,
            &[
                crate::VulkanQueueAssignment { node_id: 1, queue },
                crate::VulkanQueueAssignment { node_id: 2, queue },
            ],
        ).unwrap();

        let error = validate_initial_resources(&graph, &initial).unwrap_err();
        assert!(matches!(error, VulkanBarrierError::UnexpectedResource(ref resource) if resource.as_str() == "unused"));
        assert_eq!(plan.queue_count, 1);
    }

    #[test]
    fn receipt_rejects_unreached_timeline_completion() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected_final_timeline_value(&plan),
            completion_observed: 0,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::TimelineCompletion {
                expected: expected_final_timeline_value(&plan),
                observed: 0,
            })
        ));
    }

    #[test]
    fn receipt_rejects_wrong_timeline_api_version() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let mut receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: 1,
            completion_observed: 1,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        receipt.vulkan_api_version = vk::API_VERSION_1_2;
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::ApiVersion)
        ));
    }

    #[test]
    fn receipt_rejects_unqualified_physical_device_api_version() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: 1,
            completion_observed: 1,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: vk::API_VERSION_1_2,
            queue_family_index: 0,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::PhysicalDeviceApiVersion)
        ));
    }

    #[test]
    fn receipt_rejects_tampered_concrete_lowering_digest() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();

        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| {
                (
                    resource.clone(),
                    rounded_storage_bytes(value.as_bytes().len() as u64),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected_final_timeline_value(&plan),
            completion_observed: expected_final_timeline_value(&plan),
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        receipt.barrier_lowering_digest = String::from("tampered");

        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::BarrierDigest)
        ));
    }

    #[test]
    fn receipt_rejects_tampered_resource_storage_size() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let mut storage_sizes = final_state
            .iter()
            .map(|(resource, value)| {
                (
                    resource.clone(),
                    rounded_storage_bytes(value.as_bytes().len() as u64),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes.clone(),
            completion_expected: expected_final_timeline_value(&plan),
            completion_observed: expected_final_timeline_value(&plan),
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };

        storage_sizes.insert(ResourceId::new("mid").unwrap(), 8);
        receipt.resource_storage_sizes = storage_sizes;

        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::ResourceStorageSize(resource))
                if resource.as_str() == "mid"
        ));
    }

    #[test]
    fn receipt_rejects_tampered_barrier_digest() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();

        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| {
                (
                    resource.clone(),
                    rounded_storage_bytes(value.as_bytes().len() as u64),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected_final_timeline_value(&plan),
            completion_observed: expected_final_timeline_value(&plan),
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        receipt.barrier_digest = String::from("tampered");

        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::BarrierDigest)
        ));
    }

    #[test]
    fn completion_lowering_digest_binds_final_timeline_policy() {
        let (_, _, mut plan, _) = fixture();
        let baseline = completion_lowering_digest(
            &plan,
            expected_final_timeline_value(&plan),
            0,
        );
        plan.submissions[1].signal.value += 1;
        assert_ne!(
            baseline,
            completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            )
        );
    }

    #[test]
    fn receipt_rejects_overshot_timeline_completion() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let expected = expected_final_timeline_value(&plan);
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(&plan, expected, 0),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected,
            completion_observed: expected + 1,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::TimelineCompletion {
                expected,
                observed: expected + 1,
            })
        ));
    }

    #[test]
    fn receipt_rejects_multi_queue_plan() {
        let (graph, schedule, _, initial) = fixture();
        let queue_zero = crate::VulkanQueueId::new(0).unwrap();
        let queue_one = crate::VulkanQueueId::new(1).unwrap();
        let plan = VulkanSyncPlan::from_schedule(
            &schedule,
            &[
                crate::VulkanQueueAssignment { node_id: 1, queue: queue_zero },
                crate::VulkanQueueAssignment { node_id: 2, queue: queue_one },
            ],
        )
        .unwrap();
        assert!(plan.queue_count > 1);

        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let expected = expected_final_timeline_value(&plan);
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(&plan, expected, 0),
            node_count: schedule.nodes.len() as u32,
            barrier_count: plan.submissions.iter().map(|s| s.barriers.len() as u32).sum(),
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected,
            completion_observed: expected,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::MultipleLogicalQueues)
        ));
    }

    #[test]
    fn receipt_rejects_tampered_completion_lowering_digest() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let mut receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected_final_timeline_value(&plan),
            completion_observed: expected_final_timeline_value(&plan),
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 0,
        };
        receipt.completion_lowering_digest = String::from("tampered");
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::CompletionLoweringDigest)
        ));
    }

    #[test]
    fn receipt_rejects_tampered_queue_family_binding() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();
        let storage_sizes = final_state
            .iter()
            .map(|(resource, value)| (
                resource.clone(),
                rounded_storage_bytes(value.as_bytes().len() as u64),
            ))
            .collect::<BTreeMap<_, _>>();
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            barrier_lowering_digest: barrier_lowering_digest(&plan, &storage_sizes).unwrap(),
            completion_lowering_digest: completion_lowering_digest(
                &plan,
                expected_final_timeline_value(&plan),
                0,
            ),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected: expected_final_timeline_value(&plan),
            completion_observed: expected_final_timeline_value(&plan),
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            queue_family_index: 7,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::CompletionLoweringDigest)
        ));
    }

    fn print_receipt_evidence(label: &str, receipt: &VulkanBarrierExecutionReceipt) {
        println!("qualification_claim=workload_execution+synchronization_only");
        println!("qualification_fixture={label}");
        println!("receipt_version={}", receipt.version);
        println!("graph_digest={}", receipt.graph_digest);
        println!("schedule_digest={}", receipt.schedule_digest);
        println!("sync_plan_digest={}", receipt.sync_plan_digest);
        println!("barrier_digest={}", receipt.barrier_digest);
        println!("barrier_lowering_digest={}", receipt.barrier_lowering_digest);
        println!("completion_lowering_digest={}", receipt.completion_lowering_digest);
        println!("node_count={}", receipt.node_count);
        println!("barrier_count={}", receipt.barrier_count);
        println!("resource_storage_sizes={:?}", receipt.resource_storage_sizes);
        println!("resource_digests={:?}", receipt.resource_digests);
        println!("completion_expected={}", receipt.completion_expected);
        println!("completion_observed={}", receipt.completion_observed);
        println!("vulkan_api_version={}", receipt.vulkan_api_version);
        println!("physical_device_api_version={}", receipt.physical_device_api_version);
        println!("queue_family_index={}", receipt.queue_family_index);
    }

    #[test]
    #[ignore = "requires a Vulkan 1.3 validation runner"]
    fn real_vulkan_barrier_workload_matches_cpu_oracle() {
        let runtime = VulkanBarrierWorkloadRuntime::new()
            .expect("qualified Vulkan 1.3 synchronization2 timeline device");

        for (graph, schedule, plan, initial) in [fixture(), hazard_fixture()] {
            let (observed, receipt) = runtime
                .execute_verified(&graph, &schedule, &plan, &initial)
                .expect("Vulkan barrier workload must complete");
            receipt
                .verify_against(&graph, &schedule, &plan, &observed)
                .expect("receipt must independently verify");
            print_receipt_evidence(
                if graph.nodes.len() == 2 && plan.submissions.len() == 2 {
                    "fixture"
                } else {
                    "hazard"
                },
                &receipt,
            );
        }

        let (_, _, _, initial) = fixture();
        let (_, _, _, _) = hazard_fixture();
        assert_eq!(
            initial[&ResourceId::new("rhs").unwrap()].as_bytes(),
            &[0x33, 0xcc, 0x55, 0xaa]
        );
    }
}

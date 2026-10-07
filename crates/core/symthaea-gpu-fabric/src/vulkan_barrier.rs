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
const RECEIPT_VERSION: u16 = 1;

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
    #[error("Vulkan completion fence failed: {0:?}")]
    Fence(vk::Result),
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanBarrierExecutionReceipt {
    pub version: u16,
    pub graph_digest: String,
    pub schedule_digest: String,
    pub sync_plan_digest: String,
    pub barrier_digest: String,
    pub barrier_lowering_digest: String,
    pub node_count: u32,
    pub barrier_count: u32,
    pub resource_digests: BTreeMap<ResourceId, String>,
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
        if self.barrier_lowering_digest != barrier_lowering_digest(plan) {
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

        let mut selected = None;
        for physical in unsafe { instance.enumerate_physical_devices().map_err(VulkanBarrierError::Vk)? } {
            let props = unsafe { instance.get_physical_device_properties(physical) };
            if props.api_version < VULKAN_API_VERSION { continue; }
            let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut sync2);
            unsafe { instance.get_physical_device_features2(physical, &mut features2); }
            if sync2.synchronization2 == 0 { continue; }
            let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
                .iter().enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32);
            if let Some(family) = family { selected = Some((physical, family)); break; }
        }

        let (physical, family) = selected.ok_or(VulkanBarrierError::NoQualifiedDevice)?;
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities);
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .push_next(&mut sync2);
        let device = unsafe {
            instance.create_device(physical, &device_info, None).map_err(|e| {
                instance.destroy_instance(None);
                VulkanBarrierError::Vk(e)
            })?
        };
        let queue = unsafe { device.get_device_queue(family, 0) };

        let spirv = compile_spirv()?;
        let shader = create_shader_module(&device, &spirv)?;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None).map_err(VulkanBarrierError::Vk)? };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_layout));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).map_err(VulkanBarrierError::Vk)? };
        let entry_point = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(&entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
        let pipeline = unsafe {
            match device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None) {
                Ok(mut pipelines) => pipelines.pop().unwrap(),
                Err((_, error)) => return Err(VulkanBarrierError::Vk(error)),
            }
        };
        let command_pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None).map_err(VulkanBarrierError::Vk)? };
        let pool_size = vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count((MAX_WORKLOAD_NODES * 3) as u32);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(MAX_WORKLOAD_NODES as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).map_err(VulkanBarrierError::Vk)? };

        Ok(Self {
            instance, device, queue, command_pool, descriptor_layout, descriptor_pool,
            pipeline_layout, pipeline, shader, memory_properties,
            max_storage_buffer_range: u64::from(props.limits.max_storage_buffer_range),
            max_compute_workgroup_count_x: props.limits.max_compute_work_group_count[0],
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

        let resources = graph.nodes.iter()
            .flat_map(|n| n.resources.iter().map(|u| u.resource.clone()))
            .collect::<BTreeSet<_>>();
        for resource in &resources {
            if !initial.contains_key(resource) { return Err(VulkanBarrierError::MissingResource(resource.clone())); }
        }
        if let Some(extra) = initial.keys().find(|resource| !resources.contains(*resource)) {
            return Err(VulkanBarrierError::UnexpectedResource(extra.clone()));
        }
        let expected = simulate(graph, schedule, initial)?;
        let mut buffers = BTreeMap::new();
        for (resource, value) in initial {
            let physical = rounded_storage_bytes(value.as_bytes().len() as u64);
            if physical > self.max_storage_buffer_range { return Err(VulkanBarrierError::ResourceTooLarge(resource.clone())); }
            buffers.insert(resource.clone(), WorkloadBuffer::new(&self.device, &self.memory_properties, physical)?);
        }
        for (resource, value) in initial { buffers[resource].write(&self.device, value.as_bytes())?; }

        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info).map_err(VulkanBarrierError::Vk)?[0] };
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(command, &begin).map_err(VulkanBarrierError::Vk)?; }

        let mut sets = Vec::new();
        for scheduled in &schedule.nodes {
            let node = graph.nodes.iter().find(|n| n.id == scheduled.id).ok_or(VulkanBarrierError::UnsupportedNodeShape(scheduled.id))?;
            let reads = node.resources.iter().filter(|u| u.access == AccessKind::Read).collect::<Vec<_>>();
            let writes = node.resources.iter().filter(|u| u.access == AccessKind::Write).collect::<Vec<_>>();
            if reads.len() != 2 || writes.len() != 1 || node.resources.len() != 3 { return Err(VulkanBarrierError::UnsupportedNodeShape(node.id)); }
            let submission = plan.submissions.iter().find(|s| s.node_id == node.id).ok_or(VulkanBarrierError::UnsupportedNodeShape(node.id))?;
            record_barriers(&self.device, command, &submission.barriers, &buffers)?;
            let range = buffers[&writes[0].resource].storage_size;
            let set = allocate_set(&self.device, self.descriptor_pool, self.descriptor_layout, [&buffers[&reads[0].resource], &buffers[&reads[1].resource], &buffers[&writes[0].resource]], range)?;
            sets.push(set);
            let groups = ((range / 4) as u32).saturating_add(WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            if groups.max(1) > self.max_compute_workgroup_count_x { return Err(VulkanBarrierError::DispatchTooLarge(writes[0].resource.clone())); }
            unsafe {
                self.device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                self.device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[set], &[]);
                self.device.cmd_dispatch(command, groups.max(1), 1, 1);
            }
        }

        unsafe { self.device.end_command_buffer(command).map_err(VulkanBarrierError::Vk)?; }
        let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(VulkanBarrierError::Vk)? };
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
        if let Err(error) = unsafe { self.device.queue_submit(self.queue, std::slice::from_ref(&submit), fence) } {
            unsafe { self.device.destroy_fence(fence, None); }
            return Err(VulkanBarrierError::Vk(error));
        }
        if let Err(error) = unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) } {
            unsafe { let _ = self.device.device_wait_idle(); self.device.destroy_fence(fence, None); }
            return Err(VulkanBarrierError::Fence(error));
        }
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, std::slice::from_ref(&command));
            self.device.free_descriptor_sets(self.descriptor_pool, &sets).map_err(VulkanBarrierError::Vk)?;
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
        for (resource, value) in &observed { digests.insert(resource.clone(), resource_digest(value)); }
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().map_err(VulkanBarrierError::Graph)?,
            schedule_digest: schedule.digest_hex().map_err(|_| VulkanBarrierError::Schedule(
                crate::ScheduleError::UnsupportedVersion(schedule.version),
            ))?,
            sync_plan_digest: plan.digest_hex().map_err(|e| VulkanBarrierError::SyncPlan(e))?,
            barrier_digest: barrier_digest(plan),
            barrier_lowering_digest: barrier_lowering_digest(plan),
            node_count: schedule.nodes.len() as u32,
            barrier_count: plan.submissions.iter().map(|s| s.barriers.len() as u32).sum(),
            resource_digests: digests,
        };
        receipt.verify_against(graph, schedule, plan, &observed).map_err(VulkanBarrierError::Receipt)?;
        Ok((observed, receipt))
    }
}

fn simulate(
    graph: &ExecutionGraph,
    schedule: &ExecutionSchedule,
    initial: &BTreeMap<ResourceId, BinaryHypervector>,
) -> Result<BTreeMap<ResourceId, BinaryHypervector>, VulkanBarrierError> {
    let mut state = initial.clone();
    for scheduled in &schedule.nodes {
        let node = graph.nodes.iter().find(|n| n.id == scheduled.id).ok_or(VulkanBarrierError::UnsupportedNodeShape(scheduled.id))?;
        let reads = node.resources.iter().filter(|u| u.access == AccessKind::Read).collect::<Vec<_>>();
        let writes = node.resources.iter().filter(|u| u.access == AccessKind::Write).collect::<Vec<_>>();
        if reads.len() != 2 || writes.len() != 1 { return Err(VulkanBarrierError::UnsupportedNodeShape(node.id)); }
        let dimensions = match node.operation { GpuOperation::HdcBindXor { dimensions } => dimensions };
        if state[&reads[0].resource].dimensions != dimensions || state[&reads[1].resource].dimensions != dimensions || state[&writes[0].resource].dimensions != dimensions {
            return Err(VulkanBarrierError::ResourceDimensions { resource: writes[0].resource.clone(), actual: state[&writes[0].resource].dimensions, expected: dimensions });
        }
        let bytes = state[&reads[0].resource].as_bytes().iter().zip(state[&reads[1].resource].as_bytes()).map(|(a,b)| a ^ b).collect::<Vec<_>>();
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
            let (src_access, dst_access) = match req.kind {
                DependencyKind::ReadAfterWrite => (vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::AccessFlags2::SHADER_STORAGE_READ),
                DependencyKind::WriteAfterWrite => (vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::AccessFlags2::SHADER_STORAGE_WRITE),
                DependencyKind::WriteAfterRead => (vk::AccessFlags2::empty(), vk::AccessFlags2::SHADER_STORAGE_WRITE),
            };
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
    let set = unsafe { device.allocate_descriptor_sets(&info).map_err(VulkanBarrierError::Vk)?[0] };
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

fn barrier_lowering_digest(plan: &VulkanSyncPlan) -> String {
    let mut h = Hasher::new();
    h.update(b"symthaea.gpu-fabric.vulkan-barrier-lowering.v1\0");
    h.update(b"src-stage:compute-shader\0");
    h.update(b"dst-stage:compute-shader\0");
    h.update(b"raw-src-access:shader-storage-write\0");
    h.update(b"raw-dst-access:shader-storage-read\0");
    h.update(b"waw-src-access:shader-storage-write\0");
    h.update(b"waw-dst-access:shader-storage-write\0");
    h.update(b"war-access:empty\0");
    h.update(b"range-policy:rounded-storage-bytes\0");
    h.update(b"queue-family:ignored\0");

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
                mid,
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

    #[test]
    fn barrier_lowering_digest_is_distinct_from_semantic_barrier_digest() {
        let (_, _, plan, _) = fixture();
        assert_ne!(barrier_digest(&plan), barrier_lowering_digest(&plan));
        assert!(!barrier_lowering_digest(&plan).is_empty());
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

        let mut checked = false;
        let resources = graph.nodes.iter()
            .flat_map(|node| node.resources.iter().map(|use_| use_.resource.clone()))
            .collect::<BTreeSet<_>>();
        if let Some(extra) = initial.keys().find(|resource| !resources.contains(*resource)) {
            assert_eq!(extra.as_str(), "unused");
            checked = true;
        }
        assert!(checked);
        assert_eq!(plan.queue_count, 1);
    }

    #[test]
    fn receipt_rejects_tampered_barrier_digest() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let digests = final_state
            .iter()
            .map(|(resource, value)| (resource.clone(), resource_digest(value)))
            .collect::<BTreeMap<_, _>>();

        let mut receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: graph.digest_hex().unwrap(),
            schedule_digest: schedule.digest_hex().unwrap(),
            sync_plan_digest: plan.digest_hex().unwrap(),
            barrier_digest: barrier_digest(&plan),
            node_count: schedule.nodes.len() as u32,
            barrier_count: 1,
            resource_digests: digests,
        };
        receipt.barrier_digest = String::from("tampered");

        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::BarrierDigest)
        ));
    }

    #[test]
    #[ignore = "requires a Vulkan 1.3 validation runner"]
    fn real_vulkan_barrier_workload_matches_cpu_oracle() {
        let (graph, schedule, plan, initial) = fixture();
        let runtime = VulkanBarrierWorkloadRuntime::new()
            .expect("qualified Vulkan 1.3 synchronization2 device");
        let (observed, receipt) = runtime
            .execute_verified(&graph, &schedule, &plan, &initial)
            .expect("Vulkan barrier workload must complete");
        receipt
            .verify_against(&graph, &schedule, &plan, &observed)
            .expect("receipt must independently verify");
        assert_eq!(
            observed[&ResourceId::new("out").unwrap()].as_bytes(),
            &[0x0f, 0xf0, 0xaa, 0x55]
        );
    }
}

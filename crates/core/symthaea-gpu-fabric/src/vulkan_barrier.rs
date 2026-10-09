use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::ptr;

use ash::{vk, Device, Entry, Instance};
use blake3::Hasher;
use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    AccessKind, BinaryHypervector, DependencyKind, ExecutionGraph, ExecutionNode,
    ExecutionSchedule, GpuOperation, HDC_BIND_XOR_KERNEL_ID, ResourceId, VulkanBarrierRequirement,
    VulkanSyncPlan,
};

const MAX_WORKLOAD_NODES: usize = 64;
const WORKGROUP_SIZE: u32 = 64;
const VULKAN_API_VERSION: u32 = vk::API_VERSION_1_3;
const VULKAN_TIMELINE_TIMEOUT_NS: u64 = 5_000_000_000;
const RECEIPT_VERSION: u16 = 9;
const VULKAN_IMPLEMENTATION_IDENTITY_VERSION: &str = "symthaea.gpu-fabric.vulkan-implementation.v1";
const WGSL_ABI_MARKER: &str = "symthaea.hdc.bind_xor.storage-u32.v1";
const VULKAN_ENTRY_POINT: &str = "main";
const VULKAN_SHADER_STAGE: &str = "compute";
const DRIVER_IDENTITY_VERSION: &str = "symthaea.gpu-fabric.vulkan-driver.v1";
const QUEUE_FAMILY_IDENTITY_VERSION: &str = "symthaea.gpu-fabric.vulkan-queue-family.v1";
const SYNCHRONIZATION_FEATURE_IDENTITY_VERSION: &str =
    "symthaea.gpu-fabric.vulkan-sync-features.v1";

#[cfg(test)]
fn qualification_stage(label: &str) {
    eprintln!("qualification_stage={label}");
}

#[cfg(not(test))]
fn qualification_stage(_label: &str) {}

const WGSL: &str = include_str!("hdc_bind_xor.wgsl");

#[derive(Debug, Error)]
pub enum VulkanBarrierError {
    #[error("Vulkan loader unavailable: {0}")]
    Loader(String),
    #[error("Vulkan call failed: {0:?}")]
    Vk(vk::Result),
    #[error("no Vulkan 1.3 compute device with synchronization2")]
    NoQualifiedDevice,
    #[error("empty workload cannot produce a valid Vulkan completion witness")]
    EmptyWorkload,
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

#[derive(Debug, Error, PartialEq, Eq)]
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
    #[error("empty workload cannot produce a valid Vulkan completion receipt")]
    EmptyWorkload,
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
    #[error("receipt physical-device UUID is all zeroes")]
    DeviceUuidMissing,
    #[error("receipt implementation identity digest is missing or malformed")]
    ImplementationIdentity,
    #[error("receipt physical-device identity digest is missing or malformed")]
    PhysicalDeviceIdentity,
    #[error("receipt implementation identity does not match the execution runtime")]
    ImplementationIdentityBinding,
    #[error("receipt physical-device identity does not match the execution runtime")]
    PhysicalDeviceIdentityBinding,
    #[error("receipt driver identity digest is missing or malformed")]
    DriverIdentity,
    #[error("receipt driver identity does not match the execution runtime")]
    DriverIdentityBinding,
    #[error("receipt driver UUID does not match the execution runtime")]
    DriverUuidBinding,
    #[error("receipt driver ID does not match the execution runtime")]
    DriverIdBinding,
    #[error("receipt queue-family identity digest is missing, malformed, or inconsistent")]
    QueueFamilyIdentity,
    #[error("receipt queue-family identity does not match the execution runtime")]
    QueueFamilyIdentityBinding,
    #[error("receipt synchronization feature profile is missing, malformed, or unqualified")]
    SynchronizationFeatureIdentity,
    #[error("receipt synchronization feature profile does not match the execution runtime")]
    SynchronizationFeatureIdentityBinding,
    #[error("receipt physical-device UUID does not match the execution runtime")]
    DeviceUuidBinding,
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
pub struct VulkanSynchronizationFeatureProfile {
    pub timeline_semaphore_supported: bool,
    pub synchronization2_supported: bool,
    pub timeline_semaphore_enabled: bool,
    pub synchronization2_enabled: bool,
    pub identity_digest: String,
}

impl VulkanSynchronizationFeatureProfile {
    fn new(
        timeline_semaphore_supported: bool,
        synchronization2_supported: bool,
        timeline_semaphore_enabled: bool,
        synchronization2_enabled: bool,
    ) -> Self {
        Self {
            timeline_semaphore_supported,
            synchronization2_supported,
            timeline_semaphore_enabled,
            synchronization2_enabled,
            identity_digest: synchronization_feature_identity_digest(
                timeline_semaphore_supported,
                synchronization2_supported,
                timeline_semaphore_enabled,
                synchronization2_enabled,
            ),
        }
    }

    fn verify(&self) -> Result<(), VulkanBarrierReceiptError> {
        if !self.timeline_semaphore_supported
            || !self.synchronization2_supported
            || !self.timeline_semaphore_enabled
            || !self.synchronization2_enabled
        {
            return Err(VulkanBarrierReceiptError::SynchronizationFeatureIdentity);
        }
        if !is_sha256_hex(&self.identity_digest)
            || self.identity_digest
                != synchronization_feature_identity_digest(
                    self.timeline_semaphore_supported,
                    self.synchronization2_supported,
                    self.timeline_semaphore_enabled,
                    self.synchronization2_enabled,
                )
        {
            return Err(VulkanBarrierReceiptError::SynchronizationFeatureIdentity);
        }
        Ok(())
    }
}

fn synchronization_feature_profile_from_device_create(
    timeline_semaphore_supported: bool,
    synchronization2_supported: bool,
    timeline_semaphore_enabled: vk::Bool32,
    synchronization2_enabled: vk::Bool32,
) -> VulkanSynchronizationFeatureProfile {
    VulkanSynchronizationFeatureProfile::new(
        timeline_semaphore_supported,
        synchronization2_supported,
        timeline_semaphore_enabled != 0,
        synchronization2_enabled != 0,
    )
}

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
    pub synchronization_features: VulkanSynchronizationFeatureProfile,
    pub queue_family_identity_digest: String,
    pub queue_family_queue_flags: u32,
    pub queue_family_queue_count: u32,
    pub queue_family_timestamp_valid_bits: u32,
    pub queue_family_min_image_transfer_granularity: [u32; 3],
    pub device_uuid: [u8; 16],
    pub implementation_identity_digest: String,
    pub physical_device_identity_digest: String,
    pub driver_identity_digest: String,
    pub driver_uuid: [u8; 16],
    pub driver_id: i32,
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
        if !is_sha256_hex(&self.implementation_identity_digest) {
            return Err(VulkanBarrierReceiptError::ImplementationIdentity);
        }
        if !is_sha256_hex(&self.physical_device_identity_digest) {
            return Err(VulkanBarrierReceiptError::PhysicalDeviceIdentity);
        }
        if !is_sha256_hex(&self.driver_identity_digest) {
            return Err(VulkanBarrierReceiptError::DriverIdentity);
        }
        self.synchronization_features.verify()?;
        if !is_sha256_hex(&self.queue_family_identity_digest)
            || self.queue_family_queue_count == 0
            || (self.queue_family_queue_flags & vk::QueueFlags::COMPUTE.as_raw()) == 0
        {
            return Err(VulkanBarrierReceiptError::QueueFamilyIdentity);
        }
        if self.queue_family_identity_digest
            != queue_family_identity_digest_from_fields(
                self.queue_family_index,
                self.queue_family_queue_flags,
                self.queue_family_queue_count,
                self.queue_family_timestamp_valid_bits,
                self.queue_family_min_image_transfer_granularity,
            )
        {
            return Err(VulkanBarrierReceiptError::QueueFamilyIdentity);
        }
        if schedule.nodes.is_empty() { return Err(VulkanBarrierReceiptError::EmptyWorkload); }
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
        if self.device_uuid.iter().all(|byte| *byte == 0) {
            return Err(VulkanBarrierReceiptError::DeviceUuidMissing);
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
        device_uuid: [u8; 16],
        implementation_identity_digest: &str,
        physical_device_identity_digest: &str,
        driver_identity_digest: &str,
        driver_uuid: [u8; 16],
        driver_id: i32,
        synchronization_features: &VulkanSynchronizationFeatureProfile,
        queue_family_identity_digest: &str,
        queue_family_queue_flags: u32,
        queue_family_queue_count: u32,
        queue_family_timestamp_valid_bits: u32,
        queue_family_min_image_transfer_granularity: [u32; 3],
    ) -> Result<(), VulkanBarrierReceiptError> {
        if self.physical_device_api_version != physical_device_api_version {
            return Err(VulkanBarrierReceiptError::PhysicalDeviceApiVersionBinding);
        }
        if self.queue_family_index != queue_family_index {
            return Err(VulkanBarrierReceiptError::QueueFamilyBinding);
        }
        if self.device_uuid != device_uuid {
            return Err(VulkanBarrierReceiptError::DeviceUuidBinding);
        }
        if self.implementation_identity_digest != implementation_identity_digest {
            return Err(VulkanBarrierReceiptError::ImplementationIdentityBinding);
        }
        if self.physical_device_identity_digest != physical_device_identity_digest {
            return Err(VulkanBarrierReceiptError::PhysicalDeviceIdentityBinding);
        }
        if self.driver_identity_digest != driver_identity_digest {
            return Err(VulkanBarrierReceiptError::DriverIdentityBinding);
        }
        if self.driver_uuid != driver_uuid {
            return Err(VulkanBarrierReceiptError::DriverUuidBinding);
        }
        if self.driver_id != driver_id {
            return Err(VulkanBarrierReceiptError::DriverIdBinding);
        }
        if self.synchronization_features != *synchronization_features {
            return Err(VulkanBarrierReceiptError::SynchronizationFeatureIdentityBinding);
        }
        if self.queue_family_identity_digest != queue_family_identity_digest {
            return Err(VulkanBarrierReceiptError::QueueFamilyIdentityBinding);
        }
        if self.queue_family_queue_flags != queue_family_queue_flags
            || self.queue_family_queue_count != queue_family_queue_count
            || self.queue_family_timestamp_valid_bits != queue_family_timestamp_valid_bits
            || self.queue_family_min_image_transfer_granularity
                != queue_family_min_image_transfer_granularity
        {
            return Err(VulkanBarrierReceiptError::QueueFamilyIdentityBinding);
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
    synchronization_features: VulkanSynchronizationFeatureProfile,
    queue_family_identity_digest: String,
    queue_family_queue_flags: u32,
    queue_family_queue_count: u32,
    queue_family_timestamp_valid_bits: u32,
    queue_family_min_image_transfer_granularity: [u32; 3],
    device_uuid: [u8; 16],
    implementation_identity_digest: String,
    shader_spirv_sha256: String,
    implementation_wgsl_sha256: String,
    implementation_wgsl_hex: String,
    shader_spirv_hex: String,
    physical_device_vendor_id: u32,
    physical_device_device_id: u32,
    physical_device_type: u32,
    physical_device_driver_version: u32,
    physical_device_name_hex: String,
    physical_device_identity_digest: String,
    driver_identity_digest: String,
    driver_uuid: [u8; 16],
    driver_id: i32,
    driver_name_hex: String,
    driver_info_hex: String,
    // Must be dropped after Instance/Device because ash requires Entry to outlive them.
    _entry: Entry,
}

impl VulkanBarrierWorkloadRuntime {
    pub fn new() -> Result<Self, VulkanBarrierError> {
        qualification_stage("before_entry_load");
        let entry = unsafe { Entry::load() }.map_err(|e| VulkanBarrierError::Loader(e.to_string()))?;
        qualification_stage("entry_loaded");
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
        qualification_stage("instance_created");

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
                .map(|(i, q)| {
                    (
                        i as u32,
                        *q,
                        timeline.timeline_semaphore != 0,
                        sync2.synchronization2 != 0,
                    )
                });
            if let Some((family, queue_properties, timeline_supported, synchronization2_supported)) = family {
                selected = Some((
                    physical,
                    family,
                    queue_properties,
                    timeline_supported,
                    synchronization2_supported,
                ));
                break;
            }
        }

        let (
            physical,
            family,
            queue_family_properties,
            timeline_semaphore_supported,
            synchronization2_supported,
        ) = match selected {
            Some(value) => value,
            None => {
                unsafe { instance.destroy_instance(None); }
                return Err(VulkanBarrierError::NoQualifiedDevice);
            }
        };
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let queue_family_identity_digest =
            queue_family_identity_digest(family, &queue_family_properties);
        let queue_family_queue_flags = queue_family_properties.queue_flags.as_raw();
        let queue_family_queue_count = queue_family_properties.queue_count;
        let queue_family_timestamp_valid_bits = queue_family_properties.timestamp_valid_bits;
        let queue_family_min_image_transfer_granularity = [
            queue_family_properties.min_image_transfer_granularity.width,
            queue_family_properties.min_image_transfer_granularity.height,
            queue_family_properties.min_image_transfer_granularity.depth,
        ];
        let physical_device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_bytes();
        let physical_device_identity_digest = physical_device_identity_digest(&props);
        let mut id_properties = vk::PhysicalDeviceIDProperties::default();
        let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut id_properties)
            .push_next(&mut driver_properties);
        unsafe { instance.get_physical_device_properties2(physical, &mut properties2); }
        let device_uuid = id_properties.device_uuid;
        let driver_uuid = id_properties.driver_uuid;
        let driver_id = driver_properties.driver_id.as_raw();
        let driver_name = unsafe { CStr::from_ptr(driver_properties.driver_name.as_ptr()) }.to_bytes();
        let driver_info = unsafe { CStr::from_ptr(driver_properties.driver_info.as_ptr()) }.to_bytes();
        let driver_identity_digest = driver_identity_digest(
            driver_uuid,
            driver_id,
            driver_name,
            driver_info,
        );
        if device_uuid.iter().all(|byte| *byte == 0) {
            unsafe { instance.destroy_instance(None); }
            return Err(VulkanBarrierError::NoQualifiedDevice);
        }
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        qualification_stage(&format!("device_selected_api={}.{}.{} queue_family={family}",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version)));
        qualification_stage(&format!("driver_identity_sha256={driver_identity_digest}"));
        qualification_stage(&format!("driver_id={driver_id}"));

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities);
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        // Bind the receipt to the exact feature values passed through this device-create pNext chain.
        let synchronization_features = synchronization_feature_profile_from_device_create(
            timeline_semaphore_supported,
            synchronization2_supported,
            timeline.timeline_semaphore,
            sync2.synchronization2,
        );
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
        qualification_stage("device_created");

        let spirv = compile_spirv().map_err(|error| {
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            error
        })?;
        let shader_spirv_bytes = spirv_to_bytes(&spirv);
        let shader_spirv_hex = hex_bytes(&shader_spirv_bytes);
        let shader_spirv_sha256 = sha256_hex(&shader_spirv_bytes);
        let implementation_wgsl_hex = hex_bytes(WGSL.as_bytes());
        let implementation_wgsl_sha256 = sha256_hex(WGSL.as_bytes());
        let implementation_identity_digest = vulkan_implementation_identity_digest(&spirv);
        qualification_stage("shader_spirv_compiled");
        qualification_stage(&format!("shader_spirv_sha256={shader_spirv_sha256}"));
        qualification_stage(&format!("implementation_identity_sha256={implementation_identity_digest}"));
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
        qualification_stage("shader_module_created");
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
        qualification_stage("descriptor_layout_created");
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
        qualification_stage("pipeline_layout_created");
        let entry_point = CString::new(VULKAN_ENTRY_POINT).expect("static Vulkan entry point has no NUL bytes");
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
        qualification_stage("compute_pipeline_created");
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
        qualification_stage("command_pool_created");
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

        qualification_stage("descriptor_pool_created");
        qualification_stage("runtime_owning_entry");
        Ok(Self {
            instance, device, queue, command_pool, descriptor_layout, descriptor_pool,
            pipeline_layout, pipeline, shader, memory_properties,
            max_storage_buffer_range: u64::from(props.limits.max_storage_buffer_range),
            max_compute_workgroup_count_x: props.limits.max_compute_work_group_count[0],
            physical_device_api_version: props.api_version,
            queue_family_index: family,
            synchronization_features,
            queue_family_identity_digest,
            queue_family_queue_flags,
            queue_family_queue_count,
            queue_family_timestamp_valid_bits,
            queue_family_min_image_transfer_granularity,
            device_uuid,
            implementation_identity_digest,
            shader_spirv_sha256,
            implementation_wgsl_sha256,
            implementation_wgsl_hex,
            shader_spirv_hex,
            physical_device_vendor_id: props.vendor_id,
            physical_device_device_id: props.device_id,
            physical_device_type: props.device_type.as_raw() as u32,
            physical_device_driver_version: props.driver_version,
            physical_device_name_hex: hex_bytes(physical_device_name),
            physical_device_identity_digest,
            driver_identity_digest,
            driver_uuid,
            driver_id,
            driver_name_hex: hex_bytes(driver_name),
            driver_info_hex: hex_bytes(driver_info),
            _entry: entry,
        })
    }

    pub fn execute_verified(
        &self,
        graph: &ExecutionGraph,
        schedule: &ExecutionSchedule,
        plan: &VulkanSyncPlan,
        initial: &BTreeMap<ResourceId, BinaryHypervector>,
    ) -> Result<(BTreeMap<ResourceId, BinaryHypervector>, VulkanBarrierExecutionReceipt), VulkanBarrierError> {
        qualification_stage("execute_begin");
        schedule.verify_against(graph).map_err(VulkanBarrierError::Schedule)?;
        plan.verify_against_schedule(schedule).map_err(VulkanBarrierError::SyncPlan)?;
        if schedule.nodes.is_empty() { return Err(VulkanBarrierError::EmptyWorkload); }
        if schedule.nodes.len() > MAX_WORKLOAD_NODES { return Err(VulkanBarrierError::WorkloadNodeLimit(schedule.nodes.len())); }
        if plan.queue_count > 1 || plan.assignments.iter().any(|a| a.queue.get() != 0) { return Err(VulkanBarrierError::MultipleLogicalQueues); }

        let _resources = validate_initial_resources(graph, initial)?;
        let expected = simulate(graph, schedule, initial)?;
        let completion_expected = expected_final_timeline_value(plan);
        let submission_contract = MaterializedSubmissionContract::from_plan(
            plan,
            completion_expected,
            self.queue_family_index,
        );
        let mut buffers = BTreeMap::new();
        for (resource, value) in initial {
            let physical = rounded_storage_bytes(value.as_bytes().len() as u64);
            if physical > self.max_storage_buffer_range { return Err(VulkanBarrierError::ResourceTooLarge(resource.clone())); }
            buffers.insert(resource.clone(), WorkloadBuffer::new(&self.device, &self.memory_properties, physical)?);
        }
        for (resource, value) in initial { buffers[resource].write(&self.device, value.as_bytes())?; }
        qualification_stage("inputs_uploaded");

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
        qualification_stage("command_buffer_allocated");

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_guard.command(), &begin)
                .map_err(VulkanBarrierError::Vk)?;
        }
        qualification_stage("command_buffer_begun");

        let mut set_guard =
            DescriptorSetGuard::new(self.device.clone(), self.descriptor_pool);
        let mut materialized_barrier_batches = Vec::new();
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

            if let Some(batch) = record_barriers(
                &self.device,
                command_guard.command(),
                scheduled.id,
                &submission.barriers,
                &buffers,
            )? {
                materialized_barrier_batches.push(batch);
            }

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

            let groups = dispatch_group_count(range, self.max_compute_workgroup_count_x)
                .ok_or_else(|| VulkanBarrierError::DispatchTooLarge(writes[0].resource.clone()))?;
            qualification_stage(&format!("node_{}_dispatch_begin", node.id));
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
            qualification_stage(&format!("node_{}_dispatch_recorded", node.id));
        }

        record_host_readback_barrier(
            &self.device,
            command_guard.command(),
            &submission_contract,
        );
        qualification_stage("host_readback_barrier_recorded");

        unsafe {
            self.device
                .end_command_buffer(command_guard.command())
                .map_err(VulkanBarrierError::Vk)?;
        }
        qualification_stage("command_buffer_ended");

        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(submission_contract.semaphore_type)
            .initial_value(submission_contract.timeline_initial_value);
        let semaphore_info = vk::SemaphoreCreateInfo::default()
            .flags(submission_contract.semaphore_create_flags)
            .push_next(&mut timeline_info);
        let semaphore = unsafe {
            self.device
                .create_semaphore(&semaphore_info, None)
                .map_err(VulkanBarrierError::TimelineSemaphoreCreate)?
        };
        let mut semaphore_guard = TimelineSemaphoreGuard::new(self.device.clone(), semaphore);
        qualification_stage("timeline_semaphore_created");

        let command_buffer_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(command_guard.command())
            .device_mask(submission_contract.command_buffer_device_mask);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .value(submission_contract.signal_value)
            .stage_mask(submission_contract.signal_stage_mask)
            .device_index(submission_contract.semaphore_device_index);
        let submit = vk::SubmitInfo2::default()
            .flags(submission_contract.submit_flags)
            .wait_semaphore_infos(&[])
            .command_buffer_infos(std::slice::from_ref(&command_buffer_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        unsafe {
            self.device
                .queue_submit2(self.queue, std::slice::from_ref(&submit), vk::Fence::null())
                .map_err(VulkanBarrierError::TimelineSubmit)?;
        }
        qualification_stage(&format!("queue_submitted_expected={completion_expected}"));
        semaphore_guard.mark_submitted();

        let wait_info = vk::SemaphoreWaitInfo::default()
            .flags(submission_contract.semaphore_wait_flags)
            .semaphores(std::slice::from_ref(&semaphore))
            .values(std::slice::from_ref(&submission_contract.signal_value));
        unsafe {
            self.device
                .wait_semaphores(&wait_info, submission_contract.timeout_ns)
                .map_err(VulkanBarrierError::TimelineWait)?;
        }
        qualification_stage("timeline_wait_completed");
        semaphore_guard.mark_completed();

        let completion_observed = unsafe {
            self.device
                .get_semaphore_counter_value(semaphore)
                .map_err(VulkanBarrierError::TimelineCounter)?
        };
        qualification_stage(&format!("timeline_counter_observed={completion_observed}"));
        if completion_observed != completion_expected {
            return Err(VulkanBarrierError::TimelineCompletionNotReached {
                expected: completion_expected,
                observed: completion_observed,
            });
        }

        let mut observed = BTreeMap::new();
        for (resource, value) in initial {
            qualification_stage(&format!("readback_begin_resource={}", resource.as_str()));
            let bytes = buffers[resource].read(&self.device, value.as_bytes().len())?;
            observed.insert(resource.clone(), BinaryHypervector::from_bytes(value.dimensions, bytes)
                .map_err(|_| VulkanBarrierError::OracleMismatch(resource.clone()))?);
        }
        for (resource, expected_value) in &expected {
            if observed.get(resource) != Some(expected_value) { return Err(VulkanBarrierError::OracleMismatch(resource.clone())); }
        }
        qualification_stage("oracle_matched");
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
            barrier_lowering_digest: materialized_barrier_batches_digest(
                &materialized_barrier_batches,
            ),
            completion_lowering_digest: submission_contract.digest(),
            node_count: schedule.nodes.len() as u32,
            barrier_count: plan.submissions.iter().map(|s| s.barriers.len() as u32).sum(),
            resource_digests: digests,
            resource_storage_sizes: storage_sizes,
            completion_expected,
            completion_observed,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: self.physical_device_api_version,
            queue_family_index: self.queue_family_index,
            synchronization_features: self.synchronization_features.clone(),
            queue_family_identity_digest: self.queue_family_identity_digest.clone(),
            queue_family_queue_flags: self.queue_family_queue_flags,
            queue_family_queue_count: self.queue_family_queue_count,
            queue_family_timestamp_valid_bits: self.queue_family_timestamp_valid_bits,
            queue_family_min_image_transfer_granularity: self.queue_family_min_image_transfer_granularity,
            device_uuid: self.device_uuid,
            implementation_identity_digest: self.implementation_identity_digest.clone(),
            physical_device_identity_digest: self.physical_device_identity_digest.clone(),
            driver_identity_digest: self.driver_identity_digest.clone(),
            driver_uuid: self.driver_uuid,
            driver_id: self.driver_id,
        };
        receipt.verify_against(graph, schedule, plan, &observed).map_err(VulkanBarrierError::Receipt)?;
        qualification_stage("receipt_verified");
        receipt
            .verify_runtime_binding(
                self.physical_device_api_version,
                self.queue_family_index,
                self.device_uuid,
                &self.implementation_identity_digest,
                &self.physical_device_identity_digest,
                &self.driver_identity_digest,
                self.driver_uuid,
                self.driver_id,
                &self.synchronization_features,
                &self.queue_family_identity_digest,
                self.queue_family_queue_flags,
                self.queue_family_queue_count,
                self.queue_family_timestamp_valid_bits,
                self.queue_family_min_image_transfer_granularity,
            )
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

fn record_host_readback_barrier(
    device: &Device,
    command: vk::CommandBuffer,
    contract: &MaterializedSubmissionContract,
) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(contract.host_readback_src_stage_mask)
        .src_access_mask(contract.host_readback_src_access_mask)
        .dst_stage_mask(contract.host_readback_dst_stage_mask)
        .dst_access_mask(contract.host_readback_dst_access_mask);
    let dependency = vk::DependencyInfo::default()
        .dependency_flags(vk::DependencyFlags::empty())
        .memory_barriers(std::slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(command, &dependency); }
}

#[derive(Debug, Clone)]
struct MaterializedBarrierRecord {
    from: u32,
    to: u32,
    resource: ResourceId,
    kind: DependencyKind,
    buffer_memory: bool,
    storage_size: u64,
}

#[derive(Debug, Clone)]
struct MaterializedBarrierBatch {
    node_id: u32,
    records: Vec<MaterializedBarrierRecord>,
}

fn dependency_kind_label(kind: DependencyKind) -> &'static str {
    match kind {
        DependencyKind::ReadAfterWrite => "read_after_write",
        DependencyKind::WriteAfterRead => "write_after_read",
        DependencyKind::WriteAfterWrite => "write_after_write",
    }
}

fn access_mask_labels(kind: DependencyKind) -> (&'static str, &'static str) {
    match kind {
        DependencyKind::ReadAfterWrite => ("shader_storage_write", "shader_storage_read"),
        DependencyKind::WriteAfterRead => ("empty", "empty"),
        DependencyKind::WriteAfterWrite => ("shader_storage_write", "shader_storage_write"),
    }
}

fn materialized_barrier_batch(
    node_id: u32,
    requirements: &[VulkanBarrierRequirement],
    resource_storage_sizes: &BTreeMap<ResourceId, u64>,
) -> Result<Option<MaterializedBarrierBatch>, ResourceId> {
    if requirements.is_empty() {
        return Ok(None);
    }

    let mut records = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let buffer_memory = requirement.requires_memory_dependency();
        let storage_size = if buffer_memory {
            resource_storage_sizes
                .get(&requirement.resource)
                .copied()
                .ok_or_else(|| requirement.resource.clone())?
        } else {
            0
        };
        records.push(MaterializedBarrierRecord {
            from: requirement.from,
            to: requirement.to,
            resource: requirement.resource.clone(),
            kind: requirement.kind,
            buffer_memory,
            storage_size,
        });
    }
    Ok(Some(MaterializedBarrierBatch { node_id, records }))
}

fn materialized_barrier_batches_from_plan(
    plan: &VulkanSyncPlan,
    resource_storage_sizes: &BTreeMap<ResourceId, u64>,
) -> Result<Vec<MaterializedBarrierBatch>, ResourceId> {
    let mut batches = Vec::new();
    for submission in &plan.submissions {
        if let Some(batch) = materialized_barrier_batch(
            submission.node_id,
            &submission.barriers,
            resource_storage_sizes,
        )? {
            batches.push(batch);
        }
    }
    Ok(batches)
}

fn materialized_barrier_batches_digest(batches: &[MaterializedBarrierBatch]) -> String {
    let mut fields = vec![format!("batch_count:{}", batches.len())];
    for batch in batches {
        let memory_count = batch.records.iter().filter(|record| !record.buffer_memory).count();
        let buffer_count = batch.records.iter().filter(|record| record.buffer_memory).count();
        fields.extend([
            "batch".to_owned(),
            format!("node_id={}", batch.node_id),
            "dependency_structure=VkDependencyInfo".to_owned(),
            "pnext=null".to_owned(),
            "dependency_flags=0".to_owned(),
            format!("memory_barrier_count={memory_count}"),
            format!("buffer_memory_barrier_count={buffer_count}"),
            "image_memory_barrier_count=0".to_owned(),
        ]);
        for (ordinal, record) in batch.records.iter().enumerate() {
            let (src_access, dst_access) = access_mask_labels(record.kind);
            let (src_access_mask, dst_access_mask) = barrier_access_masks(record.kind);
            let stage_mask = vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw();
            let (barrier_type, src_queue_family_index, dst_queue_family_index) =
                if record.buffer_memory {
                    (
                        "VkBufferMemoryBarrier2",
                        vk::QUEUE_FAMILY_IGNORED.to_string(),
                        vk::QUEUE_FAMILY_IGNORED.to_string(),
                    )
                } else {
                    (
                        "VkMemoryBarrier2",
                        "not_applicable".to_owned(),
                        "not_applicable".to_owned(),
                    )
                };
            fields.extend([
                "barrier".to_owned(),
                format!("ordinal={ordinal}"),
                format!("from={}", record.from),
                format!("to={}", record.to),
                format!("resource={}", record.resource.as_str()),
                format!("kind={}", dependency_kind_label(record.kind)),
                format!("type={barrier_type}"),
                "pnext=null".to_owned(),
                "src_stage=compute_shader".to_owned(),
                format!("src_stage_mask={stage_mask}"),
                format!("src_access={src_access}"),
                format!("src_access_mask={}", src_access_mask.as_raw()),
                "dst_stage=compute_shader".to_owned(),
                format!("dst_stage_mask={stage_mask}"),
                format!("dst_access={dst_access}"),
                format!("dst_access_mask={}", dst_access_mask.as_raw()),
                format!("src_queue_family_index={src_queue_family_index}"),
                format!("dst_queue_family_index={dst_queue_family_index}"),
                "offset=0".to_owned(),
                format!("size={}", record.storage_size),
            ]);
        }
    }

    let mut digest = Sha256::new();
    digest.update(b"symthaea.gpu-fabric.vulkan-materialized-barriers.v2");
    digest.update([0]);
    for field in fields {
        let bytes = field.as_bytes();
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

fn record_barriers(
    device: &Device,
    command: vk::CommandBuffer,
    node_id: u32,
    requirements: &[VulkanBarrierRequirement],
    buffers: &BTreeMap<ResourceId, WorkloadBuffer>,
) -> Result<Option<MaterializedBarrierBatch>, VulkanBarrierError> {
    if requirements.is_empty() {
        return Ok(None);
    }
    let storage_sizes = buffers
        .iter()
        .map(|(resource, buffer)| (resource.clone(), buffer.storage_size))
        .collect::<BTreeMap<_, _>>();
    let batch = materialized_barrier_batch(node_id, requirements, &storage_sizes)
        .map_err(VulkanBarrierError::MissingResource)?
        .ok_or(VulkanBarrierError::AllocationOverflow)?;

    let mut buffer_barriers = Vec::new();
    let mut execution_barriers = Vec::new();
    for record in &batch.records {
        let (src_access, dst_access) = barrier_access_masks(record.kind);
        if record.buffer_memory {
            let buffer = buffers
                .get(&record.resource)
                .ok_or_else(|| VulkanBarrierError::MissingResource(record.resource.clone()))?;
            buffer_barriers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(dst_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.buffer)
                    .offset(0)
                    .size(record.storage_size),
            );
        } else {
            execution_barriers.push(
                vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(dst_access),
            );
        }
    }
    let dependency = vk::DependencyInfo::default()
        .dependency_flags(vk::DependencyFlags::empty())
        .memory_barriers(&execution_barriers)
        .buffer_memory_barriers(&buffer_barriers);
    unsafe { device.cmd_pipeline_barrier2(command, &dependency); }
    Ok(Some(batch))
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

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn resource_digest(value: &BinaryHypervector) -> String {
    let mut h = Hasher::new();
    h.update(b"symthaea.gpu-fabric.vulkan-barrier-resource.v1\0");
    h.update(&value.dimensions.to_le_bytes());
    h.update(value.as_bytes());
    h.finalize().to_hex().to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_len_prefixed_update(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn spirv_to_bytes(spirv: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(spirv.len() * std::mem::size_of::<u32>());
    for word in spirv {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn vulkan_implementation_identity_digest(spirv: &[u32]) -> String {
    let spirv_bytes = spirv_to_bytes(spirv);
    let mut hasher = Sha256::new();
    hasher.update(VULKAN_IMPLEMENTATION_IDENTITY_VERSION.as_bytes());
    hasher.update([0]);
    sha256_len_prefixed_update(&mut hasher, WGSL_ABI_MARKER.as_bytes());
    sha256_len_prefixed_update(&mut hasher, HDC_BIND_XOR_KERNEL_ID.as_bytes());
    sha256_len_prefixed_update(&mut hasher, VULKAN_ENTRY_POINT.as_bytes());
    sha256_len_prefixed_update(&mut hasher, VULKAN_SHADER_STAGE.as_bytes());
    sha256_len_prefixed_update(&mut hasher, WGSL.as_bytes());
    sha256_len_prefixed_update(&mut hasher, &spirv_bytes);
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn physical_device_identity_digest(props: &vk::PhysicalDeviceProperties) -> String {
    let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_bytes();
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.gpu-fabric.vulkan-device.v1\0");
    hasher.update(&props.vendor_id.to_le_bytes());
    hasher.update(&props.device_id.to_le_bytes());
    hasher.update(&(props.device_type.as_raw() as u32).to_le_bytes());
    hasher.update(&props.api_version.to_le_bytes());
    hasher.update(&props.driver_version.to_le_bytes());
    sha256_len_prefixed_update(&mut hasher, device_name);
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn synchronization_feature_identity_digest(
    timeline_semaphore_supported: bool,
    synchronization2_supported: bool,
    timeline_semaphore_enabled: bool,
    synchronization2_enabled: bool,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SYNCHRONIZATION_FEATURE_IDENTITY_VERSION.as_bytes());
    hasher.update([0]);
    hasher.update([
        u8::from(timeline_semaphore_supported),
        u8::from(synchronization2_supported),
        u8::from(timeline_semaphore_enabled),
        u8::from(synchronization2_enabled),
    ]);
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn queue_family_identity_digest(index: u32, properties: &vk::QueueFamilyProperties) -> String {
    queue_family_identity_digest_from_fields(
        index,
        properties.queue_flags.as_raw(),
        properties.queue_count,
        properties.timestamp_valid_bits,
        [
            properties.min_image_transfer_granularity.width,
            properties.min_image_transfer_granularity.height,
            properties.min_image_transfer_granularity.depth,
        ],
    )
}

fn queue_family_identity_digest_from_fields(
    index: u32,
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(QUEUE_FAMILY_IDENTITY_VERSION.as_bytes());
    hasher.update([0]);
    hasher.update(&index.to_le_bytes());
    hasher.update(&queue_flags.to_le_bytes());
    hasher.update(&queue_count.to_le_bytes());
    hasher.update(&timestamp_valid_bits.to_le_bytes());
    for value in min_image_transfer_granularity {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn driver_identity_digest(
    driver_uuid: [u8; 16],
    driver_id: i32,
    driver_name: &[u8],
    driver_info: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DRIVER_IDENTITY_VERSION.as_bytes());
    hasher.update([0]);
    sha256_len_prefixed_update(&mut hasher, &driver_uuid);
    hasher.update(&driver_id.to_le_bytes());
    sha256_len_prefixed_update(&mut hasher, driver_name);
    sha256_len_prefixed_update(&mut hasher, driver_info);
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| (b'0'..=b'9').contains(&byte) || (b'a'..=b'f').contains(&byte))
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

fn dispatch_group_count(range: u64, max_groups_x: u32) -> Option<u32> {
    let elements = range / 4;
    let groups = elements.saturating_add(u64::from(WORKGROUP_SIZE - 1)) / u64::from(WORKGROUP_SIZE);
    let groups = groups.max(1);
    if groups <= u64::from(max_groups_x) {
        Some(groups as u32)
    } else {
        None
    }
}

fn expected_final_timeline_value(plan: &VulkanSyncPlan) -> u64 {
    plan.submissions
        .iter()
        .map(|submission| submission.signal.value)
        .max()
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
struct PlannedSubmissionRecord {
    node_id: u32,
    ordinal: u32,
    queue_index: u32,
    signal_value: u64,
}

struct MaterializedSubmissionContract {
    queue_family_index: u32,
    queue_index: u32,
    semaphore_type: vk::SemaphoreType,
    semaphore_create_flags: vk::SemaphoreCreateFlags,
    timeline_initial_value: u64,
    command_buffer_device_mask: u32,
    submit_flags: vk::SubmitFlags,
    signal_value: u64,
    signal_stage_mask: vk::PipelineStageFlags2,
    semaphore_device_index: u32,
    semaphore_wait_flags: vk::SemaphoreWaitFlags,
    timeout_ns: u64,
    host_readback_src_stage_mask: vk::PipelineStageFlags2,
    host_readback_src_access_mask: vk::AccessFlags2,
    host_readback_dst_stage_mask: vk::PipelineStageFlags2,
    host_readback_dst_access_mask: vk::AccessFlags2,
    planned_submissions: Vec<PlannedSubmissionRecord>,
}

impl MaterializedSubmissionContract {
    fn from_plan(
        plan: &VulkanSyncPlan,
        signal_value: u64,
        queue_family_index: u32,
    ) -> Self {
        Self {
            queue_family_index,
            queue_index: 0,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            semaphore_create_flags: vk::SemaphoreCreateFlags::empty(),
            timeline_initial_value: 0,
            command_buffer_device_mask: 1,
            submit_flags: vk::SubmitFlags::empty(),
            signal_value,
            signal_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            semaphore_device_index: 0,
            semaphore_wait_flags: vk::SemaphoreWaitFlags::empty(),
            timeout_ns: VULKAN_TIMELINE_TIMEOUT_NS,
            host_readback_src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            host_readback_src_access_mask: vk::AccessFlags2::SHADER_STORAGE_WRITE,
            host_readback_dst_stage_mask: vk::PipelineStageFlags2::HOST,
            host_readback_dst_access_mask: vk::AccessFlags2::HOST_READ,
            planned_submissions: plan
                .submissions
                .iter()
                .map(|submission| PlannedSubmissionRecord {
                    node_id: submission.node_id,
                    ordinal: submission.ordinal,
                    queue_index: submission.queue.get() as u32,
                    signal_value: submission.signal.value,
                })
                .collect(),
        }
    }

    fn digest(&self) -> String {
        let mut fields = vec![
            "contract_version=v1".to_owned(),
            format!("queue_family_index={}", self.queue_family_index),
            format!("queue_index={}", self.queue_index),
            "semaphore_create_structure=VkSemaphoreCreateInfo".to_owned(),
            format!("semaphore_create_flags={}", self.semaphore_create_flags.as_raw()),
            "semaphore_create_pnext=VkSemaphoreTypeCreateInfo".to_owned(),
            format!("semaphore_type_raw={}", self.semaphore_type.as_raw()),
            "semaphore_type=timeline".to_owned(),
            format!("timeline_initial_value={}", self.timeline_initial_value),
            "host_readback_dependency_structure=VkDependencyInfo".to_owned(),
            "host_readback_dependency_pnext=null".to_owned(),
            "host_readback_dependency_flags=0".to_owned(),
            "host_readback_memory_barrier_count=1".to_owned(),
            "host_readback_buffer_memory_barrier_count=0".to_owned(),
            "host_readback_image_memory_barrier_count=0".to_owned(),
            "host_readback_barrier_structure=VkMemoryBarrier2".to_owned(),
            "host_readback_barrier_pnext=null".to_owned(),
            "host_readback_src_stage=compute_shader".to_owned(),
            format!("host_readback_src_stage_mask={}", self.host_readback_src_stage_mask.as_raw()),
            "host_readback_src_access=shader_storage_write".to_owned(),
            format!("host_readback_src_access_mask={}", self.host_readback_src_access_mask.as_raw()),
            "host_readback_dst_stage=host".to_owned(),
            format!("host_readback_dst_stage_mask={}", self.host_readback_dst_stage_mask.as_raw()),
            "host_readback_dst_access=host_read".to_owned(),
            format!("host_readback_dst_access_mask={}", self.host_readback_dst_access_mask.as_raw()),
            "host_readback_queue_family_indices=not_applicable".to_owned(),
            "host_readback_offset=0".to_owned(),
            "host_readback_size=0".to_owned(),
            "submit_structure=VkSubmitInfo2".to_owned(),
            "submit_pnext=null".to_owned(),
            format!("submit_flags={}", self.submit_flags.as_raw()),
            "wait_semaphore_count=0".to_owned(),
            "command_buffer_count=1".to_owned(),
            "signal_semaphore_count=1".to_owned(),
            "command_buffer_structure=VkCommandBufferSubmitInfo".to_owned(),
            "command_buffer_pnext=null".to_owned(),
            format!("command_buffer_device_mask={}", self.command_buffer_device_mask),
            "signal_structure=VkSemaphoreSubmitInfo".to_owned(),
            "signal_pnext=null".to_owned(),
            format!("signal_value={}", self.signal_value),
            "signal_stage=all_commands".to_owned(),
            format!("signal_stage_mask={}", self.signal_stage_mask.as_raw()),
            format!("signal_device_index={}", self.semaphore_device_index),
            "wait_structure=VkSemaphoreWaitInfo".to_owned(),
            "wait_pnext=null".to_owned(),
            format!("wait_flags={}", self.semaphore_wait_flags.as_raw()),
            "wait_semaphore_count=1".to_owned(),
            format!("wait_value={}", self.signal_value),
            format!("timeout_ns={}", self.timeout_ns),
            "counter_query=vkGetSemaphoreCounterValue".to_owned(),
            format!("planned_submission_count={}", self.planned_submissions.len()),
        ];
        for submission in &self.planned_submissions {
            fields.extend([
                "planned_submission".to_owned(),
                format!("node_id={}", submission.node_id),
                format!("ordinal={}", submission.ordinal),
                format!("queue_index={}", submission.queue_index),
                format!("signal_value={}", submission.signal_value),
            ]);
        }

        let mut digest = Sha256::new();
        digest.update(b"symthaea.gpu-fabric.vulkan-materialized-submission.v1");
        digest.update([0]);
        for field in fields {
            let bytes = field.as_bytes();
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
        format!("{:x}", digest.finalize())
    }
}

fn completion_lowering_digest(
    plan: &VulkanSyncPlan,
    completion_expected: u64,
    queue_family_index: u32,
) -> String {
    MaterializedSubmissionContract::from_plan(plan, completion_expected, queue_family_index).digest()
}

fn barrier_lowering_digest(
    plan: &VulkanSyncPlan,
    resource_storage_sizes: &BTreeMap<ResourceId, u64>,
) -> Result<String, ResourceId> {
    let batches = materialized_barrier_batches_from_plan(plan, resource_storage_sizes)?;
    Ok(materialized_barrier_batches_digest(&batches))
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
    const TEST_DRIVER_IDENTITY_DIGEST: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";

    const TEST_IMPLEMENTATION_IDENTITY_DIGEST: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";
    const TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const TEST_QUEUE_FAMILY_IDENTITY_DIGEST: &str =
        "6994094896a7f07fa1248d36af11e388aa7c0e6eb0ed2f99fff9b4beb4fd2978";

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
    fn dispatch_group_count_rejects_u64_to_u32_truncation() {
        let range = (u64::from(u32::MAX) + 1)
            .saturating_mul(u64::from(WORKGROUP_SIZE))
            .saturating_mul(4);
        assert_eq!(dispatch_group_count(range, u32::MAX), None);
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
    fn materialized_barrier_digest_binds_concrete_call_fields() {
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
        let mut batches =
            materialized_barrier_batches_from_plan(&plan, &storage_sizes).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].records.len(), 1);
        let baseline = materialized_barrier_batches_digest(&batches);

        batches[0].records[0].storage_size += 4;
        assert_ne!(baseline, materialized_barrier_batches_digest(&batches));
        batches[0].records[0].storage_size -= 4;

        batches[0].records[0].kind = DependencyKind::WriteAfterWrite;
        assert_ne!(baseline, materialized_barrier_batches_digest(&batches));
        batches[0].records[0].kind = DependencyKind::ReadAfterWrite;

        batches[0].node_id += 1;
        assert_ne!(baseline, materialized_barrier_batches_digest(&batches));
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
    fn receipt_rejects_empty_workload() {
        let graph = ExecutionGraph::new(Vec::new(), Vec::new()).unwrap();
        let schedule = ExecutionSchedule::from_graph(&graph).unwrap();
        let plan = VulkanSyncPlan::from_schedule(&schedule, &[]).unwrap();
        let receipt = VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: String::new(),
            schedule_digest: String::new(),
            sync_plan_digest: String::new(),
            barrier_digest: String::new(),
            barrier_lowering_digest: String::new(),
            completion_lowering_digest: String::new(),
            node_count: 0,
            barrier_count: 0,
            resource_digests: BTreeMap::new(),
            resource_storage_sizes: BTreeMap::new(),
            completion_expected: 0,
            completion_observed: 0,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
        };
        assert_eq!(
            receipt.verify_against(&graph, &schedule, &plan, &BTreeMap::new()),
            Err(VulkanBarrierReceiptError::EmptyWorkload)
        );
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
        };
        let expected = expected_final_timeline_value(&plan);
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::TimelineCompletion {
                expected: observed_expected,
                observed: 0,
            }) if observed_expected == expected
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
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

        let (_, _, mut tampered_plan, _) = fixture();
        let expected = expected_final_timeline_value(&tampered_plan);
        let baseline = completion_lowering_digest(&tampered_plan, expected, 0);
        tampered_plan.submissions[0].signal.value += 9;
        assert_ne!(
            baseline,
            completion_lowering_digest(&tampered_plan, expected, 0)
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::TimelineCompletion {
                expected: observed_expected,
                observed,
            }) if observed_expected == expected && observed == expected + 1
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
        };
        receipt.completion_lowering_digest = String::from("tampered");
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::CompletionLoweringDigest)
        ));
    }

    #[test]
    fn receipt_rejects_runtime_device_and_queue_binding_mismatch() {
        let mut receipt = minimal_receipt_for_binding_tests();
        assert!(receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ).is_ok());

        receipt.physical_device_api_version = VULKAN_API_VERSION + 1;
        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::PhysicalDeviceApiVersionBinding)
        ));

        receipt.physical_device_api_version = VULKAN_API_VERSION;
        receipt.queue_family_index = 1;
        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::QueueFamilyBinding)
        ));
    }

    #[test]
    fn receipt_rejects_runtime_provenance_binding_mismatch() {
        let mut receipt = minimal_receipt_for_binding_tests();
        assert!(receipt
            .verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            )
            .is_ok());

        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                "2222222222222222222222222222222222222222222222222222222222222222",
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::ImplementationIdentityBinding)
        ));

        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                "3333333333333333333333333333333333333333333333333333333333333333",
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::PhysicalDeviceIdentityBinding)
        ));

        receipt.device_uuid = [9; 16];
        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::DeviceUuidBinding)
        ));

        receipt.device_uuid = [1; 16];
        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [9; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::DriverUuidBinding)
        ));

        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                9,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::DriverIdBinding)
        ));

        let mut tampered_features = test_synchronization_feature_profile();
        tampered_features.synchronization2_supported = false;
        tampered_features.identity_digest = synchronization_feature_identity_digest(
            tampered_features.timeline_semaphore_supported,
            tampered_features.synchronization2_supported,
            tampered_features.timeline_semaphore_enabled,
            tampered_features.synchronization2_enabled,
        );
        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &tampered_features,
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::SynchronizationFeatureIdentityBinding)
        ));

        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                "4444444444444444444444444444444444444444444444444444444444444444",
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::QueueFamilyIdentityBinding)
        ));

        assert!(matches!(
            receipt.verify_runtime_binding(
                VULKAN_API_VERSION,
                0,
                [1; 16],
                TEST_IMPLEMENTATION_IDENTITY_DIGEST,
                TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST,
                TEST_DRIVER_IDENTITY_DIGEST,
                [2; 16],
                1,
                &test_synchronization_feature_profile(),
                TEST_QUEUE_FAMILY_IDENTITY_DIGEST,
                vk::QueueFlags::COMPUTE.as_raw(),
                2,
                0,
                [1, 1, 1],
            ),
            Err(VulkanBarrierReceiptError::QueueFamilyIdentityBinding)
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
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 7,
            queue_family_identity_digest: queue_family_identity_digest_from_fields(
                7,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            ),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
        };
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::CompletionLoweringDigest)
        ));
    }

    fn test_synchronization_feature_profile() -> VulkanSynchronizationFeatureProfile {
        VulkanSynchronizationFeatureProfile::new(true, true, true, true)
    }

    fn minimal_receipt_for_binding_tests() -> VulkanBarrierExecutionReceipt {
        VulkanBarrierExecutionReceipt {
            version: RECEIPT_VERSION,
            graph_digest: String::new(),
            schedule_digest: String::new(),
            sync_plan_digest: String::new(),
            barrier_digest: String::new(),
            barrier_lowering_digest: String::new(),
            completion_lowering_digest: String::new(),
            node_count: 0,
            barrier_count: 0,
            resource_digests: BTreeMap::new(),
            resource_storage_sizes: BTreeMap::new(),
            completion_expected: 0,
            completion_observed: 0,
            vulkan_api_version: VULKAN_API_VERSION,
            physical_device_api_version: VULKAN_API_VERSION,
            synchronization_features: test_synchronization_feature_profile(),
            queue_family_index: 0,
            queue_family_identity_digest: TEST_QUEUE_FAMILY_IDENTITY_DIGEST.to_owned(),
            queue_family_queue_flags: vk::QueueFlags::COMPUTE.as_raw(),
            queue_family_queue_count: 1,
            queue_family_timestamp_valid_bits: 0,
            queue_family_min_image_transfer_granularity: [1, 1, 1],
            device_uuid: [1; 16],
            implementation_identity_digest: TEST_IMPLEMENTATION_IDENTITY_DIGEST.to_owned(),
            physical_device_identity_digest: TEST_PHYSICAL_DEVICE_IDENTITY_DIGEST.to_owned(),
            driver_identity_digest: TEST_DRIVER_IDENTITY_DIGEST.to_owned(),
            driver_uuid: [2; 16],
            driver_id: 1,
        }
    }

    fn print_receipt_evidence(
        label: &str,
        initial: &BTreeMap<ResourceId, BinaryHypervector>,
        observed: &BTreeMap<ResourceId, BinaryHypervector>,
        receipt: &VulkanBarrierExecutionReceipt,
        runtime: &VulkanBarrierWorkloadRuntime,
    ) {
        println!("qualification_witness_version=1");
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
        println!("synchronization_feature_identity_version=1");
        println!("timeline_semaphore_supported={}", u8::from(receipt.synchronization_features.timeline_semaphore_supported));
        println!("synchronization2_supported={}", u8::from(receipt.synchronization_features.synchronization2_supported));
        println!("timeline_semaphore_enabled={}", u8::from(receipt.synchronization_features.timeline_semaphore_enabled));
        println!("synchronization2_enabled={}", u8::from(receipt.synchronization_features.synchronization2_enabled));
        println!("synchronization_feature_identity_sha256={}", receipt.synchronization_features.identity_digest);
        println!("queue_family_identity_version=1");
        println!("queue_family_identity_sha256={}", receipt.queue_family_identity_digest);
        println!("queue_family_queue_flags={}", receipt.queue_family_queue_flags);
        println!("queue_family_queue_count={}", receipt.queue_family_queue_count);
        println!("queue_family_timestamp_valid_bits={}", receipt.queue_family_timestamp_valid_bits);
        println!(
            "queue_family_min_image_transfer_granularity={},{},{}",
            receipt.queue_family_min_image_transfer_granularity[0],
            receipt.queue_family_min_image_transfer_granularity[1],
            receipt.queue_family_min_image_transfer_granularity[2],
        );
        println!("device_uuid={}", hex_bytes(&receipt.device_uuid));
        println!("implementation_identity_version=1");
        println!("implementation_identity_sha256={}", receipt.implementation_identity_digest);
        println!("implementation_abi_marker={}", WGSL_ABI_MARKER);
        println!("implementation_kernel_id={}", HDC_BIND_XOR_KERNEL_ID);
        println!("implementation_entry_point={}", VULKAN_ENTRY_POINT);
        println!("implementation_shader_stage={}", VULKAN_SHADER_STAGE);
        println!("implementation_wgsl_sha256={}", runtime.implementation_wgsl_sha256);
        println!("implementation_wgsl_hex={}", runtime.implementation_wgsl_hex);
        println!("shader_spirv_sha256={}", runtime.shader_spirv_sha256);
        println!("shader_spirv_hex={}", runtime.shader_spirv_hex);
        println!("physical_device_identity_version=1");
        println!("physical_device_identity_sha256={}", receipt.physical_device_identity_digest);
        println!("physical_device_vendor_id={}", runtime.physical_device_vendor_id);
        println!("physical_device_device_id={}", runtime.physical_device_device_id);
        println!("physical_device_type={}", runtime.physical_device_type);
        println!("physical_device_driver_version={}", runtime.physical_device_driver_version);
        println!("physical_device_name_hex={}", runtime.physical_device_name_hex);
        println!("driver_identity_version=1");
        println!("driver_identity_sha256={}", receipt.driver_identity_digest);
        println!("driver_uuid={}", hex_bytes(&receipt.driver_uuid));
        println!("driver_id={}", receipt.driver_id);
        println!("driver_name_hex={}", runtime.driver_name_hex);
        println!("driver_info_hex={}", runtime.driver_info_hex);
        for (resource, value) in initial {
            println!("resource_initial_hex={}:{}:{}", resource.as_str(), value.dimensions, hex_bytes(value.as_bytes()));
        }
        for (resource, value) in observed {
            println!("resource_observed_hex={}:{}:{}", resource.as_str(), value.dimensions, hex_bytes(value.as_bytes()));
        }
    }

    #[test]
    fn implementation_identity_digest_is_deterministic_and_spirv_bound() {
        let baseline = vulkan_implementation_identity_digest(&[0x07230203, 0x00010000]);
        assert_eq!(
            baseline,
            vulkan_implementation_identity_digest(&[0x07230203, 0x00010000])
        );
        assert_ne!(
            baseline,
            vulkan_implementation_identity_digest(&[0x07230203, 0x00010001])
        );
    }

    #[test]
    fn physical_device_identity_digest_binds_device_and_driver_fields() {
        let mut props = vk::PhysicalDeviceProperties::default();
        props.vendor_id = 1;
        props.device_id = 2;
        props.device_type = vk::PhysicalDeviceType::CPU;
        props.api_version = VULKAN_API_VERSION;
        props.driver_version = 3;
        let baseline = physical_device_identity_digest(&props);

        props.driver_version += 1;
        assert_ne!(baseline, physical_device_identity_digest(&props));

        props.driver_version = 3;
        props.vendor_id += 1;
        assert_ne!(baseline, physical_device_identity_digest(&props));
    }

    #[test]
    fn receipt_rejects_malformed_provenance_identity() {
        let (graph, schedule, plan, initial) = fixture();
        let final_state = simulate(&graph, &schedule, &initial).unwrap();
        let mut receipt = minimal_receipt_for_binding_tests();
        receipt.implementation_identity_digest = "tampered".to_owned();
        assert!(matches!(
            receipt.verify_against(&graph, &schedule, &plan, &final_state),
            Err(VulkanBarrierReceiptError::ImplementationIdentity)
        ));
    }

    #[test]
    fn synchronization_feature_profile_is_derived_from_device_create_values() {
        let timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
            .timeline_semaphore(true);
        let sync2 = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);

        let enabled = synchronization_feature_profile_from_device_create(
            true,
            true,
            timeline.timeline_semaphore,
            sync2.synchronization2,
        );
        assert!(enabled.verify().is_ok());

        let disabled = synchronization_feature_profile_from_device_create(
            true,
            true,
            timeline.timeline_semaphore,
            0,
        );
        assert_ne!(enabled.identity_digest, disabled.identity_digest);
        assert!(!disabled.synchronization2_enabled);
        assert_eq!(
            disabled.verify(),
            Err(VulkanBarrierReceiptError::SynchronizationFeatureIdentity)
        );
    }

    #[test]
    fn queue_family_identity_digest_binds_capability_tuple() {
        let mut properties = vk::QueueFamilyProperties::default()
            .queue_flags(vk::QueueFlags::COMPUTE)
            .queue_count(1)
            .timestamp_valid_bits(0)
            .min_image_transfer_granularity(vk::Extent3D { width: 1, height: 1, depth: 1 });
        let baseline = queue_family_identity_digest(3, &properties);

        properties.queue_flags |= vk::QueueFlags::TRANSFER;
        assert_ne!(baseline, queue_family_identity_digest(3, &properties));
        properties.queue_flags = vk::QueueFlags::COMPUTE;

        properties.queue_count += 1;
        assert_ne!(baseline, queue_family_identity_digest(3, &properties));
        properties.queue_count = 1;

        properties.timestamp_valid_bits = 64;
        assert_ne!(baseline, queue_family_identity_digest(3, &properties));
        properties.timestamp_valid_bits = 0;

        let mut unsupported = test_synchronization_feature_profile();
        unsupported.timeline_semaphore_supported = false;
        assert_eq!(
            unsupported.verify(),
            Err(VulkanBarrierReceiptError::SynchronizationFeatureIdentity)
        );

        properties.min_image_transfer_granularity.width = 2;
        assert_ne!(baseline, queue_family_identity_digest(3, &properties));
        assert_ne!(
            baseline,
            queue_family_identity_digest_from_fields(
                4,
                vk::QueueFlags::COMPUTE.as_raw(),
                1,
                0,
                [1, 1, 1],
            )
        );
    }

    #[test]
    fn driver_identity_digest_binds_uuid_id_and_metadata() {
        let baseline = driver_identity_digest([1; 16], 7, b"driver", b"info");
        assert_ne!(baseline, driver_identity_digest([2; 16], 7, b"driver", b"info"));
        assert_ne!(baseline, driver_identity_digest([1; 16], 8, b"driver", b"info"));
        assert_ne!(baseline, driver_identity_digest([1; 16], 7, b"driver2", b"info"));
        assert_ne!(baseline, driver_identity_digest([1; 16], 7, b"driver", b"info2"));
    }

    #[test]
    #[ignore = "requires a Vulkan 1.3 validation runner"]
    fn real_vulkan_runtime_constructs_and_drops() {
        qualification_stage("preflight_test_begin");
        qualification_stage("preflight_before_runtime_new");
        let runtime = VulkanBarrierWorkloadRuntime::new()
            .expect("qualified Vulkan 1.3 synchronization2 timeline device");
        qualification_stage("preflight_runtime_constructed");
        drop(runtime);
        qualification_stage("preflight_runtime_dropped");
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
                &initial,
                &observed,
                &receipt,
                &runtime,
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

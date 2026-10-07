//! Native Vulkan execution backend for the Symthaea GPU Fabric.
//!
//! This module implements the first concrete accelerated operation,
//! HdcBindXor, against the backend-neutral contract in the parent crate.
//! The CPU reference executor remains the semantic oracle.

use std::ffi::CString;
use std::ptr;

use ash::{vk, Device, Entry, Instance};
use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use thiserror::Error;

use crate::{
    digest_hypervector, digest_hypervectors, BackendKind, BinaryHypervector, ExecutionReceipt,
    GpuOperation, OperationPlan, ReceiptError, VectorError,
};

const WORKGROUP_SIZE: u32 = 64;
const VULKAN_IMPLEMENTATION_ABI: &[u8] = b"storage-u32-xor-v1\0";

/// WGSL for the first native Symthaea HDC-XOR kernel.
pub const VULKAN_HDC_XOR_WGSL: &str = r#"
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanDevicePolicy {
    /// Accept any compute-capable Vulkan device, including software Vulkan.
    AllowSoftware,
    /// Reject Vulkan devices reported as CPU implementations. This is a
    /// stronger hardware-oriented gate but does not by itself prove bare-metal
    /// physical attachment when a hypervisor presents a virtual GPU.
    HardwareRequired,
}

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Vulkan loader unavailable: {0}")]
    Loader(String),
    #[error("Vulkan call failed: {0:?}")]
    Vk(vk::Result),
    #[error("no Vulkan physical device with a compute queue was found")]
    NoComputeDevice,
    #[error("no non-CPU Vulkan compute device was found")]
    NoHardwareComputeDevice,
    #[error("Vulkan device name is empty")]
    MissingDeviceName,
    #[error("no host-visible Vulkan memory type is available")]
    NoHostVisibleMemory,
    #[error(
        "Vulkan device cannot support the fixed 64-thread HDC kernel workgroup"
    )]
    InsufficientComputeWorkgroupLimits,
    #[error("storage buffer range {bytes} exceeds device limit {max}")]
    StorageBufferRangeExceeded { bytes: u64, max: u64 },
    #[error("compute dispatch group count {groups} exceeds device limit {max}")]
    DispatchGroupCountExceeded { groups: u32, max: u32 },
    #[error("shader WGSL parsing failed: {0}")]
    ShaderParse(String),
    #[error("shader validation failed: {0}")]
    ShaderValidation(String),
    #[error("SPIR-V generation failed: {0}")]
    ShaderSpirv(String),
    #[error("HDC dimensions are zero")]
    ZeroDimensions,
    #[error("expected {expected} inputs, got {actual}")]
    InputCount { expected: usize, actual: usize },
    #[error("input dimensions mismatch: expected {expected}, left={left}, right={right}")]
    DimensionMismatch {
        expected: u32,
        left: u32,
        right: u32,
    },
    #[error("physical Vulkan storage allocation size overflow")]
    AllocationSizeOverflow,
    #[error("receipt verification failed: {0}")]
    Receipt(ReceiptError),
    #[error("invalid vector: {0}")]
    Vector(VectorError),
    #[error("invalid operation plan: {0}")]
    Plan(crate::PlanError),
}

/// Concrete device metadata used by accelerated receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanDeviceIdentity {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub api_version: u32,
    pub driver_version: u32,
}

impl VulkanDeviceIdentity {
    fn from_properties(properties: &vk::PhysicalDeviceProperties) -> Result<Self, VulkanError> {
        let name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_owned();

        if name.is_empty() {
            return Err(VulkanError::MissingDeviceName);
        }

        Ok(Self {
            name,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: format!("{:?}", properties.device_type),
            api_version: properties.api_version,
            driver_version: properties.driver_version,
        })
    }

    fn receipt_device_identity(&self) -> String {
        format!(
            "vulkan:vendor={:#x}:device={:#x}:type={}:name={}",
            self.vendor_id, self.device_id, self.device_type, self.name
        )
    }

    fn receipt_driver_identity(&self) -> String {
        format!(
            "vulkan-driver:vendor={:#x}:device={:#x}:api={:#x}:driver={:#x}",
            self.vendor_id,
            self.device_id,
            self.api_version,
            self.driver_version
        )
    }
}

/// First-generation native Vulkan executor.
///
/// This intentionally serializes each execution through a fence and uses
/// host-visible storage buffers. The design prioritizes semantic correctness
/// and receipt integrity before asynchronous scheduling and allocator tuning.
pub struct VulkanExecutor {
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_module: vk::ShaderModule,
    device_identity: VulkanDeviceIdentity,
    implementation_digest: String,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    claims_acceleration: bool,
    max_storage_buffer_range: u64,
    max_compute_workgroup_count_x: u32,
}

impl VulkanExecutor {
    /// Create a native Vulkan executor, allowing software Vulkan devices.
    pub fn new() -> Result<Self, VulkanError> {
        Self::new_with_policy(VulkanDevicePolicy::AllowSoftware)
    }

    /// Create a native Vulkan executor that rejects devices reported as CPU
    /// implementations. This is the hardware-oriented entry point; actual
    /// physical attachment remains an independently qualified property.
    pub fn new_hardware() -> Result<Self, VulkanError> {
        Self::new_with_policy(VulkanDevicePolicy::HardwareRequired)
    }

    pub fn new_with_policy(policy: VulkanDevicePolicy) -> Result<Self, VulkanError> {
        let spirv = compile_spirv()?;
        let implementation_digest = implementation_digest(&spirv);

        let entry =
            unsafe { Entry::load() }.map_err(|error| VulkanError::Loader(error.to_string()))?;

        let app_name =
            CString::new("symthaea-gpu-fabric").expect("static Vulkan name has no NUL bytes");
        let engine_name = CString::new("Symthaea").expect("static Vulkan name has no NUL bytes");

        let application_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_0);

        let instance_create_info =
            vk::InstanceCreateInfo::default().application_info(&application_info);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(VulkanError::Vk)?
        };

        Self::from_instance(instance, &spirv, implementation_digest, policy)
    }

    fn from_instance(
        instance: Instance,
        spirv: &[u32],
        implementation_digest: String,
        policy: VulkanDevicePolicy,
    ) -> Result<Self, VulkanError> {
        let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(VulkanError::Vk(error));
            }
        };

        let mut selected = None;

        for physical_device in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            if policy == VulkanDevicePolicy::HardwareRequired
                && properties.device_type == vk::PhysicalDeviceType::CPU
            {
                continue;
            }

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            let queue_family = queue_families
                .iter()
                .enumerate()
                .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32);

            if let Some(queue_family_index) = queue_family {
                let score = if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    2_u8
                } else {
                    1_u8
                };

                if selected
                    .as_ref()
                    .is_none_or(|(_, _, current_score)| score > *current_score)
                {
                    selected = Some((physical_device, queue_family_index, score));
                }
            }
        }

        let (physical_device, queue_family_index, _) = match selected {
            Some(selected) => selected,
            None => {
                unsafe { instance.destroy_instance(None) };
                return Err(match policy {
                    VulkanDevicePolicy::AllowSoftware => VulkanError::NoComputeDevice,
                    VulkanDevicePolicy::HardwareRequired => {
                        VulkanError::NoHardwareComputeDevice
                    }
                });
            }
        };

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        if properties.limits.max_compute_work_group_invocations < WORKGROUP_SIZE
            || properties.limits.max_compute_work_group_size[0] < WORKGROUP_SIZE
        {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::InsufficientComputeWorkgroupLimits);
        }

        let max_storage_buffer_range = u64::from(properties.limits.max_storage_buffer_range);
        let max_compute_workgroup_count_x = properties.limits.max_compute_work_group_count[0];

        let device_identity = match VulkanDeviceIdentity::from_properties(&properties) {
            Ok(identity) => identity,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device = match unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        } {
            Ok(device) => device,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(VulkanError::Vk(error));
            }
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let shader_module = match create_shader_module(&device, spirv) {
            Ok(module) => module,
            Err(error) => {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(error);
            }
        };

        let descriptor_set_layout = match create_descriptor_set_layout(&device) {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(shader_module, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(error);
            }
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = match unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanError::Vk(error));
            }
        };

        let entry_point = CString::new("main").expect("static shader entry point has no NUL bytes");
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);
        let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(pipeline_layout);

        let pipeline = match unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_create_info),
                None,
            )
        } {
            Ok(mut pipelines) => pipelines
                .pop()
                .expect("one compute pipeline was requested"),
            Err((_, error)) => {
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanError::Vk(error));
            }
        };

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);

        let command_pool = match unsafe { device.create_command_pool(&command_pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanError::Vk(error));
            }
        };

        let descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3);

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&descriptor_pool_size));

        let descriptor_pool = match unsafe {
            device.create_descriptor_pool(&descriptor_pool_info, None)
        } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(VulkanError::Vk(error));
            }
        };

        Ok(Self {
            instance,
            device,
            queue,
            command_pool,
            descriptor_set_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
            shader_module,
            device_identity,
            implementation_digest,
            memory_properties,
            max_storage_buffer_range,
            max_compute_workgroup_count_x,
            claims_acceleration: policy == VulkanDevicePolicy::HardwareRequired,
        })
    }

    pub fn device_identity(&self) -> &VulkanDeviceIdentity {
        &self.device_identity
    }

    pub fn implementation_digest(&self) -> &str {
        &self.implementation_digest
    }

    /// Execute HdcBindXor natively and return an evidence-bearing receipt.
    ///
    /// The backend verifies the receipt against both the plan and the produced
    /// output before returning success.
    pub fn execute(
        &self,
        plan: &OperationPlan,
        inputs: &[BinaryHypervector],
    ) -> Result<(BinaryHypervector, ExecutionReceipt), VulkanError> {
        plan.validate().map_err(VulkanError::Plan)?;

        let GpuOperation::HdcBindXor { dimensions } = plan.operation;

        if dimensions == 0 {
            return Err(VulkanError::ZeroDimensions);
        }
        if inputs.len() != 2 {
            return Err(VulkanError::InputCount {
                expected: 2,
                actual: inputs.len(),
            });
        }
        if inputs[0].dimensions != dimensions || inputs[1].dimensions != dimensions {
            return Err(VulkanError::DimensionMismatch {
                expected: dimensions,
                left: inputs[0].dimensions,
                right: inputs[1].dimensions,
            });
        }

        let logical_bytes = packed_bytes(dimensions);
        let physical_bytes = physical_storage_bytes(logical_bytes)?;
        if physical_bytes > self.max_storage_buffer_range {
            return Err(VulkanError::StorageBufferRangeExceeded {
                bytes: physical_bytes,
                max: self.max_storage_buffer_range,
            });
        }

        let lhs = GpuBuffer::new(&self.device, &self.memory_properties, physical_bytes)?;
        let rhs = GpuBuffer::new(&self.device, &self.memory_properties, physical_bytes)?;
        let output = GpuBuffer::new(&self.device, &self.memory_properties, physical_bytes)?;

        lhs.write_bytes(&self.device, inputs[0].as_bytes())?;
        rhs.write_bytes(&self.device, inputs[1].as_bytes())?;
        output.write_bytes(&self.device, &[])?;

        let word_count = (physical_bytes / 4) as u32;
        let group_count = word_count.saturating_add(WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        if group_count.max(1) > self.max_compute_workgroup_count_x {
            return Err(VulkanError::DispatchGroupCountExceeded {
                groups: group_count.max(1),
                max: self.max_compute_workgroup_count_x,
            });
        }

        let descriptor_set = allocate_descriptor_set(
            &self.device,
            self.descriptor_pool,
            self.descriptor_set_layout,
            [&lhs, &rhs, &output],
            physical_bytes,
        )?;

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = match unsafe {
            self.device.allocate_command_buffers(&command_buffer_info)
        } {
            Ok(mut buffers) => buffers
                .pop()
                .expect("one command buffer was requested"),
            Err(error) => {
                unsafe {
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                }
                return Err(VulkanError::Vk(error));
            }
        };

        let submit = self.record_and_submit(command_buffer, descriptor_set, group_count.max(1));

        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }

        if let Err(error) = submit {
            // A successful queue submission is fully waited on by
            // record_and_submit. On an error path no descriptor set is kept
            // alive by this method.
            unsafe {
                self.device
                    .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                    .ok();
            }
            return Err(error);
        }

        unsafe {
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                .map_err(VulkanError::Vk)?;
        }

        let bytes = output.read_bytes(&self.device, logical_bytes as usize)?;
        let result = BinaryHypervector::from_bytes(dimensions, bytes).map_err(VulkanError::Vector)?;

        let receipt = ExecutionReceipt {
            version: crate::RECEIPT_VERSION,
            backend: BackendKind::Vulkan,
            accelerated: self.claims_acceleration,
            operation: plan.operation,
            plan_digest: plan.digest_hex(),
            kernel_id: plan.operation.kernel_id().to_owned(),
            semantic_kernel_digest: plan.operation.semantic_kernel_digest(),
            implementation_digest: Some(self.implementation_digest.clone()),
            device_identity: Some(self.device_identity.receipt_device_identity()),
            driver_identity: Some(self.device_identity.receipt_driver_identity()),
            resource_limits: plan.limits,
            input_digest: digest_hypervectors(inputs),
            output_digest: digest_hypervector(&result),
            determinism: plan.determinism,
        };

        receipt.verify_plan(plan).map_err(VulkanError::Receipt)?;
        receipt.verify_output(&result).map_err(VulkanError::Receipt)?;

        Ok((result, receipt))
    }

    fn record_and_submit(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        group_count: u32,
    ) -> Result<(), VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(VulkanError::Vk)?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device
                .cmd_dispatch(command_buffer, group_count, 1, 1);

            self.device
                .end_command_buffer(command_buffer)
                .map_err(VulkanError::Vk)?;
        }

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(VulkanError::Vk)?
        };

        let submit = unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
        };

        if let Err(error) = submit {
            unsafe { self.device.destroy_fence(fence, None) };
            return Err(VulkanError::Vk(error));
        }

        let wait = unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) };

        if let Err(error) = wait {
            // A fence wait failure after a successful submission must not let
            // descriptor/buffer resources be released while work could still
            // be in flight.
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_fence(fence, None);
            }
            return Err(VulkanError::Vk(error));
        }

        unsafe { self.device.destroy_fence(fence, None) };
        Ok(())
    }
}

impl Drop for VulkanExecutor {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct GpuBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocation_size: vk::DeviceSize,
    coherent: bool,
}

impl GpuBuffer {
    fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
    ) -> Result<Self, VulkanError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(VulkanError::Vk)?
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let (memory_type_index, coherent) =
            match find_memory_type(requirements.memory_type_bits, memory_properties) {
                Some(value) => value,
                None => {
                    unsafe { device.destroy_buffer(buffer, None) };
                    return Err(VulkanError::NoHostVisibleMemory);
                }
            };

        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);

        let memory = match unsafe { device.allocate_memory(&allocation_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(VulkanError::Vk(error));
            }
        };

        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(VulkanError::Vk(error));
        }

        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            allocation_size: requirements.size,
            coherent,
        })
    }

    fn write_bytes(&self, device: &Device, bytes: &[u8]) -> Result<(), VulkanError> {
        if bytes.len() as u64 > self.allocation_size {
            return Err(VulkanError::AllocationSizeOverflow);
        }

        let mapped = unsafe {
            device
                .map_memory(
                    self.memory,
                    0,
                    self.allocation_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(VulkanError::Vk)?
        };

        if !self.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
                if bytes.len() < self.allocation_size as usize {
                    ptr::write_bytes(
                        mapped.cast::<u8>().add(bytes.len()),
                        0,
                        self.allocation_size as usize - bytes.len(),
                    );
                }
                if let Err(error) =
                    device.flush_mapped_memory_ranges(std::slice::from_ref(&range))
                {
                    device.unmap_memory(self.memory);
                    return Err(VulkanError::Vk(error));
                }
                device.unmap_memory(self.memory);
            }
        } else {
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
                if bytes.len() < self.allocation_size as usize {
                    ptr::write_bytes(
                        mapped.cast::<u8>().add(bytes.len()),
                        0,
                        self.allocation_size as usize - bytes.len(),
                    );
                }
                device.unmap_memory(self.memory);
            }
        }

        Ok(())
    }

    fn read_bytes(&self, device: &Device, len: usize) -> Result<Vec<u8>, VulkanError> {
        if len as u64 > self.allocation_size {
            return Err(VulkanError::AllocationSizeOverflow);
        }

        let mapped = unsafe {
            device
                .map_memory(
                    self.memory,
                    0,
                    self.allocation_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(VulkanError::Vk)?
        };

        if !self.coherent {
            let range = vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            unsafe {
                if let Err(error) =
                    device.invalidate_mapped_memory_ranges(std::slice::from_ref(&range))
                {
                    device.unmap_memory(self.memory);
                    return Err(VulkanError::Vk(error));
                }
            }
        }

        let mut output = vec![0_u8; len];

        unsafe {
            ptr::copy_nonoverlapping(mapped.cast::<u8>(), output.as_mut_ptr(), len);
            device.unmap_memory(self.memory);
        }

        Ok(output)
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn create_shader_module(device: &Device, spirv: &[u32]) -> Result<vk::ShaderModule, VulkanError> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(VulkanError::Vk)
    }
}

fn create_descriptor_set_layout(
    device: &Device,
) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [
        storage_binding(0),
        storage_binding(1),
        storage_binding(2),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .map_err(VulkanError::Vk)
    }
}

fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn allocate_descriptor_set(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    buffers: [&GpuBuffer; 3],
    physical_bytes: u64,
) -> Result<vk::DescriptorSet, VulkanError> {
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));

    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(&allocate_info)
            .map_err(VulkanError::Vk)?
            .into_iter()
            .next()
            .expect("one descriptor set was requested")
    };

    let infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[0].buffer)
            .offset(0)
            .range(physical_bytes),
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[1].buffer)
            .offset(0)
            .range(physical_bytes),
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[2].buffer)
            .offset(0)
            .range(physical_bytes),
    ];

    let writes = [
        storage_write(descriptor_set, 0, &infos[0]),
        storage_write(descriptor_set, 1, &infos[1]),
        storage_write(descriptor_set, 2, &infos[2]),
    ];

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    Ok(descriptor_set)
}

fn storage_write<'a>(
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}

fn compile_spirv() -> Result<Vec<u32>, VulkanError> {
    let module =
        wgsl::parse_str(VULKAN_HDC_XOR_WGSL).map_err(|error| VulkanError::ShaderParse(error.to_string()))?;

    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|error| VulkanError::ShaderValidation(error.to_string()))?;

    let options = spv::Options::default();
    let pipeline_options = spv::PipelineOptions {
        entry_point: "main".to_owned(),
        shader_stage: naga::ShaderStage::Compute,
    };

    spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .map_err(|error| VulkanError::ShaderSpirv(error.to_string()))
}

fn implementation_digest(spirv: &[u32]) -> String {
    let mut hasher = blake3::Hasher::new();

    hasher.update(b"symthaea.gpu-fabric.vulkan-implementation.v1\0");
    hasher.update(VULKAN_IMPLEMENTATION_ABI);
    hasher.update(VULKAN_HDC_XOR_WGSL.as_bytes());

    for word in &spirv {
        hasher.update(&word.to_le_bytes());
    }

    hasher.finalize().to_hex().to_string()
}

fn packed_bytes(dimensions: u32) -> u64 {
    (u64::from(dimensions).saturating_add(7)) / 8
}

fn physical_storage_bytes(logical_bytes: u64) -> Result<u64, VulkanError> {
    logical_bytes
        .checked_add(3)
        .map(|bytes| bytes / 4 * 4)
        .filter(|bytes| *bytes >= 4)
        .ok_or(VulkanError::AllocationSizeOverflow)
}

fn find_memory_type(
    type_bits: u32,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Option<(u32, bool)> {
    let count = memory_properties.memory_type_count as usize;

    for require_coherent in [true, false] {
        for index in 0..count {
            let memory_type = memory_properties.memory_types[index];
            let flags = memory_type.property_flags;
            let host_visible = flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let host_coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);

            if (type_bits & (1_u32 << index)) != 0
                && host_visible
                && (!require_coherent || host_coherent)
            {
                return Some((index as u32, host_coherent));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_is_real_spirv_and_digest_is_nonempty() {
        let words = compile_spirv().expect("WGSL kernel should compile");
        assert!(words.len() >= 5);
        assert_eq!(words[0], 0x0723_0203);
        assert!(!implementation_digest(&words).is_empty());
    }

    #[test]
    fn physical_storage_rounds_to_u32_words() {
        assert_eq!(physical_storage_bytes(1).unwrap(), 4);
        assert_eq!(physical_storage_bytes(4).unwrap(), 4);
        assert_eq!(physical_storage_bytes(5).unwrap(), 8);
    }

    #[test]
    fn memory_selection_prefers_coherent_host_visible_memory() {
        let mut properties = vk::PhysicalDeviceMemoryProperties::default();
        properties.memory_type_count = 2;
        properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        properties.memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        assert_eq!(
            find_memory_type(0b11, &properties),
            Some((1, true))
        );
    }

    #[test]
    fn empty_device_names_are_rejected() {
        let mut properties = vk::PhysicalDeviceProperties::default();
        properties.device_name = [0; vk::MAX_PHYSICAL_DEVICE_NAME_SIZE];

        assert!(matches!(
            VulkanDeviceIdentity::from_properties(&properties),
            Err(VulkanError::MissingDeviceName)
        ));
    }

    /// Real-device qualification test for a Vulkan-capable lane.
    #[test]
    #[ignore = "requires a Vulkan-capable qualification runner"]
    fn native_vulkan_matches_cpu_oracle() {
        let executor = VulkanExecutor::new().expect("Vulkan executor should initialize");

        for dimensions in [1_u32, 7, 8, 9, 31, 32, 33, 127, 128, 129, 16384] {
            let plan = plan_for(dimensions);
            let left = canonical_pattern(dimensions, 0x31, 0x0b);
            let right = canonical_pattern(dimensions, 0x8d, 0x53);

            let (cpu_output, cpu_receipt) =
                crate::CpuReferenceExecutor::execute(&plan, &[left.clone(), right.clone()])
                    .unwrap();
            cpu_receipt.verify_plan(&plan).unwrap();
            cpu_receipt.verify_output(&cpu_output).unwrap();

            let (vulkan_output, vulkan_receipt) = executor
                .execute(&plan, &[left.clone(), right.clone()])
                .expect("Vulkan execution should succeed");

            let (second_output, second_receipt) = executor
                .execute(&plan, &[left, right])
                .expect("Vulkan executor must be reusable");

            assert_eq!(vulkan_output, cpu_output);
            assert_eq!(second_output, cpu_output);
            assert_eq!(vulkan_receipt.output_digest, second_receipt.output_digest);
            assert_eq!(vulkan_receipt.input_digest, second_receipt.input_digest);
            assert_eq!(vulkan_receipt.operation, cpu_receipt.operation);
            assert_eq!(vulkan_receipt.plan_digest, cpu_receipt.plan_digest);
            assert_eq!(
                vulkan_receipt.semantic_kernel_digest,
                cpu_receipt.semantic_kernel_digest
            );
            assert_ne!(
                vulkan_receipt.implementation_digest,
                cpu_receipt.implementation_digest
            );
            assert!(!vulkan_receipt.accelerated);
            vulkan_receipt.verify_plan(&plan_for(dimensions)).unwrap();
            vulkan_receipt.verify_output(&vulkan_output).unwrap();
        }
    }

    fn plan_for(dimensions: u32) -> OperationPlan {
        OperationPlan::new(GpuOperation::HdcBindXor { dimensions })
    }

    fn canonical_pattern(dimensions: u32, multiplier: u8, offset: u8) -> BinaryHypervector {
        let byte_len = packed_bytes(dimensions) as usize;
        let mut bytes = (0..byte_len)
            .map(|index| {
                (index as u8)
                    .wrapping_mul(multiplier)
                    .wrapping_add(offset)
            })
            .collect::<Vec<_>>();

        if dimensions % 8 != 0 {
            let mask = 0xff_u8 >> (8 - (dimensions % 8));
            if let Some(last) = bytes.last_mut() {
                *last &= mask;
            }
        }

        BinaryHypervector::from_bytes(dimensions, bytes)
            .expect("canonical test pattern should satisfy vector invariants")
    }
}

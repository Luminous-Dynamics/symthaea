//! Backend-neutral semantic GPU execution contracts for Symthaea.
//!
//! The crate deliberately starts with a CPU reference executor. Native Vulkan
//! and browser WebGPU implementations can implement the same operation contract
//! later without changing the semantic identity of an operation.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PLAN_VERSION: u16 = 1;
pub const RECEIPT_VERSION: u16 = 1;
pub const HDC_BIND_XOR_KERNEL_ID: &str = "symthaea.hdc.bind_xor.v1";
pub const HDC_BIND_XOR_KERNEL_DIGEST: &str =
    "b12c0d7f6a6b7ed2a1c5d5dcb9f8e0b9f1b8b6e1b1d2f0b9e7f0d9c4e4e6f5a1";

/// Stable identifier for the kind of execution backend that actually ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    CpuReference,
    WebGpu,
    Vulkan,
}

impl BackendKind {
    pub const fn accelerated(self) -> bool {
        matches!(self, Self::WebGpu | Self::Vulkan)
    }
}

/// How strongly the executor promises reproducibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeterminismMode {
    Strict,
    BestEffort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuOperation {
    /// Bitwise HDC binding. For binary hypervectors this is XOR.
    HdcBindXor { dimensions: u32 },
}

impl GpuOperation {
    pub const fn kernel_id(self) -> &'static str {
        match self {
            Self::HdcBindXor { .. } => HDC_BIND_XOR_KERNEL_ID,
        }
    }

    pub const fn kernel_digest(self) -> &'static str {
        match self {
            Self::HdcBindXor { .. } => HDC_BIND_XOR_KERNEL_DIGEST,
        }
    }

    pub const fn input_count(self) -> usize {
        match self {
            Self::HdcBindXor { .. } => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_input_bytes: u64,
    pub max_output_bytes: u64,
    pub max_dispatch_invocations: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 64 * 1024 * 1024,
            max_output_bytes: 32 * 1024 * 1024,
            max_dispatch_invocations: 1 << 24,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferSpec {
    pub bytes: u64,
    pub alignment: u32,
}

impl BufferSpec {
    pub const fn new(bytes: u64) -> Self {
        Self { bytes, alignment: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationPlan {
    pub version: u16,
    pub operation: GpuOperation,
    pub inputs: Vec<BufferSpec>,
    pub output: BufferSpec,
    pub limits: ResourceLimits,
    pub determinism: DeterminismMode,
}

impl OperationPlan {
    pub fn new(operation: GpuOperation) -> Self {
        let bytes = match operation {
            GpuOperation::HdcBindXor { dimensions } => packed_bytes(dimensions),
        };
        Self {
            version: PLAN_VERSION,
            operation,
            inputs: (0..operation.input_count())
                .map(|_| BufferSpec::new(bytes))
                .collect(),
            output: BufferSpec::new(bytes),
            limits: ResourceLimits::default(),
            determinism: DeterminismMode::Strict,
        }
    }

    pub fn validate(&self) -> Result<(), PlanError> {
        if self.version != PLAN_VERSION {
            return Err(PlanError::UnsupportedVersion(self.version));
        }

        if self.inputs.len() != self.operation.input_count() {
            return Err(PlanError::InputCount {
                expected: self.operation.input_count(),
                actual: self.inputs.len(),
            });
        }

        if self.operation.kernel_digest() != HDC_BIND_XOR_KERNEL_DIGEST {
            return Err(PlanError::KernelDigestMismatch {
                expected: HDC_BIND_XOR_KERNEL_DIGEST,
                actual: self.operation.kernel_digest(),
            });
        }

        let required_bytes = match self.operation {
            GpuOperation::HdcBindXor { dimensions } => packed_bytes(dimensions),
        };

        for (index, input) in self.inputs.iter().enumerate() {
            if input.bytes != required_bytes {
                return Err(PlanError::ShapeMismatch {
                    field: format!("input[{index}]"),
                    expected: required_bytes,
                    actual: input.bytes,
                });
            }
        }

        if self.output.bytes != required_bytes {
            return Err(PlanError::ShapeMismatch {
                field: "output".to_owned(),
                expected: required_bytes,
                actual: self.output.bytes,
            });
        }

        let total_inputs = self.inputs.iter().map(|x| x.bytes).sum::<u64>();
        if total_inputs > self.limits.max_input_bytes {
            return Err(PlanError::InputBudgetExceeded {
                bytes: total_inputs,
                max: self.limits.max_input_bytes,
            });
        }

        if self.output.bytes > self.limits.max_output_bytes {
            return Err(PlanError::OutputBudgetExceeded {
                bytes: self.output.bytes,
                max: self.limits.max_output_bytes,
            });
        }

        let dispatch = match self.operation {
            GpuOperation::HdcBindXor { dimensions } => u64::from(dimensions),
        };
        if dispatch > self.limits.max_dispatch_invocations {
            return Err(PlanError::DispatchBudgetExceeded {
                invocations: dispatch,
                max: self.limits.max_dispatch_invocations,
            });
        }

        Ok(())
    }

    /// Digest of the exact semantic plan. Backend selection is deliberately not
    /// part of this digest: the same plan must identify the same operation
    /// regardless of whether CPU, WebGPU, or Vulkan executes it.
    pub fn digest(&self) -> [u8; 32] {
        stable_digest(self)
    }

    pub fn digest_hex(&self) -> String {
        hex_digest(self.digest())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryHypervector {
    pub dimensions: u32,
    bytes: Vec<u8>,
}

impl BinaryHypervector {
    pub fn from_bytes(dimensions: u32, bytes: Vec<u8>) -> Result<Self, VectorError> {
        let expected = packed_bytes(dimensions) as usize;
        if bytes.len() != expected {
            return Err(VectorError::ByteLength {
                dimensions,
                expected,
                actual: bytes.len(),
            });
        }

        let unused_bits = (8 - (dimensions % 8)) % 8;
        if unused_bits != 0 {
            let mask = 0xff_u8 >> unused_bits;
            if bytes.last().copied().unwrap_or_default() & !mask != 0 {
                return Err(VectorError::NonCanonicalTailBits);
            }
        }

        Ok(Self { dimensions, bytes })
    }

    pub fn zeros(dimensions: u32) -> Self {
        Self {
            dimensions,
            bytes: vec![0; packed_bytes(dimensions) as usize],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub struct CpuReferenceExecutor;

impl CpuReferenceExecutor {
    pub fn execute(
        plan: &OperationPlan,
        inputs: &[BinaryHypervector],
    ) -> Result<(BinaryHypervector, ExecutionReceipt), ExecutionError> {
        plan.validate().map_err(ExecutionError::Plan)?;

        match plan.operation {
            GpuOperation::HdcBindXor { dimensions } => {
                if inputs.len() != 2 {
                    return Err(ExecutionError::InputCount {
                        expected: 2,
                        actual: inputs.len(),
                    });
                }

                if inputs[0].dimensions != dimensions || inputs[1].dimensions != dimensions {
                    return Err(ExecutionError::DimensionMismatch {
                        expected: dimensions,
                        left: inputs[0].dimensions,
                        right: inputs[1].dimensions,
                    });
                }

                let bytes = inputs[0]
                    .as_bytes()
                    .iter()
                    .zip(inputs[1].as_bytes())
                    .map(|(a, b)| a ^ b)
                    .collect::<Vec<_>>();

                let output = BinaryHypervector::from_bytes(dimensions, bytes)
                    .map_err(ExecutionError::Vector)?;

                let input_digest = digest_bytes(
                    &inputs
                        .iter()
                        .flat_map(|vector| vector.as_bytes().iter().copied())
                        .collect::<Vec<_>>(),
                );

                let receipt = ExecutionReceipt {
                    version: RECEIPT_VERSION,
                    backend: BackendKind::CpuReference,
                    accelerated: false,
                    operation: plan.operation,
                    plan_digest: plan.digest_hex(),
                    kernel_id: plan.operation.kernel_id().to_owned(),
                    kernel_digest: plan.operation.kernel_digest().to_owned(),
                    input_digest,
                    output_digest: digest_bytes(output.as_bytes()),
                    determinism: plan.determinism,
                };

                Ok((output, receipt))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub version: u16,
    pub backend: BackendKind,
    pub accelerated: bool,
    pub operation: GpuOperation,
    pub plan_digest: String,
    pub kernel_id: String,
    pub kernel_digest: String,
    pub input_digest: String,
    pub output_digest: String,
    pub determinism: DeterminismMode,
}

impl ExecutionReceipt {
    pub fn verify_plan(&self, plan: &OperationPlan) -> Result<(), ReceiptError> {
        if self.version != RECEIPT_VERSION {
            return Err(ReceiptError::UnsupportedVersion(self.version));
        }
        plan.validate().map_err(ReceiptError::InvalidPlan)?;
        if self.operation != plan.operation {
            return Err(ReceiptError::OperationMismatch);
        }
        if self.plan_digest != plan.digest_hex() {
            return Err(ReceiptError::PlanDigestMismatch);
        }
        if self.kernel_id != plan.operation.kernel_id()
            || self.kernel_digest != plan.operation.kernel_digest()
        {
            return Err(ReceiptError::KernelMismatch);
        }
        if self.accelerated != self.backend.accelerated() {
            return Err(ReceiptError::AccelerationClaimMismatch);
        }
        Ok(())
    }

    pub fn verify_output(&self, output: &BinaryHypervector) -> Result<(), ReceiptError> {
        let digest = digest_bytes(output.as_bytes());
        if digest != self.output_digest {
            return Err(ReceiptError::OutputDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PlanError {
    #[error("unsupported plan version {0}")]
    UnsupportedVersion(u16),
    #[error("expected {expected} inputs, got {actual}")]
    InputCount { expected: usize, actual: usize },
    #[error("kernel digest mismatch: expected {expected}, got {actual}")]
    KernelDigestMismatch { expected: &'static str, actual: &'static str },
    #[error("{field} requires {expected} bytes, got {actual}")]
    ShapeMismatch { field: String, expected: u64, actual: u64 },
    #[error("input budget exceeded: {bytes} > {max}")]
    InputBudgetExceeded { bytes: u64, max: u64 },
    #[error("output budget exceeded: {bytes} > {max}")]
    OutputBudgetExceeded { bytes: u64, max: u64 },
    #[error("dispatch budget exceeded: {invocations} > {max}")]
    DispatchBudgetExceeded { invocations: u64, max: u64 },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VectorError {
    #[error("dimension {dimensions} requires {expected} bytes, got {actual}")]
    ByteLength {
        dimensions: u32,
        expected: usize,
        actual: usize,
    },
    #[error("unused tail bits must be zero")]
    NonCanonicalTailBits,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ExecutionError {
    #[error("invalid plan: {0}")]
    Plan(PlanError),
    #[error("expected {expected} inputs, got {actual}")]
    InputCount { expected: usize, actual: usize },
    #[error("input dimensions mismatch: expected {expected}, left={left}, right={right}")]
    DimensionMismatch { expected: u32, left: u32, right: u32 },
    #[error("invalid input vector: {0}")]
    Vector(VectorError),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    #[error("unsupported receipt version {0}")]
    UnsupportedVersion(u16),
    #[error("invalid plan: {0}")]
    InvalidPlan(PlanError),
    #[error("operation mismatch")]
    OperationMismatch,
    #[error("plan digest mismatch")]
    PlanDigestMismatch,
    #[error("kernel identity mismatch")]
    KernelMismatch,
    #[error("acceleration claim does not match backend")]
    AccelerationClaimMismatch,
    #[error("output digest mismatch")]
    OutputDigestMismatch,
}

fn packed_bytes(dimensions: u32) -> u64 {
    (u64::from(dimensions).saturating_add(7)) / 8
}

fn stable_digest<T: Serialize>(value: &T) -> [u8; 32] {
    let bytes = serde_json::to_vec(value).expect("semantic GPU values are serializable");
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-gpu-fabric ");
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn digest_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn hex_digest(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_execution_is_deterministic_and_strict() {
        let plan = OperationPlan::new(GpuOperation::HdcBindXor { dimensions: 13 });
        let left = BinaryHypervector::from_bytes(13, vec![0b1010_0101, 0b0001_0000]).unwrap();
        let right = BinaryHypervector::from_bytes(13, vec![0b0101_0011, 0b0000_0011]).unwrap();

        let (output_a, receipt_a) =
            CpuReferenceExecutor::execute(&plan, &[left.clone(), right.clone()]).unwrap();
        let (output_b, receipt_b) =
            CpuReferenceExecutor::execute(&plan, &[left, right]).unwrap();

        assert_eq!(output_a, output_b);
        assert_eq!(receipt_a, receipt_b);
        assert_eq!(output_a.as_bytes(), &[0b1111_0110, 0b0001_0011]);
        assert!(!receipt_a.accelerated);
        receipt_a.verify_plan(&plan).unwrap();
        receipt_a.verify_output(&output_a).unwrap();
    }

    #[test]
    fn plan_digest_is_backend_neutral() {
        let plan = OperationPlan::new(GpuOperation::HdcBindXor { dimensions: 16 });
        assert_eq!(plan.digest(), plan.digest());
        assert_ne!(plan.digest_hex(), "");
    }

    #[test]
    fn oversized_output_is_rejected_before_execution() {
        let mut plan = OperationPlan::new(GpuOperation::HdcBindXor { dimensions: 64 });
        plan.limits.max_output_bytes = 1;
        let error = plan.validate().unwrap_err();
        assert!(matches!(error, PlanError::OutputBudgetExceeded { .. }));
    }

    #[test]
    fn noncanonical_tail_bits_are_rejected() {
        let error = BinaryHypervector::from_bytes(13, vec![0, 0b1110_0000]).unwrap_err();
        assert_eq!(error, VectorError::NonCanonicalTailBits);
    }

    #[test]
    fn tampered_output_invalidates_receipt() {
        let plan = OperationPlan::new(GpuOperation::HdcBindXor { dimensions: 8 });
        let left = BinaryHypervector::from_bytes(8, vec![0xaa]).unwrap();
        let right = BinaryHypervector::from_bytes(8, vec![0x55]).unwrap();
        let (output, receipt) = CpuReferenceExecutor::execute(&plan, &[left, right]).unwrap();

        let tampered =
            BinaryHypervector::from_bytes(8, vec![output.as_bytes()[0] ^ 0x01]).unwrap();
        assert!(matches!(
            receipt.verify_output(&tampered),
            Err(ReceiptError::OutputDigestMismatch)
        ));
    }

    #[test]
    fn backend_claims_are_fail_closed() {
        let mut receipt = ExecutionReceipt {
            version: RECEIPT_VERSION,
            backend: BackendKind::CpuReference,
            accelerated: true,
            operation: GpuOperation::HdcBindXor { dimensions: 8 },
            plan_digest: String::new(),
            kernel_id: HDC_BIND_XOR_KERNEL_ID.to_owned(),
            kernel_digest: HDC_BIND_XOR_KERNEL_DIGEST.to_owned(),
            input_digest: "x".to_owned(),
            output_digest: "x".to_owned(),
            determinism: DeterminismMode::Strict,
        };
        let plan = OperationPlan::new(GpuOperation::HdcBindXor { dimensions: 8 });
        assert!(matches!(
            receipt.verify_plan(&plan),
            Err(ReceiptError::AccelerationClaimMismatch)
        ));
        receipt.accelerated = false;
        receipt.plan_digest = plan.digest_hex();
        receipt.verify_plan(&plan).unwrap();
    }
}

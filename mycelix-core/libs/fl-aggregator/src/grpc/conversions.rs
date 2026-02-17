//! Conversion functions between gRPC types and internal types.
//!
//! This module provides bidirectional conversion between:
//! - Proto Payload types <-> UnifiedPayload
//! - AggregatorStatus <-> GetStatusResponse
//! - Internal error types <-> gRPC Status

use crate::aggregator::AggregatorStatus;
use crate::payload::{
    BinaryHypervector, DenseGradient, Hypervector, HypervectorMetadata,
    PayloadType, QuantizedGradient, SparseGradient, UnifiedPayload,
};

use super::{
    EncodingMetadata, GetStatusResponse, GrpcBinaryHypervector,
    GrpcDenseGradient, GrpcHyperEncodedGradient, GrpcPayload, GrpcPayloadData,
    GrpcPayloadType, GrpcQuantizedGradient, GrpcRoundState, GrpcSparseGradient,
};

// ============================================================================
// Payload Conversions
// ============================================================================

/// Convert gRPC Payload to UnifiedPayload.
///
/// # Errors
///
/// Returns an error string if the payload is empty or invalid.
pub fn grpc_payload_to_unified(grpc: &GrpcPayload) -> Result<UnifiedPayload, String> {
    match &grpc.data {
        Some(GrpcPayloadData::Dense(d)) => {
            Ok(UnifiedPayload::Dense(grpc_dense_to_internal(d)?))
        }
        Some(GrpcPayloadData::Hyper(h)) => {
            Ok(UnifiedPayload::Hyper(grpc_hyper_to_internal(h)?))
        }
        Some(GrpcPayloadData::Binary(b)) => {
            Ok(UnifiedPayload::Binary(grpc_binary_to_internal(b)?))
        }
        Some(GrpcPayloadData::Sparse(s)) => {
            Ok(UnifiedPayload::Sparse(grpc_sparse_to_internal(s)?))
        }
        Some(GrpcPayloadData::Quantized(q)) => {
            Ok(UnifiedPayload::Quantized(grpc_quantized_to_internal(q)?))
        }
        None => Err("Payload is empty".to_string()),
    }
}

/// Convert UnifiedPayload to gRPC Payload.
pub fn unified_payload_to_grpc(unified: &UnifiedPayload) -> GrpcPayload {
    match unified {
        UnifiedPayload::Dense(d) => GrpcPayload {
            data: Some(GrpcPayloadData::Dense(internal_dense_to_grpc(d))),
        },
        UnifiedPayload::Hyper(h) => GrpcPayload {
            data: Some(GrpcPayloadData::Hyper(internal_hyper_to_grpc(h))),
        },
        UnifiedPayload::Binary(b) => GrpcPayload {
            data: Some(GrpcPayloadData::Binary(internal_binary_to_grpc(b))),
        },
        UnifiedPayload::Sparse(s) => GrpcPayload {
            data: Some(GrpcPayloadData::Sparse(internal_sparse_to_grpc(s))),
        },
        UnifiedPayload::Quantized(q) => GrpcPayload {
            data: Some(GrpcPayloadData::Quantized(internal_quantized_to_grpc(q))),
        },
    }
}

// ============================================================================
// Dense Gradient Conversions
// ============================================================================

/// Convert gRPC DenseGradient to internal DenseGradient.
fn grpc_dense_to_internal(grpc: &GrpcDenseGradient) -> Result<DenseGradient, String> {
    if grpc.values.is_empty() {
        return Err("Dense gradient values are empty".to_string());
    }

    // Validate values are finite
    for (i, &val) in grpc.values.iter().enumerate() {
        if !val.is_finite() {
            return Err(format!("Non-finite value at index {}: {}", i, val));
        }
    }

    Ok(DenseGradient::from_vec(grpc.values.clone()))
}

/// Convert internal DenseGradient to gRPC DenseGradient.
fn internal_dense_to_grpc(internal: &DenseGradient) -> GrpcDenseGradient {
    GrpcDenseGradient {
        values: internal.to_vec(),
        dimension: internal.values.len() as u64,
    }
}

// ============================================================================
// Hypervector Conversions
// ============================================================================

/// Convert gRPC HyperEncodedGradient to internal Hypervector.
fn grpc_hyper_to_internal(grpc: &GrpcHyperEncodedGradient) -> Result<Hypervector, String> {
    if grpc.hypervector.is_empty() {
        return Err("Hypervector is empty".to_string());
    }

    // Convert bytes to i8 components
    let components: Vec<i8> = grpc
        .hypervector
        .iter()
        .map(|&b| b as i8)
        .collect();

    let metadata = match &grpc.encoding {
        Some(enc) => HypervectorMetadata {
            encoder_version: enc.encoder_version.clone(),
            original_dimension: enc.original_size as usize,
            compression_ratio: enc.compression_ratio,
            projection_seed: Some(enc.projection_seed),
            use_causal: enc.use_causal,
            use_temporal: enc.use_temporal,
        },
        None => HypervectorMetadata::default(),
    };

    Ok(Hypervector::with_metadata(components, metadata))
}

/// Convert internal Hypervector to gRPC HyperEncodedGradient.
fn internal_hyper_to_grpc(internal: &Hypervector) -> GrpcHyperEncodedGradient {
    // Convert i8 components to bytes
    let hypervector: Vec<u8> = internal
        .components
        .iter()
        .map(|&c| c as u8)
        .collect();

    let encoding = Some(EncodingMetadata {
        encoder_version: internal.metadata.encoder_version.clone(),
        quantize_bits: 8,
        original_size: internal.metadata.original_dimension as u64,
        compression_ratio: internal.metadata.compression_ratio,
        projection_seed: internal.metadata.projection_seed.unwrap_or(0),
        use_causal: internal.metadata.use_causal,
        use_temporal: internal.metadata.use_temporal,
    });

    GrpcHyperEncodedGradient {
        hypervector,
        dimension: internal.components.len() as u32,
        encoding,
        phi: None, // TODO: Support phi metrics
    }
}

// ============================================================================
// Binary Hypervector Conversions
// ============================================================================

/// Convert gRPC BinaryHypervector to internal BinaryHypervector.
fn grpc_binary_to_internal(grpc: &GrpcBinaryHypervector) -> Result<BinaryHypervector, String> {
    if grpc.data.is_empty() {
        return Err("Binary hypervector data is empty".to_string());
    }

    if grpc.dimension == 0 {
        return Err("Binary hypervector dimension is zero".to_string());
    }

    // Validate byte count matches dimension
    let expected_bytes = (grpc.dimension as usize + 7) / 8;
    if grpc.data.len() != expected_bytes {
        return Err(format!(
            "Binary hypervector byte count mismatch: expected {}, got {}",
            expected_bytes,
            grpc.data.len()
        ));
    }

    Ok(BinaryHypervector::new(
        grpc.data.clone(),
        grpc.dimension as usize,
    ))
}

/// Convert internal BinaryHypervector to gRPC BinaryHypervector.
fn internal_binary_to_grpc(internal: &BinaryHypervector) -> GrpcBinaryHypervector {
    GrpcBinaryHypervector {
        data: internal.data.clone(),
        dimension: internal.bit_dimension as u32,
    }
}

// ============================================================================
// Sparse Gradient Conversions
// ============================================================================

/// Convert gRPC SparseGradient to internal SparseGradient.
fn grpc_sparse_to_internal(grpc: &GrpcSparseGradient) -> Result<SparseGradient, String> {
    if grpc.indices.len() != grpc.values.len() {
        return Err(format!(
            "Sparse gradient indices/values length mismatch: {} vs {}",
            grpc.indices.len(),
            grpc.values.len()
        ));
    }

    if grpc.full_dimension == 0 {
        return Err("Sparse gradient full dimension is zero".to_string());
    }

    // Validate indices are within bounds
    for (i, &idx) in grpc.indices.iter().enumerate() {
        if idx as u64 >= grpc.full_dimension {
            return Err(format!(
                "Sparse gradient index {} out of bounds: {} >= {}",
                i, idx, grpc.full_dimension
            ));
        }
    }

    // Validate values are finite
    for (i, &val) in grpc.values.iter().enumerate() {
        if !val.is_finite() {
            return Err(format!("Non-finite value at index {}: {}", i, val));
        }
    }

    Ok(SparseGradient::new(
        grpc.indices.clone(),
        grpc.values.clone(),
        grpc.full_dimension as usize,
    ))
}

/// Convert internal SparseGradient to gRPC SparseGradient.
fn internal_sparse_to_grpc(internal: &SparseGradient) -> GrpcSparseGradient {
    GrpcSparseGradient {
        indices: internal.indices.clone(),
        values: internal.values.clone(),
        full_dimension: internal.full_dimension as u64,
        sparsity_ratio: internal.sparsity_ratio(),
    }
}

// ============================================================================
// Quantized Gradient Conversions
// ============================================================================

/// Convert gRPC QuantizedGradient to internal QuantizedGradient.
fn grpc_quantized_to_internal(grpc: &GrpcQuantizedGradient) -> Result<QuantizedGradient, String> {
    if grpc.data.is_empty() {
        return Err("Quantized gradient data is empty".to_string());
    }

    if grpc.dimension == 0 {
        return Err("Quantized gradient dimension is zero".to_string());
    }

    // Validate bits_per_value
    if grpc.bits_per_value != 1
        && grpc.bits_per_value != 2
        && grpc.bits_per_value != 4
        && grpc.bits_per_value != 8
    {
        return Err(format!(
            "Invalid bits_per_value: {} (must be 1, 2, 4, or 8)",
            grpc.bits_per_value
        ));
    }

    // Validate scale and zero_point are finite
    if !grpc.scale.is_finite() {
        return Err("Quantized gradient scale is not finite".to_string());
    }
    if !grpc.zero_point.is_finite() {
        return Err("Quantized gradient zero_point is not finite".to_string());
    }

    Ok(QuantizedGradient {
        data: grpc.data.clone(),
        bits_per_value: grpc.bits_per_value as u8,
        scale: grpc.scale,
        zero_point: grpc.zero_point,
        dimension: grpc.dimension as usize,
    })
}

/// Convert internal QuantizedGradient to gRPC QuantizedGradient.
fn internal_quantized_to_grpc(internal: &QuantizedGradient) -> GrpcQuantizedGradient {
    GrpcQuantizedGradient {
        data: internal.data.clone(),
        bits_per_value: internal.bits_per_value as u32,
        scale: internal.scale,
        zero_point: internal.zero_point,
        dimension: internal.dimension as u64,
    }
}

// ============================================================================
// Payload Type Conversions
// ============================================================================

/// Convert internal PayloadType to gRPC GrpcPayloadType.
pub fn payload_type_to_grpc(pt: PayloadType) -> GrpcPayloadType {
    match pt {
        PayloadType::DenseGradient => GrpcPayloadType::DenseGradient,
        PayloadType::HyperEncoded => GrpcPayloadType::HyperEncoded,
        PayloadType::BinaryHypervector => GrpcPayloadType::BinaryHypervector,
        PayloadType::SparseGradient => GrpcPayloadType::SparseGradient,
        PayloadType::QuantizedGradient => GrpcPayloadType::QuantizedGradient,
    }
}

/// Convert gRPC GrpcPayloadType to internal PayloadType.
pub fn grpc_to_payload_type(grpc: GrpcPayloadType) -> Option<PayloadType> {
    match grpc {
        GrpcPayloadType::Unspecified => None,
        GrpcPayloadType::DenseGradient => Some(PayloadType::DenseGradient),
        GrpcPayloadType::HyperEncoded => Some(PayloadType::HyperEncoded),
        GrpcPayloadType::BinaryHypervector => Some(PayloadType::BinaryHypervector),
        GrpcPayloadType::SparseGradient => Some(PayloadType::SparseGradient),
        GrpcPayloadType::QuantizedGradient => Some(PayloadType::QuantizedGradient),
    }
}

// ============================================================================
// Status Conversions
// ============================================================================

/// Convert AggregatorStatus to GetStatusResponse.
pub fn aggregator_status_to_grpc_response(status: &AggregatorStatus) -> GetStatusResponse {
    let round_state = if status.is_complete {
        GrpcRoundState::Complete
    } else if status.submitted_nodes > 0 {
        GrpcRoundState::Collecting
    } else {
        GrpcRoundState::Pending
    };

    // Get current timestamp
    let round_start_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
        - status.round_elapsed.as_secs() as i64;

    GetStatusResponse {
        current_round: status.round,
        round_state: round_state as i32,
        registered_nodes: status.registered_nodes as u32,
        submissions_this_round: status.submitted_nodes as u32,
        expected_this_round: status.expected_nodes as u32,
        round_start_time,
        aggregation_algorithm: String::new(), // TODO: Get from config
    }
}

/// Convert GetStatusResponse back to partial AggregatorStatus.
///
/// Note: Some fields cannot be recovered (e.g., memory_bytes).
pub fn grpc_response_to_aggregator_status(resp: &GetStatusResponse) -> AggregatorStatus {
    use std::time::Duration;

    let round_elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
        - resp.round_start_time;

    AggregatorStatus {
        round: resp.current_round,
        registered_nodes: resp.registered_nodes as usize,
        submitted_nodes: resp.submissions_this_round as usize,
        expected_nodes: resp.expected_this_round as usize,
        memory_bytes: 0, // Cannot recover
        round_elapsed: Duration::from_secs(round_elapsed.max(0) as u64),
        is_complete: resp.round_state == GrpcRoundState::Complete as i32,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_gradient_roundtrip() {
        let original = DenseGradient::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let grpc = internal_dense_to_grpc(&original);
        let recovered = grpc_dense_to_internal(&grpc).unwrap();

        assert_eq!(original.to_vec(), recovered.to_vec());
        assert_eq!(grpc.dimension, 5);
    }

    #[test]
    fn test_dense_gradient_validation() {
        // Empty values
        let empty = GrpcDenseGradient {
            values: vec![],
            dimension: 0,
        };
        assert!(grpc_dense_to_internal(&empty).is_err());

        // NaN value
        let nan = GrpcDenseGradient {
            values: vec![1.0, f32::NAN, 3.0],
            dimension: 3,
        };
        assert!(grpc_dense_to_internal(&nan).is_err());

        // Infinity value
        let inf = GrpcDenseGradient {
            values: vec![1.0, f32::INFINITY, 3.0],
            dimension: 3,
        };
        assert!(grpc_dense_to_internal(&inf).is_err());
    }

    #[test]
    fn test_hypervector_roundtrip() {
        let original = Hypervector::new(vec![10, 20, -30, 40, -50]);
        let grpc = internal_hyper_to_grpc(&original);
        let recovered = grpc_hyper_to_internal(&grpc).unwrap();

        assert_eq!(original.components, recovered.components);
    }

    #[test]
    fn test_hypervector_with_metadata() {
        let metadata = HypervectorMetadata {
            encoder_version: "hyperfeel-v2".to_string(),
            original_dimension: 1_000_000,
            compression_ratio: 2000.0,
            projection_seed: Some(12345),
            use_causal: true,
            use_temporal: false,
        };

        let original = Hypervector::with_metadata(vec![1, 2, 3], metadata);
        let grpc = internal_hyper_to_grpc(&original);
        let recovered = grpc_hyper_to_internal(&grpc).unwrap();

        assert_eq!(recovered.metadata.encoder_version, "hyperfeel-v2");
        assert_eq!(recovered.metadata.original_dimension, 1_000_000);
        assert!(recovered.metadata.use_causal);
    }

    #[test]
    fn test_binary_hypervector_roundtrip() {
        let original = BinaryHypervector::from_bits(&[
            true, false, true, true, false, false, true, false,
        ]);
        let grpc = internal_binary_to_grpc(&original);
        let recovered = grpc_binary_to_internal(&grpc).unwrap();

        assert_eq!(original.data, recovered.data);
        assert_eq!(original.bit_dimension, recovered.bit_dimension);
    }

    #[test]
    fn test_binary_hypervector_validation() {
        // Empty data
        let empty = GrpcBinaryHypervector {
            data: vec![],
            dimension: 0,
        };
        assert!(grpc_binary_to_internal(&empty).is_err());

        // Zero dimension with non-empty data
        let zero_dim = GrpcBinaryHypervector {
            data: vec![0xFF],
            dimension: 0,
        };
        assert!(grpc_binary_to_internal(&zero_dim).is_err());

        // Byte count mismatch
        let mismatch = GrpcBinaryHypervector {
            data: vec![0xFF, 0xFF], // 2 bytes
            dimension: 100,          // Needs 13 bytes
        };
        assert!(grpc_binary_to_internal(&mismatch).is_err());
    }

    #[test]
    fn test_sparse_gradient_roundtrip() {
        let original = SparseGradient::new(
            vec![0, 5, 10],
            vec![1.0, 2.0, 3.0],
            100,
        );
        let grpc = internal_sparse_to_grpc(&original);
        let recovered = grpc_sparse_to_internal(&grpc).unwrap();

        assert_eq!(original.indices, recovered.indices);
        assert_eq!(original.values, recovered.values);
        assert_eq!(original.full_dimension, recovered.full_dimension);
    }

    #[test]
    fn test_sparse_gradient_validation() {
        // Length mismatch
        let mismatch = GrpcSparseGradient {
            indices: vec![0, 1, 2],
            values: vec![1.0, 2.0],
            full_dimension: 100,
            sparsity_ratio: 0.03,
        };
        assert!(grpc_sparse_to_internal(&mismatch).is_err());

        // Index out of bounds
        let oob = GrpcSparseGradient {
            indices: vec![0, 1, 200],
            values: vec![1.0, 2.0, 3.0],
            full_dimension: 100,
            sparsity_ratio: 0.03,
        };
        assert!(grpc_sparse_to_internal(&oob).is_err());

        // Zero dimension
        let zero = GrpcSparseGradient {
            indices: vec![],
            values: vec![],
            full_dimension: 0,
            sparsity_ratio: 0.0,
        };
        assert!(grpc_sparse_to_internal(&zero).is_err());
    }

    #[test]
    fn test_quantized_gradient_roundtrip() {
        let dense = DenseGradient::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        let original = QuantizedGradient::from_dense(&dense, 8);
        let grpc = internal_quantized_to_grpc(&original);
        let recovered = grpc_quantized_to_internal(&grpc).unwrap();

        assert_eq!(original.data, recovered.data);
        assert_eq!(original.bits_per_value, recovered.bits_per_value);
        assert!((original.scale - recovered.scale).abs() < 0.001);
    }

    #[test]
    fn test_quantized_gradient_validation() {
        // Invalid bits_per_value
        let invalid_bits = GrpcQuantizedGradient {
            data: vec![0xFF],
            bits_per_value: 3,
            scale: 1.0,
            zero_point: 0.0,
            dimension: 8,
        };
        assert!(grpc_quantized_to_internal(&invalid_bits).is_err());

        // NaN scale
        let nan_scale = GrpcQuantizedGradient {
            data: vec![0xFF],
            bits_per_value: 8,
            scale: f32::NAN,
            zero_point: 0.0,
            dimension: 1,
        };
        assert!(grpc_quantized_to_internal(&nan_scale).is_err());
    }

    #[test]
    fn test_unified_payload_roundtrip_dense() {
        let dense = DenseGradient::from_vec(vec![1.0, 2.0, 3.0]);
        let original = UnifiedPayload::Dense(dense);

        let grpc = unified_payload_to_grpc(&original);
        let recovered = grpc_payload_to_unified(&grpc).unwrap();

        match recovered {
            UnifiedPayload::Dense(d) => {
                assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected Dense payload"),
        }
    }

    #[test]
    fn test_unified_payload_roundtrip_hyper() {
        let hyper = Hypervector::new(vec![10, -20, 30]);
        let original = UnifiedPayload::Hyper(hyper);

        let grpc = unified_payload_to_grpc(&original);
        let recovered = grpc_payload_to_unified(&grpc).unwrap();

        match recovered {
            UnifiedPayload::Hyper(h) => {
                assert_eq!(h.components, vec![10, -20, 30]);
            }
            _ => panic!("Expected Hyper payload"),
        }
    }

    #[test]
    fn test_unified_payload_empty() {
        let empty = GrpcPayload { data: None };
        assert!(grpc_payload_to_unified(&empty).is_err());
    }

    #[test]
    fn test_payload_type_conversion() {
        assert_eq!(
            payload_type_to_grpc(PayloadType::DenseGradient),
            GrpcPayloadType::DenseGradient
        );
        assert_eq!(
            payload_type_to_grpc(PayloadType::HyperEncoded),
            GrpcPayloadType::HyperEncoded
        );

        assert_eq!(
            grpc_to_payload_type(GrpcPayloadType::DenseGradient),
            Some(PayloadType::DenseGradient)
        );
        assert_eq!(
            grpc_to_payload_type(GrpcPayloadType::Unspecified),
            None
        );
    }

    #[test]
    fn test_aggregator_status_conversion() {
        use std::time::Duration;

        let status = AggregatorStatus {
            round: 10,
            registered_nodes: 5,
            submitted_nodes: 3,
            expected_nodes: 5,
            memory_bytes: 1_000_000,
            round_elapsed: Duration::from_secs(60),
            is_complete: false,
        };

        let grpc_response = aggregator_status_to_grpc_response(&status);

        assert_eq!(grpc_response.current_round, 10);
        assert_eq!(grpc_response.registered_nodes, 5);
        assert_eq!(grpc_response.submissions_this_round, 3);
        assert_eq!(grpc_response.expected_this_round, 5);
        assert_eq!(grpc_response.round_state, GrpcRoundState::Collecting as i32);
    }

    #[test]
    fn test_round_state_mapping() {
        use std::time::Duration;

        // Complete state
        let complete = AggregatorStatus {
            round: 0,
            registered_nodes: 3,
            submitted_nodes: 3,
            expected_nodes: 3,
            memory_bytes: 0,
            round_elapsed: Duration::ZERO,
            is_complete: true,
        };
        let resp = aggregator_status_to_grpc_response(&complete);
        assert_eq!(resp.round_state, GrpcRoundState::Complete as i32);

        // Collecting state
        let collecting = AggregatorStatus {
            round: 0,
            registered_nodes: 3,
            submitted_nodes: 1,
            expected_nodes: 3,
            memory_bytes: 0,
            round_elapsed: Duration::ZERO,
            is_complete: false,
        };
        let resp = aggregator_status_to_grpc_response(&collecting);
        assert_eq!(resp.round_state, GrpcRoundState::Collecting as i32);

        // Pending state
        let pending = AggregatorStatus {
            round: 0,
            registered_nodes: 3,
            submitted_nodes: 0,
            expected_nodes: 3,
            memory_bytes: 0,
            round_elapsed: Duration::ZERO,
            is_complete: false,
        };
        let resp = aggregator_status_to_grpc_response(&pending);
        assert_eq!(resp.round_state, GrpcRoundState::Pending as i32);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FL Serialization
//!
//! Serialization utilities for gradient transmission.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::{AggregatedGradient, AggregationMethod, GradientMetadata, GradientUpdate};

/// Serialization errors
#[derive(Debug, Error)]
pub enum SerializationError {
    #[error("Cannot serialize empty gradients")]
    EmptyGradients,
    #[error("Invalid gradient data: length must be a multiple of 8 bytes, got {0}")]
    InvalidLength(usize),
    #[error("Base64 decode error: {0}")]
    Base64Error(String),
}

/// Serialize gradients to bytes
///
/// Each f64 is serialized as 8 bytes in little-endian format.
pub fn serialize_gradients(gradients: &[f64]) -> Result<Vec<u8>, SerializationError> {
    if gradients.is_empty() {
        return Err(SerializationError::EmptyGradients);
    }

    let mut buffer = Vec::with_capacity(gradients.len() * 8);
    for &g in gradients {
        buffer.extend_from_slice(&g.to_le_bytes());
    }

    Ok(buffer)
}

/// Deserialize gradients from bytes
pub fn deserialize_gradients(data: &[u8]) -> Result<Vec<f64>, SerializationError> {
    if !data.len().is_multiple_of(8) {
        return Err(SerializationError::InvalidLength(data.len()));
    }

    let mut gradients = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        // Safe: chunks_exact guarantees exactly 8 bytes
        let bytes: [u8; 8] = chunk
            .try_into()
            .expect("chunks_exact(8) guarantees 8-byte chunks");
        gradients.push(f64::from_le_bytes(bytes));
    }

    Ok(gradients)
}

/// Serialized gradient update for JSON transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedGradientUpdate {
    /// Participant identifier
    pub participant_id: String,
    /// Model version
    pub model_version: u64,
    /// Base64-encoded gradients
    pub gradients: String,
    /// Update metadata
    pub metadata: GradientMetadata,
}

impl SerializedGradientUpdate {
    /// Create from GradientUpdate
    pub fn from_update(update: &GradientUpdate) -> Result<Self, SerializationError> {
        let bytes = serialize_gradients(&update.gradients)?;
        let base64 = BASE64.encode(&bytes);

        Ok(Self {
            participant_id: update.participant_id.clone(),
            model_version: update.model_version,
            gradients: base64,
            metadata: update.metadata.clone(),
        })
    }

    /// Convert to GradientUpdate
    pub fn to_update(&self) -> Result<GradientUpdate, SerializationError> {
        let bytes = BASE64
            .decode(&self.gradients)
            .map_err(|e| SerializationError::Base64Error(e.to_string()))?;
        let gradients = deserialize_gradients(&bytes)?;

        Ok(GradientUpdate::with_metadata(
            self.participant_id.clone(),
            self.model_version,
            gradients,
            self.metadata.clone(),
        ))
    }
}

/// Serialized aggregated gradient for JSON transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedAggregatedGradient {
    /// Base64-encoded gradients
    pub gradients: String,
    /// Model version
    pub model_version: u64,
    /// Number of participants included
    pub participant_count: usize,
    /// Number of participants excluded
    pub excluded_count: usize,
    /// Aggregation method used
    pub aggregation_method: AggregationMethod,
    /// Timestamp
    pub timestamp: u64,
}

impl SerializedAggregatedGradient {
    /// Create from AggregatedGradient
    pub fn from_aggregated(aggregated: &AggregatedGradient) -> Result<Self, SerializationError> {
        let bytes = serialize_gradients(&aggregated.gradients)?;
        let base64 = BASE64.encode(&bytes);

        Ok(Self {
            gradients: base64,
            model_version: aggregated.model_version,
            participant_count: aggregated.participant_count,
            excluded_count: aggregated.excluded_count,
            aggregation_method: aggregated.aggregation_method,
            timestamp: aggregated.timestamp,
        })
    }

    /// Convert to AggregatedGradient
    pub fn to_aggregated(&self) -> Result<AggregatedGradient, SerializationError> {
        let bytes = BASE64
            .decode(&self.gradients)
            .map_err(|e| SerializationError::Base64Error(e.to_string()))?;
        let gradients = deserialize_gradients(&bytes)?;

        Ok(AggregatedGradient {
            gradients,
            model_version: self.model_version,
            participant_count: self.participant_count,
            excluded_count: self.excluded_count,
            aggregation_method: self.aggregation_method,
            timestamp: self.timestamp,
        })
    }
}

/// Serialize gradient update to JSON string
#[allow(dead_code)]
pub fn to_json(update: &GradientUpdate) -> Result<String, SerializationError> {
    let serialized = SerializedGradientUpdate::from_update(update)?;
    serde_json::to_string(&serialized).map_err(|e| SerializationError::Base64Error(e.to_string()))
}

/// Deserialize gradient update from JSON string
#[allow(dead_code)]
pub fn from_json(json: &str) -> Result<GradientUpdate, SerializationError> {
    let serialized: SerializedGradientUpdate =
        serde_json::from_str(json).map_err(|e| SerializationError::Base64Error(e.to_string()))?;
    serialized.to_update()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_gradients() {
        let gradients = vec![0.1, 0.2, 0.3, 0.4];
        let bytes = serialize_gradients(&gradients).expect("Serialization failed");

        assert_eq!(bytes.len(), 32); // 4 f64s * 8 bytes

        let deserialized = deserialize_gradients(&bytes).expect("Deserialization failed");
        assert_eq!(gradients.len(), deserialized.len());

        for (orig, deser) in gradients.iter().zip(deserialized.iter()) {
            assert!((orig - deser).abs() < 1e-10);
        }
    }

    #[test]
    fn test_serialize_empty() {
        let result = serialize_gradients(&[]);
        assert!(matches!(result, Err(SerializationError::EmptyGradients)));
    }

    #[test]
    fn test_deserialize_invalid_length() {
        let result = deserialize_gradients(&[0, 1, 2]); // Not multiple of 8
        assert!(matches!(result, Err(SerializationError::InvalidLength(3))));
    }

    #[test]
    fn test_serialized_gradient_update() {
        let update = GradientUpdate::new(
            "participant-1".to_string(),
            1,
            vec![0.1, 0.2, 0.3],
            100,
            0.5,
        );

        let serialized =
            SerializedGradientUpdate::from_update(&update).expect("Serialization failed");

        assert_eq!(serialized.participant_id, "participant-1");
        assert!(!serialized.gradients.is_empty());

        let deserialized = serialized.to_update().expect("Deserialization failed");
        assert_eq!(update.participant_id, deserialized.participant_id);
        assert_eq!(update.model_version, deserialized.model_version);
        assert_eq!(update.gradients.len(), deserialized.gradients.len());
    }

    #[test]
    fn test_json_roundtrip() {
        let update = GradientUpdate::new("p1".to_string(), 1, vec![1.0, 2.0, 3.0], 50, 0.25);

        let json = to_json(&update).expect("JSON serialization failed");
        let parsed = from_json(&json).expect("JSON deserialization failed");

        assert_eq!(update.participant_id, parsed.participant_id);
        assert_eq!(update.model_version, parsed.model_version);
        assert_eq!(update.metadata.batch_size, parsed.metadata.batch_size);
    }

    #[test]
    fn test_serialized_aggregated_gradient() {
        let aggregated =
            AggregatedGradient::new(vec![0.15, 0.25, 0.35], 1, 3, 1, AggregationMethod::FedAvg);

        let serialized = SerializedAggregatedGradient::from_aggregated(&aggregated)
            .expect("Serialization failed");

        let deserialized = serialized.to_aggregated().expect("Deserialization failed");

        assert_eq!(aggregated.participant_count, deserialized.participant_count);
        assert_eq!(aggregated.excluded_count, deserialized.excluded_count);
        assert_eq!(
            aggregated.aggregation_method,
            deserialized.aggregation_method
        );
    }
}

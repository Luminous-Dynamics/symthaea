// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provider-neutral dense visual feature evidence for VIS-002.
//!
//! A model name is not execution evidence. This module requires every dense feature result to
//! carry the concrete source observation, the backend class that actually executed, explicit
//! model/preprocessing artifact identity, validated tensor geometry, and `Inferred` epistemic
//! provenance. Learned features remain sensory evidence contributors, not world-model truth.

use serde::{Deserialize, Serialize};

use crate::epistemic::{VisualEvidence, VisualObservationRef, VisualOrigin};

/// Conservative ceiling for one returned dense feature tensor (64 MiB of f32 values).
pub const MAX_DENSE_FEATURE_ELEMENTS: usize = 16 * 1024 * 1024;
/// Conservative ceiling for one tightly packed source raster admitted through this contract.
pub const MAX_DENSE_INPUT_BYTES: usize = 256 * 1024 * 1024;

/// Runtime/backend family that actually produced a dense feature result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseEncoderBackend {
    /// Deterministic in-process test/reference implementation.
    DeterministicReference,
    /// ONNX Runtime or a compatible ONNX execution provider.
    Onnx,
    /// Candle-native model execution.
    Candle,
    /// LibTorch/PyTorch execution outside this crate.
    Torch,
    /// Remote or process-external inference service.
    ExternalService,
    /// Backend identity is unavailable. Downstream assurance must not infer a stronger value.
    Unknown,
}

/// Digest algorithm used to identify exact artifact bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDigest {
    Blake3([u8; 32]),
    Sha256([u8; 32]),
}

/// Whether an execution component has exact artifact identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactIdentityKind {
    /// Built into the running binary, but not independently content-addressed by this receipt.
    BuiltIn,
    /// Exact bytes are content-addressed by `digest`.
    Pinned,
    /// A human/model label is known but the exact bytes are not content-addressed.
    Unpinned,
    /// No useful artifact identity is available.
    Unknown,
}

/// Identity of a model, preprocessing graph, tokenizer/config bundle, or other execution artifact.
///
/// Fields are private so `Pinned` cannot exist without a digest and labels cannot be empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactIdentity {
    kind: ArtifactIdentityKind,
    label: String,
    revision: Option<String>,
    digest: Option<ArtifactDigest>,
}

impl ArtifactIdentity {
    pub fn pinned(
        label: impl Into<String>,
        revision: Option<String>,
        digest: ArtifactDigest,
    ) -> Result<Self, DenseEncoderError> {
        let label = validate_label(label.into())?;
        let revision = validate_optional_label(revision)?;
        Ok(Self {
            kind: ArtifactIdentityKind::Pinned,
            label,
            revision,
            digest: Some(digest),
        })
    }

    pub fn unpinned(
        label: impl Into<String>,
        revision: Option<String>,
    ) -> Result<Self, DenseEncoderError> {
        let label = validate_label(label.into())?;
        let revision = validate_optional_label(revision)?;
        Ok(Self {
            kind: ArtifactIdentityKind::Unpinned,
            label,
            revision,
            digest: None,
        })
    }

    pub fn built_in(label: impl Into<String>) -> Result<Self, DenseEncoderError> {
        Ok(Self {
            kind: ArtifactIdentityKind::BuiltIn,
            label: validate_label(label.into())?,
            revision: None,
            digest: None,
        })
    }

    pub fn unknown() -> Self {
        Self {
            kind: ArtifactIdentityKind::Unknown,
            label: "unknown".to_string(),
            revision: None,
            digest: None,
        }
    }

    pub const fn kind(&self) -> ArtifactIdentityKind {
        self.kind
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    pub const fn digest(&self) -> Option<ArtifactDigest> {
        self.digest
    }

    pub const fn exact_bytes_pinned(&self) -> bool {
        matches!(self.kind, ArtifactIdentityKind::Pinned) && self.digest.is_some()
    }
}

/// Semantic meaning of the returned dense grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseFeatureSemantics {
    /// Spatial patch/token embeddings retaining a 2-D correspondence to the input image.
    SpatialTokens,
    /// Feature pyramid level; consumers must retain provider-specific scale metadata elsewhere.
    PyramidLevel,
    /// A dense map whose exact semantics are provider-defined.
    ProviderDefined,
}

/// Execution receipt attached to a dense feature map.
///
/// This records artifact identity only. It does not claim numerical determinism across hardware,
/// runtime versions, kernels, precision modes, or accelerators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DenseEncoderReceipt {
    backend: DenseEncoderBackend,
    provider: String,
    model: ArtifactIdentity,
    preprocessing: ArtifactIdentity,
    semantics: DenseFeatureSemantics,
}

impl DenseEncoderReceipt {
    pub fn new(
        backend: DenseEncoderBackend,
        provider: impl Into<String>,
        model: ArtifactIdentity,
        preprocessing: ArtifactIdentity,
        semantics: DenseFeatureSemantics,
    ) -> Result<Self, DenseEncoderError> {
        Ok(Self {
            backend,
            provider: validate_label(provider.into())?,
            model,
            preprocessing,
            semantics,
        })
    }

    pub const fn backend(&self) -> DenseEncoderBackend {
        self.backend
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub const fn semantics(&self) -> DenseFeatureSemantics {
        self.semantics
    }

    pub const fn model(&self) -> &ArtifactIdentity {
        &self.model
    }

    pub const fn preprocessing(&self) -> &ArtifactIdentity {
        &self.preprocessing
    }

    /// True only when both model and preprocessing artifact bytes are content-addressed.
    ///
    /// This is deliberately weaker than "reproducible execution"; runtime/hardware determinism is
    /// outside this receipt.
    pub const fn model_and_preprocessing_pinned(&self) -> bool {
        self.model.exact_bytes_pinned() && self.preprocessing.exact_bytes_pinned()
    }
}

/// Borrowed input to a dense visual encoder.
#[derive(Debug, Clone, Copy)]
pub struct DenseVisualInput<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    channels: usize,
    observation: VisualObservationRef,
}

impl<'a> DenseVisualInput<'a> {
    pub fn new(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        channels: usize,
        observation: VisualObservationRef,
    ) -> Result<Self, DenseEncoderError> {
        if width == 0 || height == 0 {
            return Err(DenseEncoderError::InvalidInputDimensions);
        }
        if !matches!(channels, 1 | 3 | 4) {
            return Err(DenseEncoderError::UnsupportedChannels { channels });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or(DenseEncoderError::TensorSizeOverflow)?;
        if expected > MAX_DENSE_INPUT_BYTES {
            return Err(DenseEncoderError::InputTooLarge { bytes: expected });
        }
        if pixels.len() != expected {
            return Err(DenseEncoderError::InputLengthMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels,
            observation,
        })
    }

    pub const fn pixels(&self) -> &'a [u8] {
        self.pixels
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn observation(&self) -> VisualObservationRef {
        self.observation
    }
}

/// Validated dense feature tensor plus execution and epistemic evidence.
///
/// Layout is row-major `[grid_rows, grid_cols, feature_dim]` in `values`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DenseFeatureMap {
    grid_rows: usize,
    grid_cols: usize,
    feature_dim: usize,
    values: Vec<f32>,
    receipt: DenseEncoderReceipt,
    evidence: VisualEvidence,
}

impl DenseFeatureMap {
    pub fn new(
        grid_rows: usize,
        grid_cols: usize,
        feature_dim: usize,
        values: Vec<f32>,
        receipt: DenseEncoderReceipt,
        source_observation: VisualObservationRef,
        confidence: f32,
    ) -> Result<Self, DenseEncoderError> {
        if grid_rows == 0 || grid_cols == 0 || feature_dim == 0 {
            return Err(DenseEncoderError::InvalidOutputDimensions);
        }
        let expected = grid_rows
            .checked_mul(grid_cols)
            .and_then(|tokens| tokens.checked_mul(feature_dim))
            .ok_or(DenseEncoderError::TensorSizeOverflow)?;
        if expected > MAX_DENSE_FEATURE_ELEMENTS {
            return Err(DenseEncoderError::OutputTooLarge { elements: expected });
        }
        if values.len() != expected {
            return Err(DenseEncoderError::OutputLengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(DenseEncoderError::NonFiniteFeatureValue);
        }
        let evidence = VisualEvidence::inferred(vec![source_observation], confidence)
            .map_err(|_| DenseEncoderError::InvalidConfidence)?;
        debug_assert_eq!(evidence.origin(), VisualOrigin::Inferred);
        Ok(Self {
            grid_rows,
            grid_cols,
            feature_dim,
            values,
            receipt,
            evidence,
        })
    }

    pub const fn grid_rows(&self) -> usize {
        self.grid_rows
    }

    pub const fn grid_cols(&self) -> usize {
        self.grid_cols
    }

    pub const fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub const fn receipt(&self) -> &DenseEncoderReceipt {
        &self.receipt
    }

    pub const fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }

    pub fn token(&self, row: usize, col: usize) -> Option<&[f32]> {
        if row >= self.grid_rows || col >= self.grid_cols {
            return None;
        }
        let start = (row * self.grid_cols + col) * self.feature_dim;
        Some(&self.values[start..start + self.feature_dim])
    }
}

/// Provider-neutral dense visual encoder contract.
///
/// Implementations may use learned or deterministic models, but all successful outputs must be
/// constructible as `DenseFeatureMap` and therefore remain observation-grounded `Inferred`
/// evidence with explicit execution identity.
pub trait DenseVisualEncoder: Send {
    fn encode(
        &mut self,
        input: DenseVisualInput<'_>,
    ) -> Result<DenseFeatureMap, DenseEncoderError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseEncoderError {
    EmptyLabel,
    InvalidInputDimensions,
    UnsupportedChannels { channels: usize },
    TensorSizeOverflow,
    InputTooLarge { bytes: usize },
    InputLengthMismatch { expected: usize, actual: usize },
    InvalidOutputDimensions,
    OutputTooLarge { elements: usize },
    OutputLengthMismatch { expected: usize, actual: usize },
    NonFiniteFeatureValue,
    InvalidConfidence,
    BackendFailure(String),
}

impl std::fmt::Display for DenseEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLabel => f.write_str("dense encoder identity labels must be non-empty"),
            Self::InvalidInputDimensions => {
                f.write_str("dense visual input dimensions must be non-zero")
            }
            Self::UnsupportedChannels { channels } => {
                write!(f, "dense visual input channels must be 1, 3, or 4, got {channels}")
            }
            Self::TensorSizeOverflow => f.write_str("dense visual tensor size arithmetic overflow"),
            Self::InputTooLarge { bytes } => write!(
                f,
                "dense visual input requires {bytes} bytes, exceeding the contract ceiling"
            ),
            Self::InputLengthMismatch { expected, actual } => write!(
                f,
                "dense visual input expected {expected} bytes but received {actual}"
            ),
            Self::InvalidOutputDimensions => {
                f.write_str("dense feature grid dimensions must be non-zero")
            }
            Self::OutputTooLarge { elements } => write!(
                f,
                "dense feature output requires {elements} elements, exceeding the contract ceiling"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "dense feature output expected {expected} elements but received {actual}"
            ),
            Self::NonFiniteFeatureValue => {
                f.write_str("dense feature output contains a non-finite value")
            }
            Self::InvalidConfidence => {
                f.write_str("dense encoder confidence must be finite and within [0, 1]")
            }
            Self::BackendFailure(message) => write!(f, "dense encoder backend failed: {message}"),
        }
    }
}

impl std::error::Error for DenseEncoderError {}

fn validate_label(label: String) -> Result<String, DenseEncoderError> {
    if label.trim().is_empty() {
        return Err(DenseEncoderError::EmptyLabel);
    }
    Ok(label)
}

fn validate_optional_label(label: Option<String>) -> Result<Option<String>, DenseEncoderError> {
    label.map(validate_label).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{VisualCaptureClock, VisualStreamRef};

    fn observation() -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(51, 7).unwrap(),
            12,
            800_000,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn receipt(pinned: bool) -> DenseEncoderReceipt {
        let model = if pinned {
            ArtifactIdentity::pinned(
                "example-model",
                Some("r1".to_string()),
                ArtifactDigest::Blake3([7; 32]),
            )
            .unwrap()
        } else {
            ArtifactIdentity::unpinned("example-model", Some("r1".to_string())).unwrap()
        };
        let preprocessing = ArtifactIdentity::pinned(
            "rgb-resize-normalize-v1",
            None,
            ArtifactDigest::Blake3([9; 32]),
        )
        .unwrap();
        DenseEncoderReceipt::new(
            DenseEncoderBackend::Onnx,
            "test-provider",
            model,
            preprocessing,
            DenseFeatureSemantics::SpatialTokens,
        )
        .unwrap()
    }

    #[test]
    fn input_rejects_wrong_pixel_length() {
        assert_eq!(
            DenseVisualInput::new(&[0; 11], 2, 2, 3, observation()).unwrap_err(),
            DenseEncoderError::InputLengthMismatch {
                expected: 12,
                actual: 11,
            }
        );
    }

    #[test]
    fn feature_map_is_always_inferred_from_exact_observation() {
        let obs = observation();
        let map = DenseFeatureMap::new(2, 2, 3, vec![0.5; 12], receipt(true), obs, 1.0)
            .unwrap();
        assert_eq!(map.evidence().origin(), VisualOrigin::Inferred);
        assert_eq!(map.evidence().parent_observations(), &[obs]);
        assert_eq!(map.token(1, 1).unwrap(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn confidence_cannot_upgrade_dense_features_to_observed() {
        let map = DenseFeatureMap::new(
            1,
            1,
            2,
            vec![0.0, 1.0],
            receipt(true),
            observation(),
            1.0,
        )
        .unwrap();
        assert_ne!(map.evidence().origin(), VisualOrigin::Observed);
    }

    #[test]
    fn non_finite_feature_values_fail_closed() {
        assert_eq!(
            DenseFeatureMap::new(
                1,
                1,
                2,
                vec![0.0, f32::NAN],
                receipt(true),
                observation(),
                0.8,
            ),
            Err(DenseEncoderError::NonFiniteFeatureValue)
        );
    }

    #[test]
    fn unpinned_model_is_explicit_not_silently_reproducible() {
        let unpinned = receipt(false);
        assert!(!unpinned.model().exact_bytes_pinned());
        assert!(!unpinned.model_and_preprocessing_pinned());

        let pinned = receipt(true);
        assert!(pinned.model().exact_bytes_pinned());
        assert!(pinned.model_and_preprocessing_pinned());
    }

    #[test]
    fn pinned_identity_requires_nonempty_label() {
        assert_eq!(
            ArtifactIdentity::pinned("   ", None, ArtifactDigest::Sha256([1; 32])),
            Err(DenseEncoderError::EmptyLabel)
        );
    }

    struct ReferenceEncoder;

    impl DenseVisualEncoder for ReferenceEncoder {
        fn encode(
            &mut self,
            input: DenseVisualInput<'_>,
        ) -> Result<DenseFeatureMap, DenseEncoderError> {
            DenseFeatureMap::new(
                1,
                1,
                2,
                vec![input.pixels()[0] as f32 / 255.0, 1.0],
                DenseEncoderReceipt::new(
                    DenseEncoderBackend::DeterministicReference,
                    "reference-encoder",
                    ArtifactIdentity::built_in("reference-feature-v1")?,
                    ArtifactIdentity::built_in("identity-preprocess-v1")?,
                    DenseFeatureSemantics::SpatialTokens,
                )?,
                input.observation(),
                1.0,
            )
        }
    }

    #[test]
    fn provider_trait_preserves_source_observation() {
        let pixels = [128u8; 12];
        let input = DenseVisualInput::new(&pixels, 2, 2, 3, observation()).unwrap();
        let mut encoder = ReferenceEncoder;
        let map = encoder.encode(input).unwrap();
        assert_eq!(map.evidence().parent_observations(), &[observation()]);
        assert_eq!(map.receipt().backend(), DenseEncoderBackend::DeterministicReference);
    }
}

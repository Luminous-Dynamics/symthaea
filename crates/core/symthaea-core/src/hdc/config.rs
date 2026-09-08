// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Configuration - Centralized Dimension Management
//!
//! This module provides a centralized configuration system for HDC dimensions
//! across all Symthaea components. It ensures consistency between STT, TTS,
//! and core consciousness systems.
//!
//! It also owns the stable identity contract for HDC-LTC temporal evolution and
//! the versioned persistent learner snapshot that binds complete recurrent state
//! to that temporal semantics.
//!
//! ## Design Goals
//!
//! 1. **Single Source of Truth**: All dimension constants flow from here
//! 2. **Runtime Configurable**: Set dimension at startup before first use
//! 3. **Backward Compatible**: Defaults to 16,384 (existing behavior)
//! 4. **Compile-Time Options**: Predefined tiers for common configurations
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::config::{HdcConfig, set_hdc_config, hdc_dim, hdc_config};
//!
//! // Set configuration at startup (once, before any HDC operations)
//! set_hdc_config(HdcConfig::STANDARD);
//!
//! // Or use custom configuration
//! set_hdc_config(HdcConfig {
//!     dimension: 8192,
//!     num_levels: 16,
//!     sparse_density: 0.1,
//! });
//!
//! // Access dimension anywhere
//! let dim = hdc_dim();  // Returns configured dimension
//! let config = hdc_config();  // Returns full config
//! ```
//!
//! ## Dimension Tiers
//!
//! | Tier | Dimension | Memory/Vec | Use Case |
//! |------|-----------|------------|----------|
//! | Compact | 4,096 | 16 KB | Embedded, mobile |
//! | Standard | 16,384 | 64 KB | General use |
//! | Extended | 32,768 | 128 KB | High precision |
//! | Ultra | 65,536 | 256 KB | Maximum capacity |

use super::hdc_ltc_unified::HdcLtcUnifiedNetwork;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Global HDC configuration (set once at startup)
static HDC_CONFIG: OnceLock<HdcConfig> = OnceLock::new();

// ═══════════════════════════════════════════════════════════════════════════════
// HDC-LTC TEMPORAL SEMANTICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Stable identity for one HDC-LTC temporal evolution policy.
///
/// This enum names semantics, not a performance implementation. Two kernels may
/// share one profile only when they implement the same temporal update rule
/// within the tolerance established by their qualification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HdcLtcEvolutionProfile {
    /// Learned CfC-style gate used by `HdcLtcUnifiedNeuron::evolve_closed_form`.
    AdaptiveClosedGateV1,
    /// Single frozen-equilibrium exponential step used by
    /// `HdcLtcUnifiedNeuron::evolve_closed_form_exact`.
    FrozenEquilibriumExponentialV1,
    /// Recomputed-equilibrium sub-stepping used by
    /// `HdcLtcUnifiedNeuron::evolve_closed_form_iterative`.
    SubsteppedExponentialV1,
    /// Numerical RK4 reference integration.
    Rk4ReferenceV1,
}

impl HdcLtcEvolutionProfile {
    pub const fn schema_id(self) -> &'static str {
        match self {
            Self::AdaptiveClosedGateV1 => "symthaea.hdc-ltc.evolution.adaptive-closed-gate.v1",
            Self::FrozenEquilibriumExponentialV1 => {
                "symthaea.hdc-ltc.evolution.frozen-equilibrium-exponential.v1"
            }
            Self::SubsteppedExponentialV1 => {
                "symthaea.hdc-ltc.evolution.substepped-exponential.v1"
            }
            Self::Rk4ReferenceV1 => "symthaea.hdc-ltc.evolution.rk4-reference.v1",
        }
    }

    /// Whether this profile is intended as a numerical reference rather than a
    /// learned/control-time evolution policy.
    pub const fn is_reference(self) -> bool {
        matches!(self, Self::Rk4ReferenceV1)
    }
}

/// Versioned identity for one HDC-LTC model under one temporal semantics profile.
///
/// Digest construction is deliberately owned by the snapshot/qualification layer;
/// this type only prevents an evidence record from carrying an unbound all-zero
/// architecture or parameter identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcLtcModelIdentityV1 {
    pub schema_id: String,
    pub evolution_profile: HdcLtcEvolutionProfile,
    pub architecture_digest: [u8; 32],
    pub parameter_digest: [u8; 32],
}

impl HdcLtcModelIdentityV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.hdc-ltc.model-identity.v1";

    pub fn new(
        evolution_profile: HdcLtcEvolutionProfile,
        architecture_digest: [u8; 32],
        parameter_digest: [u8; 32],
    ) -> Self {
        Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            evolution_profile,
            architecture_digest,
            parameter_digest,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err("unsupported HDC-LTC model identity schema");
        }
        if self.architecture_digest == [0; 32] {
            return Err("architecture digest must be non-zero");
        }
        if self.parameter_digest == [0; 32] {
            return Err("parameter digest must be non-zero");
        }
        Ok(())
    }
}

/// Evidence describing one requested HDC-LTC temporal evolution interval.
///
/// This is not a safety or capability certificate. It records exactly which
/// temporal semantics and model identity were used for a finite, non-negative
/// interval so downstream evidence cannot silently conflate evolution methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdcLtcTemporalSemanticsReceiptV1 {
    pub schema_id: String,
    pub model: HdcLtcModelIdentityV1,
    pub dt_seconds: f32,
}

impl HdcLtcTemporalSemanticsReceiptV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.hdc-ltc.temporal-semantics-receipt.v1";

    pub fn new(model: HdcLtcModelIdentityV1, dt_seconds: f32) -> Result<Self, &'static str> {
        model.validate()?;
        if !dt_seconds.is_finite() || dt_seconds < 0.0 {
            return Err("dt must be finite and non-negative");
        }
        Ok(Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            model,
            dt_seconds,
        })
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err("unsupported temporal semantics receipt schema");
        }
        self.model.validate()?;
        if !self.dt_seconds.is_finite() || self.dt_seconds < 0.0 {
            return Err("dt must be finite and non-negative");
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLETE HDC-LTC LEARNER SNAPSHOT
// ═══════════════════════════════════════════════════════════════════════════════

const HDC_LTC_ARCHITECTURE_DOMAIN_V1: &[u8] = b"symthaea.hdc-ltc.architecture.v1\0";
const HDC_LTC_PARAMETER_DOMAIN_V1: &[u8] = b"symthaea.hdc-ltc.complete-state.v1\0";
const HDC_LTC_SNAPSHOT_DOMAIN_V1: &[u8] = b"symthaea.hdc-ltc.learner-snapshot.v1\0";

fn hdc_ltc_digest(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

/// Durable snapshot of the *complete* serializable HDC-LTC learner.
///
/// This is deliberately different from `NetworkStateSnapshot`, whose job is a
/// cheap allocation-reusing save/restore of mutable evolution state for pure
/// prediction and which explicitly excludes learning parameters. This snapshot
/// serializes the entire `HdcLtcUnifiedNetwork`, including recurrent weights,
/// masks, gates, momentum, running statistics, evolution clocks, layer bindings,
/// cached outputs, and network configuration.
///
/// The v1 payload uses the workspace-pinned `bincode` representation. It is an
/// implementation-lineage identity, not a claim of cross-language canonical wire
/// compatibility. A change of encoding contract requires a new schema/domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcLearnerSnapshotV1 {
    pub schema_id: String,
    pub evolution_profile: HdcLtcEvolutionProfile,
    pub architecture_digest: [u8; 32],
    pub parameter_digest: [u8; 32],
    pub snapshot_digest: [u8; 32],
    network_bytes: Vec<u8>,
}

impl HdcLtcLearnerSnapshotV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.hdc-ltc.learner-snapshot.v1";

    pub fn capture(
        network: &HdcLtcUnifiedNetwork,
        evolution_profile: HdcLtcEvolutionProfile,
    ) -> Result<Self, String> {
        let architecture_bytes = bincode::serialize(network.config())
            .map_err(|err| format!("serialize HDC-LTC architecture: {err}"))?;
        let network_bytes = bincode::serialize(network)
            .map_err(|err| format!("serialize complete HDC-LTC learner: {err}"))?;

        let architecture_digest =
            hdc_ltc_digest(HDC_LTC_ARCHITECTURE_DOMAIN_V1, &architecture_bytes);
        let parameter_digest = hdc_ltc_digest(HDC_LTC_PARAMETER_DOMAIN_V1, &network_bytes);

        let mut commitment = Vec::with_capacity(
            Self::SCHEMA_ID.len()
                + evolution_profile.schema_id().len()
                + architecture_digest.len()
                + parameter_digest.len()
                + network_bytes.len()
                + 16,
        );
        commitment.extend_from_slice(Self::SCHEMA_ID.as_bytes());
        commitment.push(0);
        commitment.extend_from_slice(evolution_profile.schema_id().as_bytes());
        commitment.push(0);
        commitment.extend_from_slice(&architecture_digest);
        commitment.extend_from_slice(&parameter_digest);
        commitment.extend_from_slice(&(network_bytes.len() as u64).to_le_bytes());
        commitment.extend_from_slice(&network_bytes);
        let snapshot_digest = hdc_ltc_digest(HDC_LTC_SNAPSHOT_DOMAIN_V1, &commitment);

        Ok(Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            evolution_profile,
            architecture_digest,
            parameter_digest,
            snapshot_digest,
            network_bytes,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err("unsupported HDC-LTC learner snapshot schema".into());
        }
        let network: HdcLtcUnifiedNetwork = bincode::deserialize(&self.network_bytes)
            .map_err(|err| format!("decode complete HDC-LTC learner: {err}"))?;
        let expected = Self::capture(&network, self.evolution_profile)?;
        if self.architecture_digest != expected.architecture_digest {
            return Err("HDC-LTC architecture digest mismatch".into());
        }
        if self.parameter_digest != expected.parameter_digest {
            return Err("HDC-LTC complete-state digest mismatch".into());
        }
        if self.snapshot_digest != expected.snapshot_digest {
            return Err("HDC-LTC learner snapshot digest mismatch".into());
        }
        Ok(())
    }

    pub fn model_identity(&self) -> Result<HdcLtcModelIdentityV1, String> {
        self.validate()?;
        Ok(HdcLtcModelIdentityV1::new(
            self.evolution_profile,
            self.architecture_digest,
            self.parameter_digest,
        ))
    }

    pub fn restore(&self) -> Result<HdcLtcUnifiedNetwork, String> {
        self.validate()?;
        bincode::deserialize(&self.network_bytes)
            .map_err(|err| format!("restore complete HDC-LTC learner: {err}"))
    }

    pub fn encoded_network(&self) -> &[u8] {
        &self.network_bytes
    }
}

/// HDC configuration parameters
///
/// Encapsulates all dimension-related settings for HDC operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdcConfig {
    /// Vector dimension (number of components)
    ///
    /// Should be a power of 2 for SIMD optimization.
    /// Common values: 4096, 8192, 16384, 32768, 65536
    pub dimension: usize,

    /// Number of quantization levels for continuous-to-discrete mapping
    ///
    /// Used when converting continuous values to discrete levels.
    /// More levels = finer granularity.
    pub num_levels: usize,

    /// Target density for sparse operations (0.0 to 1.0)
    ///
    /// Controls how many elements are non-zero in sparse representations.
    /// Lower values = more memory efficient but less precise.
    pub sparse_density: f32,
}

impl HdcConfig {
    /// Compact configuration: 4,096 dimensions
    ///
    /// Use for:
    /// - Embedded systems
    /// - Mobile applications
    /// - Memory-constrained environments
    /// - Quick prototyping
    pub const COMPACT: Self = Self {
        dimension: 4_096,
        num_levels: 8,
        sparse_density: 0.2,
    };

    /// Standard configuration: 16,384 dimensions (default)
    ///
    /// Use for:
    /// - General-purpose applications
    /// - Good balance of accuracy and memory
    /// - Compatible with existing Symthaea modules
    pub const STANDARD: Self = Self {
        dimension: 16_384,
        num_levels: 16,
        sparse_density: 0.1,
    };

    /// Extended configuration: 32,768 dimensions
    ///
    /// Use for:
    /// - High-precision applications
    /// - Complex semantic spaces
    /// - Multi-modal integration
    pub const EXTENDED: Self = Self {
        dimension: 32_768,
        num_levels: 32,
        sparse_density: 0.05,
    };

    /// Ultra configuration: 65,536 dimensions
    ///
    /// Use for:
    /// - Maximum capacity requirements
    /// - Research applications
    /// - Systems with abundant memory
    pub const ULTRA: Self = Self {
        dimension: 65_536,
        num_levels: 64,
        sparse_density: 0.025,
    };

    /// Create a custom configuration
    ///
    /// # Arguments
    ///
    /// * `dimension` - Vector dimension (should be power of 2)
    /// * `num_levels` - Quantization levels
    /// * `sparse_density` - Target sparsity (0.0 to 1.0)
    pub const fn custom(dimension: usize, num_levels: usize, sparse_density: f32) -> Self {
        Self {
            dimension,
            num_levels,
            sparse_density,
        }
    }

    /// Check if dimension is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        self.dimension > 0 && (self.dimension & (self.dimension - 1)) == 0
    }

    /// Get memory usage per continuous vector in bytes
    pub const fn memory_per_continuous_vec(&self) -> usize {
        self.dimension * 4 // f32 = 4 bytes
    }

    /// Get memory usage per binary vector in bytes
    pub const fn memory_per_binary_vec(&self) -> usize {
        self.dimension / 8 // 8 bits per byte
    }

    /// Get the expansion factor from a source dimension
    ///
    /// Returns how many times larger this config's dimension is
    /// compared to the source dimension.
    pub const fn expansion_factor_from(&self, source_dim: usize) -> usize {
        match self.dimension.checked_div(source_dim) {
            Some(v) => v,
            None => 0,
        }
    }

    /// Create configuration for a specific tier by name
    pub fn from_tier(tier: &str) -> Option<Self> {
        match tier.to_lowercase().as_str() {
            "compact" | "4k" => Some(Self::COMPACT),
            "standard" | "16k" | "default" => Some(Self::STANDARD),
            "extended" | "32k" => Some(Self::EXTENDED),
            "ultra" | "64k" => Some(Self::ULTRA),
            _ => None,
        }
    }

    /// Get tier name for this configuration
    pub fn tier_name(&self) -> &'static str {
        match self.dimension {
            4_096 => "compact",
            16_384 => "standard",
            32_768 => "extended",
            65_536 => "ultra",
            _ => "custom",
        }
    }
}

impl Default for HdcConfig {
    fn default() -> Self {
        Self::STANDARD
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL CONFIGURATION ACCESSORS
// ═══════════════════════════════════════════════════════════════════════════════

/// Set the global HDC configuration (call once at startup)
///
/// # Panics
///
/// Panics if called more than once. Configuration should be set
/// before any HDC operations begin.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::hdc::config::{HdcConfig, set_hdc_config};
///
/// // In main() or initialization:
/// set_hdc_config(HdcConfig::STANDARD);
/// ```
pub fn set_hdc_config(config: HdcConfig) {
    HDC_CONFIG
        .set(config)
        .expect("HDC configuration already set - can only be configured once at startup");
}

/// Try to set the global HDC configuration
///
/// Returns `true` if successful, `false` if already set.
/// Use this for graceful handling when configuration might already exist.
pub fn try_set_hdc_config(config: HdcConfig) -> bool {
    HDC_CONFIG.set(config).is_ok()
}

/// Get the current HDC dimension
///
/// Returns the configured dimension, or 16,384 (STANDARD) if not configured.
///
/// This is the primary accessor for dimension throughout the codebase.
#[inline]
pub fn hdc_dim() -> usize {
    hdc_config().dimension
}

/// Get the full HDC configuration
///
/// Returns the configured settings, or STANDARD defaults if not configured.
#[inline]
pub fn hdc_config() -> HdcConfig {
    HDC_CONFIG.get().copied().unwrap_or(HdcConfig::STANDARD)
}

/// Check if HDC configuration has been explicitly set
pub fn is_hdc_configured() -> bool {
    HDC_CONFIG.get().is_some()
}

// ═══════════════════════════════════════════════════════════════════════════════
// DIMENSION CONVERSION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Dimension mapping specification
///
/// Describes how to map between different HDC dimensions.
#[derive(Debug, Clone, Copy)]
pub struct DimensionMapping {
    /// Source dimension
    pub source: usize,
    /// Target dimension
    pub target: usize,
}

impl DimensionMapping {
    /// Create a new dimension mapping
    pub const fn new(source: usize, target: usize) -> Self {
        Self { source, target }
    }

    /// Check if this is an expansion (target > source)
    pub const fn is_expansion(&self) -> bool {
        self.target > self.source
    }

    /// Check if this is a compression (target < source)
    pub const fn is_compression(&self) -> bool {
        self.target < self.source
    }

    /// Check if this is an identity mapping (same dimension)
    pub const fn is_identity(&self) -> bool {
        self.target == self.source
    }

    /// Get the ratio between dimensions
    pub fn ratio(&self) -> f32 {
        if self.source == 0 {
            0.0
        } else {
            self.target as f32 / self.source as f32
        }
    }

    /// Create mapping from STT (typically 2048) to core dimension
    pub fn stt_to_core(stt_dim: usize) -> Self {
        Self::new(stt_dim, hdc_dim())
    }

    /// Create mapping from core to STT dimension
    pub fn core_to_stt(stt_dim: usize) -> Self {
        Self::new(hdc_dim(), stt_dim)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STT DIMENSION CONSTANTS (for symthaea-stt interop)
// ═══════════════════════════════════════════════════════════════════════════════

/// STT dimension: 2,048 bits (legacy BinaryHV)
///
/// This is the dimension used by symthaea-stt for speech recognition.
/// The projection layer bridges this to the core dimension.
pub const STT_DIMENSION: usize = 2_048;

/// STT binary vector size in bytes
pub const STT_BINARY_BYTES: usize = STT_DIMENSION / 8; // 256 bytes

/// Get the expansion factor from STT to core dimension
///
/// Returns how many times larger the core dimension is compared to STT.
/// Default: 16,384 / 2,048 = 8x
pub fn stt_expansion_factor() -> usize {
    hdc_dim() / STT_DIMENSION
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genesis::GenesisSeed;
    use crate::hdc::hdc_ltc_unified::{UnifiedConfig, UnifiedNetworkConfig};
    use crate::hdc::unified_hv::ContinuousHV;

    fn temporal_identity(profile: HdcLtcEvolutionProfile) -> HdcLtcModelIdentityV1 {
        HdcLtcModelIdentityV1::new(profile, [1; 32], [2; 32])
    }

    fn snapshot_test_network() -> HdcLtcUnifiedNetwork {
        HdcLtcUnifiedNetwork::from_genesis(
            UnifiedNetworkConfig {
                layer_sizes: vec![2, 2],
                neuron_config: UnifiedConfig {
                    dimension: 128,
                    ..UnifiedConfig::default()
                },
                use_layer_binding: true,
                skip_connections: false,
            },
            &GenesisSeed::from_phrase("hdc-ltc-complete-snapshot-test"),
        )
    }

    #[test]
    fn complete_learner_snapshot_round_trip_preserves_future_evolution() {
        let mut original = snapshot_test_network();
        let prefix = ContinuousHV::random(128, 11);
        original.evolve_closed_form(0.01, &prefix);

        // Mutate learning parameters as well as evolution state so this proves
        // more than the existing lightweight NetworkStateSnapshot contract.
        let learning_input = ContinuousHV::random(128, 12);
        if let Some(layer) = original.layer_mut(0) {
            layer[0].hebbian_update(&learning_input, Some(0.003));
        }

        let snapshot = HdcLtcLearnerSnapshotV1::capture(
            &original,
            HdcLtcEvolutionProfile::AdaptiveClosedGateV1,
        )
        .unwrap();
        let wire = bincode::serialize(&snapshot).unwrap();
        let decoded: HdcLtcLearnerSnapshotV1 = bincode::deserialize(&wire).unwrap();
        decoded.validate().unwrap();
        let mut restored = decoded.restore().unwrap();

        let future = ContinuousHV::random(128, 22);
        original.evolve_closed_form(0.025, &future);
        restored.evolve_closed_form(0.025, &future);

        assert_eq!(original.output().values, restored.output().values);
        assert_eq!(
            bincode::serialize(&original).unwrap(),
            bincode::serialize(&restored).unwrap()
        );
    }

    #[test]
    fn complete_learner_snapshot_tamper_fails_closed() {
        let snapshot = HdcLtcLearnerSnapshotV1::capture(
            &snapshot_test_network(),
            HdcLtcEvolutionProfile::AdaptiveClosedGateV1,
        )
        .unwrap();

        let mut tampered_digest = snapshot.clone();
        tampered_digest.snapshot_digest[0] ^= 0xff;
        assert!(tampered_digest.validate().is_err());

        let mut tampered_payload = snapshot;
        let last = tampered_payload.network_bytes.len() - 1;
        tampered_payload.network_bytes[last] ^= 0x01;
        assert!(tampered_payload.validate().is_err());
    }

    #[test]
    fn complete_learner_snapshot_binds_temporal_semantics() {
        let network = snapshot_test_network();
        let adaptive = HdcLtcLearnerSnapshotV1::capture(
            &network,
            HdcLtcEvolutionProfile::AdaptiveClosedGateV1,
        )
        .unwrap();
        let iterative = HdcLtcLearnerSnapshotV1::capture(
            &network,
            HdcLtcEvolutionProfile::SubsteppedExponentialV1,
        )
        .unwrap();

        assert_ne!(adaptive.snapshot_digest, iterative.snapshot_digest);
        assert_ne!(
            adaptive.model_identity().unwrap().evolution_profile,
            iterative.model_identity().unwrap().evolution_profile
        );
    }

    #[test]
    fn temporal_profiles_have_distinct_stable_schema_ids() {
        let profiles = [
            HdcLtcEvolutionProfile::AdaptiveClosedGateV1,
            HdcLtcEvolutionProfile::FrozenEquilibriumExponentialV1,
            HdcLtcEvolutionProfile::SubsteppedExponentialV1,
            HdcLtcEvolutionProfile::Rk4ReferenceV1,
        ];
        for (index, left) in profiles.iter().enumerate() {
            for right in profiles.iter().skip(index + 1) {
                assert_ne!(left.schema_id(), right.schema_id());
            }
        }
        assert!(HdcLtcEvolutionProfile::Rk4ReferenceV1.is_reference());
        assert!(!HdcLtcEvolutionProfile::AdaptiveClosedGateV1.is_reference());
    }

    #[test]
    fn temporal_receipt_rejects_invalid_time() {
        assert!(HdcLtcTemporalSemanticsReceiptV1::new(
            temporal_identity(HdcLtcEvolutionProfile::AdaptiveClosedGateV1),
            f32::NAN,
        )
        .is_err());
        assert!(HdcLtcTemporalSemanticsReceiptV1::new(
            temporal_identity(HdcLtcEvolutionProfile::AdaptiveClosedGateV1),
            -0.001,
        )
        .is_err());
    }

    #[test]
    fn temporal_identity_rejects_zero_digests() {
        assert!(HdcLtcModelIdentityV1::new(
            HdcLtcEvolutionProfile::AdaptiveClosedGateV1,
            [0; 32],
            [2; 32],
        )
        .validate()
        .is_err());
        assert!(HdcLtcModelIdentityV1::new(
            HdcLtcEvolutionProfile::AdaptiveClosedGateV1,
            [1; 32],
            [0; 32],
        )
        .validate()
        .is_err());
    }

    #[test]
    fn test_config_constants() {
        assert_eq!(HdcConfig::COMPACT.dimension, 4_096);
        assert_eq!(HdcConfig::STANDARD.dimension, 16_384);
        assert_eq!(HdcConfig::EXTENDED.dimension, 32_768);
        assert_eq!(HdcConfig::ULTRA.dimension, 65_536);
    }

    #[test]
    fn test_power_of_two() {
        assert!(HdcConfig::STANDARD.is_power_of_two());
        assert!(HdcConfig::COMPACT.is_power_of_two());
        assert!(!HdcConfig::custom(10_000, 16, 0.1).is_power_of_two());
    }

    #[test]
    fn test_memory_calculations() {
        let config = HdcConfig::STANDARD;
        assert_eq!(config.memory_per_continuous_vec(), 16_384 * 4); // 64 KB
        assert_eq!(config.memory_per_binary_vec(), 16_384 / 8); // 2 KB
    }

    #[test]
    fn test_expansion_factor() {
        let config = HdcConfig::STANDARD;
        assert_eq!(config.expansion_factor_from(2_048), 8);
        assert_eq!(config.expansion_factor_from(4_096), 4);
    }

    #[test]
    fn test_tier_names() {
        assert_eq!(HdcConfig::COMPACT.tier_name(), "compact");
        assert_eq!(HdcConfig::STANDARD.tier_name(), "standard");
        assert_eq!(HdcConfig::custom(10_000, 16, 0.1).tier_name(), "custom");
    }

    #[test]
    fn test_from_tier() {
        assert_eq!(HdcConfig::from_tier("standard"), Some(HdcConfig::STANDARD));
        assert_eq!(HdcConfig::from_tier("16k"), Some(HdcConfig::STANDARD));
        assert_eq!(HdcConfig::from_tier("invalid"), None);
    }

    #[test]
    fn test_dimension_mapping() {
        let mapping = DimensionMapping::new(2_048, 16_384);
        assert!(mapping.is_expansion());
        assert!(!mapping.is_compression());
        assert_eq!(mapping.ratio(), 8.0);
    }

    #[test]
    fn test_default_config() {
        let config = HdcConfig::default();
        assert_eq!(config.dimension, 16_384);
    }

    #[test]
    fn test_stt_constants() {
        assert_eq!(STT_DIMENSION, 2_048);
        assert_eq!(STT_BINARY_BYTES, 256);
    }
}

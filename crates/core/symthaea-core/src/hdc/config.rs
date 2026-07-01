// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Configuration - Centralized Dimension Management
//!
//! This module provides a centralized configuration system for HDC dimensions
//! across all Symthaea components. It ensures consistency between STT, TTS,
//! and core consciousness systems.
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

use std::sync::OnceLock;

/// Global HDC configuration (set once at startup)
static HDC_CONFIG: OnceLock<HdcConfig> = OnceLock::new();

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
        if source_dim == 0 {
            0
        } else {
            self.dimension / source_dim
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

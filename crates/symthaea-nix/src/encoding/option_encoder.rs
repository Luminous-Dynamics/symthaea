// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Option Path Encoder
//!
//! Encodes NixOS option paths (e.g., `services.nginx.enable`) into
//! ContinuousHV using hierarchical bind-permute encoding.
//!
//! ## Encoding Strategy
//!
//! An option path `A.B.C` is encoded as:
//!
//! ```text
//! HV(A.B.C) = (A ⊗ level₀) ⊕ (B ⊗ level₁) ⊕ (C ⊗ level₂)
//! ```
//!
//! Where `⊗` is binding (element-wise multiply) and `⊕` is bundling (average).
//!
//! This preserves semantic structure:
//! - `services.nginx.enable` is similar to `services.nginx.port` (shared prefix)
//! - `services.nginx.enable` is more similar to `services.postgresql.enable`
//!   than to `boot.loader.grub.enable` (shared top-level category)
//! - Values can be bound to option paths for full binding encoding

use super::codebook::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// Encodes NixOS option paths and values into hypervectors.
pub struct OptionEncoder<'a> {
    codebook: &'a mut NixCodebook,
}

impl<'a> OptionEncoder<'a> {
    /// Create a new option encoder using a shared codebook.
    pub fn new(codebook: &'a mut NixCodebook) -> Self {
        Self { codebook }
    }

    /// Encode an option path like `services.nginx.enable`.
    ///
    /// Uses hierarchical bind-permute: each segment is bound with its
    /// level marker, then all segments are bundled together.
    pub fn encode_path(&mut self, path: &str) -> ContinuousHV {
        let segments: Vec<&str> = path.split('.').collect();

        if segments.is_empty() {
            return ContinuousHV::zero(self.codebook.dim());
        }

        // Encode each segment at its hierarchy level
        let encoded_segments: Vec<ContinuousHV> = segments
            .iter()
            .enumerate()
            .map(|(level, segment)| self.codebook.encode_segment(segment, level))
            .collect();

        // Bundle all segments — weighted so earlier levels have more influence
        // (top-level category matters more than deep leaf names)
        let weights: Vec<f32> = (0..encoded_segments.len())
            .map(|i| 1.0 / (1.0 + i as f32 * 0.3))
            .collect();

        let refs: Vec<&ContinuousHV> = encoded_segments.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode an option path with its value.
    ///
    /// Result = path_hv ⊗ (value_role ⊗ value_hv)
    ///
    /// This creates a unique encoding for "this option set to this value".
    pub fn encode_binding(&mut self, path: &str, value: &str) -> ContinuousHV {
        let path_hv = self.encode_path(path);
        let value_hv = self.encode_value(value);
        let value_role = self.codebook.value_role.clone();
        let role_value = value_role.bind(&value_hv);
        path_hv.bind(&role_value)
    }

    /// Encode a NixOS option value.
    ///
    /// Booleans, numbers, and strings are encoded differently:
    /// - `true`/`false` → cached basis vectors
    /// - numeric strings → deterministic basis
    /// - other strings → token-based encoding
    pub fn encode_value(&mut self, value: &str) -> ContinuousHV {
        self.codebook.get_or_create(value).clone()
    }

    /// Encode a boolean value.
    pub fn encode_bool(&mut self, value: bool) -> ContinuousHV {
        let token = if value { "true" } else { "false" };
        self.codebook.get_or_create(token).clone()
    }

    /// Encode a list of values (e.g., `environment.systemPackages`).
    ///
    /// Each item is encoded and bundled together.
    pub fn encode_list(&mut self, items: &[&str]) -> ContinuousHV {
        if items.is_empty() {
            return ContinuousHV::zero(self.codebook.dim());
        }

        let encoded: Vec<ContinuousHV> = items
            .iter()
            .map(|item| self.codebook.get_or_create(item).clone())
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similar_paths_have_high_similarity() {
        let mut cb = NixCodebook::new();

        let nginx_enable = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.nginx.enable")
        };
        let nginx_port = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.nginx.port")
        };
        let boot_loader = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("boot.loader.grub.enable")
        };

        let sim_nginx = nginx_enable.similarity(&nginx_port);
        let sim_cross = nginx_enable.similarity(&boot_loader);

        // services.nginx.enable should be MORE similar to services.nginx.port
        // than to boot.loader.grub.enable
        assert!(
            sim_nginx > sim_cross,
            "Expected nginx siblings ({:.3}) > cross-domain ({:.3})",
            sim_nginx,
            sim_cross,
        );
    }

    #[test]
    fn test_same_category_more_similar() {
        let mut cb = NixCodebook::new();

        let nginx_enable = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.nginx.enable")
        };
        let pg_enable = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.postgresql.enable")
        };
        let boot_enable = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("boot.loader.systemd-boot.enable")
        };

        let sim_services = nginx_enable.similarity(&pg_enable);
        let sim_cross = nginx_enable.similarity(&boot_enable);

        // Same top-level category (services) should be more similar
        assert!(
            sim_services > sim_cross,
            "Expected same-category ({:.3}) > cross-category ({:.3})",
            sim_services,
            sim_cross,
        );
    }

    #[test]
    fn test_deterministic() {
        let mut cb = NixCodebook::new();

        let hv1 = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.nginx.enable")
        };
        let hv2 = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.nginx.enable")
        };

        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_binding_differs_from_path() {
        let mut cb = NixCodebook::new();

        let path_hv = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_path("services.nginx.enable")
        };
        let binding_hv = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_binding("services.nginx.enable", "true")
        };

        // Binding should be different from just the path
        let sim = path_hv.similarity(&binding_hv);
        assert!(
            sim < 0.5,
            "Path and binding should differ, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_different_values_different_bindings() {
        let mut cb = NixCodebook::new();

        let enable_true = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_binding("services.nginx.enable", "true")
        };
        let enable_false = {
            let mut enc = OptionEncoder::new(&mut cb);
            enc.encode_binding("services.nginx.enable", "false")
        };

        let sim = enable_true.similarity(&enable_false);
        assert!(
            sim < 0.5,
            "Different values should produce different bindings, got {}",
            sim
        );
    }
}

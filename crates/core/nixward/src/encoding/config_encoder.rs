// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Configuration Encoder
//!
//! Encodes an entire NixOS configuration (from parsed `NixConfig`) into a
//! single ContinuousHV ensemble. This is the "configuration fingerprint" —
//! two similar configs produce similar vectors.

use super::codebook::NixCodebook;
use super::option_encoder::OptionEncoder;
#[cfg(feature = "native")]
use crate::parser::nix_parser::{NixConfig, NixValue};
use symthaea_core::hdc::ContinuousHV;

/// Encodes a full NixOS configuration into a composite hypervector.
pub struct ConfigEncoder<'a> {
    codebook: &'a mut NixCodebook,
}

impl<'a> ConfigEncoder<'a> {
    /// Create a new config encoder using a shared codebook.
    pub fn new(codebook: &'a mut NixCodebook) -> Self {
        Self { codebook }
    }

    /// Encode a parsed NixOS configuration into a single ContinuousHV.
    ///
    /// Each option binding is encoded via `OptionEncoder`, then all bindings
    /// are bundled together into a configuration-level vector.
    #[cfg(feature = "native")]
    pub fn encode_config(&mut self, config: &NixConfig) -> ContinuousHV {
        let dim = self.codebook.dim();
        let options = &config.options;

        if options.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let mut components = Vec::new();
        let mut weights = Vec::new();

        for option in options {
            let value_str = Self::value_to_string(&option.value);
            let binding_hv = {
                let mut opt_enc = OptionEncoder::new(self.codebook);
                opt_enc.encode_binding(&option.path, &value_str)
            };
            components.push(binding_hv);

            // Weight by option "importance" — enable flags and system-level
            // options get higher weight
            let weight = Self::option_weight(&option.path);
            weights.push(weight);
        }

        // Also encode imports as part of the config fingerprint
        for import in &config.imports {
            let import_hv = {
                let mut opt_enc = OptionEncoder::new(self.codebook);
                opt_enc.encode_path(&format!("imports.{import}"))
            };
            components.push(import_hv);
            weights.push(0.5);
        }

        // Also encode module args (they tell us what kind of module this is)
        for arg in &config.module_args {
            let arg_hv = self.codebook.get_or_create(arg).clone();
            components.push(arg_hv);
            weights.push(0.2);
        }

        if components.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Convert a NixValue to a string for encoding.
    #[cfg(feature = "native")]
    fn value_to_string(value: &NixValue) -> String {
        match value {
            NixValue::Bool(b) => b.to_string(),
            NixValue::Int(i) => i.to_string(),
            NixValue::String(s) => s.clone(),
            NixValue::Path(p) => p.clone(),
            NixValue::Null => "null".to_string(),
            NixValue::List(items) => items
                .iter()
                .map(Self::value_to_string)
                .collect::<Vec<_>>()
                .join(" "),
            NixValue::AttrSet(attrs) => attrs.keys().cloned().collect::<Vec<_>>().join(" "),
            NixValue::Expression(e) => e.clone(),
            NixValue::Apply(expr) => expr.clone(),
            NixValue::With { scope, body } => {
                format!("with {} {}", scope, Self::value_to_string(body))
            }
        }
    }

    /// Weight for an option path based on its significance.
    fn option_weight(path: &str) -> f32 {
        if path.contains("enable") {
            1.0 // Enable flags are highly significant
        } else if path.starts_with("boot.") || path.starts_with("networking.") {
            0.9 // System-critical options
        } else if path.starts_with("services.") {
            0.8 // Service configuration
        } else if path.starts_with("environment.systemPackages") {
            0.7 // Package lists
        } else if path.starts_with("users.") {
            0.6
        } else {
            0.5 // Default weight
        }
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::parser::nix_parser::NixParser;

    #[test]
    fn test_encode_simple_config() {
        let mut parser = NixParser::new();
        let code = r#"
{ config, pkgs, ... }: {
    services.nginx.enable = true;
    services.nginx.port = 8080;
}
"#;
        let config = parser.parse(code).unwrap();
        let mut cb = NixCodebook::new();
        let mut enc = ConfigEncoder::new(&mut cb);
        let hv = enc.encode_config(&config);

        assert!(hv.dim() > 0);
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_similar_configs_similar_vectors() {
        let mut parser = NixParser::new();
        let mut cb = NixCodebook::new();

        let config_a = parser
            .parse(
                r#"
{ config, pkgs, ... }: {
    services.nginx.enable = true;
    services.nginx.port = 80;
}
"#,
            )
            .unwrap();

        let config_b = parser
            .parse(
                r#"
{ config, pkgs, ... }: {
    services.nginx.enable = true;
    services.nginx.port = 443;
}
"#,
            )
            .unwrap();

        let config_c = parser
            .parse(
                r#"
{ config, pkgs, ... }: {
    boot.loader.grub.enable = true;
    boot.loader.grub.device = "/dev/sda";
}
"#,
            )
            .unwrap();

        let hv_a = {
            let mut enc = ConfigEncoder::new(&mut cb);
            enc.encode_config(&config_a)
        };
        let hv_b = {
            let mut enc = ConfigEncoder::new(&mut cb);
            enc.encode_config(&config_b)
        };
        let hv_c = {
            let mut enc = ConfigEncoder::new(&mut cb);
            enc.encode_config(&config_c)
        };

        let sim_ab = hv_a.similarity(&hv_b);
        let sim_ac = hv_a.similarity(&hv_c);

        // Two nginx configs should be more similar than nginx vs boot
        assert!(
            sim_ab > sim_ac,
            "Similar configs ({:.3}) should have higher similarity than dissimilar ({:.3})",
            sim_ab,
            sim_ac,
        );
    }

    #[test]
    fn test_encode_empty_config() {
        let mut parser = NixParser::new();
        let config = parser.parse("{ ... }: { }").unwrap();
        let mut cb = NixCodebook::new();
        let mut enc = ConfigEncoder::new(&mut cb);
        let hv = enc.encode_config(&config);
        assert!(hv.dim() > 0);
        assert!(hv.norm() < 1e-6, "Empty config should produce zero vector");
    }

    #[test]
    fn test_encode_config_with_imports() {
        let mut parser = NixParser::new();
        let config = parser
            .parse(
                r#"
{ config, pkgs, ... }: {
    imports = [ ./hardware-configuration.nix ./networking.nix ];
    services.nginx.enable = true;
}
"#,
            )
            .unwrap();

        let mut cb = NixCodebook::new();
        let mut enc = ConfigEncoder::new(&mut cb);
        let hv = enc.encode_config(&config);
        assert!(hv.dim() > 0);
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_option_weight_categories() {
        // Enable flags get highest weight
        assert!((ConfigEncoder::option_weight("services.nginx.enable") - 1.0).abs() < 1e-6);
        // Boot/networking get 0.9
        assert!((ConfigEncoder::option_weight("boot.loader.grub.device") - 0.9).abs() < 1e-6);
        assert!((ConfigEncoder::option_weight("networking.firewall.enable") - 1.0).abs() < 1e-6); // has "enable"
        // Services get 0.8
        assert!((ConfigEncoder::option_weight("services.postgresql.port") - 0.8).abs() < 1e-6);
        // Packages get 0.7
        assert!((ConfigEncoder::option_weight("environment.systemPackages") - 0.7).abs() < 1e-6);
        // Users get 0.6
        assert!((ConfigEncoder::option_weight("users.users.tristan") - 0.6).abs() < 1e-6);
        // Default
        assert!(
            (ConfigEncoder::option_weight("nix.settings.auto-optimise-store") - 0.5).abs() < 1e-6
        );
    }

    #[test]
    fn test_value_to_string_variants() {
        assert_eq!(
            ConfigEncoder::value_to_string(&NixValue::Bool(true)),
            "true"
        );
        assert_eq!(ConfigEncoder::value_to_string(&NixValue::Int(42)), "42");
        assert_eq!(
            ConfigEncoder::value_to_string(&NixValue::String("hello".into())),
            "hello"
        );
        assert_eq!(ConfigEncoder::value_to_string(&NixValue::Null), "null");
        assert_eq!(
            ConfigEncoder::value_to_string(&NixValue::List(vec![
                NixValue::String("a".into()),
                NixValue::String("b".into()),
            ])),
            "a b"
        );
    }
}

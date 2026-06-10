// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System State Encoder
//!
//! Encodes the live NixOS system state (services, packages, config, store)
//! into a single ContinuousHV. This is the "system fingerprint" — the
//! world model's representation of the current system.

use super::codebook::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// Snapshot of observable system state for encoding.
#[derive(Debug, Clone, Default)]
pub struct SystemStateSnapshot {
    /// Active services and their states (name → running/stopped/failed)
    pub services: Vec<(String, ServiceState)>,
    /// Installed packages (attribute names)
    pub packages: Vec<String>,
    /// Active NixOS generation number
    pub generation: Option<u64>,
    /// Nix store size in bytes
    pub store_size_bytes: Option<u64>,
    /// Number of store paths
    pub store_path_count: Option<usize>,
    /// Key config options that are set (path → value string)
    pub config_options: Vec<(String, String)>,
}

/// State of a systemd service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServiceState {
    Running,
    Stopped,
    Failed,
    #[default]
    Inactive,
}

/// Encodes system state snapshots into HDC vectors.
pub struct SystemStateEncoder<'a> {
    codebook: &'a mut NixCodebook,
}

impl<'a> SystemStateEncoder<'a> {
    /// Create a new system state encoder.
    pub fn new(codebook: &'a mut NixCodebook) -> Self {
        Self { codebook }
    }

    /// Encode a full system state snapshot into a single ContinuousHV.
    ///
    /// The result is a weighted bundle of:
    /// - Service state encodings (highest weight — most dynamic)
    /// - Installed package encodings
    /// - Config option encodings
    /// - Store metadata
    pub fn encode_snapshot(&mut self, snapshot: &SystemStateSnapshot) -> ContinuousHV {
        let dim = self.codebook.dim();
        let mut components = Vec::new();
        let mut weights = Vec::new();

        // Encode services
        if !snapshot.services.is_empty() {
            let svc_hv = self.encode_services(&snapshot.services);
            components.push(svc_hv);
            weights.push(1.0);
        }

        // Encode packages
        if !snapshot.packages.is_empty() {
            let pkg_hv = self.encode_packages(&snapshot.packages);
            components.push(pkg_hv);
            weights.push(0.7);
        }

        // Encode config options
        if !snapshot.config_options.is_empty() {
            let cfg_hv = self.encode_config_options(&snapshot.config_options);
            components.push(cfg_hv);
            weights.push(0.8);
        }

        if components.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let refs: Vec<&ContinuousHV> = components.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode service states.
    ///
    /// Each service is: service_basis ⊗ service_role ⊗ state_basis
    fn encode_services(&mut self, services: &[(String, ServiceState)]) -> ContinuousHV {
        let service_role = self.codebook.service_role.clone();

        let encoded: Vec<ContinuousHV> = services
            .iter()
            .map(|(name, state)| {
                let name_hv = self.codebook.get_or_create(name).clone();
                let state_token = match state {
                    ServiceState::Running => "running",
                    ServiceState::Stopped => "stopped",
                    ServiceState::Failed => "failed",
                    ServiceState::Inactive => "inactive",
                };
                let state_hv = self.codebook.get_or_create(state_token).clone();
                name_hv.bind(&service_role).bind(&state_hv)
            })
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Encode installed packages as a bundled set.
    fn encode_packages(&mut self, packages: &[String]) -> ContinuousHV {
        let pkg_role = self.codebook.package_role.clone();

        let encoded: Vec<ContinuousHV> = packages
            .iter()
            .map(|pkg| {
                let pkg_hv = self.codebook.get_or_create(pkg).clone();
                pkg_hv.bind(&pkg_role)
            })
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Encode config option bindings (path=value pairs).
    fn encode_config_options(&mut self, options: &[(String, String)]) -> ContinuousHV {
        let value_role = self.codebook.value_role.clone();

        let encoded: Vec<ContinuousHV> = options
            .iter()
            .map(|(path, value)| {
                // Encode path segments hierarchically
                let segments: Vec<&str> = path.split('.').collect();
                let path_parts: Vec<ContinuousHV> = segments
                    .iter()
                    .enumerate()
                    .map(|(level, seg)| self.codebook.encode_segment(seg, level))
                    .collect();

                let path_refs: Vec<&ContinuousHV> = path_parts.iter().collect();
                let path_hv = ContinuousHV::bundle(&path_refs);

                let val_hv = self.codebook.get_or_create(value).clone();
                path_hv.bind(&value_role.bind(&val_hv))
            })
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Compute the delta between two system state encodings.
    ///
    /// The delta vector represents "what changed" — useful for the
    /// transition model in the world model.
    pub fn compute_delta(before: &ContinuousHV, after: &ContinuousHV) -> ContinuousHV {
        after.subtract(before)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> SystemStateSnapshot {
        SystemStateSnapshot {
            services: vec![
                ("nginx".to_string(), ServiceState::Running),
                ("postgresql".to_string(), ServiceState::Running),
                ("sshd".to_string(), ServiceState::Running),
            ],
            packages: vec!["firefox".to_string(), "vim".to_string(), "git".to_string()],
            config_options: vec![
                ("services.nginx.enable".to_string(), "true".to_string()),
                ("services.postgresql.enable".to_string(), "true".to_string()),
            ],
            generation: Some(42),
            store_size_bytes: None,
            store_path_count: None,
        }
    }

    #[test]
    fn test_encode_snapshot() {
        let mut cb = NixCodebook::new();
        let snapshot = sample_snapshot();

        let hv = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snapshot)
        };

        assert!(hv.dim() > 0);
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_similar_states_similar_vectors() {
        let mut cb = NixCodebook::new();

        let snap_a = SystemStateSnapshot {
            services: vec![
                ("nginx".to_string(), ServiceState::Running),
                ("sshd".to_string(), ServiceState::Running),
            ],
            packages: vec!["firefox".to_string(), "vim".to_string()],
            ..Default::default()
        };

        // snap_b: same services, one more package
        let snap_b = SystemStateSnapshot {
            services: vec![
                ("nginx".to_string(), ServiceState::Running),
                ("sshd".to_string(), ServiceState::Running),
            ],
            packages: vec!["firefox".to_string(), "vim".to_string(), "git".to_string()],
            ..Default::default()
        };

        // snap_c: completely different
        let snap_c = SystemStateSnapshot {
            services: vec![("docker".to_string(), ServiceState::Running)],
            packages: vec!["python3".to_string()],
            ..Default::default()
        };

        let hv_a = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_a)
        };
        let hv_b = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_b)
        };
        let hv_c = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_c)
        };

        let sim_ab = hv_a.similarity(&hv_b);
        let sim_ac = hv_a.similarity(&hv_c);

        assert!(
            sim_ab > sim_ac,
            "Similar states ({:.3}) should have higher similarity than different states ({:.3})",
            sim_ab,
            sim_ac,
        );
    }

    #[test]
    fn test_state_delta() {
        let mut cb = NixCodebook::new();

        let snap_before = SystemStateSnapshot {
            services: vec![("nginx".to_string(), ServiceState::Stopped)],
            ..Default::default()
        };
        let snap_after = SystemStateSnapshot {
            services: vec![("nginx".to_string(), ServiceState::Running)],
            ..Default::default()
        };

        let hv_before = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_before)
        };
        let hv_after = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_after)
        };

        let delta = SystemStateEncoder::compute_delta(&hv_before, &hv_after);
        // Delta should be non-zero (states differ)
        assert!(delta.norm() > 0.0);
    }

    #[test]
    fn test_encode_empty_snapshot() {
        let mut cb = NixCodebook::new();
        let snap = SystemStateSnapshot::default();
        let hv = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap)
        };
        assert!(hv.dim() > 0);
        assert!(
            hv.norm() < 1e-6,
            "Empty snapshot should produce zero vector"
        );
    }

    #[test]
    fn test_encode_all_service_states() {
        let mut cb = NixCodebook::new();
        let snap = SystemStateSnapshot {
            services: vec![
                ("running-svc".to_string(), ServiceState::Running),
                ("stopped-svc".to_string(), ServiceState::Stopped),
                ("failed-svc".to_string(), ServiceState::Failed),
                ("inactive-svc".to_string(), ServiceState::Inactive),
            ],
            ..Default::default()
        };
        let hv = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap)
        };
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_encode_only_config_options() {
        let mut cb = NixCodebook::new();
        let snap = SystemStateSnapshot {
            config_options: vec![
                ("services.nginx.enable".to_string(), "true".to_string()),
                (
                    "boot.loader.grub.device".to_string(),
                    "/dev/sda".to_string(),
                ),
            ],
            ..Default::default()
        };
        let hv = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap)
        };
        assert!(hv.norm() > 0.0, "Config-only snapshot should encode");
    }

    #[test]
    fn test_failed_services_differ_from_running() {
        let mut cb = NixCodebook::new();
        let snap_running = SystemStateSnapshot {
            services: vec![
                ("nginx".to_string(), ServiceState::Running),
                ("sshd".to_string(), ServiceState::Running),
            ],
            ..Default::default()
        };
        let snap_failed = SystemStateSnapshot {
            services: vec![
                ("nginx".to_string(), ServiceState::Failed),
                ("sshd".to_string(), ServiceState::Failed),
            ],
            ..Default::default()
        };

        let hv_running = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_running)
        };
        let hv_failed = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap_failed)
        };
        let sim = hv_running.similarity(&hv_failed);
        assert!(
            sim < 0.99,
            "All-running vs all-failed should differ (sim={:.3})",
            sim
        );
    }

    #[test]
    fn test_delta_same_state_is_zero() {
        let mut cb = NixCodebook::new();
        let snap = sample_snapshot();
        let hv = {
            let mut enc = SystemStateEncoder::new(&mut cb);
            enc.encode_snapshot(&snap)
        };
        let delta = SystemStateEncoder::compute_delta(&hv, &hv);
        assert!(
            delta.norm() < 1e-6,
            "Delta of same state should be zero (got {:.6})",
            delta.norm()
        );
    }
}

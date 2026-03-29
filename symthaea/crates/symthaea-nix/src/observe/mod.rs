// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System state observation -- sensory input layer.
//!
//! This module provides structured observation of the live NixOS system state
//! by parsing the output of system commands (`systemctl`, `nix-env`, `journalctl`,
//! etc.). Each sub-module handles a specific domain of observation.
//!
//! The composite `SystemObserver` ties all observers together and produces a
//! [`SystemStateSnapshot`](crate::encoding::SystemStateSnapshot) suitable for
//! encoding into the HDC world model.

pub mod flake_registry;
pub mod generations;
pub mod hardware;
pub mod journal;
pub mod store;
pub mod systemd;

// Re-export key types for ergonomic access.
pub use flake_registry::{FlakeInfo, FlakeRegistry};
pub use generations::{GenerationDiff, GenerationInfo, GenerationObserver};
pub use hardware::{DiskInfo, GpuInfo, HardwareInfo, HardwareObserver};
pub use journal::{JournalEntry, JournalObserver};
pub use store::{GcRoot, StoreInfo, StoreObserver};
pub use systemd::{SystemdObserver, UnitInfo};

use crate::encoding::{ServiceState, SystemStateSnapshot};

/// Composite observer that builds a full system state snapshot from all
/// sub-observers.
///
/// This is the primary entry point for the sensory input layer. The resulting
/// [`SystemStateSnapshot`] can be fed directly into the
/// [`SystemStateEncoder`](crate::encoding::SystemStateEncoder) to produce
/// an HDC vector representation of the system.
pub struct SystemObserver;

impl SystemObserver {
    /// Take a snapshot of the current system state by querying systemd, the
    /// nix store, and the generation history.
    ///
    /// Individual observer failures are handled gracefully: if a sub-observer
    /// fails (e.g. `systemctl` not available), that portion of the snapshot
    /// is simply left empty/None rather than failing the entire snapshot.
    pub fn snapshot() -> Result<SystemStateSnapshot, std::io::Error> {
        // Observe systemd services
        let services = match SystemdObserver::list_units() {
            Ok(units) => units
                .into_iter()
                .filter(|u| u.name.ends_with(".service"))
                .map(|u| {
                    let state = Self::map_active_state(&u.active_state, &u.sub_state);
                    (u.name, state)
                })
                .collect(),
            Err(_) => Vec::new(),
        };

        // Observe current generation
        let generation = GenerationObserver::current_generation().ok();

        // Observe store info
        let (store_size_bytes, store_path_count) = match StoreObserver::store_info() {
            Ok(info) => (Some(info.total_size_bytes), Some(info.path_count)),
            Err(_) => (None, None),
        };

        Ok(SystemStateSnapshot {
            services,
            packages: Vec::new(), // Package list requires separate nixos-rebuild / nix profile parsing
            generation,
            store_size_bytes,
            store_path_count,
            config_options: Vec::new(), // Config options come from the parser layer, not observation
        })
    }

    /// Map systemd active_state and sub_state strings to the ServiceState enum
    /// used by the encoding layer.
    fn map_active_state(active_state: &str, sub_state: &str) -> ServiceState {
        match (active_state, sub_state) {
            ("active", "running") => ServiceState::Running,
            ("active", _) => ServiceState::Running,
            ("failed", _) => ServiceState::Failed,
            ("inactive", _) => ServiceState::Inactive,
            ("deactivating", _) => ServiceState::Stopped,
            ("activating", _) => ServiceState::Running,
            _ => ServiceState::Stopped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_active_state_running() {
        assert_eq!(
            SystemObserver::map_active_state("active", "running"),
            ServiceState::Running
        );
    }

    #[test]
    fn test_map_active_state_exited() {
        // "active" + "exited" means the service started and finished (oneshot)
        assert_eq!(
            SystemObserver::map_active_state("active", "exited"),
            ServiceState::Running
        );
    }

    #[test]
    fn test_map_active_state_failed() {
        assert_eq!(
            SystemObserver::map_active_state("failed", "failed"),
            ServiceState::Failed
        );
    }

    #[test]
    fn test_map_active_state_inactive() {
        assert_eq!(
            SystemObserver::map_active_state("inactive", "dead"),
            ServiceState::Inactive
        );
    }

    #[test]
    fn test_map_active_state_unknown() {
        assert_eq!(
            SystemObserver::map_active_state("something-else", "weird"),
            ServiceState::Stopped
        );
    }

    #[test]
    fn test_map_active_state_deactivating() {
        assert_eq!(
            SystemObserver::map_active_state("deactivating", "stop-sigterm"),
            ServiceState::Stopped
        );
    }
}
